from __future__ import annotations

import asyncio
import contextlib
import re
from collections import defaultdict
from collections.abc import AsyncIterator
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

from .backend import BackendCanceled, VideoBackend
from .config import Settings
from .models import (
    DEFAULT_DURATION_SECONDS,
    MAX_DURATION_SECONDS,
    JobRecord,
    JobStatus,
    duration_seconds_contract,
    utc_now,
)
from .store import JobStore


_ELAPSED_SUFFIX_RE = re.compile(r"，已运行 \d+ 秒$")


def _with_elapsed_heartbeat(
    job: JobRecord, now: datetime | None = None
) -> JobRecord:
    if job.status != JobStatus.RUNNING or not job.started_at:
        return job
    try:
        started_at = datetime.fromisoformat(job.started_at)
    except ValueError:
        return job
    if started_at.tzinfo is None:
        started_at = started_at.replace(tzinfo=UTC)
    elapsed = max(0, int(((now or datetime.now(UTC)) - started_at).total_seconds()))
    base_message = _ELAPSED_SUFFIX_RE.sub("", job.message)
    return replace(job, message=f"{base_message}，已运行 {elapsed} 秒")


class JobManager:
    def __init__(self, settings: Settings, store: JobStore, backend: VideoBackend):
        self.settings = settings
        self.store = store
        self.backend = backend
        self._queue: asyncio.Queue[str] = asyncio.Queue()
        self._cancel_events: dict[str, asyncio.Event] = {}
        self._watchers: dict[str, set[asyncio.Queue[JobRecord]]] = defaultdict(set)
        self._workers: list[asyncio.Task[None]] = []
        self._start_lock = asyncio.Lock()
        self._started = False

    async def start(self) -> None:
        if self._started:
            return
        async with self._start_lock:
            if self._started:
                return
            self.store.recover_interrupted_jobs()
            self._workers = [
                asyncio.create_task(self._worker(index), name=f"video-worker-{index}")
                for index in range(self.settings.max_concurrent_jobs)
            ]
            self._started = True

    async def stop(self) -> None:
        for event in self._cancel_events.values():
            event.set()
        for worker in self._workers:
            worker.cancel()
        for worker in self._workers:
            with contextlib.suppress(asyncio.CancelledError):
                await worker
        self._workers.clear()
        shutdown = getattr(self.backend, "shutdown", None)
        if shutdown is not None:
            await shutdown()
        self._started = False

    async def submit(self, job: JobRecord) -> tuple[JobRecord, int]:
        await self.start()
        self.store.create(job)
        self._cancel_events[job.id] = asyncio.Event()
        await self._queue.put(job.id)
        await self._publish(job)
        return job, self._queue.qsize()

    async def cancel(self, job_id: str) -> JobRecord:
        await self.start()
        job = self.store.require(job_id)
        if job.terminal:
            return job
        event = self._cancel_events.setdefault(job_id, asyncio.Event())
        event.set()
        now = utc_now()
        if job.status == JobStatus.QUEUED:
            updated = self.store.update(
                job_id,
                status=JobStatus.CANCELED,
                progress=100.0,
                stage="canceled",
                message="任务已在队列中取消",
                finished_at=now,
            )
        else:
            updated = self.store.update(
                job_id,
                status=JobStatus.CANCEL_REQUESTED,
                stage="cancel_requested",
                message="已请求取消，正在停止 FastVideo 进程",
            )
        await self._publish(updated)
        return updated

    async def watch(self, job_id: str) -> AsyncIterator[JobRecord]:
        await self.start()
        queue: asyncio.Queue[JobRecord] = asyncio.Queue(maxsize=16)
        self._watchers[job_id].add(queue)
        current = self.store.require(job_id)
        last_update_at = current.updated_at
        try:
            yield current
            while not current.terminal:
                try:
                    current = await asyncio.wait_for(
                        queue.get(), timeout=self.settings.progress_heartbeat_seconds
                    )
                    if current.updated_at == last_update_at:
                        continue
                    last_update_at = current.updated_at
                except TimeoutError:
                    persisted = self.store.require(job_id)
                    if persisted.updated_at != last_update_at:
                        last_update_at = persisted.updated_at
                    current = _with_elapsed_heartbeat(persisted)
                yield current
        finally:
            watchers = self._watchers.get(job_id)
            if watchers is not None:
                watchers.discard(queue)
                if not watchers:
                    self._watchers.pop(job_id, None)

    def get(self, job_id: str) -> JobRecord:
        return self.store.require(job_id)

    def list(self, limit: int = 20, status: str | None = None) -> list[JobRecord]:
        return self.store.list(limit=limit, status=status)

    def status(self) -> dict[str, object]:
        return {
            "started": self._started,
            "workers": len(self._workers),
            "queue_size": self._queue.qsize(),
            "active_jobs": self.store.count_active(),
            "backend": self.backend.health(),
            "duration_seconds": duration_seconds_contract(
                getattr(
                    self.settings,
                    "default_duration_seconds",
                    DEFAULT_DURATION_SECONDS,
                ),
                getattr(
                    self.settings, "max_duration_seconds", MAX_DURATION_SECONDS
                ),
            ),
        }

    async def _worker(self, index: int) -> None:
        del index
        while True:
            job_id = await self._queue.get()
            try:
                job = self.store.require(job_id)
                if job.status == JobStatus.CANCELED:
                    continue
                cancel_event = self._cancel_events.setdefault(job_id, asyncio.Event())
                if cancel_event.is_set():
                    await self._mark_canceled(job_id, "任务在开始前被取消")
                    continue

                running = self.store.update(
                    job_id,
                    status=JobStatus.RUNNING,
                    progress=5.0,
                    stage="starting",
                    message="GPU Worker 已接收任务",
                    started_at=utc_now(),
                )
                await self._publish(running)
                request = running.to_request()
                job_dir = (self.settings.output_dir / job_id).resolve()
                if self.settings.output_dir not in job_dir.parents:
                    raise RuntimeError("invalid output path")

                async def report(progress: float, stage: str, message: str) -> None:
                    latest = self.store.require(job_id)
                    if latest.terminal:
                        return
                    next_progress = max(latest.progress, min(float(progress), 99.0))
                    next_message = message[:1000]
                    if (
                        latest.progress == next_progress
                        and latest.stage == stage
                        and latest.message == next_message
                    ):
                        return
                    updated = self.store.update(
                        job_id,
                        progress=next_progress,
                        stage=stage,
                        message=next_message,
                    )
                    await self._publish(updated)

                output = await asyncio.wait_for(
                    self.backend.generate(request, job_dir, report, cancel_event),
                    timeout=self.settings.job_timeout_seconds,
                )
                if cancel_event.is_set():
                    await self._mark_canceled(job_id, "任务已取消")
                    continue
                output = output.resolve()
                if self.settings.output_dir not in output.parents:
                    raise RuntimeError("backend returned a file outside the output directory")
                completed = self.store.update(
                    job_id,
                    status=JobStatus.COMPLETED,
                    progress=100.0,
                    stage="completed",
                    message="视频生成完成",
                    output_path=str(output),
                    finished_at=utc_now(),
                    error=None,
                )
                await self._publish(completed)
            except BackendCanceled:
                await self._mark_canceled(job_id, "FastVideo 任务已取消")
            except TimeoutError:
                failed = self.store.update(
                    job_id,
                    status=JobStatus.FAILED,
                    progress=100.0,
                    stage="timeout",
                    message="视频生成超时",
                    error=f"job exceeded {self.settings.job_timeout_seconds} seconds",
                    finished_at=utc_now(),
                )
                await self._publish(failed)
            except asyncio.CancelledError:
                with contextlib.suppress(KeyError):
                    failed = self.store.update(
                        job_id,
                        status=JobStatus.FAILED,
                        progress=100.0,
                        stage="interrupted",
                        message="Worker 停止，任务中断",
                        error="worker_stopped",
                        finished_at=utc_now(),
                    )
                    await self._publish(failed)
                raise
            except Exception as exc:
                failed = self.store.update(
                    job_id,
                    status=JobStatus.FAILED,
                    progress=100.0,
                    stage="failed",
                    message="视频生成失败",
                    error=str(exc)[-4000:],
                    finished_at=utc_now(),
                )
                await self._publish(failed)
            finally:
                self._queue.task_done()
                job = self.store.get(job_id)
                if job is not None and job.terminal:
                    self._cancel_events.pop(job_id, None)

    async def _mark_canceled(self, job_id: str, message: str) -> None:
        current = self.store.require(job_id)
        if current.terminal:
            return
        canceled = self.store.update(
            job_id,
            status=JobStatus.CANCELED,
            progress=100.0,
            stage="canceled",
            message=message,
            finished_at=utc_now(),
        )
        await self._publish(canceled)

    async def _publish(self, job: JobRecord) -> None:
        for queue in tuple(self._watchers.get(job.id, ())):
            if queue.full():
                with contextlib.suppress(asyncio.QueueEmpty):
                    queue.get_nowait()
            queue.put_nowait(job)
