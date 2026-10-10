import asyncio
import fcntl
import logging
import secrets
import shutil
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from threading import Event
from uuid import uuid4

from .backend import GenerationCanceled
from .config import Settings
from .models import TERMINAL, GenerateRequest, Job

log = logging.getLogger(__name__)


def now() -> str:
    return datetime.now(UTC).isoformat()


class JobManager:
    """One event-loop-owned store and one serial GPU worker per data directory."""

    def __init__(self, settings: Settings, backend):
        self.settings = settings
        self.backend = backend
        self.queue: asyncio.Queue[str | None] = asyncio.Queue()
        self.cancels: dict[str, Event] = {}
        self.subscribers: dict[str, set[asyncio.Queue]] = {}
        self.db = None
        self.lock_file = None
        self.worker = None
        self.stopping = False

    @property
    def output_dir(self) -> Path:
        return self.settings.data_dir / "images"

    async def start(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.lock_file = (self.settings.data_dir / "worker.lock").open("a+")
        try:
            fcntl.flock(self.lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self.lock_file.close()
            raise RuntimeError("Another worker already owns Z_IMAGE_DATA_DIR") from None
        self.db = sqlite3.connect(self.settings.data_dir / "jobs.sqlite3")
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("CREATE TABLE IF NOT EXISTS jobs (id TEXT PRIMARY KEY, data TEXT NOT NULL)")
        for (raw,) in self.db.execute("SELECT data FROM jobs").fetchall():
            job = Job.model_validate_json(raw)
            if job.status not in TERMINAL:
                job.status = "failed"
                job.stage = "interrupted"
                job.message = "Service restarted before this job completed; submit a new request"
                job.error = job.message
                job.finished_at = now()
                self._save(job)
        self.worker = asyncio.create_task(self._work(), name="z-image-gpu-worker")

    async def close(self):
        self.stopping = True
        for job_id in list(self.cancels):
            self.cancel(job_id)
        await self.queue.put(None)
        if self.worker:
            await self.worker
        await asyncio.to_thread(self.backend.unload)
        if self.db:
            self.db.close()
        if self.lock_file:
            self.lock_file.close()

    def _save(self, job: Job):
        self.db.execute(
            "INSERT OR REPLACE INTO jobs (id, data) VALUES (?, ?)",
            (job.id, job.model_dump_json()),
        )
        self.db.commit()
        for queue in self.subscribers.get(job.id, set()):
            queue.put_nowait(job.model_copy(deep=True))

    def get(self, job_id: str) -> Job:
        row = self.db.execute("SELECT data FROM jobs WHERE id = ?", (job_id,)).fetchone()
        if row is None:
            raise ValueError("Unknown job_id")
        return Job.model_validate_json(row[0])

    def submit(self, request: GenerateRequest) -> Job:
        if self.stopping:
            raise ValueError("Service is stopping; retry after restart")
        if request.width * request.height > self.settings.max_pixels:
            raise ValueError(f"Image exceeds the {self.settings.max_pixels}-pixel limit")
        if len(self.cancels) >= self.settings.max_pending_jobs:
            raise ValueError("GPU queue is full; wait for an existing job to complete")
        if shutil.disk_usage(self.output_dir).free < self.settings.min_free_disk_mb * 1024**2:
            raise ValueError("Insufficient output disk space")
        job = Job(
            id=uuid4().hex,
            status="queued",
            request=request,
            created_at=now(),
            download_token=secrets.token_urlsafe(32),
        )
        self._save(job)
        self.cancels[job.id] = Event()
        self.queue.put_nowait(job.id)
        return job

    def cancel(self, job_id: str) -> Job:
        job = self.get(job_id)
        if job.status in TERMINAL:
            return job
        self.cancels[job_id].set()
        if job.status == "queued":
            job.status = "canceled"
            job.stage = "canceled"
            job.message = "Canceled before GPU execution"
            job.finished_at = now()
        else:
            job.status = "cancel_requested"
            job.message = "Cancellation requested; waiting for the next safe GPU boundary"
        self._save(job)
        return job

    async def progress(self, job_id: str, percent: float, stage: str, message: str):
        job = self.get(job_id)
        if job.status != "running":
            return
        job.progress = max(job.progress, min(percent, 99))
        job.stage = stage
        job.message = message
        self._save(job)

    async def events(self, job_id: str):
        queue = asyncio.Queue()
        self.subscribers.setdefault(job_id, set()).add(queue)
        try:
            job = self.get(job_id)
            yield job
            while job.status not in TERMINAL:
                try:
                    job = await asyncio.wait_for(queue.get(), timeout=15)
                except TimeoutError:
                    # Heartbeats repeat the actual state; they never invent progress.
                    job = self.get(job_id)
                yield job
        finally:
            self.subscribers[job_id].discard(queue)
            if not self.subscribers[job_id]:
                del self.subscribers[job_id]

    async def _work(self):
        loop = asyncio.get_running_loop()
        while True:
            try:
                timeout = self.settings.idle_unload_seconds or None
                job_id = await asyncio.wait_for(self.queue.get(), timeout=timeout)
            except TimeoutError:
                if self.backend.loaded:
                    await asyncio.to_thread(self.backend.unload)
                    log.info("Released idle model from GPU")
                continue
            if job_id is None:
                self.queue.task_done()
                return
            try:
                job = self.get(job_id)
                if job.status in TERMINAL:
                    continue
                job.status = "running"
                job.started_at = now()
                job.stage = "starting"
                job.message = "GPU worker started"
                self._save(job)

                def report(percent, stage, message):
                    asyncio.run_coroutine_threadsafe(
                        self.progress(job_id, percent, stage, message), loop
                    ).result()

                result = await asyncio.to_thread(
                    self.backend.generate,
                    job.request,
                    self.output_dir / f"{job.id}.png",
                    report,
                    self.cancels[job.id],
                )
                job = self.get(job_id)
                if self.cancels[job.id].is_set():
                    raise GenerationCanceled()
                job.status = "succeeded"
                job.stage = "completed"
                job.progress = 100
                job.message = "Image ready"
                job.result = result
            except GenerationCanceled:
                job = self.get(job_id)
                job.status = "canceled"
                job.stage = "canceled"
                job.message = "Generation canceled"
                # A cancellation racing with PNG save must not expose a successful result.
                for suffix in (".png", ".json", ".tmp"):
                    (self.output_dir / f"{job_id}{suffix}").unlink(missing_ok=True)
            except Exception as exc:
                log.exception("Image job %s failed", job_id)
                job = self.get(job_id)
                job.status = "failed"
                job.stage = "failed"
                job.error = f"{type(exc).__name__}: {exc}"[:1500]
                job.message = "Generation failed; see error and server log"
            finally:
                # Includes jobs canceled while still queued.
                if job.status in TERMINAL:
                    job.finished_at = job.finished_at or now()
                    self._save(job)
                self.cancels.pop(job_id, None)
                self.queue.task_done()

    def public(self, job: Job) -> dict:
        data = job.model_dump(exclude={"download_token"})
        if job.status == "succeeded":
            data["result"] = {
                **job.result,
                "download_url": (
                    f"{self.settings.public_url}/images/{job.id}.png?key={job.download_token}"
                ),
            }
        return data
