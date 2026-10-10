import asyncio
import fcntl
import logging
import secrets
import shutil
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
from datetime import UTC, datetime
from pathlib import Path
from threading import Event
from uuid import uuid4

from .backend import GenerationCanceled
from .config import Settings
from .models import TERMINAL, Job, LyricsRequest, MusicRequest

log = logging.getLogger(__name__)


def now() -> str:
    return datetime.now(UTC).isoformat()


class JobManager:
    def __init__(self, settings: Settings, backend):
        self.settings = settings
        self.backend = backend
        self.queue: asyncio.Queue[str] = asyncio.Queue()
        self.cancels: dict[str, Event] = {}
        self.subscribers: dict[str, set[asyncio.Queue]] = {}
        self.db = None
        self.lock_file = None
        self.worker = None
        self.stopping = False
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="yue2-inference")

    @property
    def output_dir(self) -> Path:
        return self.settings.data_dir / "outputs"

    async def start(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.lock_file = (self.settings.data_dir / "worker.lock").open("a+")
        try:
            fcntl.flock(self.lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self.lock_file.close()
            raise RuntimeError("Another YuE2 worker owns this data directory") from None
        self.db = sqlite3.connect(self.settings.data_dir / "jobs.sqlite3")
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("CREATE TABLE IF NOT EXISTS jobs (id TEXT PRIMARY KEY, data TEXT NOT NULL)")
        for (raw,) in self.db.execute("SELECT data FROM jobs").fetchall():
            job = Job.model_validate_json(raw)
            if job.status not in TERMINAL:
                job.status = "failed"
                job.stage = "interrupted"
                job.error = "Service restarted before completion; submit a new job"
                job.message = job.error
                job.finished_at = now()
                self._save(job)
        self.worker = asyncio.create_task(self._work(), name="yue2-serial-worker")

    async def close(self):
        self.stopping = True
        for event in self.cancels.values():
            event.set()
        await self.queue.join()
        if self.worker:
            self.worker.cancel()
            with suppress(asyncio.CancelledError):
                await self.worker
        self.executor.shutdown(wait=True)
        self.backend.unload()
        if self.db:
            self.db.close()
        if self.lock_file:
            self.lock_file.close()

    def _save(self, job: Job):
        self.db.execute(
            "INSERT OR REPLACE INTO jobs VALUES (?, ?)", (job.id, job.model_dump_json())
        )
        self.db.commit()
        for queue in self.subscribers.get(job.id, set()):
            queue.put_nowait(job.model_copy(deep=True))

    def get(self, job_id: str) -> Job:
        row = self.db.execute("SELECT data FROM jobs WHERE id = ?", (job_id,)).fetchone()
        if row is None:
            raise ValueError("Unknown job_id")
        return Job.model_validate_json(row[0])

    def submit(self, kind: str, request: MusicRequest | LyricsRequest) -> Job:
        if self.stopping:
            raise ValueError("Service is stopping")
        if kind == "music" and not request.lyrics and not self.settings.lyrics_base_url:
            raise ValueError("Automatic lyrics require a configured lyrics LLM")
        if kind == "lyrics" and not self.settings.lyrics_base_url:
            raise ValueError("Lyrics LLM is not configured")
        if len(self.cancels) >= self.settings.max_pending_jobs:
            raise ValueError("Queue is full")
        if shutil.disk_usage(self.output_dir).free < self.settings.min_free_disk_mb * 1024**2:
            raise ValueError("Insufficient output disk space")
        job = Job(id=uuid4().hex, kind=kind, status="queued", request=request.model_dump(),
                  created_at=now(), download_token=secrets.token_urlsafe(32))
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
            job.message = "Canceled before execution"
            job.finished_at = now()
        else:
            job.status = "cancel_requested"
            job.message = "Cancellation requested; waiting for a safe boundary"
        self._save(job)
        return job

    def progress(self, job_id: str, percent: int, stage: str, message: str):
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
                if self.settings.idle_unload_seconds:
                    job_id = await asyncio.wait_for(
                        self.queue.get(), timeout=self.settings.idle_unload_seconds
                    )
                else:
                    job_id = await self.queue.get()
            except TimeoutError:
                if self.backend.loaded:
                    await loop.run_in_executor(self.executor, self.backend.unload)
                    log.info("Released idle YuE2 model from GPU")
                continue
            try:
                job = self.get(job_id)
                if job.status in TERMINAL:
                    continue
                job.status = "running"
                job.started_at = now()
                job.stage = "starting"
                self._save(job)

                def report(percent, stage, message):
                    loop.call_soon_threadsafe(self.progress, job_id, percent, stage, message)

                cancel = self.cancels[job_id]
                if job.kind == "lyrics":
                    self.progress(job_id, 10, "lyrics", "Generating lyrics")
                    lyrics = await loop.run_in_executor(
                        self.executor, self.backend.lyrics,
                        LyricsRequest.model_validate(job.request), cancel,
                    )
                    result = {"lyrics": lyrics}
                else:
                    request = MusicRequest.model_validate(job.request)
                    lyrics = request.lyrics
                    if not lyrics:
                        self.progress(job_id, 5, "lyrics", "Generating lyrics")
                        lyrics = await loop.run_in_executor(
                            self.executor, self.backend.lyrics,
                            LyricsRequest(prompt=request.lyrics_prompt, style=request.style,
                                          language=request.language), cancel,
                        )
                    if cancel.is_set():
                        raise GenerationCanceled()
                    result = await loop.run_in_executor(
                        self.executor, self.backend.music,
                        request, lyrics, self.output_dir / job_id,
                        report, cancel,
                    )
                if cancel.is_set():
                    raise GenerationCanceled()
                job = self.get(job_id)
                job.status = "succeeded"
                job.stage = "completed"
                job.progress = 100
                job.message = "Ready"
                job.result = result
            except (GenerationCanceled, InterruptedError):
                job = self.get(job_id)
                job.status = "canceled"
                job.stage = "canceled"
                job.message = "Generation canceled"
                shutil.rmtree(self.output_dir / job_id, ignore_errors=True)
            except Exception as exc:
                log.exception("YuE2 job %s failed", job_id)
                try:
                    await loop.run_in_executor(self.executor, self.backend.unload)
                except Exception:
                    log.exception("Could not release YuE2 model after failure")
                job = self.get(job_id)
                job.status = "failed"
                job.stage = "failed"
                job.error = f"{type(exc).__name__}: {exc}"[:1500]
                job.message = "Job failed; inspect error and server log"
                shutil.rmtree(self.output_dir / job_id, ignore_errors=True)
            finally:
                if job.status in TERMINAL:
                    job.finished_at = job.finished_at or now()
                    self._save(job)
                self.cancels.pop(job_id, None)
                self.queue.task_done()

    def public(self, job: Job) -> dict:
        data = job.model_dump(exclude={"download_token"})
        if job.status == "queued":
            queued = [key for key in self.cancels if self.get(key).status == "queued"]
            data["queue_position"] = queued.index(job.id) + 1 if job.id in queued else None
        else:
            data["queue_position"] = None
        if job.status == "succeeded" and job.kind == "music":
            base = f"{self.settings.public_url}/files/{job.id}"
            key = f"?key={job.download_token}"
            data["result"] = {
                **job.result,
                "audio_url": f"{base}/song.flac{key}",
                "preview_url": (
                    f"{base}/preview.mp3{key}" if job.result.get("preview_available") else None
                ),
                "lyrics_url": f"{base}/lyrics.txt{key}",
                "score_url": (
                    f"{base}/score.abc{key}" if job.result.get("score_available") else None
                ),
                "metadata_url": f"{base}/metadata.json{key}",
            }
        return data
