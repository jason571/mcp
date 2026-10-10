import asyncio
import sqlite3
from pathlib import Path
from threading import Event

import pytest

from yue2_mcp.backend import GenerationCanceled
from yue2_mcp.config import Settings
from yue2_mcp.jobs import JobManager
from yue2_mcp.models import LyricsRequest, MusicRequest


class FakeBackend:
    loaded = False

    def __init__(self):
        self.running = 0
        self.max_running = 0
        self.gate = Event()
        self.unloads = 0

    def unload(self):
        self.loaded = False
        self.unloads += 1

    def lyrics(self, request, cancel):
        return f"[Verse 1]\n{request.prompt}\n[Chorus]\n{request.prompt}"

    def music(self, request, lyrics, output: Path, report, cancel):
        self.running += 1
        self.max_running = max(self.max_running, self.running)
        try:
            report(25, "planning", "Planning score")
            self.gate.wait(timeout=3)
            if cancel.is_set():
                raise GenerationCanceled()
            if request.style == "fail":
                raise RuntimeError("simulated inference failure")
            output.mkdir(parents=True)
            (output / "song.flac").write_bytes(b"fake flac")
            (output / "preview.mp3").write_bytes(b"fake mp3")
            (output / "lyrics.txt").write_text(lyrics, encoding="utf-8")
            report(90, "saving", "Saving")
            return {"lyrics": lyrics, "preview_available": True,
                    "score_available": False, "sha256": "test"}
        finally:
            self.running -= 1


def settings(tmp_path):
    return Settings(_env_file=None, data_dir=tmp_path, min_free_disk_mb=0,
                    lyrics_base_url="http://localhost:8/v1",
                    lyrics_model="test")


async def terminal(manager, job_id):
    for _ in range(100):
        job = manager.get(job_id)
        if job.status in {"succeeded", "failed", "canceled"}:
            return job
        await asyncio.sleep(0.05)
    raise AssertionError(f"Job did not finish: {manager.get(job_id)}")


async def serial_queue_cancel_and_restart(tmp_path):
    backend = FakeBackend()
    manager = JobManager(settings(tmp_path), backend)
    await manager.start()
    try:
        first = manager.submit("music", MusicRequest(style="pop", lyrics="hello"))
        second = manager.submit("music", MusicRequest(style="rock", lyrics="world"))
        await asyncio.sleep(0.05)
        assert manager.get(first.id).status == "running"
        assert manager.get(second.id).status == "queued"
        assert manager.public(manager.get(second.id))["queue_position"] == 1
        assert manager.cancel(second.id).status == "canceled"
        backend.gate.set()
        assert (await terminal(manager, first.id)).status == "succeeded"
        assert (await terminal(manager, second.id)).status == "canceled"
        assert backend.max_running == 1
        assert "download_token" not in manager.public(manager.get(first.id))
        assert manager.public(manager.get(first.id))["result"]["audio_url"].endswith(
            f"?key={manager.get(first.id).download_token}"
        )
    finally:
        await manager.queue.join()
        await manager.close()
    interrupted = first.model_copy(update={"id": "interrupted-job", "status": "running"})
    with sqlite3.connect(tmp_path / "jobs.sqlite3") as db:
        db.execute("INSERT INTO jobs (id, data) VALUES (?, ?)",
                   (interrupted.id, interrupted.model_dump_json()))
    restarted = JobManager(settings(tmp_path), FakeBackend())
    await restarted.start()
    try:
        assert restarted.get(first.id).status == "succeeded"
        assert restarted.get(second.id).status == "canceled"
        assert restarted.get(interrupted.id).status == "failed"
        assert restarted.get(interrupted.id).stage == "interrupted"
    finally:
        await restarted.queue.join()
        await restarted.close()


def test_serial_queue_cancel_and_restart(tmp_path):
    asyncio.run(serial_queue_cancel_and_restart(tmp_path))


async def lyrics_and_validation(tmp_path):
    backend = FakeBackend()
    backend.gate.set()
    manager = JobManager(settings(tmp_path), backend)
    await manager.start()
    try:
        job = manager.submit("lyrics", LyricsRequest(prompt="rain", style="jazz"))
        result = await terminal(manager, job.id)
        assert result.result["lyrics"].startswith("[Verse 1]")
        failed = manager.submit("music", MusicRequest(style="fail", lyrics="words"))
        recovered = manager.submit("music", MusicRequest(style="pop", lyrics="words"))
        assert (await terminal(manager, failed.id)).status == "failed"
        assert (await terminal(manager, recovered.id)).status == "succeeded"
        automatic = manager.submit("music", MusicRequest(style="jazz", lyrics_prompt="rain"))
        assert (await terminal(manager, automatic.id)).result["lyrics"].startswith("[Verse 1]")
        with pytest.raises(ValueError):
            MusicRequest(style="pop", lyrics="")
        with pytest.raises(ValueError):
            MusicRequest(style="pop", lyrics="words", abc="X:1", cot="off")
    finally:
        await manager.queue.join()
        await manager.close()


def test_lyrics_and_validation(tmp_path):
    asyncio.run(lyrics_and_validation(tmp_path))


async def idle_unload(tmp_path):
    backend = FakeBackend()
    backend.loaded = True
    manager = JobManager(settings(tmp_path).model_copy(update={"idle_unload_seconds": 1}),
                         backend)
    await manager.start()
    try:
        await asyncio.sleep(1.2)
        assert backend.unloads == 1
        assert not backend.loaded
    finally:
        await manager.close()


def test_idle_unload(tmp_path):
    asyncio.run(idle_unload(tmp_path))
