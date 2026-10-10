import asyncio

import pytest
from conftest import FakeBackend
from pydantic import ValidationError

from z_image_mcp.jobs import JobManager
from z_image_mcp.models import GenerateRequest


@pytest.mark.parametrize(
    "values",
    [
        {"prompt": " "},
        {"prompt": "x", "width": 257},
        {"prompt": "x", "width": True},
        {"prompt": "x", "height": 4096},
        {"prompt": "x", "seed": -1},
        {"prompt": "x", "seed": 2**63},
        {"prompt": "x", "steps": 1},
        {"prompt": "x", "negative_prompt": "blur"},
        {"prompt": "x", "guidance_scale": 7},
    ],
)
def test_invalid_requests(values):
    with pytest.raises(ValidationError):
        GenerateRequest(**values)


async def terminal(manager, job_id):
    async with asyncio.timeout(10):
        async for job in manager.events(job_id):
            pass
    return job


async def test_serial_queue_failure_recovery_and_persistence(settings):
    backend = FakeBackend()
    manager = JobManager(settings, backend)
    await manager.start()
    try:
        failed = manager.submit(GenerateRequest(prompt="fail"))
        first = manager.submit(GenerateRequest(prompt="中文测试", seed=42))
        second = manager.submit(GenerateRequest(prompt="second"))
        assert (await terminal(manager, failed.id)).status == "failed"
        assert (await terminal(manager, first.id)).status == "succeeded"
        assert (await terminal(manager, second.id)).status == "succeeded"
        assert backend.max_active == 1
        saved = manager.get(first.id)
        assert saved.request.seed == 42
        assert saved.progress == 100
        assert "download_token" not in manager.public(saved)
    finally:
        await manager.close()
    restarted = JobManager(settings, FakeBackend())
    await restarted.start()
    try:
        assert restarted.get(first.id).result == saved.result
        assert restarted.get(first.id).download_token == saved.download_token
    finally:
        await restarted.close()


async def test_queued_and_running_cancellation(settings):
    backend = FakeBackend(delay=0.15)
    manager = JobManager(settings, backend)
    await manager.start()
    try:
        running = manager.submit(GenerateRequest(prompt="running"))
        queued = manager.submit(GenerateRequest(prompt="queued"))
        manager.cancel(queued.id)
        assert (await terminal(manager, queued.id)).status == "canceled"
        async for event in manager.events(running.id):
            if event.progress > 0:
                break
        assert manager.cancel(running.id).status == "cancel_requested"
        assert (await terminal(manager, running.id)).status == "canceled"
        assert len(backend.requests) == 1
        assert not list(manager.output_dir.glob("*.png"))
        assert manager.cancel(running.id).status == "canceled"
    finally:
        await manager.close()


async def test_limits_before_persistence(settings):
    settings.max_pending_jobs = 1
    manager = JobManager(settings, FakeBackend())
    await manager.start()
    try:
        with pytest.raises(ValueError, match="pixel limit"):
            manager.submit(GenerateRequest(prompt="oversized", width=2048, height=2048))
        assert manager.db.execute("SELECT COUNT(*) FROM jobs").fetchone()[0] == 0
        manager.submit(GenerateRequest(prompt="first"))
        with pytest.raises(ValueError, match="queue is full"):
            manager.submit(GenerateRequest(prompt="second"))
        with pytest.raises(ValueError, match="Unknown job"):
            manager.get("../../etc/passwd")
    finally:
        await manager.close()


async def test_restart_marks_interrupted_job_failed_and_lock_excludes_second_worker(settings):
    manager = JobManager(settings, FakeBackend())
    await manager.start()
    try:
        other = JobManager(settings, FakeBackend())
        with pytest.raises(RuntimeError, match="Another worker"):
            await other.start()
        job = manager.submit(GenerateRequest(prompt="test"))
        await terminal(manager, job.id)
        job = manager.get(job.id)
        job.status = "running"
        manager._save(job)
    finally:
        await manager.close()
    manager = JobManager(settings, FakeBackend())
    await manager.start()
    try:
        restored = manager.get(job.id)
        assert restored.status == "failed"
        assert restored.stage == "interrupted"
        assert restored.finished_at
    finally:
        await manager.close()


async def test_disk_limit_and_idle_unload(settings, monkeypatch):
    import shutil

    monkeypatch.setattr(shutil, "disk_usage", lambda _: type("Usage", (), {"free": 0})())
    settings.min_free_disk_mb = 1
    settings.idle_unload_seconds = 1
    backend = FakeBackend()
    manager = JobManager(settings, backend)
    await manager.start()
    try:
        with pytest.raises(ValueError, match="disk space"):
            manager.submit(GenerateRequest(prompt="disk full"))
        settings.min_free_disk_mb = 0
        job = manager.submit(GenerateRequest(prompt="idle"))
        await terminal(manager, job.id)
        assert backend.loaded
        await asyncio.sleep(1.2)
        assert not backend.loaded
    finally:
        await manager.close()
