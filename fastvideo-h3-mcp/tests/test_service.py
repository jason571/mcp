from __future__ import annotations

import asyncio
import tempfile
import unittest
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

from fastvideo_h3_mcp.backend import MockBackend
from fastvideo_h3_mcp.models import GenerationRequest, JobRecord, JobStatus
from fastvideo_h3_mcp.service import JobManager, _with_elapsed_heartbeat
from fastvideo_h3_mcp.store import JobStore


class JobManagerTests(unittest.IsolatedAsyncioTestCase):
    async def test_watch_heartbeat_refreshes_elapsed_time(self) -> None:
        request = GenerationRequest(
            prompt="A test scene.",
            negative_prompt=None,
            soundscape=None,
            prompt_mode="simple",
            model_id="fast",
            duration_seconds=5,
            seed=1,
            audio_enabled=True,
            aspect_ratio="16:9",
            resolution="512p",
        )
        running = replace(
            JobRecord.new(request),
            status=JobStatus.RUNNING,
            stage="joint_av_denoising",
            message="视频和音频联合生成，第 1/4 步正在计算，已运行 62 秒",
            started_at="2026-09-01T09:20:00+00:00",
        )
        heartbeat = _with_elapsed_heartbeat(
            running, datetime(2026, 9, 1, 9, 24, 25, tzinfo=UTC)
        )
        self.assertEqual(
            heartbeat.message,
            "视频和音频联合生成，第 1/4 步正在计算，已运行 265 秒",
        )

    async def test_mock_job_publishes_progress_and_completes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            settings = SimpleNamespace(
                output_dir=root / "outputs",
                max_concurrent_jobs=1,
                progress_heartbeat_seconds=1,
                job_timeout_seconds=30,
            )
            settings.output_dir.mkdir()
            store = JobStore(root / "jobs.sqlite3")
            manager = JobManager(
                settings,  # type: ignore[arg-type]
                store,
                MockBackend(step_delay_seconds=0.01),
            )
            request = GenerationRequest(
                prompt="A test scene.",
                negative_prompt=None,
                soundscape=None,
                prompt_mode="simple",
                model_id="auto",
                duration_seconds=5,
                seed=1,
                audio_enabled=True,
                aspect_ratio="16:9",
                resolution="768p",
            )
            job, _position = await manager.submit(JobRecord.new(request))
            updates = [update async for update in manager.watch(job.id)]
            self.assertEqual(updates[-1].status, JobStatus.COMPLETED)
            self.assertEqual(updates[-1].progress, 100)
            progresses = [update.progress for update in updates]
            self.assertEqual(progresses, sorted(progresses))
            self.assertGreaterEqual(len(set(progresses)), 5)
            status = manager.status()
            duration = status["duration_seconds"]
            assert isinstance(duration, dict)
            self.assertIsInstance(duration["default"], float)
            self.assertEqual(duration["default"], 15.0)
            self.assertEqual(duration["maximum"], 60.0)
            output = updates[-1].output_file()
            self.assertIsNotNone(output)
            assert output is not None
            self.assertTrue(output.is_file())
            await manager.stop()
            store.close()


if __name__ == "__main__":
    unittest.main()
