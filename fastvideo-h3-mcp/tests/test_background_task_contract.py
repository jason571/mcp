from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from fastvideo_h3_mcp.models import JobStatus
from fastvideo_h3_mcp.server import _tracked_job, get_video_job, cancel_video_job


def job(status=JobStatus.RUNNING):
    return SimpleNamespace(
        id="job-contract", status=status, progress=42.0, stage="rendering",
        message="service progress", output_file=lambda: None,
        as_public_dict=lambda: {"id": "job-contract", "status": status, "error": None},
    )


class BackgroundTaskContractTests(unittest.IsolatedAsyncioTestCase):
    def test_service_owns_stable_job_and_control_mapping(self):
        runtime = SimpleNamespace()
        for status, expected in (
            (JobStatus.QUEUED, "working"), (JobStatus.RUNNING, "working"),
            (JobStatus.CANCEL_REQUESTED, "working"), (JobStatus.COMPLETED, "completed"),
            (JobStatus.FAILED, "failed"), (JobStatus.CANCELED, "cancelled"),
        ):
            task = _tracked_job(runtime, job(status))["_meta"]["exoharness"]["background_task"]
            self.assertEqual(task["status"], expected)
            self.assertEqual(task["poll"], {"tool": "get_video_job", "arguments": {"job_id": "job-contract", "managed_tracking": True}})
            self.assertEqual(task["cancel"], {"tool": "cancel_video_job", "arguments": {"job_id": "job-contract", "managed_tracking": True}})
            self.assertEqual(task["progress"]["progress"], 42.0)

    async def test_ordinary_status_query_remains_one_shot(self):
        runtime = SimpleNamespace(manager=SimpleNamespace(get=lambda _: job()))
        with patch("fastvideo_h3_mcp.server._require_tool_auth", return_value=runtime):
            ordinary = await get_video_job("job-contract", None)
            tracked = await get_video_job("job-contract", None, managed_tracking=True)
        self.assertNotIn("_meta", ordinary)
        self.assertEqual(tracked["_meta"]["exoharness"]["background_task"]["id"], "job-contract")

    async def test_tracked_cancel_uses_existing_service_cancellation(self):
        cancel = AsyncMock(return_value=job(JobStatus.CANCELED))
        runtime = SimpleNamespace(manager=SimpleNamespace(cancel=cancel))
        with patch("fastvideo_h3_mcp.server._require_tool_auth", return_value=runtime):
            result = await cancel_video_job("job-contract", None, managed_tracking=True)
        cancel.assert_awaited_once_with("job-contract")
        self.assertEqual(result["_meta"]["exoharness"]["background_task"]["status"], "cancelled")
