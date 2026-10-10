from __future__ import annotations

import json
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from fastvideo_h3_mcp import server
from fastvideo_h3_mcp.models import JobStatus


def job(status=JobStatus.COMPLETED):
    return SimpleNamespace(
        id="visibility-job", status=status, progress=100 if status == JobStatus.COMPLETED else 42,
        stage="completed" if status == JobStatus.COMPLETED else "rendering",
        message="real service progress", terminal=status == JobStatus.COMPLETED,
        output_file=lambda: Path("/tmp/result.mp4") if status == JobStatus.COMPLETED else None,
        as_public_dict=lambda: {"id": "visibility-job", "status": status},
    )


def runtime(value):
    return SimpleNamespace(
        settings=SimpleNamespace(artifact_signing_key="fixture", artifact_url_ttl_seconds=3600),
        signer=SimpleNamespace(build_url=lambda *_: "https://media.test/result.mp4?signature=fixture"),
        manager=SimpleNamespace(get=lambda _: value, cancel=AsyncMock(return_value=value),
                                submit=AsyncMock(return_value=(value, 0))),
    )


class ArtifactVisibilityTests(unittest.IsolatedAsyncioTestCase):
    def test_internal_keeps_exact_output_and_progress_without_changing_default(self):
        value = job()
        rt = runtime(value)
        ordinary = server._public_job(rt, value)
        internal = server._public_job(rt, value, "internal")
        self.assertNotIn("_meta", ordinary)
        self.assertEqual(internal["result"], ordinary["result"])
        self.assertEqual(internal["_meta"]["exoharness"]["artifact_visibility"], "internal")
        for status in (JobStatus.QUEUED, JobStatus.RUNNING, JobStatus.COMPLETED,
                       JobStatus.FAILED, JobStatus.CANCELED):
            state = job(status)
            tagged = server._tracked_job(rt, state, "internal")
            metadata = tagged["_meta"]["exoharness"]
            self.assertEqual(metadata["artifact_visibility"], "internal")
            self.assertEqual(metadata["background_task"]["progress"]["progress"], state.progress)
            for action in ("poll", "cancel"):
                self.assertEqual(metadata["background_task"][action]["arguments"]["artifact_visibility"], "internal")

    async def test_detached_poll_and_cancel_preserve_visibility_after_json_checkpoint(self):
        value = job()
        rt = runtime(value)
        with patch.object(server, "_require_tool_auth", return_value=rt), \
             patch.object(server, "_make_request", new=AsyncMock(return_value=object())), \
             patch.object(server.JobRecord, "new", return_value=value):
            submitted = await server.submit_video_generation("original prompt", None, artifact_visibility="internal")
            control = json.loads(json.dumps(submitted["_meta"]["exoharness"]["background_task"]))
            for action, tool in (("poll", server.get_video_job), ("cancel", server.cancel_video_job)):
                result = await tool(ctx=None, **control[action]["arguments"])
                self.assertEqual(result["result"], submitted["result"])
                self.assertEqual(result["_meta"]["exoharness"]["artifact_visibility"], "internal")
            internal = await server.get_video_result(value.id, None, artifact_visibility="internal")
            public = await server.get_video_result(value.id, None)
        self.assertEqual(internal["result"], public["result"])
        self.assertNotIn("_meta", public)
        rt.manager.submit.assert_awaited_once()

    async def test_streamed_generation_still_reports_real_progress(self):
        completed = job()
        rt = runtime(completed)
        async def watch(_):
            yield job(JobStatus.RUNNING)
            yield completed
        rt.manager.watch = watch
        ctx = SimpleNamespace(report_progress=AsyncMock())
        with patch.object(server, "_require_tool_auth", return_value=rt), \
             patch.object(server, "_make_request", new=AsyncMock(return_value=object())), \
             patch.object(server.JobRecord, "new", return_value=completed):
            result = await server.generate_video("prompt", ctx, artifact_visibility="internal")
        self.assertEqual(result["_meta"]["exoharness"]["artifact_visibility"], "internal")
        self.assertEqual([c.kwargs["progress"] for c in ctx.report_progress.await_args_list], [42, 100])
        self.assertIn("download_url", result["result"])

    async def test_live_schema_declares_optional_presentation_for_all_control_paths(self):
        tools = {tool.name: tool for tool in await server.mcp.list_tools()}
        for name in ("generate_video", "submit_video_generation", "get_video_job", "get_video_result", "cancel_video_job"):
            schema = tools[name].input_schema
            self.assertEqual(schema["properties"]["artifact_visibility"]["enum"], ["chat", "internal"])
            self.assertEqual(schema["properties"]["artifact_visibility"]["default"], "chat")
            self.assertNotIn("artifact_visibility", schema.get("required", []))
        with patch.object(server, "_require_tool_auth") as auth:
            with self.assertRaises(Exception):
                await server.mcp.call_tool("submit_video_generation", {"prompt":"do not execute", "artifact_visibility":"unknown"})
            auth.assert_not_called()
