import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from media_workflow_mcp import server


class FramePresentationTests(unittest.IsolatedAsyncioTestCase):
    async def test_internal_frame_keeps_url_and_progress_and_default_is_unchanged(self):
        with tempfile.TemporaryDirectory() as directory:
            runtime = SimpleNamespace(settings=SimpleNamespace(data_dir=Path(directory), operation_timeout_seconds=5),
                                      artifacts=SimpleNamespace(add=Mock(return_value=object())))
            ctx = SimpleNamespace(report_progress=AsyncMock())
            artifact = {"filename":"last-frame.png", "content_type":"image/png", "download_url":"https://media.test/frame.png?signature=fixture"}
            with patch.object(server, "_require_tool_auth", return_value=runtime), \
                 patch.object(server, "download_source", new=AsyncMock(return_value=Path(directory)/"source.mp4")), \
                 patch.object(server, "extract_last_frame_file", new=AsyncMock(return_value={"width":896,"height":512})), \
                 patch.object(server, "_public_artifact", return_value=artifact):
                public = await server.extract_last_frame("https://media.test/source.mp4", ctx)
                ctx.report_progress.reset_mock()
                internal = await server.extract_last_frame("https://media.test/source.mp4", ctx, artifact_visibility="internal")
            self.assertNotIn("_meta", public)
            self.assertEqual(internal["result"], public["result"])
            self.assertEqual(internal["_meta"]["exoharness"]["artifact_visibility"], "internal")
            self.assertEqual([call.kwargs["progress"] for call in ctx.report_progress.await_args_list], [5,55,100])
            self.assertEqual(runtime.artifacts.add.call_count, 2, "Both choices persist the frame normally")

    async def test_live_schema_rejects_invalid_visibility_before_reading_media(self):
        tool = next(t for t in await server.mcp.list_tools() if t.name == "extract_last_frame")
        self.assertEqual(tool.input_schema["properties"]["artifact_visibility"]["enum"], ["chat","internal"])
        self.assertEqual(tool.input_schema["properties"]["artifact_visibility"]["default"], "chat")
        with patch.object(server, "_require_tool_auth") as auth:
            with self.assertRaises(Exception):
                await server.mcp.call_tool("extract_last_frame", {"video_url":"https://media.test/a.mp4", "artifact_visibility":"unknown"})
            auth.assert_not_called()
