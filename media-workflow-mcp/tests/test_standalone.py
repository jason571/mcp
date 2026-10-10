import asyncio
import copy
import shutil
import subprocess
import tempfile
import unittest
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from media_workflow_mcp.artifacts import ArtifactStore
from media_workflow_mcp.config import Settings
from media_workflow_mcp.standalone import StandalonePlan, StandaloneWorkflow
from media_workflow_mcp import server


SCHEMA = {"type": "object", "additionalProperties": False,
          "required": ["prompt", "duration_seconds", "audio_enabled", "resolution", "aspect_ratio"],
          "properties": {"prompt": {"type": "string"},
                         "duration_seconds": {"type": "number", "minimum": 1, "maximum": 30},
                         "audio_enabled": {"type": "boolean"},
                         "resolution": {"enum": ["512p"]},
                         "aspect_ratio": {"enum": ["16:9"]},
                         "first_frame_url": {"type": "string"}}}


class Generator:
    def __init__(self):
        self.calls = []
        self.fail_submit = False

    async def submission_schema(self):
        return SCHEMA

    async def submit(self, arguments, _request_state):
        self.calls.append(copy.deepcopy(arguments))
        if self.fail_submit:
            raise RuntimeError("transport lost after submission")
        return {"job_id": f"job-{len(self.calls)}", "status": "queued"}

    async def status(self, _job_id, _request_state):
        return {"status": "completed", "progress": 100,
                "result": {"download_url": "https://source.example/clip.mp4"}}

    async def cancel(self, _job_id, _request_state):
        return {"status": "cancelled"}


class StandaloneTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.settings = Settings(host="127.0.0.1", port=8790, mcp_path="/mcp",
            public_base_url="https://media.example", allowed_hosts=(), allowed_origins=(),
            allowed_fetch_origins=("https://source.example", "https://media.example"),
            api_token=None, artifact_signing_key="fixture", artifact_url_ttl_seconds=3600,
            data_dir=root, artifact_dir=root / "artifacts", ffmpeg_binary="ffmpeg",
            ffprobe_binary="ffprobe", max_download_bytes=50000000,
            operation_timeout_seconds=60)
        self.settings.prepare_directories()
        self.clip = root / "clip.mp4"
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i",
                        "color=c=green:s=64x48:r=24:d=1.1", "-f", "lavfi", "-i",
                        "sine=frequency=440:duration=1.1", "-c:v", "libx264", "-c:a", "aac",
                        "-shortest", str(self.clip)], check=True)
        self.artifacts = ArtifactStore(self.settings)
        self.workflow = StandaloneWorkflow(self.settings, self.artifacts)
        self.generator = Generator()
        self.plan = StandalonePlan(title="fixture", target_duration_seconds=2, segments=[
            {"segment_id": f"segment-{i:03d}", "duration_seconds": 1,
             "generation": {"prompt": f"shot {i}", "duration_seconds": 1,
                            "audio_enabled": True, "aspect_ratio": "16:9",
                            "resolution": "512p"}} for i in (1, 2)])

        async def download(_session, _settings, _url, target):
            shutil.copyfile(self.clip, target)
            return target

        patcher = patch("media_workflow_mcp.workflow.download_source", side_effect=download)
        patcher.start()
        self.addCleanup(patcher.stop)

    @asynccontextmanager
    async def connect(self, _settings, _request_headers=None):
        yield self.generator

    async def test_one_plan_chains_frames_preserves_audio_and_replays(self):
        digest = (await self.workflow.prepare(self.plan, self.generator))["plan_digest"]
        result = await self.workflow.run(digest, [], AsyncMock(), self.connect)
        self.assertEqual(result["status"], "completed")
        self.assertTrue(result["result"]["has_audio"])
        self.assertTrue(result["result"]["has_source_audio"])
        self.assertEqual(len(self.generator.calls), 2)
        self.assertNotIn("first_frame_url", self.generator.calls[0])
        first_frame = self.workflow.workflows.load("mv_" + digest)["segments"][0]["last_frame"]
        self.assertIn(first_frame["artifact_id"], self.generator.calls[1]["first_frame_url"])
        reopened = StandaloneWorkflow(self.settings, self.artifacts)
        replay = await reopened.run(digest, [], AsyncMock(), self.connect)
        self.assertEqual(result["result"]["sha256"], replay["result"]["sha256"])
        self.assertEqual(len(self.generator.calls), 2)

    async def test_one_mcp_call_submits_whole_plan_and_returns_final_only(self):
        runtime = SimpleNamespace(settings=self.settings, artifacts=self.artifacts)
        ctx = SimpleNamespace(report_progress=AsyncMock(), headers={})
        with patch.object(server, "_require_tool_auth", return_value=runtime), \
             patch.object(server, "connect_video_generator", self.connect), \
             patch.object(server, "peer_cancellation_check", return_value=lambda: False):
            result = await server.generate_standalone_video(self.plan, ctx)
        self.assertFalse(result.is_error)
        self.assertEqual(result.structured_content["result"]["segment_count"], 2)
        self.assertTrue(result.structured_content["result"]["has_source_audio"])
        self.assertEqual(len(self.generator.calls), 2)
        self.assertEqual(result.meta["exoharness"]["workflow_continuation"]["status"], "complete")

    async def test_schema_rejection_before_any_submit(self):
        bad = self.plan.model_copy(deep=True)
        bad.segments[1].generation["resolution"] = "unsupported"
        with self.assertRaisesRegex(ValueError, "live schema"):
            await self.workflow.prepare(bad, self.generator)
        self.assertEqual(self.generator.calls, [])

    async def test_unknown_submission_is_not_repeated(self):
        digest = (await self.workflow.prepare(self.plan, self.generator))["plan_digest"]
        self.generator.fail_submit = True
        with self.assertRaisesRegex(RuntimeError, "transport lost"):
            await self.workflow.run(digest, [], AsyncMock(), self.connect)
        self.generator.fail_submit = False
        with self.assertRaisesRegex(RuntimeError, "outcome is unknown"):
            await self.workflow.run(digest, [], AsyncMock(), self.connect)
        self.assertEqual(len(self.generator.calls), 1)

    def test_reject_wrong_timeline_and_repeated_reference_frame(self):
        bad = self.plan.model_dump()
        bad["segments"][1]["generation"]["first_frame_url"] = "https://source.example/original.jpg"
        with self.assertRaisesRegex(ValueError, "omit first_frame_url"):
            StandalonePlan.model_validate(bad)
        bad = self.plan.model_dump()
        bad["target_duration_seconds"] = 3
        with self.assertRaisesRegex(ValueError, "must sum"):
            StandalonePlan.model_validate(bad)

    def test_segments_are_generator_calls_not_storyboard_shots(self):
        schema = StandalonePlan.model_json_schema()
        self.assertIn("multiple generator calls", schema["properties"]["segments"]["description"])
        self.assertIn("not one shot", schema["$defs"]["StandaloneSegment"]["properties"]["duration_seconds"]["description"])
        bad = {"title": "five shots in one video", "target_duration_seconds": 15,
               "segments": [{"segment_id": f"shot-{index}", "duration_seconds": 3,
                             "generation": {"prompt": f"shot {index}", "duration_seconds": 5,
                                            "audio_enabled": True}} for index in range(5)]}
        with self.assertRaisesRegex(ValueError, "generation duration must equal segment window"):
            StandalonePlan.model_validate(bad)
