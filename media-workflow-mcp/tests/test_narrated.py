import asyncio
import copy
import hashlib
import logging
import shutil
import subprocess
import tempfile
import unittest
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from media_workflow_mcp import server
from media_workflow_mcp.artifacts import ArtifactStore
from media_workflow_mcp.config import Settings
from media_workflow_mcp.narrated import (
    NarratedPlan,
    NarratedWorkflow,
    NarrationAsset,
    NarrationPreflightError,
)

logging.getLogger("asyncio").setLevel(logging.CRITICAL)


SCHEMA = {"type": "object", "additionalProperties": False,
    "required": ["prompt", "duration_seconds", "audio_enabled", "resolution", "aspect_ratio"],
    "properties": {"prompt": {"type": "string"}, "prompt_mode": {"enum": ["h3"]},
        "duration_seconds": {"type": "number", "minimum": .5, "maximum": 30, "default": 1},
        "audio_enabled": {"const": False}, "resolution": {"enum": ["512p"]},
        "aspect_ratio": {"enum": ["16:9"]}, "first_frame_url": {"type": "string"}}}


class Generator:
    def __init__(self):
        self.calls, self.cancelled = [], []
        self.fail_submit = False
        self.fail_job = None
        self.wait_job = None
        self.started = asyncio.Event()

    async def submission_schema(self):
        return SCHEMA

    async def submit(self, arguments, request_state):
        self.calls.append(copy.deepcopy(arguments))
        if self.fail_submit:
            raise RuntimeError("transport lost after submission")
        return {"job_id": f"job-{len(self.calls)}", "status": "queued"}

    async def status(self, job_id, request_state):
        if self.wait_job == job_id:
            self.started.set()
            await asyncio.Event().wait()
        if self.fail_job == job_id:
            return {"status": "failed", "message": "fixture failure"}
        return {"status": "completed", "progress": 100,
                "result": {"download_url": "https://source.example/clip.mp4"}}

    async def cancel(self, job_id, request_state):
        self.cancelled.append(job_id)
        return {"status": "cancelled"}


class NarratedTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.settings = Settings(host="127.0.0.1", port=8790, mcp_path="/mcp",
            public_base_url="https://media.example", allowed_hosts=(), allowed_origins=(),
            allowed_fetch_origins=("https://source.example", "https://media.example",
                                   "http://127.0.0.1:8787"), api_token=None,
            artifact_signing_key="fixture", artifact_url_ttl_seconds=3600,
            data_dir=self.root, artifact_dir=self.root / "artifacts",
            ffmpeg_binary="ffmpeg", ffprobe_binary="ffprobe", max_download_bytes=50000000,
            operation_timeout_seconds=60)
        self.settings.prepare_directories()
        self.clip, self.audio = self.root / "clip.mp4", self.root / "audio.wav"
        self.make_clip(1.1)
        subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i",
                        "sine=frequency=440:duration=0.2", str(self.audio)], check=True)
        self.artifacts = ArtifactStore(self.settings)
        self.workflow = NarratedWorkflow(self.settings, self.artifacts)
        self.generator = Generator()
        self.plan = NarratedPlan(title="Fixture", target_duration_seconds=2, segments=[
            {"segment_id": f"segment-{i:03d}", "duration_seconds": 1,
             "generation": {"prompt": f"Exact prompt {i}", "prompt_mode": "h3", "duration_seconds": 1,
                            "audio_enabled": False, "aspect_ratio": "16:9", "resolution": "512p"},
             "narration": f"Narration {i}"} for i in range(1, 3)])
        self.assets = [NarrationAsset(segment_id=s.segment_id, audio_url="https://source.example/audio.wav",
                                     sha256=hashlib.sha256(self.audio.read_bytes()).hexdigest()) for s in self.plan.segments]
        self.downloads = []
        async def download(session, settings, url, target):
            self.downloads.append(url)
            shutil.copyfile(self.clip if url.endswith("clip.mp4") else self.audio, target)
            return target
        for module in ("media_workflow_mcp.narrated", "media_workflow_mcp.workflow"):
            patcher = patch(module + ".download_source", side_effect=download)
            patcher.start()
            self.addCleanup(patcher.stop)

    def make_clip(self, duration):
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i",
                        f"color=c=green:s=64x48:r=24:d={duration}", "-c:v", "libx264", str(self.clip)], check=True)

    @asynccontextmanager
    async def connect(self, settings, request_headers=None):
        yield self.generator

    async def prepared(self):
        return (await self.workflow.prepare(self.plan, self.generator))["plan_digest"]

    async def run_plan(self, digest, **kwargs):
        return await self.workflow.run(digest, self.assets, AsyncMock(), self.connect, **kwargs)

    async def test_once_complete_replay_uses_no_generator_and_real_frame_handoff(self):
        digest = await self.prepared()
        result = await self.run_plan(digest)
        self.assertEqual(result["status"], "completed")
        self.assertAlmostEqual(float(result["result"]["duration_seconds"]), 2, delta=.05)
        self.assertTrue(result["result"]["has_narration"])
        self.assertEqual(len(self.generator.calls), 2)
        self.assertNotIn("first_frame_url", self.generator.calls[0])
        self.assertEqual(self.generator.calls[1]["prompt_mode"], "h3")
        self.assertEqual(self.generator.calls[1]["prompt"], "Exact prompt 2")
        state = self.workflow.workflows.load("mv_" + digest)
        frame = state["segments"][0]["last_frame"]
        self.assertIn(frame["artifact_id"], self.generator.calls[1]["first_frame_url"])
        self.assertEqual(state["segments"][1]["first_frame"], frame)
        # Reopen persistent state as on a process restart; even an unavailable
        # upstream cannot prevent returning an already completed result.
        reopened = NarratedWorkflow(self.settings, self.artifacts)
        replay = await reopened.run(digest, [], AsyncMock(), lambda _: self.fail("must not reconnect"))
        self.assertEqual(replay["result"]["sha256"], result["result"]["sha256"])
        self.assertEqual(len(self.generator.calls), 2)

    async def test_unknown_submission_cannot_repeat_on_resume(self):
        digest = await self.prepared()
        self.generator.fail_submit = True
        with self.assertRaisesRegex(RuntimeError, "transport lost"):
            await self.run_plan(digest)
        self.generator.fail_submit = False
        for retry in (False, True):
            with self.assertRaisesRegex(RuntimeError, "outcome is unknown"):
                await self.run_plan(digest, retry_failed=retry)
        self.assertEqual(len(self.generator.calls), 1)

    async def test_known_failure_requires_explicit_retry_and_keeps_completed_segment(self):
        digest = await self.prepared()
        self.generator.fail_job = "job-2"
        with self.assertRaises(RuntimeError):
            await self.run_plan(digest)
        original = self.workflow.workflows.load("mv_" + digest)["segments"][0]["clip"]
        with self.assertRaisesRegex(RuntimeError, "retry_failed"):
            await self.run_plan(digest)
        self.assertEqual(len(self.generator.calls), 2)
        result = await self.run_plan(digest, retry_failed=True)
        self.assertEqual(result["status"], "completed")
        self.assertEqual(len(self.generator.calls), 3)
        state = self.workflow.workflows.load("mv_" + digest)
        self.assertEqual(state["segments"][0]["clip"], original)
        self.assertEqual(state["segments"][1]["prior_attempts"][0]["job_id"], "job-2")

    async def test_disconnect_resumes_known_job_without_resubmission(self):
        digest = await self.prepared()
        self.generator.wait_job = "job-1"
        task = asyncio.create_task(self.run_plan(digest))
        await self.generator.started.wait()
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertEqual(self.generator.cancelled, [])
        self.generator.wait_job = None
        self.assertEqual((await self.run_plan(digest))["status"], "completed")
        self.assertEqual(len(self.generator.calls), 2)

    async def test_explicit_stop_sends_child_cancel(self):
        digest = await self.prepared()
        self.generator.wait_job = "job-1"
        task = asyncio.create_task(self.run_plan(digest, cancel_requested=lambda: True))
        await self.generator.started.wait()
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertEqual(self.generator.cancelled, ["job-1"])
        state = self.workflow.workflows.load("mv_" + digest)
        self.assertTrue(state["segments"][0]["cancellation"]["delivered"])

    async def test_attachment_id_starts_generation_and_chat_url_is_rejected(self):
        digest = await self.prepared()
        attachment = "att_" + "ab" * 32
        self.assets = [
            NarrationAsset(
                segment_id=segment.segment_id,
                attachment_id=attachment,
                sha256=self.assets[0].sha256,
            )
            for segment in self.plan.segments
        ]
        result = await self.run_plan(digest)
        self.assertEqual(result["status"], "completed")
        self.assertEqual(len(self.generator.calls), 2)
        expected = f"http://127.0.0.1:8787/v1/attachment-assets/{attachment}"
        self.assertEqual([url for url in self.downloads if "attachment-assets" in url], [expected, expected])
        with self.assertRaisesRegex(ValueError, "attachment_id"):
            NarrationAsset(
                segment_id="s1",
                audio_url="https://agent.clhh.tech/v1/run-artifacts/artifact-deadbeef",
                sha256="0" * 64,
            )
        with self.assertRaisesRegex(ValueError, "exactly one"):
            NarrationAsset(segment_id="s1", sha256="0" * 64)

    async def test_invalid_binding_and_overlong_audio_prevent_all_gpu_work(self):
        digest = await self.prepared()
        self.assets[-1].audio_url = "https://denied.example/secret"
        with self.assertRaisesRegex(ValueError, "not allowed"):
            await self.run_plan(digest)
        self.assertEqual(self.downloads, [])
        self.assets[-1].audio_url = "https://source.example/audio.wav"
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i",
                        "sine=duration=2", str(self.audio)], check=True)
        for a in self.assets:
            a.sha256 = hashlib.sha256(self.audio.read_bytes()).hexdigest()
        with self.assertRaises(NarrationPreflightError) as caught:
            await self.run_plan(digest)
        self.assertEqual(
            [item["segment_id"] for item in caught.exception.repairs],
            ["segment-001", "segment-002"],
        )
        self.assertTrue(
            all(item["suggested_max_characters"] == 4 for item in caught.exception.repairs)
        )
        self.assertEqual(self.generator.calls, [])

    async def test_duration_preflight_retains_valid_audio_and_repairs_only_invalid_segment(self):
        digest = await self.prepared()
        short = self.root / "short.wav"
        long = self.root / "long.wav"
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i",
                        "sine=duration=0.2", str(short)], check=True)
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i",
                        "sine=duration=2", str(long)], check=True)
        assets = [
            NarrationAsset(segment_id="segment-001", audio_url="https://source.example/short.wav",
                           sha256=hashlib.sha256(short.read_bytes()).hexdigest()),
            NarrationAsset(segment_id="segment-002", audio_url="https://source.example/long.wav",
                           sha256=hashlib.sha256(long.read_bytes()).hexdigest()),
        ]

        async def mixed_download(_session, _settings, url, target):
            shutil.copyfile(long if url.endswith("long.wav") else short, target)
            return target

        with patch("media_workflow_mcp.narrated.download_source", side_effect=mixed_download):
            with self.assertRaises(NarrationPreflightError) as caught:
                await self.workflow.run(digest, assets, AsyncMock(), self.connect)
        self.assertEqual([item["segment_id"] for item in caught.exception.repairs], ["segment-002"])
        self.assertEqual(caught.exception.retained_segment_ids, ["segment-001"])
        state = self.workflow.workflows.load("mv_" + digest)
        self.assertIsNotNone(state["segments"][0]["audio"])
        self.assertIsNone(state["segments"][1].get("audio"))
        self.assertEqual(state["status"], "needs_narration_repair")
        self.assertEqual(self.generator.calls, [])

        replacement = NarrationAsset(
            segment_id="segment-002",
            audio_url="https://source.example/short.wav",
            sha256=hashlib.sha256(short.read_bytes()).hexdigest(),
        )
        with patch("media_workflow_mcp.narrated.download_source", side_effect=mixed_download):
            result = await self.workflow.run(
                digest, [replacement], AsyncMock(), self.connect
            )
        self.assertEqual(result["status"], "completed")
        self.assertEqual(len(self.generator.calls), 2)

    async def test_sdk_duration_repair_is_a_non_error_checkpoint(self):
        digest = await self.prepared()
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i",
                        "sine=duration=2", str(self.audio)], check=True)
        digest_value = hashlib.sha256(self.audio.read_bytes()).hexdigest()
        assets = [asset.model_copy(update={"sha256": digest_value}) for asset in self.assets]
        runtime = SimpleNamespace(settings=self.settings, artifacts=self.artifacts)
        ctx = SimpleNamespace(report_progress=AsyncMock(), headers={})
        with patch.object(server, "_require_tool_auth", return_value=runtime), \
             patch.object(server, "connect_video_generator", self.connect), \
             patch.object(server, "peer_cancellation_check", return_value=lambda: False):
            result = await server.render_narrated_video(digest, ctx, assets)
        self.assertFalse(result.is_error)
        self.assertEqual(result.structured_content["status"], "needs_narration_repair")
        self.assertEqual(
            result.structured_content["required_narration_segment_ids"],
            ["segment-001", "segment-002"],
        )
        self.assertEqual(result.structured_content["retained_narration_segment_ids"], [])
        self.assertNotIn("workflow_continuation", result.meta["exoharness"])
        self.assertEqual(self.generator.calls, [])

    async def test_live_schema_and_native_plan_validation_do_not_submit(self):
        self.plan.segments[-1].generation["resolution"] = "1080p"
        with self.assertRaisesRegex(ValueError, "live schema"):
            await self.prepared()
        self.assertEqual(self.generator.calls, [])
        tool = next(t for t in await server.mcp.list_tools() if t.name == "prepare_narrated_video")
        self.assertIn("plan", tool.input_schema["properties"])
        with patch.object(server, "_require_tool_auth") as auth:
            with self.assertRaises(Exception):
                await server.mcp.call_tool("prepare_narrated_video", {"plan": "not a JSON object"})
            auth.assert_not_called()

    async def test_video_duration_limit_is_in_schema_and_rejects_before_gpu(self):
        tool = next(t for t in await server.mcp.list_tools() if t.name == "prepare_narrated_video")
        target = tool.input_schema["$defs"]["NarratedPlan"]["properties"]["target_duration_seconds"]
        self.assertEqual(target["maximum"], 180)
        raw = self.plan.model_dump()
        raw["target_duration_seconds"] = 180 + 1 / 24
        raw["segments"] = [{
            **raw["segments"][0],
            "duration_seconds": 180 + 1 / 24,
            "generation": {**raw["segments"][0]["generation"], "duration_seconds": 180 + 1 / 24},
        }]
        with self.assertRaisesRegex(ValueError, "视频最长支持 3 分钟"):
            NarratedPlan.model_validate(raw)
        self.assertEqual(self.generator.calls, [])

        raw["target_duration_seconds"] = 180
        raw["segments"][0]["duration_seconds"] = 180
        raw["segments"][0]["generation"]["duration_seconds"] = 180
        accepted = NarratedPlan.model_validate(raw)
        self.assertEqual(accepted.target_duration_seconds, 180)

    async def test_concurrent_render_submits_each_segment_once(self):
        digest = await self.prepared()
        results = await asyncio.gather(self.run_plan(digest), self.run_plan(digest))
        self.assertEqual(results[0]["result"]["sha256"], results[1]["result"]["sha256"])
        self.assertEqual(len(self.generator.calls), 2)

    async def test_assembly_failure_reuses_all_media_even_with_upstream_unavailable(self):
        digest = await self.prepared()
        with patch("media_workflow_mcp.narrated.assemble_timeline_files", side_effect=RuntimeError("encoder fixture error")):
            with self.assertRaisesRegex(RuntimeError, "encoder fixture"):
                await self.run_plan(digest)
        self.assertEqual(len(self.generator.calls), 2)
        reopened = NarratedWorkflow(self.settings, self.artifacts)
        result = await reopened.run(digest, [], AsyncMock(), lambda _: self.fail("must not reconnect for assembly"))
        self.assertEqual(result["status"], "completed")
        self.assertEqual(len(self.generator.calls), 2)

    async def test_sdk_success_and_failure_preserve_compact_checkpoint(self):
        digest = await self.prepared()
        runtime = SimpleNamespace(settings=self.settings, artifacts=self.artifacts)
        ctx = SimpleNamespace(report_progress=AsyncMock(), headers={})
        with patch.object(server, "_require_tool_auth", return_value=runtime), \
             patch.object(server, "connect_video_generator", self.connect), \
             patch.object(server, "peer_cancellation_check", return_value=lambda: False):
            failed = await server.render_narrated_video(digest, ctx)
            self.assertTrue(failed.is_error)
            self.assertNotIn("workflow_continuation", failed.meta["exoharness"])
            self.assertNotIn("Exact prompt", str(failed))
            self.generator.fail_submit = True
            ambiguous = await server.render_narrated_video(digest, ctx, self.assets)
            self.assertTrue(ambiguous.is_error)
            self.assertEqual(ambiguous.meta["exoharness"]["workflow_continuation"]["required_arguments"], {"plan_digest": digest})
            # A separate plan tests the successful SDK envelope; do not clear
            # ambiguous production state to make a retry appear safe.
            self.generator.fail_submit = False
            self.plan.title = "separate success fixture"
            digest = await self.prepared()
            success = await server.render_narrated_video(digest, ctx, self.assets)
            self.assertFalse(success.is_error)
            self.assertEqual(success.meta["exoharness"]["workflow_continuation"]["status"], "complete")
            self.assertEqual(success.structured_content["status"], "completed")

    async def test_four_thirty_second_segments_deliver_one_final_video(self):
        self.generator.submission_schema = AsyncMock(return_value={
            **SCHEMA,
            "properties": {**SCHEMA["properties"], "duration_seconds": {
                "type": "number", "minimum": .5, "maximum": 60, "default": 30,
            }},
        })
        self.make_clip(30.25)
        first = self.plan.segments[0].model_dump()
        self.plan = NarratedPlan(title="120-second acceptance", target_duration_seconds=120,
            segments=[{**copy.deepcopy(first), "segment_id": f"segment-{i:03d}", "duration_seconds": 30,
                       "generation": {**first["generation"], "duration_seconds": 30}} for i in range(1, 5)])
        self.assets = [self.assets[0].model_copy(update={"segment_id": s.segment_id}) for s in self.plan.segments]
        result = await self.run_plan(await self.prepared())
        self.assertEqual(len(self.generator.calls), 4)
        self.assertAlmostEqual(float(result["result"]["duration_seconds"]), 120, delta=.05)
        self.assertTrue(result["result"]["has_narration"])
        self.assertEqual(len(list(self.settings.artifact_dir.glob("*/final-video.mp4"))), 1)

    async def test_silent_plan_needs_no_tts_and_bad_hash_never_submits(self):
        digest = await self.prepared()
        self.assets[0].sha256 = "0" * 64
        with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
            await self.run_plan(digest)
        self.assertEqual(self.generator.calls, [])
        for segment in self.plan.segments:
            segment.narration = ""
        digest = await self.prepared()
        self.assets = []
        result = await self.run_plan(digest)
        self.assertFalse(result["result"]["has_narration"])

    async def test_corrupt_plan_and_bad_timeline_reject_without_upstream(self):
        digest = await self.prepared()
        self.assertEqual(digest, await self.prepared())
        path = self.workflow.store.root / (digest + ".json")
        path.write_bytes(path.read_bytes() + b" ")
        with self.assertRaisesRegex(ValueError, "digest mismatch"):
            await self.run_plan(digest)
        self.assertEqual(self.generator.calls, [])
        raw = self.plan.model_dump()
        raw["target_duration_seconds"] = 120
        with self.assertRaisesRegex(ValueError, "sum"):
            NarratedPlan.model_validate(raw)

    async def test_prepare_returns_live_clip_policy_and_rejects_hardcoded_30(self):
        prepared = await self.workflow.prepare(self.plan, self.generator)
        self.assertEqual(prepared["clip_duration_seconds"], 1)
        self.assertEqual(prepared["tail_merge_max_seconds"], 1)
        self.assertEqual(
            [item["recommended_max_characters"] for item in prepared["narrations"]],
            [4, 4],
        )
        self.assertTrue(all(item["likely_too_long"] for item in prepared["narrations"]))
        self.generator.submission_schema = AsyncMock(return_value={
            **SCHEMA,
            "properties": {**SCHEMA["properties"], "duration_seconds": {
                "type": "number", "minimum": 5, "maximum": 60, "default": 15,
            }},
        })
        long = NarratedPlan(title="Yongjia", target_duration_seconds=120, segments=[
            {"segment_id": f"segment-{i:03d}", "duration_seconds": 30,
             "generation": {"prompt": f"Exact prompt {i}", "prompt_mode": "h3",
                            "duration_seconds": 30, "audio_enabled": False,
                            "aspect_ratio": "16:9", "resolution": "512p"},
             "narration": f"Narration {i}"} for i in range(1, 5)])
        with self.assertRaisesRegex(ValueError, "live generator default is 15s"):
            await self.workflow.prepare(long, self.generator)

    async def test_explicit_new_rendition_has_separate_identity_without_changing_old_plan(self):
        first = await self.prepared()
        self.plan.request_key = "explicit-second-rendition"
        second = await self.prepared()
        self.assertNotEqual(first, second)
        self.assertIsNone(self.workflow.store.get(first).request_key)
        self.assertEqual(second, await self.prepared())
