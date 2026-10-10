from __future__ import annotations

import sys
import tempfile
import unittest
import asyncio
import hashlib
import shutil
import subprocess
from pathlib import Path

from aiohttp import web

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(PROJECT / "tests"))

from media_workflow_mcp.plan import MusicVideoPlan, validate_and_render_music_video_plan
from media_workflow_mcp.plan_store import MusicVideoPlanStore
from media_workflow_mcp.artifacts import ArtifactStore
from media_workflow_mcp.config import Settings
from media_workflow_mcp.workflow import (
    MusicVideoWorkflow,
    _archive_failed_attempt,
    _artifact_ref,
    build_generation_arguments,
    workflow_id_for,
)
from media_workflow_mcp.workflow_store import MusicVideoWorkflowStore
from media_workflow_mcp.upstream import VideoGeneratorClient
from media_workflow_mcp.media import bind_reference_image, reference_image_fetch_url
from test_plan import valid_plan


class MusicVideoWorkflowTests(unittest.TestCase):
    def test_uploaded_reference_is_not_reclassified_as_opening_frame(self) -> None:
        attachment_id = "att_" + "ab" * 32
        source = bind_reference_image(
            attachment_id=attachment_id, url=None, allowed_origins=()
        )
        self.assertEqual(source, {"reference_image_attachment_id": attachment_id})
        self.assertEqual(
            reference_image_fetch_url(source, type("Settings", (), {"asset_base": "https://assets.example"})()),
            f"https://assets.example/v1/attachment-assets/{attachment_id}",
        )
        self.assertNotIn("first_frame_attachment_id", source)

    def test_terminal_attempt_is_archived_before_explicit_retry(self) -> None:
        record = {
            "segment_id": "segment-002",
            "status": "failed",
            "job_id": "job-old",
            "submitted_arguments_sha256": "a" * 64,
            "upstream_progress": 42,
            "upstream_stage": "loading_model",
            "upstream_message": "backend exited",
            "clip": None,
        }
        _archive_failed_attempt(
            record,
            {"status": "failed", "error": "backend unavailable"},
        )
        self.assertIsNone(record["job_id"])
        self.assertEqual(record["status"], "pending")
        self.assertNotIn("upstream_progress", record)
        self.assertEqual(
            record["prior_attempts"],
            [
                {
                    "job_id": "job-old",
                    "status": "failed",
                    "error": "backend unavailable",
                    "submitted_arguments_sha256": "a" * 64,
                }
            ],
        )

        record["job_id"] = "job-cancelled"
        record["status"] = "cancelled"
        _archive_failed_attempt(
            record,
            {"status": "cancelled", "message": "explicitly stopped"},
        )
        self.assertIsNone(record["job_id"])
        self.assertEqual(record["status"], "pending")
        self.assertEqual(record["prior_attempts"][-1]["status"], "cancelled")

    def test_upstream_call_does_not_invent_mcp_request_state(self) -> None:
        class FakeSession:
            def __init__(self):
                self.kwargs: dict[str, object] = {}

            async def call_tool(self, _name, _arguments, **kwargs):
                from mcp import types

                self.kwargs = kwargs
                return types.CallToolResult(
                    content=[types.TextContent(type="text", text='{"status":"ok"}')],
                    structuredContent={"status": "ok"},
                    isError=False,
                )

        async def run() -> None:
            from types import SimpleNamespace

            session = FakeSession()
            settings = SimpleNamespace(
                operation_timeout_seconds=60,
                video_generator_submit_tool="submit",
                video_generator_status_tool="get_video_job",
                video_generator_result_tool="get_video_result",
                video_generator_cancel_tool="cancel_video_job",
            )
            client = VideoGeneratorClient(settings, session)  # type: ignore[arg-type]
            result = await client.call("submit", {"prompt": "test"}, "stable-local-id")
            self.assertEqual(result["status"], "ok")
            self.assertNotIn("request_state", session.kwargs)

        asyncio.run(run())

    def test_next_segment_receives_exact_previous_frame_url(self) -> None:
        rendered = validate_and_render_music_video_plan(
            MusicVideoPlan.model_validate(valid_plan())
        )
        segment = rendered["segments"][1]
        frame_url = (
            "http://127.0.0.1:8790/artifacts/"
            "media_0123456789abcdef0123456789abcdef/segment-001-last-frame.png"
            "?expires=123&signature=abc"
        )
        arguments = build_generation_arguments(segment, frame_url)
        self.assertEqual(arguments["first_frame_url"], frame_url)
        self.assertFalse(arguments["audio_enabled"])
        self.assertEqual(arguments["duration_seconds"], 30)

    def test_first_segment_omits_first_frame_when_none_was_supplied(self) -> None:
        rendered = validate_and_render_music_video_plan(
            MusicVideoPlan.model_validate(valid_plan())
        )
        arguments = build_generation_arguments(rendered["segments"][0], None)
        self.assertNotIn("first_frame_url", arguments)
        self.assertNotIn("<Picture 1>", arguments["prompt"])

    def test_first_segment_image_to_video_uses_bound_frame_and_prefix(self) -> None:
        rendered = validate_and_render_music_video_plan(
            MusicVideoPlan.model_validate(valid_plan())
        )
        frame_url = "http://127.0.0.1:8787/v1/attachment-assets/" + "att_" + "ab" * 32
        arguments = build_generation_arguments(rendered["segments"][0], frame_url)
        self.assertEqual(arguments["first_frame_url"], frame_url)
        self.assertTrue(arguments["prompt"].startswith(
            "For the target video, at 0.00 seconds into the target video, <Picture 1>"
        ))
        with_frame = validate_and_render_music_video_plan(
            MusicVideoPlan.model_validate(valid_plan()),
            opening_frame={"first_frame_attachment_id": "att_" + "ab" * 32},
        )
        self.assertNotEqual(rendered["plan_digest"], with_frame["plan_digest"])
        self.assertEqual(with_frame["first_frame_attachment_id"], "att_" + "ab" * 32)

    def test_persistent_reference_is_distinct_from_first_frame_for_each_segment(self) -> None:
        rendered = validate_and_render_music_video_plan(
            MusicVideoPlan.model_validate(valid_plan()),
            opening_frame={"reference_image_attachment_id": "att_" + "ab" * 32},
        )
        self.assertEqual(rendered["reference_image_attachment_id"], "att_" + "ab" * 32)
        reference_url = "https://media.example/reference.png"
        first = build_generation_arguments(rendered["segments"][0], None, reference_url)
        second = build_generation_arguments(
            rendered["segments"][1], "https://media.example/previous-last.png", reference_url
        )
        self.assertEqual(first["reference_image_url"], reference_url)
        self.assertNotIn("first_frame_url", first)
        self.assertEqual(second["reference_image_url"], reference_url)
        self.assertIn("first_frame_url", second)
        self.assertFalse(second["prompt"].startswith("For the target video"))

    def test_first_segment_submits_ingested_user_still(self) -> None:
        jpeg = b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00\xff\xd9"

        class FailingAfterSubmit:
            def __init__(self) -> None:
                self.calls: list[dict[str, object]] = []

            async def submit(self, arguments, request_state):
                self.calls.append(dict(arguments))
                return {"id": "job-1", "status": "queued"}

            async def status(self, job_id, request_state):
                return {"id": job_id, "status": "failed", "error": "stop after submit"}

            async def result(self, job_id, request_state):
                raise AssertionError("result must not be requested")

            async def cancel(self, job_id, request_state):
                return {"id": job_id, "status": "cancelled"}

        async def run() -> None:
            with tempfile.TemporaryDirectory() as raw_dir:
                root = Path(raw_dir)
                jpeg_path = root / "still.jpg"
                jpeg_path.write_bytes(jpeg)
                app = web.Application()
                app.router.add_get("/still.jpg", lambda _request: web.FileResponse(jpeg_path))
                runner = web.AppRunner(app)
                await runner.setup()
                site = web.TCPSite(runner, "127.0.0.1", 0)
                await site.start()
                port = site._server.sockets[0].getsockname()[1]  # type: ignore[union-attr]
                origin = f"http://127.0.0.1:{port}"
                try:
                    settings = Settings(
                        host="127.0.0.1",
                        port=8790,
                        mcp_path="/mcp",
                        public_base_url="http://127.0.0.1:8790",
                        allowed_hosts=(),
                        allowed_origins=(),
                        allowed_fetch_origins=(origin,),
                        api_token=None,
                        artifact_signing_key="test-key",
                        artifact_url_ttl_seconds=3600,
                        data_dir=root / "data",
                        artifact_dir=root / "data" / "artifacts",
                        ffmpeg_binary=shutil.which("ffmpeg") or "ffmpeg",
                        ffprobe_binary=shutil.which("ffprobe") or "ffprobe",
                        max_download_bytes=1024 * 1024,
                        operation_timeout_seconds=60,
                    )
                    settings.prepare_directories()
                    plans = MusicVideoPlanStore(settings.data_dir)
                    rendered = validate_and_render_music_video_plan(
                        MusicVideoPlan.model_validate(valid_plan(30_000)),
                        opening_frame={"first_frame_url": f"{origin}/still.jpg"},
                    )
                    reference = plans.put(rendered)
                    audio_bytes = b"immutable-audio"
                    audio_sha = hashlib.sha256(audio_bytes).hexdigest()
                    audio_source = root / "audio.mp3"
                    audio_source.write_bytes(audio_bytes)
                    artifacts = ArtifactStore(settings)
                    audio = artifacts.add(audio_source, "master-audio.mp3", "audio/mpeg")
                    store = MusicVideoWorkflowStore(settings.data_dir)
                    workflow_id = workflow_id_for(str(reference["plan_digest"]), audio_sha)
                    state = MusicVideoWorkflow._initial_state(
                        workflow_id,
                        str(reference["plan_digest"]),
                        audio_sha,
                        rendered,
                    )
                    state["audio"] = _artifact_ref(audio)
                    store.save(state)
                    fake = FailingAfterSubmit()
                    workflow = MusicVideoWorkflow(
                        settings, artifacts, plans, store, fake  # type: ignore[arg-type]
                    )

                    async def progress(_value: float, _message: str) -> None:
                        return None

                    with self.assertRaisesRegex(RuntimeError, "stop after submit"):
                        await workflow.run(
                            str(reference["plan_digest"]),
                            f"{origin}/unused-audio.mp3",
                            audio_sha,
                            progress,
                        )
                    self.assertEqual(len(fake.calls), 1)
                    self.assertIn("first_frame_url", fake.calls[0])
                    self.assertTrue(
                        str(fake.calls[0]["first_frame_url"]).startswith(
                            "http://127.0.0.1:8790/artifacts/"
                        )
                    )
                    self.assertIn("<Picture 1>", str(fake.calls[0]["prompt"]))
                    saved = store.load(workflow_id)
                    self.assertIsNotNone(saved["opening_frame"])
                    self.assertIsNotNone(saved["segments"][0]["first_frame"])
                finally:
                    await runner.cleanup()

        asyncio.run(run())

    def test_reference_image_is_ingested_and_submitted_without_forcing_first_frame(self) -> None:
        jpeg = b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00\xff\xd9"

        class FailingAfterSubmit:
            def __init__(self) -> None:
                self.calls: list[dict[str, object]] = []

            async def submission_schema(self):
                return {"properties": {"reference_image_url": {"type": "string"}}}

            async def submit(self, arguments, request_state):
                self.calls.append(dict(arguments))
                return {"id": "job-reference", "status": "queued"}

            async def status(self, job_id, request_state):
                return {"id": job_id, "status": "failed", "error": "stop after submit"}

        async def run() -> None:
            with tempfile.TemporaryDirectory() as raw_dir:
                root = Path(raw_dir)
                image_path = root / "reference.jpg"
                image_path.write_bytes(jpeg)
                app = web.Application()
                app.router.add_get("/reference.jpg", lambda _request: web.FileResponse(image_path))
                runner = web.AppRunner(app)
                await runner.setup()
                site = web.TCPSite(runner, "127.0.0.1", 0)
                await site.start()
                port = site._server.sockets[0].getsockname()[1]  # type: ignore[union-attr]
                origin = f"http://127.0.0.1:{port}"
                try:
                    settings = Settings(
                        host="127.0.0.1", port=8790, mcp_path="/mcp",
                        public_base_url="http://127.0.0.1:8790",
                        allowed_hosts=(), allowed_origins=(),
                        allowed_fetch_origins=(origin,), api_token=None,
                        artifact_signing_key="test-key", artifact_url_ttl_seconds=3600,
                        data_dir=root / "data", artifact_dir=root / "data" / "artifacts",
                        ffmpeg_binary=shutil.which("ffmpeg") or "ffmpeg",
                        ffprobe_binary=shutil.which("ffprobe") or "ffprobe",
                        max_download_bytes=1024 * 1024, operation_timeout_seconds=60,
                    )
                    settings.prepare_directories()
                    plans = MusicVideoPlanStore(settings.data_dir)
                    rendered = validate_and_render_music_video_plan(
                        MusicVideoPlan.model_validate(valid_plan(30_000)),
                        opening_frame={"reference_image_url": f"{origin}/reference.jpg"},
                    )
                    plan_ref = plans.put(rendered)
                    audio_bytes = b"immutable-audio"
                    audio_sha = hashlib.sha256(audio_bytes).hexdigest()
                    audio_source = root / "audio.mp3"
                    audio_source.write_bytes(audio_bytes)
                    artifacts = ArtifactStore(settings)
                    audio = artifacts.add(audio_source, "master-audio.mp3", "audio/mpeg")
                    store = MusicVideoWorkflowStore(settings.data_dir)
                    workflow_id = workflow_id_for(str(plan_ref["plan_digest"]), audio_sha)
                    state = MusicVideoWorkflow._initial_state(
                        workflow_id, str(plan_ref["plan_digest"]), audio_sha, rendered,
                    )
                    state["audio"] = _artifact_ref(audio)
                    store.save(state)
                    fake = FailingAfterSubmit()
                    workflow = MusicVideoWorkflow(settings, artifacts, plans, store, fake)  # type: ignore[arg-type]

                    async def progress(_value: float, _message: str) -> None:
                        return None

                    with self.assertRaisesRegex(RuntimeError, "stop after submit"):
                        await workflow.run(
                            str(plan_ref["plan_digest"]), f"{origin}/unused-audio.mp3",
                            audio_sha, progress,
                        )
                    self.assertEqual(len(fake.calls), 1)
                    self.assertNotIn("first_frame_url", fake.calls[0])
                    self.assertNotIn("<Picture 1>", str(fake.calls[0]["prompt"]))
                    self.assertTrue(str(fake.calls[0]["reference_image_url"]).startswith(
                        "http://127.0.0.1:8790/artifacts/"
                    ))
                    saved = store.load(workflow_id)
                    self.assertIsNone(saved.get("opening_frame"))
                    self.assertIsNotNone(saved["reference_image"])
                finally:
                    await runner.cleanup()

        asyncio.run(run())

    def test_workflow_state_is_atomic_and_reopenable(self) -> None:
        with tempfile.TemporaryDirectory() as raw_dir:
            root = Path(raw_dir)
            plans = MusicVideoPlanStore(root)
            reference = plans.put(
                validate_and_render_music_video_plan(
                    MusicVideoPlan.model_validate(valid_plan())
                )
            )
            audio_sha = "b" * 64
            workflow_id = workflow_id_for(str(reference["plan_digest"]), audio_sha)
            state = {
                "workflow_id": workflow_id,
                "plan_digest": reference["plan_digest"],
                "audio_sha256": audio_sha,
                "status": "running",
                "segments": [],
            }
            store = MusicVideoWorkflowStore(root)
            store.save(state)
            self.assertEqual(store.load(workflow_id), state)
            mode = (root / "workflows" / f"{workflow_id}.json").stat().st_mode & 0o777
            self.assertEqual(mode, 0o600)

    def test_caller_disconnect_preserves_submitted_job_for_resume(self) -> None:
        self._check_caller_cancellation(explicit_stop=False)

    def test_explicit_stop_cancels_only_the_submitted_child_job(self) -> None:
        self._check_caller_cancellation(explicit_stop=True)

    def test_child_cancellation_failure_is_persisted_not_hidden(self) -> None:
        self._check_caller_cancellation(explicit_stop=True, fail_cancel=True)

    def test_explicit_stop_cleanup_survives_cancelled_sdk_scope(self) -> None:
        self._check_caller_cancellation(explicit_stop=True, sdk_scope=True)

    def _check_caller_cancellation(self, explicit_stop: bool, fail_cancel: bool = False, sdk_scope: bool = False) -> None:
        import anyio
        class WaitingGenerator:
            def __init__(self) -> None:
                self.started = asyncio.Event()
                self.cancel_calls: list[str] = []

            async def submit(self, arguments, request_state):
                return {"id": "job-live", "status": "queued"}

            async def status(self, job_id, request_state):
                self.started.set()
                await asyncio.Event().wait()

            async def result(self, job_id, request_state):
                raise AssertionError("result must not be requested")

            async def cancel(self, job_id, request_state):
                self.cancel_calls.append(job_id)
                if fail_cancel:
                    raise RuntimeError("fixture child cancellation failed")
                return {"id": job_id, "status": "cancelled"}

        async def run() -> None:
            with tempfile.TemporaryDirectory() as raw_dir:
                root = Path(raw_dir)
                settings = Settings(
                    host="127.0.0.1",
                    port=8790,
                    mcp_path="/mcp",
                    public_base_url="http://127.0.0.1:8790",
                    allowed_hosts=(),
                    allowed_origins=(),
                    allowed_fetch_origins=(),
                    api_token=None,
                    artifact_signing_key="test-key",
                    artifact_url_ttl_seconds=3600,
                    data_dir=root / "data",
                    artifact_dir=root / "data" / "artifacts",
                    ffmpeg_binary=shutil.which("ffmpeg") or "ffmpeg",
                    ffprobe_binary=shutil.which("ffprobe") or "ffprobe",
                    max_download_bytes=1024,
                    operation_timeout_seconds=60,
                )
                settings.prepare_directories()
                plans = MusicVideoPlanStore(settings.data_dir)
                rendered = validate_and_render_music_video_plan(
                    MusicVideoPlan.model_validate(valid_plan(30_000))
                )
                reference = plans.put(rendered)
                audio_bytes = b"immutable-audio"
                audio_sha = hashlib.sha256(audio_bytes).hexdigest()
                audio_source = root / "audio.mp3"
                audio_source.write_bytes(audio_bytes)
                artifacts = ArtifactStore(settings)
                audio = artifacts.add(audio_source, "master-audio.mp3", "audio/mpeg")
                store = MusicVideoWorkflowStore(settings.data_dir)
                workflow_id = workflow_id_for(str(reference["plan_digest"]), audio_sha)
                state = MusicVideoWorkflow._initial_state(
                    workflow_id,
                    str(reference["plan_digest"]),
                    audio_sha,
                    rendered,
                )
                state["audio"] = _artifact_ref(audio)
                store.save(state)
                generator = WaitingGenerator()
                workflow = MusicVideoWorkflow(
                    settings,
                    artifacts,
                    plans,
                    store,
                    generator,  # type: ignore[arg-type]
                )

                async def progress(_value: float, _message: str) -> None:
                    return None

                scopes = []

                async def invoke():
                    with anyio.CancelScope() as scope:
                        scopes.append(scope)
                        return await workflow.run(
                            str(reference["plan_digest"]),
                            "http://127.0.0.1:8788/unused",
                            audio_sha,
                            progress,
                            cancel_requested=lambda: explicit_stop,
                        )

                task = asyncio.create_task(invoke())
                await generator.started.wait()
                if sdk_scope:
                    scopes[0].cancel()
                    await task
                else:
                    task.cancel()
                    with self.assertRaises(asyncio.CancelledError):
                        await task

                saved = store.load(workflow_id)
                self.assertEqual(saved["status"], "cancelled" if explicit_stop else "interrupted")
                self.assertEqual(saved["segments"][0]["status"], "interrupted")
                self.assertEqual(saved["segments"][0]["job_id"], "job-live")
                self.assertEqual(generator.cancel_calls, ["job-live"] if explicit_stop else [])
                if explicit_stop:
                    receipt = saved["segments"][0]["cancellation"]
                    self.assertEqual(receipt["delivered"], not fail_cancel)
                    if fail_cancel:
                        self.assertIn("fixture child cancellation failed", receipt["error"])
                    else:
                        self.assertEqual(receipt["response"]["id"], "job-live")

        asyncio.run(run())

    @unittest.skipUnless(
        shutil.which("ffmpeg") and shutil.which("ffprobe"), "ffmpeg required"
    )
    def test_two_segment_workflow_persists_and_chains_boundary_frame(self) -> None:
        class FakeGenerator:
            def __init__(self, clip_urls: list[str]):
                self.clip_urls = clip_urls
                self.calls: list[dict[str, object]] = []
                self.failed_second_attempt = False

            async def submit(self, arguments, request_state):
                self.calls.append(dict(arguments))
                return {"id": f"job-{len(self.calls)}", "status": "queued"}

            async def status(self, job_id, request_state):
                index = int(job_id.rsplit("-", 1)[1]) - 1
                if index == 1 and not self.failed_second_attempt:
                    self.failed_second_attempt = True
                    return {
                        "id": job_id,
                        "status": "failed",
                        "progress": 42,
                        "stage": "backend_unavailable",
                        "error": "backend exited",
                    }
                return {
                    "id": job_id,
                    "status": "completed",
                    "progress": 100,
                    "stage": "completed",
                    "message": "done",
                    "result": {"download_url": self.clip_urls[min(index, 1)]},
                }

            async def result(self, job_id, request_state):
                return await self.status(job_id, request_state)

            async def cancel(self, job_id, request_state):
                return None

        async def run() -> None:
            with tempfile.TemporaryDirectory() as raw_dir:
                root = Path(raw_dir)
                clip_a = root / "a.mp4"
                clip_b = root / "b.mp4"
                audio = root / "song.mp3"
                ffmpeg = shutil.which("ffmpeg") or "ffmpeg"
                for color, target in (("red", clip_a), ("blue", clip_b)):
                    subprocess.run(
                        [
                            ffmpeg,
                            "-hide_banner",
                            "-loglevel",
                            "error",
                            "-f",
                            "lavfi",
                            "-i",
                            f"color=c={color}:s=320x180:d=0.5",
                            "-an",
                            "-c:v",
                            "libx264",
                            "-pix_fmt",
                            "yuv420p",
                            "-y",
                            str(target),
                        ],
                        check=True,
                    )
                subprocess.run(
                    [
                        ffmpeg,
                        "-hide_banner",
                        "-loglevel",
                        "error",
                        "-f",
                        "lavfi",
                        "-i",
                        "sine=frequency=440:duration=1",
                        "-c:a",
                        "libmp3lame",
                        "-y",
                        str(audio),
                    ],
                    check=True,
                )

                app = web.Application()
                app.router.add_get("/a.mp4", lambda _request: web.FileResponse(clip_a))
                app.router.add_get("/b.mp4", lambda _request: web.FileResponse(clip_b))
                app.router.add_get("/song.mp3", lambda _request: web.FileResponse(audio))
                runner = web.AppRunner(app)
                await runner.setup()
                site = web.TCPSite(runner, "127.0.0.1", 0)
                await site.start()
                sockets = site._server.sockets  # type: ignore[union-attr]
                port = sockets[0].getsockname()[1]
                origin = f"http://127.0.0.1:{port}"
                try:
                    settings = Settings(
                        host="127.0.0.1",
                        port=8790,
                        mcp_path="/mcp",
                        public_base_url="http://127.0.0.1:8790",
                        allowed_hosts=(),
                        allowed_origins=(),
                        allowed_fetch_origins=(origin,),
                        api_token=None,
                        artifact_signing_key="test-key",
                        artifact_url_ttl_seconds=3600,
                        data_dir=root / "data",
                        artifact_dir=root / "data" / "artifacts",
                        ffmpeg_binary=ffmpeg,
                        ffprobe_binary=shutil.which("ffprobe") or "ffprobe",
                        max_download_bytes=32 * 1024 * 1024,
                        operation_timeout_seconds=60,
                        video_generator_poll_seconds=1,
                    )
                    settings.prepare_directories()
                    plans = MusicVideoPlanStore(settings.data_dir)
                    raw_plan = valid_plan(60_000)
                    rendered = validate_and_render_music_video_plan(
                        MusicVideoPlan.model_validate(raw_plan)
                    )
                    reference = plans.put(rendered)
                    fake = FakeGenerator([f"{origin}/a.mp4", f"{origin}/b.mp4"])
                    workflow = MusicVideoWorkflow(
                        settings,
                        ArtifactStore(settings),
                        plans,
                        MusicVideoWorkflowStore(settings.data_dir),
                        fake,  # type: ignore[arg-type]
                    )
                    progress_events: list[tuple[float, str]] = []

                    async def progress(value: float, message: str) -> None:
                        progress_events.append((value, message))

                    audio_sha = hashlib.sha256(audio.read_bytes()).hexdigest()
                    with self.assertRaisesRegex(RuntimeError, "backend exited"):
                        await workflow.run(
                            str(reference["plan_digest"]),
                            f"{origin}/song.mp3",
                            audio_sha,
                            progress,
                        )
                    failed_state = MusicVideoWorkflowStore(settings.data_dir).load(
                        workflow_id_for(str(reference["plan_digest"]), audio_sha)
                    )
                    self.assertIsNotNone(failed_state)
                    self.assertIsNotNone(failed_state["segments"][0]["clip"])
                    self.assertEqual(failed_state["segments"][1]["status"], "failed")
                    self.assertEqual(len(fake.calls), 2)

                    result = await workflow.run(
                        str(reference["plan_digest"]),
                        f"{origin}/song.mp3",
                        audio_sha,
                        progress,
                    )
                    self.assertEqual(result["status"], "completed")
                    self.assertEqual(len(fake.calls), 3)
                    self.assertNotIn("first_frame_url", fake.calls[0])
                    self.assertTrue(
                        str(fake.calls[1]["first_frame_url"]).startswith(
                            "http://127.0.0.1:8790/artifacts/"
                        )
                    )
                    self.assertEqual(
                        result["segments"][1]["first_frame_artifact_id"],
                        result["segments"][0]["last_frame_artifact_id"],
                    )
                    self.assertTrue(result["result"]["audio"])
                    self.assertTrue(progress_events)
                    replayed = await workflow.run(
                        str(reference["plan_digest"]),
                        f"{origin}/song.mp3",
                        audio_sha,
                        progress,
                    )
                    self.assertEqual(replayed["result"]["sha256"], result["result"]["sha256"])
                    self.assertEqual(len(fake.calls), 3)
                finally:
                    await runner.cleanup()

        asyncio.run(run())


if __name__ == "__main__":
    unittest.main()
