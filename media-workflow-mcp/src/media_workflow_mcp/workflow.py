from __future__ import annotations

import asyncio
import hashlib
import re
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any, Awaitable, Callable

import aiohttp
import anyio

from .artifacts import Artifact, ArtifactStore
from .config import Settings
from .media import assemble_music_video_files, download_source, extract_last_frame_file, I2VA_PROMPT_PREFIX, opening_frame_fetch_url, reference_image_fetch_url
from .plan_store import MusicVideoPlanStore
from .upstream import VideoGeneratorClient
from .workflow_store import MusicVideoWorkflowStore


Progress = Callable[[float, str], Awaitable[None]]
SHA256 = re.compile(r"^[0-9a-f]{64}$")
TERMINAL_FAILURES = {"failed", "cancelled", "canceled"}


class TerminalJobError(RuntimeError):
    def __init__(self, job_id: str, status: dict[str, Any]):
        upstream_status = str(status.get("status") or "failed").lower()
        detail = status.get("error") or status.get("message") or upstream_status
        super().__init__(f"video-generator job {job_id} {upstream_status}: {detail}")
        self.job_id = job_id
        self.status = status


def workflow_id_for(plan_digest: str, audio_sha256: str) -> str:
    if not SHA256.fullmatch(plan_digest) or not SHA256.fullmatch(audio_sha256):
        raise ValueError("plan_digest and audio_sha256 must be lowercase SHA-256 values")
    digest = hashlib.sha256(f"{plan_digest}\n{audio_sha256}".encode()).hexdigest()
    return f"mv_{digest}"


def build_generation_arguments(
    segment: dict[str, Any], first_frame_url: str | None,
    reference_image_url: str | None = None,
) -> dict[str, Any]:
    controls = segment.get("controls")
    if not isinstance(controls, dict):
        raise ValueError("stored segment has no generation controls")
    arguments: dict[str, Any] = {
        "prompt": _required_string(segment, "positive_prompt"),
        "negative_prompt": _required_string(segment, "negative_prompt"),
        "duration_seconds": controls.get("duration_seconds"),
        "audio_enabled": controls.get("audio_enabled"),
        "aspect_ratio": controls.get("aspect_ratio"),
        "resolution": controls.get("resolution"),
    }
    if first_frame_url is not None:
        if not first_frame_url.startswith(("http://", "https://")):
            raise ValueError("first-frame artifact URL is not HTTP(S)")
        arguments["first_frame_url"] = first_frame_url
        prompt = str(arguments["prompt"] or "")
        if reference_image_url is None and "<Picture 1>" not in prompt:
            arguments["prompt"] = I2VA_PROMPT_PREFIX + prompt
    if reference_image_url is not None:
        if not reference_image_url.startswith(("http://", "https://")):
            raise ValueError("reference-image artifact URL is not HTTP(S)")
        arguments["reference_image_url"] = reference_image_url
    if arguments["audio_enabled"] is not False:
        raise ValueError("music-video segment must request silent video")
    if not isinstance(arguments["duration_seconds"], (int, float)):
        raise ValueError("segment duration is missing")
    if not isinstance(arguments["aspect_ratio"], str) or not isinstance(
        arguments["resolution"], str
    ):
        raise ValueError("segment canvas controls are missing")
    return arguments


class MusicVideoWorkflow:
    def __init__(
        self,
        settings: Settings,
        artifacts: ArtifactStore,
        plans: MusicVideoPlanStore,
        workflows: MusicVideoWorkflowStore,
        generator: VideoGeneratorClient,
    ):
        self.settings = settings
        self.artifacts = artifacts
        self.plans = plans
        self.workflows = workflows
        self.generator = generator

    async def run(
        self,
        plan_digest: str,
        audio_url: str,
        audio_sha256: str,
        progress: Progress,
        *,
        cancel_requested: Callable[[], bool] = lambda: False,
    ) -> dict[str, Any]:
        workflow_id = workflow_id_for(plan_digest, audio_sha256)
        plan = self.plans.get(plan_digest)
        if reference_image_fetch_url(plan, self.settings):
            schema = await self.generator.submission_schema()
            if "reference_image_url" not in schema.get("properties", {}):
                raise ValueError("selected video-generator MCP has no persistent reference_image_url input; no clip was submitted")
        state = self.workflows.load(workflow_id)
        if state is None:
            state = self._initial_state(workflow_id, plan_digest, audio_sha256, plan)
            self.workflows.save(state)
        self._validate_state(state, plan_digest, audio_sha256, plan)

        if state["status"] == "completed":
            return self._completed_result(state)
        retry_one_terminal_attempt = state["status"] in TERMINAL_FAILURES
        prior_workflow_error = state.get("error")
        state["status"] = "running"
        state["error"] = None
        self.workflows.save(state)

        active_job: tuple[dict[str, Any], str] | None = None
        try:
            await progress(1.0, "正在固定主音乐与已验证分镜计划")
            audio = await self._ensure_audio(state, audio_url, audio_sha256)
            opening = await self._ensure_opening_frame(state, plan)
            reference_image = await self._ensure_reference_image(state, plan)
            segments = plan["segments"]
            count = len(segments)
            for index, segment in enumerate(segments):
                record = state["segments"][index]
                if record.get("clip") is None:
                    prior_frame = None
                    if index:
                        prior_frame = self._artifact_from_ref(
                            state["segments"][index - 1].get("last_frame"),
                            "previous segment last frame",
                        )
                    elif opening is not None:
                        prior_frame = opening
                    first_frame_url = (
                        self.artifacts.signed_url(prior_frame)
                        if prior_frame is not None
                        else None
                    )
                    arguments = build_generation_arguments(
                        segment, first_frame_url,
                        self.artifacts.signed_url(reference_image)
                        if reference_image is not None else None,
                    )
                    if (
                        retry_one_terminal_attempt
                        and record.get("job_id") is not None
                        and str(record.get("status") or "").lower()
                        in TERMINAL_FAILURES
                    ):
                        _archive_failed_attempt(
                            record,
                            {
                                "status": record.get("status") or "failed",
                                "error": prior_workflow_error,
                            },
                        )
                        retry_one_terminal_attempt = False
                        self.workflows.save(state)

                    while True:
                        if record.get("job_id") is None:
                            record["status"] = "submitting"
                            record["first_frame"] = (
                                _artifact_ref(prior_frame)
                                if prior_frame is not None
                                else None
                            )
                            self.workflows.save(state)
                            submitted = await self.generator.submit(
                                arguments,
                                f"{workflow_id}:{record['segment_id']}:submit",
                            )
                            job_id = submitted.get("job_id") or submitted.get("id")
                            if not isinstance(job_id, str) or not job_id:
                                raise RuntimeError(
                                    "video-generator submit result has no job ID"
                                )
                            record["job_id"] = job_id
                            record["status"] = str(submitted.get("status") or "queued")
                            record["submitted_arguments_sha256"] = hashlib.sha256(
                                repr(sorted(arguments.items())).encode()
                            ).hexdigest()
                            self.workflows.save(state)
                        job_id = str(record["job_id"])
                        active_job = (record, job_id)
                        try:
                            completed = await self._wait_for_job(
                                workflow_id, state, record, index, count, progress
                            )
                        except TerminalJobError as exc:
                            active_job = None
                            if (
                                retry_one_terminal_attempt
                                and str(exc.status.get("status") or "").lower()
                                in TERMINAL_FAILURES
                            ):
                                _archive_failed_attempt(record, exc.status)
                                retry_one_terminal_attempt = False
                                self.workflows.save(state)
                                continue
                            raise
                        active_job = None
                        break
                    clip_url = _result_download_url(completed)
                    clip = await self._capture_clip(
                        state, record, clip_url, index, count, progress
                    )
                else:
                    clip = self._artifact_from_ref(record["clip"], "video clip")

                if index < count - 1 and record.get("last_frame") is None:
                    await progress(
                        _segment_progress(index, count, 96.0),
                        f"第 {index + 1}/{count} 段已完成，正在提取连续性末帧",
                    )
                    with tempfile.TemporaryDirectory(
                        dir=self.settings.data_dir
                    ) as raw_dir:
                        output = Path(raw_dir) / "last-frame.png"
                        metadata = await extract_last_frame_file(
                            self.settings, Path(clip.path), output
                        )
                        frame = self.artifacts.add(
                            output,
                            f"{record['segment_id']}-last-frame.png",
                            "image/png",
                        )
                    record["last_frame"] = _artifact_ref(frame)
                    record["last_frame_metadata"] = metadata
                    record["status"] = "boundary_ready"
                    self.workflows.save(state)

            await progress(92.0, "所有视频片段已完成，正在合并原始音乐")
            clip_artifacts = [
                self._artifact_from_ref(item.get("clip"), "video clip")
                for item in state["segments"]
            ]
            with tempfile.TemporaryDirectory(dir=self.settings.data_dir) as raw_dir:
                work_dir = Path(raw_dir)
                output = work_dir / "final-mv.mp4"

                async def assembly_progress(value: float, message: str) -> None:
                    await progress(92.0 + value * 0.08, message)

                metadata = await assemble_music_video_files(
                    self.settings,
                    [Path(item.path) for item in clip_artifacts],
                    Path(audio.path),
                    work_dir,
                    output,
                    assembly_progress,
                )
                final = self.artifacts.add(output, "final-mv.mp4", "video/mp4")
            state["final"] = _artifact_ref(final)
            state["final_metadata"] = metadata
            state["status"] = "completed"
            state["error"] = None
            self.workflows.save(state)
            await progress(100.0, "MV 工作流完成")
            return self._completed_result(state)
        except asyncio.CancelledError:
            explicit_stop = cancel_requested()
            if active_job is not None:
                record, job_id = active_job
                record["status"] = "interrupted"
                if explicit_stop:
                    # The MCP dispatcher uses a cancelled AnyIO scope. Shield
                    # bounded cleanup so cancellation reaches our child job.
                    record["cancellation"] = {"requested": True, "delivered": False}
                    self.workflows.save(state)
                    try:
                        with anyio.move_on_after(5, shield=True) as cleanup:
                            response = await self.generator.cancel(job_id, workflow_id)
                            record["cancellation"]["delivered"] = True
                            record["cancellation"]["response"] = response
                        if cleanup.cancel_called:
                            record["cancellation"]["error"] = "upstream cancellation timed out"
                    except Exception as exc:
                        record["cancellation"]["error"] = str(exc)
            state["status"] = "cancelled" if explicit_stop else "interrupted"
            state["error"] = (
                "caller requested stop; inspect child cancellation receipt before resuming"
                if explicit_stop else "caller disconnected; resume will inspect the submitted job"
            )
            self.workflows.save(state)
            raise
        except Exception as exc:
            state["status"] = "failed"
            state["error"] = str(exc)
            self.workflows.save(state)
            raise

    async def _ensure_audio(
        self, state: dict[str, Any], audio_url: str, audio_sha256: str
    ) -> Artifact:
        reference = state.get("audio")
        if reference is not None:
            return self._artifact_from_ref(reference, "master audio")
        with tempfile.TemporaryDirectory(dir=self.settings.data_dir) as raw_dir:
            target = Path(raw_dir) / "master-audio.mp3"
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(
                    total=self.settings.operation_timeout_seconds
                )
            ) as session:
                await download_source(session, self.settings, audio_url, target)
            audio = self.artifacts.add(target, "master-audio.mp3", "audio/mpeg")
        if audio.sha256 != audio_sha256:
            raise ValueError(
                f"master audio SHA-256 mismatch: expected {audio_sha256}, got {audio.sha256}"
            )
        state["audio"] = _artifact_ref(audio)
        self.workflows.save(state)
        return audio

    async def _ensure_opening_frame(
        self, state: dict[str, Any], plan: dict[str, Any]
    ):
        return await self._ensure_visual_asset(
            state, "opening_frame", opening_frame_fetch_url(plan, self.settings),
            "opening-frame",
        )

    async def _ensure_reference_image(
        self, state: dict[str, Any], plan: dict[str, Any]
    ):
        return await self._ensure_visual_asset(
            state, "reference_image", reference_image_fetch_url(plan, self.settings),
            "reference-image",
        )

    async def _ensure_visual_asset(
        self, state: dict[str, Any], state_key: str,
        source: str | None, filename: str,
    ):
        reference = state.get(state_key)
        if reference is not None:
            return self._artifact_from_ref(reference, filename)
        if not source:
            return None
        with tempfile.TemporaryDirectory(dir=self.settings.data_dir) as raw_dir:
            target = Path(raw_dir) / (filename + ".bin")
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(
                    total=self.settings.operation_timeout_seconds
                )
            ) as session:
                await download_source(session, self.settings, source, target)
            content_type = _image_content_type(target)
            suffix = {
                "image/png": ".png",
                "image/jpeg": ".jpg",
                "image/webp": ".webp",
                "image/gif": ".gif",
            }.get(content_type, ".jpg")
            named = target.with_name(filename + suffix)
            target.replace(named)
            frame = self.artifacts.add(named, named.name, content_type)
        state[state_key] = _artifact_ref(frame)
        self.workflows.save(state)
        return frame

    async def _wait_for_job(
        self,
        workflow_id: str,
        state: dict[str, Any],
        record: dict[str, Any],
        index: int,
        count: int,
        progress: Progress,
    ) -> dict[str, Any]:
        job_id = str(record["job_id"])
        while True:
            status = await self.generator.status(
                job_id, f"{workflow_id}:{record['segment_id']}:status"
            )
            upstream_status = str(status.get("status") or "unknown").lower()
            upstream_progress = _bounded_progress(status.get("progress"))
            record["status"] = upstream_status
            record["upstream_progress"] = upstream_progress
            record["upstream_stage"] = status.get("stage")
            record["upstream_message"] = status.get("message")
            self.workflows.save(state)
            mapped = _segment_progress(index, count, upstream_progress)
            stage = status.get("stage") or upstream_status
            message = status.get("message") or "视频生成服务正在执行"
            await progress(
                mapped,
                f"第 {index + 1}/{count} 段 [{stage}] {message}",
            )
            if upstream_status == "completed":
                if _has_result_url(status):
                    return status
                return await self.generator.result(
                    job_id, f"{workflow_id}:{record['segment_id']}:result"
                )
            if upstream_status in TERMINAL_FAILURES:
                raise TerminalJobError(job_id, status)
            await asyncio.sleep(self.settings.video_generator_poll_seconds)

    async def _capture_clip(
        self,
        state: dict[str, Any],
        record: dict[str, Any],
        clip_url: str,
        index: int,
        count: int,
        progress: Progress,
    ) -> Artifact:
        await progress(
            _segment_progress(index, count, 94.0),
            f"第 {index + 1}/{count} 段已生成，正在固定视频 Artifact",
        )
        with tempfile.TemporaryDirectory(dir=self.settings.data_dir) as raw_dir:
            target = Path(raw_dir) / "clip.mp4"
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(
                    total=self.settings.operation_timeout_seconds
                )
            ) as session:
                await download_source(session, self.settings, clip_url, target)
            clip = self.artifacts.add(
                target, f"{record['segment_id']}.mp4", "video/mp4"
            )
        record["clip"] = _artifact_ref(clip)
        record["status"] = "generated"
        self.workflows.save(state)
        return clip

    def _completed_result(self, state: dict[str, Any]) -> dict[str, Any]:
        final = self._artifact_from_ref(state.get("final"), "final MV")
        segments = []
        for item in state["segments"]:
            segments.append(
                {
                    "segment_id": item["segment_id"],
                    "status": item["status"],
                    "job_id": item.get("job_id"),
                    "clip_artifact_id": (item.get("clip") or {}).get("artifact_id"),
                    "first_frame_artifact_id": (item.get("first_frame") or {}).get(
                        "artifact_id"
                    ),
                    "last_frame_artifact_id": (item.get("last_frame") or {}).get(
                        "artifact_id"
                    ),
                }
            )
        return {
            "workflow_id": state["workflow_id"],
            "plan_digest": state["plan_digest"],
            "status": "completed",
            "segments": segments,
            "result": {
                "filename": final.filename,
                "content_type": final.content_type,
                "download_url": self.artifacts.signed_url(final),
                "url_ttl_seconds": self.settings.artifact_url_ttl_seconds,
                "artifact_id": final.artifact_id,
                "size_bytes": final.size_bytes,
                "sha256": final.sha256,
                **state["final_metadata"],
            },
        }

    @staticmethod
    def _initial_state(
        workflow_id: str,
        plan_digest: str,
        audio_sha256: str,
        plan: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "workflow_id": workflow_id,
            "plan_digest": plan_digest,
            "audio_sha256": audio_sha256,
            "status": "pending",
            "error": None,
            "audio": None,
            "reference_image": None,
            "segments": [
                {
                    "segment_id": item["segment_id"],
                    "status": "pending",
                    "job_id": None,
                    "first_frame": None,
                    "clip": None,
                    "last_frame": None,
                }
                for item in plan["segments"]
            ],
            "final": None,
            "final_metadata": None,
        }

    @staticmethod
    def _validate_state(
        state: dict[str, Any],
        plan_digest: str,
        audio_sha256: str,
        plan: dict[str, Any],
    ) -> None:
        expected_ids = [item.get("segment_id") for item in plan["segments"]]
        actual_ids = [item.get("segment_id") for item in state.get("segments", [])]
        if (
            state.get("plan_digest") != plan_digest
            or state.get("audio_sha256") != audio_sha256
            or actual_ids != expected_ids
        ):
            raise ValueError("stored music-video workflow does not match the request")

    def _artifact_from_ref(self, value: object, label: str) -> Artifact:
        if not isinstance(value, dict):
            raise RuntimeError(f"{label} artifact is missing")
        artifact_id = value.get("artifact_id")
        filename = value.get("filename")
        if not isinstance(artifact_id, str) or not isinstance(filename, str):
            raise RuntimeError(f"{label} artifact reference is invalid")
        artifact = self.artifacts.get(artifact_id, filename)
        if artifact is None:
            raise RuntimeError(f"{label} artifact file is unavailable")
        return artifact


def _image_content_type(path: Path) -> str:
    header = path.read_bytes()[:16]
    if header.startswith(b"\x89PNG"):
        return "image/png"
    if header.startswith(b"\xff\xd8"):
        return "image/jpeg"
    if header.startswith(b"GIF8"):
        return "image/gif"
    if header.startswith(b"RIFF") and b"WEBP" in header:
        return "image/webp"
    raise ValueError("opening frame must be a PNG, JPEG, GIF or WebP image")


def _artifact_ref(artifact: Artifact) -> dict[str, Any]:
    return asdict(artifact)


def _archive_failed_attempt(
    record: dict[str, Any], status: dict[str, Any]
) -> None:
    job_id = record.get("job_id")
    if not isinstance(job_id, str) or not job_id:
        raise ValueError("cannot archive a video-generator attempt without a job ID")
    attempts = record.setdefault("prior_attempts", [])
    if not isinstance(attempts, list):
        raise ValueError("stored prior_attempts must be a list")
    attempts.append(
        {
            "job_id": job_id,
            "status": str(status.get("status") or record.get("status") or "failed"),
            "error": status.get("error") or status.get("message"),
            "submitted_arguments_sha256": record.get("submitted_arguments_sha256"),
        }
    )
    record["job_id"] = None
    record["status"] = "pending"
    for key in ("upstream_progress", "upstream_stage", "upstream_message"):
        record.pop(key, None)


def _required_string(value: dict[str, Any], key: str) -> str:
    item = value.get(key)
    if not isinstance(item, str) or not item.strip():
        raise ValueError(f"stored segment has no {key}")
    return item


def _bounded_progress(value: object) -> float:
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(number, 100.0))


def _segment_progress(index: int, count: int, upstream_progress: float) -> float:
    return 2.0 + 88.0 * (index + upstream_progress / 100.0) / count


def _has_result_url(value: dict[str, Any]) -> bool:
    result = value.get("result")
    return isinstance(result, dict) and isinstance(result.get("download_url"), str)


def _result_download_url(value: dict[str, Any]) -> str:
    result = value.get("result")
    url = result.get("download_url") if isinstance(result, dict) else None
    if not isinstance(url, str) or not url.startswith(("http://", "https://")):
        raise RuntimeError("completed video-generator job has no HTTP(S) result URL")
    return url
