"""Persisted narrated timelines. Business/style guidance belongs to Skills.

The legacy song-MV workflow is intentionally unchanged. Reuse its media/job
primitives, but give narration timelines their own storage and execution rules.
"""
from __future__ import annotations

import asyncio
import fcntl
import hashlib
import json
import math
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import aiohttp
import anyio
from jsonschema import Draft202012Validator
from pydantic import BaseModel, ConfigDict, Field, model_validator

from .media import download_source, extract_last_frame_file, probe_file, validate_source_url
from .limits import MAX_VIDEO_DURATION_SECONDS, reject_overlong_duration
from .plan import MusicVideoPlan, duration_policy_from_schema, _expected_segment_windows
from .plan_store import MusicVideoPlanStore
from .timeline import TimelineSegment, assemble_timeline_files
from .workflow import (MusicVideoWorkflow, TERMINAL_FAILURES, _archive_failed_attempt,
                       _artifact_ref, _result_download_url)
from .workflow_store import MusicVideoWorkflowStore


@asynccontextmanager
async def no_generator(_settings):
    yield None


class NarratedSegment(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    segment_id: str = Field(pattern=r"^[A-Za-z0-9_-]{1,64}$")
    duration_seconds: float = Field(gt=0, le=3600)
    generation: dict[str, Any] = Field(description="Native arguments matching the live video submit schema, including complete prompt and canvas controls. Silent visuals; later first_frame_url is set by this service. In every shot, change place only along a path that shot shows. Do not teleport the subject into or out of a vehicle, across a gap, or off an edge.")
    narration: str = Field(default="", max_length=10000)
    audio_offset_seconds: float = Field(default=0, ge=0)


class NarratedPlan(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    title: str = Field(min_length=1, max_length=300)
    request_key: str | None = Field(default=None, min_length=1, max_length=128,
        description="Use a new unique key only for a newly requested rendition of an identical plan. Preserve it on retries; Resume uses plan_digest. Omit for idempotent reuse of this exact plan.")
    target_duration_seconds: float = Field(
        gt=0,
        le=MAX_VIDEO_DURATION_SECONDS,
        description="Final video duration. Videos longer than 3 minutes are rejected before generation.",
    )
    segments: list[NarratedSegment] = Field(min_length=1, max_length=64)

    @model_validator(mode="before")
    @classmethod
    def duration_limit(cls, value):
        if isinstance(value, dict):
            raw = value.get("target_duration_seconds")
            if isinstance(raw, (int, float)) and not isinstance(raw, bool):
                reject_overlong_duration(
                    float(raw),
                    maximum_seconds=MAX_VIDEO_DURATION_SECONDS,
                    deliverable="视频",
                )
        return value

    @model_validator(mode="after")
    def timeline(self):
        if len({s.segment_id for s in self.segments}) != len(self.segments):
            raise ValueError("segment_id values must be unique")
        if abs(sum(s.duration_seconds for s in self.segments) - self.target_duration_seconds) > .001:
            raise ValueError("segment windows must sum to target_duration_seconds")
        for index, s in enumerate(self.segments):
            if abs(s.duration_seconds * 24 - round(s.duration_seconds * 24)) > .001:
                raise ValueError(f"{s.segment_id}: window must align to 24 fps")
            if s.generation.get("duration_seconds") != s.duration_seconds or s.generation.get("audio_enabled") is not False:
                raise ValueError(f"{s.segment_id}: generation duration must equal its window and audio_enabled must be false")
            if index and "first_frame_url" in s.generation:
                raise ValueError(f"{s.segment_id}: omit first_frame_url; the service supplies the actual preceding frame")
            if s.audio_offset_seconds >= s.duration_seconds:
                raise ValueError(f"{s.segment_id}: audio offset must be inside its window")
        return self


class NarrationAsset(BaseModel):
    model_config = ConfigDict(extra="forbid")
    segment_id: str
    audio_url: str | None = None
    attachment_id: str | None = Field(default=None, pattern=r"^att_[0-9a-f]{64}$")
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def one_source(self):
        if (self.audio_url is None) == (self.attachment_id is None):
            raise ValueError("supply exactly one of audio_url or attachment_id")
        if self.audio_url and "/v1/run-artifacts/" in self.audio_url:
            raise ValueError(
                "chat download_url (/v1/run-artifacts/) cannot be fetched by the media "
                "service; pass the TTS artifact.attachment_id instead. No video submitted"
            )
        return self


def narration_fetch_url(asset: NarrationAsset, settings) -> str:
    if asset.attachment_id:
        base = (getattr(settings, "asset_base", None) or "http://127.0.0.1:8787").rstrip("/")
        return f"{base}/v1/attachment-assets/{asset.attachment_id}"
    return asset.audio_url


class NarratedStore:
    def __init__(self, root: Path):
        self.root = root / "narrated"
        self.root.mkdir(mode=0o700, parents=True, exist_ok=True)
        self.states = MusicVideoWorkflowStore(self.root)

    def put(self, plan: NarratedPlan):
        payload = json.dumps(plan.model_dump(), ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
        digest = hashlib.sha256(payload).hexdigest()
        MusicVideoPlanStore._atomic_write(self.root / f"{digest}.json", payload)
        return digest

    def get(self, digest):
        MusicVideoPlanStore._digest(digest)
        payload = (self.root / f"{digest}.json").read_bytes()
        if hashlib.sha256(payload).hexdigest() != digest:
            raise ValueError("stored narrated plan digest mismatch")
        return NarratedPlan.model_validate_json(payload)

    @asynccontextmanager
    async def lock(self, digest, progress):
        MusicVideoPlanStore._digest(digest)
        # Cross-process exclusion also covers the two deployed MCP entrances.
        with (self.root / f"{digest}.lock").open("a") as lock:
            while True:
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    await progress(0, "相同计划正在执行，等待现有执行完成")
                    await asyncio.sleep(1)
            try:
                yield
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)


def _assert_segments_match_clip_policy(plan, policy: dict[str, int]) -> None:
    master_ms = int(round(float(plan.target_duration_seconds) * 1000))
    windows = _expected_segment_windows(MusicVideoPlan.model_construct(
        master_duration_ms=master_ms,
        target_segment_duration_seconds=policy["target_segment_duration_seconds"],
        tail_merge_max_duration_seconds=policy["tail_merge_max_duration_seconds"],
        live_min_duration_seconds=policy["live_min_duration_seconds"],
        live_max_duration_seconds=policy["live_max_duration_seconds"],
        duration_step_seconds=policy["duration_step_seconds"],
    ))
    clip = policy["target_segment_duration_seconds"]
    step = policy["duration_step_seconds"]
    if len(plan.segments) != len(windows):
        expected = [
            max(policy["live_min_duration_seconds"], math.ceil((end - start) / 1000 / step) * step)
            for start, end in windows
        ]
        raise ValueError(
            f"live generator default is {clip}s (tail merge max "
            f"{policy['tail_merge_max_duration_seconds']}s); this duration needs "
            f"{len(windows)} segments with durations {expected}, got {len(plan.segments)}. "
            "Do not hardcode 15s or 30s"
        )
    for segment, (start, end) in zip(plan.segments, windows, strict=True):
        expected = max(
            policy["live_min_duration_seconds"],
            math.ceil((end - start) / 1000 / step) * step,
        )
        if abs(float(segment.duration_seconds) - expected) > .001:
            raise ValueError(
                f"{segment.segment_id}: duration must be {expected}s from live default "
                f"{clip}s (tail merge max {policy['tail_merge_max_duration_seconds']}s)"
            )


def validate_generation(plan, schema, settings):
    Draft202012Validator.check_schema(schema)
    validator = Draft202012Validator(schema)
    properties = schema.get("properties", {})
    if len(plan.segments) > 1 and "first_frame_url" not in properties:
        raise ValueError("video service does not support first-frame continuation")
    for index, segment in enumerate(plan.segments):
        arguments = dict(segment.generation)
        if index:
            arguments["first_frame_url"] = settings.public_base_url + "/artifacts/validation-frame.png"
        elif arguments.get("first_frame_url"):
            validate_source_url(arguments["first_frame_url"], settings.allowed_fetch_origins)
        errors = sorted(validator.iter_errors(arguments), key=lambda e: str(e.path))
        if errors:
            # No prompt/URL echo. The full authoring input remains in the journal.
            error = errors[0]
            path = "/".join(map(str, error.path))
            raise ValueError(f"{segment.segment_id}/generation/{path}: rejected by live schema ({error.validator}); correct this field only")


# Conservative Mandarin TTS pace is an authoring budget, not a claim about the
# provider's exact speech rate. ffprobe remains the authority for produced audio.
NARRATION_CHARS_PER_SECOND = 4.0


class NarrationPreflightError(ValueError):
    """All measured narration overruns from one pre-GPU validation pass."""

    def __init__(
        self,
        repairs: list[dict[str, Any]],
        retained_segment_ids: list[str],
    ) -> None:
        self.repairs = repairs
        self.retained_segment_ids = retained_segment_ids
        segments = ", ".join(str(item["segment_id"]) for item in repairs)
        super().__init__(
            f"narration repair is required for {segments}; valid narration audio "
            "was retained and no video was submitted"
        )


class NarratedWorkflow(MusicVideoWorkflow):
    def __init__(self, settings, artifacts):
        self.store = NarratedStore(settings.data_dir)
        super().__init__(settings, artifacts, None, self.store.states, None)

    async def prepare(self, plan, generator):
        schema = await generator.submission_schema()
        reported = None
        live_default = getattr(generator, "live_default_duration", None)
        if live_default is not None:
            reported = await live_default()
        policy = duration_policy_from_schema(
            schema,
            duration_step_seconds=getattr(self.settings, "video_generator_duration_step_seconds", 1),
            fallback_target_seconds=reported,  # None → plan.MCP_DURATION_FALLBACK_SECONDS (15)
            fallback_min_seconds=getattr(self.settings, "video_generator_min_duration_seconds", None),
            fallback_max_seconds=getattr(self.settings, "video_generator_max_duration_seconds", None),
        )
        _assert_segments_match_clip_policy(plan, policy)
        validate_generation(plan, schema, self.settings)
        digest = self.store.put(plan)
        narrations = []
        for s in plan.segments:
            if not s.narration.strip():
                continue
            window = s.duration_seconds - s.audio_offset_seconds
            entry = {"segment_id": s.segment_id, "text": s.narration, "text_chars": len(s.narration),
                     "max_duration_seconds": window,
                     "recommended_max_characters": max(
                         1, math.floor(window * NARRATION_CHARS_PER_SECOND)
                     )}
            # Advisory only: the synthesized audio is the authority. Flag text
            # that a typical TTS pace cannot finish inside the window so the
            # model trims before spending a synthesis and a render attempt.
            if len(s.narration) > entry["recommended_max_characters"]:
                entry["likely_too_long"] = True
            narrations.append(entry)
        return {"plan_digest": digest, "status": "prepared", "segment_count": len(plan.segments),
                "duration_seconds": plan.target_duration_seconds,
                "clip_duration_seconds": policy["target_segment_duration_seconds"],
                "tail_merge_max_seconds": policy["tail_merge_max_duration_seconds"],
                "narration_pace_hint": (
                    f"Typical TTS pace is about {NARRATION_CHARS_PER_SECOND:g} CJK characters (or 2.5 English words) per "
                    "second. Keep each text within max_duration_seconds at that pace; the measured audio duration decides."),
                "narrations": narrations}

    async def run(self, digest, narration, progress, connect, *, retry_failed=False, cancel_requested=lambda: False):
        async with self.store.lock(digest, progress):
            return await self._run(digest, narration, progress, connect, retry_failed, cancel_requested)

    async def _run(self, digest, narration, progress, connect, retry_failed, cancel_requested):
        plan = self.store.get(digest)
        workflow_id = "mv_" + digest
        state = self.workflows.load(workflow_id)
        if state is None:
            state = self._initial_state(workflow_id, digest, "", {"segments": [s.model_dump() for s in plan.segments]})
            self.workflows.save(state)
        self._validate_state(state, digest, "", {"segments": [s.model_dump() for s in plan.segments]})
        if state["status"] == "completed":
            return self._completed_result(state)
        active_job = None
        try:
            await progress(1, "正在检查已保存计划及全部配音，尚未提交视频")
            await self._ensure_media(plan, state, narration)
            # Block ambiguous prior submissions BEFORE any upstream call.
            for record in state["segments"]:
                if record.get("status") == "submitting" and not record.get("job_id"):
                    raise RuntimeError(f"{record['segment_id']}: previous submission outcome is unknown; reconcile the upstream job before retrying. No new job submitted")
            state.update(status="running", error=None)
            self.workflows.save(state)
            connection = connect if any(r.get("clip") is None for r in state["segments"]) else no_generator
            async with connection(self.settings) as generator:
                self.generator = generator
                if generator is not None:
                    validate_generation(plan, await generator.submission_schema(), self.settings)
                for index, segment in enumerate(plan.segments):
                    record = state["segments"][index]
                    if record.get("clip") is None:
                        previous = self._artifact_from_ref(state["segments"][index - 1]["last_frame"], "previous frame") if index else None
                        arguments = dict(segment.generation)
                        if previous is not None:
                            arguments["first_frame_url"] = self.artifacts.signed_url(previous)
                        if record.get("status") in TERMINAL_FAILURES:
                            if not retry_failed:
                                raise RuntimeError(f"{segment.segment_id}: upstream job failed; explicit retry_failed=true is required. Successful segments are retained")
                            _archive_failed_attempt(record, {"status": record["status"]})
                            retry_failed = False
                        if record.get("job_id") is None:
                            record.update(status="submitting", first_frame=_artifact_ref(previous) if previous else None)
                            self.workflows.save(state)
                            submitted = await generator.submit(arguments, f"{workflow_id}:{segment.segment_id}:submit")
                            job_id = submitted.get("job_id") or submitted.get("id")
                            if not isinstance(job_id, str) or not job_id:
                                raise RuntimeError("submission returned no job ID; outcome unknown, do not resubmit")
                            record.update(job_id=job_id, status=str(submitted.get("status") or "queued"))
                            self.workflows.save(state)
                        active_job = record
                        completed = await self._wait_for_job(workflow_id, state, record, index, len(plan.segments), progress)
                        active_job = None
                        clip = await self._capture_clip(state, record, _result_download_url(completed), index, len(plan.segments), progress)
                    else:
                        clip = self._artifact_from_ref(record["clip"], "video clip")
                    if index < len(plan.segments) - 1 and record.get("last_frame") is None:
                        await progress(2 + 88 * (index + 1) / len(plan.segments), f"第 {index + 1} 段完成，固定真实末帧")
                        with tempfile.TemporaryDirectory(dir=self.settings.data_dir) as raw:
                            target = Path(raw) / "frame.png"
                            await extract_last_frame_file(self.settings, Path(clip.path), target)
                            record["last_frame"] = _artifact_ref(self.artifacts.add(target, f"{segment.segment_id}-last-frame.png", "image/png"))
                        self.workflows.save(state)
            await progress(92, "全部片段完成，正在按计划合成画面和解说")
            final, metadata = await self._assemble(plan, state, progress)
            state.update(status="completed", error=None, final=_artifact_ref(final), final_metadata=metadata)
            self.workflows.save(state)
            await progress(100, "完整视频已生成")
            return self._completed_result(state)
        except asyncio.CancelledError:
            explicit = cancel_requested()
            if active_job is not None and explicit:
                receipt = active_job["cancellation"] = {"requested": True, "delivered": False}
                # The MCP connection's context may already have unwound. Open
                # a bounded cleanup connection rather than using a closed one.
                try:
                    with anyio.move_on_after(5, shield=True) as cleanup:
                        async with connect(self.settings) as generator:
                            receipt["response"] = await generator.cancel(active_job["job_id"], workflow_id)
                            receipt["delivered"] = True
                    if cleanup.cancel_called:
                        receipt["error"] = "upstream cancellation timed out"
                except Exception as exc:
                    receipt["error"] = str(exc)
            state.update(status="cancelled" if explicit else "interrupted")
            self.workflows.save(state)
            raise
        except NarrationPreflightError as exc:
            state.update(
                status="needs_narration_repair",
                error=str(exc),
                narration_repairs=exc.repairs,
            )
            self.workflows.save(state)
            raise
        except Exception as exc:
            state.update(status="failed", error=str(exc))
            self.workflows.save(state)
            raise

    async def _ensure_media(self, plan, state, assets):
        await self._ensure_narrations(plan, state, assets)

    async def _assemble(self, plan, state, progress):
        clips = [Path(self._artifact_from_ref(r["clip"], "video").path) for r in state["segments"]]
        audios = [Path(self._artifact_from_ref(r["audio"], "narration").path) if r.get("audio") else None for r in state["segments"]]
        windows = [TimelineSegment(video_url="persisted", duration_seconds=s.duration_seconds,
                   audio_offset_seconds=s.audio_offset_seconds) for s in plan.segments]
        async def assembly_progress(value, message):
            await progress(92 + value * .08, message)
        with tempfile.TemporaryDirectory(dir=self.settings.data_dir) as raw:
            output = Path(raw) / "final-video.mp4"
            metadata = await assemble_timeline_files(self.settings, windows, clips, audios, Path(raw), output, assembly_progress)
            final = self.artifacts.add(output, "final-video.mp4", "video/mp4")
        return final, metadata

    async def _ensure_narrations(self, plan, state, assets):
        supplied = {a.segment_id: a for a in assets}
        expected = {s.segment_id for s in plan.segments if s.narration.strip()}
        if len(supplied) != len(assets) or not set(supplied) <= expected:
            raise ValueError("narration bindings contain duplicate, unknown or silent segment IDs")
        # Preflight every binding before downloading anything or starting GPU work.
        for segment, record in zip(plan.segments, state["segments"]):
            asset = supplied.get(segment.segment_id)
            if asset:
                try:
                    validate_source_url(
                        narration_fetch_url(asset, self.settings),
                        self.settings.allowed_fetch_origins,
                    )
                except ValueError as exc:
                    if asset.attachment_id:
                        raise ValueError(
                            f"{segment.segment_id}: {exc}. Configure MEDIA_MCP_ASSET_BASE "
                            "inside allowed fetch origins. No video submitted"
                        ) from exc
                    raise ValueError(
                        f"{segment.segment_id}: {exc}. Do not pass chat/login download_url; "
                        "pass TTS artifact.attachment_id. No video submitted"
                    ) from exc
            if record.get("audio"):
                cached = self._artifact_from_ref(record["audio"], "narration")
                if asset and cached.sha256 != asset.sha256:
                    raise ValueError(f"{segment.segment_id}: narration is already fixed; cannot silently replace it")
            elif segment.segment_id in expected and not asset:
                raise ValueError(f"{segment.segment_id}: missing narration URL and SHA-256; no video submitted")
        repairs: list[dict[str, Any]] = []
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.settings.operation_timeout_seconds)) as session:
            for segment, record in zip(plan.segments, state["segments"]):
                if segment.segment_id not in expected or record.get("audio"):
                    continue
                asset = supplied[segment.segment_id]
                with tempfile.TemporaryDirectory(dir=self.settings.data_dir) as raw:
                    target = Path(raw) / "narration.audio"
                    await download_source(
                        session, self.settings, narration_fetch_url(asset, self.settings), target
                    )
                    from .media import sha256_file
                    if sha256_file(target) != asset.sha256:
                        raise ValueError(f"{segment.segment_id}: narration SHA-256 mismatch")
                    info = await probe_file(self.settings, target)
                    if not any(s.get("codec_type") == "audio" for s in info["streams"]):
                        raise ValueError(f"{segment.segment_id}: narration has no audio stream")
                    measured = float(info["duration_seconds"])
                    available = segment.duration_seconds - segment.audio_offset_seconds
                    if measured > available + .001:
                        excess = measured - available
                        # Measure every supplied segment in this pass. Returning
                        # one repair at a time made agents repeatedly synthesize
                        # already-valid audio and discover the next overrun late.
                        repairs.append({
                            "segment_id": segment.segment_id,
                            "measured_duration_seconds": round(measured, 3),
                            "max_duration_seconds": round(available, 3),
                            "window_duration_seconds": round(segment.duration_seconds, 3),
                            "audio_offset_seconds": round(segment.audio_offset_seconds, 3),
                            "minimum_cut_seconds": round(excess, 3),
                            "minimum_cut_percent": min(
                                99, math.ceil(excess / measured * 100 + 5)
                            ),
                            "suggested_max_characters": max(
                                1,
                                math.floor(
                                    available * NARRATION_CHARS_PER_SECOND
                                ),
                            ),
                            "source_text": segment.narration,
                        })
                        continue
                    record["audio"] = _artifact_ref(self.artifacts.add(target, f"{segment.segment_id}.audio", "application/octet-stream"))
                self.workflows.save(state)
        if repairs:
            retained = [
                str(record["segment_id"])
                for record in state["segments"]
                if record.get("audio") is not None
            ]
            raise NarrationPreflightError(repairs, retained)
        state.pop("narration_repairs", None)
