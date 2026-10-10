from __future__ import annotations

import asyncio
import hmac
import json
import math
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Literal, Mapping

import aiohttp
from mcp.server import MCPServer
from mcp.server.mcpserver import Context
from mcp.server.mcpserver.exceptions import ToolError
from mcp.server.transport_security import TransportSecuritySettings
from mcp.types import CallToolResult, TextContent
from pydantic import Field
from starlette.requests import Request
from starlette.responses import FileResponse, JSONResponse, Response
from starlette.types import ASGIApp, Receive, Scope, Send

from .artifacts import Artifact, ArtifactStore
from .authoring import (
    CompleteOverview, CompleteSegment, OverviewContent, SegmentContent, checkpoint as authoring_checkpoint,
    SectionWrite, create_draft, drain_sections, overview_errors, queue_sections, write_overview, write_segment,
)
from .config import Settings
from .connections import GENERATOR_CONNECTIONS
from .cancellation import peer_cancellation_check
from .media import (
    validate_source_url,
    analyze_audio_energy,
    assemble_music_video_files,
    bind_opening_frame,
    bind_reference_image,
    reference_image_fetch_url,
    download_source,
    extract_last_frame_file,
    probe_file,
    sha256_file,
)
from .plan import (
    MusicVideoPlan,
    MusicVideoPlanPatch,
    MusicVideoPlanValidationError,
    MusicVideoSegmentPlan,
    apply_music_video_plan_patches,
    missing_segment_repairs,
    normalize_music_video_plan,
    duration_policy_from_schema,
    validate_and_render_music_video_plan as validate_mv_plan,
)
from .plan_store import MusicVideoPlanStore
from .upstream import UpstreamToolError, connect_video_generator
from .workflow import MusicVideoWorkflow, workflow_id_for
from .workflow_store import MusicVideoWorkflowStore
from .timeline import TimelineSegment, assemble_timeline_files
from .narrated import (
    NarratedPlan,
    NarrationAsset,
    NarratedWorkflow,
    NarrationPreflightError,
)
from .narrated_checkpoint import prepared_checkpoint
from .standalone import StandalonePlan, StandaloneWorkflow
from .limits import MAX_MUSIC_VIDEO_DURATION_SECONDS, reject_overlong_duration


ENABLE_AUDIO_ANALYSIS = os.getenv("MEDIA_MCP_ENABLE_AUDIO_ANALYSIS", "false").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}


mcp = MCPServer(
    "Media Workflow",
    version="0.1.0",
    instructions=(
        "Probe immutable media URLs and return their duration, streams, and SHA-256. "
        "extract a completed clip's last frame for continuation, and assemble "
        "ordered video clips with source audio, narration or one master audio track. "
        "A standalone multi-clip plan can be submitted in one call for durable "
        "generation and final audio-preserving assembly. It also validates "
        "and stores complete per-segment music-video prompt plans before generation, "
        "then can execute the validated plan as one resumable workflow: detached "
        "video jobs are polled inside this MCP, boundary frames are passed into the "
        "next segment, and the final silent sequence is muxed with master audio. "
        "Source URLs must "
        "match the configured origin allowlist."
    ),
)


@dataclass(slots=True)
class Runtime:
    settings: Settings
    artifacts: ArtifactStore
    plans: MusicVideoPlanStore
    workflows: MusicVideoWorkflowStore
    workflow_locks: dict[str, asyncio.Lock] = field(default_factory=dict)


_runtime: Runtime | None = None


class ProgressIdleTimeout:
    """Cancel an operation only after it stops reporting progress."""

    def __init__(self, seconds: float) -> None:
        self.seconds = seconds
        self._timeout: asyncio.Timeout | None = None

    async def __aenter__(self) -> "ProgressIdleTimeout":
        self._timeout = asyncio.timeout(self.seconds)
        await self._timeout.__aenter__()
        return self

    async def __aexit__(self, exc_type, exc, traceback) -> bool | None:
        if self._timeout is None:
            raise RuntimeError("progress idle timeout was not entered")
        return await self._timeout.__aexit__(exc_type, exc, traceback)

    def touch(self) -> None:
        if self._timeout is None:
            raise RuntimeError("progress idle timeout was not entered")
        loop = asyncio.get_running_loop()
        self._timeout.reschedule(loop.time() + self.seconds)


def get_runtime() -> Runtime:
    global _runtime
    if _runtime is None:
        settings = Settings.from_env()
        _runtime = Runtime(
            settings=settings,
            artifacts=ArtifactStore(settings),
            plans=MusicVideoPlanStore(settings.data_dir),
            workflows=MusicVideoWorkflowStore(settings.data_dir),
        )
    return _runtime


def _bearer_token(headers: Mapping[str, str]) -> str | None:
    authorization = headers.get("authorization") or headers.get("Authorization")
    if not authorization or not authorization.startswith("Bearer "):
        return None
    return authorization[7:].strip()


def _authorized(headers: Mapping[str, str], expected: str | None) -> bool:
    if expected is None:
        return True
    supplied = _bearer_token(headers)
    return supplied is not None and hmac.compare_digest(supplied, expected)


class MCPBearerAuthMiddleware:
    def __init__(self, app: ASGIApp, api_token: str | None, mcp_path: str) -> None:
        self.app = app
        self.api_token = api_token
        self.mcp_path = mcp_path.rstrip("/") or "/"

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        path = str(scope.get("path", ""))
        protects = self.mcp_path == "/" or path == self.mcp_path or path.startswith(
            f"{self.mcp_path}/"
        )
        if scope["type"] == "http" and protects and not _authorized(
            {
                key.decode("latin-1"): value.decode("latin-1")
                for key, value in scope.get("headers", [])
            },
            self.api_token,
        ):
            response = JSONResponse(
                {"error": "missing or invalid MCP bearer token"},
                status_code=401,
                headers={"WWW-Authenticate": "Bearer"},
            )
            await response(scope, receive, send)
            return
        await self.app(scope, receive, send)


def _require_tool_auth(ctx: Context) -> Runtime:
    runtime = get_runtime()
    if not _authorized(ctx.headers or {}, runtime.settings.api_token):
        raise PermissionError("missing or invalid MCP bearer token")
    return runtime


def _public_artifact(runtime: Runtime, artifact: Artifact) -> dict[str, object]:
    return {
        "filename": artifact.filename,
        "content_type": artifact.content_type,
        "download_url": runtime.artifacts.signed_url(artifact),
        "url_ttl_seconds": runtime.settings.artifact_url_ttl_seconds,
        "artifact_id": artifact.artifact_id,
        "size_bytes": artifact.size_bytes,
        "sha256": artifact.sha256,
    }


SourceUrl = Annotated[
    str,
    Field(description="Complete HTTP(S) URL for an immutable media asset."),
]


async def live_clip_policy(
    settings, request_headers=None, *, required_submit_fields: tuple[str, ...] = ()
) -> dict[str, int]:
    """Ask the video-generator MCP for clip length, then derive windows."""
    try:
        async with connect_video_generator(
            settings, request_headers,
            required_submit_fields=required_submit_fields,
        ) as generator:
            schema = await generator.submission_schema()
            reported = await generator.live_default_duration()
    except ExceptionGroup as exc:
        raise UpstreamToolError(_exception_group_message(exc)) from exc
    return duration_policy_from_schema(
        schema,
        duration_step_seconds=getattr(settings, "video_generator_duration_step_seconds", 1),
        fallback_target_seconds=reported,  # None → plan.MCP_DURATION_FALLBACK_SECONDS (15)
        fallback_min_seconds=getattr(settings, "video_generator_min_duration_seconds", None),
        fallback_max_seconds=getattr(settings, "video_generator_max_duration_seconds", None),
    )


@mcp.tool(meta=GENERATOR_CONNECTIONS)
async def create_music_video_plan(
    audio_url: SourceUrl,
    brief: Annotated[str, Field(min_length=1, max_length=3000)],
    ctx: Context,
    aspect_ratio: Annotated[str, Field(pattern=r"^[1-9][0-9]*:[1-9][0-9]*$")] = "16:9",
    resolution: Annotated[str, Field(min_length=1, max_length=64)] = "512p",
    output_language: Literal["zh", "en"] = "zh",
    shot_mode: Literal["multi-shot", "single-shot"] = "multi-shot",
    shots_per_segment: Annotated[int, Field(ge=1, le=16, description="Writing target: at least four shots in multi-shot mode. Three complete shots are accepted; one or two are rejected. Explicit single-shot mode uses one. The service distributes saved duration across the submitted storyboard.")] = 4,
    planning_batch_size: Annotated[int, Field(ge=1, le=16, description="Maximum creative sections per model response, not concurrent videos. Default writes a typical whole song at once. Use a smaller size when the model has a small output budget.")] = 16,
    first_frame_attachment_id: Annotated[
        str | None,
        Field(
            default=None,
            pattern=r"^att_[0-9a-f]{64}$",
            description="Required when the user uploaded a still and asked to use it as the first-segment image-to-video frame. Pass the chat user_image_attachment attachment_id. Mutually exclusive with first_frame_url. Omit for text-to-video.",
        ),
    ] = None,
    first_frame_url: Annotated[
        str | None,
        Field(
            default=None,
            description="Allowlisted HTTP(S) still for first-segment image-to-video. Do not pass chat /v1/run-artifacts URLs. Prefer first_frame_attachment_id.",
        ),
    ] = None,
    reference_image_attachment_id: Annotated[
        str | None,
        Field(default=None, pattern=r"^att_[0-9a-f]{64}$", description="Uploaded image to retain as a visual reference across all MV segments; it is not a first frame. Mutually exclusive with reference_image_url."),
    ] = None,
    reference_image_url: Annotated[
        str | None,
        Field(default=None, description="Allowlisted HTTP(S) image used as a persistent reference across all segments, not an opening frame. Prefer reference_image_attachment_id for chat uploads."),
    ] = None,
) -> dict[str, object]:
    """Start an MV: measure immutable audio once and persist the exact segment skeleton.

    Does not generate video, analyze audio energy or infer lyrics. Do not construct
    a complete timed plan yourself or call probe_media first. Choose one global
    style before creating this draft, then submit its overview and current sections.
    When the user uploaded a still and asked to use it as the opening reference,
    bind first_frame_attachment_id here so the first segment is image-to-video.
    Clip length is read from the live video-generator duration_seconds.default
    (15 or 30, never assumed). Remainders at most 40% of that default merge into
    the last clip; a larger remainder stays a separate clip. Resolution/aspect
    ratio are set once here; section writes contain creative data only. Default
    planning_batch_size writes the typical whole song at once.
    """
    runtime = _require_tool_auth(ctx)
    try:
        opening = bind_opening_frame(
            attachment_id=first_frame_attachment_id,
            url=first_frame_url,
            allowed_origins=runtime.settings.allowed_fetch_origins,
        )
        reference = bind_reference_image(
            attachment_id=reference_image_attachment_id,
            url=reference_image_url,
            allowed_origins=runtime.settings.allowed_fetch_origins,
        )
    except ValueError as exc:
        raise ToolError(str(exc)) from exc
    try:
        clip_policy = await live_clip_policy(
            runtime.settings, ctx.headers,
            required_submit_fields=(
                "first_frame_url",
                *(("reference_image_url",) if reference else ()),
            ),
        )
        media = await probe_media(audio_url, ctx)
        duration_seconds = float(media["duration_seconds"])
        if not any(stream.get("codec_type") == "audio" for stream in media["streams"]):
            raise ToolError("master media must contain an audio stream")
        reject_overlong_duration(
            duration_seconds,
            maximum_seconds=MAX_MUSIC_VIDEO_DURATION_SECONDS,
            deliverable="MV",
        )
        duration_ms = math.ceil(duration_seconds * 1000)
        draft = create_draft(runtime.settings, duration_ms=duration_ms,
            audio_url=audio_url, audio_sha256=media["sha256"], brief=brief,
            aspect_ratio=aspect_ratio, resolution=resolution, output_language=output_language,
            shot_mode=shot_mode, shots_per_segment=shots_per_segment,
            planning_batch_size=planning_batch_size, opening_frame=opening or None,
            reference_image=reference or None,
            duration_policy=clip_policy)
        return authoring_checkpoint(runtime.plans, draft)
    except ExceptionGroup as exc:
        raise ToolError(_exception_group_message(exc)) from exc
    except (ValueError, UpstreamToolError, RuntimeError, OSError) as exc:
        raise ToolError(str(exc)) from exc


def _authoring_draft(runtime: Runtime, draft_id: str, draft_digest: str) -> dict:
    draft = runtime.plans.get_draft(draft_id, draft_digest)
    if draft.get("authoring_version") != 2:
        raise ToolError("This draft uses the full-plan patch protocol; use its declared next tool.")
    return draft


def _finish_authoring(runtime: Runtime, draft: dict, errors: list[dict]) -> dict:
    if not errors:
        draft, errors = drain_sections(draft)
    if draft["accepted_segments"] < len(draft["plan"]["segments"]):
        return authoring_checkpoint(runtime.plans, draft, errors)
    opening = {
        key: draft[key]
        for key in ("first_frame_attachment_id", "first_frame_url", "reference_image_attachment_id", "reference_image_url")
        if draft.get(key)
    }
    result = _store_validated_or_draft(
        runtime,
        MusicVideoPlan.model_validate(draft["plan"]),
        opening_frame=opening or None,
    )
    if result.get("valid"):
        result["_meta"]["exoharness"]["workflow_continuation"]["required_arguments"].update(
            audio_url=draft["audio_url"], audio_sha256=draft["audio_sha256"])
        # All render inputs are now bound to persisted assets. No old creative
        # proposals or style catalogs are needed to invoke this checkpoint.
        result["_meta"]["exoharness"]["workflow_continuation"]["context_scope"] = "checkpoint"
    return result


@mcp.tool()
async def write_music_video_plan_overview(
    draft_id: str, draft_digest: str, overview: CompleteOverview, ctx: Context,
) -> dict[str, object]:
    """Save the song-wide narrative, one style and continuity anchors before any segment.

    overview is a complete native JSON object. Include every required field in
    the schema. Existing saved draft errors use patch_music_video_plan_overview
    instead of rewriting the overview. No video is generated.
    """
    runtime = _require_tool_auth(ctx)
    try:
        draft, errors = write_overview(_authoring_draft(runtime, draft_id, draft_digest), overview)
        return _finish_authoring(runtime, draft, errors)
    except ValueError as error:
        raise ToolError(str(error)) from error


@mcp.tool()
async def write_music_video_plan_segment(
    draft_id: str, draft_digest: str, segment_id: str, content: CompleteSegment, ctx: Context,
) -> dict[str, object]:
    """Save creative content for only the current segment and return the next section.

    content is a native JSON object, not an encoded string. Author at least four
    shots for multi-shot mode, indexed consecutively from zero. Three are accepted
    without rewriting to reach four; one or two are rejected. Single-shot mode uses one.
    Include every required creative field. On validation errors use
    patch_music_video_plan_segment with only the reported fields. Accepted fields and
    earlier segments stay in MCP Store. Never resend accepted sections. Supply no
    timestamps, duration, segment counts or controls: the skeleton owns them. Each
    shot has an optional relative weight; the service calculates contiguous times.
    After all sections pass validation, the result declares the single rendering call.
    """
    runtime = _require_tool_auth(ctx)
    try:
        draft, errors = write_segment(_authoring_draft(runtime, draft_id, draft_digest), segment_id, content, complete=True)
        return _finish_authoring(runtime, draft, errors)
    except ValueError as error:
        raise ToolError(str(error)) from error


@mcp.tool()
async def write_music_video_plan_sections(
    draft_id: str, draft_digest: str,
    sections: Annotated[list[SectionWrite], Field(min_length=1, max_length=64)], ctx: Context,
) -> dict[str, object]:
    """Save the advertised consecutive sections in one native JSON array.

    Prefer the advertised current window to keep model output small. A longer
    already-authored consecutive batch is accepted up to the plan's remaining
    sections, so valid work is not discarded merely for crossing a window.
    Array order is the advertised timeline order. Each item contains only content;
    do not supply segment_id. IDs are used only by single-section and patch tools.
    Use one common overview and distinct story beats. Storyboard array order is
    shot order, without shot_index. Author at least four shots per multi-shot section;
    three are accepted without a retry, while fewer than three need local repair.
    An explicitly selected single-shot mode uses one continuous shot.
    All submitted sections are retained in MCP
    Store; a validation error requests only the incorrect current field. Later
    saved sections are applied automatically after that repair. Never resend
    accepted or queued sections. This writes a plan only, not video generation.
    """
    runtime = _require_tool_auth(ctx)
    try:
        draft = queue_sections(_authoring_draft(runtime, draft_id, draft_digest), sections)
        return _finish_authoring(runtime, draft, [])
    except ValueError as error:
        raise ToolError(str(error)) from error


@mcp.tool()
async def patch_music_video_plan_overview(
    draft_id: str, draft_digest: str, overview: OverviewContent, ctx: Context,
) -> dict[str, object]:
    """Repair only the reported overview fields in the persisted current draft.

    Omitted fields remain saved. Do not resend the complete overview or segments.
    """
    runtime = _require_tool_auth(ctx)
    saved = _authoring_draft(runtime, draft_id, draft_digest)
    bad = _unreported_authoring_fields(overview.model_dump(exclude_unset=True, exclude_none=True), overview_errors(saved))
    if bad:
        return authoring_checkpoint(runtime.plans, saved, bad)
    draft, errors = write_overview(saved, overview)
    return _finish_authoring(runtime, draft, errors)


@mcp.tool()
async def patch_music_video_plan_segment(
    draft_id: str, draft_digest: str, segment_id: str, content: SegmentContent, ctx: Context,
) -> dict[str, object]:
    """Repair only reported creative fields in the current segment, not earlier work.

    Supply only missing/incorrect fields. For one missing shot field, include only
    its shot_index and that field; all other saved content remains unchanged.
    """
    runtime = _require_tool_auth(ctx)
    saved = _authoring_draft(runtime, draft_id, draft_digest)
    _, current_errors = write_segment(saved, segment_id, SegmentContent())
    bad = _unreported_authoring_fields(content.model_dump(exclude_unset=True, exclude_none=True), current_errors)
    if bad:
        return authoring_checkpoint(runtime.plans, saved, bad)
    draft, errors = write_segment(saved, segment_id, content)
    return _finish_authoring(runtime, draft, errors)


def _unreported_authoring_fields(content: dict, errors: list[dict]) -> list[dict]:
    allowed = [item["path"] for item in errors]
    def paths(value, prefix=""):
        if isinstance(value, dict):
            for key, child in value.items():
                if key != "shot_index":
                    yield from paths(child, prefix+"/"+key)
        elif isinstance(value, list):
            for index, child in enumerate(value):
                slot = child.get("shot_index", index) if isinstance(child, dict) else index
                yield from paths(child, prefix+"/"+str(slot))
        else:
            yield prefix
    return [{"path":path,"code":"unreported_field","message":"Patch only reported fields; all other content is saved and unchanged."}
            for path in paths(content) if not any(path==target or path.startswith(target+"/") for target in allowed)]


@mcp.tool()
async def probe_media(source_url: SourceUrl, ctx: Context) -> dict[str, object]:
    """Return duration, stream metadata, and SHA-256 for an immutable media asset.

    Use this lightweight tool for ordinary music-video planning. It does not decode
    an energy curve or infer lyrics, beats, emotion, genre, or song structure.
    """
    runtime = _require_tool_auth(ctx)
    try:
        with tempfile.TemporaryDirectory(dir=runtime.settings.data_dir) as raw_dir:
            work_dir = Path(raw_dir)
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(
                    total=runtime.settings.operation_timeout_seconds
                )
            ) as session:
                source = await download_source(
                    session, runtime.settings, source_url, work_dir / "source.media"
                )
            result = await probe_file(runtime.settings, source)
            return {
                "duration_seconds": result["duration_seconds"],
                "sha256": sha256_file(source),
                "format": result.get("format", {}),
                "streams": result.get("streams", []),
            }
    except (ValueError, RuntimeError, OSError, aiohttp.ClientError) as exc:
        raise ToolError(str(exc)) from exc


async def analyze_music_timeline(
    audio_url: SourceUrl,
    ctx: Context,
    window_seconds: Annotated[
        float,
        Field(
            ge=0.25,
            le=5.0,
            description="Requested RMS analysis window in seconds.",
        ),
    ] = 1.0,
) -> dict[str, object]:
    """Optionally derive an energy timeline for explicitly audio-reactive editing.

    This expensive optional tool is disabled unless MEDIA_MCP_ENABLE_AUDIO_ANALYSIS
    is true. Never use it for ordinary title-based MV creation or duration probing.
    The full curve is persisted as an artifact instead of entering model context.
    """
    runtime = _require_tool_auth(ctx)
    try:
        await ctx.report_progress(progress=5, total=100, message="正在读取主音乐")
        with tempfile.TemporaryDirectory(dir=runtime.settings.data_dir) as raw_dir:
            work_dir = Path(raw_dir)
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(
                    total=runtime.settings.operation_timeout_seconds
                )
            ) as session:
                audio = await download_source(
                    session, runtime.settings, audio_url, work_dir / "master-audio.mp3"
                )
            await ctx.report_progress(
                progress=40, total=100, message="正在解码并分析能量时间线"
            )
            result = await analyze_audio_energy(
                runtime.settings, audio, window_seconds
            )
            curve = result.pop("energy_curve", [])
            timeline_path = work_dir / "energy-timeline.json"
            timeline_path.write_text(
                json.dumps({**result, "energy_curve": curve}, ensure_ascii=False),
                encoding="utf-8",
            )
            artifact = runtime.artifacts.add(
                timeline_path, "energy-timeline.json", "application/json"
            )
        await ctx.report_progress(progress=100, total=100, message="音乐时间线分析完成")
        return {
            **result,
            "energy_window_count": len(curve),
            "energy_curve_artifact": _public_artifact(runtime, artifact),
        }
    except (ValueError, RuntimeError, OSError, aiohttp.ClientError) as exc:
        raise ToolError(str(exc)) from exc


if ENABLE_AUDIO_ANALYSIS:
    mcp.tool()(analyze_music_timeline)


@mcp.tool()
async def validate_and_render_music_video_plan(
    plan: MusicVideoPlan, ctx: Context
) -> dict[str, object]:
    """Validate an already assembled MV plan and render its generation prompts.

    New audio-to-MV authoring should start with create_music_video_plan and its
    incremental section tools; this entrypoint also serves existing stored plans.

    Call this after measuring the master audio and before the first paid video
    generation. Every segment receives an independent storyboard and prompt while
    sharing an exact continuity lock. `song_context_basis` records where story
    semantics came from; per-segment `music_evidence.timing_basis` records only
    timing evidence. `plan` must be the schema-native object advertised by this
    tool; JSON encoded as a string is rejected before execution. Mechanically derived
    generator limits, segment windows, generation durations, continuity handoffs, and
    local shot timestamps are normalized by this service. If creative validation still
    fails, the result contains a persisted draft reference, compact error list, and
    exact zero-based RFC-6901 patch_targets; copy those paths into
    patch_music_video_plan instead of resending the complete plan.
    """
    runtime = _require_tool_auth(ctx)
    return _store_validated_or_draft(runtime, plan)


@mcp.tool()
async def patch_music_video_plan(
    draft_id: Annotated[
        str,
        Field(pattern=r"^mv-draft-[0-9a-f]{24}$", description="Draft ID returned by plan validation."),
    ],
    draft_digest: Annotated[
        str,
        Field(pattern=r"^[0-9a-f]{64}$", description="Exact draft digest returned by plan validation."),
    ],
    patches: Annotated[
        list[MusicVideoPlanPatch],
        Field(
            min_length=1,
            max_length=32,
            description=(
                "Only add/replace operations for exact RFC-6901 paths returned in "
                "patch_targets. Array tokens are zero-based numeric indexes."
            ),
        ),
    ],
    ctx: Context,
) -> dict[str, object]:
    """Patch only invalid fields in a persisted MV plan draft and revalidate.

    Never resend the complete plan. Use the exact draft ID and digest from the
    previous validator result, and change only exact paths named by patch_targets.
    The segments collection is an array: segment-011 uses /segments/10/..., never
    /segments/segment-011/.... One patch call replaces one full-plan regeneration turn.
    """
    runtime = _require_tool_auth(ctx)
    try:
        stored = runtime.plans.get_draft(draft_id, draft_digest)
        current = _store_validated_or_draft(runtime, MusicVideoPlan.model_validate(stored))
        allowed = current.get("patch_targets", [])
        if current.get("valid") or any(patch.path not in allowed for patch in patches):
            raise ToolError("Change only exact paths in the saved draft's patch_targets; other fields remain locked.")
        patched = apply_music_video_plan_patches(
            MusicVideoPlan.model_validate(stored), patches
        )
        return _store_validated_or_draft(runtime, patched)
    except ValueError as exc:
        raise ToolError(str(exc)) from exc


def _store_validated_or_draft(
    runtime: Runtime, plan: MusicVideoPlan, opening_frame: dict[str, str] | None = None
) -> dict[str, object]:
    settings = runtime.settings
    normalized, corrections = normalize_music_video_plan(
        plan,
        live_min_duration_seconds=plan.live_min_duration_seconds,
        live_max_duration_seconds=plan.live_max_duration_seconds,
        target_segment_duration_seconds=plan.target_segment_duration_seconds,
        duration_step_seconds=plan.duration_step_seconds,
        tail_merge_max_duration_seconds=plan.tail_merge_max_duration_seconds,
    )
    try:
        result = runtime.plans.put(validate_mv_plan(normalized, opening_frame=opening_frame))
        result["normalization_count"] = len(corrections)
        result["_meta"] = {
            "exoharness": {
                "context_preserve": ["/plan_digest", "/valid", "/segment_count"],
                "context_checkpoint": (
                    "The complete creative plan is validated and stored by plan_digest. "
                    "Keep only this compact plan reference for execution."
                ),
                "workflow_continuation": {
                    "version": 1,
                    "status": "required",
                    "required_tool": "render_validated_music_video",
                    "reason": "validated_plan_requires_execution",
                    "checkpoint": {
                        "plan_digest": result["plan_digest"],
                    },
                    "required_arguments": {
                        "plan_digest": result["plan_digest"],
                    },
                    "allowed_argument_paths": [
                        "/plan_digest",
                        "/audio_url",
                        "/audio_sha256",
                    ],
                },
            }
        }
        return result
    except MusicVideoPlanValidationError as exc:
        draft = runtime.plans.put_draft(normalized.model_dump(mode="json"))
        # Cardinality must be repaired before normalization can align timings
        # and continuity. Provide only the missing segment inputs, never make
        # the client recover or resend the complete persisted creative plan.
        missing_segments = missing_segment_repairs(normalized)
        repair_context = {
            "missing_segments": missing_segments,
            "instruction": (
                "Append the missing segments with op=add using these exact paths. "
                "Write a complete, distinct prompt_plan and storyboard only for each missing segment. "
                "Do not resend or replace existing segments. After the count is correct, the service "
                "normalizes timing and copies previous ending_keyframe into each opening_keyframe."
                if missing_segments else
                "Repair only the reported fields using the exact patch_targets."
            ),
        }
        if missing_segments:
            repair_context["missing_segment_value_schema"] = MusicVideoSegmentPlan.model_json_schema()
        return {
            "valid": False,
            **draft,
            "normalization_count": len(corrections),
            "validation_errors": list(exc.errors),
            "patch_targets": list(exc.patch_targets),
            "next_tool": "patch_music_video_plan",
            "retry_policy": "patch_reported_fields_once_do_not_resend_complete_plan",
            "_meta": {
                "exoharness": {
                    "context_preserve": ["/draft_id", "/draft_digest", "/validation_errors", "/patch_targets"],
                    "context_checkpoint": (
                        "The complete invalid plan is persisted by draft_id and draft_digest. "
                        "Drop the full plan from active context and retain only the compact "
                        "validation errors before calling patch_music_video_plan."
                    ),
                    "workflow_continuation": {
                        "version": 1,
                        "status": "required",
                        "required_tool": "patch_music_video_plan",
                        "reason": "plan_validation_failed",
                        "checkpoint": draft,
                        "required_arguments": draft,
                        "allowed_argument_paths": list(exc.patch_targets),
                        "repair_context": repair_context,
                    },
                }
            },
        }
    except ValueError as exc:
        draft = runtime.plans.put_draft(normalized.model_dump(mode="json"))
        return {
            "valid": False,
            **draft,
            "normalization_count": len(corrections),
            "validation_errors": [str(exc)],
            "patch_targets": [],
            "next_tool": "patch_music_video_plan",
            "retry_policy": "patch_reported_fields_once_do_not_resend_complete_plan",
            "_meta": {
                "exoharness": {
                    "context_preserve": ["/draft_id", "/draft_digest", "/validation_errors", "/patch_targets"],
                    "context_checkpoint": (
                        "The complete invalid plan is persisted by draft_id and draft_digest. "
                        "Drop the full plan from active context and retain only the compact "
                        "validation errors before calling patch_music_video_plan."
                    ),
                    "workflow_continuation": {
                        "version": 1,
                        "status": "required",
                        "required_tool": "patch_music_video_plan",
                        "reason": "plan_validation_failed",
                        "checkpoint": draft,
                        "required_arguments": draft,
                        "allowed_argument_paths": [],
                    },
                }
            },
        }


@mcp.tool()
async def get_music_video_plan_segment(
    plan_digest: Annotated[
        str,
        Field(
            pattern=r"^[0-9a-f]{64}$",
            description="SHA-256 plan digest returned by plan validation.",
        ),
    ],
    segment_id: Annotated[
        str,
        Field(
            pattern=r"^segment-[0-9]{3,}$",
            description="Stable segment identifier from the validation result.",
        ),
    ],
    ctx: Context,
) -> dict[str, object]:
    """Return one exact prompt and controls from a stored validated MV plan.

    Fetch only the segment that is about to be generated. This keeps complete
    plans durable without replaying every segment through each model turn.
    """
    runtime = _require_tool_auth(ctx)
    try:
        return runtime.plans.get_segment(plan_digest, segment_id)
    except ValueError as exc:
        raise ToolError(str(exc)) from exc


@mcp.tool(meta=GENERATOR_CONNECTIONS)
async def render_validated_music_video(
    plan_digest: Annotated[
        str,
        Field(
            pattern=r"^[0-9a-f]{64}$",
            description="Digest returned by validate_and_render_music_video_plan.",
        ),
    ],
    audio_url: Annotated[
        str,
        Field(description="Immutable HTTP(S) URL for the original master audio."),
    ],
    audio_sha256: Annotated[
        str,
        Field(
            pattern=r"^[0-9a-f]{64}$",
            description="Expected SHA-256 for the immutable master audio.",
        ),
    ],
    ctx: Context,
) -> dict[str, object]:
    """Execute a stored plan as one resumable, progress-capable MV workflow.

    This service submits each video job once, polls it without language-model
    turns, persists the returned clip, extracts a last-frame Artifact, and sends
    that exact Artifact URL as the next job's first_frame_url. If the saved plan
    bound a user still, the first job is image-to-video from that frame; later
    openings still use the previous clip's last frame. After all clips complete
    it concatenates the silent video and muxes the original audio.
    Repeating the same plan/audio pair resumes its durable workflow state.
    """
    runtime = _require_tool_auth(ctx)
    workflow_id = workflow_id_for(plan_digest, audio_sha256)
    cancel_requested = peer_cancellation_check(ctx)
    lock = runtime.workflow_locks.setdefault(workflow_id, asyncio.Lock())
    if lock.locked():
        await ctx.report_progress(
            progress=0,
            total=100,
            message="相同 MV 工作流正在执行，正在等待现有执行",
        )

    idle_timeout = ProgressIdleTimeout(runtime.settings.workflow_idle_timeout_seconds)

    async def progress(value: float, message: str) -> None:
        idle_timeout.touch()
        await ctx.report_progress(progress=value, total=100, message=message)

    try:
        async with idle_timeout:
            async with lock:
                plan = runtime.plans.get(plan_digest)
                required_fields = ("first_frame_url",)
                if reference_image_fetch_url(plan, runtime.settings):
                    required_fields += ("reference_image_url",)
                async with connect_video_generator(
                    runtime.settings, ctx.headers,
                    required_submit_fields=required_fields,
                ) as generator:
                    workflow = MusicVideoWorkflow(
                        runtime.settings,
                        runtime.artifacts,
                        runtime.plans,
                        runtime.workflows,
                        generator,
                    )
                    result = await workflow.run(
                        plan_digest, audio_url, audio_sha256, progress,
                        cancel_requested=cancel_requested,
                    )
        result["_meta"] = {
            "exoharness": {
                "context_checkpoint": (
                    "The durable media workflow completed. Earlier Skill instructions, "
                    "intermediate prompts, polling state, and clip details may leave the "
                    "active model context; answer from this final Artifact result."
                ),
                "context_preserve": ["/workflow_id", "/plan_digest", "/status", "/result"],
                "workflow_continuation": {
                    "version": 1,
                    "status": "complete",
                    "context_scope": "checkpoint",
                },
            }
        }
        return result
    except TimeoutError as exc:
        raise ToolError(
            "music-video workflow reported no progress for "
            f"{runtime.settings.workflow_idle_timeout_seconds} seconds"
        ) from exc
    except ExceptionGroup as exc:
        raise ToolError(_exception_group_message(exc)) from exc
    except (ValueError, RuntimeError, OSError, aiohttp.ClientError, UpstreamToolError) as exc:
        raise ToolError(str(exc)) from exc


def _exception_group_message(error: BaseExceptionGroup) -> str:
    pending: list[BaseException] = list(error.exceptions)
    messages: list[str] = []
    while pending:
        current = pending.pop(0)
        if isinstance(current, BaseExceptionGroup):
            pending[0:0] = current.exceptions
            continue
        message = str(current).strip() or type(current).__name__
        if message not in messages:
            messages.append(message)
    return "; ".join(messages) or "nested MCP operation failed"


@mcp.tool()
async def extract_last_frame(
    video_url: SourceUrl, ctx: Context,
    artifact_visibility: Annotated[Literal["chat", "internal"], Field(description="Presentation only: use internal for a frame consumed by the next generation call. It keeps its signed URL but is not a chat attachment. Use chat when the user requested the frame itself.")] = "chat",
) -> dict[str, object]:
    """Extract the final visible frame of a completed video as a signed PNG URL.

    The returned URL is suitable as the next generation call's first-frame URL.
    """
    runtime = _require_tool_auth(ctx)
    try:
        await ctx.report_progress(progress=5, total=100, message="正在读取视频")
        with tempfile.TemporaryDirectory(dir=runtime.settings.data_dir) as raw_dir:
            work_dir = Path(raw_dir)
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(
                    total=runtime.settings.operation_timeout_seconds
                )
            ) as session:
                video = await download_source(
                    session, runtime.settings, video_url, work_dir / "source.mp4"
                )
            await ctx.report_progress(progress=55, total=100, message="正在提取末帧")
            output = work_dir / "last-frame.png"
            metadata = await extract_last_frame_file(runtime.settings, video, output)
            artifact = runtime.artifacts.add(output, "last-frame.png", "image/png")
        await ctx.report_progress(progress=100, total=100, message="末帧提取完成")
        result = {"result": {**_public_artifact(runtime, artifact), **metadata}}
        if artifact_visibility == "internal":
            result["_meta"] = {"exoharness": {"artifact_visibility": "internal"}}
        return result
    except (ValueError, RuntimeError, OSError, aiohttp.ClientError) as exc:
        raise ToolError(str(exc)) from exc


@mcp.tool()
async def assemble_music_video(
    video_urls: Annotated[
        list[str],
        Field(
            min_length=1,
            max_length=64,
            description="Ordered HTTP(S) URLs for silent source video clips.",
        ),
    ],
    audio_url: Annotated[
        str,
        Field(
            description=(
                "Complete HTTP(S) URL for the immutable master audio. Its timeline "
                "and duration are authoritative for the final MV."
            )
        ),
    ],
    ctx: Context,
) -> dict[str, object]:
    """Normalize and concatenate silent clips, then mux the original master audio.

    Source order is preserved. If the visual sequence is slightly short, the last
    frame is held; if it is long, it is trimmed to the master audio duration.
    """
    runtime = _require_tool_auth(ctx)

    async def progress(value: float, message: str) -> None:
        await ctx.report_progress(progress=value, total=100, message=message)

    try:
        await progress(3.0, "正在读取主音乐与视频片段")
        with tempfile.TemporaryDirectory(dir=runtime.settings.data_dir) as raw_dir:
            work_dir = Path(raw_dir)
            timeout = aiohttp.ClientTimeout(
                total=runtime.settings.operation_timeout_seconds
            )
            async with aiohttp.ClientSession(timeout=timeout) as session:
                audio_task = download_source(
                    session, runtime.settings, audio_url, work_dir / "master-audio.mp3"
                )
                clip_tasks = [
                    download_source(
                        session,
                        runtime.settings,
                        url,
                        work_dir / f"clip-{index:04d}.mp4",
                    )
                    for index, url in enumerate(video_urls)
                ]
                downloaded = await asyncio.gather(audio_task, *clip_tasks)
            audio = downloaded[0]
            clips = list(downloaded[1:])
            await progress(20.0, "媒体读取完成，正在校验时间轴")
            output = work_dir / "final-mv.mp4"
            metadata = await assemble_music_video_files(
                runtime.settings, clips, audio, work_dir, output, progress
            )
            artifact = runtime.artifacts.add(output, "final-mv.mp4", "video/mp4")
        return {
            "result": {
                **_public_artifact(runtime, artifact),
                **metadata,
            }
        }
    except (ValueError, RuntimeError, OSError, aiohttp.ClientError) as exc:
        raise ToolError(str(exc)) from exc


@mcp.tool(meta=GENERATOR_CONNECTIONS)
async def prepare_narrated_video(plan: NarratedPlan, ctx: Context) -> dict[str, object]:
    """Validate and persist ONE complete narrated video plan, without generating video.

    Each generation object follows the live video service schema. Return compact
    plan_digest plus narration texts; do not repeatedly author/read segment prompts.
    Render later by digest and narration assets. No Skill-specific style rules.
    """
    runtime = _require_tool_auth(ctx)
    try:
        async with connect_video_generator(runtime.settings, ctx.headers) as generator:
            result = await NarratedWorkflow(runtime.settings, runtime.artifacts).prepare(plan, generator)
        result["_meta"] = {"exoharness": prepared_checkpoint(result)}
        return result
    except Exception as exc:
        detail = _exception_group_message(exc) if isinstance(exc, BaseExceptionGroup) else str(exc)
        raise ToolError(detail) from exc


@mcp.tool(meta=GENERATOR_CONNECTIONS)
async def render_narrated_video(
    plan_digest: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")],
    ctx: Context,
    narration: list[NarrationAsset] = [],
    retry_failed: bool = False,
) -> CallToolResult:
    """Execute a saved narrated plan in ONE resumable call with direct progress.

    Bind TTS artifact.attachment_id plus sha256 once (omit on Resume). Do not pass
    chat /v1/run-artifacts download URLs. Allowlisted audio_url still works.
    Validates all narration before video generation. Serially generates segments,
    passes actual boundary frames, and delivers only the final timed MP4. Repeated
    calls reuse completed segments/results. Ambiguous submissions never auto-retry.
    Explicit retry_failed applies only to a known failed job, not unknown outcomes.
    """
    runtime = _require_tool_auth(ctx)
    idle = ProgressIdleTimeout(runtime.settings.workflow_idle_timeout_seconds)
    async def progress(value, message):
        idle.touch()
        await ctx.report_progress(progress=value, total=100, message=message)
    try:
        async with idle:
            result = await NarratedWorkflow(runtime.settings, runtime.artifacts).run(
                plan_digest, narration, progress,
                lambda settings: connect_video_generator(settings, ctx.headers),
                retry_failed=retry_failed, cancel_requested=peer_cancellation_check(ctx))
        meta = {"exoharness": {
            "context_checkpoint": "Narrated workflow completed; answer only from final artifact.",
            "context_preserve": ["/workflow_id", "/plan_digest", "/status", "/result"],
            "workflow_continuation": {"version": 1, "status": "complete", "context_scope": "checkpoint"},
        }}
        return CallToolResult(content=[TextContent(type="text", text=json.dumps(result, ensure_ascii=False))],
                              structuredContent=result, _meta=meta)
    except NarrationPreflightError as exc:
        state = NarratedWorkflow(
            runtime.settings, runtime.artifacts
        ).store.states.load("mv_" + plan_digest)
        retained = exc.retained_segment_ids
        required = [str(item["segment_id"]) for item in exc.repairs]
        value = {
            "plan_digest": plan_digest,
            "status": "needs_narration_repair",
            "repairs": exc.repairs,
            "retained_narration_segment_ids": retained,
            "required_narration_segment_ids": required,
            "next_action": (
                "Shorten and synthesize only required_narration_segment_ids, then "
                "call render_narrated_video with only those replacement bindings. "
                "Omit retained narration; do not synthesize it again. No video was submitted."
            ),
            "segments": [
                {
                    "segment_id": item["segment_id"],
                    "narration_saved": bool(item.get("audio")),
                    "job_id": item.get("job_id"),
                    "video_saved": bool(item.get("clip")),
                }
                for item in (state or {}).get("segments", [])
            ],
        }
        meta = {"exoharness": {
            "context_checkpoint": (
                "Narration preflight completed before GPU work; valid audio is "
                "retained and only listed narration segments require replacement."
            ),
            "context_preserve": [
                "/plan_digest",
                "/status",
                "/repairs",
                "/retained_narration_segment_ids",
                "/required_narration_segment_ids",
                "/next_action",
            ],
        }}
        return CallToolResult(
            content=[TextContent(type="text", text=json.dumps(value, ensure_ascii=False))],
            structuredContent=value,
            _meta=meta,
            isError=False,
        )
    except Exception as exc:
        store = NarratedWorkflow(runtime.settings, runtime.artifacts).store
        state = store.states.load("mv_" + plan_digest)
        if state is None:
            raise ToolError(str(exc)) from exc
        detail = _exception_group_message(exc) if isinstance(exc, BaseExceptionGroup) else str(exc)
        value = {"plan_digest": plan_digest, "status": state["status"], "error": detail,
                 "segments": [{"segment_id": s["segment_id"], "status": s["status"],
                               "job_id": s.get("job_id"), "video_saved": bool(s.get("clip"))}
                              for s in state["segments"]]}
        meta = {"exoharness": {
            "context_checkpoint": "Narrated workflow paused; full plan and completed media retained by the MCP.",
            "context_preserve": ["/plan_digest", "/status", "/error", "/segments"],
        }}
        # Audio preflight errors may need another TTS call. Do not lock out
        # prerequisite capabilities before any video was submitted.
        if any(s.get("job_id") or s.get("clip") or s.get("status") == "submitting" for s in state["segments"]):
            meta["exoharness"]["workflow_continuation"] = {
                "version": 1, "status": "required", "context_scope": "checkpoint",
                "required_tool": "render_narrated_video", "required_arguments": {"plan_digest": plan_digest},
                "allowed_argument_paths": ["/narration", "/retry_failed"],
                "checkpoint": {"plan_digest": plan_digest},
                "reason": "Resume saved media state. Do not resubmit a plan or regenerate successful segments. Unknown submissions require reconciliation, not retry."}
        return CallToolResult(content=[TextContent(type="text", text=json.dumps(value, ensure_ascii=False))],
                              structuredContent=value, _meta=meta, isError=True)


@mcp.tool(meta=GENERATOR_CONNECTIONS)
async def generate_standalone_video(
    plan: StandalonePlan,
    ctx: Context,
) -> CallToolResult:
    """Generate one continuous multi-clip video in ONE resumable MCP call.

    Use this only when the selected video generator cannot produce the requested
    total duration in one call. Storyboard timestamps are shots within a video,
    not separate generator segments. For a duration supported by one generator
    call, send the complete multi-shot prompt directly to that generator instead.
    Supply the complete ordered multi-call segment plan once. This service validates all
    native generator arguments before paid work, submits each clip once, polls
    internally, passes real boundary frames, preserves generated source audio,
    and returns only the final assembled MP4. Progress is sent directly via MCP.
    On interruption use resume_standalone_video with the returned plan_digest;
    never rebuild or resubmit completed segments.
    """
    runtime = _require_tool_auth(ctx)
    workflow = StandaloneWorkflow(runtime.settings, runtime.artifacts)
    idle = ProgressIdleTimeout(runtime.settings.workflow_idle_timeout_seconds)

    async def progress(value, message):
        idle.touch()
        await ctx.report_progress(progress=value, total=100, message=message)

    digest = None
    try:
        async with idle:
            async with connect_video_generator(runtime.settings, ctx.headers) as generator:
                digest = (await workflow.prepare(plan, generator))["plan_digest"]
            result = await workflow.run(
                digest, [], progress,
                lambda settings: connect_video_generator(settings, ctx.headers),
                cancel_requested=peer_cancellation_check(ctx),
            )
        return _standalone_result(result, digest)
    except Exception as exc:
        if digest is None:
            raise ToolError(str(exc)) from exc
        return _standalone_error(workflow, digest, exc)


@mcp.tool(meta=GENERATOR_CONNECTIONS)
async def resume_standalone_video(
    plan_digest: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")],
    ctx: Context,
    retry_failed: bool = False,
) -> CallToolResult:
    """Resume a saved standalone video plan; keep all successful clips and real frame handoffs.

    A known failed upstream job requires explicit retry_failed=true. An unknown
    submission outcome never triggers another submit automatically.
    """
    runtime = _require_tool_auth(ctx)
    workflow = StandaloneWorkflow(runtime.settings, runtime.artifacts)
    idle = ProgressIdleTimeout(runtime.settings.workflow_idle_timeout_seconds)

    async def progress(value, message):
        idle.touch()
        await ctx.report_progress(progress=value, total=100, message=message)

    try:
        async with idle:
            result = await workflow.run(
                plan_digest, [], progress,
                lambda settings: connect_video_generator(settings, ctx.headers),
                retry_failed=retry_failed,
                cancel_requested=peer_cancellation_check(ctx),
            )
        return _standalone_result(result, plan_digest)
    except Exception as exc:
        return _standalone_error(workflow, plan_digest, exc)


def _standalone_result(result, digest):
    meta = {"exoharness": {
        "context_checkpoint": "Standalone video completed; deliver only the final artifact.",
        "context_preserve": ["/workflow_id", "/plan_digest", "/status", "/result"],
        "workflow_continuation": {"version": 1, "status": "complete", "context_scope": "checkpoint"},
    }}
    return CallToolResult(
        content=[TextContent(type="text", text=json.dumps(result, ensure_ascii=False))],
        structuredContent=result, _meta=meta,
    )


def _standalone_error(workflow, digest, exc):
    state = workflow.workflows.load("mv_" + digest)
    if state is None:
        raise ToolError(str(exc)) from exc
    detail = _exception_group_message(exc) if isinstance(exc, BaseExceptionGroup) else str(exc)
    value = {"plan_digest": digest, "status": state["status"], "error": detail,
             "segments": [{"segment_id": s["segment_id"], "status": s["status"],
                           "job_id": s.get("job_id"), "video_saved": bool(s.get("clip"))}
                          for s in state["segments"]]}
    meta = {"exoharness": {
        "context_checkpoint": "Standalone workflow paused; exact plan and completed clips remain in the MCP store.",
        "context_preserve": ["/plan_digest", "/status", "/error", "/segments"],
        "workflow_continuation": {
            "version": 1, "status": "required", "context_scope": "checkpoint",
            "required_tool": "resume_standalone_video",
            "required_arguments": {"plan_digest": digest},
            "allowed_argument_paths": ["/retry_failed"],
            "checkpoint": {"plan_digest": digest},
            "reason": "Resume saved media state; do not resubmit successful clips. Unknown submissions require reconciliation.",
        },
    }}
    return CallToolResult(
        content=[TextContent(type="text", text=json.dumps(value, ensure_ascii=False))],
        structuredContent=value, _meta=meta, isError=True,
    )


@mcp.tool()
async def assemble_video_timeline(
    segments: Annotated[list[TimelineSegment], Field(min_length=1, max_length=64,
        description="Ordered completed video references, exact windows, and either source-audio preservation or separate narration references")],
    ctx: Context,
) -> dict[str, object]:
    """Assemble completed media by URL inside the media service, without copying large videos into a code workspace.

    Trim each clip to its own planned window. Per segment, optionally preserve
    its source audio or align a separate narration track. The default remains
    silent when neither option is selected, for existing silent/MV workflows.
    Reject short video/overlong narration; never freeze-fill, loop or regenerate.
    Only administrator-allowed sources may be downloaded. Return one final MP4.
    """
    runtime = _require_tool_auth(ctx)
    async def progress(value, message):
        await ctx.report_progress(progress=value, total=100, message=message)
    try:
        # Reject an unapproved later source before downloading any earlier one.
        for segment in segments:
            validate_source_url(segment.video_url, runtime.settings.allowed_fetch_origins)
            if segment.audio_url:
                validate_source_url(segment.audio_url, runtime.settings.allowed_fetch_origins)
        with tempfile.TemporaryDirectory(dir=runtime.settings.data_dir) as raw:
            directory = Path(raw)
            clips, audios = [], []
            timeout = aiohttp.ClientTimeout(total=runtime.settings.operation_timeout_seconds)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                for index, segment in enumerate(segments):
                    await progress(2 + 16 * index / len(segments), f"正在读取第 {index + 1}/{len(segments)} 段媒体引用")
                    clips.append(await download_source(session, runtime.settings, segment.video_url, directory / f"source-{index}.mp4"))
                    audios.append(await download_source(session, runtime.settings, segment.audio_url, directory / f"speech-{index}.audio") if segment.audio_url else None)
            output = directory / "final-video.mp4"
            metadata = await assemble_timeline_files(runtime.settings, segments, clips, audios, directory, output, progress)
            artifact = runtime.artifacts.add(output, "final-video.mp4", "video/mp4")
        await progress(100, "完整视频已合成")
        return {"result": {**_public_artifact(runtime, artifact), **metadata},
                "_meta": {"exoharness": {"context_checkpoint": "timeline delivered: " + artifact.artifact_id,
                    "context_preserve": ["/result"],
                    "workflow_continuation": {"version": 1, "status": "complete", "context_scope": "checkpoint"}}}}
    except (ValueError, RuntimeError, OSError, aiohttp.ClientError) as exc:
        raise ToolError(str(exc)) from exc


@mcp.custom_route("/health", methods=["GET"], include_in_schema=False)
async def health_route(_request: Request) -> Response:
    runtime = get_runtime()
    return JSONResponse(
        {
            "status": "ok",
            "ffmpeg": runtime.settings.ffmpeg_binary,
            "ffprobe": runtime.settings.ffprobe_binary,
            "allowed_fetch_origins": list(runtime.settings.allowed_fetch_origins),
            "video_generator_configured": bool(runtime.settings.video_generator_url),
        }
    )


@mcp.custom_route(
    "/artifacts/{artifact_id}/{filename}", methods=["GET"], include_in_schema=False
)
async def artifact_route(request: Request) -> Response:
    runtime = get_runtime()
    artifact_id = request.path_params["artifact_id"]
    filename = request.path_params["filename"]
    artifact = runtime.artifacts.get(artifact_id, filename)
    if artifact is None:
        return JSONResponse({"error": "artifact not found"}, status_code=404)
    signed = runtime.artifacts.verify(
        artifact_id,
        filename,
        request.query_params.get("expires"),
        request.query_params.get("signature"),
    )
    bearer = _authorized(request.headers, runtime.settings.api_token)
    if not signed and not bearer:
        return JSONResponse({"error": "unauthorized"}, status_code=401)
    return FileResponse(
        Path(artifact.path),
        media_type=artifact.content_type,
        filename=artifact.filename,
    )


def main() -> None:
    import uvicorn

    runtime = get_runtime()
    security = TransportSecuritySettings(
        enable_dns_rebinding_protection=True,
        allowed_hosts=list(runtime.settings.allowed_hosts),
        allowed_origins=list(runtime.settings.allowed_origins),
    )
    app = mcp.streamable_http_app(
        streamable_http_path=runtime.settings.mcp_path,
        json_response=False,
        transport_security=security,
        host=runtime.settings.host,
    )
    app.add_middleware(
        MCPBearerAuthMiddleware,
        api_token=runtime.settings.api_token,
        mcp_path=runtime.settings.mcp_path,
    )
    try:
        uvicorn.run(
            app,
            host=runtime.settings.host,
            port=runtime.settings.port,
            log_level=mcp.settings.log_level.lower(),
        )
    except KeyboardInterrupt:
        return


if __name__ == "__main__":
    main()
