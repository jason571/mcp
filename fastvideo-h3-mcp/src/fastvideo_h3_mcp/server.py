from __future__ import annotations

import asyncio
import hmac
import secrets
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal, Mapping

from mcp.server import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.server.mcpserver import Context
from mcp.server.transport_security import TransportSecuritySettings
from pydantic import Field
from starlette.requests import Request
from starlette.responses import FileResponse, JSONResponse, Response
from starlette.types import ASGIApp, Receive, Scope, Send

from .artifacts import ArtifactSigner
from .assets import AssetStore
from .backend import build_backend
from .config import Settings, generation_duration_from_env
from .models import (
    DEFAULT_DURATION_SECONDS,
    DEFAULT_HEIGHT,
    DEFAULT_WIDTH,
    LONG_VIDEO_MAX_RESOLUTION,
    MAX_DURATION_SECONDS,
    MIN_DURATION_SECONDS,
    GenerationRequest,
    JobRecord,
    JobStatus,
    ASPECT_RATIOS,
    RESOLUTION_SHORT_EDGES,
    canvas_dimensions,
    duration_seconds_contract,
    duration_resolution_limits,
    normalize_resolution,
    validate_duration_resolution,
)
from .service import JobManager
from .store import JobStore


mcp = MCPServer(
    "FastVideo H3 Video Generator",
    version="0.1.0",
    instructions=(
        "Generate synchronized video and audio with FastVideo MiniMax H3. "
        "Use generate_video when the caller can receive MCP progress notifications. "
        "Use submit_video_generation plus get_video_job for detached execution. "
        "For reference-frame generation, include first_frame_url and "
        "last_frame_url together in one generation tool call. The server fetches "
        "each non-empty HTTP(S) URL before queueing the video. Never split the "
        "two frames into separate generation calls."
    ),
)


@dataclass(slots=True)
class Runtime:
    settings: Settings
    store: JobStore
    manager: JobManager
    signer: ArtifactSigner
    assets: AssetStore


_runtime: Runtime | None = None

ArtifactVisibility = Annotated[
    Literal["chat", "internal"],
    Field(description="Presentation only: chat for a requested final video; internal for a clip consumed by later editing/assembly. Internal keeps the result URL and progress but suppresses automatic chat attachments. Detached polling retains this choice; no video content or execution setting changes."),
]


FirstFrameAssetId = Annotated[
    str | None,
    Field(
        description=(
            "Optional MCP-local asset_id returned by upload_reference_image or a "
            "previous URL import. Do not pass it together with first_frame_url."
        )
    ),
]
LastFrameAssetId = Annotated[
    str | None,
    Field(
        description=(
            "Optional MCP-local asset_id returned by upload_reference_image or a "
            "previous URL import. Do not pass it together with last_frame_url."
        )
    ),
]
FirstFrameUrl = Annotated[
    str | None,
    Field(
        description=(
            "Complete http:// or https:// URL for the first-frame image. The MCP "
            "server downloads and validates it before queueing. Pass first_frame_url "
            "and last_frame_url in this same generation call."
        )
    ),
]
LastFrameUrl = Annotated[
    str | None,
    Field(
        description=(
            "Complete http:// or https:// URL for the last-frame image. The MCP "
            "server downloads and validates it before queueing. Pass last_frame_url "
            "and first_frame_url in this same generation call; do not submit a "
            "separate generation call for this image."
        )
    ),
]
Resolution = Annotated[
    str,
    Field(
        description=(
            "Output short-edge resolution. Supported values are 512p, 640p, and "
            "768p. Unknown values are automatically normalized to 512p. Requests "
            "longer than 15 seconds are also automatically normalized to 512p."
        )
    ),
]
try:
    _TOOL_DEFAULT_DURATION_SECONDS, _ = generation_duration_from_env()
except ValueError:
    _TOOL_DEFAULT_DURATION_SECONDS = DEFAULT_DURATION_SECONDS

DurationSeconds = Annotated[
    float,
    Field(
        ge=MIN_DURATION_SECONDS,
        le=MAX_DURATION_SECONDS,
        default=_TOOL_DEFAULT_DURATION_SECONDS,
        description=(
            "Requested clip length in seconds. Default is "
            "FASTVIDEO_MCP_DEFAULT_DURATION_SECONDS from the MCP process. "
            "One generation cannot exceed FASTVIDEO_MCP_MAX_DURATION_SECONDS, "
            f"itself capped at {MAX_DURATION_SECONDS:g}."
        ),
    ),
]


def get_runtime() -> Runtime:
    global _runtime
    if _runtime is None:
        settings = Settings.from_env()
        settings.prepare_runtime_directories()
        store = JobStore(settings.database_path)
        manager = JobManager(settings, store, build_backend(settings))
        _runtime = Runtime(
            settings,
            store,
            manager,
            ArtifactSigner(settings),
            AssetStore(
                settings.asset_dir,
                settings.max_reference_image_bytes,
            ),
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
    """Reject unauthenticated MCP transport requests before session handling."""

    def __init__(self, app: ASGIApp, api_token: str | None, mcp_path: str) -> None:
        self.app = app
        self.api_token = api_token
        self.mcp_path = mcp_path.rstrip("/") or "/"

    def _protects(self, path: str) -> bool:
        if self.mcp_path == "/":
            return True
        return path == self.mcp_path or path.startswith(f"{self.mcp_path}/")

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if (
            scope["type"] == "http"
            and self._protects(str(scope.get("path", "")))
            and not _authorized(
                {
                    key.decode("latin-1"): value.decode("latin-1")
                    for key, value in scope.get("headers", [])
                },
                self.api_token,
            )
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
    headers = ctx.headers or {}
    if not _authorized(headers, runtime.settings.api_token):
        raise PermissionError("missing or invalid MCP bearer token")
    return runtime


async def _resolve_reference_image(
    runtime: Runtime,
    asset_id: str | None,
    url: str | None,
    field_name: str,
) -> tuple[str | None, Path | None]:
    if asset_id and url:
        raise ValueError(
            f"pass either {field_name}_asset_id or {field_name}_url, not both"
        )
    resolved_asset_id = asset_id
    if url:
        fetched = await runtime.assets.fetch_image_url(url)
        value = fetched.get("asset_id")
        if not isinstance(value, str):
            raise RuntimeError("downloaded reference image has no asset_id")
        resolved_asset_id = value
    return resolved_asset_id, await runtime.assets.resolve_image(resolved_asset_id)


async def _make_request(
    runtime: Runtime,
    prompt: str,
    negative_prompt: str | None,
    soundscape: str | None,
    prompt_mode: str,
    duration_seconds: float | None,
    seed: int | None,
    audio_enabled: bool,
    aspect_ratio: str,
    resolution: str,
    first_frame_asset_id: str | None,
    last_frame_asset_id: str | None,
    model_id: str,
    first_frame_url: str | None = None,
    last_frame_url: str | None = None,
) -> GenerationRequest:
    if duration_seconds is None:
        duration_seconds = getattr(
            runtime.settings,
            "default_duration_seconds",
            DEFAULT_DURATION_SECONDS,
        )
    maximum_duration = getattr(
        runtime.settings, "max_duration_seconds", MAX_DURATION_SECONDS
    )
    if not MIN_DURATION_SECONDS <= duration_seconds <= maximum_duration:
        raise ValueError(
            "duration_seconds must be between "
            f"{MIN_DURATION_SECONDS:g} and {maximum_duration:g}"
        )
    selected_seed = secrets.randbelow(2**31) if seed is None else seed
    if not 0 <= selected_seed < 2**31:
        raise ValueError("seed must be between 0 and 2147483647")
    resolution = normalize_resolution(duration_seconds, resolution)
    width, height = canvas_dimensions(aspect_ratio, resolution)
    # Keep this strict invariant check after normalization so internal callers
    # cannot accidentally enqueue a memory-unsafe request.
    validate_duration_resolution(duration_seconds, resolution)
    known_model_ids = {
        "auto",
        "fast",
        runtime.settings.model_id,
        "base",
        "minimax-h3-base-fl2va",
    }
    if model_id not in known_model_ids:
        raise ValueError(f"unknown model_id: {model_id}")
    (resolved_first_id, first_frame), (resolved_last_id, last_frame) = (
        await asyncio.gather(
            _resolve_reference_image(
                runtime, first_frame_asset_id, first_frame_url, "first_frame"
            ),
            _resolve_reference_image(
                runtime, last_frame_asset_id, last_frame_url, "last_frame"
            ),
        )
    )
    resolved_model_id = "base" if first_frame or last_frame else "fast"
    request = GenerationRequest(
        prompt=prompt,
        negative_prompt=negative_prompt,
        soundscape=soundscape,
        prompt_mode=prompt_mode,
        model_id=resolved_model_id,
        duration_seconds=duration_seconds,
        seed=selected_seed,
        audio_enabled=audio_enabled,
        aspect_ratio=aspect_ratio,
        resolution=resolution,
        first_frame_asset_id=resolved_first_id,
        last_frame_asset_id=resolved_last_id,
        first_frame_path=str(first_frame) if first_frame else None,
        last_frame_path=str(last_frame) if last_frame else None,
        width=width,
        height=height,
    )
    # Force prompt validation before a task is persisted.
    _ = request.model_prompt
    return request


def _public_job(runtime: Runtime, job: JobRecord, artifact_visibility: str = "chat") -> dict[str, object]:
    result = job.as_public_dict()
    output = job.output_file()
    if job.status == JobStatus.COMPLETED and output is not None:
        result["result"] = {
            "filename": output.name,
            "content_type": "video/mp4",
            "download_url": runtime.signer.build_url(job.id, output),
            "url_ttl_seconds": runtime.settings.artifact_url_ttl_seconds
            if runtime.settings.artifact_signing_key
            else None,
        }
    if artifact_visibility == "internal":
        result["_meta"] = {"exoharness": {"artifact_visibility": "internal"}}
    return result


def _tracked_job(runtime: Runtime, job: JobRecord, artifact_visibility: str = "chat") -> dict[str, object]:
    """Explicit detached-job adapter; ordinary MCP clients may ignore metadata.

    This is an ExoHarness metadata extension, not a claim of MCP Tasks support.
    Job names, status mappings and argument bindings remain service-owned.
    """
    result = _public_job(runtime, job, artifact_visibility)
    status = {
        JobStatus.COMPLETED: "completed",
        JobStatus.FAILED: "failed",
        JobStatus.CANCELED: "cancelled",
    }.get(job.status, "working")
    control = {"job_id": job.id, "managed_tracking": True}
    if artifact_visibility == "internal":
        control["artifact_visibility"] = "internal"
    result.setdefault("_meta", {}).setdefault("exoharness", {})["background_task"] = {
        "version": 1,
        "id": job.id,
        "status": status,
        "poll": {"tool": "get_video_job", "arguments": dict(control)},
        "cancel": {"tool": "cancel_video_job", "arguments": dict(control)},
        "poll_interval_ms": 2500,
        "progress": {"progress": job.progress, "total": 100.0, "message": f"[{job.stage}] {job.message}"},
    }
    return result


@mcp.tool()
async def generate_video(
    prompt: str,
    ctx: Context,
    negative_prompt: str | None = None,
    soundscape: str | None = None,
    prompt_mode: str = "simple",
    duration_seconds: DurationSeconds = _TOOL_DEFAULT_DURATION_SECONDS,
    seed: int | None = None,
    audio_enabled: bool = True,
    aspect_ratio: str = "16:9",
    resolution: Resolution = "512p",
    first_frame_asset_id: FirstFrameAssetId = None,
    last_frame_asset_id: LastFrameAssetId = None,
    first_frame_url: FirstFrameUrl = None,
    last_frame_url: LastFrameUrl = None,
    model_id: str = "auto",
    artifact_visibility: ArtifactVisibility = "chat",
) -> dict[str, object]:
    """Generate one video and stream progress to this MCP tool caller.

    The caller must keep the tool call open and request progress notifications.
    The final response contains the completed job and a signed MP4 URL.
    Pass complete first_frame_url and last_frame_url HTTP(S) addresses in this
    same call. The server downloads both images before queueing and routes the
    request to the Base backend. Do not create one video call per image.
    """
    runtime = _require_tool_auth(ctx)
    try:
        request = await _make_request(
            runtime,
            prompt,
            negative_prompt,
            soundscape,
            prompt_mode,
            duration_seconds,
            seed,
            audio_enabled,
            aspect_ratio,
            resolution,
            first_frame_asset_id,
            last_frame_asset_id,
            model_id,
            first_frame_url,
            last_frame_url,
        )
    except ValueError as exc:
        raise ToolError(str(exc)) from exc
    job, _queue_position = await runtime.manager.submit(JobRecord.new(request))
    try:
        async for update in runtime.manager.watch(job.id):
            await ctx.report_progress(
                progress=update.progress,
                total=100.0,
                message=f"[{update.stage}] {update.message}",
            )
            if update.terminal:
                return _public_job(runtime, update, artifact_visibility)
    except asyncio.CancelledError:
        await runtime.manager.cancel(job.id)
        raise
    raise RuntimeError("job progress stream ended before a terminal state")


@mcp.tool()
async def submit_video_generation(
    prompt: str,
    ctx: Context,
    negative_prompt: str | None = None,
    soundscape: str | None = None,
    prompt_mode: str = "simple",
    duration_seconds: DurationSeconds = _TOOL_DEFAULT_DURATION_SECONDS,
    seed: int | None = None,
    audio_enabled: bool = True,
    aspect_ratio: str = "16:9",
    resolution: Resolution = "512p",
    first_frame_asset_id: FirstFrameAssetId = None,
    last_frame_asset_id: LastFrameAssetId = None,
    first_frame_url: FirstFrameUrl = None,
    last_frame_url: LastFrameUrl = None,
    model_id: str = "auto",
    artifact_visibility: ArtifactVisibility = "chat",
) -> dict[str, object]:
    """Submit one detached video job and return immediately with its job_id.

    Reference images follow the same single-call contract as generate_video:
    pass first_frame_url and last_frame_url together. The server downloads both
    HTTP(S) images before queueing; never split them into separate video jobs.
    """
    runtime = _require_tool_auth(ctx)
    try:
        request = await _make_request(
            runtime,
            prompt,
            negative_prompt,
            soundscape,
            prompt_mode,
            duration_seconds,
            seed,
            audio_enabled,
            aspect_ratio,
            resolution,
            first_frame_asset_id,
            last_frame_asset_id,
            model_id,
            first_frame_url,
            last_frame_url,
        )
    except ValueError as exc:
        raise ToolError(str(exc)) from exc
    job, queue_position = await runtime.manager.submit(JobRecord.new(request))
    result = _tracked_job(runtime, job, artifact_visibility)
    result["queue_position"] = queue_position
    return result


@mcp.tool()
async def get_video_job(job_id: str, ctx: Context, managed_tracking: bool = False,
                        artifact_visibility: ArtifactVisibility = "chat") -> dict[str, object]:
    """Return the persisted status and progress for one video job."""
    runtime = _require_tool_auth(ctx)
    job = runtime.manager.get(job_id)
    return _tracked_job(runtime, job, artifact_visibility) if managed_tracking else _public_job(runtime, job, artifact_visibility)


@mcp.tool()
async def get_video_result(job_id: str, ctx: Context, artifact_visibility: ArtifactVisibility = "chat") -> dict[str, object]:
    """Return a fresh signed download URL after a job completes."""
    runtime = _require_tool_auth(ctx)
    job = runtime.manager.get(job_id)
    if job.status != JobStatus.COMPLETED:
        return {
            "job_id": job.id,
            "status": job.status,
            "progress": job.progress,
            "message": job.message,
            "error": job.error,
        }
    return _public_job(runtime, job, artifact_visibility)


@mcp.tool()
async def cancel_video_job(job_id: str, ctx: Context, managed_tracking: bool = False,
                           artifact_visibility: ArtifactVisibility = "chat") -> dict[str, object]:
    """Cancel a queued or running video job."""
    runtime = _require_tool_auth(ctx)
    job = await runtime.manager.cancel(job_id)
    return _tracked_job(runtime, job, artifact_visibility) if managed_tracking else _public_job(runtime, job, artifact_visibility)


@mcp.tool()
async def list_video_jobs(
    ctx: Context, limit: int = 20, status: str | None = None
) -> dict[str, object]:
    """List recent persisted video jobs, optionally filtered by status."""
    runtime = _require_tool_auth(ctx)
    if status is not None and status not in {item.value for item in JobStatus}:
        raise ValueError("invalid job status")
    jobs = runtime.manager.list(limit=limit, status=status)
    return {"jobs": [_public_job(runtime, job) for job in jobs]}


@mcp.tool()
async def list_video_models(ctx: Context) -> dict[str, object]:
    """Return the configured model and its enforced generation contract."""
    runtime = _require_tool_auth(ctx)
    duration_contract = duration_seconds_contract(
        runtime.settings.default_duration_seconds,
        runtime.settings.max_duration_seconds,
    )
    duration_resolution_limits_value = duration_resolution_limits(
        runtime.settings.max_duration_seconds
    )
    return {
        "duration_seconds": duration_contract,
        "routing": (
            "routing is modality-based: every text-only request uses FastH3; "
            "Base FL2VA is used only when a first or last reference frame is present"
        ),
        "defaults": {
            "duration_seconds": runtime.settings.default_duration_seconds,
            "resolution": LONG_VIDEO_MAX_RESOLUTION,
            "aspect_ratio": "16:9",
        },
        "resolution_normalization": {
            "default": LONG_VIDEO_MAX_RESOLUTION,
            "unknown_value_fallback": LONG_VIDEO_MAX_RESOLUTION,
            "over_15_seconds_fallback": LONG_VIDEO_MAX_RESOLUTION,
        },
        "models": [
            {
                "id": runtime.settings.model_id,
                "path": runtime.settings.model_path,
                "task": "text-to-audio-video",
                "width": DEFAULT_WIDTH,
                "height": DEFAULT_HEIGHT,
                "fps": 24,
                "duration_seconds": duration_contract,
                "aspect_ratios": list(ASPECT_RATIOS),
                "resolutions": list(RESOLUTION_SHORT_EDGES),
                "duration_resolution_limits": duration_resolution_limits_value,
                "first_frame": False,
                "last_frame": False,
                "sampling": {
                    "sigma_grid_points": 5,
                    "transformer_forwards": 4,
                    "guidance_scale": 1.0,
                },
                "backend": runtime.manager.backend.health(),
            },
            {
                "id": "minimax-h3-base-fl2va",
                "path": str(runtime.settings.comfyui_models_dir)
                if runtime.settings.base_backend == "comfyui-api"
                else runtime.settings.minimax_h3_model_path,
                "backend": runtime.settings.base_backend,
                "depends_on_comfyui": (
                    runtime.settings.base_backend == "comfyui-api"
                ),
                "task": "first-last-frame-to-audio-video",
                "fps": 24,
                "duration_seconds": duration_contract,
                "aspect_ratios": list(ASPECT_RATIOS),
                "resolutions": list(RESOLUTION_SHORT_EDGES),
                "duration_resolution_limits": duration_resolution_limits_value,
                "first_frame": True,
                "last_frame": True,
                "text_only_routing": False,
                "audio_can_be_omitted_from_output": True,
                "negative_prompt_semantics": "compiled as negative_constraints",
            },
        ]
    }


@mcp.tool()
async def upload_reference_image(
    data_base64: str,
    mime_type: str,
    ctx: Context,
) -> dict[str, object]:
    """Upload a PNG, JPEG, or WebP keyframe and return its asset_id.

    Pass the returned ID as first_frame_asset_id or last_frame_asset_id to a
    generation tool. The configured maximum size defaults to 20 MiB.
    """
    runtime = _require_tool_auth(ctx)
    return runtime.assets.save_image_base64(data_base64, mime_type)


@mcp.tool()
async def get_video_service_status(ctx: Context) -> dict[str, object]:
    """Return queue, worker and FastVideo backend health."""
    runtime = _require_tool_auth(ctx)
    await runtime.manager.start()
    return runtime.manager.status()


@mcp.custom_route("/health", methods=["GET"], include_in_schema=False)
async def health_route(_request: Request) -> Response:
    runtime = get_runtime()
    health = runtime.manager.status()
    backend = health["backend"]
    ready = isinstance(backend, dict) and bool(backend.get("ready"))
    return JSONResponse(
        {"status": "ok" if ready else "degraded", **health},
        status_code=200 if ready else 503,
    )


@mcp.custom_route(
    "/artifacts/{job_id}/{filename}", methods=["GET"], include_in_schema=False
)
async def artifact_route(request: Request) -> Response:
    runtime = get_runtime()
    job_id = request.path_params["job_id"]
    filename = request.path_params["filename"]
    if Path(filename).name != filename:
        return JSONResponse({"error": "invalid filename"}, status_code=400)
    job = runtime.store.get(job_id)
    if job is None or job.status != JobStatus.COMPLETED or not job.output_path:
        return JSONResponse({"error": "artifact not found"}, status_code=404)
    output = Path(job.output_path).resolve()
    if output.name != filename or runtime.settings.output_dir not in output.parents:
        return JSONResponse({"error": "artifact not found"}, status_code=404)

    signed = runtime.signer.verify(
        job_id,
        filename,
        request.query_params.get("expires"),
        request.query_params.get("signature"),
    )
    bearer = _authorized(request.headers, runtime.settings.api_token)
    if runtime.settings.artifact_signing_key:
        allowed = signed or bearer
    else:
        allowed = bearer
    if not allowed:
        return JSONResponse({"error": "unauthorized"}, status_code=401)
    if not output.is_file():
        return JSONResponse({"error": "artifact missing from storage"}, status_code=410)
    return FileResponse(output, media_type="video/mp4", filename=output.name)


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
