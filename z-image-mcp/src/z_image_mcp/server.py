import argparse
import asyncio
import base64
import json
import logging
import secrets
from contextlib import asynccontextmanager
from typing import Any
from urllib.parse import urlsplit

import uvicorn
from mcp.server.fastmcp import Context, FastMCP
from mcp.server.fastmcp.exceptions import ToolError
from mcp.server.transport_security import TransportSecuritySettings
from mcp.types import CallToolResult, ImageContent, ResourceLink, TextContent, ToolAnnotations
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import FileResponse, JSONResponse
from starlette.routing import Mount, Route

from . import __version__
from .backend import DiffusersBackend
from .config import Settings
from .jobs import JobManager
from .models import DEFAULT_INFERENCE_STEPS, MODEL_ID, GenerateRequest


def create_app(settings: Settings | None = None, backend=None):
    settings = settings or Settings()
    manager = JobManager(settings, backend or DiffusersBackend(settings))
    origin = urlsplit(settings.public_url)
    mcp = FastMCP(
        "Z-Image-Turbo",
        instructions=(
            "Generate PNG images with Tongyi-MAI/Z-Image-Turbo. Use generate_image for a direct "
            "result with progress; for short client timeouts use submit_image_generation, then "
            "get_image_job until terminal. Retrieve the PNG with get_image_result or its download "
            "URL. After a disconnect use the reported job_id instead of submitting duplicates. "
            "Send only prompt, width and height; seed and sampling steps are server-managed. "
            "Turbo uses guidance_scale=0 and does not support negative prompts."
        ),
        stateless_http=True,
        json_response=False,
        transport_security=TransportSecuritySettings(
            allowed_hosts=[
                origin.netloc,
                f"{settings.host}:{settings.port}",
                "127.0.0.1:*",
                "localhost:*",
                "[::1]:*",
            ],
            allowed_origins=[settings.public_url, "http://127.0.0.1:*", "http://localhost:*"],
        ),
    )

    def submit(prompt, width, height):
        try:
            return manager.submit(GenerateRequest(prompt=prompt, width=width, height=height))
        except ValueError as exc:
            raise ToolError(str(exc)) from exc

    def get(job_id):
        try:
            return manager.get(job_id)
        except ValueError as exc:
            raise ToolError(str(exc)) from exc

    def result_content(job, inline=False):
        data = manager.public(job)
        contents = [TextContent(type="text", text=json.dumps(data, ensure_ascii=False))]
        if job.status == "succeeded":
            contents.append(
                ResourceLink(
                    type="resource_link",
                    uri=data["result"]["download_url"],
                    name=f"{job.id}.png",
                    mimeType="image/png",
                    description=f"Generated {job.request.width}x{job.request.height} image",
                )
            )
            if inline:
                raw = (manager.output_dir / f"{job.id}.png").read_bytes()
                if len(raw) > 12 * 1024**2:
                    raise ToolError("Image too large to inline; use the download URL")
                contents.append(
                    ImageContent(
                        type="image", mimeType="image/png", data=base64.b64encode(raw).decode()
                    )
                )
        return CallToolResult(
            content=contents, structuredContent=data, isError=job.status in {"failed", "canceled"}
        )

    read_only = ToolAnnotations(readOnlyHint=True, destructiveHint=False, openWorldHint=False)
    generate_hints = ToolAnnotations(
        readOnlyHint=False, destructiveHint=False, idempotentHint=False, openWorldHint=False
    )

    @mcp.tool(annotations=read_only)
    async def get_image_service_info() -> dict[str, Any]:
        """Return model capabilities, validated argument limits and GPU worker state."""
        return {
            "model": MODEL_ID,
            "version": __version__,
            "model_loaded": manager.backend.loaded,
            "active_jobs": len(manager.cancels),
            "max_pending_jobs": settings.max_pending_jobs,
            "defaults": {"width": 1024, "height": 1024},
            "sampling": {
                "managed_by": "server",
                "seed": "random_each_request",
                "steps": DEFAULT_INFERENCE_STEPS,
                "guidance_scale": 0.0,
            },
            "limits": {
                "min_dimension": 256,
                "max_dimension": 2048,
                "dimension_multiple": 16,
                "max_pixels": settings.max_pixels,
                "prompt_tokens": 512,
            },
            "supports": ["text_to_image", "progress", "background_jobs", "cancel", "png"],
            "idle_unload_seconds": settings.idle_unload_seconds,
        }

    @mcp.tool(annotations=generate_hints)
    async def submit_image_generation(
        prompt: str, width: int = 1024, height: int = 1024
    ) -> dict[str, Any]:
        """Queue a text-to-image job and immediately return its job_id.

        Poll get_image_job until succeeded/failed/canceled. Width/height must be multiples of 16,
        within 256..2048 and the advertised pixel limit. Seed and sampling steps are server-managed.
        """
        return manager.public(submit(prompt, width, height))

    @mcp.tool(annotations=generate_hints)
    async def generate_image(
        prompt: str,
        ctx: Context,
        width: int = 1024,
        height: int = 1024,
    ) -> CallToolResult:
        """Generate a PNG, stream real progress, and return metadata plus an image resource link.

        Configure a long client read timeout (1800 seconds recommended). The first progress
        message contains job_id for recovery. Disconnection leaves the persisted job queryable;
        call cancel_image_job to explicitly cancel. See submit_image_generation for size limits.
        Send only the prompt and optional size; seed and sampling steps are server-managed.
        """
        job = submit(prompt, width, height)
        try:
            async for job in manager.events(job.id):
                await ctx.report_progress(
                    job.progress, total=100, message=f"job_id={job.id} {job.stage}: {job.message}"
                )
        except asyncio.CancelledError:
            # The detached worker owns generation; a dropped client must not lose its result.
            raise
        return result_content(job)

    @mcp.tool(annotations=read_only)
    async def get_image_job(job_id: str) -> dict[str, Any]:
        """Get persisted job status, real progress, arguments, errors and completed download URL."""
        return manager.public(get(job_id))

    @mcp.tool(annotations=read_only)
    async def get_image_result(job_id: str, inline: bool = False) -> CallToolResult:
        """Get a completed PNG link; inline=true also returns MCP image content for vision."""
        job = get(job_id)
        if job.status not in {"succeeded", "failed", "canceled"}:
            raise ToolError("Image is not ready; poll get_image_job")
        return result_content(job, inline)

    @mcp.tool(
        annotations=ToolAnnotations(
            readOnlyHint=False, destructiveHint=True, idempotentHint=True, openWorldHint=False
        )
    )
    async def cancel_image_job(job_id: str) -> dict[str, Any]:
        """Cancel at the next safe GPU boundary; completed results remain available."""
        get(job_id)
        return manager.public(manager.cancel(job_id))

    async def health(request: Request):
        return JSONResponse(
            {
                "status": "ok" if manager.worker and not manager.worker.done() else "unavailable",
                "version": __version__,
                "model": MODEL_ID,
                "model_files_present": (settings.model_path / "model_index.json").is_file(),
                "model_loaded": manager.backend.loaded,
                "active_jobs": len(manager.cancels),
            },
            status_code=200 if manager.worker and not manager.worker.done() else 503,
        )

    async def download(request: Request):
        try:
            job = manager.get(request.path_params["job_id"])
        except ValueError:
            return JSONResponse({"error": "Image not found"}, status_code=404)
        key = request.query_params.get("key", "")
        if job.status != "succeeded" or not secrets.compare_digest(
            key.encode(), job.download_token.encode()
        ):
            return JSONResponse({"error": "Image not found"}, status_code=404)
        path = manager.output_dir / f"{job.id}.png"
        if not path.is_file():
            return JSONResponse({"error": "Image no longer available"}, status_code=410)
        return FileResponse(
            path,
            media_type="image/png",
            filename=f"{job.id}.png",
            headers={
                "Cache-Control": "private, no-store",
                "Referrer-Policy": "no-referrer",
                "X-Content-Type-Options": "nosniff",
            },
        )

    mcp_app = mcp.streamable_http_app()

    @asynccontextmanager
    async def lifespan(app):
        await manager.start()
        try:
            async with mcp.session_manager.run():
                yield
        finally:
            await manager.close()

    app = Starlette(
        routes=[
            Route("/health", health),
            Route("/images/{job_id}.png", download),
            Mount("/", app=mcp_app),
        ],
        lifespan=lifespan,
    )
    app.state.manager = manager
    app.state.mcp = mcp
    app.add_middleware(BearerMiddleware, token=settings.api_token.get_secret_value())
    return app


class BearerMiddleware:
    def __init__(self, app, token: str):
        self.app, self.token = app, token

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http" and self.token and scope["path"].rstrip("/") == "/mcp":
            headers = dict(scope["headers"])
            expected = f"Bearer {self.token}".encode()
            if not secrets.compare_digest(headers.get(b"authorization", b""), expected):
                await JSONResponse({"error": "Unauthorized"}, status_code=401)(scope, receive, send)
                return
        await self.app(scope, receive, send)


def main():
    parser = argparse.ArgumentParser(description="Z-Image-Turbo MCP server (Streamable HTTP)")
    parser.add_argument("--host")
    parser.add_argument("--port", type=int)
    args = parser.parse_args()
    overrides = {k: v for k, v in vars(args).items() if v is not None}
    settings = Settings(**overrides)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    # Download URLs are bearer capabilities; avoid logging query strings in access logs.
    uvicorn.run(create_app(settings), host=settings.host, port=settings.port, access_log=False)


if __name__ == "__main__":
    main()
