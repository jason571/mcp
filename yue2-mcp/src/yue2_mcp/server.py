import argparse
import asyncio
import importlib.util
import json
import logging
import secrets
import shutil
from contextlib import asynccontextmanager
from typing import Any, Literal
from urllib.parse import urlsplit

import uvicorn
from mcp.server.fastmcp import Context, FastMCP
from mcp.server.fastmcp.exceptions import ToolError
from mcp.server.transport_security import TransportSecuritySettings
from mcp.types import CallToolResult, ResourceLink, TextContent, ToolAnnotations
from pydantic import ValidationError
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import FileResponse, JSONResponse
from starlette.routing import Mount, Route

from . import __version__
from .backend import YuEBackend
from .config import Settings
from .jobs import JobManager
from .models import TERMINAL, LyricsRequest, MusicRequest

FILES = {
    "song.flac": "audio/flac", "preview.mp3": "audio/mpeg",
    "lyrics.txt": "text/plain; charset=utf-8",
    "score.abc": "text/plain; charset=utf-8", "metadata.json": "application/json",
}


def model_files_present(settings: Settings) -> tuple[bool, bool]:
    model = all((settings.model_path / name).is_file()
                for name in ("config.json", "model.safetensors", "qwen.tiktoken"))
    vae = all((settings.vae_path / name).is_file()
              for name in ("config.json", "model.safetensors"))
    return model, vae


def create_app(settings: Settings | None = None, backend=None):
    settings = settings or Settings()
    manager = JobManager(settings, backend or YuEBackend(settings))
    origin = urlsplit(settings.public_url)
    mcp = FastMCP(
        "YuE2 Music",
        instructions=(
            "Create original songs with YuE2. Provide style and lyrics, or lyrics_prompt if "
            "the lyrics LLM is configured. Use submit_music_generation then get_music_job for "
            "long tasks; use generate_music for MCP progress. Music jobs return one song "
            "in two encodings: a playable MP3 and original-quality 48 kHz FLAC. "
            "These are not two different songs; present one audio player by default. "
            "A separately authorized image generator may optionally create cover art, "
            "but cover failure must not block delivery of the completed song. "
            "Use generate_lyrics to draft lyrics first and "
            "edit them before music generation. A dropped connection does not cancel a job."
        ),
        stateless_http=True,
        json_response=False,
        transport_security=TransportSecuritySettings(
            allowed_hosts=[origin.netloc, f"{settings.host}:{settings.port}",
                           "127.0.0.1:*", "localhost:*", "[::1]:*"],
            allowed_origins=[settings.public_url, "http://127.0.0.1:*", "http://localhost:*"],
        ),
    )

    def submit_music(style, lyrics, lyrics_prompt, language, cot, abc, seed, cfg_scale):
        try:
            request = MusicRequest(style=style, lyrics=lyrics, lyrics_prompt=lyrics_prompt,
                                   language=language, cot=cot, abc=abc, seed=seed,
                                   cfg_scale=cfg_scale)
            return manager.submit("music", request)
        except (ValueError, ValidationError) as exc:
            raise ToolError(str(exc)) from exc

    def get(job_id):
        try:
            return manager.get(job_id)
        except ValueError as exc:
            raise ToolError(str(exc)) from exc

    def content(job, audio_format: Literal["mp3", "flac"] = "mp3"):
        data = manager.public(job)
        # The original request remains durable in the job store. Repeating lyrics
        # in every status response needlessly expands the agent's context.
        data.pop("request", None)
        blocks = [TextContent(type="text", text=json.dumps(data, ensure_ascii=False))]
        if job.status == "succeeded" and job.kind == "music":
            if audio_format == "mp3" and data["result"]["preview_url"]:
                blocks.append(ResourceLink(
                    type="resource_link", uri=data["result"]["preview_url"],
                    name=f"{job.id}-preview.mp3", mimeType="audio/mpeg",
                    description="Playable MP3 preview",
                ))
            else:
                blocks.append(ResourceLink(type="resource_link", uri=data["result"]["audio_url"],
                                           name=f"{job.id}.flac", mimeType="audio/flac",
                                           description="Original quality FLAC download"))
            blocks.append(ResourceLink(type="resource_link", uri=data["result"]["lyrics_url"],
                                       name=f"{job.id}-lyrics.txt", mimeType="text/plain",
                                       description="Song lyrics download"))
        return CallToolResult(content=blocks, structuredContent=data,
                              isError=job.status in {"failed", "canceled"})

    def background_content(job):
        result = content(job)
        status = {"succeeded": "completed", "failed": "failed",
                  "canceled": "cancelled"}.get(job.status, "working")
        result.structuredContent["_meta"] = {"exoharness": {"background_task": {
            "version": 1,
            "id": job.id,
            "status": status,
            "poll": {"tool": "get_music_job", "arguments": {"job_id": job.id}},
            "cancel": {"tool": "cancel_music_job", "arguments": {"job_id": job.id}},
            "poll_interval_ms": 2500,
            "progress": {"progress": job.progress, "total": 100,
                         "message": f"{job.stage}: {job.message}"},
        }}}
        return result

    read_only = ToolAnnotations(readOnlyHint=True, destructiveHint=False, openWorldHint=False)
    create = ToolAnnotations(readOnlyHint=False, destructiveHint=False,
                             idempotentHint=False, openWorldHint=False)

    @mcp.tool(annotations=read_only)
    async def get_music_service_info() -> dict[str, Any]:
        """Report local model, VAE, lyrics provider and queue readiness."""
        model_ready, vae_ready = model_files_present(settings)
        runtime_ready = importlib.util.find_spec("yue2") is not None
        ffmpeg_ready = shutil.which("ffmpeg") is not None
        return {
            "version": __version__, "model": "m-a-p/YuE2-3B",
            "model_files_present": model_ready,
            "vae_files_present": vae_ready,
            "runtime_package_present": runtime_ready,
            "ffmpeg_present": ffmpeg_ready,
            "model_loaded": manager.backend.loaded,
            "lyrics_configured": bool(settings.lyrics_base_url),
            "active_jobs": len(manager.cancels), "max_pending_jobs": settings.max_pending_jobs,
            "idle_unload_seconds": settings.idle_unload_seconds,
            "formats": ["flac", "mp3", "txt", "abc"],
            "cot_modes": ["full", "melody", "off"],
        }

    @mcp.tool(annotations=create)
    async def submit_music_generation(
        style: str, lyrics: str | None = None, lyrics_prompt: str | None = None,
        language: str = "Chinese", cot: str = "full", abc: str | None = None,
        seed: int | None = None, cfg_scale: float | None = None,
    ) -> CallToolResult:
        """Queue a YuE2 song. Supply lyrics or lyrics_prompt; optional ABC edits the score."""
        job = submit_music(style, lyrics, lyrics_prompt, language, cot, abc, seed,
                           cfg_scale)
        return background_content(job)

    @mcp.tool(annotations=create)
    async def generate_music(
        style: str, ctx: Context, lyrics: str | None = None,
        lyrics_prompt: str | None = None, language: str = "Chinese",
        cot: str = "full", abc: str | None = None, seed: int | None = None,
        cfg_scale: float | None = None,
    ) -> CallToolResult:
        """Queue one song, stream progress, show one MP3 player and retain FLAC metadata."""
        job = submit_music(style, lyrics, lyrics_prompt, language, cot, abc, seed,
                           cfg_scale)
        try:
            async for job in manager.events(job.id):
                await ctx.report_progress(job.progress, total=100,
                                          message=f"job_id={job.id} {job.stage}: {job.message}")
        except asyncio.CancelledError:
            raise
        return content(job)

    @mcp.tool(annotations=create)
    async def submit_lyrics_generation(
        prompt: str, style: str = "", language: str = "Chinese"
    ) -> CallToolResult:
        """Queue original lyrics through the configured OpenAI-compatible text model."""
        try:
            return background_content(manager.submit(
                "lyrics", LyricsRequest(prompt=prompt, style=style, language=language)
            ))
        except (ValueError, ValidationError) as exc:
            raise ToolError(str(exc)) from exc

    @mcp.tool(annotations=create)
    async def generate_lyrics(
        prompt: str, ctx: Context, style: str = "", language: str = "Chinese"
    ) -> CallToolResult:
        """Generate original sectioned lyrics and report queue progress."""
        try:
            job = manager.submit("lyrics", LyricsRequest(prompt=prompt, style=style,
                                                           language=language))
        except (ValueError, ValidationError) as exc:
            raise ToolError(str(exc)) from exc
        async for job in manager.events(job.id):
            await ctx.report_progress(job.progress, total=100,
                                      message=f"job_id={job.id} {job.stage}: {job.message}")
        return content(job)

    @mcp.tool(annotations=read_only)
    async def get_music_job(job_id: str) -> CallToolResult:
        """Get a queued, running or completed music or lyrics job, including its progress."""
        return background_content(get(job_id))

    @mcp.tool(annotations=read_only)
    async def get_music_result(
        job_id: str, audio_format: Literal["mp3", "flac"] = "mp3"
    ) -> CallToolResult:
        """Return one playable song format; request flac for the original-quality download."""
        job = get(job_id)
        if job.status not in TERMINAL:
            raise ToolError("Job is not ready; poll get_music_job")
        return content(job, audio_format)

    @mcp.tool(annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=True,
                                          idempotentHint=True, openWorldHint=False))
    async def cancel_music_job(job_id: str) -> CallToolResult:
        """Cancel a queued job or request interruption at the next safe model boundary."""
        get(job_id)
        return background_content(manager.cancel(job_id))

    async def health(request: Request):
        live = bool(manager.worker and not manager.worker.done())
        model_ready, vae_ready = model_files_present(settings)
        runtime_ready = importlib.util.find_spec("yue2") is not None
        ffmpeg_ready = shutil.which("ffmpeg") is not None
        ready = model_ready and vae_ready and runtime_ready and ffmpeg_ready
        return JSONResponse({"status": "ok" if live and ready else "unavailable",
                             "worker_alive": live, "model_files_present": model_ready,
                             "vae_files_present": vae_ready,
                             "runtime_package_present": runtime_ready,
                             "ffmpeg_present": ffmpeg_ready,
                             "model_loaded": manager.backend.loaded,
                             "lyrics_configured": bool(settings.lyrics_base_url)},
                            status_code=200 if live and ready else 503)

    async def download(request: Request):
        name = request.path_params["name"]
        if name not in FILES:
            return JSONResponse({"error": "Not found"}, status_code=404)
        try:
            job = manager.get(request.path_params["job_id"])
        except ValueError:
            return JSONResponse({"error": "Not found"}, status_code=404)
        if job.status != "succeeded" or job.kind != "music" or not secrets.compare_digest(
            request.query_params.get("key", "").encode(), job.download_token.encode()
        ):
            return JSONResponse({"error": "Not found"}, status_code=404)
        path = manager.output_dir / job.id / name
        if not path.is_file():
            return JSONResponse({"error": "File unavailable"}, status_code=410)
        return FileResponse(path, media_type=FILES[name], filename=name,
                            headers={"Cache-Control": "private, no-store",
                                     "Referrer-Policy": "no-referrer",
                                     "X-Content-Type-Options": "nosniff"})

    @asynccontextmanager
    async def lifespan(app):
        await manager.start()
        try:
            async with mcp.session_manager.run():
                yield
        finally:
            await manager.close()

    app = Starlette(routes=[Route("/health", health),
                            Route("/files/{job_id}/{name}", download),
                            Mount("/", app=mcp.streamable_http_app())], lifespan=lifespan)
    app.state.manager = manager
    app.state.mcp = mcp
    app.add_middleware(BearerMiddleware, token=settings.api_token.get_secret_value())
    return app


class BearerMiddleware:
    def __init__(self, app, token: str):
        self.app, self.token = app, token

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http" and self.token and scope["path"].rstrip("/") == "/mcp":
            expected = f"Bearer {self.token}".encode()
            if not secrets.compare_digest(dict(scope["headers"]).get(b"authorization", b""),
                                          expected):
                await JSONResponse({"error": "Unauthorized"}, status_code=401)(scope, receive, send)
                return
        await self.app(scope, receive, send)


def main():
    parser = argparse.ArgumentParser(description="YuE2 MCP Streamable HTTP server")
    parser.add_argument("--host")
    parser.add_argument("--port", type=int)
    args = parser.parse_args()
    settings = Settings(**{k: v for k, v in vars(args).items() if v is not None})
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    uvicorn.run(create_app(settings), host=settings.host, port=settings.port, access_log=False)


if __name__ == "__main__":
    main()
