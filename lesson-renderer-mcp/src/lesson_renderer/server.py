from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import os
import secrets
import shutil
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Literal
from urllib.parse import urlencode, urlsplit

import aiohttp
from mcp.server import MCPServer
from mcp.server.transport_security import TransportSecuritySettings
from mcp.types import CallToolResult, TextContent
from pydantic import Field
from starlette.requests import Request
from starlette.responses import FileResponse, JSONResponse

from .schema import AssetId, AudioBinding, LessonPlan
from .artifact_input import fetch_artifact
from .drafts import DraftId, DraftPatch, DraftStore, Pointer
from .store import FILES, ObjectLookupError, digest, encoded, file_digest, job_renderer_snapshot, object_path, read_json, renderer_revision, write_json


@dataclass(frozen=True)
class Settings:
    root: Path
    skill: Path
    asset_base: str
    public_base: str
    token: str
    ttl: int = 0
    artifact_base: str | None = None

    @classmethod
    def from_env(cls):
        token = Path(os.environ["LESSON_TOKEN_FILE"]).read_text().strip()
        if len(token) < 32:
            raise ValueError("lesson MCP requires a private token of at least 32 characters")
        bases = [os.environ["LESSON_ASSET_BASE"].rstrip("/"), os.environ["LESSON_PUBLIC_BASE"].rstrip("/")]
        artifact_base = os.getenv("LESSON_ARTIFACT_BASE", bases[0]).rstrip("/")
        for base in [*bases, artifact_base]:
            url = urlsplit(base)
            if url.scheme not in {"http", "https"} or not url.hostname or url.username or url.password or url.query or url.fragment:
                raise ValueError("invalid configured asset/public base URL")
        ttl = int(os.getenv("LESSON_ARTIFACT_TTL_SECONDS", "0"))
        if ttl < 0:
            raise ValueError("artifact TTL must be >= 0; 0 means permanent")
        return cls(Path(os.getenv("LESSON_DATA_DIR", "/data")), Path(os.getenv("LESSON_SKILL_DIR", "/opt/lesson-skill")), *bases, token, ttl, artifact_base)


class Runtime:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.plans = settings.root / "plans"
        self.jobs = settings.root / "jobs"
        self.plans.mkdir(parents=True, exist_ok=True)
        self.jobs.mkdir(parents=True, exist_ok=True)
        # Single controller process; the separate worker owns job execution.
        self.submit_lock = asyncio.Lock()
        self.drafts = DraftStore(self)

    def prepare(self, lesson: LessonPlan, *, lesson_id: str | None = None, snapshot: Path | None = None) -> dict:
        lesson_id = lesson_id or "lesson_" + secrets.token_hex(32)
        path = self.plans / lesson_id
        plan = lesson.model_dump()
        source = snapshot or self.settings.skill
        revision = renderer_revision(source)
        if path.exists():
            object_path(self.plans, lesson_id, "lesson_")
            if read_json(path / "lesson.json") != plan or renderer_revision(path / "snapshot") != revision:
                raise ValueError("prepared lesson identity conflict")
        else:
            # Publish a complete immutable plan atomically. Replaying a draft
            # commit after a crash reuses the same lesson ID and speech bindings.
            with tempfile.TemporaryDirectory(prefix=".prepare-", dir=self.plans) as temporary:
                staged = Path(temporary) / "plan"
                staged.mkdir(mode=0o700)
                shutil.copytree(source / "scripts", staged / "snapshot" / "scripts",
                                ignore=shutil.ignore_patterns("__pycache__", "tests"))
                if renderer_revision(staged / "snapshot") != revision:
                    raise ValueError("renderer changed during preparation; retry prepare")
                write_json(staged / "lesson.json", plan)
                write_json(staged / "metadata.json", {"renderer_revision": revision, "plan_sha256": digest(encoded(plan))})
                staged.rename(path)
                fd = os.open(self.plans, os.O_DIRECTORY)
                try: os.fsync(fd)
                finally: os.close(fd)
        return self.prepared(lesson_id)

    def prepared(self, lesson_id: str) -> dict:
        path = object_path(self.plans, lesson_id, "lesson_")
        plan = read_json(path / "lesson.json")
        metadata = read_json(path / "metadata.json")
        if digest(encoded(plan)) != metadata["plan_sha256"]:
            raise ValueError("saved lesson plan integrity check failed")
        lesson = LessonPlan.model_validate(plan)
        revision = metadata["renderer_revision"]
        speech_requests = [{"segment_id": s.id, "arguments": {"text": s.narration, "filename": f"{s.id}.mp3", "artifact_visibility": "internal"}} for s in lesson.segments]
        return {"lesson_id": lesson_id, "segment_count": len(lesson.segments), "renderer_revision": revision,
                "speech_requests": speech_requests,
                "next": "Synthesize each narration with the configured TTS tool. Submit segment_id with artifact_url equal to the exact returned audio.url, OR the original attachment_id if TTS returned a legacy attachment. Never convert artifact- IDs to att_ IDs, compute hashes or regenerate successful audio.",
                "_meta": {"exoharness": {"context_checkpoint": "lesson plan saved: " + lesson_id, "workflow_continuation": {
                    "version": 1, "status": "required", "required_tool": "submit_lesson_render",
                    "context_scope": "checkpoint", "repair_context": {"speech_requests": speech_requests},
                    "prerequisite_tools": [{"capability_id": "tool_node:global:_:model-speech-synthesis", "tool": "model.synthesize_speech"}],
                    "required_arguments": {"lesson_id": lesson_id},
                    "checkpoint": {"lesson_id": lesson_id},
                    "reason": "The lesson plan is saved, not rendered. Retain exact successful audio.url or attachment_id, then submit the render using its live AudioBinding schema. Do not rewrite IDs, prepare a second lesson or stop at a text-only summary."
                }}}}

    async def fetch(self, session, attachment_id: str, destination: Path, allowed: set[str]) -> dict:
        # Only opaque IDs at the administrator-configured asset origin. No arbitrary URLs,
        # redirects, cookies, or caller-controlled headers/paths.
        if not __import__("re").fullmatch(r"att_[0-9a-f]{64}", attachment_id):
            raise ValueError("invalid attachment ID")
        base = f"{self.settings.asset_base}/v1/attachment-assets/{attachment_id}"
        async with session.get(base + "/metadata", allow_redirects=False) as response:
            if response.status != 200:
                raise ValueError(f"attachment metadata unavailable (HTTP {response.status}); use the returned attachment_id")
            raw = await response.content.read(65537)
            if len(raw) > 65536:
                raise ValueError("attachment metadata too large")
            metadata = __import__("json").loads(raw)
        if metadata.get("attachment_id") != attachment_id or metadata.get("content_type") not in allowed:
            raise ValueError("attachment identity/type mismatch")
        size = metadata.get("size_bytes")
        if type(size) is not int or not 0 < size <= 32 * 1024 * 1024:
            raise ValueError("attachment exceeds 32 MiB")
        if destination.is_file() and file_digest(destination) == metadata.get("sha256"):
            return metadata
        temp = destination.with_suffix(".partial")
        count = 0
        try:
            async with session.get(base, allow_redirects=False) as response:
                if response.status != 200:
                    raise ValueError(f"attachment unavailable (HTTP {response.status})")
                with temp.open("wb") as stream:
                    async for block in response.content.iter_chunked(65536):
                        count += len(block)
                        if count > size:
                            raise ValueError("attachment larger than declared")
                        stream.write(block)
                    stream.flush()
                    os.fsync(stream.fileno())
            if count != size or file_digest(temp) != metadata.get("sha256"):
                raise ValueError("attachment integrity check failed")
            temp.replace(destination)
        finally:
            temp.unlink(missing_ok=True)
        return metadata

    def refresh_failed_renderer(self, job: Path) -> None:
        """Explicit repair: retain old code/request, audio and plan unchanged."""
        previous = read_json(job / "request.json")
        if renderer_revision(job_renderer_snapshot(job, previous)) != previous["renderer_revision"]:
            raise ValueError("existing renderer snapshot integrity check failed")
        if digest(encoded(read_json(job / "lesson.json"))) != previous["plan_sha256"]:
            raise ValueError("saved lesson plan integrity check failed")
        revision = renderer_revision(self.settings.skill)
        if revision == previous["renderer_revision"]:
            return
        revisions = job / "renderer-revisions"
        revisions.mkdir(exist_ok=True)
        if revisions.is_symlink():
            raise ValueError("invalid renderer revisions directory")
        target = revisions / revision
        if not target.exists():
            with tempfile.TemporaryDirectory(prefix=".prepare-", dir=revisions) as temporary:
                snapshot = Path(temporary) / "snapshot"
                shutil.copytree(self.settings.skill / "scripts", snapshot / "scripts",
                                ignore=shutil.ignore_patterns("__pycache__", "tests"))
                if renderer_revision(snapshot) != revision:
                    raise ValueError("renderer changed during repair; retry explicitly")
                snapshot.rename(target)
        updated = {**previous, "renderer_revision": revision,
                   "renderer_snapshot": f"renderer-revisions/{revision}"}
        if renderer_revision(job_renderer_snapshot(job, updated)) != revision:
            raise ValueError("replacement renderer snapshot integrity check failed")
        # Evidence precedes the atomic request-pointer commit. Interrupted repair
        # leaves a failed job, never an implicit requeue or altered original code.
        evidence = {"previous_request": previous, "next_request": updated,
                    "previous_state": read_json(job / "state.json"),
                    "reason": "explicit retry_failed with refresh_renderer", "at": time.time()}
        write_json(job / "renderer-repairs" / (secrets.token_hex(16) + ".json"), evidence)
        write_json(job / "request.json", updated)

    async def submit(self, lesson_id: str, bindings: list[AudioBinding], quality: str, fps: int,
                     retry_failed: bool, refresh_renderer: bool = False) -> dict:
        async with self.submit_lock:
            if refresh_renderer and not retry_failed:
                raise ValueError("refresh_renderer requires explicit retry_failed=true")
            plan_dir = object_path(self.plans, lesson_id, "lesson_")
            lesson = LessonPlan.model_validate(read_json(plan_dir / "lesson.json"))
            ids = {s.id for s in lesson.segments}
            by_id = {b.segment_id: b.attachment_id or b.artifact_url for b in bindings}
            binding_by_id = {b.segment_id: b for b in bindings}
            if len(by_id) != len(bindings) or set(by_id) != ids:
                raise ValueError("audio bindings must contain every segment exactly once")
            if quality not in {"l", "m", "h"} or fps not in {15, 24, 25, 30, 50, 60}:
                raise ValueError("invalid quality/fps")
            identity = {"lesson_id": lesson_id, "audio": by_id, "quality": quality, "fps": fps}
            job_id = "lessonjob_" + digest(encoded(identity))
            job = self.jobs / job_id
            job.mkdir(mode=0o700, exist_ok=True)
            if (job / "state.json").exists():
                state = read_json(job / "state.json")
                if state["status"] not in {"failed", "cancelled"} or not retry_failed:
                    return self.status(job_id, True)
                if refresh_renderer:
                    self.refresh_failed_renderer(job)
                (job / "cancel").unlink(missing_ok=True)
                write_json(job / "attempts.json", {"count": 0})
                write_json(job / "state.json", {"status": "queued", "progress": 0, "message": "已复用原配音，等待重新渲染"})
                return self.status(job_id, True)
            if refresh_renderer:
                raise ValueError("refresh_renderer requires an existing failed or cancelled job")
            records = []
            timeout = aiohttp.ClientTimeout(total=180, connect=10, sock_read=30)
            async with aiohttp.ClientSession(timeout=timeout, trust_env=False) as session:
                for segment in lesson.segments:
                    attachment_id = by_id[segment.id]
                    filename = segment.id + ".audio"
                    binding = binding_by_id[segment.id]
                    if binding.artifact_url:
                        metadata = await fetch_artifact(session, binding.artifact_url, self.settings.artifact_base or self.settings.asset_base, job / filename, {"audio/mpeg"})
                    else:
                        metadata = await self.fetch(session, attachment_id, job / filename, {"audio/mpeg"})
                    provenance = metadata.get("provenance") or {}
                    text_hash = digest(segment.narration.encode())
                    if provenance.get("input_string_sha256", {}).get("text") != text_hash:
                        raise ValueError(f"{segment.id}: audio does not match this narration. Bind the correct TTS attachment; only resynthesize if narration changed.")
                    records.append({"id": segment.id, "narration_sha256": text_hash,
                                    "invocation_id": provenance.get("invocation_id"),
                                    "artifact": {"id": metadata.get("artifact_id", attachment_id), "local_path": filename, "sha256": metadata["sha256"]}})
            metadata = read_json(plan_dir / "metadata.json")
            shutil.copytree(plan_dir / "snapshot", job / "snapshot", dirs_exist_ok=True)
            if renderer_revision(job / "snapshot") != metadata["renderer_revision"]:
                raise ValueError("renderer snapshot integrity check failed")
            write_json(job / "lesson.json", lesson.model_dump())
            write_json(job / "audio_manifest.json", {"schema_version": 1, "segments": records})
            write_json(job / "request.json", {**identity, **metadata, "kind": "render"})
            write_json(job / "state.json", {"status": "queued", "progress": 0, "message": "配音已绑定，等待教学视频渲染"})
            return self.status(job_id, True)

    async def inspect_image(self, attachment_id: str, language: str = "zh-CN") -> dict:
        async with self.submit_lock:
            job_id = "lessonjob_" + hmac.new(self.settings.token.encode(), f"ocr:{language}:{attachment_id}".encode(), hashlib.sha256).hexdigest()
            job = self.jobs / job_id
            job.mkdir(mode=0o700, exist_ok=True)
            if not (job / "state.json").exists():
                async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=120, connect=10, sock_read=30), trust_env=False) as session:
                    await self.fetch(session, attachment_id, job / "problem.image", {"image/png", "image/jpeg", "image/webp"})
                write_json(job / "request.json", {"kind": "ocr", "ocr_language": "chi_sim" if language == "zh-CN" else "eng"})
                write_json(job / "state.json", {"status": "queued", "progress": 0, "message": "等待识别题目文字"})
            return self.status(job_id, True)

    def artifact_url(self, job_id: str, filename: str, expires: int) -> str:
        signature = hmac.new(self.settings.token.encode(), f"lesson-file:{job_id}:{filename}:{expires}".encode(), hashlib.sha256).hexdigest()
        return f"{self.settings.public_base}/lessons/{job_id}/{filename}?" + urlencode({"expires": expires, "signature": signature})

    def status(self, job_id: str, observe: bool = False) -> dict:
        job = object_path(self.jobs, job_id, "lessonjob_")
        state = read_json(job / "state.json")
        status = state["status"]
        output = {"job_id": job_id, "status": status, "progress": state.get("progress", 0), "message": state.get("message", "")}
        if (job / "cancel").exists() and status not in {"completed", "failed", "cancelled"}:
            output["message"] = "正在停止渲染；等待工作进程确认"
        if status == "working" and state.get("started_at"):
            output["elapsed_seconds"] = max(0, int(time.time() - state["started_at"]))
        if status == "completed":
            result = read_json(job / "result.json")
            if result.get("kind") == "ocr":
                output["result"] = result
            else:
                # Expiry is persisted at first successful delivery; normal polling cannot extend it.
                delivery = job / "delivery.json"
                if not delivery.exists():
                    write_json(delivery, {"expires": int(time.time()) + self.settings.ttl if self.settings.ttl else 0})
                expires = read_json(delivery)["expires"]
                output["result"] = result["validation"]
                output["artifacts"] = [{**item, "download_url": self.artifact_url(job_id, item["filename"], expires),
                                        "_meta": {"exoharness": {"artifact_visibility": "chat" if item["filename"] in {"final.mp4", "speech.wav"} else "internal"}}}
                                       for item in result["files"]]
        if status == "failed":
            output["error"] = state.get("error", "render failed; inspect job evidence")
            output["recovery"] = "Correct only the failing step; existing TTS audio is retained. Explicit retry_failed=true rerenders this job without calling TTS."
            request = read_json(job / "request.json")
            if request.get("kind") == "render" and renderer_revision(self.settings.skill) != request["renderer_revision"]:
                output["recovery"] += " An administrator-installed renderer update is available. To explicitly adopt it, also set refresh_renderer=true; old snapshots are retained. Do not prepare a new plan or synthesize audio again."
        if observe:
            terminal = status if status in {"completed", "failed", "cancelled"} else "working"
            output["_meta"] = {"exoharness": {"background_task": {
                "version": 1, "id": job_id, "status": terminal,
                "poll": {"tool": "get_lesson_render", "arguments": {"job_id": job_id, "observe": True}},
                "cancel": {"tool": "cancel_lesson_render", "arguments": {"job_id": job_id}},
                "poll_interval_ms": 2500,
                "progress": {"progress": output["progress"], "total": 100, "message": output["message"]}}}}
        if status == "completed" and result.get("kind") != "ocr":
            meta = output.setdefault("_meta", {}).setdefault("exoharness", {})
            meta["context_checkpoint"] = "narrated video persisted: " + job_id
            meta["workflow_continuation"] = {"version": 1, "status": "complete", "context_scope": "checkpoint"}
        return output


mcp = MCPServer("Lesson Renderer", version="0.1.0", instructions="Render structured elementary lesson visuals with existing TTS artifacts. No TTS provider calls or arbitrary code. Prepare once, bind actual audio, submit once; wait for the declared background task to return the final narrated video.")
_runtime: Runtime | None = None


def runtime() -> Runtime:
    global _runtime
    if _runtime is None:
        _runtime = Runtime(Settings.from_env())
    return _runtime


def protocol_result(payload: dict) -> CallToolResult:
    # A dict return annotation is only text in this SDK version. Put data and
    # task metadata in their real MCP fields, not inside an encoded text block.
    structured = {key: value for key, value in payload.items() if key != "_meta"}
    return CallToolResult(content=[TextContent(text=encoded(structured).decode())],
                          structuredContent=structured, _meta=payload.get("_meta"))


@mcp.tool()
def prepare_lesson(lesson: Annotated[dict[str, Any], Field(description=
    "The lesson value IS the plan: title, question, subject, grade, segments; do not nest another lesson wrapper. "
    "The following contract describes valid authoring data, but validation runs AFTER persistence so mistakes remain locally repairable. "
    "Use only declared fields. Optional style is a color object, not prose. Authoring JSON Schema: "
    + json.dumps(LessonPlan.model_json_schema(), ensure_ascii=False, separators=(",", ":")))]) -> CallToolResult:
    """Persist one whole lesson draft BEFORE TTS. Valid drafts return status=prepared, lesson_id and speech_requests. Invalid drafts are retained with status=needs_repair, draft_id, revision and exact field errors: patch that draft, never regenerate the whole plan. Only trusted visual components, no code or guessed audio durations."""
    return protocol_result(runtime().drafts.create(lesson))


@mcp.tool()
def get_lesson_draft(draft_id: DraftId, paths: Annotated[list[Pointer], Field(max_length=8)] = [],
                     error_offset: Annotated[int, Field(ge=0)] | None = None,
                     at_revision: Annotated[int, Field(ge=1)] | None = None) -> CallToolResult:
    """Recover draft revision/errors. Read exact JSON Pointer paths and constraints only when current evidence is insufficient. at_revision retrieves original fields from a saved revision without rolling back the current draft (e.g. recover mistakenly deleted objects). Retains the last 8 focused paths, at most 16 KiB; no full-plan dump. Errors are paged 8 at a time; omitted error_offset retains the current page. Does not call TTS or render."""
    return protocol_result(runtime().drafts.get(draft_id, paths, error_offset, at_revision))


@mcp.tool()
def patch_lesson_draft(draft_id: DraftId, expected_revision: Annotated[int, Field(ge=1)],
                       patches: Annotated[list[DraftPatch], Field(min_length=1, max_length=32)]) -> CallToolResult:
    """Atomically patch only incorrect fields of the saved draft using JSON Pointers (relative to lesson, no /lesson prefix). Native JSON values; move uses from_path. Cannot replace the root/existing segments array; a missing segments field can be restored with add/move. Revision conflicts do not modify data; identical replay applies once. Revalidates the whole plan: TTS requests only on status=prepared."""
    return protocol_result(runtime().drafts.patch(draft_id, expected_revision, patches))


@mcp.tool()
def get_prepared_lesson(lesson_id: Annotated[str, Field(description="Exact opaque lesson_ ID returned by a successful prepare_lesson, never a title or invented slug.")]) -> CallToolResult:
    """Read an EXISTING saved lesson using its returned lesson_id. Does not create a lesson. If no preparation succeeded, first call prepare_lesson with the lesson plan; do not invent an ID. Reuse existing narration requests without repeating successful TTS/rendering."""
    try:
        return protocol_result(runtime().prepared(lesson_id))
    except ObjectLookupError as error:
        message = (f"Cannot read prepared lesson: {error}. lesson_id must be the exact lesson_ ID "
                   "returned by successful prepare_lesson, not a title or guessed slug. "
                   "If this task has no saved lesson, call prepare_lesson with the lesson plan first. "
                   "If preparation previously succeeded, recover that returned ID; do not recreate it. "
                   "Retrying this same lookup unchanged will not help. No TTS or rendering was started.")
        return CallToolResult(content=[TextContent(text=message)], isError=True,
                              structuredContent={"error": {"code": error.code, "field": "lesson_id",
                                                           "retryable": False, "message": message}})


@mcp.tool()
async def submit_lesson_render(lesson_id: str, audio: Annotated[list[AudioBinding], Field(min_length=1, max_length=40)],
                               quality: Literal["l", "m", "h"] = "m", fps: Literal[15, 24, 25, 30, 50, 60] = 30,
                               retry_failed: bool = False, refresh_renderer: bool = False) -> CallToolResult:
    """Bind each segment to the exact TTS return: artifact_url=audio.url for signed Artifact files, OR attachment_id for legacy attachments. Never convert artifact- IDs to att_ IDs. Render narrated MP4; same bindings reuse the job. retry_failed rerenders retained audio without new TTS. refresh_renderer with retry_failed explicitly adopts renderer fixes."""
    return protocol_result(await runtime().submit(lesson_id, audio, quality, fps, retry_failed, refresh_renderer))


@mcp.tool()
def get_lesson_render(job_id: str, observe: bool = False) -> CallToolResult:
    """Read actual progress/results; one-shot by default. observe=true lets the host continue observing the existing job without LLM polling."""
    return protocol_result(runtime().status(job_id, observe))


@mcp.tool()
def cancel_lesson_render(job_id: str) -> CallToolResult:
    """Request process-group cancellation; only report cancelled after the worker confirms it stopped."""
    rt = runtime()
    job = object_path(rt.jobs, job_id, "lessonjob_")
    if read_json(job / "state.json")["status"] not in {"completed", "failed", "cancelled"}:
        (job / "cancel").touch(mode=0o600, exist_ok=True)
    return protocol_result(rt.status(job_id, True))


@mcp.tool()
async def inspect_problem_image(attachment_id: AssetId, language: Literal["zh-CN", "en-US"] = "zh-CN") -> CallToolResult:
    """Read printed Chinese/English problem text from an uploaded image using local OCR. Not a diagram/handwriting reasoning model. Verify ambiguous numbers/symbols with the user before solving; do not invent missing conditions."""
    return protocol_result(await runtime().inspect_image(attachment_id, language))


@mcp.custom_route("/health", methods=["GET"])
async def health(_request: Request):
    heartbeat = runtime().jobs / "worker.json"
    alive = heartbeat.is_file() and time.time() - heartbeat.stat().st_mtime < 30
    return JSONResponse({"status": "ok" if alive else "worker_unavailable", "worker": alive}, status_code=200 if alive else 503)


@mcp.custom_route("/lessons/{job_id}/{filename}", methods=["GET", "HEAD"])
async def artifact(request: Request):
    rt = runtime()
    job_id, filename = request.path_params["job_id"], request.path_params["filename"]
    try:
        if filename not in FILES:
            raise ValueError("unknown file")
        job = object_path(rt.jobs, job_id, "lessonjob_")
        expires = int(request.query_params.get("expires", "-1"))
        if expires < 0 or (expires != 0 and expires < time.time()):
            raise ValueError("expired")
        expected = rt.artifact_url(job_id, filename, expires).split("signature=", 1)[1]
        if not hmac.compare_digest(expected, request.query_params.get("signature", "")):
            raise ValueError("unauthorized")
        if read_json(job / "state.json")["status"] != "completed":
            raise ValueError("not completed")
        item = next(f for f in read_json(job / "result.json")["files"] if f["filename"] == filename)
        path = job / "published" / filename
        if path.is_symlink() or path.stat().st_size != item["size_bytes"]:
            raise ValueError("invalid file")
    except (ValueError, OSError, KeyError, StopIteration):
        return JSONResponse({"error": "file unavailable"}, status_code=404)
    return FileResponse(path, media_type=FILES[filename], filename=filename,
                        content_disposition_type="inline", headers={"Cache-Control": "private, no-store", "Referrer-Policy": "no-referrer", "X-Content-Type-Options": "nosniff"})


class BearerAuth:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http" and (scope.get("path", "") == "/mcp" or scope.get("path", "").startswith("/mcp/")):
            provided = dict(scope.get("headers", [])).get(b"authorization", b"").decode("latin1")
            if not hmac.compare_digest(provided, "Bearer " + runtime().settings.token):
                await JSONResponse({"error": "unauthorized"}, status_code=401)(scope, receive, send)
                return
        await self.app(scope, receive, send)


def main():
    import uvicorn
    runtime()
    hosts = os.getenv("LESSON_ALLOWED_HOSTS", "127.0.0.1:*,localhost:*").split(",")
    app = mcp.streamable_http_app(streamable_http_path="/mcp", json_response=False,
        transport_security=TransportSecuritySettings(enable_dns_rebinding_protection=True, allowed_hosts=hosts))
    app.add_middleware(BearerAuth)
    uvicorn.run(app, host=os.getenv("LESSON_HOST", "0.0.0.0"), port=int(os.getenv("LESSON_PORT", "8793")), access_log=False)


if __name__ == "__main__":
    main()
