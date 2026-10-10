from __future__ import annotations

import hashlib
import hmac
import json
import os
import time
import threading
from functools import wraps
from pathlib import Path
from typing import Annotated, Any, Literal
from urllib.parse import parse_qs, quote, unquote, urlencode, urlsplit

from mcp.server import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.server.transport_security import TransportSecuritySettings
from mcp.types import CallToolResult, TextContent, ToolAnnotations
from pydantic import BaseModel, ConfigDict, Field, model_validator
from starlette.responses import FileResponse, JSONResponse, Response

from .common import MAX_FILE, MAX_OUTPUT, MIMES, browser_revision, encoded, job_path, package_path, read, relative, require_package_approval, validate_request, write
from . import delivery, drafts, publication, projects
from .delivery import Delivery
from .inputs import ArtifactInputs, validate_reference
from .workspaces import WORKSPACE_RE, Workspaces, merge_appends

# A handle identifies server-owned state; it is not a task label or path.
# Keep its wire format aligned with storage rather than teaching Harness a
# workspace-specific rule. Existing returned handles/revisions stay unchanged.
WorkspaceHandle = Annotated[str, Field(
    pattern=f"^{WORKSPACE_RE.pattern}$",
    description="Exact workspace_id returned by create_workspace for this task. It is a server-generated handle, not a chosen name, filename, job_id or path. If none exists, call create_workspace first; never invent an ID.")]
WorkspaceRevision = Annotated[int, Field(
    ge=0,
    description="Exact current revision returned for this workspace. Use the latest edit/execution receipt; never invent a revision or reuse one from another workspace.")]

MAX_HTML_PREVIEW_BYTES = 2 * 1024 * 1024
# Same opaque-origin boundary as Harness's authenticated HTML preview. A signed
# media URL authorizes reading this file, never accessing cookies or other APIs.
HTML_PREVIEW_CSP = "sandbox allow-scripts; default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; img-src data: blob:; font-src data:; media-src data: blob:; connect-src 'none'; frame-src 'none'; object-src 'none'; base-uri 'none'; form-action 'none'; frame-ancestors 'self'"


class SourceFile(BaseModel):
    model_config = ConfigDict(extra="forbid")
    path: str = Field(description="Relative UTF-8 source path inside /workspace; not a host path")
    content: str


class JsonFile(BaseModel):
    model_config = ConfigDict(extra="forbid")
    path: str = Field(description="Relative JSON data file path inside /workspace; not a host path")
    value: dict[str, Any] | list[Any] = Field(description="Native JSON object or array, not an encoded string. The service serializes newlines, quotes and backslashes safely. Use files.content for source code.")


class InputFile(BaseModel):
    model_config = ConfigDict(extra="forbid")
    path: str = Field(description="Relative destination inside the new /workspace; must not overwrite a source file")
    url: str | None = Field(default=None, max_length=4096, description="Exact returned artifact URL: this workspace's signed download_url or an administrator-approved input source from inspect_runtime. Chat /v1/run-artifacts links need user login: use the returned attachment_id instead. Do not guess URLs. Supply url OR attachment_id.")
    attachment_id: str | None = Field(default=None, pattern=r"^att_[0-9a-f]{64}$", description="Exact attachment_id returned by upload or a model tool, including TTS. Preferred for Harness attachments; copied from configured asset service with size and SHA-256 verification. Supply attachment_id OR url.")
    sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$", description="Optional exact hash returned with the source artifact; never invent a hash.")

    @model_validator(mode="after")
    def valid_source(self):
        validate_reference(self.model_dump(exclude_none=True))
        relative(self.path)
        return self


class FilePatch(BaseModel):
    model_config = ConfigDict(extra="forbid")
    path: str
    sha256: str = Field(pattern="^[0-9a-f]{64}$", description="Current file hash returned by read_workspace")
    before: str = Field(min_length=1)
    after: str


class Viewport(BaseModel):
    model_config = ConfigDict(extra="forbid")
    width: int = Field(ge=320, le=1920)
    height: int = Field(ge=320, le=1200)


class Runtime:
    def __init__(self, root: Path, packages: Path, token: str, public_base: str, *, asset_base="", download_bases=()):
        if len(token) < 32:
            raise ValueError("private MCP token must contain at least 32 characters")
        parsed = urlsplit(public_base)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.query or parsed.fragment:
            raise ValueError("invalid administrator public base URL")
        self.root, self.packages, self.token, self.public_base = root, packages, token, public_base.rstrip("/")
        self.jobs = root / "jobs"
        self.jobs.mkdir(parents=True, exist_ok=True)
        self.submit_lock = threading.Lock()
        self.workspaces = Workspaces(root)
        self.inputs = ArtifactInputs(asset_base, download_bases)

    def info(self):
        heartbeat = self.root / "worker.json"
        ready = heartbeat.is_file() and time.time() - heartbeat.stat().st_mtime < 30
        inventory_path = self.root / "inventory.json"
        inventory = read(inventory_path) if inventory_path.is_file() else {}
        ready = ready and bool(inventory.get("image")) and inventory["image"] == read(heartbeat).get("image")
        policy_path = self.root / "task-code.json"
        policy = read(policy_path) if policy_path.is_file() else {}
        return {"ready": ready, "image": read(heartbeat).get("image") if heartbeat.is_file() else None,
                "engineering_image": inventory.get("engineering_image") if ready else None,
                "task_code_ready": ready and bool(policy.get("enabled")) and policy.get("image") == inventory.get("image"),
                "inventory": inventory,
                "runtimes": [name for name, executable in [("python", "python"), ("shell", "bash"), ("node", "node"), ("typescript", "tsx")] if inventory.get("commands", {}).get(executable) and (name != "typescript" or inventory.get("commands", {}).get("tsc"))], "execution": "fresh OCI container per invocation; network disabled; optional read-only package at /skill; writable /workspace; no host credentials or Docker socket; never falls back to host execution",
                "libraries": {"python": inventory.get("python_packages", {}), "node": inventory.get("node_packages", {}), "commands": inventory.get("commands", {})},
                "publication_types": MIMES,
                "input_sources": self.inputs.info(),
                "browser_ready": ready and bool(inventory.get("browser_image")),
                "limits": {"seconds": 300, "memory_bytes": 2147483648, "cpus": 2, "source_bytes": 524288, "output_bytes": 67108864},
                "instructions": "Reuse the workspace ID and revision already returned for this task. A package-bound workspace supports both generated code using installed libraries and its registered scripts/resources at /skill; do not replace it with an unbound workspace just to run Python, Node or Shell. When creating a workspace, supply the exact approved package identity if any step needs package files; otherwise omit both package fields for standalone computation. An unbound workspace has no /skill mount. Both tools execute only in fresh sandbox containers, restore the private task workspace and save files after script errors. Read bounded slices and patch exact hashes before retrying; do not regenerate unchanged files. Declare only requested final downloads in outputs, keeping inspection files private. No host execution fallback or installation during calls. Missing dependencies require administrator preparation."}

    def input_bytes(self, url):
        # Own signed artifacts stay local and retain their existing validation.
        # Other input files cross a separate, administrator-scoped ingress boundary.
        parsed, base = urlsplit(url), urlsplit(self.public_base)
        prefix = base.path.rstrip("/") + "/workspace-files/"
        if (parsed.scheme, parsed.netloc) != (base.scheme, base.netloc) or parsed.fragment or not parsed.path.startswith(prefix):
            return self.inputs.url(url)
        parts = parsed.path[len(prefix):].split("/")
        if len(parts) != 2:
            raise ValueError("invalid input artifact path")
        job_id, filename = parts[0], unquote(parts[1])
        if Path(filename).name != filename or "\\" in filename:
            raise ValueError("invalid input filename")
        signature = parse_qs(parsed.query).get("signature", [])
        expected = self.artifact_url(job_id, filename).split("signature=", 1)[1]
        if len(signature) != 1 or not hmac.compare_digest(signature[0], expected):
            raise ValueError("input artifact signature is invalid")
        job = job_path(self.jobs, job_id)
        if read(job / "state.json")["status"] != "completed":
            raise ValueError("input artifact job has not completed")
        item = next((item for item in read(job / "result.json")["files"] if item["filename"] == filename), None)
        if item is None:
            raise ValueError("input artifact was not published")
        with os.fdopen(os.open(job / "published" / filename, os.O_RDONLY | os.O_NOFOLLOW), "rb") as stream:
            data = stream.read(MAX_FILE + 1)
        if not 0 < len(data) <= MAX_FILE or len(data) != item["size_bytes"] or hashlib.sha256(data).hexdigest() != item["sha256"]:
            raise ValueError("input artifact integrity mismatch")
        return data

    def input_file(self, item):
        validate_reference(item)
        data = self.inputs.attachment(item["attachment_id"]) if item.get("attachment_id") else self.input_bytes(item["url"])
        if item.get("sha256") and hashlib.sha256(data).hexdigest() != item["sha256"]:
            raise ValueError("input artifact SHA-256 mismatch")
        return data

    def submit(self, request):
        with self.submit_lock:
            return self._submit(request)

    def _submit(self, request):
        browser_check = request.get("operation") == "inspect_webpage"
        identity_request = dict(request)
        if not identity_request.get("inputs"):
            identity_request.pop("inputs", None)
        original_binding = (request.get("package_name"), request.get("package_digest"))
        # Older workspaces may have bound a markdown-only Skill by mistake.
        # Resolve only its instruction reference, without rewriting workspace
        # identity/revision or granting access to any unapproved package bytes.
        if request.get("workspace_id") and not browser_check and request.get("operation") is None and request.get("entrypoint_root", "workspace") == "workspace":
            from .execution_binding import instruction_reference
            reference = instruction_reference(self.root, self.packages, *original_binding)
            if reference:
                state = self.workspaces.state(request["workspace_id"])
                if original_binding != (state["package_name"], state["package_digest"]):
                    raise ValueError("workspace package identity cannot change")
                request = {**request, "package_name": None, "package_digest": None,
                           "instruction_reference": reference}
        try:
            validate_request(request)
        except ValueError as error:
            # Capture approved, bounded sources privately; a rejected call still
            # neither executes code nor advances the workspace revision.
            try:
                handle = request["workspace_id"]
                state = self.workspaces.state(handle, request["workspace_revision"], idle=True)
                if not browser_check and original_binding != (state["package_name"], state["package_digest"]):
                    raise ValueError("workspace identity mismatch")
                if not browser_check:
                    require_package_approval(self.root, request)
                draft = drafts.save(self.workspaces, handle, request.get("files"))
            except (ValueError, KeyError, TypeError, OSError):
                raise error
            return {"status": "rejected", "error": str(error), "code_executed": False,
                    "workspace": {"workspace_id": handle, "revision": state["revision"]}, "draft": draft,
                    "recovery": {"tool": "edit_workspace", "arguments": {"workspace_id": handle,
                        "expected_revision": state["revision"], "draft_id": draft["id"]},
                        "effect": "Restore exact saved sources only; no code executes."},
                    "_meta": {"exoharness": {"context_checkpoint": "rejected source retained: " + draft["id"],
                        "context_preserve": ["/status", "/error", "/code_executed", "/workspace", "/draft", "/recovery"]}}}
        if request["runtime"] == "typescript" and "typescript" not in self.info()["runtimes"]:
            raise ValueError("TypeScript runtime is not prepared; an administrator must install the approved toolchain")
        # Preserve identities of requests created before optional file bindings.
        request = dict(request)
        if not request.get("inputs"):
            request.pop("inputs", None)
        if browser_check:
            if request.get("package_name") or request.get("package_digest"):
                raise ValueError("browser inspection cannot mount a Skill package")
        else:
            require_package_approval(self.root, request)
        package = package_path(self.packages, request["package_name"], request["package_digest"]) if request.get("package_name") else None
        if not self.info()["ready"]:
            raise ValueError("managed runtime broker is unavailable; no execution was submitted")
        # Deterministic secret-scoped identity makes duplicate deliveries harmless.
        # Changing the code creates a distinct invocation. A successful one is reused.
        # Approval/preparation changes must not turn a duplicate delivery into a
        # new execution. Identity belongs to the caller's original invocation,
        # while request.json records the resolved, least-privilege mount set.
        job_id = "work_" + hmac.new(self.token.encode(), encoded(identity_request), hashlib.sha256).hexdigest()
        job = self.jobs / job_id
        if (job / "state.json").is_file():
            return self.status(job_id, True)
        workspace = request.get("workspace_id")
        workspace_state = None
        if workspace:
            workspace_state = self.workspaces.state(workspace, request["workspace_revision"], idle=True)
            expected_image = self.info().get("engineering_image") if workspace_state.get("engineering_owner") else self.info()["image"]
            if workspace_state.get("runtime_image") and workspace_state["runtime_image"] != expected_image:
                raise ValueError("workspace is pinned to a different runtime image; no execution was submitted")
            if not browser_check and original_binding != (workspace_state["package_name"], workspace_state["package_digest"]):
                raise ValueError("workspace package identity cannot change")
            for source in request["files"]:
                name = source["path"]
                if name in workspace_state.get("authored_files", []) and name in workspace_state["files"]:
                    if self.workspaces.bytes(workspace, name, workspace_state) != source["content"].encode():
                        raise ValueError(f"inline code would overwrite authored file {name}; no code was changed or executed. Omit code to run that saved entrypoint, use a different entrypoint for a check script, or modify the authored file explicitly with edit_workspace")
            if request.get("entrypoint_root") == "skill":
                entrypoint = package / request["entrypoint"] if package else None
                if entrypoint is None or not entrypoint.is_file() or entrypoint.is_symlink() or not entrypoint.resolve().is_relative_to(package.resolve()):
                    raise ValueError("registered Skill script not found")
            elif request["entrypoint"] not in workspace_state["files"] and request["entrypoint"] not in {item["path"] for item in request["files"]}:
                raise ValueError("workspace script not found; write it with edit_workspace first")
        input_files, total = [], 0
        if workspace_state:
            input_files = [(name, self.workspaces.bytes(workspace, name, workspace_state)) for name in workspace_state["files"]]
            total = sum(len(data) for _, data in input_files)
        for item in request.get("inputs", []):
            data = self.input_file(item)
            total += len(data)
            if total > MAX_OUTPUT:
                raise ValueError("input artifacts exceed 64 MiB")
            input_files.append((item["path"], data))
        job.mkdir(mode=0o700, exist_ok=True)
        if not (job / "state.json").exists():
            write(job / "request.json", request)
            inputs = job / "input"
            inputs.mkdir(mode=0o755, exist_ok=True)
            write(inputs / "request.json", request)
            (inputs / "request.json").chmod(0o644)
            manifest = []
            for index, (path, data) in enumerate(input_files):
                target = inputs / f"artifact-{index}"
                target.write_bytes(data)
                target.chmod(0o644)
                manifest.append({"path": path, "file": target.name, "size_bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()})
            write(inputs / "files.json", manifest)
            (inputs / "files.json").chmod(0o644)
            if workspace_state is not None:
                if request["files"]:
                    values = {name: self.workspaces.bytes(workspace, name, workspace_state) for name in workspace_state["files"]}
                    values.update({item["path"]: item["content"].encode() for item in request["files"]})
                    workspace_state["files"] = self.workspaces.persist(workspace, values)
                workspace_state["active_job"] = job_id
                write(self.workspaces.directory(workspace) / "state.json", workspace_state)
            write(job / "state.json", {"status": "queued", "progress": 0, "message": "脚本已登记，等待隔离执行"})
        return self.status(job_id, True)

    def artifact_url(self, job_id, filename):
        signature = hmac.new(self.token.encode(), f"workspace-file:{job_id}:{filename}".encode(), hashlib.sha256).hexdigest()
        return f"{self.public_base}/workspace-files/{job_id}/{quote(filename, safe='')}?" + urlencode({"signature": signature})

    def artifact_preview_url(self, job_id, filename):
        path, query = self.artifact_url(job_id, filename).split("?", 1)
        return f"{path}/preview?{query}"

    def status(self, job_id, observe=False):
        job = job_path(self.jobs, job_id)
        state = read(job / "state.json")
        output = {"job_id": job_id, **state}
        request = read(job / "request.json") if (job / "request.json").is_file() else {}
        if request.get("workspace_id"):
            workspace = self.workspaces.inspect(request["workspace_id"])
            files = workspace.pop("files")
            workspace["file_count"] = len(files)
            output["workspace"] = workspace
        if (job / "execution.json").is_file():
            execution = read(job / "execution.json")
            log = str(execution.get("log", ""))
            output["execution"] = {"exit_code": execution.get("exit_code"), "log_tail": log[-2000:], "log_truncated": len(log) > 2000}
        inspection = request.get("operation") == "inspect_webpage"
        if state["status"] == "completed":
            result = read(job / "result.json")
            output["artifacts"] = [{**item, "download_url": self.artifact_url(job_id, item["filename"])} for item in result["files"]]
            for item in output["artifacts"]:
                if item["content_type"].split(";", 1)[0].strip().lower() == "text/html" and item["size_bytes"] <= MAX_HTML_PREVIEW_BYTES:
                    item["preview_url"] = self.artifact_preview_url(job_id, item["filename"])
            output["evidence"] = {key: value for key, value in result.items() if key != "files"}
            if inspection and isinstance(output["evidence"].get("inspection"), dict):
                output["evidence"]["inspection"] = compact_inspection(output["evidence"]["inspection"])
        if observe:
            terminal = state["status"] if state["status"] in {"completed", "failed", "cancelled"} else "working"
            output["_meta"] = {"exoharness": {"background_task": {
                "version": 1, "id": job_id, "status": terminal,
                "poll": {"tool": "get_script_job", "arguments": {"job_id": job_id, "observe": True}},
                "cancel": {"tool": "cancel_script_job", "arguments": {"job_id": job_id}},
                "poll_interval_ms": 2000,
                "progress": {"progress": state.get("progress", 0), "total": 100, "message": state.get("message", "")}}}}
        if state["status"] in {"completed", "failed"}:
            meta = output.setdefault("_meta", {}).setdefault("exoharness", {})
            if state.get("artifact_visibility") == "internal":
                meta["artifact_visibility"] = "internal"
            meta["context_checkpoint"] = "script output persisted: " + job_id
            # Failed jobs have diagnostics, not published artifacts. Declare
            # only real fields, including the error even if no process started.
            # Inspection screenshots are private evidence the model cannot view;
            # only the job ID and findings must stay in the model's checkpoint.
            preserved = ("job_id", "status", "error", "message", "artifacts", "evidence", "execution", "workspace", "code_executed")
            meta["context_preserve"] = ["/" + key for key in preserved
                if key in output and not (inspection and key == "artifacts")]
            # One script finishing does not finish the user's multi-step task.
            # Compact its code/results without revoking the remaining tools.
        return output


mcp = MCPServer("Task Sandbox", version="0.1.0", instructions="Execute Python, Node, TypeScript and Shell scripts in the enabled task sandbox. Create a workspace once, then execute code using its returned ID and revision. For TypeScript sources use runtime_name=typescript; managed type checks run before execution. General computation and plotting need no Skill binding or package fields. Persist real output files for delivery. Shell runs only inside the container, never on the host.")
_runtime = None


def runtime():
    global _runtime
    if _runtime is None:
        _runtime = Runtime(Path(os.environ["WORKSPACE_DATA_DIR"]), Path(os.environ["WORKSPACE_PACKAGES_DIR"]),
                           Path(os.environ["WORKSPACE_TOKEN_FILE"]).read_text().strip(), os.environ["WORKSPACE_PUBLIC_BASE"],
                           asset_base=os.environ.get("WORKSPACE_ASSET_BASE", ""),
                           download_bases=os.environ.get("WORKSPACE_INPUT_BASES", "").split(","))
    return _runtime


def compact_inspection(inspection):
    """Explicit gate findings; raw hiding is observation, not a repair request.

    Full inventories remain in the job report. Passing covers the observer's
    automated rendering checks, not aesthetic approval or untested controls.
    """
    compact = {key: value for key, value in inspection.items() if key != "scenarios"}
    scenarios = []
    for report in inspection.get("scenarios", []):
        errors = delivery.browser_errors(report)
        missing = [key for key in delivery.BROWSER_ERROR_FIELDS if key not in report]
        entry = {"checks": {"status": "failed" if errors else "incomplete" if missing else "passed",
                            "errors": errors}}
        if missing:
            entry["checks"]["missing_fields"] = missing
        for key, value in report.items():
            if key == "headings":
                entry["heading_count"] = len(value)
            elif key == "hidden_text_blocks":
                # This bounded sample includes intentionally inactive language,
                # menu and dialog content. Only unexpected_* is actionable.
                if value:
                    entry["hidden_text_sample_count"] = len(value)
            elif key == "checks":
                continue  # Derived from raw diagnostics, never a supplied verdict.
            elif isinstance(value, list) and not value:
                continue
            else:
                entry[key] = value
        scenarios.append(entry)
    compact["scenarios"] = scenarios
    statuses = {s["checks"]["status"] for s in scenarios}
    compact["checks"] = {
        "status": "failed" if "failed" in statuses else "incomplete" if not statuses or "incomplete" in statuses else "passed",
        "scope": "automated rendering checks only",
    }
    return compact


def result(value):
    content = {key: item for key, item in value.items() if key != "_meta"}
    return CallToolResult(content=[TextContent(text=encoded(content).decode())], structuredContent=content,
                          isError=value.get("status") in {"failed", "rejected"}, _meta=value.get("_meta"))


def input_errors(function):
    """Expose deliberate input/policy validation, not unexpected exception traces."""
    @wraps(function)
    def checked(*args, **kwargs):
        try:
            return function(*args, **kwargs)
        except ValueError as error:
            raise ToolError(str(error)[:1600]) from error
    return checked


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=True, destructiveHint=False))
def inspect_runtime() -> CallToolResult:
    """Inspect Python/Node libraries, the managed TypeScript compiler/runner, Bash and sandbox readiness without running user code or installing anything. General computation needs no per-Skill setup."""
    return result(runtime().info())


class ProgramBinding(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(pattern=r"^[a-zA-Z][a-zA-Z0-9_]{0,63}$")
    tool: str = Field(min_length=1, max_length=64, description="Exact authorized Harness tool alias; only MCP tools explicitly annotated readOnlyHint=true are accepted")


@mcp.tool(meta={"exoharness": {"program_runtime": {"version": 1, "language": "python", "access": "read_only"}}})
def run_program(code: Annotated[str, Field(min_length=1, max_length=65536)],
                bindings: Annotated[list[ProgramBinding], Field(min_length=1, max_length=8)],
                input: Any = None,
                max_calls: Annotated[int, Field(ge=1, le=16)] = 8,
                timeout_ms: Annotated[int, Field(ge=1, le=300000)] = 60000,
                max_output_bytes: Annotated[int, Field(ge=1, le=65536)] = 16000) -> CallToolResult:
    """Governed cross-tool Python program in a fresh network-disabled OCI sandbox. Code is an async function body: await tools.call(binding_name, native_arguments), process JSON locally, then return a concise JSON result. Explicit bindings select authorized read-only MCP tools; no other tools, recursive programs, generated scripts, network, host paths or credentials are available. Calls use the same Run permissions, invocation journal and budgets as direct calls. Use for dependent read-only queries and result reduction, not video/music generation or writes. No model calls inside the program. Harness must provide the governed dispatcher; direct MCP clients cannot execute this operation."""
    raise ToolError("run_program requires the Harness governed dispatcher; no execution was submitted")


@mcp.tool()
@input_errors
def create_workspace(package_name: str | None = None, package_digest: str | None = None,
                     delivery: Delivery | None = None) -> CallToolResult:
    """Create a task workspace and return its generated workspace_id and revision before any execution. Only create if this task has none; reuse its returned handle across generation, checks and publication, never choose a task name as the ID. For a final file deliverable, declare delivery outputs and required checks: source writes cannot finish that workflow; publish_workspace completes it. Omit delivery for stdout-only calculation or intermediate workspace work. Pass a registered Skill's exact name and digest only if its approved scripts/resources under /skill are needed; omit both for standalone work. No code runs or dependencies install here."""
    rt = runtime()
    if bool(package_name) != bool(package_digest):
        raise ValueError("supply both package_name and package_digest, or neither")
    if package_name:
        package_path(rt.packages, package_name, package_digest)
    with rt.submit_lock:
        from .execution_binding import instruction_reference
        reference = instruction_reference(rt.root, rt.packages, package_name, package_digest)
        if reference:
            package_name = package_digest = None
        value = rt.workspaces.create(package_name, package_digest, rt.info()["image"],
                                     delivery.model_dump() if delivery else None, reference)
        return edited_workspace(value) if delivery else result(value)


@mcp.tool()
@input_errors
def read_workspace(workspace_id: WorkspaceHandle, path: str | None = None, offset: int = 0, limit: int = 8000, query: str | None = None) -> CallToolResult:
    """List saved file names/hashes when path is omitted. To read or search a file, supply path; query is a literal string, not regex. Prefer query for a known selector, function or text: it returns at most four small source snippets with exact offsets, line numbers, revision and SHA for patching. offset/limit count Unicode characters, not lines or bytes. next_offset=null means the selection is exhausted; do not restart the same scan. Use sandbox scripts for whole-file analysis and return only findings instead of paging all source through the model. Binary files stay in storage for scripts. Works after failures; keep saved code instead of regenerating it."""
    require_general_workspace(workspace_id)
    with runtime().submit_lock:
        return result(runtime().workspaces.inspect(workspace_id, path, offset, limit, query))


def edited_workspace(value):
    # Successful source writes are already durable. The next model request
    # needs hashes/revisions, not another copy of the entire authored program.
    # Newly requested source slices are not write checkpoints: they still
    # reach the model, and exact originals stay in the journal/store.
    value.setdefault("_meta", {}).setdefault("exoharness", {}).update({
        "context_checkpoint": f"workspace files saved: {value['workspace_id']} revision {value['revision']}",
        "context_preserve": ["/workspace_id", "/revision", "/package_name", "/package_digest", "/runtime_image", "/files"]
            + (["/delivery"] if value.get("delivery") else []) + (["/notes"] if "notes" in value else [])
            + (["/changes"] if value.get("changes") else [])})
    return result(delivery.pending(value, value))


def require_general_workspace(workspace_id):
    if runtime().workspaces.state(workspace_id).get("engineering_owner"):
        raise ValueError("private projects require authorized project tools")


def require_general_job(job_id):
    job = job_path(runtime().jobs, job_id)
    request = read(job / "request.json") if (job / "request.json").is_file() else read(job / "state.json").get("workspace", {})
    if request.get("operation") == "invoke_tool":
        raise ValueError("private Tool jobs are not accessible through general workspace tools")
    if request.get("workspace_id"):
        require_general_workspace(request["workspace_id"])


@mcp.tool()
@input_errors
def edit_workspace(workspace_id: WorkspaceHandle, expected_revision: WorkspaceRevision,
                   files: Annotated[list[SourceFile | JsonFile], Field(max_length=24, description="Atomic file writes: each item is either {path, content} for exact UTF-8 text/source, or {path, value} for a native JSON object/array. Supply exactly one of content or value; do not encode value as a string.")] = [],
                   patches: Annotated[list[FilePatch], Field(max_length=24)] = [],
                   inputs: Annotated[list[InputFile], Field(max_length=12)] = [],
                   draft_id: Annotated[str | None, Field(description="Exact draft ID from a rejected call. Restores its saved sources without resending code; cannot be combined with other edits or overwrite different existing content.")] = None,
                   json_files: Annotated[list[JsonFile], Field(max_length=24)] = [],
                   appends: Annotated[list[SourceFile], Field(max_length=24, description="Append exact UTF-8 text to existing files. Multiple entries with the same path are concatenated in array order and saved atomically in ONE revision, without inserting separators. Do not mix appends with writes/imports/patches for the same path. expected_revision rejects stale retries; use files to create the file first.")] = [],
                   notes: Annotated[str | None, Field(max_length=2000, description="Optional compact working notes retained verbatim with file receipts after source leaves model context. Record shared interfaces/conventions and the next unfinished step, not source or a claimed test verdict. Omit to preserve previous notes; empty text clears them. Updated atomically with these files.")] = None) -> CallToolResult:
    """Write task files, import attachments/artifacts, restore a rejected draft, or patch only erroneous code. files accepts exact source via content or native JSON object/array via value, never both; json_files remains supported for JSON-only batches. inputs accepts a returned attachment_id or an allowed artifact URL, never a login-only chat link. Multiple patches to one file use the SAME current SHA-256 and non-overlapping unique before matches in that original file. Do not mix patches and writes/appends for one path. Edits are atomic, rejecting stale revisions or busy workspaces. The receipt's changes list gives each changed text file's size, line count and final 200 characters: use it to confirm the write instead of re-reading your own output. No code executes; reuse existing files."""
    require_general_workspace(workspace_id)
    rt = runtime()
    if len(files) + len(json_files) + len(appends) > 24:
        raise ValueError("at most 24 files may be written per edit")
    if len(encoded([item.model_dump() for item in [*files, *patches, *json_files, *appends]])) > 512 * 1024:
        raise ValueError("edit request exceeds 512 KiB")
    # Explicit structured data only: never reinterpret or rewrite source text.
    files = [item if isinstance(item, SourceFile) else SourceFile(path=item.path,
        content=json.dumps(item.value, ensure_ascii=False, allow_nan=False,
                           separators=(",", ":")) + "\n") for item in [*files, *json_files]]
    if len(encoded([item.model_dump() for item in [*files, *patches, *appends]])) > 512 * 1024:
        raise ValueError("serialized edit request exceeds 512 KiB")
    appends = [SourceFile(**item) for item in merge_appends(item.model_dump() for item in appends)]
    with rt.submit_lock:
        if draft_id is not None:
            if files or patches or inputs or appends or notes is not None:
                raise ValueError("restore a draft separately from other edits")
            return edited_workspace(drafts.restore(rt.workspaces, workspace_id, expected_revision, draft_id))
        state = rt.workspaces.state(workspace_id, expected_revision, idle=True)
        names = [item.path for item in [*files, *inputs, *appends]]
        patch_names = {item.path for item in patches}
        for name in [*names, *patch_names]:
            relative(name)
        if len(set(names)) != len(names) or set(names) & patch_names:
            raise ValueError("a file may only be changed once by one edit mode; do not mix patches with writes/imports/appends")
        replaced = {item.path for item in [*files, *inputs]}
        total = sum(item["size_bytes"] for name, item in state["files"].items() if name not in replaced)
        total += sum(len(item.content.encode()) for item in files)
        total += sum(len(item.content.encode()) for item in appends)
        copied = []
        for item in inputs:
            data = rt.input_file(item.model_dump(exclude_none=True))
            total += len(data)
            if total > MAX_OUTPUT:
                raise ValueError(f"workspace total capacity exceeded: attempted {total} bytes, limit {MAX_OUTPUT} bytes. "
                                 "No files were imported by this call. Splitting the same files into more import calls does not increase total capacity. "
                                 "Retain original artifact URLs and use an authorized media/file service for large assets, or explicitly reduce saved files before retrying")
            copied.append({"path": item.path, "bytes": data})
        return edited_workspace(rt.workspaces.edit(workspace_id, expected_revision,
            [item.model_dump() for item in files], [item.model_dump() for item in patches], copied,
            [item.model_dump() for item in appends], notes=notes))


@mcp.tool()
@input_errors
def publish_workspace(workspace_id: WorkspaceHandle, expected_revision: WorkspaceRevision,
                      outputs: Annotated[list[str], Field(min_length=1, max_length=12)],
                      inspection_jobs: Annotated[list[str], Field(max_length=3, description="Returned completed inspect_webpage job IDs for declared delivery browser checks, all from the final workspace revision.")] = [],
                      artifact_visibility: Literal["final", "internal"] = "final") -> CallToolResult:
    """Publish selected existing files as immutable downloadable artifacts. Use artifact_visibility=internal for intermediate files another authorized tool needs by URL; these remain in storage/Trace, not chat, and cannot complete a declared final delivery. No script runs, dependencies install or revision changes. Supports HTML and publication_types from inspect_runtime. Does not validate page quality or deploy an application. Identical publications reuse their result."""
    require_general_workspace(workspace_id)
    rt = runtime()
    with rt.submit_lock:
        return result(publication.publish(rt, workspace_id, expected_revision, outputs, inspection_jobs, artifact_visibility))


@mcp.tool()
@input_errors
def inspect_webpage(workspace_id: WorkspaceHandle, expected_revision: WorkspaceRevision, path: str,
                    viewports: Annotated[list[Viewport], Field(min_length=1, max_length=3)] = [Viewport(width=1440, height=900), Viewport(width=390, height=844)],
                    check_without_scripts: bool = True) -> CallToolResult:
    """Inspect a saved HTML page with a real, isolated browser. Returns bounded console/layout/link diagnostics and private screenshots, not a design-quality verdict. Workspace files and revision are unchanged. Offline only: local workspace resources and data URLs; no external websites, host files, credentials or API access. Use reported paths/hashes for local repairs, not full regeneration. Screenshots are inspection evidence, not user deliverables."""
    rt = runtime()
    inventory = rt.info()["inventory"]
    if not rt.info()["browser_ready"]:
        raise ValueError("isolated browser is not prepared; browser verification was not performed")
    request = workspace_request(workspace_id, expected_revision, "python", path,
        [f"inspection-{i}.png" for i in range(len(viewports) + int(check_without_scripts))], [])
    # Browser verification reads only the immutable workspace snapshot. It
    # neither executes nor mounts the optional Skill package, whose runtime
    # approval is independent from the administrator-approved browser image.
    request.update(package_name=None, package_digest=None)
    request.update(operation="inspect_webpage", browser_image=inventory["browser_image"],
                   observer_revision=browser_revision(),
                   viewports=[v.model_dump() for v in viewports], check_without_scripts=check_without_scripts)
    return result(rt.submit(request))


def workspace_request(workspace_id, expected_revision, runtime_name, entrypoint, outputs, arguments, root="workspace", files=None):
    require_general_workspace(workspace_id)
    state = runtime().workspaces.state(workspace_id)
    return {"workspace_id": workspace_id, "workspace_revision": expected_revision,
            "runtime_image": state.get("runtime_image"),
            "package_name": state["package_name"], "package_digest": state["package_digest"],
            "runtime": runtime_name, "entrypoint": entrypoint, "entrypoint_root": root,
            "outputs": outputs, "arguments": arguments, "files": files or []}


@mcp.tool()
@input_errors
def execute_python(workspace_id: WorkspaceHandle, expected_revision: WorkspaceRevision,
                   code: Annotated[str | None, Field(description="Inline source for a NEW Python entrypoint. Omit code to execute a saved workspace file after edit_workspace; providing different code cannot overwrite an authored file.")] = None,
                   outputs: Annotated[list[str], Field(max_length=12)] = [],
                   path: str = "main.py") -> CallToolResult:
    """First call create_workspace if no workspace exists; use its returned ID/revision. Execute Python in the task sandbox to calculate, process data, plot and create files. Reuse the existing workspace, whether standalone or Skill-bound; do not create another to change languages. For a new script provide code; after read_workspace/edit_workspace, omit code and run the saved path at its current revision. Inline code cannot overwrite an authored file. Standalone computation needs no Skill binding. For text results print the answer and omit outputs; stdout/stderr return in execution.log_tail. Only list requested downloads actually written under /workspace; leave QA files private. Listing a filename does not create it. Network and host access are disabled."""
    files = [{"path": path, "content": code}] if code is not None else []
    return result(runtime().submit(workspace_request(workspace_id, expected_revision, "python", path, outputs, [], files=files)))


@mcp.tool()
@input_errors
def run_script(workspace_id: WorkspaceHandle, expected_revision: WorkspaceRevision, runtime_name: Literal["shell", "python", "node", "typescript"] = "shell",
               entrypoint: Annotated[str | None, Field(description="Saved workspace/package script path. For NEW inline code only, omission selects script.sh / script.py / script.js / script.ts from runtime_name. Explicit paths are never renamed or reinterpreted.")] = None,
               arguments: Annotated[list[str], Field(max_length=64)] = [],
               outputs: Annotated[list[str], Field(max_length=12)] = [],
               entrypoint_root: Literal["workspace", "skill"] = "workspace",
               code: Annotated[str | None, Field(description="Source for a NEW inline entrypoint. Omit code when executing a file already saved with edit_workspace. It cannot overwrite an authored file. Use a separate script path for syntax checks or QA wrappers.")] = None) -> CallToolResult:
    """Use a returned workspace ID/revision; if none exists, call create_workspace first. Execute Shell (default), Python, Node or TypeScript in that task workspace, including a Skill-bound workspace. Keep its ID/revision across languages, generation and checks. For .ts/.tsx/.mts/.cts use runtime_name=typescript; the offline compiler checks types first. Supply code for a new entrypoint, or omit code to run a saved script. Pipelines and installed programs run inside the container; no host Shell or network. arguments are literal arguments. Omit outputs for text results/private QA; execution.log_tail contains stdout/stderr. List only requested downloads that the script actually writes. Files survive failures; reuse the returned revision and patch only incorrect code."""
    if code is not None and entrypoint_root != "workspace":
        raise ValueError("inline code must stay in the task workspace; Skill packages are read-only")
    if entrypoint is None:
        if code is None and (entrypoint_root != "workspace" or runtime_name != "shell"):
            raise ValueError("supply the exact entrypoint to run a saved workspace or Skill script; no file was selected or executed")
        entrypoint = {"shell": "script.sh", "python": "script.py", "node": "script.js", "typescript": "script.ts"}[runtime_name]
    try:
        relative(entrypoint)
    except ValueError as error:
        raise ValueError("entrypoint must be a relative script file under entrypoint_root, not an interpreter path such as /bin/sh or /usr/bin/python. Choose the language with runtime_name; pass inline source in code (or use execute_python), and only script arguments in arguments. No execution was submitted") from error
    files = [{"path": entrypoint, "content": code}] if code is not None else []
    return result(runtime().submit(workspace_request(workspace_id, expected_revision, runtime_name, entrypoint, outputs, arguments, entrypoint_root, files)))


@mcp.tool()
@input_errors
def execute_skill_script(package_name: str, package_digest: str, runtime_name: Literal["python", "node", "typescript"],
                        files: Annotated[list[SourceFile], Field(min_length=1, max_length=24)], entrypoint: str,
                        outputs: Annotated[list[str], Field(max_length=12)],
                        inputs: Annotated[list[InputFile], Field(max_length=12)] = []) -> CallToolResult:
    """Execute selected Skill code in a fresh network-disabled OCI sandbox. Use its exact package identity. /skill is read-only; /workspace is fresh each call. Supply source files, entrypoint and output paths. TypeScript uses runtime_name=typescript with a .ts/.tsx/.mts/.cts entrypoint; types are checked before execution. Bundle generation, conversion and QA in one script when practical. For follow-up conversion/checks/edits reuse previous signed artifact URLs via inputs; never assume old workspace files persist or regenerate an existing artifact just to open it. Python/Node/TypeScript may run installed commands such as LibreOffice. stdout/stderr return as a bounded execution log; outputs=[] supports inspection-only scripts. Persisted outputs have downloadable URLs. Identical deliveries reuse results. No host shell, credentials or installs."""
    return result(runtime().submit({"package_name": package_name, "package_digest": package_digest,
        "runtime": runtime_name, "files": [item.model_dump() for item in files], "entrypoint": entrypoint, "outputs": outputs,
        "inputs": [item.model_dump(exclude_none=True) for item in inputs]}))


@mcp.tool()
def get_script_job(job_id: str, observe: bool = False) -> CallToolResult:
    """Retrieve persisted status, bounded execution log, exact failure or file artifacts. A completed job was already returned by execute_skill_script; only re-query if that result is unavailable. observe=true lets the host wait without model polling."""
    require_general_job(job_id)
    return result(runtime().status(job_id, observe))


@mcp.tool()
def cancel_script_job(job_id: str) -> CallToolResult:
    """Cancel only this job's isolated container; return cancelled only after the broker confirms it stopped."""
    require_general_job(job_id)
    rt = runtime()
    job = job_path(rt.jobs, job_id)
    if read(job / "state.json")["status"] not in {"completed", "failed", "cancelled"}:
        (job / "cancel").touch(mode=0o600, exist_ok=True)
    return result(rt.status(job_id, True))


projects.register(mcp, runtime, result, input_errors)


@mcp.custom_route("/health", methods=["GET"])
async def health(_request):
    ready = runtime().info()["ready"]
    return JSONResponse({"status": "ok" if ready else "worker_unavailable"}, status_code=200 if ready else 503)


def artifact_file(request):
    rt = runtime()
    job_id, filename = request.path_params["job_id"], request.path_params["filename"]
    job = job_path(rt.jobs, job_id)
    if Path(filename).name != filename:
        raise ValueError("invalid filename")
    expected = rt.artifact_url(job_id, filename).split("signature=", 1)[1]
    if not hmac.compare_digest(expected, request.query_params.get("signature", "")):
        raise ValueError("unauthorized")
    if read(job / "state.json")["status"] != "completed":
        raise ValueError("not complete")
    item = next(item for item in read(job / "result.json")["files"] if item["filename"] == filename)
    path = job / "published" / filename
    if path.is_symlink() or path.stat().st_size != item["size_bytes"]:
        raise ValueError("file unavailable")
    return path, item


@mcp.custom_route("/workspace-files/{job_id}/{filename}", methods=["GET", "HEAD"])
async def artifact(request):
    try:
        path, item = artifact_file(request)
    except (ValueError, OSError, KeyError, StopIteration):
        return JSONResponse({"error": "file unavailable"}, status_code=404)
    return FileResponse(path, filename=item["filename"], media_type=item["content_type"], content_disposition_type="attachment",
        headers={"Cache-Control": "private, no-store", "Referrer-Policy": "no-referrer", "X-Content-Type-Options": "nosniff"})


@mcp.custom_route("/workspace-files/{job_id}/{filename}/preview", methods=["GET", "HEAD"])
async def artifact_preview(request):
    try:
        path, item = artifact_file(request)
        if item["content_type"].split(";", 1)[0].strip().lower() != "text/html":
            raise ValueError("not HTML")
        if item["size_bytes"] > MAX_HTML_PREVIEW_BYTES:
            return JSONResponse({"error": "HTML exceeds the 2 MiB preview limit; use the download URL"}, status_code=413)
        with os.fdopen(os.open(path, os.O_RDONLY | os.O_NOFOLLOW), "rb") as stream:
            data = stream.read(MAX_HTML_PREVIEW_BYTES + 1)
        if len(data) != item["size_bytes"] or hashlib.sha256(data).hexdigest() != item["sha256"]:
            raise ValueError("file integrity mismatch")
    except (ValueError, OSError, KeyError, StopIteration):
        return JSONResponse({"error": "file unavailable"}, status_code=404)
    return Response(data, media_type="text/html", headers={
        "Content-Disposition": "inline", "Content-Security-Policy": HTML_PREVIEW_CSP,
        "Cache-Control": "private, no-store", "Referrer-Policy": "no-referrer",
        "X-Content-Type-Options": "nosniff",
        "Permissions-Policy": "camera=(), microphone=(), geolocation=(), clipboard-read=(), clipboard-write=()"})


class BearerAuth:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http" and (scope.get("path") == "/mcp" or scope.get("path", "").startswith("/mcp/")):
            header = dict(scope.get("headers", [])).get(b"authorization", b"").decode("latin1")
            if not hmac.compare_digest(header, "Bearer " + runtime().token):
                await JSONResponse({"error": "unauthorized"}, status_code=401)(scope, receive, send)
                return
        await self.app(scope, receive, send)


def main():
    import uvicorn
    runtime()
    app = mcp.streamable_http_app(streamable_http_path="/mcp", json_response=False,
        transport_security=TransportSecuritySettings(enable_dns_rebinding_protection=True, allowed_hosts=["127.0.0.1:*", "localhost:*"]))
    app.add_middleware(BearerAuth)
    uvicorn.run(app, host="127.0.0.1", port=int(os.getenv("WORKSPACE_PORT", "8794")), access_log=False)


if __name__ == "__main__":
    main()
