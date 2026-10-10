from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from pathlib import Path, PurePosixPath

MAX_INPUT = 512 * 1024
MAX_FILE = 32 * 1024 * 1024
MAX_OUTPUT = 64 * 1024 * 1024
MAX_WIRE = 192 * 1024 * 1024
JOB_RE = re.compile(r"work_[0-9a-f]{64}")
NAME_RE = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9_-]{0,127}")
DIGEST_RE = re.compile(r"[0-9a-f]{64}")
MIMES = {".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
         ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
         ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
         ".pdf": "application/pdf", ".png": "image/png", ".jpg": "image/jpeg", ".webp": "image/webp",
         ".txt": "text/plain", ".csv": "text/csv", ".json": "application/json",
         ".html": "text/html", ".htm": "text/html", ".css": "text/css", ".js": "text/javascript",
         ".mp4": "video/mp4", ".mp3": "audio/mpeg", ".zip": "application/zip"}


def encoded(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode()


def browser_revision():
    """Pin the mounted observer/policy as well as its immutable base image."""
    return hashlib.sha256(b"".join(Path(__file__).with_name(name).read_bytes()
        for name in ("browser_runner.py", "broker.py"))).hexdigest()


def read(path):
    return json.loads(path.read_bytes())


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".write-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(encoded(value))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def relative(value):
    if not isinstance(value, str) or not value or len(value) > 200 or "\\" in value or "\x00" in value:
        raise ValueError("invalid relative path")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in value.split("/")):
        raise ValueError("path must stay inside this job workspace")
    return value


def validate_request(data):
    if not isinstance(data, dict):
        raise ValueError("script request must be a JSON object")
    if len(encoded(data)) > MAX_INPUT:
        raise ValueError("script request exceeds 512 KiB")
    if data.get("operation") == "invoke_program":
        from .program import validate
        return validate(data)
    if data.get("operation") == "invoke_tool":
        return validate_tool_request(data)
    workspace = data.get("workspace_id")
    if workspace and (not re.fullmatch(r"ws_[0-9a-f]{64}", workspace) or type(data.get("workspace_revision")) is not int):
        raise ValueError("invalid workspace identity or revision")
    if (not workspace or data.get("package_name")) and (not NAME_RE.fullmatch(data.get("package_name", "")) or not DIGEST_RE.fullmatch(data.get("package_digest", ""))):
        raise ValueError("use the exact selected Skill package name and SHA-256")
    if data.get("runtime") not in {"python", "node", "shell", "typescript"}:
        raise ValueError("runtime must be python, node, shell, or typescript inside the approved sandbox")
    suffix = Path(data.get("entrypoint", "")).suffix.lower()
    expected = {".py": "python", ".js": "node", ".cjs": "node", ".mjs": "node", ".sh": "shell", ".bash": "shell"}.get(suffix)
    if expected and data["runtime"] != expected:
        raise ValueError(f"entrypoint {suffix} requires runtime_name={expected}; keep the source unchanged and correct the runtime. No execution was submitted")
    if data.get("runtime") == "typescript" and Path(data.get("entrypoint", "")).suffix not in {".ts", ".tsx", ".mts", ".cts"}:
        raise ValueError("typescript entrypoint must end with .ts, .tsx, .mts, or .cts")
    if data.get("runtime") == "node" and Path(data.get("entrypoint", "")).suffix in {".ts", ".tsx", ".mts", ".cts"}:
        raise ValueError("Use runtime=typescript for TypeScript sources so preparation and type checks are not skipped")
    files = data.get("files")
    if not isinstance(files, list) or not (0 if workspace else 1) <= len(files) <= 24:
        raise ValueError("provide between 1 and 24 UTF-8 source files")
    names = set()
    for item in files:
        name = relative(item["path"])
        if name in names or not isinstance(item.get("content"), str):
            raise ValueError("duplicate path or invalid source content")
        names.add(name)
    relative(data["entrypoint"])
    if data.get("operation") is not None:
        if data["operation"] != "inspect_webpage" or not workspace or suffix not in {".html", ".htm"} or files:
            raise ValueError("invalid read-only browser inspection")
        viewports = data.get("viewports")
        if not isinstance(viewports, list) or not 1 <= len(viewports) <= 3 or any(
            not isinstance(v, dict) or set(v) != {"width", "height"} or
            any(type(v[k]) is not int for k in v) or not 320 <= v["width"] <= 1920 or not 320 <= v["height"] <= 1200
            for v in viewports):
            raise ValueError("invalid browser viewports")
        if type(data.get("check_without_scripts")) is not bool or not re.fullmatch(r"sha256:[a-f0-9]{64}", data.get("browser_image", "")):
            raise ValueError("invalid browser inspection policy")
        if not re.fullmatch(r"[a-f0-9]{64}", data.get("observer_revision", "")):
            raise ValueError("invalid browser observer revision")
        count = len(viewports) + int(data["check_without_scripts"])
        if data.get("outputs") != [f"inspection-{i}.png" for i in range(count)]:
            raise ValueError("invalid browser inspection outputs")
    if not workspace and data["entrypoint"] not in names:
        raise ValueError("entrypoint must be one of the supplied files")
    if data.get("entrypoint_root", "workspace") not in {"workspace", "skill"}:
        raise ValueError("entrypoint_root must be workspace or skill")
    arguments = data.get("arguments", [])
    if not isinstance(arguments, list) or len(arguments) > 64 or any(not isinstance(v, str) or len(v) > 4096 or "\x00" in v for v in arguments):
        raise ValueError("arguments must be at most 64 literal strings")
    inputs = data.get("inputs", [])
    if not isinstance(inputs, list) or len(inputs) > 12:
        raise ValueError("provide at most 12 persisted input files")
    for item in inputs:
        from .inputs import validate_reference
        validate_reference(item)
        name = relative(item["path"])
        if name in names:
            raise ValueError("duplicate input/source path")
        names.add(name)
    outputs = data.get("outputs")
    if not isinstance(outputs, list) or len(outputs) > 12 or len(set(outputs)) != len(outputs):
        raise ValueError("declare at most 12 distinct output files; use [] for inspection-only scripts")
    for path in outputs:
        relative(path)
        if Path(path).suffix.lower() not in MIMES:
            raise ValueError("output file type is not approved for publication")
    if len({Path(path).name for path in outputs}) != len(outputs):
        raise ValueError("output basenames must be unique")
    return data


def validate_tool_request(data):
    """Trusted OCI Tool job, never an arbitrary script request from MCP tools."""
    required = {"operation", "capability_id", "capability_revision", "package_name", "package_digest", "runtime_image",
                "runtime", "entrypoint", "tool_request", "timeout_ms", "max_output_bytes"}
    if not isinstance(data, dict) or set(data) != required:
        raise ValueError("invalid isolated Tool job fields")
    if (not isinstance(data["capability_id"], str) or not re.fullmatch(r"tool_node:[a-z0-9:_-]{1,240}", data["capability_id"])
            or type(data["capability_revision"]) is not int or data["capability_revision"] < 1):
        raise ValueError("invalid immutable Tool capability identity")
    if (not isinstance(data["package_name"], str) or not NAME_RE.fullmatch(data["package_name"])
            or not isinstance(data["package_digest"], str) or not DIGEST_RE.fullmatch(data["package_digest"])):
        raise ValueError("invalid immutable Tool package identity")
    if not isinstance(data["runtime_image"], str) or not re.fullmatch(r"sha256:[a-f0-9]{64}", data["runtime_image"]):
        raise ValueError("isolated Tool image must be pinned by digest")
    entrypoint = relative(data["entrypoint"])
    suffix = Path(entrypoint).suffix.lower()
    runtime = {".py": "python", ".js": "node", ".mjs": "node", ".cjs": "node",
               ".sh": "shell", ".bash": "shell", ".ts": "typescript", ".tsx": "typescript",
               ".mts": "typescript", ".cts": "typescript"}.get(suffix)
    if data["runtime"] != runtime:
        raise ValueError("Tool entrypoint requires its declared isolated runtime")
    request = data["tool_request"]
    if not isinstance(request, dict) or set(request) != {"protocol_version", "request_id", "tool", "arguments"}:
        raise ValueError("invalid Tool request envelope")
    if request["protocol_version"] != "exoharness.tool.v1":
        raise ValueError("unsupported Tool request protocol")
    if not isinstance(request["request_id"], str) or not re.fullmatch(r"[A-Za-z0-9_.:-]{1,160}", request["request_id"]):
        raise ValueError("invalid Tool invocation ID")
    if not isinstance(request["tool"], str) or not re.fullmatch(r"[A-Za-z0-9_.-]{1,128}", request["tool"]):
        raise ValueError("invalid Tool name")
    if not isinstance(request["arguments"], dict):
        raise ValueError("Tool arguments must be a JSON object")
    if type(data["timeout_ms"]) is not int or not 1 <= data["timeout_ms"] <= 300_000:
        raise ValueError("isolated Tool timeout must be 1-300000 ms")
    if type(data["max_output_bytes"]) is not int or not 1 <= data["max_output_bytes"] <= 1_048_576:
        raise ValueError("isolated Tool output limit must be 1-1048576 bytes")
    return data


def package_path(root, name, digest):
    if not NAME_RE.fullmatch(name) or not DIGEST_RE.fullmatch(digest):
        raise ValueError("invalid package identity")
    path = root / name / digest
    if path.is_symlink() or (root / name).is_symlink() or not (path / "SKILL.md").is_file():
        raise ValueError("selected immutable Skill package is unavailable")
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError("package escaped configured store")
    return path


def require_package_approval(runtime_root, request):
    # Independent execution authorization, not merely knowing a store path.
    # Only the deployment administrator writes this policy; no MCP tool can.
    if request.get("operation") == "invoke_program":
        from .program import approved
        approved(runtime_root, request)
        return
    from .project_runtime import workspace_image
    project_image = workspace_image(runtime_root, request.get("workspace_id"))
    if project_image:
        if request.get("runtime_image") != project_image or request.get("package_name"):
            raise ValueError("project image and package binding cannot change")
        return
    image = read(runtime_root / "worker.json").get("image") if (runtime_root / "worker.json").is_file() else None
    if request.get("operation") == "invoke_tool":
        if request["runtime_image"] != image:
            raise ValueError("isolated Tool runtime image changed")
        bindings_path = runtime_root / "tool-bindings.json"
        bindings = read(bindings_path) if bindings_path.is_file() else {}
        binding = bindings.get(request["capability_id"])
        expected = {"revision": request["capability_revision"], "package_name": request["package_name"],
                    "package_digest": request["package_digest"], "image": image,
                    "runtime": request["runtime"], "entrypoint": request["entrypoint"]}
        if binding != expected:
            raise ValueError("this Tool revision is not approved for the isolated runtime")
        return
    if request.get("workspace_id") and not request.get("package_name"):
        policy = runtime_root / "task-code.json"
        approved = read(policy) if policy.is_file() else {}
        if not approved.get("enabled") or not image or approved.get("image") != image:
            raise ValueError("task code runtime is not approved for this image; ask an administrator to prepare it")
        return
    from .prepare_runtime import prepared_runtime
    if prepared_runtime(runtime_root, request, image) is not None:
        return
    bindings_path = runtime_root / "bindings.json"
    bindings = read(bindings_path) if bindings_path.is_file() else {}
    binding = bindings.get(request["package_name"] + ":" + request["package_digest"])
    if binding:
        if binding.get("image") != image:
            raise ValueError("Skill runtime image changed; an administrator must recheck its dependencies")
        if request["runtime"] != binding.get("runtime") and request["runtime"] not in binding.get("requirements", {}).get("commands", []):
            raise ValueError("runtime differs from the administrator-approved Skill binding")
        return
    policy = runtime_root / "packages.json"
    allowed = read(policy) if policy.is_file() else {}
    if request["package_digest"] not in allowed.get(request["package_name"], []):
        raise ValueError("this Skill revision is not approved for the isolated runtime; an administrator must bind the registered package")


def job_path(root, job_id):
    if not JOB_RE.fullmatch(job_id):
        raise ValueError("invalid job ID")
    path = root / job_id
    if path.is_symlink() or not path.is_dir():
        raise ValueError("job not found")
    return path
