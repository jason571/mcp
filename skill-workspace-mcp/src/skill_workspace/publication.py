"""Publish an immutable snapshot of selected workspace files without execution."""
import hashlib
import hmac

from .common import MIMES, MAX_OUTPUT, encoded, relative, write
from pathlib import Path
from . import delivery


def publish(runtime, workspace_id, revision, outputs, inspection_jobs=(), artifact_visibility="final"):
    state = runtime.workspaces.state(workspace_id, revision, idle=True)
    if artifact_visibility not in {"final", "internal"}:
        raise ValueError("artifact_visibility must be final or internal")
    if artifact_visibility == "internal" and state.get("delivery"):
        raise ValueError("internal publication cannot complete a declared final delivery")
    delivery.verify(runtime, state, outputs, inspection_jobs)
    if not outputs or len(outputs) > 12 or len(set(outputs)) != len(outputs):
        raise ValueError("select between 1 and 12 distinct existing files")
    if len({Path(p).name for p in outputs}) != len(outputs):
        raise ValueError("output basenames must be unique")
    files, total = [], 0
    for path in outputs:
        relative(path)
        mime = MIMES.get(Path(path).suffix.lower())
        if not mime:
            raise ValueError("output file type is not approved for publication")
        data = runtime.workspaces.bytes(workspace_id, path, state)
        total += len(data)
        if not data or total > MAX_OUTPUT:
            raise ValueError("empty output or publication exceeds 64 MiB")
        files.append((path, data, mime))
    identity = {"operation": "publish_workspace", "workspace_id": workspace_id, "revision": revision,
                "files": [{"path": p, "sha256": hashlib.sha256(b).hexdigest()} for p, b, _ in files]}
    if artifact_visibility == "internal":
        identity["artifact_visibility"] = "internal"
    job_id = "work_" + hmac.new(runtime.token.encode(), encoded(identity), hashlib.sha256).hexdigest()
    job = runtime.jobs / job_id
    if (job / "state.json").is_file():
        return delivery.completed(runtime.status(job_id), state.get("delivery"))
    job.mkdir(mode=0o700, exist_ok=True)
    published = job / "published"
    published.mkdir(mode=0o700, exist_ok=True)
    manifest = []
    for path, data, mime in files:
        name = Path(path).name
        target = published / name
        target.write_bytes(data)
        target.chmod(0o600)
        manifest.append({"filename": name, "content_type": mime, "size_bytes": len(data),
                         "sha256": hashlib.sha256(data).hexdigest(),
                         "_meta": {"exoharness": {"artifact_identity": {
                             "key": hashlib.sha256(f"{workspace_id}:{path}".encode()).hexdigest(),
                             "revision": revision}}}})
        if artifact_visibility == "internal":
            manifest[-1]["_meta"]["exoharness"]["artifact_visibility"] = "internal"
    write(job / "result.json", {"files": manifest, "runtime_image": state.get("runtime_image"),
                                "package_name": state.get("package_name"), "package_digest": state.get("package_digest")})
    # No request.json: the execution broker must never pick up publication-only work.
    write(job / "execution.json", {"exit_code": 0, "log": "Existing files published; no code executed."})
    snapshot = {key: state.get(key) for key in ("revision", "package_name", "package_digest", "runtime_image")}
    snapshot.update(workspace_id=workspace_id, file_count=len(state["files"]))
    write(job / "state.json", {"status": "completed", "progress": 100, "message": "文件已发布", "code_executed": False,
                               **({"artifact_visibility": "internal"} if artifact_visibility == "internal" else {}),
                               "workspace": snapshot, **({"delivery": state["delivery"]} if state.get("delivery") else {})})
    return delivery.completed(runtime.status(job_id), state.get("delivery"))
