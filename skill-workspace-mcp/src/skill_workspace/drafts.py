"""Private, bounded source retention for rejected requests. Never executes code."""
from __future__ import annotations

import hashlib
import re

from .common import MAX_INPUT, encoded, read, relative, write


def save(workspaces, handle, files):
    if not isinstance(files, list) or not 1 <= len(files) <= 24:
        raise ValueError("invalid draft files")
    names = set()
    for item in files:
        name = relative(item["path"])
        if name in names or not isinstance(item.get("content"), str):
            raise ValueError("invalid draft source")
        names.add(name)
    data = {"workspace_id": handle, "files": files}
    blob = encoded(data)
    if len(blob) > MAX_INPUT:
        raise ValueError("draft exceeds source limit")
    root = workspaces.directory(handle) / "drafts"
    root.mkdir(mode=0o700, exist_ok=True)
    digest = hashlib.sha256(blob).hexdigest()
    target = root / digest
    if not target.exists():
        if sum(p.stat().st_size for p in root.iterdir() if p.is_file()) + len(blob) > 16 * MAX_INPUT:
            raise ValueError("workspace draft storage limit reached")
        write(target, data)
        target.chmod(0o600)
    return {"id": digest, "files": [{"path": f["path"], "size_bytes": len(f["content"].encode()),
            "sha256": hashlib.sha256(f["content"].encode()).hexdigest()} for f in files]}


def load(workspaces, handle, digest):
    if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise ValueError("invalid draft ID")
    path = workspaces.directory(handle) / "drafts" / digest
    if path.is_symlink() or not path.is_file() or path.stat().st_size > MAX_INPUT:
        raise ValueError("draft not found in this workspace")
    data = read(path)
    if data.get("workspace_id") != handle or hashlib.sha256(encoded(data)).hexdigest() != digest:
        raise ValueError("draft integrity mismatch")
    return data["files"]


def restore(workspaces, handle, revision, digest):
    files = load(workspaces, handle, digest)
    state = workspaces.state(handle, revision, idle=True)
    for item in files:
        if item["path"] in state["files"] and workspaces.bytes(handle, item["path"], state) != item["content"].encode():
            raise ValueError("draft would overwrite an existing file; inspect and patch that file explicitly")
    pending = [f for f in files if f["path"] not in state["files"]]
    return workspaces.edit(handle, revision, pending, [], []) if pending else workspaces.inspect(handle)
