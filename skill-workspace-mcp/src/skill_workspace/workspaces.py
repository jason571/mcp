"""Private, versioned task workspaces. Handles are unguessable read/write capabilities.

No host path is accepted and workers never mount this store. Only bounded,
verified regular-file bytes cross the sandbox boundary. Failed executions retain
files as well as logs, so repairs need not resend the original program.
"""
from __future__ import annotations

import base64
import hashlib
import re
import secrets
from pathlib import Path

from .common import MAX_FILE, MAX_OUTPUT, job_path, read, relative, write

WORKSPACE_RE = re.compile(r"ws_[0-9a-f]{64}")
MAX_FILES = 128
RECEIPT_TAIL_CHARS = 400
RECEIPT_MAX_IDS = 40
HTML_ID_RE = re.compile(r"""<([a-zA-Z][\w-]*)\b[^>]*?\sid\s*=\s*["']([^"']+)["']""")


def merge_appends(appends):
    """One append per path, preserving every submitted chunk in array order.

    No content repair, deduplication, newline insertion or additional write is
    performed. Conflicts with other edit modes are checked after this merge.
    """
    chunks = {}
    for item in appends:
        name = relative(item["path"])
        if not isinstance(item.get("content"), str):
            raise ValueError("append content must be UTF-8 text")
        chunks.setdefault(name, []).append(item["content"])
    return [{"path": name, "content": "".join(parts)} for name, parts in chunks.items()]


def change_receipt(name, data, entry):
    """Bounded confirmation of one changed text file; binary files get none.

    Enough structure to keep authoring (where the file ends, which anchors
    exist) so the model does not re-read a file it just wrote.
    """
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        return None
    receipt = {"path": name, "sha256": entry["sha256"], "total_chars": len(text),
               "line_count": text.count("\n") + (1 if text and not text.endswith("\n") else 0),
               "tail": text[-RECEIPT_TAIL_CHARS:]}
    if Path(name).suffix.lower() in {".html", ".htm"}:
        ids = [f"{tag}#{value}" for tag, value in HTML_ID_RE.findall(text)]
        if ids:
            receipt["element_ids"] = ids[:RECEIPT_MAX_IDS]
            if len(ids) > RECEIPT_MAX_IDS:
                receipt["element_ids_truncated"] = len(ids) - RECEIPT_MAX_IDS
    return receipt


class Workspaces:
    def __init__(self, root: Path):
        self.root = root / "workspaces"
        self.root.mkdir(mode=0o700, parents=True, exist_ok=True)

    def directory(self, handle):
        if not isinstance(handle, str) or not WORKSPACE_RE.fullmatch(handle):
            raise ValueError("invalid workspace handle")
        path = self.root / handle
        if path.is_symlink() or not (path / "state.json").is_file():
            raise ValueError("workspace not found")
        return path

    def create(self, package_name=None, package_digest=None, runtime_image=None, delivery=None, instruction_reference=None):
        handle = "ws_" + secrets.token_hex(32)
        path = self.root / handle
        path.mkdir(mode=0o700)
        (path / "blobs").mkdir(mode=0o700)
        write(path / "state.json", {"workspace_id": handle, "revision": 0, "files": {},
              "package_name": package_name, "package_digest": package_digest, "runtime_image": runtime_image,
              **({"delivery": delivery} if delivery else {}),
              **({"instruction_reference": instruction_reference} if instruction_reference else {})})
        return self.inspect(handle)

    def state(self, handle, revision=None, idle=False):
        path = self.directory(handle)
        state = read(path / "state.json")
        if revision is not None and state["revision"] != revision:
            raise ValueError(f"workspace revision conflict: expected {revision}, current {state['revision']}; read it before patching")
        active = state.get("active_job")
        if idle and active:
            job = job_path(self.root.parent / "jobs", active)
            if read(job / "state.json")["status"] not in {"completed", "failed", "cancelled"}:
                raise ValueError("workspace is busy; wait or cancel its active job before editing or executing")
        return state

    def inspect(self, handle, path=None, offset=0, limit=8000, query=None):
        state = self.state(handle)
        result = {key: state[key] for key in ["workspace_id", "revision", "package_name", "package_digest"]}
        result["runtime_image"] = state.get("runtime_image")
        if state.get("instruction_reference"):
            result["instruction_reference"] = state["instruction_reference"]
        if state.get("delivery"):
            result["delivery"] = state["delivery"]
        if "notes" in state:
            result["notes"] = state["notes"]
        if path is None:
            if query is not None:
                raise ValueError("a text query requires a file path")
            result["files"] = [{"path": name, **item} for name, item in sorted(state["files"].items())]
        else:
            if not 0 <= offset or not 1 <= limit <= 16000:
                raise ValueError("invalid read range")
            data = self.bytes(handle, relative(path), state)
            try:
                text = data.decode("utf-8")
            except UnicodeDecodeError as error:
                raise ValueError("file is binary; use it in run_script without sending its bytes to the model") from error
            digest = state["files"][path]["sha256"]
            result.update(path=path, sha256=digest, total_chars=len(text), offset_unit="unicode_characters")
            if query is None:
                start, end = min(offset, len(text)), min(offset + limit, len(text))
                result.update(content=text[start:end], offset=start,
                              next_offset=end if end < len(text) else None)
                ranges = [[start, end]]
            else:
                if not isinstance(query, str) or not query or len(query) > 256:
                    raise ValueError("query must be a non-empty literal string of at most 256 characters")
                matches, remaining, cursor = [], limit, offset
                while remaining and len(matches) < 4:
                    found = text.find(query, cursor)
                    if found < 0:
                        break
                    # Bounded source snippets, not a whole-file scan through model requests.
                    start = max(0, found - min(160, remaining // 4))
                    end = min(len(text), start + min(remaining, 640))
                    matches.append({"match_offset": found, "offset": start, "end_offset": end,
                                    "line": text.count("\n", 0, found) + 1, "content": text[start:end]})
                    remaining -= end - start
                    cursor = max(end, found + len(query))
                next_match = text.find(query, cursor)
                result.update(query=query, matches=matches,
                              next_offset=cursor if next_match >= 0 else None)
                ranges = [[m["offset"], m["end_offset"]] for m in matches]
            result["_meta"] = {"exoharness": {"immutable_read": {
                "version": 1, "resource": handle + "/" + path, "revision": digest,
                "selection": "text" if query is None else "query:" + query, "ranges": ranges}}}
        return result

    def bytes(self, handle, name, state=None):
        state = state or self.state(handle)
        item = state["files"].get(name)
        if item is None:
            raise ValueError("workspace file not found: " + name)
        data = (self.directory(handle) / "blobs" / item["sha256"]).read_bytes()
        if len(data) != item["size_bytes"] or hashlib.sha256(data).hexdigest() != item["sha256"]:
            raise ValueError("workspace file integrity mismatch")
        return data

    def persist(self, handle, files):
        if len(files) > MAX_FILES or sum(len(data) for data in files.values()) > MAX_OUTPUT:
            raise ValueError("workspace exceeds 128 files or 64 MiB; remove temporary files in the sandbox")
        manifest = {}
        names = set(files)
        for name in names:
            relative(name)
            if any(parent.as_posix() in names for parent in Path(name).parents if parent != Path(".")):
                raise ValueError("workspace file conflicts with a parent directory")
        blobs = self.directory(handle) / "blobs"
        existing = {path.name: path.stat().st_size for path in blobs.iterdir() if path.is_file()}
        additions = {hashlib.sha256(data).hexdigest(): len(data) for data in files.values()}
        if sum(existing.values()) + sum(size for digest, size in additions.items() if digest not in existing) > 256 * 1024 * 1024:
            raise ValueError("workspace retained-version storage exceeds 256 MiB; ask an administrator to archive it")
        for name, data in files.items():
            relative(name)
            if len(data) > MAX_FILE:
                raise ValueError("workspace file exceeds 32 MiB")
            digest = hashlib.sha256(data).hexdigest()
            target = self.directory(handle) / "blobs" / digest
            if not target.exists():
                target.write_bytes(data)
                target.chmod(0o600)
            manifest[name] = {"sha256": digest, "size_bytes": len(data)}
        return manifest

    def edit(self, handle, revision, files, patches, inputs, appends=(), notes=None):
        if notes is not None and (not isinstance(notes, str) or len(notes) > 2000):
            raise ValueError("workspace notes must be text of at most 2000 characters; retain concise shared interfaces, not source")
        state = self.state(handle, revision, idle=True)
        appends = merge_appends(appends)
        values = {name: self.bytes(handle, name, state) for name in state["files"]}
        names = set()
        for item in [*files, *inputs, *appends]:
            name = relative(item["path"])
            if name in names:
                raise ValueError("a file may only be changed once in a single edit")
            names.add(name)
        patch_groups = {}
        for item in patches:
            name = relative(item["path"])
            if name in names:
                raise ValueError("a file may only be changed once by one edit mode; do not mix patches with writes/imports/appends")
            patch_groups.setdefault(name, []).append(item)
        names.update(patch_groups)
        for item in files:
            values[item["path"]] = item["content"].encode()
        for item in inputs:
            values[item["path"]] = item["bytes"]
        for item in appends:
            name = item["path"]
            if name not in values:
                raise ValueError("append target does not exist; create the file once with files")
            try:
                values[name].decode("utf-8")
            except UnicodeDecodeError as error:
                raise ValueError("append target must be UTF-8 text") from error
            values[name] += item["content"].encode("utf-8")
        for name, group in patch_groups.items():
            if name not in values or any(hashlib.sha256(values[name]).hexdigest() != item["sha256"] for item in group):
                raise ValueError("patch base SHA-256 mismatch; read the file before editing")
            text = values[name].decode("utf-8")
            changes = []
            # Every hunk refers to the SAME saved base, not the preceding
            # replacement. Validate the entire batch before publishing a revision.
            for item in group:
                before = item["before"]
                start = text.find(before)
                if not before or start < 0 or text.find(before, start + 1) >= 0:
                    raise ValueError("patch must match exactly once; original files were not changed")
                changes.append((start, start + len(before), item["after"]))
            changes.sort()
            if any(a[1] > b[0] for a, b in zip(changes, changes[1:])):
                raise ValueError("patch ranges overlap in the original file; original files were not changed")
            for start, end, after in reversed(changes):
                text = text[:start] + after + text[end:]
            values[name] = text.encode()
        state["files"] = self.persist(handle, values)
        if notes is not None:
            state["notes"] = notes
        # Explicitly authored/imported files are not disposable inline snippets.
        # Keep this provenance separate from their content hashes and revisions.
        state["authored_files"] = sorted(set(state.get("authored_files", [])) | names)
        state["revision"] += 1
        state.pop("active_job", None)
        write(self.directory(handle) / "state.json", state)
        result = self.inspect(handle)
        # A hash alone gives the model no way to confirm what it just wrote,
        # so it re-reads its own output page by page. A bounded tail and size
        # per changed text file answer "did the append/patch land, is the
        # file complete" without another read cycle.
        changed = [name for name in [*(item["path"] for item in files), *(item["path"] for item in appends), *patch_groups]]
        result["changes"] = [receipt for receipt in (change_receipt(name, values[name], state["files"][name]) for name in changed) if receipt]
        return result

    def capture(self, request, job_id, items):
        handle = request.get("workspace_id")
        if not handle:
            return
        state = self.state(handle)
        if state.get("last_job") == job_id:
            return
        if state.get("active_job") != job_id or state["revision"] != request["workspace_revision"]:
            raise ValueError("sandbox snapshot does not match the active workspace revision")
        values = {}
        if not isinstance(items, list) or len(items) > MAX_FILES:
            raise ValueError("invalid sandbox workspace snapshot")
        total = 0
        for item in items:
            name = relative(item["path"])
            if name in values or item["size_bytes"] > MAX_FILE or len(item["data"]) > (MAX_FILE * 4 // 3 + 4):
                raise ValueError("invalid sandbox snapshot file")
            data = base64.b64decode(item["data"], validate=True)
            total += len(data)
            if total > MAX_OUTPUT or len(data) != item["size_bytes"] or hashlib.sha256(data).hexdigest() != item["sha256"]:
                raise ValueError("sandbox snapshot size or integrity mismatch")
            values[name] = data
        state["files"] = self.persist(handle, values)
        state.update(revision=state["revision"] + 1, last_job=job_id)
        state.pop("active_job", None)
        write(self.directory(handle) / "state.json", state)
