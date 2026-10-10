from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from pathlib import Path

FILES = {"final.mp4": "video/mp4", "speech.wav": "audio/wav",
         "subtitles.srt": "application/x-subrip", "timeline.json": "application/json",
         "validation.json": "application/json"}


def digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def file_digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def encoded(value) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(encoded(value))
        stream.flush()
        os.fsync(stream.fileno())
    try:
        temporary.replace(path)
        fd = os.open(path.parent, os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    finally:
        temporary.unlink(missing_ok=True)


def read_json(path: Path):
    return json.loads(path.read_bytes())


class ObjectLookupError(ValueError):
    """Expected missing/invalid reference, not an internal service failure."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def object_path(root: Path, identifier: str, prefix: str) -> Path:
    if not re.fullmatch(re.escape(prefix) + r"[0-9a-f]{64}", identifier):
        raise ObjectLookupError("invalid_object_id", "invalid object ID")
    path = root / identifier
    if not path.is_dir() or path.is_symlink():
        raise ObjectLookupError("object_not_found", "object not found")
    return path


def renderer_revision(skill: Path) -> str:
    return digest(encoded({p.name: file_digest(p) for p in sorted((skill / "scripts").glob("*.py"))}))


def job_renderer_snapshot(job: Path, request: dict) -> Path:
    """Only original or content-addressed, job-local trusted components."""
    snapshot = request.get("renderer_snapshot", "snapshot")
    revision = request["renderer_revision"]
    if not re.fullmatch(r"[0-9a-f]{64}", revision) or snapshot not in {
        "snapshot", f"renderer-revisions/{revision}"
    }:
        raise ValueError("invalid renderer snapshot reference")
    path = job / snapshot
    if path.is_symlink() or (snapshot != "snapshot" and (job / "renderer-revisions").is_symlink()):
        raise ValueError("invalid renderer snapshot path")
    if not path.is_dir() or not path.resolve().is_relative_to(job.resolve()):
        raise ValueError("renderer snapshot not found in job")
    return path
