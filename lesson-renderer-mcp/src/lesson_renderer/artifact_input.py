"""Read a signed Artifact only at the administrator-configured asset origin."""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlsplit, urlunsplit

from .store import file_digest


def addresses(value: str, configured_base: str) -> tuple[str, str, str]:
    url, trusted = urlsplit(value), urlsplit(configured_base)
    port = lambda u: u.port or (443 if u.scheme == "https" else 80)
    if (url.scheme, url.hostname, port(url)) != (trusted.scheme, trusted.hostname, port(trusted)):
        raise ValueError("artifact URL is not at the configured asset origin")
    if url.username or url.password or url.fragment:
        raise ValueError("artifact URL must not contain credentials or a fragment")
    prefix = trusted.path.rstrip("/") + "/artifacts/"
    if not url.path.startswith(prefix):
        raise ValueError("artifact URL must use the signed artifacts endpoint")
    parts = url.path[len(prefix):].split("/")
    if len(parts) != 2 or not re.fullmatch(r"artifact-[a-f0-9]{64}", parts[0]):
        raise ValueError("invalid artifact identity; use the original TTS audio.url")
    name = unquote(parts[1])
    if not name or name in {".", ".."} or any(c in name for c in "/\\\x00\r\n"):
        raise ValueError("invalid artifact filename")
    query = parse_qs(url.query, keep_blank_values=True)
    if set(query) != {"expires", "signature"} or any(len(v) != 1 for v in query.values()):
        raise ValueError("use the exact signed audio.url, not a download or unsigned URL")
    if not re.fullmatch(r"[0-9]+", query["expires"][0]) or not re.fullmatch(r"[a-f0-9]{64}", query["signature"][0]):
        raise ValueError("invalid artifact read signature")
    return value, urlunsplit(url._replace(path=url.path + "/metadata")), parts[0]


async def fetch_artifact(session, value: str, configured_base: str, destination: Path, allowed: set[str]) -> dict:
    source, metadata_url, identity = addresses(value, configured_base)
    async with session.get(metadata_url, allow_redirects=False) as response:
        if response.status != 200:
            raise ValueError(f"signed artifact metadata unavailable (HTTP {response.status})")
        raw = await response.content.read(65537)
        if len(raw) > 65536:
            raise ValueError("artifact metadata too large")
        metadata = json.loads(raw)
    if metadata.get("artifact_id") != identity or metadata.get("content_type") not in allowed:
        raise ValueError("artifact identity/type mismatch")
    size = metadata.get("size_bytes")
    sha = metadata.get("sha256")
    if type(size) is not int or not 0 < size <= 32 * 1024 * 1024 or not isinstance(sha, str) or not re.fullmatch(r"[a-f0-9]{64}", sha):
        raise ValueError("artifact size or digest invalid")
    if destination.is_file() and destination.stat().st_size == size and file_digest(destination) == sha:
        return metadata
    temp = destination.with_suffix(".partial")
    try:
        count = 0
        async with session.get(source, allow_redirects=False) as response:
            if response.status != 200:
                raise ValueError(f"signed artifact unavailable (HTTP {response.status})")
            with temp.open("wb") as stream:
                async for block in response.content.iter_chunked(65536):
                    count += len(block)
                    if count > size:
                        raise ValueError("artifact larger than declared")
                    stream.write(block)
                stream.flush()
                os.fsync(stream.fileno())
        if count != size or file_digest(temp) != sha:
            raise ValueError("artifact integrity check failed")
        temp.replace(destination)
        return metadata
    finally:
        temp.unlink(missing_ok=True)
