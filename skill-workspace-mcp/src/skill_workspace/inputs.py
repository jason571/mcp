"""Bounded file ingress in the controller, never network access in the sandbox."""
from __future__ import annotations

import hashlib
import http.client
import json
import re
import time
from urllib.error import HTTPError, URLError
from urllib.parse import unquote, urlsplit
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener

from .common import MAX_FILE

ASSET_ID = re.compile(r"att_[0-9a-f]{64}")
DIGEST = re.compile(r"[0-9a-f]{64}")


def validate_reference(item):
    url, asset = item.get("url"), item.get("attachment_id")
    if (url is not None) == (asset is not None):
        raise ValueError("supply exactly one input url or attachment_id returned by a tool")
    if url is not None and (not isinstance(url, str) or not 0 < len(url) <= 4096):
        raise ValueError("invalid input artifact URL")
    if asset is not None and (not isinstance(asset, str) or not ASSET_ID.fullmatch(asset)):
        raise ValueError("invalid input attachment_id")
    if item.get("sha256") is not None and (not isinstance(item["sha256"], str) or not DIGEST.fullmatch(item["sha256"])):
        raise ValueError("invalid input SHA-256")


def parsed_url(url):
    if not isinstance(url, str) or not url or len(url) > 4096 or any(ord(c) <= 32 or ord(c) == 127 for c in url):
        raise ValueError("invalid artifact URL")
    parsed = urlsplit(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username is not None or parsed.password is not None or parsed.fragment:
        raise ValueError("artifact URL must use HTTP(S) without credentials or fragments")
    path = unquote(parsed.path, errors="strict")
    if "\\" in path or "%" in path or any(ord(c) < 32 or ord(c) == 127 for c in path) or any(p in {".", ".."} for p in path.split("/")):
        raise ValueError("invalid artifact URL path")
    # Normalize default ports without loosening the administrator's origin.
    origin = (parsed.scheme, parsed.hostname.lower(), parsed.port or (443 if parsed.scheme == "https" else 80))
    return parsed, origin, path


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class ArtifactInputs:
    def __init__(self, asset_base="", download_bases=()):
        self.asset_base = asset_base.rstrip("/")
        self.download_bases = tuple(base.strip().rstrip("/") for base in download_bases if base.strip())
        for base in ([self.asset_base] if self.asset_base else []) + list(self.download_bases):
            parsed, _, _ = parsed_url(base)
            if parsed.query:
                raise ValueError("administrator artifact bases must not include a query")
        for base in self.download_bases:
            if not parsed_url(base)[2].strip("/"):
                raise ValueError("artifact download bases must include a file-route prefix, not an entire service")

    def info(self):
        return {"attachment_id_ready": bool(self.asset_base), "download_bases": list(self.download_bases),
                "max_file_bytes": MAX_FILE,
                "instructions": "For uploaded or model-generated files, prefer the exact returned attachment_id (not an authenticated chat download URL). Otherwise use a returned artifact URL under download_bases, or this workspace service's signed download_url. No guessed URLs, login cookies, arbitrary URLs or host paths. Files are copied before sandbox execution."}

    @staticmethod
    def read_url(url, limit):
        # No environment proxies, cookies, forwarded authentication, redirects,
        # or decompression. Only URLs checked against administrator bases reach here.
        opener = build_opener(ProxyHandler({}), NoRedirect())
        started = time.monotonic()
        try:
            with opener.open(Request(url, headers={"Accept-Encoding": "identity"}), timeout=15) as response:
                if response.status != 200:
                    raise ValueError(f"artifact download failed (HTTP {response.status})")
                if response.headers.get("Content-Encoding", "identity").lower() != "identity":
                    raise ValueError("encoded artifact responses are not accepted")
                declared = response.headers.get("Content-Length")
                if declared is not None and (not declared.isdecimal() or not 0 < int(declared) <= limit):
                    raise ValueError("artifact exceeds input size limit or has invalid length")
                chunks, count = [], 0
                while True:
                    if time.monotonic() - started > 90:
                        raise ValueError("artifact download exceeded 90 seconds")
                    chunk = response.read1(min(65536, limit + 1 - count))
                    if not chunk:
                        break
                    count += len(chunk)
                    if count > limit:
                        raise ValueError("artifact exceeds input size limit")
                    chunks.append(chunk)
                if not count or (declared is not None and count != int(declared)):
                    raise ValueError("artifact download is empty or incomplete")
                return b"".join(chunks)
        except HTTPError as error:
            # Never echo an authenticated URL/query in tool errors.
            raise ValueError(f"artifact download failed (HTTP {error.code}); use an exact readable artifact reference") from None
        except (URLError, TimeoutError, OSError, http.client.HTTPException):
            raise ValueError("artifact download transport failed; existing workspace files were not changed") from None

    def attachment(self, asset_id):
        if not ASSET_ID.fullmatch(asset_id):
            raise ValueError("invalid input attachment_id")
        if not self.asset_base:
            raise ValueError("attachment input service is not configured; administrator must set WORKSPACE_ASSET_BASE")
        url = self.asset_base + "/v1/attachment-assets/" + asset_id
        metadata = json.loads(self.read_url(url + "/metadata", 65536))
        if not isinstance(metadata, dict) or metadata.get("attachment_id") != asset_id:
            raise ValueError("attachment metadata identity mismatch")
        size, digest = metadata.get("size_bytes"), metadata.get("sha256")
        if type(size) is not int or not 0 < size <= MAX_FILE or not isinstance(digest, str) or not DIGEST.fullmatch(digest):
            raise ValueError("attachment metadata size or SHA-256 is invalid")
        data = self.read_url(url, size)
        if len(data) != size or hashlib.sha256(data).hexdigest() != digest:
            raise ValueError("attachment integrity mismatch")
        return data

    def url(self, url):
        parsed, origin, path = parsed_url(url)
        if self.asset_base:
            _, asset_origin, asset_path = parsed_url(self.asset_base)
            prefix = asset_path.rstrip("/") + "/v1/attachment-assets/"
            if origin == asset_origin and path.startswith(prefix) and not parsed.query:
                return self.attachment(path[len(prefix):])
        if "/v1/run-artifacts/" in path:
            raise ValueError("chat download URLs require user login; supply the tool's returned attachment_id or original artifact URL instead")
        for base in self.download_bases:
            _, allowed_origin, prefix = parsed_url(base)
            if origin == allowed_origin and path.startswith(prefix.rstrip("/") + "/"):
                return self.read_url(url, MAX_FILE)
        raise ValueError("input URL is not an approved artifact source; use the returned attachment_id or inspect_runtime input_sources, not a guessed URL")
