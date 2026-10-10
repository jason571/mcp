"""Download a public, licensed GitHub repository in a disposable container.

Only the trusted broker invokes this script. The request carries repository
coordinates, never arbitrary fetch URLs or executable commands. Untrusted
archive bytes are not executed and are bounded again by the API ZIP parser.
"""
from __future__ import annotations

import http.client
import ipaddress
import json
import os
from pathlib import Path
import re
import socket
import ssl
import tempfile
from urllib.parse import quote, urlsplit


MAX_ARCHIVE = 32 * 1024 * 1024
MAX_METADATA = 256 * 1024
COORDINATE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,99}\Z")
SHA = re.compile(r"[0-9a-f]{40}\Z")
BENCHMARK_PROXY = ipaddress.ip_network("198.18.0.0/15")


def admitted_addresses(addresses):
    # The target host is fixed to GitHub and TLS still verifies that hostname.
    # Some managed networks transparently route fixed HTTPS hosts through the
    # RFC 2544 benchmark range; only that explicit proxy range is admitted.
    public = [item for item in addresses if ipaddress.ip_address(item[4][0]).is_global]
    if public:
        return public
    proxy = [item for item in addresses if ipaddress.ip_address(item[4][0]) in BENCHMARK_PROXY]
    if proxy:
        return proxy
    raise ValueError("GitHub host resolved outside the admitted HTTPS egress")


class PinnedHttps(http.client.HTTPSConnection):
    def connect(self):
        addresses = admitted_addresses(socket.getaddrinfo(self.host, self.port, type=socket.SOCK_STREAM))
        last_error = None
        for family, socktype, proto, _, address in addresses:
            raw = None
            try:
                raw = socket.socket(family, socktype, proto)
                raw.settimeout(self.timeout)
                raw.connect(address)
                self.sock = self._context.wrap_socket(raw, server_hostname=self.host)
                return
            except OSError as error:
                last_error = error
                if raw is not None:
                    raw.close()
        raise OSError("GitHub connection failed") from last_error


def get(host: str, path: str, *, archive: bool = False) -> tuple[int, dict[str, str], bytes]:
    if host not in {"api.github.com", "codeload.github.com"} or not path.startswith("/"):
        raise ValueError("download host or path is not admitted")
    connection = PinnedHttps(host, timeout=25, context=ssl.create_default_context())
    try:
        connection.request("GET", path, headers={
            "User-Agent": "ExoHarness-Source-Installer",
            "Accept": "application/vnd.github+json" if host == "api.github.com" else "application/zip",
            "X-GitHub-Api-Version": "2022-11-28",
            "Connection": "close",
        })
        response = connection.getresponse()
        headers = {key.lower(): value for key, value in response.getheaders()}
        maximum = MAX_ARCHIVE if archive else MAX_METADATA
        body = response.read(maximum + 1)
        if len(body) > maximum:
            raise ValueError("GitHub response exceeds size limit")
        return response.status, headers, body
    finally:
        connection.close()


def fetch(request: dict, output: Path) -> dict:
    owner, repository = request["owner"], request["repository"]
    ref = request.get("ref")
    if not COORDINATE.fullmatch(owner) or not COORDINATE.fullmatch(repository):
        raise ValueError("invalid GitHub repository coordinates")
    if ref is not None and (not isinstance(ref, str) or not COORDINATE.fullmatch(ref)):
        raise ValueError("invalid GitHub ref")
    prefix = f"/repos/{quote(owner)}/{quote(repository)}"
    status, _, body = get("api.github.com", prefix)
    if status != 200:
        raise ValueError(f"public GitHub repository check failed (HTTP {status})")
    info = json.loads(body)
    if info.get("private") or info.get("full_name", "").lower() != f"{owner}/{repository}".lower():
        raise ValueError("repository is not a public GitHub project")
    license_info = info.get("license") or {}
    license_id = license_info.get("spdx_id")
    if not license_id or license_id == "NOASSERTION":
        raise ValueError("GitHub repository has no recognized open-source license")
    selected_ref = ref or info.get("default_branch")
    if not isinstance(selected_ref, str) or not COORDINATE.fullmatch(selected_ref):
        raise ValueError("GitHub default branch cannot be resolved safely")
    status, _, body = get("api.github.com", prefix + "/commits/" + quote(selected_ref))
    if status != 200:
        raise ValueError(f"GitHub commit resolution failed (HTTP {status})")
    commit = json.loads(body).get("sha", "")
    if not SHA.fullmatch(commit):
        raise ValueError("GitHub did not return a fixed commit SHA")
    status, headers, body = get("api.github.com", prefix + "/zipball/" + commit, archive=True)
    if status in {301, 302, 303, 307, 308}:
        redirect = urlsplit(headers.get("location", ""))
        if redirect.scheme != "https" or redirect.hostname != "codeload.github.com" or redirect.port is not None or redirect.query or redirect.fragment:
            raise ValueError("GitHub archive redirected outside the admitted host")
        if not redirect.path.lower().startswith(f"/{owner}/{repository}/".lower()) or commit not in redirect.path:
            raise ValueError("GitHub archive redirect does not match the fixed commit")
        status, _, body = get("codeload.github.com", redirect.path, archive=True)
    if status != 200 or not body.startswith(b"PK"):
        raise ValueError(f"GitHub archive download failed (HTTP {status})")
    output.mkdir(exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=output, delete=False) as staged:
        staged.write(body)
        staged.flush()
        os.fsync(staged.fileno())
    os.replace(staged.name, output / "archive.zip")
    return {"commit": commit, "license": license_id, "archive_bytes": len(body)}


def main():
    request = json.loads(Path("/source/request.json").read_text())
    result = fetch(request, Path("/source"))
    Path("/source/result.json").write_text(json.dumps(result))


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
        raise SystemExit(str(error)[:400]) from None
