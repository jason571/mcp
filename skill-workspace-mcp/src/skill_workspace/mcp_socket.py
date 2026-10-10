"""Trusted Unix-socket bridge for isolated stdio MCP sessions.

The model and uploaded package cannot access this socket. Only the broker has
Docker authority; each authorized session gets a fresh read-only, offline OCI
container. JSON-RPC bytes are relayed unchanged, including progress and cancel.
"""
from __future__ import annotations

import json
import hashlib
import os
import re
import secrets
import socketserver
import stat
import subprocess
import tempfile
import threading
from pathlib import Path

from .broker import readable_package_snapshot
from .common import read
from .prepare_runtime import identity
from .public_https_proxy import parse_policy

HANDSHAKE_FIELDS = {"capability_id", "revision", "package_name", "package_digest", "image", "runtime", "entrypoint", "secrets"}
RUNTIMES = {"python": ("python",), "python_module": ("python", "-m"),
            "node": ("node",), "shell": ("bash", "--noprofile", "--norc"),
            "typescript": ("tsx",)}


def validate_binding(root: Path, packages: Path, image: str, request: dict):
    if not isinstance(request, dict) or set(request) != HANDSHAKE_FIELDS:
        raise ValueError("invalid isolated MCP handshake")
    if (not isinstance(request["capability_id"], str)
            or not re.fullmatch(r"mcp_server:[a-z0-9:_-]{1,240}", request["capability_id"])
            or type(request["revision"]) is not int or request["revision"] < 1):
        raise ValueError("invalid MCP capability identity")
    name, digest = request["package_name"], request["package_digest"]
    if (not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}", name)
            or not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest)
            or request["image"] != image or request["runtime"] not in RUNTIMES):
        raise ValueError("MCP package or image identity is invalid")
    from .common import relative
    entrypoint = relative(request["entrypoint"])
    suffixes = {"python": {".py"}, "node": {".js", ".mjs", ".cjs"},
                "shell": {".sh", ".bash"}, "typescript": {".ts", ".tsx", ".mts", ".cts"}}
    if request["runtime"] == "python_module":
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*", entrypoint):
            raise ValueError("invalid Python module entrypoint")
    elif Path(entrypoint).suffix not in suffixes[request["runtime"]]:
        raise ValueError("MCP entrypoint and runtime disagree")
    bindings = read(root / "mcp-bindings.json") if (root / "mcp-bindings.json").is_file() else {}
    binding = bindings.get(request["capability_id"])
    expected = {
        "revision": request["revision"], "package_name": name, "package_digest": digest,
        "image": image, "runtime": request["runtime"], "entrypoint": entrypoint,
    }
    if not isinstance(binding, dict) or any(binding.get(key) != value for key, value in expected.items()):
        raise ValueError("this MCP revision has not passed isolated preparation")
    hosts = binding.get("egress_hosts", [])
    if not isinstance(hosts, list) or (hosts and list(parse_policy(json.dumps(hosts))) != hosts):
        raise ValueError("MCP public HTTPS policy is invalid")
    secret_names = binding.get("secret_env_names", [])
    if (not isinstance(secret_names, list) or len(secret_names) > 16
            or any(not isinstance(name, str) or not re.fullmatch(r"[A-Z0-9_]{1,113}", name)
                   or name.startswith("EXOHARNESS_") for name in secret_names)
            or len(set(secret_names)) != len(secret_names)):
        raise ValueError("MCP secret policy is invalid")
    secrets = request["secrets"]
    if (not isinstance(secrets, dict) or set(secrets) != set(secret_names)
            or any(not isinstance(value, str) or not value or len(value.encode()) > 8192
                   or "\0" in value for value in secrets.values())):
        raise ValueError("MCP secret values do not match the bound names")
    package_root = packages.parent / "mcp"
    package = package_root / name / digest
    if (package_root.is_symlink() or (package_root / name).is_symlink() or package.is_symlink()
            or not package.is_dir() or not package.resolve().is_relative_to(package_root.resolve())):
        raise ValueError("selected immutable MCP package is unavailable")
    entry_file = package / (entrypoint.replace(".", "/") + "/__main__.py" if request["runtime"] == "python_module" else entrypoint)
    if entry_file.is_symlink() or not entry_file.is_file() or not entry_file.resolve().is_relative_to(package.resolve()):
        raise ValueError("MCP entrypoint escaped its immutable package")
    key = identity("mcp:" + name, digest, image)
    prepared = root / "imports" / key
    if not (prepared / "state.json").is_file() or read(prepared / "state.json").get("status") != "ready":
        raise ValueError("MCP runtime dependencies are not ready")
    if not (prepared / "runtime" / "verified.json").is_file() or read(prepared / "runtime" / "verified.json").get("status") != "ready":
        raise ValueError("MCP runtime verification is missing")
    return package, prepared / "runtime", hosts, secrets


def docker_command(name: str, package: Path, prepared: Path, image: str, request: dict,
                   network: str | None = None, has_secrets: bool = False):
    entrypoint = request["entrypoint"] if request["runtime"] == "python_module" else "/skill/" + request["entrypoint"]
    command = list(RUNTIMES[request["runtime"]]) + [entrypoint]
    command_line = ["docker", "run", "--rm", "-i", "--name", name,
            "--network", network or "none", "--read-only",
            "--user", "65532:65532", "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
            "--cpus", "2", "--memory", "2g", "--memory-swap", "2g", "--pids-limit", "128",
            "--ulimit", "nofile=256:256", "--ulimit", "fsize=67108864:67108864",
            "--tmpfs", "/tmp:rw,noexec,nosuid,size=128m,mode=1777",
            "--tmpfs", "/workspace:rw,nosuid,size=512m,uid=65532,gid=65532,mode=700",
            "--mount", f"type=bind,src={package},dst=/skill,readonly",
            "--mount", f"type=bind,src={prepared},dst=/runtime,readonly",
            "--env", "PYTHONPATH=/runtime/python:/skill", "--env", "NODE_PATH=/runtime/node/node_modules",
            "--env", "PYTHONNOUSERSITE=1", "--env", "HOME=/tmp", "--workdir", "/workspace"]
    if network:
        proxy = "http://mcp-egress:3128"
        for variable in ("HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy", "ALL_PROXY", "all_proxy"):
            command_line.extend(["--env", variable + "=" + proxy])
        command_line.extend(["--env", "NO_PROXY=", "--env", "no_proxy="])
    if has_secrets:
        script = Path(__file__).with_name("mcp_entry.py").resolve()
        command_line.extend(["--mount", f"type=bind,src={script},dst=/mcp-entry.py,readonly"])
        command = ["python", "-I", "/mcp-entry.py", *command]
    return [*command_line, image, *command]


def egress_instance(root: Path) -> str:
    return hashlib.sha256(str(root.resolve()).encode()).hexdigest()[:12]


def cleanup_orphaned_egress(root: Path):
    """The single-owner broker lock makes prior sessions orphaned on restart."""
    label = "exoharness.mcp-egress.instance=" + egress_instance(root)
    for noun in ("container", "network"):
        listing = (["docker", "ps", "-aq"] if noun == "container"
                   else ["docker", "network", "ls", "-q"])
        listed = subprocess.run([*listing, "--filter", "label=" + label],
            capture_output=True, text=True, timeout=30)
        if listed.returncode:
            raise ValueError("isolated MCP egress inventory is unavailable")
        for identifier in listed.stdout.splitlines():
            if not re.fullmatch(r"[0-9a-f]{12,64}", identifier):
                raise ValueError("isolated MCP egress inventory contains an invalid ID")
            action = ["docker", "rm", "-f", identifier] if noun == "container" else ["docker", "network", "rm", identifier]
            subprocess.run(action, capture_output=True, timeout=30)


class EgressSession:
    def __init__(self, root: Path, image: str, hosts: list[str]):
        suffix = secrets.token_hex(12)
        self.network = "exoharness-mcp-net-" + suffix
        self.proxy = "exoharness-mcp-egress-" + suffix
        self.image = image
        self.hosts = hosts
        self.instance = egress_instance(root)
        self.started = False

    def start(self):
        label = "exoharness.mcp-egress.instance=" + self.instance
        script = Path(__file__).with_name("public_https_proxy.py").resolve()
        try:
            subprocess.run(["docker", "network", "create", "--internal", "--label", label,
                self.network], check=True, capture_output=True, timeout=30)
            self.started = True
            subprocess.run(["docker", "run", "-d", "--rm", "--name", self.proxy,
                "--network", self.network, "--network-alias", "mcp-egress", "--label", label,
                "--read-only", "--user", "65532:65532", "--cap-drop", "ALL",
                "--security-opt", "no-new-privileges", "--cpus", "1", "--memory", "128m",
                "--memory-swap", "128m", "--pids-limit", "64",
                "--mount", f"type=bind,src={script},dst=/egress.py,readonly",
                "--env", "EXOHARNESS_EGRESS_HOSTS=" + json.dumps(self.hosts),
                self.image, "python", "-I", "/egress.py"],
                check=True, capture_output=True, timeout=30)
            subprocess.run(["docker", "network", "connect", "bridge", self.proxy],
                check=True, capture_output=True, timeout=30)
            for _ in range(30):
                probe = subprocess.run(["docker", "exec", self.proxy, "python", "-c",
                    "import socket; socket.create_connection(('127.0.0.1',3128),1).close()"],
                    capture_output=True, timeout=5)
                if probe.returncode == 0:
                    return self.network
                import time
                time.sleep(0.1)
            raise ValueError("public HTTPS proxy did not become ready")
        except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired, ValueError):
            self.close()
            raise

    def close(self):
        if self.started:
            subprocess.run(["docker", "rm", "-f", self.proxy], capture_output=True, timeout=30)
            subprocess.run(["docker", "network", "rm", self.network], capture_output=True, timeout=30)
            self.started = False


class McpSessionHandler(socketserver.BaseRequestHandler):
    def handle(self):
        ready = False
        name = None
        process = None
        egress = None
        acquired = False
        try:
            acquired = self.server.sessions_gate.acquire(timeout=30)
            if not acquired:
                raise ValueError("isolated MCP session capacity is full")
            wire = bytearray()
            while len(wire) <= 256 * 1024 and not wire.endswith(b"\n"):
                part = self.request.recv(1)
                if not part:
                    return
                wire.extend(part)
            if len(wire) > 256 * 1024:
                raise ValueError("MCP handshake exceeds limit")
            request = json.loads(wire)
            package, prepared, hosts, secret_values = validate_binding(self.server.root, self.server.packages,
                self.server.image, request)
            with tempfile.TemporaryDirectory(prefix=".mcp-session-", dir=self.server.sessions) as directory:
                staged = readable_package_snapshot(package, Path(directory) / "package")
                name = "exoharness-mcp-" + secrets.token_hex(16)
                if hosts:
                    egress = EgressSession(self.server.root, self.server.image, hosts)
                    egress.start()
                process = subprocess.Popen(docker_command(name, staged, prepared, self.server.image,
                    request, egress.network if egress else None, bool(secret_values)),
                    stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                    start_new_session=True)
                if secret_values:
                    process.stdin.write(json.dumps(secret_values, ensure_ascii=False).encode() + b"\n")
                    process.stdin.flush()
                self.request.sendall(b'{"status":"ready"}\n')
                ready = True
                incoming = threading.Thread(target=self._incoming, args=(process, name), daemon=True)
                incoming.start()
                total = 0
                line_bytes = 0
                while True:
                    block = os.read(process.stdout.fileno(), 8192)
                    if not block:
                        break
                    total += len(block)
                    if total > 128 * 1024 * 1024:
                        raise ValueError("MCP session output exceeded 128 MiB")
                    parts = block.split(b"\n")
                    for index, chunk in enumerate(parts):
                        line_bytes += len(chunk)
                        if line_bytes > 16 * 1024 * 1024:
                            raise ValueError("MCP response exceeded 16 MiB")
                        if index < len(parts) - 1:
                            line_bytes = 0
                    self.request.sendall(block)
                incoming.join(timeout=1)
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as error:
            if not ready:
                try:
                    self.request.sendall(json.dumps({"status":"rejected", "error":str(error)[:300]}).encode() + b"\n")
                except OSError:
                    pass
        finally:
            if process is not None:
                if process.poll() is None:
                    process.kill()
                process.wait(timeout=5)
                for pipe in (process.stdin, process.stdout):
                    try:
                        pipe.close()
                    except (OSError, ValueError):
                        pass
            if name is not None:
                subprocess.run(["docker", "rm", "-f", name], capture_output=True, timeout=30)
            if egress is not None:
                egress.close()
            if acquired:
                self.server.sessions_gate.release()

    def _incoming(self, process, name):
        total = 0
        disconnected = False
        try:
            while True:
                block = self.request.recv(8192)
                if not block:
                    disconnected = True
                    break
                total += len(block)
                if total > 128 * 1024 * 1024:
                    break
                process.stdin.write(block)
                process.stdin.flush()
        except (OSError, BrokenPipeError, ValueError):
            disconnected = True
        finally:
            try:
                process.stdin.close()
            except OSError:
                pass
            if disconnected and process.poll() is None:
                subprocess.run(["docker", "rm", "-f", name], capture_output=True, timeout=30)


class McpSessionServer(socketserver.ThreadingMixIn, socketserver.UnixStreamServer):
    daemon_threads = True
    allow_reuse_address = False

    def __init__(self, root: Path, packages: Path, image: str):
        path = root / "mcp.sock"
        if path.exists() or path.is_symlink():
            mode = path.lstat().st_mode
            if not stat.S_ISSOCK(mode) or path.lstat().st_uid != os.getuid():
                raise ValueError("MCP broker socket path is not an owned socket")
            path.unlink()
        self.root, self.packages, self.image = root, packages, image
        self.sessions = root / "mcp-sessions"
        self.sessions.mkdir(mode=0o700, exist_ok=True)
        self.sessions_gate = threading.BoundedSemaphore(4)
        super().__init__(str(path), McpSessionHandler)
        os.chmod(path, 0o600)


def start(root: Path, packages: Path, image: str):
    server = McpSessionServer(root, packages, image)
    cleanup_orphaned_egress(root)
    threading.Thread(target=server.serve_forever, name="isolated-mcp-sessions", daemon=True).start()
    return server
