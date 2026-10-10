"""Trusted import-time dependency preparation, never a model-callable tool.

Only the management API queues requests. Package installers run in a disposable
container without host credentials; wheels only / npm lifecycle scripts disabled.
Normal task containers remain offline. No Docker image is built or mutated.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import re
import subprocess
import time

from .common import package_path, read, write


def identity(name, digest, image):
    return hashlib.sha256(f"{name}\0{digest}\0{image}".encode()).hexdigest()


def prepared_runtime(root, request, image):
    if not request.get("package_name"):
        return None
    prefix = "tool:" if request.get("operation") == "invoke_tool" else "mcp:" if request.get("operation") == "invoke_mcp" else ""
    name = prefix + request["package_name"]
    key = identity(name, request["package_digest"], image)
    path = root / "imports" / key
    if not (path / "state.json").is_file():
        return None
    state = read(path / "state.json")
    if state.get("status") != "ready" or state.get("image") != image:
        raise ValueError("Skill dependencies are not ready for this sandbox image")
    if not (path / "runtime" / "verified.json").is_file():
        raise ValueError("Skill dependency verification is missing")
    if read(path / "runtime" / "verified.json").get("status") != "ready":
        raise ValueError("Skill dependency verification is invalid")
    return path / "runtime"


def preparation_args(job, package, image, script, offline=False, prepare=False):
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", image):
        raise ValueError("dependency preparation requires the approved immutable image")
    # Match the non-root broker only for writes into its private preparation
    # directory. No host home or credentials are mounted. Task UID stays 65532.
    uid, gid = os.getuid(), os.getgid()
    if uid == 0:
        raise ValueError("dependency preparation broker must run as a non-root user")
    dns_order = os.environ.get("WORKSPACE_DEPENDENCY_DNS_ORDER", "ipv4first")
    if dns_order not in {"ipv4first", "ipv6first", "verbatim"}:
        raise ValueError("WORKSPACE_DEPENDENCY_DNS_ORDER must be ipv4first, ipv6first, or verbatim")
    args = ["run", "--rm", "--name", "exoharness-prepare-" + job.name,
            "--network", "none" if offline else "bridge", "--read-only", "--user", f"{uid}:{gid}",
            "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
            "--cpus", "2", "--memory", "2g", "--memory-swap", "2g", "--pids-limit", "128",
            "--ulimit", "fsize=268435456:268435456",
            "--tmpfs", "/tmp:rw,nosuid,size=512m,mode=1777",
            "--mount", f"type=bind,src={package},dst=/skill,readonly",
            "--mount", f"type=bind,src={job / 'runtime'},dst=/runtime" + (",readonly" if offline and not prepare else ""),
            "--mount", f"type=bind,src={script},dst=/prepare.py,readonly",
            "--mount", f"type=bind,src={script.with_name('typescript.cjs')},dst=/typescript.cjs,readonly",
            "--env", "HOME=/tmp", "--env", "PYTHONPATH=", "--env", "PYTHONNOUSERSITE=1",
            "--env", "NPM_CONFIG_GIT=/bin/false", "--workdir", "/tmp",
            "--env", "NODE_OPTIONS=--dns-result-order=" + dns_order,
            image, "python", "-I", "/prepare.py", "prepare" if prepare else "check" if offline else "install"]
    return args


def process_pending(root, packages, image, heartbeat):
    from .broker import readable_package_snapshot

    queue = root / "imports"
    if not queue.is_dir():
        return
    for job in sorted(queue.iterdir()):
        if job.is_symlink() or not job.is_dir() or not re.fullmatch(r"[0-9a-f]{64}", job.name):
            continue
        if not (job / "request.json").is_file() or not (job / "state.json").is_file():
            continue
        try:
            if read(job / "state.json").get("status") not in {"queued", "preparing"}:
                continue
            request = read(job / "request.json")
            name, digest = request["package_name"], request["package_digest"]
            kind = request.get("package_kind", "skill")
            if kind not in {"skill", "tool", "mcp"}:
                raise ValueError("unsupported package preparation kind")
            package_identity = (kind + ":" if kind != "skill" else "") + name
            if job.name != identity(package_identity, digest, image) or request.get("image") != image:
                raise ValueError("import preparation snapshot no longer matches the approved image")
            if kind in {"tool", "mcp"}:
                package_root = packages.parent / ("tools" if kind == "tool" else "mcp")
                package = package_root / name / digest
                if (package_root.is_symlink() or (package_root / name).is_symlink() or package.is_symlink()
                        or not package.is_dir() or not package.resolve().is_relative_to(package_root.resolve())):
                    raise ValueError("selected immutable executable package is unavailable")
            else:
                package = package_path(packages, name, digest)
            package = readable_package_snapshot(package, job / "skill")
            target = job / "runtime"
            target.mkdir(mode=0o755, exist_ok=True)
            write(job / "state.json", {"status": "preparing", "image": image, "message": "正在准备并检查沙盒依赖"})
            # A restart may have left a detached installer. Remove only this
            # exact preparation container before the idempotent install phase.
            subprocess.run(["docker", "rm", "-f", "exoharness-prepare-" + job.name],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=30)
            for offline, prepare in [(False, False), (True, True), (True, False)]:
                args = preparation_args(job, package, image, Path(__file__).with_name("runtime_installer.py").resolve(), offline, prepare)
                log_path = job / ("prepare.log" if prepare else "check.log" if offline else "install.log")
                with log_path.open("wb") as output:
                    process = subprocess.Popen(["docker", *args], stdout=output, stderr=subprocess.STDOUT)
                    started = time.monotonic()
                    next_size_check = started
                    try:
                        while process.poll() is None:
                            write(heartbeat, {"at": time.time(), "image": image})
                            if time.monotonic() - started > (300 if offline else 900) or output.tell() > 2 * 1024 * 1024:
                                raise ValueError("dependency preparation exceeded time/output limit")
                            if time.monotonic() >= next_size_check:
                                count, size = 0, 0
                                for file in target.rglob("*"):
                                    if file.is_file() and not file.is_symlink():
                                        size += file.stat().st_size
                                    count += 1
                                    if size > 2 * 1024 ** 3 or count > 100000:
                                        raise ValueError("dependency preparation exceeds 2 GiB / 100000 files")
                                next_size_check = time.monotonic() + 5
                            time.sleep(1)
                        if process.returncode:
                            with log_path.open("rb") as log:
                                log.seek(max(0, log_path.stat().st_size - 1600))
                                detail = log.read(1600).decode(errors="replace")
                            raise ValueError("依赖准备或检查失败：" + detail)
                    finally:
                        if process.poll() is None:
                            process.kill()
                            process.wait()
                            subprocess.run(["docker", "rm", "-f", "exoharness-prepare-" + job.name],
                                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=30)
            verified = read(target / "verified.json")
            if verified.get("status") != "ready":
                raise ValueError("dependency verifier did not confirm readiness")
            target.chmod(0o755)
            # Approval is scoped to the exact package AND base image. No global
            # interpreter grant and no business code execution occurs here.
            write(job / "state.json", {"status": "ready", "image": image,
                  "message": "沙盒依赖检查通过", "checks": verified})
        except Exception as error:
            write(job / "state.json", {"status": "failed", "image": image, "message": str(error)[:2000]})
