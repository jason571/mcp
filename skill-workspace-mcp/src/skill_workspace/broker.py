"""Trusted host broker. Only this process can access Docker, never uploaded code.

Uses the administrator's existing Docker authority. It accepts no image,
command, host mount, network or Docker options from MCP arguments.
"""
from __future__ import annotations

import base64
import fcntl
import hashlib
import json
import os
import re
import stat
import subprocess
import tempfile
import time
from pathlib import Path

from .common import MAX_FILE, MAX_OUTPUT, MAX_WIRE, MIMES, browser_revision, encoded, job_path, package_path, read, require_package_approval, validate_request, write
from .workspaces import Workspaces


def docker(*args, check=True, timeout=30):
    result = subprocess.run(["docker", *args], capture_output=True, timeout=timeout)
    if check and result.returncode:
        raise RuntimeError("container runtime: " + result.stderr.decode(errors="replace")[:2000])
    return result


def container_args(job, package, image, runner, prepared=None, *, browser=False):
    # Immutable image ID is a deployment policy, never a model-controlled tag.
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", image):
        raise ValueError("configure an inspected immutable WORKSPACE_IMAGE_ID")
    args = ["create", "--name", "exoharness-" + job.name,
            "--label", "exoharness.skill-workspace=" + job.name,
            "--network", "none", "--read-only", "--user", "65532:65532",
            "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
            "--cpus", "2", "--memory", "2g", "--memory-swap", "2g", "--pids-limit", "256" if browser else "128",
            "--ulimit", "nofile=256:256", "--ulimit", "fsize=67108864:67108864",
            # Chromium uses /tmp for its profile and raster shared memory.
            # A vector-heavy page exceeded 128 MiB with a measured 177 MiB peak.
            # Keep this browser-only allocation bounded by the same 2 GiB cgroup.
            "--tmpfs", f"/tmp:rw,noexec,nosuid,size={'512m' if browser else '128m'},mode=1777",
            "--tmpfs", "/workspace:rw,nosuid,size=512m,uid=65532,gid=65532,mode=700",
            "--mount", f"type=bind,src={job / 'input'},dst=/input,readonly",
            "--mount", f"type=bind,src={runner},dst=/runner/sandbox.py,readonly",
            "--mount", f"type=bind,src={runner.with_name('typescript.cjs')},dst=/runner/typescript.cjs,readonly",
            "--log-driver", "local", "--log-opt", "max-size=192m", "--log-opt", "max-file=1", "--log-opt", "compress=false",
            "--workdir", "/workspace"]
    if package:
        args.extend(["--mount", f"type=bind,src={package},dst=/skill,readonly"])
    if prepared:
        args.extend(["--mount", f"type=bind,src={prepared},dst=/runtime,readonly"])
    if browser:
        args.extend(["--env", "LP_NUM_THREADS=2", "--env", "OMP_NUM_THREADS=2"])
    if runner.name == "program_runner.py":
        args.extend(["--mount", f"type=bind,src={job / 'rpc'},dst=/rpc"])
    return [*args, image, "python", "/runner/sandbox.py"]


def inspect(name):
    result = docker("inspect", name, check=False)
    if result.returncode:
        return None
    return json.loads(result.stdout)[0]


def readable_package_snapshot(package, destination):
    """Mount a private copy readable by the sandbox UID, never chmod the registry."""
    if destination.exists():
        if destination.is_symlink() or not destination.is_dir():
            raise ValueError("invalid staged Skill package")
        return destination  # Atomic snapshot from this exact immutable Job, e.g. after broker restart.
    count, total = 0, 0
    with tempfile.TemporaryDirectory(prefix=".skill-stage-", dir=destination.parent) as directory:
        staged = Path(directory) / "content"

        def copy_tree(source, target):
            nonlocal count, total
            target.mkdir(mode=0o755)
            for item in sorted(source.iterdir()):
                count += 1
                if count > 4096:
                    raise ValueError("Skill package exceeds entry limit")
                mode = item.lstat().st_mode
                copied = target / item.name
                if stat.S_ISDIR(mode):
                    copy_tree(item, copied)
                elif stat.S_ISREG(mode):
                    with os.fdopen(os.open(item, os.O_RDONLY | os.O_NOFOLLOW), "rb") as stream:
                        if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
                            raise ValueError("Skill package entry is not a regular file")
                        data = stream.read(8 * 1024 * 1024 + 1)
                    total += len(data)
                    if len(data) > 8 * 1024 * 1024 or total > 64 * 1024 * 1024:
                        raise ValueError("Skill package exceeds size limit")
                    copied.write_bytes(data)
                    copied.chmod(0o444)
                else:
                    raise ValueError("Skill package links and special files cannot be mounted")
            # The broker owns this copy; the different sandbox UID has no write
            # bits and Docker additionally mounts the entire tree read-only.
            target.chmod(0o755)

        copy_tree(package, staged)
        os.replace(staged, destination)
    return destination


def runtime_inventory(image):
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", image):
        raise ValueError("runtime inventory requires an immutable image ID")
    probe = """import importlib.metadata, json, pathlib, shutil, sys
packages = {d.metadata['Name'].lower().replace('_','-'): d.version for d in importlib.metadata.distributions() if d.metadata['Name']}
node = {}
root = pathlib.Path('/usr/local/lib/node_modules')
for p in root.glob('*/package.json'):
 d = json.loads(p.read_text()); node[d['name']] = d['version']
commands = {name: shutil.which(name) is not None for name in ['python','bash','node','tsc','tsx','ffmpeg','libreoffice','soffice','pdftoppm']}
typescript = {}
for name in ['typescript','tsx']:
 p = pathlib.Path('/opt/exoharness-typescript/node_modules') / name / 'package.json'
 if p.is_file(): typescript[name] = json.loads(p.read_text())['version']
import subprocess
node_version = subprocess.check_output(['node','--version'], text=True).strip() if commands['node'] else None
print(json.dumps({'python_packages': packages, 'node_packages': node, 'commands': commands, 'python_version': sys.version.split()[0], 'node_version':node_version,'typescript':typescript}))
"""
    response = docker("run", "--rm", "--network", "none", "--read-only", "--user", "65532:65532",
                      "--cap-drop", "ALL", "--security-opt", "no-new-privileges", "--memory", "256m",
                      image, "python", "-c", probe, timeout=60)
    return json.loads(response.stdout)


def finish(job, request, wire):
    result = json.loads(wire)
    write(job / "execution.json", {"exit_code": result.get("exit_code"), "log": str(result.get("log", ""))[-6000:]})
    if request.get("operation") == "invoke_program":
        if len(wire) > request["max_output_bytes"] + 8192:
            raise ValueError("program output exceeded its byte limit")
        if result.get("exit_code") != 0:
            raise ValueError("program failed: " + str(result.get("error", "runtime error"))[:2000])
        if "value" not in result:
            raise ValueError("program returned no JSON value")
        write(job / "result.json", result)
        write(job / "state.json", {"status": "completed", "progress": 100})
        return
    if request.get("operation") == "invoke_tool":
        if result.get("exit_code") != 0:
            raise ValueError("isolated Tool failed: " + str(result.get("log", ""))[-2000:])
        if "tool_output" not in result or len(encoded(result["tool_output"])) > request["max_output_bytes"]:
            raise ValueError("isolated Tool returned no bounded JSON output")
        write(job / "result.json", {"tool_output": result["tool_output"],
              "runtime_image": request["runtime_image"],
              "package_name": request["package_name"], "package_digest": request["package_digest"]})
        write(job / "state.json", {"status": "completed", "message": "隔离工具执行完成", "progress": 100})
        return
    if request.get("workspace_id") and "workspace_files" in result:
        Workspaces(job.parent.parent).capture(request, job.name, result["workspace_files"])
    if result.get("exit_code") != 0:
        raise ValueError("script failed: " + str(result.get("log", ""))[-4000:])
    items = result.get("files", [])
    if not isinstance(items, list) or {v.get("path") for v in items} != set(request["outputs"]) or len(items) != len(request["outputs"]):
        raise ValueError("worker did not return the exact declared outputs")
    total, files = 0, []
    published = job / "published"
    published.mkdir(exist_ok=True)
    for item in items:
        data = base64.b64decode(item["data"], validate=True)
        total += len(data)
        sha = hashlib.sha256(data).hexdigest()
        if not 0 < len(data) <= MAX_FILE or total > MAX_OUTPUT or len(data) != item["size_bytes"] or sha != item["sha256"]:
            raise ValueError("worker artifact size or integrity mismatch")
        filename = Path(item["path"]).name
        target = published / filename
        # Only this trusted broker writes published files; no worker mount points here.
        target.write_bytes(data)
        artifact = {"filename": filename, "content_type": MIMES[Path(filename).suffix.lower()], "size_bytes": len(data), "sha256": sha}
        if request.get("operation") == "inspect_webpage":
            artifact["_meta"] = {"exoharness": {"artifact_visibility": "internal"}}
        if request.get("workspace_id"):
            artifact.setdefault("_meta", {}).setdefault("exoharness", {})["artifact_identity"] = {
                "key": hashlib.sha256(f"{request['workspace_id']}:{item['path']}".encode()).hexdigest(),
                "revision": request["workspace_revision"] + 1,
            }
        files.append(artifact)
    evidence = {"files": files, "runtime_image": os.environ["WORKSPACE_IMAGE_ID"],
          "package_name": request.get("package_name"), "package_digest": request.get("package_digest")}
    if request.get("operation") == "inspect_webpage":
        report = result.get("inspection")
        if not isinstance(report, dict) or len(encoded(report)) > 24000:
            raise ValueError("invalid bounded browser report")
        evidence.update(inspection=report, browser_image=request["browser_image"],
                        observer_revision=request["observer_revision"],
                        inspected_revision=request["workspace_revision"], source_path=request["entrypoint"])
    write(job / "result.json", evidence)
    write(job / "state.json", {"status": "completed", "message": "文件已生成并保存" if files else "代码执行完成", "progress": 100})


def run_job(job, packages, image, heartbeat):
    broker_image = image
    request_bytes = (job / "request.json").read_bytes()
    request = validate_request(json.loads(request_bytes))
    if request.get("operation") == "invoke_program":
        expected_job = "work_" + hashlib.sha256(request["invocation_id"].encode()).hexdigest()
        if job.name != expected_job or (job / "input" / "request.json").read_bytes() != request_bytes:
            raise ValueError("program job identity or input changed")
    if request.get("operation") == "invoke_tool":
        identity = f"{request['capability_id']}\0{request['capability_revision']}\0{request['tool_request']['request_id']}".encode()
        expected_job = "work_" + hashlib.sha256(identity).hexdigest()
        if job.name != expected_job or (job / "input" / "request.json").read_bytes() != request_bytes:
            raise ValueError("isolated Tool job identity or staged request changed")
    from .project_runtime import workspace_image
    image = workspace_image(job.parent.parent, request.get("workspace_id")) or image
    if request.get("runtime_image") and request["runtime_image"] != image:
        raise ValueError("job runtime image differs from its immutable workspace snapshot")
    browser_check = request.get("operation") == "inspect_webpage"
    if browser_check:
        if request.get("package_name") or request.get("package_digest"):
            raise ValueError("browser inspection cannot mount a Skill package")
    else:
        require_package_approval(job.parent.parent, request)
    execution_image = image
    if browser_check:
        execution_image = os.environ.get("WORKSPACE_BROWSER_IMAGE_ID", "")
        if request["observer_revision"] != browser_revision():
            raise ValueError("browser observer revision changed; inspect again against the new observer, sources were not modified")
        if not re.fullmatch(r"sha256:[a-f0-9]{64}", execution_image) or execution_image != request["browser_image"]:
            raise ValueError("browser image changed or is not approved; no browser execution was submitted")
    if request.get("operation") == "invoke_tool":
        tool_packages = packages.parent / "tools"
        package = tool_packages / request["package_name"] / request["package_digest"]
        if (tool_packages.is_symlink() or (tool_packages / request["package_name"]).is_symlink()
                or package.is_symlink() or not package.is_dir()
                or not package.resolve().is_relative_to(tool_packages.resolve())):
            raise ValueError("selected immutable Tool package is unavailable")
    else:
        package = package_path(packages, request["package_name"], request["package_digest"]) if request.get("package_name") else None
    name = "exoharness-" + job.name
    info = inspect(name)
    if info is None and read(job / "state.json").get("status") == "working":
        # A missing container after submission is an unknown execution outcome,
        # not permission to run the saved code again. Retain exact inputs and
        # require explicit recovery outside this broker.
        raise ValueError("submitted worker container is missing; execution is not automatically repeated")
    if info is None:
        if package and not browser_check:
            package = readable_package_snapshot(package, job / "skill")
        if request.get("operation") == "invoke_tool":
            entrypoint = package / request["entrypoint"]
            if entrypoint.is_symlink() or not entrypoint.is_file() or not entrypoint.resolve().is_relative_to(package.resolve()):
                raise ValueError("isolated Tool entrypoint is not in its immutable package")
        from .prepare_runtime import prepared_runtime
        program = request.get("operation") == "invoke_program"
        prepared = None if browser_check or program else prepared_runtime(job.parent.parent, request, image)
        runner = "program_runner.py" if program else "browser_runner.py" if browser_check else "sandbox.py"
        args = container_args(job, None if browser_check else package, execution_image, Path(__file__).with_name(runner).resolve(), prepared, browser=browser_check)
        docker(*args)
        info = inspect(name)
    if info["Config"]["Labels"].get("exoharness.skill-workspace") != job.name or info["Image"] != execution_image:
        raise ValueError("container identity does not match this job and configured image")
    state = read(job / "state.json")
    started = state.get("started_at", time.time())
    if (job / "cancel").exists():
        docker("rm", "-f", name, check=False)
        write(job / "state.json", {"status": "cancelled", "message": "已停止脚本执行", "progress": 0})
        return
    write(job / "state.json", {"status": "working", "message": "正在沙盒内执行代码", "started_at": started, "progress": 0})
    if info["State"]["Status"] == "created":
        docker("start", name)
    while True:
        write(heartbeat, {"at": time.time(), "image": broker_image})
        info = inspect(name)
        if info is None:
            raise ValueError("worker container disappeared; execution is not automatically repeated")
        runtime_limit = request["timeout_ms"] / 1000 + 30 if request.get("operation") == "invoke_program" else 90 if browser_check else 330
        if (job / "cancel").exists() or time.time() - started > runtime_limit:
            docker("rm", "-f", name, check=False)
            cancelled = (job / "cancel").exists()
            write(job / "state.json", {"status": "cancelled" if cancelled else "failed", "progress": 0,
                  "message": "已停止执行", "error": "cancelled" if cancelled else ("browser inspection time limit exceeded; source files are unchanged" if browser_check else "script runtime limit exceeded")})
            return
        if not info["State"]["Running"]:
            break
        time.sleep(1)
    if info["State"].get("OOMKilled"):
        raise ValueError("worker exceeded its isolated memory budget")
    # Bound untrusted bytes before parsing; no data from this pipe is sent to the LLM as base64.
    with tempfile.TemporaryFile() as output:
        process = subprocess.Popen(["docker", "logs", name], stdout=output, stderr=subprocess.DEVNULL)
        deadline = time.monotonic() + 30
        while process.poll() is None:
            if output.seek(0, 2) > MAX_WIRE or time.monotonic() > deadline:
                process.kill()
                process.wait()
                raise ValueError("worker transport exceeded output/time budget")
            time.sleep(0.05)
        if output.seek(0, 2) > MAX_WIRE or process.returncode != 0:
            raise ValueError("invalid worker output")
        output.seek(0)
        wire = output.read(MAX_WIRE + 1)
    finish(job, request, wire)
    docker("rm", name, check=False)


def process_sources(root, image, heartbeat):
    """Run fixed GitHub-only downloader code, never repository code."""
    if os.getuid() == 0:
        raise ValueError("GitHub source broker must run as a non-root user")
    sources = root / "sources"
    sources.mkdir(exist_ok=True)
    for job in sorted(sources.iterdir()):
        if job.is_symlink() or not job.is_dir() or not re.fullmatch(r"[0-9a-f]{64}", job.name):
            continue
        if not (job / "request.json").is_file() or not (job / "state.json").is_file():
            continue
        try:
            if read(job / "state.json").get("status") not in {"queued", "working"}:
                continue
            if (job / "cancel").exists():
                write(job / "state.json", {"status": "cancelled", "message": "已取消 GitHub 来源下载"})
                continue
            request = read(job / "request.json")
            if set(request) != {"owner", "repository", "ref"}:
                raise ValueError("invalid GitHub source request")
            if not all(isinstance(request[key], str) for key in ("owner", "repository")):
                raise ValueError("invalid GitHub coordinates")
            if request["ref"] is not None and not isinstance(request["ref"], str):
                raise ValueError("invalid GitHub ref")
            name = "exoharness-source-" + job.name
            write(job / "state.json", {"status": "working", "message": "正在隔离下载公开 GitHub 项目"})
            write(heartbeat, {"at": time.time(), "image": image})
            subprocess.run(["docker", "rm", "-f", name], capture_output=True, timeout=30)
            args = ["run", "--rm", "--name", name,
                    "--network", "bridge", "--read-only", "--user", f"{os.getuid()}:{os.getgid()}",
                    "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
                    "--cpus", "1", "--memory", "256m", "--memory-swap", "256m", "--pids-limit", "64",
                    "--tmpfs", "/tmp:rw,nosuid,noexec,size=64m,mode=1777",
                    "--mount", f"type=bind,src={job},dst=/source",
                    "--mount", f"type=bind,src={Path(__file__).with_name('github_fetch.py').resolve()},dst=/fetch.py,readonly",
                    image, "python", "-I", "/fetch.py"]
            process = subprocess.Popen(["docker", *args], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            started = time.monotonic()
            try:
                while True:
                    if (job / "cancel").exists():
                        docker("rm", "-f", name, check=False, timeout=30)
                        process.kill()
                        process.communicate()
                        write(job / "state.json", {"status": "cancelled", "message": "已取消 GitHub 来源下载"})
                        break
                    if time.monotonic() - started > 120:
                        raise ValueError("GitHub 来源下载超过 120 秒")
                    write(heartbeat, {"at": time.time(), "image": image})
                    try:
                        _, errors = process.communicate(timeout=0.25)
                        if process.returncode:
                            raise ValueError("GitHub 来源下载失败：" + errors.decode(errors="replace")[-1400:])
                        break
                    except subprocess.TimeoutExpired:
                        continue
                if read(job / "state.json").get("status") == "cancelled":
                    continue
            finally:
                if process.poll() is None:
                    docker("rm", "-f", name, check=False, timeout=30)
                    process.kill()
                    process.communicate()
            metadata = read(job / "result.json")
            if not re.fullmatch(r"[0-9a-f]{40}", metadata.get("commit", "")):
                raise ValueError("GitHub 固定提交验证失败")
            if not (job / "archive.zip").is_file() or (job / "archive.zip").stat().st_size > 32 * 1024 * 1024:
                raise ValueError("GitHub 压缩包不存在或超出限制")
            write(job / "state.json", {"status": "completed", "message": "已获取固定版本", "commit": metadata["commit"]})
        except Exception as error:
            write(job / "state.json", {"status": "failed", "message": str(error)[:2000]})


def main():
    root = Path(os.environ["WORKSPACE_DATA_DIR"]).resolve()
    packages = Path(os.environ["WORKSPACE_PACKAGES_DIR"]).resolve()
    image = os.environ["WORKSPACE_IMAGE_ID"]
    root.mkdir(parents=True, exist_ok=True)
    jobs = root / "jobs"
    jobs.mkdir(exist_ok=True)
    lock = (root / "broker.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    inventory = {"image": image, **runtime_inventory(image)}
    if (root / "engineering-runtime.json").is_file():
        from .project_runtime import approved_image
        try:
            engineering_image = approved_image(root)
            docker("run", "--rm", "--network", "none", "--read-only", "--user", "65532:65532", engineering_image, "git", "--version")
            inventory["engineering_image"] = engineering_image
        except (RuntimeError, ValueError, subprocess.TimeoutExpired):
            inventory["engineering_error"] = "Software-engineering image is unavailable; existing task runtime is unchanged"
    browser_image = os.environ.get("WORKSPACE_BROWSER_IMAGE_ID")
    if browser_image:
        try:
            if not re.fullmatch(r"sha256:[a-f0-9]{64}", browser_image):
                raise ValueError("invalid browser image policy")
            docker("run", "--rm", "--network", "none", "--read-only", "--user", "65532:65532", browser_image,
                   "python", "-c", "from playwright.sync_api import sync_playwright; import shutil; assert shutil.which('chromium')")
            inventory["browser_image"] = browser_image
        except (RuntimeError, ValueError, subprocess.TimeoutExpired):
            # An optional browser profile must not disable existing Python,
            # Node, TypeScript or package-bound execution.
            inventory["browser_error"] = "Configured isolated browser is unavailable; administrator preparation required"
    write(root / "inventory.json", inventory)
    from .mcp_socket import start as start_mcp_sessions
    mcp_sessions = start_mcp_sessions(root, packages, image)
    while True:
        write(root / "worker.json", {"at": time.time(), "image": image})
        from .prepare_runtime import process_pending
        process_pending(root, packages, image, root / "worker.json")
        process_sources(root, image, root / "worker.json")
        for path in sorted(jobs.iterdir()):
            if not path.is_dir() or not (path / "state.json").is_file():
                continue
            try:
                job = job_path(jobs, path.name)
                if read(job / "state.json")["status"] not in {"queued", "working"}:
                    continue
                run_job(job, packages, image, root / "worker.json")
            except Exception as error:
                write(path / "state.json", {"status": "failed", "progress": 0, "error": str(error)[:4000], "message": "脚本执行失败；查看错误后只修复失败部分"})
                if re.fullmatch(r"work_[0-9a-f]{64}", path.name):
                    docker("rm", "-f", "exoharness-" + path.name, check=False)
        time.sleep(1)


if __name__ == "__main__":
    main()
