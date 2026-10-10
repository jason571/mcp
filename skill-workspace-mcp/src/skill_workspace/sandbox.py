"""Runs INSIDE a fresh network-disabled OCI container. Never a host entrypoint."""
import base64
import hashlib
import json
import os
import selectors
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path


def main():
    request = json.loads(Path("/input/request.json").read_bytes())
    work = Path("/workspace")
    manifest_path = Path("/input/files.json")
    for item in json.loads(manifest_path.read_bytes()) if manifest_path.is_file() else []:
        source = Path("/input") / item["file"]
        if source.stat().st_size != item["size_bytes"] or hashlib.sha256(source.read_bytes()).hexdigest() != item["sha256"]:
            raise ValueError("staged input integrity mismatch")
        path = work / item["path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, path)
    for item in request.get("files", []):
        path = work / item["path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(item["content"])
    tool_job = request.get("operation") == "invoke_tool"
    root = Path("/skill") if tool_job or request.get("entrypoint_root") == "skill" else work
    interpreters = {"python": ["python"], "node": ["node"], "shell": ["bash", "--noprofile", "--norc"]}
    command = None if request["runtime"] == "typescript" else [*interpreters[request["runtime"]], str(root / request["entrypoint"]), *request.get("arguments", [])]
    # Runtime defaults, not instructions injected into Harness or individual Skills.
    # The fixed image includes this CJK font. A caller can still choose its own style.
    plotting = Path("/tmp/matplotlib")
    plotting.mkdir(exist_ok=True)
    (plotting / "matplotlibrc").write_text("backend: Agg\nfont.family: sans-serif\nfont.sans-serif: Noto Sans CJK JP, DejaVu Sans\naxes.unicode_minus: True\n")
    os.environ["MPLCONFIGDIR"] = str(plotting)
    # A missing glyph is a broken deliverable, not a publishable success.
    os.environ["PYTHONWARNINGS"] = "error:Glyph:UserWarning"
    if Path("/runtime/verified.json").is_file():
        os.environ["PYTHONPATH"] = "/runtime/python" + (":" + os.environ["PYTHONPATH"] if os.environ.get("PYTHONPATH") else "")
        os.environ["NODE_PATH"] = "/runtime/node/node_modules" + (":" + os.environ["NODE_PATH"] if os.environ.get("NODE_PATH") else "")
        os.environ["PATH"] = "/runtime/python/bin:/runtime/node/node_modules/.bin:" + os.environ.get("PATH", "")
    output = {"exit_code": 1, "log": "", "files": []}
    try:
        if request["runtime"] == "typescript":
            checked = subprocess.run(["/usr/local/bin/node", "/runner/typescript.cjs", "plan", str(root), "/runtime", request["entrypoint"]],
                                     capture_output=True, text=True, timeout=120, stdin=subprocess.DEVNULL)
            if checked.returncode:
                raise ValueError("TypeScript check failed before execution: " + checked.stderr[-6000:])
            command = [*json.loads(checked.stdout), *request.get("arguments", [])]
        output.update(execute_tool(command, work, request) if tool_job else execute(command, work))
    except Exception as caught:
        output["log"] = str(caught)[:6000]
    if request.get("workspace_id"):
        output["workspace_files"] = snapshot(work)
    if output["exit_code"] != 0:
        return output
    if tool_job:
        return output
    try:
        total = 0
        for name in request["outputs"]:
            item = file_item(work, name)
            total += item["size_bytes"]
            if not item["size_bytes"] or total > 64 * 1024 * 1024:
                raise ValueError("declared files exceed output limits")
            output["files"].append(item)
    except Exception as caught:
        output.update(exit_code=1, log=output["log"] + "\n" + str(caught), files=[])
    return output


def execute(command, work):
    process = subprocess.Popen(command, cwd=work, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, start_new_session=True)
    selector = selectors.DefaultSelector()
    selector.register(process.stdout, selectors.EVENT_READ)
    log = bytearray()
    started = time.monotonic()
    try:
        while selector.get_map():
            if time.monotonic() - started > 300:
                raise TimeoutError("script exceeded 300 seconds")
            for key, _ in selector.select(0.2):
                block = os.read(key.fileobj.fileno(), 8192)
                if not block:
                    selector.unregister(key.fileobj)
                else:
                    log.extend(block)
                    if len(log) > 65536:
                        raise ValueError("script output exceeds 64 KiB; write deliverables to files")
        return {"exit_code": process.wait(timeout=2), "log": log.decode(errors="replace")[-6000:]}
    finally:
        # Stop descendants before reading files, including detached children.
        import signal
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=2)
        selector.close()


def execute_tool(command, work, request):
    """Deliver one typed request over stdin; stdout is data, not a chat log."""
    wire = json.dumps(request["tool_request"], ensure_ascii=False, separators=(",", ":")).encode() + b"\n"
    stdout, stderr = bytearray(), bytearray()
    with tempfile.TemporaryFile() as input_file:
        input_file.write(wire)
        input_file.seek(0)
        process = subprocess.Popen(command, cwd=work, stdin=input_file, stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE, start_new_session=True)
        selector = selectors.DefaultSelector()
        selector.register(process.stdout, selectors.EVENT_READ, stdout)
        selector.register(process.stderr, selectors.EVENT_READ, stderr)
        deadline = time.monotonic() + request["timeout_ms"] / 1000
        try:
            while selector.get_map():
                if time.monotonic() > deadline:
                    raise TimeoutError("isolated Tool execution exceeded its time limit")
                for key, _ in selector.select(0.05):
                    block = os.read(key.fileobj.fileno(), 8192)
                    if not block:
                        selector.unregister(key.fileobj)
                    else:
                        key.data.extend(block)
                        limit = request["max_output_bytes"] if key.data is stdout else 65536
                        if len(key.data) > limit:
                            raise ValueError("isolated Tool output exceeded its byte limit")
            code = process.wait(timeout=2)
            detail = stderr.decode(errors="replace")[-6000:]
            if code:
                return {"exit_code": code, "log": detail or "isolated Tool exited unsuccessfully"}
            try:
                value = json.loads(stdout)
            except (ValueError, UnicodeDecodeError):
                raise ValueError("isolated Tool stdout is not one valid JSON value") from None
            return {"exit_code": 0, "log": detail, "tool_output": value}
        finally:
            import signal
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait(timeout=2)
            selector.close()
            process.stdout.close()
            process.stderr.close()


def file_item(work, name):
    path = work / name
    if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(work):
        raise ValueError("output missing, symlinked, or outside workspace: " + name)
    if path.stat().st_size > 32 * 1024 * 1024:
        raise ValueError("workspace file exceeds 32 MiB")
    data = path.read_bytes()
    return {"path": name, "size_bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(), "data": base64.b64encode(data).decode()}


def snapshot(work):
    files, total = [], 0
    for root, directories, names in os.walk(work, followlinks=False):
        directories[:] = [name for name in directories if name not in {"__pycache__", ".cache"} and not (Path(root) / name).is_symlink()]
        for name in sorted(names):
            path = Path(root) / name
            if path.is_symlink():
                continue
            item = file_item(work, path.relative_to(work).as_posix())
            total += item["size_bytes"]
            if len(files) >= 128 or total > 64 * 1024 * 1024:
                raise ValueError("workspace snapshot exceeds 128 files or 64 MiB")
            files.append(item)
    return files


if __name__ == "__main__":
    try:
        result = main()
    except Exception as error:
        result = {"exit_code": 1, "log": str(error)[:6000], "files": []}
    sys.stdout.write(json.dumps(result, separators=(",", ":")))
    sys.stdout.flush()
    # Any child processes are terminated when PID 1 exits; none may survive a job.
