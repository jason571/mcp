"""Runs only in the existing fresh OCI sandbox. RPC is a bounded job-local
mailbox, not a socket, Shell, URL fetcher, package store or secret store.
"""
import asyncio
import contextlib
import io
import json
import os
from pathlib import Path
import tempfile
import time


class Bindings:
    def __init__(self, request):
        self.names = frozenset(request["bindings"])
        self.limit = request["max_calls"]
        self.sequence = 0
        self.deadline = time.monotonic() + request["timeout_ms"] / 1000
        self.lock = asyncio.Lock()

    async def call(self, name, arguments):
        # Serial calls deliberately avoid parallel external side effects and
        # make the persisted order unambiguous, even with asyncio.gather.
        async with self.lock:
            if name not in self.names or not isinstance(arguments, dict):
                raise ValueError("use an explicit binding and native object arguments")
            if self.sequence >= self.limit:
                raise ValueError("program tool-call limit reached")
            wire = json.dumps({"binding": name, "arguments": arguments}, allow_nan=False).encode()
            if len(wire) > 65536:
                raise ValueError("program tool arguments exceed 64 KiB")
            sequence = self.sequence
            self.sequence += 1
            base = Path("/rpc") / str(sequence)
            fd, staged = tempfile.mkstemp(prefix=".call-", dir="/rpc")
            with os.fdopen(fd, "wb") as output:
                os.fchmod(output.fileno(), 0o644)
                output.write(wire)
            os.replace(staged, base.with_suffix(".request.json"))
            response = base.with_suffix(".response.json")
            while not response.exists():
                if time.monotonic() >= self.deadline:
                    raise TimeoutError("program binding timed out")
                await asyncio.sleep(.02)
            if response.is_symlink() or response.stat().st_size > 131072:
                raise ValueError("invalid program binding response")
            value = json.loads(response.read_bytes())
            if value.get("is_error"):
                raise RuntimeError(str(value.get("result", "tool failed"))[:2000])
            return value["result"]


class BoundedLog(io.StringIO):
    def write(self, text):
        if self.tell() + len(text) > 6000:
            raise ValueError("program logs exceed 6000 characters")
        return super().write(text)


async def main():
    request = json.loads(Path("/input/request.json").read_bytes())
    log = BoundedLog()
    try:
        namespace = {"tools": Bindings(request), "input": request["input"]}
        source = "async def __program__():\n" + "\n".join("    " + line for line in request["code"].splitlines())
        with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            exec(compile(source, "<isolated-program>", "exec"), namespace)
            value = await asyncio.wait_for(namespace["__program__"](), request["timeout_ms"] / 1000)
        output = {"exit_code": 0, "value": value, "log": log.getvalue()}
        wire = json.dumps(output, ensure_ascii=False, allow_nan=False).encode()
        if len(wire) > request["max_output_bytes"]:
            raise ValueError("program output exceeds its byte limit")
        return wire
    except Exception as error:
        return json.dumps({"exit_code": 1, "error": str(error)[:2000], "log": log.getvalue()}, ensure_ascii=False).encode()


if __name__ == "__main__":
    os.write(1, asyncio.run(main()))
