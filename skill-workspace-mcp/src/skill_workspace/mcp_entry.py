"""Trusted OCI entrypoint for a user MCP that requires private environment values.

The broker sends one bounded JSON line over stdin before any MCP JSON-RPC bytes.
Reading one byte at a time avoids buffering away the first protocol message
before exec replaces this process with the package's declared entrypoint.
"""
from __future__ import annotations

import json
import os
import sys


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("MCP entrypoint is missing")
    wire = bytearray()
    while len(wire) <= 256 * 1024:
        byte = os.read(0, 1)
        if not byte:
            raise SystemExit("MCP secret preamble is incomplete")
        if byte == b"\n":
            break
        wire.extend(byte)
    else:
        raise SystemExit("MCP secret preamble exceeded limit")
    values = json.loads(wire)
    if not isinstance(values, dict) or not values:
        raise SystemExit("MCP secret preamble is invalid")
    for name, value in values.items():
        if (not isinstance(name, str) or not name.isascii() or not name.replace("_", "").isalnum()
                or name != name.upper() or name.startswith("EXOHARNESS_")
                or not isinstance(value, str) or not value or "\0" in value):
            raise SystemExit("MCP secret preamble is invalid")
        os.environ[name] = value
    os.execvp(sys.argv[1], sys.argv[1:])


if __name__ == "__main__":
    main()
