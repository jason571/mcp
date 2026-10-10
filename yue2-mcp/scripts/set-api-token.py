"""Replace the service and private import token with one read from stdin."""

import json
import os
import re
import sys
from pathlib import Path

from yue2_mcp.config import Settings


def private_write(path: Path, content: str):
    staging = path.with_name(path.name + ".new")
    fd = os.open(staging, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as stream:
        stream.write(content)
    os.replace(staging, path)


def main():
    token = sys.stdin.readline().strip()
    if not re.fullmatch(r"[0-9a-f]{64}", token):
        raise SystemExit("Expected a 64-character lowercase hexadecimal token on stdin")
    root = Path(__file__).resolve().parent.parent
    environment = root / ".env"
    lines = environment.read_text(encoding="utf-8").splitlines()
    found = False
    for index, line in enumerate(lines):
        if line.startswith("YUE2_API_TOKEN="):
            lines[index] = f"YUE2_API_TOKEN={token}"
            found = True
    if not found:
        lines.append(f"YUE2_API_TOKEN={token}")
    private_write(environment, "\n".join(lines) + "\n")

    settings = Settings(_env_file=environment)
    config_path = root / "artifacts/mcp-config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    entry = config["mcpServers"]["yue2"]
    entry["headers"]["Authorization"] = f"Bearer {settings.api_token.get_secret_value()}"
    private_write(config_path, json.dumps(config, ensure_ascii=False, indent=2) + "\n")
    print("Service and private import token updated")


if __name__ == "__main__":
    main()
