#!/usr/bin/env bash
set -euo pipefail

project_dir=$(cd "$(dirname "$0")/.." && pwd)
cd "$project_dir"

if [[ ! -x .venv/bin/yue2-mcp ]]; then
  echo "Install the YuE2 runtime and MCP package into .venv first" >&2
  exit 1
fi

if [[ ! -f .env ]]; then
  .venv/bin/python - <<'PY'
import os
import secrets
from pathlib import Path

source = Path('.env.example').read_text(encoding='utf-8')
source = source.replace('YUE2_API_TOKEN=\n', f'YUE2_API_TOKEN={secrets.token_hex(32)}\n')
target = Path('.env')
target.write_text(source, encoding='utf-8')
os.chmod(target, 0o600)
PY
fi

.venv/bin/python - <<'PY'
import json
import os
from pathlib import Path
from yue2_mcp.config import Settings

settings = Settings()
if not settings.api_token.get_secret_value():
    raise SystemExit('Set YUE2_API_TOKEN in .env before installing the service')
artifacts = Path('artifacts')
artifacts.mkdir(mode=0o700, exist_ok=True)
config = {'mcpServers': {'yue2': {
    'type': 'streamableHttp',
    'url': f'{settings.public_url}/mcp',
    'headers': {'Authorization': f'Bearer {settings.api_token.get_secret_value()}'},
    'description': 'YuE2 音乐生成：歌词、歌曲、排队进度、取消、乐谱和 FLAC 下载',
}}}
path = artifacts / 'mcp-config.json'
fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
with os.fdopen(fd, 'w', encoding='utf-8') as stream:
    json.dump(config, stream, ensure_ascii=False, indent=2)
    stream.write('\n')
PY

mkdir -p "$HOME/.config/systemd/user"
install -m 644 deploy/yue2-mcp.service.example "$HOME/.config/systemd/user/yue2-mcp.service"
systemctl --user daemon-reload
systemctl --user enable --now yue2-mcp.service
systemctl --user is-active yue2-mcp.service
health_url=$(.venv/bin/python -c 'from yue2_mcp.config import Settings; print(Settings().public_url + "/health")')
for attempt in {1..30}; do
  if curl --fail --silent --output /dev/null "$health_url"; then
    echo "YuE2 MCP health check passed"
    exit 0
  fi
  sleep 1
done
echo "YuE2 MCP did not become healthy within 30 seconds" >&2
exit 1
