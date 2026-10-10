#!/usr/bin/env bash
# Installs only this service in the current user's systemd manager.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
project_dir="$PWD"
if [[ ! -x .venv/bin/z-image-mcp || ! -f .env ]]; then
  echo 'Install the package in .venv and create .env first.' >&2
  exit 1
fi
if [[ "$project_dir" == *'%'* || "$project_dir" == *'"'* || "$project_dir" == *$'\n'* ]]; then
  echo 'Unsupported systemd path characters.' >&2
  exit 1
fi
unit_dir="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user"
mkdir -p "$unit_dir"
cat > "$unit_dir/z-image-mcp.service" <<EOF
[Unit]
Description=Z-Image-Turbo MCP image generation
After=network.target

[Service]
Type=simple
WorkingDirectory=$project_dir
EnvironmentFile=$project_dir/.env
Environment=PYTHONUNBUFFERED=1
ExecStart="$project_dir/.venv/bin/z-image-mcp"
Restart=on-failure
RestartSec=5
TimeoutStopSec=1800
KillMode=control-group
UMask=0077

[Install]
WantedBy=default.target
EOF
systemctl --user daemon-reload
systemctl --user enable --now z-image-mcp.service
systemctl --user --no-pager status z-image-mcp.service
