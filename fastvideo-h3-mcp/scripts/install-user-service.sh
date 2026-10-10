#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
secret_dir="$project_dir/var/secrets"
user_unit_dir="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user"
unit_name="fastvideo-h3-mcp.service"

install -d -m 700 "$secret_dir"
if [[ ! -s "$secret_dir/api-token" ]]; then
  openssl rand -hex -out "$secret_dir/api-token" 32
fi
if [[ ! -s "$secret_dir/artifact-signing-key" ]]; then
  openssl rand -hex -out "$secret_dir/artifact-signing-key" 32
fi
chmod 600 "$secret_dir/api-token" "$secret_dir/artifact-signing-key"

install -d -m 755 "$user_unit_dir"
install -m 644 "$project_dir/deploy/fastvideo-h3-mcp.user.service" "$user_unit_dir/$unit_name"
systemctl --user daemon-reload
systemctl --user enable --now "$unit_name"

for attempt in {1..20}; do
  if curl --fail --silent --show-error --max-time 2 \
    http://127.0.0.1:8765/health >/dev/null
  then
    break
  fi
  if [[ "$attempt" -eq 20 ]]; then
    echo "$unit_name did not become healthy within 20 seconds." >&2
    exit 1
  fi
  sleep 1
done

echo "Installed and started $unit_name. API token remains in $secret_dir/api-token."
