#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "$project_dir/deploy/production.env"

mcp_port="${H3_MCP_PORT:-8766}"
test -x /home/clhh/minimax-h3-mcp/.venv/bin/python
echo "mcp_python=ok"
curl -fsS --connect-timeout 3 --max-time 10 "http://127.0.0.1:${mcp_port}/health" >/dev/null
echo "mcp_health=ok url=http://127.0.0.1:${mcp_port}/health"
