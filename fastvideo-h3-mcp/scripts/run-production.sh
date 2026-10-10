#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
config_file="$project_dir/deploy/production.env"
secret_dir="$project_dir/var/secrets"
api_token_file="$secret_dir/api-token"
signing_key_file="$secret_dir/artifact-signing-key"

for required_file in "$config_file" "$api_token_file" "$signing_key_file"; do
  if [[ ! -s "$required_file" ]]; then
    echo "Required production file is missing or empty: $required_file" >&2
    exit 1
  fi
done

set -a
# shellcheck disable=SC1090
source "$config_file"
set +a
export FASTVIDEO_MCP_API_TOKEN="$(<"$api_token_file")"
export FASTVIDEO_MCP_ARTIFACT_SIGNING_KEY="$(<"$signing_key_file")"

exec "$project_dir/.venv/bin/fastvideo-h3-mcp"
