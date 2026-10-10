#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_dir"
python_binary="$project_dir/.venv/bin/python"
if [[ ! -x "$python_binary" ]]; then
  shared_python="$project_dir/../fastvideo-h3-mcp/.venv/bin/python"
  if [[ -x "$shared_python" ]]; then
    python_binary="$shared_python"
  else
    python_binary="python3"
  fi
fi

if ! "$python_binary" -c 'import aiohttp, mcp, PIL, pydantic' 2>/dev/null; then
  echo "test runtime is missing project dependencies: $python_binary" >&2
  echo "create .venv or install dependencies before running tests" >&2
  exit 2
fi

PYTHONPATH="$project_dir/src${PYTHONPATH:+:$PYTHONPATH}" \
  "$python_binary" -m compileall -q src tests
PYTHONPATH="$project_dir/src${PYTHONPATH:+:$PYTHONPATH}" \
  "$python_binary" -m unittest discover -s tests -v
