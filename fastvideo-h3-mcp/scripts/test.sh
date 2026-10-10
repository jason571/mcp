#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_dir"
python_binary="$project_dir/.venv/bin/python"
if [[ ! -x "$python_binary" ]]; then
  python_binary="python"
fi

PYTHONPATH="$project_dir/src${PYTHONPATH:+:$PYTHONPATH}" \
  "$python_binary" -m compileall -q src examples tests
PYTHONPATH="$project_dir/src${PYTHONPATH:+:$PYTHONPATH}" \
  "$python_binary" -m unittest discover -s tests -v
