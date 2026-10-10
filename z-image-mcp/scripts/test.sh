#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
exec "${Z_IMAGE_PYTHON:-.venv/bin/python}" -m pytest "$@"
