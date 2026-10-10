#!/usr/bin/env bash
set -euo pipefail
workspace_image_id="$(docker image inspect skill-workspace:0.1.0 --format '{{.Id}}')"
export WORKSPACE_IMAGE_ID="$workspace_image_id"
export SKILL_WORKSPACE_INTEGRATION=1
export PYTHONPATH=examples/skill-workspace-mcp/src
export PYTHONDONTWRITEBYTECODE=1
python3 -m unittest discover -s examples/skill-workspace-mcp/tests -p test_broker.py -v
python3 -m unittest discover -s examples/skill-workspace-mcp/tests -p test_tool_jobs.py -v
python3 -m unittest discover -s examples/skill-workspace-mcp/tests -p test_local_mcp.py -v
python3 -m unittest discover -s examples/skill-workspace-mcp/tests -p test_github_fetch.py -v
python3 -m unittest discover -s examples/skill-workspace-mcp/tests -p test_typescript.py -v
docker run --rm --network none --volume "$PWD:/src:ro" --workdir /src \
  --env PYTHONPATH=/src/examples/skill-workspace-mcp/src "$workspace_image_id" \
  python -m unittest discover -s examples/skill-workspace-mcp/tests -p test_inputs.py -v
docker run --rm --network none --volume "$PWD:/src:ro" --workdir /src \
  --env PYTHONPATH=/src/examples/skill-workspace-mcp/src "$workspace_image_id" \
  python -m unittest discover -s examples/skill-workspace-mcp/tests -p test_server.py -v
docker run --rm --network none --volume "$PWD:/src:ro" --workdir /src \
  --env PYTHONDONTWRITEBYTECODE=1 "$workspace_image_id" \
  python -m unittest discover -s examples/skill-workspace-mcp/tests -p test_pptx_layout.py -v
