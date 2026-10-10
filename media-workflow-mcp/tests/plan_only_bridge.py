"""Opt-in real-LLM bridge. Run with Docker --network none, without GPU access.

Production schemas, SDK validation and stores; recorded audio metadata fixture.
Only authoring tools can dispatch, and no video upstream is configured.
"""
import asyncio
import json
import sys
import shutil
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from media_workflow_mcp import server
from media_workflow_mcp.plan_store import MusicVideoPlanStore

ALLOWED = {"create_music_video_plan", "write_music_video_plan_overview", "write_music_video_plan_segment", "write_music_video_plan_sections", "patch_music_video_plan_overview", "patch_music_video_plan_segment"}


async def main():
    for line in sys.stdin:
        try:
            request = json.loads(line)
            if request["op"] == "init":
                if request.get("resume"):
                    shutil.copytree(Path("/seed") / request["round"], Path("/evidence") / request["round"])
                settings = SimpleNamespace(api_token=None,
                    video_generator_min_duration_seconds=5, video_generator_max_duration_seconds=60,
                    video_generator_target_duration_seconds=30, video_generator_duration_step_seconds=1,
                    video_generator_tail_merge_max_duration_seconds=40)
                server._runtime = SimpleNamespace(settings=settings, plans=MusicVideoPlanStore(Path("/evidence") / request["round"]))
                server.probe_media = AsyncMock(return_value={"duration_seconds":request["duration_ms"]/1000,
                    "sha256":request["audio_sha256"], "streams":[{"codec_type":"audio"}]})
                response = {"tools":[tool.model_dump(mode="json", by_alias=True) for tool in await server.mcp.list_tools() if tool.name in ALLOWED]}
            elif request["op"] == "call":
                if request["name"] not in ALLOWED:
                    raise ValueError("PLAN-ONLY GATE: tool cannot execute in this isolated test")
                result = await server.mcp.call_tool(request["name"], request["arguments"], context=SimpleNamespace(headers={}))
                response = result.model_dump(mode="json", by_alias=True)
                if result.structured_content and result.structured_content.get("valid"):
                    response["validated_plan"] = server._runtime.plans.get(result.structured_content["plan_digest"])
            else:
                raise ValueError("Unknown test operation")
        except Exception as error:
            response = {"isError":True, "content":[{"type":"text", "text":str(error)}]}
        print(json.dumps(response, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    asyncio.run(main())
