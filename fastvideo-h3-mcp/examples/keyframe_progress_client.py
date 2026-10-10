from __future__ import annotations

import asyncio
import copy
import json
import os

import httpx2
from mcp import Client
from mcp.client.streamable_http import streamable_http_client


async def on_progress(progress: float, total: float | None, message: str | None) -> None:
    total_text = "?" if total is None else f"{total:.0f}"
    print(f"进度 {progress:.1f}/{total_text}: {message or ''}", flush=True)


async def main() -> None:
    url = os.getenv("FASTVIDEO_MCP_URL", "http://192.168.108.50:8765/mcp")
    token = os.environ["FASTVIDEO_MCP_API_TOKEN"]
    first_frame_url = os.getenv("FASTVIDEO_FIRST_FRAME_URL")
    last_frame_url = os.getenv("FASTVIDEO_LAST_FRAME_URL")
    first_frame_asset_id = os.getenv("FASTVIDEO_FIRST_FRAME_ASSET_ID")
    last_frame_asset_id = os.getenv("FASTVIDEO_LAST_FRAME_ASSET_ID")
    if not first_frame_url and not first_frame_asset_id:
        raise RuntimeError(
            "set FASTVIDEO_FIRST_FRAME_URL or FASTVIDEO_FIRST_FRAME_ASSET_ID"
        )

    async with httpx2.AsyncClient(
        headers={"Authorization": f"Bearer {token}"},
        timeout=httpx2.Timeout(30.0, read=7200.0),
        follow_redirects=True,
    ) as http_client:
        transport = streamable_http_client(url, http_client=http_client)
        async with Client(transport) as client:
            arguments = {
                "prompt": os.getenv(
                    "FASTVIDEO_TEST_PROMPT",
                    "A cinematic transition from the first frame to the last frame, "
                    "with smooth natural motion and consistent subjects.",
                ),
                "soundscape": os.getenv(
                    "FASTVIDEO_TEST_SOUNDSCAPE", "Natural synchronized ambience."
                ),
                "negative_prompt": os.getenv(
                    "FASTVIDEO_TEST_NEGATIVE_PROMPT",
                    "No text, no watermark, no flicker, no deformed subjects.",
                ),
                "duration_seconds": float(
                    os.getenv("FASTVIDEO_TEST_DURATION_SECONDS", "5")
                ),
                "audio_enabled": os.getenv(
                    "FASTVIDEO_TEST_AUDIO_ENABLED", "true"
                ).lower()
                in {"1", "true", "yes", "on"},
                "aspect_ratio": os.getenv(
                    "FASTVIDEO_TEST_ASPECT_RATIO", "9:16"
                ),
                "resolution": os.getenv(
                    "FASTVIDEO_TEST_RESOLUTION", "512p"
                ),
                "model_id": "base",
            }
            test_seed = os.getenv("FASTVIDEO_TEST_SEED")
            if test_seed:
                arguments["seed"] = int(test_seed)
            if first_frame_url:
                arguments["first_frame_url"] = first_frame_url
            if last_frame_url:
                arguments["last_frame_url"] = last_frame_url
            if first_frame_asset_id:
                arguments["first_frame_asset_id"] = first_frame_asset_id
            if last_frame_asset_id:
                arguments["last_frame_asset_id"] = last_frame_asset_id
            result = await client.call_tool(
                "generate_video",
                arguments,
                read_timeout_seconds=7200,
                progress_callback=on_progress,
            )
            printable_content = copy.deepcopy(result.structured_content)
            if isinstance(printable_content, dict):
                metadata = printable_content.get("result")
                if isinstance(metadata, dict) and "download_url" in metadata:
                    metadata["download_url"] = "<redacted>"
            print(json.dumps(printable_content, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
