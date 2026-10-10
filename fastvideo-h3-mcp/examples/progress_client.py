from __future__ import annotations

import asyncio
import copy
import json
import os
from datetime import UTC, datetime

import httpx2
from mcp import Client
from mcp.client.streamable_http import streamable_http_client


progress_events: list[dict[str, object]] = []


async def on_progress(progress: float, total: float | None, message: str | None) -> None:
    timestamp = datetime.now(UTC).isoformat(timespec="seconds")
    progress_events.append(
        {
            "timestamp": timestamp,
            "progress": progress,
            "total": total,
            "message": message or "",
        }
    )
    total_text = "?" if total is None else f"{total:.0f}"
    print(
        f"{timestamp} 进度 {progress:.1f}/{total_text}: {message or ''}",
        flush=True,
    )


def validate_progress() -> dict[str, object]:
    values = [float(event["progress"]) for event in progress_events]
    if not values:
        raise RuntimeError("MCP client did not receive any progress notifications")
    if any(current < previous for previous, current in zip(values, values[1:])):
        raise RuntimeError("MCP progress moved backwards")
    messages = "\n".join(str(event["message"]) for event in progress_events)
    if "1/20" in messages:
        raise RuntimeError("received invalid 1/20 sampling progress")
    if "Transformer " not in messages:
        raise RuntimeError("missing real transformer block progress")
    expected_stages = (
        "joint_av_denoising",
        "video_decoding",
        "audio_decoding",
        "completed",
    )
    missing = [stage for stage in expected_stages if f"[{stage}]" not in messages]
    if missing:
        raise RuntimeError(f"missing progress stages: {', '.join(missing)}")
    return {
        "event_count": len(progress_events),
        "monotonic": True,
        "first_progress": values[0],
        "last_progress": values[-1],
        "required_stages": list(expected_stages),
    }


async def main() -> None:
    url = os.getenv("FASTVIDEO_MCP_URL", "http://192.168.108.50:8765/mcp")
    token = os.environ["FASTVIDEO_MCP_API_TOKEN"]
    async with httpx2.AsyncClient(
        headers={"Authorization": f"Bearer {token}"},
        timeout=httpx2.Timeout(30.0, read=7200.0),
        follow_redirects=True,
    ) as http_client:
        transport = streamable_http_client(url, http_client=http_client)
        async with Client(transport) as client:
            result = await client.call_tool(
                "generate_video",
                {
                    "prompt": "A cinematic tracking shot of a red fox running through snow.",
                    "soundscape": "Wind, soft footsteps in snow, no music.",
                    "negative_prompt": "No text, no watermark, no flicker.",
                    "duration_seconds": float(
                        os.getenv("FASTVIDEO_TEST_DURATION_SECONDS", "5")
                    ),
                    "audio_enabled": os.getenv(
                        "FASTVIDEO_TEST_AUDIO_ENABLED", "true"
                    ).lower()
                    in {"1", "true", "yes", "on"},
                    "aspect_ratio": os.getenv(
                        "FASTVIDEO_TEST_ASPECT_RATIO", "16:9"
                    ),
                    "resolution": os.getenv(
                        "FASTVIDEO_TEST_RESOLUTION", "768p"
                    ),
                    "model_id": os.getenv("FASTVIDEO_TEST_MODEL_ID", "auto"),
                },
                read_timeout_seconds=7200,
                progress_callback=on_progress,
            )
            content = result.structured_content
            if not isinstance(content, dict) or content.get("status") != "completed":
                raise RuntimeError(f"generation did not complete: {content}")
            printable_content = copy.deepcopy(content)
            result_metadata = printable_content.get("result")
            if isinstance(result_metadata, dict) and "download_url" in result_metadata:
                result_metadata["download_url"] = "<redacted>"
            print(json.dumps(printable_content, ensure_ascii=False, indent=2))
            print(
                json.dumps(
                    {"progress_validation": validate_progress()},
                    ensure_ascii=False,
                    indent=2,
                )
            )


if __name__ == "__main__":
    asyncio.run(main())
