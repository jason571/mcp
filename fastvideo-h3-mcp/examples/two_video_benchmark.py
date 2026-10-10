from __future__ import annotations

import asyncio
import json
import os
import time
from datetime import datetime
from typing import Any

import httpx2
from mcp import Client
from mcp.client.streamable_http import streamable_http_client


TERMINAL_STATUSES = {"completed", "failed", "canceled"}


def payload_of(result: object) -> dict[str, Any]:
    structured = getattr(result, "structured_content", None)
    if isinstance(structured, dict):
        return structured
    for block in getattr(result, "content", []):
        text = getattr(block, "text", None)
        if isinstance(text, str):
            value = json.loads(text)
            if isinstance(value, dict):
                return value
    raise RuntimeError("MCP tool did not return a JSON object")


def seconds_between(start: str | None, end: str | None) -> float | None:
    if not start or not end:
        return None
    return round((datetime.fromisoformat(end) - datetime.fromisoformat(start)).total_seconds(), 3)


async def main() -> None:
    url = os.getenv("FASTVIDEO_MCP_URL", "http://192.168.108.50:8765/mcp")
    token = os.environ["FASTVIDEO_MCP_API_TOKEN"]
    requests = [
        {
            "prompt": (
                "A hummingbird hovers beside bright red flowers in a misty forest, "
                "slow cinematic camera movement, realistic feathers and natural light."
            ),
            "negative_prompt": "watermark, subtitles, deformed anatomy, flicker",
            "soundscape": "Rapid wing beats, soft forest birds and a distant stream.",
            "duration_seconds": 5,
            "seed": 90101,
            "audio_enabled": True,
            "aspect_ratio": "16:9",
            "resolution": "768p",
            "model_id": "fast",
        },
        {
            "prompt": (
                "A red sports car drives along a winding ocean cliff road at sunset, "
                "smooth tracking shot, realistic reflections and cinematic motion."
            ),
            "negative_prompt": "watermark, subtitles, warped wheels, flicker",
            "soundscape": "A refined engine note, tire noise and ocean waves below.",
            "duration_seconds": 5,
            "seed": 90102,
            "audio_enabled": True,
            "aspect_ratio": "16:9",
            "resolution": "768p",
            "model_id": "fast",
        },
    ]
    benchmark_started = time.monotonic()
    async with httpx2.AsyncClient(
        headers={"Authorization": f"Bearer {token}"},
        timeout=httpx2.Timeout(30.0, read=120.0),
        follow_redirects=True,
    ) as http_client:
        transport = streamable_http_client(url, http_client=http_client)
        async with Client(transport) as client:
            jobs: list[dict[str, Any]] = []
            for number, request in enumerate(requests, start=1):
                result = await client.call_tool("submit_video_generation", request)
                job = payload_of(result)
                jobs.append(job)
                print(
                    f"submitted #{number}: {job['id']} "
                    f"queue_position={job.get('queue_position')} "
                    f"elapsed={time.monotonic() - benchmark_started:.2f}s",
                    flush=True,
                )

            latest = {str(job["id"]): job for job in jobs}
            last_state: dict[str, tuple[object, ...]] = {}
            while True:
                all_terminal = True
                for number, job in enumerate(jobs, start=1):
                    job_id = str(job["id"])
                    result = await client.call_tool("get_video_job", {"job_id": job_id})
                    current = payload_of(result)
                    latest[job_id] = current
                    state = (
                        current.get("status"),
                        current.get("stage"),
                        current.get("progress"),
                        current.get("message"),
                    )
                    if last_state.get(job_id) != state:
                        print(
                            f"#{number} {job_id}: status={state[0]} stage={state[1]} "
                            f"progress={state[2]} elapsed={time.monotonic() - benchmark_started:.1f}s "
                            f"message={state[3]}",
                            flush=True,
                        )
                        last_state[job_id] = state
                    if current.get("status") not in TERMINAL_STATUSES:
                        all_terminal = False
                if all_terminal:
                    break
                await asyncio.sleep(2)

    summaries = []
    for number, job in enumerate(jobs, start=1):
        current = latest[str(job["id"])]
        summaries.append(
            {
                "number": number,
                "job_id": current["id"],
                "status": current["status"],
                "queue_wait_seconds": seconds_between(
                    current.get("created_at"), current.get("started_at")
                ),
                "run_seconds": seconds_between(
                    current.get("started_at"), current.get("finished_at")
                ),
                "created_to_finished_seconds": seconds_between(
                    current.get("created_at"), current.get("finished_at")
                ),
                "result": current.get("result"),
                "error": current.get("error"),
            }
        )
    print(
        json.dumps(
            {
                "total_wall_seconds": round(time.monotonic() - benchmark_started, 3),
                "jobs": summaries,
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    if any(item["status"] != "completed" for item in summaries):
        raise SystemExit(1)


if __name__ == "__main__":
    asyncio.run(main())
