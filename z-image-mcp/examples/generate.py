"""Real MCP smoke client: progress, detached recovery, PNG download and integrity verification."""

import argparse
import asyncio
import hashlib
import io
import json
import os
import time
from datetime import timedelta
from pathlib import Path

import httpx
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client
from PIL import Image


async def run(args):
    headers = {}
    if token := os.getenv("Z_IMAGE_API_TOKEN"):
        headers["Authorization"] = f"Bearer {token}"
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    begin = time.monotonic()
    events = []

    async def progress(value, total, message):
        event = {
            "progress": value,
            "total": total,
            "message": message,
            "elapsed_seconds": round(time.monotonic() - begin, 3),
        }
        events.append(event)
        print(json.dumps(event, ensure_ascii=False), flush=True)

    # Set BOTH HTTP read timeout and MCP response timeout. One does not replace the other.
    async with httpx.AsyncClient(headers=headers, timeout=httpx.Timeout(1800, connect=15)) as http:
        async with streamable_http_client(args.url, http_client=http) as streams:
            async with ClientSession(
                streams[0], streams[1], read_timeout_seconds=timedelta(seconds=1800)
            ) as client:
                await client.initialize()
                tools = [tool.name for tool in (await client.list_tools()).tools]
                print(json.dumps({"tools": tools}), flush=True)
                payload = {
                    "prompt": args.prompt,
                    "width": args.width,
                    "height": args.height,
                }
                if args.job_id:
                    job_id = args.job_id
                elif args.detached:
                    result = await client.call_tool("submit_image_generation", payload)
                    if result.isError:
                        raise RuntimeError(str(result.content))
                    job_id = result.structuredContent["id"]
                    print(json.dumps({"job_id": job_id}), flush=True)
                else:
                    job_id = None
                if job_id:
                    async with asyncio.timeout(1800):
                        while True:
                            status = await client.call_tool("get_image_job", {"job_id": job_id})
                            if status.isError:
                                raise RuntimeError(str(status.content))
                            data = status.structuredContent
                            await progress(
                                data["progress"], 100, f"job_id={job_id} {data['stage']}"
                            )
                            if data["status"] in {"succeeded", "failed", "canceled"}:
                                break
                            await asyncio.sleep(1)
                    result = await client.call_tool("get_image_result", {"job_id": job_id})
                else:
                    result = await client.call_tool(
                        "generate_image", payload, progress_callback=progress
                    )
                data = result.structuredContent
                if result.isError or not data or data["status"] != "succeeded":
                    raise RuntimeError(json.dumps(data or str(result.content), ensure_ascii=False))
        # Download URL contains its own capability; no MCP authorization header is needed.
        async with httpx.AsyncClient(timeout=60) as downloader:
            download = await downloader.get(data["result"]["download_url"])
            download.raise_for_status()
        digest = hashlib.sha256(download.content).hexdigest()
        assert digest == data["result"]["sha256"], "PNG integrity mismatch"
        with Image.open(io.BytesIO(download.content)) as image:
            image.load()
            assert image.format == "PNG"
            assert image.size == (data["request"]["width"], data["request"]["height"])
        output.write_bytes(download.content)
        evidence = {
            "job": data,
            "progress_events": events,
            "sha256_verified": True,
            "client_seconds": round(time.monotonic() - begin, 3),
        }
        output.with_suffix(".json").write_text(
            json.dumps(evidence, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(json.dumps(evidence, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8766/mcp")
    parser.add_argument(
        "--prompt", default="一只橘猫坐在窗边的木桌上，窗外是雨后的绿色花园，柔和的自然光，真实摄影"
    )
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--detached", action="store_true")
    parser.add_argument("--job-id", help="Resume polling an existing job without submitting again")
    parser.add_argument("--output", default="artifacts/example.png")
    asyncio.run(run(parser.parse_args()))
