"""Submit a YuE2 song through MCP, poll, download FLAC, and verify SHA-256."""

import argparse
import asyncio
import hashlib
import os
from pathlib import Path

import httpx
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client


async def run(args):
    headers = {"Authorization": f"Bearer {args.token}"} if args.token else {}
    async with httpx.AsyncClient(headers=headers, timeout=1800, trust_env=False) as http:
        async with streamable_http_client(args.url, http_client=http) as streams:
            async with ClientSession(streams[0], streams[1]) as client:
                await client.initialize()
                parameters = {"style": args.style}
                if args.lyrics_file:
                    parameters["lyrics"] = args.lyrics_file.read_text(encoding="utf-8")
                if args.lyrics_prompt:
                    parameters["lyrics_prompt"] = args.lyrics_prompt
                response = await client.call_tool("submit_music_generation", parameters)
                if response.isError:
                    raise RuntimeError(response.content)
                job = response.structuredContent
                print(f"job_id={job['id']}", flush=True)
                while job["status"] not in {"succeeded", "failed", "canceled"}:
                    await asyncio.sleep(3)
                    response = await client.call_tool("get_music_job", {"job_id": job["id"]})
                    job = response.structuredContent
                    print(f"{job['progress']}% {job['stage']} {job['message']}", flush=True)
                if job["status"] != "succeeded":
                    raise RuntimeError(job.get("error") or job["message"])
                result = job["result"]
                download = await http.get(result["audio_url"])
                download.raise_for_status()
                digest = hashlib.sha256(download.content).hexdigest()
                if digest != result["sha256"]:
                    raise RuntimeError("FLAC SHA-256 mismatch")
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_bytes(download.content)
                print(f"saved={args.output} duration_seconds={result['duration_seconds']}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8767/mcp")
    parser.add_argument("--style", required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--lyrics-file", type=Path)
    group.add_argument("--lyrics-prompt")
    parser.add_argument("--output", type=Path, default=Path("artifacts/song.flac"))
    parser.add_argument("--token", default=os.getenv("YUE2_API_TOKEN", ""))
    asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    main()
