from __future__ import annotations

import asyncio
import json
import os

import httpx2
from mcp import Client
from mcp.client.streamable_http import streamable_http_client


async def main() -> None:
    url = os.getenv("FASTVIDEO_MCP_URL", "http://192.168.108.50:8765/mcp")
    token = os.environ["FASTVIDEO_MCP_API_TOKEN"]
    async with httpx2.AsyncClient(
        headers={"Authorization": f"Bearer {token}"},
        timeout=httpx2.Timeout(30.0, read=120.0),
        follow_redirects=True,
    ) as http_client:
        transport = streamable_http_client(url, http_client=http_client)
        async with Client(transport) as client:
            result = await client.call_tool("get_video_service_status", {})
            payload = result.structured_content
            if payload is None:
                payload = [
                    block.model_dump(mode="json")
                    if hasattr(block, "model_dump")
                    else str(block)
                    for block in result.content
                ]
            print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
