"""Check live health, MCP authentication, and tool discovery."""

import asyncio

import httpx
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

from yue2_mcp.config import Settings


async def main():
    settings = Settings()
    token = settings.api_token.get_secret_value()
    if not token:
        raise RuntimeError("YUE2_API_TOKEN is required for this service check")
    async with httpx.AsyncClient(timeout=30, trust_env=False) as http:
        health = await http.get(f"{settings.public_url}/health")
        health.raise_for_status()
        print(f"health={health.status_code} {health.json()}")
        anonymous = await http.get(f"{settings.public_url}/mcp")
        if anonymous.status_code != 401:
            raise RuntimeError(f"Expected anonymous HTTP 401, got {anonymous.status_code}")
        print("anonymous_mcp=401")
    headers = {"Authorization": f"Bearer {token}"}
    async with httpx.AsyncClient(headers=headers, timeout=30, trust_env=False) as http:
        endpoint = f"{settings.public_url}/mcp"
        async with streamable_http_client(endpoint, http_client=http) as streams:
            async with ClientSession(streams[0], streams[1]) as session:
                await session.initialize()
                tools = await session.list_tools()
                print("tools=" + ",".join(sorted(tool.name for tool in tools.tools)))


if __name__ == "__main__":
    asyncio.run(main())
