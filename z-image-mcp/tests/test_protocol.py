import asyncio
import base64
import hashlib
import io
from contextlib import asynccontextmanager
from datetime import timedelta

import httpx
from conftest import FakeBackend
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client
from PIL import Image

from z_image_mcp.server import create_app


@asynccontextmanager
async def transport(url):
    async with httpx.AsyncClient(timeout=30) as http:
        async with streamable_http_client(url, http_client=http) as streams:
            yield streams


async def test_real_mcp_progress_resource_link_and_inline_image(live_server):
    url, app, backend = live_server
    notifications = []

    async def progress(value, total, message):
        notifications.append((value, total, message))

    async with transport(f"{url}/mcp") as streams:
        async with ClientSession(
            streams[0], streams[1], read_timeout_seconds=timedelta(seconds=30)
        ) as client:
            await client.initialize()
            tools = (await client.list_tools()).tools
            names = {tool.name for tool in tools}
            assert names == {
                "generate_image",
                "submit_image_generation",
                "get_image_job",
                "get_image_result",
                "cancel_image_job",
                "get_image_service_info",
            }
            for tool in tools:
                if tool.name in {"generate_image", "submit_image_generation"}:
                    assert set(tool.inputSchema["properties"]) == {"prompt", "width", "height"}
                    assert tool.inputSchema["required"] == ["prompt"]
            invalid = await client.call_tool(
                "submit_image_generation", {"prompt": "x", "width": 257}
            )
            assert invalid.isError
            assert not backend.requests
            result = await client.call_tool(
                "generate_image", {"prompt": "中文图片"}, progress_callback=progress
            )
            assert not result.isError
            data = result.structuredContent
            assert data["status"] == "succeeded"
            assert data["request"]["steps"] == 9
            assert isinstance(data["request"]["seed"], int)
            assert any(item.type == "resource_link" for item in result.content)
            assert [n[0] for n in notifications] == sorted(n[0] for n in notifications)
            assert notifications[0][0] == 0
            assert notifications[-1][0] == 100
            assert any("denoising" in n[2] for n in notifications)
            assert any("starting: GPU worker started" in n[2] for n in notifications)
            assert all(data["id"] in n[2] for n in notifications)
            inline = await client.call_tool(
                "get_image_result", {"job_id": data["id"], "inline": True}
            )
            encoded = next(item.data for item in inline.content if item.type == "image")
            image = Image.open(io.BytesIO(base64.b64decode(encoded)))
            assert image.size == (1024, 1024)
            async with httpx.AsyncClient() as http:
                download = await http.get(data["result"]["download_url"])
                assert download.status_code == 200
                assert hashlib.sha256(download.content).hexdigest() == data["result"]["sha256"]
                assert (
                    await http.get(f"{url}/images/{data['id']}.png?key=wrong")
                ).status_code == 404
                assert (await http.get(f"{url}/health")).json()["status"] == "ok"
                assert (
                    await http.get(f"{url}/images/{data['id']}.png", params={"key": "错误"})
                ).status_code == 404


async def test_detached_job_survives_client_session_and_failure_is_error(live_server):
    url, app, backend = live_server
    async with transport(f"{url}/mcp") as streams:
        async with ClientSession(streams[0], streams[1]) as client:
            await client.initialize()
            result = await client.call_tool("submit_image_generation", {"prompt": "background"})
            job_id = result.structuredContent["id"]
    async with asyncio.timeout(10):
        async for job in app.state.manager.events(job_id):
            pass
    assert job.status == "succeeded"
    async with transport(f"{url}/mcp") as streams:
        async with ClientSession(streams[0], streams[1]) as client:
            await client.initialize()
            result = await client.call_tool("get_image_job", {"job_id": job_id})
            assert result.structuredContent["status"] == "succeeded"
            failed = await client.call_tool("generate_image", {"prompt": "fail"})
            assert failed.isError
            assert failed.structuredContent["status"] == "failed"


async def test_auth_and_host_protection(settings):
    from pydantic import SecretStr

    settings.api_token = SecretStr("test-token")
    app = create_app(settings, FakeBackend())
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app), base_url="http://localhost:8766"
        ) as http:
            assert (await http.get("/mcp")).status_code == 401
            assert (await http.get("/health")).status_code == 200
            response = await http.post(
                "/mcp",
                headers={
                    "authorization": "Bearer test-token",
                    "host": "attacker.example",
                    "accept": "application/json, text/event-stream",
                },
                json={"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}},
            )
            assert response.status_code == 421


async def test_cancel_tool_stops_gpu_and_next_request_succeeds(live_server):
    url, app, backend = live_server
    backend.delay = 0.15
    async with transport(f"{url}/mcp") as streams:
        async with ClientSession(streams[0], streams[1]) as client:
            await client.initialize()
            result = await client.call_tool("submit_image_generation", {"prompt": "cancel me"})
            job_id = result.structuredContent["id"]
            async for event in app.state.manager.events(job_id):
                if event.progress > 0:
                    break
            canceled = await client.call_tool("cancel_image_job", {"job_id": job_id})
            assert canceled.structuredContent["status"] == "cancel_requested"
            async for event in app.state.manager.events(job_id):
                pass
            assert event.status == "canceled"
            assert not (app.state.manager.output_dir / f"{job_id}.png").exists()
            assert not (await client.call_tool("generate_image", {"prompt": "next"})).isError


async def test_sampling_is_chosen_by_server_for_each_request(live_server, monkeypatch):
    url, app, backend = live_server
    seeds = iter([101, 202])
    monkeypatch.setattr("z_image_mcp.models.secrets.randbelow", lambda limit: next(seeds))
    async with transport(f"{url}/mcp") as streams:
        async with ClientSession(streams[0], streams[1]) as client:
            await client.initialize()
            direct = await client.call_tool("generate_image", {"prompt": "test"})
            background = await client.call_tool("submit_image_generation", {"prompt": "test"})
            assert direct.structuredContent["request"]["seed"] == 101
            assert background.structuredContent["request"]["seed"] == 202
            assert direct.structuredContent["request"]["steps"] == 9
            assert background.structuredContent["request"]["steps"] == 9
            info = (await client.call_tool("get_image_service_info")).structuredContent
            assert info["sampling"]["managed_by"] == "server"
            assert "seed" not in info["limits"]
            assert "steps" not in info["limits"]
