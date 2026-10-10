import asyncio
import socket

import httpx
import uvicorn
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client
from test_jobs import FakeBackend, terminal

from yue2_mcp.config import Settings
from yue2_mcp.models import MusicRequest
from yue2_mcp.server import create_app


async def http_contract(tmp_path):
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    settings = Settings(_env_file=None, data_dir=tmp_path, min_free_disk_mb=0,
                        api_token="test-token",
                        port=port, public_url=f"http://127.0.0.1:{port}")
    backend = FakeBackend()
    backend.gate.set()
    app = create_app(settings, backend)
    assert len(app.state.mcp._tool_manager.list_tools()) == 8
    server = uvicorn.Server(uvicorn.Config(app, log_level="error", access_log=False))
    task = asyncio.create_task(server.serve(sockets=[sock]))
    try:
        async with asyncio.timeout(10):
            while not server.started:
                if task.done():
                    await task
                await asyncio.sleep(0.01)
        async with httpx.AsyncClient(base_url=settings.public_url, timeout=10) as client:
            health = await client.get("/health")
            assert health.status_code == 503  # Local CPU test has no model files.
            unauth = await client.post("/mcp", json={})
            assert unauth.status_code == 401

            job = app.state.manager.submit("music", MusicRequest(style="pop", lyrics="rain"))
            assert (await terminal(app.state.manager, job.id)).status == "succeeded"
            result = app.state.manager.public(app.state.manager.get(job.id))
            valid = await client.get(result["result"]["audio_url"])
            assert valid.status_code == 200
            assert valid.content == b"fake flac"
            preview = await client.get(result["result"]["preview_url"])
            assert preview.status_code == 200
            assert preview.content == b"fake mp3"
            lyrics = await client.get(result["result"]["lyrics_url"])
            assert lyrics.status_code == 200
            assert lyrics.text == "rain"
            denied = await client.get(f"/files/{job.id}/song.flac?key=wrong")
            assert denied.status_code == 404
        progress = []

        async def on_progress(value, total, message):
            progress.append((value, message))

        async with httpx.AsyncClient(
            headers={"Authorization": "Bearer test-token"}, timeout=30
        ) as authorized:
            async with streamable_http_client(
                f"{settings.public_url}/mcp", http_client=authorized
            ) as streams:
                async with ClientSession(streams[0], streams[1]) as session:
                    await session.initialize()
                    names = {tool.name for tool in (await session.list_tools()).tools}
                    assert "generate_music" in names
                    assert "generate_lyrics" in names
                    generated = await session.call_tool(
                        "generate_music", {"style": "jazz", "lyrics": "[Verse 1] rain"},
                        progress_callback=on_progress,
                    )
                    assert not generated.isError
                    assert generated.structuredContent["status"] == "succeeded"
                    links = [item for item in generated.content if item.type == "resource_link"]
                    assert {item.mimeType for item in links} == {
                        "audio/mpeg", "text/plain"
                    }
                    visible_audio = [item.mimeType for item in links
                                     if item.mimeType.startswith("audio/") and
                                     (item.meta or {}).get("exoharness", {}).get(
                                         "artifact_visibility") != "internal"]
                    assert visible_audio == ["audio/mpeg"]
                    original = await session.call_tool(
                        "get_music_result", {"job_id": job.id, "audio_format": "flac"},
                    )
                    original_links = [item for item in original.content
                                      if item.type == "resource_link"]
                    original_visible = [item.mimeType for item in original_links
                                        if item.mimeType.startswith("audio/") and
                                        (item.meta or {}).get("exoharness", {}).get(
                                            "artifact_visibility") != "internal"]
                    assert original_visible == ["audio/flac"]
                    assert progress[0][0] == 0
                    assert progress[-1][0] == 100
                    backend.gate.clear()
                    submitted = await session.call_tool(
                        "submit_music_generation",
                        {"style": "folk", "lyrics": "[Verse 1] mountain"},
                    )
                    assert not submitted.isError
                    data = submitted.structuredContent
                    assert "request" not in data
                    job_id = data["id"]
                    declared = data["_meta"]["exoharness"]["background_task"]
                    assert declared["id"] == job_id
                    assert declared["status"] == "working"
                    assert declared["poll"] == {
                        "tool": "get_music_job", "arguments": {"job_id": job_id}
                    }
                    assert declared["cancel"] == {
                        "tool": "cancel_music_job", "arguments": {"job_id": job_id}
                    }
                    polled = await session.call_tool("get_music_job", {"job_id": job_id})
                    assert "request" not in polled.structuredContent
                    assert polled.structuredContent["_meta"]["exoharness"][
                        "background_task"]["id"] == job_id
                    cancelled = await session.call_tool("cancel_music_job", {"job_id": job_id})
                    assert cancelled.structuredContent["_meta"]["exoharness"][
                        "background_task"]["id"] == job_id
                    backend.gate.set()
                    assert (await terminal(app.state.manager, job_id)).status == "canceled"
    finally:
        server.should_exit = True
        await asyncio.wait_for(task, 15)
        sock.close()


def test_http_auth_and_artifact(tmp_path):
    asyncio.run(http_contract(tmp_path))
