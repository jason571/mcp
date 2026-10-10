from __future__ import annotations

import asyncio
import json
import sys
import unittest
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from aiohttp import web

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from media_workflow_mcp.upstream import (
    UpstreamToolError,
    VideoGeneratorClient,
    _numeric_duration_from_mcp,
    bind_generator_contract,
    connect_video_generator,
    generator_urls,
)


class UpstreamTransportTests(unittest.IsolatedAsyncioTestCase):
    async def test_reference_workflow_selects_compatible_registered_generator(self):
        @asynccontextmanager
        async def connected(settings):
            fields = {"prompt": {}, "duration_seconds": {}, "first_frame_url": {}}
            if settings.video_generator_url.endswith("compatible/mcp"):
                fields["reference_image_url"] = {}
            yield SimpleNamespace(
                submission_schema=AsyncMock(return_value={"properties": fields}),
                selected_url=settings.video_generator_url,
            )

        settings = SimpleNamespace(
            video_generator_url=(
                "https://example.test/other/mcp,"
                "https://example.test/compatible/mcp"
            )
        )
        with patch("media_workflow_mcp.upstream._connect_generator_url", connected):
            async with connect_video_generator(
                settings,
                required_submit_fields=("first_frame_url", "reference_image_url"),
            ) as client:
                self.assertEqual(
                    client.selected_url, "https://example.test/compatible/mcp"
                )

    async def test_workflow_error_is_not_rewritten_as_discovery_failure(self):
        @asynccontextmanager
        async def connected(_settings):
            yield SimpleNamespace()

        settings = SimpleNamespace(video_generator_url="http://generator.test/mcp")
        with patch(
            "media_workflow_mcp.upstream._connect_generator_url", connected
        ):
            with self.assertRaisesRegex(
                ValueError,
                "media source origin is not allowed",
            ):
                async with connect_video_generator(settings):
                    raise ValueError(
                        "media source origin is not allowed: http://example.test"
                    )

    def test_numeric_duration_reads_status_and_models_payloads(self) -> None:
        self.assertEqual(_numeric_duration_from_mcp({"default_duration_seconds": 15}), 15)
        self.assertEqual(_numeric_duration_from_mcp({"defaults": {"duration_seconds": 15}}), 15)
        self.assertEqual(_numeric_duration_from_mcp({"duration_seconds": {"default": 15}}), 15)
        self.assertIsNone(_numeric_duration_from_mcp({"description": "defaults to 30"}))
        self.assertIsNone(_numeric_duration_from_mcp({"duration_seconds": None}))

    def test_bind_generator_picks_closest_description_when_several_match(self) -> None:
        def tool(name, description, *fields):
            return SimpleNamespace(
                name=name,
                description=description,
                input_schema={"properties": {field: {} for field in fields}},
            )

        preferred = {
            "submit": "submit_video_generation",
            "status": "get_video_job",
            "result": "get_video_result",
            "cancel": "cancel_video_job",
        }
        bound = bind_generator_contract(
            [
                tool("preview_clip", "Create a short video clip preview.", "prompt", "duration_seconds"),
                tool(
                    "generate_video",
                    "Submit a MiniMax H3 video generation job from a prompt.",
                    "prompt",
                    "duration_seconds",
                    "audio_enabled",
                    "aspect_ratio",
                    "resolution",
                ),
                tool("get_video_job", "Get the status of a video job.", "job_id"),
                tool("get_video_result", "Fetch the completed video job result.", "job_id"),
                tool("cancel_video_job", "Cancel an in-flight video job.", "job_id"),
            ],
            preferred,
        )
        self.assertEqual(bound["submit"], "generate_video")
        self.assertEqual(bound["status"], "get_video_job")
        self.assertEqual(bound["result"], "get_video_result")
        self.assertEqual(bound["cancel"], "cancel_video_job")

    def test_generator_urls_split_header_and_skip_duplicates(self) -> None:
        settings = SimpleNamespace(
            video_generator_url="http://127.0.0.1:8765/mcp,http://127.0.0.1:8766/mcp"
        )
        self.assertEqual(
            generator_urls(
                settings,
                {
                    "X-Exoharness-Video-Generator-Url": (
                        "http://127.0.0.1:8766/mcp,http://127.0.0.1:8765/mcp"
                    )
                },
            ),
            ["http://127.0.0.1:8766/mcp", "http://127.0.0.1:8765/mcp"],
        )

    async def test_submission_can_wait_six_seconds_without_default_http_timeout(self):
        result, calls = await self.call_delayed_fixture(delay=6, timeout=15)
        self.assertEqual(result["job_id"],"fixture-job")
        self.assertEqual(calls,1)

    async def test_configured_timeout_is_still_enforced_without_resubmission(self):
        with self.assertRaisesRegex(UpstreamToolError, "timed out"):
            await self.call_delayed_fixture(delay=0.5, timeout=0.1)
        self.assertEqual(self.fixture_calls,1)

    async def test_connect_skips_stopped_generator_url(self):
        result, calls = await self.call_delayed_fixture(
            delay=0.1, timeout=15, extra_dead_url="http://127.0.0.1:1/mcp"
        )
        self.assertEqual(result["job_id"], "fixture-job")
        self.assertEqual(calls, 1)

    async def call_delayed_fixture(self, *, delay, timeout, extra_dead_url=None):
        calls = 0
        self.fixture_calls = 0
        async def endpoint(request):
            nonlocal calls
            if request.method == "GET":
                return web.Response(status=405)
            if request.method == "DELETE":
                return web.Response(status=200)
            body = await request.json()
            if body["method"] == "initialize":
                return web.json_response({"jsonrpc":"2.0","id":body["id"],"result":{
                    "protocolVersion":body["params"]["protocolVersion"],"capabilities":{"tools":{}},
                    "serverInfo":{"name":"slow-fixture","version":"1"}}})
            if "id" not in body:
                return web.Response(status=202)
            if body["method"] == "tools/list":
                return web.json_response({"jsonrpc":"2.0","id":body["id"],"result":{
                    "tools":[
                        {"name":"submit","description":"Submit a video generation job","inputSchema":{"type":"object","properties":{"prompt":{"type":"string"},"duration_seconds":{"type":"number"}}}},
                        {"name":"get_video_job","description":"Get the status of a video job","inputSchema":{"type":"object","properties":{"job_id":{"type":"string"}}}},
                        {"name":"get_video_result","description":"Fetch the completed video job result","inputSchema":{"type":"object","properties":{"job_id":{"type":"string"}}}},
                        {"name":"cancel_video_job","description":"Cancel an in-flight video job","inputSchema":{"type":"object","properties":{"job_id":{"type":"string"}}}},
                    ]}})
            self.assertEqual(body["method"],"tools/call")
            calls += 1
            self.fixture_calls = calls
            response = web.StreamResponse(headers={"Content-Type":"text/event-stream"})
            await response.prepare(request)
            await asyncio.sleep(delay)
            payload = {"jsonrpc":"2.0","id":body["id"],"result":{
                "content":[{"type":"text","text":"accepted"}],
                "structuredContent":{"job_id":"fixture-job","status":"queued"},"isError":False}}
            try:
                await response.write(("event: message\ndata: "+json.dumps(payload)+"\n\n").encode())
                await response.write_eof()
            except ConnectionResetError:
                pass
            return response
        app = web.Application()
        app.router.add_route("*","/mcp",endpoint)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner,"127.0.0.1",0)
        await site.start()
        port = site._server.sockets[0].getsockname()[1]
        live = f"http://127.0.0.1:{port}/mcp"
        settings = SimpleNamespace(
            video_generator_url=f"{extra_dead_url},{live}" if extra_dead_url else live,
            video_generator_authorization=None,operation_timeout_seconds=timeout,
            video_generator_submit_tool="submit",
            video_generator_status_tool="get_video_job",
            video_generator_result_tool="get_video_result",
            video_generator_cancel_tool="cancel_video_job")
        try:
            with patch.object(VideoGeneratorClient,"validate_contract",AsyncMock()):
                async with connect_video_generator(settings) as client:
                    # Exercise the real SDK wire method, not a permissive fake
                    # generator. SDK v2 uses params=, not cursor=.
                    schema = await client.submission_schema()
                    self.assertEqual(schema.get("type"), "object")
                    self.assertIn("prompt", schema.get("properties", {}))
                    self.assertIn("duration_seconds", schema.get("properties", {}))
                    result = await client.submit({"prompt":"fixture only"},"local-id")
            return result, calls
        finally:
            await runner.cleanup()
