import asyncio
import json
import logging
import os
import socket
import subprocess
import sys
import time
import unittest
from pathlib import Path

import aiohttp
import httpx2
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

from test_renderer import PipelineTests, PLAN, SKILL

logging.getLogger("httpx2").setLevel(logging.WARNING)
logging.getLogger("mcp.client.streamable_http").setLevel(logging.WARNING)


def load_tests(loader, tests, pattern):
    # Reuse the fixture without running inherited unit cases a second time.
    return unittest.TestSuite([HttpTests("test_mcp_transport_completion_and_private_range_delivery")])


class HttpTests(PipelineTests):
    async def test_mcp_transport_completion_and_private_range_delivery(self):
        with socket.socket() as socket_:
            socket_.bind(("127.0.0.1", 0))
            port = socket_.getsockname()[1]
        origin = f"http://127.0.0.1:{port}"
        token = "local-fixture-token-" * 4
        secret = self.root / "token"
        secret.write_text(token)
        secret.chmod(0o600)
        env = {**os.environ, "LESSON_TOKEN_FILE": str(secret), "LESSON_ASSET_BASE": self.base,
               "LESSON_PUBLIC_BASE": origin, "LESSON_SKILL_DIR": str(SKILL), "LESSON_DATA_DIR": str(self.root / "live"),
               "LESSON_JOB_DIR": str(self.root / "live/jobs"), "LESSON_PORT": str(port), "LESSON_HOST": "127.0.0.1"}
        processes = []
        with (self.root / "server.log").open("wb") as log:
            try:
                for module in ["lesson_renderer.server", "lesson_renderer.worker"]:
                    processes.append(subprocess.Popen([sys.executable, "-m", module], env=env, stdout=log, stderr=subprocess.STDOUT))
                async with aiohttp.ClientSession() as http:
                    for _ in range(100):
                        try:
                            async with http.get(origin + "/health") as response:
                                if response.status == 200:
                                    break
                        except aiohttp.ClientError:
                            pass
                        await asyncio.sleep(.1)
                    else:
                        self.fail((self.root / "server.log").read_text()[-2000:])
                    async with http.post(origin + "/mcp", json={}) as response:
                        self.assertEqual(response.status, 401)
                async with httpx2.AsyncClient(headers={"Authorization": "Bearer " + token}, timeout=httpx2.Timeout(120)) as http:
                    async with streamable_http_client(origin + "/mcp", http_client=http) as streams:
                        async with ClientSession(*streams, read_timeout_seconds=120) as session:
                            await session.initialize()
                            tools = await session.list_tools()
                            self.assertIn("submit_lesson_render", [t.name for t in tools.tools])
                            async def call(name, arguments):
                                result = await session.call_tool(name, arguments)
                                wire = result.model_dump(by_alias=True, exclude_none=True)
                                self.assertFalse(wire.get("isError"), wire)
                                self.assertIsInstance(wire.get("structuredContent"), dict)
                                self.assertEqual(json.loads(wire["content"][0]["text"]), wire["structuredContent"])
                                return {**wire["structuredContent"], "_meta": wire.get("_meta", {})}
                            broken = json.loads(json.dumps(PLAN))
                            broken["segments"][0]["heading"] = broken["segments"][0]["visual"].pop("heading")
                            draft = await call("prepare_lesson", {"lesson": broken})
                            self.assertEqual(draft["status"], "needs_repair")
                            self.assertNotIn("speech_requests", draft)
                            focused = await call("get_lesson_draft", {"draft_id": draft["draft_id"], "paths": ["/segments/0/heading"]})
                            self.assertEqual(focused["focus"][0]["value"], PLAN["segments"][0]["visual"]["heading"])
                            arguments = {"draft_id": draft["draft_id"], "expected_revision": draft["revision"], "patches": [
                                {"op": "move", "from_path": "/segments/0/heading", "path": "/segments/0/visual/heading"}]}
                            plan = await call("patch_lesson_draft", arguments)
                            replay = await call("patch_lesson_draft", arguments)
                            self.assertEqual(plan["lesson_id"], replay["lesson_id"])
                            self.assertTrue(replay["idempotent_replay"])
                            submitted = await call("submit_lesson_render", {"lesson_id": plan["lesson_id"], "audio": [b.model_dump() for b in self.bindings], "quality": "l"})
                            task = submitted["_meta"]["exoharness"]["background_task"]
                            progress = []
                            for _ in range(100):
                                result = await call(task["poll"]["tool"], task["poll"]["arguments"])
                                progress.append(result["progress"])
                                if result["status"] in {"completed", "failed", "cancelled"}:
                                    break
                                await asyncio.sleep(.3)
                            self.assertEqual(result["status"], "completed", result)
                            self.assertEqual(result["_meta"]["exoharness"]["background_task"]["id"], task["id"])
                            video = next(f for f in result["artifacts"] if f["content_type"] == "video/mp4")
                            self.assertIn(100, progress)
                async with aiohttp.ClientSession() as http:
                    async with http.get(video["download_url"].split("?", 1)[0]) as response:
                        self.assertEqual(response.status, 404)
                    async with http.get(video["download_url"], headers={"Range": "bytes=0-15"}) as response:
                        self.assertEqual(response.status, 206)
                        self.assertEqual(len(await response.read()), 16)
                    async with http.get(video["download_url"]) as response:
                        from lesson_renderer.store import digest
                        self.assertEqual(digest(await response.read()), video["sha256"])
            finally:
                for process in processes:
                    process.terminate()
                for process in processes:
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
