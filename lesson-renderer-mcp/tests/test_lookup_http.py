"""Exercise real MCP wire errors and successful recovery without TTS/rendering."""
import asyncio
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import unittest

import httpx2
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client


class LookupHttpTests(unittest.IsolatedAsyncioTestCase):
    async def test_lookup_wire_contract_and_prepare_recovery(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with socket.socket() as sock:
                sock.bind(("127.0.0.1", 0))
                port = sock.getsockname()[1]
            origin = f"http://127.0.0.1:{port}"
            token = "local-read-contract-fixture-" * 3
            secret = root / "token"
            secret.write_text(token)
            secret.chmod(0o600)
            env = {**os.environ, "LESSON_TOKEN_FILE":str(secret), "LESSON_ASSET_BASE":origin,
                   "LESSON_PUBLIC_BASE":origin, "LESSON_DATA_DIR":str(root / "data"),
                   "LESSON_SKILL_DIR":str(Path(os.getenv("TEST_LESSON_SKILL", Path(__file__).resolve().parents[3] / "skills/elementary-lesson-video"))),
                   "LESSON_HOST":"127.0.0.1", "LESSON_PORT":str(port)}
            with (root / "server.log").open("wb") as log:
                process = subprocess.Popen([sys.executable,"-m","lesson_renderer.server"],env=env,stdout=log,stderr=subprocess.STDOUT)
                try:
                    async with httpx2.AsyncClient(headers={"Authorization":"Bearer " + token},timeout=httpx2.Timeout(30)) as http:
                        for _ in range(100):
                            try:
                                # No worker: a reachable 503 health response is expected.
                                response = await http.get(origin + "/health")
                                if response.status_code in (200,503): break
                            except httpx2.TransportError:
                                pass
                            await asyncio.sleep(.1)
                        else:
                            self.fail((root / "server.log").read_text()[-2000:])
                        async with streamable_http_client(origin + "/mcp", http_client=http) as streams:
                            async with ClientSession(*streams,read_timeout_seconds=30) as session:
                                await session.initialize()
                                for identifier, code in [("pirate-100-gold-10-pirates","invalid_object_id"),("lesson_"+"a"*64,"object_not_found")]:
                                    result = await session.call_tool("get_prepared_lesson",{"lesson_id":identifier})
                                    wire = result.model_dump(by_alias=True,exclude_none=True)
                                    self.assertTrue(wire["isError"])
                                    self.assertEqual(wire["structuredContent"]["error"]["code"],code)
                                    self.assertIn("prepare_lesson",wire["content"][0]["text"])
                                    self.assertNotIn(directory,json.dumps(wire))
                                lesson = {"title":"光与影","question":"为什么会有影子？","subject":"科学","grade":"小学一年级",
                                          "segments":[{"id":"intro","narration":"物体挡住光，会形成影子。", "visual":{"type":"title","title":"光与影","subtitle":"为什么会有影子？"}}]}
                                prepared = await session.call_tool("prepare_lesson",{"lesson":lesson})
                                self.assertFalse(prepared.is_error, f"{prepared}\n{(root / 'server.log').read_text()[-3000:]}")
                                payload = prepared.model_dump(by_alias=True)["structuredContent"]
                                identifier = payload["lesson_id"]
                                self.assertRegex(identifier,r"^lesson_[0-9a-f]{64}$")
                                recovered = await session.call_tool("get_prepared_lesson",{"lesson_id":identifier})
                                self.assertFalse(recovered.is_error,recovered)
                                recovered_wire = recovered.model_dump(by_alias=True)
                                self.assertEqual(recovered_wire["structuredContent"]["speech_requests"],payload["speech_requests"])
                                self.assertEqual(recovered_wire["_meta"]["exoharness"]["workflow_continuation"]["required_tool"],"submit_lesson_render")
                                self.assertEqual(len(list((root / "data/plans").iterdir())),1)
                                self.assertEqual(list((root / "data/jobs").iterdir()),[])
                finally:
                    process.terminate()
                    try: process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
