from __future__ import annotations

import asyncio
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from media_workflow_mcp.connections import GENERATOR_CONNECTIONS
from media_workflow_mcp.server import mcp
from media_workflow_mcp.upstream import generator_urls
from types import SimpleNamespace


class ConnectionDeclarationTests(unittest.TestCase):
    def test_live_tools_list_exposes_connections_only_for_generator_operations(self):
        tools = {t.name: t for t in asyncio.run(mcp.list_tools())}
        operations = {
            "create_music_video_plan", "render_validated_music_video",
            "prepare_narrated_video", "render_narrated_video",
            "generate_standalone_video", "resume_standalone_video",
        }
        for name in operations:
            self.assertEqual(tools[name].meta, GENERATOR_CONNECTIONS)
        for name in ("probe_media", "extract_last_frame", "assemble_music_video", "assemble_video_timeline"):
            self.assertIsNone(tools[name].meta)

    def test_public_json_schema_remains_native_and_has_no_connection_controls(self):
        tools = {t.name: t for t in asyncio.run(mcp.list_tools())}
        for name in ("generate_standalone_video", "prepare_narrated_video"):
            schema = tools[name].input_schema
            self.assertEqual(schema["type"], "object")
            self.assertEqual(schema["properties"]["plan"].get("$ref", "")[:2], "#/")
            self.assertNotIn("connection_bindings", schema["properties"])

    def test_declared_header_keeps_existing_upstream_fallback_and_order(self):
        binding = GENERATOR_CONNECTIONS["exoharness"]["connection_bindings"]["headers"][0]
        settings = SimpleNamespace(video_generator_url="https://configured.test/mcp")
        self.assertFalse(binding["required"])
        self.assertEqual(generator_urls(settings), ["https://configured.test/mcp"])
        self.assertEqual(generator_urls(settings, {binding["name"]: "https://first.test/mcp,https://second.test/mcp"}),
                         ["https://first.test/mcp", "https://second.test/mcp", "https://configured.test/mcp"])


if __name__ == "__main__":
    unittest.main()
