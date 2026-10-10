"""Metadata and read-only recovery regression; no model, TTS or rendering."""
import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from lesson_renderer.server import get_prepared_lesson, mcp
from lesson_renderer.store import ObjectLookupError, object_path


class LookupTests(unittest.TestCase):
    def test_bad_and_missing_ids_have_actionable_mcp_errors(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rt = Mock()
            rt.prepared.side_effect = lambda identifier: object_path(root, identifier, "lesson_")
            with patch("lesson_renderer.server.runtime", return_value=rt):
                for identifier, code in [("pirate-100-gold-10-pirates", "invalid_object_id"),
                                         ("lesson_" + "a" * 64, "object_not_found")]:
                    wire = get_prepared_lesson(identifier).model_dump(by_alias=True, exclude_none=True)
                    self.assertTrue(wire["isError"])
                    self.assertEqual(wire["structuredContent"]["error"]["code"], code)
                    self.assertFalse(wire["structuredContent"]["error"]["retryable"])
                    self.assertIn("prepare_lesson", wire["content"][0]["text"])
                    self.assertIn("do not recreate", wire["content"][0]["text"])
                    self.assertNotIn(directory, json.dumps(wire))
            self.assertEqual(list(root.iterdir()), [])

    def test_valid_recovery_keeps_exact_checkpoint_and_speech_requests(self):
        payload = {"lesson_id":"lesson_" + "b" * 64, "speech_requests":[{"segment_id":"one"}],
                   "_meta":{"exoharness":{"workflow_continuation":{"status":"required","required_tool":"submit_lesson_render"}}}}
        rt = Mock()
        rt.prepared.return_value = payload
        with patch("lesson_renderer.server.runtime", return_value=rt):
            wire = get_prepared_lesson(payload["lesson_id"]).model_dump(by_alias=True, exclude_none=True)
        self.assertFalse(wire.get("isError", False))
        self.assertEqual(wire["_meta"], payload["_meta"])
        self.assertEqual(wire["structuredContent"]["speech_requests"], payload["speech_requests"])
        rt.prepared.assert_called_once_with(payload["lesson_id"])

    def test_integrity_failures_are_not_disguised_as_missing_plans(self):
        rt = Mock()
        rt.prepared.side_effect = ValueError("saved lesson plan integrity check failed")
        with patch("lesson_renderer.server.runtime", return_value=rt):
            with self.assertRaisesRegex(ValueError, "integrity"):
                get_prepared_lesson("lesson_" + "a" * 64)

    def test_schema_distinguishes_recovery_from_creation(self):
        async def inspect():
            tool = next(t for t in await mcp.list_tools() if t.name == "get_prepared_lesson")
            self.assertIn("EXISTING", tool.description)
            self.assertIn("prepare_lesson", tool.input_schema["properties"]["lesson_id"]["description"])
        asyncio.run(inspect())

    def test_lookup_errors_remain_value_errors_for_existing_callers(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ObjectLookupError):
                object_path(Path(directory), "wrong", "lesson_")
        self.assertTrue(issubclass(ObjectLookupError, ValueError))
