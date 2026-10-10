import copy
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
from media_workflow_mcp.narrated_checkpoint import prepared_checkpoint


class HandoffTests(unittest.TestCase):
    def receipt(self):
        return {"plan_digest": "a" * 64, "status": "prepared", "segment_count": 8,
                "duration_seconds": 120, "clip_duration_seconds": 15,
                "tail_merge_max_seconds": 21, "narration_pace_hint": "Exact service hint",
                "narrations": [{"segment_id": f"segment-{i:03d}", "text": f"逐字保留第{i}段旁白", "recommended_max_characters": 56} for i in range(8)]}

    def test_saved_narration_declares_handoff_without_changing_any_content(self):
        receipt = self.receipt()
        original = copy.deepcopy(receipt)
        meta = prepared_checkpoint(receipt)
        directive = meta["workflow_continuation"]
        self.assertEqual(directive["required_tool"], "render_narrated_video")
        self.assertEqual(directive["required_arguments"], {"plan_digest": receipt["plan_digest"]})
        self.assertEqual(directive["prerequisite_tools"], ["model.synthesize_speech"])
        self.assertEqual(directive["context_scope"], "checkpoint")
        self.assertIn("artifact_visibility=internal", directive["reason"])
        self.assertEqual(receipt, original)
        for path in meta["context_preserve"]:
            self.assertIn(path[1:], receipt)

    def test_silent_plan_has_no_speech_prerequisite(self):
        receipt = self.receipt()
        receipt["narrations"] = []
        self.assertEqual(prepared_checkpoint(receipt)["workflow_continuation"]["prerequisite_tools"], [])

    def test_warning_keeps_reauthoring_available_and_preserves_exact_narration(self):
        receipt = self.receipt()
        receipt["narrations"][3]["likely_too_long"] = True
        original = copy.deepcopy(receipt)
        meta = prepared_checkpoint(receipt)
        self.assertNotIn("workflow_continuation", meta)
        self.assertEqual(receipt, original)
        self.assertIn("/narrations", meta["context_preserve"])


class SdkHandoffTests(unittest.IsolatedAsyncioTestCase):
    async def test_actual_sdk_function_adds_only_handoff_metadata_without_submitting_generation(self):
        from contextlib import asynccontextmanager
        from media_workflow_mcp import server
        from media_workflow_mcp.narrated import NarratedPlan
        receipt = HandoffTests().receipt()
        original = copy.deepcopy(receipt)
        plan = NarratedPlan(title="fixture", target_duration_seconds=15, segments=[{
            "segment_id":"segment-001", "duration_seconds":15,
            "generation":{"prompt":"EXACT FOUR SHOT PROMPT", "duration_seconds":15,"audio_enabled":False},
            "narration":"EXACT NARRATION"}])
        exact_plan = copy.deepcopy(plan.model_dump())
        generator = SimpleNamespace(submit=AsyncMock())
        @asynccontextmanager
        async def connection(*args):
            yield generator
        workflow = SimpleNamespace(prepare=AsyncMock(return_value=receipt))
        runtime = SimpleNamespace(settings=object(), artifacts=object())
        with patch.object(server,"_require_tool_auth",return_value=runtime), \
             patch.object(server,"connect_video_generator",connection), \
             patch.object(server,"NarratedWorkflow",return_value=workflow):
            result = await server.prepare_narrated_video(plan,SimpleNamespace(headers={}))
        self.assertEqual({k:v for k,v in result.items() if k!="_meta"},original)
        self.assertEqual(plan.model_dump(),exact_plan)
        generator.submit.assert_not_called()
        workflow.prepare.assert_awaited_once_with(plan,generator)
        self.assertEqual(result["_meta"]["exoharness"]["workflow_continuation"]["required_tool"],"render_narrated_video")
