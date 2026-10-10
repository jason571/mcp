from __future__ import annotations

import asyncio
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(PROJECT / "tests"))

from media_workflow_mcp.plan import (
    MusicVideoPlan,
    MusicVideoPlanPatch,
    apply_music_video_plan_patches,
)
from media_workflow_mcp.plan_store import MusicVideoPlanStore
from media_workflow_mcp.server import ProgressIdleTimeout, _store_validated_or_draft
from test_plan import valid_plan


class ProgressIdleTimeoutTests(unittest.IsolatedAsyncioTestCase):
    async def test_progress_extends_deadline(self) -> None:
        # Test deadline movement against a controlled monotonic clock, not a
        # 20 ms scheduler margin on a shared CI host. The silence test below
        # still exercises real cancellation by asyncio's timeout machinery.
        loop = asyncio.get_running_loop()
        now = [loop.time()]
        with patch.object(loop, "time", side_effect=lambda: now[0]):
            async with ProgressIdleTimeout(0.05) as watchdog:
                initial_deadline = watchdog._timeout.when()
                for _ in range(3):
                    now[0] += 0.03
                    self.assertLess(now[0], watchdog._timeout.when())
                    watchdog.touch()
                    self.assertAlmostEqual(watchdog._timeout.when(), now[0] + 0.05)
                self.assertGreater(now[0], initial_deadline)
                self.assertFalse(watchdog._timeout.expired())

    async def test_silence_expires_deadline(self) -> None:
        with self.assertRaises(TimeoutError):
            async with ProgressIdleTimeout(0.02):
                await asyncio.sleep(0.05)


class IncrementalPlanRepairTests(unittest.TestCase):
    def test_missing_segment_is_appended_without_resending_existing_ten(self) -> None:
        raw = valid_plan(313_992)
        missing = raw["segments"].pop()
        original_prompts = [segment["prompt_plan"] for segment in raw["segments"]]
        settings = SimpleNamespace(
            video_generator_min_duration_seconds=5,
            video_generator_max_duration_seconds=60,
            video_generator_target_duration_seconds=30,
            video_generator_duration_step_seconds=1,
            video_generator_tail_merge_max_duration_seconds=40,
        )
        with tempfile.TemporaryDirectory() as raw_dir:
            runtime = SimpleNamespace(settings=settings, plans=MusicVideoPlanStore(Path(raw_dir)))
            invalid = _store_validated_or_draft(runtime, MusicVideoPlan.model_validate(raw))
            self.assertFalse(invalid["valid"])
            self.assertIn("/segments/10", invalid["patch_targets"])
            metadata = invalid["_meta"]["exoharness"]
            self.assertIn("/validation_errors", metadata["context_preserve"])
            repairs = metadata["workflow_continuation"]["repair_context"]["missing_segments"]
            self.assertEqual(repairs[0]["op"], "add")
            self.assertEqual(repairs[0]["start_ms"], 300_000)
            self.assertEqual(repairs[0]["end_ms"], 313_992)
            stored = runtime.plans.get_draft(invalid["draft_id"], invalid["draft_digest"])
            patched = apply_music_video_plan_patches(MusicVideoPlan.model_validate(stored), [
                MusicVideoPlanPatch(op="add", path="/segments/10", value=missing),
            ])
            valid = _store_validated_or_draft(runtime, patched)
            self.assertTrue(valid["valid"])
            self.assertEqual(valid["segment_count"], 11)
            self.assertEqual([segment.prompt_plan.model_dump(mode="json") for segment in patched.segments[:10]], original_prompts)

    def test_invalid_full_plan_is_persisted_and_repaired_with_one_small_patch(self) -> None:
        raw = valid_plan(60_000)
        raw["segments"][1]["visual_role"] = raw["segments"][0]["visual_role"]
        settings = SimpleNamespace(
            video_generator_min_duration_seconds=5,
            video_generator_max_duration_seconds=60,
            video_generator_target_duration_seconds=30,
            video_generator_duration_step_seconds=1,
            video_generator_tail_merge_max_duration_seconds=40,
        )

        with tempfile.TemporaryDirectory() as raw_dir:
            plans = MusicVideoPlanStore(Path(raw_dir))
            runtime = SimpleNamespace(settings=settings, plans=plans)
            invalid = _store_validated_or_draft(
                runtime, MusicVideoPlan.model_validate(raw)
            )
            self.assertFalse(invalid["valid"])
            self.assertEqual(invalid["next_tool"], "patch_music_video_plan")
            self.assertEqual(invalid["patch_targets"], ["/segments/1/visual_role"])
            continuation = invalid["_meta"]["exoharness"]["workflow_continuation"]
            self.assertEqual(continuation["status"], "required")
            self.assertEqual(continuation["required_tool"], "patch_music_video_plan")
            self.assertEqual(continuation["required_arguments"], {
                "draft_id": invalid["draft_id"],
                "draft_digest": invalid["draft_digest"],
            })
            self.assertEqual(
                continuation["allowed_argument_paths"],
                invalid["patch_targets"],
            )
            self.assertNotIn("segments", invalid)
            self.assertLess(len(str(invalid)), 2_000)

            stored = plans.get_draft(
                invalid["draft_id"], invalid["draft_digest"]
            )
            patched = apply_music_video_plan_patches(
                MusicVideoPlan.model_validate(stored),
                [
                    MusicVideoPlanPatch(
                        op="replace",
                        path=invalid["patch_targets"][0],
                        value="A distinct second chapter and emotional turn",
                    )
                ],
            )
            valid = _store_validated_or_draft(runtime, patched)

            self.assertTrue(valid["valid"])
            self.assertEqual(valid["segment_count"], 2)
            self.assertIn("plan_digest", valid)
            continuation = valid["_meta"]["exoharness"]["workflow_continuation"]
            self.assertEqual(continuation["status"], "required")
            self.assertEqual(
                continuation["required_tool"],
                "render_validated_music_video",
            )
            self.assertEqual(
                continuation["required_arguments"],
                {"plan_digest": valid["plan_digest"]},
            )


if __name__ == "__main__":
    unittest.main()
