from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(PROJECT / "tests"))

from media_workflow_mcp.plan import MusicVideoPlan, validate_and_render_music_video_plan
from media_workflow_mcp.plan_store import MusicVideoPlanStore
from test_plan import valid_plan


class MusicVideoPlanStoreTests(unittest.TestCase):
    def test_returns_compact_plan_reference_and_one_exact_segment(self) -> None:
        rendered = validate_and_render_music_video_plan(
            MusicVideoPlan.model_validate(valid_plan())
        )
        full_size = len(str(rendered))

        with tempfile.TemporaryDirectory() as raw_dir:
            store = MusicVideoPlanStore(Path(raw_dir))
            reference = store.put(rendered)
            self.assertEqual(reference["segment_count"], 8)
            self.assertEqual(reference["segment_tool"], "get_music_video_plan_segment")
            self.assertLess(len(str(reference)), full_size // 5)
            self.assertNotIn("positive_prompt", str(reference))

            fetched = store.get_segment(
                str(reference["plan_digest"]), "segment-003"
            )
            self.assertEqual(fetched["segment"]["segment_id"], "segment-003")
            self.assertIn("positive_prompt", fetched["segment"])
            self.assertEqual(
                fetched["segment"]["positive_prompt"],
                rendered["segments"][2]["positive_prompt"],
            )
            self.assertEqual(store.get(str(reference["plan_digest"])), rendered)

    def test_rejects_invalid_references(self) -> None:
        with tempfile.TemporaryDirectory() as raw_dir:
            store = MusicVideoPlanStore(Path(raw_dir))
            with self.assertRaisesRegex(ValueError, "plan_digest"):
                store.get_segment("../invalid", "segment-001")
            with self.assertRaisesRegex(ValueError, "segment_id"):
                store.get_segment("a" * 64, "../segment-001")

    def test_persists_content_addressed_draft_for_small_followup_patch(self) -> None:
        plan = MusicVideoPlan.model_validate(valid_plan()).model_dump(mode="json")
        with tempfile.TemporaryDirectory() as raw_dir:
            store = MusicVideoPlanStore(Path(raw_dir))
            reference = store.put_draft(plan)
            restored = store.get_draft(
                reference["draft_id"], reference["draft_digest"]
            )

            self.assertEqual(restored, plan)
            mode = (
                Path(raw_dir) / "plan-drafts" / f"{reference['draft_id']}.json"
            ).stat().st_mode & 0o777
            self.assertEqual(mode, 0o600)


if __name__ == "__main__":
    unittest.main()
