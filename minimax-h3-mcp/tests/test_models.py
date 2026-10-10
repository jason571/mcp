from __future__ import annotations

import unittest
import sqlite3
import tempfile
from pathlib import Path

from minimax_h3_mcp.models import (
    GenerationRequest,
    MODEL_ID,
    aligned_num_frames,
    build_h3_prompt,
    canvas_dimensions,
    duration_resolution_limits,
    duration_seconds_contract,
    normalize_resolution,
)
from minimax_h3_mcp.store import JobStore


class ModelContractTests(unittest.TestCase):
    def test_frame_count_uses_h3_alignment(self) -> None:
        frames = aligned_num_frames(5)
        self.assertEqual((frames - 5) % 17, 0)
        self.assertGreaterEqual(frames, 5 * 24)

    def test_duration_limits_are_enforced(self) -> None:
        with self.assertRaises(ValueError):
            aligned_num_frames(4.99)
        with self.assertRaises(ValueError):
            aligned_num_frames(60.01)
        self.assertEqual(aligned_num_frames(30), 736)
        self.assertEqual(aligned_num_frames(60), 1450)

    def test_duration_contract_exposes_default_and_maximum(self) -> None:
        self.assertEqual(
            duration_seconds_contract(),
            {"default": 15.0, "minimum": 5.0, "maximum": 60.0},
        )
        self.assertEqual(
            duration_seconds_contract(15.0, 30.0),
            {"default": 15.0, "minimum": 5.0, "maximum": 30.0},
        )
        limits = duration_resolution_limits(30.0)
        self.assertEqual(limits[0]["duration_seconds"]["maximum"], 15.0)  # type: ignore[index]
        self.assertEqual(limits[1]["duration_seconds"]["maximum"], 30.0)  # type: ignore[index]
        self.assertEqual(len(duration_resolution_limits(15.0)), 1)

    def test_requested_ratios_and_resolution(self) -> None:
        self.assertEqual(canvas_dimensions("9:16", "768p"), (768, 1344))
        self.assertEqual(canvas_dimensions("16:9", "768p"), (1344, 768))
        with self.assertRaisesRegex(ValueError, "16:9, 9:16"):
            canvas_dimensions("19:9", "768p")

    def test_resolution_normalization_is_batch_safe(self) -> None:
        self.assertEqual(normalize_resolution(15, " 768P "), "768p")
        self.assertEqual(normalize_resolution(15, "1080p"), "512p")
        self.assertEqual(normalize_resolution(15, ""), "512p")
        self.assertEqual(normalize_resolution(15, None), "512p")
        self.assertEqual(normalize_resolution(15.01, "640p"), "512p")
        self.assertEqual(normalize_resolution(60, "768p"), "512p")

    def test_negative_prompt_is_an_h3_semantic_constraint(self) -> None:
        prompt = build_h3_prompt(
            "A crane shot over a forest.",
            "No watermark, no flicker.",
            "Birds and distant wind.",
            "simple",
        )
        self.assertIn("integrated_multimodal_description:", prompt)
        self.assertIn("overall_soundscape:", prompt)
        self.assertIn("negative_constraints: No watermark, no flicker.", prompt)

    def test_generation_request_exposes_keyframe_capability(self) -> None:
        request = GenerationRequest(
            prompt="test",
            negative_prompt=None,
            soundscape=None,
            prompt_mode="simple",
            model_id=MODEL_ID,
            duration_seconds=5,
            seed=1,
            audio_enabled=True,
            aspect_ratio="16:9",
            resolution="768p",
            first_frame_path="/tmp/first.png",
        )
        self.assertTrue(request.has_keyframes)

    def test_reference_image_is_persisted_separately_from_keyframes(self) -> None:
        request = GenerationRequest(
            prompt="test", negative_prompt=None, soundscape=None,
            prompt_mode="simple", model_id=MODEL_ID, duration_seconds=5,
            seed=1, audio_enabled=False, aspect_ratio="16:9", resolution="512p",
            reference_image_asset_id="img_" + "a" * 24,
            reference_image_path="/tmp/reference.png",
        )
        self.assertIn("subject_definitions: <Picture 1>", request.model_prompt)
        self.assertIn("detailed_description: test", request.model_prompt)
        self.assertIn("overall_soundscape: N/A", request.model_prompt)
        self.assertNotIn("at 0.00 seconds", request.model_prompt)
        from minimax_h3_mcp.models import JobRecord
        with tempfile.TemporaryDirectory() as temp:
            store = JobStore(Path(temp) / "jobs.sqlite3")
            saved = store.create(JobRecord.new(request))
            loaded = store.require(saved.id)
            self.assertEqual(loaded.reference_image_asset_id, request.reference_image_asset_id)
            self.assertEqual(loaded.to_request().reference_image_path, request.reference_image_path)
            self.assertIsNone(loaded.first_frame_path)
            self.assertNotIn("reference_image_path", loaded.as_public_dict())
            store.close()

    def test_structured_reference_prompt_keeps_the_image_binding(self) -> None:
        prompt = (
            "subject_definitions: A traveler waits at a station.\n"
            "summary: The traveler starts a quiet journey.\n"
            "retention_analysis: Keep the setting consistent.\n"
            "detailed_description: Camera follows the traveler.\n"
            "overall_soundscape: N/A\n"
            "non_diegetic_music: N/A"
        )
        actual = build_h3_prompt(
            prompt, None, None, "h3", reference_image=True, audio_enabled=False
        )
        self.assertIn("<Picture 1>", actual)
        self.assertIn("A traveler waits at a station.", actual)
        self.assertNotIn("frame zero", actual)
        self.assertEqual(actual.count("subject_definitions:"), 1)

    def test_existing_job_database_is_migrated_without_dropping_jobs(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "jobs.sqlite3"
            with sqlite3.connect(path) as connection:
                connection.execute("CREATE TABLE jobs (id TEXT PRIMARY KEY, status TEXT NOT NULL, created_at TEXT NOT NULL)")
                connection.execute("INSERT INTO jobs (id,status,created_at) VALUES ('old-job','completed','old')")
            store = JobStore(path)
            columns = {row[1] for row in store._connection.execute("PRAGMA table_info(jobs)")}
            self.assertIn("reference_image_asset_id", columns)
            self.assertIn("reference_image_path", columns)
            self.assertEqual(store._connection.execute("SELECT status FROM jobs WHERE id='old-job'").fetchone()[0], "completed")
            store.close()


if __name__ == "__main__":
    unittest.main()
