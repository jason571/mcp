from __future__ import annotations

import unittest

from fastvideo_h3_mcp.models import (
    GenerationRequest,
    aligned_num_frames,
    build_h3_prompt,
    canvas_dimensions,
    duration_resolution_limits,
    duration_seconds_contract,
    normalize_resolution,
)


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
            model_id="auto",
            duration_seconds=5,
            seed=1,
            audio_enabled=True,
            aspect_ratio="16:9",
            resolution="768p",
            first_frame_path="/tmp/first.png",
        )
        self.assertTrue(request.has_keyframes)


if __name__ == "__main__":
    unittest.main()
