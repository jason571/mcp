from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from fastvideo_h3_mcp.h3_fastvideo_base_runner import (
    _LORA_TARGET_MODULES,
    _load_request,
    _scheduler_grid_points,
)


class FastVideoBaseRunnerTests(unittest.TestCase):
    def test_four_nfe_uses_five_scheduler_grid_points(self) -> None:
        self.assertEqual(_scheduler_grid_points(4), 5)

    def test_eight_nfe_uses_nine_scheduler_grid_points(self) -> None:
        self.assertEqual(_scheduler_grid_points(8), 9)

    def test_lora_targets_only_trained_attention_and_ffn_layers(self) -> None:
        self.assertEqual(
            set(_LORA_TARGET_MODULES),
            {
                ".attn.to_q",
                ".attn.to_k",
                ".attn.to_v",
                ".attn.to_out",
                ".ff.fc_in",
                ".ff.fc_out",
            },
        )
        self.assertFalse(any("time_embedder" in name for name in _LORA_TARGET_MODULES))

    def test_accepts_last_frame_only(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            request_file = Path(directory) / "request.json"
            request_file.write_text(
                json.dumps(
                    {
                        "model_path": "/models/base",
                        "lora_path": "/models/lora/4step.safetensors",
                        "lora_alpha": 8,
                        "lora_strength": 1.0,
                        "prompt": "test",
                        "first_frame_path": None,
                        "last_frame_path": "/tmp/last.jpg",
                        "width": 896,
                        "height": 512,
                        "num_frames": 362,
                        "fps": 24,
                        "steps": 4,
                        "seed": 1,
                        "num_gpus": 1,
                        "output_path": "/tmp/result.mp4",
                    }
                ),
                encoding="utf-8",
            )
            request = _load_request(request_file)

        self.assertIsNone(request["first_frame_path"])
        self.assertEqual(request["last_frame_path"], "/tmp/last.jpg")

    def test_rejects_request_without_reference_frame(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            request_file = Path(directory) / "request.json"
            request_file.write_text(
                json.dumps(
                    {
                        "model_path": "/models/base",
                        "lora_path": "/models/lora/4step.safetensors",
                        "lora_alpha": 8,
                        "lora_strength": 1.0,
                        "prompt": "test",
                        "width": 896,
                        "height": 512,
                        "num_frames": 362,
                        "fps": 24,
                        "steps": 4,
                        "seed": 1,
                        "num_gpus": 1,
                        "output_path": "/tmp/result.mp4",
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "requires"):
                _load_request(request_file)


if __name__ == "__main__":
    unittest.main()
