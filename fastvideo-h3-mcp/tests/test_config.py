from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastvideo_h3_mcp.config import Settings


class SettingsTests(unittest.TestCase):
    def test_vsa_profile_defaults_match_published_checkpoint(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            settings = Settings.from_env()

        self.assertEqual(settings.fastvideo_attention_backend, "VIDEO_SPARSE_ATTN_H3")
        self.assertEqual(settings.fastvideo_sage_attention, "auto")
        self.assertEqual(settings.fastvideo_vsa_sparsity, 0.9)
        self.assertEqual(settings.fastvideo_vsa_tile_size, 64)
        self.assertEqual(settings.fastvideo_vsa_kernel, "triton")
        self.assertFalse(settings.fastvideo_vsa_cache_tile_buffer)
        self.assertEqual(
            settings.fastvideo_vsa_attention_residual_chunk_tokens, 32768
        )
        self.assertFalse(settings.enable_fa4)
        self.assertIn("VSA-DataFree", settings.model_path)
        self.assertEqual(settings.base_backend, "fastvideo-native")
        self.assertEqual(
            settings.minimax_h3_python,
            Path("/home/clhh/FastVideo/.venv/bin/python"),
        )
        self.assertTrue(settings.minimax_h3_text_encoder_fp8)
        self.assertTrue(settings.minimax_h3_direct_device_load)
        self.assertTrue(settings.minimax_h3_release_dit_after_denoise)
        self.assertEqual(settings.minimax_h3_ff_chunk_tokens, 32768)
        self.assertEqual(settings.minimax_h3_steps, 8)
        self.assertEqual(settings.minimax_h3_legacy_steps, 20)
        self.assertEqual(settings.minimax_h3_lora_alpha, 8)
        self.assertEqual(settings.minimax_h3_lora_strength, 1.0)
        self.assertTrue(
            settings.minimax_h3_lora_path.endswith(
                "minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors"
            )
        )
        self.assertTrue(settings.resource_admission_enabled)
        self.assertEqual(settings.resource_admission_max_gpu_used_mib, 4096)
        self.assertEqual(
            settings.resource_admission_min_system_available_mib, 45000
        )
        self.assertEqual(settings.resource_admission_ready_samples, 2)
        self.assertEqual(settings.gpu_worker_idle_timeout_seconds, 60)
        self.assertEqual(settings.gpu_worker_warm_max_duration_seconds, 30)
        self.assertEqual(settings.default_duration_seconds, 15.0)
        self.assertEqual(settings.max_duration_seconds, 60.0)
        self.assertTrue(settings.fastvideo_release_dit_after_denoise)

    def test_invalid_vsa_profile_is_rejected(self) -> None:
        invalid_values = (
            {"FASTVIDEO_VSA_SPARSITY": "1.0"},
            {"FASTVIDEO_VSA_SPARSITY": "nan"},
            {"FASTVIDEO_VSA_TILE_SIZE": "128"},
            {"FASTVIDEO_VSA_KERNEL": "sm120"},
            {"FASTVIDEO_SAGE_ATTENTION": "maybe"},
            {"FASTVIDEO_VSA_CACHE_TILE_BUFFER": "sometimes"},
            {"FASTVIDEO_VSA_ATTENTION_RESIDUAL_CHUNK_TOKENS": "-1"},
            {"FASTVIDEO_RESOURCE_ADMISSION_POLL_SECONDS": "0"},
            {"FASTVIDEO_RESOURCE_ADMISSION_READY_SAMPLES": "0"},
            {"MINIMAX_H3_FF_CHUNK_TOKENS": "-1"},
            {"MINIMAX_H3_LORA_ALPHA": "0"},
            {"MINIMAX_H3_LORA_STRENGTH": "0"},
            {"MINIMAX_H3_BASE_BACKEND": "unknown"},
            {"FASTVIDEO_MCP_GPU_IDLE_TIMEOUT_SECONDS": "0"},
            {"FASTVIDEO_MCP_WARM_MAX_DURATION_SECONDS": "4"},
            {"FASTVIDEO_MCP_MAX_DURATION_SECONDS": "61"},
            {"FASTVIDEO_MCP_MAX_DURATION_SECONDS": "4"},
            {"FASTVIDEO_MCP_DEFAULT_DURATION_SECONDS": "4"},
            {
                "FASTVIDEO_MCP_DEFAULT_DURATION_SECONDS": "45",
                "FASTVIDEO_MCP_MAX_DURATION_SECONDS": "30",
            },
        )
        for environment in invalid_values:
            with self.subTest(environment=environment):
                with patch.dict(os.environ, environment, clear=True):
                    with self.assertRaises(ValueError):
                        Settings.from_env()

    def test_python_venv_entrypoint_symlink_is_preserved(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            target = root / "python-real"
            target.touch()
            entrypoint = root / "venv-python"
            entrypoint.symlink_to(target)

            with patch.dict(
                os.environ,
                {
                    "FASTVIDEO_PYTHON": str(entrypoint),
                    "MINIMAX_H3_PYTHON": str(entrypoint),
                },
                clear=True,
            ):
                settings = Settings.from_env()

            self.assertEqual(settings.fastvideo_python, entrypoint.absolute())
            self.assertEqual(settings.minimax_h3_python, entrypoint.absolute())
            self.assertTrue(settings.fastvideo_python.is_symlink())

    def test_generation_duration_limits_are_read_from_env(self) -> None:
        with patch.dict(
            os.environ,
            {
                "FASTVIDEO_MCP_DEFAULT_DURATION_SECONDS": "15",
                "FASTVIDEO_MCP_MAX_DURATION_SECONDS": "30",
            },
            clear=True,
        ):
            settings = Settings.from_env()

        self.assertEqual(settings.default_duration_seconds, 15.0)
        self.assertEqual(settings.max_duration_seconds, 30.0)
        self.assertEqual(settings.gpu_worker_warm_max_duration_seconds, 30)

    def test_generation_duration_from_env_returns_numeric_default(self) -> None:
        from fastvideo_h3_mcp.config import generation_duration_from_env

        with patch.dict(os.environ, {}, clear=True):
            default, maximum = generation_duration_from_env()
        self.assertIsInstance(default, float)
        self.assertEqual(default, 15.0)
        self.assertEqual(maximum, 60.0)


if __name__ == "__main__":
    unittest.main()
