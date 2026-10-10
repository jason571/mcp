from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from minimax_h3_mcp.config import Settings, generation_duration_from_env


class SettingsTests(unittest.TestCase):
    def test_generation_duration_defaults_are_15_and_60(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            settings = Settings.from_env()
        self.assertEqual(settings.default_duration_seconds, 15.0)
        self.assertEqual(settings.max_duration_seconds, 60.0)
        self.assertEqual(settings.port, 8766)

    def test_generation_duration_limits_are_read_from_env(self) -> None:
        with patch.dict(
            os.environ,
            {
                "H3_MCP_DEFAULT_DURATION_SECONDS": "15",
                "H3_MCP_MAX_DURATION_SECONDS": "30",
            },
            clear=True,
        ):
            settings = Settings.from_env()
        self.assertEqual(settings.default_duration_seconds, 15.0)
        self.assertEqual(settings.max_duration_seconds, 30.0)

    def test_invalid_generation_duration_is_rejected(self) -> None:
        invalid_values = (
            {"H3_MCP_MAX_DURATION_SECONDS": "61"},
            {"H3_MCP_MAX_DURATION_SECONDS": "4"},
            {"H3_MCP_DEFAULT_DURATION_SECONDS": "4"},
            {
                "H3_MCP_DEFAULT_DURATION_SECONDS": "45",
                "H3_MCP_MAX_DURATION_SECONDS": "30",
            },
        )
        for environment in invalid_values:
            with self.subTest(environment=environment):
                with patch.dict(os.environ, environment, clear=True):
                    with self.assertRaises(ValueError):
                        Settings.from_env()

    def test_generation_duration_from_env_returns_numeric_default(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            default, maximum = generation_duration_from_env()
        self.assertIsInstance(default, float)
        self.assertEqual(default, 15.0)
        self.assertEqual(maximum, 60.0)

    def test_invalid_transport_configuration_is_rejected(self) -> None:
        invalid_values = (
            {"H3_MCP_PORT": "65536"},
            {"H3_MCP_PATH": "mcp"},
            {"H3_MCP_PUBLIC_BASE_URL": "not-a-url"},
            {"COMFYUI_BASE_URL": "ws://127.0.0.1:8189"},
            {"COMFYUI_HEALTH_TIMEOUT_SECONDS": "0"},
        )
        for environment in invalid_values:
            with self.subTest(environment=environment):
                with patch.dict(os.environ, environment, clear=True):
                    with self.assertRaises(ValueError):
                        Settings.from_env()


if __name__ == "__main__":
    unittest.main()
