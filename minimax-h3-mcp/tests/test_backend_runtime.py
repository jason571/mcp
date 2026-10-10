from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

import aiohttp

from minimax_h3_mcp.backend import ComfyUIBackend
from minimax_h3_mcp.config import Settings


def _settings(root: Path, **overrides: object) -> Settings:
    models = root / "models"
    workflow = root / "workflow.json"
    for directory in ("diffusion_models", "loras", "text_encoders", "vae"):
        (models / directory).mkdir(parents=True, exist_ok=True)
    names = {
        "unet_name": "unet.safetensors",
        "lora_name": "lora.safetensors",
        "clip_name": "clip.safetensors",
        "video_vae_name": "video.safetensors",
        "audio_vae_name": "audio.safetensors",
    }
    for relative in (
        Path("diffusion_models") / names["unet_name"],
        Path("loras") / names["lora_name"],
        Path("text_encoders") / names["clip_name"],
        Path("vae") / names["video_vae_name"],
        Path("vae") / names["audio_vae_name"],
    ):
        (models / relative).write_bytes(b"fixture")
    workflow.write_text("{}", encoding="utf-8")
    values = dict(
        host="127.0.0.1",
        port=8766,
        mcp_path="/mcp",
        public_base_url="http://127.0.0.1:8766",
        allowed_hosts=("127.0.0.1:8766",),
        allowed_origins=(),
        api_token=None,
        artifact_signing_key=None,
        artifact_url_ttl_seconds=3600,
        data_dir=root / "var",
        output_dir=root / "var" / "outputs",
        asset_dir=root / "var" / "assets",
        database_path=root / "var" / "jobs.sqlite3",
        max_concurrent_jobs=1,
        progress_heartbeat_seconds=1.0,
        job_timeout_seconds=3600,
        default_duration_seconds=15.0,
        max_duration_seconds=60.0,
        max_reference_image_bytes=20 * 1024 * 1024,
        max_output_video_bytes=1024,
        comfyui_base_url="http://127.0.0.1:8189",
        comfyui_health_timeout_seconds=0.5,
        comfyui_manage_process=False,
        comfyui_python=Path("/usr/bin/python3"),
        comfyui_root=root,
        comfyui_startup_timeout_seconds=10,
        comfyui_models_dir=models,
        comfyui_workflow_template=workflow,
        comfyui_api_key=None,
        comfyui_allow_interrupt=False,
        comfyui_unload_after_job=True,
        lora_strength=1.0,
        steps=8,
        sampler_name="euler",
        scheduler_name="beta",
        **names,
    )
    values.update(overrides)
    return Settings(**values)  # type: ignore[arg-type]


class BackendHealthTests(unittest.IsolatedAsyncioTestCase):
    async def test_static_file_checks_are_not_reported_as_live_ready(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            backend = ComfyUIBackend(_settings(Path(directory)))
            health = backend.health()
        self.assertTrue(health["configured"])
        self.assertFalse(health["ready"])
        self.assertIsNone(health["reachable"])

    async def test_probe_requires_a_reachable_comfyui_queue_api(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            backend = ComfyUIBackend(_settings(Path(directory)))

            async def response(
                _session: object, _method: str, path: str, **_kwargs: object
            ) -> dict[str, object]:
                if path == "/queue":
                    return {
                        "queue_running": [[1]],
                        "queue_pending": [[2], [3]],
                    }
                node = path.rsplit("/", 1)[-1]
                return {node: {}}

            with patch.object(
                backend,
                "_json_request",
                AsyncMock(side_effect=response),
            ):
                health = await backend.probe_health()
        self.assertTrue(health["ready"])
        self.assertTrue(health["reachable"])
        self.assertEqual(health["queue_running"], 1)
        self.assertEqual(health["queue_pending"], 2)

    async def test_probe_reports_dependency_failure_without_exposing_details(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            backend = ComfyUIBackend(_settings(Path(directory)))
            with patch.object(
                backend,
                "_json_request",
                AsyncMock(side_effect=aiohttp.ClientConnectionError("secret detail")),
            ):
                health = await backend.probe_health()
        self.assertFalse(health["ready"])
        self.assertFalse(health["reachable"])
        self.assertNotIn("secret detail", str(health.get("error")))


class CancellationContractTests(unittest.IsolatedAsyncioTestCase):
    async def test_running_shared_workflow_is_not_claimed_canceled_without_interrupt(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            backend = ComfyUIBackend(_settings(Path(directory)))
            request = AsyncMock(return_value={})
            with patch.object(backend, "_json_request", request):
                stopped = await backend._cancel_prompt(object(), "prompt-1", True)  # type: ignore[arg-type]
        self.assertFalse(stopped)
        self.assertEqual(request.await_count, 1)

    async def test_running_workflow_is_stopped_when_interrupt_is_enabled(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            backend = ComfyUIBackend(
                _settings(Path(directory), comfyui_allow_interrupt=True)
            )
            request = AsyncMock(return_value={})
            with patch.object(backend, "_json_request", request):
                stopped = await backend._cancel_prompt(object(), "prompt-1", True)  # type: ignore[arg-type]
        self.assertTrue(stopped)
        self.assertEqual(request.await_count, 2)


if __name__ == "__main__":
    unittest.main()
