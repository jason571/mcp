from __future__ import annotations

import json
import unittest
from pathlib import Path
from unittest.mock import patch

from minimax_h3_mcp.backend import ComfyUIBackend
from minimax_h3_mcp.config import Settings
from minimax_h3_mcp.models import GenerationRequest, MODEL_ID


WORKFLOW = Path(__file__).resolve().parents[1] / "workflows" / "minimax_h3_8step_api.json"


def _settings(**overrides: object) -> Settings:
    project_root = Path(__file__).resolve().parents[1]
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
        data_dir=project_root / "var",
        output_dir=project_root / "var" / "outputs",
        asset_dir=project_root / "var" / "assets",
        database_path=project_root / "var" / "jobs.sqlite3",
        max_concurrent_jobs=1,
        progress_heartbeat_seconds=5.0,
        job_timeout_seconds=3600,
        default_duration_seconds=15.0,
        max_duration_seconds=60.0,
        max_reference_image_bytes=20 * 1024 * 1024,
        max_output_video_bytes=2 * 1024 * 1024 * 1024,
        comfyui_base_url="http://127.0.0.1:8189",
        comfyui_health_timeout_seconds=3.0,
        comfyui_manage_process=False,
        comfyui_python=Path("/usr/bin/python3"),
        comfyui_root=project_root,
        comfyui_startup_timeout_seconds=120,
        comfyui_models_dir=Path("/mnt/data/ComfyUI-aki-v3/ComfyUI/models"),
        comfyui_workflow_template=WORKFLOW,
        comfyui_api_key=None,
        comfyui_allow_interrupt=False,
        comfyui_unload_after_job=True,
        unet_name="minimax_h3_hybrid_fl2va_ref2va_b25-49-int8.safetensors",
        clip_name="qwen3vl_32b_minimax_h3_int8_convrot.safetensors",
        video_vae_name="minimax_h3_video_vae_fp16.safetensors",
        audio_vae_name="minimax_h3_audio_vae_fp32.safetensors",
        lora_name="minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors",
        lora_strength=1.0,
        steps=8,
        sampler_name="euler",
        scheduler_name="beta",
    )
    values.update(overrides)
    return Settings(**values)  # type: ignore[arg-type]


def _request(**overrides: object) -> GenerationRequest:
    values = dict(
        prompt="a forest",
        negative_prompt=None,
        soundscape=None,
        prompt_mode="simple",
        model_id=MODEL_ID,
        duration_seconds=15.0,
        seed=1,
        audio_enabled=True,
        aspect_ratio="16:9",
        resolution="512p",
        width=896,
        height=512,
    )
    values.update(overrides)
    return GenerationRequest(**values)  # type: ignore[arg-type]


class EightStepWorkflowTests(unittest.TestCase):
    def test_text_to_video_drops_load_image_nodes(self) -> None:
        template = json.loads(WORKFLOW.read_text(encoding="utf-8"))
        workflow = ComfyUIBackend(_settings()).render_workflow(
            template, _request(), first_name="", last_name=""
        )
        class_types = {
            node["class_type"]
            for node in workflow.values()
            if isinstance(node, dict)
        }
        self.assertNotIn("LoadImage", class_types)
        self.assertEqual(workflow["8"]["inputs"]["steps"], 8)
        self.assertEqual(workflow["7"]["inputs"]["sampler_name"], "euler")
        self.assertEqual(workflow["8"]["inputs"]["scheduler"], "beta")
        self.assertEqual(
            workflow["1"]["inputs"]["unet_name"],
            "minimax_h3_hybrid_fl2va_ref2va_b25-49-int8.safetensors",
        )
        self.assertEqual(
            workflow["18"]["inputs"]["lora_name"],
            "minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors",
        )
        self.assertNotIn("first_frame", workflow["5"]["inputs"])
        self.assertNotIn("last_frame", workflow["5"]["inputs"])

    def test_image_to_video_keeps_first_and_last_frames(self) -> None:
        template = json.loads(WORKFLOW.read_text(encoding="utf-8"))
        workflow = ComfyUIBackend(_settings()).render_workflow(
            template,
            _request(first_frame_path="/tmp/first.png", last_frame_path="/tmp/last.png"),
            first_name="minimax_h3_mcp/first.png",
            last_name="minimax_h3_mcp/last.png",
        )
        self.assertEqual(workflow["15"]["inputs"]["image"], "minimax_h3_mcp/first.png")
        self.assertEqual(workflow["16"]["inputs"]["image"], "minimax_h3_mcp/last.png")
        self.assertEqual(workflow["5"]["inputs"]["first_frame"], ["15", 0])
        self.assertEqual(workflow["5"]["inputs"]["last_frame"], ["16", 0])

    def test_default_steps_are_eight(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            settings = Settings.from_env()
        self.assertEqual(settings.steps, 8)
        self.assertEqual(settings.sampler_name, "euler")
        self.assertTrue(settings.unet_name.endswith("b25-49-int8.safetensors"))

    def test_audio_disabled_prunes_audio_decode_and_video_audio_input(self) -> None:
        template = json.loads(WORKFLOW.read_text(encoding="utf-8"))
        workflow = ComfyUIBackend(_settings()).render_workflow(
            template, _request(audio_enabled=False), first_name="", last_name=""
        )
        class_types = {
            node["class_type"]
            for node in workflow.values()
            if isinstance(node, dict)
        }
        self.assertNotIn("VAEDecodeAudio", class_types)
        self.assertNotIn("audio", workflow["13"]["inputs"])
        self.assertNotIn("4", workflow)

    def test_first_frame_only_does_not_leave_last_frame_link(self) -> None:
        template = json.loads(WORKFLOW.read_text(encoding="utf-8"))
        workflow = ComfyUIBackend(_settings()).render_workflow(
            template,
            _request(first_frame_path="/tmp/first.png"),
            first_name="minimax_h3_mcp/first.png",
            last_name="",
        )
        self.assertEqual(workflow["5"]["inputs"]["first_frame"], ["15", 0])
        self.assertNotIn("last_frame", workflow["5"]["inputs"])

    def test_reference_image_is_not_mistaken_for_first_frame(self) -> None:
        template = json.loads(WORKFLOW.read_text(encoding="utf-8"))
        workflow = ComfyUIBackend(_settings()).render_workflow(
            template,
            _request(reference_image_path="/tmp/reference.png"),
            first_name="", last_name="",
            reference_name="minimax_h3_mcp/reference.png",
        )
        self.assertEqual(workflow["5"]["class_type"], "MiniMaxH3ReferenceToVideo")
        self.assertEqual(workflow["5"]["inputs"]["ref_images.ref_image_0"], ["19", 0])
        self.assertEqual(workflow["5"]["inputs"]["ref_image_size"], "match")
        self.assertNotIn("15", workflow)
        self.assertNotIn("20", workflow)
        self.assertEqual(workflow["9"]["inputs"]["conditioning"], ["5", 0])

    def test_reference_and_explicit_frame_guide_can_coexist(self) -> None:
        template = json.loads(WORKFLOW.read_text(encoding="utf-8"))
        workflow = ComfyUIBackend(_settings()).render_workflow(
            template,
            _request(reference_image_path="/tmp/reference.png", first_frame_path="/tmp/first.png"),
            first_name="minimax_h3_mcp/first.png", last_name="",
            reference_name="minimax_h3_mcp/reference.png",
        )
        self.assertEqual(workflow["20"]["class_type"], "MiniMaxH3AddGuide")
        self.assertEqual(workflow["20"]["inputs"]["image"], ["15", 0])
        self.assertEqual(workflow["20"]["inputs"]["frame_idx"], 0)
        self.assertEqual(workflow["9"]["inputs"]["conditioning"], ["20", 0])


if __name__ == "__main__":
    unittest.main()
