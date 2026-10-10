from __future__ import annotations

import asyncio
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from fastvideo_h3_mcp.backend import (
    BackendRouter,
    ComfyUIBackend,
    FastVideoBaseFL2VABackend,
    FastVideoCliBackend,
    ResourceAdmission,
    ResourceSnapshot,
    _PersistentWorker,
    _Stage,
    render_workflow_template,
)
from fastvideo_h3_mcp.models import GenerationRequest
from fastvideo_h3_mcp.sage_attention import SageAttentionProbe


def make_request(**changes: object) -> GenerationRequest:
    values: dict[str, object] = {
        "prompt": "A fox running through snow.",
        "negative_prompt": "No watermark.",
        "soundscape": "Wind and footsteps.",
        "prompt_mode": "simple",
        "model_id": "base",
        "duration_seconds": 5,
        "seed": 123,
        "audio_enabled": True,
        "aspect_ratio": "16:9",
        "resolution": "768p",
        "width": 1344,
        "height": 768,
    }
    values.update(changes)
    return GenerationRequest(**values)  # type: ignore[arg-type]


class WorkflowTemplateTests(unittest.TestCase):
    def test_exact_placeholder_keeps_json_type(self) -> None:
        template = {
            "node": {
                "inputs": {
                    "width": "{{WIDTH}}",
                    "enabled": "{{AUDIO_ENABLED}}",
                    "label": "size={{WIDTH}}",
                }
            }
        }
        rendered = render_workflow_template(
            template, {"{{WIDTH}}": 1344, "{{AUDIO_ENABLED}}": False}
        )
        self.assertEqual(rendered["node"]["inputs"]["width"], 1344)  # type: ignore[index]
        self.assertIs(rendered["node"]["inputs"]["enabled"], False)  # type: ignore[index]
        self.assertEqual(rendered["node"]["inputs"]["label"], "size=1344")  # type: ignore[index]

    def test_comfy_render_and_output_discovery(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            common = root / "workflow.json"
            common.write_text("{}", encoding="utf-8")
            settings = SimpleNamespace(
                comfyui_models_dir=root,
                comfyui_base_url="http://127.0.0.1:8189",
                comfyui_t2va_workflow_template=common,
                comfyui_fl2va_workflow_template=common,
                comfyui_first_frame_workflow_template=common,
                comfyui_last_frame_workflow_template=common,
                comfyui_first_last_workflow_template=common,
                comfyui_api_key=None,
                comfyui_allow_interrupt=False,
                minimax_h3_legacy_steps=20,
                progress_heartbeat_seconds=1,
                ffmpeg_binary="ffmpeg",
            )
            backend = ComfyUIBackend(settings)  # type: ignore[arg-type]
            request = make_request(
                first_frame_path="/tmp/first.png",
                last_frame_path="/tmp/last.png",
            )
            template = {
                "1": {
                    "inputs": {
                        "text": "{{H3_PROMPT}}",
                        "width": "{{WIDTH}}",
                        "height": "{{HEIGHT}}",
                        "frames": "{{NUM_FRAMES}}",
                        "first": "{{FIRST_FRAME}}",
                        "last": "{{LAST_FRAME}}",
                    },
                    "class_type": "TestNode",
                }
            }
            rendered = backend._render_workflow(
                template,
                request,
                first_name="fastvideo_h3_mcp/first.png",
                last_name="fastvideo_h3_mcp/last.png",
            )
            inputs = rendered["1"]["inputs"]  # type: ignore[index]
            self.assertEqual(inputs["width"], 1344)
            self.assertEqual(inputs["frames"], request.num_frames)
            self.assertIn("negative_constraints:", inputs["text"])
            self.assertEqual(
                backend._select_workflow(request),
                settings.comfyui_first_last_workflow_template,
            )
            history = {
                "prompt": {
                    "outputs": {
                        "9": {
                            "gifs": [
                                {
                                    "filename": "generated.mp4",
                                    "subfolder": "h3",
                                    "type": "output",
                                }
                            ]
                        }
                    }
                }
            }
            self.assertEqual(
                backend._find_video_output(history, "prompt")["filename"],
                "generated.mp4",
            )

    def test_absent_keyframe_node_and_link_are_pruned(self) -> None:
        settings = SimpleNamespace(minimax_h3_legacy_steps=20)
        backend = ComfyUIBackend(settings)  # type: ignore[arg-type]
        template = {
            "5": {
                "class_type": "MiniMaxH3ImageToVideo",
                "inputs": {
                    "first_frame": ["15", 0],
                    "last_frame": ["16", 0],
                },
            },
            "15": {
                "class_type": "LoadImage",
                "inputs": {"image": "{{FIRST_FRAME}}"},
            },
            "16": {
                "class_type": "LoadImage",
                "inputs": {"image": "{{LAST_FRAME}}"},
            },
        }
        rendered = backend._render_workflow(
            template,
            make_request(first_frame_path="/tmp/first.png"),
            first_name="fastvideo_h3_mcp/first.png",
            last_name="",
        )
        self.assertIn("15", rendered)
        self.assertNotIn("16", rendered)
        self.assertIn("first_frame", rendered["5"]["inputs"])  # type: ignore[index]
        self.assertNotIn("last_frame", rendered["5"]["inputs"])  # type: ignore[index]

    def test_unresolved_placeholder_fails(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            common = root / "workflow.json"
            common.write_text("{}", encoding="utf-8")
            settings = SimpleNamespace(
                comfyui_models_dir=root,
                comfyui_base_url="http://127.0.0.1:8189",
                comfyui_t2va_workflow_template=common,
                comfyui_fl2va_workflow_template=common,
                comfyui_first_frame_workflow_template=common,
                comfyui_last_frame_workflow_template=common,
                comfyui_first_last_workflow_template=common,
                comfyui_api_key=None,
                comfyui_allow_interrupt=False,
                minimax_h3_legacy_steps=20,
                progress_heartbeat_seconds=1,
                ffmpeg_binary="ffmpeg",
            )
            backend = ComfyUIBackend(settings)  # type: ignore[arg-type]
            with self.assertRaisesRegex(RuntimeError, "unresolved"):
                backend._render_workflow(
                    {"1": {"inputs": {"x": "{{UNKNOWN}}"}}},
                    make_request(),
                    first_name="",
                    last_name="",
                )


class FastVideoCommandTests(unittest.TestCase):
    def test_progress_classifier_does_not_treat_loading_vae_as_decoding(self) -> None:
        backend = FastVideoCliBackend(SimpleNamespace())  # type: ignore[arg-type]
        loading = backend._classify_line(
            "Loading audio_vae using diffusers from /models/audio_vae", 10.0
        )
        self.assertIsNone(loading)

        pipeline = backend._classify_line(
            "Running pipeline stages: ['conditioning_stage', 'denoising_stage']",
            42.0,
        )
        self.assertIsNotNone(pipeline)
        assert pipeline is not None
        self.assertEqual(pipeline.progress, 43.0)
        self.assertEqual(pipeline.name, "preparing_generation")

        step_started = backend._classify_line(
            "MCP_PROGRESS h3_joint_denoise step=2/4 phase=started", 58.0
        )
        self.assertIsNotNone(step_started)
        assert step_started is not None
        self.assertEqual(step_started.progress, 58.0)
        self.assertEqual(step_started.name, "joint_av_denoising")
        self.assertIn("2/4", step_started.message)

        block_completed = backend._classify_line(
            "MCP_PROGRESS h3_transformer_block step=2/4 block=25/50 phase=completed",
            58.0,
        )
        self.assertIsNotNone(block_completed)
        assert block_completed is not None
        self.assertEqual(block_completed.progress, 62.0)
        self.assertEqual(block_completed.name, "joint_av_denoising")
        self.assertIn("25/50", block_completed.message)

        step_completed = backend._classify_line(
            "MCP_PROGRESS h3_joint_denoise step=2/4 phase=completed", 58.0
        )
        self.assertIsNotNone(step_completed)
        assert step_completed is not None
        self.assertEqual(step_completed.progress, 66.0)

        audio_decode = backend._classify_line(
            "MCP_PROGRESS pipeline_stage=audio_decoding_stage phase=started",
            87.0,
        )
        self.assertIsNotNone(audio_decode)
        assert audio_decode is not None
        self.assertEqual(audio_decode.progress, 88.0)
        self.assertEqual(audio_decode.name, "audio_decoding")

        completed_sampling = backend._classify_line(
            "Generated successfully in 66.22 seconds", 55.0
        )
        self.assertIsNotNone(completed_sampling)
        assert completed_sampling is not None
        self.assertEqual(completed_sampling.progress, 92.0)
        self.assertEqual(completed_sampling.name, "postprocessing")

        after_mux = backend._classify_line(
            "Output written to: /tmp/result.mp4", 98.0
        )
        self.assertIsNotNone(after_mux)
        assert after_mux is not None
        self.assertEqual(after_mux.progress, 98.0)
        self.assertEqual(after_mux.name, "muxing")

    def test_command_matches_dense_fasth3_cli(self) -> None:
        settings = SimpleNamespace(
            fastvideo_python=Path("/home/clhh/FastVideo/.venv/bin/python"),
            fastvideo_script=Path(
                "/home/clhh/FastVideo/examples/inference/basic/"
                "basic_minimax_h3_t2v.py"
            ),
            model_path="/models/fasth3/snapshots/master",
            num_gpus=1,
            fastvideo_attention_backend="TORCH_SDPA",
            fastvideo_vsa_sparsity=0.9,
            fastvideo_vsa_tile_size=64,
            fastvideo_torch_compile=False,
            fastvideo_text_encoder_fp8=False,
            extra_args=(),
        )
        backend = FastVideoCliBackend(settings)  # type: ignore[arg-type]
        command = backend.build_command(make_request(model_id="fast"), Path("/tmp/job"))
        self.assertIn("basic_minimax_h3_t2v.py", command[1])
        self.assertIn("--steps", command)
        self.assertEqual(command[command.index("--steps") + 1], "5")
        self.assertNotIn("--profile", command)
        self.assertNotIn("--no-warmup", command)
        self.assertNotIn("--fa4", command)
        self.assertFalse(
            backend.worker_payload(
                make_request(model_id="fast", duration_seconds=30),
                Path("/tmp/short.mp4"),
            )["release_dit_after_denoise"]
        )
        self.assertTrue(
            backend.worker_payload(
                make_request(model_id="fast", duration_seconds=60),
                Path("/tmp/long.mp4"),
            )["release_dit_after_denoise"]
        )

    def test_router_uses_fast_for_every_text_only_canvas(self) -> None:
        fast = FastVideoCliBackend(
            SimpleNamespace(model_id="fasth3-checkpoint")  # type: ignore[arg-type]
        )
        base = SimpleNamespace(name="base")
        router = BackendRouter(fast, base)  # type: ignore[arg-type]

        for request in (
            make_request(model_id="auto", resolution="512p", width=896, height=512),
            make_request(
                model_id="base",
                aspect_ratio="9:16",
                resolution="640p",
                width=640,
                height=1152,
            ),
        ):
            self.assertIs(router._select(request), fast)

    def test_router_reserves_base_for_reference_frames(self) -> None:
        fast = FastVideoCliBackend(
            SimpleNamespace(model_id="fasth3-checkpoint")  # type: ignore[arg-type]
        )
        base = SimpleNamespace(name="base")
        router = BackendRouter(fast, base)  # type: ignore[arg-type]
        request = make_request(
            model_id="fast",
            first_frame_path="/tmp/first.png",
        )
        self.assertIs(router._select(request), base)

    def test_command_enables_text_encoder_fp8_compatibility_flag(self) -> None:
        settings = SimpleNamespace(
            fastvideo_python=Path("/opt/fv/.venv/bin/python"),
            fastvideo_script=Path("/opt/fv/basic_minimax_h3_t2v.py"),
            model_path="/models/fasth3",
            num_gpus=1,
            fastvideo_attention_backend="TORCH_SDPA",
            fastvideo_vsa_sparsity=0.9,
            fastvideo_vsa_tile_size=64,
            fastvideo_torch_compile=False,
            fastvideo_text_encoder_fp8=True,
            extra_args=(),
        )
        command = FastVideoCliBackend(settings).build_command(  # type: ignore[arg-type]
            make_request(model_id="fast"), Path("/tmp/job")
        )
        self.assertIn("--text-encoder-fp8", command)

    def test_command_and_environment_enable_pro6000_vsa_profile(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fastvideo_repo = Path(directory) / "FastVideo"
            (fastvideo_repo / "fastvideo-kernel" / "python" / "fastvideo_kernel").mkdir(
                parents=True
            )
            settings = SimpleNamespace(
                fastvideo_repo=fastvideo_repo,
                fastvideo_python=fastvideo_repo / ".venv/bin/python",
                fastvideo_script=fastvideo_repo / "examples/inference/basic/basic_minimax_h3_t2v.py",
                model_path="/models/fasth3-vsa",
                num_gpus=1,
                fastvideo_attention_backend="VIDEO_SPARSE_ATTN_H3",
                fastvideo_vsa_sparsity=0.9,
                fastvideo_vsa_tile_size=64,
                fastvideo_vsa_kernel="triton",
                fastvideo_vsa_cache_tile_buffer=False,
                fastvideo_vsa_attention_residual_chunk_tokens=32768,
                fastvideo_torch_compile=False,
                fastvideo_text_encoder_fp8=True,
                fastvideo_direct_device_load=True,
                enable_fa4=False,
                extra_args=(),
            )
            backend = FastVideoCliBackend(settings)  # type: ignore[arg-type]
            command = backend.build_command(
                make_request(model_id="fast"), Path("/tmp/job")
            )
            environment = backend.build_environment()

        self.assertEqual(command[command.index("--vsa-sparsity") + 1], "0.9")
        self.assertEqual(command[command.index("--vsa-tile-size") + 1], "64")
        self.assertEqual(environment["FASTVIDEO_ATTENTION_BACKEND"], "VIDEO_SPARSE_ATTN_H3")
        self.assertEqual(environment["FASTVIDEO_VSA_SM100A"], "0")
        self.assertEqual(environment["FASTVIDEO_VSA_CUTEDSL"], "0")
        self.assertEqual(environment["FASTVIDEO_VSA_TRITON"], "1")
        self.assertEqual(environment["FASTVIDEO_VSA_CACHE_TILE_BUFFER"], "0")
        self.assertEqual(
            environment["FASTVIDEO_VSA_ATTENTION_RESIDUAL_CHUNK_TOKENS"],
            "32768",
        )
        self.assertEqual(environment["FASTVIDEO_FA4"], "0")
        self.assertTrue(
            environment["PYTHONPATH"].startswith(
                str(fastvideo_repo / "fastvideo-kernel" / "python")
            )
        )

    def test_dense_environment_forces_vsa_memory_options_off(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fastvideo_repo = Path(directory) / "FastVideo"
            fastvideo_repo.mkdir()
            settings = SimpleNamespace(
                fastvideo_repo=fastvideo_repo,
                fastvideo_attention_backend="TORCH_SDPA",
                fastvideo_vsa_kernel="triton",
                fastvideo_vsa_cache_tile_buffer=True,
                fastvideo_vsa_attention_residual_chunk_tokens=32768,
                enable_fa4=False,
                fastvideo_direct_device_load=True,
            )
            environment = FastVideoCliBackend(  # type: ignore[arg-type]
                settings
            ).build_environment()

        self.assertEqual(environment["FASTVIDEO_VSA_TRITON"], "0")
        self.assertEqual(environment["FASTVIDEO_VSA_CACHE_TILE_BUFFER"], "0")
        self.assertEqual(
            environment["FASTVIDEO_VSA_ATTENTION_RESIDUAL_CHUNK_TOKENS"], "0"
        )
        self.assertEqual(environment["FASTVIDEO_ATTENTION_BACKEND"], "TORCH_SDPA")

    def test_sage_auto_does_not_rewrite_vsa_backend(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fastvideo_repo = Path(directory) / "FastVideo"
            fastvideo_repo.mkdir()
            settings = SimpleNamespace(
                fastvideo_repo=fastvideo_repo,
                fastvideo_python=fastvideo_repo / "missing-python",
                fastvideo_attention_backend="VIDEO_SPARSE_ATTN_H3",
                fastvideo_sage_attention="auto",
                fastvideo_vsa_kernel="triton",
                fastvideo_vsa_cache_tile_buffer=False,
                fastvideo_vsa_attention_residual_chunk_tokens=32768,
                enable_fa4=False,
                fastvideo_direct_device_load=True,
            )
            with patch(
                "fastvideo_h3_mcp.backend.probe_sage_attention",
                return_value=SageAttentionProbe(
                    sage_attn_available=True,
                    sage_attn_three_available=True,
                    h3_supported_backends=("SAGE_ATTN", "SAGE_ATTN_THREE"),
                    detail="unused",
                ),
            ):
                environment = FastVideoCliBackend(  # type: ignore[arg-type]
                    settings
                ).build_environment()

        self.assertEqual(
            environment["FASTVIDEO_ATTENTION_BACKEND"], "VIDEO_SPARSE_ATTN_H3"
        )
        self.assertEqual(environment["FASTVIDEO_VSA_TRITON"], "1")

    def test_sage_auto_selects_sage_attn_for_dense_fasth3(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fastvideo_repo = Path(directory) / "FastVideo"
            python = fastvideo_repo / "python"
            fastvideo_repo.mkdir()
            python.write_text("", encoding="utf-8")
            settings = SimpleNamespace(
                fastvideo_repo=fastvideo_repo,
                fastvideo_python=python,
                fastvideo_attention_backend="TORCH_SDPA",
                fastvideo_sage_attention="auto",
                fastvideo_vsa_kernel="triton",
                fastvideo_vsa_cache_tile_buffer=True,
                fastvideo_vsa_attention_residual_chunk_tokens=32768,
                enable_fa4=False,
                fastvideo_direct_device_load=True,
            )
            with patch(
                "fastvideo_h3_mcp.backend.probe_sage_attention",
                return_value=SageAttentionProbe(
                    sage_attn_available=True,
                    sage_attn_three_available=False,
                    h3_supported_backends=("TORCH_SDPA", "FLASH_ATTN", "SAGE_ATTN"),
                    detail="ok",
                ),
            ):
                environment = FastVideoCliBackend(  # type: ignore[arg-type]
                    settings
                ).build_environment()

        self.assertEqual(environment["FASTVIDEO_ATTENTION_BACKEND"], "SAGE_ATTN")
        self.assertEqual(environment["FASTVIDEO_VSA_TRITON"], "0")
        self.assertEqual(
            environment["FASTVIDEO_VSA_ATTENTION_RESIDUAL_CHUNK_TOKENS"], "0"
        )

    def test_sage_auto_keeps_torch_sdpa_when_h3_does_not_list_sage(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fastvideo_repo = Path(directory) / "FastVideo"
            python = fastvideo_repo / "python"
            fastvideo_repo.mkdir()
            python.write_text("", encoding="utf-8")
            settings = SimpleNamespace(
                fastvideo_repo=fastvideo_repo,
                fastvideo_python=python,
                fastvideo_attention_backend="TORCH_SDPA",
                fastvideo_sage_attention="auto",
                fastvideo_vsa_kernel="triton",
                fastvideo_vsa_cache_tile_buffer=False,
                fastvideo_vsa_attention_residual_chunk_tokens=32768,
                enable_fa4=False,
                fastvideo_direct_device_load=True,
            )
            with patch(
                "fastvideo_h3_mcp.backend.probe_sage_attention",
                return_value=SageAttentionProbe(
                    sage_attn_available=True,
                    sage_attn_three_available=True,
                    h3_supported_backends=(
                        "TORCH_SDPA",
                        "FLASH_ATTN",
                        "VIDEO_SPARSE_ATTN_H3",
                    ),
                    detail="ok",
                ),
            ):
                environment = FastVideoCliBackend(  # type: ignore[arg-type]
                    settings
                ).build_environment()

        self.assertEqual(environment["FASTVIDEO_ATTENTION_BACKEND"], "TORCH_SDPA")

    def test_base_fastvideo_environment_is_dense_and_independent_of_comfyui(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fastvideo_repo = Path(directory) / "FastVideo"
            fastvideo_repo.mkdir()
            settings = SimpleNamespace(
                fastvideo_repo=fastvideo_repo,
                minimax_h3_model_path="/models/MiniMax-H3-r16",
                minimax_h3_lora_path=(
                    "/models/MiniMax-H3-Turbo/"
                    "minimax_h3_fl2v_turbo_4step_v0.1.safetensors"
                ),
                minimax_h3_lora_alpha=8,
                minimax_h3_lora_strength=1.0,
                minimax_h3_text_encoder_fp8=True,
                minimax_h3_direct_device_load=True,
                minimax_h3_release_dit_after_denoise=True,
                minimax_h3_ff_chunk_tokens=32768,
                minimax_h3_steps=4,
                num_gpus=1,
            )
            backend = FastVideoBaseFL2VABackend(settings)  # type: ignore[arg-type]
            environment = backend.build_environment()
            request = make_request(
                first_frame_path="/tmp/first.jpg",
                last_frame_path="/tmp/last.webp",
                resolution="512p",
                width=896,
                height=512,
            )
            payload = backend.request_payload(request, Path("/tmp/result.mp4"))
            long_payload = backend.request_payload(
                make_request(
                    duration_seconds=60,
                    first_frame_path="/tmp/first.jpg",
                    resolution="512p",
                    width=896,
                    height=512,
                ),
                Path("/tmp/long.mp4"),
            )

        self.assertEqual(environment["FASTVIDEO_ATTENTION_BACKEND"], "TORCH_SDPA")
        self.assertEqual(environment["FASTVIDEO_VSA_TRITON"], "0")
        self.assertEqual(environment["FASTVIDEO_VSA_CACHE_TILE_BUFFER"], "0")
        self.assertEqual(environment["FASTVIDEO_LOAD_WEIGHTS_DIRECT_TO_DEVICE"], "1")
        self.assertEqual(
            environment["FASTVIDEO_MINIMAX_H3_RELEASE_DIT_AFTER_DENOISE"], "1"
        )
        self.assertEqual(environment["FASTVIDEO_MINIMAX_H3_FF_CHUNK_TOKENS"], "32768")
        self.assertEqual(environment["FASTVIDEO_LORA_CPU_BACKUP"], "0")
        self.assertEqual(environment["FASTVIDEO_REQUIRE_COMPLETE_LORA"], "1")
        self.assertEqual(payload["first_frame_path"], "/tmp/first.jpg")
        self.assertEqual(payload["last_frame_path"], "/tmp/last.webp")
        self.assertEqual(payload["width"], 896)
        self.assertEqual(payload["height"], 512)
        self.assertEqual(
            payload["lora_path"],
            "/models/MiniMax-H3-Turbo/"
            "minimax_h3_fl2v_turbo_4step_v0.1.safetensors",
        )
        self.assertEqual(payload["steps"], 4)
        self.assertEqual(payload["lora_alpha"], 8)
        self.assertEqual(payload["lora_strength"], 1.0)
        self.assertFalse(payload["release_dit_after_denoise"])
        self.assertTrue(long_payload["release_dit_after_denoise"])


class ResourceAdmissionTests(unittest.IsolatedAsyncioTestCase):
    async def test_waits_until_two_consecutive_safe_samples(self) -> None:
        snapshots = [
            ResourceSnapshot(gpu_used_mib=50_000, system_available_mib=10_000),
            ResourceSnapshot(gpu_used_mib=2_000, system_available_mib=50_000),
            ResourceSnapshot(gpu_used_mib=1_500, system_available_mib=51_000),
        ]

        async def probe() -> ResourceSnapshot:
            return snapshots.pop(0)

        settings = SimpleNamespace(
            resource_admission_enabled=True,
            resource_admission_max_gpu_used_mib=4096,
            resource_admission_min_system_available_mib=45000,
            resource_admission_ready_samples=2,
            resource_admission_poll_seconds=0,
        )
        admission = ResourceAdmission(settings, probe=probe)  # type: ignore[arg-type]
        reports: list[tuple[float, str, str]] = []

        async def report(progress: float, stage: str, message: str) -> None:
            reports.append((progress, stage, message))

        await admission.wait(report, asyncio.Event())

        self.assertEqual(len(snapshots), 0)
        self.assertEqual(reports[0][1], "waiting_for_resources")
        self.assertEqual(reports[-1][1], "checking_resources")
        self.assertIn("2/2", reports[-1][2])

    def test_resource_probe_parsers_reject_unknown_values(self) -> None:
        self.assertEqual(ResourceAdmission._parse_gpu_memory("1234\n"), [1234])
        self.assertEqual(
            ResourceAdmission._parse_gpu_process_memory("100, 2048\n200, 1024\n"),
            [(100, 2048), (200, 1024)],
        )
        with self.assertRaisesRegex(RuntimeError, "invalid GPU memory"):
            ResourceAdmission._parse_gpu_memory("N/A\n")

        with tempfile.TemporaryDirectory() as directory:
            meminfo = Path(directory) / "meminfo"
            meminfo.write_text(
                "MemTotal:       65536000 kB\nMemAvailable:   49152000 kB\n",
                encoding="utf-8",
            )
            self.assertEqual(
                ResourceAdmission._read_system_available_mib(meminfo), 48000
            )

    async def test_runtime_monitor_excludes_worker_group_and_rejects_comfyui(self) -> None:
        settings = SimpleNamespace(
            resource_admission_enabled=True,
            resource_admission_max_gpu_used_mib=4096,
        )
        admission = ResourceAdmission(settings)  # type: ignore[arg-type]
        admission._run_nvidia_smi = AsyncMock(  # type: ignore[method-assign]
            return_value="101, 60000\n202, 45000\n303, 900\n"
        )
        process_groups = {101: 500, 202: 202, 303: 303}

        with patch("os.getpgid", side_effect=lambda pid: process_groups[pid]):
            with self.assertRaisesRegex(RuntimeError, "external GPU workload"):
                await admission.assert_worker_exclusive(worker_pgid=500)


class PersistentWorkerTests(unittest.IsolatedAsyncioTestCase):
    class Admission:
        def __init__(self) -> None:
            self.wait_calls = 0
            self.exclusive_calls = 0

        async def wait(self, report, cancel_event) -> None:  # type: ignore[no-untyped-def]
            self.wait_calls += 1

        async def assert_worker_exclusive(self, worker_pgid: int) -> None:
            self.exclusive_calls += 1

    @staticmethod
    def _settings() -> SimpleNamespace:
        return SimpleNamespace(
            gpu_worker_idle_timeout_seconds=1,
            gpu_worker_warm_max_duration_seconds=30,
            resource_admission_poll_seconds=0.05,
            progress_heartbeat_seconds=0.05,
        )

    @staticmethod
    async def _report(progress: float, stage: str, message: str) -> None:
        return None

    async def test_reuses_one_process_then_releases_it_after_idle_timeout(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            script = root / "fake_worker.py"
            script.write_text(
                "import json, os, sys\n"
                "from pathlib import Path\n"
                "for raw in sys.stdin:\n"
                "    request = json.loads(raw)\n"
                "    Path(request['output_path']).write_text(str(os.getpid()))\n"
                "    result = {'request_id': request['request_id'], 'ok': True, "
                "'reusable': request.get('reusable', True)}\n"
                "    print('MCP_WORKER_RESULT ' + json.dumps(result), flush=True)\n"
                "    if not result['reusable']:\n"
                "        break\n",
                encoding="utf-8",
            )
            admission = self.Admission()
            worker = _PersistentWorker(  # type: ignore[arg-type]
                self._settings(), admission, "test"
            )
            pids: list[str] = []
            try:
                for index in range(2):
                    job_dir = root / f"job-{index}"
                    job_dir.mkdir()
                    output = job_dir / "result.mp4"
                    await worker.run(
                        command=[sys.executable, str(script)],
                        cwd=root,
                        environment={},
                        payload={"output_path": str(output), "reusable": True},
                        job_dir=job_dir,
                        report=self._report,
                        cancel_event=asyncio.Event(),
                        initial_stage=_Stage(10.0, "test", "test"),
                        classify_line=lambda _line, _progress: None,
                    )
                    pids.append(output.read_text(encoding="utf-8"))
                self.assertEqual(pids[0], pids[1])
                self.assertEqual(admission.wait_calls, 1)
                self.assertTrue(worker.is_warm)
                await asyncio.sleep(1.2)
                self.assertFalse(worker.is_warm)
            finally:
                await worker.shutdown()

    async def test_non_reusable_result_and_cancellation_discard_worker(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            script = root / "fake_worker.py"
            script.write_text(
                "import json, sys, time\n"
                "from pathlib import Path\n"
                "for raw in sys.stdin:\n"
                "    request = json.loads(raw)\n"
                "    if request.get('sleep'):\n"
                "        time.sleep(10)\n"
                "    Path(request['output_path']).write_bytes(b'video')\n"
                "    result = {'request_id': request['request_id'], 'ok': True, "
                "'reusable': request.get('reusable', False)}\n"
                "    print('MCP_WORKER_RESULT ' + json.dumps(result), flush=True)\n"
                "    if not result['reusable']:\n"
                "        break\n",
                encoding="utf-8",
            )
            admission = self.Admission()
            worker = _PersistentWorker(  # type: ignore[arg-type]
                self._settings(), admission, "test"
            )
            first_dir = root / "one-shot"
            first_dir.mkdir()
            await worker.run(
                command=[sys.executable, str(script)],
                cwd=root,
                environment={},
                payload={
                    "output_path": str(first_dir / "result.mp4"),
                    "reusable": False,
                },
                job_dir=first_dir,
                report=self._report,
                cancel_event=asyncio.Event(),
                initial_stage=_Stage(10.0, "test", "test"),
                classify_line=lambda _line, _progress: None,
            )
            self.assertFalse(worker.is_warm)

            second_dir = root / "cancel"
            second_dir.mkdir()
            cancel_event = asyncio.Event()
            task = asyncio.create_task(
                worker.run(
                    command=[sys.executable, str(script)],
                    cwd=root,
                    environment={},
                    payload={
                        "output_path": str(second_dir / "result.mp4"),
                        "sleep": True,
                    },
                    job_dir=second_dir,
                    report=self._report,
                    cancel_event=cancel_event,
                    initial_stage=_Stage(10.0, "test", "test"),
                    classify_line=lambda _line, _progress: None,
                )
            )
            await asyncio.sleep(0.1)
            cancel_event.set()
            with self.assertRaisesRegex(Exception, "job canceled"):
                await task
            self.assertFalse(worker.is_warm)


class BackendRouterWorkerLifecycleTests(unittest.IsolatedAsyncioTestCase):
    class StubBackend:
        def __init__(self, name: str, *, warm: bool = False) -> None:
            self.name = name
            self.settings = SimpleNamespace(model_id="fasth3-checkpoint")
            self.has_warm_worker = warm
            self.shutdown_calls = 0

        def health(self) -> dict[str, object]:
            return {"ready": True}

        async def shutdown(self) -> None:
            self.shutdown_calls += 1
            self.has_warm_worker = False

        async def generate(
            self, request, job_dir, report, cancel_event  # type: ignore[no-untyped-def]
        ) -> Path:
            output = job_dir / "result.mp4"
            output.write_bytes(b"video")
            return output

    async def test_switching_models_releases_old_worker_before_generation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fast = self.StubBackend("fastvideo-cli", warm=True)
            base = self.StubBackend("fastvideo-native-fl2va")
            router = BackendRouter(fast, base)  # type: ignore[arg-type]
            reports: list[str] = []

            async def report(progress: float, stage: str, message: str) -> None:
                reports.append(stage)

            request = make_request(
                model_id="base", first_frame_path="/tmp/first.png"
            )
            output = await router.generate(
                request, root, report, asyncio.Event()
            )

            self.assertTrue(output.is_file())
            self.assertEqual(fast.shutdown_calls, 1)
            self.assertIn("switching_model", reports)

if __name__ == "__main__":
    unittest.main()
