from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path

from fastvideo_h3_mcp.models import (
    DEFAULT_DURATION_SECONDS,
    MAX_DURATION_SECONDS,
    MIN_DURATION_SECONDS,
)


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean")


def _env_int(name: str, default: int, minimum: int = 0) -> int:
    raw = os.getenv(name)
    value = default if raw is None else int(raw)
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return value


def _env_float(
    name: str,
    default: float,
    minimum: float | None = None,
    maximum: float | None = None,
    maximum_exclusive: float | None = None,
) -> float:
    raw = os.getenv(name)
    value = default if raw is None else float(raw)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    if maximum is not None and value > maximum:
        raise ValueError(f"{name} must be <= {maximum}")
    if maximum_exclusive is not None and value >= maximum_exclusive:
        raise ValueError(f"{name} must be < {maximum_exclusive}")
    return value


def _env_csv(name: str, default: tuple[str, ...] = ()) -> tuple[str, ...]:
    raw = os.getenv(name)
    if raw is None:
        return default
    return tuple(item.strip() for item in raw.split(",") if item.strip())


def _env_json_args(name: str) -> tuple[str, ...]:
    raw = os.getenv(name, "[]")
    value = json.loads(raw)
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"{name} must be a JSON array of strings")
    if any("\x00" in item for item in value):
        raise ValueError(f"{name} must not contain NUL bytes")
    return tuple(value)


def generation_duration_from_env() -> tuple[float, float]:
    """Return ``(default, maximum)`` clip seconds read at process start."""
    max_duration_seconds = _env_float(
        "FASTVIDEO_MCP_MAX_DURATION_SECONDS",
        MAX_DURATION_SECONDS,
        minimum=MIN_DURATION_SECONDS,
        maximum=MAX_DURATION_SECONDS,
    )
    default_duration_seconds = _env_float(
        "FASTVIDEO_MCP_DEFAULT_DURATION_SECONDS",
        DEFAULT_DURATION_SECONDS,
        minimum=MIN_DURATION_SECONDS,
        maximum=max_duration_seconds,
    )
    return default_duration_seconds, max_duration_seconds


@dataclass(frozen=True, slots=True)
class Settings:
    host: str
    port: int
    mcp_path: str
    public_base_url: str
    allowed_hosts: tuple[str, ...]
    allowed_origins: tuple[str, ...]
    api_token: str | None
    artifact_signing_key: str | None
    artifact_url_ttl_seconds: int
    data_dir: Path
    output_dir: Path
    asset_dir: Path
    database_path: Path
    max_concurrent_jobs: int
    progress_heartbeat_seconds: int
    job_timeout_seconds: int
    gpu_worker_idle_timeout_seconds: int
    gpu_worker_warm_max_duration_seconds: int
    default_duration_seconds: float
    max_duration_seconds: float
    resource_admission_enabled: bool
    resource_admission_poll_seconds: int
    resource_admission_ready_samples: int
    resource_admission_max_gpu_used_mib: int
    resource_admission_min_system_available_mib: int
    nvidia_smi_binary: str
    backend: str
    fastvideo_repo: Path
    fastvideo_python: Path
    fastvideo_script: Path
    model_path: str
    model_id: str
    num_gpus: int
    fastvideo_attention_backend: str
    fastvideo_sage_attention: str
    fastvideo_vsa_sparsity: float
    fastvideo_vsa_tile_size: int
    fastvideo_vsa_kernel: str
    fastvideo_vsa_cache_tile_buffer: bool
    fastvideo_vsa_attention_residual_chunk_tokens: int
    enable_fa4: bool
    fastvideo_torch_compile: bool
    fastvideo_direct_device_load: bool
    fastvideo_text_encoder_fp8: bool
    fastvideo_release_dit_after_denoise: bool
    extra_args: tuple[str, ...]
    minimax_h3_python: Path
    minimax_h3_model_path: str
    minimax_h3_lora_path: str
    minimax_h3_lora_alpha: int
    minimax_h3_lora_strength: float
    minimax_h3_steps: int
    minimax_h3_legacy_steps: int
    minimax_h3_text_encoder_fp8: bool
    minimax_h3_direct_device_load: bool
    minimax_h3_release_dit_after_denoise: bool
    minimax_h3_ff_chunk_tokens: int
    ffmpeg_binary: str
    max_reference_image_bytes: int
    base_backend: str
    comfyui_base_url: str
    comfyui_models_dir: Path
    comfyui_t2va_workflow_template: Path
    comfyui_fl2va_workflow_template: Path
    comfyui_first_frame_workflow_template: Path
    comfyui_last_frame_workflow_template: Path
    comfyui_first_last_workflow_template: Path
    comfyui_api_key: str | None
    comfyui_allow_interrupt: bool
    comfyui_unload_after_job: bool

    @classmethod
    def from_env(cls) -> "Settings":
        project_root = Path(__file__).resolve().parents[2]
        data_dir = Path(os.getenv("FASTVIDEO_MCP_DATA_DIR", project_root / "var")).expanduser().resolve()
        output_dir = Path(os.getenv("FASTVIDEO_MCP_OUTPUT_DIR", data_dir / "outputs")).expanduser().resolve()
        mcp_path = os.getenv("FASTVIDEO_MCP_PATH", "/mcp").strip()
        if not mcp_path.startswith("/"):
            raise ValueError("FASTVIDEO_MCP_PATH must start with '/'")

        backend = os.getenv("FASTVIDEO_MCP_BACKEND", "mock").strip().lower()
        if backend not in {
            "mock",
            "fastvideo-cli",
            "fastvideo-native",
            "minimax-h3-diffusers",
            "comfyui-api",
            "auto",
        }:
            raise ValueError(
                "FASTVIDEO_MCP_BACKEND must be mock, fastvideo-cli, "
                "fastvideo-native, minimax-h3-diffusers, comfyui-api, or auto"
            )

        base_backend = os.getenv(
            "MINIMAX_H3_BASE_BACKEND", "fastvideo-native"
        ).strip().lower()
        if base_backend not in {
            "fastvideo-native",
            "comfyui-api",
            "minimax-h3-diffusers",
        }:
            raise ValueError(
                "MINIMAX_H3_BASE_BACKEND must be fastvideo-native, comfyui-api, "
                "or minimax-h3-diffusers"
            )

        attention_backend = os.getenv(
            "FASTVIDEO_ATTENTION_BACKEND", "VIDEO_SPARSE_ATTN_H3"
        ).strip().upper()
        if attention_backend not in {
            "AUTO",
            "FLASH_ATTN",
            "SAGE_ATTN",
            "SAGE_ATTN_THREE",
            "TORCH_SDPA",
            "VIDEO_SPARSE_ATTN_H3",
        }:
            raise ValueError(
                "FASTVIDEO_ATTENTION_BACKEND must be AUTO, FLASH_ATTN, "
                "SAGE_ATTN, SAGE_ATTN_THREE, TORCH_SDPA, or VIDEO_SPARSE_ATTN_H3"
            )
        sage_attention = os.getenv("FASTVIDEO_SAGE_ATTENTION", "auto").strip().lower()
        if sage_attention not in {"auto", "on", "off"}:
            raise ValueError("FASTVIDEO_SAGE_ATTENTION must be auto, on, or off")

        default_duration_seconds, max_duration_seconds = generation_duration_from_env()

        vsa_tile_size = _env_int("FASTVIDEO_VSA_TILE_SIZE", 64, 1)
        if vsa_tile_size not in {64, 256}:
            raise ValueError("FASTVIDEO_VSA_TILE_SIZE must be 64 or 256")
        vsa_kernel = os.getenv("FASTVIDEO_VSA_KERNEL", "triton").strip().lower()
        if vsa_kernel not in {"triton", "sm100a"}:
            raise ValueError("FASTVIDEO_VSA_KERNEL must be triton or sm100a")

        token = os.getenv("FASTVIDEO_MCP_API_TOKEN", "").strip() or None
        signing_key = os.getenv("FASTVIDEO_MCP_ARTIFACT_SIGNING_KEY", "").strip() or None

        return cls(
            host=os.getenv("FASTVIDEO_MCP_HOST", "127.0.0.1").strip(),
            port=_env_int("FASTVIDEO_MCP_PORT", 8765, 1),
            mcp_path=mcp_path,
            public_base_url=os.getenv(
                "FASTVIDEO_MCP_PUBLIC_BASE_URL", "http://127.0.0.1:8765"
            ).strip().rstrip("/"),
            allowed_hosts=_env_csv(
                "FASTVIDEO_MCP_ALLOWED_HOSTS",
                ("localhost:*", "127.0.0.1:*", "[::1]:*"),
            ),
            allowed_origins=_env_csv("FASTVIDEO_MCP_ALLOWED_ORIGINS"),
            api_token=token,
            artifact_signing_key=signing_key,
            artifact_url_ttl_seconds=_env_int(
                "FASTVIDEO_MCP_ARTIFACT_URL_TTL_SECONDS", 3600, 60
            ),
            data_dir=data_dir,
            output_dir=output_dir,
            asset_dir=data_dir / "assets",
            database_path=data_dir / "jobs.sqlite3",
            max_concurrent_jobs=_env_int("FASTVIDEO_MCP_MAX_CONCURRENT_JOBS", 1, 1),
            progress_heartbeat_seconds=_env_int(
                "FASTVIDEO_MCP_PROGRESS_HEARTBEAT_SECONDS", 5, 1
            ),
            job_timeout_seconds=_env_int("FASTVIDEO_MCP_JOB_TIMEOUT_SECONDS", 3600, 30),
            gpu_worker_idle_timeout_seconds=_env_int(
                "FASTVIDEO_MCP_GPU_IDLE_TIMEOUT_SECONDS", 60, 1
            ),
            gpu_worker_warm_max_duration_seconds=_env_int(
                "FASTVIDEO_MCP_WARM_MAX_DURATION_SECONDS", 30, 5
            ),
            default_duration_seconds=default_duration_seconds,
            max_duration_seconds=max_duration_seconds,
            resource_admission_enabled=_env_bool(
                "FASTVIDEO_RESOURCE_ADMISSION_ENABLED", True
            ),
            resource_admission_poll_seconds=_env_int(
                "FASTVIDEO_RESOURCE_ADMISSION_POLL_SECONDS", 5, 1
            ),
            resource_admission_ready_samples=_env_int(
                "FASTVIDEO_RESOURCE_ADMISSION_READY_SAMPLES", 2, 1
            ),
            resource_admission_max_gpu_used_mib=_env_int(
                "FASTVIDEO_RESOURCE_ADMISSION_MAX_GPU_USED_MIB", 4096
            ),
            resource_admission_min_system_available_mib=_env_int(
                "FASTVIDEO_RESOURCE_ADMISSION_MIN_SYSTEM_AVAILABLE_MIB", 45000
            ),
            nvidia_smi_binary=os.getenv(
                "FASTVIDEO_NVIDIA_SMI_BINARY", "nvidia-smi"
            ).strip(),
            backend=backend,
            fastvideo_repo=Path(
                os.getenv("FASTVIDEO_REPO", "/home/clhh/FastVideo")
            ).expanduser().resolve(),
            fastvideo_python=Path(
                os.getenv(
                    "FASTVIDEO_PYTHON", "/home/clhh/FastVideo/.venv/bin/python"
                )
            ).expanduser().absolute(),
            fastvideo_script=Path(
                os.getenv(
                    "FASTVIDEO_SCRIPT",
                    "/home/clhh/FastVideo/examples/inference/basic/"
                    "basic_minimax_h3_t2v.py",
                )
            ).expanduser().resolve(),
            model_path=os.getenv(
                "FASTVIDEO_MODEL_PATH",
                "/home/clhh/.cache/modelscope/models/"
                "FastVideo--FastVideo-FastH3-4-step-Preview-v1-VSA-DataFree/"
                "snapshots/master",
            ).strip(),
            model_id=os.getenv(
                "FASTVIDEO_MODEL_ID", "fasth3-4-step-preview-v1-vsa-datafree"
            ).strip(),
            num_gpus=_env_int("FASTVIDEO_NUM_GPUS", 1, 1),
            fastvideo_attention_backend=attention_backend,
            fastvideo_sage_attention=sage_attention,
            fastvideo_vsa_sparsity=_env_float(
                "FASTVIDEO_VSA_SPARSITY",
                0.9,
                minimum=0.0,
                maximum_exclusive=1.0,
            ),
            fastvideo_vsa_tile_size=vsa_tile_size,
            fastvideo_vsa_kernel=vsa_kernel,
            fastvideo_vsa_cache_tile_buffer=_env_bool(
                "FASTVIDEO_VSA_CACHE_TILE_BUFFER", False
            ),
            fastvideo_vsa_attention_residual_chunk_tokens=_env_int(
                "FASTVIDEO_VSA_ATTENTION_RESIDUAL_CHUNK_TOKENS", 32768
            ),
            enable_fa4=_env_bool("FASTVIDEO_ENABLE_FA4", False),
            fastvideo_torch_compile=_env_bool("FASTVIDEO_TORCH_COMPILE", False),
            fastvideo_direct_device_load=_env_bool(
                "FASTVIDEO_DIRECT_DEVICE_LOAD", False
            ),
            fastvideo_text_encoder_fp8=_env_bool(
                "FASTVIDEO_TEXT_ENCODER_FP8", False
            ),
            fastvideo_release_dit_after_denoise=_env_bool(
                "FASTVIDEO_MINIMAX_H3_RELEASE_DIT_AFTER_DENOISE", True
            ),
            extra_args=_env_json_args("FASTVIDEO_EXTRA_ARGS_JSON"),
            minimax_h3_python=Path(
                os.getenv(
                    "MINIMAX_H3_PYTHON",
                    "/home/clhh/FastVideo/.venv/bin/python",
                )
            ).expanduser().absolute(),
            minimax_h3_model_path=os.getenv(
                "MINIMAX_H3_MODEL_PATH", "/mnt/data/models/MiniMax-H3-r16"
            ).strip(),
            minimax_h3_lora_path=os.getenv(
                "MINIMAX_H3_LORA_PATH",
                "/mnt/data/ComfyUI-aki-v3/ComfyUI/models/loras/"
                "minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors",
            ).strip(),
            minimax_h3_lora_alpha=_env_int("MINIMAX_H3_LORA_ALPHA", 8, 1),
            minimax_h3_lora_strength=_env_float(
                "MINIMAX_H3_LORA_STRENGTH", 1.0, minimum=0.01
            ),
            minimax_h3_steps=_env_int("MINIMAX_H3_STEPS", 8, 2),
            minimax_h3_legacy_steps=_env_int(
                "MINIMAX_H3_LEGACY_STEPS", 20, 2
            ),
            minimax_h3_text_encoder_fp8=_env_bool(
                "MINIMAX_H3_TEXT_ENCODER_FP8", True
            ),
            minimax_h3_direct_device_load=_env_bool(
                "MINIMAX_H3_DIRECT_DEVICE_LOAD", True
            ),
            minimax_h3_release_dit_after_denoise=_env_bool(
                "MINIMAX_H3_RELEASE_DIT_AFTER_DENOISE", True
            ),
            minimax_h3_ff_chunk_tokens=_env_int(
                "MINIMAX_H3_FF_CHUNK_TOKENS", 32768
            ),
            ffmpeg_binary=os.getenv("FFMPEG_BINARY", "ffmpeg").strip(),
            max_reference_image_bytes=_env_int(
                "FASTVIDEO_MCP_MAX_REFERENCE_IMAGE_BYTES", 20 * 1024 * 1024, 1024
            ),
            base_backend=base_backend,
            comfyui_base_url=os.getenv(
                "COMFYUI_BASE_URL", "http://127.0.0.1:8189"
            ).strip().rstrip("/"),
            comfyui_models_dir=Path(
                os.getenv(
                    "COMFYUI_MODELS_DIR",
                    "/mnt/data/ComfyUI-aki-v3/ComfyUI/models",
                )
            ).expanduser().resolve(),
            comfyui_t2va_workflow_template=Path(
                os.getenv(
                    "COMFYUI_T2VA_WORKFLOW_TEMPLATE",
                    str(project_root / "workflows" / "video_minimax_h3_t2va_api.json"),
                )
            ).expanduser().resolve(),
            comfyui_fl2va_workflow_template=Path(
                os.getenv(
                    "COMFYUI_FL2VA_WORKFLOW_TEMPLATE",
                    str(project_root / "workflows" / "video_minimax_h3_fl2va_api.json"),
                )
            ).expanduser().resolve(),
            comfyui_first_frame_workflow_template=Path(
                os.getenv(
                    "COMFYUI_FIRST_FRAME_WORKFLOW_TEMPLATE",
                    os.getenv(
                        "COMFYUI_FL2VA_WORKFLOW_TEMPLATE",
                        str(project_root / "workflows" / "video_minimax_h3_fl2va_api.json"),
                    ),
                )
            ).expanduser().resolve(),
            comfyui_last_frame_workflow_template=Path(
                os.getenv(
                    "COMFYUI_LAST_FRAME_WORKFLOW_TEMPLATE",
                    os.getenv(
                        "COMFYUI_FL2VA_WORKFLOW_TEMPLATE",
                        str(project_root / "workflows" / "video_minimax_h3_fl2va_api.json"),
                    ),
                )
            ).expanduser().resolve(),
            comfyui_first_last_workflow_template=Path(
                os.getenv(
                    "COMFYUI_FIRST_LAST_WORKFLOW_TEMPLATE",
                    os.getenv(
                        "COMFYUI_FL2VA_WORKFLOW_TEMPLATE",
                        str(project_root / "workflows" / "video_minimax_h3_fl2va_api.json"),
                    ),
                )
            ).expanduser().resolve(),
            comfyui_api_key=os.getenv("COMFYUI_API_KEY", "").strip() or None,
            comfyui_allow_interrupt=_env_bool("COMFYUI_ALLOW_INTERRUPT", False),
            comfyui_unload_after_job=_env_bool(
                "COMFYUI_UNLOAD_AFTER_JOB", True
            ),
        )

    def prepare_runtime_directories(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.asset_dir.mkdir(parents=True, exist_ok=True)
