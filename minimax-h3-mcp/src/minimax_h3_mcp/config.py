from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

from .models import (
    DEFAULT_DURATION_SECONDS,
    MAX_DURATION_SECONDS,
    MIN_DURATION_SECONDS,
    MODEL_ID,
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


def _env_int(
    name: str, default: int, minimum: int = 0, maximum: int | None = None
) -> int:
    raw = os.getenv(name)
    value = default if raw is None else int(raw)
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    if maximum is not None and value > maximum:
        raise ValueError(f"{name} must be <= {maximum}")
    return value


def _env_float(
    name: str,
    default: float,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    raw = os.getenv(name)
    value = default if raw is None else float(raw)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    if maximum is not None and value > maximum:
        raise ValueError(f"{name} must be <= {maximum}")
    return value


def _env_csv(name: str, default: tuple[str, ...] = ()) -> tuple[str, ...]:
    raw = os.getenv(name)
    if raw is None:
        return default
    return tuple(item.strip() for item in raw.split(",") if item.strip())


def generation_duration_from_env() -> tuple[float, float]:
    max_duration_seconds = _env_float(
        "H3_MCP_MAX_DURATION_SECONDS",
        MAX_DURATION_SECONDS,
        minimum=MIN_DURATION_SECONDS,
        maximum=MAX_DURATION_SECONDS,
    )
    default_duration_seconds = _env_float(
        "H3_MCP_DEFAULT_DURATION_SECONDS",
        DEFAULT_DURATION_SECONDS,
        minimum=MIN_DURATION_SECONDS,
        maximum=max_duration_seconds,
    )
    return default_duration_seconds, max_duration_seconds


def _http_url(name: str, value: str) -> str:
    normalized = value.strip().rstrip("/")
    parsed = urlsplit(normalized)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError(f"{name} must be an http(s) URL")
    return normalized


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
    progress_heartbeat_seconds: float
    job_timeout_seconds: int
    default_duration_seconds: float
    max_duration_seconds: float
    max_reference_image_bytes: int
    max_output_video_bytes: int
    comfyui_base_url: str
    comfyui_health_timeout_seconds: float
    comfyui_manage_process: bool
    comfyui_python: Path
    comfyui_root: Path
    comfyui_startup_timeout_seconds: int
    comfyui_models_dir: Path
    comfyui_workflow_template: Path
    comfyui_api_key: str | None
    comfyui_allow_interrupt: bool
    comfyui_unload_after_job: bool
    unet_name: str
    clip_name: str
    video_vae_name: str
    audio_vae_name: str
    lora_name: str
    lora_strength: float
    steps: int
    sampler_name: str
    scheduler_name: str

    @property
    def model_id(self) -> str:
        return MODEL_ID

    def prepare_runtime_directories(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.asset_dir.mkdir(parents=True, exist_ok=True)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_env(cls) -> "Settings":
        project_root = Path(__file__).resolve().parents[2]
        data_dir = Path(
            os.getenv("H3_MCP_DATA_DIR", str(project_root / "var"))
        ).expanduser().resolve()
        default_duration_seconds, max_duration_seconds = generation_duration_from_env()
        api_token = os.getenv("H3_MCP_API_TOKEN", "").strip() or None
        return cls(
            host=os.getenv("H3_MCP_HOST", "127.0.0.1").strip(),
            port=_env_int("H3_MCP_PORT", 8766, 1, 65535),
            mcp_path=_mcp_path(os.getenv("H3_MCP_PATH", "/mcp")),
            public_base_url=_http_url(
                "H3_MCP_PUBLIC_BASE_URL",
                os.getenv("H3_MCP_PUBLIC_BASE_URL", "http://127.0.0.1:8766"),
            ),
            allowed_hosts=_env_csv(
                "H3_MCP_ALLOWED_HOSTS",
                ("127.0.0.1:8766", "localhost:8766"),
            ),
            allowed_origins=_env_csv("H3_MCP_ALLOWED_ORIGINS"),
            api_token=api_token,
            artifact_signing_key=os.getenv("H3_MCP_ARTIFACT_SIGNING_KEY", "").strip()
            or None,
            artifact_url_ttl_seconds=_env_int(
                "H3_MCP_ARTIFACT_URL_TTL_SECONDS", 3600, 60
            ),
            data_dir=data_dir,
            output_dir=data_dir / "outputs",
            asset_dir=data_dir / "assets",
            database_path=data_dir / "jobs.sqlite3",
            max_concurrent_jobs=_env_int("H3_MCP_MAX_CONCURRENT_JOBS", 1, 1),
            progress_heartbeat_seconds=_env_float(
                "H3_MCP_PROGRESS_HEARTBEAT_SECONDS", 5.0, minimum=1.0
            ),
            job_timeout_seconds=_env_int("H3_MCP_JOB_TIMEOUT_SECONDS", 3600, 30),
            default_duration_seconds=default_duration_seconds,
            max_duration_seconds=max_duration_seconds,
            max_reference_image_bytes=_env_int(
                "H3_MCP_MAX_REFERENCE_IMAGE_BYTES", 20 * 1024 * 1024, 1024
            ),
            max_output_video_bytes=_env_int(
                "H3_MCP_MAX_OUTPUT_VIDEO_BYTES", 2 * 1024 * 1024 * 1024, 1024
            ),
            comfyui_base_url=_http_url(
                "COMFYUI_BASE_URL",
                os.getenv("COMFYUI_BASE_URL", "http://127.0.0.1:8189"),
            ),
            comfyui_health_timeout_seconds=_env_float(
                "COMFYUI_HEALTH_TIMEOUT_SECONDS", 3.0, minimum=0.25, maximum=30.0
            ),
            comfyui_manage_process=_env_bool("COMFYUI_MANAGE_PROCESS", True),
            comfyui_python=Path(
                os.getenv(
                    "COMFYUI_PYTHON",
                    "/home/clhh/miniconda3/envs/comfyui/bin/python",
                )
            ).expanduser().resolve(),
            comfyui_root=Path(
                os.getenv("COMFYUI_ROOT", "/mnt/data/ComfyUI-aki-v3/ComfyUI")
            ).expanduser().resolve(),
            comfyui_startup_timeout_seconds=_env_int(
                "COMFYUI_STARTUP_TIMEOUT_SECONDS", 120, 10
            ),
            comfyui_models_dir=Path(
                os.getenv(
                    "COMFYUI_MODELS_DIR",
                    "/mnt/data/ComfyUI-aki-v3/ComfyUI/models",
                )
            ).expanduser().resolve(),
            comfyui_workflow_template=Path(
                os.getenv(
                    "H3_COMFYUI_WORKFLOW_TEMPLATE",
                    str(project_root / "workflows" / "minimax_h3_8step_api.json"),
                )
            ).expanduser().resolve(),
            comfyui_api_key=os.getenv("COMFYUI_API_KEY", "").strip() or None,
            comfyui_allow_interrupt=_env_bool("COMFYUI_ALLOW_INTERRUPT", False),
            comfyui_unload_after_job=_env_bool("COMFYUI_UNLOAD_AFTER_JOB", True),
            unet_name=os.getenv(
                "H3_UNET_NAME",
                "minimax_h3_hybrid_fl2va_ref2va_b25-49-int8.safetensors",
            ).strip(),
            clip_name=os.getenv(
                "H3_CLIP_NAME",
                "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
            ).strip(),
            video_vae_name=os.getenv(
                "H3_VIDEO_VAE_NAME",
                "minimax_h3_video_vae_fp16.safetensors",
            ).strip(),
            audio_vae_name=os.getenv(
                "H3_AUDIO_VAE_NAME",
                "minimax_h3_audio_vae_fp32.safetensors",
            ).strip(),
            lora_name=os.getenv(
                "H3_LORA_NAME",
                "minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors",
            ).strip(),
            lora_strength=_env_float("H3_LORA_STRENGTH", 1.0, minimum=0.01),
            steps=_env_int("H3_STEPS", 8, 1),
            sampler_name=os.getenv("H3_SAMPLER_NAME", "euler").strip() or "euler",
            scheduler_name=os.getenv("H3_SCHEDULER_NAME", "beta").strip() or "beta",
        )


def _mcp_path(value: str) -> str:
    normalized = value.strip() or "/mcp"
    if not normalized.startswith("/"):
        raise ValueError("H3_MCP_PATH must start with /")
    return normalized
