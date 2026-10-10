from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path


def _env_int(name: str, default: int, minimum: int = 0) -> int:
    value = int(os.getenv(name, str(default)))
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return value


def _env_csv(name: str, default: tuple[str, ...] = ()) -> tuple[str, ...]:
    raw = os.getenv(name)
    if raw is None:
        return default
    return tuple(item.strip().rstrip("/") for item in raw.split(",") if item.strip())


def _secret(name: str, required: bool = False) -> str | None:
    value = os.getenv(name, "").strip()
    file_name = os.getenv(f"{name}_FILE", "").strip()
    if value and file_name:
        raise ValueError(f"configure either {name} or {name}_FILE, not both")
    if file_name:
        try:
            value = Path(file_name).read_text(encoding="utf-8").strip()
        except OSError as exc:
            raise ValueError(f"could not read {name}_FILE: {exc}") from exc
    if required and not value:
        raise ValueError(f"{name} or {name}_FILE must be configured")
    return value or None


@dataclass(frozen=True, slots=True)
class Settings:
    host: str
    port: int
    mcp_path: str
    public_base_url: str
    allowed_hosts: tuple[str, ...]
    allowed_origins: tuple[str, ...]
    allowed_fetch_origins: tuple[str, ...]
    api_token: str | None
    artifact_signing_key: str
    artifact_url_ttl_seconds: int
    data_dir: Path
    artifact_dir: Path
    ffmpeg_binary: str
    ffprobe_binary: str
    max_download_bytes: int
    operation_timeout_seconds: int
    video_generator_url: str | None = None
    video_generator_authorization: str | None = None
    video_generator_submit_tool: str = "submit_video_generation"
    video_generator_status_tool: str = "get_video_job"
    video_generator_result_tool: str = "get_video_result"
    video_generator_cancel_tool: str = "cancel_video_job"
    video_generator_poll_seconds: int = 3
    workflow_idle_timeout_seconds: int = 1800
    video_generator_min_duration_seconds: int = 5
    video_generator_max_duration_seconds: int = 60
    video_generator_target_duration_seconds: int = 15
    video_generator_duration_step_seconds: int = 1
    video_generator_tail_merge_max_duration_seconds: int = 21
    asset_base: str = "http://127.0.0.1:8787"

    @classmethod
    def from_env(cls) -> "Settings":
        data_dir = Path(
            os.getenv("MEDIA_MCP_DATA_DIR", "/tmp/media-workflow-mcp")
        ).expanduser().resolve()
        mcp_path = os.getenv("MEDIA_MCP_PATH", "/mcp").strip()
        if not mcp_path.startswith("/"):
            raise ValueError("MEDIA_MCP_PATH must start with '/'")
        signing_key = _secret("MEDIA_MCP_ARTIFACT_SIGNING_KEY", required=True)
        ffmpeg = os.getenv("MEDIA_MCP_FFMPEG", "ffmpeg").strip()
        ffprobe = os.getenv("MEDIA_MCP_FFPROBE", "ffprobe").strip()
        if shutil.which(ffmpeg) is None:
            raise ValueError(f"ffmpeg executable was not found: {ffmpeg}")
        if shutil.which(ffprobe) is None:
            raise ValueError(f"ffprobe executable was not found: {ffprobe}")
        settings = cls(
            host=os.getenv("MEDIA_MCP_HOST", "127.0.0.1").strip(),
            port=_env_int("MEDIA_MCP_PORT", 8790, 1),
            mcp_path=mcp_path,
            public_base_url=os.getenv(
                "MEDIA_MCP_PUBLIC_BASE_URL", "http://127.0.0.1:8790"
            ).strip().rstrip("/"),
            allowed_hosts=_env_csv(
                "MEDIA_MCP_ALLOWED_HOSTS",
                ("localhost:*", "127.0.0.1:*", "[::1]:*"),
            ),
            allowed_origins=_env_csv("MEDIA_MCP_ALLOWED_ORIGINS"),
            allowed_fetch_origins=_env_csv(
                "MEDIA_MCP_ALLOWED_FETCH_ORIGINS",
                ("http://127.0.0.1:8787", "http://127.0.0.1:8765"),
            ),
            api_token=_secret("MEDIA_MCP_API_TOKEN"),
            artifact_signing_key=signing_key or "",
            artifact_url_ttl_seconds=_env_int(
                "MEDIA_MCP_ARTIFACT_URL_TTL_SECONDS", 0, 0
            ),
            data_dir=data_dir,
            artifact_dir=data_dir / "artifacts",
            ffmpeg_binary=ffmpeg,
            ffprobe_binary=ffprobe,
            max_download_bytes=_env_int(
                "MEDIA_MCP_MAX_DOWNLOAD_BYTES", 2 * 1024 * 1024 * 1024, 1
            ),
            operation_timeout_seconds=_env_int(
                "MEDIA_MCP_OPERATION_TIMEOUT_SECONDS", 1800, 30
            ),
            video_generator_url=os.getenv(
                "MEDIA_MCP_VIDEO_GENERATOR_URL", ""
            ).strip()
            or None,
            video_generator_authorization=_secret(
                "MEDIA_MCP_VIDEO_GENERATOR_AUTHORIZATION"
            ),
            video_generator_submit_tool=os.getenv(
                "MEDIA_MCP_VIDEO_GENERATOR_SUBMIT_TOOL",
                "submit_video_generation",
            ).strip(),
            video_generator_status_tool=os.getenv(
                "MEDIA_MCP_VIDEO_GENERATOR_STATUS_TOOL", "get_video_job"
            ).strip(),
            video_generator_result_tool=os.getenv(
                "MEDIA_MCP_VIDEO_GENERATOR_RESULT_TOOL", "get_video_result"
            ).strip(),
            video_generator_cancel_tool=os.getenv(
                "MEDIA_MCP_VIDEO_GENERATOR_CANCEL_TOOL", "cancel_video_job"
            ).strip(),
            video_generator_poll_seconds=_env_int(
                "MEDIA_MCP_VIDEO_GENERATOR_POLL_SECONDS", 3, 1
            ),
            workflow_idle_timeout_seconds=_env_int(
                "MEDIA_MCP_WORKFLOW_IDLE_TIMEOUT_SECONDS", 1800, 60
            ),
            video_generator_min_duration_seconds=_env_int(
                "MEDIA_MCP_VIDEO_MIN_DURATION_SECONDS", 5, 1
            ),
            video_generator_max_duration_seconds=_env_int(
                "MEDIA_MCP_VIDEO_MAX_DURATION_SECONDS", 60, 1
            ),
            video_generator_target_duration_seconds=_env_int(
                "MEDIA_MCP_VIDEO_TARGET_DURATION_SECONDS", 15, 1
            ),
            video_generator_duration_step_seconds=_env_int(
                "MEDIA_MCP_VIDEO_DURATION_STEP_SECONDS", 1, 1
            ),
            video_generator_tail_merge_max_duration_seconds=_env_int(
                "MEDIA_MCP_VIDEO_TAIL_MERGE_MAX_DURATION_SECONDS", 21, 1
            ),
            asset_base=os.getenv("MEDIA_MCP_ASSET_BASE", "http://127.0.0.1:8787")
            .strip()
            .rstrip("/"),
        )
        if not (
            settings.video_generator_min_duration_seconds
            <= settings.video_generator_target_duration_seconds
            <= settings.video_generator_tail_merge_max_duration_seconds
            <= settings.video_generator_max_duration_seconds
        ):
            raise ValueError(
                "video duration settings must satisfy minimum <= target <= "
                "tail merge maximum <= maximum"
            )
        if (
            settings.video_generator_target_duration_seconds
            % settings.video_generator_duration_step_seconds
            or settings.video_generator_tail_merge_max_duration_seconds
            % settings.video_generator_duration_step_seconds
        ):
            raise ValueError(
                "video target and tail merge durations must align to the duration step"
            )
        settings.prepare_directories()
        return settings

    def prepare_directories(self) -> None:
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        try:
            self.data_dir.chmod(0o700)
            self.artifact_dir.chmod(0o700)
        except OSError:
            pass
