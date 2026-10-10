from __future__ import annotations

import secrets
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any


FPS = 24
DEFAULT_WIDTH = 1344
DEFAULT_HEIGHT = 768
MIN_DURATION_SECONDS = 5.0
DEFAULT_DURATION_SECONDS = 15.0
MAX_DURATION_SECONDS = 60.0
LONG_VIDEO_THRESHOLD_SECONDS = 15.0
LONG_VIDEO_MAX_RESOLUTION = "512p"
FRAMES_PER_CHUNK = 17
LATENTS_PER_CHUNK = 5
TERMINAL_STATUSES = {"completed", "failed", "canceled"}
ASPECT_RATIOS = {
    "16:9": (16, 9),
    "9:16": (9, 16),
}
RESOLUTION_SHORT_EDGES = {"512p": 512, "640p": 640, "768p": 768}
MODEL_ID = "minimax-h3-8step"


def normalize_resolution(duration_seconds: float, resolution: object) -> str:
    if not isinstance(resolution, str):
        return LONG_VIDEO_MAX_RESOLUTION
    normalized = resolution.strip().lower()
    if normalized not in RESOLUTION_SHORT_EDGES:
        return LONG_VIDEO_MAX_RESOLUTION
    if (
        duration_seconds > LONG_VIDEO_THRESHOLD_SECONDS
        and normalized != LONG_VIDEO_MAX_RESOLUTION
    ):
        return LONG_VIDEO_MAX_RESOLUTION
    return normalized


def validate_duration_resolution(duration_seconds: float, resolution: str) -> None:
    if (
        duration_seconds > LONG_VIDEO_THRESHOLD_SECONDS
        and resolution != LONG_VIDEO_MAX_RESOLUTION
    ):
        raise ValueError(
            f"resolution must be {LONG_VIDEO_MAX_RESOLUTION} when duration_seconds "
            f"is greater than {LONG_VIDEO_THRESHOLD_SECONDS:g}"
        )


def duration_seconds_contract(
    default: float = DEFAULT_DURATION_SECONDS,
    maximum: float = MAX_DURATION_SECONDS,
) -> dict[str, float]:
    return {
        "default": default,
        "minimum": MIN_DURATION_SECONDS,
        "maximum": maximum,
    }


def duration_resolution_limits(maximum: float) -> list[dict[str, object]]:
    first_maximum = min(LONG_VIDEO_THRESHOLD_SECONDS, maximum)
    limits: list[dict[str, object]] = [
        {
            "duration_seconds": {
                "minimum": MIN_DURATION_SECONDS,
                "maximum": first_maximum,
            },
            "resolutions": list(RESOLUTION_SHORT_EDGES),
        }
    ]
    if maximum > LONG_VIDEO_THRESHOLD_SECONDS:
        limits.append(
            {
                "duration_seconds": {
                    "exclusive_minimum": LONG_VIDEO_THRESHOLD_SECONDS,
                    "maximum": maximum,
                },
                "resolutions": [LONG_VIDEO_MAX_RESOLUTION],
            }
        )
    return limits


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def make_job_id() -> str:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"vid_{timestamp}_{secrets.token_hex(4)}"


def aligned_num_frames(duration_seconds: float) -> int:
    if not MIN_DURATION_SECONDS <= duration_seconds <= MAX_DURATION_SECONDS:
        raise ValueError(
            f"duration_seconds must be between {MIN_DURATION_SECONDS:g} and "
            f"{MAX_DURATION_SECONDS:g}"
        )
    requested_frames = round(duration_seconds * FPS)
    chunks = max(0, -(-(requested_frames - LATENTS_PER_CHUNK) // FRAMES_PER_CHUNK))
    return LATENTS_PER_CHUNK + chunks * FRAMES_PER_CHUNK


def canvas_dimensions(aspect_ratio: str, resolution: str) -> tuple[int, int]:
    if aspect_ratio not in ASPECT_RATIOS:
        raise ValueError("aspect_ratio must be one of " + ", ".join(ASPECT_RATIOS))
    if resolution not in RESOLUTION_SHORT_EDGES:
        raise ValueError(
            "resolution must be one of " + ", ".join(RESOLUTION_SHORT_EDGES)
        )
    ratio_width, ratio_height = ASPECT_RATIOS[aspect_ratio]
    short_edge = RESOLUTION_SHORT_EDGES[resolution]

    def align_32(value: float) -> int:
        return max(32, round(value / 32) * 32)

    if ratio_width >= ratio_height:
        width = align_32(short_edge * ratio_width / ratio_height)
        height = short_edge
    else:
        width = short_edge
        height = align_32(short_edge * ratio_height / ratio_width)
    if resolution == "768p" and aspect_ratio == "16:9":
        return 1344, 768
    if resolution == "768p" and aspect_ratio == "9:16":
        return 768, 1344
    return width, height


def build_h3_prompt(
    prompt: str,
    negative_prompt: str | None,
    soundscape: str | None,
    prompt_mode: str,
    *,
    reference_image: bool = False,
    audio_enabled: bool = True,
) -> str:
    clean_prompt = prompt.strip()
    if not clean_prompt:
        raise ValueError("prompt must not be empty")
    if len(clean_prompt) > 12_000:
        raise ValueError("prompt must not exceed 12000 characters")
    if reference_image:
        if prompt_mode not in {"simple", "h3"}:
            raise ValueError("prompt_mode must be 'simple' or 'h3'")
        if clean_prompt.startswith("subject_definitions:"):
            result = clean_prompt
            if "<Picture 1>" not in result:
                result = result.replace(
                    "subject_definitions:",
                    "subject_definitions: <Picture 1> is the supplied persistent "
                    "visual reference. Keep its depicted subject and setting "
                    "recognizable throughout the clip unless the user's "
                    "instructions explicitly change them. ",
                    1,
                )
        else:
            visual = clean_prompt
            if visual.startswith("integrated_multimodal_description:"):
                visual = visual.split(":", 1)[1].split("\noverall_soundscape:", 1)[0].strip()
            ambience = (
                (soundscape or "Natural synchronized ambience matching the scene.").strip()
                if audio_enabled else "N/A"
            )
            result = (
                "subject_definitions: <Picture 1> is the supplied persistent visual reference; "
                "its depicted subject and setting are the baseline unless the user explicitly changes them.\n"
                "summary: Generate the requested scene using <Picture 1> throughout, not as frame zero.\n"
                "retention_analysis: Keep the visible subject, location, and distinguishing details "
                "recognizable across shots; use visible transitions for any user-requested change.\n"
                f"detailed_description: {visual}\n"
                f"overall_soundscape: {ambience}\n"
                "non_diegetic_music: N/A"
            )
        if negative_prompt and negative_prompt.strip():
            result += f"\nnegative_constraints: {negative_prompt.strip()}"
        return result
    if prompt_mode == "h3":
        result = clean_prompt
        if negative_prompt and negative_prompt.strip():
            result += f"\nnegative_constraints: {negative_prompt.strip()}"
        return result
    if prompt_mode != "simple":
        raise ValueError("prompt_mode must be 'simple' or 'h3'")
    clean_soundscape = (
        soundscape or "Natural synchronized ambience matching the scene."
    ).strip()
    if len(clean_soundscape) > 4_000:
        raise ValueError("soundscape must not exceed 4000 characters")
    result = (
        f"integrated_multimodal_description: {clean_prompt}\n"
        f"overall_soundscape: {clean_soundscape}"
    )
    if negative_prompt and negative_prompt.strip():
        if len(negative_prompt.strip()) > 4_000:
            raise ValueError("negative_prompt must not exceed 4000 characters")
        result += f"\nnegative_constraints: {negative_prompt.strip()}"
    return result


class JobStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    CANCEL_REQUESTED = "cancel_requested"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELED = "canceled"


@dataclass(frozen=True, slots=True)
class GenerationRequest:
    prompt: str
    negative_prompt: str | None
    soundscape: str | None
    prompt_mode: str
    model_id: str
    duration_seconds: float
    seed: int
    audio_enabled: bool
    aspect_ratio: str
    resolution: str
    first_frame_asset_id: str | None = None
    last_frame_asset_id: str | None = None
    reference_image_asset_id: str | None = None
    first_frame_path: str | None = None
    last_frame_path: str | None = None
    reference_image_path: str | None = None
    width: int = DEFAULT_WIDTH
    height: int = DEFAULT_HEIGHT
    fps: int = FPS

    @property
    def num_frames(self) -> int:
        return aligned_num_frames(self.duration_seconds)

    @property
    def actual_duration_seconds(self) -> float:
        return self.num_frames / self.fps

    @property
    def model_prompt(self) -> str:
        return build_h3_prompt(
            self.prompt, self.negative_prompt, self.soundscape, self.prompt_mode,
            reference_image=bool(self.reference_image_path),
            audio_enabled=self.audio_enabled,
        )

    @property
    def has_keyframes(self) -> bool:
        return bool(self.first_frame_path or self.last_frame_path)


@dataclass(slots=True)
class JobRecord:
    id: str
    status: str
    progress: float
    stage: str
    message: str
    prompt: str
    negative_prompt: str | None
    soundscape: str | None
    prompt_mode: str
    model_id: str
    requested_duration_seconds: float
    num_frames: int
    width: int
    height: int
    fps: int
    seed: int
    audio_enabled: bool
    aspect_ratio: str
    resolution: str
    first_frame_asset_id: str | None
    last_frame_asset_id: str | None
    reference_image_asset_id: str | None
    first_frame_path: str | None
    last_frame_path: str | None
    reference_image_path: str | None
    created_at: str
    updated_at: str
    started_at: str | None = None
    finished_at: str | None = None
    output_path: str | None = None
    error: str | None = None

    @property
    def terminal(self) -> bool:
        return self.status in TERMINAL_STATUSES

    @property
    def actual_duration_seconds(self) -> float:
        return self.num_frames / self.fps

    def as_public_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data.pop("output_path", None)
        data.pop("first_frame_path", None)
        data.pop("last_frame_path", None)
        data.pop("reference_image_path", None)
        data["actual_duration_seconds"] = round(self.actual_duration_seconds, 3)
        return data

    @classmethod
    def new(cls, request: GenerationRequest) -> "JobRecord":
        now = utc_now()
        return cls(
            id=make_job_id(),
            status=JobStatus.QUEUED,
            progress=0.0,
            stage="queued",
            message="任务已进入队列",
            prompt=request.prompt,
            negative_prompt=request.negative_prompt,
            soundscape=request.soundscape,
            prompt_mode=request.prompt_mode,
            model_id=request.model_id,
            requested_duration_seconds=request.duration_seconds,
            num_frames=request.num_frames,
            width=request.width,
            height=request.height,
            fps=request.fps,
            seed=request.seed,
            audio_enabled=request.audio_enabled,
            aspect_ratio=request.aspect_ratio,
            resolution=request.resolution,
            first_frame_asset_id=request.first_frame_asset_id,
            last_frame_asset_id=request.last_frame_asset_id,
            reference_image_asset_id=request.reference_image_asset_id,
            first_frame_path=request.first_frame_path,
            last_frame_path=request.last_frame_path,
            reference_image_path=request.reference_image_path,
            created_at=now,
            updated_at=now,
        )

    def to_request(self) -> GenerationRequest:
        return GenerationRequest(
            prompt=self.prompt,
            negative_prompt=self.negative_prompt,
            soundscape=self.soundscape,
            prompt_mode=self.prompt_mode,
            model_id=self.model_id,
            duration_seconds=self.requested_duration_seconds,
            seed=self.seed,
            audio_enabled=self.audio_enabled,
            aspect_ratio=self.aspect_ratio,
            resolution=self.resolution,
            first_frame_asset_id=self.first_frame_asset_id,
            last_frame_asset_id=self.last_frame_asset_id,
            reference_image_asset_id=self.reference_image_asset_id,
            first_frame_path=self.first_frame_path,
            last_frame_path=self.last_frame_path,
            reference_image_path=self.reference_image_path,
            width=self.width,
            height=self.height,
            fps=self.fps,
        )

    def output_file(self) -> Path | None:
        return Path(self.output_path) if self.output_path else None
