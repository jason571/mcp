"""One-call, durable generation of a complete standalone video timeline."""
from __future__ import annotations

import hashlib
import tempfile
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .limits import MAX_VIDEO_DURATION_SECONDS, reject_overlong_duration
from .narrated import NarratedStore, NarratedWorkflow, validate_generation
from .plan_store import MusicVideoPlanStore
from .timeline import TimelineSegment, assemble_timeline_files


class StandaloneSegment(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    segment_id: str = Field(pattern=r"^[A-Za-z0-9_-]{1,64}$")
    duration_seconds: float = Field(
        gt=0, le=3600,
        description="Duration of one complete generator call, not one shot within its prompt. "
        "Must equal generation.duration_seconds and satisfy the live generator schema.",
    )
    generation: dict[str, Any] = Field(
        description="Complete native arguments for the selected live video submit schema. "
        "The workflow replaces first_frame_url on later segments with the preceding real last frame."
    )


class StandalonePlan(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    title: str = Field(min_length=1, max_length=300)
    request_key: str | None = Field(
        default=None, min_length=1, max_length=128,
        description="Unique key only for a newly requested rendition of identical content. Omit on recovery.",
    )
    target_duration_seconds: float = Field(gt=0, le=MAX_VIDEO_DURATION_SECONDS)
    segments: list[StandaloneSegment] = Field(
        min_length=2, max_length=64,
        description="Use only when the final video needs multiple generator calls. "
        "Storyboard timestamps remain shots inside a generator prompt, not separate segments.",
    )

    @model_validator(mode="before")
    @classmethod
    def duration_limit(cls, value):
        if isinstance(value, dict):
            raw = value.get("target_duration_seconds")
            if isinstance(raw, (int, float)) and not isinstance(raw, bool):
                reject_overlong_duration(float(raw), maximum_seconds=MAX_VIDEO_DURATION_SECONDS,
                                         deliverable="视频")
        return value

    @model_validator(mode="after")
    def timeline(self):
        if len({s.segment_id for s in self.segments}) != len(self.segments):
            raise ValueError("segment_id values must be unique")
        if abs(sum(s.duration_seconds for s in self.segments) - self.target_duration_seconds) > .001:
            raise ValueError("segment durations must sum to target_duration_seconds")
        audio_flags = set()
        for index, segment in enumerate(self.segments):
            if abs(segment.duration_seconds * 24 - round(segment.duration_seconds * 24)) > .001:
                raise ValueError(f"{segment.segment_id}: duration must align to 24 fps")
            arguments = segment.generation
            if arguments.get("duration_seconds") != segment.duration_seconds:
                raise ValueError(f"{segment.segment_id}: generation duration must equal segment window")
            if not isinstance(arguments.get("audio_enabled"), bool):
                raise ValueError(f"{segment.segment_id}: set audio_enabled explicitly")
            audio_flags.add(arguments["audio_enabled"])
            if index and "first_frame_url" in arguments:
                raise ValueError(f"{segment.segment_id}: omit first_frame_url; the workflow supplies the prior real last frame")
        if len(audio_flags) != 1:
            raise ValueError("all segments must use the same audio_enabled setting")
        return self


class StandaloneStore(NarratedStore):
    def __init__(self, root: Path):
        super().__init__(root / "standalone-video")

    def get(self, digest):
        MusicVideoPlanStore._digest(digest)
        payload = (self.root / f"{digest}.json").read_bytes()
        if hashlib.sha256(payload).hexdigest() != digest:
            raise ValueError("stored standalone plan digest mismatch")
        return StandalonePlan.model_validate_json(payload)


class StandaloneWorkflow(NarratedWorkflow):
    def __init__(self, settings, artifacts):
        super().__init__(settings, artifacts)
        self.store = StandaloneStore(settings.data_dir)
        self.workflows = self.store.states

    async def prepare(self, plan: StandalonePlan, generator):
        schema = await generator.submission_schema()
        validate_generation(plan, schema, self.settings)
        digest = self.store.put(plan)
        return {"plan_digest": digest, "status": "prepared", "segment_count": len(plan.segments),
                "duration_seconds": plan.target_duration_seconds}

    async def _ensure_media(self, plan, state, assets):
        if assets:
            raise ValueError("standalone video has no separate narration bindings")

    async def _assemble(self, plan, state, progress):
        clips = [Path(self._artifact_from_ref(r["clip"], "video").path) for r in state["segments"]]
        preserve = plan.segments[0].generation["audio_enabled"]
        windows = [TimelineSegment(video_url="persisted", duration_seconds=s.duration_seconds,
                                   preserve_source_audio=preserve) for s in plan.segments]

        async def assembly_progress(value, message):
            await progress(92 + value * .08, message)

        with tempfile.TemporaryDirectory(dir=self.settings.data_dir) as raw:
            output = Path(raw) / "final-video.mp4"
            metadata = await assemble_timeline_files(
                self.settings, windows, clips, [None] * len(clips), Path(raw), output,
                assembly_progress,
            )
            final = self.artifacts.add(output, "final-video.mp4", "video/mp4")
        return final, metadata
