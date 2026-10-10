from __future__ import annotations

import hashlib
import json
import math
import re
from typing import Any, Literal

# One rule for every shot. The writing model sees it on the action field;
# the video model sees the same sentence in every rendered prompt.
PHYSICAL_CONTINUITY = (
    "Change place only along a path this shot itself shows. "
    "Do not teleport the subject into or out of a vehicle, across a gap, or off an edge."
)

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictPlanModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class ContinuityAnchors(StrictPlanModel):
    subject_identity: str = Field(min_length=1, max_length=1200)
    wardrobe_or_materials: str = Field(
        default="Keep established wardrobe, materials, and surface details unchanged across segments.",
        min_length=1,
        max_length=1200,
    )
    setting_geometry: str = Field(
        min_length=1,
        max_length=1200,
        description=(
            "Spatial layout of recurring locations when they appear again; "
            "a reference image does not require every segment to stay in its depicted location."
        ),
    )
    palette_and_lighting: str = Field(min_length=1, max_length=1200)
    camera_language: str = Field(min_length=1, max_length=1200)
    recurring_motif: str = Field(
        min_length=1,
        max_length=1200,
        description="A motif to use at story-relevant moments, not a required object in every shot.",
    )
    screen_direction: str = Field(min_length=1, max_length=1200)


class MusicEvidence(StrictPlanModel):
    timing_basis: Literal["measured-energy", "duration-only", "user-timed-lyrics"] = Field(
        description=(
            "Timing evidence only. Use duration-only when story semantics come from a song "
            "title/model knowledge; title-informed-model-knowledge belongs only in the "
            "top-level song_context_basis field."
        )
    )
    description: str = Field(min_length=1, max_length=2000)


class StoryboardShot(StrictPlanModel):
    start_ms: int = Field(
        ge=0,
        description=(
            "Segment-local start time in milliseconds. Every segment restarts at 0; "
            "never place the master-audio timestamp here."
        ),
    )
    end_ms: int = Field(
        gt=0,
        description=(
            "Segment-local end time in milliseconds, contiguous with the next shot and "
            "ending at generation_duration_seconds * 1000."
        ),
    )
    framing_and_lens: str = Field(min_length=1, max_length=1000)
    subject_action: str = Field(min_length=1, max_length=1600)
    camera_movement: str = Field(min_length=1, max_length=1000)
    environment_motion: str = Field(min_length=1, max_length=1000)
    lighting: str = Field(min_length=1, max_length=1000)
    transition: str = Field(min_length=1, max_length=1000)


class VideoControls(StrictPlanModel):
    duration_seconds: int = Field(ge=1, le=3600)
    aspect_ratio: str = Field(pattern=r"^[1-9][0-9]*:[1-9][0-9]*$")
    resolution: str = Field(min_length=1, max_length=64)
    audio_enabled: Literal[False]


class TemplateProvenance(StrictPlanModel):
    planning_mode: Literal["read-only-resources", "executable-tools", "user-specified"]
    pack_id: str | None = Field(default=None, min_length=1, max_length=200)
    document_id: str | None = Field(default=None, min_length=1, max_length=300)
    model_profile: str = Field(min_length=1, max_length=200)

    @model_validator(mode="after")
    def require_actual_library_source(self) -> "TemplateProvenance":
        if self.planning_mode == "user-specified":
            if self.pack_id is not None or self.document_id is not None:
                raise ValueError("user-specified style must not claim a library pack or document")
        elif self.pack_id is None or self.document_id is None:
            raise ValueError("library style requires its actual pack_id and document_id")
        return self


class SegmentPromptPlan(StrictPlanModel):
    shot_mode: Literal["multi-shot", "single-shot"] = "multi-shot"
    negative_prompt: str = Field(min_length=10, max_length=4000)
    soundscape: None = None
    opening_keyframe: str = Field(min_length=1, max_length=2000)
    ending_keyframe: str = Field(min_length=1, max_length=2000)
    storyboard: list[StoryboardShot] = Field(min_length=1, max_length=64)
    controls: VideoControls


class MusicVideoSegmentPlan(StrictPlanModel):
    segment_id: str = Field(pattern=r"^segment-[0-9]{3,}$")
    start_ms: int = Field(
        ge=0, description="Start of this segment on the master-audio timeline in milliseconds."
    )
    end_ms: int = Field(
        gt=0, description="End of this segment on the master-audio timeline in milliseconds."
    )
    generation_duration_seconds: int = Field(ge=1, le=3600)
    visual_role: str = Field(min_length=1, max_length=2000)
    music_evidence: MusicEvidence
    prompt_plan: SegmentPromptPlan


class MusicVideoPlan(StrictPlanModel):
    schema_version: Literal[1]
    plan_id: str = Field(min_length=1, max_length=200)
    master_duration_ms: int = Field(gt=0)
    target_segment_duration_seconds: int = Field(ge=1, le=3600, default=15)
    live_min_duration_seconds: int = Field(ge=1, le=3600, default=5)
    live_max_duration_seconds: int = Field(ge=1, le=3600, default=60)
    duration_step_seconds: int = Field(ge=1, le=3600, default=1)
    tail_merge_max_duration_seconds: int = Field(
        ge=1,
        le=3600,
        default=21,
        description=(
            "Maximum duration of a final segment after absorbing a short trailing "
            "remainder (default clip plus 40%). Remainder beyond this stays a "
            "separate clip, for example 15+7."
        ),
    )
    output_language: Literal["zh", "en"] = "zh"
    song_context_basis: Literal[
        "title-informed-model-knowledge",
        "user-lyrics",
        "user-timed-lyrics",
        "energy-only",
    ] = Field(
        description=(
            "Semantic story source for the whole song. This is separate from each "
            "segment music_evidence.timing_basis."
        )
    )
    continuity_anchors: ContinuityAnchors
    provenance: TemplateProvenance
    segments: list[MusicVideoSegmentPlan] = Field(min_length=1, max_length=64)


class MusicVideoPlanPatch(StrictPlanModel):
    op: Literal["add", "replace"]
    path: str = Field(
        min_length=2,
        max_length=512,
        pattern=r"^/(continuity_anchors|provenance|segments|plan_id|output_language|song_context_basis)(/.*)?$",
        description=(
            "Exact RFC-6901 path returned in patch_targets. Array positions are "
            "zero-based numeric tokens: use /segments/10/... for segment-011, "
            "never /segments/segment-011/...."
        ),
    )
    value: Any


class MusicVideoPlanValidationError(ValueError):
    """Plan validation failure with machine-usable patch targets."""

    def __init__(self, errors: list[str], patch_targets: list[str]) -> None:
        self.errors = tuple(errors)
        self.patch_targets = tuple(dict.fromkeys(patch_targets))
        super().__init__("music video plan validation failed: " + "; ".join(errors))


MAX_SERIALIZED_PLAN_BYTES = 256 * 1024
MAX_PATCH_BYTES = 64 * 1024


def apply_music_video_plan_patches(
    plan: MusicVideoPlan, patches: list[MusicVideoPlanPatch]
) -> MusicVideoPlan:
    """Apply bounded RFC-6901 add/replace operations to creative plan fields."""
    if not patches:
        raise ValueError("at least one plan patch is required")
    if len(patches) > 32:
        raise ValueError("at most 32 plan patches are allowed")
    if len(
        json.dumps(
            [patch.model_dump(mode="json") for patch in patches],
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
    ) > MAX_PATCH_BYTES:
        raise ValueError(f"plan patches exceed {MAX_PATCH_BYTES} bytes")

    raw = plan.model_dump(mode="json")
    for patch in patches:
        parts = [_decode_json_pointer_part(part) for part in patch.path.split("/")[1:]]
        parent: Any = raw
        for part in parts[:-1]:
            if isinstance(parent, dict):
                if part not in parent:
                    raise ValueError(f"plan patch parent does not exist: {patch.path}")
                parent = parent[part]
            elif isinstance(parent, list):
                index = _json_pointer_index(part, len(parent), patch.path)
                parent = parent[index]
            else:
                raise ValueError(f"plan patch parent is not a container: {patch.path}")
        leaf = parts[-1]
        if isinstance(parent, dict):
            if patch.op == "replace" and leaf not in parent:
                raise ValueError(f"plan patch target does not exist: {patch.path}")
            parent[leaf] = patch.value
        elif isinstance(parent, list):
            if patch.op == "add":
                # RFC-6902 add inserts an array item, including at len(array).
                # Replacing here made it impossible to repair a missing segment.
                index = len(parent) if leaf == "-" else _json_pointer_index(
                    leaf, len(parent) + 1, patch.path
                )
                parent.insert(index, patch.value)
            else:
                index = _json_pointer_index(leaf, len(parent), patch.path)
                parent[index] = patch.value
        else:
            raise ValueError(f"plan patch target is not a container: {patch.path}")
    return MusicVideoPlan.model_validate(raw)


def _decode_json_pointer_part(value: str) -> str:
    if re.search(r"~(?![01])", value):
        raise ValueError("plan patch contains an invalid JSON pointer escape")
    output = value.replace("~1", "/").replace("~0", "~")
    return output


def _json_pointer_index(value: str, length: int, path: str) -> int:
    try:
        index = int(value)
    except ValueError as exc:
        raise ValueError(f"plan patch list index is invalid: {path}") from exc
    if index < 0 or index >= length:
        raise ValueError(f"plan patch list index is outside the plan: {path}")
    return index


def missing_segment_repairs(plan: MusicVideoPlan) -> list[dict[str, object]]:
    windows = _expected_segment_windows(plan)
    step_ms = plan.duration_step_seconds * 1000
    return [
        {
            "path": f"/segments/{index}",
            "op": "add",
            "segment_id": f"segment-{index + 1:03d}",
            "start_ms": start_ms,
            "end_ms": end_ms,
            "generation_duration_seconds": max(
                plan.live_min_duration_seconds,
                math.ceil((end_ms - start_ms) / step_ms) * plan.duration_step_seconds,
            ),
            "previous_ending_keyframe": (
                plan.segments[-1].prompt_plan.ending_keyframe
                if index == len(plan.segments) else "Use the preceding newly-added segment's ending_keyframe"
            ),
        }
        for index, (start_ms, end_ms) in enumerate(windows)
        if index >= len(plan.segments)
    ]


def normalize_music_video_plan(
    plan: MusicVideoPlan,
    *,
    live_min_duration_seconds: int,
    live_max_duration_seconds: int,
    target_segment_duration_seconds: int,
    duration_step_seconds: int,
    tail_merge_max_duration_seconds: int,
) -> tuple[MusicVideoPlan, list[str]]:
    """Replace mechanically derived timing fields with the service contract.

    Creative text remains untouched. This prevents a caller from having to
    regenerate a complete plan merely because it used master timestamps inside
    a segment or copied stale generator limits.
    """
    raw = plan.model_dump(mode="json")
    corrections: list[str] = []

    authoritative = {
        "live_min_duration_seconds": live_min_duration_seconds,
        "live_max_duration_seconds": live_max_duration_seconds,
        "target_segment_duration_seconds": target_segment_duration_seconds,
        "duration_step_seconds": duration_step_seconds,
        "tail_merge_max_duration_seconds": tail_merge_max_duration_seconds,
    }
    for field, value in authoritative.items():
        if raw.get(field) != value:
            corrections.append(field)
            raw[field] = value

    normalized = MusicVideoPlan.model_validate(raw)
    windows = _expected_segment_windows(normalized)
    if len(normalized.segments) != len(windows):
        return normalized, corrections

    raw_segments = raw["segments"]
    prior_ending: str | None = None
    step_ms = duration_step_seconds * 1000
    for index, ((start_ms, end_ms), segment) in enumerate(
        zip(windows, raw_segments, strict=True)
    ):
        label = f"segments[{index}]"
        logical_duration_ms = end_ms - start_ms
        generation_ms = max(
            live_min_duration_seconds * 1000,
            math.ceil(logical_duration_ms / step_ms) * step_ms,
        )
        generation_seconds = generation_ms // 1000
        derived = {
            "segment_id": f"segment-{index + 1:03d}",
            "start_ms": start_ms,
            "end_ms": end_ms,
            "generation_duration_seconds": generation_seconds,
        }
        for field, value in derived.items():
            if segment.get(field) != value:
                corrections.append(f"{label}.{field}")
                segment[field] = value

        prompt_plan = segment["prompt_plan"]
        controls = prompt_plan["controls"]
        if controls.get("duration_seconds") != generation_seconds:
            corrections.append(f"{label}.prompt_plan.controls.duration_seconds")
            controls["duration_seconds"] = generation_seconds
        if prior_ending is not None and prompt_plan.get("opening_keyframe") != prior_ending:
            corrections.append(f"{label}.prompt_plan.opening_keyframe")
            prompt_plan["opening_keyframe"] = prior_ending
        prior_ending = prompt_plan["ending_keyframe"]

        storyboard = prompt_plan["storyboard"]
        boundaries = _normalized_storyboard_boundaries(storyboard, generation_ms)
        for shot_index, shot in enumerate(storyboard):
            expected_start = boundaries[shot_index]
            expected_end = boundaries[shot_index + 1]
            if shot.get("start_ms") != expected_start:
                corrections.append(
                    f"{label}.prompt_plan.storyboard[{shot_index}].start_ms"
                )
                shot["start_ms"] = expected_start
            if shot.get("end_ms") != expected_end:
                corrections.append(
                    f"{label}.prompt_plan.storyboard[{shot_index}].end_ms"
                )
                shot["end_ms"] = expected_end

    return MusicVideoPlan.model_validate(raw), corrections


def _normalized_storyboard_boundaries(
    storyboard: list[dict[str, Any]], generation_ms: int
) -> list[int]:
    original = [storyboard[0]["start_ms"]] + [shot["end_ms"] for shot in storyboard]
    contiguous = all(
        storyboard[index]["start_ms"] == original[index]
        and storyboard[index]["end_ms"] > storyboard[index]["start_ms"]
        for index in range(len(storyboard))
    )
    if contiguous and original[0] == 0 and original[-1] == generation_ms:
        return original
    span = original[-1] - original[0]
    if contiguous and span > 0:
        return [
            round((boundary - original[0]) * generation_ms / span)
            for boundary in original
        ]
    return [
        round(generation_ms * shot_index / len(storyboard))
        for shot_index in range(len(storyboard) + 1)
    ]


def tail_merge_limit(target_seconds: int, live_max_seconds: int, fraction: float = 0.4) -> int:
    """Absorb a remainder only when it is at most 40% of the live default clip."""
    extra = int(target_seconds * fraction)
    limit = min(live_max_seconds, target_seconds + extra)
    return max(target_seconds, limit)


def _positive_int(value) -> int | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        return None
    return max(1, int(value))


def _schema_nodes(field: dict) -> list[dict]:
    nodes = [field]
    for key in ("anyOf", "oneOf", "allOf"):
        items = field.get(key)
        if isinstance(items, list):
            nodes.extend(item for item in items if isinstance(item, dict))
    return nodes


def duration_numeric_default(field: dict) -> int | None:
    """Use only numeric JSON Schema defaults. Ignore null and description text."""
    for node in _schema_nodes(field):
        for key in ("default", "const"):
            value = _positive_int(node.get(key))
            if value is not None:
                return value
        enumerated = node.get("enum")
        if isinstance(enumerated, list):
            numbers = [_positive_int(item) for item in enumerated]
            numbers = [item for item in numbers if item is not None]
            if len(set(numbers)) == 1:
                return numbers[0]
    return None


def duration_bounds(field: dict) -> tuple[int | None, int | None]:
    live_min = live_max = None
    for node in _schema_nodes(field):
        live_min = live_min or _positive_int(node.get("minimum"))
        live_max = live_max or _positive_int(node.get("maximum"))
    return live_min, live_max


# Last-resort clip length when the video-generator MCP omits a numeric default.
MCP_DURATION_FALLBACK_SECONDS = 15


def duration_policy_from_schema(
    schema: dict,
    *,
    duration_step_seconds: int = 1,
    fallback_target_seconds: int | None = None,
    fallback_min_seconds: int | None = None,
    fallback_max_seconds: int | None = None,
) -> dict[str, int]:
    """Prefer the live MCP numeric default; if MCP omits one, use 15s. Never use env 30."""
    if not isinstance(schema, dict):
        raise ValueError("video-generator submit schema is missing")
    props = schema.get("properties") if isinstance(schema.get("properties"), dict) else {}
    field = props.get("duration_seconds")
    if not isinstance(field, dict):
        raise ValueError("video-generator submit schema must declare duration_seconds")

    live_min, live_max = duration_bounds(field)
    target = duration_numeric_default(field)
    if target is None and live_min is not None and live_min == live_max:
        target = live_min
    if target is None:
        target = _positive_int(fallback_target_seconds)
    if target is None:
        target = MCP_DURATION_FALLBACK_SECONDS

    live_min = live_min or _positive_int(fallback_min_seconds) or 1
    live_max = live_max or _positive_int(fallback_max_seconds) or max(target, tail_merge_limit(target, 10**9))
    if live_min > live_max:
        raise ValueError("video-generator duration_seconds minimum exceeds maximum")
    target = min(max(target, live_min), live_max)
    step = max(1, int(duration_step_seconds))
    tail = tail_merge_limit(target, live_max)
    tail -= tail % step
    if tail < target:
        tail = target
    return {
        "live_min_duration_seconds": live_min,
        "live_max_duration_seconds": live_max,
        "target_segment_duration_seconds": target,
        "duration_step_seconds": step,
        "tail_merge_max_duration_seconds": tail,
    }


def _expected_segment_windows(plan: MusicVideoPlan) -> list[tuple[int, int]]:
    target_ms = plan.target_segment_duration_seconds * 1000
    tail_merge_max_ms = plan.tail_merge_max_duration_seconds * 1000
    full_count, remainder_ms = divmod(plan.master_duration_ms, target_ms)

    if full_count == 0:
        return [(0, plan.master_duration_ms)]
    if remainder_ms == 0:
        return [
            (index * target_ms, (index + 1) * target_ms)
            for index in range(full_count)
        ]
    if target_ms + remainder_ms <= tail_merge_max_ms:
        windows = [
            (index * target_ms, (index + 1) * target_ms)
            for index in range(full_count - 1)
        ]
        windows.append(((full_count - 1) * target_ms, plan.master_duration_ms))
        return windows
    return [
        (
            index * target_ms,
            min((index + 1) * target_ms, plan.master_duration_ms),
        )
        for index in range(full_count + 1)
    ]


def parse_music_video_plan(plan: MusicVideoPlan | str) -> MusicVideoPlan:
    """Normalize either schema-native input or an explicitly serialized JSON plan."""
    if isinstance(plan, MusicVideoPlan):
        return plan
    if len(plan.encode("utf-8")) > MAX_SERIALIZED_PLAN_BYTES:
        raise ValueError(
            f"serialized music video plan exceeds {MAX_SERIALIZED_PLAN_BYTES} bytes"
        )
    try:
        value = json.loads(plan)
    except json.JSONDecodeError as exc:
        raise ValueError(f"serialized music video plan is not valid JSON: {exc.msg}") from exc
    if not isinstance(value, dict):
        raise ValueError("serialized music video plan must decode to a JSON object")
    return MusicVideoPlan.model_validate(value)


def _seconds(milliseconds: int) -> str:
    value = milliseconds / 1000
    return f"{value:g}"


def _render_prompt(
    plan: MusicVideoPlan, segment: MusicVideoSegmentPlan
) -> str:
    prompt = segment.prompt_plan
    anchors = plan.continuity_anchors
    if plan.output_language == "zh":
        lines = [
            (
                f"{segment.segment_id}，主音乐时间窗 "
                f"{_seconds(segment.start_ms)}-{_seconds(segment.end_ms)} 秒；"
                f"本段视频使用局部时间轴 0-{segment.generation_duration_seconds} 秒。"
            ),
            f"本段叙事职责：{segment.visual_role}",
            (
                "跨段视觉连续性："
                f"主体={anchors.subject_identity}；"
                f"服装或材质={anchors.wardrobe_or_materials}；"
                f"色彩与光线={anchors.palette_and_lighting}；"
                f"摄影语言={anchors.camera_language}；"
                f"屏幕运动方向={anchors.screen_direction}。"
            ),
            f"开场关键帧：{prompt.opening_keyframe}",
            "逐镜头分镜：",
        ]
        for shot in prompt.storyboard:
            lines.append(
                f"[{_seconds(shot.start_ms)}-{_seconds(shot.end_ms)} 秒] "
                f"景别与镜头={shot.framing_and_lens}；"
                f"主体动作={shot.subject_action}；"
                f"摄影机运动={shot.camera_movement}；"
                f"环境运动={shot.environment_motion}；"
                f"光线={shot.lighting}；"
                f"转场={shot.transition}。"
            )
        lines.extend(
            [
                f"结尾关键帧：{prompt.ending_keyframe}",
                "换位置只沿本镜头里看得见的路径进行。不要把主体瞬移进车、瞬移出车、跨过空隙或迈下边缘。",
                "只生成无声画面，不生成音乐、对白、环境音、字幕、标题、标志或水印。",
            ]
        )
        return "\n".join(lines)

    lines = [
        (
            f"{segment.segment_id}; master-audio window "
            f"{_seconds(segment.start_ms)}-{_seconds(segment.end_ms)} seconds; "
            f"use a local video timeline of 0-{segment.generation_duration_seconds} seconds."
        ),
        f"Narrative responsibility: {segment.visual_role}",
        (
            "Cross-segment visual continuity: "
            f"subject={anchors.subject_identity}; "
            f"wardrobe/materials={anchors.wardrobe_or_materials}; "
            f"palette/lighting={anchors.palette_and_lighting}; "
            f"camera language={anchors.camera_language}; "
            f"screen direction={anchors.screen_direction}."
        ),
        f"Opening keyframe: {prompt.opening_keyframe}",
        "Shot-by-shot storyboard:",
    ]
    for shot in prompt.storyboard:
        lines.append(
            f"[{_seconds(shot.start_ms)}-{_seconds(shot.end_ms)}s] "
            f"framing/lens={shot.framing_and_lens}; "
            f"subject action={shot.subject_action}; "
            f"camera movement={shot.camera_movement}; "
            f"environment motion={shot.environment_motion}; "
            f"lighting={shot.lighting}; transition={shot.transition}."
        )
    lines.extend(
        [
            f"Ending keyframe: {prompt.ending_keyframe}",
            PHYSICAL_CONTINUITY,
            "Silent visuals only. Do not generate music, dialogue, ambience, subtitles, titles, logos, or watermarks.",
        ]
    )
    return "\n".join(lines)


def validate_and_render_music_video_plan(
    plan: MusicVideoPlan, *, opening_frame: dict[str, str] | None = None
) -> dict[str, object]:
    errors: list[str] = []
    patch_targets: list[str] = []
    target_ms = plan.target_segment_duration_seconds * 1000
    step_ms = plan.duration_step_seconds * 1000

    if plan.live_min_duration_seconds > plan.live_max_duration_seconds:
        errors.append("live_min_duration_seconds must not exceed live_max_duration_seconds")
    if not (
        plan.live_min_duration_seconds
        <= plan.target_segment_duration_seconds
        <= plan.live_max_duration_seconds
    ):
        errors.append("target segment duration is outside the live generation range")
    if plan.target_segment_duration_seconds % plan.duration_step_seconds != 0:
        errors.append("target segment duration is not aligned to duration_step_seconds")
    if not (
        plan.target_segment_duration_seconds
        <= plan.tail_merge_max_duration_seconds
        <= plan.live_max_duration_seconds
    ):
        errors.append(
            "tail_merge_max_duration_seconds must be between the target and live maximum durations"
        )
    if plan.tail_merge_max_duration_seconds % plan.duration_step_seconds != 0:
        errors.append(
            "tail_merge_max_duration_seconds is not aligned to duration_step_seconds"
        )

    expected_windows = _expected_segment_windows(plan)
    expected_count = len(expected_windows)
    if len(plan.segments) != expected_count:
        errors.append(
            f"expected {expected_count} segments for {plan.master_duration_ms} ms, "
            f"received {len(plan.segments)}"
        )
        patch_targets.extend(
            f"/segments/{index}" for index in range(len(plan.segments), expected_count)
        )

    rendered: list[dict[str, object]] = []
    prior_ending: str | None = None
    rendered_prompts: set[str] = set()
    storyboard_signatures: set[str] = set()
    visual_roles: set[str] = set()

    for index, segment in enumerate(plan.segments):
        label = segment.segment_id
        expected_id = f"segment-{index + 1:03d}"
        if index < expected_count:
            expected_start, expected_end = expected_windows[index]
        else:
            expected_start, expected_end = segment.start_ms, segment.end_ms
        logical_duration_ms = expected_end - expected_start
        required_generation_ms = max(
            plan.live_min_duration_seconds * 1000,
            math.ceil(logical_duration_ms / step_ms) * step_ms,
        )

        if label != expected_id:
            errors.append(f"segment {index + 1} must use id {expected_id}, received {label}")
        if segment.start_ms != expected_start or segment.end_ms != expected_end:
            errors.append(
                f"{label} must cover master window {expected_start}-{expected_end} ms"
            )
        if segment.generation_duration_seconds * 1000 != required_generation_ms:
            errors.append(
                f"{label} generation duration must be {required_generation_ms // 1000} seconds"
            )
        if segment.generation_duration_seconds > plan.live_max_duration_seconds:
            errors.append(f"{label} generation duration exceeds the live maximum")

        prompt = segment.prompt_plan
        controls = prompt.controls
        if controls.duration_seconds != segment.generation_duration_seconds:
            errors.append(f"{label} controls.duration_seconds differs from the segment")
        if controls.audio_enabled is not False or prompt.soundscape is not None:
            errors.append(f"{label} must be a silent prompt plan")
        if prior_ending is not None and prompt.opening_keyframe != prior_ending:
            errors.append(
                f"{label} opening_keyframe must exactly equal the previous ending_keyframe"
            )
            patch_targets.append(f"/segments/{index}/prompt_plan/opening_keyframe")
        prior_ending = prompt.ending_keyframe

        generation_ms = segment.generation_duration_seconds * 1000
        if prompt.shot_mode == "multi-shot" and len(prompt.storyboard) < 3:
            errors.append(f"{label} needs at least three storyboard shots")
            patch_targets.append(f"/segments/{index}/prompt_plan/storyboard")
        if prompt.shot_mode == "single-shot" and len(prompt.storyboard) != 1:
            errors.append(f"{label} single-shot mode requires exactly one storyboard item")
            patch_targets.append(f"/segments/{index}/prompt_plan/storyboard")

        normalized_role = " ".join(segment.visual_role.split()).casefold()
        if normalized_role in visual_roles:
            errors.append(f"{label} must have an independent visual_role")
            patch_targets.append(f"/segments/{index}/visual_role")
        visual_roles.add(normalized_role)

        storyboard_signature = json.dumps(
            [
                {
                    "framing_and_lens": shot.framing_and_lens,
                    "subject_action": shot.subject_action,
                    "camera_movement": shot.camera_movement,
                    "environment_motion": shot.environment_motion,
                    "lighting": shot.lighting,
                    "transition": shot.transition,
                }
                for shot in prompt.storyboard
            ],
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).casefold()
        if storyboard_signature in storyboard_signatures:
            errors.append(f"{label} must not reuse another segment's storyboard")
            patch_targets.append(f"/segments/{index}/prompt_plan/storyboard")
        storyboard_signatures.add(storyboard_signature)

        cursor = 0
        for shot_index, shot in enumerate(prompt.storyboard, start=1):
            if shot.start_ms != cursor:
                errors.append(
                    f"{label} shot {shot_index} must start at {cursor} ms, "
                    f"received {shot.start_ms}"
                )
                patch_targets.append(f"/segments/{index}/prompt_plan/storyboard")
            if shot.end_ms <= shot.start_ms:
                errors.append(f"{label} shot {shot_index} has a non-positive duration")
                patch_targets.append(f"/segments/{index}/prompt_plan/storyboard")
            cursor = shot.end_ms
        if cursor != generation_ms:
            errors.append(
                f"{label} local storyboard must end at {generation_ms} ms, received {cursor}"
            )
            patch_targets.append(f"/segments/{index}/prompt_plan/storyboard")

        positive_prompt = _render_prompt(plan, segment)
        normalized_prompt = " ".join(positive_prompt.split()).casefold()
        if normalized_prompt in rendered_prompts:
            errors.append(f"{label} renders the same positive prompt as another segment")
            patch_targets.append(f"/segments/{index}/prompt_plan/storyboard")
        rendered_prompts.add(normalized_prompt)
        rendered.append(
            {
                "segment_id": label,
                "master_start_ms": segment.start_ms,
                "master_end_ms": segment.end_ms,
                "generation_duration_seconds": segment.generation_duration_seconds,
                "storyboard_shot_count": len(prompt.storyboard),
                "positive_prompt": positive_prompt,
                "negative_prompt": prompt.negative_prompt,
                "soundscape": None,
                "controls": controls.model_dump(mode="json"),
                "opening_keyframe": prompt.opening_keyframe,
                "ending_keyframe": prompt.ending_keyframe,
            }
        )

    if errors:
        raise MusicVideoPlanValidationError(errors, patch_targets)

    payload = plan.model_dump(mode="json")
    frame = {
        key: value
        for key, value in (opening_frame or {}).items()
        if key in {"first_frame_attachment_id", "first_frame_url", "reference_image_attachment_id", "reference_image_url"} and value
    }
    if frame:
        payload.update(frame)
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    result = {
        "valid": True,
        "plan_id": plan.plan_id,
        "plan_digest": hashlib.sha256(canonical).hexdigest(),
        "master_duration_ms": plan.master_duration_ms,
        "segment_count": len(rendered),
        "provenance": plan.provenance.model_dump(mode="json"),
        "segments": rendered,
    }
    result.update(frame)
    return result
