"""Incremental MV authoring. Creative fields belong to the model; clocks do not.

Every accepted/partially completed section is an immutable, digest-addressed
draft. The next-step response contains only the working section, never the
accumulating full storyboard. This module performs no generation or network IO.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
from typing import Literal

from pydantic import Field, ValidationError, create_model

from .plan import (
    PHYSICAL_CONTINUITY, ContinuityAnchors, MusicVideoPlan, MusicVideoSegmentPlan, StrictPlanModel,
    TemplateProvenance, _expected_segment_windows,
)


def _partial_model(name, model):
    fields = {}
    for key, field in model.model_fields.items():
        optional = copy.deepcopy(field)
        optional.default = None
        fields[key] = (field.annotation | None, optional)
    return create_model(name, __base__=StrictPlanModel, **fields)


OverviewAnchors = _partial_model("OverviewAnchors", ContinuityAnchors)
OverviewProvenance = _partial_model("OverviewProvenance", TemplateProvenance)


class OverviewContent(StrictPlanModel):
    narrative_arc: str | None = Field(default=None, min_length=1, max_length=3000)
    style_summary: str | None = Field(default=None, min_length=1, max_length=1600)
    song_context_basis: Literal[
        "title-informed-model-knowledge", "user-lyrics", "user-timed-lyrics"
    ] | None = None
    # Partial nested objects allow one missing field to be supplied alone.
    continuity_anchors: OverviewAnchors | None = None
    provenance: OverviewProvenance | None = None


class ShotContent(StrictPlanModel):
    shot_index: int = Field(ge=0, le=63, description="Zero-based slot from current_section.")
    weight: float | None = Field(default=None, gt=0, le=100, description="Relative shot duration; the service calculates timestamps.")
    framing_and_lens: str | None = Field(default=None, min_length=1, max_length=1000)
    subject_action: str | None = Field(default=None, min_length=1, max_length=1600, description="One continuous visible action. The FIRST shot starts at the exact opening_keyframe location, pose and objects. The LAST shot finishes at the exact ending_keyframe state. " + PHYSICAL_CONTINUITY)
    camera_movement: str | None = Field(default=None, min_length=1, max_length=1000)
    environment_motion: str | None = Field(default=None, min_length=1, max_length=1000)
    lighting: str | None = Field(default=None, min_length=1, max_length=1000)
    transition: str | None = Field(default=None, min_length=1, max_length=1000)


class SegmentContent(StrictPlanModel):
    visual_role: str | None = Field(default=None, min_length=1, max_length=2000)
    music_description: str | None = Field(default=None, min_length=1, max_length=2000)
    negative_prompt: str | None = Field(default=None, min_length=10, max_length=4000)
    opening_keyframe: str | None = Field(default=None, min_length=1, max_length=2000, description="Only for segment-001; later openings are copied by the service. Describe a SINGLE static frame: location, subject pose, objects and light. The first shot begins exactly here.")
    ending_keyframe: str | None = Field(default=None, min_length=1, max_length=2000, description="Describe a SINGLE static final frame, not a sequence or camera movement. It MUST be the terminal state of this segment's last shot and is copied as the next segment's opening. Author the last shot and this frame together.")
    storyboard: list[ShotContent] | None = Field(default=None, min_length=1, max_length=64)


def _complete_model(name, model, optional_fields=(), replacements=None):
    fields = {}
    for key, field in model.model_fields.items():
        if key in optional_fields:
            continue
        required = copy.deepcopy(field)
        required.default = ...
        annotation = (replacements or {}).get(key)
        if annotation is None:
            from typing import get_args, Union
            types = tuple(t for t in get_args(field.annotation) if t is not type(None))
            annotation = Union[types] if types else field.annotation
        fields[key] = (annotation, required)
    return create_model(name, __base__=model, **fields)


CompleteShot = _complete_model("CompleteShot", ShotContent, ("weight",))
CompleteSegment = _complete_model("CompleteSegment", SegmentContent, ("opening_keyframe",), {"storyboard": list[CompleteShot]})
CompleteOverview = _complete_model("CompleteOverview", OverviewContent, replacements={"continuity_anchors":ContinuityAnchors,"provenance":TemplateProvenance})

# Full writes use array order; only focused patches need explicit slot indexes.
# Keep the per-field constraints from the ordinary complete-write schema.
OrderedShot = create_model("OrderedShot", __base__=StrictPlanModel, **{
    name: (field.annotation, copy.deepcopy(field))
    for name, field in CompleteShot.model_fields.items() if name != "shot_index"
})
OrderedSegment = _complete_model("OrderedSegment", SegmentContent, ("opening_keyframe",),
    {"storyboard": list[OrderedShot]})


class SectionWrite(StrictPlanModel):
    content: OrderedSegment


def section_window(draft: dict) -> list[dict]:
    """Use the persisted writing budget, independent of rendering concurrency."""
    window, shots = [], 0
    section_limit = draft.get("planning_batch_size", 4)
    shot_limit = 64 if "planning_batch_size" in draft else 16
    for segment in draft["plan"]["segments"][draft["accepted_segments"]:]:
        count = len(segment["prompt_plan"]["storyboard"])
        if window and (not draft.get("authoring_batch_version") or len(window) == section_limit or shots + count > shot_limit):
            break
        window.append(segment)
        shots += count
    return window


def queue_sections(draft: dict, sections: list[SectionWrite]) -> dict:
    # The advertised window keeps model responses small. If a caller has already
    # authored more valid consecutive sections, retain them rather than throwing
    # away the work and asking the model to generate it again.
    remaining = [s["segment_id"] for s in draft["plan"]["segments"][draft["accepted_segments"]:]]
    if overview_errors(draft) or draft.get("pending_sections") or not sections or len(sections) > len(remaining):
        raise ValueError("Write consecutive remaining sections in array order; saved data is unchanged.")
    updated = copy.deepcopy(draft)
    pending = updated.setdefault("pending_sections", {})
    for segment_id, section in zip(remaining, sections):
        data = section.content.model_dump(exclude_unset=True, exclude_none=True)
        for index, shot in enumerate(data["storyboard"]):
            shot["shot_index"] = index
        pending[segment_id] = data
    return updated


def drain_sections(draft: dict) -> tuple[dict, list[dict]]:
    """After a local repair, reuse later saved sections without another model call."""
    updated = copy.deepcopy(draft)
    while updated["accepted_segments"] < len(updated["plan"]["segments"]):
        segment_id = updated["plan"]["segments"][updated["accepted_segments"]]["segment_id"]
        content = updated.get("pending_sections", {}).pop(segment_id, None)
        if content is None:
            break
        updated, errors = write_segment(updated, segment_id, SegmentContent.model_validate(content), complete=True)
        if errors:
            return updated, errors
    return updated, []


def create_draft(settings, *, duration_ms: int, audio_url: str, audio_sha256: str,
                 brief: str, aspect_ratio: str, resolution: str,
                 output_language: str, shot_mode: str, shots_per_segment: int,
                 planning_batch_size: int = 4, opening_frame: dict | None = None,
                 reference_image: dict | None = None,
                 duration_policy: dict | None = None) -> dict:
    if not 1 <= planning_batch_size <= 16:
        raise ValueError("planning_batch_size must be between 1 and 16")
    clocks = {
        "target_segment_duration_seconds": settings.video_generator_target_duration_seconds,
        "live_min_duration_seconds": settings.video_generator_min_duration_seconds,
        "live_max_duration_seconds": settings.video_generator_max_duration_seconds,
        "duration_step_seconds": settings.video_generator_duration_step_seconds,
        "tail_merge_max_duration_seconds": settings.video_generator_tail_merge_max_duration_seconds,
    }
    if duration_policy:
        clocks.update({key: duration_policy[key] for key in clocks if key in duration_policy})
    policy = {"master_duration_ms": duration_ms, **clocks}
    # Reuse the exact window calculation used by the final renderer.
    windows = _expected_segment_windows(MusicVideoPlan.model_construct(**policy))
    if not 1 <= len(windows) <= 64:
        raise ValueError("audio requires more than 64 segments")
    identity_parts = [audio_sha256, brief, policy, aspect_ratio, resolution, output_language, shot_mode, shots_per_segment]
    if opening_frame:
        identity_parts.append(opening_frame)
    if reference_image:
        identity_parts.append(reference_image)
    identity = json.dumps(identity_parts, sort_keys=True)
    segments = []
    for index, (start, end) in enumerate(windows):
        duration = max(policy["live_min_duration_seconds"], math.ceil((end-start)/1000/policy["duration_step_seconds"])*policy["duration_step_seconds"])
        count = 1 if shot_mode == "single-shot" else max(3, shots_per_segment)
        segments.append({
            "segment_id": f"segment-{index+1:03d}", "start_ms": start, "end_ms": end,
            "generation_duration_seconds": duration,
            "prompt_plan": {"shot_mode": shot_mode, "soundscape": None,
                "controls": {"duration_seconds": duration, "aspect_ratio": aspect_ratio, "resolution": resolution, "audio_enabled": False},
                "storyboard": [{} for _ in range(count)]},
        })
    draft = {"authoring_version": 2, "authoring_batch_version": 1, "planning_batch_size": planning_batch_size,
        "audio_url": audio_url, "audio_sha256": audio_sha256,
        "brief": brief, "overview": {}, "accepted_segments": 0, "shot_weights": {},
        "plan": {"schema_version": 1, "plan_id": "mv-"+hashlib.sha256(identity.encode()).hexdigest()[:24],
            **policy, "output_language": output_language, "segments": segments}}
    if opening_frame:
        draft.update(opening_frame)
    if reference_image:
        draft.update(reference_image)
    return draft


def _errors(error: ValidationError, prefix: str = "") -> list[dict]:
    return [{"path": prefix + "/" + "/".join(str(part).replace("~", "~0").replace("/", "~1") for part in item["loc"]),
             "code": item["type"], "message": item["msg"]}
            for item in error.errors(include_url=False, include_input=False)]


def overview_errors(draft: dict) -> list[dict]:
    errors = []
    for field in ("narrative_arc", "style_summary", "song_context_basis"):
        if not draft["overview"].get(field):
            errors.append({"path": "/"+field, "code": "missing", "message": "Field required"})
    for field, model in (("continuity_anchors", ContinuityAnchors), ("provenance", TemplateProvenance)):
        try:
            model.model_validate(draft["overview"].get(field, {}))
        except ValidationError as error:
            errors.extend(_errors(error, "/"+field))
    return errors


def write_overview(draft: dict, content: OverviewContent) -> tuple[dict, list[dict]]:
    updated = copy.deepcopy(draft)
    if draft["accepted_segments"]:
        return updated, [{"path": "/overview", "code": "locked", "message": "Overview is locked after segment acceptance; continue the current section."}]
    for field, value in content.model_dump(exclude_unset=True, exclude_none=True).items():
        if isinstance(value, dict):
            model = ContinuityAnchors if field == "continuity_anchors" else TemplateProvenance
            unknown = set(value) - set(model.model_fields)
            if unknown:
                return copy.deepcopy(draft), [{"path": "/"+field+"/"+name, "code": "unknown_field", "message": "Unknown field; saved data is unchanged."} for name in sorted(unknown)]
            updated["overview"].setdefault(field, {}).update(value)
        else:
            updated["overview"][field] = value
    errors = overview_errors(updated)
    if not errors:
        for field in ("song_context_basis", "continuity_anchors", "provenance"):
            updated["plan"][field] = copy.deepcopy(updated["overview"][field])
    return updated, errors


def _materialize_segment(draft: dict, index: int) -> None:
    segment = draft["plan"]["segments"][index]
    prompt = segment["prompt_plan"]
    if index:
        prompt["opening_keyframe"] = draft["plan"]["segments"][index-1]["prompt_plan"]["ending_keyframe"]
    shots = prompt["storyboard"]
    weights = draft["shot_weights"].get(segment["segment_id"], {})
    values = [weights.get(str(i), 1) for i in range(len(shots))]
    duration = segment["generation_duration_seconds"]*1000
    cursor, accumulated = 0, 0.0
    for i, shot in enumerate(shots):
        accumulated += values[i]
        end = duration if i == len(shots)-1 else round(duration*accumulated/sum(values))
        shot.update(start_ms=cursor, end_ms=end)
        cursor = end


def write_segment(draft: dict, segment_id: str, content: SegmentContent, *, complete: bool = False) -> tuple[dict, list[dict]]:
    updated = copy.deepcopy(draft)
    index = draft["accepted_segments"]
    segments = updated["plan"]["segments"]
    if overview_errors(draft) or index >= len(segments) or segments[index]["segment_id"] != segment_id:
        return updated, [{"path": "/segment_id", "code": "wrong_section", "message": "Only the current unaccepted section may be edited."}]
    segment, prompt = segments[index], segments[index]["prompt_plan"]
    fields = content.model_dump(exclude_unset=True, exclude_none=True)
    # Reject foreign slots before applying any part of the update.
    slots = [item["shot_index"] for item in fields.get("storyboard", [])]
    if complete:
        # A full creative write chooses its shot count. Empty scaffold slots are
        # suggestions, not missing work. Focused patches keep saved addresses.
        if not slots or sorted(slots) != list(range(len(slots))):
            return updated, [{"path": "/storyboard", "code": "invalid_slot", "message": "A complete storyboard needs one or more shots indexed consecutively from zero."}]
        minimum = 1 if prompt["shot_mode"] == "single-shot" else 3
        # Keep supplied creative fields even below the acceptance floor. The
        # empty missing slots become focused repair targets, not a full retry.
        prompt["storyboard"] = [{} for _ in range(max(minimum, len(slots)))]
        updated["shot_weights"].pop(segment_id, None)
    if len(slots) != len(set(slots)) or any(i >= len(prompt["storyboard"]) for i in slots):
        return updated, [{"path": "/storyboard", "code": "invalid_slot", "message": "Use each current_section shot_index at most once."}]
    if index:
        # This field is service-owned after the first segment, just like the
        # persisted clocks/controls. A redundant model echo cannot change it.
        fields.pop("opening_keyframe", None)
    for field, value in fields.items():
        if field == "storyboard":
            for item in value:
                slot = item.pop("shot_index")
                if "weight" in item:
                    updated["shot_weights"].setdefault(segment_id, {})[str(slot)] = item.pop("weight")
                prompt["storyboard"][slot].update(item)
        elif field == "visual_role":
            segment[field] = value
        elif field == "music_description":
            segment["music_evidence"] = {"description": value,
                "timing_basis": "user-timed-lyrics" if updated["overview"]["song_context_basis"] == "user-timed-lyrics" else "duration-only"}
        else:
            prompt[field] = value
    _materialize_segment(updated, index)
    try:
        parsed = MusicVideoSegmentPlan.model_validate(segment)
    except ValidationError as error:
        errors = _errors(error)
        # Report the public content fields, not private plan nesting.
        for item in errors:
            if item["path"].startswith("/prompt_plan/"):
                item["path"] = item["path"][len("/prompt_plan"):]
            if item["path"].startswith("/music_evidence"):
                item["path"] = "/music_description"
        return updated, errors
    errors = []
    def signature(s):
        return json.dumps([{k:v for k,v in shot.items() if k not in ("start_ms", "end_ms")} for shot in s["prompt_plan"]["storyboard"]], sort_keys=True).casefold()
    for earlier in segments[:index]:
        if " ".join(earlier["visual_role"].split()).casefold() == " ".join(parsed.visual_role.split()).casefold():
            errors.append({"path": "/visual_role", "code": "duplicate", "message": "Give this segment its own narrative role."})
        if signature(earlier) == signature(segment):
            errors.append({"path": "/storyboard", "code": "duplicate", "message": "This storyboard repeats an accepted segment."})
    if any(shot.end_ms <= shot.start_ms for shot in parsed.prompt_plan.storyboard):
        errors.append({"path": "/storyboard", "code": "duration", "message": "Shot weights must allow positive millisecond durations."})
    if not errors:
        updated["accepted_segments"] += 1
    return updated, errors


def _section_brief(segment: dict, total: int) -> dict:
    """Planning metadata, not an example of the internal executable plan."""
    number = int(segment["segment_id"].rsplit("-", 1)[1])
    return {**{k: segment[k] for k in ("segment_id", "start_ms", "end_ms", "generation_duration_seconds")},
        "suggested_shot_count": len(segment["prompt_plan"]["storyboard"]),
        "minimum_shots": 1 if segment["prompt_plan"]["shot_mode"] == "single-shot" else 3,
        "opening_keyframe_required": number == 1, "is_final": number == total}


def _saved_content(segment: dict) -> dict:
    """Project saved creative fields to the SAME public input contract."""
    prompt = segment["prompt_plan"]
    data = {k: copy.deepcopy(prompt[k]) for k in ("negative_prompt", "opening_keyframe", "ending_keyframe") if k in prompt}
    if "visual_role" in segment:
        data["visual_role"] = segment["visual_role"]
    if "music_evidence" in segment:
        data["music_description"] = segment["music_evidence"]["description"]
    shots = []
    for index, shot in enumerate(prompt["storyboard"]):
        fields = {k: v for k, v in shot.items() if k in ShotContent.model_fields and k != "shot_index"}
        if fields:
            shots.append({"shot_index": index, **fields})
    if shots:
        data["storyboard"] = shots
    return data


def checkpoint(store, draft: dict, errors: list[dict] | None = None) -> dict:
    reference = store.put_draft(draft)
    overview_pending = bool(overview_errors(draft))
    index = draft["accepted_segments"]
    count = len(draft["plan"]["segments"])
    if index == count:
        raise ValueError("completed draft must be finalized before returning a checkpoint")
    operation = "patch" if errors else "write"
    required = f"{operation}_music_video_plan_overview" if overview_pending else f"{operation}_music_video_plan_segment"
    arguments = dict(reference)
    working = {"brief": draft["brief"], "master_duration_ms": draft["plan"]["master_duration_ms"], "segment_count": count,
        "clip_duration_seconds": draft["plan"]["target_segment_duration_seconds"],
        "tail_merge_max_seconds": draft["plan"]["tail_merge_max_duration_seconds"],
        "output_language": draft["plan"]["output_language"],
        "authoring_guidance": "Use one work-wide style and narrative. Advance each current section's distinct story beat within that overview. Keep each creative field to one concrete sentence; do not repeat global style/wardrobe paragraphs. Start the first shot at the exact opening frame and finish the last shot at the ending frame. " + PHYSICAL_CONTINUITY + " Use the native input schema and its shot addressing. Honor the brief; plan silent visuals without subtitles, titles, logos or credits. Timing and controls are already saved; no audio-energy analysis or invented lyric timestamps. The final section resolves the story. Clip length is the live generator default in clip_duration_seconds; do not assume 15s or 30s."}
    if draft.get("reference_image_attachment_id") or draft.get("reference_image_url"):
        working["authoring_guidance"] += (
            " A persistent reference image supplies identity/style cues, not a reset frame"
            " or a permanent location. Continue from the previous segment's ending;"
            " change place through visible story actions. Revisit the source composition"
            " only when this section's distinct story beat calls for it."
        )
    if overview_pending:
        working["saved_fields"] = draft["overview"]
        working["required_fields"] = [item["path"] for item in overview_errors(draft)]
    else:
        segment = draft["plan"]["segments"][index]
        arguments["segment_id"] = segment["segment_id"]
        working.update(overview=draft["overview"], current_section=_section_brief(segment, count))
        if errors:
            working["saved_content"] = _saved_content(segment)
            working["current_section"]["saved_shot_count"] = len(segment["prompt_plan"]["storyboard"])
        _, missing = write_segment(draft, segment["segment_id"], SegmentContent())
        working["required_fields"] = [item["path"] for item in missing]
        working["section_position"] = {"index": index, "count": count, "is_final": index == count-1}
        working["shot_indexes"] = list(range(len(segment["prompt_plan"]["storyboard"])))
        if index:
            working["previous_ending_keyframe"] = draft["plan"]["segments"][index-1]["prompt_plan"]["ending_keyframe"]
        window = section_window(draft)
        if not errors and len(window) > 1:
            required = "write_music_video_plan_sections"
            arguments.pop("segment_id")
            working.pop("current_section")
            working.pop("shot_indexes")
            # A single-segment list of missing internal paths was misleading in
            # a batch. Publish only the window and ONE public input shape.
            working.pop("required_fields")
            working["current_sections"] = [_section_brief(s, count) for s in window]
            example = {"visual_role":"<this section's story beat>", "music_description":"<title-informed mood, not measured audio>",
                "negative_prompt":"<unwanted visual content>", "ending_keyframe":"<static state after the last shot>",
                "storyboard":[{name:"<"+name+">" for name in OrderedShot.model_fields if name != "weight"}]}
            if index == 0:
                example["opening_keyframe"] = "<static state where the very first shot begins>"
            working["input_shape_example"] = {"sections":[{"content":example}]}
            working["batch_guidance"] = f"Write all {len(window)} current_sections in one call, in displayed order. input_shape_example illustrates nesting only: replace its placeholders. For multi-shot segments, author at least four shots; three complete shots are accepted without rewriting merely to reach four. Fewer than minimum_shots must be repaired only in that segment. An explicitly selected single-shot mode keeps one continuous shot. sections and storyboard are arrays; each section contains a content object. The first section of the entire work requires opening_keyframe; later openings are copied from preceding endings. On validation failure, later submitted sections stay saved; patch only the reported field."
        elif not errors:
            working["shot_count_guidance"] = "For multi-shot mode author at least four complete shots, indexed consecutively from zero. Three are accepted without rewriting to reach four; fewer than minimum_shots are rejected. Explicit single-shot mode keeps one shot. The service distributes the segment time across the submitted shots."
            working.pop("shot_indexes")
            working.pop("required_fields")
        if draft.get("pending_sections"):
            working["saved_later_sections"] = list(draft["pending_sections"])
    result = {**reference, "valid": False, "accepted_segments": index, "segment_count": count,
        "validation_errors": errors or [], "next_tool": required,
        "_meta": {"exoharness": {
            "context_checkpoint": "Creative draft persisted. Keep only this current section; accepted sections remain in MCP Store.",
            "context_preserve": ["/draft_id", "/draft_digest", "/accepted_segments", "/segment_count", "/validation_errors"],
            "workflow_continuation": {"version": 1, "status": "required", "required_tool": required,
                # Before the overview is committed, style-query results are
                # still needed. Never declare this input self-contained early.
                **({"context_scope": "checkpoint"} if not overview_pending else {}),
                "checkpoint": reference, "required_arguments": arguments,
                "reason": "author_current_section", "repair_context": working}}}}
    # Section-sized bound, independent of already accepted segment count.
    if len(json.dumps(result, ensure_ascii=False).encode()) > 30*1024:
        # A large partially written section remains durable. On a repair turn
        # only the error paths and mechanical slots are needed, not its prose.
        if working.pop("saved_content", None):
            working["saved_creative_fields_retained_in_store"] = True
        working.pop("saved_fields", None)
        if len(json.dumps(result, ensure_ascii=False).encode()) > 30*1024:
            raise ValueError("global overview exceeds current-section context budget")
    return result
