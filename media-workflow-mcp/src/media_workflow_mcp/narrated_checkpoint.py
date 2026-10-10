"""Declare saved-plan handoff using the existing durable-workflow protocol.

No plan, generation setting, narration or rendering algorithm is modified.
The host resolves names against its own authorized live schema catalogue.
"""


def prepared_checkpoint(result: dict[str, object]) -> dict[str, object]:
    meta: dict[str, object] = {
        "context_checkpoint": "Full narrated plan persisted; use digest and narration texts, not full generation arguments.",
        "context_preserve": ["/plan_digest", "/status", "/segment_count", "/duration_seconds", "/narrations",
                             "/clip_duration_seconds", "/tail_merge_max_seconds", "/narration_pace_hint"],
    }
    narrations = result.get("narrations", [])
    # Warnings may require reauthoring before speech. Do not lock a warned plan
    # to rendering or remove the opportunity to correct its exact narration.
    if any(item.get("likely_too_long") for item in narrations):
        return meta
    speech = any(item.get("text", "").strip() for item in narrations)
    meta["workflow_continuation"] = {
        "version": 1,
        "status": "required",
        "context_scope": "checkpoint",
        "required_tool": "render_narrated_video",
        "required_arguments": {"plan_digest": result["plan_digest"]},
        "checkpoint": {"plan_digest": result["plan_digest"]},
        "prerequisite_tools": ["model.synthesize_speech"] if speech else [],
        "reason": (
            "The complete plan is saved, not rendered. Reuse the preserved narration texts unchanged. "
            "Synthesize each required narration with artifact_visibility=internal and retain the exact returned audio URL and SHA-256. "
            "Call render_narrated_video by the saved digest and bindings; do not repeat model lookup, plan creation or successful speech. "
            "A silent plan has no speech prerequisite. Tool names and arguments must follow the host's authorized live schemas."
        ),
    }
    return meta
