from __future__ import annotations

import copy
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(PROJECT / "tests"))

from media_workflow_mcp.authoring import (
    OverviewContent, SegmentContent, SectionWrite, checkpoint, create_draft, drain_sections,
    queue_sections, section_window, write_overview, write_segment,
)
from media_workflow_mcp.plan_store import MusicVideoPlanStore
from media_workflow_mcp.server import _finish_authoring
from media_workflow_mcp import server
from test_plan import valid_plan


def settings():
    return SimpleNamespace(video_generator_min_duration_seconds=5,
        video_generator_max_duration_seconds=60, video_generator_target_duration_seconds=30,
        video_generator_duration_step_seconds=1, video_generator_tail_merge_max_duration_seconds=40,
        allowed_fetch_origins=())


def skeleton(duration=313_992):
    return create_draft(settings(), duration_ms=duration, audio_url="https://media.example/song.mp3",
        audio_sha256="a"*64, brief="Create a homecoming MV from the song title.",
        aspect_ratio="16:9", resolution="512p", output_language="zh", shot_mode="multi-shot", shots_per_segment=4)


def overview():
    plan = valid_plan()
    return dict(narrative_arc="Wandering, remembering, returning home; each chapter advances the same journey.",
        style_summary="One natural pastoral style, warm light and restrained motion.",
        **{k:plan[k] for k in ("song_context_basis", "continuity_anchors", "provenance")})


def content(segment):
    p = segment["prompt_plan"]
    return dict(visual_role=segment["visual_role"], music_description=segment["music_evidence"]["description"],
        negative_prompt=p["negative_prompt"], ending_keyframe=p["ending_keyframe"],
        **({"opening_keyframe":p["opening_keyframe"]} if segment["segment_id"] == "segment-001" else {}),
        storyboard=[{"shot_index":i, **{k:v for k,v in shot.items() if k not in ("start_ms", "end_ms")}}
                    for i,shot in enumerate(p["storyboard"])])


def section_content(segment):
    data = content(segment)
    for shot in data["storyboard"]:
        shot.pop("shot_index")
    return {"content": data}


class IncrementalAuthoringTests(unittest.TestCase):
    def test_reference_guidance_does_not_reset_each_section_to_source_scene(self):
        draft = create_draft(
            settings(), duration_ms=60_000,
            audio_url="https://media.example/song.mp3", audio_sha256="a" * 64,
            brief="Create a journey from this reference.", aspect_ratio="16:9",
            resolution="512p", output_language="en", shot_mode="multi-shot",
            shots_per_segment=4,
            reference_image={"reference_image_attachment_id": "att_" + "ab" * 32},
        )
        draft, errors = write_overview(draft, OverviewContent.model_validate(overview()))
        self.assertFalse(errors)
        with tempfile.TemporaryDirectory() as temp:
            working = checkpoint(MusicVideoPlanStore(Path(temp)), draft)["_meta"]["exoharness"]["workflow_continuation"]["repair_context"]
        guidance = working["authoring_guidance"]
        self.assertIn("not a reset frame", guidance)
        self.assertIn("Continue from the previous segment's ending", guidance)

    def test_user_specified_style_overview_needs_no_template_pack(self):
        custom = overview()
        custom["style_summary"] = "User brief: restrained ink-wash landscapes and warm lantern light."
        custom["provenance"] = {
            "planning_mode": "user-specified", "model_profile": "not-selected"
        }
        draft, errors = write_overview(
            skeleton(60_000), OverviewContent.model_validate(custom)
        )
        self.assertEqual(errors, [])
        self.assertEqual(draft["plan"]["provenance"]["planning_mode"], "user-specified")
        self.assertNotIn("pack_id", draft["plan"]["provenance"])

    def test_scope_only_after_global_inputs_are_saved(self):
        with tempfile.TemporaryDirectory() as temp:
            store = MusicVideoPlanStore(Path(temp))
            before = checkpoint(store, skeleton())
            self.assertNotIn("context_scope", before["_meta"]["exoharness"]["workflow_continuation"])
            draft, _ = write_overview(skeleton(), OverviewContent.model_validate(overview()))
            after = checkpoint(store, draft)["_meta"]["exoharness"]["workflow_continuation"]
            self.assertEqual(after["context_scope"], "checkpoint")
            self.assertEqual(after["repair_context"]["overview"], overview())
            self.assertEqual(after["repair_context"]["output_language"], "zh")
            self.assertEqual(len(after["repair_context"]["current_sections"]), 4)

    def test_batch_checkpoint_uses_public_input_shape_without_private_plan_nesting(self):
        draft, _ = write_overview(skeleton(), OverviewContent.model_validate(overview()))
        with tempfile.TemporaryDirectory() as temp:
            working = checkpoint(MusicVideoPlanStore(Path(temp)), draft)["_meta"]["exoharness"]["workflow_continuation"]["repair_context"]
        self.assertNotIn("required_fields", working)
        self.assertNotIn("prompt_plan", json.dumps(working))
        for item in working["current_sections"]:
            self.assertEqual(item["suggested_shot_count"], 4)
            self.assertEqual(item["minimum_shots"], 3)
        self.assertTrue(working["current_sections"][0]["opening_keyframe_required"])
        self.assertFalse(working["current_sections"][1]["opening_keyframe_required"])
        shape = working["input_shape_example"]["sections"]
        self.assertIsInstance(shape, list)
        self.assertIsInstance(shape[0]["content"]["storyboard"], list)
        SectionWrite.model_validate(shape[0])

    def test_repair_checkpoint_preserves_exact_public_fields_and_shot_addresses(self):
        draft, _ = write_overview(skeleton(60_000), OverviewContent.model_validate(overview()))
        first = content(valid_plan(60_000)["segments"][0])
        first["storyboard"][2].pop("lighting")
        draft, errors = write_segment(draft, "segment-001", SegmentContent.model_validate(first))
        with tempfile.TemporaryDirectory() as temp:
            working = checkpoint(MusicVideoPlanStore(Path(temp)), draft, errors)["_meta"]["exoharness"]["workflow_continuation"]["repair_context"]
        self.assertEqual(working["saved_content"], first)
        self.assertNotIn("prompt_plan", json.dumps(working))
        self.assertEqual(working["required_fields"], ["/storyboard/2/lighting"])

    def test_batch_rejects_ids_in_schema_and_invalid_windows_without_changes(self):
        draft, _ = write_overview(skeleton(60_000), OverviewContent.model_validate(overview()))
        original = copy.deepcopy(draft)
        sections = [SectionWrite.model_validate(section_content(s)) for s in valid_plan(60_000)["segments"]]
        for identifier in (None, "segment-002", "unknown"):
            with self.assertRaisesRegex(ValueError, "segment_id"):
                SectionWrite.model_validate({**section_content(valid_plan(60_000)["segments"][0]), "segment_id": identifier})
        for bad in ([], sections + sections):
            with self.assertRaisesRegex(ValueError, "consecutive"):
                queue_sections(draft, bad)
            self.assertEqual(draft, original)

    def test_batch_uses_only_array_order_without_rewriting_creative_data(self):
        draft, _ = write_overview(skeleton(60_000), OverviewContent.model_validate(overview()))
        original = copy.deepcopy(draft)
        sections = [SectionWrite.model_validate(section_content(s)) for s in valid_plan(60_000)["segments"]]
        expected = queue_sections(draft, sections)
        for index, section in enumerate(sections):
            saved = copy.deepcopy(expected["pending_sections"][f"segment-{index+1:03d}"])
            for shot in saved["storyboard"]:
                shot.pop("shot_index")
            self.assertEqual(saved, section.content.model_dump(exclude_unset=True, exclude_none=True))
        self.assertEqual(draft, original)
        self.assertFalse(drain_sections(expected)[1])

    def test_already_authored_sections_beyond_advertised_window_are_retained(self):
        draft = create_draft(settings(), duration_ms=315_000,
            audio_url="https://media.example/song.mp3", audio_sha256="a"*64,
            brief="One continuous story", aspect_ratio="16:9", resolution="512p",
            output_language="en", shot_mode="multi-shot", shots_per_segment=4,
            planning_batch_size=16,
            opening_frame={"first_frame_attachment_id": "att_" + "ab" * 32},
            duration_policy={"target_segment_duration_seconds": 15})
        draft, _ = write_overview(draft, OverviewContent.model_validate(overview()))
        self.assertEqual(len(draft["plan"]["segments"]), 21)
        self.assertEqual(len(section_window(draft)), 16)
        source = section_content(valid_plan(60_000)["segments"][0])
        sections = [SectionWrite.model_validate(copy.deepcopy(source)) for _ in range(21)]
        queued = queue_sections(draft, sections)
        self.assertEqual(len(queued["pending_sections"]), 21)
        self.assertIn("segment-021", queued["pending_sections"])
        self.assertEqual(queued["first_frame_attachment_id"], "att_" + "ab" * 32)
        self.assertEqual(draft.get("pending_sections"), None)

    def test_batch_limit_also_bounds_shot_count(self):
        draft = skeleton()
        for segment in draft["plan"]["segments"]:
            segment["prompt_plan"]["storyboard"] = [{} for _ in range(16)]
        self.assertEqual(len(section_window(draft)), 4)
        draft.pop("planning_batch_size")
        self.assertEqual(len(section_window(draft)), 1)

    def test_existing_draft_does_not_require_a_new_tool_outside_its_run_snapshot(self):
        draft, _ = write_overview(skeleton(), OverviewContent.model_validate(overview()))
        draft.pop("authoring_batch_version")
        with tempfile.TemporaryDirectory() as temp:
            step = checkpoint(MusicVideoPlanStore(Path(temp)), draft)
            self.assertEqual(step["next_tool"], "write_music_video_plan_segment")
            self.assertIn("current_section", step["_meta"]["exoharness"]["workflow_continuation"]["repair_context"])

    def test_service_builds_exact_skeleton_including_tail_merge(self):
        for duration, count, tail in ((313_992,11,14),(241_894,8,32),(30_000,1,30),(1000,1,5)):
            draft = skeleton(duration)
            segments = draft["plan"]["segments"]
            self.assertEqual(len(segments), count)
            self.assertEqual(segments[-1]["end_ms"], duration)
            self.assertEqual(segments[-1]["generation_duration_seconds"], tail)
            self.assertTrue(all(s["prompt_plan"]["controls"]["resolution"] == "512p" for s in segments))
            self.assertTrue(all(s["prompt_plan"]["controls"]["audio_enabled"] is False for s in segments))

    def test_missing_overview_field_is_saved_then_patched_without_rewriting(self):
        partial = overview()
        subject = partial["continuity_anchors"].pop("subject_identity")
        draft, errors = write_overview(skeleton(), OverviewContent.model_validate(partial))
        self.assertEqual([e["path"] for e in errors], ["/continuity_anchors/subject_identity"])
        before = copy.deepcopy(draft["overview"])
        draft, errors = write_overview(draft, OverviewContent(continuity_anchors={"subject_identity":subject}))
        self.assertFalse(errors)
        self.assertEqual(draft["overview"]["narrative_arc"], before["narrative_arc"])
        self.assertEqual(draft["overview"]["continuity_anchors"]["camera_language"], before["continuity_anchors"]["camera_language"])

    def test_one_missing_shot_field_only_requires_that_field(self):
        draft, _ = write_overview(skeleton(60_000), OverviewContent.model_validate(overview()))
        first = content(valid_plan(60_000)["segments"][0])
        light = first["storyboard"][2].pop("lighting")
        draft, errors = write_segment(draft, "segment-001", SegmentContent.model_validate(first))
        self.assertEqual(draft["accepted_segments"], 0)
        self.assertIn("/storyboard/2/lighting", [e["path"] for e in errors])
        saved = copy.deepcopy(draft["plan"]["segments"][0])
        draft, errors = write_segment(draft, "segment-001", SegmentContent.model_validate({"storyboard":[{"shot_index":2,"lighting":light}]}))
        self.assertFalse(errors)
        self.assertEqual(draft["accepted_segments"], 1)
        self.assertEqual(saved["prompt_plan"]["storyboard"][0], draft["plan"]["segments"][0]["prompt_plan"]["storyboard"][0])
        locked, errors = write_segment(draft, "segment-001", SegmentContent(visual_role="overwrite"))
        self.assertTrue(errors)
        self.assertEqual(locked, draft)

    def test_entire_11_segment_plan_survives_reopen_and_small_checkpoints(self):
        raw = valid_plan(313_992)
        with tempfile.TemporaryDirectory() as temp:
            runtime = SimpleNamespace(settings=settings(), plans=MusicVideoPlanStore(Path(temp)))
            draft, errors = write_overview(skeleton(), OverviewContent.model_validate(overview()))
            self.assertFalse(errors)
            selected_overview = copy.deepcopy(draft["overview"])
            receipts = []
            for i, expected in enumerate(raw["segments"]):
                cp = checkpoint(runtime.plans, draft)
                text = json.dumps(cp, ensure_ascii=False)
                self.assertLess(len(text), 6500)
                # Field names may occur in the example, not accepted creative prose.
                if i:
                    self.assertNotIn(raw["segments"][i-1]["prompt_plan"]["storyboard"][0]["subject_action"], text)
                self.assertEqual(cp["accepted_segments"], i)
                # Every section gets the SAME work-wide style and narrative,
                # including after reopening the persisted draft.
                working = cp["_meta"]["exoharness"]["workflow_continuation"]["repair_context"]
                self.assertEqual(working["overview"], selected_overview)
                runtime.plans = MusicVideoPlanStore(Path(temp))
                restored = runtime.plans.get_draft(cp["draft_id"], cp["draft_digest"])
                self.assertEqual(restored["overview"], selected_overview)
                before = copy.deepcopy(restored["plan"]["segments"][:i])
                draft, errors = write_segment(restored, expected["segment_id"], SegmentContent.model_validate(content(expected)))
                self.assertFalse(errors)
                self.assertEqual(draft["overview"], selected_overview)
                self.assertEqual(before, draft["plan"]["segments"][:i])
                # Identical delivery from an immutable checkpoint is deterministic.
                repeated, _ = write_segment(restored, expected["segment_id"], SegmentContent.model_validate(content(expected)))
                self.assertEqual(repeated, draft)
                receipts.append(_finish_authoring(runtime, draft, errors))
            final = receipts[-1]
            self.assertTrue(final["valid"])
            self.assertEqual(final["segment_count"], 11)
            required = final["_meta"]["exoharness"]["workflow_continuation"]["required_arguments"]
            self.assertEqual(final["_meta"]["exoharness"]["workflow_continuation"]["context_scope"], "checkpoint")
            self.assertEqual(required["audio_sha256"], "a"*64)
            self.assertEqual(required["audio_url"], "https://media.example/song.mp3")
            saved = runtime.plans.get(final["plan_digest"])
            for i, segment in enumerate(saved["segments"]):
                self.assertEqual(segment["storyboard_shot_count"], 4)
                if i:
                    self.assertEqual(segment["opening_keyframe"], saved["segments"][i-1]["ending_keyframe"])

    def test_out_of_order_duplicate_and_invalid_slots_do_not_change_saved_draft(self):
        draft, _ = write_overview(skeleton(60_000), OverviewContent.model_validate(overview()))
        for segment, data in (("segment-002", {"visual_role":"later"}),
                              ("segment-001", {"storyboard":[{"shot_index":12,"lighting":"test"}]})):
            after, errors = write_segment(draft, segment, SegmentContent.model_validate(data))
            self.assertTrue(errors)
            self.assertEqual(after, draft)
        with self.assertRaisesRegex(ValueError, "Extra inputs"):
            OverviewContent(continuity_anchors={"invented":"bad"})


class AuthoringProtocolTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.clip_policy = patch.object(server, "live_clip_policy", AsyncMock(return_value={
            "live_min_duration_seconds": 5,
            "live_max_duration_seconds": 60,
            "target_segment_duration_seconds": 30,
            "duration_step_seconds": 1,
            "tail_merge_max_duration_seconds": 40,
        }))
        self.clip_policy.start()
        self.addCleanup(self.clip_policy.stop)

    async def test_reference_requirement_is_checked_before_audio_or_draft(self):
        with tempfile.TemporaryDirectory() as temp:
            configured = settings()
            configured.api_token = None
            runtime = SimpleNamespace(settings=configured, plans=MusicVideoPlanStore(Path(temp)))
            probe = AsyncMock()
            with patch.object(server, "_runtime", runtime), patch.object(
                server, "probe_media", probe
            ), patch.object(
                server, "live_clip_policy",
                AsyncMock(side_effect=server.UpstreamToolError(
                    "no compatible generator accepts reference_image_url"
                )),
            ) as policy:
                with self.assertRaisesRegex(server.ToolError, "reference_image_url"):
                    await server.mcp.call_tool("create_music_video_plan", {
                        "audio_url": "https://media.example/song.mp3",
                        "brief": "Use the described subject and station setting.",
                        "reference_image_attachment_id": "att_" + "ab" * 32,
                    }, context=SimpleNamespace(headers={}))
            policy.assert_awaited_once_with(
                configured, {},
                required_submit_fields=("first_frame_url", "reference_image_url"),
            )
            probe.assert_not_awaited()
            self.assertEqual(list(runtime.plans.draft_root.glob("*.json")), [])

    async def test_mv_duration_limit_rejects_before_persisting_a_draft(self):
        with tempfile.TemporaryDirectory() as temp:
            configured = settings()
            configured.api_token = None
            runtime = SimpleNamespace(settings=configured, plans=MusicVideoPlanStore(Path(temp)))
            context = SimpleNamespace(headers={})
            media = {"duration_seconds": 360.001, "sha256": "a" * 64,
                     "streams": [{"codec_type": "audio"}]}
            with patch.object(server, "_runtime", runtime), patch.object(
                server, "probe_media", AsyncMock(return_value=media)
            ):
                with self.assertRaisesRegex(server.ToolError, "MV最长支持 6 分钟.*未创建计划"):
                    await server.mcp.call_tool("create_music_video_plan", {
                        "audio_url": "https://media.example/song.mp3",
                        "brief": "One coherent style.",
                    }, context=context)
            self.assertEqual(list(runtime.plans.draft_root.glob("*.json")), [])

    async def test_mv_duration_limit_accepts_exactly_six_minutes(self):
        with tempfile.TemporaryDirectory() as temp:
            configured = settings()
            configured.api_token = None
            runtime = SimpleNamespace(settings=configured, plans=MusicVideoPlanStore(Path(temp)))
            context = SimpleNamespace(headers={})
            media = {"duration_seconds": 360, "sha256": "a" * 64,
                     "streams": [{"codec_type": "audio"}]}
            with patch.object(server, "_runtime", runtime), patch.object(
                server, "probe_media", AsyncMock(return_value=media)
            ):
                result = await server.mcp.call_tool("create_music_video_plan", {
                    "audio_url": "https://media.example/song.mp3",
                    "brief": "One coherent style.",
                }, context=context)
            self.assertFalse(result.is_error, str(result))
            continuation = result.structured_content["_meta"]["exoharness"]["workflow_continuation"]
            self.assertEqual(continuation["repair_context"]["master_duration_ms"], 360000)

    async def test_full_writes_accept_fewer_shots_and_allocate_exact_timing(self):
        for batched in (False, True):
            for count in (3, 4):
                with self.subTest(batched=batched, count=count), tempfile.TemporaryDirectory() as temp:
                    configured = settings()
                    configured.api_token = None
                    runtime = SimpleNamespace(settings=configured, plans=MusicVideoPlanStore(Path(temp)))
                    draft, _ = write_overview(skeleton(30_000), OverviewContent.model_validate(overview()))
                    step = checkpoint(runtime.plans, draft)
                    data = content(valid_plan(30_000)["segments"][0])
                    data["storyboard"] = data["storyboard"][:count]
                    arguments = {key:step[key] for key in ("draft_id", "draft_digest")}
                    if batched:
                        for shot in data["storyboard"]:
                            shot.pop("shot_index")
                        name = "write_music_video_plan_sections"
                        arguments["sections"] = [{"content":data}]
                    else:
                        name = "write_music_video_plan_segment"
                        arguments.update(segment_id="segment-001", content=data)
                    original = copy.deepcopy(arguments)
                    with patch.object(server, "_runtime", runtime):
                        result = await server.mcp.call_tool(name, arguments, context=SimpleNamespace(headers={}))
                    self.assertFalse(result.is_error, str(result))
                    self.assertTrue(result.structured_content["valid"], str(result))
                    self.assertEqual(arguments, original)
                    saved = runtime.plans.get(result.structured_content["plan_digest"])
                    self.assertEqual(saved["segments"][0]["storyboard_shot_count"], count)
                    self.assertEqual(saved["segments"][0]["generation_duration_seconds"], 30)

    async def test_three_shot_floor_and_focused_repair_preserve_saved_fields(self):
        draft = create_draft(settings(), duration_ms=30_000, audio_url="https://media.example/song.mp3",
            audio_sha256="a"*64, brief="A quiet scene", aspect_ratio="16:9", resolution="512p",
            output_language="zh", shot_mode="multi-shot", shots_per_segment=2)
        self.assertEqual(len(draft["plan"]["segments"][0]["prompt_plan"]["storyboard"]), 3)
        draft, _ = write_overview(draft, OverviewContent.model_validate(overview()))
        data = content(valid_plan(30_000)["segments"][0])
        data["storyboard"] = data["storyboard"][:3]
        data["storyboard"][2].pop("lighting")
        partial, errors = write_segment(draft, "segment-001", SegmentContent.model_validate(data), complete=True)
        self.assertEqual([e["path"] for e in errors], ["/storyboard/2/lighting"])
        first = copy.deepcopy(partial["plan"]["segments"][0]["prompt_plan"]["storyboard"][0])
        patched, errors = write_segment(partial, "segment-001", SegmentContent(storyboard=[{"shot_index":2,"lighting":"Warm daylight"}]))
        self.assertFalse(errors)
        shots = patched["plan"]["segments"][0]["prompt_plan"]["storyboard"]
        self.assertEqual(len(shots), 3)
        self.assertEqual(shots[0], first)
        self.assertEqual([(s["start_ms"],s["end_ms"]) for s in shots], [(0,10000),(10000,20000),(20000,30000)])

    async def test_two_shots_require_only_the_missing_third_shot(self):
        draft, _ = write_overview(skeleton(30_000), OverviewContent.model_validate(overview()))
        data = content(valid_plan(30_000)["segments"][0])
        original = copy.deepcopy(data["storyboard"][:2])
        data["storyboard"] = copy.deepcopy(original)
        partial, errors = write_segment(draft, "segment-001", SegmentContent.model_validate(data), complete=True)
        self.assertEqual(partial["accepted_segments"], 0)
        self.assertTrue(errors)
        self.assertTrue(all(e["path"].startswith("/storyboard/2/") for e in errors), errors)
        for actual, expected in zip(partial["plan"]["segments"][0]["prompt_plan"]["storyboard"], original):
            for key, value in expected.items():
                if key != "shot_index":
                    self.assertEqual(actual[key], value)
        third = content(valid_plan(30_000)["segments"][0])["storyboard"][2]
        repaired, errors = write_segment(partial, "segment-001", SegmentContent(storyboard=[third]))
        self.assertFalse(errors)
        self.assertEqual(repaired["accepted_segments"], 1)
        self.assertEqual(len(repaired["plan"]["segments"][0]["prompt_plan"]["storyboard"]), 3)

    async def test_short_first_section_preserves_later_section_until_local_repair(self):
        with tempfile.TemporaryDirectory() as temp:
            configured = settings()
            configured.api_token = None
            runtime = SimpleNamespace(settings=configured, plans=MusicVideoPlanStore(Path(temp)))
            draft, _ = write_overview(skeleton(60_000), OverviewContent.model_validate(overview()))
            step = checkpoint(runtime.plans, draft)
            sections = [section_content(s) for s in valid_plan(60_000)["segments"]]
            missing = content(valid_plan(60_000)["segments"][0])["storyboard"][2]
            sections[0]["content"]["storyboard"] = sections[0]["content"]["storyboard"][:2]
            sections[1]["content"]["storyboard"] = sections[1]["content"]["storyboard"][:3]
            with patch.object(server, "_runtime", runtime):
                response = await server.mcp.call_tool("write_music_video_plan_sections", {
                    **{k:step[k] for k in ("draft_id","draft_digest")}, "sections":sections,
                }, context=SimpleNamespace(headers={}))
                self.assertFalse(response.is_error)
                step = response.structured_content
                directive = step["_meta"]["exoharness"]["workflow_continuation"]
                self.assertEqual(directive["required_tool"], "patch_music_video_plan_segment")
                self.assertEqual(directive["repair_context"]["saved_later_sections"], ["segment-002"])
                self.assertTrue(all(e["path"].startswith("/storyboard/2/") for e in step["validation_errors"]))
                response = await server.mcp.call_tool(directive["required_tool"], {
                    **directive["required_arguments"], "content":{"storyboard":[missing]},
                }, context=SimpleNamespace(headers={}))
                self.assertFalse(response.is_error)
                self.assertTrue(response.structured_content["valid"], str(response))
                saved = runtime.plans.get(response.structured_content["plan_digest"])
                self.assertEqual([s["storyboard_shot_count"] for s in saved["segments"]], [3,3])

    @unittest.skipUnless(os.getenv("EXOHARNESS_BATCH_CONTEXT_REPLAY"), "requires private 150k-token trace; no inference or generation")
    async def test_recorded_batch_errors_accept_exact_creative_payloads_and_reach_render(self):
        trace = json.loads(Path(os.environ["EXOHARNESS_BATCH_CONTEXT_REPLAY"]).read_text())
        records = trace["invocations"]
        self.assertEqual(len(records), 12)
        with tempfile.TemporaryDirectory() as temp:
            configured = settings()
            configured.api_token = None
            runtime = SimpleNamespace(settings=configured, plans=MusicVideoPlanStore(Path(temp)))
            context = SimpleNamespace(headers={})
            measured = {"duration_seconds":313.992,"sha256":"a"*64,"streams":[{"codec_type":"audio"}]}
            originals = []
            with patch.object(server, "_runtime", runtime), patch.object(server, "probe_media", AsyncMock(return_value=measured)):
                result = await server.mcp.call_tool("create_music_video_plan", records[2]["arguments"], context=context)
                step = result.structured_content
                result = await server.mcp.call_tool("write_music_video_plan_overview", {
                    **step["_meta"]["exoharness"]["workflow_continuation"]["required_arguments"],
                    "overview":records[3]["arguments"]["overview"],
                },context=context)
                step = result.structured_content
                for record_index in (5,7,10):
                    # Use the FIRST native arrays, including the actual reversed
                    # IDs and omitted IDs. No regenerated or repaired prose.
                    proposed = records[record_index]["arguments"]["sections"]
                    original = copy.deepcopy(proposed)
                    ordered = sorted(proposed, key=lambda s:s["segment_id"]) if all(s.get("segment_id") for s in proposed) else proposed
                    originals.extend(s["content"] for s in ordered)
                    continuation = step["_meta"]["exoharness"]["workflow_continuation"]
                    result = await server.mcp.call_tool(continuation["required_tool"], {
                        **continuation["required_arguments"],"sections":[{"content":s["content"]} for s in ordered],
                    }, context=context)
                    self.assertFalse(result.is_error)
                    self.assertEqual(proposed, original)
                    step = result.structured_content
                    if record_index == 5:
                        # This original rejected batch also lacked its first
                        # opening. Keep the four saved sections; reuse only the
                        # exact opening present in the later recorded correction.
                        self.assertEqual([e["path"] for e in step["validation_errors"]],["/opening_keyframe"])
                        repair = step["_meta"]["exoharness"]["workflow_continuation"]
                        opening = records[6]["arguments"]["sections"][0]["content"]["opening_keyframe"]
                        result = await server.mcp.call_tool(repair["required_tool"], {
                            **repair["required_arguments"],"content":{"opening_keyframe":opening},
                        },context=context)
                        self.assertFalse(result.is_error)
                        step = result.structured_content
                    self.assertFalse(step.get("validation_errors"))
                self.assertTrue(step["valid"])
                self.assertEqual(step["segment_count"],11)
                continuation = step["_meta"]["exoharness"]["workflow_continuation"]
                self.assertEqual(continuation["context_scope"],"checkpoint")
                self.assertEqual(continuation["required_tool"],"render_validated_music_video")
                self.assertEqual(set(continuation["required_arguments"]), {"plan_digest","audio_url","audio_sha256"})
                saved = runtime.plans.get(step["plan_digest"])
                for actual, old in zip(saved["segments"], originals, strict=True):
                    self.assertEqual(actual["ending_keyframe"],old["ending_keyframe"])
                    self.assertIn(old["visual_role"],actual["positive_prompt"])
                    # Rendered prompt contains the unmodified concrete shots.
                    for shot in old["storyboard"]:
                        for name, text in shot.items():
                            if name != "weight":
                                self.assertIn(text,actual["positive_prompt"])
                output = os.getenv("EXOHARNESS_TOKEN_REPLAY_OUTPUT")
                if output:
                    tools = await server.mcp.list_tools()
                    Path(output).write_text(json.dumps({"tools":[{"name":"media-workflow-"+t.name,"description":t.description,"parameters":t.input_schema} for t in tools],"render_checkpoint":step}),encoding="utf-8")
                print("BATCH_REPLAY: 11 recorded sections accepted in 3 writes plus 1 opening-field patch; first rejected native arrays reused; creative shots unchanged; render inputs bound; no generation")

    @unittest.skipUnless(os.getenv("EXOHARNESS_CONTEXT_REPLAY"), "requires private recorded trace; no inference or generation")
    async def test_recorded_creative_content_uses_three_batches_without_rewriting(self):
        trace = json.loads(Path(os.environ["EXOHARNESS_CONTEXT_REPLAY"]).read_text())
        records = trace["invocations"]
        creation = next(r for r in records if r["tool"].endswith(".create_music_video_plan"))
        global_write = next(r for r in records if r["tool"].endswith(".write_music_video_plan_overview"))
        duration = creation["result"]["details"]["_meta"]["exoharness"]["workflow_continuation"]["repair_context"]["master_duration_ms"]
        original = {}
        for record in records:
            if record["tool"].endswith(".write_music_video_plan_segment"):
                args = record["arguments"]
                # The first segment-006 proposal had no mechanical shot indexes.
                # The new ordered full-write schema must accept its creative data.
                original.setdefault(args["segment_id"], copy.deepcopy(args["content"]))
        with tempfile.TemporaryDirectory() as temp:
            configured = settings()
            configured.api_token = None
            runtime = SimpleNamespace(settings=configured, plans=MusicVideoPlanStore(Path(temp)))
            context = SimpleNamespace(headers={})
            measured = {"duration_seconds":duration/1000,"sha256":"a"*64,"streams":[{"codec_type":"audio"}]}
            with patch.object(server, "_runtime", runtime), patch.object(server, "probe_media", AsyncMock(return_value=measured)):
                result = await server.mcp.call_tool("create_music_video_plan", creation["arguments"], context=context)
                step = result.structured_content
                result = await server.mcp.call_tool("write_music_video_plan_overview", {
                    **step["_meta"]["exoharness"]["workflow_continuation"]["required_arguments"],
                    "overview":global_write["arguments"]["overview"],
                },context=context)
                step = result.structured_content
                ids = list(original)
                writes = 0
                for start in range(0,len(ids),4):
                    sections = []
                    for segment_id in ids[start:start+4]:
                        data = copy.deepcopy(original[segment_id])
                        for shot in data["storyboard"]:
                            shot.pop("shot_index",None)
                        sections.append({"content":data})
                    continuation = step["_meta"]["exoharness"]["workflow_continuation"]
                    self.assertEqual(continuation["required_tool"],"write_music_video_plan_sections")
                    result = await server.mcp.call_tool(continuation["required_tool"], {
                        **continuation["required_arguments"],"sections":sections,
                    },context=context)
                    step = result.structured_content
                    self.assertFalse(result.is_error)
                    self.assertFalse(step["validation_errors"])
                    writes += 1
                self.assertEqual((writes,step["accepted_segments"],step["segment_count"]),(3,10,11))
                restored = runtime.plans.get_draft(step["draft_id"],step["draft_digest"])
                self.assertEqual(restored["overview"],global_write["arguments"]["overview"])
                for i,segment in enumerate(restored["plan"]["segments"][:10]):
                    proposed = original[segment["segment_id"]]
                    self.assertEqual(segment["visual_role"],proposed["visual_role"])
                    self.assertEqual(segment["prompt_plan"]["ending_keyframe"],proposed["ending_keyframe"])
                    for actual,old in zip(segment["prompt_plan"]["storyboard"],proposed["storyboard"],strict=True):
                        for key,value in old.items():
                            if key not in ("shot_index","weight"):
                                self.assertEqual(actual[key],value)
                    if i:
                        self.assertEqual(segment["prompt_plan"]["opening_keyframe"],restored["plan"]["segments"][i-1]["prompt_plan"]["ending_keyframe"])
                print("RECORDED_MCP_REPLAY: 10 recorded sections accepted in 3 batches; creative data unchanged; last section absent in original stopped Run; no generation")

    async def test_missing_first_opening_uses_only_one_field_patch(self):
        with tempfile.TemporaryDirectory() as temp:
            configured = settings()
            configured.api_token = None
            runtime = SimpleNamespace(settings=configured, plans=MusicVideoPlanStore(Path(temp)))
            draft, _ = write_overview(skeleton(30_000), OverviewContent.model_validate(overview()))
            ref = checkpoint(runtime.plans, draft)
            args = ref["_meta"]["exoharness"]["workflow_continuation"]["required_arguments"]
            full = content(valid_plan(30_000)["segments"][0])
            opening = full.pop("opening_keyframe")
            with patch.object(server, "_runtime", runtime):
                partial = await server.mcp.call_tool("write_music_video_plan_segment", {**args,"content":full},context=SimpleNamespace(headers={}))
                payload = partial.structured_content
                self.assertEqual(payload["next_tool"], "patch_music_video_plan_segment")
                self.assertEqual(payload["validation_errors"][0]["path"], "/opening_keyframe")
                args = payload["_meta"]["exoharness"]["workflow_continuation"]["required_arguments"]
                repaired = await server.mcp.call_tool("patch_music_video_plan_segment", {**args,"content":{"opening_keyframe":opening}},context=SimpleNamespace(headers={}))
                self.assertTrue(repaired.structured_content["valid"])

    async def test_legacy_patch_cannot_rewrite_unreported_fields(self):
        with tempfile.TemporaryDirectory() as temp:
            configured = settings()
            configured.api_token = None
            runtime = SimpleNamespace(settings=configured, plans=MusicVideoPlanStore(Path(temp)))
            raw = valid_plan(60_000)
            raw["segments"][1]["visual_role"] = raw["segments"][0]["visual_role"]
            current = server._store_validated_or_draft(runtime, server.MusicVideoPlan.model_validate(raw))
            arguments = {k:current[k] for k in ("draft_id", "draft_digest")}
            with patch.object(server, "_runtime", runtime):
                with self.assertRaisesRegex(server.ToolError, "only exact paths"):
                    await server.mcp.call_tool("patch_music_video_plan", {**arguments, "patches":[
                        {"op":"replace", "path":"/continuity_anchors/subject_identity", "value":"unnecessary rewrite"}
                    ]}, context=SimpleNamespace(headers={}))
                accepted = await server.mcp.call_tool("patch_music_video_plan", {**arguments, "patches":[
                    {"op":"replace", "path":"/segments/1/visual_role", "value":"A distinct second chapter"}
                ]}, context=SimpleNamespace(headers={}))
                self.assertFalse(accepted.is_error)
                self.assertTrue(accepted.structured_content["valid"])

    async def test_typed_mcp_calls_create_fill_and_validate_without_generation(self):
        with tempfile.TemporaryDirectory() as temp:
            configured = settings()
            configured.api_token = None
            runtime = SimpleNamespace(settings=configured, plans=MusicVideoPlanStore(Path(temp)))
            context = SimpleNamespace(headers={})
            async def call(name, arguments):
                result = await server.mcp.call_tool(name, arguments, context=context)
                self.assertFalse(result.is_error, str(result))
                return result.structured_content
            measured = {"duration_seconds":60.0,"sha256":"a"*64,"streams":[{"codec_type":"audio"}]}
            with patch.object(server, "_runtime", runtime), patch.object(server, "probe_media", AsyncMock(return_value=measured)) as probe:
                step = await call("create_music_video_plan", {"audio_url":"https://media.example/song.mp3", "brief":"Return home, one pastoral style."})
                self.assertEqual(step["segment_count"], 2)
                args = step["_meta"]["exoharness"]["workflow_continuation"]["required_arguments"]
                step = await call("write_music_video_plan_overview", {**args, "overview":overview()})
                for segment in valid_plan(60_000)["segments"]:
                    args = step["_meta"]["exoharness"]["workflow_continuation"]["required_arguments"]
                    step = await call("write_music_video_plan_segment", {**args,"segment_id":segment["segment_id"],"content":content(segment)})
                self.assertTrue(step["valid"])
                self.assertEqual(step["_meta"]["exoharness"]["workflow_continuation"]["required_tool"], "render_validated_music_video")
                probe.assert_awaited_once()

    async def test_live_schema_uses_native_sections_not_encoded_json(self):
        tools = {tool.name:tool for tool in await server.mcp.list_tools()}
        segment = tools["write_music_video_plan_segment"].input_schema
        self.assertIn("content", segment["required"])
        self.assertNotIn("ctx", segment["properties"])
        overview_schema = tools["write_music_video_plan_overview"].input_schema
        self.assertIn("overview", overview_schema["required"])
        self.assertIn("read-only-resources", json.dumps(overview_schema))
        self.assertIn("subject_identity", json.dumps(overview_schema))
        required = segment["$defs"]["CompleteSegment"]["required"]
        self.assertIn("ending_keyframe", required)
        self.assertIn("negative_prompt", required)
        patch_schema = tools["patch_music_video_plan_segment"].input_schema
        self.assertNotIn("required", patch_schema["$defs"]["SegmentContent"])
        batch_schema = tools["write_music_video_plan_sections"].input_schema
        self.assertEqual(batch_schema["properties"]["sections"]["maxItems"], 64)
        self.assertEqual(set(batch_schema["$defs"]["SectionWrite"]["properties"]), {"content"})
        self.assertFalse(batch_schema["$defs"]["SectionWrite"]["additionalProperties"])
        self.assertNotIn("shot_index", batch_schema["$defs"]["OrderedShot"]["properties"])
        self.assertIn("subject_action", batch_schema["$defs"]["OrderedShot"]["required"])

    async def test_eleven_segments_use_three_batches_and_one_style_without_generation(self):
        with tempfile.TemporaryDirectory() as temp:
            configured = settings()
            configured.api_token = None
            runtime = SimpleNamespace(settings=configured, plans=MusicVideoPlanStore(Path(temp)))
            context = SimpleNamespace(headers={})
            draft, _ = write_overview(skeleton(), OverviewContent.model_validate(overview()))
            step = checkpoint(runtime.plans, draft)
            segments = valid_plan(313_992)["segments"]
            with patch.object(server, "_runtime", runtime):
                for start in (0, 4, 8):
                    continuation = step["_meta"]["exoharness"]["workflow_continuation"]
                    self.assertEqual(continuation["required_tool"], "write_music_video_plan_sections")
                    self.assertEqual(continuation["repair_context"]["overview"], overview())
                    result = await server.mcp.call_tool(continuation["required_tool"], {
                        **continuation["required_arguments"],
                        "sections": [section_content(s) for s in segments[start:start+4]],
                    }, context=context)
                    self.assertFalse(result.is_error)
                    step = result.structured_content
                    runtime.plans = MusicVideoPlanStore(Path(temp))
                self.assertTrue(step["valid"])
                self.assertEqual(step["segment_count"], 11)
                saved = runtime.plans.get(step["plan_digest"])
                for index, segment in enumerate(saved["segments"]):
                    self.assertEqual(segment["storyboard_shot_count"], 4)
                    if index:
                        self.assertEqual(segment["opening_keyframe"], saved["segments"][index-1]["ending_keyframe"])

    async def test_default_whole_song_writes_eleven_sections_once_with_small_batch_option(self):
        for requested_size, expected_window in ((None, 11), (2, 2)):
            with self.subTest(batch=requested_size), tempfile.TemporaryDirectory() as temp:
                configured = settings()
                configured.api_token = None
                runtime = SimpleNamespace(settings=configured, plans=MusicVideoPlanStore(Path(temp)))
                ctx = SimpleNamespace(headers={})
                with patch.object(server,"_runtime",runtime), patch.object(server,"probe_media", AsyncMock(return_value={"duration_seconds":313.992,"sha256":"a"*64,"streams":[{"codec_type":"audio"}]})):
                    args = {"audio_url":"https://media.example/song.mp3","brief":"Homecoming"}
                    if requested_size is not None:
                        args["planning_batch_size"] = requested_size
                    result = await server.mcp.call_tool("create_music_video_plan",args,context=ctx)
                    step = result.structured_content
                    result = await server.mcp.call_tool("write_music_video_plan_overview", {**step["_meta"]["exoharness"]["workflow_continuation"]["required_arguments"],"overview":overview()},context=ctx)
                    step = result.structured_content
                    work = step["_meta"]["exoharness"]["workflow_continuation"]["repair_context"]
                    self.assertEqual(work["clip_duration_seconds"], 30)
                    self.assertEqual(len(work["current_sections"]),expected_window)
                    if requested_size is None:
                        result = await server.mcp.call_tool("write_music_video_plan_sections", {**step["_meta"]["exoharness"]["workflow_continuation"]["required_arguments"],"sections":[section_content(s) for s in valid_plan(313_992)["segments"]]},context=ctx)
                        self.assertFalse(result.is_error)
                        self.assertTrue(result.structured_content["valid"])
                        self.assertEqual(result.structured_content["segment_count"],11)

    async def test_create_plan_windows_follow_live_15s_default_not_hardcoded_30(self):
        policy = {
            "live_min_duration_seconds": 5,
            "live_max_duration_seconds": 60,
            "target_segment_duration_seconds": 15,
            "duration_step_seconds": 1,
            "tail_merge_max_duration_seconds": 21,
        }
        with tempfile.TemporaryDirectory() as temp, patch.object(
            server, "live_clip_policy", AsyncMock(return_value=policy)
        ):
            configured = settings()
            configured.api_token = None
            runtime = SimpleNamespace(settings=configured, plans=MusicVideoPlanStore(Path(temp)))
            ctx = SimpleNamespace(headers={})
            with patch.object(server, "_runtime", runtime), patch.object(
                server, "probe_media", AsyncMock(return_value={
                    "duration_seconds": 246.0, "sha256": "a" * 64,
                    "streams": [{"codec_type": "audio"}],
                })
            ):
                result = await server.mcp.call_tool("create_music_video_plan", {
                    "audio_url": "https://media.example/song.mp3",
                    "brief": "Homecoming",
                    "first_frame_attachment_id": "att_" + "ab" * 32,
                }, context=ctx)
            step = result.structured_content
            work = step["_meta"]["exoharness"]["workflow_continuation"]["repair_context"]
            self.assertEqual(work["clip_duration_seconds"], 15)
            self.assertEqual(work["tail_merge_max_seconds"], 21)
            self.assertEqual(step["segment_count"], 16)
            draft = runtime.plans.get_draft(step["draft_id"], step["draft_digest"])
            self.assertEqual(draft["first_frame_attachment_id"], "att_" + "ab" * 32)
            last = draft["plan"]["segments"][-1]
            self.assertEqual((last["start_ms"], last["end_ms"]), (225_000, 246_000))
            self.assertEqual(last["generation_duration_seconds"], 21)

    async def test_live_clip_policy_reads_video_generator_schema_default(self):
        from contextlib import asynccontextmanager

        class FakeGenerator:
            def __init__(self, default):
                self.default = default

            async def submission_schema(self):
                return {"properties": {"duration_seconds": {
                    "type": "number", "default": self.default, "minimum": 5, "maximum": 60,
                }}}

            async def live_default_duration(self):
                return None

        self.clip_policy.stop()
        try:
            for default, tail in ((15, 21), (30, 42)):
                with self.subTest(default=default):
                    @asynccontextmanager
                    async def fake_connect(_settings, _request_headers=None, **_kwargs):
                        yield FakeGenerator(default)

                    with patch.object(server, "connect_video_generator", fake_connect):
                        policy = await server.live_clip_policy(SimpleNamespace(
                            video_generator_duration_step_seconds=1,
                        ))
                    self.assertEqual(policy["target_segment_duration_seconds"], default)
                    self.assertEqual(policy["tail_merge_max_duration_seconds"], tail)
        finally:
            self.clip_policy.start()

    async def test_live_clip_policy_uses_mcp_status_duration_when_schema_default_is_null(self):
        from contextlib import asynccontextmanager

        class FakeGenerator:
            async def submission_schema(self):
                return {"properties": {"duration_seconds": {
                    "anyOf": [
                        {"type": "number", "minimum": 5, "maximum": 60,
                         "description": "Omit to use FASTVIDEO_MCP_DEFAULT_DURATION_SECONDS, which defaults to 30."},
                        {"type": "null"},
                    ],
                    "default": None,
                }}}

            async def live_default_duration(self):
                return 15

        self.clip_policy.stop()
        try:
            @asynccontextmanager
            async def fake_connect(_settings, _request_headers=None, **_kwargs):
                yield FakeGenerator()

            with patch.object(server, "connect_video_generator", fake_connect):
                policy = await server.live_clip_policy(SimpleNamespace(
                    video_generator_duration_step_seconds=1,
                    video_generator_target_duration_seconds=30,
                    video_generator_min_duration_seconds=5,
                    video_generator_max_duration_seconds=60,
                ))
            self.assertEqual(policy["target_segment_duration_seconds"], 15)
            self.assertEqual(policy["tail_merge_max_duration_seconds"], 21)
        finally:
            self.clip_policy.start()

    async def test_live_clip_policy_falls_back_to_15_when_mcp_omits_duration(self):
        from contextlib import asynccontextmanager

        class FakeGenerator:
            async def submission_schema(self):
                return {"properties": {"duration_seconds": {
                    "anyOf": [
                        {"type": "number", "minimum": 5, "maximum": 60,
                         "description": "Omit to use FASTVIDEO_MCP_DEFAULT_DURATION_SECONDS, which defaults to 30."},
                        {"type": "null"},
                    ],
                    "default": None,
                }}}

            async def live_default_duration(self):
                return None

        self.clip_policy.stop()
        try:
            @asynccontextmanager
            async def fake_connect(_settings, _request_headers=None, **_kwargs):
                yield FakeGenerator()

            with patch.object(server, "connect_video_generator", fake_connect):
                policy = await server.live_clip_policy(SimpleNamespace(
                    video_generator_duration_step_seconds=1,
                    video_generator_target_duration_seconds=30,
                    video_generator_min_duration_seconds=5,
                    video_generator_max_duration_seconds=60,
                ))
            self.assertEqual(policy["target_segment_duration_seconds"], 15)
            self.assertEqual(policy["tail_merge_max_duration_seconds"], 21)
        finally:
            self.clip_policy.start()

    async def test_batch_repair_reuses_saved_later_sections_after_reopen(self):
        with tempfile.TemporaryDirectory() as temp:
            configured = settings()
            configured.api_token = None
            runtime = SimpleNamespace(settings=configured, plans=MusicVideoPlanStore(Path(temp)))
            draft, _ = write_overview(skeleton(90_000), OverviewContent.model_validate(overview()))
            step = checkpoint(runtime.plans, draft)
            sections = [section_content(s) for s in valid_plan(90_000)["segments"]]
            opening = sections[0]["content"].pop("opening_keyframe")
            context = SimpleNamespace(headers={})
            with patch.object(server, "_runtime", runtime):
                result = await server.mcp.call_tool("write_music_video_plan_sections", {
                    **step["_meta"]["exoharness"]["workflow_continuation"]["required_arguments"],
                    "sections": sections,
                }, context=context)
                step = result.structured_content
                continuation = step["_meta"]["exoharness"]["workflow_continuation"]
                self.assertEqual(continuation["required_tool"], "patch_music_video_plan_segment")
                self.assertEqual(step["validation_errors"][0]["path"], "/opening_keyframe")
                self.assertEqual(continuation["repair_context"]["saved_later_sections"], ["segment-002", "segment-003"])
                runtime.plans = MusicVideoPlanStore(Path(temp))
                saved_before = runtime.plans.get_draft(step["draft_id"], step["draft_digest"])
                result = await server.mcp.call_tool("patch_music_video_plan_segment", {
                    **continuation["required_arguments"], "content":{"opening_keyframe":opening},
                }, context=context)
                step = result.structured_content
                self.assertTrue(step["valid"])
                saved = runtime.plans.get(step["plan_digest"])
                self.assertEqual(len(saved["segments"]), 3)
                # Repair retained the submitted later content exactly.
                for i in (1, 2):
                    self.assertEqual(saved["segments"][i]["ending_keyframe"],
                        saved_before["pending_sections"][f"segment-{i+1:03d}"]["ending_keyframe"])


if __name__ == "__main__":
    unittest.main()
