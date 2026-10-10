from __future__ import annotations

import copy
import json
import sys
import unittest
from pathlib import Path

from pydantic import ValidationError

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from media_workflow_mcp.plan import (
    MAX_SERIALIZED_PLAN_BYTES,
    MusicVideoPlan,
    MusicVideoPlanPatch,
    MusicVideoPlanValidationError,
    apply_music_video_plan_patches,
    duration_policy_from_schema,
    normalize_music_video_plan,
    parse_music_video_plan,
    validate_and_render_music_video_plan,
    _expected_segment_windows,
)


ANCHORS = {
    "subject_identity": "同一位成年东方女性，鹅蛋脸、长黑发和稳定五官",
    "wardrobe_or_materials": "始终穿无文字的素白纱衣，手持同一枝莲花",
    "setting_geometry": "同一座山水荷塘与古寺，空间方位保持稳定",
    "palette_and_lighting": "青灰水墨主色，柔和侧逆光，暖金烛光只在寺内出现",
    "camera_language": "克制的缓慢推轨与环绕，不使用无动机跳切",
    "recurring_motif": "莲花、薄雾、水面涟漪与一盏青灯",
    "screen_direction": "人物和摄影机整体由画面左侧向右侧推进",
}


def valid_plan(duration_ms: int = 222_014) -> dict[str, object]:
    target_ms = 30_000
    tail_merge_max_ms = 40_000
    full_count, remainder_ms = divmod(duration_ms, target_ms)
    if full_count == 0:
        windows = [(0, duration_ms)]
    elif remainder_ms == 0:
        windows = [
            (index * target_ms, (index + 1) * target_ms)
            for index in range(full_count)
        ]
    elif target_ms + remainder_ms <= tail_merge_max_ms:
        windows = [
            (index * target_ms, (index + 1) * target_ms)
            for index in range(full_count - 1)
        ]
        windows.append(((full_count - 1) * target_ms, duration_ms))
    else:
        windows = [
            (index * target_ms, min((index + 1) * target_ms, duration_ms))
            for index in range(full_count + 1)
        ]
    segments: list[dict[str, object]] = []
    prior_ending: str | None = None
    for index, (start, end) in enumerate(windows):
        logical_ms = end - start
        generation_seconds = max(5, (logical_ms + 999) // 1000)
        generation_ms = generation_seconds * 1000
        opening = prior_ending or "薄雾中的荷塘水面，一片花瓣停在画面中央"
        ending = f"边界关键帧 {index + 1}：人物与镜头停在向右运动的稳定构图"
        prior_ending = ending
        boundaries = [round(generation_ms * part / 4) for part in range(5)]
        storyboard = []
        for shot in range(4):
            storyboard.append(
                {
                    "start_ms": boundaries[shot],
                    "end_ms": boundaries[shot + 1],
                    "framing_and_lens": f"镜头 {shot + 1} 的中景，50mm 等效焦段",
                    "subject_action": f"人物完成第 {index + 1} 段的动作 {shot + 1}",
                    "camera_movement": "摄影机沿既定屏幕方向缓慢推轨",
                    "environment_motion": "薄雾、衣摆与水面涟漪保持因果连续",
                    "lighting": "青灰侧逆光方向保持不变",
                    "transition": "以主体动作连续匹配到下一镜头",
                }
            )
        segments.append(
            {
                "segment_id": f"segment-{index + 1:03d}",
                "start_ms": start,
                "end_ms": end,
                "generation_duration_seconds": generation_seconds,
                "visual_role": f"推进完整故事的第 {index + 1} 个独立章节",
                "music_evidence": {
                    "timing_basis": "duration-only",
                    "description": "仅使用已测得的主音乐时间窗；叙事来自歌名与用户要求",
                },
                "prompt_plan": {
                    "shot_mode": "multi-shot",
                    "negative_prompt": "人物漂移、服装变化、场景跳变、闪烁、模糊、字幕、标志、水印",
                    "soundscape": None,
                    "opening_keyframe": opening,
                    "ending_keyframe": ending,
                    "storyboard": storyboard,
                    "controls": {
                        "duration_seconds": generation_seconds,
                        "aspect_ratio": "16:9",
                        "resolution": "512p",
                        "audio_enabled": False,
                    },
                },
            }
        )
    return {
        "schema_version": 1,
        "plan_id": "mv_test_001",
        "master_duration_ms": duration_ms,
        "target_segment_duration_seconds": 30,
        "live_min_duration_seconds": 5,
        "live_max_duration_seconds": 60,
        "duration_step_seconds": 1,
        "tail_merge_max_duration_seconds": 40,
        "output_language": "zh",
        "song_context_basis": "title-informed-model-knowledge",
        "continuity_anchors": copy.deepcopy(ANCHORS),
        "provenance": {
            "planning_mode": "read-only-resources",
            "pack_id": "classic-chinese-mv",
            "document_id": "classic-chinese-mv--h3-video",
            "model_profile": "minimax-h3",
        },
        "segments": segments,
    }


class MusicVideoPlanTests(unittest.TestCase):
    def test_user_supplied_style_needs_no_library_lookup_provenance(self) -> None:
        plan = valid_plan()
        plan["provenance"] = {
            "planning_mode": "user-specified",
            "model_profile": "not-selected",
        }
        MusicVideoPlan.model_validate(plan)

    def test_library_provenance_requires_real_source_and_user_style_cannot_claim_one(self) -> None:
        plan = valid_plan()
        plan["provenance"].pop("document_id")
        with self.assertRaises(ValueError):
            MusicVideoPlan.model_validate(plan)
        plan["provenance"]["planning_mode"] = "user-specified"
        with self.assertRaises(ValueError):
            MusicVideoPlan.model_validate(plan)

    def test_clip_policy_uses_live_default_not_a_hardcoded_15_or_30(self) -> None:
        fifteen = duration_policy_from_schema({
            "properties": {"duration_seconds": {"type": "number", "default": 15, "minimum": 5, "maximum": 60}},
        })
        self.assertEqual(fifteen["target_segment_duration_seconds"], 15)
        self.assertEqual(fifteen["tail_merge_max_duration_seconds"], 21)
        thirty = duration_policy_from_schema({
            "properties": {"duration_seconds": {"type": "number", "default": 30, "minimum": 5, "maximum": 60}},
        })
        self.assertEqual(thirty["target_segment_duration_seconds"], 30)
        self.assertEqual(thirty["tail_merge_max_duration_seconds"], 42)
        missing = duration_policy_from_schema(
            {"properties": {"duration_seconds": {"type": "number", "maximum": 60}}}
        )
        self.assertEqual(missing["target_segment_duration_seconds"], 15)
        self.assertEqual(missing["tail_merge_max_duration_seconds"], 21)
        reported = duration_policy_from_schema(
            {"properties": {"duration_seconds": {
                "anyOf": [
                    {"type": "number", "minimum": 5, "maximum": 60,
                     "description": "Omit to use FASTVIDEO_MCP_DEFAULT_DURATION_SECONDS, which defaults to 30."},
                    {"type": "null"},
                ],
                "default": None,
            }}},
            fallback_target_seconds=15,
        )
        self.assertEqual(reported["target_segment_duration_seconds"], 15)
        self.assertEqual(reported["tail_merge_max_duration_seconds"], 21)
        reported_thirty = duration_policy_from_schema(
            {"properties": {"duration_seconds": {"type": "number", "default": None, "minimum": 5, "maximum": 60}}},
            fallback_target_seconds=30,
        )
        self.assertEqual(reported_thirty["target_segment_duration_seconds"], 30)
        schema_wins = duration_policy_from_schema(
            {"properties": {"duration_seconds": {"type": "number", "default": 15, "minimum": 5, "maximum": 60}}},
            fallback_target_seconds=30,
        )
        self.assertEqual(schema_wins["target_segment_duration_seconds"], 15)

    def test_windows_follow_live_default_and_40_percent_tail_merge(self) -> None:
        def clocks(target: int, tail: int, duration_ms: int):
            return MusicVideoPlan.model_construct(
                master_duration_ms=duration_ms,
                target_segment_duration_seconds=target,
                tail_merge_max_duration_seconds=tail,
                live_min_duration_seconds=5,
                live_max_duration_seconds=60,
                duration_step_seconds=1,
            )

        four_min_at_15 = _expected_segment_windows(clocks(15, 21, 240_000))
        self.assertEqual(len(four_min_at_15), 16)
        merged = _expected_segment_windows(clocks(15, 21, 246_000))
        self.assertEqual(len(merged), 16)
        self.assertEqual(merged[-1], (225_000, 246_000))
        split = _expected_segment_windows(clocks(15, 21, 247_000))
        self.assertEqual(len(split), 17)
        self.assertEqual(split[-2], (225_000, 240_000))
        self.assertEqual(split[-1], (240_000, 247_000))
        merged_30 = _expected_segment_windows(clocks(30, 42, 252_000))
        self.assertEqual(len(merged_30), 8)
        self.assertEqual(merged_30[-1], (210_000, 252_000))
        split_30 = _expected_segment_windows(clocks(30, 42, 253_000))
        self.assertEqual(len(split_30), 9)
        self.assertEqual(split_30[-1], (240_000, 253_000))

    def test_normalizes_service_owned_timing_without_rewriting_creative_plan(self) -> None:
        raw = valid_plan(241_894)
        raw["live_max_duration_seconds"] = 30
        raw["continuity_anchors"].pop("wardrobe_or_materials")
        for segment in raw["segments"]:
            offset = segment["start_ms"]
            for shot in segment["prompt_plan"]["storyboard"]:
                shot["start_ms"] += offset
                shot["end_ms"] += offset

        normalized, corrections = normalize_music_video_plan(
            MusicVideoPlan.model_validate(raw),
            live_min_duration_seconds=5,
            live_max_duration_seconds=60,
            target_segment_duration_seconds=30,
            duration_step_seconds=1,
            tail_merge_max_duration_seconds=40,
        )
        result = validate_and_render_music_video_plan(normalized)

        self.assertTrue(result["valid"])
        self.assertEqual(result["segment_count"], 8)
        self.assertEqual(result["segments"][-1]["generation_duration_seconds"], 32)
        self.assertIn("live_max_duration_seconds", corrections)
        self.assertGreater(len(corrections), 20)

    def test_applies_only_requested_creative_field_patch(self) -> None:
        plan = MusicVideoPlan.model_validate(valid_plan(30_000))
        patched = apply_music_video_plan_patches(
            plan,
            [
                MusicVideoPlanPatch(
                    op="replace",
                    path="/segments/0/visual_role",
                    value="A corrected opening chapter",
                )
            ],
        )

        self.assertEqual(patched.segments[0].visual_role, "A corrected opening chapter")
        self.assertEqual(patched.segments[0].prompt_plan, plan.segments[0].prompt_plan)

    def test_normalization_preserves_an_already_valid_uneven_storyboard(self) -> None:
        raw = valid_plan(30_000)
        boundaries = [0, 4_000, 11_000, 19_000, 30_000]
        for index, shot in enumerate(raw["segments"][0]["prompt_plan"]["storyboard"]):
            shot["start_ms"] = boundaries[index]
            shot["end_ms"] = boundaries[index + 1]

        normalized, corrections = normalize_music_video_plan(
            MusicVideoPlan.model_validate(raw),
            live_min_duration_seconds=5,
            live_max_duration_seconds=60,
            target_segment_duration_seconds=30,
            duration_step_seconds=1,
            tail_merge_max_duration_seconds=40,
        )

        actual = [normalized.segments[0].prompt_plan.storyboard[0].start_ms] + [
            shot.end_ms for shot in normalized.segments[0].prompt_plan.storyboard
        ]
        self.assertEqual(actual, boundaries)
        self.assertFalse(any("storyboard" in item for item in corrections))

    def test_accepts_an_explicit_json_serialization_with_identical_validation(self) -> None:
        raw = valid_plan()
        parsed = parse_music_video_plan(json.dumps(raw, ensure_ascii=False))

        self.assertEqual(parsed.model_dump(), MusicVideoPlan.model_validate(raw).model_dump())

    def test_rejects_invalid_or_oversized_serialized_plans(self) -> None:
        with self.assertRaisesRegex(ValueError, "not valid JSON"):
            parse_music_video_plan("{not-json}")
        with self.assertRaisesRegex(ValueError, "must decode to a JSON object"):
            parse_music_video_plan("[]")
        with self.assertRaisesRegex(ValueError, "exceeds"):
            parse_music_video_plan('"' + ("x" * MAX_SERIALIZED_PLAN_BYTES) + '"')

    def test_renders_independent_prompt_and_storyboard_for_every_segment(self) -> None:
        result = validate_and_render_music_video_plan(MusicVideoPlan.model_validate(valid_plan()))
        self.assertTrue(result["valid"])
        self.assertEqual(result["segment_count"], 8)
        segments = result["segments"]
        assert isinstance(segments, list)
        self.assertEqual(segments[-1]["master_start_ms"], 210_000)
        self.assertEqual(segments[-1]["master_end_ms"], 222_014)
        self.assertEqual(segments[-1]["generation_duration_seconds"], 13)
        self.assertTrue(all(item["storyboard_shot_count"] == 4 for item in segments))
        prompts = {item["positive_prompt"] for item in segments}
        self.assertEqual(len(prompts), 8)
        self.assertTrue(all(item["negative_prompt"] for item in segments))
        self.assertIn("不要把主体瞬移进车", segments[0]["positive_prompt"])

    def test_recurring_location_and_motif_are_not_forced_into_every_clip(self) -> None:
        raw = valid_plan(60_000)
        raw["output_language"] = "en"
        raw["continuity_anchors"]["setting_geometry"] = "SOURCE_STATION_LAYOUT"
        raw["continuity_anchors"]["recurring_motif"] = "SOURCE_SCENE_OBJECT"
        rendered = validate_and_render_music_video_plan(MusicVideoPlan.model_validate(raw))
        second_prompt = rendered["segments"][1]["positive_prompt"]
        self.assertIn(raw["continuity_anchors"]["subject_identity"], second_prompt)
        self.assertNotIn("SOURCE_STATION_LAYOUT", second_prompt)
        self.assertNotIn("SOURCE_SCENE_OBJECT", second_prompt)
        self.assertIn(raw["segments"][1]["visual_role"], second_prompt)

    def test_merges_a_short_tail_into_one_30_to_40_second_final_segment(self) -> None:
        result = validate_and_render_music_video_plan(
            MusicVideoPlan.model_validate(valid_plan(241_894))
        )

        self.assertEqual(result["segment_count"], 8)
        segments = result["segments"]
        assert isinstance(segments, list)
        self.assertEqual(segments[-1]["master_start_ms"], 210_000)
        self.assertEqual(segments[-1]["master_end_ms"], 241_894)
        self.assertEqual(segments[-1]["generation_duration_seconds"], 32)

    def test_keeps_a_separate_tail_when_merging_would_exceed_40_seconds(self) -> None:
        result = validate_and_render_music_video_plan(
            MusicVideoPlan.model_validate(valid_plan(250_001))
        )

        self.assertEqual(result["segment_count"], 9)
        segments = result["segments"]
        assert isinstance(segments, list)
        self.assertEqual(segments[-1]["master_start_ms"], 240_000)
        self.assertEqual(segments[-1]["master_end_ms"], 250_001)
        self.assertEqual(segments[-1]["generation_duration_seconds"], 11)

    def test_rejects_continuity_handoff_drift(self) -> None:
        raw = valid_plan()
        raw["segments"][1]["prompt_plan"]["opening_keyframe"] = "unrelated opening"
        with self.assertRaisesRegex(ValueError, "opening_keyframe"):
            validate_and_render_music_video_plan(MusicVideoPlan.model_validate(raw))

    def test_rejects_missing_negative_prompt(self) -> None:
        raw = valid_plan()
        raw["segments"][0]["prompt_plan"]["negative_prompt"] = ""
        with self.assertRaises(ValidationError):
            MusicVideoPlan.model_validate(raw)

    def test_rejects_incomplete_local_storyboard(self) -> None:
        raw = valid_plan(30_000)
        raw["segments"][0]["prompt_plan"]["storyboard"][-1]["end_ms"] = 29_000
        with self.assertRaisesRegex(ValueError, "must end at 30000 ms"):
            validate_and_render_music_video_plan(MusicVideoPlan.model_validate(raw))

    def test_multi_shot_accepts_three_but_rejects_one_or_two_shots(self) -> None:
        for count in (1, 2, 3, 4):
            with self.subTest(count=count):
                raw = valid_plan(30_000)
                shots = raw["segments"][0]["prompt_plan"]["storyboard"][:count]
                for index, shot in enumerate(shots):
                    shot["start_ms"] = round(30_000 * index / count)
                    shot["end_ms"] = round(30_000 * (index + 1) / count)
                raw["segments"][0]["prompt_plan"]["storyboard"] = shots
                if count < 3:
                    with self.assertRaisesRegex(MusicVideoPlanValidationError, "at least three"):
                        validate_and_render_music_video_plan(MusicVideoPlan.model_validate(raw))
                else:
                    result = validate_and_render_music_video_plan(MusicVideoPlan.model_validate(raw))
                    self.assertEqual(result["segments"][0]["storyboard_shot_count"], count)
        raw["segments"][0]["prompt_plan"]["storyboard"] = []
        with self.assertRaises(ValidationError):
            MusicVideoPlan.model_validate(raw)

    def test_rejects_missing_song_segment(self) -> None:
        raw = valid_plan()
        raw["segments"].pop()
        with self.assertRaisesRegex(ValueError, "expected 8 segments"):
            validate_and_render_music_video_plan(MusicVideoPlan.model_validate(raw))

    def test_rejects_reused_segment_storyboard(self) -> None:
        raw = valid_plan()
        raw["segments"][1]["prompt_plan"]["storyboard"] = copy.deepcopy(
            raw["segments"][0]["prompt_plan"]["storyboard"]
        )
        with self.assertRaisesRegex(ValueError, "must not reuse"):
            validate_and_render_music_video_plan(MusicVideoPlan.model_validate(raw))

    def test_incomplete_segment_reports_exact_zero_based_patch_target(self) -> None:
        raw = valid_plan(322_038)
        final_storyboard = raw["segments"][10]["prompt_plan"]["storyboard"]
        final_storyboard[0]["end_ms"] = 22_000
        raw["segments"][10]["prompt_plan"]["storyboard"] = [final_storyboard[0]]

        with self.assertRaises(MusicVideoPlanValidationError) as caught:
            validate_and_render_music_video_plan(MusicVideoPlan.model_validate(raw))

        self.assertEqual(
            caught.exception.patch_targets,
            ("/segments/10/prompt_plan/storyboard",),
        )
        self.assertIn(
            "segment-011 local storyboard must end at 23000 ms, received 22000",
            caught.exception.errors,
        )


if __name__ == "__main__":
    unittest.main()
