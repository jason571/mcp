import copy
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from jsonschema import Draft202012Validator

from lesson_renderer.drafts import Contract, DraftPatch, apply_patches
from lesson_renderer.server import Runtime, Settings
from lesson_renderer.store import encoded, read_json, write_json
from test_plan_contract import sample, SKILL


def patches(*operations):
    return [DraftPatch.model_validate(p) for p in operations]


class DraftTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.settings = Settings(Path(self.tmp.name), SKILL, "http://unused", "https://unused", "fixture" * 8)
        self.runtime = Runtime(self.settings)
        self.store = self.runtime.drafts

    def broken(self):
        plan = sample()
        plan["segments"][1]["heading"] = plan["segments"][1]["visual"].pop("heading")
        plan["segments"][3]["visual"]["actions"][0].pop("degrees")
        return plan

    def repairs(self):
        return patches({"op": "move", "from_path": "/segments/1/heading", "path": "/segments/1/visual/heading"},
            {"op": "add", "path": "/segments/3/visual/actions/0/degrees", "value": sample()["segments"][3]["visual"]["actions"][0]["degrees"]})

    def raw(self, draft):
        return self.store.load(self.store.root / draft["draft_id"])["plan"]

    def test_visibility_conflict_has_two_exact_actions_and_needs_only_one_field_patch(self):
        plan = sample()
        actions = [{"op":"show", "target":"b", "start":.2},
                   {"op":"highlight", "target":"b", "start":.2, "end":.45},
                   {"op":"hide", "target":"b", "start":.45}]
        plan["segments"][1]["visual"]["actions"] = actions
        original = copy.deepcopy(plan)
        initial = self.store.create(plan)
        self.assertEqual(initial["status"], "needs_repair")
        self.assertNotIn("speech_requests", initial)
        self.assertEqual(self.raw(initial), original)
        error = initial["errors"][0]
        self.assertEqual(error["code"], "action_overlap")
        self.assertEqual(error["objects"], ["b"])
        self.assertEqual([r["path"] for r in error["related"]],
            ["/segments/1/visual/actions/0", "/segments/1/visual/actions/2"])
        self.assertEqual(error["related"][0]["effective_timing"], {"start":.2, "end":1, "defaulted":["end"]})
        directive = initial["_meta"]["exoharness"]["workflow_continuation"]
        self.assertEqual(directive["prerequisite_tools"], [])
        self.assertEqual([item["value"] for item in directive["repair_context"]["focus"]], [actions[0], actions[2]])
        self.assertLess(len(encoded(directive["repair_context"])), 2000)
        # An unrelated model edit must not conceal the original conflict.
        wrong = self.store.patch(initial["draft_id"], 1, patches({"op":"replace", "path":"/title", "value":"updated title"}))
        self.assertEqual(wrong["errors"][0]["related"], error["related"])
        self.assertNotIn("speech_requests", wrong)
        self.store = Runtime(self.settings).drafts
        result = self.store.patch(initial["draft_id"], 2, patches({"op":"add", "path":"/segments/1/visual/actions/0/end", "value":.45}))
        self.assertEqual(result["status"], "prepared")
        expected = copy.deepcopy(original)
        expected["title"] = "updated title"
        expected["segments"][1]["visual"]["actions"][0]["end"] = .45
        self.assertEqual(self.raw(result), expected)
        self.assertEqual([s["arguments"]["text"] for s in result["speech_requests"]],
                         [s["narration"] for s in original["segments"]])
        self.assertEqual(list(self.runtime.jobs.iterdir()), [])

    def test_missing_array_error_includes_element_type_before_a_patch(self):
        plan = sample()
        plan["segments"][0]["visual"] = {"type":"steps", "steps":[{"title":"观察", "detail":"数一数"}]}
        initial = self.store.create(plan)
        missing = next(e for e in initial["errors"] if e["path"] == "/segments/0/visual/items")
        self.assertEqual(missing["expected"]["items"]["type"], "string")
        result = self.store.patch(initial["draft_id"], 1, patches(
            {"op":"remove", "path":"/segments/0/visual/steps"},
            {"op":"add", "path":"/segments/0/visual/items", "value":["观察：数一数"]}))
        self.assertEqual(result["status"], "prepared")

    def test_independent_conflicts_reported_together_despite_bad_segment_id(self):
        plan = sample()
        plan["segments"][1]["id"] = "不支持的标识符"
        for segment in plan["segments"][1:]:
            segment["visual"]["actions"] = [{"op":"show","target":"b","start":.1},
                                               {"op":"hide","target":"b","start":.6}]
        initial = self.store.create(plan)
        conflicts = [e for e in initial["errors"] if e["code"] == "action_overlap"]
        self.assertEqual({e["path"] for e in conflicts},
                         {f"/segments/{i}/visual/actions/1" for i in (1,2,3)})
        self.assertEqual(len(conflicts), 3)
        result = self.store.patch(initial["draft_id"], 1, patches(
            {"op":"replace","path":"/segments/1/id","value":"observe"},
            *[{"op":"add","path":f"/segments/{i}/visual/actions/0/end","value":.6} for i in (1,2,3)]))
        self.assertEqual(result["status"], "prepared")

    def test_unwrap_saved_plan_by_moving_exact_fields_without_rewriting_segments(self):
        original = sample()
        initial = self.store.create({"lesson": original})
        result = self.store.patch(initial["draft_id"], 1, patches(*[
            {"op":"move", "from_path":"/lesson/"+key, "path":"/"+key} for key in original],
            {"op":"remove", "path":"/lesson"}))
        self.assertEqual(result["status"], "prepared")
        self.assertEqual(self.raw(result), original)
        with self.assertRaisesRegex(ValueError, "existing segments"):
            apply_patches(original, patches({"op":"add", "path":"/segments", "value":[]}))

    def test_real_failure_saved_then_only_two_fields_repaired(self):
        broken = self.broken()
        initial = self.store.create(broken)
        self.assertEqual(initial["status"], "needs_repair")
        self.assertEqual({e["path"] for e in initial["errors"]}, {
            "/segments/1/heading", "/segments/3/visual/actions/0/degrees"})
        self.assertNotIn("speech_requests", initial)
        self.assertEqual(list(self.runtime.plans.iterdir()), [])
        directive = initial["_meta"]["exoharness"]["workflow_continuation"]
        self.assertEqual(directive["prerequisite_tools"], [])
        self.assertEqual({f["path"] for f in directive["repair_context"]["focus"]}, {"/segments/1", "/segments/3"})
        self.assertEqual(self.raw(initial), broken)
        final = self.store.patch(initial["draft_id"], 1, self.repairs())
        self.assertEqual(final["status"], "prepared")
        self.assertEqual(final["revision"], 2)
        self.assertEqual(self.raw(final), sample())
        self.assertEqual(len(final["speech_requests"]), len(sample()["segments"]))
        self.assertEqual(list(self.runtime.jobs.iterdir()), [])

    def test_duplicate_patch_and_restart_reuse_single_commit(self):
        initial = self.store.create(self.broken())
        final = self.store.patch(initial["draft_id"], 1, self.repairs())
        self.store = Runtime(self.settings).drafts
        replay = self.store.patch(initial["draft_id"], 1, self.repairs())
        self.assertTrue(replay["idempotent_replay"])
        self.assertEqual(final["lesson_id"], replay["lesson_id"])
        self.assertEqual(final["speech_requests"], replay["speech_requests"])
        self.assertEqual(self.store.get(initial["draft_id"])["lesson_id"], final["lesson_id"])
        self.assertEqual(len(list(self.runtime.plans.iterdir())), 1)

    def test_atomic_failure_stale_revision_and_committed_freeze(self):
        initial = self.store.create(self.broken())
        bad = self.repairs() + patches({"op": "remove", "path": "/segments/99"})
        rejected = self.store.patch(initial["draft_id"], 1, bad)
        self.assertIn("patch_rejected", rejected)
        self.assertEqual(rejected["revision"], 1)
        self.assertEqual(self.raw(initial), self.broken())
        partial = self.store.patch(initial["draft_id"], 1, self.repairs()[:1])
        self.assertNotIn("speech_requests", partial)
        rejected = self.store.patch(initial["draft_id"], 1, self.repairs()[1:])
        self.assertIn("revision_conflict", rejected["patch_rejected"])
        final = self.store.patch(initial["draft_id"], 2, self.repairs()[1:])
        rejected = self.store.patch(initial["draft_id"], 3, patches({"op": "replace", "path": "/title", "value": "different"}))
        self.assertIn("immutable", rejected["patch_rejected"])
        self.assertEqual(rejected["lesson_id"], final["lesson_id"])

    def test_duplicate_array_insert_does_not_insert_twice_or_rollback_newer_edits(self):
        initial = self.store.create(self.broken())
        # Example has no groups: add a harmless unknown list first, retained as a draft error.
        operation = patches({"op": "add", "path": "/extra", "value": []}, {"op": "add", "path": "/extra/-", "value": 1})
        self.store.patch(initial["draft_id"], 1, operation)
        two = self.store.patch(initial["draft_id"], 2, patches({"op": "replace", "path": "/title", "value": "updated"}))
        replay = self.store.patch(initial["draft_id"], 1, operation)
        self.assertEqual(replay["revision"], two["revision"])
        self.assertEqual(self.raw(replay)["extra"], [1])
        self.assertEqual(self.raw(replay)["title"], "updated")

    def test_concurrent_patches_do_not_lose_updates(self):
        initial = self.store.create(self.broken())
        def run(value):
            return Runtime(self.settings).drafts.patch(initial["draft_id"], 1, patches({"op": "replace", "path": "/title", "value": value}))
        with ThreadPoolExecutor(2) as pool:
            results = list(pool.map(run, ["a", "b"]))
        self.assertEqual(sum("patch_rejected" in r for r in results), 1)
        self.assertEqual(self.store.get(initial["draft_id"])["revision"], 2)

    def test_commit_interrupted_after_draft_save_is_recoverable(self):
        initial = self.store.create(self.broken())
        with patch.object(self.runtime, "prepare", side_effect=RuntimeError("simulated crash")):
            with self.assertRaises(RuntimeError): self.store.patch(initial["draft_id"], 1, self.repairs())
        self.store = Runtime(self.settings).drafts
        result = self.store.patch(initial["draft_id"], 1, self.repairs())
        self.assertEqual(result["status"], "prepared")
        self.assertTrue(result["idempotent_replay"])

    def test_focused_read_and_bounded_context_metadata(self):
        initial = self.store.create(self.broken())
        result = self.store.get(initial["draft_id"], ["/segments/3/visual/actions/0", "/segments/3/visual/actions/0/degrees"])
        self.assertEqual(result["focus"][0]["value"]["op"], "rotate")
        self.assertIn("degrees", result["focus"][0]["expected"]["required"])
        self.assertTrue(result["focus"][1]["missing"])
        self.assertEqual(result["focus"][1]["expected"]["maximum"], 1440)
        self.assertLess(len(encoded(initial)), 8000)
        self.assertNotIn("plan", initial)
        continuation = result["_meta"]["exoharness"]["workflow_continuation"]
        self.assertEqual(continuation["required_tool"], "patch_lesson_draft")
        self.assertEqual(continuation["repair_context"]["read_focus"], result["focus"])
        self.assertEqual(continuation["context_scope"], "checkpoint")

    def test_pagination_and_oversize_focus_are_explicit(self):
        bad = self.broken()
        for i in range(20): bad[f"wrong{i}"] = "x" * 1000
        initial = self.store.create(bad)
        self.assertEqual(len(initial["errors"]), 8)
        self.assertTrue(initial["has_more_errors"])
        second = self.store.get(initial["draft_id"], error_offset=8)
        self.assertNotEqual(initial["errors"], second["errors"])
        self.assertLess(len(encoded(initial["_meta"]["exoharness"]["workflow_continuation"])), 32768)
        with self.assertRaisesRegex(ValueError, "error_offset"):
            self.store.get(initial["draft_id"], error_offset=200)

    def test_successive_focus_queries_keep_bounded_workspace_and_error_page(self):
        initial = self.store.create(self.broken())
        self.store.get(initial["draft_id"], ["/segments/1/heading"], error_offset=1)
        result = self.store.get(initial["draft_id"], ["/segments/3/visual/actions/0"])
        self.assertEqual(result["error_offset"], 1)
        self.assertEqual({f["path"] for f in result["focus"]}, {"/segments/1/heading", "/segments/3/visual/actions/0"})
        self.store = Runtime(self.settings).drafts
        self.assertEqual(self.store.get(initial["draft_id"])["focus"], result["focus"])
        partial = self.store.patch(initial["draft_id"], 1, self.repairs()[:1])
        self.assertEqual(partial["revision"], 2)
        current = self.store.get(initial["draft_id"])
        self.assertNotIn("focus", current)
        self.assertEqual(current["error_offset"], 0)

    def test_identical_submissions_have_unpredictable_independent_drafts(self):
        one = self.store.create(self.broken())
        two = self.store.create(self.broken())
        self.assertNotEqual(one["draft_id"], two["draft_id"])
        self.store.patch(one["draft_id"], 1, self.repairs()[:1])
        self.assertEqual(self.raw(two), self.broken())

    def test_historical_field_recovery_does_not_rollback_current_draft(self):
        initial = self.store.create(self.broken())
        updated = self.store.patch(initial["draft_id"], 1, patches({"op": "remove", "path": "/segments/1/visual/objects"}))
        self.assertEqual(updated["revision"], 2)
        old = self.store.get(initial["draft_id"], ["/segments/1/visual/objects"], at_revision=1)
        self.assertEqual(old["revision"], 2)
        self.assertEqual(old["focus"][0]["source_revision"], 1)
        self.assertEqual(old["focus"][0]["value"], self.broken()["segments"][1]["visual"]["objects"])
        self.assertNotIn("objects", self.raw(old)["segments"][1]["visual"])
        restored = self.store.patch(initial["draft_id"], 2, patches({"op": "add", "path": "/segments/1/visual/objects", "value": old["focus"][0]["value"]}))
        self.assertEqual(self.raw(restored), self.broken())
        with self.assertRaisesRegex(ValueError, "revision does not exist"):
            self.store.get(initial["draft_id"], ["/title"], at_revision=100)

    def test_invalid_json_shape_limits_and_integrity(self):
        for bad in [json.dumps(sample()), {"title": "x" * 262145}, {"a": float("nan")}]:
            with self.assertRaises(ValueError): self.store.create(bad)
        initial = self.store.create(self.broken())
        path = self.store.root / initial["draft_id"]
        head = read_json(path / "head.json")
        record = read_json(path / (head["record"] + ".json"))
        record["plan"]["title"] = "tampered"
        write_json(path / (head["record"] + ".json"), record)
        with self.assertRaisesRegex(ValueError, "integrity"): self.store.get(initial["draft_id"])

    def test_contract_change_retains_draft_but_does_not_reinterpret_it(self):
        initial = self.store.create(self.broken())
        self.store.contract.version = "new-contract"
        with self.assertRaisesRegex(ValueError, "contract changed"): self.store.get(initial["draft_id"])
        self.assertTrue((self.store.root / initial["draft_id"] / "head.json").is_file())

    def test_semantic_errors_are_saved_and_report_real_json_paths(self):
        bad = sample()
        bad["segments"][1]["visual"]["objects"][0]["kind"] = "text"
        initial = self.store.create(bad)
        self.assertEqual(initial["status"], "needs_repair")
        self.assertEqual(initial["errors"][0]["path"], "/segments/1/visual/objects/0")
        self.assertNotIn("speech_requests", initial)
        self.assertTrue(initial["_meta"]["exoharness"]["workflow_continuation"]["prerequisite_tools"])

    def test_shape_and_independent_semantic_errors_are_reported_together(self):
        bad = self.broken()
        bad["segments"][1]["visual"]["objects"][0]["kind"] = "text"
        result = self.store.create(bad)
        self.assertIn("value_error", {e["code"] for e in result["errors"]})
        self.assertIn("unexpected_field", {e["code"] for e in result["errors"]})
        self.assertIn("missing_field", {e["code"] for e in result["errors"]})

    def test_safe_native_patch_contract(self):
        plan = {"a/b": {"x~y": [1, 2]}}
        changed = apply_patches(plan, patches({"op": "replace", "path": "/a~1b/x~0y/1", "value": {"native": True}}))
        self.assertEqual(changed["a/b"]["x~y"][1], {"native": True})
        for path in ["", "/segments", "/a~2b", "/a~1b/x~0y/01", "/a~1b/x~0y/99"]:
            with self.subTest(path=path), self.assertRaises(ValueError):
                apply_patches(plan, patches({"op": "replace", "path": path, "value": []}))
        for op in [{"op": "add", "path": "/x"}, {"op": "move", "path": "/x"}, {"op": "remove", "path": "/x", "value": 1}]:
            with self.assertRaises(ValueError): patches(op)
            self.assertFalse(Draft202012Validator(DraftPatch.model_json_schema()).is_valid(op))


if __name__ == "__main__": unittest.main()
