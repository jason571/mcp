"""The production 19-patch loop: missing declarations, lost inheritance and ID guessing."""
import copy
import unittest
import test_drafts
from test_drafts import patches


class SceneRepairRegression(unittest.TestCase):
    setUp = test_drafts.DraftTests.setUp
    raw = test_drafts.DraftTests.raw
    def chain(self):
        from test_plan_contract import sample
        plan = sample()
        plan["segments"][2]["visual"]["actions"] = [{"op": "show", "target": "detail"}]
        # The following segment still needs the objects from the original scene.
        plan["segments"][3]["visual"]["actions"] = [{"op": "highlight", "target": "pair"}]
        return plan

    def test_missing_declaration_is_fixed_at_source_in_one_patch_without_losing_content(self):
        plan = self.chain()
        initial = self.store.create(plan)
        error = next(e for e in initial["errors"] if e["code"] == "missing_scene_target")
        self.assertEqual(error["targets"], ["detail"])
        self.assertEqual(error["scene"]["source_path"], "/segments/1/visual")
        self.assertEqual(error["scene"]["object_ids"], ["a", "b"])
        self.assertEqual(error["scene"]["groups"], [{"id": "pair", "members": ["a", "b"]}])
        self.assertEqual(error["scene"]["dependent_paths"], ["/segments/2/visual", "/segments/3/visual"])
        directive = initial["_meta"]["exoharness"]["workflow_continuation"]
        self.assertEqual(directive["prerequisite_tools"], [])
        self.assertTrue(directive["repair_context"]["focus_complete"])
        self.assertIn("/segments/1/visual/objects", [f["path"] for f in directive["repair_context"]["focus"]])
        addition = {"id": "detail", "kind": "circle", "at": [0, 1], "visible": False}
        result = self.store.patch(initial["draft_id"], 1, patches(
            {"op": "add", "path": error["scene"]["source_path"] + "/objects/-", "value": addition}))
        self.assertEqual(result["status"], "prepared")
        expected = copy.deepcopy(plan)
        expected["segments"][1]["visual"]["objects"].append(addition)
        self.assertEqual(self.raw(result), expected)  # Exact narration, original objects and all actions.
        self.assertEqual(len(result["speech_requests"]), len(plan["segments"]))
        self.assertEqual(list(self.runtime.jobs.iterdir()), [])

    def test_missing_groups_returns_native_array_contract_before_model_patch(self):
        plan = self.chain()
        plan["segments"][1]["visual"].pop("groups")
        initial = self.store.create(plan)
        focus = initial["_meta"]["exoharness"]["workflow_continuation"]["repair_context"]["focus"]
        group = next(f for f in focus if f["path"] == "/segments/1/visual/groups")
        self.assertTrue(group["missing"])
        self.assertEqual(group["expected"]["type"], "array")
        self.assertEqual(group["expected"]["items"]["required"], ["id", "members"])
        result = self.store.patch(initial["draft_id"], 1, patches(
            {"op":"add", "path":"/segments/1/visual/objects/-", "value":{"id":"detail", "kind":"circle", "visible":False}},
            {"op":"add", "path":"/segments/1/visual/groups", "value":[{"id":"pair", "members":["a","b"]}]}))
        self.assertEqual(result["status"], "prepared")

    def test_declarations_in_continuation_include_source_and_can_be_moved_without_regeneration(self):
        plan = self.chain()
        obj = {"id":"detail", "kind":"circle", "visible":False, "at":[2,1]}
        plan["segments"][2]["visual"]["objects"] = [obj]
        initial = self.store.create(plan)
        error = next(e for e in initial["errors"] if e["code"] == "inherited_scene_declarations")
        self.assertEqual(error["scene"]["source_path"], "/segments/1/visual")
        directive = initial["_meta"]["exoharness"]["workflow_continuation"]
        self.assertEqual(directive["prerequisite_tools"], [])
        result = self.store.patch(initial["draft_id"], 1, patches(
            {"op":"move", "from_path":"/segments/2/visual/objects/0", "path":"/segments/1/visual/objects/-"},
            {"op":"remove", "path":"/segments/2/visual/objects"}))
        self.assertEqual(result["status"], "prepared")
        expected = copy.deepcopy(plan)
        expected["segments"][1]["visual"]["objects"].append(obj)
        expected["segments"][2]["visual"].pop("objects")
        self.assertEqual(self.raw(result), expected)

    def test_whole_visual_replacement_cannot_silently_drop_inheritance(self):
        initial = self.store.create(self.chain())
        replacement = {"type": "scene", "objects": [{"id": "detail", "kind": "circle"}], "actions": []}
        result = self.store.patch(initial["draft_id"], 1, patches(
            {"op": "replace", "path": "/segments/2/visual", "value": replacement}))
        self.assertEqual(result["revision"], 1)
        self.assertIn("omits continue_previous", result["patch_rejected"])
        self.assertEqual(self.raw(result), self.chain())
        self.assertIn("patch_rejected", result["_meta"]["exoharness"]["workflow_continuation"]["repair_context"])
        # Explicit authoring is allowed, but all newly broken dependencies must
        # be reported. No automatic copying of old objects into a new scene.
        replacement["continue_previous"] = False
        result = self.store.patch(initial["draft_id"], 1, patches(
            {"op": "replace", "path": "/segments/2/visual", "value": replacement}))
        self.assertEqual(result["revision"], 2)
        self.assertEqual(result["status"], "needs_repair")
        error = next(e for e in result["errors"] if e["code"] == "missing_scene_target")
        self.assertEqual(error["targets"], ["pair"])
        self.assertEqual(error["scene"]["object_ids"], ["detail"])
        self.assertEqual(error["scene"]["source_path"], "/segments/2/visual")
        self.assertNotIn("speech_requests", result)

    def test_noop_id_guessing_and_focus_queries_do_not_fabricate_progress(self):
        initial = self.store.create(self.chain())
        def progress(r): return r["_meta"]["exoharness"]["workflow_continuation"]["repair_progress"]
        no_change = self.store.patch(initial["draft_id"], 1, patches(
            {"op": "replace", "path": "/segments/2/visual/actions/0/target", "value": "detail"}))
        self.assertEqual(no_change["revision"], 1)
        self.assertIn("no_change", no_change["patch_rejected"])
        result = initial
        for name in ["unknown1", "unknown2", "detail"]:
            result = self.store.patch(initial["draft_id"], result["revision"], patches(
                {"op": "replace", "path": "/segments/2/visual/actions/0/target", "value": name}))
            self.assertEqual(progress(result)["issues"], progress(initial)["issues"])
            read = self.store.get(initial["draft_id"], ["/segments/1/visual/objects"])
            self.assertEqual(progress(read), progress(result))
        self.assertEqual(progress(result)["candidate_digest"], progress(initial)["candidate_digest"])

    def test_non_scene_boundary_never_exposes_stale_object_ids(self):
        plan = self.chain()
        plan["segments"][2]["visual"] = {"type": "title", "title": "A new page"}
        result = self.store.create(plan)
        self.assertFalse(any(e["code"] == "missing_scene_target" for e in result["errors"]))
        self.assertTrue(any("immediately preceding" in e.get("message", "") for e in result["errors"]))

    def test_independent_missing_targets_are_reported_together(self):
        plan = self.chain()
        plan["segments"][3]["visual"]["actions"].append({"op": "show", "target": "other_missing"})
        result = self.store.create(plan)
        errors = [e for e in result["errors"] if e["code"] == "missing_scene_target"]
        self.assertEqual(len(errors), 2)
        self.assertEqual({e["targets"][0] for e in errors}, {"detail", "other_missing"})

    def test_unrelated_shape_error_does_not_hide_declaration_source(self):
        plan = self.chain()
        plan["segments"][1]["visual"]["objects"][0]["rotate"] = 45
        plan["segments"][2]["visual"]["objects"] = [{"id":"detail", "kind":"circle", "visible":False}]
        result = self.store.create(plan)
        self.assertEqual(result["status"], "needs_repair")
        self.assertTrue(any(e["code"] == "unexpected_field" for e in result["errors"]))
        error = next(e for e in result["errors"] if e["code"] == "inherited_scene_declarations")
        self.assertEqual(error["scene"]["source_path"], "/segments/1/visual")
        self.assertNotIn("speech_requests", result)

    def test_ambiguous_identity_boundary_does_not_resolve_stale_objects(self):
        plan = self.chain()
        plan["segments"][1]["visual"]["objects"][1]["id"] = "a"
        result = self.store.create(plan)
        self.assertFalse(any(e["code"] == "missing_scene_target" for e in result["errors"]))

    def test_invalid_action_interval_is_local_and_needs_no_repeated_browsing(self):
        from test_plan_contract import sample
        plan = sample()
        action = plan["segments"][3]["visual"]["actions"][0]
        action.update(start=0, end=0)
        initial = self.store.create(plan)
        self.assertEqual(initial["errors"][0]["code"], "action_interval")
        directive = initial["_meta"]["exoharness"]["workflow_continuation"]
        self.assertEqual(directive["prerequisite_tools"], [])
        self.assertEqual(directive["repair_context"]["focus"][0]["value"], action)
        result = self.store.patch(initial["draft_id"], 1, patches(
            {"op":"replace", "path":"/segments/3/visual/actions/0/end", "value":.5}))
        self.assertEqual(result["status"], "prepared")

    def test_recorded_spike_targets_preserve_capsid_core_and_all_following_actions(self):
        # Minimal reproducer of run-1a099232ae8-7-3, not a topic template.
        plan = self.chain()
        source = plan["segments"][1]["visual"]
        source["objects"] = [{"id":"capsid", "kind":"circle", "size":[3.2,3.2]},
                             {"id":"core", "kind":"ellipse", "size":[1.4,1.4], "visible":False}]
        source.pop("groups")
        source["actions"] = [{"op":"highlight", "target":"capsid"}]
        targets = ["spk1","spk2","spk3","spk4"]
        plan["segments"][2]["visual"]["actions"] = [{"op":"show", "target":t} for t in targets] + [{"op":"highlight", "target":"spikes"}]
        plan["segments"][3]["visual"]["actions"] = [{"op":"highlight", "target":t} for t in ("capsid","core")]
        initial = self.store.create(plan)
        error = next(e for e in initial["errors"] if e["code"] == "missing_scene_target")
        self.assertEqual(set(error["targets"]), {*targets,"spikes"})
        additions = [{"id":t,"kind":"circle","visible":False,"at":list(at),"size":[.3,.3]}
                     for t, at in zip(targets, [(0,1.7),(1.7,0),(0,-1.7),(-1.7,0)])]
        operations = [{"op":"add","path":error["scene"]["source_path"]+"/objects/-","value":o} for o in additions]
        group = {"id":"spikes","members":targets}
        operations.append({"op":"add","path":error["scene"]["source_path"]+"/groups","value":[group]})
        repaired = self.store.patch(initial["draft_id"],1,patches(*operations))
        self.assertEqual(repaired["status"],"prepared")
        expected = copy.deepcopy(plan)
        expected["segments"][1]["visual"]["objects"].extend(additions)
        expected["segments"][1]["visual"]["groups"] = [group]
        self.assertEqual(self.raw(repaired), expected)
