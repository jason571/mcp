import copy
import hashlib
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

from mcp.server.mcpserver.exceptions import ToolError
from skill_workspace import server
from skill_workspace.common import browser_revision, read, write
from skill_workspace.delivery import Delivery


class DeliveryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        self.rt = server.Runtime(root / "data", root / "packages", "private-test-key" * 5, "http://testserver")
        write(self.rt.root / "worker.json", {"at": time.time(), "image": "sha256:" + "a" * 64})
        write(self.rt.root / "inventory.json", {"image": "sha256:" + "a" * 64, "commands": {"python": True}})
        server._runtime = self.rt

    def tearDown(self):
        server._runtime = None
        self.temp.cleanup()

    def create(self, name="index.html", checks=True):
        created = server.create_workspace(delivery=Delivery(outputs=[name], browser_checks=[name] if checks else []))
        handle = created.structured_content["workspace_id"]
        self.assertEqual(created.meta["exoharness"]["workflow_continuation"]["required_tool"], "publish_workspace")
        return handle

    def inspect(self, handle, revision=1):
        # Trusted broker receipt fixture; no model-provided inspection verdict.
        identity = f"{handle}:{revision}"
        job_id = "work_" + hashlib.sha256(identity.encode()).hexdigest()
        job = self.rt.jobs / job_id
        job.mkdir(exist_ok=True)
        write(job / "state.json", {"status": "completed"})
        write(job / "request.json", {"operation": "inspect_webpage", "workspace_id": handle,
                                     "workspace_revision": revision, "entrypoint": "index.html",
                                     "observer_revision": browser_revision()})
        write(job / "result.json", {"inspection": {
            "source_sha256": self.rt.workspaces.state(handle)["files"]["index.html"]["sha256"],
            "scenarios": [{"visible_text_chars": 200, "horizontal_overflow": False,
                           "console_errors": [], "broken_images": [], "missing_anchors": []}]}})
        return job_id

    def test_source_is_not_delivery_and_success_requires_verified_publication(self):
        handle = self.create()
        saved = server.edit_workspace(handle, 0, files=[server.SourceFile(path="index.html", content="<html><body>saved source</body></html>")])
        self.assertEqual(saved.meta["exoharness"]["workflow_continuation"]["status"], "required")
        with self.assertRaisesRegex(ToolError, "requires browser inspection"):
            server.publish_workspace(handle, 1, ["index.html"])
        self.assertEqual(list(self.rt.jobs.iterdir()), [])
        job = self.inspect(handle)
        completed = server.publish_workspace(handle, 1, ["index.html"], [job])
        self.assertEqual(completed.meta["exoharness"]["workflow_continuation"]["status"], "complete")
        value = completed.structured_content
        self.assertTrue(value["artifacts"][0]["preview_url"])
        self.assertFalse((self.rt.jobs / value["job_id"] / "request.json").exists())
        self.assertEqual(server.publish_workspace(handle, 1, ["index.html"], [job]).structured_content["job_id"], value["job_id"])
        self.assertNotIn("workflow_continuation", self.rt.status(value["job_id"])["_meta"]["exoharness"],
                         "Reading a prior publication to locate sources must not complete a new user request")
        self.assertEqual(self.rt.workspaces.state(handle)["revision"], 1)

    def test_stale_cross_workspace_failed_and_incorrect_check_cannot_publish(self):
        handle = self.create()
        server.edit_workspace(handle, 0, files=[server.SourceFile(path="index.html", content="<h1>First</h1>")])
        job_id = self.inspect(handle)
        job = self.rt.jobs / job_id
        original = read(job / "result.json")
        for key, bad in [("console_errors", ["TypeError"]), ("horizontal_overflow", True),
                         ("broken_images", ["missing.png"]), ("missing_anchors", ["#gone"]), ("visible_text_chars", 0),
                         ("unexpected_hidden_text_blocks", ["#content"]),
                         ("unreadable_headings", [{"selector": "h1", "ratio": 1.01, "minimum": 3}])]:
            report = copy.deepcopy(original)
            report["inspection"]["scenarios"][0][key] = bad
            summary = server.compact_inspection(report["inspection"])
            self.assertEqual(summary["checks"]["status"], "failed")
            self.assertIn(key, summary["scenarios"][0]["checks"]["errors"])
            write(job / "result.json", report)
            with self.assertRaisesRegex(ToolError, "rendering errors"):
                server.publish_workspace(handle, 1, ["index.html"], [job_id])
        write(job / "result.json", original)
        write(job / "state.json", {"status": "failed"})
        with self.assertRaisesRegex(ToolError, "not a completed"):
            server.publish_workspace(handle, 1, ["index.html"], [job_id])
        write(job / "state.json", {"status": "completed"})
        other = self.create()
        server.edit_workspace(other, 0, files=[server.SourceFile(path="index.html", content="<h1>First</h1>")])
        with self.assertRaisesRegex(ToolError, "different workspace"):
            server.publish_workspace(other, 1, ["index.html"], [job_id])
        edited = server.edit_workspace(handle, 1, appends=[server.SourceFile(path="index.html", content="<p>Final</p>")])
        self.assertEqual(edited.meta["exoharness"]["workflow_continuation"]["status"], "required")
        with self.assertRaisesRegex(ToolError, "stale revision"):
            server.publish_workspace(handle, 2, ["index.html"], [job_id])
        fresh = self.inspect(handle, 2)
        server.publish_workspace(handle, 2, ["index.html"], [fresh])

    def test_intentionally_hidden_ui_is_not_a_publication_failure(self):
        handle = self.create()
        server.edit_workspace(handle, 0, files=[server.SourceFile(path="index.html", content="<h1>Visible title</h1><p hidden>Inactive translation</p>")])
        job_id = self.inspect(handle)
        job = self.rt.jobs / job_id
        report = read(job / "result.json")
        report["inspection"]["scenarios"][0].update(hidden_text_blocks=["p"],
            unexpected_hidden_text_blocks=[], unreadable_headings=[])
        write(job / "result.json", report)
        self.assertEqual(server.compact_inspection(report["inspection"])["checks"]["status"], "passed")
        result = server.publish_workspace(handle, 1, ["index.html"], [job_id])
        self.assertEqual(result.meta["exoharness"]["workflow_continuation"]["status"], "complete")

    def test_old_observer_receipt_requires_reinspection_not_source_regeneration(self):
        handle = self.create()
        server.edit_workspace(handle, 0, files=[server.SourceFile(path="index.html", content="<h1>Saved page</h1>")])
        job_id = self.inspect(handle)
        job = self.rt.jobs / job_id
        request = read(job / "request.json")
        request["observer_revision"] = "0" * 64
        write(job / "request.json", request)
        with self.assertRaisesRegex(ToolError, "policy has changed"):
            server.publish_workspace(handle, 1, ["index.html"], [job_id])
        self.assertEqual(self.rt.workspaces.state(handle)["revision"], 1)
        self.assertEqual(self.rt.workspaces.bytes(handle, "index.html"), b"<h1>Saved page</h1>")

    def test_no_contract_keeps_stdout_and_old_file_publication_behavior(self):
        created = server.create_workspace()
        self.assertIsNone(created.meta)
        handle = created.structured_content["workspace_id"]
        saved = server.edit_workspace(handle, 0, files=[server.SourceFile(path="answer.txt", content="100! = ...")])
        self.assertNotIn("workflow_continuation", saved.meta["exoharness"])
        value = server.publish_workspace(handle, 1, ["answer.txt"])
        self.assertNotIn("workflow_continuation", value.meta["exoharness"])

    def test_internal_publication_is_durable_hidden_and_does_not_complete_final_delivery(self):
        handle = server.create_workspace().structured_content["workspace_id"]
        server.edit_workspace(handle, 0, files=[server.SourceFile(path="speech.mp3", content="audio-fixture")])
        value = server.publish_workspace(handle, 1, ["speech.mp3"], artifact_visibility="internal")
        self.assertEqual(value.meta["exoharness"]["artifact_visibility"], "internal")
        job = value.structured_content["job_id"]
        for observe in (False, True):
            replay = self.rt.status(job, observe=observe)
            self.assertEqual(replay["_meta"]["exoharness"]["artifact_visibility"], "internal")
            self.assertEqual(replay["artifacts"][0]["_meta"]["exoharness"]["artifact_visibility"], "internal")
        final = server.publish_workspace(handle, 1, ["speech.mp3"])
        self.assertNotEqual(final.structured_content["job_id"], job)
        self.assertNotIn("artifact_visibility", final.meta["exoharness"])
        self.assertEqual(server.publish_workspace(handle, 1, ["speech.mp3"], artifact_visibility="internal").structured_content["job_id"], job)
        bound = self.create("video.mp4", checks=False)
        server.edit_workspace(bound, 0, files=[server.SourceFile(path="video.mp4", content="video-fixture")])
        with self.assertRaisesRegex(ToolError, "cannot complete"):
            server.publish_workspace(bound, 1, ["video.mp4"], artifact_visibility="internal")

    def test_capacity_is_aggregate_and_failed_import_is_atomic(self):
        handle = server.create_workspace().structured_content["workspace_id"]
        media = lambda name: server.InputFile(path=name, attachment_id="att_" + "a" * 64)
        with patch.object(server, "MAX_OUTPUT", 10), patch.object(self.rt, "input_file", return_value=b"123456"):
            with self.assertRaisesRegex(ToolError, "Splitting.*does not increase total capacity"):
                server.edit_workspace(handle, 0, inputs=[media("one.mp4"), media("two.mp4")])
            self.assertEqual(self.rt.workspaces.state(handle)["revision"], 0)
            self.assertEqual(self.rt.workspaces.state(handle)["files"], {})
            server.edit_workspace(handle, 0, inputs=[media("one.mp4")])
            with self.assertRaisesRegex(ToolError, "Splitting.*does not increase total capacity"):
                server.edit_workspace(handle, 1, inputs=[media("two.mp4")])
            self.assertEqual(self.rt.workspaces.state(handle)["revision"], 1)
            self.assertEqual(set(self.rt.workspaces.state(handle)["files"]), {"one.mp4"})

    def test_file_contract_is_not_tied_to_html_and_rejects_wrong_outputs(self):
        for path in ["slides.pptx", "report.docx", "data.xlsx", "paper.pdf", "plot.png", "video.mp4", "voice.mp3"]:
            handle = self.create(path, checks=False)
            server.edit_workspace(handle, 0, files=[server.SourceFile(path=path, content="fixture"), server.SourceFile(path="qa.txt", content="internal")])
            with self.assertRaisesRegex(ToolError, "exactly the declared"):
                server.publish_workspace(handle, 1, ["qa.txt"])
            completed = server.publish_workspace(handle, 1, [path])
            self.assertEqual(completed.meta["exoharness"]["workflow_continuation"]["status"], "complete")
            self.assertEqual([a["filename"] for a in completed.structured_content["artifacts"]], [path])

    def test_bad_contracts_rejected_before_creating_workspace(self):
        for data in [{"outputs": ["../escape.html"]}, {"outputs": ["/abs.html"]},
                     {"outputs": ["a.html", "sub/a.html"]}, {"outputs": ["run.exe"]},
                     {"outputs": ["a.html"], "browser_checks": ["different.html"]}]:
            with self.assertRaises(ValueError):
                Delivery(**data)
        self.assertEqual(list(self.rt.workspaces.root.iterdir()), [])

    def test_bounded_working_notes_survive_receipts_and_failed_edits_atomically(self):
        handle = self.create()
        notes = "Exact shared interface: .container, .hero; translations use data-en/data-zh. main is open; footer remains."
        saved = server.edit_workspace(handle, 0, files=[server.SourceFile(path="index.html", content="<main>")], notes=notes)
        self.assertEqual(saved.structured_content["notes"], notes)
        self.assertIn("/notes", saved.meta["exoharness"]["context_preserve"])
        # Intermediate saves never narrow the model context: the just-written
        # source and recent reads must stay visible for the next append.
        self.assertNotIn("context_scope", saved.meta["exoharness"]["workflow_continuation"])
        appended = server.edit_workspace(handle, 1, appends=[server.SourceFile(path="index.html", content="<h1 id=\"title\">Title</h1>")])
        receipt = appended.structured_content["changes"][0]
        self.assertEqual(receipt["element_ids"], ["h1#title"])
        self.assertTrue(receipt["tail"].endswith("</h1>"))
        self.assertEqual(server.read_workspace(handle).structured_content["notes"], notes)
        for revision, bad_notes in [(2, "x" * 2001), (1, "stale change")]:
            with self.assertRaises(ToolError):
                server.edit_workspace(handle, revision, files=[server.SourceFile(path="index.html", content="wrong")], notes=bad_notes)
            self.assertEqual(self.rt.workspaces.state(handle)["notes"], notes)
            self.assertEqual(self.rt.workspaces.bytes(handle, "index.html"), b"<main><h1 id=\"title\">Title</h1>")
        cleared = server.edit_workspace(handle, 2, notes="")
        self.assertEqual(cleared.structured_content["notes"], "")
        self.assertIn("/notes", cleared.meta["exoharness"]["context_preserve"])
        self.assertNotIn("context_scope", cleared.meta["exoharness"]["workflow_continuation"])

    def test_authoring_receipts_remain_small_and_exact_source_can_be_read_on_demand(self):
        handle = self.create()
        source = "<style>.existing{color:green}</style><main>"
        server.edit_workspace(handle, 0, files=[server.SourceFile(path="index.html", content=source)], notes="main open; use .existing")
        for revision in range(1, 4):
            chunk = "<p class='existing'>中文</p>" * 300
            source += chunk
            result = server.edit_workspace(handle, revision, appends=[server.SourceFile(path="index.html", content=chunk)])
            self.assertNotIn("recent_source", result.structured_content)
            self.assertLess(len(json.dumps(result.structured_content).encode()), 4096)
            self.assertEqual(self.rt.workspaces.bytes(handle, "index.html"), source.encode())
        found = server.read_workspace(handle, path="index.html", query=".existing{").structured_content
        self.assertEqual(found["revision"], 4)
        self.assertEqual(found["sha256"], hashlib.sha256(source.encode()).hexdigest())
        self.assertIn(".existing{color:green}", found["matches"][0]["content"])
