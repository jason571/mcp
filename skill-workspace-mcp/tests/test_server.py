import asyncio
import copy
import hashlib
import json
import tempfile
import time
import unittest
from unittest.mock import patch
from pathlib import Path
from urllib.parse import urlsplit

from starlette.testclient import TestClient
from skill_workspace import server
from skill_workspace.common import read, write
from mcp.server.mcpserver.exceptions import ToolError, UnexpectedToolError


class ServerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        package = self.root / "packages" / "fixture" / ("a" * 64)
        package.mkdir(parents=True)
        (package / "SKILL.md").write_text("Test instructions")
        self.rt = server.Runtime(self.root / "data", self.root / "packages", "test-key-" * 8, "http://testserver")
        write(self.rt.root / "worker.json", {"at": time.time(), "image": "sha256:" + "a" * 64})
        write(self.rt.root / "inventory.json", {"image": "sha256:" + "a" * 64, "commands":{"python":True,"node":True}})
        write(self.rt.root / "packages.json", {"fixture": ["a" * 64]})
        self.request = {"package_name": "fixture", "package_digest": "a" * 64, "runtime": "python",
            "files": [{"path": "main.py", "content": "print('test')"}], "entrypoint": "main.py", "outputs": ["report.pdf"]}

    def tearDown(self):
        server._runtime = None
        self.temp.cleanup()

    def test_duplicates_do_not_reset_working_or_failed_jobs(self):
        first = self.rt.submit(self.request)
        job = self.rt.jobs / first["job_id"]
        self.assertIn("background_task", first["_meta"]["exoharness"])
        self.assertEqual(self.rt.submit(self.request)["job_id"], first["job_id"])
        self.assertEqual(self.rt.submit({**self.request, "inputs": []})["job_id"], first["job_id"])
        for status in ["working", "failed"]:
            write(job / "state.json", {"status": status, "error": "test"})
            self.assertEqual(self.rt.submit(self.request)["status"], status)
        self.request["files"][0]["content"] = "print('corrected')"
        self.assertNotEqual(self.rt.submit(self.request)["job_id"], first["job_id"])

    def test_unready_runtime_and_missing_package_do_not_submit(self):
        self.request["package_digest"] = "b" * 64
        with self.assertRaises(ValueError): self.rt.submit(self.request)
        self.request["package_digest"] = "a" * 64
        (self.rt.root / "worker.json").unlink()
        with self.assertRaises(ValueError): self.rt.submit(self.request)
        self.assertEqual(list(self.rt.jobs.iterdir()), [])

    def test_live_mcp_schema_is_generic_and_typed(self):
        tools = asyncio.run(server.mcp.list_tools())
        execute = next(tool for tool in tools if tool.name == "execute_skill_script")
        schema = execute.input_schema
        self.assertIn("package_digest", schema["required"])
        self.assertEqual(schema["properties"]["runtime_name"]["enum"], ["python", "node", "typescript"])
        self.assertEqual(schema["properties"]["files"]["type"], "array")
        names = {tool.name for tool in tools}
        self.assertTrue({"execute_python", "run_script", "create_workspace", "read_workspace", "edit_workspace"}.issubset(names))
        python = next(tool for tool in tools if tool.name == "execute_python").input_schema
        self.assertIn("code", python["properties"])
        self.assertNotIn("code", python["required"], "saved entrypoints must be executable without inline code")
        shell = next(tool for tool in tools if tool.name == "run_script").input_schema
        self.assertEqual(shell["properties"]["runtime_name"]["enum"], ["shell", "python", "node", "typescript"])
        self.assertEqual(shell["properties"]["runtime_name"]["default"], "shell")
        self.assertIn("code", shell["properties"])
        self.assertNotIn("package_name", shell["properties"])
        self.assertTrue({"publish_workspace", "inspect_webpage"}.issubset(names))
        edit = next(tool for tool in tools if tool.name == "edit_workspace").input_schema
        self.assertIn("appends", edit["properties"])
        variants = edit["properties"]["files"]["items"]["anyOf"]
        required = [set(edit["$defs"][v["$ref"].rsplit("/", 1)[-1]]["required"]) for v in variants]
        self.assertEqual(required, [{"path", "content"}, {"path", "value"}])
        self.assertTrue(all(edit["$defs"][v["$ref"].rsplit("/", 1)[-1]]["additionalProperties"] is False for v in variants))
        reader = next(tool for tool in tools if tool.name == "read_workspace").input_schema
        self.assertIn("query", reader["properties"])
        self.assertNotIn("query", reader["required"], "existing pagination calls remain compatible")

    def test_workspace_preconditions_are_in_catalog_preview_and_wire_schema(self):
        from pydantic import TypeAdapter, ValidationError
        tools = {tool.name: tool for tool in asyncio.run(server.mcp.list_tools())}
        for name in ("execute_python", "run_script"):
            self.assertIn("create_workspace", tools[name].description[:96],
                          "deferred tool metadata must expose the prerequisite, not hide it after truncation")
        for name in ("read_workspace", "edit_workspace", "publish_workspace", "inspect_webpage", "execute_python", "run_script"):
            schema = tools[name].input_schema
            handle = schema["properties"]["workspace_id"]
            self.assertEqual(handle["pattern"], "^ws_[0-9a-f]{64}$")
            self.assertIn("create_workspace", handle["description"])
            self.assertIn("workspace_id", schema["required"])
            if name != "read_workspace":
                self.assertEqual(schema["properties"]["expected_revision"]["minimum"], 0)
        validator = TypeAdapter(server.WorkspaceHandle)
        for placeholder in ("factorial-100", "workspace", "../ws_" + "a" * 64, "ws_" + "a" * 63):
            with self.assertRaises(ValidationError):
                validator.validate_python(placeholder)
        self.assertEqual(validator.validate_python("ws_" + "a" * 64), "ws_" + "a" * 64)
        with self.assertRaises(ValidationError):
            TypeAdapter(server.WorkspaceRevision).validate_python(-1)

    def test_unknown_handle_does_not_create_workspace_or_execute(self):
        server._runtime = self.rt
        handle = "ws_" + "9" * 64
        with self.assertRaises(ToolError):
            server.execute_python(handle, 0, "print(42)")
        self.assertEqual(list(self.rt.workspaces.root.iterdir()), [])
        self.assertEqual(list(self.rt.jobs.iterdir()), [])

    def test_mcp_rejects_invented_handle_before_execution_and_accepts_returned_handle(self):
        self.standalone()
        existing = list(self.rt.workspaces.root.iterdir())
        for name, arguments in (
            ("execute_python", {"workspace_id": "factorial-100", "expected_revision": 0, "code": "print(42)"}),
            ("run_script", {"workspace_id": "factorial-100", "expected_revision": 0, "code": "echo 42"}),
        ):
            with self.assertRaises(ToolError):
                asyncio.run(server.mcp.call_tool(name, arguments))
        self.assertEqual(list(self.rt.workspaces.root.iterdir()), existing)
        self.assertEqual(list(self.rt.jobs.iterdir()), [])
        created = asyncio.run(server.mcp.call_tool("create_workspace", {})).structured_content
        handle = created["workspace_id"]
        with self.assertRaises(ToolError):
            asyncio.run(server.mcp.call_tool("execute_python", {
                "workspace_id": handle, "expected_revision": -1, "code": "print(42)"}))
        self.assertEqual(list(self.rt.jobs.iterdir()), [])
        job = asyncio.run(server.mcp.call_tool("execute_python", {
            "workspace_id": handle, "expected_revision": created["revision"], "code": "print(42)"})).structured_content
        self.assertEqual(job["status"], "queued")
        self.assertEqual(len(list(self.rt.jobs.iterdir())), 1)

    def test_incremental_source_append_is_exact_versioned_and_atomic(self):
        server._runtime = self.rt
        handle = server.create_workspace().structured_content["workspace_id"]
        first = '<!doctype html><html lang="zh"><body>永嘉'
        server.edit_workspace(handle, 0, files=[server.SourceFile(path="index.html", content=first)])
        chunk = ' / Yongjia\n</body></html>'
        receipt = server.edit_workspace(handle, 1, appends=[server.SourceFile(path="index.html", content=chunk)])
        self.assertEqual(self.rt.workspaces.bytes(handle, "index.html"), (first + chunk).encode())
        self.assertEqual(receipt.structured_content["revision"], 2)
        with self.assertRaisesRegex(ToolError, "revision conflict"):
            server.edit_workspace(handle, 1, appends=[server.SourceFile(path="index.html", content=chunk)])
        for appends in [
            [server.SourceFile(path="missing.html", content="x")],
            [server.SourceFile(path="../escape", content="x")],
        ]:
            with self.assertRaises(ToolError):
                server.edit_workspace(handle, 2, files=[server.SourceFile(path="should-not-exist", content="x")], appends=appends)
            self.assertEqual(self.rt.workspaces.state(handle)["revision"], 2)
            self.assertEqual(self.rt.workspaces.bytes(handle, "index.html"), (first + chunk).encode())
            self.assertNotIn("should-not-exist", self.rt.workspaces.state(handle)["files"])
        self.assertEqual(list(self.rt.jobs.iterdir()), [], "Appending does not execute code")

    def test_same_path_appends_merge_in_order_with_one_revision_and_one_receipt(self):
        server._runtime = self.rt
        handle = server.create_workspace().structured_content["workspace_id"]
        prefix = '<!doctype html><html><body><main>'
        server.edit_workspace(handle, 0, files=[server.SourceFile(path="index.html", content=prefix),
                                               server.SourceFile(path="style.css", content="/* base */")])
        chunks = [server.SourceFile(path="index.html", content="<section>永嘉 / Yongjia</section>"),
                  server.SourceFile(path="style.css", content="body{color:#123}"),
                  server.SourceFile(path="index.html", content="</main><footer>页脚</footer>"),
                  server.SourceFile(path="index.html", content="</body></html>")]
        receipt = server.edit_workspace(handle, 1, appends=chunks, notes="Page closed; checks pending").structured_content
        expected = prefix + "".join(item.content for item in chunks if item.path == "index.html")
        self.assertEqual(self.rt.workspaces.bytes(handle, "index.html"), expected.encode())
        self.assertEqual(self.rt.workspaces.bytes(handle, "style.css"), b"/* base */body{color:#123}")
        self.assertEqual(receipt["revision"], 2)
        self.assertEqual([item["path"] for item in receipt["changes"]], ["index.html", "style.css"])
        self.assertEqual(receipt["changes"][0]["sha256"], hashlib.sha256(expected.encode()).hexdigest())
        with self.assertRaisesRegex(ToolError, "revision conflict"):
            server.edit_workspace(handle, 1, appends=chunks)
        self.assertEqual(self.rt.workspaces.bytes(handle, "index.html"), expected.encode())
        self.assertEqual(list(self.rt.jobs.iterdir()), [])

    def test_merge_appends_keeps_atomicity_conflicts_limits_and_exact_duplicate_text(self):
        handle = self.standalone()
        saved = server.edit_workspace(handle, 0, files=[server.SourceFile(path="text.txt", content="base")]).structured_content
        chunks = [server.SourceFile(path="text.txt", content="同"), server.SourceFile(path="text.txt", content="同")]
        for edits in [
            {"files": [server.SourceFile(path="text.txt", content="overwrite")]},
            {"patches": [server.FilePatch(path="text.txt", sha256=saved["files"][0]["sha256"], before="base", after="patch")]},
            {"inputs": [server.InputFile(path="text.txt", url="https://untrusted.example/file")]},
        ]:
            with self.assertRaisesRegex(ToolError, "one edit mode"):
                server.edit_workspace(handle, 1, appends=chunks, **edits)
            self.assertEqual(self.rt.workspaces.state(handle)["revision"], 1)
            self.assertEqual(self.rt.workspaces.bytes(handle, "text.txt"), b"base")
        with patch.object(self.rt, "input_file", side_effect=AssertionError("must not download")):
            with self.assertRaisesRegex(ToolError, "one edit mode"):
                server.edit_workspace(handle, 1, appends=chunks, inputs=[server.InputFile(path="text.txt", url="https://untrusted.example/file")])
        with self.assertRaises(ToolError):
            server.edit_workspace(handle, 1, appends=[*chunks, server.SourceFile(path="missing.txt", content="invalid")], notes="not committed")
        self.assertEqual(self.rt.workspaces.state(handle)["revision"], 1)
        self.assertNotIn("notes", self.rt.workspaces.state(handle))
        for limit in ("MAX_FILE", "MAX_OUTPUT"):
            with patch("skill_workspace.workspaces." + limit, 9), self.assertRaises(ToolError):
                server.edit_workspace(handle, 1, appends=chunks)
            self.assertEqual(self.rt.workspaces.bytes(handle, "text.txt"), b"base")
        with self.assertRaisesRegex(ToolError, "at most 24"):
            server.edit_workspace(handle, 1, appends=[server.SourceFile(path="text.txt", content="x")] * 25)
        self.rt.workspaces.edit(handle, 1, [], [], [], [item.model_dump() for item in chunks])
        self.assertEqual(self.rt.workspaces.bytes(handle, "text.txt"), "base同同".encode())
        self.assertEqual(self.rt.workspaces.state(handle)["revision"], 2)

    def test_append_rejects_binary_and_keeps_existing_patch_contract(self):
        server._runtime = self.rt
        handle = server.create_workspace().structured_content["workspace_id"]
        self.rt.workspaces.edit(handle, 0, [{"path":"text.txt", "content":"original"}], [], [{"path":"binary", "bytes":b"\xff"}])
        with self.assertRaisesRegex(ToolError, "UTF-8"):
            server.edit_workspace(handle, 1, appends=[server.SourceFile(path="binary", content="x")])
        with self.assertRaisesRegex(ToolError, "once"):
            server.edit_workspace(handle, 1, files=[server.SourceFile(path="text.txt", content="replacement")], appends=[server.SourceFile(path="text.txt", content="suffix")])
        digest = self.rt.workspaces.state(handle)["files"]["text.txt"]["sha256"]
        server.edit_workspace(handle, 1, patches=[server.FilePatch(path="text.txt", sha256=digest, before="original", after="fixed")])
        self.assertEqual(self.rt.workspaces.bytes(handle, "text.txt"), b"fixed")

    def test_append_obeys_busy_and_size_limits_without_changing_revision(self):
        server._runtime = self.rt
        handle = server.create_workspace().structured_content["workspace_id"]
        server.edit_workspace(handle, 0, files=[server.SourceFile(path="text.txt", content="12345678")])
        for limit in ("MAX_FILE", "MAX_OUTPUT"):
            with patch("skill_workspace.workspaces." + limit, 8), self.assertRaises(ToolError):
                server.edit_workspace(handle, 1, appends=[server.SourceFile(path="text.txt", content="9")])
            self.assertEqual(self.rt.workspaces.state(handle)["revision"], 1)
            self.assertEqual(self.rt.workspaces.bytes(handle, "text.txt"), b"12345678")
        job_id = "work_" + "b" * 64
        job = self.rt.jobs / job_id
        job.mkdir()
        write(job / "state.json", {"status":"working"})
        state = self.rt.workspaces.state(handle)
        state["active_job"] = job_id
        write(self.rt.workspaces.directory(handle) / "state.json", state)
        with self.assertRaisesRegex(ToolError, "busy"):
            server.edit_workspace(handle, 1, appends=[server.SourceFile(path="text.txt", content="9")])
        self.assertEqual(self.rt.workspaces.state(handle)["revision"], 1)

    def standalone(self):
        server._runtime = self.rt
        write(self.rt.root / "task-code.json", {"enabled": True, "image": "sha256:" + "a" * 64})
        return server.create_workspace().structured_content["workspace_id"]

    def test_execute_python_reruns_edited_file_without_overwriting_source(self):
        handle = self.standalone()
        server.edit_workspace(handle, 0, files=[server.SourceFile(path="main.py", content="print('fixed')")])
        with self.assertRaisesRegex(ToolError, "overwrite"):
            server.execute_python(handle, 1, code="print('replacement')")
        self.assertEqual(self.rt.workspaces.bytes(handle, "main.py"), b"print('fixed')")
        submitted = server.execute_python(handle, 1, outputs=["chart.png"]).structured_content
        request = read(self.rt.jobs / submitted["job_id"] / "request.json")
        self.assertEqual(request["entrypoint"], "main.py")
        self.assertEqual(request["files"], [])
        self.assertEqual(request["workspace_revision"], 1)
        self.assertEqual(self.rt.workspaces.bytes(handle, "main.py"), b"print('fixed')")

    def test_publish_existing_html_is_immutable_idempotent_and_never_executes(self):
        handle = self.standalone()
        html = '<!doctype html><meta charset="utf-8"><h1>茶园</h1><script>console.log(42)</script>'
        server.edit_workspace(handle, 0, files=[server.SourceFile(path="index.html", content=html)])
        result = server.publish_workspace(handle, 1, ["index.html"])
        first = result.structured_content
        self.assertFalse(result.is_error)
        self.assertFalse(first["code_executed"])
        self.assertEqual(first["workspace"]["revision"], 1)
        self.assertEqual(first["artifacts"][0]["content_type"], "text/html")
        self.assertEqual(self.rt.input_bytes(first["artifacts"][0]["download_url"]), html.encode())
        self.assertFalse((self.rt.jobs / first["job_id"] / "request.json").exists())
        self.assertEqual(server.publish_workspace(handle, 1, ["index.html"]).structured_content["job_id"], first["job_id"])
        self.assertEqual(self.rt.workspaces.state(handle)["revision"], 1)
        changed = html.replace("茶园", "山谷")
        server.edit_workspace(handle, 1, files=[server.SourceFile(path="index.html", content=changed)])
        second = server.publish_workspace(handle, 2, ["index.html"]).structured_content
        self.assertNotEqual(first["job_id"], second["job_id"])
        self.assertEqual(self.rt.input_bytes(first["artifacts"][0]["download_url"]), html.encode())
        a, b = [v["artifacts"][0]["_meta"]["exoharness"]["artifact_identity"] for v in [first, second]]
        self.assertEqual(a["key"], b["key"])
        self.assertEqual((a["revision"], b["revision"]), (1, 2))
        self.assertEqual(self.rt.status(first["job_id"])["workspace"]["revision"], 1)
        for invalid in [["../index.html"], ["absent.html"], ["index.html", "index.html"]]:
            with self.assertRaises(ToolError): server.publish_workspace(handle, 2, invalid)
        with self.assertRaisesRegex(ToolError, "revision conflict"): server.publish_workspace(handle, 1, ["index.html"])

    def test_publication_preserves_office_media_and_download_security(self):
        handle = self.standalone()
        names = ["slides.pptx", "report.docx", "data.xlsx", "paper.pdf", "video.mp4", "voice.mp3", "plot.png", "page.html"]
        server.edit_workspace(handle, 0, files=[server.SourceFile(path=n, content="fixture") for n in names])
        published = server.publish_workspace(handle, 1, names).structured_content
        client = TestClient(server.mcp.streamable_http_app())
        item = next(f for f in published["artifacts"] if f["filename"] == "page.html")
        response = client.get(item["download_url"])
        self.assertEqual(response.status_code, 200)
        self.assertIn("attachment", response.headers["content-disposition"])
        self.assertEqual(response.headers["x-content-type-options"], "nosniff")
        self.assertEqual(client.get(item["download_url"].split("?")[0]).status_code, 404)
        self.assertEqual(len(published["artifacts"]), len(names))

    def test_followup_recovers_workspace_from_publication_and_patches_only_requested_text(self):
        handle = self.standalone()
        original = '<!doctype html><title>Original</title><main>Retain this exact content.</main>'
        server.edit_workspace(handle, 0, files=[server.SourceFile(path="index.html", content=original)])
        published = server.publish_workspace(handle, 1, ["index.html"]).structured_content
        # A new turn may retain only the earlier download, not the full tool history.
        # Its exact Job ID resolves the source; never synthesize a Workspace ID.
        url = published["artifacts"][0]["download_url"]
        job_id = urlsplit(url).path.split("/")[-2]
        receipt = server.get_script_job(job_id).structured_content
        self.assertEqual(receipt["workspace"]["workspace_id"], handle)
        source = server.read_workspace(receipt["workspace"]["workspace_id"], "index.html", limit=100).structured_content
        patch = server.FilePatch(path="index.html", sha256=source["sha256"],
                                 before="<title>Original</title>", after="<title>Updated</title>")
        server.edit_workspace(handle, source["revision"], patches=[patch])
        current = server.publish_workspace(handle, 2, ["index.html"]).structured_content
        self.assertEqual(self.rt.input_bytes(current["artifacts"][0]["download_url"]),
                         original.replace("<title>Original</title>", "<title>Updated</title>").encode())
        self.assertEqual(self.rt.input_bytes(url), original.encode(), "Prior download stays immutable")
        # Old publication receipts keep their original revision; source reads return current state.
        self.assertEqual(server.get_script_job(job_id).structured_content["workspace"]["revision"], 1)
        self.assertEqual(server.read_workspace(handle).structured_content["revision"], 2)

    def test_signed_html_preview_is_inline_authenticated_and_integrity_checked(self):
        handle = self.standalone()
        html = '<!doctype html><title>Preview</title><h1>网页</h1><script>window.ready=true</script>'
        server.edit_workspace(handle, 0, files=[server.SourceFile(path="index.html", content=html),
                                               server.SourceFile(path="report.pdf", content="%PDF-fixture")])
        published = server.publish_workspace(handle, 1, ["index.html", "report.pdf"]).structured_content
        item, pdf = published["artifacts"]
        # URLs are projected for existing publications too; no regeneration or manifest rewrite.
        stored = read(self.rt.jobs / published["job_id"] / "result.json")
        self.assertNotIn("preview_url", stored["files"][0])
        self.assertNotIn("preview_url", pdf)
        client = TestClient(server.mcp.streamable_http_app())
        response = client.get(item["preview_url"])
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.content, html.encode())
        self.assertEqual(response.headers["content-disposition"], "inline")
        self.assertEqual(response.headers["content-security-policy"], server.HTML_PREVIEW_CSP)
        self.assertNotIn("allow-same-origin", server.HTML_PREVIEW_CSP)
        self.assertIn("connect-src 'none'", server.HTML_PREVIEW_CSP)
        self.assertEqual(response.headers["referrer-policy"], "no-referrer")
        head = client.head(item["preview_url"])
        self.assertEqual(head.status_code, 200)
        self.assertEqual(head.content, b"")
        self.assertEqual(int(head.headers["content-length"]), len(html.encode()))
        self.assertIn("attachment", client.get(item["download_url"]).headers["content-disposition"])
        for invalid in [item["preview_url"].split("?")[0], item["preview_url"].replace("signature=", "signature=wrong"),
                        self.rt.artifact_preview_url(published["job_id"], "report.pdf")]:
            self.assertEqual(client.get(invalid).status_code, 404)
        path = self.rt.jobs / published["job_id"] / "published/index.html"
        path.write_bytes(html.encode().replace(b"Preview", b"Tamper!"))
        self.assertEqual(client.get(item["preview_url"]).status_code, 404, "Same-length tampering must be rejected")

    def test_large_or_nonterminal_html_does_not_become_a_preview(self):
        handle = self.standalone()
        # Exercise the preview boundary independently of the edit request size limit.
        with patch.object(server, "MAX_HTML_PREVIEW_BYTES", 128):
            server.edit_workspace(handle, 0, files=[server.SourceFile(path="large.html", content="x" * 129)])
            published = server.publish_workspace(handle, 1, ["large.html"]).structured_content
            self.assertNotIn("preview_url", published["artifacts"][0])
            client = TestClient(server.mcp.streamable_http_app())
            url = self.rt.artifact_preview_url(published["job_id"], "large.html")
            self.assertEqual(client.get(url).status_code, 413)
            write(self.rt.jobs / published["job_id"] / "state.json", {"status":"cancelled"})
            self.assertEqual(client.get(url).status_code, 404)

    def test_rejected_source_can_be_restored_without_resending_or_execution(self):
        handle = self.standalone()
        source = "from pathlib import Path\nPath('result.txt').write_text('中文\\n')\n"
        failed = server.execute_python(handle, 0, source, outputs=["result.unsupported"])
        self.assertTrue(failed.is_error)
        value = failed.structured_content
        self.assertEqual(value["status"], "rejected")
        self.assertEqual(value["recovery"]["tool"], "edit_workspace")
        self.assertEqual(self.rt.workspaces.inspect(handle)["files"], [])
        self.assertEqual(list(self.rt.jobs.iterdir()), [])
        self.assertNotIn(source, str(value))
        restored = server.edit_workspace(**value["recovery"]["arguments"]).structured_content
        self.assertEqual(restored["revision"], 1)
        self.assertEqual(self.rt.workspaces.inspect(handle, "main.py")["content"], source)
        self.assertEqual(list(self.rt.jobs.iterdir()), [])
        self.assertEqual(server.edit_workspace(handle, 1, draft_id=value["draft"]["id"]).structured_content["revision"], 1)
        with self.assertRaises(ToolError):
            server.edit_workspace(self.standalone(), 0, draft_id=value["draft"]["id"])
        server.edit_workspace(handle, 1, files=[server.SourceFile(path="main.py", content="changed")])
        with self.assertRaisesRegex(ToolError, "overwrite"):
            server.edit_workspace(handle, 2, draft_id=value["draft"]["id"])
        self.assertEqual(self.rt.workspaces.inspect(handle, "main.py")["content"], "changed")

    def test_successful_edit_has_exact_file_checkpoint_but_reads_still_return_source(self):
        handle = self.standalone()
        code = "# retained exact source\n" * 4000
        edited = server.edit_workspace(handle, 0, files=[server.SourceFile(path="large.py", content=code)])
        wire = edited.model_dump(by_alias=True)
        meta = wire["_meta"]["exoharness"]
        self.assertIn("context_checkpoint", meta)
        for pointer in meta["context_preserve"]:
            self.assertIn(pointer[1:], wire["structuredContent"])
        self.assertNotIn("workflow_continuation", meta)
        self.assertNotIn(code, str(wire))
        self.assertEqual(self.rt.workspaces.bytes(handle,"large.py"), code.encode())
        page = server.read_workspace(handle,"large.py",0,160).model_dump(by_alias=True)
        self.assertEqual(page["structuredContent"]["content"], code[:160])
        self.assertEqual(set(page["_meta"]["exoharness"]), {"immutable_read"})
        self.assertEqual(page["_meta"]["exoharness"]["immutable_read"]["ranges"], [[0,160]])
        self.assertNotIn("context_checkpoint", page["_meta"]["exoharness"], "A requested source slice must reach the model before checkpoint compaction")

    def test_source_search_returns_bounded_exact_unicode_slices_and_revision_identity(self):
        handle = self.standalone()
        code = "茶园🌿\n" + (".hero {width:100%}\n" + "正文。" * 500 + "\n") * 9
        server.edit_workspace(handle, 0, files=[server.SourceFile(path="index.html",content=code)])
        page = server.read_workspace(handle, "index.html", query=".hero", limit=1300).model_dump(by_alias=True)
        value, meta = page["structuredContent"], page["_meta"]["exoharness"]["immutable_read"]
        self.assertEqual(value["total_chars"], len(code))
        self.assertEqual(value["offset_unit"], "unicode_characters")
        self.assertLessEqual(sum(len(m["content"]) for m in value["matches"]), 1300)
        self.assertLessEqual(len(value["matches"]), 4)
        for m in value["matches"]:
            self.assertEqual(m["content"], code[m["offset"]:m["end_offset"]])
            self.assertEqual(m["line"], code.count("\n",0,m["match_offset"])+1)
            self.assertTrue(code[m["match_offset"]:].startswith(".hero"))
        self.assertIsNotNone(value["next_offset"])
        self.assertEqual(meta["revision"], hashlib.sha256(code.encode()).hexdigest())
        next_page = server.read_workspace(handle,"index.html", offset=value["next_offset"],query=".hero").structured_content
        self.assertGreater(next_page["matches"][0]["match_offset"], value["matches"][-1]["match_offset"])
        self.assertEqual(server.read_workspace(handle,"index.html",query="not-found").structured_content["matches"], [])
        final = server.read_workspace(handle,"index.html",offset=len(code)-3,limit=30).structured_content
        self.assertEqual(final["content"], code[-3:])
        self.assertIsNone(final["next_offset"])
        for kwargs in [{"query":""},{"query":"x"*257},{"query":"hero","limit":0},{"offset":-1}]:
            with self.assertRaises(ToolError): server.read_workspace(handle,"index.html",**kwargs)
        with self.assertRaises(ToolError): server.read_workspace(handle,query="hero")

    def test_drafts_never_bypass_authorization_or_accept_invalid_source(self):
        handle = self.standalone()
        write(self.rt.root / "task-code.json", {"enabled": False})
        with self.assertRaises(ToolError): server.execute_python(handle, 0, "print('x')", outputs=["x.bad"])
        self.assertFalse((self.rt.workspaces.directory(handle) / "drafts").exists())
        write(self.rt.root / "task-code.json", {"enabled": True, "image": "sha256:" + "a" * 64})
        with self.assertRaises(ToolError): server.execute_python(handle, 0, "print('x')", path="../x.py")
        self.assertEqual(list(self.rt.jobs.iterdir()), [])
        value = server.execute_python(handle, 0, "print('exact')", outputs=["x.bad"]).structured_content
        path = self.rt.workspaces.directory(handle) / "drafts" / value["draft"]["id"]
        self.assertEqual(path.stat().st_mode & 0o777, 0o600)
        path.write_text('{"files":[],"workspace_id":"tampered"}')
        with self.assertRaisesRegex(ToolError, "integrity"):
            server.edit_workspace(handle, 0, draft_id=value["draft"]["id"])

    def test_browser_requires_independent_admin_image_and_retains_source(self):
        handle = self.standalone()
        server.edit_workspace(handle, 0, files=[server.SourceFile(path="index.html", content="<h1>Test</h1>")])
        with self.assertRaisesRegex(ToolError, "not prepared"): server.inspect_webpage(handle, 1, "index.html")
        inventory = read(self.rt.root / "inventory.json")
        inventory["browser_image"] = "sha256:" + "b" * 64
        write(self.rt.root / "inventory.json", inventory)
        job = server.inspect_webpage(handle, 1, "index.html").structured_content
        request = read(self.rt.jobs / job["job_id"] / "request.json")
        self.assertEqual(request["operation"], "inspect_webpage")
        self.assertEqual(request["runtime_image"], "sha256:" + "a" * 64)
        self.assertEqual(request["browser_image"], "sha256:" + "b" * 64)
        self.assertEqual(request["observer_revision"], server.browser_revision())
        self.assertEqual(self.rt.workspaces.state(handle)["revision"], 1)
        self.assertEqual(request["files"], [])
        with self.assertRaisesRegex(ToolError, "busy"): server.publish_workspace(handle, 1, ["index.html"])
        self.assertIn("cancel", self.rt.status(job["job_id"], True)["_meta"]["exoharness"]["background_task"])

    def test_browser_does_not_require_or_mount_bound_skill_runtime(self):
        server._runtime = self.rt
        write(self.rt.root / "packages.json", {})
        workspace = server.create_workspace("fixture", "a" * 64).structured_content
        handle = workspace["workspace_id"]
        server.edit_workspace(handle, 0, files=[server.SourceFile(
            path="index.html", content="<h1>Package-independent preview</h1>")])
        inventory = read(self.rt.root / "inventory.json")
        inventory["browser_image"] = "sha256:" + "b" * 64
        write(self.rt.root / "inventory.json", inventory)
        result = server.inspect_webpage(handle, 1, "index.html").structured_content
        request = read(self.rt.jobs / result["job_id"] / "request.json")
        self.assertIsNone(request["package_name"])
        self.assertIsNone(request["package_digest"])
        self.assertEqual(request["operation"], "inspect_webpage")
        with self.assertRaisesRegex(ToolError, "not approved"):
            server.execute_python(handle, 1, "print('still requires Skill approval')")

    def test_inline_shell_needs_only_task_workspace_not_a_skill(self):
        server._runtime = self.rt
        write(self.rt.root / "task-code.json", {"enabled": True, "image": "sha256:" + "a" * 64})
        write(self.rt.root / "packages.json", {})
        workspace = server.create_workspace().structured_content
        result = server.run_script(workspace["workspace_id"], 0, code="printf 'hello' | tr a-z A-Z > result.txt", outputs=["result.txt"]).structured_content
        request = read(self.rt.jobs / result["job_id"] / "request.json")
        self.assertEqual(request["runtime"], "shell")
        self.assertIsNone(request["package_name"])
        self.assertEqual(request["entrypoint"], "script.sh")
        self.assertEqual(self.rt.workspaces.inspect(workspace["workspace_id"], "script.sh")["content"], request["files"][0]["content"])
        self.assertEqual(server.run_script(workspace["workspace_id"], 0, code=request["files"][0]["content"], outputs=["result.txt"]).structured_content["job_id"], result["job_id"])

    def test_typescript_advertises_only_ready_toolchain_and_uses_generic_workspace(self):
        server._runtime = self.rt
        write(self.rt.root / "task-code.json", {"enabled": True, "image": "sha256:" + "a" * 64})
        workspace = server.create_workspace().structured_content
        self.assertNotIn("typescript", self.rt.info()["runtimes"])
        with self.assertRaisesRegex(ToolError, "TypeScript runtime is not prepared"):
            server.run_script(workspace["workspace_id"], 0, runtime_name="typescript", entrypoint="main.ts", code="console.log(42)")
        self.assertEqual(list(self.rt.jobs.iterdir()), [])
        write(self.rt.root / "inventory.json", {"image": "sha256:" + "a" * 64, "commands":{"node":True,"tsc":True,"tsx":True}})
        self.assertIn("typescript", self.rt.info()["runtimes"])
        result = server.run_script(workspace["workspace_id"], 0, runtime_name="typescript", entrypoint="main.ts", code="const answer: number=42; console.log(answer)").structured_content
        request = read(self.rt.jobs / result["job_id"] / "request.json")
        self.assertEqual(request["runtime"], "typescript")
        self.assertIsNone(request["package_name"])

    def test_private_workspaces_atomic_patch_and_revision_conflict(self):
        a, b = self.rt.workspaces.create(), self.rt.workspaces.create()
        self.assertNotEqual(a["workspace_id"], b["workspace_id"])
        handle = a["workspace_id"]
        changed = self.rt.workspaces.edit(handle, 0, [{"path":"main.py", "content":"print('bad')\n"}], [], [])
        sha = changed["files"][0]["sha256"]
        with self.assertRaisesRegex(ValueError, "not found"):
            self.rt.workspaces.inspect(b["workspace_id"], "main.py")
        with self.assertRaisesRegex(ValueError, "revision conflict"):
            self.rt.workspaces.edit(handle, 0, [], [], [])
        with self.assertRaisesRegex(ValueError, "exactly once"):
            self.rt.workspaces.edit(handle, 1, [{"path":"other.py", "content":"no"}], [{"path":"main.py", "sha256":sha, "before":"missing", "after":"ok"}], [])
        self.assertEqual(len(self.rt.workspaces.inspect(handle)["files"]), 1)
        corrected = self.rt.workspaces.edit(handle, 1, [], [{"path":"main.py", "sha256":sha, "before":"'bad'", "after":"'ok'"}], [])
        self.assertEqual(corrected["revision"], 2)
        self.assertEqual(self.rt.workspaces.inspect(handle, "main.py")["content"], "print('ok')\n")
        for bad in ["../secret", "/etc/passwd", "a//b"]:
            with self.assertRaises(ValueError):
                self.rt.workspaces.edit(handle, 2, [{"path":bad, "content":"escape"}], [], [])

    def test_multiple_exact_patches_share_one_base_and_commit_atomically(self):
        server._runtime = self.rt
        handle = server.create_workspace().structured_content["workspace_id"]
        original = "标题\n.grid{columns:3}\n.mobile{columns:2}\n结尾\n"
        server.edit_workspace(handle, 0, files=[server.SourceFile(path="page.css", content=original)])
        sha = self.rt.workspaces.inspect(handle, "page.css")["sha256"]
        def hunk(before, after, digest=sha):
            return server.FilePatch(path="page.css", sha256=digest, before=before, after=after)
        for hunks in ([hunk("columns:3", "columns:1"), hunk("missing", "x")],
                      [hunk(".grid{columns:3}", "x"), hunk("columns:3", "y")],
                      [hunk("columns:3", "x"), hunk("columns:2", "y", "0" * 64)]):
            with self.assertRaises(ToolError):
                server.edit_workspace(handle, 1, patches=hunks, notes="not committed")
            self.assertEqual(self.rt.workspaces.inspect(handle, "page.css")["content"], original)
            self.assertEqual(self.rt.workspaces.state(handle)["revision"], 1)
            self.assertNotIn("notes", self.rt.workspaces.state(handle))
        changed = server.edit_workspace(handle, 1, patches=[hunk("columns:3", "columns:2"), hunk("columns:2", "columns:1")])
        self.assertEqual(changed.structured_content["revision"], 2)
        self.assertEqual(self.rt.workspaces.inspect(handle, "page.css")["content"],
                         original.replace(".grid{columns:3}", ".grid{columns:2}").replace(".mobile{columns:2}", ".mobile{columns:1}"))
        with self.assertRaisesRegex(ToolError, "revision conflict"):
            server.edit_workspace(handle, 1, patches=[hunk("columns:3", "x")])

    def test_overlapping_occurrences_are_not_a_unique_patch(self):
        server._runtime = self.rt
        handle = server.create_workspace().structured_content["workspace_id"]
        server.edit_workspace(handle, 0, files=[server.SourceFile(path="code.txt", content="aaa")])
        sha = self.rt.workspaces.inspect(handle, "code.txt")["sha256"]
        with self.assertRaisesRegex(ToolError, "exactly once"):
            server.edit_workspace(handle, 1, patches=[server.FilePatch(path="code.txt", sha256=sha, before="aa", after="b")])
        self.assertEqual(self.rt.workspaces.inspect(handle, "code.txt")["content"], "aaa")

    def test_general_code_requires_admin_and_shares_cancel_and_idempotency(self):
        handle = self.rt.workspaces.create()["workspace_id"]
        server._runtime = self.rt
        req = server.workspace_request(handle, 0, "python", "main.py", [], [], files=[{"path":"main.py", "content":"print(1)"}])
        with self.assertRaisesRegex(ValueError, "not approved"):
            self.rt.submit(req)
        self.assertEqual(list(self.rt.jobs.iterdir()), [])
        write(self.rt.root / "task-code.json", {"enabled":True, "image":"sha256:" + "a" * 64})
        first = self.rt.submit(req)
        self.assertEqual(self.rt.submit(req)["job_id"], first["job_id"])
        self.assertEqual(first["workspace"]["file_count"], 1)
        self.assertEqual(self.rt.workspaces.inspect(handle)["files"][0]["path"], "main.py")
        with self.assertRaisesRegex(ValueError, "busy"):
            self.rt.workspaces.edit(handle, 0, [], [], [])
        result = server.cancel_script_job(first["job_id"])
        self.assertTrue((self.rt.jobs / first["job_id"] / "cancel").is_file())
        self.assertEqual(result.structured_content["status"], "queued")
        self.assertIn("cancel", first["_meta"]["exoharness"]["background_task"])

    def test_runtime_image_change_revokes_general_and_skill_approval(self):
        write(self.rt.root / "bindings.json", {"fixture:" + "a" * 64:{"image":"sha256:" + "b" * 64, "runtime":"python"}})
        with self.assertRaisesRegex(ValueError, "image changed"):
            self.rt.submit(self.request)

    def test_existing_unapproved_skill_cannot_execute(self):
        write(self.rt.root / "packages.json", {})
        with self.assertRaisesRegex(ValueError, "not approved"):
            self.rt.submit(self.request)
        self.assertEqual(list(self.rt.jobs.iterdir()), [])

    def test_mcp_returns_policy_and_runtime_errors_without_hiding_or_executing(self):
        server._runtime = self.rt
        write(self.rt.root / "packages.json", {})
        # A real executable/resource package must never be downgraded to an
        # instruction reference, even when general task code is permitted.
        package = self.root / "packages" / "fixture" / ("a" * 64)
        (package / "scripts").mkdir()
        workspace = server.create_workspace("fixture", "a" * 64).structured_content
        with self.assertRaisesRegex(ToolError, "not approved for the isolated runtime") as denied:
            asyncio.run(server.mcp.call_tool("run_script", {
                "workspace_id": workspace["workspace_id"], "expected_revision": 0,
                "code": "echo test", "entrypoint": "check.sh"}))
        self.assertNotIsInstance(denied.exception, UnexpectedToolError)
        with self.assertRaisesRegex(ToolError, "requires runtime_name=node"):
            asyncio.run(server.mcp.call_tool("execute_skill_script", {
                "package_name": "fixture", "package_digest": "a" * 64,
                "runtime_name": "python", "entrypoint": "build.js", "outputs": [],
                "files": [{"path": "build.js", "content": "console.log(1)"}]}))
        self.assertEqual(list(self.rt.jobs.iterdir()), [])
        self.assertEqual(self.rt.workspaces.inspect(workspace["workspace_id"])["files"], [])
        # Unexpected server errors still must not leak paths, tokens or stacks.
        with patch.object(self.rt.workspaces, "inspect", side_effect=RuntimeError("private-secret-sentinel")):
            with self.assertRaises(UnexpectedToolError) as crashed:
                asyncio.run(server.mcp.call_tool("read_workspace", {"workspace_id": workspace["workspace_id"]}))
            self.assertNotIn("private-secret-sentinel", str(crashed.exception))

    def test_instruction_only_reference_uses_task_policy_without_package_mount(self):
        server._runtime = self.rt
        write(self.rt.root / "packages.json", {})
        write(self.rt.root / "task-code.json", {"enabled": True, "image": "sha256:" + "a" * 64})
        for language, entrypoint, source in [("python", "check.py", "print(42)"),
                                             ("node", "check.js", "console.log(42)"),
                                             ("shell", "check.sh", "printf 42")]:
            with self.subTest(language=language):
                workspace = server.create_workspace("fixture", "a" * 64).structured_content
                self.assertIsNone(workspace["package_name"])
                self.assertEqual(workspace["instruction_reference"], {"package_name": "fixture", "package_digest": "a" * 64})
                job = server.run_script(workspace["workspace_id"], 0, runtime_name=language,
                                        entrypoint=entrypoint, code=source).structured_content
                request = read(self.rt.jobs / job["job_id"] / "request.json")
                self.assertIsNone(request["package_name"])
                self.assertIsNone(request["package_digest"])
                from skill_workspace.broker import container_args
                args = container_args(self.rt.jobs / job["job_id"], None, request["runtime_image"],
                                      Path(server.__file__).with_name("sandbox.py"))
                self.assertEqual(args[args.index("--network") + 1], "none")
                self.assertIn("--read-only", args)
                self.assertFalse(any("dst=/skill" in value or "docker.sock" in value for value in args))

    def test_legacy_instruction_binding_preserves_sources_identity_revision_and_duplicate_job(self):
        server._runtime = self.rt
        write(self.rt.root / "packages.json", {})
        write(self.rt.root / "task-code.json", {"enabled": True, "image": "sha256:" + "a" * 64})
        workspace = self.rt.workspaces.create("fixture", "a" * 64, "sha256:" + "a" * 64)
        handle = workspace["workspace_id"]
        saved = server.edit_workspace(handle, 0, files=[server.SourceFile(path="index.html", content="<h1>永嘉</h1>")]).structured_content
        before = copy.deepcopy(self.rt.workspaces.state(handle))
        job = server.execute_python(handle, 1, "from pathlib import Path\nassert '永嘉' in Path('index.html').read_text()").structured_content
        request = read(self.rt.jobs / job["job_id"] / "request.json")
        self.assertIsNone(request["package_name"])
        self.assertEqual(request["instruction_reference"]["package_name"], "fixture")
        after = self.rt.workspaces.state(handle)
        self.assertEqual(after["revision"], saved["revision"])
        self.assertEqual(after["package_name"], before["package_name"])
        self.assertEqual(self.rt.workspaces.bytes(handle, "index.html"), b"<h1>\xe6\xb0\xb8\xe5\x98\x89</h1>")
        self.assertEqual(server.execute_python(handle, 1, request["files"][0]["content"]).structured_content["job_id"], job["job_id"])
        write(self.rt.jobs / job["job_id"] / "state.json", {"status": "failed", "error": "fixture"})
        write(self.rt.root / "packages.json", {"fixture": ["a" * 64]})
        self.assertEqual(server.execute_python(handle, 1, request["files"][0]["content"]).structured_content["job_id"], job["job_id"])
        self.assertEqual(len(list(self.rt.jobs.iterdir())), 1)

    def test_instruction_reference_cannot_grant_task_execution_or_skill_entrypoints(self):
        server._runtime = self.rt
        write(self.rt.root / "packages.json", {})
        legacy = self.rt.workspaces.create("fixture", "a" * 64, "sha256:" + "a" * 64)
        handle = legacy["workspace_id"]
        for policy in [{}, {"enabled": False, "image": "sha256:" + "a" * 64},
                       {"enabled": True, "image": "sha256:" + "b" * 64}]:
            write(self.rt.root / "task-code.json", policy)
            with self.assertRaisesRegex(ToolError, "task code runtime is not approved"):
                server.execute_python(handle, 0, "print(42)")
        write(self.rt.root / "task-code.json", {"enabled": True, "image": "sha256:" + "a" * 64})
        with self.assertRaisesRegex(ToolError, "not approved for the isolated runtime"):
            server.run_script(handle, 0, runtime_name="python", entrypoint="main.py", entrypoint_root="skill")
        self.assertEqual(list(self.rt.jobs.iterdir()), [])
        self.assertEqual(self.rt.workspaces.state(handle)["revision"], 0)
        self.assertEqual(self.rt.workspaces.state(handle)["files"], {})

    def test_instruction_resolution_keeps_pending_preparation_and_symlinks_bound(self):
        server._runtime = self.rt
        write(self.rt.root / "packages.json", {})
        write(self.rt.root / "task-code.json", {"enabled": True, "image": "sha256:" + "a" * 64})
        from skill_workspace.prepare_runtime import identity
        preparation = self.rt.root / "imports" / identity("fixture", "a" * 64, "sha256:" + "a" * 64)
        write(preparation / "state.json", {"status": "preparing", "image": "sha256:" + "a" * 64})
        workspace = server.create_workspace("fixture", "a" * 64).structured_content
        self.assertEqual(workspace["package_name"], "fixture")
        with self.assertRaisesRegex(ToolError, "not ready"):
            server.execute_python(workspace["workspace_id"], 0, "print(42)")
        (preparation / "state.json").unlink()
        package = self.root / "packages" / "fixture" / ("a" * 64)
        instruction = package / "SKILL.md"
        instruction.unlink()
        target = self.root / "outside.md"
        target.write_text("external instructions")
        instruction.symlink_to(target)
        workspace = server.create_workspace("fixture", "a" * 64).structured_content
        self.assertEqual(workspace["package_name"], "fixture")
        with self.assertRaisesRegex(ToolError, "not approved"):
            server.execute_python(workspace["workspace_id"], 0, "print(42)")
        self.assertEqual(list(self.rt.jobs.iterdir()), [])

    def test_additional_package_entries_and_runtime_bindings_keep_exact_approval(self):
        server._runtime = self.rt
        write(self.rt.root / "packages.json", {})
        write(self.rt.root / "task-code.json", {"enabled": True, "image": "sha256:" + "a" * 64})
        package = self.root / "packages" / "fixture" / ("a" * 64)
        for name in ["scripts", "resources", "runtime.json", "requirements.txt"]:
            item = package / name
            if name in {"scripts", "resources"}: item.mkdir()
            else: item.write_text("{}")
            workspace = server.create_workspace("fixture", "a" * 64).structured_content
            self.assertEqual(workspace["package_name"], "fixture")
            with self.assertRaisesRegex(ToolError, "not approved for the isolated runtime"):
                server.execute_python(workspace["workspace_id"], 0, "print(42)")
            if item.is_dir(): item.rmdir()
            else: item.unlink()
        write(self.rt.root / "bindings.json", {"fixture:" + "a" * 64: {"image": "sha256:" + "b" * 64, "runtime": "python"}})
        workspace = server.create_workspace("fixture", "a" * 64).structured_content
        with self.assertRaisesRegex(ToolError, "image changed"):
            server.execute_python(workspace["workspace_id"], 0, "print(42)")
        self.assertEqual(list(self.rt.jobs.iterdir()), [])

    def test_mcp_preserves_code_strings_exactly_and_rejects_wrong_interpreter(self):
        server._runtime = self.rt
        workspace = server.create_workspace("fixture", "a" * 64).structured_content
        code = 'console.log("first\\nlast");\nconst x = `actual\nnewline`;\n'
        changed = asyncio.run(server.mcp.call_tool("edit_workspace", {
            "workspace_id": workspace["workspace_id"], "expected_revision": 0,
            "files": [{"path": "build.js", "content": code}]})).structured_content
        self.assertEqual(self.rt.workspaces.inspect(workspace["workspace_id"], "build.js")["content"], code)
        with self.assertRaisesRegex(ToolError, "requires runtime_name=node"):
            asyncio.run(server.mcp.call_tool("run_script", {
                "workspace_id": workspace["workspace_id"], "expected_revision": changed["revision"],
                "runtime_name": "python", "entrypoint": "build.js"}))
        result = asyncio.run(server.mcp.call_tool("run_script", {
            "workspace_id": workspace["workspace_id"], "expected_revision": changed["revision"],
            "runtime_name": "node", "entrypoint": "build.js"})).structured_content
        self.assertEqual(result["status"], "queued")
        self.assertEqual(self.rt.workspaces.inspect(workspace["workspace_id"], "build.js")["content"], code)

    def test_inline_default_entrypoints_follow_explicit_runtime_through_mcp(self):
        self.standalone()
        inventory = read(self.rt.root / "inventory.json")
        inventory["commands"].update(tsx=True, tsc=True)
        write(self.rt.root / "inventory.json", inventory)
        for runtime_name, path, code in [("shell", "script.sh", "printf ok"),
                                         ("python", "script.py", 'import json\nprint(json.dumps({"ok": True}))\n'),
                                         ("node", "script.js", "console.log('ok')"),
                                         ("typescript", "script.ts", "const ok: boolean=true; console.log(ok)")]:
            with self.subTest(runtime=runtime_name):
                handle = server.create_workspace().structured_content["workspace_id"]
                arguments = {"workspace_id": handle, "expected_revision": 0, "runtime_name": runtime_name, "code": code}
                result = asyncio.run(server.mcp.call_tool("run_script", arguments)).structured_content
                self.assertEqual(result["status"], "queued")
                request = read(self.rt.jobs / result["job_id"] / "request.json")
                self.assertEqual((request["runtime"], request["entrypoint"]), (runtime_name, path))
                self.assertEqual(request["files"], [{"path": path, "content": code}])
                self.assertEqual(asyncio.run(server.mcp.call_tool("run_script", arguments)).structured_content["job_id"], result["job_id"])
                self.assertFalse((self.rt.workspaces.directory(handle) / "drafts").exists())

    def test_native_json_files_preserve_values_and_source_bytes_through_mcp(self):
        handle = self.standalone()
        value = {"prompt": '中文\n"quoted"\r\nnext\ttab', "literal": r'C:\new\nfile',
                 "segments": [{"duration": 30, "enabled": True, "optional": None}], "empty": []}
        source = 'const text = `first\nlast`; // do not interpret \\n\n'
        raw = '{"text":"raw\nnewline"}'
        response = asyncio.run(server.mcp.call_tool("edit_workspace", {
            "workspace_id": handle, "expected_revision": 0,
            "json_files": [{"path": "plan.json", "value": value}, {"path": "array.json", "value": [value]}],
            "files": [{"path": "build.js", "content": source}, {"path": "raw.json", "content": raw}]})).structured_content
        self.assertEqual(response["revision"], 1)
        for path, expected in (("plan.json", value), ("array.json", [value])):
            content = self.rt.workspaces.inspect(handle, path)["content"]
            self.assertEqual(json.loads(content), expected)
        self.assertEqual(self.rt.workspaces.inspect(handle, "build.js")["content"], source)
        self.assertEqual(self.rt.workspaces.inspect(handle, "raw.json")["content"], raw)
        self.assertEqual(list(self.rt.jobs.iterdir()), [])

    def test_production_mixed_write_replay_succeeds_without_repair_or_execution(self):
        fixture = json.loads((Path(__file__).parent / "fixtures" / "mixed_file_write_failure.json").read_text())
        handle = self.standalone()
        arguments = {"workspace_id": handle, "expected_revision": 0, "files": fixture["files"]}
        response = asyncio.run(server.mcp.call_tool("edit_workspace", arguments)).structured_content
        self.assertEqual(response["revision"], 1)
        for item in fixture["files"]:
            saved = self.rt.workspaces.inspect(handle, item["path"])["content"]
            if "content" in item:
                self.assertEqual(saved, item["content"])
            else:
                self.assertEqual(json.loads(saved), item["value"])
        self.assertEqual(list(self.rt.jobs.iterdir()), [], "writing files must not execute code")

    def test_mixed_native_json_arrays_and_legacy_batches_preserve_exact_sources(self):
        handle = self.standalone()
        source = '{"invalid JSON preserved literally":"raw\nnewline"}'
        value = [{"text": '中文\n"quoted"', "path": r'C:\new\nfile', "optional": None}]
        response = asyncio.run(server.mcp.call_tool("edit_workspace", {
            "workspace_id": handle, "expected_revision": 0,
            "files": [{"path": "source.json", "content": source}, {"path": "array.json", "value": value}],
            "json_files": [{"path": "legacy.json", "value": {"keep": True}}]})).structured_content
        self.assertEqual(response["revision"], 1)
        self.assertEqual(self.rt.workspaces.inspect(handle, "source.json")["content"], source)
        self.assertEqual(json.loads(self.rt.workspaces.inspect(handle, "array.json")["content"]), value)
        self.assertEqual(json.loads(self.rt.workspaces.inspect(handle, "legacy.json")["content"]), {"keep": True})

    def test_native_json_rejects_unsafe_ambiguous_or_oversized_edits_atomically(self):
        handle = self.standalone()
        original = server.edit_workspace(handle, 0, files=[server.SourceFile(path="keep.txt", content="exact")]).structured_content
        cases = [
            {"files": [{"path": "ambiguous.json", "content": "keep", "value": {}}]},
            {"files": [{"path": "missing.json"}]},
            {"files": [{"path": "extra.json", "value": {}, "unknown": True}]},
            {"files": [{"path": "string.json", "value": '{"nested":"string"}'}]},
            {"files": [{"path": "../outside.json", "value": {}}]},
            {"files": [{"path": "/tmp/outside.json", "value": {}}]},
            {"files": [{"path": "keep.txt", "value": {}}], "json_files": [{"path": "keep.txt", "value": []}]},
            {"files": [{"path": "keep.txt", "value": {}}, {"path": "keep.txt", "content": "wrong"}]},
            {"files": [{"path": "huge.json", "value": {"text": "x" * (512 * 1024)}}]},
            {"files": [{"path": "finite.json", "value": {"value": float("inf")}}]},
            {"files": [{"path": "ok.json", "value": {}}], "draft_id": "a" * 64},
            {"json_files": [{"path": "../outside.json", "value": {}}]},
            {"json_files": [{"path": "/tmp/outside.json", "value": {}}]},
            {"json_files": [{"path": "keep.txt", "value": {}}], "files": [{"path": "keep.txt", "content": "wrong"}]},
            {"json_files": [{"path": "string.json", "value": '{"nested":"string"}'}]},
            {"json_files": [{"path": "huge.json", "value": {"text": "x" * (512 * 1024)}}]},
            {"json_files": [{"path": "finite.json", "value": {"value": float("nan")}}]},
            {"json_files": [{"path": "ok.json", "value": {}}], "draft_id": "a" * 64},
            {"json_files": [{"path": f"{i}.json", "value": {}} for i in range(24)], "files": [{"path": "extra", "content": ""}]},
        ]
        for arguments in cases:
            with self.subTest(keys=list(arguments)):
                with self.assertRaises(ToolError):
                    asyncio.run(server.mcp.call_tool("edit_workspace", {"workspace_id": handle,
                        "expected_revision": 1, **arguments}))
                current = self.rt.workspaces.inspect(handle)
                self.assertEqual(current["revision"], original["revision"])
                self.assertEqual(current["files"], original["files"])
        with self.assertRaisesRegex(ToolError, "revision conflict"):
            server.edit_workspace(handle, 0, json_files=[server.JsonFile(path="data.json", value={})])
        with self.assertRaisesRegex(ToolError, "revision conflict"):
            server.edit_workspace(handle, 0, files=[server.JsonFile(path="data.json", value={})])
        server.run_script(handle, 1, code="echo unchanged")
        with self.assertRaisesRegex(ToolError, "busy"):
            server.edit_workspace(handle, 1, json_files=[server.JsonFile(path="data.json", value={})])
        with self.assertRaisesRegex(ToolError, "busy"):
            server.edit_workspace(handle, 1, files=[server.JsonFile(path="data.json", value={})])

    def test_interpreter_entrypoint_rejected_without_job_or_workspace_mutation(self):
        handle = self.standalone()
        for entrypoint in ("/bin/sh", "/usr/bin/python", "../outside.py", "a//script.sh"):
            with self.subTest(entrypoint=entrypoint), self.assertRaisesRegex(ToolError, "relative script file.*runtime_name.*No execution was submitted"):
                asyncio.run(server.mcp.call_tool("run_script", {"workspace_id": handle,
                    "expected_revision": 0, "runtime_name": "shell", "entrypoint": entrypoint,
                    "arguments": ["-c", "import json; print('never execute')"]}))
        self.assertEqual(self.rt.workspaces.inspect(handle)["files"], [])
        self.assertEqual(self.rt.workspaces.state(handle)["revision"], 0)
        self.assertEqual(list(self.rt.jobs.iterdir()), [])

    def test_saved_scripts_are_not_guessed_and_explicit_paths_are_not_changed(self):
        handle = self.standalone()
        for runtime_name in ("python", "node", "typescript"):
            with self.assertRaisesRegex(ToolError, "exact entrypoint"):
                asyncio.run(server.mcp.call_tool("run_script", {"workspace_id": handle, "expected_revision": 0,
                    "runtime_name": runtime_name}))
        with self.assertRaisesRegex(ToolError, "exact entrypoint"):
            asyncio.run(server.mcp.call_tool("run_script", {"workspace_id": handle, "expected_revision": 0,
                "entrypoint_root": "skill"}))
        source = "print('preserved Python')\n"
        rejected = asyncio.run(server.mcp.call_tool("run_script", {"workspace_id": handle, "expected_revision": 0,
            "runtime_name": "python", "entrypoint": "explicit.sh", "code": source})).structured_content
        self.assertEqual(rejected["status"], "rejected")
        self.assertFalse(rejected["code_executed"])
        self.assertEqual(rejected["draft"]["files"][0]["path"], "explicit.sh")
        self.assertEqual(list(self.rt.jobs.iterdir()), [])
        explicit = asyncio.run(server.mcp.call_tool("run_script", {"workspace_id": handle, "expected_revision": 0,
            "runtime_name": "python", "entrypoint": "check.py", "code": source})).structured_content
        request = read(self.rt.jobs / explicit["job_id"] / "request.json")
        self.assertEqual(request["files"], [{"path": "check.py", "content": source}])

    def test_inline_checks_cannot_overwrite_authored_generator_but_scratch_still_reusable(self):
        server._runtime = self.rt
        write(self.rt.root / "task-code.json", {"enabled": True, "image": "sha256:" + "a" * 64})
        workspace = server.create_workspace().structured_content
        handle = workspace["workspace_id"]
        code = "console.log('original generator')"
        saved = server.edit_workspace(handle, 0, files=[server.SourceFile(path="build.js", content=code)]).structured_content
        with self.assertRaisesRegex(ToolError, "would overwrite authored file build.js"):
            asyncio.run(server.mcp.call_tool("run_script", {"workspace_id": handle, "expected_revision": saved["revision"],
                "runtime_name": "node", "entrypoint": "build.js", "code": "console.log('syntax check')"}))
        self.assertEqual(list(self.rt.jobs.iterdir()), [])
        self.assertEqual(self.rt.workspaces.inspect(handle, "build.js")["content"], code)
        self.assertEqual(self.rt.workspaces.state(handle)["revision"], saved["revision"])
        # Ordinary inline commands retain their prior overwrite/reuse semantics.
        first = server.run_script(handle, saved["revision"], code="echo one").structured_content
        write(self.rt.jobs / first["job_id"] / "state.json", {"status": "failed"})
        second = server.run_script(handle, saved["revision"], code="echo two").structured_content
        self.assertNotEqual(first["job_id"], second["job_id"])
        self.assertEqual(self.rt.workspaces.inspect(handle, "script.sh")["content"], "echo two")
        self.assertEqual(self.rt.workspaces.inspect(handle, "build.js")["content"], code)

    def publish(self):
        first = self.rt.submit(self.request)
        job = self.rt.jobs / first["job_id"]
        (job / "published").mkdir()
        data = b"%PDF-test-content"
        (job / "published/report.pdf").write_bytes(data)
        write(job / "result.json", {"files": [{"filename": "report.pdf", "content_type": "application/pdf",
            "size_bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}]})
        write(job / "state.json", {"status": "completed", "progress": 100})
        write(job / "execution.json", {"exit_code": 0, "log": "verified output"})
        return self.rt.status(first["job_id"])

    def test_edit_receipt_confirms_changed_text_files_without_a_read_cycle(self):
        server._runtime = self.rt
        write(self.rt.root / "task-code.json", {"enabled": True, "image": "sha256:" + "a" * 64})
        handle = server.create_workspace().structured_content["workspace_id"]
        body = "<html>\n" + "<p>line</p>\n" * 50 + "</html>\n"
        saved = server.edit_workspace(handle, 0, files=[server.SourceFile(path="index.html", content=body),
            server.SourceFile(path="a.bin", content="ok")]).structured_content
        receipts = {item["path"]: item for item in saved["changes"]}
        self.assertEqual(receipts["index.html"]["total_chars"], len(body))
        self.assertEqual(receipts["index.html"]["line_count"], 52)
        self.assertTrue(receipts["index.html"]["tail"].endswith("</html>\n"))
        self.assertLessEqual(len(receipts["index.html"]["tail"]), 400)
        self.assertNotIn("<html>", receipts["index.html"]["tail"])
        self.assertNotIn("element_ids", receipts["index.html"], "no id attributes, no outline")
        self.assertEqual(receipts["index.html"]["sha256"], hashlib.sha256(body.encode()).hexdigest())
        wire = server.edited_workspace(dict(saved))
        self.assertIn("/changes", wire.meta["exoharness"]["context_preserve"])
        appended = server.edit_workspace(handle, saved["revision"], appends=[server.SourceFile(path="index.html", content="<!-- end -->\n")]).structured_content
        self.assertEqual([c["path"] for c in appended["changes"]], ["index.html"])
        self.assertTrue(appended["changes"][0]["tail"].endswith("<!-- end -->\n"))
        listing = self.rt.workspaces.inspect(handle)
        self.assertNotIn("changes", listing)

    def test_inspection_evidence_is_compact_and_screenshots_stay_out_of_the_checkpoint(self):
        first = self.rt.submit(self.request)
        job = self.rt.jobs / first["job_id"]
        # A completed browser inspection job, as recorded by inspect_webpage.
        write(job / "request.json", {**read(job / "request.json"), "operation": "inspect_webpage"})
        (job / "published").mkdir()
        (job / "published/inspection-0.png").write_bytes(b"png")
        headings = [{"text": f"Section {i} heading", "visible": i != 3} for i in range(25)]
        write(job / "result.json", {"files": [{"filename": "inspection-0.png", "content_type": "image/png",
            "size_bytes": 3, "sha256": hashlib.sha256(b"png").hexdigest()}],
            "inspection": {"source_sha256": "s", "limits": "offline", "scenarios": [{
                "title": "Home", "headings": headings, "console_errors": ["TypeError: boom"], "broken_images": [],
                "missing_anchors": [], "overflowing_elements": [], "horizontal_overflow": False, "page_width": 1440}]}})
        write(job / "state.json", {"status": "completed", "progress": 100})
        completed = self.rt.status(first["job_id"])
        scenario = completed["evidence"]["inspection"]["scenarios"][0]
        self.assertEqual(scenario["heading_count"], 25)
        self.assertNotIn("hidden_headings", scenario)
        self.assertEqual(scenario["checks"]["status"], "failed")
        self.assertIn("console_errors", scenario["checks"]["errors"])
        self.assertNotIn("headings", scenario)
        self.assertNotIn("broken_images", scenario)
        self.assertEqual(scenario["console_errors"], ["TypeError: boom"])
        self.assertEqual(scenario["page_width"], 1440)
        paths = completed["_meta"]["exoharness"]["context_preserve"]
        self.assertNotIn("/artifacts", paths)
        self.assertIn("/evidence", paths)
        self.assertIn("/job_id", paths)
        self.assertEqual(completed["artifacts"][0]["filename"], "inspection-0.png")
        self.assertLess(len(json.dumps(completed["evidence"])), 1000)
        self.assertEqual(read(job / "result.json")["inspection"]["scenarios"][0]["headings"], headings)

    def test_real_bilingual_report_passes_without_turning_hidden_translations_into_repairs(self):
        fixture = json.loads((Path(__file__).parent / "fixtures/bilingual_inspection_pass.json").read_text())
        raw = fixture["inspection"]
        original = copy.deepcopy(raw)
        compact = server.compact_inspection(raw)
        self.assertEqual(compact["checks"]["status"], "passed")
        self.assertEqual(len(compact["scenarios"]), 3)
        for scenario in compact["scenarios"]:
            self.assertEqual(scenario["checks"], {"status": "passed", "errors": []})
            self.assertNotIn("hidden_headings", scenario)
            self.assertNotIn("hidden_text_blocks", scenario)
        self.assertEqual(compact["scenarios"][0]["hidden_text_sample_count"], 20)
        self.assertLess(len(json.dumps(compact)), len(json.dumps(raw)) / 3)
        self.assertEqual(raw, original, "exact report and source receipts must stay intact")

    def test_summary_never_claims_pass_for_missing_observations_or_trusts_supplied_verdict(self):
        compact = server.compact_inspection({"scenarios": [{"visible_text_chars": 200,
            "hidden_text_blocks": ["h2"], "checks": {"status": "passed", "errors": []}}]})
        self.assertEqual(compact["checks"]["status"], "incomplete")
        self.assertEqual(compact["scenarios"][0]["checks"]["errors"], [])
        self.assertIn("unexpected_hidden_text_blocks", compact["scenarios"][0]["checks"]["missing_fields"])
        self.assertEqual(server.compact_inspection({"scenarios": []})["checks"]["status"], "incomplete")

    def test_summary_uses_exact_publication_error_predicate_for_all_defect_types(self):
        fixture = json.loads((Path(__file__).parent / "fixtures/bilingual_inspection_pass.json").read_text())
        for field, bad in [("console_errors", ["TypeError"]), ("broken_images", ["missing.png"]),
                           ("missing_anchors", ["#missing"]), ("horizontal_overflow", True),
                           ("unexpected_hidden_text_blocks", [".reveal"]),
                           ("unreadable_headings", [{"selector": "h1", "ratio": 1.01}]),
                           ("visible_text_chars", 0)]:
            with self.subTest(field=field):
                raw = copy.deepcopy(fixture["inspection"])
                raw["scenarios"][1][field] = bad
                compact = server.compact_inspection(raw)
                self.assertEqual(compact["checks"]["status"], "failed")
                self.assertEqual(compact["scenarios"][1]["checks"]["errors"], [field])
                self.assertEqual(compact["scenarios"][1][field], bad)
                self.assertEqual(compact["scenarios"][0]["checks"]["status"], "passed")

    def test_script_checkpoint_does_not_end_outer_workflow_and_returns_log(self):
        completed = self.publish()
        self.assertIn("context_checkpoint", completed["_meta"]["exoharness"])
        self.assertNotIn("workflow_continuation", completed["_meta"]["exoharness"])
        self.assertEqual(completed["execution"]["log_tail"], "verified output")

    def test_terminal_checkpoint_paths_exist_and_keep_failure_diagnostics(self):
        first = self.rt.submit(self.request)
        job = self.rt.jobs / first["job_id"]
        for has_execution in [False, True]:
            with self.subTest(has_execution=has_execution):
                write(job / "state.json", {"status": "failed", "error": "declared output missing", "message": "repair the output"})
                if has_execution:
                    write(job / "execution.json", {"exit_code": 1, "log": "computed value\noutput missing"})
                value = self.rt.status(first["job_id"], True)
                wire = server.result(value)
                self.assertTrue(wire.is_error)
                paths = value["_meta"]["exoharness"]["context_preserve"]
                for path in paths:
                    self.assertIn(path[1:], wire.structured_content)
                self.assertIn("/error", paths)
                self.assertNotIn("/artifacts", paths)
                self.assertNotIn("/evidence", paths)
                self.assertEqual("/execution" in paths, has_execution)
                self.assertEqual(value["_meta"]["exoharness"]["background_task"]["status"], "failed")
        completed = self.publish()
        for path in completed["_meta"]["exoharness"]["context_preserve"]:
            self.assertIn(path[1:], completed)

    def test_stdout_only_python_and_shell_do_not_declare_downloads(self):
        server._runtime = self.rt
        write(self.rt.root / "task-code.json", {"enabled": True, "image": "sha256:" + "a" * 64})
        for tool, code in [(server.execute_python, "print(123)"), (server.run_script, "printf 123")]:
            workspace = server.create_workspace().structured_content
            submitted = tool(workspace["workspace_id"], 0, code=code).structured_content
            request = read(self.rt.jobs / submitted["job_id"] / "request.json")
            self.assertEqual(request["outputs"], [])
            self.assertIn("execution.log_tail", tool.__doc__)

    def test_reuse_signed_artifact_without_network_or_regeneration(self):
        completed = self.publish()
        url = completed["artifacts"][0]["download_url"]
        request = copy.deepcopy(self.request)
        request["inputs"] = [{"path": "input/report.pdf", "url": url}]
        request["outputs"] = []
        followup = self.rt.submit(request)
        job = self.rt.jobs / followup["job_id"]
        manifest = read(job / "input/files.json")
        self.assertEqual(manifest[0]["path"], "input/report.pdf")
        self.assertEqual((job / "input/artifact-0").read_bytes(), b"%PDF-test-content")
        self.assertEqual(self.rt.submit(request)["job_id"], followup["job_id"])

    def test_reuse_rejects_forgery_traversal_cross_service_and_tampering(self):
        completed = self.publish()
        url = completed["artifacts"][0]["download_url"]
        bad_urls = [url.replace("testserver", "otherhost"), url.split("?", 1)[0],
                    url + "&signature=extra", url.replace("signature=", "signature=wrong"),
                    "file:///etc/passwd", url.replace("report.pdf", "%2Fetc%2Fpasswd")]
        for bad in bad_urls:
            with self.assertRaises(ValueError):
                self.rt.input_bytes(bad)
        request = copy.deepcopy(self.request)
        request["inputs"] = [{"path": "../escape.pdf", "url": url}]
        with self.assertRaises(ValueError): self.rt.submit(request)
        request["inputs"][0]["path"] = "main.py"
        with self.assertRaises(ValueError): self.rt.submit(request)
        (self.rt.jobs / completed["job_id"] / "published/report.pdf").write_bytes(b"altered")
        with self.assertRaisesRegex(ValueError, "integrity"):
            self.rt.input_bytes(url)

    def test_permanent_signed_download_and_mcp_authentication(self):
        first = self.rt.submit(self.request)
        job = self.rt.jobs / first["job_id"]
        published = job / "published"
        published.mkdir()
        data = b"%PDF-test-content"
        (published / "report.pdf").write_bytes(data)
        write(job / "result.json", {"files": [{"filename": "report.pdf", "content_type": "application/pdf",
            "size_bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}]})
        write(job / "state.json", {"status": "completed", "progress": 100})
        server._runtime = self.rt
        app = server.mcp.streamable_http_app(streamable_http_path="/mcp")
        app.add_middleware(server.BearerAuth)
        with TestClient(app) as client:
            url = self.rt.status(first["job_id"])["artifacts"][0]["download_url"]
            self.assertNotIn("expires", url)
            response = client.get(url)
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.content, data)
            self.assertTrue(response.headers["content-disposition"].startswith("attachment"))
            self.assertEqual(client.get(url, headers={"Range": "bytes=0-3"}).status_code, 206)
            self.assertEqual(client.get(url.split("?", 1)[0]).status_code, 404)
            self.assertEqual(client.post("/mcp", json={}).status_code, 401)


if __name__ == "__main__":
    unittest.main()
