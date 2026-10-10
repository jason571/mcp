import asyncio
import hashlib
import json
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

from starlette.testclient import TestClient
from skill_workspace import server
from skill_workspace.projects import Projects, ProjectFile, ProjectPatch
from skill_workspace.common import read, write


class ProjectTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.rt = server.Runtime(self.root / "data", self.root / "packages", "fixture-key-" * 6, "http://testserver")
        server._runtime = self.rt
        image = "sha256:" + "a" * 64
        write(self.rt.root / "worker.json", {"image": image})
        write(self.rt.root / "inventory.json", {"image": image, "engineering_image": image, "commands": {"python": True, "bash": True}})
        write(self.rt.root / "engineering-runtime.json", {"enabled": True, "image": image})
        write(self.rt.root / "task-code.json", {"enabled": True, "image": image})
        self.ctx = SimpleNamespace(request_context=SimpleNamespace(meta={"exoharness": {"subject": "a" * 64, "software_engineering": True}}))

    def tearDown(self):
        server._runtime = None
        self.temp.cleanup()

    def call(self, tool_name, **args):
        return server.mcp._tool_manager.get_tool(tool_name).fn(ctx=self.ctx, **args)

    def test_open_is_idempotent_scoped_and_never_exposes_internal_handle(self):
        first = self.call("open_project", name="计数器").structured_content
        self.assertEqual(first, self.call("open_project", name="计数器").structured_content)
        self.assertNotIn("ws_", json.dumps(first))
        project = first["project_id"]
        self.ctx.request_context.meta["exoharness"]["subject"] = "b" * 64
        self.assertEqual(self.call("list_projects").structured_content["projects"], [])
        with self.assertRaisesRegex(Exception, "not found"):
            self.call("read_project", project_id=project)
        self.assertNotEqual(self.call("open_project", name="计数器").structured_content["project_id"], project)

    def test_atomic_multifile_edit_diff_restore_keeps_exact_data_and_no_host_access(self):
        project = self.call("open_project", name="demo").structured_content["project_id"]
        self.call("edit_project", project_id=project, expected_revision=0, files=[
            ProjectFile(path="calc.py", content="def add(a,b): return a-b\n"),
            ProjectFile(path="test_calc.py", content="from calc import add\nassert add(2,3)==5\n")])
        cp = self.call("checkpoint_project", project_id=project, expected_revision=1, label="before").structured_content["checkpoint"]
        file = self.call("read_project", project_id=project, path="calc.py").structured_content
        patch = ProjectPatch(path="calc.py", sha256=file["sha256"], before="a-b", after="a+b")
        with self.assertRaisesRegex(Exception, "SHA-256"):
            self.call("edit_project", project_id=project, expected_revision=1, patches=[patch, ProjectPatch(path="test_calc.py", sha256="0"*64,before="5",after="7")])
        self.assertEqual(self.call("read_project",project_id=project,path="calc.py").structured_content["sha256"],file["sha256"])
        result = self.call("edit_project", project_id=project, expected_revision=1, patches=[patch])
        self.assertNotIn("ws_", json.dumps(result.model_dump()))
        for pointer in result.meta["exoharness"]["context_preserve"]:
            self.assertIn(pointer[1:], result.structured_content)
        diff = self.call("diff_project",project_id=project,checkpoint=cp,path="calc.py").structured_content
        self.assertEqual(len(diff["changes"]),1)
        self.assertIn("+def add(a,b): return a+b",diff["diff"])
        for path in ("/etc/passwd", "../escape", "a/../../b"):
            with self.assertRaises(Exception): self.call("edit_project",project_id=project,expected_revision=2,files=[ProjectFile(path=path,content="bad")])
        with self.assertRaises(Exception): self.call("restore_project",project_id=project,expected_revision=2,checkpoint=cp,confirm=False)
        self.call("restore_project",project_id=project,expected_revision=2,checkpoint=cp,confirm=True)
        restored=self.call("read_project",project_id=project,path="calc.py").structured_content
        self.assertEqual(restored["sha256"],file["sha256"])
        self.assertEqual(restored["revision"],3)

    def test_instructions_are_bounded_nested_and_scoped_not_permission(self):
        project = self.call("open_project",name="instructions").structured_content["project_id"]
        files={"AGENTS.md":"root", "src/AGENTS.md":"parent", "src/lib/AGENTS.md":"ignored override", "src/lib/AGENTS.override.md":"nested", "other/AGENTS.md":"unrelated"}
        self.call("edit_project",project_id=project,expected_revision=0,files=[ProjectFile(path=p,content=t) for p,t in files.items()])
        value=self.call("read_project_instructions",project_id=project,path="src/lib/add.py").structured_content
        self.assertEqual([d["content"] for d in value["instructions"]],["root","parent","nested"])
        self.assertIn("cannot grant",value["scope"])

    def test_general_tools_cannot_bypass_project_ownership_or_reveal_handles(self):
        project = self.call("open_project",name="private").structured_content["project_id"]
        state=Projects(self.rt).state("a"*64,project)
        handle=state["workspace_id"]
        for fn in (lambda: server.read_workspace(handle), lambda: server.edit_workspace(handle,0),
                   lambda: server.execute_python(handle,0,"print('forbidden')"), lambda:server.publish_workspace(handle,0,["index.html"])):
            with self.assertRaisesRegex(Exception,"private projects"): fn()
        self.call("edit_project",project_id=project,expected_revision=0,files=[ProjectFile(path="index.html",content="<h1>Hello</h1>")])
        publication=self.call("publish_project",project_id=project,expected_revision=1,outputs=["index.html"])
        self.assertNotIn(handle,json.dumps(publication.model_dump()))
        with self.assertRaisesRegex(Exception,"private projects"):
            server.get_script_job(publication.structured_content["job_id"])

    def test_real_mcp_wire_requires_host_subject_and_keeps_feature_out_of_schema(self):
        tools=asyncio.run(server.mcp.list_tools())
        project_tools=[t for t in tools if (t.meta or {}).get("exoharness",{}).get("required_feature")=="software_engineering"]
        self.assertGreaterEqual(len(project_tools),12)
        for tool in project_tools:
            self.assertNotIn("subject",tool.input_schema.get("properties",{}))
            self.assertNotIn("ctx",tool.input_schema.get("properties",{}))
        app=server.mcp.streamable_http_app(streamable_http_path="/mcp",stateless_http=True,json_response=True)
        app.add_middleware(server.BearerAuth)
        with TestClient(app, base_url="http://127.0.0.1:8794") as client:
            headers={"Authorization":"Bearer "+self.rt.token,"Accept":"application/json, text/event-stream"}
            def call(meta=None):
                params={"name":"open_project","arguments":{"name":"wire-test"}}
                if meta is not None: params["_meta"]=meta
                response=client.post("/mcp",headers=headers,json={"jsonrpc":"2.0","id":1,"method":"tools/call","params":params})
                self.assertEqual(response.status_code,200,response.text)
                return response.json()["result"]
            self.assertTrue(call()["isError"])
            value=call(self.ctx.request_context.meta)
            self.assertFalse(value.get("isError",False),value)
            self.assertTrue(value["structuredContent"]["project_id"].startswith("project_"))

    def test_project_job_metadata_supports_wait_cancel_and_no_encoded_payload(self):
        project=self.call("open_project",name="job").structured_content["project_id"]
        response=self.call("run_project",project_id=project,expected_revision=0,code="printf ok",entrypoint="check.sh")
        meta=response.meta["exoharness"]["background_task"]
        self.assertEqual(meta["poll"]["tool"],"get_project_job")
        self.assertEqual(meta["cancel"]["tool"],"cancel_project_job")
        self.assertEqual(meta["poll"]["arguments"]["project_id"],project)
        job=response.structured_content["job_id"]
        with self.assertRaises(Exception): self.call("edit_project",project_id=project,expected_revision=0)
        self.call("cancel_project_job",project_id=project,job_id=job)
        self.assertTrue((self.rt.jobs/job/"cancel").exists())
        with self.assertRaisesRegex(Exception,"private projects"): server.cancel_script_job(job)
        other=self.call("open_project",name="other").structured_content["project_id"]
        with self.assertRaisesRegex(Exception,"not found"): self.call("get_project_job",project_id=other,job_id=job)

    def test_invalid_project_code_request_preserves_draft_in_its_own_authorized_scope(self):
        project=self.call("open_project",name="draft").structured_content["project_id"]
        result=self.call("run_project",project_id=project,expected_revision=0,runtime_name="shell",entrypoint="main.py",code="print(42)").structured_content
        self.assertEqual(result["status"],"rejected")
        self.assertEqual(result["recovery"]["tool"],"edit_project")
        self.assertNotIn("ws_",json.dumps(result))
        self.call("edit_project",**result["recovery"]["arguments"])
        self.assertEqual(self.call("read_project",project_id=project,path="main.py").structured_content["content"],"print(42)")

    def test_project_policy_is_independent_from_general_image_and_cannot_be_changed_by_code(self):
        from skill_workspace.project_runtime import workspace_image
        project=self.call("open_project",name="runtime-pinned").structured_content["project_id"]
        handle=Projects(self.rt).state("a"*64,project)["workspace_id"]
        self.assertEqual(workspace_image(self.rt.root,handle),"sha256:"+"a"*64)
        write(self.rt.root/"engineering-runtime.json",{"enabled":True,"image":"sha256:"+"b"*64})
        with self.assertRaisesRegex(ValueError,"image changed"): workspace_image(self.rt.root,handle)
        normal=server.create_workspace().structured_content["workspace_id"]
        self.assertIsNone(workspace_image(self.rt.root,normal))
        self.assertEqual(self.rt.workspaces.state(normal)["runtime_image"],"sha256:"+"a"*64)
