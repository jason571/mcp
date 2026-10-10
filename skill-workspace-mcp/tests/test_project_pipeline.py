"""Real MCP -> private project -> broker -> OCI -> patch/review/publish.

The controller never receives a Docker socket; authored code only runs in the
existing network-disabled worker. Fixtures use no production account/model.
"""
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
import unittest
from urllib.request import Request, urlopen

from skill_workspace.broker import docker, run_job
from skill_workspace.common import read, write


@unittest.skipUnless(os.getenv("SKILL_WORKSPACE_INTEGRATION") == "1", "requires local Docker and built immutable image")
class ProjectPipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="exoharness-engineering-")
        self.root = Path(self.temp.name)
        self.root.chmod(0o755)
        self.data = self.root / "data"
        self.packages = self.root / "packages"
        self.packages.mkdir()
        self.image = os.environ["WORKSPACE_IMAGE_ID"]
        self.engineering_image = docker("image","inspect","skill-workspace-engineering:0.1.0","--format","{{.Id}}").stdout.decode().strip()
        write(self.data / "worker.json", {"image":self.image})
        write(self.data / "inventory.json", {"image":self.image,"engineering_image":self.engineering_image,"commands":{"python":True,"bash":True,"node":True}})
        write(self.data / "engineering-runtime.json", {"image":self.engineering_image,"enabled":True})
        write(self.data / "task-code.json", {"enabled":True,"image":self.image})
        source=Path(__file__).resolve().parents[1] / "src"
        code="from pathlib import Path; import uvicorn; from skill_workspace import server; server._runtime=server.Runtime(Path('/fixture/data'),Path('/fixture/packages'),'project-test-token-'*4,'http://localhost'); app=server.mcp.streamable_http_app(streamable_http_path='/mcp',stateless_http=True,json_response=True); app.add_middleware(server.BearerAuth); uvicorn.run(app,host='0.0.0.0',port=8794,log_level='error')"
        self.container=docker("run","--rm","-d","--network","bridge","-p","127.0.0.1::8794",
            "--user",f"{os.getuid()}:{os.getgid()}",
            "--mount",f"type=bind,src={source},dst=/src,readonly","--mount",f"type=bind,src={self.root},dst=/fixture",
            "-e","PYTHONPATH=/src","-e","PYTHONDONTWRITEBYTECODE=1","--entrypoint","python",self.image,"-B","-c",code).stdout.decode().strip()
        address=docker("port",self.container,"8794/tcp").stdout.decode().strip()
        self.url=f"http://{address}/mcp"
        self.jobs=[]
        for _ in range(60):
            try:
                self.call("list_projects")
                break
            except OSError:
                time.sleep(.2)
        else:
            self.fail(docker("logs",self.container).stdout.decode())

    def tearDown(self):
        for job in self.jobs:
            docker("rm","-f","exoharness-"+job,check=False)
        docker("rm","-f",self.container,check=False)
        self.temp.cleanup()

    def call(self, tool_name, owner="a", **arguments):
        body={"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":tool_name,"arguments":arguments,
            "_meta":{"exoharness":{"subject":owner*64,"software_engineering":True}}}}
        request=Request(self.url,data=json.dumps(body).encode(),headers={"Content-Type":"application/json","Accept":"application/json, text/event-stream","Authorization":"Bearer "+"project-test-token-"*4})
        with urlopen(request,timeout=20) as response:
            result=json.load(response)["result"]
        if result.get("isError"):
            # Terminal failed jobs are expected in the local repair test.
            if result.get("structuredContent",{}).get("status") != "failed":
                raise ValueError(result)
        return result

    def execute(self, project, revision, code, outputs=None):
        write(self.data/"worker.json",{"image":self.image})
        submitted=self.call("run_project",project_id=project,expected_revision=revision,entrypoint=f"check-{revision}.sh",code=code,outputs=outputs or [])
        job=submitted["structuredContent"]["job_id"]
        self.jobs.append(job)
        try:
            run_job(self.data/"jobs"/job,self.packages,self.image,self.data/"worker.json")
        except ValueError as error:
            # Same failure boundary as the broker's queue loop. Preserve its
            # real execution log and captured workspace for the local repair.
            write(self.data/"jobs"/job/"state.json",{"status":"failed","error":str(error)})
        return self.call("get_project_job",project_id=project,job_id=job,observe=True)

    def test_multifile_failure_local_patch_tests_git_diff_restore_and_publication(self):
        project=self.call("open_project",name="tiny-library")["structuredContent"]["project_id"]
        self.call("edit_project",project_id=project,expected_revision=0,files=[
            {"path":"mathlib.py","content":"def add(a,b):\n    return a-b\n"},
            {"path":"test_mathlib.py","content":"from mathlib import add\nassert add(2,3)==5\nprint('TESTS_PASSED')\n"},
            {"path":"AGENTS.md","content":"Preserve public API; run test_mathlib.py after changes."}])
        checkpoint=self.call("checkpoint_project",project_id=project,expected_revision=1,label="before fix")["structuredContent"]["checkpoint"]
        first=self.execute(project,1,"set -eu\npython -B test_mathlib.py\n")
        self.assertTrue(first["isError"])
        self.assertIn("AssertionError",first["structuredContent"]["execution"]["log_tail"])
        current=self.call("read_project",project_id=project,path="mathlib.py")["structuredContent"]
        self.assertEqual(current["revision"],2)
        test_sha=self.call("read_project",project_id=project,path="test_mathlib.py")["structuredContent"]["sha256"]
        self.call("edit_project",project_id=project,expected_revision=2,patches=[{"path":"mathlib.py","sha256":current["sha256"],"before":"a-b","after":"a+b"}])
        second=self.execute(project,3,"set -eu\npython -B test_mathlib.py\ngit init -q\ngit -c user.name=Fixture -c user.email=fixture@example.test add mathlib.py test_mathlib.py AGENTS.md\ngit -c user.name=Fixture -c user.email=fixture@example.test commit -qm 'verified library'\ngit log -1 --format=%s\nprintf 'verified build' > build.txt\n",["build.txt"])
        self.assertFalse(second.get("isError",False),second)
        content=second["structuredContent"]
        self.assertEqual(content["execution"]["exit_code"],0)
        self.assertIn("TESTS_PASSED",content["execution"]["log_tail"])
        self.assertIn("verified library",content["execution"]["log_tail"])
        self.assertEqual([a["filename"] for a in content["artifacts"]],["build.txt"])
        self.assertNotIn("ws_",json.dumps(second))
        self.assertEqual(self.call("read_project",project_id=project,path="test_mathlib.py")["structuredContent"]["sha256"],test_sha)
        diff=self.call("diff_project",project_id=project,checkpoint=checkpoint,path="mathlib.py")["structuredContent"]
        self.assertIn("+    return a+b",diff["diff"])
        for owner in ["b"]:
            with self.assertRaises(ValueError): self.call("read_project",owner=owner,project_id=project)
            with self.assertRaises(ValueError): self.call("get_project_job",owner=owner,project_id=project,job_id=content["job_id"])
        self.call("restore_project",project_id=project,expected_revision=4,checkpoint=checkpoint,confirm=True)
        self.assertEqual(self.call("read_project",project_id=project,path="mathlib.py")["structuredContent"]["sha256"],current["sha256"])

    def test_project_cancel_uses_broker_without_running_the_queued_script(self):
        project=self.call("open_project",name="cancel-fixture")["structuredContent"]["project_id"]
        submitted=self.call("run_project",project_id=project,expected_revision=0,code="printf forbidden > marker.txt",outputs=["marker.txt"])
        job=submitted["structuredContent"]["job_id"]
        self.jobs.append(job)
        self.call("cancel_project_job",project_id=project,job_id=job)
        run_job(self.data/"jobs"/job,self.packages,self.image,self.data/"worker.json")
        result=self.call("get_project_job",project_id=project,job_id=job,observe=True)["structuredContent"]
        self.assertEqual(result["status"],"cancelled")
        self.assertNotIn("marker.txt",json.dumps(self.call("read_project",project_id=project)))
