import copy
import hashlib
import json
import os
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from skill_workspace.common import read, validate_request, write
from skill_workspace.broker import container_args, finish, run_job
from skill_workspace.program import approved


def request(image="sha256:" + "a" * 64):
    return {"operation":"invoke_program", "runtime_image":image, "invocation_id":"program-test",
            "code":"value = await tools.call('query', {'n': input})\nreturn {'sum': sum(value['items'])}",
            "input":3, "bindings":["query"], "max_calls":3, "timeout_ms":10000, "max_output_bytes":4096}


class ProgramContractTests(unittest.TestCase):
    def test_contract_does_not_accept_host_policy_from_code(self):
        validate_request(request())
        for key, value in [("runtime_image","latest"),("timeout_ms",300001),("max_calls",0),
                           ("max_calls",True),("max_output_bytes",65537),("bindings",["../secret"]),
                           ("bindings",["query","query"]),("code","")]:
            data=request(); data[key]=value
            with self.subTest(key=key), self.assertRaises(ValueError): validate_request(data)
        for key in ["host_mount","environment","network","shell","package_name","tool_url"]:
            data=request(); data[key]="untrusted"
            with self.subTest(key=key), self.assertRaises(ValueError): validate_request(data)

    def test_disabled_or_changed_image_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); data=request()
            write(root / "worker.json", {"image":data["runtime_image"]})
            write(root / "task-code.json", {"image":data["runtime_image"], "enabled":False})
            with self.assertRaises(ValueError): approved(root,data)
            write(root / "task-code.json", {"image":data["runtime_image"], "enabled":True})
            approved(root,data)
            data["runtime_image"]="sha256:" + "b"*64
            with self.assertRaises(ValueError): approved(root,data)

    def test_program_mount_is_only_the_job_mailbox(self):
        image=request()["runtime_image"]
        args=container_args(Path("/owned/job"),None,image,Path("/trusted/program_runner.py"))
        self.assertIn("none",args); self.assertIn("--read-only",args)
        self.assertIn("type=bind,src=/owned/job/rpc,dst=/rpc",args)
        self.assertFalse(any("docker.sock" in arg or "dst=/skill" in arg for arg in args))

    def test_rejected_identity_never_calls_docker(self):
        with tempfile.TemporaryDirectory() as directory, patch("skill_workspace.broker.docker") as docker:
            job=Path(directory); data=request()
            write(job / "request.json",data); write(job / "input/request.json",data)
            with self.assertRaisesRegex(ValueError,"identity"): run_job(job,job,data["runtime_image"],job / "worker.json")
            docker.assert_not_called()

    def test_output_is_native_json_and_intermediates_are_not_published(self):
        with tempfile.TemporaryDirectory() as directory:
            job=Path(directory); data=request()
            finish(job,data,json.dumps({"exit_code":0,"value":{"sum":6},"log":"done"}).encode())
            self.assertEqual(read(job / "result.json")["value"],{"sum":6})
            self.assertFalse((job / "published").exists())
            with self.assertRaisesRegex(ValueError,"program failed"):
                finish(job,data,json.dumps({"exit_code":1,"error":"bad binding"}).encode())

    def test_missing_submitted_container_is_not_recreated(self):
        with tempfile.TemporaryDirectory() as directory, patch("skill_workspace.broker.inspect", return_value=None), patch("skill_workspace.broker.docker") as docker:
            root=Path(directory); data=request()
            job=root / "jobs" / ("work_" + hashlib.sha256(data["invocation_id"].encode()).hexdigest())
            write(job / "request.json",data); write(job / "input/request.json",data)
            write(job / "state.json",{"status":"working","started_at":time.time()})
            write(root / "worker.json",{"image":data["runtime_image"]})
            write(root / "task-code.json",{"image":data["runtime_image"],"enabled":True})
            with self.assertRaisesRegex(ValueError,"not automatically repeated"):
                run_job(job,root / "packages",data["runtime_image"],root / "worker.json")
            docker.assert_not_called()
            self.assertTrue((job / "request.json").is_file())


@unittest.skipUnless(os.environ.get("SKILL_WORKSPACE_INTEGRATION") == "1", "requires existing OCI runtime")
class ProgramOciTests(unittest.TestCase):
    def invoke(self, code=None, cancel=False):
        with tempfile.TemporaryDirectory(prefix="exoharness-program-") as directory:
            root=Path(directory); image=os.environ["WORKSPACE_IMAGE_ID"]; data=request(image)
            if code is not None: data["code"]=code
            job=root / "jobs" / ("work_" + hashlib.sha256(data["invocation_id"].encode()).hexdigest())
            write(job / "request.json",data); write(job / "input/request.json",data)
            (job / "input").chmod(0o755); (job / "input/request.json").chmod(0o444)
            (job / "rpc").mkdir(mode=0o777); (job / "rpc").chmod(0o777)
            write(job / "state.json",{"status":"queued"})
            write(root / "worker.json",{"image":image})
            write(root / "task-code.json",{"image":image,"enabled":True})
            errors=[]; calls=[]
            def broker():
                try: run_job(job,root / "packages",image,root / "worker.json")
                except Exception as error:
                    errors.append(str(error))
                    from skill_workspace.broker import docker
                    docker("rm","-f","exoharness-" + job.name,check=False)
            worker=threading.Thread(target=broker); worker.start()
            try:
                deadline=time.monotonic()+30
                responded=set()
                while worker.is_alive() and time.monotonic()<deadline:
                    for path in (job / "rpc").glob("*.request.json"):
                        if path.name in responded: continue
                        call=read(path); calls.append(call); responded.add(path.name)
                        target=path.with_name(path.name.replace(".request.json",".response.json"))
                        write(target,{"is_error":False,"result":{"items":[1,2,3]}}); target.chmod(0o444)
                    if cancel and read(job / "state.json")["status"]=="working": (job / "cancel").touch()
                    time.sleep(.02)
                self.assertLess(time.monotonic(),deadline,"program did not terminate")
            finally:
                if worker.is_alive(): (job / "cancel").touch()
                worker.join(10)
                self.assertFalse(worker.is_alive())
            result=read(job / "result.json") if (job / "result.json").exists() else None
            return result,errors,calls,read(job / "state.json")

    def test_real_program_calls_and_reduces_results(self):
        result,errors,calls,_=self.invoke()
        self.assertEqual(errors,[]); self.assertEqual(result["value"],{"sum":6})
        self.assertEqual(calls,[{"binding":"query","arguments":{"n":3}}])

    def test_real_program_has_no_network_host_socket_or_package(self):
        result,errors,_,_=self.invoke("import os, socket\nassert not os.path.exists('/var/run/docker.sock')\nassert not os.path.exists('/skill')\ns = socket.socket()\ns.settimeout(.2)\ntry:\n    s.connect(('1.1.1.1', 80))\n    raise AssertionError('network escaped')\nexcept OSError:\n    pass\nreturn {'isolated': True}")
        self.assertEqual(errors,[]); self.assertEqual(result["value"],{"isolated":True})

    def test_real_program_rejects_undeclared_bindings(self):
        result,errors,calls,_=self.invoke("return await tools.call('undeclared', {})")
        self.assertIsNone(result); self.assertIn("explicit binding"," ".join(errors)); self.assertEqual(calls,[])

    def test_real_program_stops_at_call_limit(self):
        result,errors,calls,_=self.invoke("for i in range(4):\n    await tools.call('query', {'n':i})\nreturn 1")
        self.assertIsNone(result); self.assertIn("tool-call limit"," ".join(errors)); self.assertEqual(len(calls),3)

    def test_real_program_is_cancelled_without_host_fallback(self):
        result,errors,calls,state=self.invoke("import asyncio\nawait asyncio.sleep(60)\nreturn 1",cancel=True)
        self.assertIsNone(result); self.assertEqual(errors,[]); self.assertEqual(calls,[]); self.assertEqual(state["status"],"cancelled")

    def test_real_program_rejects_oversized_output(self):
        result,errors,calls,_=self.invoke("return 'x' * 20000")
        self.assertIsNone(result); self.assertIn("byte limit"," ".join(errors)); self.assertEqual(calls,[])


if __name__ == "__main__": unittest.main()
