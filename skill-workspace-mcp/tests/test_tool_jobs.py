import json
import hashlib
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from skill_workspace.broker import finish, run_job
from skill_workspace.common import read, validate_request, write
from skill_workspace.sandbox import execute_tool


def tool_job():
    return {
        "operation": "invoke_tool",
        "capability_id": "tool_node:user:test:user-tool-fixture",
        "capability_revision": 1,
        "package_name": "user-tool-fixture",
        "package_digest": "a" * 64,
        "runtime_image": "sha256:" + "b" * 64,
        "runtime": "python",
        "entrypoint": "scripts/tool.py",
        "tool_request": {
            "protocol_version": "exoharness.tool.v1",
            "request_id": "invocation-123",
            "tool": "calculate",
            "arguments": {"value": 7},
        },
        "timeout_ms": 1000,
        "max_output_bytes": 1024,
    }


def tool_job_id(request):
    identity = f"{request['capability_id']}\0{request['capability_revision']}\0{request['tool_request']['request_id']}".encode()
    return "work_" + hashlib.sha256(identity).hexdigest()


class IsolatedToolJobTests(unittest.TestCase):
    def test_contract_requires_pinned_package_image_and_typed_input(self):
        validate_request(tool_job())
        with self.assertRaisesRegex(ValueError, "JSON object"):
            validate_request([])
        for mutate in (
            lambda job: job.update(entrypoint="../host.py"),
            lambda job: job.update(runtime="shell"),
            lambda job: job.update(runtime_image="latest"),
            lambda job: job.update(runtime_image=None),
            lambda job: job.update(package_name=42),
            lambda job: job.update(capability_revision=0),
            lambda job: job.update(extra_mount="/"),
            lambda job: job["tool_request"].update(arguments="{}"),
            lambda job: job["tool_request"].update(request_id="bad\nrequest"),
            lambda job: job.update(timeout_ms=300001),
            lambda job: job.update(max_output_bytes=0),
        ):
            with self.subTest(mutate=mutate):
                job = tool_job()
                mutate(job)
                with self.assertRaises(ValueError):
                    validate_request(job)

    def test_stdout_is_exact_json_result_and_stderr_is_separate(self):
        with tempfile.TemporaryDirectory() as directory:
            command = [sys.executable, "-c",
                       "import json,sys; r=json.load(sys.stdin); print(json.dumps({'seen':r['arguments']['value']})); print('diagnostic',file=sys.stderr)"]
            result = execute_tool(command, Path(directory), tool_job())
            self.assertEqual(result["exit_code"], 0)
            self.assertEqual(result["tool_output"], {"seen": 7})
            self.assertEqual(result["log"], "diagnostic\n")

    def test_broker_persists_json_without_publishing_intermediate_files(self):
        with tempfile.TemporaryDirectory() as directory:
            job = Path(directory)
            request = tool_job()
            finish(job, request, json.dumps({"exit_code": 0, "tool_output": {"answer": 42}, "log": ""}).encode())
            self.assertEqual(read(job / "result.json")["tool_output"], {"answer": 42})
            self.assertEqual(read(job / "state.json")["status"], "completed")
            self.assertFalse((job / "published").exists())
            with self.assertRaisesRegex(ValueError, "bounded JSON"):
                finish(job, request, json.dumps({"exit_code": 0, "tool_output": "x" * 2000}).encode())

    def test_changed_job_identity_or_staged_request_never_starts_docker(self):
        with tempfile.TemporaryDirectory() as directory, patch("skill_workspace.broker.docker") as docker:
            root = Path(directory)
            request = tool_job()
            job = root / ("work_" + "c" * 64)
            write(job / "request.json", request)
            write(job / "input" / "request.json", request)
            with self.assertRaisesRegex(ValueError, "identity"):
                run_job(job, root / "packages", request["runtime_image"], root / "worker.json")
            job = root / tool_job_id(request)
            write(job / "request.json", request)
            changed = tool_job()
            changed["tool_request"]["arguments"] = {"value": 8}
            write(job / "input" / "request.json", changed)
            with self.assertRaisesRegex(ValueError, "staged request"):
                run_job(job, root / "packages", request["runtime_image"], root / "worker.json")
            docker.assert_not_called()

    @unittest.skipUnless(os.environ.get("SKILL_WORKSPACE_INTEGRATION") == "1", "requires local immutable OCI image")
    def test_real_tool_protocol_runs_only_inside_offline_container(self):
        image = subprocess.check_output(
            ["docker", "image", "inspect", "skill-workspace:0.1.0", "--format", "{{.Id}}"],
            text=True,
        ).strip()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            request = tool_job()
            request["runtime_image"] = image
            package = root / "tools" / request["package_name"] / request["package_digest"]
            (package / "scripts").mkdir(parents=True)
            (package / "scripts" / "tool.py").write_text(
                "import json,sys\nr=json.load(sys.stdin)\nprint(json.dumps({'answer':r['arguments']['value']*2}))\n"
            )
            write(root / "worker.json", {"at": 0, "image": image})
            write(root / "tool-bindings.json", {request["capability_id"]: {
                "revision": request["capability_revision"], "package_name": request["package_name"],
                "package_digest": request["package_digest"], "image": image,
                "runtime": request["runtime"], "entrypoint": request["entrypoint"]}})
            job = root / "jobs" / tool_job_id(request)
            write(job / "request.json", request)
            write(job / "input" / "request.json", request)
            (job / "input").chmod(0o755)
            (job / "input" / "request.json").chmod(0o644)
            write(job / "state.json", {"status": "queued"})
            name = "exoharness-" + job.name
            try:
                run_job(job, root / "packages", image, root / "worker.json")
                self.assertEqual(read(job / "state.json")["status"], "completed")
                self.assertEqual(read(job / "result.json")["tool_output"], {"answer": 14})
            finally:
                subprocess.run(["docker", "rm", "-f", name], capture_output=True, timeout=30)


if __name__ == "__main__":
    unittest.main()
