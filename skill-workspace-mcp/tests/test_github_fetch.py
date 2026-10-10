import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from skill_workspace import broker, github_fetch


class GitHubFetchTests(unittest.TestCase):
    def test_interrupted_working_source_restarts_from_persisted_request(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "sources" / ("a" * 64)
            source.mkdir(parents=True)
            (source / "request.json").write_text(json.dumps({
                "owner": "example", "repository": "skill", "ref": "main",
            }))
            (source / "state.json").write_text(json.dumps({"status": "working"}))

            class FinishedDocker:
                returncode = 0

                def communicate(self, timeout=None):
                    return b"", b""

                def poll(self):
                    return 0

            def restart_download(command, **_kwargs):
                self.assertEqual(command[:2], ["docker", "run"])
                (source / "result.json").write_text(json.dumps({"commit": "b" * 40}))
                (source / "archive.zip").write_bytes(b"pinned source archive")
                return FinishedDocker()

            with patch.object(broker.subprocess, "run") as remove_old, patch.object(
                broker.subprocess, "Popen", side_effect=restart_download
            ):
                broker.process_sources(root, "sha256:" + "b" * 64, root / "worker.json")
            self.assertEqual(remove_old.call_args.args[0][:3],
                             ["docker", "rm", "-f"])
            self.assertEqual(json.loads((source / "state.json").read_text())["status"], "completed")
            self.assertEqual(json.loads((source / "state.json").read_text())["commit"], "b" * 40)

    def test_cancelled_source_does_not_start_a_download_container(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "sources" / ("a" * 64)
            source.mkdir(parents=True)
            (source / "request.json").write_text(json.dumps({
                "owner": "example", "repository": "skill", "ref": None,
            }))
            (source / "state.json").write_text(json.dumps({"status": "queued"}))
            (source / "cancel").write_bytes(b"")
            with patch.object(broker, "docker") as docker:
                broker.process_sources(root, "sha256:" + "b" * 64, root / "worker.json")
            docker.assert_not_called()
            self.assertEqual(json.loads((source / "state.json").read_text())["status"], "cancelled")

    def test_cancel_during_source_transfer_stops_its_container(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "sources" / ("a" * 64)
            source.mkdir(parents=True)
            (source / "request.json").write_text(json.dumps({
                "owner": "example", "repository": "skill", "ref": None,
            }))
            (source / "state.json").write_text(json.dumps({"status": "queued"}))

            class RunningDocker:
                returncode = None

                def communicate(self, timeout=None):
                    if timeout is not None:
                        (source / "cancel").write_bytes(b"")
                        raise subprocess.TimeoutExpired("docker", timeout)
                    return b"", b""

                def poll(self):
                    return self.returncode

                def kill(self):
                    self.returncode = -9

            process = RunningDocker()
            with patch.object(broker.subprocess, "run"), patch.object(broker.subprocess, "Popen", return_value=process), patch.object(broker, "docker") as docker:
                broker.process_sources(root, "sha256:" + "b" * 64, root / "worker.json")
            self.assertEqual(json.loads((source / "state.json").read_text())["status"], "cancelled")
            self.assertTrue(any(call.args[:2] == ("rm", "-f") for call in docker.call_args_list))

    def test_egress_filters_private_dns_but_supports_verified_managed_proxy(self):
        address = lambda ip: (2, 1, 6, "", (ip, 443))
        for ip in ["127.0.0.1", "10.1.2.3", "169.254.169.254", "192.168.1.2"]:
            with self.assertRaisesRegex(ValueError, "admitted HTTPS egress"):
                github_fetch.admitted_addresses([address(ip)])
        self.assertEqual(github_fetch.admitted_addresses([address("198.18.0.25")]),
                         [address("198.18.0.25")])
        self.assertEqual(github_fetch.admitted_addresses([address("140.82.112.5"), address("198.18.0.25")]),
                         [address("140.82.112.5")])

    def test_fixes_commit_and_accepts_only_codeload_redirect(self):
        sha = "a" * 40
        calls = []

        def response(host, path, *, archive=False):
            calls.append((host, path, archive))
            if path == "/repos/example/skill":
                return 200, {}, json.dumps({"private": False, "full_name": "example/skill",
                    "default_branch": "main", "license": {"spdx_id": "MIT"}}).encode()
            if path.endswith("/commits/main"):
                return 200, {}, json.dumps({"sha": sha}).encode()
            if path.endswith("/zipball/" + sha):
                return 302, {"location": f"https://codeload.github.com/example/skill/legacy.zip/{sha}"}, b""
            return 200, {}, b"PK\x03\x04archive"

        with tempfile.TemporaryDirectory() as directory, patch.object(github_fetch, "get", response):
            result = github_fetch.fetch({"owner": "example", "repository": "skill", "ref": None}, Path(directory))
            self.assertEqual(result["commit"], sha)
            self.assertEqual((Path(directory) / "archive.zip").read_bytes(), b"PK\x03\x04archive")
        self.assertEqual(calls[-1][0], "codeload.github.com")
        self.assertIn(sha, calls[-1][1])

    def test_rejects_unlicensed_repository_and_unsafe_redirect(self):
        info = {"private": False, "full_name": "example/skill", "default_branch": "main", "license": None}
        with tempfile.TemporaryDirectory() as directory, patch.object(github_fetch, "get", return_value=(200, {}, json.dumps(info).encode())):
            with self.assertRaisesRegex(ValueError, "open-source license"):
                github_fetch.fetch({"owner": "example", "repository": "skill", "ref": None}, Path(directory))
        sha = "b" * 40
        def redirect(host, path, *, archive=False):
            if path == "/repos/example/skill":
                return 200, {}, json.dumps({**info, "license": {"spdx_id": "MIT"}}).encode()
            if path.endswith("/commits/main"):
                return 200, {}, json.dumps({"sha": sha}).encode()
            return 302, {"location": "https://internal.example.test/private"}, b""
        with tempfile.TemporaryDirectory() as directory, patch.object(github_fetch, "get", redirect):
            with self.assertRaisesRegex(ValueError, "redirected outside"):
                github_fetch.fetch({"owner": "example", "repository": "skill", "ref": None}, Path(directory))

    @unittest.skipUnless(os.environ.get("GITHUB_FETCH_INTEGRATION") == "1", "requires public GitHub and Docker")
    def test_public_repository_is_downloaded_in_disposable_container(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)
            (source / "request.json").write_text(json.dumps({
                "owner": "microsoft", "repository": "playwright-mcp", "ref": None,
            }))
            result = subprocess.run([
                "docker", "run", "--rm", "--network", "bridge", "--read-only",
                "--user", f"{os.getuid()}:{os.getgid()}", "--cap-drop", "ALL",
                "--security-opt", "no-new-privileges", "--memory", "256m",
                "--mount", f"type=bind,src={source},dst=/source",
                "--mount", f"type=bind,src={Path(github_fetch.__file__).resolve()},dst=/fetch.py,readonly",
                "skill-workspace:0.1.0", "python", "-I", "/fetch.py",
            ], capture_output=True, text=True, timeout=120)
            self.assertEqual(result.returncode, 0, result.stderr[-2000:])
            metadata = json.loads((source / "result.json").read_text())
            self.assertRegex(metadata["commit"], r"^[0-9a-f]{40}$")
            self.assertGreater((source / "archive.zip").stat().st_size, 1000)


if __name__ == "__main__":
    unittest.main()
