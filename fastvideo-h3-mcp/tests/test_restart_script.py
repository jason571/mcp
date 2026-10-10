from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class RestartScriptTests(unittest.TestCase):
    """Exercise service-control decisions without touching the host systemd."""

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="fastvideo-restart-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.script = self.root / "restart.sh"
        shutil.copyfile(Path(__file__).resolve().parents[1] / "restart.sh", self.script)
        (self.root / "deploy").mkdir()
        (self.root / "deploy" / "production.env").write_text(
            "FASTVIDEO_MCP_PORT=8765\n", encoding="utf-8"
        )
        (self.root / ".venv" / "bin").mkdir(parents=True)
        (self.root / ".venv" / "bin" / "python").symlink_to(sys.executable)
        binaries = self.root / "bin"
        binaries.mkdir()
        stub = f"#!{sys.executable}\n" + '''
import json, os, sys
from pathlib import Path
name = Path(sys.argv[0]).name
args = sys.argv[1:]
with open(os.environ["RESTART_TEST_LOG"], "a") as log:
    log.write(json.dumps([name, *args]) + "\\n")
if name == "stat":
    print("root")  # Keep root-run CI tests in this isolated fixture.
elif name == "systemctl":
    if "--property=LoadState" in args:
        print(os.environ.get("RESTART_TEST_LOAD", "loaded"))
    elif "--property=MainPID" in args:
        print("12345")
    elif "restart" in args:
        sys.exit(int(os.environ.get("RESTART_TEST_FAIL", "0")))
elif name == "curl":
    print(os.environ.get("RESTART_TEST_HEALTH", '{"status":"ok","active_jobs":0,"queue_size":0}'))
elif name == "journalctl":
    print("example startup failure")
'''
        for name in ("stat", "systemctl", "curl", "journalctl"):
            executable = binaries / name
            executable.write_text(stub, encoding="utf-8")
            executable.chmod(0o755)
        self.log = self.root / "commands.jsonl"
        self.environment = {
            **os.environ,
            "PATH": f"{binaries}{os.pathsep}{os.environ['PATH']}",
            "RESTART_TEST_LOG": str(self.log),
        }

    def run_script(self, *arguments: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["bash", str(self.script), *arguments],
            cwd="/tmp",  # Entry point must work outside the project directory.
            env=self.environment,
            capture_output=True,
            text=True,
            timeout=10,
        )

    def commands(self) -> list[list[str]]:
        return [json.loads(line) for line in self.log.read_text().splitlines()]

    def test_restart_waits_for_health_and_reports_pid(self) -> None:
        result = self.run_script()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("PID=12345", result.stdout)
        self.assertIn(
            ["systemctl", "--user", "restart", "fastvideo-h3-mcp.service"],
            self.commands(),
        )

    def test_active_or_queued_jobs_require_force(self) -> None:
        for active, queued in ((1, 0), (0, 2)):
            with self.subTest(active=active, queued=queued):
                self.log.write_text("")
                self.environment["RESTART_TEST_HEALTH"] = json.dumps(
                    {"status": "ok", "active_jobs": active, "queue_size": queued}
                )
                result = self.run_script()
                self.assertEqual(result.returncode, 2)
                self.assertIn("--force", result.stderr)
                self.assertFalse(any("restart" in call for call in self.commands()))
                forced = self.run_script("--force")
                self.assertEqual(forced.returncode, 0, forced.stderr)
                self.assertTrue(any("restart" in call for call in self.commands()))

    def test_check_never_restarts(self) -> None:
        result = self.run_script("--check")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse(any("restart" in call for call in self.commands()))

    def test_restart_failure_prints_logs_and_exits_nonzero(self) -> None:
        self.environment["RESTART_TEST_FAIL"] = "1"
        result = self.run_script()
        self.assertEqual(result.returncode, 1)
        self.assertIn("example startup failure", result.stdout)

    def test_missing_service_does_not_create_or_restart_it(self) -> None:
        self.environment["RESTART_TEST_LOAD"] = "not-found"
        result = self.run_script()
        self.assertEqual(result.returncode, 1)
        self.assertIn("install-user-service.sh", result.stderr)
        self.assertFalse(any("restart" in call for call in self.commands()))


if __name__ == "__main__":
    unittest.main()
