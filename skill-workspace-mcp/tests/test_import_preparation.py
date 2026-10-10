import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from skill_workspace.common import require_package_approval, write
from skill_workspace.prepare_runtime import identity, preparation_args, prepared_runtime, process_pending
from skill_workspace.runtime_installer import requirements, check


class PreparationTests(unittest.TestCase):
    def test_native_dependency_files_and_no_skill_rewrite(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "SKILL.md").write_text("unchanged instructions")
            (root / "requirements.txt").write_text("# comment\npackaging>=20\n")
            (root / "pyproject.toml").write_text('[project]\ndependencies=["packaging>=20"]\n')
            (root / "package.json").write_text(json.dumps({"dependencies": {"example-lib": "^1.2.0"}, "scripts": {"postinstall": "touch /host"}}))
            self.assertEqual(requirements(root), (["packaging>=20"], {"example-lib": "^1.2.0"}))
            self.assertEqual((root / "SKILL.md").read_text(), "unchanged instructions")

    def test_untrusted_installer_options_and_urls_rejected(self):
        for value in ["-r /secret", "--index-url https://evil.invalid", "lib @ https://evil.invalid/a.whl", "../setup.py"]:
            with self.subTest(value=value), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                (root / "requirements.txt").write_text(value)
                with self.assertRaises(Exception):
                    requirements(root)
        for value in ["git+ssh://example.invalid/a", "file:/host", "https://example.invalid/a.tgz", "latest"]:
            with self.subTest(value=value), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                (root / "package.json").write_text(json.dumps({"dependencies": {"library": value}}))
                with self.assertRaises(ValueError):
                    requirements(root)

    def test_checks_imports_without_executing_skill(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "scripts").mkdir()
            marker = root / "must-not-exist"
            script = root / "scripts" / "main.py"
            script.write_text(f"import json\nfrom pathlib import Path\nPath({str(marker)!r}).write_text('bad')\n")
            result = check(root, root / "runtime", [], {})
            self.assertFalse(result["business_code_executed"])
            self.assertFalse(marker.exists())
            script.write_text("import nonexistent_fixture_dependency_xyz\n")
            with self.assertRaisesRegex(ValueError, "nonexistent_fixture_dependency_xyz"):
                check(root, root / "runtime", [], {})

    def test_optional_and_local_imports_are_not_false_missing_packages(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "scripts").mkdir()
            (root / "scripts/helper.py").write_text("value=1")
            (root / "scripts/main.py").write_text("import helper\ntry:\n import optional_fixture_xyz\nexcept ImportError:\n pass\n")
            self.assertEqual(check(root, root / "runtime", [], {})["status"], "ready")

    def test_declared_missing_and_wrong_versions_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for requirement in ["nonexistent-fixture-xyz==1", "packaging<0"]:
                with self.assertRaisesRegex(ValueError, "Python 依赖未满足"):
                    check(root, root, [requirement], {})

    def test_nested_modules_resolve_ancestor_entrypoint_helpers_without_execution(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cli = root / "scripts" / "office"
            (cli / "validators").mkdir(parents=True)
            (cli / "helpers").mkdir()
            (cli / "helpers" / "__init__.py").write_text("raise RuntimeError('must not import during checking')")
            (cli / "validate.py").write_text("from validators import report\n")
            (cli / "validators" / "report.py").write_text("from helpers import render\n")
            self.assertEqual(check(root, root / "runtime", [], {})["status"], "ready")
            (cli / "validators" / "report.py").write_text("import not_shipped_fixture_dependency\n")
            with self.assertRaisesRegex(ValueError, "not_shipped_fixture_dependency"):
                check(root, root / "runtime", [], {})

    def test_installer_and_execution_network_boundaries(self):
        image = "sha256:" + "a" * 64
        job = Path("/approved") / ("b" * 64)
        with patch("skill_workspace.prepare_runtime.os.getuid", return_value=1000), patch("skill_workspace.prepare_runtime.os.getgid", return_value=1000):
            online = preparation_args(job, Path("/package"), image, Path("/script"))
            offline = preparation_args(job, Path("/package"), image, Path("/script"), True)
        self.assertEqual(online[online.index("--network") + 1], "bridge")
        self.assertEqual(offline[offline.index("--network") + 1], "none")
        self.assertIn("--read-only", online)
        self.assertIn("1000:1000", online)
        self.assertIn("PYTHONPATH=", online)
        self.assertIn("NODE_OPTIONS=--dns-result-order=ipv4first", online)
        self.assertEqual(online[-4:], ["python", "-I", "/prepare.py", "install"])
        self.assertFalse(any("docker.sock" in item for item in online))
        self.assertIn("src=/package,dst=/skill,readonly", " ".join(online))
        with self.assertRaises(ValueError):
            preparation_args(job, Path("/package"), "arbitrary:latest", Path("/script"))
        with patch.dict(os.environ, {"WORKSPACE_DEPENDENCY_DNS_ORDER":"verbatim --require=/skill/evil.js"}), patch("skill_workspace.prepare_runtime.os.getuid", return_value=1000):
            with self.assertRaisesRegex(ValueError, "WORKSPACE_DEPENDENCY_DNS_ORDER"):
                preparation_args(job, Path("/package"), image, Path("/script"))

    def test_exact_package_and_image_approval_no_global_execution_grant(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = "sha256:" + "a" * 64
            request = {"package_name": "new-skill", "package_digest": "b" * 64, "runtime": "python"}
            write(root / "worker.json", {"image": image})
            job = root / "imports" / identity("new-skill", "b" * 64, image)
            write(job / "state.json", {"status": "preparing", "image": image})
            with self.assertRaisesRegex(ValueError, "not ready"):
                require_package_approval(root, request)
            write(job / "state.json", {"status": "ready", "image": image})
            with self.assertRaisesRegex(ValueError, "verification is missing"):
                require_package_approval(root, request)
            write(job / "runtime/verified.json", {"status": "ready"})
            require_package_approval(root, request)
            self.assertEqual(prepared_runtime(root, request, image), job / "runtime")
            with self.assertRaises(ValueError):
                require_package_approval(root, {**request, "package_digest": "c" * 64})
            write(root / "worker.json", {"image": "sha256:" + "d" * 64})
            with self.assertRaises(ValueError):
                require_package_approval(root, request)
            self.assertFalse((root / "task-code.json").exists())

    def test_bad_queue_identity_never_launches_a_container(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            job = root / "imports" / ("b" * 64)
            write(job / "state.json", {"status": "queued"})
            write(job / "request.json", {"package_name": "fixture", "package_digest": "c" * 64, "image": "bad"})
            with patch("skill_workspace.prepare_runtime.subprocess.run") as run:
                process_pending(root, root / "packages", "sha256:" + "a" * 64, root / "worker.json")
                run.assert_not_called()
            self.assertEqual(json.loads((job / "state.json").read_text())["status"], "failed")


@unittest.skipUnless(os.getenv("IMPORT_PREPARATION_INTEGRATION") == "1", "requires Docker and public dependency download")
class RealPreparationTests(unittest.TestCase):
    def test_prepare_then_run_original_skill_offline_and_reuse(self):
        import hashlib
        from skill_workspace.broker import docker, run_job
        from skill_workspace.common import read

        image = os.environ.get("WORKSPACE_IMAGE_ID") or docker("image", "inspect", "skill-workspace:0.1.0", "--format", "{{.Id}}").stdout.decode().strip()
        with tempfile.TemporaryDirectory(prefix="exoharness-import-test-") as directory, patch.dict(os.environ, {"WORKSPACE_IMAGE_ID": image}):
            root = Path(directory)
            digest = "f" * 64
            package = root / "packages" / "brand-new-fixture" / digest
            (package / "scripts").mkdir(parents=True)
            (package / "SKILL.md").write_text("---\nname: brand-new-fixture\ndescription: Independent fixture\n---\nUse scripts/query.py\n")
            (package / "requirements.txt").write_text("colorama==0.4.6\n")
            (package / "package.json").write_text(json.dumps({"dependencies": {"is-number": "7.0.0"},
                "scripts": {"postinstall": "touch /runtime/UNSAFE_NPM_HOOK"}}))
            (package / "sitecustomize.py").write_text("from pathlib import Path\nif Path('/prepare.py').is_file():\n Path('/runtime/UNSAFE_STARTUP').write_text('executed')\n")
            script = "import colorama\nimport subprocess\nfrom pathlib import Path\nnode = subprocess.check_output(['node', '-e', \"process.stdout.write(String(require('is-number')(42)))\"]).decode()\nPath('output.txt').write_text(colorama.__version__ + ':' + node)\n"
            (package / "scripts/query.py").write_text(script)
            original = hashlib.sha256((package / "SKILL.md").read_bytes()).hexdigest()
            key = identity("brand-new-fixture", digest, image)
            preparation = root / "imports" / key
            write(root / "worker.json", {"image": image})
            write(preparation / "request.json", {"package_name": "brand-new-fixture", "package_digest": digest, "image": image})
            write(preparation / "state.json", {"status": "queued"})
            process_pending(root, root / "packages", image, root / "worker.json")
            result = read(preparation / "state.json")
            self.assertEqual(result["status"], "ready", result)
            self.assertFalse((package / "output.txt").exists())
            self.assertEqual(hashlib.sha256((package / "SKILL.md").read_bytes()).hexdigest(), original)
            self.assertTrue((preparation / "runtime/python/colorama").is_dir())
            self.assertFalse((preparation / "runtime/UNSAFE_STARTUP").exists(), "package startup hooks must not run during preparation")
            self.assertFalse((preparation / "runtime/UNSAFE_NPM_HOOK").exists())
            # An identical verified package/image is a cache hit, not a reinstall.
            with patch("skill_workspace.prepare_runtime.subprocess.Popen") as process:
                process_pending(root, root / "packages", image, root / "worker.json")
                process.assert_not_called()
            job = root / "jobs" / ("work_" + os.urandom(32).hex())
            request = {"package_name": "brand-new-fixture", "package_digest": digest, "runtime": "python",
                       "files": [{"path": "placeholder.py", "content": ""}], "entrypoint_root": "skill",
                       "entrypoint": "scripts/query.py", "outputs": ["output.txt"]}
            # validate_request requires a declared workspace entrypoint for old
            # non-workspace requests. Use an actual private workspace here.
            from skill_workspace.workspaces import Workspaces
            workspace = Workspaces(root).create("brand-new-fixture", digest, image)
            request.update(workspace_id=workspace["workspace_id"], workspace_revision=0, runtime_image=image)
            write(job / "request.json", request)
            write(job / "input/request.json", request)
            (job / "input/request.json").chmod(0o644)
            write(job / "state.json", {"status": "queued"})
            workspaces = Workspaces(root)
            workspace_state = workspaces.state(workspace["workspace_id"])
            workspace_state["active_job"] = job.name
            write(workspaces.directory(workspace["workspace_id"]) / "state.json", workspace_state)
            try:
                run_job(job, root / "packages", image, root / "worker.json")
                self.assertEqual((job / "published/output.txt").read_text(), "0.4.6:true")
            finally:
                docker("rm", "-f", "exoharness-" + job.name, check=False)


if __name__ == "__main__":
    unittest.main()
