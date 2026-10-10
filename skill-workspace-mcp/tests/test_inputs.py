import hashlib
import json
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

from skill_workspace import server
from skill_workspace.common import read, write
from skill_workspace.inputs import ArtifactInputs


class InputTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.data = b"ID3-existing-TTS-bytes"
        self.asset = "att_" + "a" * 64
        self.path = "/v1/attachment-assets/" + self.asset
        self.metadata = {"attachment_id": self.asset, "size_bytes": len(self.data),
                         "sha256": hashlib.sha256(self.data).hexdigest()}
        self.routes, self.requests = {}, []
        fixture = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                fixture.requests.append((self.path, dict(self.headers)))
                if self.path == fixture.path + "/metadata":
                    status, body, headers = 200, json.dumps(fixture.metadata).encode(), {}
                elif self.path == fixture.path:
                    status, body, headers = 200, fixture.data, {}
                else:
                    status, body, headers = fixture.routes.get(self.path, (404, b"not found", {}))
                self.send_response(status)
                for key, value in headers.items(): self.send_header(key, value)
                if "Content-Length" not in headers: self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            def log_message(self, *_args): pass

        self.http = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.http.serve_forever, daemon=True)
        self.thread.start()
        self.base = f"http://127.0.0.1:{self.http.server_port}"
        self.rt = server.Runtime(self.root / "data", self.root / "packages", "key-" * 16,
                                 "https://workspace.test", asset_base=self.base,
                                 download_bases=[self.base + "/artifacts/"])
        server._runtime = self.rt
        image = "sha256:" + "b" * 64
        write(self.rt.root / "worker.json", {"image": image, "at": time.time()})
        write(self.rt.root / "inventory.json", {"image": image, "commands": {"python": True}})
        write(self.rt.root / "task-code.json", {"enabled": True, "image": image})
        self.ws = server.create_workspace().structured_content["workspace_id"]

    def tearDown(self):
        server._runtime = None
        self.http.shutdown()
        self.http.server_close()
        self.thread.join()
        self.temp.cleanup()

    def test_real_attachment_import_and_offline_script_staging(self):
        result = server.edit_workspace(self.ws, 0, inputs=[server.InputFile(
            path="audio/segment-001.mp3", attachment_id=self.asset)]).structured_content
        self.assertEqual(result["revision"], 1)
        self.assertEqual(self.rt.workspaces.bytes(self.ws, "audio/segment-001.mp3"), self.data)
        self.assertEqual([path for path, _ in self.requests], [self.path + "/metadata", self.path])
        self.requests.clear()
        job = server.execute_python(self.ws, 1, code="print('reuse audio')", outputs=[]).structured_content
        self.assertEqual(self.requests, [], "Executing saved inputs must not download or regenerate them")
        inputs = self.rt.jobs / job["job_id"] / "input"
        manifest = read(inputs / "files.json")
        item = next(v for v in manifest if v["path"] == "audio/segment-001.mp3")
        self.assertEqual((inputs / item["file"]).read_bytes(), self.data)
        self.assertEqual(item["sha256"], self.metadata["sha256"])

    def test_attachment_url_compatibility_and_generic_mcp_file(self):
        self.assertEqual(self.rt.input_bytes(self.base + self.path), self.data)
        url = self.base + "/artifacts/job/video.mp4?signature=fixture"
        self.routes["/artifacts/job/video.mp4?signature=fixture"] = (200, b"video-fixture", {})
        imported = server.edit_workspace(self.ws, 0, inputs=[server.InputFile(path="video/clip.mp4", url=url,
            sha256=hashlib.sha256(b"video-fixture").hexdigest())]).structured_content
        self.assertEqual(imported["revision"], 1)
        self.assertEqual(self.rt.workspaces.bytes(self.ws, "video/clip.mp4"), b"video-fixture")
        self.assertFalse(any(k.lower() in {"cookie", "authorization", "proxy-authorization"} for _, headers in self.requests for k in headers))

    def test_id_and_url_are_exclusive_and_paths_are_validated_before_fetch(self):
        for args in [{}, {"url": "x", "attachment_id": self.asset}, {"attachment_id": "guessed"},
                     {"url": "x", "sha256": "wrong"}]:
            with self.assertRaises(ValueError): server.InputFile(path="ok.mp3", **args)
        with self.assertRaises(ValueError): server.InputFile(path="../escape.mp3", attachment_id=self.asset)
        with self.assertRaises(Exception): server.edit_workspace(self.ws, 99, inputs=[server.InputFile(path="ok.mp3", attachment_id=self.asset)])
        with self.assertRaises(Exception): server.edit_workspace(self.ws, 0,
            files=[server.SourceFile(path="ok.mp3", content="conflict")], inputs=[server.InputFile(path="ok.mp3", attachment_id=self.asset)])
        self.assertEqual(self.requests, [])

    def test_identity_size_and_hash_failures_preserve_workspace(self):
        server.edit_workspace(self.ws, 0, files=[server.SourceFile(path="plan.json", content='{"saved":true}')])
        original = self.rt.workspaces.state(self.ws)
        for metadata in [{**self.metadata, "sha256": "f" * 64}, {**self.metadata, "size_bytes": 0},
                         {**self.metadata, "size_bytes": 33 * 1024 * 1024}, {**self.metadata, "attachment_id": "other"}]:
            with self.subTest(metadata=metadata), patch.object(self, "metadata", metadata):
                with self.assertRaises(Exception): server.edit_workspace(self.ws, 1, inputs=[server.InputFile(path="audio.mp3", attachment_id=self.asset)])
                self.assertEqual(self.rt.workspaces.state(self.ws), original)
        self.assertEqual(self.rt.workspaces.bytes(self.ws, "plan.json"), b'{"saved":true}')

    def test_no_partial_edit_if_later_import_fails(self):
        with self.assertRaises(Exception): server.edit_workspace(self.ws, 0, inputs=[
            server.InputFile(path="one.mp3", attachment_id=self.asset),
            server.InputFile(path="two.mp3", url=self.base + "/artifacts/missing.mp3")])
        self.assertEqual(self.rt.workspaces.state(self.ws)["revision"], 0)
        self.assertEqual(self.rt.workspaces.state(self.ws)["files"], {})

    def test_native_json_sources_and_artifacts_share_one_atomic_edit(self):
        value = {"narration": "第一行\n第二行", "audio": "audio.mp3"}
        arguments = {"json_files": [server.JsonFile(path="plan.json", value=value)],
                     "files": [server.SourceFile(path="check.py", content="print('exact source')\n")]}
        with self.assertRaises(Exception):
            server.edit_workspace(self.ws, 0, **arguments, inputs=[
                server.InputFile(path="audio.mp3", attachment_id=self.asset),
                server.InputFile(path="bad.mp4", url=self.base + "/artifacts/missing.mp4")])
        self.assertEqual(self.rt.workspaces.state(self.ws)["files"], {})
        result = server.edit_workspace(self.ws, 0, **arguments,
            inputs=[server.InputFile(path="audio.mp3", attachment_id=self.asset)]).structured_content
        self.assertEqual(result["revision"], 1)
        self.assertEqual(json.loads(self.rt.workspaces.bytes(self.ws, "plan.json")), value)
        self.assertEqual(self.rt.workspaces.bytes(self.ws, "check.py"), b"print('exact source')\n")
        self.assertEqual(self.rt.workspaces.bytes(self.ws, "audio.mp3"), self.data)

    def test_url_security_denies_other_origins_paths_credentials_and_redirects(self):
        bad = ["file:///etc/passwd", "http://169.254.169.254/latest/meta-data/", self.base + "/admin",
               self.base + "/artifacts-evil/x", self.base.replace("127.0.0.1", "localhost") + "/artifacts/x",
               self.base + "/artifacts/%2e%2e/admin", self.base + "/artifacts/%252e%252e/admin",
               self.base + "/artifacts/%5cadmin", self.base + "/artifacts/x#fragment",
               self.base.replace("://", "://user:secret@") + "/artifacts/x"]
        for url in bad:
            with self.subTest(url=url), self.assertRaises(ValueError): self.rt.input_bytes(url)
        self.assertEqual(self.requests, [])
        self.routes["/artifacts/redirect"] = (302, b"redirect", {"Location": self.base + "/admin"})
        with self.assertRaisesRegex(ValueError, "HTTP 302"): self.rt.input_bytes(self.base + "/artifacts/redirect")
        self.assertEqual([p for p, _ in self.requests], ["/artifacts/redirect"])

    def test_default_deny_private_chat_guidance_and_invalid_configuration(self):
        ingress = ArtifactInputs()
        with self.assertRaisesRegex(ValueError, "not configured"): ingress.attachment(self.asset)
        with self.assertRaisesRegex(ValueError, "not an approved"): ingress.url(self.base + "/artifacts/x")
        with self.assertRaisesRegex(ValueError, "attachment_id"): ingress.url("https://agent.test/v1/run-artifacts/artifact-x")
        for base in [self.base, self.base + "/?key=secret", "file:///tmp/files"]:
            with self.assertRaises(ValueError): ArtifactInputs(download_bases=[base])
        self.assertEqual(self.requests, [])

    def test_bounded_download_rejects_large_compressed_or_corrupt_inputs(self):
        for name, body, headers in [("large", b"too much", {}), ("declared", b"x", {"Content-Length": "999"}),
                                     ("gzip", b"x", {"Content-Encoding": "gzip"})]:
            self.routes["/artifacts/" + name] = (200, body, headers)
            with self.assertRaises(ValueError): ArtifactInputs.read_url(self.base + "/artifacts/" + name, 4)
        self.routes["/artifacts/hash"] = (200, b"data", {})
        with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
            self.rt.input_file({"url": self.base + "/artifacts/hash", "sha256": "0" * 64})

    def test_forged_own_signature_never_falls_back_to_network(self):
        with self.assertRaises(ValueError): self.rt.input_bytes("https://workspace.test/workspace-files/work_" + "a" * 64 + "/x.mp4?signature=wrong")
        self.assertEqual(self.requests, [])


if __name__ == "__main__": unittest.main()
