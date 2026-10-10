import json
import os
import socket
import subprocess
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from skill_workspace.common import write
from skill_workspace.mcp_socket import McpSessionServer, docker_command, validate_binding
from skill_workspace.prepare_runtime import identity
from skill_workspace.public_https_proxy import admitted_addresses, parse_policy


class LocalMcpSandboxTests(unittest.TestCase):
    def test_public_https_policy_rejects_internal_hosts_and_private_dns(self):
        self.assertEqual(parse_policy('["example.com","api.example.com"]'),
                         ("example.com", "api.example.com"))
        self.assertEqual(parse_policy('["*"]'), ("*",))
        for policy in ['[]', '["*","example.com"]', '["localhost"]',
                       '["metadata.google.internal"]', '["EXAMPLE.COM"]',
                       '["198.18.1.187"]', '["8.8.8.8"]']:
            with self.assertRaises(ValueError, msg=policy):
                parse_policy(policy)
        private = [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("169.254.169.254", 443))]
        with patch("skill_workspace.public_https_proxy.socket.getaddrinfo", return_value=private):
            with self.assertRaisesRegex(ValueError, "private or reserved"):
                admitted_addresses("example.com")

    def test_unbound_or_changed_identity_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            request = {"capability_id": "mcp_server:user:test:fixture", "revision": 1,
                       "package_name": "fixture", "package_digest": "a" * 64,
                       "image": "sha256:" + "b" * 64, "runtime": "python", "entrypoint": "server.py", "secrets": {}}
            with self.assertRaisesRegex(ValueError, "not passed isolated preparation"):
                validate_binding(root, root / "packages" / "skills", request["image"], request)
            request["entrypoint"] = "../host.py"
            with self.assertRaises(ValueError):
                validate_binding(root, root / "packages" / "skills", request["image"], request)

    def test_python_module_entrypoint_is_bound_to_package_main(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            packages = root / "packages" / "skills"
            digest, image = "a" * 64, "sha256:" + "b" * 64
            request = {"capability_id": "mcp_server:user:test:module", "revision": 1,
                       "package_name": "module", "package_digest": digest,
                       "image": image, "runtime": "python_module", "entrypoint": "fixture_server", "secrets": {}}
            package = root / "packages" / "mcp" / "module" / digest
            (package / "fixture_server").mkdir(parents=True)
            (package / "fixture_server" / "__main__.py").write_text("print('ready')")
            prepared = root / "imports" / identity("mcp:module", digest, image)
            write(prepared / "state.json", {"status": "ready", "image": image})
            write(prepared / "runtime" / "verified.json", {"status": "ready"})
            write(root / "mcp-bindings.json", {request["capability_id"]: {
                "revision": 1, "package_name": "module", "package_digest": digest,
                "image": image, "runtime": "python_module", "entrypoint": "fixture_server"}})
            selected, runtime, hosts, secret_values = validate_binding(root, packages, image, request)
            self.assertEqual(selected, package)
            self.assertEqual(hosts, [])
            self.assertEqual(secret_values, {})
            self.assertEqual(docker_command("test", selected, runtime, image, request)[-3:],
                             ["python", "-m", "fixture_server"])
            binding = json.loads((root / "mcp-bindings.json").read_text())
            binding[request["capability_id"]]["egress_hosts"] = ["example.com"]
            write(root / "mcp-bindings.json", binding)
            self.assertEqual(validate_binding(root, packages, image, request)[2], ["example.com"])
            binding[request["capability_id"]]["egress_hosts"] = ["metadata.google.internal"]
            write(root / "mcp-bindings.json", binding)
            with self.assertRaisesRegex(ValueError, "policy"):
                validate_binding(root, packages, image, request)

    @unittest.skipUnless(os.environ.get("SKILL_WORKSPACE_INTEGRATION") == "1", "requires local OCI image")
    def test_real_secret_reaches_only_the_isolated_mcp_process(self):
        image = subprocess.check_output(["docker", "image", "inspect", "skill-workspace:0.1.0",
                                         "--format", "{{.Id}}"], text=True).strip()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            packages = root / "packages" / "skills"
            name, digest = "secretfixture", "c" * 64
            request = {"capability_id": "mcp_server:user:test:secretfixture", "revision": 1,
                       "package_name": name, "package_digest": digest, "image": image,
                       "runtime": "python", "entrypoint": "server.py",
                       "secrets": {"DEMO_API_KEY": "fixture-private-value"}}
            package = root / "packages" / "mcp" / name / digest
            package.mkdir(parents=True)
            (package / "server.py").write_text(
                "import json,os,sys\n"
                "for line in sys.stdin:\n"
                " r=json.loads(line); m=r.get('method'); i=r.get('id')\n"
                " if m=='initialize': v={'protocolVersion':'2025-03-26','capabilities':{'tools':{}},'serverInfo':{'name':'secretfixture','version':'1'}}\n"
                " elif m=='tools/list': v={'tools':[{'name':'check','description':'fixture','inputSchema':{'type':'object'}}]}\n"
                " elif m=='tools/call': v={'content':[{'type':'text','text':'configured='+str(os.getenv('DEMO_API_KEY')=='fixture-private-value')}]}\n"
                " else: continue\n"
                " print(json.dumps({'jsonrpc':'2.0','id':i,'result':v}),flush=True)\n"
            )
            prepared = root / "imports" / identity("mcp:" + name, digest, image)
            write(prepared / "state.json", {"status": "ready", "image": image})
            write(prepared / "runtime" / "verified.json", {"status": "ready"})
            write(root / "mcp-bindings.json", {request["capability_id"]: {
                "revision": 1, "package_name": name, "package_digest": digest,
                "image": image, "runtime": "python", "entrypoint": "server.py",
                "secret_env_names": ["DEMO_API_KEY"]}})
            command = docker_command("secret-test", package, prepared / "runtime", image,
                                     request, has_secrets=True)
            self.assertNotIn("fixture-private-value", " ".join(command))
            self.assertIn("/mcp-entry.py", " ".join(command))
            server = McpSessionServer(root, packages, image)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                with socket.socket(socket.AF_UNIX) as client:
                    client.settimeout(15)
                    client.connect(str(root / "mcp.sock"))
                    stream = client.makefile("rwb", buffering=0)
                    stream.write(json.dumps(request).encode() + b"\n")
                    self.assertEqual(json.loads(stream.readline())["status"], "ready")
                    stream.write(b'{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-03-26","capabilities":{},"clientInfo":{"name":"test","version":"1"}}}\n')
                    self.assertEqual(json.loads(stream.readline())["result"]["serverInfo"]["name"], name)
                    stream.write(b'{"jsonrpc":"2.0","method":"notifications/initialized"}\n')
                    stream.write(b'{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"check","arguments":{}}}\n')
                    self.assertEqual(json.loads(stream.readline())["result"]["content"][0]["text"], "configured=True")
            finally:
                server.shutdown()
                server.server_close()

    @unittest.skipUnless(os.environ.get("SKILL_WORKSPACE_INTEGRATION") == "1", "requires local OCI image")
    def test_real_mcp_initialize_discover_call_and_progress_are_relayed(self):
        image = subprocess.check_output(["docker", "image", "inspect", "skill-workspace:0.1.0",
                                         "--format", "{{.Id}}"], text=True).strip()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            packages = root / "packages" / "skills"
            name, digest = "fixture", "a" * 64
            request = {"capability_id": "mcp_server:user:test:fixture", "revision": 1,
                       "package_name": name, "package_digest": digest, "image": image,
                       "runtime": "python", "entrypoint": "server.py", "secrets": {}}
            package = root / "packages" / "mcp" / name / digest
            package.mkdir(parents=True)
            (package / "server.py").write_text(
                "import json,sys\n"
                "for line in sys.stdin:\n"
                " r=json.loads(line); m=r.get('method'); i=r.get('id')\n"
                " if m=='initialize': v={'protocolVersion':'2025-03-26','capabilities':{'tools':{}},'serverInfo':{'name':'fixture','version':'1'}}\n"
                " elif m=='tools/list': v={'tools':[{'name':'echo','description':'fixture','inputSchema':{'type':'object'}}]}\n"
                " elif m=='tools/call':\n"
                "  print(json.dumps({'jsonrpc':'2.0','method':'notifications/progress','params':{'progressToken':'test','progress':1,'total':1}}),flush=True)\n"
                "  v={'content':[{'type':'text','text':r['params']['arguments']['text']}]}\n"
                " else: continue\n"
                " print(json.dumps({'jsonrpc':'2.0','id':i,'result':v}),flush=True)\n"
            )
            prepared = root / "imports" / identity("mcp:" + name, digest, image)
            write(prepared / "state.json", {"status": "ready", "image": image})
            write(prepared / "runtime" / "verified.json", {"status": "ready"})
            write(root / "mcp-bindings.json", {request["capability_id"]: {
                "revision": 1, "package_name": name, "package_digest": digest,
                "image": image, "runtime": "python", "entrypoint": "server.py"}})
            server = McpSessionServer(root, packages, image)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                with socket.socket(socket.AF_UNIX) as client:
                    client.settimeout(15)
                    client.connect(str(root / "mcp.sock"))
                    stream = client.makefile("rwb", buffering=0)
                    stream.write(json.dumps(request).encode() + b"\n")
                    self.assertEqual(json.loads(stream.readline())["status"], "ready")
                    stream.write(b'{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-03-26","capabilities":{},"clientInfo":{"name":"test","version":"1"}}}\n')
                    self.assertEqual(json.loads(stream.readline())["result"]["serverInfo"]["name"], "fixture")
                    stream.write(b'{"jsonrpc":"2.0","method":"notifications/initialized"}\n')
                    stream.write(b'{"jsonrpc":"2.0","id":2,"method":"tools/list","params":{}}\n')
                    self.assertEqual(json.loads(stream.readline())["result"]["tools"][0]["name"], "echo")
                    stream.write(b'{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"echo","arguments":{"text":"hello"}}}\n')
                    self.assertEqual(json.loads(stream.readline())["method"], "notifications/progress")
                    self.assertEqual(json.loads(stream.readline())["result"]["content"][0]["text"], "hello")
            finally:
                server.shutdown()
                server.server_close()


if __name__ == "__main__":
    unittest.main()
