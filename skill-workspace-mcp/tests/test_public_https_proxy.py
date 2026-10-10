"""Opt-in real Docker/network proof for the local MCP egress boundary."""
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

from skill_workspace.mcp_socket import EgressSession, cleanup_orphaned_egress


@unittest.skipUnless(os.environ.get("SKILL_WORKSPACE_NETWORK_INTEGRATION") == "1",
                     "requires local Docker and public HTTPS")
class PublicHttpsEgressIntegration(unittest.TestCase):
    def test_restarted_broker_cleans_only_its_own_orphaned_egress_resources(self):
        image = subprocess.check_output([
            "docker", "image", "inspect", "skill-workspace:0.1.0", "--format", "{{.Id}}",
        ], text=True).strip()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            session = EgressSession(root, image, ["example.com"])
            session.start()
            try:
                cleanup_orphaned_egress(root)
                self.assertNotEqual(subprocess.run(["docker", "inspect", session.proxy],
                    capture_output=True).returncode, 0)
                self.assertNotEqual(subprocess.run(["docker", "network", "inspect", session.network],
                    capture_output=True).returncode, 0)
            finally:
                session.close()

    def test_private_internal_network_uses_only_per_session_public_https_proxy(self):
        image = subprocess.check_output([
            "docker", "image", "inspect", "skill-workspace:0.1.0", "--format", "{{.Id}}",
        ], text=True).strip()
        with tempfile.TemporaryDirectory() as directory:
            session = EgressSession(Path(directory), image, ["example.com"])
            network = session.start()
            try:
                script = """import socket,urllib.request
try:
    socket.create_connection(('1.1.1.1',443),2)
except OSError:
    pass
else:
    raise AssertionError('untrusted MCP unexpectedly reached direct Internet')
with urllib.request.urlopen('https://example.com',timeout=20) as response:
    assert 200 <= response.status < 400, response.status
try:
    urllib.request.urlopen('https://169.254.169.254/latest/meta-data/',timeout=3)
except Exception:
    pass
else:
    raise AssertionError('untrusted MCP reached metadata through the proxy')
for disallowed in ('https://www.python.org', 'http://example.com'):
    try:
        urllib.request.urlopen(disallowed,timeout=3)
    except Exception:
        pass
    else:
        raise AssertionError('untrusted MCP used a disallowed proxy target: '+disallowed)
print('isolated public HTTPS only')
"""
                result = subprocess.run([
                    "docker", "run", "--rm", "--network", network,
                    "--read-only", "--user", "65532:65532", "--cap-drop", "ALL",
                    "--security-opt", "no-new-privileges",
                    "--env", "HTTPS_PROXY=http://mcp-egress:3128",
                    "--env", "HTTP_PROXY=http://mcp-egress:3128",
                    "--env", "ALL_PROXY=http://mcp-egress:3128",
                    "--env", "NO_PROXY=", image, "python", "-c", script,
                ], capture_output=True, text=True, timeout=60)
                self.assertEqual(result.returncode, 0, result.stderr[-3000:])
                self.assertIn("isolated public HTTPS only", result.stdout)
            finally:
                session.close()


if __name__ == "__main__":
    unittest.main()
