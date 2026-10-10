from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from urllib.parse import parse_qs, urlsplit

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from media_workflow_mcp.artifacts import Artifact, ArtifactStore
from media_workflow_mcp.config import _env_int


class ArtifactExpiryTests(unittest.TestCase):
    def store(self, ttl):
        return ArtifactStore(SimpleNamespace(
            artifact_url_ttl_seconds=ttl, artifact_signing_key="test-only-key",
            public_base_url="https://media.example.test",
        ))

    def signed(self, store):
        artifact = Artifact("media_" + "a" * 32, "final.mp4", "video/mp4", "", 1, "", 1)
        params = parse_qs(urlsplit(store.signed_url(artifact)).query)
        return artifact, params["expires"][0], params["signature"][0]

    def test_permanent_url_survives_time_and_still_requires_exact_signature(self):
        store = self.store(0)
        artifact, expires, signature = self.signed(store)
        self.assertEqual(expires, "0")
        with patch("media_workflow_mcp.artifacts.time.time", return_value=10**12):
            self.assertTrue(store.verify(artifact.artifact_id, artifact.filename, expires, signature))
            self.assertFalse(store.verify(artifact.artifact_id, "other.mp4", expires, signature))
            self.assertFalse(store.verify(artifact.artifact_id, artifact.filename, expires, "bad"))

    def test_finite_url_expires_and_cannot_be_changed_to_permanent(self):
        store = self.store(3600)
        with patch("media_workflow_mcp.artifacts.time.time", return_value=100):
            artifact, expires, signature = self.signed(store)
            self.assertTrue(store.verify(artifact.artifact_id, artifact.filename, expires, signature))
        self.assertFalse(store.verify(artifact.artifact_id, artifact.filename, "0", signature))
        with patch("media_workflow_mcp.artifacts.time.time", return_value=3700):
            self.assertFalse(store.verify(artifact.artifact_id, artifact.filename, expires, signature))
        self.assertFalse(store.verify(artifact.artifact_id, artifact.filename, "-1", signature))

    def test_ttl_config_accepts_zero_but_rejects_negative_values(self):
        with patch.dict("os.environ", {"MEDIA_MCP_ARTIFACT_URL_TTL_SECONDS": "0"}):
            self.assertEqual(_env_int("MEDIA_MCP_ARTIFACT_URL_TTL_SECONDS", 0, 0), 0)
        with patch.dict("os.environ", {"MEDIA_MCP_ARTIFACT_URL_TTL_SECONDS": "-1"}):
            with self.assertRaises(ValueError):
                _env_int("MEDIA_MCP_ARTIFACT_URL_TTL_SECONDS", 0, 0)


if __name__ == "__main__":
    unittest.main()
