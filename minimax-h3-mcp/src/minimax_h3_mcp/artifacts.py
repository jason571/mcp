from __future__ import annotations

import hashlib
import hmac
import time
from pathlib import Path
from urllib.parse import urlencode

from .config import Settings


class ArtifactSigner:
    def __init__(self, settings: Settings):
        self.settings = settings

    def build_url(self, job_id: str, output_file: Path) -> str:
        base = (
            f"{self.settings.public_base_url}/artifacts/{job_id}/"
            f"{output_file.name}"
        )
        if not self.settings.artifact_signing_key:
            return base
        expires = int(time.time()) + self.settings.artifact_url_ttl_seconds
        signature = self._signature(job_id, output_file.name, expires)
        return f"{base}?{urlencode({'expires': expires, 'signature': signature})}"

    def verify(
        self, job_id: str, filename: str, expires: str | None, signature: str | None
    ) -> bool:
        if not self.settings.artifact_signing_key:
            return False
        if not expires or not signature:
            return False
        try:
            expires_value = int(expires)
        except ValueError:
            return False
        if expires_value < int(time.time()):
            return False
        expected = self._signature(job_id, filename, expires_value)
        return hmac.compare_digest(expected, signature)

    def _signature(self, job_id: str, filename: str, expires: int) -> str:
        key = (self.settings.artifact_signing_key or "").encode("utf-8")
        payload = f"{job_id}\n{filename}\n{expires}".encode("utf-8")
        return hmac.new(key, payload, hashlib.sha256).hexdigest()
