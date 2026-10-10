from __future__ import annotations

import hashlib
import hmac
import json
import secrets
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from urllib.parse import urlencode

from .config import Settings


@dataclass(frozen=True, slots=True)
class Artifact:
    artifact_id: str
    filename: str
    content_type: str
    path: str
    size_bytes: int
    sha256: str
    created_at: int


class ArtifactStore:
    def __init__(self, settings: Settings):
        self.settings = settings

    def add(self, source: Path, filename: str, content_type: str) -> Artifact:
        artifact_id = f"media_{secrets.token_hex(16)}"
        directory = self.settings.artifact_dir / artifact_id
        directory.mkdir(mode=0o700)
        target = directory / Path(filename).name
        source.replace(target)
        artifact = Artifact(
            artifact_id=artifact_id,
            filename=target.name,
            content_type=content_type,
            path=str(target),
            size_bytes=target.stat().st_size,
            sha256=_sha256_file(target),
            created_at=int(time.time()),
        )
        (directory / "artifact.json").write_text(
            json.dumps(asdict(artifact), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return artifact

    def get(self, artifact_id: str, filename: str) -> Artifact | None:
        if (
            not artifact_id.startswith("media_")
            or len(artifact_id) != 38
            or not all(char in "0123456789abcdef" for char in artifact_id[6:])
        ):
            return None
        if Path(filename).name != filename:
            return None
        metadata = self.settings.artifact_dir / artifact_id / "artifact.json"
        try:
            artifact = Artifact(**json.loads(metadata.read_text(encoding="utf-8")))
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            return None
        path = Path(artifact.path).resolve()
        expected_root = (self.settings.artifact_dir / artifact_id).resolve()
        if (
            artifact.artifact_id != artifact_id
            or artifact.filename != filename
            or path.parent != expected_root
            or not path.is_file()
        ):
            return None
        return artifact

    def signed_url(self, artifact: Artifact) -> str:
        ttl = self.settings.artifact_url_ttl_seconds
        expires = int(time.time()) + ttl if ttl else 0
        signature = self._signature(artifact.artifact_id, artifact.filename, expires)
        base = (
            f"{self.settings.public_base_url}/artifacts/"
            f"{artifact.artifact_id}/{artifact.filename}"
        )
        return f"{base}?{urlencode({'expires': expires, 'signature': signature})}"

    def verify(
        self,
        artifact_id: str,
        filename: str,
        expires: str | None,
        signature: str | None,
    ) -> bool:
        if not expires or not signature:
            return False
        try:
            expires_value = int(expires)
        except ValueError:
            return False
        if expires_value < 0 or (expires_value != 0 and expires_value <= int(time.time())):
            return False
        expected = self._signature(artifact_id, filename, expires_value)
        return hmac.compare_digest(expected, signature)

    def _signature(self, artifact_id: str, filename: str, expires: int) -> str:
        payload = f"{artifact_id}\n{filename}\n{expires}".encode()
        return hmac.new(
            self.settings.artifact_signing_key.encode(), payload, hashlib.sha256
        ).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
