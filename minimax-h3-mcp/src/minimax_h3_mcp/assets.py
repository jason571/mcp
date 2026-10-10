from __future__ import annotations

import base64
import binascii
import io
import re
import secrets
from pathlib import Path
from urllib.parse import urlsplit

import aiohttp
from PIL import Image


_ASSET_ID_RE = re.compile(r"^img_[a-f0-9]{24}$")
_IMAGE_FORMATS = {
    "image/png": (".png", b"\x89PNG\r\n\x1a\n", "PNG"),
    "image/jpeg": (".jpg", b"\xff\xd8\xff", "JPEG"),
    "image/webp": (".webp", b"RIFF", "WEBP"),
}
_DOWNLOAD_CHUNK_BYTES = 64 * 1024


class AssetStore:
    def __init__(self, asset_dir: Path, max_image_bytes: int):
        self.asset_dir = asset_dir.resolve()
        self.max_image_bytes = max_image_bytes
        self.asset_dir.mkdir(parents=True, exist_ok=True)

    def save_image_base64(self, data_base64: str, mime_type: str) -> dict[str, object]:
        if mime_type not in _IMAGE_FORMATS:
            raise ValueError("mime_type must be image/png, image/jpeg, or image/webp")
        try:
            payload = base64.b64decode(data_base64, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise ValueError("data_base64 is not valid base64") from exc
        return self._save_new_image(payload, mime_type)

    async def fetch_image_url(self, url: str) -> dict[str, object]:
        self._validate_image_url(url)
        timeout = aiohttp.ClientTimeout(total=30)
        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(url, allow_redirects=False) as response:
                    if response.status != 200:
                        raise ValueError(
                            f"reference image fetch failed with HTTP {response.status}"
                        )
                    declared_length = response.content_length
                    if (
                        declared_length is not None
                        and declared_length > self.max_image_bytes
                    ):
                        raise ValueError(
                            f"reference image exceeds {self.max_image_bytes} bytes"
                        )
                    payload_buffer = bytearray()
                    async for chunk in response.content.iter_chunked(
                        _DOWNLOAD_CHUNK_BYTES
                    ):
                        if len(payload_buffer) + len(chunk) > self.max_image_bytes:
                            raise ValueError(
                                f"reference image exceeds {self.max_image_bytes} bytes"
                            )
                        payload_buffer.extend(chunk)
                    payload = bytes(payload_buffer)
                    mime_type = (
                        response.headers.get("Content-Type", "")
                        .split(";", 1)[0]
                        .strip()
                        .lower()
                    )
        except (aiohttp.ClientError, TimeoutError) as exc:
            raise ValueError("reference image HTTP fetch failed") from exc
        return self._save_new_image(payload, mime_type)

    @staticmethod
    def _validate_image_url(url: str) -> None:
        try:
            parsed = urlsplit(url)
            port = parsed.port
        except ValueError as exc:
            raise ValueError("reference image URL is invalid") from exc
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
            or parsed.fragment
            or (port is not None and not 1 <= port <= 65535)
        ):
            raise ValueError(
                "reference image URL must use http:// or https:// without credentials or a fragment"
            )

    def _save_new_image(self, payload: bytes, mime_type: str) -> dict[str, object]:
        if not payload:
            raise ValueError("reference image is empty")
        if len(payload) > self.max_image_bytes:
            raise ValueError(
                f"reference image exceeds {self.max_image_bytes} bytes"
            )
        image_format = _IMAGE_FORMATS.get(mime_type)
        if image_format is None:
            raise ValueError(
                "reference image must be image/png, image/jpeg, or image/webp"
            )
        extension, magic, expected_format = image_format
        if not payload.startswith(magic):
            raise ValueError("reference image content does not match mime_type")
        if mime_type == "image/webp" and payload[8:12] != b"WEBP":
            raise ValueError("invalid WebP reference image")
        self._validate_decodable_image(payload, expected_format)
        asset_id = f"img_{secrets.token_hex(12)}"
        path = self.asset_dir / f"{asset_id}{extension}"
        path.write_bytes(payload)
        return {
            "asset_id": asset_id,
            "mime_type": mime_type,
            "size_bytes": len(payload),
        }

    @staticmethod
    def _validate_decodable_image(payload: bytes, expected_format: str) -> None:
        try:
            with Image.open(io.BytesIO(payload)) as image:
                actual_format = image.format
                image.verify()
            with Image.open(io.BytesIO(payload)) as image:
                image.load()
        except (OSError, SyntaxError, ValueError) as exc:
            raise ValueError("reference image is truncated or invalid") from exc
        if actual_format != expected_format:
            raise ValueError("reference image content does not match mime_type")

    async def resolve_image(self, asset_id: str | None) -> Path | None:
        if asset_id is None:
            return None
        if not _ASSET_ID_RE.fullmatch(asset_id):
            raise ValueError("invalid reference image asset_id")
        path = self._find_asset(asset_id)
        if path is None:
            raise ValueError(f"reference image asset not found: {asset_id}")
        return path

    def _find_asset(self, asset_id: str) -> Path | None:
        matches = list(self.asset_dir.glob(f"{asset_id}.*"))
        if len(matches) != 1:
            return None
        path = matches[0].resolve()
        if self.asset_dir not in path.parents or not path.is_file():
            return None
        return path
