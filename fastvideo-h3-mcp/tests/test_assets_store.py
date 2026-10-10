from __future__ import annotations

import base64
import asyncio
import io
import random
import socket
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from aiohttp import web
from PIL import Image

from fastvideo_h3_mcp.assets import AssetStore
from fastvideo_h3_mcp.models import GenerationRequest, JobRecord
from fastvideo_h3_mcp.store import JobStore


class AssetStoreTests(unittest.TestCase):
    def test_png_round_trip_and_id_validation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = AssetStore(Path(directory), 1024)
            png = _make_image("PNG")
            saved = store.save_image_base64(
                base64.b64encode(png).decode("ascii"), "image/png"
            )
            resolved = asyncio.run(store.resolve_image(str(saved["asset_id"])))
            self.assertIsNotNone(resolved)
            assert resolved is not None
            self.assertEqual(resolved.read_bytes(), png)
            with self.assertRaises(ValueError):
                asyncio.run(store.resolve_image("../../etc/passwd"))

    def test_rejects_mime_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = AssetStore(Path(directory), 1024)
            payload = base64.b64encode(b"not-a-png").decode("ascii")
            with self.assertRaises(ValueError):
                store.save_image_base64(payload, "image/png")

    def test_accepts_complete_png_jpeg_and_webp_images(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = AssetStore(Path(directory), 64 * 1024)
            for image_format, mime_type in (
                ("PNG", "image/png"),
                ("JPEG", "image/jpeg"),
                ("WEBP", "image/webp"),
            ):
                with self.subTest(image_format=image_format):
                    payload = _make_image(image_format)
                    saved = store.save_image_base64(
                        base64.b64encode(payload).decode("ascii"), mime_type
                    )
                    path = asyncio.run(
                        store.resolve_image(str(saved["asset_id"]))
                    )
                    assert path is not None
                    self.assertEqual(path.read_bytes(), payload)

    def test_rejects_truncated_png_jpeg_and_webp_images(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = AssetStore(Path(directory), 64 * 1024)
            for image_format, mime_type in (
                ("PNG", "image/png"),
                ("JPEG", "image/jpeg"),
                ("WEBP", "image/webp"),
            ):
                with self.subTest(image_format=image_format):
                    payload = _make_image(image_format)[:-16]
                    with self.assertRaisesRegex(
                        ValueError, "truncated or invalid"
                    ):
                        store.save_image_base64(
                            base64.b64encode(payload).decode("ascii"), mime_type
                        )


def _make_image(image_format: str, *, noisy: bool = False) -> bytes:
    if noisy:
        pixels = random.Random(0).randbytes(512 * 512 * 3)
        image = Image.frombytes("RGB", (512, 512), pixels)
    else:
        image = Image.new("RGB", (8, 8), (32, 96, 160))
    output = io.BytesIO()
    image.save(output, format=image_format, quality=95)
    return output.getvalue()


class _FakeContent:
    def __init__(self, payload: bytes, network_chunk_bytes: int):
        self.payload = payload
        self.network_chunk_bytes = network_chunk_bytes
        self.chunks_yielded = 0

    async def iter_chunked(self, limit: int):
        chunk_size = min(limit, self.network_chunk_bytes)
        for offset in range(0, len(self.payload), chunk_size):
            self.chunks_yielded += 1
            yield self.payload[offset : offset + chunk_size]


class _FakeResponse:
    def __init__(
        self,
        status: int,
        payload: bytes,
        mime_type: str,
        *,
        declare_content_length: bool = True,
        network_chunk_bytes: int = 64 * 1024,
    ):
        self.status = status
        self.content_length = len(payload) if declare_content_length else None
        self.content = _FakeContent(payload, network_chunk_bytes)
        self.headers = {"Content-Type": mime_type}

    async def __aenter__(self) -> "_FakeResponse":
        return self

    async def __aexit__(self, *_args: object) -> None:
        return None


class _FakeSession:
    def __init__(self, response: _FakeResponse):
        self.response = response

    async def __aenter__(self) -> "_FakeSession":
        return self

    async def __aexit__(self, *_args: object) -> None:
        return None

    def get(self, _url: str, *, allow_redirects: bool) -> _FakeResponse:
        assert not allow_redirects
        return self.response


class AssetStoreHttpTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.store = AssetStore(Path(self.temporary_directory.name), 1024)

    async def asyncTearDown(self) -> None:
        self.temporary_directory.cleanup()

    async def test_fetches_http_image_and_returns_local_asset_id(self) -> None:
        payload = _make_image("PNG")
        response = _FakeResponse(200, payload, "image/png")
        with patch(
            "fastvideo_h3_mcp.assets.aiohttp.ClientSession",
            return_value=_FakeSession(response),
        ):
            saved = await self.store.fetch_image_url(
                "http://127.0.0.1:8787/image?expires=1&signature=test"
            )
        asset_id = str(saved["asset_id"])
        self.assertRegex(asset_id, r"^img_[a-f0-9]{24}$")
        path = await self.store.resolve_image(asset_id)
        self.assertIsNotNone(path)
        assert path is not None
        self.assertEqual(path.read_bytes(), payload)

    async def test_fetches_complete_image_across_many_network_chunks(self) -> None:
        payload = _make_image("PNG", noisy=True)
        self.assertGreater(len(payload), 127_836)
        self.store = AssetStore(
            Path(self.temporary_directory.name), len(payload) + 1024
        )
        response = _FakeResponse(
            200,
            payload,
            "image/png",
            network_chunk_bytes=127_836,
        )
        with patch(
            "fastvideo_h3_mcp.assets.aiohttp.ClientSession",
            return_value=_FakeSession(response),
        ):
            saved = await self.store.fetch_image_url(
                "http://127.0.0.1:8787/v1/attachment-assets/test"
            )
        path = await self.store.resolve_image(str(saved["asset_id"]))
        assert path is not None
        self.assertGreater(response.content.chunks_yielded, 1)
        self.assertEqual(path.read_bytes(), payload)
        self.assertEqual(saved["size_bytes"], len(payload))

    async def test_fetches_complete_jpeg_from_real_chunked_http_server(self) -> None:
        payload = _make_image("JPEG", noisy=True)
        self.assertGreater(len(payload), 127_836)
        self.store = AssetStore(
            Path(self.temporary_directory.name), len(payload) + 1024
        )

        async def serve_image(request: web.Request) -> web.StreamResponse:
            response = web.StreamResponse(
                status=200, headers={"Content-Type": "image/jpeg"}
            )
            await response.prepare(request)
            for offset in range(0, len(payload), 127_836):
                await response.write(payload[offset : offset + 127_836])
            await response.write_eof()
            return response

        app = web.Application()
        app.router.add_get("/large.jpg", serve_image)
        runner = web.AppRunner(app)
        await runner.setup()
        listener = socket.socket()
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
        site = web.SockSite(runner, listener)
        await site.start()
        try:
            saved = await self.store.fetch_image_url(
                f"http://127.0.0.1:{port}/large.jpg"
            )
        finally:
            await runner.cleanup()

        path = await self.store.resolve_image(str(saved["asset_id"]))
        assert path is not None
        self.assertEqual(path.read_bytes(), payload)
        self.assertEqual(saved["size_bytes"], len(payload))

    async def test_rejects_stream_that_exceeds_limit_without_content_length(self) -> None:
        payload = _make_image("PNG", noisy=True)
        response = _FakeResponse(
            200,
            payload,
            "image/png",
            declare_content_length=False,
            network_chunk_bytes=256,
        )
        with patch(
            "fastvideo_h3_mcp.assets.aiohttp.ClientSession",
            return_value=_FakeSession(response),
        ):
            with self.assertRaisesRegex(ValueError, "exceeds 1024 bytes"):
                await self.store.fetch_image_url(
                    "https://attachments.example/oversized-image"
                )
        self.assertLess(response.content.chunks_yielded * 256, len(payload))

    async def test_rejects_non_http_urls_and_redirects(self) -> None:
        with self.assertRaisesRegex(ValueError, "must use http:// or https://"):
            await self.store.fetch_image_url("file:///tmp/first.png")
        response = _FakeResponse(302, b"", "text/plain")
        with patch(
            "fastvideo_h3_mcp.assets.aiohttp.ClientSession",
            return_value=_FakeSession(response),
        ):
            with self.assertRaisesRegex(ValueError, "HTTP 302"):
                await self.store.fetch_image_url(
                    "https://attachments.example/redirect"
                )


class JobStoreTests(unittest.TestCase):
    def test_job_round_trip_preserves_generation_fields(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = JobStore(Path(directory) / "jobs.sqlite3")
            request = GenerationRequest(
                prompt="A boat at dawn.",
                negative_prompt="No text.",
                soundscape="Waves.",
                prompt_mode="simple",
                model_id="base",
                duration_seconds=5,
                seed=42,
                audio_enabled=False,
                aspect_ratio="9:16",
                resolution="768p",
                first_frame_asset_id="img_000000000000000000000000",
                first_frame_path="/tmp/first.png",
                width=768,
                height=1344,
            )
            created = JobRecord.new(request)
            store.create(created)
            loaded = store.require(created.id)
            self.assertEqual(loaded.prompt, request.prompt)
            self.assertEqual(loaded.aspect_ratio, "9:16")
            self.assertFalse(loaded.audio_enabled)
            self.assertEqual(loaded.first_frame_asset_id, request.first_frame_asset_id)
            self.assertNotIn("first_frame_path", loaded.as_public_dict())
            store.close()


if __name__ == "__main__":
    unittest.main()
