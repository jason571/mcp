from __future__ import annotations

import inspect
import unittest
from pathlib import Path
import asyncio
from types import SimpleNamespace
from typing import Any, get_type_hints

from fastvideo_h3_mcp.server import (
    MCPBearerAuthMiddleware,
    _make_request,
    generate_video,
    submit_video_generation,
)


class MCPBearerAuthMiddlewareTests(unittest.IsolatedAsyncioTestCase):
    async def _request(
        self, path: str, authorization: str | None
    ) -> tuple[bool, list[dict[str, Any]]]:
        called = False

        async def app(_scope: object, _receive: object, send: object) -> None:
            nonlocal called
            called = True
            await send({"type": "http.response.start", "status": 204, "headers": []})  # type: ignore[operator]
            await send({"type": "http.response.body", "body": b""})  # type: ignore[operator]

        middleware = MCPBearerAuthMiddleware(app, "test-token", "/mcp")
        headers = []
        if authorization is not None:
            headers.append((b"authorization", authorization.encode("ascii")))
        scope = {
            "type": "http",
            "asgi": {"version": "3.0"},
            "http_version": "1.1",
            "method": "POST",
            "scheme": "http",
            "path": path,
            "raw_path": path.encode("ascii"),
            "query_string": b"",
            "headers": headers,
            "client": ("127.0.0.1", 1234),
            "server": ("127.0.0.1", 8765),
        }
        messages: list[dict[str, Any]] = []

        async def receive() -> dict[str, Any]:
            return {"type": "http.request", "body": b"", "more_body": False}

        async def send(message: dict[str, Any]) -> None:
            messages.append(message)

        await middleware(scope, receive, send)  # type: ignore[arg-type]
        return called, messages

    async def test_missing_token_is_rejected_before_mcp_app(self) -> None:
        called, messages = await self._request("/mcp", None)
        self.assertFalse(called)
        self.assertEqual(messages[0]["status"], 401)
        self.assertIn(
            (b"www-authenticate", b"Bearer"), messages[0]["headers"]
        )

    async def test_invalid_token_is_rejected(self) -> None:
        called, messages = await self._request("/mcp", "Bearer wrong")
        self.assertFalse(called)
        self.assertEqual(messages[0]["status"], 401)

    async def test_valid_token_reaches_mcp_app(self) -> None:
        called, messages = await self._request("/mcp", "Bearer test-token")
        self.assertTrue(called)
        self.assertEqual(messages[0]["status"], 204)

    async def test_health_route_remains_public(self) -> None:
        called, messages = await self._request("/health", None)
        self.assertTrue(called)
        self.assertEqual(messages[0]["status"], 204)


class ModelRoutingNormalizationTests(unittest.TestCase):
    def test_generation_tool_defaults_are_numeric_duration_and_512p(self) -> None:
        for tool in (generate_video, submit_video_generation):
            with self.subTest(tool=tool.__name__):
                parameters = inspect.signature(tool).parameters
                default = parameters["duration_seconds"].default
                self.assertIsInstance(default, float)
                self.assertEqual(default, 15.0)
                self.assertEqual(parameters["resolution"].default, "512p")

    def test_reference_frame_fields_have_single_call_schema_descriptions(self) -> None:
        for tool in (generate_video, submit_video_generation):
            with self.subTest(tool=tool.__name__):
                parameters = inspect.signature(tool).parameters
                annotations = get_type_hints(tool, include_extras=True)
                for field_name, peer_name in (
                    ("first_frame_url", "last_frame_url"),
                    ("last_frame_url", "first_frame_url"),
                ):
                    self.assertIsNone(parameters[field_name].default)
                    metadata = annotations[field_name].__metadata__
                    description = metadata[0].description
                    self.assertIn(peer_name, description)
                    self.assertIn("same generation call", description)
                    self.assertIn("http://", description)

    def _request(
        self,
        model_id: str,
        first_frame_asset_id: str | None = None,
        last_frame_asset_id: str | None = None,
        first_frame_url: str | None = None,
        last_frame_url: str | None = None,
        duration_seconds: float | None = 5,
        aspect_ratio: str = "16:9",
        resolution: str = "512p",
    ) -> object:
        async def resolve_image(asset_id: str | None) -> Path | None:
            return Path("/tmp/first.png") if asset_id is not None else None

        async def fetch_image_url(url: str) -> dict[str, object]:
            suffix = "1" if "first" in url else "2"
            return {"asset_id": f"img_{suffix * 24}"}

        assets = SimpleNamespace(
            resolve_image=resolve_image,
            fetch_image_url=fetch_image_url,
        )
        runtime = SimpleNamespace(
            settings=SimpleNamespace(
                model_id="fasth3-checkpoint",
                default_duration_seconds=15.0,
                max_duration_seconds=60.0,
            ),
            assets=assets,
        )
        return asyncio.run(
            _make_request(
                runtime,  # type: ignore[arg-type]
                prompt="A test scene",
                negative_prompt=None,
                soundscape=None,
                prompt_mode="simple",
                duration_seconds=duration_seconds,
                seed=1,
                audio_enabled=True,
                aspect_ratio=aspect_ratio,
                resolution=resolution,
                first_frame_asset_id=first_frame_asset_id,
                last_frame_asset_id=last_frame_asset_id,
                model_id=model_id,
                first_frame_url=first_frame_url,
                last_frame_url=last_frame_url,
            )
        )

    def test_text_only_base_hint_is_normalized_to_fast(self) -> None:
        request = self._request("base")
        self.assertEqual(request.model_id, "fast")  # type: ignore[attr-defined]

    def test_reference_frame_is_normalized_to_base(self) -> None:
        request = self._request("fast", first_frame_asset_id="asset-id")
        self.assertEqual(request.model_id, "base")  # type: ignore[attr-defined]

    def test_first_and_last_frames_share_one_generation_request(self) -> None:
        request = self._request(
            "auto",
            first_frame_asset_id="first-asset",
            last_frame_asset_id="last-asset",
        )
        self.assertEqual(request.model_id, "base")  # type: ignore[attr-defined]
        self.assertEqual(  # type: ignore[attr-defined]
            request.first_frame_asset_id, "first-asset"
        )
        self.assertEqual(  # type: ignore[attr-defined]
            request.last_frame_asset_id, "last-asset"
        )

    def test_first_and_last_http_urls_are_downloaded_in_one_request(self) -> None:
        request = self._request(
            "auto",
            first_frame_url="http://127.0.0.1:8787/first",
            last_frame_url="https://attachments.example/last?signature=test",
        )
        self.assertEqual(request.model_id, "base")  # type: ignore[attr-defined]
        self.assertEqual(  # type: ignore[attr-defined]
            request.first_frame_asset_id, "img_111111111111111111111111"
        )
        self.assertEqual(  # type: ignore[attr-defined]
            request.last_frame_asset_id, "img_222222222222222222222222"
        )

    def test_rejects_asset_id_and_url_for_same_frame(self) -> None:
        with self.assertRaisesRegex(
            ValueError, "either first_frame_asset_id or first_frame_url"
        ):
            self._request(
                "auto",
                first_frame_asset_id="img_111111111111111111111111",
                first_frame_url="http://127.0.0.1:8787/first",
            )

    def test_sixty_second_request_is_accepted(self) -> None:
        request = self._request("auto", duration_seconds=60)
        self.assertEqual(request.num_frames, 1450)  # type: ignore[attr-defined]
        self.assertEqual(request.actual_duration_seconds, 1450 / 24)  # type: ignore[attr-defined]

    def test_omitted_duration_uses_startup_default(self) -> None:
        request = self._request("auto", duration_seconds=None)
        self.assertEqual(request.duration_seconds, 15.0)  # type: ignore[attr-defined]

    def test_configured_max_duration_is_enforced(self) -> None:
        async def resolve_image(asset_id: str | None) -> Path | None:
            return None

        runtime = SimpleNamespace(
            settings=SimpleNamespace(
                model_id="fasth3-checkpoint",
                default_duration_seconds=15.0,
                max_duration_seconds=30.0,
            ),
            assets=SimpleNamespace(resolve_image=resolve_image, fetch_image_url=None),
        )
        with self.assertRaisesRegex(ValueError, "between 5 and 30"):
            asyncio.run(
                _make_request(
                    runtime,  # type: ignore[arg-type]
                    prompt="A test scene",
                    negative_prompt=None,
                    soundscape=None,
                    prompt_mode="simple",
                    duration_seconds=60,
                    seed=1,
                    audio_enabled=True,
                    aspect_ratio="16:9",
                    resolution="512p",
                    first_frame_asset_id=None,
                    last_frame_asset_id=None,
                    model_id="auto",
                )
            )

    def test_fifteen_second_high_resolution_is_accepted_for_both_ratios(self) -> None:
        for first_frame_asset_id in (None, "asset-id"):
            for aspect_ratio in ("16:9", "9:16"):
                for resolution in ("640p", "768p"):
                    with self.subTest(
                        first_frame_asset_id=first_frame_asset_id,
                        aspect_ratio=aspect_ratio,
                        resolution=resolution,
                    ):
                        request = self._request(
                            "auto",
                            first_frame_asset_id=first_frame_asset_id,
                            duration_seconds=15,
                            aspect_ratio=aspect_ratio,
                            resolution=resolution,
                        )
                        self.assertEqual(request.resolution, resolution)  # type: ignore[attr-defined]

    def test_over_fifteen_seconds_normalizes_high_resolution_for_both_ratios(self) -> None:
        for first_frame_asset_id in (None, "asset-id"):
            for aspect_ratio in ("16:9", "9:16"):
                for resolution in ("640p", "768p"):
                    with self.subTest(
                        first_frame_asset_id=first_frame_asset_id,
                        aspect_ratio=aspect_ratio,
                        resolution=resolution,
                    ):
                        request = self._request(
                            "auto",
                            first_frame_asset_id=first_frame_asset_id,
                            duration_seconds=15.01,
                            aspect_ratio=aspect_ratio,
                            resolution=resolution,
                        )
                        self.assertEqual(request.resolution, "512p")  # type: ignore[attr-defined]

    def test_unknown_resolution_normalizes_to_512p_for_both_backends(self) -> None:
        for first_frame_asset_id in (None, "asset-id"):
            for aspect_ratio in ("16:9", "9:16"):
                for resolution in ("1080p", "highest", ""):
                    with self.subTest(
                        first_frame_asset_id=first_frame_asset_id,
                        aspect_ratio=aspect_ratio,
                        resolution=resolution,
                    ):
                        request = self._request(
                            "auto",
                            first_frame_asset_id=first_frame_asset_id,
                            duration_seconds=5,
                            aspect_ratio=aspect_ratio,
                            resolution=resolution,
                        )
                        self.assertEqual(request.resolution, "512p")  # type: ignore[attr-defined]
