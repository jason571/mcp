from __future__ import annotations

import asyncio
import inspect
import json
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any, get_type_hints
from unittest.mock import AsyncMock, patch

from minimax_h3_mcp.models import MODEL_ID, JobStatus
from minimax_h3_mcp.server import (
    MCPBearerAuthMiddleware,
    _make_request,
    _public_job,
    _tracked_job,
    cancel_video_job,
    generate_video,
    get_video_job,
    get_video_result,
    mcp,
    submit_video_generation,
)


def _schema_nodes(field: dict[str, Any]) -> list[dict[str, Any]]:
    nodes = [field]
    for key in ("anyOf", "oneOf", "allOf"):
        items = field.get(key)
        if isinstance(items, list):
            nodes.extend(item for item in items if isinstance(item, dict))
    return nodes


def _numeric_schema_default(field: dict[str, Any]) -> float | None:
    for node in _schema_nodes(field):
        value = node.get("default")
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
            continue
        return float(value)
    return None


def _tool_fields(tool: Any) -> set[str]:
    schema = getattr(tool, "input_schema", None)
    if not isinstance(schema, dict):
        return set()
    props = schema.get("properties")
    return set(props) if isinstance(props, dict) else set()


def _description(tool: Any) -> str:
    return " ".join(str(getattr(tool, "description", "") or "").lower().split())


def job(status: str = JobStatus.COMPLETED):
    completed = status == JobStatus.COMPLETED
    return SimpleNamespace(
        id="job-contract",
        status=status,
        progress=100.0 if completed else 42.0,
        stage="completed" if completed else "rendering",
        message="service progress",
        terminal=completed,
        output_file=lambda: Path("/tmp/result.mp4") if completed else None,
        as_public_dict=lambda: {"id": "job-contract", "status": status, "error": None},
    )


def runtime(value):
    return SimpleNamespace(
        settings=SimpleNamespace(
            artifact_signing_key="fixture", artifact_url_ttl_seconds=3600
        ),
        signer=SimpleNamespace(
            build_url=lambda *_: "https://media.test/result.mp4?signature=fixture"
        ),
        manager=SimpleNamespace(
            get=lambda _: value,
            cancel=AsyncMock(return_value=value),
            submit=AsyncMock(return_value=(value, 0)),
        ),
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
            "server": ("127.0.0.1", 8766),
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

    async def test_valid_token_reaches_mcp_app(self) -> None:
        called, messages = await self._request("/mcp", "Bearer test-token")
        self.assertTrue(called)
        self.assertEqual(messages[0]["status"], 204)

    async def test_health_route_remains_public(self) -> None:
        called, messages = await self._request("/health", None)
        self.assertTrue(called)
        self.assertEqual(messages[0]["status"], 204)


class LiveToolContractTests(unittest.IsolatedAsyncioTestCase):
    async def test_generation_tools_publish_numeric_duration_default(self) -> None:
        tools = {tool.name: tool for tool in await mcp.list_tools()}
        for name in ("generate_video", "submit_video_generation"):
            with self.subTest(tool=name):
                schema = tools[name].input_schema
                duration = schema["properties"]["duration_seconds"]
                self.assertEqual(_numeric_schema_default(duration), 15.0)
                self.assertEqual(
                    next(
                        node.get("minimum")
                        for node in _schema_nodes(duration)
                        if node.get("minimum") is not None
                    ),
                    5.0,
                )
                self.assertEqual(
                    next(
                        node.get("maximum")
                        for node in _schema_nodes(duration)
                        if node.get("maximum") is not None
                    ),
                    60.0,
                )
                self.assertNotIn("duration_seconds", schema.get("required", []))

    async def test_tools_match_music_video_planner_descriptions(self) -> None:
        tools = {tool.name: tool for tool in await mcp.list_tools()}
        required = {
            "generate_video",
            "submit_video_generation",
            "get_video_job",
            "get_video_result",
            "cancel_video_job",
            "list_video_models",
            "get_video_service_status",
        }
        self.assertTrue(required <= set(tools))

        submit = tools["submit_video_generation"]
        self.assertTrue({"prompt", "duration_seconds"} <= _tool_fields(submit))
        submit_text = _description(submit)
        self.assertTrue(any(word in submit_text for word in ("video", "clip", "视频")))
        self.assertTrue(
            any(
                word in submit_text
                for word in ("generat", "submit", "create", "生成", "提交")
            )
        )

        status = tools["get_video_job"]
        self.assertIn("job_id", _tool_fields(status))
        status_text = _description(status)
        self.assertTrue(any(word in status_text for word in ("job", "video", "任务", "视频")))
        self.assertTrue(
            any(
                word in status_text
                for word in ("status", "progress", "poll", "状态", "进度")
            )
        )

        result = tools["get_video_result"]
        self.assertIn("job_id", _tool_fields(result))
        result_text = _description(result)
        self.assertTrue(any(word in result_text for word in ("job", "video", "任务", "视频")))
        self.assertTrue(
            any(
                word in result_text
                for word in ("result", "output", "artifact", "download", "结果", "成片")
            )
        )

        cancel = tools["cancel_video_job"]
        self.assertIn("job_id", _tool_fields(cancel))
        self.assertTrue(
            any(word in _description(cancel) for word in ("cancel", "abort", "取消"))
        )

        models_text = _description(tools["list_video_models"])
        status_service_text = _description(tools["get_video_service_status"])
        self.assertTrue("list" in models_text and "model" in models_text)
        self.assertTrue("default" in models_text and "duration" in models_text)
        self.assertTrue(
            "service" in status_service_text and "status" in status_service_text
        )

    async def test_live_schema_declares_optional_presentation(self) -> None:
        tools = {tool.name: tool for tool in await mcp.list_tools()}
        for name in (
            "generate_video",
            "submit_video_generation",
            "get_video_job",
            "get_video_result",
            "cancel_video_job",
        ):
            schema = tools[name].input_schema
            self.assertEqual(
                schema["properties"]["artifact_visibility"]["enum"],
                ["chat", "internal"],
            )
            self.assertEqual(
                schema["properties"]["artifact_visibility"]["default"], "chat"
            )
            self.assertNotIn("artifact_visibility", schema.get("required", []))

    async def test_reference_image_is_a_separate_optional_input(self) -> None:
        tools = {tool.name: tool for tool in await mcp.list_tools()}
        for name in ("generate_video", "submit_video_generation"):
            schema = tools[name].input_schema
            self.assertIn("reference_image_url", schema["properties"])
            self.assertIn("reference_image_asset_id", schema["properties"])
            self.assertNotIn("reference_image_url", schema.get("required", []))
            self.assertIn("first_frame_url", schema["properties"])


class PythonSignatureTests(unittest.TestCase):
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


class JobEnvelopeTests(unittest.IsolatedAsyncioTestCase):
    def test_public_job_exposes_documented_job_id(self) -> None:
        value = job(JobStatus.RUNNING)
        public = _public_job(runtime(value), value)
        self.assertEqual(public["id"], value.id)
        self.assertEqual(public["job_id"], value.id)

    def test_completed_job_exposes_result_download_url(self) -> None:
        value = job()
        rt = runtime(value)
        ordinary = _public_job(rt, value)
        self.assertIn("download_url", ordinary["result"])
        self.assertNotIn("result_url", ordinary)
        self.assertNotIn("_meta", ordinary)

    def test_tracked_job_uses_exoharness_background_task(self) -> None:
        value = job(JobStatus.RUNNING)
        tagged = _tracked_job(runtime(value), value)
        task = tagged["_meta"]["exoharness"]["background_task"]
        self.assertEqual(task["status"], "working")
        self.assertEqual(
            task["poll"],
            {
                "tool": "get_video_job",
                "arguments": {"job_id": "job-contract", "managed_tracking": True},
            },
        )
        self.assertEqual(
            task["cancel"],
            {
                "tool": "cancel_video_job",
                "arguments": {"job_id": "job-contract", "managed_tracking": True},
            },
        )

    def test_internal_keeps_download_url_and_tags_visibility(self) -> None:
        value = job()
        rt = runtime(value)
        ordinary = _public_job(rt, value)
        internal = _public_job(rt, value, "internal")
        self.assertEqual(internal["result"], ordinary["result"])
        self.assertEqual(internal["_meta"]["exoharness"]["artifact_visibility"], "internal")
        tagged = _tracked_job(rt, value, "internal")
        self.assertEqual(
            tagged["_meta"]["exoharness"]["background_task"]["poll"]["arguments"][
                "artifact_visibility"
            ],
            "internal",
        )

    async def test_detached_poll_and_cancel_preserve_visibility(self) -> None:
        value = job()
        rt = runtime(value)
        with (
            patch(
                "minimax_h3_mcp.server._require_tool_auth", return_value=rt
            ),
            patch(
                "minimax_h3_mcp.server._make_request",
                new=AsyncMock(return_value=object()),
            ),
            patch("minimax_h3_mcp.server.JobRecord.new", return_value=value),
        ):
            submitted = await submit_video_generation(
                "original prompt", None, artifact_visibility="internal"
            )
            control = json.loads(
                json.dumps(submitted["_meta"]["exoharness"]["background_task"])
            )
            for action, tool in (
                ("poll", get_video_job),
                ("cancel", cancel_video_job),
            ):
                result = await tool(ctx=None, **control[action]["arguments"])
                self.assertEqual(result["result"], submitted["result"])
                self.assertEqual(
                    result["_meta"]["exoharness"]["artifact_visibility"], "internal"
                )
            internal = await get_video_result(
                value.id, None, artifact_visibility="internal"
            )
            public = await get_video_result(value.id, None)
        self.assertEqual(internal["result"], public["result"])
        self.assertNotIn("_meta", public)


class RequestNormalizationTests(unittest.TestCase):
    def _request(
        self,
        model_id: str,
        first_frame_asset_id: str | None = None,
        last_frame_asset_id: str | None = None,
        first_frame_url: str | None = None,
        last_frame_url: str | None = None,
        reference_image_url: str | None = None,
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
        runtime_value = SimpleNamespace(
            settings=SimpleNamespace(
                model_id=MODEL_ID,
                default_duration_seconds=15.0,
                max_duration_seconds=60.0,
            ),
            assets=assets,
        )
        return asyncio.run(
            _make_request(
                runtime_value,  # type: ignore[arg-type]
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
                reference_image_url=reference_image_url,
            )
        )

    def test_text_and_reference_frames_share_the_8step_model(self) -> None:
        text = self._request("auto")
        self.assertEqual(text.model_id, MODEL_ID)  # type: ignore[attr-defined]
        framed = self._request("fast", first_frame_asset_id="asset-id")
        self.assertEqual(framed.model_id, MODEL_ID)  # type: ignore[attr-defined]

    def test_reference_image_does_not_become_first_frame(self) -> None:
        request = self._request(
            "auto", reference_image_url="https://example.test/reference.png"
        )
        self.assertIsNone(request.first_frame_asset_id)  # type: ignore[attr-defined]
        self.assertIsNone(request.first_frame_path)  # type: ignore[attr-defined]
        self.assertEqual(request.reference_image_asset_id, "img_" + "2" * 24)  # type: ignore[attr-defined]

    def test_omitted_duration_uses_startup_default(self) -> None:
        request = self._request("auto", duration_seconds=None)
        self.assertEqual(request.duration_seconds, 15.0)  # type: ignore[attr-defined]

    def test_sixty_second_request_is_accepted(self) -> None:
        request = self._request("auto", duration_seconds=60)
        self.assertEqual(request.num_frames, 1450)  # type: ignore[attr-defined]

    def test_over_fifteen_seconds_normalizes_high_resolution(self) -> None:
        request = self._request(
            "auto", duration_seconds=15.01, resolution="768p"
        )
        self.assertEqual(request.resolution, "512p")  # type: ignore[attr-defined]


if __name__ == "__main__":
    unittest.main()
