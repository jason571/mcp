from __future__ import annotations

import json
from contextlib import asynccontextmanager
from dataclasses import is_dataclass, replace
from types import SimpleNamespace
from typing import Any, AsyncIterator, Mapping

import httpx2
from mcp import ClientSession, types
from mcp.client.streamable_http import streamable_http_client

from .config import Settings


class UpstreamToolError(RuntimeError):
    pass


class _WorkflowBodyFailure(BaseException):
    """Sentinel that survives AnyIO ExceptionGroup wrapping during cleanup."""

    def __init__(self, primary: BaseException):
        super().__init__(str(primary))
        self.primary = primary


def _workflow_body_failure(error: BaseException) -> _WorkflowBodyFailure | None:
    if isinstance(error, _WorkflowBodyFailure):
        return error
    if isinstance(error, BaseExceptionGroup):
        for nested in error.exceptions:
            found = _workflow_body_failure(nested)
            if found is not None:
                return found
    return None


def _numeric_duration_from_mcp(payload: Any) -> int | None:
    """Read a live numeric default from status/models payloads. Ignore description text."""
    if not isinstance(payload, dict):
        return None
    defaults = payload.get("defaults") if isinstance(payload.get("defaults"), dict) else {}
    candidates = (
        payload.get("default_duration_seconds"),
        payload.get("duration_seconds"),
        defaults.get("duration_seconds"),
        defaults.get("default_duration_seconds"),
    )
    for value in candidates:
        if isinstance(value, dict):
            value = value.get("default")
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
            continue
        return max(1, int(value))
    return None


def _tool_payload(result: types.CallToolResult) -> dict[str, Any]:
    if result.is_error:
        detail = ""
        for item in result.content:
            if isinstance(item, types.TextContent):
                detail = item.text
                break
        raise UpstreamToolError(detail or "upstream MCP tool returned an error")
    if isinstance(result.structured_content, dict):
        return result.structured_content
    for item in result.content:
        if not isinstance(item, types.TextContent):
            continue
        try:
            payload = json.loads(item.text)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    raise UpstreamToolError("upstream MCP tool returned no JSON object")


def _tool_fields(tool: Any) -> set[str]:
    schema = getattr(tool, "input_schema", None)
    if not isinstance(schema, dict):
        return set()
    props = schema.get("properties")
    return set(props) if isinstance(props, dict) else set()


def _tool_description(tool: Any) -> str:
    return " ".join(str(getattr(tool, "description", "") or "").lower().split())


def _description_score(
    tool: Any,
    *,
    fields: set[str],
    description_all_groups: tuple[tuple[str, ...], ...],
    description_none: tuple[str, ...] = (),
    preferred_name: str | None = None,
) -> int | None:
    if fields - _tool_fields(tool):
        return None
    text = _tool_description(tool)
    group_hits = [sum(1 for needle in group if needle in text) for group in description_all_groups]
    extra = len(
        _tool_fields(tool)
        & {"audio_enabled", "aspect_ratio", "resolution", "first_frame_url", "prompt_mode"}
    )
    none_hits = sum(1 for needle in description_none if needle in text)
    score = sum(group_hits) * 10 + extra - none_hits * 20
    if all(hits > 0 for hits in group_hits):
        score += 100
    if "preview" in text or "预览" in text:
        score -= 30
    if preferred_name and getattr(tool, "name", None) == preferred_name:
        score += 50
    return score


def _closest_described_tool(
    tools: list[Any],
    *,
    role: str,
    fields: set[str],
    description_all_groups: tuple[tuple[str, ...], ...],
    description_none: tuple[str, ...] = (),
    preferred_name: str | None = None,
) -> str:
    ranked: list[tuple[int, Any]] = []
    for tool in tools:
        score = _description_score(
            tool,
            fields=fields,
            description_all_groups=description_all_groups,
            description_none=description_none,
            preferred_name=preferred_name,
        )
        if score is None:
            continue
        ranked.append((score, tool))
    if not ranked:
        raise UpstreamToolError(
            f"video-generator MCP has no {role} tool whose description and schema match"
        )
    ranked.sort(key=lambda item: item[0], reverse=True)
    return ranked[0][1].name


def bind_generator_contract(tools: list[Any], preferred: Mapping[str, str]) -> dict[str, str]:
    """Bind submit/status/result/cancel from live descriptions, not guessed names."""
    specs = {
        "submit": {
            "fields": {"prompt", "duration_seconds"},
            "description_all_groups": (
                ("video", "clip", "视频"),
                ("generat", "submit", "create", "生成", "提交"),
            ),
            "description_none": ("cancel", "取消"),
        },
        "status": {
            "fields": {"job_id"},
            "description_all_groups": (
                ("job", "video", "任务", "视频"),
                ("status", "progress", "poll", "状态", "进度"),
            ),
            "description_none": ("cancel", "取消"),
        },
        "result": {
            "fields": {"job_id"},
            "description_all_groups": (
                ("job", "video", "任务", "视频"),
                ("result", "output", "artifact", "download", "结果", "成片"),
            ),
            "description_none": ("cancel", "取消"),
        },
        "cancel": {
            "fields": {"job_id"},
            "description_all_groups": (("cancel", "abort", "取消"),),
            "description_none": (),
        },
    }
    by_name = {tool.name: tool for tool in tools if getattr(tool, "name", None)}
    bound: dict[str, str] = {}
    for role, spec in specs.items():
        try:
            bound[role] = _closest_described_tool(
                tools,
                role=role,
                preferred_name=preferred.get(role),
                **spec,
            )
        except UpstreamToolError:
            name = preferred.get(role)
            if name in by_name:
                bound[role] = name
            else:
                raise
    return bound


def duration_query_tools(tools: list[Any]) -> list[str]:
    names: list[str] = []
    for tool in tools:
        text = _tool_description(tool)
        if any(
            all(part in text for part in pair)
            for pair in (
                ("default", "duration"),
                ("service", "status"),
                ("list", "model"),
                ("默认", "时长"),
                ("服务", "状态"),
            )
        ):
            names.append(tool.name)
    return names


def _split_generator_urls(raw: str) -> list[str]:
    return [item.strip() for item in str(raw).split(",") if item.strip()]


def generator_urls(settings: Any, request_headers: Mapping[str, str] | None = None) -> list[str]:
    ordered: list[str] = []
    if request_headers:
        for key, value in request_headers.items():
            if str(key).lower() == "x-exoharness-video-generator-url" and str(value).strip():
                ordered.extend(_split_generator_urls(str(value)))
    raw = getattr(settings, "video_generator_url", None) or ""
    ordered.extend(_split_generator_urls(str(raw)))
    unique: list[str] = []
    seen: set[str] = set()
    for url in ordered:
        if url not in seen:
            seen.add(url)
            unique.append(url)
    return unique


class VideoGeneratorClient:
    """Typed adapter for a detached video-generator MCP contract.

    Polling stays inside this service. The calling Agent sees one MCP invocation
    and receives ordinary progress notifications instead of spending a model
    turn for every status check.
    """

    def __init__(self, settings: Settings, session: ClientSession):
        self.settings = settings
        self.session = session
        self.submit_tool = settings.video_generator_submit_tool
        self.status_tool = settings.video_generator_status_tool
        self.result_tool = settings.video_generator_result_tool
        self.cancel_tool = settings.video_generator_cancel_tool
        self._listed_tools: list[Any] = []

    async def _list_all_tools(self) -> list[Any]:
        tools: list[Any] = []
        cursor = None
        while True:
            result = await self.session.list_tools(
                params=types.PaginatedRequestParams(cursor=cursor) if cursor else None
            )
            tools.extend(result.tools)
            cursor = result.next_cursor
            if not cursor:
                return tools

    async def validate_contract(self) -> None:
        tools = await self._list_all_tools()
        self._listed_tools = tools
        bound = bind_generator_contract(
            tools,
            {
                "submit": self.settings.video_generator_submit_tool,
                "status": self.settings.video_generator_status_tool,
                "result": self.settings.video_generator_result_tool,
                "cancel": self.settings.video_generator_cancel_tool,
            },
        )
        self.submit_tool = bound["submit"]
        self.status_tool = bound["status"]
        self.result_tool = bound["result"]
        self.cancel_tool = bound["cancel"]

    async def submission_schema(self) -> dict[str, Any]:
        """Read the actual provider contract; never invent model-specific controls."""
        tools = self._listed_tools or await self._list_all_tools()
        for tool in tools:
            if tool.name == self.submit_tool:
                return tool.input_schema
        raise UpstreamToolError("configured video submit tool is unavailable")

    async def live_default_duration(self) -> int | None:
        """Numeric default from generator status/models tools. Ignore schema description text."""
        tools = self._listed_tools or await self._list_all_tools()
        for tool_name in duration_query_tools(tools):
            try:
                payload = await self.call(tool_name, {}, "")
            except UpstreamToolError:
                continue
            value = _numeric_duration_from_mcp(payload)
            if value is not None:
                return value
        return None

    async def call(
        self, tool_name: str, arguments: dict[str, Any], _request_state: str
    ) -> dict[str, Any]:
        try:
            result = await self.session.call_tool(
                tool_name,
                arguments,
                read_timeout_seconds=float(self.settings.operation_timeout_seconds),
            )
        except Exception as exc:
            raise UpstreamToolError(
                f"video-generator tool {tool_name!r} transport failed: {exc}"
            ) from exc
        if not isinstance(result, types.CallToolResult):
            raise UpstreamToolError("upstream MCP tool did not complete")
        return _tool_payload(result)

    async def submit(
        self, arguments: dict[str, Any], request_state: str
    ) -> dict[str, Any]:
        return await self.call(self.submit_tool, arguments, request_state)

    async def status(self, job_id: str, request_state: str) -> dict[str, Any]:
        return await self.call(self.status_tool, {"job_id": job_id}, request_state)

    async def result(self, job_id: str, request_state: str) -> dict[str, Any]:
        return await self.call(self.result_tool, {"job_id": job_id}, request_state)

    async def cancel(self, job_id: str, request_state: str) -> dict[str, Any]:
        # The workflow persists this receipt (or failure) instead of claiming
        # the child stopped when the upstream cancellation could not be sent.
        return await self.call(self.cancel_tool, {"job_id": job_id}, request_state)


def _settings_with_url(settings: Any, url: str) -> Any:
    try:
        if is_dataclass(settings):
            return replace(settings, video_generator_url=url)
    except TypeError:
        pass
    payload = dict(getattr(settings, "__dict__", {}))
    payload["video_generator_url"] = url
    return SimpleNamespace(**payload)


@asynccontextmanager
async def _connect_generator_url(settings: Any) -> AsyncIterator[VideoGeneratorClient]:
    headers: dict[str, str] = {}
    if getattr(settings, "video_generator_authorization", None):
        headers["Authorization"] = settings.video_generator_authorization
    # ClientSession's deadline does not configure HTTP socket reads. Use the
    # same operation policy at both layers, otherwise httpx2 ends a legitimate
    # quiet SSE response after its default five seconds. Keep a finite timeout.
    async with httpx2.AsyncClient(
        headers=headers,
        timeout=httpx2.Timeout(float(settings.operation_timeout_seconds), connect=2.0),
    ) as http_client:
        async with streamable_http_client(
            settings.video_generator_url, http_client=http_client
        ) as streams:
            read_stream, write_stream = streams
            async with ClientSession(
                read_stream,
                write_stream,
                read_timeout_seconds=float(settings.operation_timeout_seconds),
            ) as session:
                await session.initialize()
                client = VideoGeneratorClient(settings, session)
                await client.validate_contract()
                yield client


@asynccontextmanager
async def connect_video_generator(
    settings: Settings,
    request_headers: Mapping[str, str] | None = None,
    *,
    required_submit_fields: tuple[str, ...] = (),
) -> AsyncIterator[VideoGeneratorClient]:
    urls = generator_urls(settings, request_headers)
    if not urls:
        raise UpstreamToolError("MEDIA_MCP_VIDEO_GENERATOR_URL is not configured")
    errors: list[str] = []
    for url in urls:
        try:
            async with _connect_generator_url(
                _settings_with_url(settings, url)
            ) as client:
                if required_submit_fields:
                    schema = await client.submission_schema()
                    properties = schema.get("properties", {})
                    if not isinstance(properties, dict):
                        properties = {}
                    missing = [name for name in required_submit_fields if name not in properties]
                    if missing:
                        errors.append(
                            f"{url}: submit schema lacks {', '.join(missing)}"
                        )
                        continue
                try:
                    yield client
                except BaseException as primary:
                    # AnyIO transport cleanup can wrap an exception raised by
                    # the caller in a BaseExceptionGroup. Mark it so the outer
                    # discovery fallback never mistakes a workflow failure for
                    # a connection/contract failure.
                    raise _WorkflowBodyFailure(primary) from primary
                return
        except BaseException as exc:
            body_failure = _workflow_body_failure(exc)
            if body_failure is not None:
                raise body_failure.primary
            if not isinstance(exc, Exception):
                raise
            errors.append(f"{url}: {exc}")
    raise UpstreamToolError(
        "no video-generator MCP matched the live tool descriptions: " + "; ".join(errors)
    )
