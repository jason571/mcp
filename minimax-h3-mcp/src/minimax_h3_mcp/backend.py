from __future__ import annotations

import asyncio
import copy
import json
import os
import re
import signal
import uuid
from collections.abc import Awaitable, Callable
from pathlib import Path
from urllib.parse import urlencode, urlsplit, urlunsplit

import aiohttp

from .config import Settings
from .models import GenerationRequest


ProgressReporter = Callable[[float, str, str], Awaitable[None]]
_WORKFLOW_PLACEHOLDER_RE = re.compile(r"\{\{[A-Z0-9_]+\}\}")


class BackendCanceled(RuntimeError):
    pass


def render_workflow_template(
    template: object, replacements: dict[str, object]
) -> object:
    if isinstance(template, dict):
        return {
            key: render_workflow_template(value, replacements)
            for key, value in template.items()
        }
    if isinstance(template, list):
        return [render_workflow_template(value, replacements) for value in template]
    if isinstance(template, str):
        if template in replacements:
            return replacements[template]
        rendered = template
        for placeholder, value in replacements.items():
            rendered = rendered.replace(placeholder, str(value))
        return rendered
    return template


class ComfyUIBackend:
    """Run one MiniMax H3 8-step workflow for both text-to-video and image-to-video."""

    name = "comfyui-8step"
    _VIDEO_EXTENSIONS = {".mp4"}
    _REQUIRED_NODES = (
        "MiniMaxH3ImageToVideo",
        "MiniMaxH3MemoryEfficientSageAttentionPatch",
        "LoraLoaderModelOnly",
        "VAEDecodeAudio",
        "CreateVideo",
        "SaveVideo",
    )

    def __init__(self, settings: Settings):
        self.settings = settings
        self._process: asyncio.subprocess.Process | None = None
        self._process_lock = asyncio.Lock()
        self._log_handle = None

    def health(self) -> dict[str, object]:
        models = self.settings.comfyui_models_dir
        unet = models / "diffusion_models" / self.settings.unet_name
        lora = models / "loras" / self.settings.lora_name
        clip = models / "text_encoders" / self.settings.clip_name
        video_vae = models / "vae" / self.settings.video_vae_name
        audio_vae = models / "vae" / self.settings.audio_vae_name
        checks = {
            "models_dir_exists": models.is_dir(),
            "workflow_exists": self.settings.comfyui_workflow_template.is_file(),
            "unet_exists": unet.is_file() if models.is_dir() else None,
            "lora_exists": lora.is_file() if models.is_dir() else None,
            "clip_exists": clip.is_file() if models.is_dir() else None,
            "video_vae_exists": video_vae.is_file() if models.is_dir() else None,
            "audio_vae_exists": audio_vae.is_file() if models.is_dir() else None,
        }
        configured = all(value is not False for value in checks.values())
        return {
            "backend": self.name,
            "ready": False,
            "configured": configured,
            "reachable": None,
            "dependency_checked": False,
            "runtime_owned_by_mcp": self.settings.comfyui_manage_process,
            "runtime_state": (
                "running"
                if self._process is not None and self._process.returncode is None
                else "stopped"
            ),
            "checks": checks,
            "base_url": self.settings.comfyui_base_url,
            "models_dir": str(models),
            "unet_name": self.settings.unet_name,
            "lora_name": self.settings.lora_name,
            "steps": self.settings.steps,
            "sampler_name": self.settings.sampler_name,
            "scheduler_name": self.settings.scheduler_name,
            "supports_first_frame": True,
            "supports_last_frame": True,
            "supports_reference_image": False,
            "supports_text_only": True,
            "unload_after_job": self.settings.comfyui_unload_after_job,
        }

    async def probe_health(self) -> dict[str, object]:
        health = self.health()
        if not health["configured"]:
            health["reachable"] = False
            health["dependency_checked"] = True
            health["error"] = "required model or workflow files are missing"
            return health
        if self.settings.comfyui_manage_process:
            try:
                await self._ensure_process()
            except RuntimeError as exc:
                health["reachable"] = False
                health["dependency_checked"] = True
                health["error"] = str(exc)[-1000:]
                return health
        timeout = aiohttp.ClientTimeout(
            total=self.settings.comfyui_health_timeout_seconds,
            connect=self.settings.comfyui_health_timeout_seconds,
            sock_connect=self.settings.comfyui_health_timeout_seconds,
            sock_read=self.settings.comfyui_health_timeout_seconds,
        )
        try:
            async with aiohttp.ClientSession(
                headers=self._headers(), timeout=timeout
            ) as session:
                queue = await self._json_request(session, "GET", "/queue")
                for node in self._REQUIRED_NODES:
                    value = await self._json_request(
                        session, "GET", f"/object_info/{node}"
                    )
                    if node not in value:
                        raise RuntimeError(f"required ComfyUI node is missing: {node}")
                reference_nodes = (
                    "MiniMaxH3ReferenceToVideo", "MiniMaxH3AddGuide"
                )
                try:
                    supports_reference = True
                    for node in reference_nodes:
                        value = await self._json_request(
                            session, "GET", f"/object_info/{node}"
                        )
                        supports_reference &= node in value
                    health["supports_reference_image"] = supports_reference
                except (aiohttp.ClientError, TimeoutError, RuntimeError):
                    health["supports_reference_image"] = False
        except (aiohttp.ClientError, TimeoutError, RuntimeError) as exc:
            health["reachable"] = False
            health["dependency_checked"] = True
            health["error"] = f"ComfyUI health probe failed: {type(exc).__name__}"
            return health
        health["reachable"] = True
        health["dependency_checked"] = True
        health["ready"] = True
        health["queue_running"] = len(queue.get("queue_running") or [])
        health["queue_pending"] = len(queue.get("queue_pending") or [])
        return health

    async def shutdown(self) -> None:
        async with self._process_lock:
            await self._stop_process_locked()

    async def _ensure_process(self) -> None:
        if await self._api_available():
            return
        async with self._process_lock:
            if await self._api_available():
                return
            if self._process is not None and self._process.returncode is None:
                raise RuntimeError("managed ComfyUI process is running but API is unavailable")
            if not self.settings.comfyui_python.is_file():
                raise RuntimeError("COMFYUI_PYTHON does not exist")
            main = self.settings.comfyui_root / "main.py"
            if not main.is_file():
                raise RuntimeError("COMFYUI_ROOT/main.py does not exist")
            split = urlsplit(self.settings.comfyui_base_url)
            if split.scheme != "http" or split.hostname not in {"127.0.0.1", "localhost"}:
                raise RuntimeError(
                    "managed ComfyUI requires a loopback http COMFYUI_BASE_URL"
                )
            port = split.port or 80
            log_path = self.settings.data_dir / "comfyui-worker.log"
            self._log_handle = log_path.open("ab", buffering=0)
            self._process = await asyncio.create_subprocess_exec(
                str(self.settings.comfyui_python),
                str(main),
                "--listen",
                "127.0.0.1",
                "--port",
                str(port),
                "--disable-auto-launch",
                cwd=self.settings.comfyui_root,
                stdout=self._log_handle,
                stderr=asyncio.subprocess.STDOUT,
                start_new_session=True,
            )
            deadline = (
                asyncio.get_running_loop().time()
                + self.settings.comfyui_startup_timeout_seconds
            )
            try:
                while asyncio.get_running_loop().time() < deadline:
                    if self._process.returncode is not None:
                        raise RuntimeError(
                            "managed ComfyUI exited during startup with code "
                            f"{self._process.returncode}"
                        )
                    if await self._api_available():
                        return
                    await asyncio.sleep(0.5)
                raise RuntimeError("managed ComfyUI did not become ready before timeout")
            except BaseException:
                await self._stop_process_locked()
                raise

    async def _stop_process_locked(self) -> None:
        process, self._process = self._process, None
        if process is not None and process.returncode is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                await asyncio.wait_for(process.wait(), timeout=20)
            except TimeoutError:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                await process.wait()
        if self._log_handle is not None:
            self._log_handle.close()
            self._log_handle = None

    async def _api_available(self) -> bool:
        timeout = aiohttp.ClientTimeout(total=1.0, connect=1.0, sock_read=1.0)
        try:
            async with aiohttp.ClientSession(
                headers=self._headers(), timeout=timeout
            ) as session:
                await self._json_request(session, "GET", "/queue")
            return True
        except (aiohttp.ClientError, TimeoutError, RuntimeError):
            return False

    async def generate(
        self,
        request: GenerationRequest,
        job_dir: Path,
        report: ProgressReporter,
        cancel_event: asyncio.Event,
    ) -> Path:
        health = await self.probe_health()
        if not health["ready"]:
            raise RuntimeError(
                "ComfyUI 8-step backend is not ready: "
                f"checks={health['checks']}, reachable={health.get('reachable')}"
            )
        if request.reference_image_path and not health["supports_reference_image"]:
            raise RuntimeError("ComfyUI backend does not expose reference-image conditioning")
        try:
            template = json.loads(
                self.settings.comfyui_workflow_template.read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                f"invalid ComfyUI workflow template: {self.settings.comfyui_workflow_template}"
            ) from exc
        if not isinstance(template, dict):
            raise RuntimeError("ComfyUI workflow template must be a JSON object")

        job_dir.mkdir(parents=True, exist_ok=False)
        headers = self._headers()
        timeout = aiohttp.ClientTimeout(
            total=None, connect=30, sock_connect=30, sock_read=None
        )
        client_id = uuid.uuid4().hex
        prompt_id: str | None = None
        started = False
        await report(8.0, "connecting_comfyui", "正在连接 ComfyUI")

        async with aiohttp.ClientSession(headers=headers, timeout=timeout) as session:
            first_name = await self._upload_image(
                session, request.first_frame_path, "first"
            )
            last_name = await self._upload_image(
                session, request.last_frame_path, "last"
            )
            reference_name = await self._upload_image(
                session, request.reference_image_path, "reference"
            )
            workflow = self.render_workflow(
                template, request, first_name=first_name, last_name=last_name,
                reference_name=reference_name,
            )
            (job_dir / "workflow.rendered.json").write_text(
                json.dumps(workflow, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            ws_url = self._websocket_url(client_id)
            await report(12.0, "submitting", "正在向 ComfyUI 提交 8 步工作流")
            try:
                async with session.ws_connect(ws_url, heartbeat=30) as websocket:
                    response = await self._json_request(
                        session,
                        "POST",
                        "/prompt",
                        json={"prompt": workflow, "client_id": client_id},
                    )
                    prompt_id = str(response.get("prompt_id", "")).strip()
                    if not prompt_id:
                        raise RuntimeError(
                            f"ComfyUI did not return prompt_id: {response}"
                        )
                    await report(18.0, "queued_in_comfyui", "工作流已进入 ComfyUI 队列")
                    started = await self._wait_for_completion(
                        session, websocket, prompt_id, report, cancel_event
                    )
            except aiohttp.ClientError as exc:
                if cancel_event.is_set():
                    raise BackendCanceled("job canceled") from exc
                raise RuntimeError(f"ComfyUI connection failed: {exc}") from exc
            finally:
                if cancel_event.is_set() and prompt_id:
                    await self._cancel_prompt(session, prompt_id, started)

            if cancel_event.is_set():
                raise BackendCanceled("job canceled")
            assert prompt_id is not None
            await report(94.0, "collecting_output", "正在读取 ComfyUI 输出")
            history = await self._json_request(session, "GET", f"/history/{prompt_id}")
            output_info = self._find_video_output(history, prompt_id)
            output = job_dir / "result.mp4"
            await self._download_output(session, output_info, output)
            await self._maybe_unload(session, report)
        if not output.is_file() or output.stat().st_size == 0:
            raise RuntimeError("ComfyUI produced no usable MP4 file")
        return output

    def render_workflow(
        self,
        template: dict[str, object],
        request: GenerationRequest,
        *,
        first_name: str,
        last_name: str,
        reference_name: str = "",
    ) -> dict[str, object]:
        prepared = copy.deepcopy(template)
        if not first_name:
            self._prune_optional_image(prepared, "{{FIRST_FRAME}}")
        if not last_name:
            self._prune_optional_image(prepared, "{{LAST_FRAME}}")
        if request.reference_image_path:
            if not reference_name:
                raise ValueError("reference_image_path requires an uploaded reference image")
            prepared["19"] = {
                "class_type": "LoadImage",
                "inputs": {"image": reference_name},
                "_meta": {"title": "Persistent reference image"},
            }
            prepared["5"]["class_type"] = "MiniMaxH3ReferenceToVideo"
            prepared["5"]["inputs"] = {
                "clip": ["2", 0], "vae": ["3", 0],
                "prompt": "{{H3_PROMPT}}", "width": "{{WIDTH}}",
                "height": "{{HEIGHT}}", "length": "{{NUM_FRAMES}}",
                "ref_image_size": "match",
                "ref_images.ref_image_0": ["19", 0],
            }
            conditioning_source = "5"
            for node_id, frame_node, frame_index in (
                ("20", "15", 0), ("21", "16", -1)
            ):
                if (frame_node == "15" and not first_name) or (frame_node == "16" and not last_name):
                    continue
                prepared[node_id] = {
                    "class_type": "MiniMaxH3AddGuide",
                    "inputs": {
                        "positive": [conditioning_source, 0],
                        "latent": ["5", 1], "vae": ["3", 0],
                        "image": [frame_node, 0], "frame_idx": frame_index,
                    },
                }
                conditioning_source = node_id
            prepared["9"]["inputs"]["conditioning"] = [conditioning_source, 0]
        if not request.audio_enabled:
            self._prune_audio_output(prepared)
        replacements: dict[str, object] = {
            "{{H3_PROMPT}}": request.model_prompt,
            "{{WIDTH}}": request.width,
            "{{HEIGHT}}": request.height,
            "{{NUM_FRAMES}}": request.num_frames,
            "{{SEED}}": request.seed,
            "{{STEPS}}": self.settings.steps,
            "{{SAMPLER_NAME}}": self.settings.sampler_name,
            "{{SCHEDULER_NAME}}": self.settings.scheduler_name,
            "{{UNET_NAME}}": self.settings.unet_name,
            "{{CLIP_NAME}}": self.settings.clip_name,
            "{{VIDEO_VAE_NAME}}": self.settings.video_vae_name,
            "{{AUDIO_VAE_NAME}}": self.settings.audio_vae_name,
            "{{LORA_NAME}}": self.settings.lora_name,
            "{{LORA_STRENGTH}}": self.settings.lora_strength,
            "{{FIRST_FRAME}}": first_name,
            "{{LAST_FRAME}}": last_name,
        }
        rendered = render_workflow_template(prepared, replacements)
        serialized = json.dumps(rendered, ensure_ascii=False)
        unresolved = sorted(set(_WORKFLOW_PLACEHOLDER_RE.findall(serialized)))
        if unresolved:
            raise RuntimeError(
                "unresolved ComfyUI workflow placeholders: " + ", ".join(unresolved)
            )
        assert isinstance(rendered, dict)
        return rendered

    @staticmethod
    def _prune_optional_image(workflow: dict[str, object], placeholder: str) -> None:
        removable: set[str] = set()
        for node_id, value in workflow.items():
            if not isinstance(value, dict):
                continue
            inputs = value.get("inputs")
            if (
                value.get("class_type") == "LoadImage"
                and isinstance(inputs, dict)
                and inputs.get("image") == placeholder
            ):
                removable.add(str(node_id))
        for node_id in removable:
            workflow.pop(node_id, None)
        for value in workflow.values():
            if not isinstance(value, dict):
                continue
            inputs = value.get("inputs")
            if not isinstance(inputs, dict):
                continue
            for input_name, input_value in tuple(inputs.items()):
                if (
                    isinstance(input_value, list)
                    and len(input_value) == 2
                    and str(input_value[0]) in removable
                ):
                    inputs.pop(input_name)

    @staticmethod
    def _prune_audio_output(workflow: dict[str, object]) -> None:
        removable: set[str] = set()
        for node_id, value in workflow.items():
            if not isinstance(value, dict):
                continue
            class_type = value.get("class_type")
            meta = value.get("_meta")
            title = str(meta.get("title", "")).lower() if isinstance(meta, dict) else ""
            if class_type == "VAEDecodeAudio" or (
                class_type == "VAELoader" and "audio" in title
            ):
                removable.add(str(node_id))
        for node_id in removable:
            workflow.pop(node_id, None)
        for value in workflow.values():
            if not isinstance(value, dict):
                continue
            inputs = value.get("inputs")
            if not isinstance(inputs, dict):
                continue
            for input_name, input_value in tuple(inputs.items()):
                if (
                    isinstance(input_value, list)
                    and len(input_value) == 2
                    and str(input_value[0]) in removable
                ):
                    inputs.pop(input_name)

    def _headers(self) -> dict[str, str]:
        if not self.settings.comfyui_api_key:
            return {}
        return {"Authorization": f"Bearer {self.settings.comfyui_api_key}"}

    def _url(self, path: str) -> str:
        return f"{self.settings.comfyui_base_url}{path}"

    def _websocket_url(self, client_id: str) -> str:
        split = urlsplit(self.settings.comfyui_base_url)
        if split.scheme not in {"http", "https"} or not split.netloc:
            raise RuntimeError("COMFYUI_BASE_URL must be an http(s) URL")
        scheme = "wss" if split.scheme == "https" else "ws"
        base_path = split.path.rstrip("/")
        return urlunsplit(
            (
                scheme,
                split.netloc,
                f"{base_path}/ws",
                urlencode({"clientId": client_id}),
                "",
            )
        )

    async def _json_request(
        self,
        session: aiohttp.ClientSession,
        method: str,
        path: str,
        **kwargs: object,
    ) -> dict[str, object]:
        async with session.request(method, self._url(path), **kwargs) as response:
            text = await response.text()
            if response.status >= 400:
                raise RuntimeError(
                    f"ComfyUI {method} {path} returned {response.status}: {text[-4000:]}"
                )
            try:
                payload = json.loads(text) if text else {}
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"ComfyUI returned invalid JSON for {path}") from exc
            if not isinstance(payload, dict):
                raise RuntimeError(f"ComfyUI returned non-object JSON for {path}")
            return payload

    async def _upload_image(
        self,
        session: aiohttp.ClientSession,
        source_path: str | None,
        role: str,
    ) -> str:
        if source_path is None:
            return ""
        source = Path(source_path)
        if not source.is_file():
            raise RuntimeError(f"{role} reference image is missing")
        form = aiohttp.FormData()
        form.add_field(
            "image",
            source.read_bytes(),
            filename=f"{uuid.uuid4().hex}_{role}{source.suffix.lower()}",
            content_type={
                ".png": "image/png",
                ".jpg": "image/jpeg",
                ".jpeg": "image/jpeg",
                ".webp": "image/webp",
            }.get(source.suffix.lower(), "application/octet-stream"),
        )
        form.add_field("type", "input")
        form.add_field("subfolder", "minimax_h3_mcp")
        form.add_field("overwrite", "false")
        response = await self._json_request(session, "POST", "/upload/image", data=form)
        name = str(response.get("name", "")).strip()
        subfolder = str(response.get("subfolder", "")).strip().strip("/\\")
        if not name or Path(name).name != name:
            raise RuntimeError(
                f"ComfyUI returned invalid uploaded image name: {response}"
            )
        normalized_subfolder = subfolder.replace("\\", "/")
        if normalized_subfolder and (
            normalized_subfolder.startswith("/")
            or ".." in normalized_subfolder.split("/")
        ):
            raise RuntimeError(
                f"ComfyUI returned invalid upload subfolder: {response}"
            )
        return f"{normalized_subfolder}/{name}" if normalized_subfolder else name

    async def _wait_for_completion(
        self,
        session: aiohttp.ClientSession,
        websocket: aiohttp.ClientWebSocketResponse,
        prompt_id: str,
        report: ProgressReporter,
        cancel_event: asyncio.Event,
    ) -> bool:
        started = False
        cancel_deferred = False
        current_progress = 20.0
        while True:
            if cancel_event.is_set():
                stopped = await self._cancel_prompt(session, prompt_id, started)
                if stopped:
                    raise BackendCanceled("job canceled")
                if not cancel_deferred:
                    cancel_deferred = True
                    await report(
                        current_progress,
                        "cancel_waiting",
                        "共享 ComfyUI 未允许中断运行中任务；等待其结束后丢弃结果",
                    )
            try:
                message = await asyncio.wait_for(
                    websocket.receive(),
                    timeout=self.settings.progress_heartbeat_seconds,
                )
            except TimeoutError:
                continue
            if message.type in {
                aiohttp.WSMsgType.CLOSE,
                aiohttp.WSMsgType.CLOSED,
                aiohttp.WSMsgType.ERROR,
            }:
                raise RuntimeError("ComfyUI WebSocket closed before completion")
            if message.type != aiohttp.WSMsgType.TEXT:
                continue
            try:
                event = json.loads(message.data)
            except (TypeError, json.JSONDecodeError):
                continue
            if not isinstance(event, dict):
                continue
            data = event.get("data")
            if not isinstance(data, dict):
                data = {}
            event_prompt_id = data.get("prompt_id")
            if event_prompt_id is not None and str(event_prompt_id) != prompt_id:
                continue
            event_type = str(event.get("type", ""))
            if event_type in {"execution_error", "execution_interrupted"}:
                raise RuntimeError(
                    "ComfyUI execution failed: "
                    + json.dumps(data, ensure_ascii=False)[-4000:]
                )
            if event_type == "execution_start":
                started = True
                current_progress = max(current_progress, 25.0)
                await report(current_progress, "comfyui_started", "ComfyUI 已开始执行 8 步工作流")
                continue
            if event_type == "executing":
                node = data.get("node")
                if node is None:
                    await report(93.0, "comfyui_completed", "ComfyUI 工作流执行完成")
                    return started
                started = True
                current_progress = max(current_progress, min(85.0, current_progress + 1.0))
                await report(
                    current_progress, "comfyui_node", f"ComfyUI 正在执行节点 {node}"
                )
                continue
            if event_type == "progress":
                try:
                    value = float(data.get("value", 0))
                    maximum = max(1.0, float(data.get("max", 1)))
                except (TypeError, ValueError):
                    continue
                node = data.get("node", "unknown")
                current_progress = max(
                    current_progress, min(92.0, 30.0 + 62.0 * value / maximum)
                )
                await report(
                    current_progress,
                    "sampling",
                    f"ComfyUI 节点 {node} 采样进度 {int(value)}/{int(maximum)}",
                )
                continue
            if event_type in {"execution_success", "execution_complete"}:
                await report(93.0, "comfyui_completed", "ComfyUI 工作流执行完成")
                return started

    async def _cancel_prompt(
        self,
        session: aiohttp.ClientSession,
        prompt_id: str,
        started: bool,
    ) -> bool:
        removed_from_queue = False
        try:
            await self._json_request(
                session, "POST", "/queue", json={"delete": [prompt_id]}
            )
            removed_from_queue = not started
        except Exception:
            pass
        if started and (
            self.settings.comfyui_allow_interrupt
            or (
                self.settings.comfyui_manage_process
                and self._process is not None
                and self._process.returncode is None
            )
        ):
            try:
                await self._json_request(session, "POST", "/interrupt", json={})
                return True
            except Exception:
                pass
        return removed_from_queue

    def _find_video_output(
        self, history: dict[str, object], prompt_id: str
    ) -> dict[str, str]:
        root: object = history.get(prompt_id, history)
        candidates: list[dict[str, str]] = []

        def visit(value: object) -> None:
            if isinstance(value, dict):
                filename = value.get("filename")
                if isinstance(filename, str):
                    extension = Path(filename).suffix.lower()
                    if extension in self._VIDEO_EXTENSIONS:
                        candidates.append(
                            {
                                "filename": filename,
                                "subfolder": str(value.get("subfolder", "")),
                                "type": str(value.get("type", "output")),
                            }
                        )
                for child in value.values():
                    visit(child)
            elif isinstance(value, list):
                for child in value:
                    visit(child)

        visit(root)
        if not candidates:
            raise RuntimeError(
                "ComfyUI history contains no MP4 output; configure the workflow to save MP4"
            )
        return candidates[-1]

    async def _download_output(
        self,
        session: aiohttp.ClientSession,
        output_info: dict[str, str],
        destination: Path,
    ) -> None:
        parameters = urlencode(output_info)
        async with session.get(self._url(f"/view?{parameters}")) as response:
            if response.status >= 400:
                text = await response.text()
                raise RuntimeError(
                    f"ComfyUI output download failed with {response.status}: {text[-2000:]}"
                )
            with destination.open("wb") as target:
                received = 0
                async for chunk in response.content.iter_chunked(1024 * 1024):
                    received += len(chunk)
                    if received > self.settings.max_output_video_bytes:
                        target.close()
                        destination.unlink(missing_ok=True)
                        raise RuntimeError(
                            "ComfyUI output exceeds H3_MCP_MAX_OUTPUT_VIDEO_BYTES"
                        )
                    target.write(chunk)

    async def _maybe_unload(
        self, session: aiohttp.ClientSession, report: ProgressReporter
    ) -> None:
        if not self.settings.comfyui_unload_after_job:
            return
        queue = await self._json_request(session, "GET", "/queue")
        running = queue.get("queue_running")
        pending = queue.get("queue_pending")
        if running or pending:
            await report(96.0, "retaining_models", "ComfyUI 仍有其他任务，暂不卸载共享模型")
            return
        await report(96.0, "freeing_vram", "正在卸载 ComfyUI 模型并释放显存")
        await self._json_request(
            session,
            "POST",
            "/free",
            json={"unload_models": True, "free_memory": True},
        )
