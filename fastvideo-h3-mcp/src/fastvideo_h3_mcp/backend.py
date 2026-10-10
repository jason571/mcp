from __future__ import annotations

import asyncio
import copy
import json
import os
import re
import signal
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol
from urllib.parse import urlencode, urlsplit, urlunsplit

import aiohttp

from .config import Settings
from .models import GenerationRequest
from .sage_attention import (
    AttentionResolution,
    empty_sage_probe,
    probe_sage_attention,
    resolve_attention_backend,
)


ProgressReporter = Callable[[float, str, str], Awaitable[None]]
_WORKFLOW_PLACEHOLDER_RE = re.compile(r"\{\{[A-Z0-9_]+\}\}")


class BackendCanceled(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class ResourceSnapshot:
    gpu_used_mib: int
    system_available_mib: int


class ResourceAdmission:
    """Wait until GPU and host memory are safe for a large model worker.

    JobManager serializes jobs owned by this MCP process. This guard covers
    CUDA consumers outside that queue, notably a separately launched ComfyUI.
    The probes intentionally fail closed: starting a 40-60 GiB worker with an
    unknown resource state is less safe than returning a diagnostic error.
    """

    def __init__(
        self,
        settings: Settings,
        probe: Callable[[], Awaitable[ResourceSnapshot]] | None = None,
    ):
        self.settings = settings
        self._probe = probe or self._probe_resources

    def health(self) -> dict[str, object]:
        return {
            "enabled": self.settings.resource_admission_enabled,
            "max_gpu_used_mib": self.settings.resource_admission_max_gpu_used_mib,
            "min_system_available_mib": (
                self.settings.resource_admission_min_system_available_mib
            ),
            "ready_samples": self.settings.resource_admission_ready_samples,
            "poll_seconds": self.settings.resource_admission_poll_seconds,
        }

    async def wait(
        self,
        report: ProgressReporter,
        cancel_event: asyncio.Event,
    ) -> None:
        if not self.settings.resource_admission_enabled:
            return
        consecutive_ready = 0
        while consecutive_ready < self.settings.resource_admission_ready_samples:
            if cancel_event.is_set():
                raise BackendCanceled("job canceled")
            snapshot = await self._probe()
            gpu_ready = (
                snapshot.gpu_used_mib
                <= self.settings.resource_admission_max_gpu_used_mib
            )
            memory_ready = (
                snapshot.system_available_mib
                >= self.settings.resource_admission_min_system_available_mib
            )
            if gpu_ready and memory_ready:
                consecutive_ready += 1
                await report(
                    7.0,
                    "checking_resources",
                    "GPU 与主机内存可用，正在确认资源稳定 "
                    f"{consecutive_ready}/{self.settings.resource_admission_ready_samples} "
                    f"(GPU 已用 {snapshot.gpu_used_mib} MiB，"
                    f"系统可用 {snapshot.system_available_mib} MiB)",
                )
            else:
                consecutive_ready = 0
                reasons: list[str] = []
                if not gpu_ready:
                    reasons.append(
                        f"GPU 已用 {snapshot.gpu_used_mib} MiB，"
                        "可能有 ComfyUI 或其他 CUDA 任务"
                    )
                if not memory_ready:
                    reasons.append(
                        f"系统可用内存仅 {snapshot.system_available_mib} MiB"
                    )
                await report(
                    7.0,
                    "waiting_for_resources",
                    "等待外部任务释放资源：" + "；".join(reasons),
                )
            if consecutive_ready >= self.settings.resource_admission_ready_samples:
                return
            try:
                await asyncio.wait_for(
                    cancel_event.wait(),
                    timeout=self.settings.resource_admission_poll_seconds,
                )
            except TimeoutError:
                pass

    async def _probe_resources(self) -> ResourceSnapshot:
        system_available_mib = self._read_system_available_mib(
            Path("/proc/meminfo")
        )
        stdout = await self._run_nvidia_smi(
            "--query-gpu=memory.used",
            "--format=csv,noheader,nounits",
            "--id=0",
        )
        gpu_values = self._parse_gpu_memory(stdout)
        if len(gpu_values) != 1:
            raise RuntimeError(
                "resource admission expected exactly one GPU from nvidia-smi --id=0"
            )
        return ResourceSnapshot(
            gpu_used_mib=gpu_values[0],
            system_available_mib=system_available_mib,
        )

    async def assert_worker_exclusive(self, worker_pgid: int) -> None:
        """Fail if a GPU process outside the worker group crosses the limit."""
        if not self.settings.resource_admission_enabled:
            return
        stdout = await self._run_nvidia_smi(
            "--query-compute-apps=pid,used_gpu_memory",
            "--format=csv,noheader,nounits",
            "--id=0",
        )
        external_used_mib = 0
        for pid, used_mib in self._parse_gpu_process_memory(stdout):
            try:
                process_group = os.getpgid(pid)
            except ProcessLookupError:
                continue
            except PermissionError:
                process_group = -1
            if process_group != worker_pgid:
                external_used_mib += used_mib
        if external_used_mib > self.settings.resource_admission_max_gpu_used_mib:
            raise RuntimeError(
                "external GPU workload exceeded the safe coexistence limit "
                f"({external_used_mib} MiB > "
                f"{self.settings.resource_admission_max_gpu_used_mib} MiB); "
                "the FastVideo worker was stopped to prevent system OOM"
            )

    async def _run_nvidia_smi(self, *arguments: str) -> str:
        try:
            process = await asyncio.create_subprocess_exec(
                self.settings.nvidia_smi_binary,
                *arguments,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except OSError as exc:
            raise RuntimeError(
                f"resource admission could not start nvidia-smi: {exc}"
            ) from exc
        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(), timeout=10
            )
        except TimeoutError as exc:
            process.kill()
            await process.wait()
            raise RuntimeError("resource admission nvidia-smi probe timed out") from exc
        if process.returncode != 0:
            detail = stderr.decode("utf-8", errors="replace").strip()
            raise RuntimeError(
                "resource admission nvidia-smi probe failed: " + detail[-1000:]
            )
        return stdout.decode("utf-8", errors="replace")

    @staticmethod
    def _read_system_available_mib(path: Path) -> int:
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except OSError as exc:
            raise RuntimeError(f"resource admission could not read {path}: {exc}") from exc
        for line in lines:
            if line.startswith("MemAvailable:"):
                fields = line.split()
                if len(fields) >= 2 and fields[1].isdigit():
                    return int(fields[1]) // 1024
        raise RuntimeError(f"resource admission found no MemAvailable in {path}")

    @staticmethod
    def _parse_gpu_memory(output: str) -> list[int]:
        values: list[int] = []
        for line in output.splitlines():
            normalized = line.strip()
            if not normalized:
                continue
            if not normalized.isdigit():
                raise RuntimeError(
                    f"resource admission received invalid GPU memory value: {normalized}"
                )
            values.append(int(normalized))
        return values

    @staticmethod
    def _parse_gpu_process_memory(output: str) -> list[tuple[int, int]]:
        values: list[tuple[int, int]] = []
        for line in output.splitlines():
            normalized = line.strip()
            if not normalized:
                continue
            fields = [field.strip() for field in normalized.split(",")]
            if len(fields) != 2 or not all(field.isdigit() for field in fields):
                raise RuntimeError(
                    "resource admission received invalid GPU process value: "
                    + normalized
                )
            values.append((int(fields[0]), int(fields[1])))
        return values


async def _remove_audio_track(ffmpeg_binary: str, output: Path) -> None:
    silent_output = output.with_name("result.silent.mp4")
    process = await asyncio.create_subprocess_exec(
        ffmpeg_binary,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(output),
        "-map",
        "0:v:0",
        "-c:v",
        "copy",
        "-an",
        str(silent_output),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    _stdout, stderr = await process.communicate()
    if process.returncode != 0:
        raise RuntimeError(
            "failed to remove audio track: "
            + stderr.decode("utf-8", errors="replace")[-2000:]
        )
    silent_output.replace(output)


class VideoBackend(Protocol):
    name: str

    async def generate(
        self,
        request: GenerationRequest,
        job_dir: Path,
        report: ProgressReporter,
        cancel_event: asyncio.Event,
    ) -> Path: ...

    def health(self) -> dict[str, object]: ...


class MockBackend:
    name = "mock"

    def __init__(self, step_delay_seconds: float = 0.05):
        self.step_delay_seconds = step_delay_seconds

    async def generate(
        self,
        request: GenerationRequest,
        job_dir: Path,
        report: ProgressReporter,
        cancel_event: asyncio.Event,
    ) -> Path:
        job_dir.mkdir(parents=True, exist_ok=True)
        stages = (
            (10.0, "preparing", "正在准备模拟后端"),
            (25.0, "loading_model", "正在加载模拟模型"),
            (55.0, "generating", "正在模拟视频生成"),
            (85.0, "encoding", "正在模拟视频编码"),
            (95.0, "verifying", "正在校验模拟输出"),
        )
        for progress, stage, message in stages:
            if cancel_event.is_set():
                raise BackendCanceled("job canceled")
            await report(progress, stage, message)
            await asyncio.sleep(self.step_delay_seconds)
        output = job_dir / "result.mp4"
        output.write_bytes(b"fastvideo-h3-mcp mock artifact\n")
        return output

    def health(self) -> dict[str, object]:
        return {"backend": self.name, "ready": True, "real_video": False}


@dataclass(slots=True)
class _Stage:
    progress: float
    name: str
    message: str


_WORKER_RESULT_PREFIX = "MCP_WORKER_RESULT "


async def _terminate_process(process: asyncio.subprocess.Process) -> None:
    if process.returncode is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        await asyncio.wait_for(process.wait(), timeout=10)
    except TimeoutError:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            return
        await process.wait()


class _PersistentWorker:
    """Own one reusable CUDA subprocess for one model profile.

    Requests are newline-delimited JSON. The runner prints normal FastVideo
    logs plus one ``MCP_WORKER_RESULT`` record for each request. A healthy
    worker is retained only while it is idle and only for the configured
    timeout. Cancellation, runner errors, external GPU conflicts, and model
    switches all terminate the complete process group.
    """

    def __init__(
        self,
        settings: Settings,
        admission: ResourceAdmission,
        profile: str,
    ):
        self.settings = settings
        self.admission = admission
        self.profile = profile
        self._process: asyncio.subprocess.Process | None = None
        self._lock = asyncio.Lock()
        self._idle_task: asyncio.Task[None] | None = None
        self._idle_since: float | None = None
        self._busy = False

    @property
    def is_warm(self) -> bool:
        process = self._process
        return process is not None and process.returncode is None

    def health(self) -> dict[str, object]:
        process = self._process
        alive = process is not None and process.returncode is None
        return {
            "profile": self.profile,
            "state": "running" if alive and self._busy else "idle" if alive else "stopped",
            "pid": process.pid if alive else None,
            "idle_timeout_seconds": getattr(
                self.settings, "gpu_worker_idle_timeout_seconds", 60
            ),
            "warm_max_duration_seconds": (
                getattr(self.settings, "gpu_worker_warm_max_duration_seconds", 30)
            ),
        }

    async def run(
        self,
        *,
        command: list[str],
        cwd: Path,
        environment: dict[str, str],
        payload: dict[str, object],
        job_dir: Path,
        report: ProgressReporter,
        cancel_event: asyncio.Event,
        initial_stage: _Stage,
        classify_line: Callable[[str, float], _Stage | None],
    ) -> dict[str, object]:
        await self._cancel_idle_task()
        async with self._lock:
            reused = self.is_warm
            logs: list[str] = [
                f"worker_profile={self.profile}",
                f"worker_reused={'true' if reused else 'false'}",
            ]
            try:
                if not reused:
                    await self.admission.wait(report, cancel_event)
                    await report(8.0, "starting_backend", "正在启动 FastVideo GPU Worker")
                    self._process = await asyncio.create_subprocess_exec(
                        *command,
                        cwd=cwd,
                        env=environment,
                        stdin=asyncio.subprocess.PIPE,
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.STDOUT,
                        start_new_session=True,
                    )
                else:
                    await report(8.0, "reusing_model", "正在复用已加载的 FastVideo 模型")

                process = self._process
                assert process is not None
                assert process.stdin is not None
                assert process.stdout is not None
                self._busy = True
                self._idle_since = None
                logs.extend(
                    (
                        f"worker_pid={process.pid}",
                        f"worker_command={command}",
                    )
                )
                request_id = uuid.uuid4().hex
                request_payload = dict(payload)
                request_payload["request_id"] = request_id
                process.stdin.write(
                    (json.dumps(request_payload, ensure_ascii=False) + "\n").encode(
                        "utf-8"
                    )
                )
                await process.stdin.drain()

                current = initial_stage
                await report(current.progress, current.name, current.message)
                next_resource_check = 0.0
                result: dict[str, object] | None = None
                while result is None:
                    if cancel_event.is_set():
                        await self._terminate_locked()
                        raise BackendCanceled("job canceled")
                    now = asyncio.get_running_loop().time()
                    if now >= next_resource_check:
                        try:
                            await self.admission.assert_worker_exclusive(process.pid)
                        except RuntimeError as exc:
                            await report(
                                current.progress,
                                "resource_conflict",
                                "检测到外部 GPU 任务，正在释放 FastVideo 模型以避免 OOM",
                            )
                            await self._terminate_locked()
                            raise exc
                        next_resource_check = (
                            now + self.settings.resource_admission_poll_seconds
                        )
                    try:
                        raw = await asyncio.wait_for(
                            process.stdout.readline(),
                            timeout=self.settings.progress_heartbeat_seconds,
                        )
                    except TimeoutError:
                        continue
                    if not raw:
                        return_code = await process.wait()
                        raise RuntimeError(
                            "FastVideo GPU Worker exited before returning a result "
                            f"(code {return_code})\n" + "\n".join(logs[-20:])
                        )
                    line = raw.decode("utf-8", errors="replace").rstrip()
                    logs.append(line)
                    if line.startswith(_WORKER_RESULT_PREFIX):
                        try:
                            candidate = json.loads(line[len(_WORKER_RESULT_PREFIX) :])
                        except json.JSONDecodeError as exc:
                            raise RuntimeError(
                                "FastVideo GPU Worker returned invalid result JSON"
                            ) from exc
                        if not isinstance(candidate, dict):
                            raise RuntimeError(
                                "FastVideo GPU Worker result must be an object"
                            )
                        if candidate.get("request_id") != request_id:
                            raise RuntimeError(
                                "FastVideo GPU Worker returned a mismatched request id"
                            )
                        result = candidate
                        continue
                    classified = classify_line(line, current.progress)
                    if classified is not None and classified.progress >= current.progress:
                        current = classified
                        await report(current.progress, current.name, current.message)

                if not bool(result.get("ok")):
                    detail = str(result.get("error") or "unknown worker error")
                    await self._terminate_locked()
                    raise RuntimeError(detail[-4000:])

                reusable = bool(result.get("reusable"))
                if reusable and process.returncode is None:
                    self._busy = False
                    self._idle_since = asyncio.get_running_loop().time()
                    self._schedule_idle_task(process)
                else:
                    await self._terminate_locked()
                return result
            except (BackendCanceled, asyncio.CancelledError):
                await self._terminate_locked()
                raise
            except Exception:
                await self._terminate_locked()
                raise
            finally:
                self._busy = False
                (job_dir / "backend.log").write_text(
                    "\n".join(logs)[-1_000_000:], encoding="utf-8"
                )

    async def shutdown(self) -> None:
        await self._cancel_idle_task()
        async with self._lock:
            await self._terminate_locked()

    def _schedule_idle_task(self, process: asyncio.subprocess.Process) -> None:
        self._idle_task = asyncio.create_task(
            self._idle_monitor(process),
            name=f"fastvideo-idle-{self.profile}",
        )

    async def _cancel_idle_task(self) -> None:
        task = self._idle_task
        self._idle_task = None
        if task is None or task.done():
            return
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    async def _idle_monitor(self, process: asyncio.subprocess.Process) -> None:
        interval = min(
            float(self.settings.resource_admission_poll_seconds),
            float(self.settings.gpu_worker_idle_timeout_seconds),
        )
        try:
            while True:
                await asyncio.sleep(interval)
                async with self._lock:
                    if self._process is not process or self._busy:
                        return
                    idle_since = self._idle_since
                    if idle_since is None:
                        return
                    elapsed = asyncio.get_running_loop().time() - idle_since
                    if elapsed >= self.settings.gpu_worker_idle_timeout_seconds:
                        await self._terminate_locked()
                        return
                    try:
                        await self.admission.assert_worker_exclusive(process.pid)
                    except RuntimeError:
                        await self._terminate_locked()
                        return
        except asyncio.CancelledError:
            raise

    async def _terminate_locked(self) -> None:
        process = self._process
        self._process = None
        self._idle_since = None
        self._busy = False
        if process is not None:
            await _terminate_process(process)


class FastVideoCliBackend:
    """Runs FastVideo's official basic_fasth3.py without a shell.

    The upstream CLI does not expose a stable per-denoising-step callback. This
    adapter reports real process/stage transitions. JobManager supplies the
    single heartbeat stream while a long CUDA operation is in flight.
    """

    name = "fastvideo-cli"
    _STEP_RE = re.compile(r"(?P<current>\d+)\s*/\s*(?P<total>\d+)")
    _H3_STEP_RE = re.compile(
        r"mcp_progress\s+h3_joint_denoise\s+"
        r"step=(?P<current>\d+)\s*/\s*(?P<total>\d+)\s+"
        r"phase=(?P<phase>started|completed)"
    )
    _H3_BLOCK_RE = re.compile(
        r"mcp_progress\s+h3_transformer_block\s+"
        r"step=(?P<step>\d+)\s*/\s*(?P<steps>\d+)\s+"
        r"block=(?P<block>\d+)\s*/\s*(?P<blocks>\d+)\s+phase=completed"
    )
    _PIPELINE_STAGE_RE = re.compile(
        r"mcp_progress\s+pipeline_stage=(?P<stage>[a-z0-9_]+)\s+"
        r"phase=(?P<phase>started|completed)"
    )

    def __init__(
        self,
        settings: Settings,
        admission: ResourceAdmission | None = None,
    ):
        self.settings = settings
        self.runner = Path(__file__).with_name("h3_fastvideo_runner.py")
        self._resource_admission = admission or ResourceAdmission(settings)
        self._worker = _PersistentWorker(
            settings,
            self._resource_admission,
            "fasth3",
        )
        self._attention_resolution: AttentionResolution | None = None

    def attention_resolution(self) -> AttentionResolution:
        if self._attention_resolution is None:
            configured = self.settings.fastvideo_attention_backend
            sage_mode = getattr(self.settings, "fastvideo_sage_attention", "auto")
            if configured in {
                "FLASH_ATTN",
                "SAGE_ATTN",
                "SAGE_ATTN_THREE",
                "VIDEO_SPARSE_ATTN_H3",
            }:
                probe = empty_sage_probe(
                    "explicit or VSA backend skips SageAttention probe"
                )
            else:
                probe = probe_sage_attention(
                    getattr(self.settings, "fastvideo_python", None)
                )
            self._attention_resolution = resolve_attention_backend(
                configured, sage_mode, probe
            )
        return self._attention_resolution

    def health(self) -> dict[str, object]:
        checks = {
            "python_exists": self.settings.fastvideo_python.is_file(),
            "script_exists": self.settings.fastvideo_script.is_file(),
            "runner_exists": self.runner.is_file(),
            "repo_exists": self.settings.fastvideo_repo.is_dir(),
        }
        if self.settings.fastvideo_attention_backend == "VIDEO_SPARSE_ATTN_H3":
            checks["vsa_kernel_source_exists"] = (
                self.settings.fastvideo_repo
                / "fastvideo-kernel"
                / "python"
                / "fastvideo_kernel"
            ).is_dir()
        local_model = Path(self.settings.model_path).expanduser()
        checks["model_path_exists"] = local_model.exists() if local_model.is_absolute() else None
        attention = self.attention_resolution()
        if attention.required_missing:
            checks["sage_attention_ready"] = False
        return {
            "backend": self.name,
            "ready": all(value is not False for value in checks.values()),
            "checks": checks,
            "model_id": self.settings.model_id,
            "model_path": self.settings.model_path,
            "num_gpus": self.settings.num_gpus,
            "attention_backend": attention.resolved,
            "attention_backend_configured": attention.configured,
            "sage_attention": {
                "mode": attention.sage_mode,
                "available": attention.sage_available,
                "sage_attn": attention.sage_attn_available,
                "sage_attn_three": attention.sage_attn_three_available,
                "h3_supports_sage": attention.h3_supports_sage,
                "reason": attention.reason,
            },
            "warm_worker": self._worker.health(),
        }

    @property
    def has_warm_worker(self) -> bool:
        return self._worker.is_warm

    async def shutdown(self) -> None:
        await self._worker.shutdown()

    def build_command(self, request: GenerationRequest, job_dir: Path) -> list[str]:
        command = [
            str(self.settings.fastvideo_python),
            str(self.settings.fastvideo_script),
            "--model-path",
            self.settings.model_path,
            "--prompt",
            request.model_prompt,
            "--output",
            str(job_dir),
            "--height",
            str(request.height),
            "--width",
            str(request.width),
            "--num-frames",
            str(request.num_frames),
            "--steps",
            "5",
            "--seed",
            str(request.seed),
            "--repeats",
            "1",
            "--num-gpus",
            str(self.settings.num_gpus),
        ]
        if self.settings.fastvideo_torch_compile:
            command.append("--torch-compile")
        if self.settings.fastvideo_text_encoder_fp8:
            command.append("--text-encoder-fp8")
        if self.settings.fastvideo_attention_backend == "VIDEO_SPARSE_ATTN_H3":
            command.extend(
                [
                    "--vsa-sparsity",
                    f"{self.settings.fastvideo_vsa_sparsity:g}",
                    "--vsa-tile-size",
                    str(self.settings.fastvideo_vsa_tile_size),
                ]
            )
        command.extend(self.settings.extra_args)
        return command

    def build_environment(self) -> dict[str, str]:
        environment = os.environ.copy()
        environment["PYTHONUNBUFFERED"] = "1"
        is_vsa = (
            self.settings.fastvideo_attention_backend
            == "VIDEO_SPARSE_ATTN_H3"
        )
        # The upstream script lives below examples/, so make the repository
        # package importable even when FastVideo was installed editable into a
        # venv that reuses another environment's site-packages.
        inherited_pythonpath = environment.get("PYTHONPATH", "")
        kernel_python = self.settings.fastvideo_repo / "fastvideo-kernel" / "python"
        environment["PYTHONPATH"] = os.pathsep.join(
            part
            for part in (
                str(kernel_python) if kernel_python.is_dir() else "",
                str(self.settings.fastvideo_repo),
                inherited_pythonpath,
            )
            if part
        )
        environment["FASTVIDEO_ATTENTION_BACKEND"] = (
            self.attention_resolution().resolved
        )
        environment["FASTVIDEO_FA4"] = "1" if self.settings.enable_fa4 else "0"
        environment["FASTVIDEO_LOAD_WEIGHTS_DIRECT_TO_DEVICE"] = (
            "1" if self.settings.fastvideo_direct_device_load else "0"
        )
        environment["FASTVIDEO_VSA_SM100A"] = (
            "1" if self.settings.fastvideo_vsa_kernel == "sm100a" else "0"
        )
        environment["FASTVIDEO_VSA_CUTEDSL"] = "0"
        environment["FASTVIDEO_VSA_TRITON"] = (
            "1"
            if (
                is_vsa
                and self.settings.fastvideo_vsa_kernel == "triton"
            )
            else "0"
        )
        environment["FASTVIDEO_VSA_CACHE_TILE_BUFFER"] = (
            "1"
            if is_vsa and self.settings.fastvideo_vsa_cache_tile_buffer
            else "0"
        )
        environment["FASTVIDEO_VSA_ATTENTION_RESIDUAL_CHUNK_TOKENS"] = (
            str(self.settings.fastvideo_vsa_attention_residual_chunk_tokens)
            if is_vsa
            else "0"
        )
        environment["FASTVIDEO_MCP_PROGRESS"] = "1"
        return environment

    def worker_payload(
        self, request: GenerationRequest, output: Path
    ) -> dict[str, object]:
        warm_safe = (
            request.duration_seconds
            <= getattr(self.settings, "gpu_worker_warm_max_duration_seconds", 30)
        )
        release_after_denoise = (
            getattr(self.settings, "fastvideo_release_dit_after_denoise", True)
            and not warm_safe
        )
        return {
            "model_path": self.settings.model_path,
            "prompt": request.model_prompt,
            "width": request.width,
            "height": request.height,
            "num_frames": request.num_frames,
            "fps": request.fps,
            "steps": 5,
            "seed": request.seed,
            "num_gpus": self.settings.num_gpus,
            "text_encoder_fp8": self.settings.fastvideo_text_encoder_fp8,
            "torch_compile": self.settings.fastvideo_torch_compile,
            "vsa_sparsity": (
                self.settings.fastvideo_vsa_sparsity
                if self.settings.fastvideo_attention_backend
                == "VIDEO_SPARSE_ATTN_H3"
                else 0.0
            ),
            "vsa_tile_size": self.settings.fastvideo_vsa_tile_size,
            "release_dit_after_denoise": release_after_denoise,
            "output_path": str(output),
        }

    async def generate(
        self,
        request: GenerationRequest,
        job_dir: Path,
        report: ProgressReporter,
        cancel_event: asyncio.Event,
    ) -> Path:
        if request.has_keyframes:
            raise ValueError(
                "FastH3 distilled checkpoint does not support first/last-frame conditioning"
            )
        health = self.health()
        if not health["ready"]:
            raise RuntimeError(f"FastVideo backend is not ready: {health['checks']}")

        job_dir.mkdir(parents=True, exist_ok=False)
        output = job_dir / "result.mp4"
        command = [
            str(self.settings.fastvideo_python),
            str(self.runner),
            "--server",
        ]
        environment = self.build_environment()
        await self._worker.run(
            command=command,
            cwd=self.settings.fastvideo_repo,
            environment=environment,
            payload=self.worker_payload(request, output),
            job_dir=job_dir,
            report=report,
            cancel_event=cancel_event,
            initial_stage=_Stage(
                10.0, "loading_model", "FastVideo GPU Worker 已接收任务"
            ),
            classify_line=self._classify_line,
        )
        if not output.is_file() or output.stat().st_size == 0:
            raise RuntimeError("FastVideo produced no usable MP4 file")
        if not request.audio_enabled:
            await report(98.0, "removing_audio", "正在按请求移除输出音轨")
            await _remove_audio_track(self.settings.ffmpeg_binary, output)
        await report(99.0, "verifying", "FastVideo 输出完成，正在校验结果文件")
        return output

    def _classify_line(self, line: str, current_progress: float) -> _Stage | None:
        normalized = line.lower()
        if "loading pipeline modules" in normalized or "loading required modules" in normalized:
            return _Stage(max(current_progress, 15.0), "loading_model", "正在加载 FastH3 组件")
        if "loading text_encoder" in normalized or "loading text encoder" in normalized:
            return _Stage(max(current_progress, 20.0), "loading_model", "正在加载提示词编码器")
        if "loading transformer" in normalized or "loading model from 13 safetensors" in normalized:
            return _Stage(max(current_progress, 30.0), "loading_model", "正在加载 FastH3 transformer")
        if "loaded model with" in normalized and "parameters" in normalized:
            return _Stage(max(current_progress, 38.0), "loading_model", "FastH3 transformer 加载完成")
        if "loading vae using" in normalized:
            return _Stage(max(current_progress, 42.0), "loading_model", "正在加载视频与音频解码器")
        if "running pipeline stages" in normalized:
            return _Stage(max(current_progress, 43.0), "preparing_generation", "模型加载完成，正在准备联合生成")
        h3_block = self._H3_BLOCK_RE.search(normalized)
        if h3_block:
            total_steps = max(1, int(h3_block.group("steps")))
            step = min(max(1, int(h3_block.group("step"))), total_steps)
            total_blocks = max(1, int(h3_block.group("blocks")))
            block = min(max(1, int(h3_block.group("block"))), total_blocks)
            completed_fraction = (step - 1 + block / total_blocks) / total_steps
            progress = 50.0 + 32.0 * completed_fraction
            return _Stage(
                max(current_progress, progress),
                "joint_av_denoising",
                f"视频和音频联合生成，第 {step}/{total_steps} 步，"
                f"Transformer {block}/{total_blocks} 层完成",
            )
        h3_step = self._H3_STEP_RE.search(normalized)
        if h3_step:
            current_step = int(h3_step.group("current"))
            total_steps = max(1, int(h3_step.group("total")))
            current_step = min(max(1, current_step), total_steps)
            phase = h3_step.group("phase")
            completed_steps = current_step if phase == "completed" else current_step - 1
            progress = 50.0 + 32.0 * completed_steps / total_steps
            message = (
                f"视频和音频联合生成，第 {current_step}/{total_steps} 步完成"
                if phase == "completed"
                else f"视频和音频联合生成，第 {current_step}/{total_steps} 步正在计算"
            )
            return _Stage(max(current_progress, progress), "joint_av_denoising", message)
        pipeline_stage = self._PIPELINE_STAGE_RE.search(normalized)
        if pipeline_stage:
            key = (pipeline_stage.group("stage"), pipeline_stage.group("phase"))
            stages = {
                ("input_preparation_stage", "started"): _Stage(
                    44.0, "preparing_inputs", "正在准备联合视频和音频输入"
                ),
                ("input_preparation_stage", "completed"): _Stage(
                    45.0, "preparing_inputs", "联合视频和音频输入准备完成"
                ),
                ("conditioning_stage", "started"): _Stage(
                    46.0, "conditioning", "正在编码提示词与声音条件"
                ),
                ("conditioning_stage", "completed"): _Stage(
                    47.0, "conditioning", "提示词与声音条件编码完成"
                ),
                ("latent_preparation_stage", "started"): _Stage(
                    48.0, "preparing_latents", "正在准备视频与音频联合潜变量"
                ),
                ("latent_preparation_stage", "completed"): _Stage(
                    49.0, "preparing_latents", "视频与音频联合潜变量准备完成"
                ),
                ("denoising_stage", "started"): _Stage(
                    50.0, "joint_av_denoising", "开始联合生成视频和音频"
                ),
                ("denoising_stage", "completed"): _Stage(
                    83.0, "joint_av_denoising", "视频和音频联合生成完成"
                ),
                ("video_decoding_stage", "started"): _Stage(
                    84.0, "video_decoding", "正在解码视频画面"
                ),
                ("video_decoding_stage", "completed"): _Stage(
                    87.0, "video_decoding", "视频画面解码完成"
                ),
                ("audio_decoding_stage", "started"): _Stage(
                    88.0, "audio_decoding", "正在解码音频波形"
                ),
                ("audio_decoding_stage", "completed"): _Stage(
                    91.0, "audio_decoding", "音频波形解码完成"
                ),
            }
            stage = stages.get(key)
            if stage is not None:
                return _Stage(
                    max(current_progress, stage.progress), stage.name, stage.message
                )
        if "denois" in normalized or "sampling" in normalized:
            progress = max(current_progress, 55.0)
            match = self._STEP_RE.search(normalized)
            if match:
                current_step = int(match.group("current"))
                total_steps = max(1, int(match.group("total")))
                progress = max(progress, min(80.0, 40.0 + 40.0 * current_step / total_steps))
                return _Stage(progress, "denoising", f"正在执行蒸馏采样 {current_step}/{total_steps}")
            return _Stage(progress, "denoising", "正在执行 FastH3 蒸馏采样")
        if "generated successfully" in normalized:
            return _Stage(max(current_progress, 92.0), "postprocessing", "视频和音频生成完成，正在整理输出")
        if "postdecode" in normalized:
            return _Stage(max(current_progress, 94.0), "postprocessing", "正在整理已解码的视频帧")
        if "wrote frames to ffmpeg" in normalized:
            return _Stage(max(current_progress, 96.0), "encoding", "视频帧编码完成，正在封装音频")
        if "saved video to" in normalized or "output written to:" in normalized:
            if current_progress >= 98.0:
                return _Stage(current_progress, "muxing", "音视频封装完成，正在校验输出")
            return _Stage(max(current_progress, 97.0), "encoding", "视频文件已经写入，正在完成音视频封装")
        if "audiomuxstage completed" in normalized:
            return _Stage(max(current_progress, 98.0), "muxing", "音视频封装完成")
        return None

    @staticmethod
    async def _terminate(process: asyncio.subprocess.Process) -> None:
        await _terminate_process(process)

class FastVideoBaseFL2VABackend:
    """Run Base MiniMax H3 FL2VA through a reusable FastVideo worker."""

    name = "fastvideo-native-fl2va"
    _PROGRESS_PREFIX = "MCP_PROGRESS "

    def __init__(
        self,
        settings: Settings,
        admission: ResourceAdmission | None = None,
    ):
        self.settings = settings
        self.runner = Path(__file__).with_name("h3_fastvideo_base_runner.py")
        self._line_classifier = FastVideoCliBackend(settings)
        self._resource_admission = admission or ResourceAdmission(settings)
        self._worker = _PersistentWorker(
            settings,
            self._resource_admission,
            "minimax-h3-base-fl2va",
        )

    def health(self) -> dict[str, object]:
        local_model = Path(self.settings.minimax_h3_model_path).expanduser()
        local_lora = Path(self.settings.minimax_h3_lora_path).expanduser()
        checks: dict[str, bool | None] = {
            "python_exists": self.settings.minimax_h3_python.is_file(),
            "runner_exists": self.runner.is_file(),
            "fastvideo_repo_exists": self.settings.fastvideo_repo.is_dir(),
            "model_path_exists": (
                local_model.exists() if local_model.is_absolute() else None
            ),
            "lora_path_exists": (
                local_lora.is_file() if local_lora.is_absolute() else None
            ),
        }
        if local_model.is_dir():
            for component in (
                "transformer",
                "text_encoder",
                "vae",
                "audio_vae",
                "modular_model_index.json",
            ):
                checks[f"model_{component}_exists"] = (local_model / component).exists()
        return {
            "backend": self.name,
            "ready": all(value is not False for value in checks.values()),
            "checks": checks,
            "model_id": "minimax-h3-base-fl2va-turbo-8step",
            "model_path": self.settings.minimax_h3_model_path,
            "lora_path": self.settings.minimax_h3_lora_path,
            "lora_alpha": self.settings.minimax_h3_lora_alpha,
            "lora_strength": self.settings.minimax_h3_lora_strength,
            "inference_steps": self.settings.minimax_h3_steps,
            "video_scheduler_shift": 12,
            "audio_scheduler_shift": 3,
            "attention_backend": "TORCH_SDPA",
            "text_encoder_fp8": self.settings.minimax_h3_text_encoder_fp8,
            "direct_device_load": self.settings.minimax_h3_direct_device_load,
            "supports_first_frame": True,
            "supports_last_frame": True,
            "depends_on_comfyui": False,
            "warm_worker": self._worker.health(),
        }

    @property
    def has_warm_worker(self) -> bool:
        return self._worker.is_warm

    async def shutdown(self) -> None:
        await self._worker.shutdown()

    def build_environment(self) -> dict[str, str]:
        environment = os.environ.copy()
        environment["PYTHONUNBUFFERED"] = "1"
        inherited_pythonpath = environment.get("PYTHONPATH", "")
        kernel_python = self.settings.fastvideo_repo / "fastvideo-kernel" / "python"
        environment["PYTHONPATH"] = os.pathsep.join(
            part
            for part in (
                str(kernel_python) if kernel_python.is_dir() else "",
                str(self.settings.fastvideo_repo),
                inherited_pythonpath,
            )
            if part
        )
        # Base FL2VA was trained with dense attention. Never inherit FastH3's
        # optional VSA settings into this independent model process.
        environment["FASTVIDEO_ATTENTION_BACKEND"] = "TORCH_SDPA"
        environment["FASTVIDEO_FA4"] = "0"
        environment["FASTVIDEO_VSA_SM100A"] = "0"
        environment["FASTVIDEO_VSA_CUTEDSL"] = "0"
        environment["FASTVIDEO_VSA_TRITON"] = "0"
        environment["FASTVIDEO_VSA_CACHE_TILE_BUFFER"] = "0"
        environment["FASTVIDEO_VSA_ATTENTION_RESIDUAL_CHUNK_TOKENS"] = "0"
        environment["FASTVIDEO_LOAD_WEIGHTS_DIRECT_TO_DEVICE"] = (
            "1" if self.settings.minimax_h3_direct_device_load else "0"
        )
        environment["FASTVIDEO_MINIMAX_H3_RELEASE_DIT_AFTER_DENOISE"] = (
            "1" if self.settings.minimax_h3_release_dit_after_denoise else "0"
        )
        environment["FASTVIDEO_MINIMAX_H3_FF_CHUNK_TOKENS"] = str(
            self.settings.minimax_h3_ff_chunk_tokens
        )
        # This backend is a one-shot worker and never swaps or unmerges LoRA.
        # Avoid duplicating the full Base DiT in system RAM just for rollback.
        environment["FASTVIDEO_LORA_CPU_BACKUP"] = "0"
        # Distilled turbo inference is valid only when every tensor in the
        # FL2VA adapter reaches its matching transformer layer. Never silently
        # fall back to running the undistilled Base model with few steps.
        environment["FASTVIDEO_REQUIRE_COMPLETE_LORA"] = "1"
        environment["FASTVIDEO_MCP_PROGRESS"] = "1"
        return environment

    def request_payload(
        self, request: GenerationRequest, output: Path
    ) -> dict[str, object]:
        warm_safe = (
            request.duration_seconds
            <= getattr(self.settings, "gpu_worker_warm_max_duration_seconds", 30)
        )
        return {
            "model_path": self.settings.minimax_h3_model_path,
            "lora_path": self.settings.minimax_h3_lora_path,
            "lora_alpha": self.settings.minimax_h3_lora_alpha,
            "lora_strength": self.settings.minimax_h3_lora_strength,
            "prompt": request.model_prompt,
            "first_frame_path": request.first_frame_path,
            "last_frame_path": request.last_frame_path,
            "width": request.width,
            "height": request.height,
            "num_frames": request.num_frames,
            "fps": request.fps,
            "steps": self.settings.minimax_h3_steps,
            "seed": request.seed,
            "text_encoder_fp8": self.settings.minimax_h3_text_encoder_fp8,
            "num_gpus": self.settings.num_gpus,
            "release_dit_after_denoise": (
                self.settings.minimax_h3_release_dit_after_denoise
                and not warm_safe
            ),
            "output_path": str(output),
        }

    async def generate(
        self,
        request: GenerationRequest,
        job_dir: Path,
        report: ProgressReporter,
        cancel_event: asyncio.Event,
    ) -> Path:
        if not request.has_keyframes:
            raise ValueError("Base FL2VA requires a first or last reference frame")
        health = self.health()
        if not health["ready"]:
            raise RuntimeError(
                f"FastVideo Base FL2VA backend is not ready: {health['checks']}"
            )
        job_dir.mkdir(parents=True, exist_ok=False)
        output = job_dir / "result.mp4"
        environment = self.build_environment()
        command = [
            str(self.settings.minimax_h3_python),
            str(self.runner),
            "--server",
        ]

        def classify(line: str, current_progress: float) -> _Stage | None:
            if line.startswith(self._PROGRESS_PREFIX):
                try:
                    payload = json.loads(line[len(self._PROGRESS_PREFIX) :])
                    return _Stage(
                        float(payload["progress"]),
                        str(payload["stage"]),
                        str(payload["message"]),
                    )
                except (ValueError, KeyError, TypeError, json.JSONDecodeError):
                    return None
            classified = self._line_classifier._classify_line(
                line, current_progress
            )
            if classified is None:
                return None
            return _Stage(
                classified.progress,
                classified.name,
                classified.message.replace("FastH3", "Base FL2VA"),
            )

        await self._worker.run(
            command=command,
            cwd=self.settings.fastvideo_repo,
            environment=environment,
            payload=self.request_payload(request, output),
            job_dir=job_dir,
            report=report,
            cancel_event=cancel_event,
            initial_stage=_Stage(
                10.0, "loading_model", "Base FL2VA GPU Worker 已接收任务"
            ),
            classify_line=classify,
        )
        if not output.is_file() or output.stat().st_size == 0:
            raise RuntimeError("FastVideo Base FL2VA produced no usable MP4 file")
        if not request.audio_enabled:
            await report(98.0, "removing_audio", "正在按请求移除输出音轨")
            await _remove_audio_track(self.settings.ffmpeg_binary, output)
        await report(99.0, "verifying", "FastVideo Base FL2VA 输出完成，正在校验结果")
        return output


class MiniMaxH3DiffusersBackend:
    """Base MiniMax H3 FL2VA backend for text and first/last keyframes."""

    name = "minimax-h3-diffusers"
    _PROGRESS_PREFIX = "MCP_PROGRESS "

    def __init__(self, settings: Settings):
        self.settings = settings
        self.runner = Path(__file__).with_name("h3_diffusers_runner.py")

    def health(self) -> dict[str, object]:
        local_model = Path(self.settings.minimax_h3_model_path).expanduser()
        checks = {
            "python_exists": self.settings.minimax_h3_python.is_file(),
            "runner_exists": self.runner.is_file(),
            "model_path_exists": local_model.exists() if local_model.is_absolute() else None,
        }
        return {
            "backend": self.name,
            "ready": all(value is not False for value in checks.values()),
            "checks": checks,
            "model_id": "minimax-h3-base-fl2va",
            "model_path": self.settings.minimax_h3_model_path,
            "supports_first_frame": True,
            "supports_last_frame": True,
        }

    async def generate(
        self,
        request: GenerationRequest,
        job_dir: Path,
        report: ProgressReporter,
        cancel_event: asyncio.Event,
    ) -> Path:
        health = self.health()
        if not health["ready"]:
            raise RuntimeError(f"MiniMax H3 Base backend is not ready: {health['checks']}")
        job_dir.mkdir(parents=True, exist_ok=False)
        output = job_dir / "result.mp4"
        request_file = job_dir / "request.json"
        request_file.write_text(
            json.dumps(
                {
                    "model_path": self.settings.minimax_h3_model_path,
                    "prompt": request.model_prompt,
                    "first_frame_path": request.first_frame_path,
                    "last_frame_path": request.last_frame_path,
                    "width": request.width,
                    "height": request.height,
                    "num_frames": request.num_frames,
                    "steps": self.settings.minimax_h3_legacy_steps,
                    "seed": request.seed,
                    "audio_enabled": request.audio_enabled,
                    "output_path": str(output),
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        await report(8.0, "starting_backend", "正在启动 MiniMax H3 Base FL2VA")
        process = await asyncio.create_subprocess_exec(
            str(self.settings.minimax_h3_python),
            str(self.runner),
            "--request-json",
            str(request_file),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            start_new_session=True,
        )
        assert process.stdout is not None
        current = _Stage(10.0, "loading_model", "正在加载 MiniMax H3 Base FL2VA")
        logs: list[str] = []
        await report(current.progress, current.name, current.message)
        try:
            while True:
                if cancel_event.is_set():
                    await FastVideoCliBackend._terminate(process)
                    raise BackendCanceled("job canceled")
                try:
                    raw = await asyncio.wait_for(
                        process.stdout.readline(),
                        timeout=self.settings.progress_heartbeat_seconds,
                    )
                except TimeoutError:
                    continue
                if not raw:
                    break
                line = raw.decode("utf-8", errors="replace").rstrip()
                logs.append(line)
                if line.startswith(self._PROGRESS_PREFIX):
                    try:
                        payload = json.loads(line[len(self._PROGRESS_PREFIX) :])
                        current = _Stage(
                            float(payload["progress"]),
                            str(payload["stage"]),
                            str(payload["message"]),
                        )
                    except (ValueError, KeyError, TypeError, json.JSONDecodeError):
                        continue
                    await report(current.progress, current.name, current.message)
            return_code = await process.wait()
        except asyncio.CancelledError:
            await FastVideoCliBackend._terminate(process)
            raise
        finally:
            (job_dir / "backend.log").write_text("\n".join(logs)[-1_000_000:], encoding="utf-8")
        if return_code != 0:
            raise RuntimeError(
                f"MiniMax H3 runner exited with code {return_code}\n"
                + "\n".join(logs[-20:])
            )
        if not output.is_file() or output.stat().st_size == 0:
            raise RuntimeError("MiniMax H3 runner produced no usable MP4 file")
        return output


def render_workflow_template(
    template: object, replacements: dict[str, object]
) -> object:
    """Render explicit placeholders without evaluating code or templates."""
    if isinstance(template, dict):
        return {
            key: render_workflow_template(value, replacements)
            for key, value in template.items()
        }
    if isinstance(template, list):
        return [render_workflow_template(value, replacements) for value in template]
    if not isinstance(template, str):
        return template
    if template in replacements:
        return replacements[template]
    rendered = template
    for placeholder, value in replacements.items():
        rendered = rendered.replace(placeholder, str(value))
    return rendered


class ComfyUIBackend:
    """Submit API-format MiniMax H3 workflows to an existing ComfyUI instance."""

    name = "comfyui-api"
    _VIDEO_EXTENSIONS = {".mp4"}

    def __init__(self, settings: Settings):
        self.settings = settings

    def health(self) -> dict[str, object]:
        checks = {
            "models_dir_exists": self.settings.comfyui_models_dir.is_dir(),
            "t2va_workflow_exists": self.settings.comfyui_t2va_workflow_template.is_file(),
            "first_frame_workflow_exists": self.settings.comfyui_first_frame_workflow_template.is_file(),
            "last_frame_workflow_exists": self.settings.comfyui_last_frame_workflow_template.is_file(),
            "first_last_workflow_exists": self.settings.comfyui_first_last_workflow_template.is_file(),
        }
        return {
            "backend": self.name,
            "ready": all(checks.values()),
            "checks": checks,
            "base_url": self.settings.comfyui_base_url,
            "models_dir": str(self.settings.comfyui_models_dir),
            "supports_first_frame": True,
            "supports_last_frame": True,
            "unload_after_job": self.settings.comfyui_unload_after_job,
            "network_checked": False,
        }

    async def generate(
        self,
        request: GenerationRequest,
        job_dir: Path,
        report: ProgressReporter,
        cancel_event: asyncio.Event,
    ) -> Path:
        template_path = self._select_workflow(request)
        if not self.settings.comfyui_models_dir.is_dir():
            raise RuntimeError(
                f"ComfyUI models directory does not exist: {self.settings.comfyui_models_dir}"
            )
        if not template_path.is_file():
            raise RuntimeError(
                "ComfyUI API workflow template not found: " + str(template_path)
            )
        try:
            template = json.loads(template_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"invalid ComfyUI workflow template: {template_path}") from exc
        if not isinstance(template, dict):
            raise RuntimeError("ComfyUI API workflow template must be a JSON object")

        job_dir.mkdir(parents=True, exist_ok=False)
        headers = self._headers()
        timeout = aiohttp.ClientTimeout(
            total=None,
            connect=30,
            sock_connect=30,
            sock_read=None,
        )
        client_id = uuid.uuid4().hex
        prompt_id: str | None = None
        started = False
        await report(8.0, "connecting_comfyui", "正在连接 ComfyUI Base 服务")

        async with aiohttp.ClientSession(headers=headers, timeout=timeout) as session:
            first_name = await self._upload_image(
                session, request.first_frame_path, "first"
            )
            last_name = await self._upload_image(
                session, request.last_frame_path, "last"
            )
            workflow = self._render_workflow(
                template, request, first_name=first_name, last_name=last_name
            )
            (job_dir / "workflow.rendered.json").write_text(
                json.dumps(workflow, ensure_ascii=False, indent=2), encoding="utf-8"
            )

            ws_url = self._websocket_url(client_id)
            await report(12.0, "submitting", "正在向 ComfyUI 提交工作流")
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
                        raise RuntimeError(f"ComfyUI did not return prompt_id: {response}")
                    await report(18.0, "queued_in_comfyui", "工作流已进入 ComfyUI 队列")
                    started = await self._wait_for_completion(
                        session,
                        websocket,
                        prompt_id,
                        report,
                        cancel_event,
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
            await self._maybe_unload_models(session, report)

        if not output.is_file() or output.stat().st_size == 0:
            raise RuntimeError("ComfyUI produced no usable MP4 file")
        if not request.audio_enabled:
            await report(97.0, "removing_audio", "正在按请求移除输出音轨")
            await _remove_audio_track(self.settings.ffmpeg_binary, output)
        await report(98.0, "verifying", "ComfyUI 输出完成，正在校验结果")
        return output

    async def _maybe_unload_models(
        self,
        session: aiohttp.ClientSession,
        report: ProgressReporter,
    ) -> None:
        if not self.settings.comfyui_unload_after_job:
            return
        queue = await self._json_request(session, "GET", "/queue")
        running = queue.get("queue_running")
        pending = queue.get("queue_pending")
        if running or pending:
            await report(
                96.0,
                "retaining_models",
                "ComfyUI 仍有其他任务，暂不卸载共享模型",
            )
            return
        await report(96.0, "freeing_vram", "正在卸载 ComfyUI 模型并释放显存")
        await self._json_request(
            session,
            "POST",
            "/free",
            json={"unload_models": True, "free_memory": True},
        )

    def _select_workflow(self, request: GenerationRequest) -> Path:
        if request.first_frame_path and request.last_frame_path:
            return self.settings.comfyui_first_last_workflow_template
        if request.first_frame_path:
            return self.settings.comfyui_first_frame_workflow_template
        if request.last_frame_path:
            return self.settings.comfyui_last_frame_workflow_template
        return self.settings.comfyui_t2va_workflow_template

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
            (scheme, split.netloc, f"{base_path}/ws", urlencode({"clientId": client_id}), "")
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
            content_type={".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp"}.get(source.suffix.lower(), "application/octet-stream"),
        )
        form.add_field("type", "input")
        form.add_field("subfolder", "fastvideo_h3_mcp")
        form.add_field("overwrite", "false")
        response = await self._json_request(
            session, "POST", "/upload/image", data=form
        )
        name = str(response.get("name", "")).strip()
        subfolder = str(response.get("subfolder", "")).strip().strip("/\\")
        if not name or Path(name).name != name:
            raise RuntimeError(f"ComfyUI returned invalid uploaded image name: {response}")
        normalized_subfolder = subfolder.replace("\\", "/")
        if normalized_subfolder and (
            normalized_subfolder.startswith("/")
            or ".." in normalized_subfolder.split("/")
        ):
            raise RuntimeError(f"ComfyUI returned invalid upload subfolder: {response}")
        return f"{normalized_subfolder}/{name}" if normalized_subfolder else name

    def _render_workflow(
        self,
        template: dict[str, object],
        request: GenerationRequest,
        *,
        first_name: str,
        last_name: str,
    ) -> dict[str, object]:
        prepared = copy.deepcopy(template)
        if not first_name:
            self._prune_optional_image(prepared, "{{FIRST_FRAME}}")
        if not last_name:
            self._prune_optional_image(prepared, "{{LAST_FRAME}}")
        replacements: dict[str, object] = {
            "{{POSITIVE_PROMPT}}": request.prompt,
            "{{NEGATIVE_PROMPT}}": request.negative_prompt or "",
            "{{H3_PROMPT}}": request.model_prompt,
            "{{SOUNDSCAPE}}": request.soundscape or "",
            "{{WIDTH}}": request.width,
            "{{HEIGHT}}": request.height,
            "{{NUM_FRAMES}}": request.num_frames,
            "{{DURATION_SECONDS}}": request.duration_seconds,
            "{{SEED}}": request.seed,
            "{{STEPS}}": self.settings.minimax_h3_legacy_steps,
            "{{AUDIO_ENABLED}}": request.audio_enabled,
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
    def _prune_optional_image(
        workflow: dict[str, object], placeholder: str
    ) -> None:
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

    async def _wait_for_completion(
        self,
        session: aiohttp.ClientSession,
        websocket: aiohttp.ClientWebSocketResponse,
        prompt_id: str,
        report: ProgressReporter,
        cancel_event: asyncio.Event,
    ) -> bool:
        started = False
        current_progress = 20.0
        current_stage = "waiting_comfyui"
        current_message = "等待 ComfyUI 执行工作流"
        while True:
            if cancel_event.is_set():
                await self._cancel_prompt(session, prompt_id, started)
                raise BackendCanceled("job canceled")
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
                current_stage = "comfyui_started"
                current_message = "ComfyUI 已开始执行 Base 工作流"
                await report(current_progress, current_stage, current_message)
                continue
            if event_type == "executing":
                node = data.get("node")
                if node is None:
                    await report(93.0, "comfyui_completed", "ComfyUI 工作流执行完成")
                    return started
                started = True
                current_progress = max(current_progress, min(85.0, current_progress + 1.0))
                current_stage = "comfyui_node"
                current_message = f"ComfyUI 正在执行节点 {node}"
                await report(current_progress, current_stage, current_message)
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
                current_stage = "sampling"
                current_message = f"ComfyUI 节点 {node} 采样进度 {int(value)}/{int(maximum)}"
                await report(current_progress, current_stage, current_message)
                continue
            if event_type in {"execution_success", "execution_complete"}:
                await report(93.0, "comfyui_completed", "ComfyUI 工作流执行完成")
                return started

    async def _cancel_prompt(
        self,
        session: aiohttp.ClientSession,
        prompt_id: str,
        started: bool,
    ) -> None:
        try:
            await self._json_request(
                session, "POST", "/queue", json={"delete": [prompt_id]}
            )
        except Exception:
            pass
        if started and self.settings.comfyui_allow_interrupt:
            try:
                await self._json_request(session, "POST", "/interrupt", json={})
            except Exception:
                pass

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
                async for chunk in response.content.iter_chunked(1024 * 1024):
                    target.write(chunk)


class ResourceGuardedBackend:
    def __init__(self, backend: VideoBackend, admission: ResourceAdmission):
        self.backend = backend
        self.admission = admission
        self.name = backend.name

    def health(self) -> dict[str, object]:
        result = dict(self.backend.health())
        result["resource_admission"] = self.admission.health()
        return result

    async def generate(
        self,
        request: GenerationRequest,
        job_dir: Path,
        report: ProgressReporter,
        cancel_event: asyncio.Event,
    ) -> Path:
        await self.admission.wait(report, cancel_event)
        return await self.backend.generate(request, job_dir, report, cancel_event)


class BackendRouter:
    name = "auto"

    def __init__(
        self,
        fast: FastVideoCliBackend,
        base: VideoBackend,
        admission: ResourceAdmission | None = None,
    ):
        self.fast = fast
        self.base = base
        self.admission = admission
        self._model_lock = asyncio.Lock()

    def health(self) -> dict[str, object]:
        result: dict[str, object] = {
            "backend": self.name,
            "ready": bool(self.fast.health()["ready"] or self.base.health()["ready"]),
            "fast": self.fast.health(),
            "base_fl2va": self.base.health(),
        }
        if self.admission is not None:
            result["resource_admission"] = self.admission.health()
        return result

    async def generate(
        self,
        request: GenerationRequest,
        job_dir: Path,
        report: ProgressReporter,
        cancel_event: asyncio.Event,
    ) -> Path:
        async with self._model_lock:
            backend = self._select(request)
            await report(6.0, "routing", f"任务已路由到 {backend.name}")
            other = self.base if backend is self.fast else self.fast
            if bool(getattr(other, "has_warm_worker", False)):
                await report(
                    7.0,
                    "switching_model",
                    "任务需要另一模型，正在释放已加载的旧模型",
                )
                shutdown = getattr(other, "shutdown", None)
                if shutdown is not None:
                    await shutdown()
            if (
                self.admission is not None
                and backend.name != "comfyui-api"
                and not isinstance(
                    backend, (FastVideoCliBackend, FastVideoBaseFL2VABackend)
                )
            ):
                await self.admission.wait(report, cancel_event)
            return await backend.generate(request, job_dir, report, cancel_event)

    async def shutdown(self) -> None:
        async with self._model_lock:
            for backend in (self.fast, self.base):
                shutdown = getattr(backend, "shutdown", None)
                if shutdown is not None:
                    await shutdown()

    def _select(self, request: GenerationRequest) -> VideoBackend:
        known_model_ids = {
            "auto",
            "fast",
            self.fast.settings.model_id,
            "base",
            "minimax-h3-base-fl2va",
        }
        if request.model_id not in known_model_ids:
            raise ValueError(f"unknown model_id: {request.model_id}")
        # Routing is modality-based, not resolution-based. FastH3 accepts the
        # requested text-to-video canvas; Base is reserved for reference-frame
        # generation because the distilled checkpoint has no keyframe input.
        if request.has_keyframes:
            return self.base
        return self.fast


def build_backend(settings: Settings) -> VideoBackend:
    if settings.backend == "mock":
        return MockBackend()
    admission = ResourceAdmission(settings)
    if settings.backend == "fastvideo-cli":
        return FastVideoCliBackend(settings, admission)
    if settings.backend == "fastvideo-native":
        return FastVideoBaseFL2VABackend(settings, admission)
    if settings.backend == "minimax-h3-diffusers":
        return ResourceGuardedBackend(MiniMaxH3DiffusersBackend(settings), admission)
    if settings.backend == "comfyui-api":
        return ComfyUIBackend(settings)
    base: VideoBackend
    if settings.base_backend == "fastvideo-native":
        base = FastVideoBaseFL2VABackend(settings, admission)
    elif settings.base_backend == "comfyui-api":
        base = ComfyUIBackend(settings)
    else:
        base = MiniMaxH3DiffusersBackend(settings)
    return BackendRouter(FastVideoCliBackend(settings, admission), base, admission)
