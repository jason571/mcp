from __future__ import annotations

import asyncio
import hashlib
import json
import math
import re
import sys
from array import array
from pathlib import Path
from typing import Any, Awaitable, Callable
from urllib.parse import urlsplit

import aiohttp
import anyio

from .config import Settings


Progress = Callable[[float, str], Awaitable[None]]
ATTACHMENT_ID = re.compile(r"^att_[0-9a-f]{64}$")
I2VA_PROMPT_PREFIX = (
    "For the target video, at 0.00 seconds into the target video, "
    "<Picture 1> (from [Shot 1]) is fully referenced.\n\n"
)


def bind_opening_frame(
    *,
    attachment_id: str | None,
    url: str | None,
    allowed_origins: tuple[str, ...],
) -> dict[str, str]:
    """Persist at most one user still for the first segment's image-to-video job."""
    attachment_id = (attachment_id or "").strip() or None
    url = (url or "").strip() or None
    if attachment_id and url:
        raise ValueError("supply exactly one of first_frame_attachment_id or first_frame_url")
    if not attachment_id and not url:
        return {}
    if attachment_id:
        if not ATTACHMENT_ID.fullmatch(attachment_id):
            raise ValueError("first_frame_attachment_id is invalid")
        return {"first_frame_attachment_id": attachment_id}
    if "/v1/run-artifacts/" in url:
        raise ValueError(
            "chat download_url (/v1/run-artifacts/) cannot be fetched by the media "
            "service; pass first_frame_attachment_id. First segment stays text-to-video"
        )
    validate_source_url(url, allowed_origins)
    return {"first_frame_url": url}


def opening_frame_fetch_url(source: dict[str, Any], settings: Settings) -> str | None:
    attachment_id = source.get("first_frame_attachment_id")
    if isinstance(attachment_id, str) and attachment_id:
        base = (getattr(settings, "asset_base", None) or "http://127.0.0.1:8787").rstrip("/")
        return f"{base}/v1/attachment-assets/{attachment_id}"
    url = source.get("first_frame_url")
    return url if isinstance(url, str) and url else None


def bind_reference_image(
    *, attachment_id: str | None, url: str | None,
    allowed_origins: tuple[str, ...],
) -> dict[str, str]:
    """Persist a visual reference without assigning it the first-frame role."""
    attachment_id = (attachment_id or "").strip() or None
    url = (url or "").strip() or None
    if attachment_id and url:
        raise ValueError("supply exactly one of reference_image_attachment_id or reference_image_url")
    if attachment_id:
        if not ATTACHMENT_ID.fullmatch(attachment_id):
            raise ValueError("reference_image_attachment_id is invalid")
        return {"reference_image_attachment_id": attachment_id}
    if url:
        if "/v1/run-artifacts/" in url:
            raise ValueError("chat download URL cannot be used as a reference image; pass its attachment ID")
        validate_source_url(url, allowed_origins)
        return {"reference_image_url": url}
    return {}


def reference_image_fetch_url(source: dict[str, Any], settings: Settings) -> str | None:
    return opening_frame_fetch_url({
        "first_frame_attachment_id": source.get("reference_image_attachment_id"),
        "first_frame_url": source.get("reference_image_url"),
    }, settings)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_source_url(url: str, allowed_origins: tuple[str, ...]) -> str:
    parsed = urlsplit(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("media source URL must use http or https")
    if parsed.username or parsed.password or parsed.fragment:
        raise ValueError("media source URL must not contain credentials or a fragment")
    default_port = 443 if parsed.scheme == "https" else 80
    origin = f"{parsed.scheme}://{parsed.hostname.lower()}:{parsed.port or default_port}"
    normalized_allowed = {
        normalize_origin(item) for item in allowed_origins if item.strip()
    }
    if origin not in normalized_allowed:
        raise ValueError(f"media source origin is not allowed: {origin}")
    return url


def normalize_origin(value: str) -> str:
    parsed = urlsplit(value)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError(f"invalid allowed media origin: {value!r}")
    default_port = 443 if parsed.scheme == "https" else 80
    return f"{parsed.scheme}://{parsed.hostname.lower()}:{parsed.port or default_port}"


async def download_source(
    session: aiohttp.ClientSession,
    settings: Settings,
    url: str,
    target: Path,
) -> Path:
    validate_source_url(url, settings.allowed_fetch_origins)
    size = 0
    async with session.get(url, allow_redirects=False) as response:
        if response.status != 200:
            raise RuntimeError(f"media download returned HTTP {response.status}")
        content_length = response.headers.get("Content-Length")
        if content_length and int(content_length) > settings.max_download_bytes:
            raise RuntimeError("media download exceeds configured size limit")
        with target.open("xb") as output:
            async for chunk in response.content.iter_chunked(1024 * 1024):
                size += len(chunk)
                if size > settings.max_download_bytes:
                    raise RuntimeError("media download exceeds configured size limit")
                output.write(chunk)
    return target


async def _cancel_process(process: asyncio.subprocess.Process) -> None:
    # Reap only the subprocess owned by this request, including when the MCP
    # dispatcher has already cancelled its surrounding AnyIO scope.
    with anyio.move_on_after(3, shield=True):
        if process.returncode is None:
            try:
                process.kill()
            except ProcessLookupError:
                pass
        await process.wait()


async def run_process(settings: Settings, command: list[str]) -> None:
    process = await asyncio.create_subprocess_exec(
        *command,
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        _stdout, stderr = await asyncio.wait_for(
            process.communicate(), timeout=settings.operation_timeout_seconds
        )
    except TimeoutError:
        process.kill()
        await process.wait()
        raise RuntimeError("media operation timed out") from None
    except asyncio.CancelledError:
        await _cancel_process(process)
        raise
    if process.returncode != 0:
        detail = stderr.decode("utf-8", errors="replace")[-4000:]
        raise RuntimeError(f"media command failed: {detail}")


async def probe_file(settings: Settings, path: Path) -> dict[str, Any]:
    process = await asyncio.create_subprocess_exec(
        settings.ffprobe_binary,
        "-v",
        "error",
        "-show_entries",
        "format=duration,format_name,size,bit_rate:stream=index,codec_type,codec_name,width,height,r_frame_rate,sample_rate,channels",
        "-of",
        "json",
        str(path),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(
            process.communicate(), timeout=settings.operation_timeout_seconds
        )
    except TimeoutError:
        process.kill()
        await process.wait()
        raise RuntimeError("ffprobe timed out") from None
    except asyncio.CancelledError:
        await _cancel_process(process)
        raise
    if process.returncode != 0:
        raise RuntimeError(
            "ffprobe failed: " + stderr.decode("utf-8", errors="replace")[-4000:]
        )
    payload = json.loads(stdout)
    duration = float(payload.get("format", {}).get("duration", 0))
    if not math.isfinite(duration) or duration <= 0:
        raise RuntimeError("media duration is missing or invalid")
    payload["duration_seconds"] = duration
    return payload


async def analyze_audio_energy(
    settings: Settings, path: Path, requested_window_seconds: float
) -> dict[str, Any]:
    if not 0.25 <= requested_window_seconds <= 5.0:
        raise ValueError("window_seconds must be between 0.25 and 5")
    probe = await probe_file(settings, path)
    duration = float(probe["duration_seconds"])
    effective_window = max(requested_window_seconds, duration / 1200.0)
    sample_rate = 100
    process = await asyncio.create_subprocess_exec(
        settings.ffmpeg_binary,
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-f",
        "f32le",
        "pipe:1",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(
            process.communicate(), timeout=settings.operation_timeout_seconds
        )
    except TimeoutError:
        process.kill()
        await process.wait()
        raise RuntimeError("audio analysis timed out") from None
    except asyncio.CancelledError:
        await _cancel_process(process)
        raise
    if process.returncode != 0:
        raise RuntimeError(
            "audio decode failed: " + stderr.decode("utf-8", errors="replace")[-4000:]
        )
    samples = array("f")
    samples.frombytes(stdout[: len(stdout) - len(stdout) % 4])
    if sys.byteorder != "little":
        samples.byteswap()
    if not samples:
        raise RuntimeError("audio decode returned no samples")
    window_samples = max(1, round(effective_window * sample_rate))
    raw_energy: list[float] = []
    for offset in range(0, len(samples), window_samples):
        window = samples[offset : offset + window_samples]
        if not window:
            continue
        rms = math.sqrt(sum(float(value) ** 2 for value in window) / len(window))
        raw_energy.append(rms)
    maximum = max(raw_energy) or 1.0
    normalized = [value / maximum for value in raw_energy]
    curve = []
    for index, energy in enumerate(normalized):
        start = index * effective_window
        end = min(duration, start + effective_window)
        previous = normalized[index - 1] if index else energy
        curve.append(
            {
                "start_ms": round(start * 1000),
                "end_ms": round(end * 1000),
                "energy": round(energy, 4),
                "rise": round(max(0.0, energy - previous), 4),
            }
        )
    ranked = sorted(
        range(len(curve)),
        key=lambda index: curve[index]["energy"] * 0.75
        + curve[index]["rise"] * 0.25,
        reverse=True,
    )
    selected: list[int] = []
    for index in ranked:
        if all(abs(index - other) > 1 for other in selected):
            selected.append(index)
        if len(selected) >= 12:
            break
    peak_windows = [curve[index] for index in sorted(selected)]
    transitions = sorted(curve[1:], key=lambda item: item["rise"], reverse=True)[:12]
    return {
        "duration_seconds": duration,
        "analysis": "decoded mono RMS energy and positive energy change; no lyric, BPM, beat, or emotion inference",
        "requested_window_seconds": requested_window_seconds,
        "effective_window_seconds": effective_window,
        "energy_curve": curve,
        "peak_windows": peak_windows,
        "rising_windows": sorted(transitions, key=lambda item: item["start_ms"]),
    }


def primary_video_stream(probe: dict[str, Any]) -> dict[str, Any]:
    stream = next(
        (item for item in probe.get("streams", []) if item.get("codec_type") == "video"),
        None,
    )
    if not isinstance(stream, dict):
        raise RuntimeError("media has no video stream")
    return stream


async def extract_last_frame_file(
    settings: Settings, video: Path, output: Path
) -> dict[str, Any]:
    probe = await probe_file(settings, video)
    stream = primary_video_stream(probe)
    duration = float(probe["duration_seconds"])
    seek = max(0.0, duration - 0.05)
    await run_process(
        settings,
        [
            settings.ffmpeg_binary,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-ss",
            f"{seek:.6f}",
            "-i",
            str(video),
            "-frames:v",
            "1",
            "-compression_level",
            "6",
            str(output),
        ],
    )
    if not output.is_file() or output.stat().st_size == 0:
        raise RuntimeError("last-frame extraction produced no image")
    return {
        "source_duration_seconds": duration,
        "width": stream.get("width"),
        "height": stream.get("height"),
        "position_seconds": seek,
    }


def even_dimension(value: Any, fallback: int) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        number = fallback
    if number < 2:
        number = fallback
    return number - number % 2


async def assemble_music_video_files(
    settings: Settings,
    clips: list[Path],
    audio: Path,
    work_dir: Path,
    output: Path,
    progress: Progress,
) -> dict[str, Any]:
    if not clips:
        raise ValueError("at least one video clip is required")
    audio_probe = await probe_file(settings, audio)
    audio_duration = float(audio_probe["duration_seconds"])
    clip_probes = [await probe_file(settings, clip) for clip in clips]
    first_stream = primary_video_stream(clip_probes[0])
    width = even_dimension(first_stream.get("width"), 896)
    height = even_dimension(first_stream.get("height"), 512)
    fps = 24
    normalized: list[Path] = []
    for index, clip in enumerate(clips):
        await progress(25.0 + 35.0 * index / len(clips), f"正在统一第 {index + 1}/{len(clips)} 段视频规格")
        target = work_dir / f"normalized-{index:04d}.mp4"
        filter_value = (
            f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
            f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:black,"
            f"fps={fps},setsar=1"
        )
        await run_process(
            settings,
            [
                settings.ffmpeg_binary,
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-i",
                str(clip),
                "-map",
                "0:v:0",
                "-an",
                "-vf",
                filter_value,
                "-c:v",
                "libx264",
                "-preset",
                "medium",
                "-crf",
                "18",
                "-pix_fmt",
                "yuv420p",
                str(target),
            ],
        )
        normalized.append(target)

    concat_file = work_dir / "concat.txt"
    concat_file.write_text(
        "".join(f"file '{path.as_posix()}'\n" for path in normalized),
        encoding="utf-8",
    )
    joined = work_dir / "joined.mp4"
    await progress(62.0, "正在按顺序拼接视频片段")
    await run_process(
        settings,
        [
            settings.ffmpeg_binary,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(concat_file),
            "-c",
            "copy",
            str(joined),
        ],
    )
    joined_probe = await probe_file(settings, joined)
    joined_duration = float(joined_probe["duration_seconds"])
    visual_master = joined
    if joined_duration + 0.01 < audio_duration:
        visual_master = work_dir / "visual-master.mp4"
        gap = audio_duration - joined_duration + 0.05
        await progress(76.0, "视频短于主音乐，正在保持末帧补齐时间轴")
        await run_process(
            settings,
            [
                settings.ffmpeg_binary,
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-i",
                str(joined),
                "-vf",
                f"tpad=stop_mode=clone:stop_duration={gap:.6f}",
                "-an",
                "-t",
                f"{audio_duration:.6f}",
                "-c:v",
                "libx264",
                "-preset",
                "medium",
                "-crf",
                "18",
                "-pix_fmt",
                "yuv420p",
                str(visual_master),
            ],
        )

    await progress(88.0, "正在混入原始音乐并封装最终 MV")
    await run_process(
        settings,
        [
            settings.ffmpeg_binary,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(visual_master),
            "-i",
            str(audio),
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-t",
            f"{audio_duration:.6f}",
            "-movflags",
            "+faststart",
            str(output),
        ],
    )
    result_probe = await probe_file(settings, output)
    result_stream = primary_video_stream(result_probe)
    has_audio = any(
        item.get("codec_type") == "audio" for item in result_probe.get("streams", [])
    )
    if not has_audio:
        raise RuntimeError("assembled MV has no audio stream")
    await progress(100.0, "MV 合成完成")
    return {
        "duration_seconds": float(result_probe["duration_seconds"]),
        "master_audio_duration_seconds": audio_duration,
        "source_video_duration_seconds": sum(
            float(item["duration_seconds"]) for item in clip_probes
        ),
        "width": result_stream.get("width"),
        "height": result_stream.get("height"),
        "fps": fps,
        "audio": True,
        "clip_count": len(clips),
    }
