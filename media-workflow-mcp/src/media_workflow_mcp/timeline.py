"""URL-referenced, timed media assembly; independent of prompt authoring/GPU jobs."""
from pydantic import BaseModel, ConfigDict, Field

from .media import probe_file, primary_video_stream, run_process


class TimelineSegment(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    video_url: str = Field(min_length=1, description="Exact completed source clip URL, not a local path")
    duration_seconds: float = Field(gt=0, le=3600, description="Exact planned window; short video is rejected, never looped or frozen")
    audio_url: str | None = Field(default=None, description="Optional separate narration audio URL. By default source video audio is discarded")
    preserve_source_audio: bool = Field(default=False, description="Keep this video's own audio when no separate audio_url is supplied. Reject source clips without an audio stream")
    audio_offset_seconds: float = Field(default=0, ge=0, description="Delay of this segment's narration within its own window")


async def assemble_timeline_files(settings, segments, clips, audios, directory, output, progress):
    if not segments or not (len(segments) == len(clips) == len(audios)):
        raise ValueError("timeline media count mismatch")
    fps, cursor, shape = 24, 0.0, None
    probes = []
    # Validate all sources before encoding any segment.
    for index, (segment, clip, audio) in enumerate(zip(segments, clips, audios)):
        duration = segment.duration_seconds
        if abs(duration * fps - round(duration * fps)) > .001:
            raise ValueError(f"segment {index + 1}: duration must align to {fps} fps frames")
        video = await probe_file(settings, clip)
        stream = primary_video_stream(video)
        if segment.preserve_source_audio and segment.audio_url is not None:
            raise ValueError(f"segment {index + 1}: choose source audio or separate audio_url, not both")
        if segment.preserve_source_audio and segment.audio_offset_seconds:
            raise ValueError(f"segment {index + 1}: audio_offset_seconds applies only to separate narration")
        if segment.preserve_source_audio and not any(s.get("codec_type") == "audio" for s in video["streams"]):
            raise ValueError(f"segment {index + 1}: source video has no audio stream")
        current = (stream["width"], stream["height"])
        if shape is not None and current != shape:
            raise ValueError(f"segment {index + 1}: video dimensions differ")
        shape = current
        if float(video["duration_seconds"]) + .001 < duration:
            raise ValueError(f"segment {index + 1}: video is shorter than its window; no freeze-fill")
        if audio is not None:
            speech = await probe_file(settings, audio)
            if not any(s.get("codec_type") == "audio" for s in speech["streams"]):
                raise ValueError(f"segment {index + 1}: narration has no audio stream")
            if float(speech["duration_seconds"]) + segment.audio_offset_seconds > duration + .001:
                raise ValueError(f"segment {index + 1}: narration exceeds its window; correct this audio only")
        elif segment.audio_url is not None:
            raise ValueError(f"segment {index + 1}: missing downloaded narration")
        cursor += duration
        probes.append({"segment": index + 1, "duration_seconds": duration})
    spoken = any(audio is not None for audio in audios)
    has_source_audio = any(segment.preserve_source_audio for segment in segments)
    has_audio = spoken or has_source_audio
    normalized = []
    for index, (segment, clip, audio) in enumerate(zip(segments, clips, audios)):
        await progress(20 + 60 * index / len(segments), f"正在对齐第 {index + 1}/{len(segments)} 段画面与解说")
        target = directory / f"segment-{index:04d}.mkv"
        command = [settings.ffmpeg_binary, "-nostdin", "-hide_banner", "-loglevel", "error", "-i", str(clip)]
        filters = f"[0:v:0]trim=duration={segment.duration_seconds},setpts=PTS-STARTPTS,fps={fps},format=yuv420p[v]"
        if has_audio:
            if segment.preserve_source_audio:
                audio_input = "[0:a:0]"
            elif audio is None:
                command += ["-f", "lavfi", "-i", "anullsrc=r=48000:cl=stereo"]
                audio_input = "[1:a:0]"
            else:
                command += ["-i", str(audio)]
                audio_input = "[1:a:0]"
            delay = round(segment.audio_offset_seconds * 1000)
            filters += f";{audio_input}aresample=48000,asetpts=PTS-STARTPTS,adelay={delay}:all=1,apad,atrim=duration={segment.duration_seconds}[a]"
        command += ["-filter_complex", filters, "-map", "[v]", "-c:v", "libx264", "-preset", "fast", "-crf", "20", "-threads", "2"]
        command += ["-map", "[a]", "-ac", "2", "-c:a", "pcm_s16le"] if has_audio else ["-an"]
        await run_process(settings, [*command, str(target)])
        normalized.append(target)
    listing = directory / "concat.txt"
    listing.write_text("".join(f"file '{p.name}'\n" for p in normalized))
    await progress(85, "正在拼接并校验完整时间轴")
    command = [settings.ffmpeg_binary, "-nostdin", "-hide_banner", "-loglevel", "error", "-f", "concat", "-safe", "1", "-i", str(listing), "-map", "0:v:0", "-c:v", "copy"]
    command += ["-map", "0:a:0", "-c:a", "aac", "-b:a", "128k"] if has_audio else ["-an"]
    await run_process(settings, [*command, "-movflags", "+faststart", str(output)])
    metadata = await probe_file(settings, output)
    if abs(float(metadata["duration_seconds"]) - cursor) > 1 / fps + .05:
        raise ValueError("assembled duration differs from the planned timeline")
    if has_audio and not any(s.get("codec_type") == "audio" for s in metadata["streams"]):
        raise ValueError("assembled video lost its audio stream")
    return {"duration_seconds": metadata["duration_seconds"], "segment_count": len(segments),
            "has_audio": has_audio, "has_narration": spoken, "has_source_audio": has_source_audio,
            "width": shape[0], "height": shape[1], "fps": fps,
            "timeline": probes}
