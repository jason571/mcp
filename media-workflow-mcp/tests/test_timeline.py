import asyncio
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from media_workflow_mcp import server
from media_workflow_mcp.artifacts import ArtifactStore
from media_workflow_mcp.config import Settings
from media_workflow_mcp.media import probe_file
from media_workflow_mcp.timeline import TimelineSegment, assemble_timeline_files


class TimelineTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.settings = Settings(host="127.0.0.1", port=8790, mcp_path="/mcp",
            public_base_url="https://media.example", allowed_hosts=(), allowed_origins=(),
            allowed_fetch_origins=("https://source.example",), api_token=None,
            artifact_signing_key="test-key", artifact_url_ttl_seconds=3600,
            data_dir=self.root, artifact_dir=self.root / "artifacts",
            ffmpeg_binary="ffmpeg", ffprobe_binary="ffprobe", max_download_bytes=512 * 1024 * 1024,
            operation_timeout_seconds=60)
        self.settings.prepare_directories()
        self.clips = [self.root / "red.mp4", self.root / "blue.mp4"]
        for color, path in zip(("red", "blue"), self.clips):
            subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-f", "lavfi", "-i",
                f"color=c={color}:s=160x90:r=24:d=1.25", "-an", "-c:v", "libx264", str(path)], check=True)
        self.audio = self.root / "speech.wav"
        subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-f", "lavfi", "-i",
            "sine=frequency=440:duration=0.25", str(self.audio)], check=True)
        self.segments = [TimelineSegment(video_url="https://source.example/red.mp4", duration_seconds=.5,
            audio_url="https://source.example/speech.wav", audio_offset_seconds=.1),
            TimelineSegment(video_url="https://source.example/blue.mp4", duration_seconds=1,
            audio_url="https://source.example/speech.wav", audio_offset_seconds=.5)]

    async def test_exact_segment_windows_and_separate_narration_reach_final_tool_artifact(self):
        sources = dict(zip([s.video_url for s in self.segments], self.clips))
        sources[self.segments[0].audio_url] = self.audio
        async def download(session, settings, url, target):
            shutil.copyfile(sources[url], target)
            return target
        runtime = SimpleNamespace(settings=self.settings, artifacts=ArtifactStore(self.settings))
        ctx = SimpleNamespace(report_progress=AsyncMock())
        with patch.object(server, "_require_tool_auth", return_value=runtime), patch.object(server, "download_source", side_effect=download) as fetch:
            result = await server.assemble_video_timeline(self.segments, ctx)
        self.assertEqual(fetch.await_count, 4)
        artifact = result["result"]
        self.assertEqual(artifact["segment_count"], 2)
        self.assertTrue(artifact["has_narration"])
        self.assertAlmostEqual(float(artifact["duration_seconds"]), 1.5, delta=.05)
        output = next(self.settings.artifact_dir.glob("*/final-video.mp4"))
        metadata = await probe_file(self.settings, output)
        self.assertTrue(any(s["codec_type"] == "audio" for s in metadata["streams"]))
        # Clip 1 has extra source frames. The second planned window must start
        # at .5s, not at the first source file's original 1.25s duration.
        pixel = subprocess.check_output(["ffmpeg", "-nostdin", "-v", "error", "-ss", "0.65", "-i", str(output),
            "-frames:v", "1", "-vf", "scale=1:1", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"])
        self.assertGreater(pixel[2], pixel[0] + 100)
        self.assertEqual(result["_meta"]["exoharness"]["workflow_continuation"]["status"], "complete")
        self.assertTrue(all(p.is_file() for p in [*self.clips, self.audio]))

    async def test_short_video_and_long_audio_fail_before_any_encoding(self):
        for change in ("short_video", "long_audio", "unaligned"):
            segments = [s.model_copy() for s in self.segments]
            if change == "short_video": segments[1].duration_seconds = 2
            elif change == "long_audio": segments[0].audio_offset_seconds = .4
            else: segments[0].duration_seconds = .51
            with patch("media_workflow_mcp.timeline.run_process", new_callable=AsyncMock) as encode:
                with self.assertRaises(ValueError):
                    await assemble_timeline_files(self.settings, segments, self.clips, [self.audio] * 2,
                        self.root, self.root / "invalid.mp4", AsyncMock())
                encode.assert_not_awaited()
            self.assertFalse((self.root / "invalid.mp4").exists())

    async def test_silent_timeline_has_no_audio_and_native_schema_rejects_bad_values(self):
        for invalid in (float("nan"), -1, 0):
            with self.assertRaises(ValueError):
                TimelineSegment(video_url="https://source.example/a", duration_seconds=invalid)
        segments = [TimelineSegment(video_url=s.video_url, duration_seconds=s.duration_seconds) for s in self.segments]
        output = self.root / "silent.mp4"
        await assemble_timeline_files(self.settings, segments, self.clips, [None, None], self.root, output, AsyncMock())
        self.assertFalse(any(s["codec_type"] == "audio" for s in (await probe_file(self.settings, output))["streams"]))

    async def test_source_audio_survives_multi_clip_assembly_without_separate_audio(self):
        voiced_clips = [self.root / "voiced-red.mp4", self.root / "voiced-blue.mp4"]
        for color, frequency, path in zip(("red", "blue"), (440, 660), voiced_clips):
            subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-f", "lavfi", "-i",
                f"color=c={color}:s=160x90:r=24:d=1.25", "-f", "lavfi", "-i",
                f"sine=frequency={frequency}:duration=1.25", "-c:v", "libx264", "-c:a", "aac",
                "-shortest", str(path)], check=True)
        segments = [TimelineSegment(video_url=f"https://source.example/{index}.mp4",
            duration_seconds=1, preserve_source_audio=True) for index in range(2)]
        output = self.root / "with-source-audio.mp4"
        report = await assemble_timeline_files(self.settings, segments, voiced_clips,
                                               [None, None], self.root, output, AsyncMock())
        self.assertTrue(report["has_audio"])
        self.assertTrue(report["has_source_audio"])
        self.assertFalse(report["has_narration"])
        metadata = await probe_file(self.settings, output)
        self.assertTrue(any(s["codec_type"] == "audio" for s in metadata["streams"]))
        samples = subprocess.check_output(["ffmpeg", "-nostdin", "-v", "error", "-i", str(output),
            "-vn", "-t", "0.5", "-ac", "1", "-ar", "8000", "-f", "s16le", "-"])
        self.assertTrue(any(byte != 0 for byte in samples))

    async def test_source_audio_requires_track_and_cannot_mix_with_separate_narration(self):
        for segment in (
            TimelineSegment(video_url="https://source.example/red.mp4", duration_seconds=.5,
                            preserve_source_audio=True),
            TimelineSegment(video_url="https://source.example/red.mp4", duration_seconds=.5,
                            preserve_source_audio=True, audio_url="https://source.example/speech.wav"),
            TimelineSegment(video_url="https://source.example/red.mp4", duration_seconds=.5,
                            preserve_source_audio=True, audio_offset_seconds=.1),
        ):
            with self.subTest(segment=segment.model_dump()):
                with patch("media_workflow_mcp.timeline.run_process", new_callable=AsyncMock) as encode:
                    with self.assertRaises(ValueError):
                        await assemble_timeline_files(self.settings, [segment], [self.clips[0]],
                            [self.audio if segment.audio_url else None], self.root,
                            self.root / "invalid-source-audio.mp4", AsyncMock())
                    encode.assert_not_awaited()

    async def test_disallowed_later_source_and_cancellation_publish_nothing(self):
        runtime = SimpleNamespace(settings=self.settings, artifacts=ArtifactStore(self.settings))
        ctx = SimpleNamespace(report_progress=AsyncMock())
        self.segments[-1].video_url = "http://unapproved.example/private"
        with patch.object(server, "_require_tool_auth", return_value=runtime), patch.object(server, "download_source", new_callable=AsyncMock) as fetch:
            with self.assertRaisesRegex(Exception, "origin is not allowed"):
                await server.assemble_video_timeline(self.segments, ctx)
            fetch.assert_not_awaited()
        self.segments[-1].video_url = "https://source.example/blue.mp4"
        with patch.object(server, "_require_tool_auth", return_value=runtime), patch.object(server, "download_source", side_effect=asyncio.CancelledError):
            with self.assertRaises(asyncio.CancelledError):
                await server.assemble_video_timeline(self.segments, ctx)
        self.assertEqual(list(self.settings.artifact_dir.iterdir()), [])
        self.assertEqual(set(self.root.iterdir()), {*self.clips, self.audio, self.settings.artifact_dir})

    async def test_mcp_schema_and_dispatch_use_native_segments(self):
        tool = next(t for t in await server.mcp.list_tools() if t.name == "assemble_video_timeline")
        self.assertEqual(tool.input_schema["properties"]["segments"]["type"], "array")
        self.assertIn("preserve_source_audio", str(tool.input_schema))
        with patch.object(server, "_require_tool_auth") as auth:
            with self.assertRaises(Exception):
                await server.mcp.call_tool("assemble_video_timeline", {"segments": '[{"video_url":"wrong"}]'})
            auth.assert_not_called()
        sources = dict(zip([s.video_url for s in self.segments], self.clips))
        sources[self.segments[0].audio_url] = self.audio
        async def download(session, settings, url, target):
            shutil.copyfile(sources[url], target)
            return target
        runtime = SimpleNamespace(settings=self.settings, artifacts=ArtifactStore(self.settings))
        with patch.object(server, "_require_tool_auth", return_value=runtime), patch.object(server, "download_source", side_effect=download), patch.object(server.Context, "report_progress", new_callable=AsyncMock):
            await server.mcp.call_tool("assemble_video_timeline", {"segments": [s.model_dump() for s in self.segments]})
        self.assertEqual(len(list(self.settings.artifact_dir.glob("*/final-video.mp4"))), 1)

    async def test_four_thirty_second_windows_deliver_one_two_minute_video(self):
        clip = self.root / "long.mp4"
        subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-f", "lavfi", "-i",
            "color=c=green:s=64x48:r=24:d=30.25", "-an", "-c:v", "libx264", str(clip)], check=True)
        segments = [TimelineSegment(video_url=f"https://source.example/{i}.mp4", duration_seconds=30,
            audio_url="https://source.example/speech.wav", audio_offset_seconds=i) for i in range(4)]
        output = self.root / "two-minutes.mp4"
        report = await assemble_timeline_files(self.settings, segments, [clip] * 4, [self.audio] * 4,
                                              self.root, output, AsyncMock())
        self.assertEqual(report["segment_count"], 4)
        self.assertAlmostEqual(float(report["duration_seconds"]), 120, delta=.05)
        self.assertTrue(report["has_narration"])
