from __future__ import annotations

import asyncio
import hashlib
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from media_workflow_mcp.config import Settings
from media_workflow_mcp.media import (
    analyze_audio_energy,
    assemble_music_video_files,
    bind_opening_frame,
    extract_last_frame_file,
    opening_frame_fetch_url,
    probe_file,
    sha256_file,
    validate_source_url,
    run_process,
)


class ProcessCancellationTests(unittest.IsolatedAsyncioTestCase):
    async def test_cancel_reaps_only_the_owned_subprocess(self) -> None:
        started = asyncio.Event()
        children = []
        spawn = asyncio.create_subprocess_exec

        async def capture(*args, **kwargs):
            child = await spawn(*args, **kwargs)
            children.append(child)
            started.set()
            return child

        with patch("media_workflow_mcp.media.asyncio.create_subprocess_exec", capture):
            task = asyncio.create_task(run_process(
                SimpleNamespace(operation_timeout_seconds=120),
                [sys.executable, "-c", "import time; time.sleep(60)"],
            ))
            await started.wait()
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await asyncio.wait_for(task, 5)
            self.assertEqual(len(children), 1)
            self.assertIsNotNone(children[0].returncode)


class UrlPolicyTests(unittest.TestCase):
    def test_sha256_file_returns_the_exact_immutable_asset_digest(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            source = Path(raw) / "asset.bin"
            source.write_bytes(b"immutable-media")
            self.assertEqual(
                sha256_file(source), hashlib.sha256(b"immutable-media").hexdigest()
            )

    def test_exact_origin_allowlist_rejects_redirect_targets_and_credentials(self) -> None:
        allowed = ("http://127.0.0.1:8787", "https://media.example.com")
        self.assertEqual(
            validate_source_url("http://127.0.0.1:8787/v1/a", allowed),
            "http://127.0.0.1:8787/v1/a",
        )
        with self.assertRaises(ValueError):
            validate_source_url("http://127.0.0.1:8765/private", allowed)
        with self.assertRaises(ValueError):
            validate_source_url("https://user:pass@media.example.com/a.mp3", allowed)


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "ffmpeg required")
class MediaIntegrationTests(unittest.TestCase):
    def test_extract_and_assemble_preserve_master_audio_duration(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            settings = Settings(
                host="127.0.0.1",
                port=8790,
                mcp_path="/mcp",
                public_base_url="http://127.0.0.1:8790",
                allowed_hosts=(),
                allowed_origins=(),
                allowed_fetch_origins=(),
                api_token=None,
                artifact_signing_key="test-key",
                artifact_url_ttl_seconds=3600,
                data_dir=root,
                artifact_dir=root / "artifacts",
                ffmpeg_binary=shutil.which("ffmpeg") or "ffmpeg",
                ffprobe_binary=shutil.which("ffprobe") or "ffprobe",
                max_download_bytes=32 * 1024 * 1024,
                operation_timeout_seconds=60,
            )
            settings.prepare_directories()
            clip_a = root / "a.mp4"
            clip_b = root / "b.mp4"
            audio = root / "song.mp3"
            subprocess.run(
                [
                    settings.ffmpeg_binary,
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-f",
                    "lavfi",
                    "-i",
                    "color=c=red:s=320x180:d=1",
                    "-an",
                    "-c:v",
                    "libx264",
                    "-pix_fmt",
                    "yuv420p",
                    "-y",
                    str(clip_a),
                ],
                check=True,
            )
            subprocess.run(
                [
                    settings.ffmpeg_binary,
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-f",
                    "lavfi",
                    "-i",
                    "color=c=blue:s=320x180:d=1",
                    "-an",
                    "-c:v",
                    "libx264",
                    "-pix_fmt",
                    "yuv420p",
                    "-y",
                    str(clip_b),
                ],
                check=True,
            )
            subprocess.run(
                [
                    settings.ffmpeg_binary,
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-f",
                    "lavfi",
                    "-i",
                    "sine=frequency=440:duration=2.4",
                    "-c:a",
                    "libmp3lame",
                    "-y",
                    str(audio),
                ],
                check=True,
            )

            async def run() -> None:
                last_frame = root / "last.png"
                frame_meta = await extract_last_frame_file(settings, clip_a, last_frame)
                self.assertTrue(last_frame.is_file())
                self.assertEqual(frame_meta["width"], 320)
                output = root / "final.mp4"

                async def progress(_value: float, _message: str) -> None:
                    return None

                metadata = await assemble_music_video_files(
                    settings, [clip_a, clip_b], audio, root, output, progress
                )
                result = await probe_file(settings, output)
                analysis = await analyze_audio_energy(settings, audio, 0.5)
                self.assertTrue(metadata["audio"])
                self.assertEqual(metadata["clip_count"], 2)
                self.assertAlmostEqual(result["duration_seconds"], 2.4, delta=0.12)
                self.assertAlmostEqual(analysis["duration_seconds"], 2.4, delta=0.12)
                self.assertTrue(analysis["energy_curve"])
                self.assertIn("no lyric", analysis["analysis"])

            asyncio.run(run())


class OpeningFrameBindingTests(unittest.TestCase):
    def test_attachment_id_xor_url(self) -> None:
        attachment = "att_" + "ab" * 32
        bound = bind_opening_frame(
            attachment_id=attachment, url=None, allowed_origins=("https://media.example",)
        )
        self.assertEqual(bound, {"first_frame_attachment_id": attachment})
        self.assertEqual(
            opening_frame_fetch_url(bound, SimpleNamespace(asset_base="http://127.0.0.1:8787")),
            f"http://127.0.0.1:8787/v1/attachment-assets/{attachment}",
        )
        with self.assertRaisesRegex(ValueError, "exactly one"):
            bind_opening_frame(
                attachment_id=attachment,
                url="https://media.example/still.jpg",
                allowed_origins=("https://media.example",),
            )
        with self.assertRaisesRegex(ValueError, "run-artifacts"):
            bind_opening_frame(
                attachment_id=None,
                url="https://agent.clhh.tech/v1/run-artifacts/still.jpg",
                allowed_origins=("https://agent.clhh.tech",),
            )
        self.assertEqual(
            bind_opening_frame(attachment_id=None, url=None, allowed_origins=()),
            {},
        )


if __name__ == "__main__":
    unittest.main()
