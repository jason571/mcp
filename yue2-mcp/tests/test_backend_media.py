import hashlib
from threading import Event
from types import SimpleNamespace

import numpy as np
import soundfile as sf

from yue2_mcp.backend import YuEBackend
from yue2_mcp.config import Settings
from yue2_mcp.models import MusicRequest


class ShortAudioPipeline:
    def plan(self, **kwargs):
        return SimpleNamespace(abc=None, truncated=False)

    def generate_semantic(self, plan, **kwargs):
        return SimpleNamespace(truncated=False)

    def synthesize(self, semantic, **kwargs):
        return None

    def decode(self, latents):
        samples = np.sin(2 * np.pi * 440 * np.arange(24000) / 48000).astype("float32") * 0.1
        return np.column_stack((samples, samples))


def test_music_writes_playable_preview_and_original(tmp_path):
    backend = YuEBackend(Settings(_env_file=None, data_dir=tmp_path))
    backend._load = lambda: ShortAudioPipeline()
    output = tmp_path / "song"
    result = backend.music(MusicRequest(style="acoustic", lyrics="[Verse] hello"),
                           "[Verse] hello", output, lambda *args: None, Event())
    assert result["preview_available"]
    assert sf.info(output / "preview.mp3").channels == 2
    assert sf.info(output / "preview.mp3").samplerate == 48000
    assert sf.info(output / "song.flac").duration == 0.5
    assert result["preview_sha256"] == hashlib.sha256(
        (output / "preview.mp3").read_bytes()).hexdigest()
    assert (output / "lyrics.txt").read_text() == "[Verse] hello"
