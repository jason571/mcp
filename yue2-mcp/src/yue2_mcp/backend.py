import hashlib
import json
import os
import subprocess
from pathlib import Path
from threading import Event

import httpx

from .config import Settings
from .models import LyricsRequest, MusicRequest


class GenerationCanceled(Exception):
    pass


class YuEBackend:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.pipe = None

    @property
    def loaded(self) -> bool:
        return self.pipe is not None

    def unload(self):
        if self.pipe is not None:
            self.pipe.close()
            self.pipe = None

    def _load(self):
        if self.pipe is None:
            if not (self.settings.model_path / "config.json").is_file():
                raise FileNotFoundError(f"YuE2 model missing: {self.settings.model_path}")
            if not (self.settings.vae_path / "config.json").is_file():
                raise FileNotFoundError(f"YuE2 VAE missing: {self.settings.vae_path}")
            from yue2 import YuE2Pipeline

            self.pipe = YuE2Pipeline.from_pretrained(
                str(self.settings.model_path), vae=str(self.settings.vae_path),
                device=self.settings.device, local_files_only=True, progress=False,
                memory_budget_gib=self.settings.memory_budget_gib,
                quantization=self.settings.quantization,
                offload_ar=self.settings.offload_ar,
            )
        return self.pipe

    def lyrics(self, request: LyricsRequest, cancel: Event) -> str:
        if cancel.is_set():
            raise GenerationCanceled()
        if not self.settings.lyrics_base_url:
            raise RuntimeError(
                "Automatic lyrics require YUE2_LYRICS_BASE_URL and YUE2_LYRICS_MODEL"
            )
        url = self.settings.lyrics_base_url.rstrip("/") + "/chat/completions"
        headers = {}
        key = self.settings.lyrics_api_key.get_secret_value()
        if key:
            headers["Authorization"] = f"Bearer {key}"
        body = {
            "model": self.settings.lyrics_model,
            "temperature": 0.8,
            "messages": [
                {"role": "system", "content": (
                    "Write original, singable song lyrics only. Use clear section labels such as "
                    "[Verse 1], [Chorus], [Bridge]. Respect the requested language and style. "
                    "Do not include commentary or markdown fences."
                )},
                {"role": "user", "content": (
                    f"Language: {request.language}\nStyle: {request.style}\n"
                    f"Song brief: {request.prompt}"
                )},
            ],
        }
        with httpx.Client(timeout=180) as client:
            response = client.post(url, headers=headers, json=body)
            response.raise_for_status()
            data = response.json()
        if cancel.is_set():
            raise GenerationCanceled()
        content = data["choices"][0]["message"]["content"]
        if isinstance(content, list):
            content = "\n".join(item.get("text", "") for item in content if isinstance(item, dict))
        lyrics = str(content).strip()
        if not lyrics or len(lyrics) > 16000:
            raise ValueError("Lyrics provider returned empty or oversized lyrics")
        return lyrics

    def music(
        self, request: MusicRequest, lyrics: str, output: Path, report, cancel: Event
    ) -> dict:
        def check():
            if cancel.is_set():
                raise GenerationCanceled()

        check()
        report(12, "loading", "Loading YuE2 and VAE model files")
        pipe = self._load()
        check()
        seed = request.seed if request.seed is not None else int.from_bytes(os.urandom(4), "big")
        kwargs = {"style": request.style, "lyrics": lyrics, "cot": request.cot,
                  "seed": seed, "id": output.name}
        if request.abc:
            kwargs["abc"] = request.abc
        if request.cfg_scale is not None:
            kwargs["cfg_scale"] = request.cfg_scale

        def token(phase, value):
            # The runtime supplies tokens but no total token count. Keep percentage at the
            # last completed boundary and report observed output count instead.
            counts[phase] = counts.get(phase, 0) + 1
            if counts[phase] % 32 == 0:
                report(
                    20 if phase == "abc" else 45,
                    phase,
                    f"Generated {counts[phase]} {phase} tokens",
                )
            check()

        counts = {}
        report(20, "planning", "Planning score")
        plan = pipe.plan(**kwargs, cancelled=cancel.is_set, on_token=token)
        check()
        report(40, "semantic", "Generating semantic music tokens")
        semantic = pipe.generate_semantic(plan, cancelled=cancel.is_set, on_token=token)
        check()
        report(65, "synthesis", "Synthesizing acoustic latents")
        latents = pipe.synthesize(semantic, cancelled=cancel.is_set)
        check()
        report(85, "decoding", "Decoding stereo audio")
        audio = pipe.decode(latents)
        check()
        report(95, "saving", "Saving FLAC, MP3 preview and score")
        import soundfile as sf

        output.mkdir(parents=True, exist_ok=True)
        tmp = output / "song.tmp.flac"
        sf.write(tmp, audio, 48000, format="FLAC")
        check()
        tmp.replace(output / "song.flac")
        preview_tmp = output / "preview.tmp.mp3"
        subprocess.run(
            ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
             "-i", str(output / "song.flac"), "-vn", "-codec:a", "libmp3lame",
             "-b:a", "160k", str(preview_tmp)],
            check=True, capture_output=True, timeout=300,
        )
        check()
        preview_tmp.replace(output / "preview.mp3")
        (output / "lyrics.txt").write_text(lyrics, encoding="utf-8")
        if plan.abc:
            (output / "score.abc").write_text(plan.abc, encoding="utf-8")
        digest = hashlib.sha256((output / "song.flac").read_bytes()).hexdigest()
        result = {
            "seed": seed, "sample_rate": 48000, "channels": 2,
            "duration_seconds": round(len(audio) / 48000, 3), "sha256": digest,
            "lyrics": lyrics, "preview_available": True,
            "preview_sha256": hashlib.sha256((output / "preview.mp3").read_bytes()).hexdigest(),
            "score_available": bool(plan.abc),
            "score_truncated": bool(plan.truncated),
            "semantic_truncated": bool(semantic.truncated),
        }
        (output / "metadata.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
        return result
