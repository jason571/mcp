"""GPU imports are lazy: protocol tests and service startup do not load a model."""

import gc
import hashlib
import json
import time
from collections.abc import Callable
from pathlib import Path
from threading import Event

from .config import Settings
from .models import MODEL_ID, GenerateRequest


class GenerationCanceled(Exception):
    pass


def check_canceled(cancel: Event):
    if cancel.is_set():
        raise GenerationCanceled("Generation canceled")


class DiffusersBackend:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.pipe = None

    @property
    def loaded(self) -> bool:
        return self.pipe is not None

    def unload(self):
        self.pipe = None
        gc.collect()
        # Do not import torch just to shut down an unused service.
        import sys

        if "torch" in sys.modules:
            sys.modules["torch"].cuda.empty_cache()

    def _load(self):
        if self.pipe is not None:
            return
        import torch
        from diffusers import ZImagePipeline

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; install a CUDA-enabled PyTorch build")
        model_path = self.settings.model_path.resolve()
        if not (model_path / "model_index.json").is_file():
            raise RuntimeError("Z_IMAGE_MODEL_PATH must point to a complete Diffusers snapshot")
        options = dict(
            torch_dtype=getattr(torch, self.settings.dtype),
            local_files_only=True,
            low_cpu_mem_usage=True,
        )
        if not self.settings.cpu_offload:
            # Load onto GPU directly to avoid a full model-sized CPU intermediate copy.
            torch.cuda.set_device(self.settings.device)
            options["device_map"] = "cuda"
        pipe = ZImagePipeline.from_pretrained(str(model_path), **options)
        if self.settings.cpu_offload:
            pipe.enable_model_cpu_offload(device=self.settings.device)
        pipe.set_progress_bar_config(disable=True)
        pipe.vae.enable_tiling()
        self.pipe = pipe

    def generate(
        self,
        request: GenerateRequest,
        output: Path,
        progress: Callable[[float, str, str], None],
        cancel: Event,
    ) -> dict:
        import torch
        from PIL.PngImagePlugin import PngInfo

        begin = time.monotonic()
        cold = not self.loaded
        check_canceled(cancel)
        progress(1, "loading_model", "Loading local weights" if cold else "Model already loaded")
        try:
            self._load()
            load_seconds = time.monotonic() - begin
            check_canceled(cancel)
            # Reject silent tokenizer truncation; callers can shorten and retry.
            tokens = self.pipe.tokenizer.apply_chat_template(
                [{"role": "user", "content": request.prompt}],
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=True,
            )
            if len(tokens) > 512:
                raise ValueError("Prompt exceeds the model's 512-token limit; shorten the prompt")
            progress(5, "encoding_prompt", "Encoding prompt")
            inference_start = time.monotonic()

            def on_step(pipe, step, timestep, callback_kwargs):
                check_canceled(cancel)
                count = len(pipe.scheduler.timesteps)
                progress(
                    10 + 75 * (step + 1) / count,
                    "denoising",
                    f"Completed sampling step {step + 1}/{count}",
                )
                return callback_kwargs

            with torch.inference_mode():
                image = self.pipe(
                    prompt=request.prompt,
                    width=request.width,
                    height=request.height,
                    num_inference_steps=request.steps,
                    guidance_scale=0.0,
                    max_sequence_length=512,
                    generator=torch.Generator(self.settings.device).manual_seed(request.seed),
                    callback_on_step_end=on_step,
                    callback_on_step_end_tensor_inputs=[],
                ).images[0]
            torch.cuda.synchronize(self.settings.device)
            inference_seconds = time.monotonic() - inference_start
            check_canceled(cancel)
            progress(95, "saving_image", "Decoded image; saving PNG")
            metadata = PngInfo()
            metadata.add_text("model", MODEL_ID)
            metadata.add_text("generation", request.model_dump_json())
            temporary = output.with_suffix(".tmp")
            image.save(temporary, format="PNG", pnginfo=metadata)
            temporary.replace(output)
            result = {
                "model": MODEL_ID,
                "width": image.width,
                "height": image.height,
                "seed": request.seed,
                "steps": request.steps,
                "guidance_scale": 0.0,
                "mime_type": "image/png",
                "bytes": output.stat().st_size,
                "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
                "cold_start": cold,
                "model_load_seconds": round(load_seconds, 3),
                "inference_seconds": round(inference_seconds, 3),
                "worker_seconds": round(time.monotonic() - begin, 3),
            }
            output.with_suffix(".json").write_text(
                json.dumps(
                    {"request": request.model_dump(), **result}, ensure_ascii=False, indent=2
                ),
                encoding="utf-8",
            )
            return result
        except GenerationCanceled:
            raise
        except Exception:
            # A broken load / CUDA OOM must not poison the next queued request.
            self.unload()
            raise
