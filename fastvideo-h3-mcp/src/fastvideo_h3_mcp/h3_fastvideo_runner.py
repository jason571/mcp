from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any


def _load_request_value(value: object) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("worker request must be an object")
    required = {
        "model_path",
        "prompt",
        "width",
        "height",
        "num_frames",
        "fps",
        "steps",
        "seed",
        "num_gpus",
        "output_path",
    }
    missing = sorted(required - value.keys())
    if missing:
        raise ValueError("worker request is missing: " + ", ".join(missing))
    return value


class FastH3Worker:
    def __init__(self) -> None:
        self.generator: Any | None = None
        self.config_key: tuple[object, ...] | None = None

    def _ensure_generator(self, request: dict[str, Any]) -> None:
        # Keep CUDA and FastVideo out of the MCP web process.
        from fastvideo import VideoGenerator
        from fastvideo.api import (
            CompileConfig,
            ComponentConfig,
            EngineConfig,
            GeneratorConfig,
            OffloadConfig,
            ParallelismConfig,
            PipelineSelection,
            QuantizationConfig,
        )

        model_path = str(request["model_path"])
        num_gpus = int(request["num_gpus"])
        text_encoder_fp8 = bool(request.get("text_encoder_fp8", False))
        torch_compile = bool(request.get("torch_compile", False))
        vsa_sparsity = float(request.get("vsa_sparsity", 0.0))
        vsa_tile_size = int(request.get("vsa_tile_size", 64))
        config_key = (
            model_path,
            num_gpus,
            text_encoder_fp8,
            torch_compile,
            vsa_sparsity,
            vsa_tile_size,
        )
        if self.generator is not None:
            if self.config_key != config_key:
                raise RuntimeError("worker model configuration changed")
            return

        experimental: dict[str, object] = {}
        if vsa_sparsity > 0.0:
            experimental["VSA_sparsity"] = vsa_sparsity
            experimental["VSA_tile_size"] = vsa_tile_size
        self.generator = VideoGenerator.from_config(
            GeneratorConfig(
                model_path=model_path,
                pipeline=PipelineSelection(
                    components=ComponentConfig(
                        text_encoder_weights=(
                            str(Path(model_path) / "text_encoder")
                            if text_encoder_fp8
                            else None
                        ),
                    ),
                    experimental=experimental,
                ),
                engine=EngineConfig(
                    num_gpus=num_gpus,
                    use_fsdp_inference=num_gpus > 1,
                    parallelism=ParallelismConfig(tp_size=1, sp_size=num_gpus),
                    offload=OffloadConfig(
                        dit=False,
                        dit_layerwise=False,
                        text_encoder=True,
                        vae=True,
                        pin_cpu_memory=False,
                    ),
                    quantization=(
                        QuantizationConfig(text_encoder_quant="AbsMaxFP8")
                        if text_encoder_fp8
                        else None
                    ),
                    compile=CompileConfig(enabled=torch_compile),
                ),
            )
        )
        self.config_key = config_key

    def generate(self, request: dict[str, Any]) -> tuple[Path, bool]:
        from fastvideo.api import GenerationRequest, OutputConfig, SamplingConfig

        release_after_denoise = bool(request.get("release_dit_after_denoise", True))
        os.environ["FASTVIDEO_MINIMAX_H3_RELEASE_DIT_AFTER_DENOISE"] = (
            "1" if release_after_denoise else "0"
        )
        output_path = Path(str(request["output_path"]))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_generator(request)
        assert self.generator is not None
        result = self.generator.generate(
            GenerationRequest(
                prompt=str(request["prompt"]),
                negative_prompt="",
                sampling=SamplingConfig(
                    height=int(request["height"]),
                    width=int(request["width"]),
                    num_frames=int(request["num_frames"]),
                    fps=int(request["fps"]),
                    num_inference_steps=int(request["steps"]),
                    guidance_scale=1.0,
                    batch_cfg=False,
                    seed=int(request["seed"]),
                ),
                output=OutputConfig(
                    output_path=str(output_path),
                    save_video=True,
                    return_frames=False,
                ),
            )
        )
        result_path = Path(str(result.video_path))
        if result_path != output_path:
            if not result_path.is_file():
                raise RuntimeError(f"FastVideo returned missing output: {result_path}")
            result_path.replace(output_path)
        reusable = not release_after_denoise
        if not reusable:
            self.shutdown()
        return output_path, reusable

    def shutdown(self) -> None:
        if self.generator is None:
            return
        try:
            self.generator.shutdown()
        finally:
            self.generator = None
            self.config_key = None


def serve() -> None:
    worker = FastH3Worker()
    try:
        for raw in sys.stdin:
            request_id: object = None
            try:
                request = _load_request_value(json.loads(raw))
                request_id = request.get("request_id")
                output_path, reusable = worker.generate(request)
                result = {
                    "request_id": request_id,
                    "ok": True,
                    "output_path": str(output_path),
                    "reusable": reusable,
                }
            except Exception as exc:
                worker.shutdown()
                result = {
                    "request_id": request_id,
                    "ok": False,
                    "error": f"{exc}\n{traceback.format_exc()}",
                    "reusable": False,
                }
                print("MCP_WORKER_RESULT " + json.dumps(result), flush=True)
                return
            print("MCP_WORKER_RESULT " + json.dumps(result), flush=True)
            if not reusable:
                return
    finally:
        worker.shutdown()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", action="store_true", required=True)
    parser.parse_args()
    serve()


if __name__ == "__main__":
    main()
