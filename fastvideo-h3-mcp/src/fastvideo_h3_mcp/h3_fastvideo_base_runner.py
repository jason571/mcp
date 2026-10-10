from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any


_LORA_TARGET_MODULES = (
    ".attn.to_q",
    ".attn.to_k",
    ".attn.to_v",
    ".attn.to_out",
    ".ff.fc_in",
    ".ff.fc_out",
)


def _progress(progress: float, stage: str, message: str) -> None:
    print(
        "MCP_PROGRESS "
        + json.dumps(
            {"progress": progress, "stage": stage, "message": message},
            ensure_ascii=False,
        ),
        flush=True,
    )


def _load_request(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    return _load_request_value(value)


def _load_request_value(value: object) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("request JSON must be an object")
    required = {
        "model_path",
        "lora_path",
        "lora_alpha",
        "lora_strength",
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
        raise ValueError("request JSON is missing: " + ", ".join(missing))
    if not value.get("first_frame_path") and not value.get("last_frame_path"):
        raise ValueError("Base FL2VA requires first_frame_path or last_frame_path")
    return value


def _load_rgb_image(path: str | None):
    if not path:
        return None
    from PIL import Image

    with Image.open(path) as source:
        return source.convert("RGB")


def _assert_diffusers_lora(lora_path: str) -> str:
    # FastVideo venv runs this file as a script, so relative package imports
    # are unavailable. Load the sibling converter from disk; the converter
    # only imports torch inside prepare_fastvideo_lora_path().
    try:
        from .lora_native_to_diffusers import prepare_fastvideo_lora_path
    except ImportError:
        import importlib.util

        helper = Path(__file__).with_name("lora_native_to_diffusers.py")
        spec = importlib.util.spec_from_file_location(
            "_fastvideo_h3_mcp_lora_native", helper
        )
        if spec is None or spec.loader is None:
            raise RuntimeError(f"unable to load LoRA converter from {helper}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        prepare_fastvideo_lora_path = module.prepare_fastvideo_lora_path
    return str(prepare_fastvideo_lora_path(lora_path))


def _scheduler_grid_points(transformer_evaluations: int) -> int:
    if transformer_evaluations < 1:
        raise ValueError("transformer evaluations must be positive")
    # MiniMaxH3Scheduler treats num_inference_steps as sigma grid points and
    # excludes the terminal zero from transformer evaluation. Four NFE needs
    # q=[1,.75,.5,.25] followed by the terminal q=0 point.
    return transformer_evaluations + 1


class BaseFL2VAWorker:
    def __init__(self) -> None:
        self.generator: Any | None = None
        self.config_key: tuple[object, ...] | None = None

    def _ensure_generator(self, request: dict[str, Any]) -> None:
        # Convert ComfyUI fused-QKV adapters before importing FastVideo so a
        # bad LoRA fails without loading the DiT.
        lora_path = _assert_diffusers_lora(str(request["lora_path"]))
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
        text_encoder_fp8 = bool(request.get("text_encoder_fp8", True))
        config_key = (
            model_path,
            lora_path,
            int(request["lora_alpha"]),
            float(request["lora_strength"]),
            num_gpus,
            text_encoder_fp8,
        )
        if self.generator is not None:
            if self.config_key != config_key:
                raise RuntimeError("worker model configuration changed")
            return

        if lora_path != str(request["lora_path"]):
            _progress(11.0, "converting_lora", "正在把 ComfyUI/原生 H3 LoRA 转成 FastVideo Diffusers 键")
        _progress(12.0, "loading_model", "正在由 FastVideo 加载 Base FL2VA 组件")
        self.generator = VideoGenerator.from_config(
            GeneratorConfig(
                model_path=model_path,
                pipeline=PipelineSelection(
                    components=ComponentConfig(
                        lora_path=lora_path,
                        lora_alpha=int(request["lora_alpha"]),
                        lora_strength=float(request["lora_strength"]),
                        text_encoder_weights=(
                            str(Path(model_path) / "text_encoder")
                            if text_encoder_fp8
                            else None
                        ),
                    ),
                    experimental={"lora_target_modules": list(_LORA_TARGET_MODULES)},
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
                    compile=CompileConfig(enabled=False),
                ),
            )
        )
        self.config_key = config_key

    def generate(self, request: dict[str, Any]) -> tuple[Path, bool]:
        from fastvideo.api import (
            GenerationRequest,
            InputConfig,
            OutputConfig,
            SamplingConfig,
        )

        release_after_denoise = bool(request.get("release_dit_after_denoise", True))
        os.environ["FASTVIDEO_MINIMAX_H3_RELEASE_DIT_AFTER_DENOISE"] = (
            "1" if release_after_denoise else "0"
        )
        first_image = _load_rgb_image(request.get("first_frame_path"))
        last_image = _load_rgb_image(request.get("last_frame_path"))
        output_path = Path(str(request["output_path"]))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self._ensure_generator(request)
            assert self.generator is not None
            _progress(44.0, "preparing_inputs", "Base FL2VA 模型就绪，正在准备参考帧")
            result = self.generator.generate(
                GenerationRequest(
                    prompt=str(request["prompt"]),
                    negative_prompt="",
                    inputs=InputConfig(pil_image=first_image, last_image=last_image),
                    sampling=SamplingConfig(
                        height=int(request["height"]),
                        width=int(request["width"]),
                        num_frames=int(request["num_frames"]),
                        fps=int(request["fps"]),
                        num_inference_steps=_scheduler_grid_points(int(request["steps"])),
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
            _progress(96.0, "encoding", "Base FL2VA 已完成生成与音视频封装")
            result_path = Path(str(result.video_path))
            if result_path != output_path:
                if not result_path.is_file():
                    raise RuntimeError(f"FastVideo returned missing output: {result_path}")
                result_path.replace(output_path)
            reusable = not release_after_denoise
            if not reusable:
                self.shutdown()
            return output_path, reusable
        finally:
            if first_image is not None:
                first_image.close()
            if last_image is not None:
                last_image.close()

    def shutdown(self) -> None:
        if self.generator is None:
            return
        _progress(97.0, "releasing_model", "正在释放 Base FL2VA 模型资源")
        try:
            self.generator.shutdown()
        finally:
            self.generator = None
            self.config_key = None


def run(request: dict[str, Any]) -> None:
    worker = BaseFL2VAWorker()
    try:
        worker.generate(request)
    finally:
        worker.shutdown()


def serve() -> None:
    worker = BaseFL2VAWorker()
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
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--request-json", type=Path)
    group.add_argument("--server", action="store_true")
    args = parser.parse_args()
    if args.server:
        serve()
    else:
        assert args.request_json is not None
        run(_load_request(args.request_json))


if __name__ == "__main__":
    main()
