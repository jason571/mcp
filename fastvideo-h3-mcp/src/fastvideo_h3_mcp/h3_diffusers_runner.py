from __future__ import annotations

import argparse
import json
from pathlib import Path


def progress(value: float, stage: str, message: str) -> None:
    print(
        "MCP_PROGRESS "
        + json.dumps(
            {"progress": value, "stage": stage, "message": message},
            ensure_ascii=False,
        ),
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request-json", required=True)
    args = parser.parse_args()
    request = json.loads(Path(args.request_json).read_text(encoding="utf-8"))

    progress(12, "importing", "正在载入 PyTorch 与 Diffusers")
    import torch
    from diffusers import ComponentsManager, ModularPipeline
    from diffusers.utils import load_image
    from diffusers.utils.export_utils import encode_video

    progress(18, "loading_model", "正在创建 MiniMax H3 FL2VA Pipeline")
    manager = ComponentsManager()
    manager.enable_auto_cpu_offload(device="cuda")
    pipe = ModularPipeline.from_pretrained(
        request["model_path"], components_manager=manager
    )
    pipe.load_components(workflow="fl2va", dtype=torch.bfloat16)
    progress(35, "loading_keyframes", "模型加载完成，正在处理首尾参考帧")

    first_image = (
        load_image(request["first_frame_path"])
        if request.get("first_frame_path")
        else None
    )
    last_image = (
        load_image(request["last_frame_path"])
        if request.get("last_frame_path")
        else None
    )
    call_args = {
        "prompt": request["prompt"],
        "height": int(request["height"]),
        "width": int(request["width"]),
        "num_frames": int(request["num_frames"]),
        "num_inference_steps": int(request["steps"]),
        "generator": torch.Generator().manual_seed(int(request["seed"])),
        "output": ["videos", "audio", "sampling_rate"],
    }
    if first_image is not None:
        call_args["image"] = first_image
    if last_image is not None:
        call_args["last_image"] = last_image

    progress(42, "generating", "正在生成视频与同步音频")
    results = pipe(**call_args)
    progress(88, "encoding", "生成完成，正在编码 MP4")
    audio = results["audio"][0] if request["audio_enabled"] else None
    encode_video(
        results["videos"][0],
        fps=24,
        output_path=request["output_path"],
        audio=audio,
        audio_sample_rate=results["sampling_rate"] if audio is not None else None,
    )
    progress(97, "verifying", "MP4 已写入，正在返回结果")


if __name__ == "__main__":
    main()
