#!/usr/bin/env bash
set -euo pipefail

base_backend="${MINIMAX_H3_BASE_BACKEND:-fastvideo-native}"
base_model="${MINIMAX_H3_MODEL_PATH:-/mnt/data/models/MiniMax-H3-r16}"
base_lora="${MINIMAX_H3_LORA_PATH:-/mnt/data/ComfyUI-aki-v3/ComfyUI/models/loras/minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors}"
fastvideo_python="${MINIMAX_H3_PYTHON:-/home/clhh/FastVideo/.venv/bin/python}"

if [[ "$base_backend" == "fastvideo-native" ]]; then
  test -x "$fastvideo_python"
  echo "fastvideo_python=ok path=$fastvideo_python"
  test -d "$base_model/transformer"
  test -d "$base_model/text_encoder"
  test -d "$base_model/vae"
  test -d "$base_model/audio_vae"
  test -f "$base_model/modular_model_index.json"
  echo "base_model=ok path=$base_model"
  test -f "$base_lora"
  echo "base_lora=ok path=$base_lora steps=${MINIMAX_H3_STEPS:-8}"
else
  models_dir="${COMFYUI_MODELS_DIR:-/mnt/data/ComfyUI-aki-v3/ComfyUI/models}"
  t2va_workflow="${COMFYUI_T2VA_WORKFLOW_TEMPLATE:-/mnt/data/ComfyUI-aki-v3/ComfyUI/user/default/workflows/video_minimax_h3_t2va_api.json}"
  fl2va_workflow="${COMFYUI_FL2VA_WORKFLOW_TEMPLATE:-/mnt/data/ComfyUI-aki-v3/ComfyUI/user/default/workflows/video_minimax_h3_fl2va_api.json}"
  comfyui_url="${COMFYUI_BASE_URL:-http://127.0.0.1:8189}"

  test -d "$models_dir"
  echo "models_dir=ok path=$models_dir"
  test -f "$t2va_workflow"
  test -f "$fl2va_workflow"
  curl -fsS --connect-timeout 3 --max-time 10 "$comfyui_url/system_stats" >/dev/null
  echo "comfyui_api=ok url=$comfyui_url"
fi

command -v nvidia-smi >/dev/null
nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits --id=0 >/dev/null
echo "nvidia_smi=ok"
command -v ffmpeg >/dev/null
command -v ffprobe >/dev/null
echo "ffmpeg=ok"
