#!/usr/bin/env bash
set -euo pipefail

fastvideo_dir="${FASTVIDEO_REPO:-/home/clhh/FastVideo}"
source_model="${1:-/mnt/data/ComfyUI-aki-v3/ComfyUI/models/diffusers/MiniMax-H3}"
target_model="${2:-/mnt/data/models/MiniMax-H3-r16}"
python_binary="${MINIMAX_H3_PYTHON:-$fastvideo_dir/.venv/bin/python}"
converter="$fastvideo_dir/scripts/checkpoint_conversion/convert_minimax_h3_adaln_rank.py"
lora_model_id="lightx2v/Minimax-h3-Turbo"
lora_filename="minimax_h3_fl2v_turbo_8step_v1.0_bf16.safetensors"
lora_sha256="e16ac20824d6e6649b193806f8fb095639bd9946c97b1bb84b4248eab1cc807f"
lora_dir="${MINIMAX_H3_LORA_DIR:-/mnt/data/models/MiniMax-H3-Turbo}"
lora_path="${MINIMAX_H3_LORA_PATH:-$lora_dir/$lora_filename}"
lora_parent="$(dirname "$lora_path")"

test -x "$python_binary"
test -f "$converter"
test -f "$source_model/transformer/diffusion_pytorch_model.safetensors.index.json"

if [[ ! -f "$target_model/transformer/config.json" ]]; then
  if [[ -e "$target_model/transformer" ]]; then
    echo "Refusing to overwrite incomplete target: $target_model/transformer" >&2
    exit 1
  fi
  mkdir -p "$target_model"
  staging_transformer="$target_model/.transformer-r16-building"
  if [[ -e "$staging_transformer" ]]; then
    echo "Refusing to overwrite an earlier conversion staging directory: $staging_transformer" >&2
    exit 1
  fi
  "$python_binary" "$converter" \
    --src "$source_model/transformer" \
    --dst "$staging_transformer" \
    --rank 16 \
    --skip-error-report
  mv "$staging_transformer" "$target_model/transformer"
fi

components=(
  audio_scheduler
  audio_vae
  configuration.json
  modular_model_index.json
  processor
  scheduler
  text_encoder
  tokenizer
  vae
)
for component in "${components[@]}"; do
  source_path="$source_model/$component"
  target_path="$target_model/$component"
  test -e "$source_path"
  if [[ -e "$target_path" || -L "$target_path" ]]; then
    continue
  fi
  ln -s "$source_path" "$target_path"
done

mkdir -p "$lora_parent"
if [[ -f "$lora_path" ]]; then
  actual_lora_sha256="$(sha256sum "$lora_path" | awk '{print $1}')"
  if [[ "$actual_lora_sha256" != "$lora_sha256" ]]; then
    echo "Refusing incompatible LoRA at $lora_path (sha256=$actual_lora_sha256)" >&2
    exit 1
  fi
else
  lora_staging_dir="$lora_parent/.modelscope-download"
  mkdir -p "$lora_staging_dir"
  "$python_binary" - "$lora_model_id" "$lora_filename" "$lora_staging_dir" <<'PY'
import sys

from modelscope.hub.file_download import model_file_download

downloaded = model_file_download(
    model_id=sys.argv[1],
    file_path=sys.argv[2],
    revision="master",
    local_dir=sys.argv[3],
)
print(f"modelscope_lora_download={downloaded}")
PY
  lora_staging="$lora_staging_dir/$lora_filename"
  actual_lora_sha256="$(sha256sum "$lora_staging" | awk '{print $1}')"
  if [[ "$actual_lora_sha256" != "$lora_sha256" ]]; then
    echo "Downloaded LoRA failed sha256 validation: $actual_lora_sha256" >&2
    exit 1
  fi
  mv "$lora_staging" "$lora_path"
fi

"$python_binary" - "$target_model/transformer/config.json" <<'PY'
import json
import sys

config = json.load(open(sys.argv[1], encoding="utf-8"))
rank = config.get("adaln_rank")
if rank != 16:
    raise SystemExit(f"unexpected adaln_rank: {rank!r}")
print(f"base_model=ready adaln_rank={rank}")
PY

echo "base_model_path=$target_model"
echo "base_lora_path=$lora_path"
