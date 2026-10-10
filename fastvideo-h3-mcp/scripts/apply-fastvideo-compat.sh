#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fastvideo_dir="${1:-/home/clhh/FastVideo}"
apply_patch_if_missing() {
  local patch_name="$1"
  local target_file="$2"
  local marker="$3"
  local patch_file="$project_dir/patches/$patch_name"

  if grep -Fq -- "$marker" "$fastvideo_dir/$target_file"; then
    echo "$patch_name is already applied."
    return
  fi

  patch --batch --force --dry-run --silent -d "$fastvideo_dir" -p1 < "$patch_file"
  patch --batch --force --silent -d "$fastvideo_dir" -p1 < "$patch_file"
  echo "Applied $patch_name."
}

apply_patch_if_missing \
  fastvideo-direct-device-load.patch \
  fastvideo/models/loader/fsdp_load.py \
  FASTVIDEO_LOAD_WEIGHTS_DIRECT_TO_DEVICE
apply_patch_if_missing \
  fastvideo-h3-text-encoder-fp8.patch \
  examples/inference/basic/basic_minimax_h3_t2v.py \
  --text-encoder-fp8
apply_patch_if_missing \
  fastvideo-h3-fp8-cpu-offload.patch \
  examples/inference/basic/basic_minimax_h3_t2v.py \
  'text_encoder=True'
apply_patch_if_missing \
  fastvideo-h3-vsa-profile.patch \
  examples/inference/basic/basic_minimax_h3_t2v.py \
  --vsa-tile-size
apply_patch_if_missing \
  fastvideo-h3-vsa-cache-tile-buffer.patch \
  fastvideo/attention/backends/video_sparse_attn_h3.py \
  FASTVIDEO_VSA_CACHE_TILE_BUFFER
apply_patch_if_missing \
  fastvideo-absmax-fp8-fsdp-metadata.patch \
  fastvideo/layers/quantization/absmax_fp8.py \
  'output_dtype: torch.dtype = getattr('
apply_patch_if_missing \
  fastvideo-pipeline-stage-progress.patch \
  fastvideo/pipelines/composed_pipeline_base.py \
  'MCP_PROGRESS pipeline_stage=%s phase=started'
apply_patch_if_missing \
  fastvideo-h3-joint-progress.patch \
  fastvideo/pipelines/basic/minimax_h3/stages/minimax_h3_denoising.py \
  'MCP_PROGRESS h3_joint_denoise step=%d/%d phase=started'
apply_patch_if_missing \
  fastvideo-h3-release-dit-before-decode.patch \
  fastvideo/pipelines/basic/minimax_h3/stages/minimax_h3_denoising.py \
  FASTVIDEO_MINIMAX_H3_RELEASE_DIT_AFTER_DENOISE
apply_patch_if_missing \
  fastvideo-h3-60s-duration.patch \
  fastvideo/pipelines/basic/minimax_h3/packing.py \
  'MINIMAX_H3_MAX_DURATION = 60.0'
apply_patch_if_missing \
  fastvideo-h3-ff-chunking.patch \
  fastvideo/models/dits/minimax_h3.py \
  'FASTVIDEO_MINIMAX_H3_FF_CHUNK_TOKENS'
apply_patch_if_missing \
  fastvideo-h3-attention-residual-chunking.patch \
  fastvideo/models/dits/minimax_h3.py \
  'FASTVIDEO_VSA_ATTENTION_RESIDUAL_CHUNK_TOKENS'
apply_patch_if_missing \
  fastvideo-h3-block-progress.patch \
  fastvideo/models/dits/minimax_h3.py \
  'MCP_PROGRESS h3_transformer_block step=%d/%d block=%d/%d phase=completed'
apply_patch_if_missing \
  fastvideo-h3-block-progress-throttle.patch \
  fastvideo/models/dits/minimax_h3.py \
  'block_progress_stride = ('
apply_patch_if_missing \
  fastvideo-h3-rank-conversion-skip-error-report.patch \
  scripts/checkpoint_conversion/convert_minimax_h3_adaln_rank.py \
  --skip-error-report
apply_patch_if_missing \
  fastvideo-diffusers-lora-default-key.patch \
  fastvideo/pipelines/lora_pipeline.py \
  '.lora_A.default.'
apply_patch_if_missing \
  fastvideo-inference-lora-alpha-strength.patch \
  fastvideo/api/compat.py \
  'components.lora_alpha'
apply_patch_if_missing \
  fastvideo-lora-optional-cpu-backup.patch \
  fastvideo/layers/lora/linear.py \
  FASTVIDEO_LORA_CPU_BACKUP
apply_patch_if_missing \
  fastvideo-minimax-h3-lora-pipeline.patch \
  fastvideo/pipelines/basic/minimax_h3/minimax_h3_pipeline.py \
  'class MiniMaxH3BasePipeline(LoRAPipeline)'
