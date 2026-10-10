"""Convert native/ComfyUI MiniMax H3 LoRA keys into FastVideo Diffusers keys.

ComfyUI MiniMax H3 uses fused ``blocks.*.attn.qkv_proj``. FastVideo's r16 DiT
uses split ``transformer_blocks.*.attn.to_q/k/v``. The 8-step Turbo adapter is
the former; this rewrite is done in the GPU worker so the MCP web process never
imports torch.
"""

from __future__ import annotations

import json
import struct
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

_MAX_SAFETENSORS_HEADER_BYTES = 16 * 1024 * 1024
_NATIVE_PREFIXES = ("diffusion_model.",)
_LAYER_RENAMES = (
    (".attn.out_proj.", ".attn.to_out."),
)
_QKV_PARTS = ("to_q", "to_k", "to_v")


def read_safetensors_tensor_names(path: Path) -> tuple[str, ...]:
    with path.open("rb") as handle:
        size_bytes = handle.read(8)
        if len(size_bytes) != 8:
            raise ValueError("file is too short to be a safetensors LoRA")
        header_len = struct.unpack("<Q", size_bytes)[0]
        if header_len < 2 or header_len > _MAX_SAFETENSORS_HEADER_BYTES:
            raise ValueError(f"invalid safetensors header size: {header_len}")
        raw = handle.read(header_len)
        if len(raw) != header_len:
            raise ValueError("truncated safetensors header")
    header = json.loads(raw)
    if not isinstance(header, dict):
        raise ValueError("safetensors header is not an object")
    return tuple(key for key in header if key != "__metadata__")


def is_native_minimax_h3_lora(keys: tuple[str, ...] | list[str]) -> bool:
    stripped = tuple(_strip_prefix(key) for key in keys)
    has_qkv = any("qkv_proj" in key for key in stripped)
    has_diffusers = any(
        "transformer_blocks" in key and ".to_q" in key for key in stripped
    )
    return has_qkv and not has_diffusers


def convert_native_minimax_h3_lora(
    state: Mapping[str, Any],
) -> dict[str, Any]:
    converted: dict[str, Any] = {}
    for name, tensor in state.items():
        mapped = _convert_one(name, tensor)
        for new_name, new_tensor in mapped.items():
            if new_name in converted:
                raise ValueError(f"native LoRA mapped two tensors onto {new_name}")
            converted[new_name] = new_tensor
    if not converted:
        raise ValueError("native MiniMax H3 LoRA produced no Diffusers tensors")
    if not any(".attn.to_q." in key for key in converted):
        raise ValueError("native MiniMax H3 LoRA did not yield attn.to_q tensors")
    return converted


def prepare_fastvideo_lora_path(lora_path: str | Path) -> Path:
    source = Path(lora_path).expanduser()
    if not source.is_file():
        raise FileNotFoundError(f"MINIMAX_H3_LORA_PATH does not exist: {source}")
    keys = read_safetensors_tensor_names(source)
    if not is_native_minimax_h3_lora(keys):
        return source
    cached = _cache_path(source)
    if _cache_is_current(source, cached):
        return cached
    from safetensors.torch import load_file, save_file

    converted = convert_native_minimax_h3_lora(load_file(str(source)))
    for candidate in (cached, Path(tempfile.gettempdir()) / cached.name):
        try:
            staging = candidate.with_suffix(candidate.suffix + ".building")
            save_file(converted, str(staging))
            staging.replace(candidate)
            return candidate
        except OSError:
            continue
    raise RuntimeError(f"unable to write converted FastVideo LoRA for {source}")


def _convert_one(name: str, tensor: Any) -> dict[str, Any]:
    stripped = _strip_prefix(name)
    if _should_drop(stripped):
        return {}
    if ".attn.qkv_proj." in stripped:
        return _split_qkv(_to_transformer_blocks(stripped), tensor)
    renamed = _rename_layer(_to_transformer_blocks(stripped))
    if renamed is None:
        return {}
    return {renamed: tensor}


def _split_qkv(name: str, tensor: Any) -> dict[str, Any]:
    kind = _lora_side(name)
    chunks = _qkv_chunks(tensor, kind)
    converted: dict[str, Any] = {}
    for part, chunk in zip(_QKV_PARTS, chunks, strict=True):
        converted[name.replace(".attn.qkv_proj.", f".attn.{part}.")] = chunk
    return converted


def _qkv_chunks(tensor: Any, kind: str) -> list[Any]:
    if kind == "alpha":
        return [_copy_tensor(tensor), _copy_tensor(tensor), _copy_tensor(tensor)]
    if kind == "A":
        # Fused nn.Linear(in, 3*hidden) LoRA keeps one A and a stacked B.
        return [_copy_tensor(tensor), _copy_tensor(tensor), _copy_tensor(tensor)]
    if tensor.shape[0] % 3 != 0:
        raise ValueError(
            "fused qkv_proj LoRA B cannot be split into to_q/to_k/to_v: "
            f"shape={tuple(tensor.shape)}"
        )
    width = tensor.shape[0] // 3
    return [tensor[index * width : (index + 1) * width] for index in range(3)]


def _lora_side(name: str) -> str:
    if ".lora_alpha" in name or name.endswith(".alpha"):
        return "alpha"
    if ".lora_A." in name or ".lora_down." in name:
        return "A"
    if ".lora_B." in name or ".lora_up." in name:
        return "B"
    raise ValueError(f"unsupported fused qkv LoRA tensor: {name}")


def _to_transformer_blocks(name: str) -> str:
    if name.startswith("blocks."):
        return "transformer_blocks." + name[len("blocks.") :]
    return name


def _rename_layer(name: str) -> str | None:
    if not name.startswith("transformer_blocks."):
        return None
    if "mlp." in name:
        return None
    for old, new in _LAYER_RENAMES:
        name = name.replace(old, new)
    if "qkv_proj" in name:
        return None
    return name


def _should_drop(name: str) -> bool:
    lowered = name.lower()
    if name.startswith("token_refiner.") or ".token_refiner." in name:
        return True
    if name.startswith("time_embedder.") or ".time_embedder." in name:
        return True
    if "adaln" in lowered:
        return True
    return False


def _strip_prefix(name: str) -> str:
    for prefix in _NATIVE_PREFIXES:
        if name.startswith(prefix):
            return name[len(prefix) :]
    return name


def _copy_tensor(tensor: Any) -> Any:
    clone = getattr(tensor, "clone", None)
    if callable(clone):
        return clone()
    copy = getattr(tensor, "copy", None)
    if callable(copy):
        return copy()
    return tensor


def _cache_path(source: Path) -> Path:
    stat = source.stat()
    stamp = f"{stat.st_mtime_ns:x}-{stat.st_size:x}"
    return source.with_name(f"{source.stem}.fastvideo-diffusers.{stamp}.safetensors")


def _cache_is_current(source: Path, cached: Path) -> bool:
    if not cached.is_file():
        return False
    try:
        keys = read_safetensors_tensor_names(cached)
    except (OSError, ValueError, json.JSONDecodeError, UnicodeDecodeError):
        return False
    return not is_native_minimax_h3_lora(keys)
