from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path


EXPLICIT_BACKENDS = {
    "FLASH_ATTN",
    "SAGE_ATTN",
    "SAGE_ATTN_THREE",
    "VIDEO_SPARSE_ATTN_H3",
}
SAGE_BACKENDS = ("SAGE_ATTN_THREE", "SAGE_ATTN")
_PROBE_TIMEOUT_SECONDS = 8

_PROBE_SOURCE = """
import importlib.util
import json

def _has(name):
    return importlib.util.find_spec(name) is not None

sage_attn = _has("sageattention")
sage_attn_three = _has("sageattn3") or _has("sageattention3")
supported = None
error = None
try:
    from fastvideo.configs.models.dits.minimax_h3 import MiniMaxH3ArchConfig
    supported = [
        backend.name
        for backend in MiniMaxH3ArchConfig()._supported_attention_backends
    ]
except Exception as exc:
    error = f"{type(exc).__name__}: {exc}"

print(
    json.dumps(
        {
            "sage_attn": sage_attn,
            "sage_attn_three": sage_attn_three,
            "supported": supported,
            "error": error,
        }
    )
)
"""


@dataclass(frozen=True, slots=True)
class SageAttentionProbe:
    sage_attn_available: bool
    sage_attn_three_available: bool
    h3_supported_backends: tuple[str, ...] | None
    detail: str

    def usable_backend(self) -> str | None:
        supported = self.h3_supported_backends
        if supported is None:
            return None
        if self.sage_attn_three_available and "SAGE_ATTN_THREE" in supported:
            return "SAGE_ATTN_THREE"
        if self.sage_attn_available and "SAGE_ATTN" in supported:
            return "SAGE_ATTN"
        return None


@dataclass(frozen=True, slots=True)
class AttentionResolution:
    configured: str
    resolved: str
    sage_mode: str
    sage_available: bool
    sage_attn_available: bool
    sage_attn_three_available: bool
    h3_supports_sage: bool | None
    required_missing: bool
    reason: str


def empty_sage_probe(detail: str = "SageAttention was not probed") -> SageAttentionProbe:
    return SageAttentionProbe(
        sage_attn_available=False,
        sage_attn_three_available=False,
        h3_supported_backends=None,
        detail=detail,
    )


def probe_sage_attention(python: Path | None) -> SageAttentionProbe:
    if python is None or not Path(python).is_file():
        return empty_sage_probe("FastVideo Python is missing; SageAttention was not probed")
    try:
        completed = subprocess.run(
            [str(python), "-c", _PROBE_SOURCE],
            check=False,
            capture_output=True,
            text=True,
            timeout=_PROBE_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return empty_sage_probe(f"SageAttention probe failed: {exc}")
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "").strip()[-500:]
        return empty_sage_probe(
            "SageAttention probe failed"
            + (f": {detail}" if detail else "")
        )
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError:
        return empty_sage_probe("SageAttention probe returned invalid JSON")
    supported_raw = payload.get("supported")
    supported = (
        tuple(str(item) for item in supported_raw)
        if isinstance(supported_raw, list)
        else None
    )
    detail = str(payload.get("error") or "SageAttention probe succeeded")
    return SageAttentionProbe(
        sage_attn_available=bool(payload.get("sage_attn")),
        sage_attn_three_available=bool(payload.get("sage_attn_three")),
        h3_supported_backends=supported,
        detail=detail,
    )


def resolve_attention_backend(
    configured: str,
    sage_mode: str,
    probe: SageAttentionProbe | None = None,
) -> AttentionResolution:
    probe = probe or empty_sage_probe()
    h3_supports_sage = None
    if probe.h3_supported_backends is not None:
        h3_supports_sage = any(
            backend in probe.h3_supported_backends for backend in SAGE_BACKENDS
        )
    preferred = probe.usable_backend()

    def result(
        resolved: str,
        reason: str,
        *,
        required_missing: bool = False,
        sage_available: bool | None = None,
    ) -> AttentionResolution:
        return AttentionResolution(
            configured=configured,
            resolved=resolved,
            sage_mode=sage_mode,
            sage_available=(
                preferred is not None if sage_available is None else sage_available
            ),
            sage_attn_available=probe.sage_attn_available,
            sage_attn_three_available=probe.sage_attn_three_available,
            h3_supports_sage=h3_supports_sage,
            required_missing=required_missing,
            reason=reason,
        )

    if configured == "VIDEO_SPARSE_ATTN_H3":
        return result(
            configured,
            "VSA profile is unchanged; SageAttention auto-detect does not apply",
        )
    if configured in EXPLICIT_BACKENDS:
        return result(configured, "explicit attention backend is unchanged")

    fallback = "TORCH_SDPA"
    if sage_mode == "off":
        return result(
            fallback,
            "SageAttention is disabled; dense FastH3 uses TORCH_SDPA",
        )
    if preferred is not None:
        return result(
            preferred,
            f"SageAttention auto-detect selected {preferred}",
        )

    required_missing = sage_mode == "on"
    if required_missing:
        reason = (
            "SageAttention is required but not usable for MiniMax H3: "
            + probe.detail
        )
    elif probe.sage_attn_available or probe.sage_attn_three_available:
        reason = (
            "SageAttention is present in the worker venv, but MiniMax H3 "
            "does not list SAGE_ATTN/SAGE_ATTN_THREE; dense FastH3 keeps "
            "TORCH_SDPA"
        )
    else:
        reason = (
            "SageAttention is not usable for MiniMax H3; dense FastH3 keeps "
            "TORCH_SDPA: "
            + probe.detail
        )
    return result(
        fallback,
        reason,
        required_missing=required_missing,
        sage_available=False,
    )
