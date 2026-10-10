from __future__ import annotations

import unittest

from fastvideo_h3_mcp.sage_attention import (
    SageAttentionProbe,
    resolve_attention_backend,
)


def _probe(
    *,
    sage_attn: bool = False,
    sage_attn_three: bool = False,
    supported: tuple[str, ...] | None = None,
) -> SageAttentionProbe:
    return SageAttentionProbe(
        sage_attn_available=sage_attn,
        sage_attn_three_available=sage_attn_three,
        h3_supported_backends=supported,
        detail="test probe",
    )


class SageAttentionResolutionTests(unittest.TestCase):
    def test_vsa_is_never_rewritten(self) -> None:
        resolution = resolve_attention_backend(
            "VIDEO_SPARSE_ATTN_H3",
            "auto",
            _probe(
                sage_attn=True,
                sage_attn_three=True,
                supported=("SAGE_ATTN", "SAGE_ATTN_THREE", "VIDEO_SPARSE_ATTN_H3"),
            ),
        )
        self.assertEqual(resolution.resolved, "VIDEO_SPARSE_ATTN_H3")
        self.assertFalse(resolution.required_missing)

    def test_flash_attn_explicit_is_unchanged(self) -> None:
        resolution = resolve_attention_backend(
            "FLASH_ATTN",
            "auto",
            _probe(sage_attn=True, supported=("FLASH_ATTN", "SAGE_ATTN")),
        )
        self.assertEqual(resolution.resolved, "FLASH_ATTN")

    def test_auto_selects_sage_attn_three_when_h3_supports_it(self) -> None:
        resolution = resolve_attention_backend(
            "TORCH_SDPA",
            "auto",
            _probe(
                sage_attn=True,
                sage_attn_three=True,
                supported=("TORCH_SDPA", "SAGE_ATTN", "SAGE_ATTN_THREE"),
            ),
        )
        self.assertEqual(resolution.resolved, "SAGE_ATTN_THREE")
        self.assertTrue(resolution.sage_available)

    def test_auto_keeps_torch_sdpa_when_h3_does_not_list_sage(self) -> None:
        resolution = resolve_attention_backend(
            "TORCH_SDPA",
            "auto",
            _probe(
                sage_attn=True,
                sage_attn_three=True,
                supported=("TORCH_SDPA", "FLASH_ATTN", "VIDEO_SPARSE_ATTN_H3"),
            ),
        )
        self.assertEqual(resolution.resolved, "TORCH_SDPA")
        self.assertFalse(resolution.sage_available)
        self.assertFalse(resolution.required_missing)

    def test_on_marks_required_missing_when_sage_is_unusable(self) -> None:
        resolution = resolve_attention_backend(
            "AUTO",
            "on",
            _probe(supported=("TORCH_SDPA", "FLASH_ATTN")),
        )
        self.assertEqual(resolution.resolved, "TORCH_SDPA")
        self.assertTrue(resolution.required_missing)

    def test_off_keeps_torch_sdpa_even_when_sage_is_usable(self) -> None:
        resolution = resolve_attention_backend(
            "TORCH_SDPA",
            "off",
            _probe(
                sage_attn=True,
                supported=("TORCH_SDPA", "SAGE_ATTN"),
            ),
        )
        self.assertEqual(resolution.resolved, "TORCH_SDPA")


if __name__ == "__main__":
    unittest.main()
