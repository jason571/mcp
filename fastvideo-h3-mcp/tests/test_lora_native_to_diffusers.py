from __future__ import annotations

import unittest

from fastvideo_h3_mcp.lora_native_to_diffusers import (
    convert_native_minimax_h3_lora,
    is_native_minimax_h3_lora,
)


class FakeTensor:
    def __init__(self, rows: int, cols: int, tag: str = "") -> None:
        self.tag = tag
        self._data = [[(row, col) for col in range(cols)] for row in range(rows)]

    @property
    def shape(self) -> tuple[int, int]:
        return (len(self._data), len(self._data[0]) if self._data else 0)

    def __getitem__(self, slc: slice) -> FakeTensor:
        rows = self._data[slc]
        cloned = FakeTensor(0, 0, self.tag)
        cloned._data = [list(row) for row in rows]
        return cloned

    def clone(self) -> FakeTensor:
        cloned = FakeTensor(0, 0, self.tag)
        cloned._data = [list(row) for row in self._data]
        return cloned


class NativeLoRAConversionTests(unittest.TestCase):
    def test_detects_comfyui_fused_qkv_keys(self) -> None:
        self.assertTrue(
            is_native_minimax_h3_lora(
                (
                    "diffusion_model.blocks.0.attn.qkv_proj.lora_A.weight",
                    "diffusion_model.blocks.0.attn.qkv_proj.lora_B.weight",
                )
            )
        )

    def test_leaves_diffusers_to_q_adapters_alone(self) -> None:
        self.assertFalse(
            is_native_minimax_h3_lora(
                (
                    "transformer_blocks.0.attn.to_q.lora_A.weight",
                    "transformer_blocks.0.attn.to_k.lora_A.weight",
                )
            )
        )

    def test_splits_fused_qkv_and_renames_companion_layers(self) -> None:
        state = {
            "diffusion_model.blocks.0.attn.qkv_proj.lora_A.weight": FakeTensor(
                2, 8, "A"
            ),
            "diffusion_model.blocks.0.attn.qkv_proj.lora_B.weight": FakeTensor(
                12, 2, "B"
            ),
            "diffusion_model.blocks.0.attn.out_proj.lora_A.weight": FakeTensor(
                2, 4, "O"
            ),
            "diffusion_model.blocks.0.mlp.fc1.lora_A.weight": FakeTensor(2, 4, "F"),
            "diffusion_model.blocks.0.mlp.fc2.lora_A.weight": FakeTensor(2, 4, "G"),
            "token_refiner.0.attn.qkv_proj.lora_A.weight": FakeTensor(2, 8, "R"),
            "blocks.0.adaln_modulation.lora_A.weight": FakeTensor(2, 4, "N"),
        }

        converted = convert_native_minimax_h3_lora(state)

        to_q_b = converted["transformer_blocks.0.attn.to_q.lora_B.weight"]
        to_k_b = converted["transformer_blocks.0.attn.to_k.lora_B.weight"]
        self.assertEqual(to_q_b.shape, (4, 2))
        self.assertEqual(to_k_b.shape, (4, 2))
        self.assertEqual(to_q_b._data[0][0], (0, 0))
        self.assertEqual(to_k_b._data[0][0], (4, 0))
        self.assertEqual(
            converted["transformer_blocks.0.attn.to_q.lora_A.weight"].shape, (2, 8)
        )
        self.assertEqual(
            converted["transformer_blocks.0.attn.to_out.lora_A.weight"].shape,
            (2, 4),
        )
        self.assertNotIn(
            "transformer_blocks.0.ff.fc_in.lora_A.weight", converted
        )
        self.assertFalse(
            any(
                "qkv_proj" in key or "token_refiner" in key or "adaln" in key or ".mlp." in key
                for key in converted
            )
        )


if __name__ == "__main__":
    unittest.main()
