from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import unittest
from unittest import mock

from vllm_infinicore.ops import custom_ops

ROOT = Path(__file__).resolve().parents[1]


class CustomOpsTests(unittest.TestCase):
    def test_default_loader_does_not_import_torch(self) -> None:
        code = """
import os
import sys
os.environ.pop("VLLM_INFINICORE_ENABLE_CUSTOM_OPS", None)
from vllm_infinicore.ops import load_custom_ops
status = load_custom_ops()
print(status.available)
print(status.registered_ops)
print("torch" in sys.modules)
"""
        result = subprocess.run(
            [sys.executable, "-c", code],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )

        self.assertEqual(
            result.stdout.strip().splitlines(),
            ["False", "()", "False"],
        )

    def test_rms_norm_prototype_matches_torch_formula(self) -> None:
        try:
            import torch
            import torch.nn.functional as F
        except Exception as exc:  # pragma: no cover - depends on local install
            self.skipTest(f"torch unavailable: {exc}")

        with mock.patch.dict(
            os.environ,
            {custom_ops.CUSTOM_OP_ENABLE_ENV: "1"},
            clear=False,
        ):
            status = custom_ops.load_custom_ops()
            self.assertTrue(status.available, status.reason)
            for op_name in custom_ops.ALL_CUSTOM_OPS:
                self.assertIn(op_name, status.registered_ops)

            input_tensor = torch.tensor(
                [[1.0, -2.0, 3.0], [0.5, 1.5, -2.5]],
                dtype=torch.float32,
            )
            weight = torch.tensor([1.0, 0.5, 2.0], dtype=torch.float32)
            eps = 1e-6

            output = custom_ops.rms_norm(input_tensor, weight, eps)
            expected = input_tensor.float() * torch.rsqrt(
                input_tensor.float().pow(2).mean(dim=-1, keepdim=True) + eps
            )
            expected = expected.to(dtype=input_tensor.dtype) * weight

            torch.testing.assert_close(output, expected)

            silu_input = torch.tensor(
                [[0.5, -1.0, 2.0, 3.0, -4.0, 5.0]],
                dtype=torch.float32,
            )
            silu_output = custom_ops.silu_and_mul(silu_input)
            silu_expected = F.silu(silu_input[..., :3]) * silu_input[..., 3:]
            torch.testing.assert_close(silu_output, silu_expected)

            linear_input = torch.tensor(
                [[1.0, -2.0, 0.5, 3.0], [-1.0, 2.0, 1.5, 0.0]],
                dtype=torch.float32,
            )
            linear_weight = torch.tensor(
                [[0.5, -1.0, 0.25, 2.0], [1.0, 0.0, -0.5, 0.75]],
                dtype=torch.float32,
            )
            linear_bias = torch.tensor([0.25, -0.75], dtype=torch.float32)
            linear_expected = F.linear(linear_input, linear_weight, linear_bias)
            torch.testing.assert_close(
                custom_ops.linear(linear_input, linear_weight, linear_bias),
                linear_expected,
            )
            torch.testing.assert_close(
                custom_ops.lm_head(linear_input, linear_weight, linear_bias),
                linear_expected,
            )

            embedding_input = torch.tensor([0, 2, 4], dtype=torch.long)
            embedding_weight = torch.arange(15, dtype=torch.float32).view(5, 3)
            torch.testing.assert_close(
                custom_ops.embedding(embedding_input, embedding_weight),
                F.embedding(embedding_input, embedding_weight),
            )

            positions = torch.tensor([0, 1], dtype=torch.long)
            angles = torch.arange(6, dtype=torch.float32).view(3, 2) * 0.1
            cos_sin_cache = torch.cat((angles.cos(), angles.sin()), dim=-1)
            query = torch.arange(16, dtype=torch.float32).view(2, 8) / 10
            key = query + 1.0
            rotated_query, rotated_key = custom_ops.rotary_embedding(
                positions,
                query,
                key,
                4,
                4,
                cos_sin_cache,
                True,
            )
            expected_query = _rotate_neox(positions, query, cos_sin_cache)
            expected_key = _rotate_neox(positions, key, cos_sin_cache)
            torch.testing.assert_close(rotated_query, expected_query)
            torch.testing.assert_close(rotated_key, expected_key)

    def test_force_load_does_not_enable_direct_api_without_env(self) -> None:
        try:
            import torch
        except Exception as exc:  # pragma: no cover - depends on local install
            self.skipTest(f"torch unavailable: {exc}")

        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop(custom_ops.CUSTOM_OP_ENABLE_ENV, None)
            status = custom_ops.load_custom_ops(
                force=True,
                required_ops=(custom_ops.RMS_NORM_OP,),
            )
            self.assertTrue(status.available, status.reason)

            input_tensor = torch.ones((1, 4), dtype=torch.float32)
            weight = torch.ones((4,), dtype=torch.float32)
            with self.assertRaisesRegex(RuntimeError, custom_ops.CUSTOM_OP_ENABLE_ENV):
                custom_ops.rms_norm(input_tensor, weight)


def _rotate_neox(positions, tensor, cos_sin_cache):
    import torch

    cos, sin = cos_sin_cache.index_select(0, positions).chunk(2, dim=-1)
    view = tensor.view(positions.shape[0], -1, 4)
    first, second = view.chunk(2, dim=-1)
    cos = cos.unsqueeze(-2)
    sin = sin.unsqueeze(-2)
    rotated = torch.cat(
        (first * cos - second * sin, second * cos + first * sin),
        dim=-1,
    )
    return rotated.reshape_as(tensor)


if __name__ == "__main__":
    unittest.main()
