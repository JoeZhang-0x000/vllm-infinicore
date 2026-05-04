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
        except Exception as exc:  # pragma: no cover - depends on local install
            self.skipTest(f"torch unavailable: {exc}")

        with mock.patch.dict(
            os.environ,
            {custom_ops.CUSTOM_OP_ENABLE_ENV: "1"},
            clear=False,
        ):
            status = custom_ops.load_custom_ops()
            self.assertTrue(status.available, status.reason)
            self.assertEqual(status.registered_ops, (custom_ops.RMS_NORM_OP,))

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


if __name__ == "__main__":
    unittest.main()
