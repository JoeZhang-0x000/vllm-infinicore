"""Shared benchmark helpers must preserve platform routing and lazy imports."""

from __future__ import annotations

import os
import subprocess
import sys
import unittest
from pathlib import Path

from vllm_infinicore.benchmarks.common import (
    ASCEND_FALLBACK_REASONS,
    mode_environment,
    verify_workers,
)

ROOT = Path(__file__).resolve().parents[1]


class BenchmarkRuntimeTests(unittest.TestCase):
    def test_mode_environment_preserves_vendor_plugins_and_parent_environment(self):
        parent = {
            "VLLM_PLUGINS": "vendor_model,vllm_infinicore,vendor_device",
            "VLLM_INFINICORE_ROUTES": "all",
        }
        for platform in ("ascend", "kunlun", "metax"):
            with self.subTest(platform=platform):
                native = mode_environment("native", parent, platform, ("Embedding",))
                plugin = mode_environment("infinicore", parent, platform, ("Embedding",))
                self.assertEqual(native["VLLM_PLUGINS"], "vendor_model,vendor_device")
                self.assertEqual(native["VLLM_INFINICORE_ENABLE_PATCHES"], "0")
                self.assertEqual(
                    plugin["VLLM_PLUGINS"], "vendor_model,vendor_device,vllm_infinicore"
                )
                self.assertEqual(plugin["VLLM_INFINICORE_OPERATOR_BACKEND"], platform)
                self.assertEqual(plugin["VLLM_INFINICORE_ROUTES"], "Embedding")
                self.assertEqual(plugin["VLLM_INFINICORE_CPP_BRIDGE_ROUTES"], "Embedding")
                self.assertEqual(plugin["VLLM_INFINICORE_STRICT_BACKEND"], "1")
        self.assertEqual(parent["VLLM_PLUGINS"], "vendor_model,vllm_infinicore,vendor_device")
        self.assertEqual(parent["VLLM_INFINICORE_ROUTES"], "all")
        self.assertNotIn("VLLM_INFINICORE_ENABLE_PATCHES", parent)

    def test_subset_routes_allow_unrequested_native_silu_on_ascend(self):
        worker = dict(
            rank=0,
            platform="ascend",
            graph_captures=1,
            graph_replays=3,
            registration=dict(installed_routes=["Embedding"]),
            backend_calls={"embedding": 1},
            bridge_calls={},
            attention_calls={},
            fallback_calls={},
            attention_fallbacks={},
            native_attention_calls={},
            known_native_paths={"silu_and_mul": ASCEND_FALLBACK_REASONS["silu_and_mul"]},
        )
        verify_workers("infinicore", [worker], False, True, routes=("Embedding",))

    def test_benchmark_imports_do_not_load_inference_dependencies(self):
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import sys; "
                "from vllm_infinicore.benchmarks import common, gsm8k, kunlun_static; "
                "assert not any(name.split('.')[0] in "
                "{'torch', 'vllm', 'kunlun_ops', 'transformers', 'modelscope', 'pyarrow'} "
                "for name in sys.modules)",
            ],
            cwd=ROOT,
            env=os.environ,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
