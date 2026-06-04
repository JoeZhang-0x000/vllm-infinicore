from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
THROUGHPUT_SCRIPT = ROOT / "scripts" / "qwen3_three_engine_throughput.py"


def load_throughput_module():
    spec = importlib.util.spec_from_file_location(
        "qwen3_three_engine_throughput",
        THROUGHPUT_SCRIPT,
    )
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class QwenThroughputHarnessTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.throughput = load_throughput_module()

    def test_worker_env_uses_configured_no_metax_plugins(self) -> None:
        manifest = {
            "infinilm_root": "/root/InfiniLM",
            "vllm_infinicore_plugins": "infinicore,vllm_infinicore",
            "vllm_native_plugins": "infinicore",
        }

        infinicore_env = self.throughput._worker_env(
            manifest,
            "vllm-infinicore",
        )
        native_env = self.throughput._worker_env(manifest, "vllm-native")

        self.assertEqual(
            infinicore_env["VLLM_PLUGINS"],
            "infinicore,vllm_infinicore",
        )
        self.assertEqual(native_env["VLLM_PLUGINS"], "infinicore")


if __name__ == "__main__":
    unittest.main()
