from __future__ import annotations

import argparse
import importlib.util
import os
from pathlib import Path
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SMOKE_SCRIPT = ROOT / "scripts" / "qwen3_128_32_smoke.py"


def load_smoke_module():
    spec = importlib.util.spec_from_file_location("qwen3_128_32_smoke", SMOKE_SCRIPT)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class QwenSmokeHarnessTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.smoke = load_smoke_module()

    def test_runtime_environment_respects_existing_vllm_plugins(self) -> None:
        args = argparse.Namespace(plugins="")

        with patch.dict(os.environ, {"VLLM_PLUGINS": "infinicore,vllm_infinicore"}, clear=True):
            self.smoke.configure_runtime_environment(args)
            self.assertEqual(os.environ["VLLM_PLUGINS"], "infinicore,vllm_infinicore")

    def test_no_metax_cases_select_infinicore_platform_and_forbid_metax(self) -> None:
        args = argparse.Namespace(
            plugins="",
            custom_routes="",
            force_native_fallback=False,
            disabled_routes="",
            cpp_bridge_routes="",
        )

        with patch.dict(os.environ, {}, clear=True):
            self.smoke.configure_runtime_environment(args)
            self.smoke.configure_case_environment("no-metax-graph", args)

            self.assertEqual(os.environ["VLLM_PLUGINS"], "infinicore,vllm_infinicore")
            self.assertEqual(os.environ["VLLM_INFINICORE_ENABLE_PATCHES"], "1")
            self.assertEqual(os.environ["VLLM_INFINICORE_ROUTES"], "all")
            self.assertEqual(os.environ["VLLM_INFINICORE_FORCE_NATIVE_FALLBACK"], "0")
            self.assertTrue(
                self.smoke.CASE_SPECS["no-metax-graph"]["forbid_metax_load"]
            )

    def test_child_command_forwards_no_metax_flags(self) -> None:
        args = argparse.Namespace(
            model="/model",
            input_len=128,
            output_len=32,
            warmup=1,
            repeats=2,
            gpu_memory_utilization=0.9,
            max_model_len=1024,
            trust_remote_code=True,
            skip_warmup=False,
            custom_routes="",
            disabled_routes="",
            cpp_bridge_routes="",
            force_native_fallback=False,
            plugins="infinicore,vllm_infinicore",
            forbid_metax_load=True,
        )

        command = self.smoke._child_command(
            args,
            "custom-graph",
            Path("prompt.json"),
            Path("case.json"),
        )

        self.assertIn("--plugins", command)
        self.assertIn("infinicore,vllm_infinicore", command)
        self.assertIn("--forbid-metax-load", command)


if __name__ == "__main__":
    unittest.main()
