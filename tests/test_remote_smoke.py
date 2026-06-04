from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
REMOTE_SMOKE = ROOT / "tests" / "remote" / "run_qwen_smoke.py"


def load_remote_smoke_module():
    spec = importlib.util.spec_from_file_location("run_qwen_smoke", REMOTE_SMOKE)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class RemoteSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.smoke = load_remote_smoke_module()

    def test_forbid_metax_selects_no_metax_plugins_by_default(self) -> None:
        with patch.dict(os.environ, {"VLLM_SMOKE_FORBID_METAX_LOAD": "1"}, clear=True):
            environment = self.smoke.configure_environment()

        self.assertEqual(environment["VLLM_PLUGINS"], "infinicore,vllm_infinicore")
        self.assertEqual(environment["VLLM_INFINICORE_ENABLE_PATCHES"], "1")
        self.assertEqual(environment["VLLM_INFINICORE_ROUTES"], "all")
        self.assertEqual(environment["VLLM_INFINICORE_FORCE_NATIVE_FALLBACK"], "0")
        self.assertEqual(environment["VLLM_INFINICORE_STRICT_BACKEND"], "1")

    def test_explicit_plugins_are_respected(self) -> None:
        with patch.dict(
            os.environ,
            {
                "VLLM_SMOKE_FORBID_METAX_LOAD": "1",
                "VLLM_PLUGINS": "custom_platform,vllm_infinicore",
            },
            clear=True,
        ):
            environment = self.smoke.configure_environment()

        self.assertEqual(environment["VLLM_PLUGINS"], "custom_platform,vllm_infinicore")

    def test_default_remains_metax_compatible(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            environment = self.smoke.configure_environment()

        self.assertEqual(environment["VLLM_PLUGINS"], "metax,vllm_infinicore")


if __name__ == "__main__":
    unittest.main()
