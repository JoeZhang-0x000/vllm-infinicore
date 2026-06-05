from __future__ import annotations

from pathlib import Path
import os
import subprocess
import sys
import unittest
from unittest import mock

from vllm_infinicore import platform

ROOT = Path(__file__).resolve().parents[1]


class PlatformPluginTests(unittest.TestCase):
    def test_register_platform_returns_infinicore_platform_class_path(self) -> None:
        self.assertEqual(
            platform.register_platform(),
            "vllm_infinicore.platform.InfiniCorePlatform",
        )

    def test_platform_entry_point_does_not_import_vllm_or_torch(self) -> None:
        code = """
import sys
from vllm_infinicore.platform import register_platform
print(register_platform())
print("torch" in sys.modules)
print("vllm" in sys.modules)
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
            [
                "vllm_infinicore.platform.InfiniCorePlatform",
                "False",
                "False",
            ],
        )

    def test_platform_attention_registration_respects_explicit_routes(self) -> None:
        try:
            from vllm_infinicore.ops import vllm_attention_backend
        except ModuleNotFoundError as exc:
            self.skipTest(f"attention backend dependencies unavailable: {exc}")

        vllm_attention_backend._ACTIVE_ROUTES.clear()
        with (
            mock.patch.object(vllm_attention_backend, "_register_backend_once"),
            mock.patch.dict(
                os.environ,
                {
                    "VLLM_INFINICORE_ENABLE_PATCHES": "1",
                    "VLLM_INFINICORE_ROUTES": "throughput",
                },
            ),
        ):
            vllm_attention_backend.install_platform_attention_backend()

        self.assertEqual(vllm_attention_backend._ACTIVE_ROUTES, set())

    def test_platform_only_attention_registration_keeps_no_metax_default(self) -> None:
        try:
            from vllm_infinicore.ops import vllm_attention_backend
        except ModuleNotFoundError as exc:
            self.skipTest(f"attention backend dependencies unavailable: {exc}")

        vllm_attention_backend._ACTIVE_ROUTES.clear()
        with (
            mock.patch.object(vllm_attention_backend, "_register_backend_once"),
            mock.patch.dict(os.environ, {}, clear=True),
        ):
            vllm_attention_backend.install_platform_attention_backend()

        self.assertEqual(
            vllm_attention_backend._ACTIVE_ROUTES,
            set(vllm_attention_backend.INFINICORE_ATTENTION_BACKEND_ROUTES),
        )
        vllm_attention_backend._ACTIVE_ROUTES.clear()


if __name__ == "__main__":
    unittest.main()
