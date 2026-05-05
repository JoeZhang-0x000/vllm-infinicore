from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import unittest
from unittest import mock

from vllm_infinicore import plugin
from vllm_infinicore.patching import (
    FORCE_NATIVE_FALLBACK_ENV,
    PatchInstallResult,
    PatchRegistry,
    PatchUninstallResult,
    QWEN3_OPERATOR_ROUTES,
    ROUTE_DISABLE_ENV,
    ROUTE_STATE_DISABLED,
    ROUTE_STATE_NATIVE_FALLBACK,
)

ROOT = Path(__file__).resolve().parents[1]


class RegistrationTests(unittest.TestCase):
    def setUp(self) -> None:
        plugin._REGISTERED = False
        plugin._REGISTRATION_RESULT = None

    def test_register_is_default_off_and_idempotent(self) -> None:
        with mock.patch.dict(
            os.environ,
            {
                "VLLM_INFINICORE_ENABLE_PATCHES": "",
                "VLLM_INFINICORE_ROUTES": "",
            },
        ):
            first = plugin.register()
            second = plugin.register()

        self.assertIs(first, second)
        self.assertEqual(first.route_count, 9)
        self.assertFalse(first.patching_enabled)
        self.assertEqual(first.enabled_routes, ())
        self.assertEqual(first.requested_routes, ())
        self.assertEqual(first.installed_routes, ())
        self.assertEqual(first.skipped_routes, ())
        self.assertIsNone(first.failure_reason)
        self.assertEqual(len(first.route_states), 9)
        self.assertEqual(
            tuple(state.status for state in first.route_states),
            (ROUTE_STATE_DISABLED,) * 9,
        )

    def test_route_selector_installs_explicit_rmsnorm(self) -> None:
        calls: list[str] = []

        def install_rmsnorm() -> PatchInstallResult:
            calls.append("RMSNorm")
            return PatchInstallResult(installed=True, reason="installed")

        registry = PatchRegistry(
            QWEN3_OPERATOR_ROUTES,
            installers={"RMSNorm": install_rmsnorm},
        )
        with mock.patch.dict(
            os.environ,
            {
                "VLLM_INFINICORE_ENABLE_PATCHES": "1",
                "VLLM_INFINICORE_ROUTES": "RMSNorm",
            },
        ):
            result = registry.register_from_environment()

        self.assertTrue(result.patching_enabled)
        self.assertEqual(result.requested_routes, ("RMSNorm",))
        self.assertEqual(result.installed_routes, ("RMSNorm",))
        self.assertEqual(result.skipped_routes, ())
        self.assertIsNone(result.failure_reason)
        self.assertEqual(calls, ["RMSNorm"])

    def test_route_selector_expands_throughput_profile(self) -> None:
        calls: list[str] = []

        def make_installer(route_name: str):
            def install() -> PatchInstallResult:
                calls.append(route_name)
                return PatchInstallResult(installed=True, reason="installed")

            return install

        registry = PatchRegistry(
            QWEN3_OPERATOR_ROUTES,
            installers={
                "RMSNorm": make_installer("RMSNorm"),
                "SiluAndMul": make_installer("SiluAndMul"),
                "Embedding": make_installer("Embedding"),
            },
        )
        with mock.patch.dict(
            os.environ,
            {
                "VLLM_INFINICORE_ENABLE_PATCHES": "1",
                "VLLM_INFINICORE_ROUTES": "throughput",
                ROUTE_DISABLE_ENV: "",
                FORCE_NATIVE_FALLBACK_ENV: "0",
            },
        ):
            result = registry.register_from_environment()

        self.assertTrue(result.patching_enabled)
        self.assertEqual(result.requested_routes, ("RMSNorm", "SiluAndMul", "Embedding"))
        self.assertEqual(result.installed_routes, ("RMSNorm", "SiluAndMul", "Embedding"))
        self.assertEqual(result.skipped_routes, ())
        self.assertIsNone(result.failure_reason)
        self.assertEqual(calls, ["RMSNorm", "SiluAndMul", "Embedding"])

    def test_attention_routes_register_backend_override(self) -> None:
        try:
            from vllm.v1.attention.backends.registry import AttentionBackendEnum
        except Exception as exc:  # pragma: no cover - depends on vLLM install
            self.skipTest(f"vLLM attention registry unavailable: {exc}")

        AttentionBackendEnum.FLASH_ATTN.clear_override()
        registry = PatchRegistry(QWEN3_OPERATOR_ROUTES)
        with mock.patch.dict(
            os.environ,
            {
                "VLLM_INFINICORE_ENABLE_PATCHES": "1",
                "VLLM_INFINICORE_ROUTES": "StoreKVCache,PagedAttentionPrefill,PagedAttentionDecode",
                ROUTE_DISABLE_ENV: "",
                FORCE_NATIVE_FALLBACK_ENV: "0",
            },
        ):
            result = registry.register_from_environment()

        self.assertTrue(result.patching_enabled)
        self.assertEqual(
            result.installed_routes,
            ("StoreKVCache", "PagedAttentionPrefill", "PagedAttentionDecode"),
        )
        self.assertTrue(AttentionBackendEnum.FLASH_ATTN.is_overridden())
        self.assertEqual(
            AttentionBackendEnum.FLASH_ATTN.get_path(),
            "vllm_infinicore.ops.vllm_attention_backend.InfiniCoreFlashAttentionBackend",
        )
        AttentionBackendEnum.FLASH_ATTN.clear_override()

    def test_all_routes_can_force_native_fallback(self) -> None:
        registry = PatchRegistry(
            QWEN3_OPERATOR_ROUTES,
            installers={
                "RMSNorm": lambda: PatchInstallResult(
                    installed=True, reason="should not run"
                )
            },
        )
        with mock.patch.dict(
            os.environ,
            {
                "VLLM_INFINICORE_ENABLE_PATCHES": "1",
                "VLLM_INFINICORE_ROUTES": "all",
                FORCE_NATIVE_FALLBACK_ENV: "1",
                ROUTE_DISABLE_ENV: "",
            },
        ):
            result = registry.register_from_environment()

        expected_routes = tuple(route.name for route in QWEN3_OPERATOR_ROUTES)
        self.assertFalse(result.patching_enabled)
        self.assertEqual(result.requested_routes, expected_routes)
        self.assertEqual(result.installed_routes, ())
        self.assertEqual(result.skipped_routes, expected_routes)
        self.assertEqual(result.native_fallback_routes, expected_routes)
        self.assertTrue(
            all(state.status == ROUTE_STATE_NATIVE_FALLBACK for state in result.route_states)
        )

    def test_route_disable_env_removes_selected_route(self) -> None:
        registry = PatchRegistry(
            QWEN3_OPERATOR_ROUTES,
            installers={
                "RMSNorm": lambda: PatchInstallResult(
                    installed=True, reason="should not run"
                )
            },
        )
        with mock.patch.dict(
            os.environ,
            {
                "VLLM_INFINICORE_ENABLE_PATCHES": "1",
                "VLLM_INFINICORE_ROUTES": "RMSNorm",
                ROUTE_DISABLE_ENV: "RMSNorm",
                FORCE_NATIVE_FALLBACK_ENV: "0",
            },
        ):
            result = registry.register_from_environment()

        self.assertFalse(result.patching_enabled)
        self.assertEqual(result.requested_routes, ("RMSNorm",))
        self.assertEqual(result.installed_routes, ())
        self.assertEqual(result.skipped_routes, ("RMSNorm",))
        rms_state = next(state for state in result.route_states if state.name == "RMSNorm")
        self.assertEqual(rms_state.status, ROUTE_STATE_DISABLED)
        self.assertTrue(rms_state.disabled_by_env)

    def test_route_selector_rejects_unknown_route(self) -> None:
        registry = PatchRegistry(
            QWEN3_OPERATOR_ROUTES,
            installers={
                "RMSNorm": lambda: PatchInstallResult(
                    installed=True, reason="should not run"
                )
            },
        )
        with mock.patch.dict(
            os.environ,
            {
                "VLLM_INFINICORE_ENABLE_PATCHES": "1",
                "VLLM_INFINICORE_ROUTES": "RMSNorm,UnknownRoute",
            },
        ):
            result = registry.register_from_environment()

        self.assertFalse(result.patching_enabled)
        self.assertEqual(result.requested_routes, ("RMSNorm", "UnknownRoute"))
        self.assertEqual(result.installed_routes, ())
        self.assertEqual(result.skipped_routes, ("RMSNorm", "UnknownRoute"))
        self.assertIn("UnknownRoute", result.failure_reason or "")

    def test_plugin_register_keeps_first_result(self) -> None:
        first_result = PatchRegistry(
            QWEN3_OPERATOR_ROUTES,
            installers={
                "RMSNorm": lambda: PatchInstallResult(installed=True, reason="installed")
            },
        )

        with (
            mock.patch.object(plugin, "get_default_registry", return_value=first_result),
            mock.patch.dict(
                os.environ,
                {
                    "VLLM_INFINICORE_ENABLE_PATCHES": "1",
                    "VLLM_INFINICORE_ROUTES": "RMSNorm",
                },
            ),
        ):
            first = plugin.register()
            second = plugin.register()

        self.assertIs(first, second)
        self.assertEqual(first.installed_routes, ("RMSNorm",))

    def test_dry_register_does_not_import_torch(self) -> None:
        code = """
import os
import sys
os.environ.pop("VLLM_INFINICORE_ENABLE_PATCHES", None)
os.environ.pop("VLLM_INFINICORE_ROUTES", None)
import vllm_infinicore
result = vllm_infinicore.register()
print(result.route_count)
print(result.patching_enabled)
print(result.installed_routes)
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
            ["9", "False", "()", "False", "False"],
        )

    def test_uninstall_routes_is_idempotent(self) -> None:
        calls: list[str] = []

        def uninstall_rmsnorm() -> PatchUninstallResult:
            calls.append("RMSNorm")
            return PatchUninstallResult(uninstalled=False, reason="not installed")

        registry = PatchRegistry(
            QWEN3_OPERATOR_ROUTES,
            installers={},
            uninstallers={"RMSNorm": uninstall_rmsnorm},
        )

        first = registry.uninstall_routes(("RMSNorm",))
        second = registry.uninstall_routes(("RMSNorm",))

        self.assertEqual(first.uninstalled_routes, ())
        self.assertEqual(first.skipped_routes, ("RMSNorm",))
        self.assertIsNone(first.failure_reason)
        self.assertEqual(second.skipped_routes, ("RMSNorm",))
        self.assertEqual(calls, ["RMSNorm", "RMSNorm"])


if __name__ == "__main__":
    unittest.main()
