from __future__ import annotations

import os
import subprocess
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest import mock

from vllm_infinicore import platform, platform_support, plugin
from vllm_infinicore.patching import (
    PatchInstallResult,
    PatchRegistry,
    QWEN3_OPERATOR_ROUTES,
    get_default_registry,
)


class AscendPlatformSupportTests(unittest.TestCase):
    def setUp(self):
        platforms = ModuleType("vllm.platforms")
        platforms._current_platform = None
        self.platforms = platforms
        modules = mock.patch.dict(sys.modules, {"vllm.platforms": platforms})
        modules.start()
        self.addCleanup(modules.stop)
        environment = mock.patch.dict(
            os.environ,
            {
                "VLLM_PLUGINS": "ascend,vllm_infinicore",
                "VLLM_INFINICORE_ENABLE_PATCHES": "1",
                "VLLM_INFINICORE_ROUTES": "all",
                "VLLM_INFINICORE_STRICT_BACKEND": "1",
            },
            clear=True,
        )
        environment.start()
        self.addCleanup(environment.stop)

    def test_explicit_ascend_selection_declines_competing_platform(self):
        self.assertIsNone(platform.register_platform())

    def test_explicit_other_platform_preserves_existing_routes(self):
        os.environ["VLLM_PLUGINS"] = "infinicore,vllm_infinicore"
        with mock.patch.object(
            platform_support, "_package_available", return_value=True
        ):
            self.assertFalse(platform_support.ascend_platform_selected())
            self.assertEqual(platform.register_platform(), platform.PLATFORM_CLASS_PATH)
            self.assertEqual(get_default_registry()._native_fallback_reasons, {})

    def test_resolved_platform_takes_precedence_over_environment(self):
        self.platforms._current_platform = SimpleNamespace(device_type="cuda")
        self.assertFalse(platform_support.ascend_platform_selected())
        self.platforms._current_platform = SimpleNamespace(device_type="npu")
        os.environ["VLLM_PLUGINS"] = "infinicore,vllm_infinicore"
        self.assertTrue(platform_support.ascend_platform_selected())

    def test_automatic_discovery_defers_only_when_ascend_stack_is_present(self):
        os.environ.pop("VLLM_PLUGINS")
        with mock.patch.object(
            platform_support, "_package_available", return_value=True
        ):
            self.assertIsNone(platform.register_platform())
        with mock.patch.object(
            platform_support, "_package_available", return_value=False
        ):
            self.assertEqual(platform.register_platform(), platform.PLATFORM_CLASS_PATH)

    def test_all_unsupported_routes_preserve_native_without_importing_installers(self):
        registry = get_default_registry()
        installer = mock.Mock(
            side_effect=AssertionError("must not install an unsupported adapter")
        )
        registry._installers = {name: installer for name in registry.routes}
        result = registry.register_from_environment()
        self.assertEqual(result.native_fallback_routes, tuple(registry.routes))
        self.assertEqual(result.installed_routes, ())
        self.assertIsNone(result.failure_reason)
        self.assertFalse(result.patching_enabled)
        for state in result.route_states:
            self.assertIn("not supported", state.reason)
            self.assertIn("vLLM-Ascend", state.native_fallback)
            self.assertEqual(state.graph_policy, "native_platform")
        installer.assert_not_called()

    def test_capability_fallback_does_not_disable_supported_routes(self):
        os.environ["VLLM_INFINICORE_ROUTES"] = "RMSNorm,SiluAndMul"
        unsupported = mock.Mock(
            side_effect=AssertionError("unsupported installer called")
        )
        supported = mock.Mock(
            return_value=PatchInstallResult(True, "supported adapter")
        )
        registry = PatchRegistry(
            QWEN3_OPERATOR_ROUTES,
            installers={"RMSNorm": unsupported, "SiluAndMul": supported},
            native_fallback_reasons={"RMSNorm": "kernel unavailable"},
        )
        result = registry.register_from_environment()
        self.assertEqual(result.installed_routes, ("SiluAndMul",))
        self.assertEqual(result.native_fallback_routes, ("RMSNorm",))
        self.assertIsNone(result.failure_reason)
        supported.assert_called_once()
        unsupported.assert_not_called()

    def test_disabled_and_unknown_routes_keep_existing_semantics(self):
        os.environ["VLLM_INFINICORE_DISABLED_ROUTES"] = "RMSNorm"
        result = get_default_registry().register_from_environment()
        self.assertEqual(result.disabled_routes, ("RMSNorm",))
        self.assertEqual(len(result.native_fallback_routes), 8)
        os.environ["VLLM_INFINICORE_ROUTES"] = "Typo"
        result = get_default_registry().register_from_environment()
        self.assertIn("Typo", result.failure_reason)
        self.assertEqual(result.native_fallback_routes, ())

    def test_ascend_registration_skips_metax_compatibility_mutations(self):
        with (
            mock.patch.object(plugin, "_REGISTERED", False),
            mock.patch.object(plugin, "_REGISTRATION_RESULT", None),
            mock.patch.object(plugin, "register_vllm_environment"),
            mock.patch.object(plugin, "apply_vllm_020_compat_patches") as compat,
        ):
            first = plugin.register()
            self.assertIs(plugin.register(), first)
            self.assertEqual(len(first.native_fallback_routes), 9)
            compat.assert_not_called()
            self.assertEqual(plugin.unregister().uninstalled_routes, ())

    def test_explicit_ascend_dry_registration_does_not_import_frameworks(self):
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                """
import sys
from vllm_infinicore import register
from vllm_infinicore.platform import register_platform
assert register_platform() is None
assert len(register().native_fallback_routes) == 9
assert not any(name in sys.modules for name in ('torch', 'torch_npu', 'vllm', 'vllm_ascend', 'infinicore'))
""",
            ],
            env=os.environ.copy(),
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
