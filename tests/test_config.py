from __future__ import annotations

import copy
from pathlib import Path
import tomllib
import unittest

from vllm_infinicore.config import (
    ConfigValidationError,
    load_config,
    parse_config,
    validate_config_against_registry,
)

ROOT = Path(__file__).resolve().parents[1]


class ConfigTests(unittest.TestCase):
    def test_default_config_matches_route_registry(self) -> None:
        config = load_config()

        self.assertEqual(config.plugin.patch_enable_env, "VLLM_INFINICORE_ENABLE_PATCHES")
        self.assertEqual(config.plugin.route_select_env, "VLLM_INFINICORE_ROUTES")
        self.assertEqual(
            config.plugin.route_disable_env,
            "VLLM_INFINICORE_DISABLED_ROUTES",
        )
        self.assertEqual(
            config.plugin.force_native_fallback_env,
            "VLLM_INFINICORE_FORCE_NATIVE_FALLBACK",
        )
        self.assertEqual(config.cuda_graph.enforce_eager, False)
        self.assertEqual(
            config.cuda_graph.vllm_compilation_config["backend"],
            "eager",
        )
        self.assertEqual(len(config.routes), 9)
        self.assertEqual(tuple(route.default_enabled for route in config.routes), (False,) * 9)
        self.assertTrue(all(route.native_fallback for route in config.routes))
        self.assertTrue(all(route.validation for route in config.routes))

        validate_config_against_registry(config)

    def test_duplicate_route_is_rejected(self) -> None:
        config = load_config(validate_registry=False)
        raw_config = {
            "plugin": config.plugin.__dict__,
            "target": config.target.__dict__,
            "cuda_graph": {
                **config.cuda_graph.__dict__,
                "vllm_compilation_config": dict(
                    config.cuda_graph.vllm_compilation_config
                ),
            },
            "routes": [route.__dict__ for route in config.routes],
            "benchmark_rules": {
                "prompt_ids": config.benchmark_rules.prompt_ids,
                "metric": config.benchmark_rules.metric,
                "sampling": config.benchmark_rules.sampling.__dict__,
                "vllm_tokens": config.benchmark_rules.vllm_tokens.__dict__,
                "validation": config.benchmark_rules.validation.__dict__,
            },
        }
        raw_config["routes"][1] = copy.deepcopy(raw_config["routes"][0])

        with self.assertRaisesRegex(ConfigValidationError, "duplicate route"):
            parse_config(raw_config)

    def test_pyproject_declares_vllm_entry_points(self) -> None:
        data = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))

        general_plugins = data["project"]["entry-points"]["vllm.general_plugins"]
        self.assertEqual(general_plugins["vllm_infinicore"], "vllm_infinicore:register")

        platform_plugins = data["project"]["entry-points"]["vllm.platform_plugins"]
        self.assertEqual(
            platform_plugins["infinicore"],
            "vllm_infinicore.platform:register_platform",
        )


if __name__ == "__main__":
    unittest.main()
