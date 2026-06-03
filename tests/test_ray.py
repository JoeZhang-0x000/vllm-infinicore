from __future__ import annotations

import os
import sys
import types
import unittest
from unittest import mock

from vllm_infinicore.ray import (
    RAY_BACKEND_ENV,
    RAY_NOSET_CUDA_VISIBLE_DEVICES_ENV,
    configure_ray_environment,
)


class RayEnvironmentTests(unittest.TestCase):
    def test_non_ray_backend_is_noop(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=True):
            status = configure_ray_environment(
                distributed_executor_backend="mp",
            )

        self.assertFalse(status.enabled)
        self.assertEqual(status.registered_env_vars, ())
        self.assertNotIn(RAY_NOSET_CUDA_VISIBLE_DEVICES_ENV, os.environ)

    def test_ray_backend_sets_noset_and_registers_vllm_envs(self) -> None:
        fake_vllm = types.ModuleType("vllm")
        fake_envs = types.ModuleType("vllm.envs")
        fake_envs.environment_variables = {}
        fake_vllm.envs = fake_envs

        with (
            mock.patch.dict(
                sys.modules,
                {
                    "vllm": fake_vllm,
                    "vllm.envs": fake_envs,
                },
            ),
            mock.patch.dict(
                os.environ,
                {
                    "VLLM_INFINICORE_ENABLE_PATCHES": "1",
                    "VLLM_INFINICORE_ROUTES": "all",
                    "VLLM_INFINICORE_DISABLE_RAY_STORE_KV_CACHE": "1",
                },
                clear=True,
            ),
        ):
            status = configure_ray_environment(
                distributed_executor_backend="ray",
                extra_env_vars=("EXTRA_TEST_ENV",),
            )
            noset_env = os.environ[RAY_NOSET_CUDA_VISIBLE_DEVICES_ENV]
            ray_backend_env = os.environ[RAY_BACKEND_ENV]
            registered_routes = fake_envs.environment_variables[
                "VLLM_INFINICORE_ROUTES"
            ]()

        self.assertTrue(status.enabled)
        self.assertEqual(status.noset_cuda_visible_devices, "1")
        self.assertEqual(noset_env, "1")
        self.assertEqual(ray_backend_env, "1")
        self.assertIn("VLLM_INFINICORE_ENABLE_PATCHES", status.registered_env_vars)
        self.assertIn("VLLM_INFINICORE_ROUTES", status.registered_env_vars)
        self.assertIn(
            "VLLM_INFINICORE_DISABLE_RAY_STORE_KV_CACHE",
            status.registered_env_vars,
        )
        self.assertIn(RAY_BACKEND_ENV, status.registered_env_vars)
        self.assertIn("EXTRA_TEST_ENV", status.registered_env_vars)
        self.assertEqual(registered_routes, "all")


if __name__ == "__main__":
    unittest.main()
