import importlib.util
from pathlib import Path
import unittest


spec = importlib.util.spec_from_file_location(
    "ascend_smoke", Path(__file__).parent / "remote" / "run_ascend_smoke.py"
)
smoke = importlib.util.module_from_spec(spec)
spec.loader.exec_module(smoke)


class AscendSmokeValidationTests(unittest.TestCase):
    def state(self, *, installed=False, count=0):
        return {
            "registration": {
                "requested_routes": ["RMSNorm"],
                "installed_routes": ["RMSNorm"] if installed else [],
                "route_states": [
                    {
                        "name": "RMSNorm",
                        "status": "installed" if installed else "native_fallback",
                        "reason": "adapter unsupported",
                        "native_fallback": "vLLM-Ascend native RMSNorm",
                    }
                ],
                "failure_reason": None,
            },
            "backend_call_counts": {"rms_norm": count},
        }

    def test_native_fallback_requires_explicit_acceptance(self):
        state = self.state()
        self.assertEqual(
            smoke.validate_worker_routes([state], allow_native_fallback=True), []
        )
        self.assertIn(
            "requested_routes_not_covered",
            smoke.validate_worker_routes([state], allow_native_fallback=False),
        )

    def test_fallback_permission_cannot_mask_falsely_installed_routes(self):
        self.assertIn(
            "installed_route_has_no_backend_calls:RMSNorm",
            smoke.validate_worker_routes(
                [self.state(installed=True)], allow_native_fallback=True
            ),
        )
        self.assertEqual(
            smoke.validate_worker_routes(
                [self.state(installed=True, count=2)], allow_native_fallback=True
            ),
            [],
        )

    def test_missing_registration_and_unexplained_fallback_fail(self):
        self.assertTrue(smoke.validate_worker_routes([], allow_native_fallback=True))
        self.assertTrue(smoke.validate_worker_routes([{}], allow_native_fallback=True))
        state = self.state()
        state["registration"]["route_states"][0]["reason"] = ""
        self.assertIn(
            "requested_routes_not_covered",
            smoke.validate_worker_routes([state], allow_native_fallback=True),
        )
