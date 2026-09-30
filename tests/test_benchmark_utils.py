"""CPU regression tests for the benchmark correctness gates."""

import math
import unittest

import numpy as np
import torch

from tests.benchmarks.accuracy import error_stats
from tests.benchmarks.common import compare_outputs


class BenchmarkCorrectnessTests(unittest.TestCase):
    def test_tolerance_exact_and_nonfinite_checks(self):
        reference = torch.tensor([0.0, 2.0])
        actual = torch.tensor([0.015, 2.03])
        self.assertTrue(compare_outputs(actual, reference, dtype=torch.bfloat16)["passed"])
        self.assertFalse(compare_outputs(actual, reference, dtype=torch.float16)["passed"])
        self.assertEqual(compare_outputs(actual, reference, dtype=torch.bfloat16,
                                         exact=True)["failed_elements"], 2)
        bad = compare_outputs(torch.tensor([float("nan"), float("inf"), 1.0]),
                              torch.tensor([0.0, float("inf"), 1.0]), dtype=torch.float16)
        self.assertFalse(bad["passed"])
        self.assertEqual(bad["failed_elements"], 2)

    def test_missing_or_misshaped_outputs_cannot_pass(self):
        one = torch.ones(2)
        for actual, expected in (([one], [one, one]), ([], []),
                                 (one, one.reshape(1, 2)), (torch.empty(0), torch.empty(0))):
            with self.subTest(actual=actual, expected=expected), self.assertRaises(ValueError):
                compare_outputs(actual, expected, dtype=torch.float16)

    def test_full_reference_error_statistics(self):
        result = error_stats(np.array([0.0, 2.0, 3.0, np.nan]), np.array([0.0, 1.0, 3.0, 4.0]))
        self.assertEqual(result["failed_elements"], 2)
        self.assertEqual(result["nonfinite_elements"], 1)
        self.assertEqual(result["max_abs_error"], 1.0)
        self.assertEqual(result["mean_abs_error"], 1.0 / 3)
        self.assertAlmostEqual(result["relative_l2"], math.sqrt(0.1))
        self.assertEqual(result["worst_elements"][0]["flat_index"], 1)
        exact = error_stats(np.array([1.0, 1.001]), np.ones(2), exact=True)
        self.assertEqual(exact["failed_elements"], 1)
        self.assertTrue(error_stats(np.ones(3), np.ones(3), exact=True)["passed"])


if __name__ == "__main__":
    unittest.main()
