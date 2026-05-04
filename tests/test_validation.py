from __future__ import annotations

from pathlib import Path
import subprocess
import sys
import unittest

from vllm_infinicore.validation import (
    BenchmarkResult,
    GraphEvidence,
    check_graph_evidence,
    compute_text_health,
    detect_degenerate_repetition,
    validate_token_counts,
)

ROOT = Path(__file__).resolve().parents[1]


class ValidationTests(unittest.TestCase):
    def test_module_import_does_not_import_torch_or_vllm(self) -> None:
        code = """
import sys
from vllm_infinicore.validation import BenchmarkResult, GraphEvidence
print(BenchmarkResult.__name__)
print(GraphEvidence.__name__)
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
            ["BenchmarkResult", "GraphEvidence", "False", "False"],
        )

    def test_token_count_validation_reports_exact_mismatches(self) -> None:
        errors = validate_token_counts(
            expected_input_tokens=128,
            actual_input_tokens=127,
            expected_output_tokens=32,
            actual_output_tokens=31,
        )

        self.assertEqual(
            errors,
            [
                "input_tokens=127, expected=128",
                "generated_tokens=31, expected=32",
            ],
        )

    def test_text_health_counts_decode_signals(self) -> None:
        health = compute_text_health(" ok\x01\ufffd\n ", [1, 1, 2, 3])

        self.assertEqual(health.chars, 7)
        self.assertEqual(health.stripped_length, 4)
        self.assertEqual(health.replacement_chars, 1)
        self.assertEqual(health.control_chars, 1)
        self.assertEqual(health.newline_chars, 1)
        self.assertEqual(health.unique_token_count, 3)
        self.assertEqual(health.token_count, 4)
        self.assertEqual(
            health.validation_errors(),
            ["replacement_chars=1", "control_chars=1"],
        )

    def test_repetition_detection_flags_token_runs(self) -> None:
        repetition = detect_degenerate_repetition([9] * 8)

        self.assertTrue(repetition.is_degenerate)
        self.assertEqual(repetition.longest_token_run, 8)
        self.assertEqual(repetition.repeated_token_id, 9)
        self.assertIn(
            "token 9 repeated 8 consecutive times",
            repetition.reasons,
        )

    def test_repetition_detection_flags_consecutive_ngrams(self) -> None:
        repetition = detect_degenerate_repetition([1, 2, 3, 4] * 4)

        self.assertTrue(repetition.is_degenerate)
        self.assertEqual(repetition.repeated_ngram_size, 4)
        self.assertEqual(repetition.repeated_ngram_repetitions, 4)
        self.assertEqual(repetition.repeated_ngram, (1, 2, 3, 4))

    def test_graph_evidence_accepts_conservative_cudagraph_record(self) -> None:
        evidence = GraphEvidence.from_compilation_config(
            {
                "cudagraph_mode": "PIECEWISE",
                "backend": "eager",
                "cudagraph_capture_sizes": [1, 2, 4, 8],
            },
            enforce_eager=False,
            log_snippets=("captured cudagraph sizes [1, 2, 4, 8]",),
        )

        self.assertEqual(evidence.cudagraph_mode, "PIECEWISE")
        self.assertEqual(evidence.backend, "eager")
        self.assertEqual(evidence.capture_sizes, (1, 2, 4, 8))
        self.assertEqual(evidence.validation_errors(require_cudagraph=True), [])

    def test_graph_evidence_reports_required_cudagraph_failures(self) -> None:
        evidence = GraphEvidence(
            cudagraph_mode="NONE",
            enforce_eager=True,
            backend="inductor",
            capture_sizes=(),
        )

        errors = check_graph_evidence(evidence, require_cudagraph=True)
        self.assertIn("cudagraph_mode is missing or disabled", errors)
        self.assertIn("enforce_eager must be False for cudagraph validation", errors)
        self.assertIn(
            "backend='inductor' is not the conservative eager backend",
            errors,
        )
        self.assertIn("cudagraph_capture_sizes are required", errors)
        self.assertIn("graph evidence string or log snippet is required", errors)

    def test_healthy_benchmark_result_has_no_validation_errors(self) -> None:
        result = BenchmarkResult(
            name="baseline",
            expected_input_tokens=128,
            actual_input_tokens=128,
            expected_output_tokens=4,
            actual_output_tokens=4,
            elapsed_seconds=2.0,
            output_token_ids=[10, 11, 12, 13],
            decoded_text="valid decoded output",
            graph_evidence=GraphEvidence(
                cudagraph_mode="PIECEWISE",
                enforce_eager=False,
                backend="eager",
                capture_sizes=[1, 2, 4, 8],
                evidence_strings=["capture completed"],
            ),
            require_cudagraph=True,
        )

        self.assertEqual(result.output_tps, 2.0)
        self.assertEqual(result.text_health.unique_token_count, 4)
        self.assertEqual(result.validation_errors, [])
        self.assertEqual(result.as_dict()["validation_errors"], [])

    def test_benchmark_result_flags_unhealthy_output(self) -> None:
        result = BenchmarkResult(
            name="candidate",
            expected_input_tokens=128,
            actual_input_tokens=128,
            expected_output_tokens=8,
            actual_output_tokens=8,
            elapsed_seconds=1.0,
            output_token_ids=[5] * 8,
            decoded_text="repeat repeat",
        )

        self.assertEqual(result.output_tps, 8.0)
        self.assertIn(
            "degenerate repetition: token 5 repeated 8 consecutive times",
            result.validation_errors,
        )

    def test_benchmark_result_requires_recorded_output_token_ids(self) -> None:
        result = BenchmarkResult(
            name="candidate",
            expected_input_tokens=128,
            actual_input_tokens=128,
            expected_output_tokens=3,
            actual_output_tokens=3,
            elapsed_seconds=1.0,
            output_token_ids=[1, 2],
            decoded_text="ok",
        )

        self.assertEqual(
            result.validation_errors,
            ["output_token_ids length=2, actual_output_tokens=3"],
        )


if __name__ == "__main__":
    unittest.main()
