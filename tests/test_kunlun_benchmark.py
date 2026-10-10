"""Reject incomplete, mismatched, or silently bypassed throughput evidence."""

from __future__ import annotations

import copy
import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from vllm_infinicore.benchmarks.kunlun_static import (
    benchmark_matrix,
    compare_results,
    llm_kwargs,
    parse_args,
    timed_rows,
)

ROUTES = "Embedding,RoPE,MatMul,LMHead,StoreKVCache"


class StaticProtocolTests(unittest.TestCase):
    def arguments(self, *extra):
        return parse_args(
            [
                "--mode",
                "native",
                "--model",
                "/model",
                "--tp",
                "1",
                "--devices",
                "0",
                "--output",
                "/unused/native.json",
                *extra,
            ]
        )

    def test_defaults_keep_the_measured_matrix_and_graph_protocol(self):
        args = self.arguments()
        self.assertEqual(benchmark_matrix(args), ([1, 4, 16, 32, 64], [(128, 128), (2048, 512)]))
        self.assertEqual(args.repeats, 3)
        for tp in (1, 2, 4, 8):
            with self.subTest(tp=tp):
                args.tp = tp
                kwargs = llm_kwargs(args)
                self.assertEqual(kwargs["tensor_parallel_size"], tp)
                self.assertEqual(kwargs["gpu_memory_utilization"], 0.85 if tp <= 2 else 0.70)
                self.assertEqual(kwargs["max_model_len"], 2816)
                self.assertEqual(kwargs["max_num_seqs"], 64)
                self.assertEqual(kwargs["block_size"], 128)
                self.assertEqual(
                    kwargs["compilation_config"],
                    dict(
                        cudagraph_capture_sizes=[1, 2, 4, 8, 16, 32, 64], cudagraph_num_of_warmups=1
                    ),
                )
        args.enforce_eager = True
        args.memory = 0.3
        self.assertNotIn("compilation_config", llm_kwargs(args))
        self.assertEqual(llm_kwargs(args)["gpu_memory_utilization"], 0.3)

    def test_invalid_matrix_and_attention_are_rejected_before_loading_a_model(self):
        for option, value in (
            ("--batches", ""),
            ("--batches", "65"),
            ("--lengths", "128"),
            ("--lengths", "2048:1024"),
            ("--routes", "PagedAttentionDecode"),
            ("--repeats", "0"),
        ):
            with self.subTest(option=option, value=value), redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    self.arguments(option, value)
                self.assertEqual(error.exception.code, 2)

    def test_timings_exclude_warmup_and_require_complete_fixed_length_outputs(self):
        output = SimpleNamespace(token_ids=[1, 2, 3, 4], text="example")
        llm = Mock()
        llm.generate.return_value = [SimpleNamespace(outputs=[output])]
        case = dict(input_len=128, output_len=4, batch=1)
        with patch(
            "vllm_infinicore.benchmarks.kunlun_static.time.perf_counter",
            side_effect=[0, 2, 10, 14, 20, 25],
        ):
            rows = list(timed_rows(llm, [{}], object(), case, repeats=3))
        self.assertEqual([row["output_tps"] for row in rows], [2.0, 1.0, 0.8])
        self.assertEqual(llm.generate.call_count, 4)
        self.assertTrue(all(len(row["output_token_hashes"]) == 1 for row in rows))
        output.token_ids = [1]
        with self.assertRaisesRegex(RuntimeError, "fixed output length"):
            list(timed_rows(llm, [{}], object(), case, repeats=3))
        llm.generate.return_value = []
        with self.assertRaisesRegex(RuntimeError, "Missing requests"):
            list(timed_rows(llm, [{}], object(), case, repeats=3))


class BenchmarkEvidenceTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.native = dict(
            completed=True,
            errors=[],
            arguments=dict(
                mode="native",
                tp=2,
                devices="0,1",
                routes=ROUTES,
                batches="1",
                lengths="128:128",
                repeats=3,
                enforce_eager=False,
            ),
            llm_kwargs=dict(tensor_parallel_size=2),
            model_config_sha256="model",
            prompt_sha256="prompt",
            vendor_cache_sha256="vendor",
            versions={"vllm": "0.11.0"},
            final_workers=[dict(rank=rank, graph_captures=1, graph_replays=3) for rank in range(2)],
            rows=[
                dict(input_len=128, output_len=128, batch=1, repeat=i, output_tps=100.0)
                for i in range(3)
            ],
            summaries=[dict(input_len=128, output_len=128, batch=1, median_tps=100.0)],
        )
        self.plugin = copy.deepcopy(self.native)
        self.plugin["arguments"]["mode"] = "infinicore"
        for worker in self.plugin["final_workers"]:
            worker.update(
                platform="kunlun",
                registration=dict(installed_routes=ROUTES.split(",")),
                bridge_calls={name: 1 for name in ROUTES.split(",")},
                attention_calls={},
                fallback_calls={},
                attention_fallbacks={},
                native_attention_calls={},
            )
        for row in self.plugin["rows"]:
            row["output_tps"] = 92.0
        self.plugin["summaries"][0]["median_tps"] = 92.0

    def compare(self):
        paths = [self.root / f"{name}.json" for name in ("native", "infinicore")]
        for path, data in zip(paths, (self.native, self.plugin)):
            path.write_text(json.dumps(data))
        return compare_results(*paths)

    def test_valid_pair_and_failed_threshold_are_reported(self):
        self.assertTrue(self.compare()["all_cases_reached_90_percent"])
        for row in self.plugin["rows"]:
            row["output_tps"] = 89.0
        self.plugin["summaries"][0]["median_tps"] = 89.0
        self.assertFalse(self.compare()["all_cases_reached_90_percent"])

    def test_mismatched_devices_and_vendor_fix_are_rejected(self):
        self.plugin["arguments"]["devices"] = "2,3"
        with self.assertRaisesRegex(ValueError, "conditions differ: devices"):
            self.compare()
        self.plugin["arguments"]["devices"] = "0,1"
        self.plugin["vendor_cache_sha256"] = "different"
        with self.assertRaisesRegex(ValueError, "Vendor cache implementation differs"):
            self.compare()

    def test_each_rank_must_execute_requested_routes_and_graphs(self):
        worker = self.plugin["final_workers"][1]
        worker["bridge_calls"]["Embedding"] = 0
        with self.assertRaisesRegex(RuntimeError, "Missing actual plugin route calls"):
            self.compare()
        worker["bridge_calls"]["Embedding"] = 1
        worker["graph_replays"] = 0
        with self.assertRaisesRegex(RuntimeError, "did not capture/replay"):
            self.compare()

    def test_unrequested_attention_and_missing_samples_are_rejected(self):
        self.plugin["final_workers"][1]["attention_calls"]["PagedAttentionDecode"] = 1
        with self.assertRaisesRegex(RuntimeError, "Unrequested InfiniCore Attention"):
            self.compare()
        self.plugin["final_workers"][1]["attention_calls"].clear()
        self.plugin["rows"].pop()
        with self.assertRaisesRegex(ValueError, "Missing timing repetitions"):
            self.compare()

    def test_planned_native_attention_is_allowed_but_native_store_is_rejected(self):
        worker = self.plugin["final_workers"][1]
        worker["native_attention_calls"] = {"PagedAttentionPrefill": 4, "PagedAttentionDecode": 128}
        self.assertTrue(self.compare()["all_cases_reached_90_percent"])
        worker["native_attention_calls"]["StoreKVCache"] = 1
        with self.assertRaisesRegex(RuntimeError, "Unexpected plugin fallback"):
            self.compare()


if __name__ == "__main__":
    unittest.main()
