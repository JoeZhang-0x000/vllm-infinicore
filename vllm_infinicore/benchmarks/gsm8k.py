"""GSM8K numeric exact match with ModelScope data and fresh vLLM processes.

python -m vllm_infinicore.benchmarks.gsm8k --model /path/to/Qwen3-8B \
    --mode both --output-dir results/gsm8k
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import importlib.metadata
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from collections.abc import Mapping
from pathlib import Path

from .common import (
    ROUTES,
    mode_environment,
    sha256,
    verify_workers,
    worker_state,
    write_json,
)
from .gsm8k_grading import METRIC, extract_answer

DATASET_ID = "AI-ModelScope/gsm8k"
PROMPT = (
    "Solve the following math problem step by step. "
    "Write your final answer on the last line in the form #### <number>.\n\n{question}"
)


def make_prompts(tokenizer, records: list[dict]) -> list[list[int]]:
    prompts = []
    for row in records:
        encoded = tokenizer.apply_chat_template(
            [{"role": "user", "content": PROMPT.format(question=row["question"])}],
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
            return_dict=False,
        )
        prompts.append(list(encoded["input_ids"] if isinstance(encoded, Mapping) else encoded))
    return prompts


def read_test_records(snapshot: Path) -> tuple[list[dict], list[Path]]:
    candidates = [
        p
        for p in snapshot.rglob("*")
        if p.is_file()
        and "test" in p.stem.lower()
        and p.suffix in (".jsonl", ".parquet", ".json")
        and "socratic" not in p.parts
    ]
    main = [p for p in candidates if "main" in p.relative_to(snapshot).parts]
    candidates = main or candidates
    for suffix in (".jsonl", ".parquet", ".json"):
        paths = sorted(p for p in candidates if p.suffix == suffix)
        if paths:
            break
    else:
        raise RuntimeError(f"No main/test data files in ModelScope snapshot: {snapshot}")
    records = []
    for path in paths:
        if path.suffix == ".parquet":
            import pyarrow.parquet as parquet

            rows = parquet.read_table(path, columns=["question", "answer"]).to_pylist()
        elif path.suffix == ".jsonl":
            rows = [
                json.loads(line)
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        else:
            rows = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(rows, dict):
                rows = rows["test"]
        for row in rows:
            question, answer = row["question"], row["answer"]
            target = extract_answer(answer)
            if not isinstance(question, str) or not question.strip() or target is None:
                raise ValueError(f"Invalid GSM8K question/answer in {path}, row {len(records)}")
            records.append(
                {"index": len(records), "question": question, "answer": answer, "target": target}
            )
    if len(records) != 1319:
        raise ValueError(
            f"Expected the complete GSM8K main/test split (1319 rows), got {len(records)}"
        )
    return records, paths


def prepare_dataset(args) -> Path:
    if args.dataset_file:
        path = args.dataset_file.resolve()
        data = json.loads(path.read_text(encoding="utf-8"))
        if len(data["records"]) != 1319:
            raise ValueError("Prepared dataset must contain all 1319 test questions")
        for row in data["records"]:
            if extract_answer(row["answer"]) != row["target"]:
                raise ValueError("Prepared dataset contains an invalid gold answer")
        return path
    from modelscope import dataset_snapshot_download

    snapshot = Path(
        dataset_snapshot_download(
            args.dataset_id,
            revision=args.revision,
            cache_dir=str(args.dataset_cache),
            allow_patterns=["*test*.jsonl", "*test*.parquet", "*test*.json"],
        )
    )
    records, paths = read_test_records(snapshot)
    path = args.output_dir / "dataset.json"
    write_json(
        path,
        {
            "source": {
                "hub": "modelscope",
                "dataset_id": args.dataset_id,
                "revision": args.revision,
                "subset": "main",
                "split": "test",
                "count": len(records),
                "snapshot": str(snapshot),
                "files": {str(p.relative_to(snapshot)): sha256(p) for p in paths},
            },
            "records": records,
        },
    )
    print(
        "DATASET",
        json.dumps({"path": str(path), "count": len(records), "sha256": sha256(path)}),
        flush=True,
    )
    return path


def run_mode(args, dataset_path: Path) -> int:
    os.environ.update(mode_environment(args.mode, os.environ, args.platform, args.routes))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    score_path = args.output_dir / f"{args.mode}.json"
    prediction_path = args.output_dir / f"{args.mode}.predictions.jsonl"
    if score_path.exists() or prediction_path.exists():
        raise FileExistsError(f"Use a fresh output directory; {args.mode} results already exist")
    os.environ["VLLM_CACHE_ROOT"] = tempfile.mkdtemp(
        prefix=f"compile-{args.mode}-", dir=args.output_dir
    )
    data = json.loads(dataset_path.read_text(encoding="utf-8"))
    records = data["records"][: args.limit] if args.limit else data["records"]
    result = {
        "benchmark": "gsm8k",
        "mode": args.mode,
        "platform": args.platform,
        "completed": False,
        "errors": [],
        "started_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "dataset": {
            **data["source"],
            "prepared_sha256": sha256(dataset_path),
            "evaluated_count": len(records),
        },
        "protocol": {
            "few_shot": 0,
            "enable_thinking": False,
            "temperature": 0.0,
            "max_tokens": args.max_tokens,
            "prompt": PROMPT,
            "metric": METRIC,
            "scorer_sha256": sha256(Path(__file__).with_name("gsm8k_grading.py")),
        },
        "environment": {
            key: os.getenv(key)
            for key in (
                "CUDA_VISIBLE_DEVICES",
                "XPU_VISIBLE_DEVICES",
                "ASCEND_RT_VISIBLE_DEVICES",
                "INFINI_ROOT",
                "INFINI_OPS_ROOT",
                "INFINI_RT_ROOT",
                "VLLM_INFINICORE_ASCEND_LIBRARY",
                "VLLM_INFINICORE_OPERATOR_BACKEND",
                "VLLM_PLUGINS",
                "VLLM_INFINICORE_ENABLE_PATCHES",
                "VLLM_INFINICORE_ROUTES",
                "VLLM_CACHE_ROOT",
            )
        },
        "sources": {
            str(p.relative_to(Path(__file__).parents[1])): sha256(p)
            for p in sorted(Path(__file__).parents[1].rglob("*"))
            if p.is_file() and p.suffix in (".py", ".cpp", ".h", ".json")
        },
    }
    llm = None
    try:
        import torch
        from transformers import AutoTokenizer
        from vllm import LLM, SamplingParams

        result["versions"] = {}
        for name in (
            "vllm",
            "vllm-metax",
            "vllm-ascend",
            "vllm-kunlun",
            "torch",
            "torch-npu",
            "transformers",
            "modelscope",
            "modelscope-hub",
            "pyarrow",
        ):
            try:
                result["versions"][name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                pass
        device_api = torch.npu if args.platform == "ascend" else torch.cuda
        result["device_name"] = device_api.get_device_name(0)
        if args.platform == "ascend":
            from ..operators.platforms.ascend import backend as ascend_backend

            if os.environ.get(ascend_backend.LIBRARY_ENV):
                library = Path(os.environ[ascend_backend.LIBRARY_ENV])
                result["ascend_library"] = {"path": str(library), "sha256": sha256(library)}
            result["ascend_infini_core"] = json.loads(
                (Path(__file__).parents[1] / "infinicore.lock.json").read_text()
            )["legacy_ascend"]
            if args.mode == "infinicore":
                ascend_backend.library()
        model_config = Path(args.model) / "config.json"
        result["model_config_sha256"] = sha256(model_config) if model_config.is_file() else None
        torch.set_num_threads(4)
        tokenizer = AutoTokenizer.from_pretrained(
            args.model, local_files_only=True, trust_remote_code=True
        )
        prompts = make_prompts(tokenizer, records)
        if max(map(len, prompts)) + args.max_tokens > args.max_model_len:
            raise ValueError(
                "Prompt plus max_tokens exceeds max_model_len; increase --max-model-len"
            )
        result["prompt_token_ids_sha256"] = hashlib.sha256(json.dumps(prompts).encode()).hexdigest()
        kwargs = dict(
            model=args.model,
            dtype="bfloat16",
            tensor_parallel_size=1,
            seed=args.seed,
            enforce_eager=args.enforce_eager,
            max_model_len=args.max_model_len,
            max_num_seqs=args.batch_size,
            max_num_batched_tokens=args.max_num_batched_tokens,
            gpu_memory_utilization=args.memory,
            enable_prefix_caching=False,
            trust_remote_code=True,
        )
        if args.platform in {"ascend", "kunlun"}:
            kwargs["block_size"] = 128
        if not args.enforce_eager:
            sizes = sorted(
                {1, args.batch_size}
                | {2**i for i in range(args.batch_size.bit_length()) if 2**i <= args.batch_size}
            )
            kwargs["compilation_config"] = dict(
                cudagraph_capture_sizes=sizes, cudagraph_num_of_warmups=1
            )
            if args.platform == "ascend":
                from vllm.config import CompilationMode, CUDAGraphMode

                kwargs["compilation_config"].update(
                    mode=CompilationMode.VLLM_COMPILE, cudagraph_mode=CUDAGraphMode.FULL_DECODE_ONLY
                )
        result["llm_kwargs"] = kwargs
        write_json(score_path, result)
        llm = LLM(**kwargs)
        result["startup_workers"] = llm.collective_rpc(worker_state)
        verify_workers(
            args.mode,
            result["startup_workers"],
            args.enforce_eager,
            finished=False,
            routes=args.routes,
        )
        sampling = SamplingParams(
            temperature=0.0, top_p=1.0, top_k=1, max_tokens=args.max_tokens, seed=args.seed
        )
        correct = invalid = truncated = completed = output_tokens = 0
        started = time.perf_counter()
        with prediction_path.open("w", encoding="utf-8") as output_file:
            for offset in range(0, len(records), args.batch_size):
                batch = records[offset : offset + args.batch_size]
                outputs = llm.generate(
                    [{"prompt_token_ids": p} for p in prompts[offset : offset + args.batch_size]],
                    sampling,
                    use_tqdm=False,
                )
                if len(outputs) != len(batch):
                    raise RuntimeError("Missing vLLM outputs in evaluation batch")
                for row, generated in zip(batch, outputs):
                    output = generated.outputs[0]
                    predicted = extract_answer(output.text)
                    passed = predicted is not None and predicted == row["target"]
                    correct += passed
                    invalid += predicted is None
                    truncated += output.finish_reason == "length"
                    output_tokens += len(output.token_ids)
                    completed += 1
                    output_file.write(
                        json.dumps(
                            {
                                "index": row["index"],
                                "question": row["question"],
                                "target": row["target"],
                                "predicted": predicted,
                                "correct": passed,
                                "text": output.text,
                                "token_ids": list(output.token_ids),
                                "finish_reason": output.finish_reason,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                output_file.flush()
                elapsed = time.perf_counter() - started
                result["score"] = {
                    "correct": correct,
                    "total": completed,
                    "expected_total": len(records),
                    "accuracy": correct / completed,
                    "accuracy_percent": 100 * correct / completed,
                    "invalid_answers": invalid,
                    "length_limited": truncated,
                    "generation_seconds": elapsed,
                    "output_tokens": output_tokens,
                }
                write_json(score_path, result)
                print(
                    "PROGRESS",
                    json.dumps(
                        {
                            "mode": args.mode,
                            "done": completed,
                            "total": len(records),
                            "correct": correct,
                            "invalid": invalid,
                            "length_limited": truncated,
                            "elapsed_s": round(elapsed, 1),
                        }
                    ),
                    flush=True,
                )
        result["final_workers"] = llm.collective_rpc(worker_state)
        verify_workers(
            args.mode,
            result["final_workers"],
            args.enforce_eager,
            finished=True,
            routes=args.routes,
        )
        result["predictions_sha256"] = sha256(prediction_path)
        result["completed"] = completed == len(records)
    except Exception:
        result["errors"].append(traceback.format_exc())
        print(result["errors"][-1], file=sys.stderr, flush=True)
    finally:
        result["finished_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        write_json(score_path, result)
        print("RESULT", str(score_path), json.dumps(result.get("score")), flush=True)
        if llm is not None:
            del llm
    return 0 if result["completed"] and not result["errors"] else 1


def compare_scores(output_dir: Path) -> dict:
    native, infini = [
        json.loads((output_dir / f"{mode}.json").read_text()) for mode in ("native", "infinicore")
    ]
    for data in (native, infini):
        if not data["completed"] or data["errors"]:
            raise RuntimeError("Cannot compare incomplete/failed evaluations")
    for key in (
        "dataset",
        "protocol",
        "prompt_token_ids_sha256",
        "llm_kwargs",
        "model_config_sha256",
    ):
        if native[key] != infini[key]:
            raise RuntimeError(f"Evaluation conditions differ: {key}")
    if native.get("platform", "metax") != infini.get("platform", "metax"):
        raise RuntimeError("Evaluations ran on different platforms")
    device_env = (
        "ASCEND_RT_VISIBLE_DEVICES"
        if native.get("platform") == "ascend"
        else "CUDA_VISIBLE_DEVICES"
    )
    if native["environment"][device_env] != infini["environment"][device_env]:
        raise RuntimeError("Evaluations ran on different devices")
    if native.get("platform") == "kunlun" and native["environment"].get(
        "XPU_VISIBLE_DEVICES"
    ) != infini["environment"].get("XPU_VISIBLE_DEVICES"):
        raise RuntimeError("Evaluations ran with different XPU device mappings")
    off, on = native["score"], infini["score"]
    return {
        "benchmark": "gsm8k",
        "platform": native.get("platform", "metax"),
        "dataset": native["dataset"],
        "protocol": native["protocol"],
        "model": native["llm_kwargs"]["model"],
        "native": off,
        "infinicore": on,
        "accuracy_delta_percentage_points": on["accuracy_percent"] - off["accuracy_percent"],
        "accuracy_gap_below_1pp": abs(on["accuracy_percent"] - off["accuracy_percent"]) < 1.0,
    }


def rescore_result(output_dir: Path, mode: str) -> dict:
    """Regrade saved answers without rerunning inference; retain generation evidence."""
    score_path = output_dir / f"{mode}.json"
    prediction_path = output_dir / f"{mode}.predictions.jsonl"
    result = json.loads(score_path.read_text(encoding="utf-8"))
    if not result["completed"] or result["errors"]:
        raise RuntimeError("Cannot rescore an incomplete/failed evaluation")
    if sha256(prediction_path) != result["predictions_sha256"]:
        raise ValueError("Prediction file checksum differs from the completed evaluation")
    dataset_path = output_dir / "dataset.json"
    if not dataset_path.is_file():
        dataset_path = Path(result["arguments"]["dataset_file"])
    if sha256(dataset_path) != result["dataset"]["prepared_sha256"]:
        raise ValueError("Frozen dataset checksum differs from the completed evaluation")
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    rows = [
        json.loads(line)
        for line in prediction_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(rows) != result["dataset"]["evaluated_count"]:
        raise ValueError("Prediction count differs from the completed evaluation")
    correct = invalid = truncated = tokens = 0
    generated = []
    for index, row in enumerate(rows):
        gold = dataset["records"][index]
        if (
            row["index"] != gold["index"]
            or row["question"] != gold["question"]
            or row["target"] != gold["target"]
        ):
            raise ValueError(f"Prediction and frozen question/gold do not match: {index}")
        generated.append([row["index"], row["text"], row["token_ids"], row["finish_reason"]])
        row["predicted"] = extract_answer(row["text"])
        row["correct"] = row["predicted"] is not None and row["predicted"] == gold["target"]
        correct += row["correct"]
        invalid += row["predicted"] is None
        truncated += row["finish_reason"] == "length"
        tokens += len(row["token_ids"])
    for source, backup in (
        (score_path, output_dir / f"{mode}.generation.json"),
        (prediction_path, output_dir / f"{mode}.generated.jsonl"),
    ):
        if not backup.exists():
            shutil.copyfile(source, backup)
    temporary = prediction_path.with_name(prediction_path.name + ".tmp")
    temporary.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
    )
    temporary.replace(prediction_path)
    count = len(rows)
    result["score"].update(
        correct=correct,
        total=count,
        accuracy=correct / count,
        accuracy_percent=100 * correct / count,
        invalid_answers=invalid,
        length_limited=truncated,
        output_tokens=tokens,
    )
    result["protocol"]["metric"] = METRIC
    result["protocol"]["scorer_sha256"] = sha256(Path(__file__).with_name("gsm8k_grading.py"))
    result["scoring"] = {
        "metric": METRIC,
        "scorer_sha256": sha256(Path(__file__).with_name("gsm8k_grading.py")),
        "rescored_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "generated_outputs_sha256": hashlib.sha256(
            json.dumps(generated, ensure_ascii=False).encode()
        ).hexdigest(),
        "original_predictions_sha256": sha256(output_dir / f"{mode}.generated.jsonl"),
        "original_result_sha256": sha256(output_dir / f"{mode}.generation.json"),
    }
    result["predictions_sha256"] = sha256(prediction_path)
    write_json(score_path, result)
    print("RESCORED", mode, json.dumps(result["score"]), flush=True)
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", help="Local Qwen3 model directory")
    parser.add_argument("--platform", choices=("metax", "ascend", "kunlun"), default="metax")
    parser.add_argument(
        "--routes",
        help="Comma-separated InfiniCore routes; Kunlun defaults to supported non-Attention routes",
    )
    parser.add_argument("--mode", choices=("both", "native", "infinicore"), default="both")
    parser.add_argument("--output-dir", type=Path, default=Path("results/gsm8k"))
    parser.add_argument("--dataset-id", default=DATASET_ID)
    parser.add_argument("--revision", default="master")
    parser.add_argument("--dataset-cache", type=Path, default=Path.home() / ".cache/modelscope")
    parser.add_argument(
        "--dataset-file",
        type=Path,
        help="Reuse the frozen dataset.json from a previous preparation",
    )
    parser.add_argument(
        "--prepare-only", action="store_true", help="Download/validate data without loading a model"
    )
    parser.add_argument(
        "--rescore-only",
        action="store_true",
        help="Regrade saved responses without loading a model",
    )
    parser.add_argument("--limit", type=int, help="First N test questions; omit for all 1319")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--max-num-batched-tokens", type=int, default=4096)
    parser.add_argument("--memory", type=float, default=0.55)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--enforce-eager", action="store_true")
    args = parser.parse_args(argv)
    if args.routes is None:
        if args.platform == "kunlun":
            from ..operators.platforms.kunlun import SUPPORTED_ROUTES

            args.routes = tuple(
                route
                for route in ROUTES
                if route in SUPPORTED_ROUTES
                and route not in {"PagedAttentionPrefill", "PagedAttentionDecode"}
            )
        else:
            args.routes = ROUTES
    else:
        args.routes = tuple(
            dict.fromkeys(item.strip() for item in args.routes.split(",") if item.strip())
        )
        if not args.routes or set(args.routes) - set(ROUTES):
            parser.error("--routes must contain known, nonempty operator route names")
    if args.prepare_only and args.rescore_only:
        parser.error("--prepare-only and --rescore-only are mutually exclusive")
    if not args.prepare_only and not args.rescore_only and not args.model:
        parser.error("--model is required unless --prepare-only is used")
    if any(
        value < 1
        for value in (
            args.batch_size,
            args.max_tokens,
            args.max_model_len,
            args.max_num_batched_tokens,
        )
    ):
        parser.error("Batch and token limits must be positive")
    if args.limit is not None and not 1 <= args.limit <= 1319:
        parser.error("--limit must be between 1 and 1319")
    if not 0 < args.memory < 1:
        parser.error("--memory must be between 0 and 1")
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.rescore_only:
        for mode in ("native", "infinicore") if args.mode == "both" else (args.mode,):
            rescore_result(args.output_dir, mode)
        if args.mode == "both":
            comparison = compare_scores(args.output_dir)
            write_json(args.output_dir / "comparison.json", comparison)
            print("COMPARISON", json.dumps(comparison), flush=True)
        return 0
    dataset_path = prepare_dataset(args)
    if args.prepare_only:
        return 0
    if args.mode != "both":
        return run_mode(args, dataset_path)
    # One frozen data file, same visible GPU, and no shared live engine state.
    arguments = list(sys.argv[1:] if argv is None else argv)
    for mode in ("native", "infinicore"):
        command = [
            sys.executable,
            "-m",
            "vllm_infinicore.benchmarks.gsm8k",
            *arguments,
            "--mode",
            mode,
            "--dataset-file",
            str(dataset_path),
        ]
        status = subprocess.run(
            command, env=mode_environment(mode, os.environ, args.platform, args.routes)
        ).returncode
        if status:
            return status
    comparison = compare_scores(args.output_dir)
    write_json(args.output_dir / "comparison.json", comparison)
    print("COMPARISON", json.dumps(comparison), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
