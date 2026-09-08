"""Isolated Ascend graph throughput case with worker capture/replay evidence."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import time
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def graph_state(worker):
    from vllm.compilation.counter import compilation_counter
    from vllm_ascend.compilation import acl_graph
    from vllm_infinicore.ops import infinicore_backend as backend
    cfg = worker.vllm_config
    return dict(
        pid=os.getpid(), rank=worker.rank,
        captures=compilation_counter.num_cudagraph_captured,
        replays=getattr(acl_graph, '_benchmark_replays', 0),
        graph_mode=str(cfg.compilation_config.cudagraph_mode),
        capture_sizes=cfg.compilation_config.cudagraph_capture_sizes,
        calls=backend.backend_call_counts(),
        fallbacks=backend.backend_fallback_counts(),
        fallback_reasons=backend.backend_fallback_reasons(),
    )


def instrument_replay(worker):
    from vllm_ascend.compilation import acl_graph
    from vllm.config import CUDAGraphMode
    from vllm.forward_context import get_forward_context
    if not hasattr(acl_graph, '_benchmark_replays'):
        acl_graph._benchmark_replays = 0
        original = acl_graph.ACLGraphWrapper.__call__

        def counted(self, *args, **kwargs):
            ctx = get_forward_context()
            entry = self.concrete_aclgraph_entries.get(ctx.batch_descriptor)
            replay = (ctx.cudagraph_runtime_mode != CUDAGraphMode.NONE
                      and ctx.cudagraph_runtime_mode == self.runtime_mode
                      and entry is not None and entry.aclgraph is not None)
            result = original(self, *args, **kwargs)
            if replay:
                acl_graph._benchmark_replays += 1
            return result
        acl_graph.ACLGraphWrapper.__call__ = counted
    return graph_state(worker)


def sync_worker(worker):
    import torch
    torch.npu.synchronize()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('case', choices=['prepare', 'native', 'infinicore'])
    p.add_argument('--root', required=True)
    p.add_argument('--model', default='/models/Qwen3.8-27B')
    p.add_argument('--tp', type=int, default=2)
    p.add_argument('--batches', default='1,4,16,32')
    p.add_argument('--max-num-seqs', type=int, default=32)
    p.add_argument('--memory', type=float, default=0.95)
    p.add_argument('--input-len', type=int, default=1024)
    p.add_argument('--output-len', type=int, default=1024)
    p.add_argument('--repeats', type=int, default=3)
    p.add_argument('--warmups', type=int, default=1)
    p.add_argument('--library', default='/workspace/work/infinicore-ascend-20260907/build/libvllm_infinicore_ascend.so')
    a = p.parse_args()
    root = Path(a.root)
    root.mkdir(parents=True, exist_ok=True)
    batches = [int(x) for x in a.batches.split(',')]
    for key in list(os.environ):
        if key.startswith('VLLM_INFINICORE_'):
            del os.environ[key]
    os.environ.update(
        ASCEND_RT_VISIBLE_DEVICES=','.join(map(str, range(a.tp))),
        HF_HUB_OFFLINE='1', VLLM_WORKER_MULTIPROC_METHOD='spawn',
        VLLM_ENABLE_V1_MULTIPROCESSING='0',
        VLLM_PLUGINS='ascend,ascend_kv_connector,ascend_model,ascend_model_loader,ascend_service_profiling'
        + (',vllm_infinicore' if a.case == 'infinicore' else ''),
        VLLM_INFINICORE_ENABLE_PATCHES='1' if a.case == 'infinicore' else '0',
        VLLM_INFINICORE_ROUTES='Embedding,MatMul,LMHead',
        VLLM_INFINICORE_ASCEND_LIBRARY=a.library,
        VLLM_INFINICORE_STRICT_BACKEND='1',
    )
    result = dict(case=a.case, args=vars(a), completed=False, validation_errors=[], rows=[])
    dest = root / f'{a.case}-tp{a.tp}.json'

    def save():
        dest.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    llm = None
    try:
        if a.case == 'prepare':
            from transformers import AutoTokenizer
            tok = AutoTokenizer.from_pretrained(a.model, local_files_only=True)
            # A coherent long-writing request gives 1K output room without forcing
            # a one-sentence answer to continue beyond EOS.
            message = ('Write a detailed educational chapter on how computers process information. '
                       'Explain processors, memory, storage, networking, algorithms, practical examples, '
                       'limitations, and future developments in connected prose. ')
            seed = tok.encode(message, add_special_tokens=False)
            ids = (seed * (a.input_len // len(seed) + 1))[:a.input_len]
            (root/'prompts.json').write_text(json.dumps(dict(token_ids=ids, input_len=a.input_len)))
            result['completed'] = True
            return 0
        import importlib.metadata
        from vllm import LLM, SamplingParams
        from vllm.config import CompilationMode, CUDAGraphMode
        from vllm_infinicore.validation import compute_text_health, detect_degenerate_repetition
        result['versions'] = {n: importlib.metadata.version(n) for n in ['vllm','vllm_ascend','torch','torch_npu']}
        prompt_data = json.loads((root/'prompts.json').read_text())
        ids = prompt_data['token_ids']
        assert len(ids) == a.input_len
        result['prompt_sha256'] = hashlib.sha256(json.dumps(ids).encode()).hexdigest()
        sizes = [x for x in [1,2,4,8,16,32] if x <= a.max_num_seqs]
        kwargs = dict(model=a.model, tensor_parallel_size=a.tp, dtype='bfloat16',
                      max_model_len=a.input_len+a.output_len, max_num_seqs=a.max_num_seqs,
                      max_num_batched_tokens=1024, gpu_memory_utilization=a.memory,
                      enforce_eager=False, enable_prefix_caching=False, seed=0,
                      limit_mm_per_prompt={'image':0,'video':0},
                      compilation_config=dict(mode=CompilationMode.VLLM_COMPILE,
                          cudagraph_mode=CUDAGraphMode.FULL_DECODE_ONLY,
                          cudagraph_capture_sizes=sizes, cudagraph_num_of_warmups=1))
        result['llm_kwargs'] = json.loads(json.dumps(kwargs, default=str))
        save()
        llm = LLM(**kwargs)
        result['startup_workers'] = llm.collective_rpc(instrument_replay)
        assert len(result['startup_workers']) == a.tp
        params = SamplingParams(temperature=0.0, top_p=1.0, top_k=1,
                                ignore_eos=True, min_tokens=a.output_len, max_tokens=a.output_len)
        for bs in batches:
            requests = [{'prompt_token_ids':ids} for _ in range(bs)]
            for _ in range(a.warmups):
                preview = llm.generate(requests, params, use_tqdm=False)
            print('PREVIEW', bs, preview[0].outputs[0].text[:500], flush=True)
            for rep in range(a.repeats):
                llm.collective_rpc(sync_worker)
                before = llm.collective_rpc(graph_state)
                started = time.perf_counter()
                outputs = llm.generate(requests, params, use_tqdm=False)
                elapsed = time.perf_counter()-started
                after = llm.collective_rpc(graph_state)
                records = []
                for output in outputs:
                    completion = output.outputs[0]
                    tokens = list(completion.token_ids)
                    health = compute_text_health(completion.text, tokens)
                    repetition = detect_degenerate_repetition(tokens)
                    errors = health.validation_errors()
                    if repetition.is_degenerate:
                        errors += list(repetition.reasons)
                    if len(tokens) != a.output_len or len(output.prompt_token_ids) != a.input_len:
                        errors.append('token_count_mismatch')
                    result['validation_errors'].extend(errors)
                    records.append(dict(input_tokens=len(output.prompt_token_ids), output_tokens=len(tokens),
                                        token_ids=tokens, text=completion.text, text_health=health.as_dict(),
                                        repetition=repetition.as_dict()))
                generated = sum(x['output_tokens'] for x in records)
                row = dict(batch_size=bs, repeat=rep, elapsed_s=elapsed,
                           output_tokens=generated, output_tps=generated/elapsed,
                           workers_before=before, workers_after=after, outputs=records)
                for b, c in zip(before, after):
                    if c['captures'] <= 0 or c['replays'] <= b['replays']:
                        result['validation_errors'].append('no_graph_capture_or_replay_evidence')
                result['rows'].append(row)
                print('MEASURE', json.dumps({k:v for k,v in row.items() if k not in ['outputs','workers_before','workers_after']}), flush=True)
                save()
        result['summary'] = {str(bs):dict(median_output_tps=statistics.median(
            x['output_tps'] for x in result['rows'] if x['batch_size']==bs)) for bs in batches}
        result['final_workers'] = llm.collective_rpc(graph_state)
        result['completed'] = True
    except Exception:
        result['exception'] = traceback.format_exc()
        print(result['exception'], flush=True)
    finally:
        if llm is not None:
            try:
                llm.llm_engine.engine_core.shutdown(timeout=10.0)
            except Exception:
                result['shutdown_exception'] = traceback.format_exc()
        save()
        print('RESULT_PATH', dest, flush=True)
    return 0 if result['completed'] and not result['validation_errors'] else 1


if __name__ == '__main__':
    sys.exit(main())
