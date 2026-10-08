# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Strict operator parity, then three-arm latency on a frozen Flux baseline.

No pretrained weights, no model integration, no end-to-end speed claim.
"""

import argparse
import gc
import hashlib
import importlib
import importlib.metadata
import inspect
import itertools
import json
import os
from pathlib import Path
import statistics
import subprocess
import time

import torch

from probe_flux_bitwise import difference


BASELINE_SHA = "34787d3065731491a2992989850b2fa51ff2ad147c86cc1737aaa83d6d65457a"
VARIANT_SHA = {
    "vllm_c": "2fefb2156cb8dbe45dc81157b10b27883ac052caccb289a392d018b222c8559d",
    "native": "19576b3e3a4763b9d1dcc7af525129740f7bb49618349de159cf7ff0374017cb",
}
PERF_SHAPES = [(1, 512, 4096, 24), (1, 512, 4096, 48), (2, 512, 4096, 24), (1, 512, 16384, 48)]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def gpu_state():
    result = subprocess.run(
        ["nvidia-smi", "-i", "0", "--query-gpu=name,uuid,driver_version,pstate,clocks.sm,clocks.mem,power.draw,temperature.gpu,utilization.gpu", "--format=csv,noheader"],
        capture_output=True, text=True, check=False,
    )
    return {"returncode": result.returncode, "gpu0": result.stdout.strip(), "stderr": result.stderr.strip()}


def runtime(provider):
    from tests.model_executor.helpers import bootstrap_vllm_layer_custom_op_modules
    bootstrap_vllm_layer_custom_op_modules()
    from vllm import ir
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.model_executor.layers.layernorm import RMSNorm
    from vllm_omni.diffusion.layers import fused_qk_norm_rope as current
    from vllm_omni.diffusion.layers.rope import RotaryEmbedding, apply_rope_to_qk

    assert torch.cuda.is_available() and "H200" in torch.cuda.get_device_name()
    versions = {n: importlib.metadata.version(n) for n in ("torch", "triton", "vllm")}
    assert versions["vllm"] == "0.29.0" and torch.version.cuda == "13.0", versions
    assert os.environ.get("VLLM_BATCH_INVARIANT", "0") in ("", "0")
    assert sha(inspect.getfile(current)) == BASELINE_SHA
    mode = "cuda" if provider == "vllm_c" else "native"
    variant = importlib.import_module(f"candidate_{mode}_r3")
    assert sha(inspect.getfile(variant)) == VARIANT_SHA[provider]
    ir.ops.rms_norm.set_default([provider])
    with set_current_vllm_config(VllmConfig()):
        norms = [RMSNorm(128, eps=1e-6).cuda().to(torch.bfloat16).eval() for _ in range(4)]
    rope = RotaryEmbedding(is_neox_style=False)
    assert rope.apply_rotary_emb_vllm_flash_attn is not None
    return {
        "provider": provider, "ir": ir, "current": current, "candidate": variant,
        "norms": norms, "rope": rope, "apply_rope": apply_rope_to_qk, "versions": versions,
    }


def workload(rt, shape, seed, weights="random", scale=1.0):
    batch, text, image, heads = shape
    total = text + image
    torch.manual_seed(seed)

    def project(seq):
        packed = torch.randn(batch, seq, heads * 128 * 3, device="cuda", dtype=torch.bfloat16)
        packed.mul_(scale)
        return tuple(t.unflatten(-1, (heads, 128)) for t in packed.chunk(3, dim=-1))

    q0, k0, v0 = project(text)
    q1, k1, v1 = project(image)
    sq, sk, _ = project(total)
    norms = rt["norms"]
    for norm in norms:
        if weights == "ones":
            norm.weight.fill_(1)
        elif weights == "signed":
            norm.weight.uniform_(-1.5, 1.5)
        else:
            norm.weight.uniform_(0.5, 1.5)
    for x, norm in zip((q0, k0, q1, k1), norms):
        assert rt["ir"].ops.rms_norm.dispatch(x, norm.weight, 1e-6, None).provider == rt["provider"]
    angles = torch.randn(total, 64, device="cuda")
    cos, sin = angles.cos(), angles.sin()
    table = rt["current"].pack_qk_norm_rope_table(cos, sin, batch, dtype=torch.bfloat16, min_tokens=0)
    assert table is not None
    flat_q = sq.reshape(-1, heads, 128)
    flat_k = sk.reshape(-1, heads, 128)
    assert flat_q.data_ptr() == sq.data_ptr() and flat_k.data_ptr() == sk.data_ptr()
    assert rt["current"]._fused_cuda_supported(q0, k0, 128, 128, True)
    assert rt["candidate"]._fused_cuda_supported(q0, k0, 128, 128, True)

    def original_joint():
        q = torch.cat((norms[0](q0), norms[2](q1)), dim=1)
        k = torch.cat((norms[1](k0), norms[3](k1)), dim=1)
        v = torch.cat((v0, v1), dim=1)
        q, k = rt["apply_rope"](rt["rope"], q, k, (cos, sin))
        return q, k, v

    def original_single():
        return rt["apply_rope"](rt["rope"], norms[0](sq), norms[1](sk), (cos, sin))

    def joint(module):
        return module.fused_joint_qkv_norm_rope(
            q0, k0, v0, q1, k1, v1, *(n.weight for n in norms), table, 1e-6,
        )

    def single(module):
        q, k = module.fused_qk_norm_rope(flat_q, flat_k, norms[0].weight, norms[1].weight, table, 1e-6, interleaved=True)
        return q.view_as(sq), k.view_as(sk)

    return {
        "joint": {"original": original_joint, "current_fused": lambda: joint(rt["current"]), "exact_candidate": lambda: joint(rt["candidate"])},
        "single": {"original": original_single, "current_fused": lambda: single(rt["current"]), "exact_candidate": lambda: single(rt["candidate"])},
        "pack": lambda: rt["current"].pack_qk_norm_rope_table(cos, sin, batch, dtype=torch.bfloat16, min_tokens=0),
    }


def compare(actual, expected):
    assert len(actual) == len(expected)
    return {name: difference(a, e) for name, a, e in zip(("q", "k", "v"), actual, expected)}


def exact(record):
    return all(x["bit_mismatches"] == 0 and x["finite"] for x in record.values())


def measure(fn, iterations):
    torch.cuda.synchronize()
    begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    begin.record()
    for _ in range(iterations):
        value = fn()
    end.record()
    end.synchronize()
    del value
    return begin.elapsed_time(end) / iterations


def stats(samples):
    ordered = sorted(samples)
    return {
        "median_ms": statistics.median(ordered), "min_ms": min(ordered), "max_ms": max(ordered),
        "mean_ms": statistics.mean(ordered), "samples_ms": samples,
    }


def capture(fn, inner_calls=10):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        for _ in range(inner_calls):
            outputs = fn()
    graph.replay()
    torch.cuda.synchronize()
    return graph, outputs


def bench_group(functions, warmup, iterations, calls_per_invocation=1):
    names = tuple(functions)
    assert names == ("original", "current_fused", "exact_candidate")
    for fn in functions.values():
        for _ in range(warmup):
            fn()
    rounds = list(itertools.permutations(names))
    samples = {name: [] for name in names}
    for order in rounds:
        for name in order:
            samples[name].append(measure(functions[name], iterations) / calls_per_invocation)
    result = {name: stats(values) for name, values in samples.items()}
    result["orders"] = rounds
    result["exact_speedup_vs_original"] = result["original"]["median_ms"] / result["exact_candidate"]["median_ms"]
    result["exact_speedup_vs_current"] = result["current_fused"]["median_ms"] / result["exact_candidate"]["median_ms"]
    result["exact_latency_change_vs_current_pct"] = 100 * (result["exact_candidate"]["median_ms"] / result["current_fused"]["median_ms"] - 1)
    return result


def run_accuracy(rt, output):
    cases = [(shape, seed, weight, 1.0) for shape, seed, weight in itertools.product(PERF_SHAPES, (0, 3), ("ones", "random"))]
    cases += [((2, 5, 7, 6), 29, "signed", scale) for scale in (0.0, 1e-4, 1.0, 1e3)]
    cases += [((2, 77, 1030, 24), 37, "signed", 1.0)]
    passed = True
    records = []
    with (output / "accuracy.jsonl").open("w") as log:
        for shape, seed, weights, scale in cases:
            data = workload(rt, shape, seed, weights, scale)
            for path in ("joint", "single"):
                fns = data[path]
                reference = fns["original"]()
                current = fns["current_fused"]()
                candidate = fns["exact_candidate"]()
                record = {
                    "shape": shape, "seed": seed, "weights": weights, "scale": scale, "path": path,
                    "exact_vs_original": compare(candidate, reference),
                    "current_vs_original": compare(current, reference),
                    "reference_repeatability": compare(fns["original"](), reference),
                    "candidate_repeatability": compare(fns["exact_candidate"](), candidate),
                }
                record["pass"] = all(exact(record[key]) for key in ("exact_vs_original", "reference_repeatability", "candidate_repeatability"))
                passed &= record["pass"]
                records.append(record)
                log.write(json.dumps(record) + "\n")
                log.flush()
                print(json.dumps({"accuracy": record}), flush=True)
                del reference, current, candidate
            del data
            gc.collect()
    return passed, records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", choices=("vllm_c", "native"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    assert args.warmup >= 10 and args.iterations >= 50
    output = args.output_dir.resolve()
    allowed = Path("/inspire/ssd/project/video-generation/public/huangyuwei")
    if not output.is_relative_to(allowed):
        raise RuntimeError("Output outside authorized personal storage")
    output.mkdir(parents=True, exist_ok=False)
    rt = runtime(args.provider)
    metadata = {
        "provider": args.provider, "versions": rt["versions"], "cuda": torch.version.cuda, "gpu": gpu_state(),
        "script_sha256": sha(__file__), "baseline_sha256": BASELINE_SHA, "variant_sha256": VARIANT_SHA[args.provider],
        "compile": False, "dtype": "bfloat16", "head_dim": 128, "epsilon": 1e-6,
        "warmup": args.warmup, "iterations_per_sample": args.iterations, "samples_per_arm": 6,
        "graph_calls_per_replay": 10,
        "scope": "operator Q/K/V preparation only; no model weights, attention, projections or end-to-end claim",
        "measurement": "eager CUDA-event latency and CUDA-graph replay reported separately; no profiler",
        "table_cost": "packed table reused by fused blocks; packing cost reported separately per forward",
        "rmsnorm_priority": rt["ir"].ops.rms_norm.get_priority(),
        "rmsnorm_forward": repr(rt["norms"][0]._forward_method), "rope_forward": repr(rt["rope"]._forward_method),
    }
    (output / "environment.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata), flush=True)
    with torch.inference_mode():
        passed, accuracy = run_accuracy(rt, output)
        if not passed:
            result = {"status": "QUALITY_FAILED_PERF_SKIPPED", "accuracy_cases": len(accuracy), "failed_cases": sum(not r["pass"] for r in accuracy)}
            (output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
            print(json.dumps(result), flush=True)
            return
        if args.verify_only:
            result = {"status": "ACCURACY_ONLY", "provider": args.provider, "accuracy_cases": len(accuracy), "accuracy_passed": True}
            (output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
            print(json.dumps(result), flush=True)
            return
        measurements = []
        with (output / "performance.jsonl").open("w") as log:
            for shape in PERF_SHAPES:
                data = workload(rt, shape, 2026)
                for path in ("joint", "single"):
                    fns = data[path]
                    before = gpu_state()
                    # The measured seed must pass too, not only the accuracy seeds.
                    assert exact(compare(fns["exact_candidate"](), fns["original"]()))
                    eager = bench_group(fns, args.warmup, args.iterations)
                    graphs = {name: capture(fn) for name, fn in fns.items()}
                    for name, (_, outputs) in graphs.items():
                        assert exact(compare(outputs, fns[name]())), (shape, path, name)
                    assert exact(compare(graphs["exact_candidate"][1], graphs["original"][1]))
                    replay = {name: g.replay for name, (g, _) in graphs.items()}
                    graph_timing = bench_group(replay, args.warmup, args.iterations, calls_per_invocation=10)
                    assert exact(compare(fns["exact_candidate"](), fns["original"]()))
                    record = {"shape": shape, "path": path, "bitwise": True, "eager": eager, "cuda_graph": graph_timing, "gpu_before": before, "gpu_after": gpu_state()}
                    measurements.append(record)
                    log.write(json.dumps(record) + "\n")
                    log.flush()
                    print(json.dumps({"performance": record}), flush=True)
                    del replay, graphs
                    gc.collect()
                for _ in range(args.warmup):
                    data["pack"]()
                pack = {"shape": shape, "kind": "one_time_table_pack", "eager": stats([measure(data["pack"], args.iterations) for _ in range(6)])}
                log.write(json.dumps(pack) + "\n")
                log.flush()
                print(json.dumps(pack), flush=True)
                del data
                gc.collect()
    result = {
        "status": "COMPLETE_OPERATOR_ONLY", "provider": args.provider, "accuracy_cases": len(accuracy),
        "accuracy_passed": passed, "benchmark_cases": len(measurements), "measurements": measurements,
        "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "measurements"}), flush=True)


if __name__ == "__main__":
    main()
