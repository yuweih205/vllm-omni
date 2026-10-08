# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Operator-level bitwise and latency check for the six QK/RoPE PRs.

The exact CUDA candidate is an isolated experiment; this script does not
claim a full-model or compiled-path result. Each process uses one H200.
"""

import argparse
import hashlib
import itertools
import json
import os
from pathlib import Path
import statistics
import subprocess

import torch

from benchmark_precision_r3 import capture, compare, exact, measure, runtime


PROFILES = {
    "flux2": (1, 512, 4096, 48, "both", "full"),
    "flux1": (1, 512, 4096, 24, "both", "full"),
    "hunyuan15": (1, 2048, 256, 16, "joint", "identity_second"),
    "ovis": (1, 256, 4096, 24, "both", "full"),
    "zimage": (1, 0, 4096, 30, "single", "full"),
    "qwen": (1, 512, 4096, 24, "joint", "per_stream_fused"),
}


def gpu_state():
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0].strip()
    result = subprocess.run(
        [
            "nvidia-smi", "-i", visible,
            "--query-gpu=name,uuid,driver_version,pstate,clocks.sm,clocks.mem,power.draw,temperature.gpu,utilization.gpu",
            "--format=csv,noheader",
        ],
        capture_output=True, text=True, check=False,
    )
    return {
        "returncode": result.returncode,
        "physical_device": visible,
        "gpu": result.stdout.strip(),
        "stderr": result.stderr.strip(),
    }


def tensor_bits_equal(actual, expected):
    return exact(compare(actual, expected))


def prepared(rt, profile, seed):
    batch, first_len, second_len, heads, path, mode = PROFILES[profile]
    epsilon = rt["epsilon"]
    torch.manual_seed(seed)
    device = torch.device("cuda")

    def project(seq):
        packed = torch.randn(batch, seq, heads * 128 * 3, device=device, dtype=torch.bfloat16)
        return tuple(t.unflatten(-1, (heads, 128)) for t in packed.chunk(3, dim=-1))

    q0, k0, v0 = project(first_len)
    q1, k1, v1 = project(second_len)
    qs, ks, _ = project(first_len + second_len)
    norms = rt["norms"]
    for norm in norms:
        norm.weight.uniform_(0.5, 1.5)
    for x, norm in zip((q0, k0, q1, k1), norms):
        if x.numel():
            if profile == "zimage":
                assert norm._forward_method.__name__ == "forward_cuda"
                # The model's omni RMSNorm calls the direct vLLM CUDA op;
                # verify it works before the fallback-catching forward runs.
                norm._forward_fused(x)
            else:
                assert rt["ir"].ops.rms_norm.dispatch(x, norm.weight, epsilon, None).provider == "vllm_c"

    if mode == "identity_second":
        angles = torch.randn(first_len, 64, device=device)
        cos, sin = angles.cos(), angles.sin()
        table = rt["current"].pack_qk_norm_rope_table(
            cos, sin, batch, dtype=torch.bfloat16, min_tokens=0, identity_rows=second_len
        )
    else:
        angles = torch.randn(first_len + second_len, 64, device=device)
        cos, sin = angles.cos(), angles.sin()
        table_dtype = torch.float32 if mode == "per_stream_fused" else torch.bfloat16
        table = rt["current"].pack_qk_norm_rope_table(cos, sin, batch, dtype=table_dtype, min_tokens=0)
    assert table is not None and table.shape == (batch * (first_len + second_len), 128)

    def original_joint():
        if mode == "per_stream_fused":
            t0, t1 = table[:first_len], table[first_len:]
            a, b = rt["current"].fused_qk_norm_rope(
                q0.reshape(-1, heads, 128), k0.reshape(-1, heads, 128),
                norms[0].weight, norms[1].weight, t0, epsilon, interleaved=True,
            )
            c, d = rt["current"].fused_qk_norm_rope(
                q1.reshape(-1, heads, 128), k1.reshape(-1, heads, 128),
                norms[2].weight, norms[3].weight, t1, epsilon, interleaved=True,
            )
            return torch.cat((a.view_as(q0), c.view_as(q1)), dim=1), torch.cat(
                (b.view_as(k0), d.view_as(k1)), dim=1
            ), torch.cat((v0, v1), dim=1)
        if mode == "identity_second":
            a, b = rt["apply_rope"](rt["rope"], norms[0](q0), norms[1](k0), (cos, sin))
            return torch.cat((a, norms[2](q1)), dim=1), torch.cat(
                (b, norms[3](k1)), dim=1
            ), torch.cat((v0, v1), dim=1)
        q = torch.cat((norms[0](q0), norms[2](q1)), dim=1)
        k = torch.cat((norms[1](k0), norms[3](k1)), dim=1)
        q, k = rt["apply_rope"](rt["rope"], q, k, (cos, sin))
        return q, k, torch.cat((v0, v1), dim=1)

    def joint(module):
        return module.fused_joint_qkv_norm_rope(
            q0, k0, v0, q1, k1, v1, *(n.weight for n in norms), table, epsilon
        )

    def original_single():
        return rt["apply_rope"](rt["rope"], norms[0](qs), norms[1](ks), (cos, sin))

    def single(module):
        q, k = module.fused_qk_norm_rope(
            qs.reshape(-1, heads, 128), ks.reshape(-1, heads, 128),
            norms[0].weight, norms[1].weight, table, epsilon, interleaved=True,
        )
        return q.view_as(qs), k.view_as(ks)

    paths = {}
    if path in ("joint", "both"):
        paths["joint"] = {
            "original": original_joint,
            "current": lambda: joint(rt["current"]),
        }
        if mode != "per_stream_fused":
            paths["joint"]["exact"] = lambda: joint(rt["candidate"])
    if path in ("single", "both"):
        paths["single"] = {
            "original": original_single,
            "current": lambda: single(rt["current"]),
            "exact": lambda: single(rt["candidate"]),
        }
    return paths


def timings(functions, *, graph, warmup=20, repeats=100):
    names = tuple(functions)
    if graph:
        captured = {name: capture(fn) for name, fn in functions.items()}
        for name, (_, out) in captured.items():
            assert tensor_bits_equal(out, functions[name]())
        calls = {name: graph_obj.replay for name, (graph_obj, _) in captured.items()}
        divisor = 10
    else:
        calls, divisor = functions, 1
    for fn in calls.values():
        for _ in range(warmup):
            fn()
    samples = {name: [] for name in names}
    orders = list(itertools.permutations(names))
    if len(names) == 2:
        orders *= 3
    for order in orders:
        for name in order:
            samples[name].append(measure(calls[name], repeats) / divisor)
    return {name: {"median_ms": statistics.median(values), "samples_ms": values} for name, values in samples.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=PROFILES, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261008)
    parser.add_argument("--include-cuda-graph", action="store_true")
    args = parser.parse_args()
    root = Path("/inspire/ssd/project/video-generation/public/huangyuwei")
    output = args.output.resolve()
    assert output.is_relative_to(root) and not output.exists()
    output.parent.mkdir(parents=True, exist_ok=True)
    rt = runtime("vllm_c")
    rt["epsilon"] = 1e-5 if args.profile == "zimage" else 1e-6
    if args.profile == "zimage":
        from vllm_omni.diffusion.layers.norm import RMSNorm

        rt["norms"] = [
            RMSNorm(128, eps=rt["epsilon"]).cuda().to(torch.bfloat16).eval()
            for _ in range(4)
        ]
    norm_class = type(rt["norms"][0])
    result = {
        "profile": args.profile,
        "shape": PROFILES[args.profile],
        "seed": args.seed,
        "gpu": gpu_state(),
        "visible_gpu": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "torch_device": torch.cuda.get_device_name(0),
        "versions": rt["versions"],
        "runtime_python": str(Path(os.sys.executable).resolve()),
        "image": os.environ.get("OMNI_BENCH_IMAGE", "unknown"),
        "include_cuda_graph": args.include_cuda_graph,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "current_source_sha256": hashlib.sha256(Path(rt["current"].__file__).read_bytes()).hexdigest(),
        "candidate_source_sha256": hashlib.sha256(Path(rt["candidate"].__file__).read_bytes()).hexdigest(),
        "epsilon": rt["epsilon"],
        "norm_implementation": f"{norm_class.__module__}.{norm_class.__qualname__}",
        "norm_forward": rt["norms"][0]._forward_method.__qualname__,
        "scope": "Q/K/V preparation only, eager Python model path; no full-model or compiled claim",
        "paths": {},
    }
    with torch.inference_mode():
        paths = prepared(rt, args.profile, args.seed)
        for label, functions in paths.items():
            reference = functions["original"]()
            current = functions["current"]()
            bitwise = {
                "current_vs_original": compare(current, reference),
                "original_repeat": compare(functions["original"](), reference),
            }
            assert exact(bitwise["original_repeat"])
            if "exact" in functions:
                candidate = functions["exact"]()
                bitwise["exact_vs_original"] = compare(candidate, reference)
                bitwise["candidate_repeat"] = compare(functions["exact"](), candidate)
                passed = exact(bitwise["exact_vs_original"]) and exact(bitwise["candidate_repeat"])
            else:
                passed = exact(bitwise["current_vs_original"])
            if not passed:
                result["paths"][label] = {"bitwise": bitwise, "status": "PARITY_FAILED_PERF_SKIPPED"}
                result["status"] = "PARITY_FAILED_PERF_SKIPPED"
                output.write_text(json.dumps(result, indent=2) + "\n")
                print(json.dumps({"profile": args.profile, "path": label, "bitwise": bitwise, "status": result["status"]}), flush=True)
                raise SystemExit(2)
            eager = timings(functions, graph=False)
            result["paths"][label] = {"bitwise": bitwise, "eager": eager}
            output.write_text(json.dumps(result, indent=2) + "\n")
            if args.include_cuda_graph:
                replay = timings(functions, graph=True)
                result["paths"][label]["cuda_graph"] = replay
            output.write_text(json.dumps(result, indent=2) + "\n")
            print(json.dumps({"profile": args.profile, "path": label, **result["paths"][label]}), flush=True)
    result["status"] = "COMPLETE"
    output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
