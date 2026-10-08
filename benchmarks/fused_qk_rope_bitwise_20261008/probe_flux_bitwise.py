"""Isolated accuracy experiment; does not modify a PR or installed packages."""

import argparse
import hashlib
import importlib.metadata
import inspect
import itertools
import json
import os
from pathlib import Path
import time

import torch
import triton
import triton.language as tl


@triton.jit
def candidate(
    X, W, TABLE, OUT, NORM,
    stride_t: tl.constexpr, stride_h: tl.constexpr,
    HEADS: tl.constexpr, DIM: tl.constexpr, HPP: tl.constexpr,
    ROUND_BEFORE_WEIGHT: tl.constexpr, EPS: tl.constexpr,
):
    token = tl.program_id(0).to(tl.int64)
    heads = tl.program_id(1) * HPP + tl.arange(0, HPP)
    dims = tl.arange(0, DIM)
    mask = heads[:, None] < HEADS
    offsets = token * stride_t + heads[:, None] * stride_h + dims[None, :]
    x = tl.load(X + offsets, mask=mask, other=0).to(tl.float32)
    w = tl.load(W + dims).to(tl.float32)
    rstd = tl.rsqrt(tl.sum(x * x, 1) / DIM + EPS)
    n = x * rstd[:, None]
    if ROUND_BEFORE_WEIGHT:
        n = n.to(tl.bfloat16).to(tl.float32)
    n = (n * w[None, :]).to(tl.bfloat16).to(tl.float32)
    partner = tl.load(X + token * stride_t + heads[:, None] * stride_h + (dims ^ 1)[None, :], mask=mask, other=0).to(tl.float32)
    partner_w = tl.load(W + (dims ^ 1)).to(tl.float32)
    pn = partner * rstd[:, None]
    if ROUND_BEFORE_WEIGHT:
        pn = pn.to(tl.bfloat16).to(tl.float32)
    pn = (pn * partner_w[None, :]).to(tl.bfloat16).to(tl.float32)
    cos = tl.load(TABLE + token * DIM + dims // 2).to(tl.float32)
    sin = tl.load(TABLE + token * DIM + DIM // 2 + dims // 2).to(tl.float32)
    first = n * cos - pn * sin
    second = n * cos + pn * sin
    output = tl.where((dims % 2)[None, :] == 0, first, second)
    target = (token * HEADS + heads[:, None]) * DIM + dims[None, :]
    tl.store(OUT + target, output, mask=mask)
    tl.store(NORM + target, n, mask=mask)


def difference(actual, expected):
    # Integer views also catch signed-zero differences; no tolerance is used.
    a, e = actual.contiguous(), expected.contiguous()
    return {
        "bit_mismatches": int((a.view(torch.int16) != e.view(torch.int16)).sum().item()),
        "elements": a.numel(),
        "max_abs": float((a.float() - e.float()).abs().max().item()),
        "finite": bool(torch.isfinite(a).all() and torch.isfinite(e).all()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--provider", choices=("vllm_c", "native"), required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    allowed = Path("/inspire/ssd/project/video-generation/public/huangyuwei")
    if not output.is_relative_to(allowed):
        raise RuntimeError("Output path outside authorized personal storage")
    output.mkdir(parents=True, exist_ok=False)
    from tests.model_executor.helpers import bootstrap_vllm_layer_custom_op_modules
    bootstrap_vllm_layer_custom_op_modules()
    from vllm import ir
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.model_executor.layers.layernorm import RMSNorm
    from vllm_omni.diffusion.layers.rope import RotaryEmbedding
    from vllm_omni.diffusion.layers import fused_qk_norm_rope as module

    source = Path(inspect.getfile(module))
    expected_hash = "34787d3065731491a2992989850b2fa51ff2ad147c86cc1737aaa83d6d65457a"
    if hashlib.sha256(source.read_bytes()).hexdigest() != expected_hash:
        raise RuntimeError("Shared reference source changed")
    versions = {name: importlib.metadata.version(name) for name in ("torch", "triton", "vllm")}
    if versions["vllm"] != "0.29.0" or torch.version.cuda != "13.0":
        raise RuntimeError(f"Unexpected runtime baseline: {versions}")
    assert torch.cuda.is_available() and "H200" in torch.cuda.get_device_name()
    metadata = {
        "versions": versions, "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(), "source": str(source),
        "source_sha256": expected_hash, "compile": False,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "scope": "Q/K preparation only; no model weights, no end-to-end claim",
    }
    with set_current_vllm_config(VllmConfig()):
        norm = RMSNorm(128, eps=1e-6).cuda().to(torch.bfloat16).eval()
    # CUDA diffusion workers prefer vllm_c (CudaOmniPlatform); native is a
    # separately labelled control, never substituted for the production arm.
    ir.ops.rms_norm.set_default([args.provider])
    metadata["reference_provider"] = args.provider
    rope = RotaryEmbedding(is_neox_style=False)
    metadata["rmsnorm_forward_method"] = repr(norm._forward_method)
    metadata["rotary_forward_method"] = repr(rope._forward_method)
    metadata["rotary_cuda_backend"] = repr(rope.apply_rotary_emb_vllm_flash_attn)
    metadata["rmsnorm_ir_priority"] = ir.ops.rms_norm.get_priority()
    (output / "environment.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata), flush=True)
    candidates = list(itertools.product((False, True), (1, 4, 8), (1, 4), (False, True)))
    totals = {str(config): {"norm": 0, "rope": 0} for config in candidates}
    cases = [(1, 77, 24), (2, 512, 24), (1, 4608, 24), (1, 4608, 48)]
    with torch.inference_mode(), (output / "cases.jsonl").open("w") as log:
        for seed, (batch, seq, heads), weight_kind in itertools.product((0, 3), cases, ("ones", "random")):
            torch.manual_seed(seed)
            qkv = torch.randn(batch, seq, heads * 128 * 3, device="cuda", dtype=torch.bfloat16)
            x = qkv.chunk(3, dim=-1)[0].unflatten(-1, (heads, 128))
            if weight_kind == "ones":
                norm.weight.fill_(1)
            else:
                norm.weight.uniform_(0.5, 1.5)
            angles = torch.randn(seq, 64, device="cuda")
            cos, sin = angles.cos().to(x.dtype), angles.sin().to(x.dtype)
            table = torch.cat((cos, sin), -1).unsqueeze(0).expand(batch, -1, -1).reshape(-1, 128).contiguous()
            expected_norm = norm(x)
            expected = rope(expected_norm, cos, sin)
            rows = x.view(batch * seq, heads, 128)
            # Compare the actual dispatch against both rounding hypotheses.
            xf = x.float()
            xn = xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + 1e-6)
            mirrors = {
                "double_round": difference((xn.to(x.dtype) * norm.weight).to(x.dtype), expected_norm),
                "single_round": difference((xn * norm.weight.float()).to(x.dtype), expected_norm),
            }
            case = {"seed": seed, "batch": batch, "seq": seq, "heads": heads, "weights": weight_kind}
            record = {
                "case": case, "mirrors_vs_actual_norm": mirrors, "candidates": [],
                "rmsnorm_ir_provider": ir.ops.rms_norm.dispatch(x, norm.weight, 1e-6, None).provider,
            }
            out = torch.empty(expected.shape, device=expected.device, dtype=expected.dtype)
            norm_out = torch.empty(expected_norm.shape, device=expected_norm.device, dtype=expected_norm.dtype)
            for round_before, hpp, warps, fma in candidates:
                config = (round_before, hpp, warps, fma)
                candidate[(batch * seq, triton.cdiv(heads, hpp))](
                    rows, norm.weight, table, out, norm_out,
                    rows.stride(0), rows.stride(1), heads, 128, hpp,
                    round_before, 1e-6, num_warps=warps, enable_fp_fusion=fma,
                )
                nd, rd = difference(norm_out, expected_norm), difference(out, expected)
                totals[str(config)]["norm"] += nd["bit_mismatches"]
                totals[str(config)]["rope"] += rd["bit_mismatches"]
                record["candidates"].append({"config": config, "norm": nd, "rope": rd})
            log.write(json.dumps(record) + "\n")
            log.flush()
            print(json.dumps({"case": case, "provider": record["rmsnorm_ir_provider"], "mirrors": mirrors, "best": sorted(record["candidates"], key=lambda r: (r["rope"]["bit_mismatches"], r["norm"]["bit_mismatches"]))[:3]}), flush=True)
    result = {
        "status": "EXPERIMENT_COMPLETE", "totals": totals,
        "zero_difference_candidates": [key for key, value in totals.items() if value["norm"] == value["rope"] == 0],
        "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
