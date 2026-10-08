# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Current-source eager reproduction of the existing Qwen bitwise case.

Adapted from qwen_joint_bench.py (SHA256
29b91501138cc20a64843cd1f9fb253562b1407de14114095683c5594ea6ab63),
which remains the unmodified script for the historical measurement. This
adaptation has not been executed and provides no new measured result.

Both arms set MIN_TOKENS=0 so the current per-stream threshold of 2048 does
not switch the reference to unfused RMSNorm/BF16 RoPE. The per-stream arm
suppresses only the joint table; the joint arm uses the real table packer.
The model, seeds, parameters, inputs and eager timing procedure are unchanged.
"""

import argparse
import json
import os
import statistics
import time

import torch


def _log(m):
    print(f"[qjoint {time.strftime('%H:%M:%S')}] {m}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, default=60)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--txt", type=int, default=512)
    ap.add_argument("--img", type=int, default=4096)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--out", default="")
    ap.add_argument("--compile", action="store_true")
    args = ap.parse_args()
    if args.compile:
        ap.error("This reproduction is eager only; --compile is prohibited.")

    from tests.model_executor.helpers import bootstrap_vllm_layer_custom_op_modules

    bootstrap_vllm_layer_custom_op_modules()

    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.distributed.parallel_state import init_distributed_environment, initialize_model_parallel

    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", str(29500 + os.getpid() % 1000))
    with set_current_vllm_config(VllmConfig()):
        init_distributed_environment(world_size=1, rank=0, local_rank=0, distributed_init_method="env://")
        initialize_model_parallel()
        from vllm_omni.diffusion.data import OmniDiffusionConfig
        from vllm_omni.diffusion.models.qwen_image import qwen_image_transformer as qt

        od_config = OmniDiffusionConfig(model="test", dtype=torch.bfloat16)
        torch.manual_seed(0)
        with torch.device("cuda"):
            model = qt.QwenImageTransformer2DModel(od_config, num_layers=args.layers)
        model = model.to(torch.bfloat16).eval()
        g = torch.Generator(device="cuda").manual_seed(1)
        with torch.no_grad():
            for name, p in model.named_parameters():
                if p.dim() == 1 and "norm" in name and "weight" in name:
                    p.copy_(1.0 + 0.05 * torch.randn(p.shape, device="cuda", generator=g))
                elif p.dim() == 1:
                    p.copy_(0.01 * torch.randn(p.shape, device="cuda", generator=g))
                else:
                    p.copy_((0.02 * torch.randn(p.shape, device="cuda", dtype=torch.float32, generator=g)).to(p.dtype))

        B, txt, img = args.batch, args.txt, args.img
        side = int(img**0.5)
        hidden = torch.randn(B, img, 64, device="cuda", dtype=torch.bfloat16, generator=g)
        encoder = torch.randn(B, txt, 3584, device="cuda", dtype=torch.bfloat16, generator=g)
        mask = torch.ones(B, txt, device="cuda", dtype=torch.bool)
        timestep = torch.full((B,), 0.5, device="cuda", dtype=torch.bfloat16)
        img_shapes = [[(1, side, side)]] * B

        from vllm_omni.diffusion.forward_context import set_forward_context

        def fwd():
            with torch.no_grad(), set_forward_context(omni_diffusion_config=od_config):
                return model(
                    hidden_states=hidden,
                    encoder_hidden_states=encoder,
                    encoder_hidden_states_mask=mask,
                    timestep=timestep,
                    img_shapes=img_shapes,
                    txt_seq_lens=[txt] * B,
                    return_dict=False,
                )[0]

        # Truth gate: count joint-op calls.
        calls = {"joint": 0, "single": 0}
        real_joint = qt.fused_joint_qkv_norm_rope
        real_single = qt.fused_qk_norm_rope

        def cj(*a, **k):
            calls["joint"] += 1
            return real_joint(*a, **k)

        def cs(*a, **k):
            calls["single"] += 1
            return real_single(*a, **k)

        qt.fused_joint_qkv_norm_rope = cj
        qt.fused_qk_norm_rope = cs
        real_packer = qt._packed_qk_norm_rope_table

        def per_stream_packer(*_a, **_k):
            return None

        def select_arm(arm):
            assert arm in ("per_stream", "joint")
            os.environ["VLLM_OMNI_FUSED_QK_NORM_ROPE_MIN_TOKENS"] = "0"
            qt._packed_qk_norm_rope_table = real_packer if arm == "joint" else per_stream_packer

        outs, counts = {}, {}
        for arm in ("per_stream", "joint"):
            select_arm(arm)
            calls["joint"] = calls["single"] = 0
            outs[arm] = fwd().float().cpu()
            counts[arm] = dict(calls)
            _log(
                f"{arm}: joint_calls={calls['joint']} single_calls={calls['single']} norm={outs[arm].norm().item():.6f}"
            )
        assert counts["per_stream"]["joint"] == 0, "per-stream arm used the joint op"
        assert counts["per_stream"]["single"] == 2 * args.layers, (
            f"per-stream arm: {counts['per_stream']['single']} != {2 * args.layers}"
        )
        assert counts["joint"]["joint"] == args.layers, f"joint arm: {counts['joint']['joint']} != {args.layers}"
        assert counts["joint"]["single"] == 0, "joint arm still used per-stream launches"
        _log(f"TRUTH-GATE ok {json.dumps(counts)}")

        diff = outs["joint"] - outs["per_stream"]
        rel = (diff.double().norm() / outs["per_stream"].double().norm()).item()
        _log(
            f"joint vs per-stream: bitwise={torch.equal(outs['joint'], outs['per_stream'])} "
            f"rel_l2={rel:.3e} max_abs={diff.abs().max().item():.3e}"
        )

        def timed(arm):
            select_arm(arm)
            for _ in range(args.warmup):
                fwd()
            torch.accelerator.synchronize()
            ev = [
                (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(args.iters)
            ]
            for a, b in ev:
                a.record()
                fwd()
                b.record()
            torch.accelerator.synchronize()
            return [a.elapsed_time(b) for a, b in ev]

        samples = {"per_stream": [], "joint": []}
        for arm in ("per_stream", "joint", "joint", "per_stream"):
            samples[arm] += timed(arm)
        t = {a: statistics.median(v) for a, v in samples.items()}
        t["saving_pct"] = 100.0 * (t["per_stream"] - t["joint"]) / t["per_stream"]
        rec = {
            "compiled": False,
            "min_tokens_both_arms": 0,
            "reference_joint_table": "suppressed",
            "layers": args.layers,
            "batch": B,
            "txt": txt,
            "img": img,
            "timing_ms": t,
            "joint_vs_per_stream_rel_l2": rel,
            "bitwise": bool(torch.equal(outs["joint"], outs["per_stream"])),
            "counts": counts,
            "peak_gib": torch.accelerator.max_memory_allocated() / 2**30,
        }
        print("ROW " + json.dumps(rec, sort_keys=True), flush=True)
        if args.out:
            with open(args.out, "w") as f:
                json.dump(rec, f, indent=2)
        qt._packed_qk_norm_rope_table = real_packer
        qt.fused_joint_qkv_norm_rope = real_joint
        qt.fused_qk_norm_rope = real_single


if __name__ == "__main__":
    main()
