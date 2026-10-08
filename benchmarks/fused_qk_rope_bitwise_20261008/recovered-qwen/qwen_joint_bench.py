"""Qwen-Image: the joint launch vs main's per-stream fused path.

main (#5931/#7513) fuses Q/K RMSNorm + RoPE per stream and then concatenates.
This measures what replacing those two launches and three cats with one joint
launch is worth, on the full 60-block transformer, and checks the outputs.
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

    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.distributed.parallel_state import init_distributed_environment, initialize_model_parallel

    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", str(29500 + os.getpid() % 1000))
    with set_current_vllm_config(VllmConfig()):
        init_distributed_environment(world_size=1, rank=0, local_rank=0, distributed_init_method="env://")
        initialize_model_parallel()
        from vllm_omni.diffusion.models.qwen_image import qwen_image_transformer as qt

        from vllm_omni.diffusion.data import OmniDiffusionConfig

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

        if args.compile:
            from vllm_omni.diffusion.compile import regionally_compile

            n_before = len(model.transformer_blocks)
            regionally_compile(model)
            _log(f"regional compile applied over {n_before} blocks")

        B, txt, img = args.batch, args.txt, args.img
        side = int(img**0.5)
        hidden = torch.randn(B, img, 64, device="cuda", dtype=torch.bfloat16, generator=g)
        encoder = torch.randn(B, txt, 3584, device="cuda", dtype=torch.bfloat16, generator=g)
        mask = torch.ones(B, txt, device="cuda", dtype=torch.bool)
        timestep = torch.full((B,), 0.5, device="cuda", dtype=torch.bfloat16)
        img_shapes = [[(1, side, side)]] * B

        from vllm_omni.diffusion.data import OmniDiffusionConfig as _ODC  # noqa: F401
        from vllm_omni.diffusion.forward_context import set_forward_context

        def fwd():
            with torch.no_grad(), set_forward_context(omni_diffusion_config=od_config):
                return model(
                    hidden_states=hidden, encoder_hidden_states=encoder,
                    encoder_hidden_states_mask=mask, timestep=timestep,
                    img_shapes=img_shapes, txt_seq_lens=[txt] * B, return_dict=False,
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

        outs, counts = {}, {}
        for arm, gate in (("per_stream", "1000000000"), ("joint", "0")):
            os.environ["VLLM_OMNI_FUSED_QK_NORM_ROPE_MIN_TOKENS"] = gate
            calls["joint"] = calls["single"] = 0
            outs[arm] = fwd().float().cpu()
            counts[arm] = dict(calls)
            _log(f"{arm}: joint_calls={calls['joint']} single_calls={calls['single']} "
                 f"norm={outs[arm].norm().item():.6f}")
        assert counts["per_stream"]["joint"] == 0, "gate-off arm used the joint op"
        assert counts["joint"]["joint"] == args.layers, f"joint arm: {counts['joint']['joint']} != {args.layers}"
        assert counts["joint"]["single"] == 0, "joint arm still used per-stream launches"
        _log(f"TRUTH-GATE ok {json.dumps(counts)}")

        diff = (outs["joint"] - outs["per_stream"])
        rel = (diff.double().norm() / outs["per_stream"].double().norm()).item()
        _log(f"joint vs per-stream: bitwise={torch.equal(outs['joint'], outs['per_stream'])} "
             f"rel_l2={rel:.3e} max_abs={diff.abs().max().item():.3e}")

        def timed(gate):
            os.environ["VLLM_OMNI_FUSED_QK_NORM_ROPE_MIN_TOKENS"] = gate
            for _ in range(args.warmup):
                fwd()
            torch.accelerator.synchronize()
            ev = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
                  for _ in range(args.iters)]
            for a, b in ev:
                a.record()
                fwd()
                b.record()
            torch.accelerator.synchronize()
            return [a.elapsed_time(b) for a, b in ev]

        samples = {"per_stream": [], "joint": []}
        for arm, gate in (("per_stream", "1000000000"), ("joint", "0"), ("joint", "0"), ("per_stream", "1000000000")):
            samples[arm] += timed(gate)
        t = {a: statistics.median(v) for a, v in samples.items()}
        t["saving_pct"] = 100.0 * (t["per_stream"] - t["joint"]) / t["per_stream"]
        rec = {"compiled": bool(args.compile), "layers": args.layers, "batch": B, "txt": txt, "img": img, "timing_ms": t,
               "joint_vs_per_stream_rel_l2": rel,
               "bitwise": bool(torch.equal(outs["joint"], outs["per_stream"])),
               "counts": counts, "peak_gib": torch.cuda.max_memory_allocated() / 2**30}
        print("ROW " + json.dumps(rec, sort_keys=True), flush=True)
        if args.out:
            with open(args.out, "w") as f:
                json.dump(rec, f, indent=2)


if __name__ == "__main__":
    main()
