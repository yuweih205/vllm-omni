"""Z-Image: fused QK-norm+RoPE vs the eager chain, with and without Ulysses SP.

Two questions, one run:
  1. SP off (world 1): does enabling the gate change anything it did not change
     before? (regression guard for the packer refactor)
  2. SP on (world 2, ulysses_degree=2): is the fused path still equal to the
     eager chain on every rank, and is the SP output still equal to the SP=1
     output?

The model's ``_sp_plan`` gathers the final layer, so each rank returns the full
image and the tensors are directly comparable.
"""

import json
import os
import sys

import torch

_MODEL_SEED = 42
_INPUT_SEED = 123
_TINY = dict(
    all_patch_size=(2,), all_f_patch_size=(1,), in_channels=16, dim=512, n_layers=2,
    n_refiner_layers=1, n_heads=4, n_kv_heads=4, cap_feat_dim=64,
)
_GATE = "VLLM_OMNI_FUSED_QK_NORM_ROPE_MIN_TOKENS"


def _log(m):
    print(f"[zsp] {m}", flush=True)


def _inputs(device, dtype, bsz=2):
    g = torch.Generator().manual_seed(_INPUT_SEED)
    x = [torch.randn(16, 1, 16, 16, generator=g).to(device=device, dtype=dtype) for _ in range(bsz)]
    cap = [torch.randn(32, 64, generator=g).to(device=device, dtype=dtype) for _ in range(bsz)]
    t = torch.full((bsz,), 0.5).to(device=device, dtype=dtype)
    return x, t, cap


def _worker(local_rank, world_size, sp_size, port, queue, mode="ulysses"):
    from vllm.config import VllmConfig, set_current_vllm_config

    from vllm_omni.diffusion.config import set_current_diffusion_config
    from vllm_omni.diffusion.data import (
        AttentionConfig, AttentionSpec, DiffusionParallelConfig, OmniDiffusionConfig,
    )
    from vllm_omni.diffusion.distributed import get_sp_plan_from_model
    from vllm_omni.diffusion.distributed.parallel_state import (
        destroy_distributed_env, get_sequence_parallel_world_size,
        init_distributed_environment, initialize_model_parallel,
    )
    from vllm_omni.diffusion.forward_context import set_forward_context
    from vllm_omni.platforms import current_omni_platform

    device = torch.device(f"{current_omni_platform.device_type}:{local_rank}")
    current_omni_platform.set_device(device)
    os.environ.update(RANK=str(local_rank), LOCAL_RANK=str(local_rank), WORLD_SIZE=str(world_size),
                      MASTER_ADDR="localhost", MASTER_PORT=str(port))
    ulysses = sp_size if mode == "ulysses" else 1
    ring = sp_size if mode == "ring" else 1
    init_distributed_environment()
    initialize_model_parallel(sequence_parallel_size=sp_size, ulysses_degree=ulysses, ring_degree=ring)
    assert get_sequence_parallel_world_size() == sp_size

    dtype = torch.bfloat16
    vllm_cfg_ctx = set_current_vllm_config(VllmConfig())
    vllm_cfg_ctx.__enter__()
    od_config = OmniDiffusionConfig(
        model="test", dtype=dtype,
        parallel_config=DiffusionParallelConfig(
            pipeline_parallel_size=1, data_parallel_size=1, tensor_parallel_size=1,
            sequence_parallel_size=sp_size, ulysses_degree=ulysses, ring_degree=ring, cfg_parallel_size=1,
        ),
        diffusion_attention_config=AttentionConfig(default=AttentionSpec(backend="TORCH_SDPA")),
    )
    with set_current_diffusion_config(od_config):
        from vllm_omni.diffusion.models.z_image.z_image_transformer import ZImageTransformer2DModel

        torch.manual_seed(_MODEL_SEED)
        with torch.device(device):
            model = ZImageTransformer2DModel(**_TINY).eval()
        for _, p in sorted(model.named_parameters()):
            torch.nn.init.normal_(p, mean=0.0, std=0.02, generator=torch.Generator(device=device).manual_seed(7))
        model = model.to(device=device, dtype=dtype)
        if sp_size > 1:
            from vllm_omni.diffusion.distributed import SequenceParallelConfig
            from vllm_omni.diffusion.hooks.sequence_parallel import apply_sequence_parallel

            plan = get_sp_plan_from_model(model)
            assert plan is not None, "z-image must expose _sp_plan"
            apply_sequence_parallel(model, SequenceParallelConfig(ulysses_degree=ulysses, ring_degree=ring), plan)
            _log(f"rank{local_rank}: SP hooks applied (sp={sp_size} mode={mode})")

    # Runtime truth gate: count real fused-op calls so a silent fallback cannot
    # masquerade as "fused == eager".
    from vllm_omni.diffusion.models.z_image import z_image_transformer as zt

    calls = {"n": 0}
    _real_fused = zt.fused_qk_norm_rope

    def _counting_fused(*a, **k):
        calls["n"] += 1
        return _real_fused(*a, **k)

    zt.fused_qk_norm_rope = _counting_fused

    x, t, cap = _inputs(device, dtype)
    outs = {}
    counts = {}
    for arm, gate in (("eager", "1000000000"), ("fused", "0")):
        os.environ[_GATE] = gate
        calls["n"] = 0
        with torch.no_grad(), set_current_diffusion_config(od_config), set_forward_context(
            omni_diffusion_config=od_config
        ):
            out = model(x, t, cap)[0]
        out = out[0] if isinstance(out, (list, tuple)) else out
        outs[arm] = out.float().cpu()
        counts[arm] = calls["n"]
        _log(f"rank{local_rank} sp={sp_size}/{mode} {arm}: shape={tuple(outs[arm].shape)} "
             f"norm={outs[arm].norm().item():.6f} fused_op_calls={counts[arm]}")
    assert counts["eager"] == 0, f"eager arm called the fused op {counts['eager']} times"
    assert counts["fused"] > 0, "GATE DID NOT ENGAGE: fused arm never called the fused op"
    _log(f"rank{local_rank} sp={sp_size}/{mode} TRUTH-GATE ok: eager={counts['eager']} fused={counts['fused']} calls")

    # Timing: A B B A over the same model and inputs, CUDA events, p50.
    import statistics

    def _run(gate):
        os.environ[_GATE] = gate
        with torch.no_grad(), set_current_diffusion_config(od_config), set_forward_context(
            omni_diffusion_config=od_config
        ):
            model(x, t, cap)

    for gate in ("1000000000", "0"):
        for _ in range(3):
            _run(gate)
    torch.accelerator.synchronize()
    samples = {"eager": [], "fused": []}
    for arm, gate in (("eager", "1000000000"), ("fused", "0"), ("fused", "0"), ("eager", "1000000000")):
        ev = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(10)]
        for a, b in ev:
            a.record()
            _run(gate)
            b.record()
        torch.accelerator.synchronize()
        samples[arm] += [a.elapsed_time(b) * 1000.0 for a, b in ev]
    timing = {arm: statistics.median(v) for arm, v in samples.items()}
    timing["saving_pct"] = 100.0 * (timing["eager"] - timing["fused"]) / timing["eager"]
    _log(f"rank{local_rank} sp={sp_size}/{mode} TIMING eager={timing['eager'] / 1000:.3f} ms "
         f"fused={timing['fused'] / 1000:.3f} ms saving={timing['saving_pct']:+.2f}%")
    queue.put((local_rank, outs["eager"], outs["fused"], counts["fused"], timing))
    vllm_cfg_ctx.__exit__(None, None, None)
    destroy_distributed_env()


def _rel(a, b):
    return ((a - b).double().norm() / b.double().norm()).item() if b.norm() else float("nan")


def main():
    ctx = torch.multiprocessing.get_context("spawn")
    mgr = ctx.Manager()
    # (sp_size, mode); sp=1 is the single-process baseline.
    configs = [(1, "ulysses")] + [
        (int(sp), mode)
        for spec in (sys.argv[1] if len(sys.argv) > 1 else "2:ulysses,4:ulysses,8:ulysses,2:ring").split(",")
        for sp, mode in [spec.split(":")]
    ]
    res = {}
    port = 29570
    for sp, mode in configs:
        key = f"{sp}{'' if mode == 'ulysses' else '-ring'}"
        q = mgr.Queue()
        port += 1
        try:
            torch.multiprocessing.spawn(_worker, args=(sp, sp, port, q, mode), nprocs=sp)
        except Exception as exc:  # a config the model refuses is a result, not a crash
            _log(f"sp={sp} mode={mode} FAILED: {type(exc).__name__}: {str(exc)[:300]}")
            continue
        res[key] = {r[0]: (r[1], r[2], r[3], r[4]) for r in [q.get() for _ in range(sp)]}
        _log(f"sp={sp} mode={mode} done, ranks={sorted(res[key])}")

    report = {}
    e1, f1, _, _ = res["1"][0]
    for key, ranks in res.items():
        for rank, (eager, fused, nfused, timing) in sorted(ranks.items()):
            report[f"sp{key}_rank{rank}"] = {
                "fused_vs_eager_rel_l2": _rel(fused, eager),
                "fused_vs_eager_bitwise": bool(torch.equal(fused, eager)),
                "fused_vs_eager_max_abs": (fused - eager).abs().max().item(),
                "fused_op_calls": nfused,
                "eager_us": timing["eager"], "fused_us": timing["fused"],
                "saving_pct": timing["saving_pct"],
                "eager_vs_sp1_eager_rel_l2": _rel(eager, e1),
                "fused_vs_sp1_fused_rel_l2": _rel(fused, f1),
            }
    out_path = os.environ.get("ZSP_REPORT", "")
    if out_path:
        with open(out_path, "w") as f:
            json.dump(report, f, indent=2)
        _log(f"report written to {out_path}")
    # One log record per row: the platform log store reorders lines, so a
    # pretty-printed blob cannot be attributed back to its key.
    for key in sorted(report):
        print("ROW " + key + " " + json.dumps(report[key], sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
