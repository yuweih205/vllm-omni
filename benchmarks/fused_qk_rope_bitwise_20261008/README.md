# Eager bitwise configurations and retained benchmark records

These artifacts supplement PRs #7560, #7594, #7595, #7596, #7597 and #7600. Existing PR performance tables retain their original revisions and measurement scope.

## Parameterized CUDA RMSNorm reference

`parameterized_operator.py` exposes the two numerical controls used by the isolated experiment:

| Parameter | Current fusion | CUDA RMSNorm reference mode |
| --- | --- | --- |
| `rms_norm_reduction` | `"triton"` | `"vllm_cuda_128"` |
| `enable_fp_fusion` | `True` | `False` |

The second reduction reproduces vLLM 0.29's CUDA RMSNorm accumulation order: eight explicit FMA accumulations per lane and the matching fixed 16-lane reduction. The normalized intermediate retains BF16 rounding. The tested launch parameters remain four heads per program, one warp and two stages.

This is an isolated parameterized variant, rather than a runtime option already exposed by the model PRs. It requires CUDA BF16 activations and weights, head dimension 128, full interleaved RoPE and the indicated reference provider. It rejects unsupported reference-mode settings.

The default numerical path was checked against the shared operator source by AST; the reference reduction was checked against the previously validated `candidate_cuda_r3.py` by AST. Their source hashes are respectively `34787d3065731491a2992989850b2fa51ff2ad147c86cc1737aaa83d6d65457a` and `2fefb2156cb8dbe45dc81157b10b27883ac052caccb289a392d018b222c8559d`.

## Newly retained operator measurements

H200, Python 3.12, vLLM 0.29.0, Torch 2.13.0+cu130, CUDA 13.0, Triton 3.7.1; BF16, head dimension 128, epsilon `1e-6`, seed `20261008`, world/TP/SP size 1. The CUDA-reference candidate has zero integer-view bit mismatches for all tested Q/K and routed V elements against the original eager CUDA RMSNorm/RoPE chain. The original and CUDA-reference candidate arms also repeat exactly.

vLLM 0.29's method named `RMSNorm.forward_native` calls its IR dispatcher; the experiment explicitly asserts the selected IR provider is `vllm_c` for the reference tensors.

Each duration is milliseconds per Q/K(/V) preparation call: median of six balanced-order samples, 20 warmup calls and 100 calls per sample, CUDA events around eager calls. Packed RoPE tables are reused, so packing cost is excluded. The eight processes ran on separate GPUs in the same node. These are operator results; full model or pipeline gains require their own measurements.

| PR / geometry `(B, first tokens, second tokens, heads)` | Path | Original | Current | CUDA reference mode | Saving vs original | Extra latency vs current |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| #7560 / `(1,512,4096,48)` | Joint | 0.597912 | 0.111704 | 0.122705 | 79.48% | 9.85% |
| #7560 / same | Single | 0.367308 | 0.075905 | 0.101037 | 72.49% | 33.11% |
| #7595 / `(1,512,4096,24)` | Joint | 0.317764 | 0.106983 | 0.107405 | 66.20% | 0.39% |
| #7595 / same | Single | 0.194527 | 0.066775 | 0.067396 | 65.35% | 0.93% |
| #7596 / `(1,2048,256,16)` | Joint, identity text rows | 0.246770 | 0.109039 | 0.110161 | 55.36% | 1.03% |
| #7600 / `(1,256,4096,24)` | Joint | 0.306889 | 0.106426 | 0.106892 | 65.17% | 0.44% |
| #7600 / same | Single | 0.185626 | 0.066534 | 0.067409 | 63.69% | 1.32% |

Raw records, sample times, GPU state, implementation hashes and mismatch counts are in `operator_results/`. These measurements use `candidate_cuda_r3.py`; its reference branch is identical to the selected branch in `parameterized_operator.py`. The parameterized wrapper/schema still requires GPU integration validation.

To reproduce these operator cases with the pinned runtime and a source checkout whose shared operator hash matches above:

```bash
export PYTHONPATH="$SOURCE:$ARTIFACTS"
export VLLM_BATCH_INVARIANT=0
python "$ARTIFACTS/benchmark_series.py" --profile hunyuan15 --output "$OUTPUT/hunyuan15.json"
```

Use `flux2`, `flux1`, `hunyuan15` or `ovis`; outputs must be fresh. The retained Inspire harness restricts its output root to `/inspire/ssd/project/video-generation/public/huangyuwei`. When reproducing on another host, replace that path check with the experiment's own output root; it does not enter numerical computation.

## Existing Qwen eager bitwise result

`recovered-qwen/` contains the original three results, the surviving historical script, exact source overlay and provenance. The original measurement is from 2026-09-17, using vLLM 0.28.0 and a 60-block random-init transformer. The baseline makes 120 per-stream fused calls and the optimized arm makes 60 joint calls. All three whole returned tensors passed the original `torch.equal` check.

| CLI settings, with `--layers 60 --txt 512` | Per-stream | Joint | Saving |
| --- | ---: | ---: | ---: |
| `--batch 1 --img 4096` | 164.278221 ms | 158.815163 ms | 3.325492% |
| `--batch 2 --img 4096` | 324.814789 ms | 314.622177 ms | 3.137976% |
| `--batch 2 --img 1024` | 105.247505 ms | 102.388176 ms | 2.716767% |

The current main's per-stream threshold is 2048. `qwen_joint_bench_current.py` therefore sets `VLLM_OMNI_FUSED_QK_NORM_ROPE_MIN_TOKENS=0` in both arms and disables only the joint table packer in the reference arm. It retains the model, seeds and inputs and asserts 120 per-stream versus 60 joint calls. This adapted eager reproduction script has been syntax checked; the historical figures above were retained without a new measurement. See `recovered-qwen/REPRODUCTION.md` for the pinned historical-source recipe.

## Existing Z-Image eager bitwise result

`recovered-zimage/` contains the original script, source overlay, base-source archive, 17 per-rank records and provenance. The original tiny transformer uses `dim=512`, `n_layers=2`, `n_refiner_layers=1`, `n_heads=n_kv_heads=4`, `cap_feat_dim=64`, head dimension 128, BF16 and TORCH_SDPA. Batch size is 2; each image input is `[16,1,16,16]`, each caption is `[32,64]`, and both timesteps are 0.5.

Model construction uses seed 42. Every sorted named parameter is independently initialized with a fresh CUDA generator seeded 7, `normal_(0,0.02)`. A CPU generator seeded 123 creates the inputs in the retained script's order.

Switching the token threshold from `1000000000` to `0` changes the fused-call count from 0 to 4. The first returned image tensor passes `torch.equal`, with relative L2 and maximum absolute error both zero. The retained single-GPU measurement is **6.647759914 → 6.167151928 ms, saving 7.229623%**. The recorded comparison covers the first output image of the full batch forward. The original PR's other performance figures remain retained at their original scope.

See `recovered-zimage/REPRODUCTION.md` for the exact historical-source/runtime recipe. Both recovered historical harnesses compare BF16 outputs after conversion to FP32 using `torch.equal`; integer-view signed-zero checks were not recorded in those historical runs.

## Four pending full shallow-model checks

`model_bitwise.py` compares original/current/reference-mode outputs from the same initialized model and inputs, checks raw output bytes and actual fused-call counts, includes model RoPE generation and per-forward packing, and measures eager ABBA latency. The four profiles are Flux.2, FLUX.1, HunyuanVideo-1.5 and Ovis-Image; Qwen and Z-Image use the retained evidence above.

Current model source revisions: #7560 `e702a5253e2f58476468ddaaba836c63bcd3850b`, #7595 `34276f858c50c36290b2139c6fd86fb6f52b0db7`, #7596 `4b25e592bf33ca683a70d1f09fea9dbc2c537875`, #7600 `96ec51b2b15cde3aa0e2938247bddc2432d3dfcf`. Models retain actual head widths/head dimension 128 and use two dual and two single blocks, or two Hunyuan blocks plus one text-refiner layer. Inputs use 64 text tokens and a 16×16 image/video grid, B1/B2, seed `20261008`, world/TP/SP size 1, eager execution.

The H200 eight-GPU job `hyw-omni-bitwise-model-1008-r3` is queued in MOVA2.0纯交付分区. This section records a prepared experiment and supplies no full-model performance claim yet.

`prepare_model_sources.py` archives the four pinned Git revisions and generates the exact source manifests required by the model harness. Set its repository path to your local checkout, with those revisions fetched, and run it from a fresh artifact directory. `run_models.sh` contains the submitted eight-worker profile map and full command. `source_manifests/` retains the staged Python-source hashes. For another host, replace only the script's repository/output/runtime paths; keep the source revisions, model/input settings and numerical controls fixed.


## Prepared production preset (not yet pushed to the six PRs)

The common production patch in `production_validation/shared-numerics.patch` adds `VLLM_OMNI_FUSED_QK_NORM_ROPE_NUMERICS=fast|vllm_cuda_128`. Unset/`fast` retains the original arithmetic; `vllm_cuda_128` selects the two reference kwargs above for existing model callers. It does not change other model RMSNorm provider priorities. For the model benchmark, the original reference explicitly selects `ir.ops.rms_norm.set_default(["vllm_c"])` after entering the forward context.

Actual pinned Torch custom-op schemas and fake single/joint output shapes were checked on the CPU preparation notebook; the 20 CPU parameter tests passed using actual imported source with no dependency stubs. The CUDA reference provider is not executed by these CPU checks. Changed-file pre-commit checks passed. The same six common patch files are identical in all six prepared branches; 15 pairwise merges and 30 ordered first-squash simulations are clean. These records are in `production_validation/`; the four-model GPU job remains the pending gate before pushing the production patch.
