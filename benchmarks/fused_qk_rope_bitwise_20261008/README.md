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

## Completed production eager model measurements

The six B1/B2 cases for Flux.2, FLUX.1 and Ovis-Image use `model_production_bitwise.py` with the prepared production source and its actual environment preset. The normal model functions are wrapped only for counting. All six CUDA-reference cases match every returned tensor's raw bytes, repeat exactly, and match again after timing. The fast arm repeats but has nonzero byte differences against the original chain.

The comparison holds the initialized model, state and inputs fixed, includes model RoPE generation and per-forward table packing, and checks `0 joint + 0 single` calls in the original versus `2 joint + 2 single` in either fused arm. Recorded original providers are `vllm_c` RMSNorm and `vllm_flash_attn` RoPE, with FLASH_ATTN attention.

| Model | Batch | Original for bitwise pair (ms) | Current fast (ms) | Bitwise (ms) | Bitwise saving vs original | Bitwise extra latency vs current |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Flux.2 | B1 | 5.407557 | 4.783005 | 4.851823 | 10.28% | 1.44% |
| Flux.2 | B2 | 5.974448 | 5.634958 | 5.708642 | 4.45% | 1.31% |
| FLUX.1 | B1 | 4.902914 | 4.339882 | 4.393066 | 10.40% | 1.23% |
| FLUX.1 | B2 | 5.158642 | 4.580354 | 4.642147 | 10.01% | 1.35% |
| Ovis-Image | B1 | 4.860845 | 4.271122 | 4.331773 | 10.88% | 1.42% |
| Ovis-Image | B2 | 5.147882 | 4.538938 | 4.599460 | 10.65% | 1.33% |

These are full shallow random-init transformer forward times: two dual plus two single blocks, 64 text tokens and 256 image tokens (16×16 grid), actual head widths/head dimension 128, BF16, eager, world/TP/SP size 1, seed 20261008. Each pair uses CUDA-event ABBA, 20 warmups, 20 calls/sample, three rounds, medians of six samples per arm. Original/current and original/bitwise are separate pairs; the table's baseline and saving come from the bitwise pair. The extra-latency ratio describes the separately measured optimized medians. The measured bitwise medians cost about 1.2%–1.4% more than the fast medians, while retaining a 4.45%–10.88% saving versus the original chain. These figures supplement the original PR performance tables.

`model_results_r3/` contains six unchanged raw records and the validation summary. The measured script SHA256 is `1d4eaff2c5de50b0c09a098adfab736bd936dba1c01653f870bd7dbb4b3efed9`; production operator SHA256 is `0ebacf9d83d89a08419e92db9cdf5ffdbba3f37222a32a104db0b1bc29ad91b4`.

### PR #7560 settings and reproduction

The original PR performance results remain unchanged. These additional measurements cover a **full forward of a shallow random-init Flux.2 transformer** with two dual blocks and two single blocks, including model RoPE generation and per-forward table packing.

**Configuration that passed bitwise:** H200; Python 3.12.12, vLLM 0.29.0, Torch 2.13.0 / CUDA 13.0, Triton 3.7.1, Diffusers 0.40.0; eager (`enforce_eager=True`), BF16 activations/norm weights/RoPE, FLASH_ATTN, world/TP/SP size 1, seed `20261008`; B1 and B2, 64 text tokens, 256 image tokens (16×16 grid), 48 heads × 128 dimensions, image channels 128, text feature dimension 15360; guidance 3.5. Matrices use `normal_(0, 0.02)` and Q/K norm weights use `uniform_(0.5, 1.5)`. The baseline Q/K RMSNorm provider is explicitly pinned to `vllm_c` inside the forward context; recorded original RoPE dispatch is `vllm_flash_attn`, with full interleaved RoPE and epsilon `1e-6`.

With the prepared production patch, keep `VLLM_OMNI_FUSED_QK_NORM_ROPE_MIN_TOKENS=0` and change **`VLLM_OMNI_FUSED_QK_NORM_ROPE_NUMERICS=fast` → `vllm_cuda_128`**. This resolves **`rms_norm_reduction="triton"` → `"vllm_cuda_128"` and `enable_fp_fusion=True` → `False`**. Keep `VLLM_BATCH_INVARIANT=0`. The launch stays at four heads/program, one warp and two stages.

| Arm | `MIN_TOKENS` | `NUMERICS` |
| --- | ---: | --- |
| Original Q/K norm + RoPE chain | 1000000000000 | `fast` |
| Current fast fusion | 0 | `fast` |
| Eager bitwise fusion | 0 | `vllm_cuda_128` |

Both B1/B2 have **zero output-byte mismatches, zero maximum absolute error and finite outputs** versus the original chain. All three arms repeat exactly; the bitwise check also passes after timing. Observed fused calls per forward are original `0 joint + 0 single`, optimized `2 joint + 2 single`. The wrappers only count normal production calls and forward their arguments unchanged.

| Batch | Original for bitwise pair (ms) | Current fast fusion (ms) | Bitwise fusion (ms) | Bitwise latency saving vs original | Bitwise extra latency vs current |
| --- | ---: | ---: | ---: | ---: | ---: |
| B1 | 5.407557 | 4.783005 | 4.851823 | 10.28% | 1.44% |
| B2 | 5.974448 | 5.634958 | 5.708642 | 4.45% | 1.31% |

**The measured bitwise medians are slower than the current fast-fusion medians.** Each original/optimized pair uses eager CUDA-event ABBA ordering, 20 warmup calls, 20 calls/sample and three rounds (six samples per arm). The original column and saving use the original/bitwise pair; current fast fusion was measured in a separate original/current pair. Extra latency is the descriptive ratio of those two optimized medians. These timings apply to the shallow configuration above; the earlier PR performance tables retain their own model/input settings.

**Measured source:** prepared revision `032950ed9533ae3565046defb2bf4d2f859938f9`, reconstructed as current PR `e0afca0bc5fc458694bd28f2101cb5fb49c16fa7` plus the [common numerical-preset patch](./production_validation/shared-numerics.patch). **That patch has not yet been pushed to this PR**, so reproduction currently requires applying it. The production operator SHA256 is `0ebacf9d83d89a08419e92db9cdf5ffdbba3f37222a32a104db0b1bc29ad91b4`; the [harness](./model_production_bitwise.py) SHA256 is `1d4eaff2c5de50b0c09a098adfab736bd936dba1c01653f870bd7dbb4b3efed9`. It verifies the entire pinned Python-source manifest before executing.

Reproduce with the pinned runtime, a fresh checkout at the stated current PR revision, and downloaded artifacts:

```bash
git -C "$SOURCE" apply "$ARTIFACTS/production_validation/shared-numerics.patch"
cp "$ARTIFACTS/production_source_manifests/pr7560/source_info.json" "$SOURCE/source_info.json"
export PYTHONPATH="$SOURCE:$ARTIFACTS"
export VLLM_BATCH_INVARIANT=0
# Run once with --batch-size 1, then with 2, using a fresh output for each.
CUDA_VISIBLE_DEVICES=0 "$PYTHON" "$ARTIFACTS/model_production_bitwise.py" \
  --profile flux2 --source "$SOURCE" \
  --expected-commit 032950ed9533ae3565046defb2bf4d2f859938f9 \
  --seed 20261008 --batch-size 1 --text-tokens 64 --image-side 16 \
  --warmup 20 --iterations 20 --rounds 3 --output "$OUTPUT/flux2_b1.json"
```

`SOURCE` and `OUTPUT` use the documented personal Inspire output root; paths can be adapted as described in the artifact README. The harness sets all three arms' environment presets and reference-provider priority internally. [B1 raw results](./model_results_r3/flux2_b1.json), [B2 raw results](./model_results_r3/flux2_b2.json) include individual timing samples, dispatch providers, source hashes, state/input fingerprints and all bitwise checks.

### PR #7595 settings and reproduction

The original PR performance results remain unchanged. These additional measurements cover a **full forward of a shallow random-init FLUX.1 transformer** with two dual blocks and two single blocks, including model RoPE generation and per-forward table packing.

**Configuration that passed bitwise:** H200; Python 3.12.12, vLLM 0.29.0, Torch 2.13.0 / CUDA 13.0, Triton 3.7.1, Diffusers 0.40.0; eager (`enforce_eager=True`), BF16 activations/norm weights/RoPE, FLASH_ATTN, world/TP/SP size 1, seed `20261008`; B1 and B2, 64 text tokens, 256 image tokens (16×16 grid), 24 heads × 128 dimensions, image channels 64, text feature dimension 4096; pooled projection dimension 768; guidance 3.5. Matrices use `normal_(0, 0.02)` and Q/K norm weights use `uniform_(0.5, 1.5)`. The baseline Q/K RMSNorm provider is explicitly pinned to `vllm_c` inside the forward context; recorded original RoPE dispatch is `vllm_flash_attn`, with full interleaved RoPE and epsilon `1e-6`.

With the prepared production patch, keep `VLLM_OMNI_FUSED_QK_NORM_ROPE_MIN_TOKENS=0` and change **`VLLM_OMNI_FUSED_QK_NORM_ROPE_NUMERICS=fast` → `vllm_cuda_128`**. This resolves **`rms_norm_reduction="triton"` → `"vllm_cuda_128"` and `enable_fp_fusion=True` → `False`**. Keep `VLLM_BATCH_INVARIANT=0`. The launch stays at four heads/program, one warp and two stages.

| Arm | `MIN_TOKENS` | `NUMERICS` |
| --- | ---: | --- |
| Original Q/K norm + RoPE chain | 1000000000000 | `fast` |
| Current fast fusion | 0 | `fast` |
| Eager bitwise fusion | 0 | `vllm_cuda_128` |

Both B1/B2 have **zero output-byte mismatches, zero maximum absolute error and finite outputs** versus the original chain. All three arms repeat exactly; the bitwise check also passes after timing. Observed fused calls per forward are original `0 joint + 0 single`, optimized `2 joint + 2 single`. The wrappers only count normal production calls and forward their arguments unchanged.

| Batch | Original for bitwise pair (ms) | Current fast fusion (ms) | Bitwise fusion (ms) | Bitwise latency saving vs original | Bitwise extra latency vs current |
| --- | ---: | ---: | ---: | ---: | ---: |
| B1 | 4.902914 | 4.339882 | 4.393066 | 10.40% | 1.23% |
| B2 | 5.158642 | 4.580354 | 4.642147 | 10.01% | 1.35% |

**The measured bitwise medians are slower than the current fast-fusion medians.** Each original/optimized pair uses eager CUDA-event ABBA ordering, 20 warmup calls, 20 calls/sample and three rounds (six samples per arm). The original column and saving use the original/bitwise pair; current fast fusion was measured in a separate original/current pair. Extra latency is the descriptive ratio of those two optimized medians. These timings apply to the shallow configuration above; the earlier PR performance tables retain their own model/input settings.

**Measured source:** prepared revision `6b40588161000b4942abe6e6d062a80d42228831`, reconstructed as current PR `c97af7759b37c107560a0cf58b31fced5e0a1882` plus the [common numerical-preset patch](./production_validation/shared-numerics.patch). **That patch has not yet been pushed to this PR**, so reproduction currently requires applying it. The production operator SHA256 is `0ebacf9d83d89a08419e92db9cdf5ffdbba3f37222a32a104db0b1bc29ad91b4`; the [harness](./model_production_bitwise.py) SHA256 is `1d4eaff2c5de50b0c09a098adfab736bd936dba1c01653f870bd7dbb4b3efed9`. It verifies the entire pinned Python-source manifest before executing.

Reproduce with the pinned runtime, a fresh checkout at the stated current PR revision, and downloaded artifacts:

```bash
git -C "$SOURCE" apply "$ARTIFACTS/production_validation/shared-numerics.patch"
cp "$ARTIFACTS/production_source_manifests/pr7595/source_info.json" "$SOURCE/source_info.json"
export PYTHONPATH="$SOURCE:$ARTIFACTS"
export VLLM_BATCH_INVARIANT=0
# Run once with --batch-size 1, then with 2, using a fresh output for each.
CUDA_VISIBLE_DEVICES=0 "$PYTHON" "$ARTIFACTS/model_production_bitwise.py" \
  --profile flux1 --source "$SOURCE" \
  --expected-commit 6b40588161000b4942abe6e6d062a80d42228831 \
  --seed 20261008 --batch-size 1 --text-tokens 64 --image-side 16 \
  --warmup 20 --iterations 20 --rounds 3 --output "$OUTPUT/flux1_b1.json"
```

`SOURCE` and `OUTPUT` use the documented personal Inspire output root; paths can be adapted as described in the artifact README. The harness sets all three arms' environment presets and reference-provider priority internally. [B1 raw results](./model_results_r3/flux1_b1.json), [B2 raw results](./model_results_r3/flux1_b2.json) include individual timing samples, dispatch providers, source hashes, state/input fingerprints and all bitwise checks.

### PR #7600 settings and reproduction

The original PR performance results remain unchanged. These additional measurements cover a **full forward of a shallow random-init Ovis-Image transformer** with two dual blocks and two single blocks, including model RoPE generation and per-forward table packing.

**Configuration that passed bitwise:** H200; Python 3.12.12, vLLM 0.29.0, Torch 2.13.0 / CUDA 13.0, Triton 3.7.1, Diffusers 0.40.0; eager (`enforce_eager=True`), BF16 activations/norm weights/RoPE, FLASH_ATTN, world/TP/SP size 1, seed `20261008`; B1 and B2, 64 text tokens, 256 image tokens (16×16 grid), 24 heads × 128 dimensions, image channels 64, text feature dimension 2048. Matrices use `normal_(0, 0.02)` and Q/K norm weights use `uniform_(0.5, 1.5)`. The baseline Q/K RMSNorm provider is explicitly pinned to `vllm_c` inside the forward context; recorded original RoPE dispatch is `vllm_flash_attn`, with full interleaved RoPE and epsilon `1e-6`.

With the prepared production patch, keep `VLLM_OMNI_FUSED_QK_NORM_ROPE_MIN_TOKENS=0` and change **`VLLM_OMNI_FUSED_QK_NORM_ROPE_NUMERICS=fast` → `vllm_cuda_128`**. This resolves **`rms_norm_reduction="triton"` → `"vllm_cuda_128"` and `enable_fp_fusion=True` → `False`**. Keep `VLLM_BATCH_INVARIANT=0`. The launch stays at four heads/program, one warp and two stages.

| Arm | `MIN_TOKENS` | `NUMERICS` |
| --- | ---: | --- |
| Original Q/K norm + RoPE chain | 1000000000000 | `fast` |
| Current fast fusion | 0 | `fast` |
| Eager bitwise fusion | 0 | `vllm_cuda_128` |

Both B1/B2 have **zero output-byte mismatches, zero maximum absolute error and finite outputs** versus the original chain. All three arms repeat exactly; the bitwise check also passes after timing. Observed fused calls per forward are original `0 joint + 0 single`, optimized `2 joint + 2 single`. The wrappers only count normal production calls and forward their arguments unchanged.

| Batch | Original for bitwise pair (ms) | Current fast fusion (ms) | Bitwise fusion (ms) | Bitwise latency saving vs original | Bitwise extra latency vs current |
| --- | ---: | ---: | ---: | ---: | ---: |
| B1 | 4.860845 | 4.271122 | 4.331773 | 10.88% | 1.42% |
| B2 | 5.147882 | 4.538938 | 4.599460 | 10.65% | 1.33% |

**The measured bitwise medians are slower than the current fast-fusion medians.** Each original/optimized pair uses eager CUDA-event ABBA ordering, 20 warmup calls, 20 calls/sample and three rounds (six samples per arm). The original column and saving use the original/bitwise pair; current fast fusion was measured in a separate original/current pair. Extra latency is the descriptive ratio of those two optimized medians. These timings apply to the shallow configuration above; the earlier PR performance tables retain their own model/input settings.

**Measured source:** prepared revision `53b1d8da4cf07fc40f55613a381f7b401873bbea`, reconstructed as current PR `aba7d60a30747bde93894080543a9f6eeb0c2959` plus the [common numerical-preset patch](./production_validation/shared-numerics.patch). **That patch has not yet been pushed to this PR**, so reproduction currently requires applying it. The production operator SHA256 is `0ebacf9d83d89a08419e92db9cdf5ffdbba3f37222a32a104db0b1bc29ad91b4`; the [harness](./model_production_bitwise.py) SHA256 is `1d4eaff2c5de50b0c09a098adfab736bd936dba1c01653f870bd7dbb4b3efed9`. It verifies the entire pinned Python-source manifest before executing.

Reproduce with the pinned runtime, a fresh checkout at the stated current PR revision, and downloaded artifacts:

```bash
git -C "$SOURCE" apply "$ARTIFACTS/production_validation/shared-numerics.patch"
cp "$ARTIFACTS/production_source_manifests/pr7600/source_info.json" "$SOURCE/source_info.json"
export PYTHONPATH="$SOURCE:$ARTIFACTS"
export VLLM_BATCH_INVARIANT=0
# Run once with --batch-size 1, then with 2, using a fresh output for each.
CUDA_VISIBLE_DEVICES=0 "$PYTHON" "$ARTIFACTS/model_production_bitwise.py" \
  --profile ovis --source "$SOURCE" \
  --expected-commit 53b1d8da4cf07fc40f55613a381f7b401873bbea \
  --seed 20261008 --batch-size 1 --text-tokens 64 --image-side 16 \
  --warmup 20 --iterations 20 --rounds 3 --output "$OUTPUT/ovis_b1.json"
```

`SOURCE` and `OUTPUT` use the documented personal Inspire output root; paths can be adapted as described in the artifact README. The harness sets all three arms' environment presets and reference-provider priority internally. [B1 raw results](./model_results_r3/ovis_b1.json), [B2 raw results](./model_results_r3/ovis_b2.json) include individual timing samples, dispatch providers, source hashes, state/input fingerprints and all bitwise checks.


## Prepared production preset (not yet pushed to the six PRs)

The common production patch in `production_validation/shared-numerics.patch` adds `VLLM_OMNI_FUSED_QK_NORM_ROPE_NUMERICS=fast|vllm_cuda_128`. Unset/`fast` retains the original arithmetic; `vllm_cuda_128` selects the two reference kwargs above for existing model callers. It does not change other model RMSNorm provider priorities. For the model benchmark, the original reference explicitly selects `ir.ops.rms_norm.set_default(["vllm_c"])` after entering the forward context.

Actual pinned Torch custom-op schemas and fake single/joint output shapes were checked on the CPU preparation notebook; the 20 CPU parameter tests passed using actual imported source with no dependency stubs. The CUDA reference provider is not executed by these CPU checks. Changed-file pre-commit checks passed. The same six common patch files are identical in all six prepared branches; 15 pairwise merges and 30 ordered first-squash simulations are clean. These records are in `production_validation/`; The three documented model profiles now have six validated CUDA runs. The common patch is still awaiting push to the six PRs.
