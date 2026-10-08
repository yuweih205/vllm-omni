# Recovered historical Z-Image eager case

These are retained September 17 measurements, recovered October 8 without rerunning CUDA. They measure a complete tiny-model eager forward, with equality checked only on the first returned image tensor. The harness used `torch.equal` after converting the BF16 output to FP32; no integer-view signed-zero check was recorded.

The base is public commit `ff6e906a25de1ca3122c0d0d5250fde5a0ddbc7b`. Every one of the 1,503 archived base files matches that commit. The eight-file source overlay comes from `6dc3f91353a2f879c5a0981788f148b29a252ced`, except that its environment-variable inventory removes `VLLM_OMNI_ABORT_TIMEOUT` and `VLLM_OMNI_BREEZE_GOLDEN_DIR` for old-base compatibility. The retained exact overlay and file hashes are in `source-overlay/` and `overlay-manifest.json`.

Prepare a fresh directory using the public base and the recovered overlay:

```bash
mkdir -p "$RUN_DIR"
git -C "$REPO" archive ff6e906a25de1ca3122c0d0d5250fde5a0ddbc7b vllm_omni pyproject.toml | tar -x -C "$RUN_DIR"
cp -R "$ARTIFACT_DIR/source-overlay/." "$RUN_DIR/"
cp "$ARTIFACT_DIR/zimage_sp_check.py" "$RUN_DIR/"
```

The original invocation was:

```bash
PYTHONPATH="$RUN_DIR" ZSP_REPORT="$RUN_DIR/report.json" "$PYTHON" "$RUN_DIR/zimage_sp_check.py" '2:ulysses,4:ulysses,8:ulysses,2:ring'
```

Runtime: H200 141 GB, Python 3.12.3, vLLM 0.28.0, Torch 2.13.0+cu130/CUDA 13.0, Triton 3.7.1, Diffusers 0.40.0. No pretrained checkpoint is used. No `torch.compile` or regional compile is enabled. Attention is explicitly `TORCH_SDPA`; TP/PP/DP/CFG degrees are all 1.

The exact tiny constructor, parameter initialization, input generation and gate settings are in the recovered script and `provenance.json`. The input batch is 2. Set the fusion gate from `1000000000` to `0`; the counted fused operator calls change from 0 to 4. Both the output comparison and timing use the same model and inputs.

Retained SP1 result: 6.647759914 ms to 6.167151928 ms, a 7.229623% saving; `torch.equal=True`, relative L2 and maximum absolute difference both 0. All 17 retained SP/rank records have equality and 4 fused calls. Equality is within the same SP configuration, not between different SP sizes. The SP configurations themselves differ from SP1 numerically. Timings use CUDA-event medians, three warmups per arm, and A B B A with ten calls per segment.

This is a constructed random-weight tiny-model case, not a pretrained end-to-end image-quality result or a new-head measurement.
