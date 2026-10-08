# Recovered historical Qwen-Image eager case

These are retained September 17 measurements, recovered October 8 without rerunning CUDA. They measure the full 60-block random-weight transformer and compare its entire returned tensor using `torch.equal` after converting the BF16 output to FP32. No integer-view signed-zero check was recorded.

The base is public commit `ff6e906a25de1ca3122c0d0d5250fde5a0ddbc7b`. Every one of the 1,503 archived base files matches that commit. All 21 files in the recovered Qwen overlay match `f3afda8aa8b841e2d72a842dea534d63fc65555a` exactly. File hashes are in `overlay-manifest.json`; safe source files are in `source-overlay/`.

Prepare a fresh directory using the public base and the recovered overlay:

```bash
mkdir -p "$RUN_DIR"
git -C "$REPO" archive ff6e906a25de1ca3122c0d0d5250fde5a0ddbc7b vllm_omni pyproject.toml | tar -x -C "$RUN_DIR"
cp -R "$ARTIFACT_DIR/source-overlay/." "$RUN_DIR/"
cp "$ARTIFACT_DIR/qwen_joint_bench.py" "$RUN_DIR/"
```

The original eager invocations were:

```bash
PYTHONPATH="$RUN_DIR" "$PYTHON" "$RUN_DIR/qwen_joint_bench.py" --layers 60 --batch 1 --txt 512 --img 4096 --out "$RUN_DIR/b1_4096.json"
PYTHONPATH="$RUN_DIR" "$PYTHON" "$RUN_DIR/qwen_joint_bench.py" --layers 60 --batch 2 --txt 512 --img 4096 --out "$RUN_DIR/b2_4096.json"
PYTHONPATH="$RUN_DIR" "$PYTHON" "$RUN_DIR/qwen_joint_bench.py" --layers 60 --batch 2 --txt 512 --img 1024 --out "$RUN_DIR/b2_1024.json"
```

Do not pass `--compile`. The retained script includes that optional flag added after the eager measurements. The current saved script hash therefore identifies the recovered copy, not a hash recorded contemporaneously at the original eager run. The raw eager records do not contain a `compiled` field.

Runtime: H200 141 GB, Python 3.12.3, vLLM 0.28.0, Torch 2.13.0+cu130/CUDA 13.0, Triton 3.7.1, Diffusers 0.40.0. The constructor is `QwenImageTransformer2DModel(OmniDiffusionConfig(model="test", dtype=torch.bfloat16), num_layers=60)`, with all other model and attention defaults. No pretrained checkpoint is used.

Seed 0 initializes construction; then one CUDA generator with seed 1 sequentially fills every named parameter and subsequently generates the inputs. Its exact parameter rules and input sequence are preserved in the script; generating inputs with a separately reset seed-1 generator would produce different values. Set the joint gate from `1000000000` to `0`: counted launches change from 120 per-stream/0 joint to 0 per-stream/60 joint. At this historical revision, the large joint gate does not disable the per-stream fused baseline. Current main changed that gate behavior, so this command must use the pinned historical source.

| Batch / image tokens | Per-stream baseline | Joint | Saving |
| --- | ---: | ---: | ---: |
| 1 / 4096 | 164.278221 ms | 158.815163 ms | 3.325492% |
| 2 / 4096 | 324.814789 ms | 314.622177 ms | 3.137976% |
| 2 / 1024 | 105.247505 ms | 102.388176 ms | 2.716767% |

All three retained records report `torch.equal=True` and relative L2 exactly 0, with the launch counts above. CUDA-event p50 timing uses three warmups before each timed segment and A B B A, ten iterations per segment. Peak allocated memory recorded by the harness is 76.182684 GiB.

These are historical eager full-transformer results; the compiled arms are not equal, and no pretrained end-to-end or new-head result is inferred.
