# Eager bitwise speed results, 2026-10-09

Fresh H200 measurements, vLLM 0.29.0 / Torch 2.13.0+cu130 / Triton 3.7.1 / CUDA 13.0.
BF16, head dimension 128, world/TP/SP 1. All timings use eager execution.
The CUDA-order numerical preset is a prepared patch and has not been pushed to the six public PR branches.

## Pretrained Z-Image full request

Pinned Z-Image-Turbo checkpoint, 1024x1024, fixed prompt and seed 7, guidance 0.
Wall latency includes text encoding, denoising, VAE and PIL output; model loading is excluded.
Full VAE output and RGB storage bits match before and after timing. Every arm repeats exactly.
One complete warmup per arm; three ABBA rounds per comparison, six samples per arm.

| Steps | Original (s) | Exact preset (s) | Latency reduction | Speedup | Bitwise |
| --- | ---: | ---: | ---: | ---: | --- |
| 8 | 1.050633 | 1.002427 | 4.588% | 1.0481x | 0 byte mismatches |
| 28 | 3.338334 | 3.186369 | 4.552% | 1.0477x | 0 byte mismatches |

The current fast arm is also measured in [raw Z-Image results](results/zimage.json).
Its bitwise outcome and separate paired timings are retained without substituting it for the exact arm.

## Shallow full-width transformer forwards

Random weights; two joint/dual blocks, and two single blocks for FLUX/Ovis; Hunyuan text-refiner depth 1.
4096 image/video tokens, 512 configured text tokens; Hunyuan includes additional condition streams.
Model RoPE generation and table packing are included. Text encoder, sampler and VAE are excluded.
Ten warmups per arm, three ABBA rounds, ten calls per sample. Full returned BF16 storage bits match.
Provider hooks and counter wrappers are removed for timing. State/input hashes remain unchanged.

| PR / model | Batch | Original (ms) | Exact preset (ms) | Latency reduction | Bitwise |
| --- | ---: | ---: | ---: | ---: | --- |
| #7560 FLUX.2 | 1 | 38.450858 | 37.274188 | 3.060% | 0 byte mismatches |
| #7560 FLUX.2 | 2 | 76.559448 | 74.354437 | 2.880% | 0 byte mismatches |
| #7595 FLUX.1 | 1 | 12.256840 | 11.525142 | 5.970% | 0 byte mismatches |
| #7595 FLUX.1 | 2 | 23.859587 | 23.211646 | 2.716% | 0 byte mismatches |
| #7596 HunyuanVideo1.5 | 1 | 5.832320 | 5.540894 | 4.997% | 0 byte mismatches |
| #7596 HunyuanVideo1.5 | 2 | 9.858630 | 9.311595 | 5.549% | 0 byte mismatches |
| #7600 Ovis-Image | 1 | 15.091319 | 14.552750 | 3.569% | 0 byte mismatches |
| #7600 Ovis-Image | 2 | 29.961728 | 29.195374 | 2.558% | 0 byte mismatches |

## Qwen-Image full 60-layer transformer

Random weights, full width, 512 text + 4096 image tokens, B1/B2.
Baseline: existing per-stream fusion plus cats; optimized: joint fusion. Both use the fast preset and token gate 0.
60 joint launches replace 120 per-stream launches. Full BF16 storage bits, including signed zero, match.
Three warmups per arm, three ABBA rounds, ten forwards per sample; counter wrappers removed for timing.

| Batch | Per-stream (ms) | Joint (ms) | Latency reduction | Bitwise |
| --- | ---: | ---: | ---: | --- |
| 1 | 167.449280 | 161.765442 | 3.394% | 0 byte mismatches |
| 2 | 329.690540 | 319.401697 | 3.121% | 0 byte mismatches |

## Scope and records

Z-Image is pretrained full generation for one fixed prompt/seed at two step counts.
The other results apply to the recorded random-weight transformers; they do not establish full generated image/video parity.
No universal input, model or runtime-version guarantee is inferred from these measurements.
Performance and exactness use the same implementation in every row.

The initial Qwen r1 attempt used a dummy model identifier that triggered an offline model lookup.
It produced no timing data. The r2 retry uses model=None and an explicit synthetic transformer config.
The other models use r1 results. Original failed records and logs are retained in the task evidence.
The earlier Hunyuan harness timestep mismatch was fixed in this run before submission.

[Exact reproduction procedure](REPRODUCTION.md), [runner](run.sh),
[Z-Image harness](zimage_speed.py), [shallow model harness](model_production_bitwise.py), [Qwen harness](qwen_speed.py).

### Raw completed results

- [flux1_b1.json](results/flux1_b1.json), SHA256 `32533d23222d293d30e23962b908166ca935026b6034c2e1b4df6a46e01a70ad`
- [flux1_b2.json](results/flux1_b2.json), SHA256 `ae4d1ea17c6f8233e3866d29b4a33230643b6431f19530a34a8cbc7bfae956c3`
- [flux2_b1.json](results/flux2_b1.json), SHA256 `51bed45a90e3b13df068a5f3306dd390a270342edc26e9939e7fd35a4ad86ca3`
- [flux2_b2.json](results/flux2_b2.json), SHA256 `1f14fc7e18e9506b7017129221e3cbbef972d7fca21ed91d10322c7e8a02e147`
- [hunyuan15_b1.json](results/hunyuan15_b1.json), SHA256 `7f0b416678010feb056dc9e10b524c6770004b6410bc8c77883663d1c6ab95b7`
- [hunyuan15_b2.json](results/hunyuan15_b2.json), SHA256 `6582320b0603e6fa1cc70e81529b68800b74f12d99e5bbc3cb27b973dcd97126`
- [ovis_b1.json](results/ovis_b1.json), SHA256 `5e5bd81e952b105f55e90d808c0b75660a1f828777f8c0d9468dc164f9749ad3`
- [ovis_b2.json](results/ovis_b2.json), SHA256 `1fafbbe5d9e66e9b432f1ad61c949040afa83e2f4fbe260b44b90085d1b12c1a`
- [qwen_b1.json](results/qwen_b1.json), SHA256 `1fec472588274aea998eb087e6facb14a876d679d1354eef601db84f2b1256d9`
- [qwen_b2.json](results/qwen_b2.json), SHA256 `094af2d03bb1856156ed9080e0778b2000b224f11103b0eb62defb17002e469d`
- [zimage.json](results/zimage.json), SHA256 `4bceb936f241a556926c3faf1122f77c6dea320d75ae8dcab76de9e47bbeaff3`

AI assistance: Codex prepared the isolated harnesses, ran the GPU tests, checked strict byte equality, and retained the raw timing evidence.
