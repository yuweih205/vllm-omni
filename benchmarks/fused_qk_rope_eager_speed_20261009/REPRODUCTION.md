# Eager bitwise performance follow-up (2026-10-09)

The `results/` records contain fresh measurements. Each record identifies the
tested source, implementation, geometry, software stack and GPU, retains raw
ABBA samples, and checks strict byte equality before and after timing.

## Source preparation

The exact numerical preset remains a prepared patch. The six public PR branches
have not been changed by this benchmark. For each PR, `sources/prNNNN/source_info.json`
identifies its public base and prepared source commit and hashes every imported
Python file. Apply `prepared-patches/prNNNN.patch` to that public base. The six
patches are identical and preserve the default `fast` mode. Use the corresponding
manifest as `source_info.json` in the source root; the runners validate file hashes
before importing. Reproduction needs the prepared tree's file contents, so the
unpublished prepared commit object is not required.

## Runtime and execution

H200 141 GB, CUDA 13.0, vLLM 0.29.0, Torch 2.13.0+cu130, Triton 3.7.1,
Diffusers 0.40.0. One GPU per process, BF16, world/TP/SP size 1. All measurements
use eager execution. No torch.compile or CUDA graph is requested. `run.sh` records
the exact personal paths, cache isolation, GPU assignments, ports and commands.
Adapt those paths to an equivalent existing environment and set offline cache
paths under your own writable directory.

## Z-Image: pretrained full request

`zimage_speed.py`: Tongyi-MAI/Z-Image-Turbo, checkpoint revision
`f332072aa78be7aecdf3ee76d5c247082da564a6`, 1024x1024, 8 and 28 inference steps,
guidance 0, seed 7, one fixed prompt recorded in the script and JSON. The prepared
source is `cbe0e7f03f39f54b13e46ba3084fc085cf89425e` over public base
`ffdb71041baeefe3242edcc4d36efbcab949fda5`.

Original: MIN_TOKENS=1000000000000 / NUMERICS=fast. Current fusion:
MIN_TOKENS=0 / NUMERICS=fast. Exact fusion: MIN_TOKENS=0 /
NUMERICS=vllm_cuda_128, resolving to CUDA-order reduction and disabled implicit
FP fusion. The operator receives ordinary model arguments; it is not substituted
or numerically patched by the harness.

Both the complete VAE float output and RGB uint8 image must have zero byte
mismatches for exact versus original. Each arm also repeats exactly. Runtime
counts verify 0 original fused calls and 34 x steps calls in both fusion arms.
After those checks, counters are removed. One complete warmup request per arm
precedes three ABBA rounds per optimized arm, giving six request samples per arm
in each comparison. CUDA synchronization brackets each request. Wall latency
includes text encoding, denoising, VAE and PIL image output; model loading,
correctness comparisons and output artifact writes are excluded. The separate
pipeline CUDA-event field ends before PIL postprocessing. After timing, original
and exact outputs must still match the retained original output bit for bit.

The MODEL_READY marker refers to the verified weights from the preceding
numerical-attribution experiment. To reproduce in another directory, obtain
that exact checkpoint and change the marker check together with the paths;
do not silently select a different model revision.

## FLUX.2 / FLUX.1 / HunyuanVideo1.5 / Ovis-Image

`model_production_bitwise.py`: random weights, full model width/head dimension
but shallow depth (two dual/joint blocks; FLUX/Ovis also two single blocks;
Hunyuan has one configured text refiner). These are transformer-forward timings.
They include model RoPE generation and table packing, and exclude text encoders,
samplers and VAE. Image/video grid 64x64 (4096 tokens), 512 configured text tokens,
batches 1 and 2, seed 20261008. The JSON lists the additional Hunyuan condition
streams. Its timestep is BF16, matching the BF16 time-embedding linear layers.

The same original/current/exact environment controls are used. The original
vLLM RMSNorm IR provider is pinned and checked as `vllm_c`. Norm/RoPE dispatch,
fused-call counts, state and input hashes are recorded. Full returned BF16
tensor storage bytes must match for exact versus original; every arm must repeat
exactly. Provider hooks and counter wrappers are removed before timing. Ten
warmup calls per arm precede three ABBA rounds, ten calls per sample, six samples
per arm. Timings are CUDA-event milliseconds per complete shallow forward.
Final parity and unchanged state/input hashes are checked after timing.

## Qwen-Image

`qwen_speed.py`: 60-layer full-width random-weight transformer, 24 heads,
head dimension 128, 512 text and 4096 image tokens, B1/B2. The baseline is the
existing per-stream fused preparation plus concatenation; the optimized path
uses joint fusion. Both use MIN_TOKENS=0 and NUMERICS=fast. Only the baseline
joint table packer is suppressed. Truth counters must be 120 per-stream calls
and 0 joint calls versus 0 per-stream and 60 joint calls. Constructor seed 0;
one CUDA generator seed 1 initializes parameters and then inputs using the
retained historical initialization sequence.

The complete returned BF16 storage bytes, including signed zero, must match.
Both arms repeat exactly. Counters are removed for timing: three warmup forwards
per arm, three ABBA rounds, ten forwards per sample, six CUDA-event samples per
arm. Parity is rechecked after timing. This is a random-weight transformer
measurement.

## Interpretation

Performance comparisons apply to their recorded scopes and supported eager
reference. The Z-Image measurements establish pretrained full-request parity
for one prompt/seed at two step counts. Other models' measurements establish
parity for their recorded random-weight forwards. They do not establish
universal input, model or runtime-version guarantees.

AI assistance: Codex prepared the isolated harnesses, ran the GPU tests,
checked byte equality and raw measurements, and published this evidence.
