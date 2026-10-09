#!/usr/bin/env bash
set -euo pipefail
task=/inspire/ssd/project/video-generation/public/huangyuwei/experiments/eager-bitwise-speed-20261008
runtime=/inspire/ssd/project/video-generation/public/huangyuwei/experiments/vomni-fusion-i64-20260921-r1/env-v029/bin/python
test "$(readlink -f "$task")" = "$task"
cd "$task"
sha256sum --quiet --check SHA256SUMS
mkdir -p "$task/results" "$task/cache"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false VLLM_LOGGING_LEVEL=WARNING
export OMNI_BENCH_IMAGE=cuda:13.0.2-cudnn-devel-ubuntu22.04 MASTER_ADDR=127.0.0.1
unset VLLM_OMNI_FUSED_QK_NORM_ROPE_MIN_TOKENS VLLM_OMNI_FUSED_QK_NORM_ROPE_NUMERICS
function configure_worker() {
    local name=$1 gpu=$2 pr=$3 port=$4
    export CUDA_VISIBLE_DEVICES=$gpu MASTER_PORT=$port
    export PYTHONPATH="$task/sources/pr$pr:$task"
    local cache="$task/cache/$name"
    mkdir -p "$cache/tmp"
    export TMPDIR="$cache/tmp" XDG_CACHE_HOME="$cache/xdg" TRITON_CACHE_DIR="$cache/triton"
    export TORCHINDUCTOR_CACHE_DIR="$cache/inductor" CUDA_CACHE_PATH="$cache/cuda"
    export HF_HOME="$cache/hf" VLLM_CACHE_ROOT="$cache/vllm" FLASHINFER_WORKSPACE_BASE="$cache/flashinfer"
}
pids=()
names=()
(
    configure_worker zimage 0 7597 37897
    "$runtime" -u "$task/zimage_speed.py" --source "$task/sources/pr7597" \
        --expected-commit cbe0e7f03f39f54b13e46ba3084fc085cf89425e \
        --model /inspire/ssd/project/video-generation/public/huangyuwei/experiments/zimage-numerics-20261008-r1/models/Z-Image-Turbo \
        --output "$task/results/zimage.json"
) > "$task/results/zimage.log" 2>&1 &
pids+=("$!"); names+=(zimage)
for batch in 1 2; do
    (
        configure_worker "qwen_b$batch" "$batch" 7594 "$((37940+batch))"
        "$runtime" -u "$task/qwen_speed.py" --source "$task/sources/pr7594" \
            --expected-commit 8b57eb5986c93f6562fd730e3552f2c04e418b47 \
            --batch "$batch" --output "$task/results/qwen_b$batch.json"
    ) > "$task/results/qwen_b$batch.log" 2>&1 &
    pids+=("$!"); names+=("qwen_b$batch")
done
profiles=(flux2 flux1 hunyuan15 ovis)
prs=(7560 7595 7596 7600)
commits=(032950ed9533ae3565046defb2bf4d2f859938f9 6b40588161000b4942abe6e6d062a80d42228831 cce0fdfd9a3d810be67cda07eff537ef11cb99c5 53b1d8da4cf07fc40f55613a381f7b401873bbea)
for i in 0 1 2 3; do
    (
        for batch in 1 2; do
            name="${profiles[$i]}_b$batch"
            configure_worker "$name" "$((i+3))" "${prs[$i]}" "$((38000+i*10+batch))"
            "$runtime" -u "$task/model_production_bitwise.py" --profile "${profiles[$i]}" \
                --batch-size "$batch" --text-tokens 512 --image-side 64 --warmup 10 --iterations 10 --rounds 3 \
                --source "$task/sources/pr${prs[$i]}" --expected-commit "${commits[$i]}" \
                --output "$task/results/$name.json" > "$task/results/$name.log" 2>&1
        done
    ) &
    pids+=("$!"); names+=("${profiles[$i]}")
done
failed=0
for i in "${!pids[@]}"; do
    if wait "${pids[$i]}"; then
        echo "${names[$i]}: complete"
    else
        echo "${names[$i]}: failed" >&2
        failed=1
    fi
done
"$runtime" - "$task/results" <<'PY'
import json
from pathlib import Path
import sys
root=Path(sys.argv[1])
names=['zimage','qwen_b1','qwen_b2']+[f'{profile}_b{batch}' for profile in ('flux2','flux1','hunyuan15','ovis') for batch in (1,2)]
failed=False
for name in names:
    path=root/f'{name}.json'
    status=json.loads(path.read_text()).get('status') if path.exists() else 'MISSING'
    print(name,status)
    failed |= not status.startswith('COMPLETE')
if not failed:
    (root/'COMPLETE').write_text('COMPLETE\n')
sys.exit(1 if failed else 0)
PY
exit "$failed"
