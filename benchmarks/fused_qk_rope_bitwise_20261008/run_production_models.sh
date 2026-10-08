#!/usr/bin/env bash
set -euo pipefail

task=/inspire/ssd/project/video-generation/public/huangyuwei/experiments/omni-bitwise-20261008
runtime=/inspire/ssd/project/video-generation/public/huangyuwei/experiments/vomni-fusion-i64-20260921-r1
test "$(readlink -f "$task")" = "$task"
cd "$task"
run="$task/model-production"
mkdir -p "$run/results" "$run/cache"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
export OMNI_BENCH_IMAGE=cuda:13.0.2-cudnn-devel-ubuntu22.04
unset VLLM_OMNI_FUSED_QK_NORM_ROPE_MIN_TOKENS VLLM_OMNI_FUSED_QK_NORM_ROPE_NUMERICS

profiles=(flux2 flux1 hunyuan15 ovis flux2 flux1 hunyuan15 ovis)
prs=(7560 7595 7596 7600 7560 7595 7596 7600)
commits=(032950ed9533ae3565046defb2bf4d2f859938f9 6b40588161000b4942abe6e6d062a80d42228831 cce0fdfd9a3d810be67cda07eff537ef11cb99c5 53b1d8da4cf07fc40f55613a381f7b401873bbea)
pids=()
names=()
for gpu in 0 1 2 3 4 5 6 7; do
    batch=$((gpu / 4 + 1))
    name="${profiles[$gpu]}_b$batch"
    names+=("$name")
    source_root="$task/model-production/sources/pr${prs[$gpu]}"
    cache="$run/cache/$name"
    mkdir -p "$cache" "$cache/tmp"
    (
        export CUDA_VISIBLE_DEVICES=$gpu
        export PYTHONPATH="$source_root:$task"
        export TMPDIR="$cache/tmp"
        export XDG_CACHE_HOME="$cache/xdg"
        export TORCHINDUCTOR_CACHE_DIR="$cache/inductor"
        export TRITON_CACHE_DIR="$cache/triton"
        export CUDA_CACHE_PATH="$cache/cuda"
        export HF_HOME="$cache/hf"
        export VLLM_CACHE_ROOT="$cache/vllm"
        export FLASHINFER_WORKSPACE_BASE="$cache/flashinfer"
        export MASTER_ADDR=127.0.0.1
        export MASTER_PORT=$((36560 + gpu))
        "$runtime/env-v029/bin/python" -u "$task/model_production_bitwise.py" \
            --profile "${profiles[$gpu]}" --batch-size "$batch" --seed 20261008 \
            --source "$source_root" --expected-commit "${commits[$((gpu % 4))]}" \
            --output "$run/results/$name.json"
    ) > "$run/results/$name.log" 2>&1 &
    pids+=("$!")
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
"$runtime/env-v029/bin/python" - "$run/results" <<'PY'
import json
import pathlib
import sys
root = pathlib.Path(sys.argv[1])
failed = False
for profile in ('flux2', 'flux1', 'hunyuan15', 'ovis'):
    for batch in (1, 2):
        path = root / f'{profile}_b{batch}.json'
        if not path.exists():
            print(path.name, 'MISSING')
            failed = True
            continue
        result = json.loads(path.read_text())
        print(path.name, result.get('status'))
        failed |= result.get('status') != 'COMPLETE_SHALLOW_MODEL_EAGER_BITWISE'
sys.exit(1 if failed else 0)
PY
exit "$failed"
