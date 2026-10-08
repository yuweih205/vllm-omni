#!/usr/bin/env bash
set -euo pipefail

task=/inspire/ssd/project/video-generation/public/huangyuwei/experiments/omni-bitwise-20261008
runtime=/inspire/ssd/project/video-generation/public/huangyuwei/experiments/vomni-fusion-i64-20260921-r1
test "$(readlink -f "$task")" = "$task"
cd "$task"
export PYTHONDONTWRITEBYTECODE=1

verify_source() {
    "$runtime/env-v029/bin/python" - "$1" "$2" <<'PY'
import hashlib
import json
import pathlib
import sys

source = pathlib.Path(sys.argv[1]).resolve()
manifest_path = pathlib.Path(sys.argv[2]).resolve()
task = pathlib.Path('/inspire/ssd/project/video-generation/public/huangyuwei/experiments/omni-bitwise-20261008').resolve()
assert source.is_relative_to(task), source
assert manifest_path.is_relative_to(task), manifest_path
expected = json.loads(manifest_path.read_text())
actual = {
    path.relative_to(source).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
    for path in sorted(source.rglob('*.py'))
}
assert actual == expected['files'], {
    'missing': sorted(set(expected['files']) - set(actual)),
    'extra': sorted(set(actual) - set(expected['files'])),
    'changed': sorted(name for name in actual.keys() & expected['files'].keys() if actual[name] != expected['files'][name]),
}
print(expected['pr'], expected['commit'], len(actual), 'source verified')
PY
}

common=(
    docs/configuration/environment_variables.md
    tests/diffusion/layers/test_fused_qk_norm_rope.py
    tests/diffusion/layers/test_fused_qk_norm_rope_numerics.py
    vllm_omni/config/environment_variable_inventory.py
    vllm_omni/diffusion/envs.py
    vllm_omni/diffusion/layers/fused_qk_norm_rope.py
)
mkdir -p "$task/model-production/sources"
for pr in 7560 7595 7596 7600; do
    source_root="$task/model-production/sources/pr$pr"
    manifest="$task/model-production/manifests/pr$pr/source_info.json"
    if test -e "$source_root"; then
        verify_source "$source_root" "$manifest"
        continue
    fi
    stage="$source_root.staging"
    test ! -e "$stage"
    cp -a "$task/model-r2/sources/pr$pr" "$stage"
    for path in "${common[@]}"; do
        mkdir -p "$stage/$(dirname "$path")"
        cp "$task/model-production/shared/$path" "$stage/$path"
    done
    cp "$manifest" "$stage/source_info.json"
    verify_source "$stage" "$manifest"
    mv -T "$stage" "$source_root"
done

"$runtime/env-v029/bin/python" - "$task/model-production" <<'PY'
import hashlib
import json
import pathlib
import sys

run = pathlib.Path(sys.argv[1])
info = json.loads((run / 'stage-info.json').read_text())
for profile in info['profiles']:
    root = run / 'sources' / f"pr{profile['pr']}"
    manifest = root / 'source_info.json'
    assert hashlib.sha256(manifest.read_bytes()).hexdigest() == profile['manifest_sha256']
    assert json.loads(manifest.read_text())['commit'] == profile['commit']
    for name, expected in info['shared'].items():
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == expected, (profile['pr'], name)
print('All four production snapshots and all six shared files verified.')
PY
