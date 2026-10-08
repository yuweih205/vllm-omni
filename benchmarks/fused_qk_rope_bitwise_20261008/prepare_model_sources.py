"""Archive the four current PR sources for offline eager model checks."""
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile

REPO = Path('/Users/shushu/.codex-work/perf-pr-scout.dTQ3Au/vllm-omni')
TASK = Path(__file__).resolve().parent
HEADS = {
    7560: 'e702a5253e2f58476468ddaaba836c63bcd3850b',
    7595: '34276f858c50c36290b2139c6fd86fb6f52b0db7',
    7596: '4b25e592bf33ca683a70d1f09fea9dbc2c537875',
    7600: '96ec51b2b15cde3aa0e2938247bddc2432d3dfcf',
}

for pr, commit in HEADS.items():
    destination = TASK / 'model-r2' / 'sources' / f'pr{pr}'
    assert not destination.exists()
    destination.mkdir(parents=True)
    archive = subprocess.check_output(
        ['git', 'archive', '--format=tar', commit, 'vllm_omni', 'tests', 'pyproject.toml'],
        cwd=REPO,
    )
    with tarfile.open(fileobj=io.BytesIO(archive)) as tf:
        tf.extractall(destination, filter='data')
    files = {
        str(path.relative_to(destination)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(destination.rglob('*.py'))
    }
    manifest = {'pr': pr, 'commit': commit, 'files': files}
    (destination / 'source_info.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(pr, commit, len(files))
