import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import os

ALLOWED = Path('/inspire/ssd/project/video-generation/public/huangyuwei')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_source(source, expected_commit):
    source = source.resolve()
    assert source.is_relative_to(ALLOWED)
    info = json.loads((source / 'source_info.json').read_text())
    assert info['commit'] == expected_commit
    for relative, digest in info['files'].items():
        target = (source / relative).resolve()
        assert target.is_relative_to(source) and sha(target) == digest, relative
    return {'commit': info['commit'], 'public_base': info['public_base'],
            'manifest_sha256': sha(source / 'source_info.json'),
            'verified_files': len(info['files']),
            'operator_sha256': info['files']['vllm_omni/diffusion/layers/fused_qk_norm_rope.py']}


def compare(a, b):
    import torch
    assert a.shape == b.shape and a.dtype == b.dtype
    a, b = a.detach().cpu().contiguous(), b.detach().cpu().contiguous()
    byte_mismatches = int((a.view(torch.uint8) != b.view(torch.uint8)).sum())
    finite = bool(torch.isfinite(a).all() and torch.isfinite(b).all())
    return {'bitwise': byte_mismatches == 0 and finite, 'byte_mismatches': byte_mismatches,
            'finite': finite, 'shape': list(a.shape), 'dtype': str(a.dtype),
            'max_abs': float((a.float() - b.float()).abs().max())}


def fingerprint(items):
    import torch
    digest = hashlib.sha256()
    for name, value in items:
        digest.update(name.encode())
        digest.update(str((value.shape, value.dtype)).encode())
        digest.update(value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def gpu_state():
    physical = os.environ.get('CUDA_VISIBLE_DEVICES', '0').split(',')[0]
    p = subprocess.run(['nvidia-smi', '-i', physical,
         '--query-gpu=name,uuid,driver_version,pstate,clocks.sm,clocks.mem,power.draw,temperature.gpu,utilization.gpu',
         '--format=csv,noheader'], text=True, capture_output=True)
    return {'physical_device': physical, 'gpu': p.stdout.strip(), 'stderr': p.stderr.strip()}


def summary(samples, baseline, optimized):
    medians = {arm: statistics.median(values) for arm, values in samples.items()}
    return {'samples': samples, 'median': medians,
            'speedup': medians[baseline] / medians[optimized],
            'latency_saving_pct': 100 * (1 - medians[optimized] / medians[baseline])}
