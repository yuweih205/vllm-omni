"""Pretrained Z-Image eager parity plus interleaved ABBA request latency.

Uses prepared production operators through their numerical environment preset.
No operator substitution, trace, tensor saves, counter wrappers, or hooks in
timed calls. Wall latency includes text encoding, denoising, VAE, and PIL output.
Model loading and correctness comparisons are outside measured intervals.
"""
import argparse
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
from types import SimpleNamespace
import time
import traceback

from bench_support import ALLOWED, compare, gpu_state, sha, summary, verify_source

GATE = 'VLLM_OMNI_FUSED_QK_NORM_ROPE_MIN_TOKENS'
NUMERICS = 'VLLM_OMNI_FUSED_QK_NORM_ROPE_NUMERICS'
PROMPT = 'a red fox standing in fresh snow at the edge of a pine forest'
REVISION = 'f332072aa78be7aecdf3ee76d5c247082da564a6'


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source', type=Path, required=True)
    ap.add_argument('--expected-commit', required=True)
    ap.add_argument('--model', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--rounds', type=int, default=3)
    args = ap.parse_args()
    output = args.output.resolve()
    assert output.is_relative_to(ALLOWED) and not output.exists() and args.rounds >= 3
    output.parent.mkdir(parents=True, exist_ok=True)
    source_info = verify_source(args.source, args.expected_commit)
    assert args.model.resolve().is_relative_to(ALLOWED)
    assert (args.model.parent.parent / 'MODEL_READY').read_text().strip() == REVISION
    import sys
    sys.path.insert(0, str(args.source.resolve()))
    import torch
    import numpy as np
    from tests.model_executor.helpers import bootstrap_vllm_layer_custom_op_modules
    bootstrap_vllm_layer_custom_op_modules()
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.config.load import LoadConfig
    from vllm_omni.diffusion.config import set_current_diffusion_config
    from vllm_omni.diffusion.data import OmniDiffusionConfig
    from vllm_omni.diffusion.distributed import parallel_state
    from vllm_omni.diffusion.forward_context import set_forward_context
    from vllm_omni.diffusion.model_loader.diffusers_loader import DiffusersPipelineLoader
    from vllm_omni.diffusion.models.z_image.pipeline_z_image import ZImagePipeline
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams
    module = importlib.import_module('vllm_omni.diffusion.models.z_image.z_image_transformer')
    operator = importlib.import_module('vllm_omni.diffusion.layers.fused_qk_norm_rope')
    assert Path(module.__file__).resolve().is_relative_to(args.source.resolve())
    assert torch.cuda.device_count() == 1 and 'H200' in torch.cuda.get_device_name(0)
    assert torch.version.cuda == '13.0' and importlib.metadata.version('vllm') == '0.29.0'
    old = {key: os.environ.get(key) for key in (GATE, NUMERICS)}
    real_fusion = module.fused_qk_norm_rope
    result = {'status': 'RUNNING', 'pr': 7597, 'source': source_info,
              'scope': 'Pretrained full eager request through pipeline and PIL postprocessing; model loading excluded',
              'script_sha256': sha(__file__), 'support_sha256': sha(Path(__file__).parent / 'bench_support.py'),
              'checkpoint': {'repo': 'Tongyi-MAI/Z-Image-Turbo', 'revision': REVISION},
              'versions': {n: importlib.metadata.version(n) for n in ('torch','vllm','triton','diffusers','transformers')},
              'gpu_before': gpu_state(), 'cuda': torch.version.cuda, 'seed': 7, 'prompt': PROMPT,
              'height': 1024, 'width': 1024, 'guidance_scale': 0.0, 'dtype': 'bfloat16',
              'parallelism': {'world': 1, 'tp': 1, 'sp': 1}, 'compile': False, 'cuda_graph': False,
              'rounds': args.rounds, 'warmup_full_requests_per_arm_per_steps': 1,
              'arms': {'original': {'min_tokens': 10**12, 'numerics': 'fast'},
                       'current': {'min_tokens': 0, 'numerics': 'fast'},
                       'exact': {'min_tokens': 0, 'numerics': 'vllm_cuda_128',
                                 'rms_norm_reduction': 'vllm_cuda_128', 'enable_fp_fusion': False}},
              'cases': []}
    def save():
        output.write_text(json.dumps(result, indent=2) + '\n')
    save()
    calls = [0]
    def counted(*values, **kwargs):
        calls[0] += 1
        return real_fusion(*values, **kwargs)
    def select(arm, counting=False):
        os.environ[GATE] = str(10**12 if arm == 'original' else 0)
        os.environ[NUMERICS] = 'vllm_cuda_128' if arm == 'exact' else 'fast'
        assert operator._resolve_numerical_parameters(None, None) == (
            ('vllm_cuda_128', False) if arm == 'exact' else ('triton', True))
        module.fused_qk_norm_rope = counted if counting else real_fusion
        calls[0] = 0
    def equal_outputs(actual, baseline):
        records = {key: compare(actual[key], baseline[key]) for key in ('vae', 'rgb')}
        return {'bitwise': all(v['bitwise'] for v in records.values()), 'tensors': records}
    try:
        config = VllmConfig()
        od = OmniDiffusionConfig(model=str(args.model), dtype=torch.bfloat16, enforce_eager=True)
        with set_current_vllm_config(config), set_current_diffusion_config(od):
            parallel_state.init_distributed_environment(world_size=1, rank=0, local_rank=0,
                                                       distributed_init_method='env://')
            parallel_state.initialize_model_parallel(sequence_parallel_size=1, tensor_parallel_size=1)
            torch.manual_seed(7)
            pipeline = DiffusersPipelineLoader(LoadConfig(), od).load_model(
                load_device='cuda', load_format='custom_pipeline', custom_pipeline_name=ZImagePipeline,
                device=torch.device('cuda:0'))
            pipeline.eval()
            print('PRETRAINED_PIPELINE_LOADED', flush=True)
            result['attention_backends'] = sorted({c.attn_backend.get_name() for c in pipeline.modules()
                                                 if getattr(c, 'attn_backend', None) is not None})
            norm_sites = [(name, c) for name, c in pipeline.transformer.named_modules()
                          if name.rsplit('.', 1)[-1] in ('norm_q', 'norm_k')]
            assert len(norm_sites) == 68, len(norm_sites)
            result['qk_norm_dispatch'] = [{'site': name, 'type': type(c).__module__ + '.' + type(c).__name__,
                'method': c._forward_method.__qualname__, 'epsilon': c.variance_epsilon} for name, c in norm_sites]
            assert all(c._forward_method.__name__ == 'forward_cuda' for _, c in norm_sites)
            with torch.inference_mode(), set_forward_context(vllm_config=config, omni_diffusion_config=od):
                for steps in (8, 28):
                    case = {'steps': steps, 'parity_before': {}, 'repeatability': {}, 'fused_call_counts': {}}
                    result['cases'].append(case)
                    def generate(capture=False):
                        params = OmniDiffusionSamplingParams(height=1024, width=1024, seed=7,
                            generator=torch.Generator(device='cuda').manual_seed(7),
                            num_inference_steps=steps, num_outputs_per_prompt=1, guidance_scale=0.0)
                        request = SimpleNamespace(prompts=[PROMPT], sampling_params=params)
                        assert od.enforce_eager and not torch.compiler.is_compiling()
                        torch.cuda.synchronize()
                        begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                        start = time.perf_counter()
                        begin.record()
                        generated = pipeline(request).output
                        end.record()
                        images = pipeline.image_processor.postprocess(generated, output_type='pil')
                        images[0].convert('RGB')
                        torch.cuda.synchronize()
                        wall_ms = (time.perf_counter() - start) * 1000
                        answer = None
                        if capture:
                            answer = {'vae': generated.detach().cpu().clone(),
                                      'rgb': torch.from_numpy(np.asarray(images[0].convert('RGB')).copy())}
                        return answer, {'wall_ms': wall_ms, 'pipeline_cuda_event_ms': begin.elapsed_time(end)}
                    outputs = {}
                    for arm in ('original', 'current', 'exact'):
                        select(arm, counting=True)
                        outputs[arm], _ = generate(capture=True)
                        count = calls[0]
                        expected = 0 if arm == 'original' else 34 * steps
                        assert count == expected, (arm, count, expected)
                        case['fused_call_counts'][arm] = count
                        calls[0] = 0
                        repeated, _ = generate(capture=True)
                        assert calls[0] == expected
                        case['repeatability'][arm] = equal_outputs(repeated, outputs[arm])
                        assert case['repeatability'][arm]['bitwise']
                    case['parity_before'] = {a: equal_outputs(outputs[a], outputs['original'])
                                             for a in ('current', 'exact')}
                    assert case['parity_before']['exact']['bitwise'], case['parity_before']
                    save()
                    print('PARITY_BEFORE_PASSED', steps, flush=True)
                    for arm in ('original', 'current', 'exact'):
                        select(arm)
                        generate()
                    case['timing'] = {}
                    for optimized in ('current', 'exact'):
                        samples = {a: [] for a in ('original', optimized)}
                        for round_id in range(args.rounds):
                            for arm in ('original', optimized, optimized, 'original'):
                                select(arm)
                                assert module.fused_qk_norm_rope is real_fusion
                                _, elapsed = generate()
                                samples[arm].append(elapsed)
                                print('TIMED', steps, optimized, round_id, arm, json.dumps(elapsed), flush=True)
                        case['timing'][optimized] = {'order': ['original', optimized, optimized, 'original'],
                            'wall_ms': summary({a: [s['wall_ms'] for s in v] for a, v in samples.items()},
                                               'original', optimized),
                            'pipeline_cuda_event_ms': summary({a: [s['pipeline_cuda_event_ms'] for s in v]
                                                             for a, v in samples.items()}, 'original', optimized)}
                        save()
                    case['parity_after'] = {}
                    for arm in ('original', 'exact'):
                        select(arm)
                        answer, _ = generate(capture=True)
                        case['parity_after'][arm] = equal_outputs(answer, outputs['original'])
                        assert case['parity_after'][arm]['bitwise']
                    save()
                    print('CASE_COMPLETE', steps, json.dumps(case['timing']['exact']['wall_ms']), flush=True)
            result['gpu_after'] = gpu_state()
            result['peak_allocated_gib'] = torch.cuda.max_memory_allocated() / 2**30
        result['status'] = 'COMPLETE_PRETRAINED_EAGER_BITWISE_SPEED'
        save()
        return 0
    except Exception:
        result['status'] = 'FAILED'
        result['error'] = traceback.format_exc()
        save()
        print(result['error'], flush=True)
        return 1
    finally:
        module.fused_qk_norm_rope = real_fusion
        for key, value in old.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


if __name__ == '__main__':
    raise SystemExit(main())
