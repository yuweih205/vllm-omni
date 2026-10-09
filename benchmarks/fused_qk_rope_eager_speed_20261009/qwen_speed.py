"""Current prepared Qwen source, full 60-layer random-weight eager ABBA.

The reference is Qwen's existing per-stream fusion plus concatenation. Joint
packing is suppressed only in that reference arm, with the token gate zero in
both arms. Full BF16 output storage bits, including signed zero, are checked.
"""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import traceback

from bench_support import ALLOWED, compare, fingerprint, gpu_state, sha, summary, verify_source


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source', type=Path, required=True)
    ap.add_argument('--expected-commit', required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--batch', type=int, choices=(1,2), required=True)
    args = ap.parse_args()
    output = args.output.resolve()
    assert output.is_relative_to(ALLOWED) and not output.exists()
    output.parent.mkdir(parents=True, exist_ok=True)
    info = verify_source(args.source, args.expected_commit)
    import sys
    sys.path.insert(0, str(args.source.resolve()))
    import torch
    from tests.model_executor.helpers import bootstrap_vllm_layer_custom_op_modules
    bootstrap_vllm_layer_custom_op_modules()
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm_omni.diffusion.config import set_current_diffusion_config
    from vllm_omni.diffusion.data import OmniDiffusionConfig, TransformerConfig
    from vllm_omni.diffusion.distributed import parallel_state
    from vllm_omni.diffusion.forward_context import set_forward_context
    from vllm_omni.diffusion.models.qwen_image import qwen_image_transformer as qt
    assert Path(qt.__file__).resolve().is_relative_to(args.source.resolve())
    assert torch.cuda.device_count() == 1 and 'H200' in torch.cuda.get_device_name(0)
    result = {'status': 'RUNNING', 'pr': 7594, 'source': info,
        'scope': 'Full 60-layer random-weight transformer; existing per-stream fused chain vs joint fusion; no pretrained generation',
        'script_sha256': sha(__file__), 'support_sha256': sha(Path(__file__).parent / 'bench_support.py'),
        'versions': {n: importlib.metadata.version(n) for n in ('torch','vllm','triton','diffusers')},
        'gpu_before': gpu_state(), 'dtype': 'bfloat16', 'compile': False, 'cuda_graph': False,
        'parallelism': {'world':1,'tp':1,'sp':1}, 'layers':60, 'heads':24,'head_dim':128,
        'batch':args.batch, 'text_tokens':512,'image_tokens':4096,
        'constructor_seed':0, 'parameter_and_input_generator_seed':1,
        'num_iterations_per_sample':10, 'rounds':3, 'warmup_calls_per_arm':3,
        'gate_both_arms':0, 'numerics_both_arms':'fast'}
    def save():
        output.write_text(json.dumps(result, indent=2) + '\n')
    save()
    packer, joint, single = qt._packed_qk_norm_rope_table, qt.fused_joint_qkv_norm_rope, qt.fused_qk_norm_rope
    counts = {'joint':0,'single':0}
    def counted(fn, kind):
        def call(*values, **kwargs):
            counts[kind] += 1
            return fn(*values, **kwargs)
        return call
    def select(arm, counting=False):
        os.environ['VLLM_OMNI_FUSED_QK_NORM_ROPE_MIN_TOKENS'] = '0'
        os.environ['VLLM_OMNI_FUSED_QK_NORM_ROPE_NUMERICS'] = 'fast'
        qt._packed_qk_norm_rope_table = packer if arm == 'joint' else lambda *_a, **_k: None
        qt.fused_joint_qkv_norm_rope = counted(joint,'joint') if counting else joint
        qt.fused_qk_norm_rope = counted(single,'single') if counting else single
        counts.update(joint=0,single=0)
    try:
        cfg = VllmConfig()
        od = OmniDiffusionConfig(model=None,dtype=torch.bfloat16,enforce_eager=True)
        od.set_tf_model_config(TransformerConfig.from_dict({'num_layers':60}))
        with set_current_vllm_config(cfg), set_current_diffusion_config(od):
            parallel_state.init_distributed_environment(world_size=1,rank=0,local_rank=0,
                                                       distributed_init_method='env://')
            parallel_state.initialize_model_parallel(sequence_parallel_size=1,tensor_parallel_size=1)
            torch.manual_seed(0)
            with torch.device('cuda'):
                model = qt.QwenImageTransformer2DModel(od, num_layers=60)
            model = model.to(torch.bfloat16).eval()
            g = torch.Generator(device='cuda').manual_seed(1)
            with torch.no_grad():
                for name, p in model.named_parameters():
                    if p.dim()==1 and 'norm' in name and 'weight' in name:
                        p.copy_(1 + .05*torch.randn(p.shape,device='cuda',generator=g))
                    elif p.dim()==1:
                        p.copy_(.01*torch.randn(p.shape,device='cuda',generator=g))
                    else:
                        p.copy_((.02*torch.randn(p.shape,device='cuda',dtype=torch.float32,generator=g)).to(p.dtype))
            batch=args.batch
            inputs = {'hidden_states':torch.randn(batch,4096,64,device='cuda',dtype=torch.bfloat16,generator=g),
                      'encoder_hidden_states':torch.randn(batch,512,3584,device='cuda',dtype=torch.bfloat16,generator=g),
                      'encoder_hidden_states_mask':torch.ones(batch,512,device='cuda',dtype=torch.bool),
                      'timestep':torch.full((batch,),.5,device='cuda',dtype=torch.bfloat16),
                      'img_shapes':[[(1,64,64)]]*batch,'txt_seq_lens':[512]*batch,'return_dict':False}
            result['input_sha256']=fingerprint((n,v) for n,v in inputs.items() if isinstance(v,torch.Tensor))
            result['parameter_count']=sum(p.numel() for p in model.parameters())
            result['attention_backends']=sorted({c.attn_backend.get_name() for c in model.modules()
                                                if getattr(c,'attn_backend',None) is not None})
            with torch.inference_mode(),set_forward_context(vllm_config=cfg,omni_diffusion_config=od):
                def forward():
                    assert od.enforce_eager and not torch.compiler.is_compiling()
                    return model(**inputs)[0]
                outputs={}
                result['counts']={}
                result['repeatability']={}
                for arm in ('per_stream','joint'):
                    select(arm,counting=True)
                    outputs[arm]=forward().detach().cpu()
                    result['counts'][arm]=counts.copy()
                    assert counts == ({'joint':60,'single':0} if arm=='joint' else {'joint':0,'single':120}), counts
                    repeated=forward()
                    result['repeatability'][arm]=compare(repeated,outputs[arm])
                    assert result['repeatability'][arm]['bitwise']
                result['parity_before']=compare(outputs['joint'],outputs['per_stream'])
                assert result['parity_before']['bitwise'],result['parity_before']
                save()
                for arm in ('per_stream','joint'):
                    select(arm)
                    for _ in range(3):
                        forward()
                samples={'per_stream':[],'joint':[]}
                for round_id in range(3):
                    for arm in ('per_stream','joint','joint','per_stream'):
                        select(arm)
                        assert qt.fused_joint_qkv_norm_rope is joint and qt.fused_qk_norm_rope is single
                        torch.cuda.synchronize()
                        start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                        start.record()
                        for _ in range(10):
                            forward()
                        end.record();end.synchronize()
                        value=start.elapsed_time(end)/10
                        samples[arm].append(value)
                        print('TIMED',round_id,arm,value,flush=True)
                result['eager_abba_ms']=summary(samples,'per_stream','joint')
                select('joint')
                result['parity_after']=compare(forward(),outputs['per_stream'])
                assert result['parity_after']['bitwise']
            result['gpu_after']=gpu_state()
            result['peak_allocated_gib']=torch.cuda.max_memory_allocated()/2**30
        result['status']='COMPLETE_QWEN_EAGER_BITWISE_SPEED'
        save()
        print('COMPLETE',json.dumps(result['eager_abba_ms']),flush=True)
        return 0
    except Exception:
        result['status']='FAILED';result['error']=traceback.format_exc();save()
        print(result['error'],flush=True)
        return 1
    finally:
        qt._packed_qk_norm_rope_table=packer
        qt.fused_joint_qkv_norm_rope=joint
        qt.fused_qk_norm_rope=single
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


if __name__=='__main__':
    raise SystemExit(main())
