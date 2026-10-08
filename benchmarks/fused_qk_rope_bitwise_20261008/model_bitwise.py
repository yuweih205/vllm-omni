# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Full shallow-transformer eager parity and ABBA timing for four fusion PRs.

Uses one current PR source snapshot per process, one random model and fixed inputs.
Includes model RoPE generation and table packing. No pretrained weights, sampler,
text encoder, VAE, torch.compile, or CUDA graph is used.
"""

import argparse
from contextlib import contextmanager
import hashlib
import importlib
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import traceback


ALLOWED = Path("/inspire/ssd/project/video-generation/public/huangyuwei")
PROFILES = {
    "flux2": (7560, "flux2.flux2_transformer", "Flux2Transformer2DModel", 48),
    "flux1": (7595, "flux.flux_transformer", "FluxTransformer2DModel", 24),
    "hunyuan15": (7596, "hunyuan_video.hunyuan_video_15_transformer", "HunyuanVideo15Transformer3DModel", 16),
    "ovis": (7600, "ovis_image.ovis_image_transformer", "OvisImageTransformer2DModel", 24),
}
EXACT_CONFIG = {"rms_norm_reduction": "vllm_cuda_128", "enable_fp_fusion": False}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def gpu_state():
    physical = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0].strip()
    result = subprocess.run(
        ["nvidia-smi", "-i", physical,
         "--query-gpu=name,uuid,driver_version,pstate,clocks.sm,clocks.mem,power.draw,temperature.gpu,utilization.gpu",
         "--format=csv,noheader"],
        text=True, capture_output=True, check=False,
    )
    return {"physical_device": physical, "returncode": result.returncode,
            "gpu": result.stdout.strip(), "stderr": result.stderr.strip()}


def verify_source(source, expected_commit, model_relative):
    manifest_path = source / "source_info.json"
    manifest = json.loads(manifest_path.read_text())
    assert manifest["commit"] == expected_commit
    required = (model_relative, "vllm_omni/diffusion/layers/fused_qk_norm_rope.py")
    assert all(relative in manifest["files"] for relative in required)
    for relative, expected in manifest["files"].items():
        path = (source / relative).resolve()
        assert path.is_relative_to(source), relative
        assert sha(path) == expected, relative
    return {"commit": manifest["commit"], "manifest_sha256": sha(manifest_path),
            "verified_files": len(manifest["files"]),
            "model_sha256": manifest["files"][model_relative],
            "operator_sha256": manifest["files"][required[1]]}


def tensor_leaves(value):
    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, (tuple, list)):
        for child in value:
            yield from tensor_leaves(child)
    elif isinstance(value, dict):
        for key in sorted(value):
            yield from tensor_leaves(value[key])
    elif value is not None:
        raise TypeError(f"Unexpected output/input type: {type(value)}")


def fingerprint(named_tensors):
    digest = hashlib.sha256()
    for name, tensor in named_tensors:
        digest.update(name.encode())
        digest.update(str((tuple(tensor.shape), tensor.dtype)).encode())
        digest.update(tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def compare(actual, reference):
    actual, reference = tuple(tensor_leaves(actual)), tuple(tensor_leaves(reference))
    assert len(actual) == len(reference) and actual
    records = []
    for a, b in zip(actual, reference):
        assert a.shape == b.shape and a.dtype == b.dtype and a.device == b.device
        byte_mismatches = int((a.contiguous().view(torch.uint8) != b.contiguous().view(torch.uint8)).sum().item())
        records.append({"shape": list(a.shape), "dtype": str(a.dtype),
                        "byte_mismatches": byte_mismatches,
                        "finite": bool(torch.isfinite(a).all() and torch.isfinite(b).all()),
                        "max_abs": float((a.float() - b.float()).abs().max().item())})
    return {"bitwise": all(r["byte_mismatches"] == 0 and r["finite"] for r in records),
            "tensors": records}


class Arms:
    def __init__(self, model_module, exact_module):
        self.arm = "original"
        self.counting = True
        self.counts = {"joint": 0, "single": 0}
        self.original_functions = {}
        self.module = model_module
        self.exact_module = exact_module
        for name, kind in (("fused_joint_qkv_norm_rope", "joint"), ("fused_qk_norm_rope", "single")):
            if hasattr(model_module, name):
                original = getattr(model_module, name)
                self.original_functions[name] = original
                setattr(model_module, name, self.wrapper(original, getattr(exact_module, name), kind))

    def wrapper(self, original, exact, kind):
        def call(*args, **kwargs):
            assert self.arm != "original", "Original arm unexpectedly entered fusion"
            if self.counting:
                self.counts[kind] += 1
            if self.arm == "exact":
                return exact(*args, **kwargs, **EXACT_CONFIG)
            return original(*args, **kwargs)
        return call

    def select(self, arm, *, counting=False):
        assert arm in ("original", "current", "exact")
        self.arm, self.counting = arm, counting
        self.counts = {"joint": 0, "single": 0}
        os.environ["VLLM_OMNI_FUSED_QK_NORM_ROPE_MIN_TOKENS"] = str(10**12 if arm == "original" else 0)

    def restore(self):
        for name, original in self.original_functions.items():
            setattr(self.module, name, original)


def build_model(module, args, od_config):
    _, _, class_name, heads = PROFILES[args.profile]
    common = {"od_config": od_config, "num_layers": 2, "num_attention_heads": heads, "attention_head_dim": 128}
    if args.profile == "hunyuan15":
        settings = {**common, "num_refiner_layers": 1, "in_channels": 65, "out_channels": 32,
                    "text_embed_dim": 3584, "text_embed_2_dim": 1472, "image_embed_dim": 1152,
                    "patch_size": 1, "patch_size_t": 1, "rope_axes_dim": (16, 56, 56)}
    elif args.profile == "flux2":
        settings = {**common, "num_single_layers": 2, "in_channels": 128, "joint_attention_dim": 15360,
                    "guidance_embeds": True, "timestep_guidance_channels": 256, "axes_dims_rope": (32, 32, 32, 32)}
    elif args.profile == "flux1":
        settings = {**common, "num_single_layers": 2, "in_channels": 64, "joint_attention_dim": 4096,
                    "pooled_projection_dim": 768, "guidance_embeds": True, "axes_dims_rope": (16, 56, 56)}
    else:
        settings = {**common, "num_single_layers": 2, "in_channels": 64, "out_channels": 64,
                    "joint_attention_dim": 2048, "axes_dims_rope": (16, 56, 56)}
    model = getattr(module, class_name)(**settings).cuda().to(torch.bfloat16).eval()
    # vLLM linear layers allocate empty tensors: initialize every parameter.
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if parameter.ndim > 1:
                parameter.normal_(0, 0.02)
            elif name.endswith("weight") and "norm" in name:
                parameter.fill_(1)
            else:
                parameter.zero_()
        for name, child in model.named_modules():
            if name.rsplit(".", 1)[-1] in ("norm_q", "norm_k", "norm_added_q", "norm_added_k"):
                child.weight.uniform_(0.5, 1.5)
    settings.pop("od_config")
    return model, settings


def build_inputs(args, settings):
    batch, text, side = args.batch_size, args.text_tokens, args.image_side
    image = side * side

    def normal(*shape):
        return torch.randn(*shape, device="cuda", dtype=torch.bfloat16)

    if args.profile == "hunyuan15":
        # All three condition streams are valid; mask construction stays in the model.
        inputs = {"hidden_states": normal(batch, 65, 1, side, side),
                  "timestep": torch.full((batch,), 500, device="cuda", dtype=torch.float32),
                  "encoder_hidden_states": normal(batch, text, 3584),
                  "encoder_attention_mask": torch.ones(batch, text, device="cuda", dtype=torch.bool),
                  "encoder_hidden_states_2": normal(batch, 16, 1472),
                  "encoder_attention_mask_2": torch.ones(batch, 16, device="cuda", dtype=torch.bool),
                  "image_embeds": normal(batch, 16, 1152),
                  "image_embeds_mask": torch.ones(batch, 16, device="cuda", dtype=torch.bool),
                  "return_dict": False}
    else:
        axes = len(settings["axes_dims_rope"])
        img_ids = torch.zeros(image, axes, device="cuda", dtype=torch.float32)
        idx = torch.arange(image, device="cuda")
        img_ids[:, 1], img_ids[:, 2] = idx // side, idx % side
        txt_ids = torch.zeros(text, axes, device="cuda", dtype=torch.float32)
        txt_ids[:, 0] = torch.arange(text, device="cuda")
        inputs = {"hidden_states": normal(batch, image, settings["in_channels"]),
                  "encoder_hidden_states": normal(batch, text, settings["joint_attention_dim"]),
                  "timestep": torch.full((batch,), 0.5, device="cuda", dtype=torch.float32),
                  "img_ids": img_ids, "txt_ids": txt_ids, "return_dict": False}
        if args.profile in ("flux1", "flux2"):
            inputs["guidance"] = torch.full((batch,), 3.5, device="cuda", dtype=torch.float32)
        if args.profile == "flux1":
            inputs["pooled_projections"] = normal(batch, 768)
    return inputs


def input_tensor_items(inputs):
    return [(key, value) for key, value in sorted(inputs.items()) if isinstance(value, torch.Tensor)]


def measure(fn, iterations):
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        output = fn()
    end.record()
    end.synchronize()
    del output
    return start.elapsed_time(end) / iterations


def benchmark(arms, forward, args):
    for arm in ("original", "current", "exact"):
        arms.select(arm)
        for _ in range(args.warmup):
            forward()
    pairs = {}
    for optimized in ("current", "exact"):
        samples = {"original": [], optimized: []}
        for _ in range(args.rounds):
            for arm in ("original", optimized, optimized, "original"):
                arms.select(arm)
                samples[arm].append(measure(forward, args.iterations))
        medians = {arm: statistics.median(values) for arm, values in samples.items()}
        pairs[optimized] = {"order": ["original", optimized, optimized, "original"],
                            "samples_ms": samples, "median_ms": medians,
                            "latency_saving_pct": 100 * (1 - medians[optimized] / medians["original"])}
    return pairs


@contextmanager
def checked_norm_provider(model, ir, records):
    hooks = []

    def check(norm, inputs):
        x = inputs[0]
        provider = ir.ops.rms_norm.dispatch(x, norm.weight, norm.variance_epsilon, None).provider
        assert provider == "vllm_c", provider
        records.append({"module": type(norm).__module__, "forward": repr(norm._forward_method),
                        "provider": provider, "epsilon": norm.variance_epsilon,
                        "input_shape": list(x.shape), "input_stride": list(x.stride())})

    for name, child in model.named_modules():
        if name.rsplit(".", 1)[-1] in ("norm_q", "norm_k", "norm_added_q", "norm_added_k"):
            # In vLLM 0.29 both forward_native and ordinary forward_cuda
            # delegate to the IR op; its selected provider decides the kernel.
            assert type(child).__module__ == "vllm.model_executor.layers.layernorm"
            native_source = inspect.getsource(type(child).forward_native)
            assert "ir.ops.rms_norm" in native_source, native_source
            hooks.append(child.register_forward_pre_hook(check))
    assert hooks
    try:
        yield
    finally:
        for hook in hooks:
            hook.remove()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=PROFILES, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261008)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--text-tokens", type=int, default=64)
    parser.add_argument("--image-side", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--rounds", type=int, default=3)
    args = parser.parse_args()
    assert args.batch_size in (1, 2) and args.text_tokens > 0 and args.image_side > 0
    assert args.warmup >= 10 and args.iterations >= 10 and args.rounds >= 3
    source, output = args.source.resolve(), args.output.resolve()
    assert source.is_relative_to(ALLOWED) and output.is_relative_to(ALLOWED) and not output.exists()
    output.parent.mkdir(parents=True, exist_ok=True)
    pr, suffix, _, _ = PROFILES[args.profile]
    module_name = "vllm_omni.diffusion.models." + suffix
    relative = module_name.replace(".", "/") + ".py"
    source_record = verify_source(source, args.expected_commit, relative)
    sys.path.insert(0, str(source))
    for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
        os.environ[key] = "1"
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", str(33000 + pr % 1000 + args.batch_size * 1000))
    old_gate = os.environ.get("VLLM_OMNI_FUSED_QK_NORM_ROPE_MIN_TOKENS")
    os.environ["VLLM_OMNI_FUSED_QK_NORM_ROPE_MIN_TOKENS"] = "0"
    global torch
    import torch

    assert torch.cuda.is_available() and torch.cuda.device_count() == 1
    assert "H200" in torch.cuda.get_device_name(0)
    assert importlib.metadata.version("vllm") == "0.29.0" and torch.version.cuda == "13.0"
    assert os.environ.get("VLLM_BATCH_INVARIANT", "0") in ("", "0")
    from tests.model_executor.helpers import bootstrap_vllm_layer_custom_op_modules
    bootstrap_vllm_layer_custom_op_modules()
    from vllm import ir
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm_omni.diffusion.config import set_current_diffusion_config
    from vllm_omni.diffusion.data import OmniDiffusionConfig, TransformerConfig
    from vllm_omni.diffusion.distributed import parallel_state
    from vllm_omni.diffusion.forward_context import set_forward_context

    model_module = importlib.import_module(module_name)
    current_module = importlib.import_module("vllm_omni.diffusion.layers.fused_qk_norm_rope")
    exact_module = importlib.import_module("parameterized_operator")
    assert Path(model_module.__file__).resolve() == source / relative
    assert Path(current_module.__file__).resolve().is_relative_to(source)
    result = {"status": "RUNNING", "pr": pr, "profile": args.profile,
              "scope": "Full shallow random-init transformer forward, including model RoPE and per-forward packing; no pretrained model or end-to-end pipeline claim",
              "source": source_record, "source_root": str(source), "script_sha256": sha(__file__),
              "exact_operator_sha256": sha(exact_module.__file__),
              "image": os.environ.get("OMNI_BENCH_IMAGE", "unknown"),
              "versions": {name: importlib.metadata.version(name) for name in ("torch", "triton", "vllm", "diffusers")},
              "cuda": torch.version.cuda, "gpu_before": gpu_state(),
              "settings": vars(args) | {"source": str(source), "output": str(output)},
              "arms": {"original": {"min_tokens": 10**12, "norm_provider": "vllm_c"},
                       "current": {"min_tokens": 0, "rms_norm_reduction": "triton", "enable_fp_fusion": True},
                       "exact": {"min_tokens": 0, **EXACT_CONFIG}},
              "compile": False, "cuda_graph": False, "dtype": "bfloat16", "head_dim": 128,
              "parallelism": {"world": 1, "tp": 1, "sp": 1}, "seed": args.seed}

    def save():
        output.write_text(json.dumps(result, indent=2) + "\n")

    save()
    arms = None
    try:
        config = VllmConfig()
        od_config = OmniDiffusionConfig(model=None, enforce_eager=True)
        od_config.set_tf_model_config(TransformerConfig.from_dict({"num_layers": 2}))
        with set_current_vllm_config(config), set_current_diffusion_config(od_config):
            parallel_state.init_distributed_environment(world_size=1, rank=0, local_rank=0, distributed_init_method="env://")
            parallel_state.initialize_model_parallel(sequence_parallel_size=1, tensor_parallel_size=1)
            torch.manual_seed(args.seed)
            model, model_settings = build_model(model_module, args, od_config)
            inputs = build_inputs(args, model_settings)
            result["model_settings"] = model_settings
            result["input_tensors"] = {name: {"shape": list(value.shape), "dtype": str(value.dtype)}
                                       for name, value in input_tensor_items(inputs)}
            result["state_sha256"] = fingerprint(model.state_dict().items())
            result["input_sha256"] = fingerprint(input_tensor_items(inputs))
            result["attention_backends"] = {name: child.attn_backend.get_name()
                                           for name, child in model.named_modules()
                                           if getattr(child, "attn_backend", None) is not None}
            arms = Arms(model_module, exact_module)
            expected_counts = {"joint": 2, "single": 0 if args.profile == "hunyuan15" else 2}
            result["expected_fused_calls"] = expected_counts
            with torch.inference_mode(), set_forward_context(vllm_config=config, omni_diffusion_config=od_config):
                # Entering VllmConfig/ForwardContext applies priorities: override afterward.
                ir.ops.rms_norm.set_default(["vllm_c"])
                result["rmsnorm_priority"] = repr(ir.ops.rms_norm.get_priority())

                def forward():
                    return model(**inputs)

                outputs, counts = {}, {}
                provider_records = []
                for arm in ("original", "current", "exact"):
                    arms.select(arm, counting=True)
                    if arm == "original":
                        with checked_norm_provider(model, ir, provider_records):
                            outputs[arm] = forward()
                    else:
                        outputs[arm] = forward()
                    counts[arm] = dict(arms.counts)
                    assert counts[arm] == ({"joint": 0, "single": 0} if arm == "original" else expected_counts), counts
                    repeat = forward()
                    result.setdefault("repeatability", {})[arm] = compare(repeat, outputs[arm])
                    assert result["repeatability"][arm]["bitwise"], (arm, result["repeatability"][arm])
                result["observed_fused_calls"] = counts
                result["baseline_norm_dispatch"] = provider_records
                assert provider_records
                result["parity"] = {arm: compare(outputs[arm], outputs["original"]) for arm in ("current", "exact")}
                save()
                if not result["parity"]["exact"]["bitwise"]:
                    result["status"] = "PARITY_FAILED_PERF_SKIPPED"
                    save()
                    return 2
                del outputs
                result["eager_abba"] = benchmark(arms, forward, args)
                arms.select("original")
                baseline_after = forward()
                arms.select("exact")
                result["parity_after_timing"] = compare(forward(), baseline_after)
                assert result["parity_after_timing"]["bitwise"]
            assert fingerprint(model.state_dict().items()) == result["state_sha256"], "Model state mutated"
            assert fingerprint(input_tensor_items(inputs)) == result["input_sha256"], "Inputs mutated"
        result["gpu_after"] = gpu_state()
        result["status"] = "COMPLETE_SHALLOW_MODEL_EAGER_BITWISE"
        save()
        print(json.dumps({"pr": pr, "status": result["status"], "parity": result["parity"],
                          "eager_abba": result["eager_abba"]}), flush=True)
        return 0
    except Exception:
        result["status"] = "FAILED"
        result["error"] = traceback.format_exc()
        save()
        print(result["error"], flush=True)
        return 1
    finally:
        if arms is not None:
            arms.restore()
        if old_gate is None:
            os.environ.pop("VLLM_OMNI_FUSED_QK_NORM_ROPE_MIN_TOKENS", None)
        else:
            os.environ["VLLM_OMNI_FUSED_QK_NORM_ROPE_MIN_TOKENS"] = old_gate
        if torch.distributed.is_initialized():
            parallel_state.destroy_distributed_env()


if __name__ == "__main__":
    sys.exit(main())
