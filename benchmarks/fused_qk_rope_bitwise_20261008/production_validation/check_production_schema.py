"""Verify actual pinned Torch custom-op schemas without allocating a GPU."""

import importlib
import json

import torch
from torch._subclasses.fake_tensor import FakeTensorMode

module = importlib.import_module("vllm_omni.diffusion.layers.fused_qk_norm_rope")
single = torch.ops.vllm_omni.fused_qk_norm_rope
joint = torch.ops.vllm_omni.fused_joint_qkv_norm_rope
with FakeTensorMode():
    q = torch.empty((5, 4, 128), dtype=torch.bfloat16)
    weight = torch.empty((128,), dtype=torch.bfloat16)
    table = torch.empty((5, 128), dtype=torch.bfloat16)
    output = single(q, q, weight, weight, table, 1e-6, 128, 128, True, "vllm_cuda_128", False)
    assert tuple(tuple(t.shape) for t in output) == ((5, 4, 128), (5, 4, 128))
    q0 = torch.empty((2, 3, 4, 128), dtype=torch.bfloat16)
    q1 = torch.empty((2, 5, 4, 128), dtype=torch.bfloat16)
    table = torch.empty((16, 128), dtype=torch.bfloat16)
    output = joint(q0, q0, q0, q1, q1, q1, weight, weight, weight, weight, table,
                   1e-6, 128, 128, True, "vllm_cuda_128", False)
    assert tuple(tuple(t.shape) for t in output) == ((2, 8, 4, 128),) * 3
print(json.dumps({"status": "CPU_SCHEMA_AND_FAKE_SHAPES_PASS", "torch": torch.__version__,
                  "module": module.__file__, "single": str(single.default._schema),
                  "joint": str(joint.default._schema)}))
