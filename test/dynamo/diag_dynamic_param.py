# Targeted diagnostic: monkey-patch to trace if _automatic_dynamic sees DYNAMIC
import torch
import torch._dynamo
import torch._dynamo.testing
import torch._dynamo.variables.builder as builder

# Monkey-patch _automatic_dynamic to trace its output
_original_auto_dynamic = builder._automatic_dynamic

def _traced_auto_dynamic(e, tx, source, static_shapes, **kwargs):
    result = _original_auto_dynamic(e, tx, source, static_shapes, **kwargs)
    from torch.fx.experimental.symbolic_shapes import DimDynamic
    if hasattr(result, 'dynamic_sizes'):
        print(f"[TRACE] _automatic_dynamic source={source.name} static_shapes={static_shapes} "
              f"type={type(e).__name__} "
              f"dynamic_sizes={result.dynamic_sizes} "
              f"has_dynamo_dynamic_indices={hasattr(e, '_dynamo_dynamic_indices')} "
              f"_dynamo_dynamic_indices={getattr(e, '_dynamo_dynamic_indices', 'N/A')}")
    return result

builder._automatic_dynamic = _traced_auto_dynamic

# Also patch tensor_always_has_static_shape
from torch._dynamo import utils as dynamo_utils
_original_static = dynamo_utils.tensor_always_has_static_shape

def _traced_static(tensor, is_tensor, tensor_source):
    result = _original_static(tensor, is_tensor, tensor_source)
    if type(tensor) is torch.nn.Parameter:
        print(f"[TRACE] tensor_always_has_static_shape type=Parameter "
              f"has_dynamo_dynamic_indices={hasattr(tensor, '_dynamo_dynamic_indices')} "
              f"result={result}")
    return result

dynamo_utils.tensor_always_has_static_shape = _traced_static

cnts = torch._dynamo.testing.CompileCounter()

@torch.compile(backend=cnts, fullgraph=True)
def fn(x):
    return x.cos()

p = torch.nn.Parameter(torch.ones(2, 2))
for d in range(p.dim()):
    torch._dynamo.mark_dynamic(p, d)

print("=== First call with (2,2) ===")
fn(p)
print(f"frame_count: {cnts.frame_count}")

p2 = torch.nn.Parameter(torch.ones(3, 3))
for d in range(p2.dim()):
    torch._dynamo.mark_dynamic(p2, d)

print("=== Second call with (3,3) ===")
fn(p2)
print(f"frame_count: {cnts.frame_count}")
