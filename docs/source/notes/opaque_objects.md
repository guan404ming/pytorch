---
file_format: mystnb
kernelspec:
  name: python3
mystnb:
  execution_timeout: 120
  execution_show_tb: True
  merge_streams: True
---

(opaque_objects)=

# Opaque Objects

```{note}
Opaque objects are an experimental feature. The APIs live under
`torch._library.opaque_object` and may change.
```

Previously, {ref}`custom operators <torch-library-docs>` only supported passing
in basic inputs like tensors and Python constants. Opaque objects are a way to
pass arbitrary user-defined "black box" objects into custom operators. Some
examples include ``ProcessGroup``s, communication objects, or custom callables.

Before opaque objects, the only way to pass such objects through the PyTorch
stack was via
[torch::CustomClassHolder](https://docs.pytorch.org/tutorials/advanced/custom_classes.html)
(aka TorchBind), which required implementing a C++ class with TorchScript
bindings and then a
["fake" version of the class](https://docs.pytorch.org/tutorials/advanced/custom_class_pt2.html)
in Python. Opaque objects provide a simpler, pure-Python alternative: subclass
{class}`CustomClassBase`, register the class with `register_custom_class`, and
use it directly as a custom operator argument.

## In eager mode

When dispatching a custom operator with an opaque object, the object is treated
as a ``PyObject`` IValue and passed through the dispatcher as-is. Nothing
special happens -- the operator implementation receives the original Python
object.

```{code-cell}
import torch
from torch._custom_class_base import CustomClassBase
from torch._library.opaque_object import register_custom_class

# 1. Define the opaque class
class RNGState(CustomClassBase):
    def __init__(self, seed):
        self.seed = seed

# 2. Register it as an opaque type
register_custom_class(RNGState, typ="symbolic")

# 3. Define a custom op that accepts it
@torch.library.custom_op("mylib::noisy_add", mutates_args=())
def noisy_add(x: torch.Tensor, rng: RNGState) -> torch.Tensor:
    torch.manual_seed(rng.seed)
    return x + torch.randn_like(x)

# 4. Use it
rng = RNGState(42)
result = torch.ops.mylib.noisy_add(torch.ones(3), rng)
print(result)
```

## In torch.compile

### Constant vs. Symbolic Type

There are some complexities in how the object should be treated when building
the FX graph. We split opaque objects into two kinds -- **constant** types and
**symbolic** types -- which are analogous to constant integers vs. symbolic
integers.

```{list-table}
:header-rows: 1

* -
  - Constant Type
  - Symbolic Type
* - Semantic
  - Immutable constant
  - Mutable stateful object
* - `torch.compile` behavior
  - Specialized / baked into the graph as a constant (similar to a Python
    `int` or `str`)
  - Lifted as a graph input
* - Required methods
  - `__eq__`, `__hash__`, `__fx_repr__`
  - None
* - Guarding
  - Based on `__eq__`
  - Based on type + optional `guard_fn`
* - Caching
  - The actual value is in the cache key
  - Object identity/value is in the cache key
* - Attribute/method access
  - Dynamo inlines through the real object
  - Must explicitly register members via
    `register_custom_class(members=...)`
* - Creation in compiled region
  - Allowed -- the object must have a trivial `__init__`
  - Not allowed -- must be created before `torch.compile`, or created within a
    custom op
* - Example use case
  - Configs like enums, placements
  - `ProcessGroup`, leaf modules
```

The examples below use a small backend that prints the captured Dynamo graph so
you can see how each kind of object shows up.

```{code-cell}
def print_graph(gm, example_inputs):
    gm.print_readable()
    return gm.forward
```

#### Constant type

A constant type is baked into the graph. It must implement `__eq__` and
`__hash__` (used for guarding and caching) and `__fx_repr__`, which returns an
evaluable representation used when the value is emitted into the FX graph.

```{code-cell}
class ValueConfig(CustomClassBase):
    def __init__(self, mode: str):
        self.mode = mode

    def __eq__(self, other):
        return isinstance(other, ValueConfig) and self.mode == other.mode

    def __hash__(self):
        return hash(self.mode)

    def __fx_repr__(self):
        # (repr_string, {name: type}) -- repr_string must reconstruct the object
        return f"ValueConfig(mode={self.mode!r})", {"ValueConfig": ValueConfig}

register_custom_class(ValueConfig, typ="constant")

@torch.library.custom_op("mylib::process_with_config", mutates_args=())
def process_with_config(x: torch.Tensor, cfg: ValueConfig) -> torch.Tensor:
    if cfg.mode == "double":
        return x * 2
    return x

@process_with_config.register_fake
def _(x, cfg):
    return torch.empty_like(x)

@torch.compile(fullgraph=True, backend=print_graph)
def f(x, cfg):
    return torch.ops.mylib.process_with_config(x, cfg)

# The ValueConfig is baked into the graph as a constant.
print(f(torch.ones(3), ValueConfig("double")))
```

#### Symbolic type

A symbolic type is lifted as a graph input instead of being baked in. It needs
no extra methods.

```{code-cell}
class AddModule(CustomClassBase, torch.nn.Module):
    def forward(self, x, y):
        return x * y

register_custom_class(AddModule, typ="symbolic")

@torch.library.custom_op("mylib::module_mul", mutates_args=())
def module_mul(mod: AddModule, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return mod(x, y)

@module_mul.register_fake
def _(mod, x, y):
    return torch.empty_like(x)

class M(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.moo = AddModule()

    def forward(self, x, y):
        return torch.ops.mylib.module_mul(self.moo, x, y)

# self.moo is lifted as a graph input.
compiled = torch.compile(M(), fullgraph=True, backend=print_graph)
print(compiled(torch.randn(3), torch.tensor(4.0)))
```

### Hoisted Constant Types

Constant types can optionally be marked as *hoisted*
(`register_custom_class(..., typ="constant", hoist=True)`). Instead of baking
the value as a constant in the graph -- which triggers a recompilation whenever
the value changes -- hoisting lifts it as a graph input so the graph stays the
same across different values. This is useful, for example, when dealing with
different string arguments to a custom op.

Guards on a hoisted object are removed so new values do not trigger
recompilation. The value is instead reconstructed at runtime via *pregraph
bytecode* (Python bytecode emitted before each graph invocation that calls the
constructor with the current arguments). The AOTAutograd cache key includes only
the *type* of the object, not the actual value, so two different hoisted values
produce identical cache keys and share the same compiled artifact.

```{code-cell}
class HoistedString(CustomClassBase):
    def __init__(self, val):
        self.val = val

    def __eq__(self, other):
        return isinstance(other, HoistedString) and self.val == other.val

    def __hash__(self):
        return hash(self.val)

    def __fx_repr__(self):
        return f"HoistedString({self.val!r})", {"HoistedString": HoistedString}

register_custom_class(HoistedString, typ="constant", hoist=True)

@torch.library.custom_op("mylib::op_with_string", mutates_args=())
def op_with_string(x: torch.Tensor, s: HoistedString) -> torch.Tensor:
    if s.val == "double":
        return x * 2
    elif s.val == "square":
        return x**2
    raise AssertionError("expected double or square")

@op_with_string.register_fake
def _(x, s):
    return torch.empty_like(x)

@torch.compile(fullgraph=True, backend=print_graph)
def g(x):
    return torch.ops.mylib.op_with_string(x, HoistedString("double"))

# HoistedString is lifted as a synthetic graph input.
print(g(torch.tensor(3.0)))
```

### Members

Symbolic opaque objects are treated as black boxes -- we don't inspect or guard
on their internals unless told to -- which lets custom ops do whatever they want
with the object under the hood. But user code often needs to access attributes
or call methods on these objects. The `members` registration tells Dynamo which
members are safe to access and how to behave when tracing them:

- **`MemberType.USE_REAL`**: Evaluate the member with the real object at trace
  time and treat the result as a constant. Use this when the member returns a
  constant or calls into C++ that Dynamo can't trace. For soundness, guard on
  the object by implementing `guard_fn`.
- **`MemberType.INLINED`**: Trace into the method and inline its operations into
  the FX graph, the same way Dynamo handles regular Python method calls.
- **Unregistered members**: Accessing an unregistered member raises a hard error
  during tracing.

```{code-cell}
import random
from torch._library.opaque_object import MemberType

class RNGStateWithMembers(CustomClassBase):
    def __init__(self, seed):
        self.seed = seed
        self.rng = random.Random(self.seed)

    def get_seed(self):
        return self.seed

    def noisy_inject(self, x):
        return torch.ops.mylib.noisy_inject(x, self)

@torch.library.custom_op("mylib::noisy_inject", mutates_args=())
def noisy_inject(x: torch.Tensor, rng: RNGStateWithMembers) -> torch.Tensor:
    torch.manual_seed(rng.seed)
    return x + torch.randn_like(x)

@noisy_inject.register_fake
def _(x, rng):
    return torch.empty_like(x)

register_custom_class(
    RNGStateWithMembers,
    typ="symbolic",
    guard_fn=lambda obj: [obj.seed],
    members={
        "get_seed": MemberType.USE_REAL,
        "noisy_inject": MemberType.INLINED,
    },
)

@torch.compile(fullgraph=True, backend=print_graph)
def foo(rng_state, x):
    seed1 = rng_state.get_seed()
    x = rng_state.noisy_inject(x)
    x = x * (seed1 + 1)
    return x

print(foo(RNGStateWithMembers(0), torch.tensor([3.0])))
```

Notice that `get_seed()` (`USE_REAL`) is evaluated at trace time and its value
is folded into a constant, while `noisy_inject` (`INLINED`) appears as actual op
calls in the graph that execute at runtime.

## Current users of opaque objects

**Symbolic types:**

- **DeviceMesh**: Making `DeviceMesh` an opaque object simplifies Dynamo
  tracing. Instead of capturing closures that close over the mesh, we can insert
  `DTensor.from_local` directly into the graph and pass the `DeviceMesh` as a
  graph input.
- **ProcessGroup**: Making `ProcessGroup` an opaque object allows passing it
  directly into custom ops, instead of looking up a PG string name and mapping
  it back to a `ProcessGroup`. This helps enable compile-on-one-rank work, since
  the graph no longer has hard-coded PG names.
- **`@leaf_function`**: Wraps arbitrary callables that should be treated as
  opaque leaf nodes during tracing.

**Constant types:**

- **Placements** (e.g. `Shard`, `Replicate`): DTensor placement descriptors that
  are immutable and constant during tracing.
- **Enums**: Similar constant configuration values.

**Hoisted constant types:**

- String arguments to custom ops that change frequently and would otherwise
  cause recompilations.

## Opaque objects vs. PyTree-able types

Opaque objects are treated as pytree leaves, which means any tensors stored
inside an opaque object are invisible to the compiler infrastructure. This leads
to behaviors like no dispatching based on the inner tensor's dispatch keys and
no CUDA graph integration. Tensors are fundamentally *not* opaque to PyTorch, so
storing them inside an opaque object -- which is meant to be opaque -- creates a
mismatch.

If you want to store tensors in a data structure, registering that structure as
a pytree node is the better solution.
