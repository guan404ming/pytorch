# mypy: allow-untyped-defs
"""Lower FlexGEMM grouped local reductions into QuACK/CuTeDSL epilogues.

FlexGEMM recognizes a narrow local-reduction contract inside the GEMM output
tile: an epilogue reshapes the accumulator to expose contiguous groups along M
or N, then reduces only that grouped dimension. N-axis groups that fit in one
32-lane TensorSSA fragment lower as ordinary in-fragment TensorSSA reductions;
larger N groups produce TensorSSA partials that QuACK combines physically.
M-axis groups currently always use QuACK's physical row-lane/warp combine path,
even when the group is small enough to fit in one fragment. Inductor owns the
FX pattern matching and output contracts; these helpers describe the supported
TensorSSA shapes and generated combine/finalize expressions QuACK needs.
"""

import dataclasses
import math
import operator
from typing import Any

import torch
from torch._inductor import inductor_prims
from torch._inductor.codegen.cutedsl.cutedsl_op_overrides import CuteDSLOpOverrides
from torch._inductor.kernel.flex_gemm.constraints import (
    grouped_reduce_dims_match,
    LOCAL_REDUCE_EXPLICIT_DTYPE_ERROR,
    LOCAL_REDUCE_INNERMOST_GROUPED_DIM_ERROR,
    LOCAL_REDUCE_MIXED_GROUPED_LAYOUT_ERROR,
    LOCAL_REDUCE_PARTIAL_OUTPUT_CONTRACT_ERROR,
    validate_local_reduce_tensorssa_group_size,
)
from torch._inductor.shape_propagation import get_broadcasted_shape
from torch._inductor.virtualized import V
from torch.utils._ordered_set import OrderedSet


def normalize_shape(shape: Any) -> Any:
    return tuple(shape) if isinstance(shape, (list, tuple, torch.Size)) else shape


@dataclasses.dataclass(frozen=True)
class GroupedTensorSSALayout:
    """Describe a grouped M/N TensorSSA view inside the generated epilogue.

    Attributes:
        axis: GEMM output dimension being grouped: 0 for M, 1 for N.
        group_size: Number of contiguous output elements reduced as one group.
    """

    axis: int
    group_size: int

    @property
    def reduce_dims(self) -> tuple[int, ...]:
        return (-1, 2) if self.axis == 1 else (-2, 1)

    @property
    def fragment_group_size(self) -> int:
        return min(self.group_size, 32)

    @property
    def tensorssa_shape(self) -> str:
        if self.axis == 1:
            return f"((1, {self.fragment_group_size}, {32 // self.fragment_group_size}), 1, 1)"
        return (
            f"(({self.fragment_group_size}, 1, {32 // self.fragment_group_size}), 1, 1)"
        )

    @property
    def keepdim_shape(self) -> str:
        return f"((1, 1, {32 // self.fragment_group_size}), 1, 1)"

    @property
    def needs_physical_combine(self) -> bool:
        return self.axis == 0 or self.group_size > self.fragment_group_size

    @property
    def reduction_profile(self) -> str:
        if self.axis == 1:
            return "((None, 1, None), 1, 1)"
        return "((1, None, None), 1, 1)"


def grouped_tensor_layout(shape: Any) -> GroupedTensorSSALayout | None:
    """Recognize reshapes that encode a grouped M/N local-reduction domain."""
    shape = normalize_shape(shape)
    if not isinstance(shape, tuple) or len(shape) not in (3, 4):
        return None
    if isinstance(shape[-1], int) and shape[-1] > 0 and shape[-2] == -1:
        return GroupedTensorSSALayout(axis=1, group_size=shape[-1])
    if shape[-3] == -1 and isinstance(shape[-2], int) and shape[-2] > 0:
        return GroupedTensorSSALayout(axis=0, group_size=shape[-2])
    return None


def _cute_op_name(target: Any) -> str | None:
    if isinstance(target, torch._ops.OpOverload):
        op_name = target.overloadpacket.__name__
    elif isinstance(target, str):
        op_name = target
    else:
        op_name = target.__name__ if callable(target) else None
    return "truediv" if op_name == "div" else op_name


def partial_vector_reduction_error() -> NotImplementedError:
    return NotImplementedError(LOCAL_REDUCE_PARTIAL_OUTPUT_CONTRACT_ERROR)


@dataclasses.dataclass(frozen=True)
class GroupedTensorSSAInfo:
    """Carry grouped-layout provenance for an FX node during lowering.

    Attributes:
        layout: TensorSSA grouping that still applies to the node's value.
    """

    layout: GroupedTensorSSALayout

    @property
    def axis(self) -> int:
        return self.layout.axis

    @property
    def group_size(self) -> int:
        return self.layout.group_size


@dataclasses.dataclass(frozen=True)
class FlexGemmReductionSpec:
    """Store the CuTe/QuACK code-generation recipe for a local reduction.

    Attributes:
        cute_op: CuTe reduction enum used for in-fragment TensorSSA reductions.
        init_val: Initial value expression for the TensorSSA reducer.
        combine_expr: QuACK physical-combine expression using ``lhs`` and ``rhs``.
        scale_by_group_size: Whether TensorSSA-only reductions divide by group size.
        finalize_expr: QuACK finalizer expression using ``value`` after combining.
    """

    cute_op: str
    init_val: str
    combine_expr: str
    scale_by_group_size: bool = False
    finalize_expr: str = "value"


FlexGemmPhysicalReduction = FlexGemmReductionSpec


FLEX_GEMM_REDUCTIONS = {
    "sum": FlexGemmReductionSpec("cute.ReductionOp.ADD", "0.0", "lhs + rhs"),
    "mean": FlexGemmReductionSpec(
        "cute.ReductionOp.ADD",
        "0.0",
        "lhs + rhs",
        scale_by_group_size=True,
        finalize_expr="value / {group_size}.0",
    ),
    "prod": FlexGemmReductionSpec("cute.ReductionOp.MUL", "1.0", "lhs * rhs"),
    "max": FlexGemmReductionSpec(
        "cute.ReductionOp.MAX", 'float("-inf")', "cute.arch.fmax(lhs, rhs)"
    ),
    "min": FlexGemmReductionSpec(
        "cute.ReductionOp.MIN", 'float("inf")', "cute.arch.fmin(lhs, rhs)"
    ),
}


FLEX_GEMM_POINTWISE_OP_NAMES = frozenset(
    OrderedSet(
        [
            "_to_copy",
            "clamp",
            "clamp_max",
            "clamp_min",
            "convert_element_type",
            "mx_e8m0_scale",
            "nvfp4_e4m3_scale",
        ]
    )
)

PYTHON_POINTWISE_TARGETS = frozenset(
    OrderedSet(
        [
            operator.add,
            operator.sub,
            operator.mul,
            operator.truediv,
            operator.neg,
            operator.eq,
            operator.ne,
            operator.lt,
            operator.le,
            operator.gt,
            operator.ge,
        ]
    )
)


def _cute_scale_expr(
    op_name: str, source: Any, max_power: Any = 8, *, tensorssa: bool = False
) -> str:
    """Render scale encoders as numeric CuTeDSL expressions before output casting."""
    if op_name == "mx_e8m0_scale":
        if tensorssa:
            scale_exp = f"(cute.math.floor(cute.math.log2({source})) - {max_power})"
            return (
                "cute.math.exp2("
                f"cute.where({scale_exp} < -127.0, -127.0, "
                f"cute.where({scale_exp} > 128.0, 128.0, {scale_exp}))"
                ")"
            )
        exponent = (
            f"(((cutlass.Float32({source}).bitcast(cutlass.Int32) >> 23) "
            f"& 0xFF) - 127 - {max_power})"
        )
        clamped = f"cutlass.max(cutlass.min({exponent}, 128), -127)"
        return f"cutlass.Float32(cute.math.exp2(cutlass.Float32({clamped})))"
    scale = f"({source} / 6.0)"
    if tensorssa:
        return (
            f"cute.where({scale} < 0.015625, 0.015625, "
            f"cute.where({scale} > 448.0, 448.0, {scale}))"
        )
    return f"cutlass.Float32(cutlass.max(cutlass.min({scale}, 448.0), 0.015625))"


def _cute_scale_call(
    op_name: str, args: tuple[Any, ...], kwargs: dict[str, Any]
) -> Any:
    """Lower FlexGEMM scale custom ops for TensorSSA values and scalar finalizers."""
    if len(args) != 1:
        raise NotImplementedError(f"unsupported FlexGEMM epilogue op: {op_name}")
    if op_name == "nvfp4_e4m3_scale" and kwargs:
        raise NotImplementedError(f"unsupported FlexGEMM epilogue op kwargs: {op_name}")
    max_power = kwargs.get("max_power", 8)
    if op_name == "mx_e8m0_scale" and not isinstance(max_power, (int, float)):
        raise NotImplementedError("FlexGEMM mx_e8m0_scale requires static max_power")
    source = args[0]
    cse_var = CuteDSLOpOverrides._get_cse_var(source)
    expr = _cute_scale_expr(op_name, source, max_power, tensorssa=cse_var is not None)
    if cse_var is None:
        return expr
    return V.kernel.cse.generate(
        V.kernel.body,
        expr,
        bounds=cse_var.bounds,
        dtype=cse_var.dtype,
        shape=cse_var.shape,
    )


def _cute_arg(value: Any, env: dict[torch.fx.Node, Any]) -> Any:
    """Translate FX node references and constants into CuTeDSL epilogue values."""
    if isinstance(value, torch.fx.Node):
        if value in env:
            return env[value]
        raise NotImplementedError(
            f"unsupported FlexGEMM epilogue dependency: {value.format_node()}"
        )
    if isinstance(value, float) and not math.isfinite(value):
        if math.isnan(value):
            return 'float("nan")'
        return 'float("inf")' if value > 0 else 'float("-inf")'
    if isinstance(
        value,
        (
            int,
            float,
            bool,
            torch.dtype,
            torch.device,
            torch.layout,
            torch.memory_format,
        ),
    ):
        return value
    if isinstance(value, (tuple, list)):
        return type(value)(_cute_arg(item, env) for item in value)
    raise NotImplementedError(f"unsupported FlexGEMM epilogue constant: {value!r}")


def _generate_like(
    kernel: Any, expr: str, ref: Any, shape_ref: Any | None = None
) -> Any:
    """Emit CuTeDSL while preserving dtype and shape metadata from references."""
    if shape_ref is None:
        shape_ref = ref
    return kernel.cse.generate(
        kernel.body,
        expr,
        dtype=getattr(ref, "dtype", None),
        shape=getattr(shape_ref, "shape", None),
    )


def _keepdim_and_broadcast(
    kernel: Any, reduced: Any, info: GroupedTensorSSAInfo, source: Any
) -> tuple[Any, Any]:
    """Materialize keepdim and store-shaped forms of a grouped reduction."""
    keepdim_source = _generate_like(
        kernel, f"{reduced}.reshape({info.layout.keepdim_shape})", reduced
    )
    return keepdim_source, _generate_like(
        kernel, f"{keepdim_source}.broadcast_to({source}.shape)", keepdim_source, source
    )


def _cute_call(target: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
    op_name = _cute_op_name(target)
    if op_name in {
        "all",
        "any",
        "argmax",
        "argmin",
        "mean",
        "amin",
        "prod",
        "std",
        "var",
    }:
        raise NotImplementedError(
            "unsupported FlexGEMM epilogue local reduction without a grouped "
            f"TensorSSA lowering: {target}"
        )
    if op_name is None:
        raise NotImplementedError(f"unsupported FlexGEMM epilogue op: {target}")
    if op_name in ("mx_e8m0_scale", "nvfp4_e4m3_scale"):
        return _cute_scale_call(op_name, args, kwargs)
    try:
        op = getattr(V.get_ops_handler(), op_name)
    except AttributeError:
        raise NotImplementedError(
            f"unsupported FlexGEMM epilogue op: {target}"
        ) from None
    return op(*args, **kwargs)


def _local_reduce_store_arg(
    value: Any, env: dict[torch.fx.Node, Any], sources: dict[torch.fx.Node, Any]
) -> Any:
    if isinstance(value, torch.fx.Node) and value in sources:
        return sources[value]
    if isinstance(value, (tuple, list)):
        return type(value)(
            _local_reduce_store_arg(item, env, sources) for item in value
        )
    return _cute_arg(value, env)


def has_local_reduce_store_source(
    value: Any, sources: dict[torch.fx.Node, Any]
) -> bool:
    if isinstance(value, torch.fx.Node):
        return value in sources
    if isinstance(value, (tuple, list)):
        return any(has_local_reduce_store_source(item, sources) for item in value)
    return False


def tensor_meta_shape(node: torch.fx.Node) -> tuple[Any, ...] | None:
    """Return fake-tensor shape metadata when the FX value is tensor-like."""
    meta = node.meta.get("val")
    if isinstance(meta, torch.Tensor):
        return tuple(meta.shape)
    return None


def canonical_shape(shape: tuple[Any, ...]) -> tuple[str, ...]:
    """Canonicalize tensor metadata shapes for symbolic broadcast comparisons."""
    return tuple(str(dim) for dim in shape)


def shapes_match(lhs: tuple[Any, ...], rhs: tuple[Any, ...]) -> bool:
    """Compare shape metadata using Inductor's symbolic-shape string convention."""
    return canonical_shape(lhs) == canonical_shape(rhs)


def is_broadcastable_to_shape(
    shape: tuple[Any, ...], output_shape: tuple[Any, ...]
) -> bool:
    """Return whether Inductor shape propagation expands a tensor to output shape."""
    canonical_output_shape = canonical_shape(output_shape)
    try:
        broadcast_shape = get_broadcasted_shape(
            canonical_shape(shape), canonical_output_shape
        )
    except AssertionError:
        return False
    if broadcast_shape is None:
        return False
    return tuple(broadcast_shape) == canonical_output_shape


def node_preserves_tensor_shapes(node: torch.fx.Node) -> bool:
    """Reject pointwise broadcasts that cannot preserve a grouped TensorSSA input."""
    output_shape = tensor_meta_shape(node)
    if output_shape is None:
        return False
    has_matching_tensor_input = False
    for input_node in iter_fx_node_inputs((node.args, node.kwargs)):
        input_shape = tensor_meta_shape(input_node)
        if not input_shape:
            continue
        if shapes_match(input_shape, output_shape):
            has_matching_tensor_input = True
        elif not is_broadcastable_to_shape(input_shape, output_shape):
            return False
    return has_matching_tensor_input


def is_pointwise_node(node: torch.fx.Node) -> bool:
    if node.op == "call_method":
        return True
    if node.op != "call_function":
        return False
    if isinstance(node.target, torch._ops.OpOverload):
        return (
            torch.Tag.pointwise in node.target.tags
            or _cute_op_name(node.target) in FLEX_GEMM_POINTWISE_OP_NAMES
        )
    return (
        node.target in PYTHON_POINTWISE_TARGETS
        or _cute_op_name(node.target) in FLEX_GEMM_POINTWISE_OP_NAMES
    )


def is_shape_preserving_pointwise_node(node: torch.fx.Node) -> bool:
    return is_pointwise_node(node) and node_preserves_tensor_shapes(node)


def iter_fx_node_inputs(value: Any):
    """Yield FX node inputs nested in args/kwargs-style containers."""
    if isinstance(value, torch.fx.Node):
        yield value
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from iter_fx_node_inputs(item)
    elif isinstance(value, dict):
        for item in value.values():
            yield from iter_fx_node_inputs(item)


def propagate_grouped_tensorssa_info(
    node: torch.fx.Node, grouped_tensors: dict[torch.fx.Node, GroupedTensorSSAInfo]
) -> GroupedTensorSSAInfo | None:
    input_infos = [
        grouped_tensors[arg]
        for arg in iter_fx_node_inputs((node.args, node.kwargs))
        if arg in grouped_tensors
    ]
    if not input_infos:
        return None
    layout = input_infos[0].layout
    if any(info.layout != layout for info in input_infos):
        raise NotImplementedError(LOCAL_REDUCE_MIXED_GROUPED_LAYOUT_ERROR)
    return GroupedTensorSSAInfo(layout)


def lower_view_or_reshape(
    node: torch.fx.Node,
    env: dict[torch.fx.Node, Any],
    kernel: Any,
    grouped_tensors: dict[torch.fx.Node, GroupedTensorSSAInfo],
    local_reduce_store_sources: dict[torch.fx.Node, Any],
    preserve_value_layout: bool = False,
) -> Any | None:
    if node.op == "call_method" and node.target in ("view", "reshape"):
        source_node = node.args[0]
        shape = node.args[1:]
    elif node.op == "call_function" and node.target in (
        torch.ops.aten.view.default,
        torch.ops.aten.reshape.default,
    ):
        source_node = node.args[0]
        shape = node.args[1]
    else:
        return None
    if (
        isinstance(source_node, torch.fx.Node)
        and source_node in local_reduce_store_sources
    ):
        local_reduce_store_sources[node] = local_reduce_store_sources[source_node]
        return _cute_arg(source_node, env)
    grouped_layout = grouped_tensor_layout(shape)
    if not isinstance(source_node, torch.fx.Node):
        return None
    if grouped_layout is None:
        if source_node in grouped_tensors:
            return _cute_arg(source_node, env)
        return None
    validate_local_reduce_tensorssa_group_size(
        grouped_layout.axis, grouped_layout.group_size
    )
    source = _cute_arg(source_node, env)
    grouped_tensors[node] = GroupedTensorSSAInfo(grouped_layout)
    if preserve_value_layout:
        return source
    return _generate_like(
        kernel, f"{source}.reshape({grouped_layout.tensorssa_shape})", source
    )


def reduction_from_node(node: torch.fx.Node) -> tuple[Any, Any, Any, Any, str] | None:
    if node.op == "call_method" and node.target in (
        "sum",
        "mean",
        "prod",
        "amax",
        "amin",
    ):
        input_node = node.args[0]
        dim = node.args[1] if len(node.args) > 1 else node.kwargs.get("dim")
        keepdim = (
            node.args[2] if len(node.args) > 2 else node.kwargs.get("keepdim", False)
        )
        dtype = node.args[3] if len(node.args) > 3 else node.kwargs.get("dtype")
        reduction_type = {"amax": "max", "amin": "min"}.get(node.target, node.target)
        return input_node, dim, keepdim, dtype, reduction_type
    if node.op == "call_function" and node.target in (
        torch.ops.aten.sum.dim_IntList,
        torch.ops.aten.mean.dim,
        torch.mean,
    ):
        input_node = node.args[0]
        dim = node.args[1] if len(node.args) > 1 else node.kwargs.get("dim")
        keepdim = (
            node.args[2] if len(node.args) > 2 else node.kwargs.get("keepdim", False)
        )
        dtype = node.args[3] if len(node.args) > 3 else node.kwargs.get("dtype")
        reduction_type = (
            "mean" if node.target in (torch.ops.aten.mean.dim, torch.mean) else "sum"
        )
        return input_node, dim, keepdim, dtype, reduction_type
    if node.op == "call_function" and node.target == torch.ops.aten.prod.dim_int:
        input_node = node.args[0]
        dim = node.args[1] if len(node.args) > 1 else node.kwargs.get("dim")
        keepdim = (
            node.args[2] if len(node.args) > 2 else node.kwargs.get("keepdim", False)
        )
        dtype = node.args[3] if len(node.args) > 3 else node.kwargs.get("dtype")
        return input_node, dim, keepdim, dtype, "prod"
    if node.op == "call_function" and node.target in (
        torch.ops.aten.amax.default,
        torch.amax,
    ):
        input_node = node.args[0]
        dim = node.args[1] if len(node.args) > 1 else node.kwargs.get("dim")
        keepdim = (
            node.args[2] if len(node.args) > 2 else node.kwargs.get("keepdim", False)
        )
        return input_node, dim, keepdim, None, "max"
    if node.op == "call_function" and node.target in (
        torch.ops.aten.amin.default,
        torch.amin,
    ):
        input_node = node.args[0]
        dim = node.args[1] if len(node.args) > 1 else node.kwargs.get("dim")
        keepdim = (
            node.args[2] if len(node.args) > 2 else node.kwargs.get("keepdim", False)
        )
        return input_node, dim, keepdim, None, "min"
    return None


def moment_reduction_from_node(
    node: torch.fx.Node,
) -> tuple[Any, Any, Any, Any, str] | None:
    if node.op == "call_method" and node.target in ("var", "std"):
        input_node = node.args[0]
        dim = node.args[1] if len(node.args) > 1 else node.kwargs.get("dim")
        if "correction" in node.kwargs:
            correction = node.kwargs["correction"]
        elif len(node.args) > 2:
            correction = 1 if node.args[2] else 0
        else:
            correction = 1 if node.kwargs.get("unbiased", True) else 0
        keepdim = (
            node.args[3] if len(node.args) > 3 else node.kwargs.get("keepdim", False)
        )
        return input_node, dim, correction, keepdim, node.target
    if node.op == "call_function" and node.target in (
        torch.ops.aten.var.dim,
        torch.ops.aten.std.dim,
    ):
        input_node = node.args[0]
        dim = node.args[1] if len(node.args) > 1 else node.kwargs.get("dim")
        unbiased = (
            node.args[2] if len(node.args) > 2 else node.kwargs.get("unbiased", True)
        )
        keepdim = (
            node.args[3] if len(node.args) > 3 else node.kwargs.get("keepdim", False)
        )
        reduction_type = "std" if node.target is torch.ops.aten.std.dim else "var"
        return input_node, dim, 1 if unbiased else 0, keepdim, reduction_type
    if node.op == "call_function" and node.target in (
        torch.ops.aten.var.correction,
        torch.ops.aten.std.correction,
    ):
        input_node = node.args[0]
        dim = node.args[1] if len(node.args) > 1 else node.kwargs.get("dim")
        correction = node.kwargs.get(
            "correction", node.args[2] if len(node.args) > 2 else 1
        )
        keepdim = node.kwargs.get(
            "keepdim", node.args[3] if len(node.args) > 3 else False
        )
        reduction_type = (
            "std" if node.target is torch.ops.aten.std.correction else "var"
        )
        return input_node, dim, correction, keepdim, reduction_type
    return None


def unsupported_reduction_from_node(node: torch.fx.Node) -> str | None:
    target = node.target
    if node.op == "call_method" and node.target in (
        "all",
        "any",
        "argmax",
        "argmin",
    ):
        return str(node.target)
    if node.op == "call_function" and target in (
        torch.ops.aten.all.dim,
        torch.ops.aten.all.dims,
        torch.ops.aten.all.default,
        torch.ops.aten.any.dim,
        torch.ops.aten.any.dims,
        torch.ops.aten.any.default,
        torch.ops.aten.argmax.default,
        torch.ops.aten.argmin.default,
    ):
        return str(target)
    return None


def lower_full_scalar(node: torch.fx.Node) -> Any | None:
    if node.op != "call_function" or node.target is not torch.ops.aten.full.default:
        return None
    shape = normalize_shape(node.args[0])
    if shape != ():
        return None
    return node.args[1]


def lower_squeeze(
    node: torch.fx.Node,
    env: dict[torch.fx.Node, Any],
    local_reduce_store_sources: dict[torch.fx.Node, Any],
) -> Any | None:
    if node.op == "call_method" and node.target == "squeeze":
        source_node = node.args[0]
    elif node.op == "call_function" and node.target in (
        torch.ops.aten.squeeze.dim,
        torch.ops.aten.squeeze.dims,
        torch.ops.aten.squeeze.default,
    ):
        source_node = node.args[0]
    else:
        return None
    if (
        isinstance(source_node, torch.fx.Node)
        and source_node in local_reduce_store_sources
    ):
        local_reduce_store_sources[node] = local_reduce_store_sources[source_node]
        return _cute_arg(source_node, env)
    return None


def lower_getitem(
    node: torch.fx.Node,
    env: dict[torch.fx.Node, Any],
    local_reduce_store_sources: dict[torch.fx.Node, Any],
) -> Any | None:
    if node.op != "call_function" or node.target is not operator.getitem:
        return None
    source_node, index = node.args
    if not isinstance(source_node, torch.fx.Node) or not isinstance(index, int):
        return None
    source = _cute_arg(source_node, env)
    if source_node in local_reduce_store_sources:
        local_reduce_store_sources[node] = local_reduce_store_sources[source_node][
            index
        ]
    return source[index]


def lower_prepare_softmax_online(
    node: torch.fx.Node,
    env: dict[torch.fx.Node, Any],
    kernel: Any,
    grouped_tensors: dict[torch.fx.Node, GroupedTensorSSAInfo],
    local_reduce_store_sources: dict[torch.fx.Node, Any],
) -> Any | None:
    if (
        node.op != "call_function"
        or node.target is not inductor_prims.prepare_softmax_online
    ):
        return None
    input_node = node.args[0]
    dim = node.args[1] if len(node.args) > 1 else node.kwargs.get("dim")
    if not isinstance(input_node, torch.fx.Node):
        return None
    if input_node not in grouped_tensors:
        raise partial_vector_reduction_error()
    info = grouped_tensors[input_node]
    if info.layout.needs_physical_combine:
        raise NotImplementedError(
            "unsupported FlexGEMM physical local reduction: prepare_softmax_online "
            "needs a multi-value generated physical reducer"
        )
    if not grouped_reduce_dims_match(dim, info.layout.reduce_dims):
        raise NotImplementedError(
            "unsupported FlexGEMM epilogue local reduction: prepare_softmax_online "
            "currently supports only the grouped dimension"
        )
    source = _cute_arg(input_node, env)
    max_reduced = _generate_like(
        kernel,
        f'{source}.reduce(cute.ReductionOp.MAX, init_val=float("-inf"), reduction_profile={info.layout.reduction_profile})',
        source,
    )
    _, max_store = _keepdim_and_broadcast(kernel, max_reduced, info, source)
    centered = _generate_like(kernel, f"({source} - {max_store})", source)
    exp_centered = CuteDSLOpOverrides.exp(centered)
    sum_reduced = _generate_like(
        kernel,
        f"{exp_centered}.reduce(cute.ReductionOp.ADD, init_val=0.0, reduction_profile={info.layout.reduction_profile})",
        exp_centered,
    )
    _, sum_store = _keepdim_and_broadcast(kernel, sum_reduced, info, source)
    local_reduce_store_sources[node] = (max_store, sum_store)
    return max_store, sum_store


def lower_tensorssa_moment_reduce(
    node: torch.fx.Node,
    env: dict[torch.fx.Node, Any],
    kernel: Any,
    grouped_tensors: dict[torch.fx.Node, GroupedTensorSSAInfo],
    local_reduce_store_sources: dict[torch.fx.Node, Any],
) -> Any | None:
    """Lower grouped var/std as generated mean plus squared residual reductions."""
    reduction = moment_reduction_from_node(node)
    if reduction is None:
        return None
    input_node, dim, correction, keepdim, reduction_type = reduction
    if not isinstance(input_node, torch.fx.Node):
        return None
    if input_node not in grouped_tensors:
        raise partial_vector_reduction_error()
    info = grouped_tensors[input_node]
    if info.layout.needs_physical_combine:
        raise NotImplementedError(
            "unsupported FlexGEMM physical local reduction: moment reductions "
            "need matching QuACK combine/finalize callbacks"
        )
    if not grouped_reduce_dims_match(dim, info.layout.reduce_dims):
        raise NotImplementedError(
            "unsupported FlexGEMM epilogue local reduction: moment reductions "
            "currently support only the innermost grouped dimension"
        )
    if correction is None:
        correction = 1
    if not isinstance(correction, (int, float)):
        raise NotImplementedError(
            "unsupported FlexGEMM epilogue local reduction: dynamic variance correction"
        )
    denominator = info.group_size - correction
    if denominator <= 0:
        raise NotImplementedError(
            "unsupported FlexGEMM epilogue local reduction: variance correction "
            "must be smaller than the group size"
        )
    source = _cute_arg(input_node, env)
    sum_reduced = _generate_like(
        kernel,
        f"{source}.reduce(cute.ReductionOp.ADD, init_val=0.0, reduction_profile={info.layout.reduction_profile})",
        source,
    )
    mean_reduced = _generate_like(
        kernel, f"{sum_reduced} / {float(info.group_size)!r}", sum_reduced
    )
    _, mean_store = _keepdim_and_broadcast(kernel, mean_reduced, info, source)
    centered = _generate_like(kernel, f"({source} - {mean_store})", source)
    squared = _generate_like(kernel, f"({centered} * {centered})", centered)
    var_reduced = _generate_like(
        kernel,
        f"{squared}.reduce(cute.ReductionOp.ADD, init_val=0.0, reduction_profile={info.layout.reduction_profile}) / {float(denominator)!r}",
        squared,
    )
    if reduction_type == "std":
        result = _generate_like(kernel, f"cute.math.sqrt({var_reduced})", var_reduced)
    else:
        result = var_reduced
    keepdim_source, local_reduce_store_sources[node] = _keepdim_and_broadcast(
        kernel, result, info, source
    )
    if keepdim:
        return keepdim_source
    return result


def lower_tensorssa_reduce(
    node: torch.fx.Node,
    env: dict[torch.fx.Node, Any],
    kernel: Any,
    grouped_tensors: dict[torch.fx.Node, GroupedTensorSSAInfo],
    local_reduce_store_sources: dict[torch.fx.Node, Any],
    local_reduce_physical_reductions: dict[torch.fx.Node, FlexGemmPhysicalReduction]
    | None = None,
) -> Any | None:
    """Lower value reductions while deferring cross-fragment finalization to QuACK."""
    reduction = reduction_from_node(node)
    if reduction is None:
        return None
    input_node, dim, keepdim, dtype, reduction_type = reduction
    if dtype is not None:
        raise NotImplementedError(LOCAL_REDUCE_EXPLICIT_DTYPE_ERROR)
    if not isinstance(input_node, torch.fx.Node):
        return None
    if input_node not in grouped_tensors:
        raise partial_vector_reduction_error()
    info = grouped_tensors[input_node]
    if not grouped_reduce_dims_match(dim, info.layout.reduce_dims):
        raise NotImplementedError(LOCAL_REDUCE_INNERMOST_GROUPED_DIM_ERROR)
    desc = FLEX_GEMM_REDUCTIONS.get(reduction_type)
    if desc is None:
        raise NotImplementedError(
            "unsupported FlexGEMM epilogue local reduction: "
            f"reduction_type={reduction_type!r}"
        )
    source = _cute_arg(input_node, env)
    if info.layout.needs_physical_combine:
        if local_reduce_physical_reductions is not None:
            local_reduce_physical_reductions[node] = dataclasses.replace(
                desc,
                finalize_expr=desc.finalize_expr.format(group_size=info.group_size),
            )
        if info.axis == 0:
            local_reduce_store_sources[node] = source
            return source
    reduced = _generate_like(
        kernel,
        f"{source}.reduce({desc.cute_op}, init_val={desc.init_val}, reduction_profile={info.layout.reduction_profile})",
        source,
    )
    if desc.scale_by_group_size and not info.layout.needs_physical_combine:
        reduced = _generate_like(
            kernel, f"{reduced} / {float(info.group_size)!r}", reduced
        )
    keepdim_source, local_reduce_store_sources[node] = _keepdim_and_broadcast(
        kernel, reduced, info, source
    )
    if keepdim:
        return keepdim_source
    return reduced
