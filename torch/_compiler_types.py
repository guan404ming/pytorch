from typing import Literal, TypeAlias


StanceStr: TypeAlias = Literal[
    "default",
    "eager_then_compile",
    "aot_eager_then_compile",
    "force_eager",
    "eager_on_recompile",
    "fail_on_recompile",
]

SetSubgraphInputs: TypeAlias = Literal[
    "automatic",
    "automatic_with_forced_inputs",
    "flatten_manual",
    "manual",
]

StaticAddressType: TypeAlias = Literal["guarded", "unguarded"]

OptimizeDdpMode: TypeAlias = Literal[
    "ddp_optimizer",
    "python_reducer",
    "python_reducer_without_compiled_forward",
    "no_optimization",
]

HookType: TypeAlias = Literal[
    "unpack_hook",
    "tensor_pre_hook",
    "pre_hook",
    "post_hook",
    "post_acc_grad_hook",
]
