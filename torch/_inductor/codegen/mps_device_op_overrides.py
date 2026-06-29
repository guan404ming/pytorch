from __future__ import annotations

from .common import DeviceOpOverrides, register_device_op_overrides


class MPSDeviceOpOverrides(DeviceOpOverrides):
    # MPS is single-device, so the device index is ignored.
    def device_guard(self, device_idx: int | str) -> str:
        return "torch._ops.contextlib.nullcontext()"

    def set_device(self, device_idx: int | str) -> str:
        return "pass  # MPS set device"

    def kernel_driver(self) -> str:
        return """
            #include <ATen/native/mps/MetalShaderLibrary.h>
        """

    def cpp_kernel_type(self) -> str:
        return "MTLFunction_t"


register_device_op_overrides("mps", MPSDeviceOpOverrides())
