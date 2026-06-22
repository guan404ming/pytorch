# Owner(s): ["oncall: distributed"]

import os
import sys
import types
import unittest

import torch
from torch.testing._internal.common_distributed import (
    MultiProcessTestCase,
    MultiThreadedTestCase,
)
from torch.testing._internal.common_utils import run_tests, TestCase


# The filter under test lives in test/conftest.py. Make it importable whether the
# file is run under pytest (test/ already on sys.path) or directly.
_TEST_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _TEST_ROOT not in sys.path:
    sys.path.insert(0, _TEST_ROOT)

from conftest import MinGpuFilterPlugin


class _FakeMP(MultiProcessTestCase):
    pass


class _FakeMTT(MultiThreadedTestCase):
    pass


class _FakePlain(unittest.TestCase):
    pass


_UNSET = object()


def _func(min_gpus=None, required_world_size=None):
    def f():
        pass

    if min_gpus is not None:
        f._min_gpus_required = min_gpus
    if required_world_size is not None:
        f._required_world_size = required_world_size
    return f


def _item(
    *,
    cls=None,
    obj=None,
    world_size=None,
    name="test_x",
    module_name="test_foo",
    device=_UNSET,
    device_type=_UNSET,
):
    inst = types.SimpleNamespace()
    if world_size is not None:
        inst.world_size = world_size
    if device is not _UNSET:
        inst.device = device
    if device_type is not _UNSET:
        inst.device_type = device_type
    return types.SimpleNamespace(
        obj=obj if obj is not None else _func(),
        cls=cls,
        instance=inst,
        name=name,
        module=types.SimpleNamespace(__name__=module_name),
    )


class TestMinGpuFilter(TestCase):
    THRESHOLD = 4

    def setUp(self):
        self.plugin = MinGpuFilterPlugin(self.THRESHOLD)
        # Pin the accelerator token so the thread-based heuristic is
        # deterministic regardless of the host's available accelerators.
        self.plugin._accel_tokens = ("cuda",)

    def _keep(self, item):
        return self.plugin._required_gpus(item) >= self.THRESHOLD

    # --- process-based ---
    def test_process_world_size_below_threshold_dropped(self):
        item = _item(cls=_FakeMP, world_size=2, module_name="test_c10d_nccl")
        self.assertFalse(self._keep(item))

    def test_process_world_size_at_threshold_kept(self):
        item = _item(cls=_FakeMP, world_size=4, module_name="test_c10d_nccl")
        self.assertTrue(self._keep(item))

    def test_process_gpu_device_index_kept(self):
        # NCCL-style `device` returning a rank int -> treated as accelerator.
        item = _item(cls=_FakeMP, world_size=4, device=0, module_name="test_c10d_nccl")
        self.assertTrue(self._keep(item))

    # --- process-based, CPU-backed gate ---
    def test_cpu_device_world_size_4_dropped(self):
        item = _item(cls=_FakeMP, world_size=4, device=torch.device("cpu"))
        self.assertFalse(self._keep(item))

    def test_cpu_device_type_world_size_4_dropped(self):
        item = _item(cls=_FakeMP, world_size=4, device_type="cpu")
        self.assertFalse(self._keep(item))

    def test_gloo_module_world_size_4_dropped(self):
        item = _item(cls=_FakeMP, world_size=4, module_name="test_c10d_gloo")
        self.assertFalse(self._keep(item))

    def test_cpu_backed_with_decorator_kept(self):
        # An explicit decorator overrides the backend/device gate.
        item = _item(
            cls=_FakeMP,
            obj=_func(min_gpus=4),
            world_size=4,
            device=torch.device("cpu"),
        )
        self.assertTrue(self._keep(item))

    # --- thread-based (MultiThreadedTestCase) ---
    def test_thread_accelerator_variant_world_size_4_kept(self):
        item = _item(cls=_FakeMTT, world_size=4, name="test_broadcast_device_cuda")
        self.assertTrue(self._keep(item))

    def test_thread_cpu_variant_dropped(self):
        item = _item(cls=_FakeMTT, world_size=4, name="test_broadcast_device_cpu")
        self.assertFalse(self._keep(item))

    def test_thread_no_device_variant_dropped(self):
        item = _item(cls=_FakeMTT, world_size=4, name="test_expand_1d_rank_list")
        self.assertFalse(self._keep(item))

    def test_thread_accelerator_variant_world_size_2_dropped(self):
        item = _item(cls=_FakeMTT, world_size=2, name="test_broadcast_device_cuda")
        self.assertFalse(self._keep(item))

    def test_thread_with_decorator_below_threshold_dropped(self):
        item = _item(
            cls=_FakeMTT,
            obj=_func(min_gpus=2),
            world_size=4,
            name="test_broadcast_device_cuda",
        )
        self.assertFalse(self._keep(item))

    # --- plain TestCase / decorators ---
    def test_decorated_single_process_kept(self):
        item = _item(cls=_FakePlain, obj=_func(min_gpus=4))
        self.assertTrue(self._keep(item))

    def test_requires_world_size_attr_kept(self):
        item = _item(cls=_FakePlain, obj=_func(required_world_size=4))
        self.assertTrue(self._keep(item))

    def test_plain_testcase_without_signal_dropped(self):
        item = _item(cls=_FakePlain)
        self.assertFalse(self._keep(item))

    def test_no_class_without_signal_kept(self):
        # Non class-based items are kept unless a decorator says otherwise.
        item = _item(cls=None)
        self.assertTrue(self._keep(item))


if __name__ == "__main__":
    run_tests()
