"""Native frontend buffer placement is separate from requested GPU dynamics."""
from types import SimpleNamespace
import sys

import numpy as np
import pytest

from bench.simulation.isaacsim_backend.env import FR3DexterousIsaacEnv


@pytest.fixture
def torch_frontend(monkeypatch):
    # The API/unit-test interpreter deliberately has no native Torch runtime.
    # Enforce the receiver's CPU buffer contract without allocating a GPU.
    def as_tensor(value, *, device):
        assert str(device) == 'cpu', 'A CUDA launch device cannot override the CPU native frontend'
        value = np.ascontiguousarray(value)
        tensor = SimpleNamespace(device='cpu', dtype=value.dtype, is_contiguous=lambda: value.flags.c_contiguous)
        tensor.contiguous = lambda: tensor
        return tensor
    monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace(as_tensor=as_tensor))


def test_cpu_readback_reset_does_not_allocate_buffers_on_requested_cuda_device(torch_frontend):
    env = FR3DexterousIsaacEnv.__new__(FR3DexterousIsaacEnv)
    env.device = 'cuda:0'
    env._native_tensor_device = 'cpu'
    indices = env._tensor(np.array([0], dtype=np.int32))
    poses = env._tensor(np.zeros((1, 7), dtype=np.float32))
    assert indices.device == poses.device == 'cpu'
    assert indices.dtype == np.int32 and poses.dtype == np.float32
    assert indices.is_contiguous() and poses.is_contiguous()


def test_explicit_native_parameter_device_stays_independent(torch_frontend):
    env = FR3DexterousIsaacEnv.__new__(FR3DexterousIsaacEnv)
    env.device = env._native_tensor_device = 'cuda:0'
    assert env._tensor(np.zeros((1, 2), dtype=np.float32), 'cpu').device == 'cpu'


def test_recreated_view_refreshes_declared_transport_before_initializing_subviews(monkeypatch):
    from bench.simulation.isaacsim_backend import readback_profile
    class StopBeforeNativeSubviews(Exception):
        pass
    observed = []
    def stop_at_guard(view, device, profile):
        observed.append((view.device, device, profile))
        raise StopBeforeNativeSubviews
    monkeypatch.setattr(readback_profile, 'tensor_pipeline', stop_at_guard)
    env = FR3DexterousIsaacEnv.__new__(FR3DexterousIsaacEnv)
    env.device = 'cuda:0'
    for device, profile in [('cuda:0', 'cuda_tensors_v1'), ('cpu', 'gpu_solver_usd_host_v1')]:
        env.native_readback_profile = profile
        env.world = SimpleNamespace(physics_sim_view=SimpleNamespace(device=device))
        with pytest.raises(StopBeforeNativeSubviews):
            env._initialize_views()
        assert observed[-1] == (device, 'cuda:0', profile)
