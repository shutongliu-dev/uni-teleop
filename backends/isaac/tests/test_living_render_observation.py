"""Ensure live renderer observation cannot feed USD state into the solver."""
import sys
from types import ModuleType, SimpleNamespace
import numpy as np
import pytest
from pxr import Usd, UsdGeom, Gf
from scripts.media.render_living_task import body_states


def setup(monkeypatch, playing):
    omni, physx, fabric = [ModuleType(n) for n in ('omni', 'omni.physx', 'omni.physxfabric')]
    omni.physx, omni.physxfabric = physx, fabric
    calls = []
    velocity = np.array([[.1, .2, .3, 1., 2., 3.]])
    physx.get_physx_simulation_interface = lambda: SimpleNamespace(fetch_results=lambda: calls.append('fetch'))
    physx.get_physx_interface = lambda: SimpleNamespace(update_transformations=lambda *args: calls.append('update'))
    def save():
        calls.append('save')
        velocity[:] = 0  # Reproduce the observed native export side effect.
    fabric.get_physx_fabric_interface = lambda: SimpleNamespace(force_update=lambda *args: None, save_to_usd=save)
    for mod in (omni, physx, fabric): monkeypatch.setitem(sys.modules, mod.__name__, mod)
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(stage, '/Body').AddTranslateOp().Set(Gf.Vec3d(1, 2, 3))
    view = SimpleNamespace(get_transforms=lambda: np.array([[1, 2, 3, 0, 0, 0, 1]]), get_velocities=lambda: velocity)
    world = SimpleNamespace(is_playing=lambda: playing, get_physics_dt=lambda: 1/960, current_time=1.)
    return (stage, {'bodies': [{'path': '/Body'}]}, {'/Body': view}, world), calls


def test_live_read_never_publishes_usd(monkeypatch):
    args, calls = setup(monkeypatch, True)
    row = body_states(*args)['/Body']
    assert calls == ['fetch']
    assert row['velocity_m_s'] == [.1, .2, .3]
    assert not row['usd_published']


def test_live_export_is_rejected_before_writes(monkeypatch):
    args, calls = setup(monkeypatch, True)
    with pytest.raises(RuntimeError, match='pausing'):
        body_states(*args, publish_paused=True)
    assert calls == ['fetch']


def test_paused_export_keeps_pre_export_native_velocity(monkeypatch):
    args, calls = setup(monkeypatch, False)
    row = body_states(*args, publish_paused=True)['/Body']
    assert calls == ['fetch', 'update', 'save']
    assert row['angular_velocity_rad_s'] == [1., 2., 3.]
    assert row['usd_native_position_error_m'] == 0
