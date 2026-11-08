import numpy as np
import pytest
from scripts.validation import rigid_freeflight_metrics as metrics
from scripts.validation.rigid_freeflight import author, BODY, TABLE
from scripts.validation.sponge_freeflight_metrics import force_for_step


def ideal():
    dt = float(np.float32(.0005)); time = np.arange(401)*dt
    force = np.asarray([force_for_step(i) for i in range(401)], dtype=np.float32)
    v = np.zeros((401, 6)); v[1:, :3] = np.cumsum((force[1:]/metrics.MASS_KG+[0., 0., -9.81])*dt, axis=0)
    q = np.tile([*metrics.CENTER_M, 0., 0., 0., 1.], (401, 1))
    q[1:, :3] += np.cumsum(.5*(v[:-1, :3]+v[1:, :3])*dt, axis=0)
    pos = np.zeros((401, 3), np.float32); pos[1:] = q[:-1, :3]
    return {'time_s': time, 'native_step': np.arange(401)+2, 'body_q': q, 'body_qd': v,
            'force_world_n': force, 'force_position_world_m': pos, 'torque_world_nm': np.zeros((401, 3))}


def test_rigid_analytic_momentum_and_writable_COM():
    raw = ideal(); checks, measured, derived = metrics.evaluate(raw, metrics.MASS_KG, [0., 0., 0.], -9.81)
    assert all(checks.values()), checks
    assert np.isclose(measured['actual_force_pulse_impulse_ns'][0], .02)
    assert measured['rigid_only_no_fem_mass_approximation'] is True
    assert derived['momentum_ns'].dtype == np.float64


@pytest.mark.parametrize('kind', ['lost_impulse', 'release_loss', 'bad_gravity', 'rotation', 'frozen_pose'])
def test_rigid_physical_counterexamples(kind):
    raw = ideal()
    if kind == 'lost_impulse': raw['body_qd'][:, 0] *= .5; key = 'horizontal_pulse_momentum'
    elif kind == 'release_loss': raw['body_qd'][201:, 0] *= .8; key = 'release_horizontal_momentum'
    elif kind == 'bad_gravity': raw['body_qd'][:, 2] *= 1.5; key = 'gravity_momentum_all_windows'
    elif kind == 'rotation': raw['body_qd'][200, 3] = 1.; key = 'no_unforced_rotation'
    else:
        raw['body_q'][:, :3] = raw['body_q'][0, :3]
        raw['force_position_world_m'][1:] = raw['body_q'][:-1, :3]
        key = 'actual_horizontal_com_motion'
    checks, _, _ = metrics.evaluate(raw, metrics.MASS_KG, [0., 0., 0.], -9.81)
    assert not checks[key]


@pytest.mark.parametrize('kind', ['missing_step', 'wrong_COM_force', 'wrong_mass', 'force_outside_pulse'])
def test_rigid_bad_archive_rejected(kind):
    raw = ideal(); mass = metrics.MASS_KG
    if kind == 'missing_step': raw['native_step'][2] += 1
    elif kind == 'wrong_COM_force': raw['force_position_world_m'][101, 1] += .1
    elif kind == 'wrong_mass': mass *= 2.
    else: raw['force_world_n'][100, 0] = .4
    with pytest.raises(ValueError): metrics.evaluate(raw, mass, [0., 0., 0.], -9.81)


def test_rigid_author_exact_mass_box_inertia_and_no_bond():
    Usd = pytest.importorskip('pxr.Usd'); UsdPhysics = pytest.importorskip('pxr.UsdPhysics')
    stage = Usd.Stage.CreateInMemory(); author(stage)
    prim = stage.GetPrimAtPath(BODY); mass = UsdPhysics.MassAPI(prim)
    assert np.isclose(mass.GetMassAttr().Get(), .0792)
    assert np.allclose(mass.GetDiagonalInertiaAttr().Get(), metrics.INERTIA_KG_M2)
    assert np.allclose(mass.GetCenterOfMassAttr().Get(), 0.)
    assert UsdPhysics.RigidBodyAPI(prim).GetKinematicEnabledAttr().Get() is False
    assert prim.GetAttribute('physxRigidBody:solverPositionIterationCount').Get() == 32
    assert prim.GetAttribute('physxRigidBody:solverVelocityIterationCount').Get() == 1
    assert prim.GetAttribute('physxRigidBody:linearDamping').Get() == 0.
    assert np.isclose(prim.GetAttribute('physxRigidBody:angularDamping').Get(), .05)
    assert not UsdPhysics.RigidBodyAPI(stage.GetPrimAtPath(TABLE))
    assert not any(p.IsA(UsdPhysics.Joint) or p.GetTypeName().endswith('Attachment') for p in stage.Traverse())
    assert sorted(str(p.GetPath()) for p in stage.Traverse() if p.HasAPI(UsdPhysics.CollisionAPI)) == sorted([BODY, TABLE])
