"""Readback and impulse-unit controls without a native SimulationApp."""
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.diagnostics.probe_workroom_base import MOUNT_PATH, base_contacts, native_static_readback, stage_relationships


BASE = '/World/Robot/fr3_link0'


def contact(a=BASE, b=MOUNT_PATH):
    return {'actor0': a, 'actor1': b, 'impulse_magnitude_sum_ns': 2.,
            'impulse_vector_ns': [0., 0., -2.], 'min_separation_m': -.00001}


def test_conversion_uses_physics_substep_and_keeps_native_impulse():
    row = contact()
    before = deepcopy(row)
    result = base_contacts([row, contact(BASE + '1', MOUNT_PATH), contact(BASE, MOUNT_PATH + '_aux')], BASE, .0005)
    assert len(result) == 1
    assert result[0]['average_normal_force_magnitude_sum_n'] == 4000.
    assert result[0]['average_normal_force_vector_n'] == [0., 0., -4000.]
    assert result[0]['impulse_magnitude_sum_ns'] == 2.
    assert row == before
    assert base_contacts([contact(MOUNT_PATH, BASE)], BASE, .001)[0]['average_normal_force_magnitude_sum_n'] == 2000.


@pytest.mark.parametrize('dt', [0., -.01, np.inf, np.nan])
def test_invalid_timestep_rejected(dt):
    with pytest.raises(ValueError, match='finite and positive'):
        base_contacts([contact()], BASE, dt)


@pytest.mark.parametrize('impulse', [-1., np.nan, np.inf])
def test_invalid_native_impulses_are_not_sanitized(impulse):
    row = contact()
    row['impulse_magnitude_sum_ns'] = impulse
    with pytest.raises(ValueError, match='Invalid native'):
        base_contacts([row], BASE, .0005)


@pytest.mark.parametrize('fixed', [False, True])
def test_reads_both_metatypes_without_conflating_false_with_collection_failure(fixed):
    meta = SimpleNamespace(fixed_base=fixed, link_names=['fr3_link0'], link_parents=[''], dof_names=[])
    view = SimpleNamespace(shared_metatype=meta, get_metatype=lambda index: meta, count=1,
                           prim_paths=[BASE], get_masses=lambda: np.array([[2.3]]),
                           get_root_transforms=lambda: np.array([[0., 0., .78, 0., 0., 0., 1.]]),
                           get_root_velocities=lambda: np.zeros((1, 6)))
    env = SimpleNamespace(_robot_view=view, body_names=['fr3_link0'],
                          scene_info={'articulation_path': BASE, 'robot_link_paths': {'fr3_link0': BASE}})
    result = native_static_readback(env)
    assert result['shared_metatype_fixed_base'] is fixed
    assert result['get_metatype_0_fixed_base'] is fixed
    assert result['native_robot_link_masses_kg'] == {'fr3_link0': 2.3}


def test_usd_relationship_readback_distinguishes_world_fixed_joint():
    pytest.importorskip('pxr')
    from pxr import Sdf, Usd, UsdGeom, UsdPhysics
    stage = Usd.Stage.CreateInMemory()
    base = UsdGeom.Xform.Define(stage, BASE)
    UsdPhysics.ArticulationRootAPI.Apply(base.GetPrim())
    joint = UsdPhysics.FixedJoint.Define(stage, '/World/Robot/fixed_world')
    joint.CreateBody1Rel().SetTargets([Sdf.Path(BASE)])
    env = SimpleNamespace(stage=stage, scene_info={'robot_path': '/World/Robot'})
    result = stage_relationships(env)
    assert result['articulation_roots'][0]['path'] == BASE
    actual = result['robot_related_joints'][0]
    assert actual['type'] == 'PhysicsFixedJoint'
    assert actual['body0_is_world'] and not actual['body1_is_world']
    assert actual['body1'] == [BASE]
