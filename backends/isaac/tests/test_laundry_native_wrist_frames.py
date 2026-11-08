"""Actual pure USD frames and synthetic native poses; no Kit or GPU startup."""
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
from pxr import UsdGeom, UsdPhysics

from bench.simulation.isaacsim_backend.robot_frames import nonphysical_frame_paths
from scenes.workrooms.laundry_room.native_observer import CookedRobotGeometry, wrist_frame_bindings
from test_l_mount_camera_frames import actual_stage
from test_laundry_robot_native_runtime import geometry, towel


@pytest.fixture(params=['single', 'dual'])
def actual_laundry_frames(request):
    stage, robot, source, rigid = actual_stage.__wrapped__(SimpleNamespace(param=request.param))
    frames = nonphysical_frame_paths(stage, robot, rigid, source)
    env = SimpleNamespace(stage=stage, body_names=list(rigid),
                          scene_info={'robot_link_paths': rigid, 'robot_frame_paths': frames})
    return env, ('right',) if request.param == 'single' else ('left', 'right')


def auditor_inputs(env, sides):
    export, _, topology, _ = geometry()
    export['colliders'] = [r for r in export['colliders'] if r['body_name'] in env.body_names]
    # All other native bodies remain declared; the synthetic cloth is far away.
    poses = np.tile([10., 10., 10., 0., 0., 0., 1.], (len(env.body_names), 1))
    return export, topology, poses


def test_pure_wrist_tracks_same_native_parent_sample_and_preserves_collision_audit(actual_laundry_frames):
    env, sides = actual_laundry_frames
    export, topology, poses = auditor_inputs(env, sides)
    with pytest.raises(ValueError):  # The historical rigid-only assumption fails.
        CookedRobotGeometry(export, env.body_names, topology, {'towel': .003}, active_sides=sides)
    bindings = wrist_frame_bindings(env, sides)
    original = deepcopy(bindings)
    auditor = CookedRobotGeometry(export, env.body_names, topology, {'towel': .003},
                                 active_sides=sides, wrist_frames=bindings)
    rotation = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    for side, binding in bindings.items():
        poses[env.body_names.index(binding['body_name'])] = [1., 2., 3., 0., 0., 2**-.5, 2**-.5]
    before = poses.copy()
    result = auditor.check(towel(.014), poses)
    assert result['fully_separated'] and not result['hand_near_cloth']
    for side, binding in bindings.items():
        hand = result['hands'][side]
        expected = [1., 2., 3.] + rotation @ binding['body_from_frame'][:3, 3]
        np.testing.assert_allclose(hand['wrist_position_m'], expected, rtol=0., atol=1.e-12)
        assert hand['wrist_body_name'] == binding['body_name']
        assert hand['wrist_frame_name'] == side+'_hand_flange'
        assert side+'_hand_flange' not in env.body_names
        np.testing.assert_array_equal(binding['body_from_frame'], original[side]['body_from_frame'])
    np.testing.assert_array_equal(poses, before)
    # The USD stage remains unchanged. A later tensor sample, without render or
    # USD updates, must still move the wrist exactly with its native parent.
    for binding in bindings.values():
        poses[env.body_names.index(binding['body_name']), 0] += .123
    after = auditor.check(towel(.014), poses)
    for side in sides:
        np.testing.assert_allclose(np.asarray(after['hands'][side]['wrist_position_m']) -
                                   result['hands'][side]['wrist_position_m'], [.123, 0., 0.], atol=1.e-12)
    unchanged = lambda value: {k: v for k, v in value.items() if k != 'hands'}
    assert unchanged(result) == unchanged(after)


@pytest.mark.parametrize('fault', ['missing', 'alias', 'parent', 'reset', 'mass', 'scaled'])
def test_pure_wrist_binding_rejects_missing_ambiguous_or_nonrigid_frames(actual_laundry_frames, fault):
    env, sides = actual_laundry_frames
    name = sides[0]+'_hand_flange'
    frames = env.scene_info['robot_frame_paths']
    prim = env.stage.GetPrimAtPath(frames[name])
    if fault == 'missing':
        del frames[name]
    elif fault == 'alias':
        env.scene_info['robot_link_paths'][name] = frames[name]
    elif fault == 'parent':
        env.body_names.remove(prim.GetParent().GetName())
    elif fault == 'reset':
        UsdGeom.Xformable(prim).SetResetXformStack(True)
    elif fault == 'mass':
        UsdPhysics.MassAPI.Apply(prim)
    else:
        UsdGeom.Xformable(prim).AddScaleOp().Set((2., 1., 1.))
    with pytest.raises(ValueError):
        bindings = wrist_frame_bindings(env, sides)
        export, topology, _ = auditor_inputs(env, sides)
        CookedRobotGeometry(export, env.body_names, topology, {'towel': .003},
                            active_sides=sides, wrist_frames=bindings)


@pytest.mark.parametrize('sides', [('right',), ('left', 'right')])
def test_historical_rigid_flange_is_exactly_unchanged(sides):
    export, names, topology, poses = geometry()
    if len(sides) == 1:
        keep = [i for i, name in enumerate(names) if not name.startswith('left_')]
        names, poses = [names[i] for i in keep], poses[keep]
        export['colliders'] = [r for r in export['colliders'] if r['body_name'] in names]
    env = SimpleNamespace(body_names=names, scene_info={}, stage=None)
    assert wrist_frame_bindings(env, sides) == {}
    auditor = CookedRobotGeometry(export, names, topology, {'towel': .003}, active_sides=sides)
    result = auditor.check(towel(.014), poses)
    for side in sides:
        assert result['hands'][side] == {'wrist_body_name': side+'_hand_flange',
            'wrist_position_m': poses[names.index(side+'_hand_flange'), :3].tolist(),
            'near_towel': False, 'near_any_cloth': False}
