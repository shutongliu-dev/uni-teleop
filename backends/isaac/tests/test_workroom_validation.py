"""Negative CPU checks for native-evidence interpretation, not physics acceptance."""
from copy import deepcopy

import numpy as np
import pytest

from scripts.validation.validate_workrooms import (
    THRESHOLDS, assess_objects, assess_robot, check_window, smooth_targets,
    support_contact,
)


OBJECT = '/World/Objects/cube_floor'
SUPPORT = '/World/Room/Geometry/room_floor/Collider'


def pose(z, vz=0.):
    return {'position_m': [.3, -.7, z], 'quaternion_wxyz': [1., 0., 0., 0.],
            'linear_velocity_m_s': [0., 0., vz], 'angular_velocity_rad_s': [0., 0., 0.]}


def settled_fixture():
    initial = {'objects': {'cube_floor': pose(.25)}}
    diagnostics = {'cube_floor': {'half_extents_m': [.016, .02, .03],
        'support_top_z_m': 0., 'expected_support_path': SUPPORT}}
    rows = []
    for i, z in enumerate([.24, .20, .10, .03]+[.03]*20):
        contact = [] if i < 3 else [{'actor0': OBJECT, 'actor1': SUPPORT,
            'impulse_magnitude_sum_ns': .01, 'min_separation_m': -.0001, 'time_s': (i+1)*.02}]
        rows.append({'observation': {'objects': {'cube_floor': pose(z, -.2 if i < 3 else 0.)},
                                    'contacts': {'rigid': contact}}})
    return initial, rows, diagnostics, {'cube_floor': OBJECT}


def test_native_gravity_support_and_rest_require_independent_observed_signals():
    assert assess_objects(*settled_fixture())['passed']
    initial, rows, spec, paths = settled_fixture()
    # A cube frozen in the air must fail even if a positive contact is claimed.
    for row in rows:
        row['observation']['objects']['cube_floor'] = pose(.25)
    result = assess_objects(initial, rows, spec, paths)
    assert not result['passed']
    checks = result['objects']['cube_floor']['checks']
    assert not checks['gravity_descent']
    assert not checks['observed_downward_velocity']
    assert not checks['geometry_support_gap']


@pytest.mark.parametrize('wrong_actor', ['/World/Robot/finger', '/World/Objects/other_cube'])
def test_unrelated_contact_cannot_qualify_cube_support(wrong_actor):
    initial, rows, spec, paths = settled_fixture()
    for row in rows:
        for contact in row['observation']['contacts']['rigid']:
            contact['actor0'] = wrong_actor
    result = assess_objects(initial, rows, spec, paths)
    assert not result['passed']
    assert result['objects']['cube_floor']['support_impulse_ns'] == 0.


def test_one_contact_in_final_window_does_not_claim_continuous_support():
    initial, rows, spec, paths = settled_fixture()
    for row in rows[-THRESHOLDS['rest_window_steps']:-1]:
        row['observation']['contacts']['rigid'] = []
    result = assess_objects(initial, rows, spec, paths)
    assert not result['passed']
    assert result['objects']['cube_floor']['supported_rest_windows'] == 1


def test_penetration_uses_native_separation_and_rotated_actual_extents():
    initial, rows, spec, paths = settled_fixture()
    rows[-1]['observation']['contacts']['rigid'][0]['min_separation_m'] = -.004
    assert not assess_objects(initial, rows, spec, paths)['objects']['cube_floor']['checks']['solver_penetration']
    # A 90 degree Y rotation puts the X half extent along world Z.
    initial, rows, spec, paths = settled_fixture()
    for row in rows[4:]:
        obj = row['observation']['objects']['cube_floor']
        obj['position_m'][2] = .016
        obj['quaternion_wxyz'] = [2**-.5, 0., 2**-.5, 0.]
    assert assess_objects(initial, rows, spec, paths)['passed']


def test_contact_duration_counts_actual_unique_substeps_only():
    _, rows, _, _ = settled_fixture()
    observation = deepcopy(rows[-1]['observation'])
    observation['contacts']['rigid'] *= 2
    result = support_contact(observation, OBJECT, SUPPORT)
    assert result['contact_substeps'] == 1


def test_sampling_contract_rejects_missing_substep_even_with_finite_state():
    previous = {'native_physics_step_index': 0, 'native_world_time_s': 0., 'time_s': 0.}
    state = {'native_physics_step_index': 40, 'native_world_time_s': .02, 'time_s': .02}
    observation = {'sim_time_s': .02, 'contacts': {'errors': [], 'sampling': {
        'complete': True, 'sampled_substeps': 40, 'native_start_step_index': 0,
        'native_end_step_index': 40, 'start_sim_time_s': 0., 'end_sim_time_s': .02}}}
    check_window(observation, state, previous, .0005)
    observation['contacts']['sampling']['sampled_substeps'] = 39
    with pytest.raises(RuntimeError, match='sampling window'):
        check_window(observation, state, previous, .0005)


def test_robot_targets_alone_cannot_prove_motion():
    home = {'fr3_joint1': 0., 'right_index': 0.}
    motion = {'arm_joint': 'fr3_joint1', 'finger_joint': 'right_index',
              'offsets_rad': {'fr3_joint1': .04, 'right_index': .06}}
    spec = {'initial_joint_positions_rad': home,
            'joint_limits_rad': {name: [-1., 1.] for name in home}}
    first = {'root_pose_xyzw': [0., 0., 0., 0., 0., 0., 1.],
             'observation': {'joint_positions_rad': home, 'metrics': {'state_finite': True}, 'terminated': False}}
    rows = [{**deepcopy(first), 'phase': phase,
             'command': {'joint_positions_rad': smooth_targets(home, motion['offsets_rad'], 1.)}}
            for phase in ['outward', 'return', 'post_motion']]
    result = assess_robot(first, rows, spec, motion)
    assert not result['passed']
    assert not result['checks']['arm_moved'] and not result['checks']['finger_moved']
    rows[0]['observation']['joint_positions_rad'] = smooth_targets(home, motion['offsets_rad'], 1.)
    assert assess_robot(first, rows, spec, motion)['passed']


def test_minimum_jerk_joint_command_preserves_home_and_returns_exactly():
    home = {'fr3_joint1': 1.5, 'finger': .1, 'unchanged': -.5}
    offsets = {'fr3_joint1': .04, 'finger': .06}
    assert smooth_targets(home, offsets, 0.) == home
    assert smooth_targets(home, offsets, 1.) == {'fr3_joint1': 1.54, 'finger': .16, 'unchanged': -.5}
    values = [smooth_targets(home, offsets, t)['fr3_joint1'] for t in np.linspace(0, 1, 51)]
    assert np.all(np.diff(values) >= 0)


@pytest.mark.parametrize('damage', [None, 'moves', 'undeclared', 'missing'])
def test_stage_inventory_only_accepts_declared_kinematic_supports(damage):
    pytest.importorskip('pxr.Usd')
    from pxr import Usd, UsdGeom, UsdPhysics
    from types import SimpleNamespace
    from scripts.validation.validate_workrooms import stage_inventory

    stage = Usd.Stage.CreateInMemory()
    robot = UsdGeom.Xform.Define(stage, '/World/Robot').GetPrim()
    UsdPhysics.ArticulationRootAPI.Apply(robot)
    UsdPhysics.Scene.Define(stage, '/physicsScene')
    cube = UsdGeom.Cube.Define(stage, '/World/Objects/cube').GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(cube)
    support_path = '/World/Room/Support'
    if damage != 'missing':
        support = UsdGeom.Xform.Define(stage, support_path).GetPrim()
        rigid = UsdPhysics.RigidBodyAPI.Apply(support)
        rigid.CreateRigidBodyEnabledAttr(True)
        rigid.CreateKinematicEnabledAttr(damage != 'moves')
    env = SimpleNamespace(stage=stage, joint_names=list(range(29)),
        _robot_view=SimpleNamespace(count=1), scene_info={
            'robot_path': '/World/Robot', 'objects': {'cube': {'path': str(cube.GetPath())}},
            'kinematic_body_paths': [] if damage == 'undeclared' else [support_path]})
    result = stage_inventory(env)
    assert result['passed'] is (damage is None)
