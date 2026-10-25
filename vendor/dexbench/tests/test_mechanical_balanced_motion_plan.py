"""CPU contracts for a backup grasp candidate; no native contact acceptance."""
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from bench.geometry import quaternion_matrix
from bench.protocol.schemas import DirectJointAction
from bench.runtime.service import EpisodeSession
from scenes.workrooms.mechanical_assembly.robot_layout import (
    geometry_model, ready_configuration, robot_configuration,
)
from scenes.workrooms.mechanical_assembly.tasks.fastener_assembly.balanced_motion_plan import (
    balanced_grasp_candidate, candidate_balanced_motion_plan,
)
from scenes.workrooms.mechanical_assembly.tasks.fastener_assembly.motion_plan import _side_rotation
from scenes.workrooms.mechanical_assembly.tasks.fastener_assembly.public_motion_plan import (
    four_finger_candidate, public_move_arguments,
)


@pytest.fixture(scope='module')
def configured():
    robot = robot_configuration()
    model = geometry_model(robot)
    caps = {'direct_joint_execution': 'min_jerk_velocity_preflight_v2',
            'arm_joints': model.arm.names, 'hand_joints': model.hand.names,
            'control_period_s': .02, 'joint_limits': {
                j.name: {'lower_rad': j.lower, 'upper_rad': j.upper,
                         'max_velocity_rad_s': min(j.velocity, .65 if j.name in model.arm.names else 1.)}
                for j in (*model.arm.active, *model.hand.active)}}
    return robot, model, caps, candidate_balanced_motion_plan(caps, robot)


def test_original_ready_base_assets_and_old_candidate_are_preserved(configured):
    robot, _, _, plan = configured
    original = deepcopy(four_finger_candidate())
    candidate = balanced_grasp_candidate()
    candidate['closed_hand_rad']['right_thumb_CMC_FE'] = 0.
    assert balanced_grasp_candidate()['closed_hand_rad']['right_thumb_CMC_FE'] > 1.
    assert four_finger_candidate() == original
    assert robot['base_position_m'] == [-.23, 0., .78]
    assert plan['initial_joint_positions_rad'] == ready_configuration(robot)[0]['joint_positions_rad']
    assert plan['phases'][-1]['joint_positions_rad'] == plan['initial_joint_positions_rad']
    assert plan['tool_asset_id'] == 'threaded_sharpa/v3'
    assert plan['native_qualified'] is False and plan['manipulation_qualified'] is False
    assert plan['contains_turn'] is False


def test_nominal_side_shift_is_eight_mm_along_tool_y(configured):
    _, _, _, plan = configured
    rotation = quaternion_matrix(np.asarray(plan['tool_quaternion_xyzw'])[[3, 0, 1, 2]])
    delta = np.asarray(plan['grasp_candidate']['axis_in_hand_xz_m'])-four_finger_candidate()['axis_in_hand_xz_m']
    shift_world = -_side_rotation(rotation) @ np.array([delta[0], 0., delta[1]])
    assert rotation.T @ shift_world == pytest.approx([0., .008, 0.], abs=1.e-12)


def test_full_targets_respect_limits_posture_prior_and_small_finger_spread(configured):
    _, model, _, plan = configured
    for phase in plan['phases']:
        q = phase['joint_positions_rad']
        assert set(q) == set(model.names)
        model.checked_joints(q, enforce_limits=True)
        assert not any(model.hand_posture_prior.violation(f, q) for f in model.finger_joints)
    for posture in ('preclose_hand_rad', 'closed_hand_rad'):
        q = plan['grasp_candidate'][posture]
        assert all(abs(q[f'right_{finger}_MCP_AA']) <= .005000001
                   for finger in ('index', 'middle', 'ring'))


def test_actual_public_ramp_speed_and_retreat_endpoint_hold(configured):
    _, model, caps, plan = configured
    capability_object = SimpleNamespace(control_period_s=.02,
        joint_limits={name: SimpleNamespace(**row) for name, row in caps['joint_limits'].items()})
    previous = plan['initial_joint_positions_rad']
    for index, phase in enumerate(plan['phases']):
        arguments = public_move_arguments(phase, caps, previous,
            observation_sequence=index, action_id=f'balanced_{index}')
        session = SimpleNamespace(capabilities=capability_object,
            _acknowledged_drive_targets=lambda: previous)
        _, target, steps, diagnostic = EpisodeSession._direct_trajectory(
            session, DirectJointAction(**arguments))
        assert diagnostic is None
        ramp = steps*.02
        assert ramp == pytest.approx(phase['direct_ramp_duration_s'])
        assert arguments['duration_s'] <= 10.
        for name in model.names:
            cap = phase['arm_speed_cap_rad_s'] if name in model.arm.names else phase['hand_speed_cap_rad_s']
            assert 1.875*abs(target[name]-previous[name])/ramp <= cap+1.e-12
        if phase['public_function_hint'] == 'move_joints':
            assert arguments['duration_s']-ramp >= .20-1.e-12
        previous = target


def test_source_pad_directions_are_more_opposed_without_extra_overlap(configured):
    from scripts.diagnostics.probe_mechanical_tool_motion import source_meshes
    _, model, _, plan = configured
    candidate = plan['grasp_candidate']
    axis = np.asarray(candidate['axis_in_hand_xz_m'])
    poses = model.hand.forward(candidate['closed_hand_rad'])
    tool_r = quaternion_matrix(np.asarray(plan['tool_quaternion_xyzw'])[[3, 0, 1, 2]])
    to_tool = tool_r.T @ _side_rotation(tool_r)
    directions = {}
    for row in source_meshes(model):
        if row['name'] not in {f'right_{f}_elastomer' for f in ('thumb', 'index', 'middle', 'ring')}:
            continue
        pose = poses[row['name']]
        points = row['vertices'] @ pose[:3, :3].T+pose[:3, 3]
        radial = points[:, (0, 2)]-axis
        lengths = np.linalg.norm(radial, axis=1)
        which = int(np.argmin(lengths))
        assert .0013 <= .022-lengths[which] <= .00151
        assert abs(points[which, 1]) < .048
        vector = radial[which]/lengths[which]
        directions[row['name']] = to_tool @ np.array([vector[0], 0., vector[1]])
    thumb = directions['right_thumb_elastomer']
    assert thumb[1] < 0.
    for finger in ('index', 'middle', 'ring'):
        opposing = directions[f'right_{finger}_elastomer']
        assert opposing[1] > 0.
        assert np.dot(thumb, opposing) < -.975


def test_thirty_degree_turn_retains_original_joint_margin(configured):
    from scripts.validation.validate_mechanical_functions import turn_path
    _, model, _, plan = configured
    closed = next(p['joint_positions_rad'] for p in plan['phases'] if p['name'] == 'hold_grasp')
    phases, _ = turn_path(model, closed,
        [*plan['tool_tip_position_m'], *plan['tool_quaternion_xyzw']], 30)
    assert len(phases) == 10
    assert phases[-1]['nominal_angle_rad'] == pytest.approx(-np.pi/6)
    assert min(p['ik']['minimum_arm_joint_margin_rad'] for p in phases) > .34
