"""Real-asset CPU target checks; the native controller owns grasp acceptance."""
from pathlib import Path

import numpy as np
import pytest

from bench.geometry import quaternion_matrix
from bench.paths import DATA_ROOT
from scenes.workrooms.mechanical_assembly.robot_layout import geometry_model, robot_configuration
from scenes.workrooms.mechanical_assembly.tasks.fastener_assembly.motion_plan import (
    candidate_motion_plan, c2_fraction,
)


@pytest.fixture(scope='module')
def candidate():
    if not Path(f'{DATA_ROOT}/assets/robots/franka/fr3/fr3_no_hand.absolute.urdf').exists():
        pytest.skip('Pinned original robot assets unavailable')
    robot = robot_configuration()
    return geometry_model(robot), candidate_motion_plan(robot)


def test_reaches_actual_tilted_handle_with_full_bounded_joint_commands(candidate):
    model, plan = candidate
    by_name = {row['name']: row for row in plan['phases']}
    q = by_name['approach']['joint_positions_rad']
    pose = model.public_geometry(q)['wrist']
    rotation = quaternion_matrix(pose['quaternion_wxyz'])
    axis = plan['grasp_candidate']['axis_in_hand_xz_m']
    center_from_wrist = np.asarray(pose['position_m'])+rotation @ np.array([axis[0], 0., axis[1]])
    assert np.linalg.norm(center_from_wrist-plan['handle_center_world_m']) < .00006
    tool_rotation = quaternion_matrix(np.asarray(plan['tool_quaternion_xyzw'])[[3, 0, 1, 2]])
    assert np.linalg.norm(rotation[:, 1]-tool_rotation[:, 2]) < .001
    for phase in plan['phases']:
        assert len(model.checked_joints(phase['joint_positions_rad'], enforce_limits=True)) == 29
    assert by_name['close']['joint_positions_rad'] != plan['initial_joint_positions_rad']
    assert by_name['final_hold']['joint_positions_rad'] == plan['initial_joint_positions_rad']


def test_safe_segmented_finger_route_is_reversed_before_retreat(candidate):
    _, plan = candidate
    phases = {p['name']: p for p in plan['phases']}
    hand = lambda name: {k: v for k, v in phases[name]['joint_positions_rad'].items() if k.startswith('right_')}
    assert hand('thumb_base_prepare') == hand('open_thumb_fingers')
    assert hand('preclose') == hand('release_preclose')
    assert hand('approach') == hand('open') == hand('retreat')
    assert hand('preclose') != hand('close')
    assert plan['required_native_grasp_gate']['minimum_opposed_contact_duration_s'] == .3
    assert not plan['contains_turn'] and not plan['native_qualified']
    assert not plan['manipulation_qualified']


def test_actual_c2_speed_and_phase_timing_are_bounded(candidate):
    model, plan = candidate
    previous = plan['initial_joint_positions_rad']
    for phase in plan['phases']:
        duration, q = phase['duration_s'], phase['joint_positions_rad']
        assert .02 <= duration <= 10.
        assert duration/.02 == pytest.approx(round(duration/.02), abs=1e-9)
        # Differentiate the actual interpolated reference independently.
        times = np.linspace(0., duration, 1001)
        weights = np.array([c2_fraction(t/duration) for t in times])
        for names in (model.arm.names, model.hand.names):
            delta = max(abs(q[n]-previous[n]) for n in names)
            velocities = np.diff(weights*delta)/np.diff(times)
            assert float(abs(velocities).max()) <= 1.200001
        previous = q
    assert plan['total_duration_s'] < 17.
    assert c2_fraction(0.) == 0. and c2_fraction(1.) == 1.
    eps = 1e-5
    assert c2_fraction(eps)/eps < 1e-7
    assert (1.-c2_fraction(1.-eps))/eps < 1e-7


def test_contact_stages_keep_arm_fixed_and_allow_a_full_hold_gate_window(candidate):
    _, plan = candidate
    phases = {p['name']: p for p in plan['phases']}
    close = phases['close']['joint_positions_rad']
    approach_arm = {n: q for n, q in phases['approach']['joint_positions_rad'].items() if n.startswith('fr3_')}
    for name in ('close', 'hold_grasp', 'release_preclose'):
        assert {n: q for n, q in phases[name]['joint_positions_rad'].items() if n.startswith('fr3_')} == approach_arm
    assert phases['hold_grasp']['duration_s'] > plan['required_native_grasp_gate']['minimum_opposed_contact_duration_s']
    assert phases['hold_grasp']['joint_positions_rad'] == close


@pytest.mark.parametrize('position,quat', [
    ((.38, 0., float('nan')), (0., 0., 0., 1.)),
    ((.38, 0., .88), (0., 0., 0., 2.)),
    ((.38, 0.), (0., 0., 0., 1.)),
])
def test_invalid_tool_pose_is_rejected(candidate, position, quat):
    _, _ = candidate
    with pytest.raises(ValueError):
        candidate_motion_plan(tool_tip_position_m=position, tool_quaternion_xyzw=quat)
