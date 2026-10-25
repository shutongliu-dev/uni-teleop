import copy
import numpy as np
import pytest

from scripts.validation.validate_mechanical_tool_motion import opposed_handle_contacts, grip_window, validate_motion_phases
from scenes.workrooms.mechanical_assembly.threaded_reset import released_pose_seed


def contact(actor, x, impulse=1e-4):
    return {'actor0': '/tool', 'actor1': actor, 'normal_impulses_ns': [impulse],
            'points_m': [[x, 0., .156]], 'separations_m': [-1e-5]}


def test_grip_requires_positive_opposing_handle_contacts():
    pads = {'thumb': ['/thumb'], 'index': ['/index'], 'middle': [], 'ring': [], 'pinky': []}
    pose = np.array([0., 0., 0., 0., 0., 0., 1.])
    good = [contact('/thumb', -.022), contact('/index', .022)]
    assert opposed_handle_contacts(good, '/tool', pose, pads)['qualified']
    assert not opposed_handle_contacts([contact('/thumb', .022), contact('/index', .022)], '/tool', pose, pads)['qualified']
    assert not opposed_handle_contacts([contact('/thumb', -.022, -1e-4), good[1]], '/tool', pose, pads)['qualified']
    wrong_height = copy.deepcopy(good)
    wrong_height[0]['points_m'][0][2] = .04
    assert not opposed_handle_contacts(wrong_height, '/tool', pose, pads)['qualified']


def test_grip_window_rejects_gap_and_insufficient_duration():
    good = [{'phase': 'hold_grasp', 'native_step_index': i+1, 'simulation_time_s': (i+1)*.0005,
             'grip': {'qualified': True}} for i in range(600)]
    assert grip_window(good, .0005, .30)['passed']
    assert not grip_window(good[:-1], .0005, .30)['passed']
    failed = copy.deepcopy(good)
    for row in failed[-21:]:
        row['grip']['qualified'] = False
    assert not grip_window(failed, .0005, .30)['passed']


def test_grip_window_cannot_borrow_close_contact_or_skipped_native_steps():
    rows = [{'phase': 'close' if i < 560 else 'hold_grasp', 'native_step_index': i+1,
             'simulation_time_s': (i+1)*.0005, 'grip': {'qualified': True}} for i in range(600)]
    assert not grip_window(rows, .0005, .30)['passed']
    for row in rows:
        row['phase'] = 'hold_grasp'
    rows[300]['native_step_index'] += 1
    assert not grip_window(rows, .0005, .30)['passed']


def test_motion_plan_requires_measurable_hold_and_release_phases():
    home = {'fr3_joint1': 0.}
    phases = [{'name': name, 'duration_s': duration, 'joint_positions_rad': home}
              for name, duration in [('approach', 1.), ('close', 1.), ('hold_grasp', .3), ('open', 1.), ('retreat', 1.), ('final_hold', .2)]]
    validate_motion_phases(phases, home)
    short = copy.deepcopy(phases)
    short[2]['duration_s'] = .02
    with pytest.raises(ValueError, match='enough'):
        validate_motion_phases(short, home)
    with pytest.raises(ValueError, match='in order'):
        validate_motion_phases(phases[:-1], home)
    wrong_ready = copy.deepcopy(phases)
    wrong_ready[-1]['joint_positions_rad']['fr3_joint1'] = .1
    with pytest.raises(ValueError, match='frozen ready'):
        validate_motion_phases(wrong_ready, home)


def test_reset_seed_translates_only_once_and_never_exposes_replay():
    one, audit = released_pose_seed()
    two, _ = released_pose_seed((.48, .1, .88))
    for name in one:
        assert set(one[name]) == {'position_m', 'quaternion_wxyz'}
        assert np.allclose(np.asarray(two[name]['position_m'])-one[name]['position_m'], [.1, .1, .1])
        assert two[name]['quaternion_wxyz'] == one[name]['quaternion_wxyz']
    assert audit['reset_velocities'] == 'zero'
    assert audit['trajectory_replay'] is False
