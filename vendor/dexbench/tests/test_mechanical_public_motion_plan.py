"""Public timing and source-robot constraints; these do not qualify contact."""
from copy import deepcopy
from types import SimpleNamespace

import pytest

from bench.protocol.schemas import DirectJointAction
from bench.runtime.service import EpisodeSession
from scenes.workrooms.mechanical_assembly.robot_layout import geometry_model, robot_configuration
from scenes.workrooms.mechanical_assembly.tasks.fastener_assembly.public_motion_plan import (
    candidate_public_motion_plan, four_finger_candidate, public_move_arguments,
)


def runtime_ramp_duration(arguments, capabilities, acknowledged):
    """Exercise the real admission/timing method without starting a worker."""
    caps = SimpleNamespace(control_period_s=capabilities['control_period_s'],
        joint_limits={n: SimpleNamespace(**limit) for n, limit in capabilities['joint_limits'].items()})
    session = SimpleNamespace(capabilities=caps, _acknowledged_drive_targets=lambda: acknowledged)
    start, goal, steps, diagnostic = EpisodeSession._direct_trajectory(
        session, DirectJointAction(**arguments))
    assert diagnostic is None
    assert start == acknowledged
    assert goal == arguments['arm_joint_positions_rad'] | arguments['hand_joint_positions_rad']
    return steps * caps.control_period_s


@pytest.fixture(scope='module')
def configured():
    robot = robot_configuration()
    model = geometry_model(robot)
    caps = {'direct_joint_execution': 'min_jerk_velocity_preflight_v2',
            'arm_joints': model.arm.names, 'hand_joints': model.hand.names,
            'control_period_s': .02, 'joint_limits': {
                joint.name: {'lower_rad': joint.lower, 'upper_rad': joint.upper,
                    'max_velocity_rad_s': min(joint.velocity, .65 if joint.name in model.arm.names else 1.)}
                for joint in (*model.arm.active, *model.hand.active)}}
    return model, caps, candidate_public_motion_plan(caps, robot)


def test_all_phases_obey_public_velocity_and_posture_constraints(configured):
    model, caps, plan = configured
    previous = plan['initial_joint_positions_rad']
    for i, phase in enumerate(plan['phases']):
        args = public_move_arguments(phase, caps, previous,
            observation_sequence=i, action_id=f'mechanical_{i}')
        target = {**args['arm_joint_positions_rad'], **args['hand_joint_positions_rad']}
        assert set(target) == set(model.names)
        ramp_duration = runtime_ramp_duration(args, caps, previous)
        assert phase['direct_ramp_duration_s'] == ramp_duration
        for name in target:
            peak = 1.875*abs(target[name]-previous[name])/ramp_duration
            assert peak <= caps['joint_limits'][name]['max_velocity_rad_s'] + 1e-12
            phase_cap = phase['arm_speed_cap_rad_s'] if name in model.arm.names else phase['hand_speed_cap_rad_s']
            assert peak <= phase_cap + 1e-12
        if phase['name'] in ('close', 'release_preclose'):
            assert max(1.875*abs(target[n]-previous[n])/ramp_duration
                       for n in model.hand.names) <= .12 + 1e-12
        assert not any(model.hand_posture_prior.violation(f, target) for f in model.finger_joints)
        previous = target
    assert previous == plan['initial_joint_positions_rad']
    assert not plan['contains_turn'] and not plan['native_qualified']
    assert plan['schema'] == 'mechanical_public_tool_motion_candidate.v2'


def test_old_close_duration_reproduces_excess_speed_and_new_duration_bounds_real_ramp(configured):
    model, caps, plan = configured
    phase = next(p for p in plan['phases'] if p['name'] == 'close')
    previous = next(p for p in plan['phases'] if p['name'] == 'preclose')['joint_positions_rad']
    args = public_move_arguments(phase, caps, previous, observation_sequence=0, action_id='regression')
    old = {**args, 'duration_s': 1.52}
    old_ramp = runtime_ramp_duration(old, caps, previous)
    assert old_ramp == pytest.approx(1.22)
    peak_distance = max(1.875*abs(phase['joint_positions_rad'][n]-previous[n]) for n in model.hand.names)
    assert peak_distance / old_ramp > .147
    ramp = runtime_ramp_duration(args, caps, previous)
    assert peak_distance / ramp <= .12
    assert args['duration_s'] == pytest.approx(1.88)


def test_long_requested_action_still_uses_actual_ramp_for_reported_speeds(configured):
    _, caps, plan = configured
    phase = deepcopy(next(p for p in plan['phases'] if p['name'] == 'close'))
    phase['duration_s'] = 4.
    previous = next(p for p in plan['phases'] if p['name'] == 'preclose')['joint_positions_rad']
    args = public_move_arguments(phase, caps, previous, observation_sequence=0, action_id='long')
    assert args['duration_s'] == 4.
    assert runtime_ramp_duration(args, caps, previous) == pytest.approx(3.2)


def test_actual_ack_and_lower_native_velocity_extend_dispatch(configured):
    _, caps, plan = configured
    phase = deepcopy(next(p for p in plan['phases'] if p['name'] == 'close'))
    before = deepcopy(phase)
    actual_ack = dict(phase['joint_positions_rad'])
    actual_ack['right_thumb_CMC_FE'] -= .5
    lower_caps = deepcopy(caps)
    lower_caps['joint_limits']['right_thumb_CMC_FE']['max_velocity_rad_s'] = .05
    args = public_move_arguments(phase, lower_caps, actual_ack,
        observation_sequence=21, action_id='from_real_ack')
    assert args['duration_s'] >= 1.875*.5/.05
    actual_ramp = runtime_ramp_duration(args, lower_caps, actual_ack)
    assert 1.875*.5/actual_ramp <= .05
    assert phase == before
    assert args['observation_sequence'] == 21


def test_engineering_episode_budget_admits_the_actual_slow_approach_and_return(configured):
    from bench.control.factory import configured_policy, robot_configuration as task_robot_configuration
    from scenes.workrooms.mechanical_assembly.tasks.fastener_assembly.task import build_task_inputs
    from scripts.validation.validate_mechanical_functions import episode_request
    _, caps, plan = configured
    policy = configured_policy(task_robot_configuration(build_task_inputs()['scene']))
    request = episode_request('fastener_assembly', policy, 'a'*64)
    previous = plan['initial_joint_positions_rad']
    durations = []
    for index, phase in enumerate(plan['phases']):
        args = public_move_arguments(phase, caps, previous,
            observation_sequence=index, action_id=f'budget_{index}')
        durations.append(args['duration_s'])
        assert args['duration_s'] <= request.budgets.max_action_sim_s
        previous = args['arm_joint_positions_rad'] | args['hand_joint_positions_rad']
    # This slow path really needs more than the general API's default 5 seconds.
    assert max(durations) > 5.
    # Settling occurs while holding the reset reference, before any approach.
    # Reserve the entire bounded extra wait (2 s), including the 1.5 s observed
    # for the clocked reset, in addition to the initial engineering hold.
    assert sum(durations) + .5 + 2. < request.budgets.sim_time_s


@pytest.mark.parametrize('value', [None, 0., float('nan'), True])
def test_missing_or_invalid_public_velocity_is_not_invented(configured, value):
    _, caps, plan = configured
    broken = deepcopy(caps)
    broken['joint_limits']['right_index_PIP']['max_velocity_rad_s'] = value
    with pytest.raises(ValueError, match='explicit positive velocity'):
        public_move_arguments(plan['phases'][0], broken, plan['initial_joint_positions_rad'],
            observation_sequence=0, action_id='invalid')


def test_legacy_step_target_execution_is_rejected(configured):
    _, caps, plan = configured
    broken = deepcopy(caps)
    broken['direct_joint_execution'] = 'step_target_v1'
    with pytest.raises(ValueError, match='min_jerk_velocity_preflight_v2'):
        public_move_arguments(plan['phases'][0], broken, plan['initial_joint_positions_rad'],
            observation_sequence=0, action_id='legacy')


def test_candidate_returns_independent_waypoints():
    first = four_finger_candidate()
    first['closed_hand_rad']['right_middle_PIP'] = 0.
    assert four_finger_candidate()['closed_hand_rad']['right_middle_PIP'] > .7
