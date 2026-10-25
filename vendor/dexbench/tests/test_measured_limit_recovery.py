"""Recorded contact-driven limit excursion: no controller dead end; no simulator."""
from pathlib import Path

import numpy as np
import pytest

from bench.paths import load_json
from bench.protocol.schemas import (AgentObservation, MEASURED_JOINT_LIMIT_SAFETY_BAND_RAD,
                                 RobotJointCommand)
from bench.control.fingertip import GeometryIKError, MEASURED_JOINT_LIMIT_TOLERANCE_RAD
from test_classical_fingertip_control import (
    geometry, ready_policy, timed_hold, action, executed_step, assert_continuous_command,
)


@pytest.fixture
def recorded(geometry):
    row = load_json(Path(__file__).parent / 'fixtures/recorded_pinky_limit_excursion.json')
    obs = AgentObservation(episode_id='public_test', sequence=row['sequence'], sim_time_s=row['sim_time_s'],
        instruction='Recorded robot-only recovery regression', profile='vision_proprio', cameras=[],
        proprioception=row['proprioception'])
    policy = ready_policy(geometry)
    command = RobotJointCommand.model_validate(row['last_command'])
    previous = row['previous_command']['joint_positions_rad']
    velocity = {name:(value-previous[name])/.02 for name,value in command.joint_positions_rad.items()}
    policy.adopt_direct_reference(command, velocity_rad_s=velocity,
                                 observation_sequence=obs.sequence, sim_time_s=obs.sim_time_s)
    return policy, obs


def next_observation(original, tick, *, positions=None):
    obs = original.model_copy(deep=True)
    obs.sequence += tick
    obs.sim_time_s += tick * .02
    if positions is not None:
        obs.proprioception.joint_positions_rad = dict(positions)
    return obs


def test_controller_band_is_the_service_safety_band():
    assert MEASURED_JOINT_LIMIT_TOLERANCE_RAD == MEASURED_JOINT_LIMIT_SAFETY_BAND_RAD == .03


def test_recorded_excursion_accepts_geometry_without_changing_raw_feedback(recorded):
    """The recorded G28 pinky PIP state no longer deadlocks geometry actions."""
    policy, obs = recorded
    raw = obs.model_dump()
    assert obs.proprioception.joint_positions_rad['right_pinky_PIP'] == -0.0018969556549564004
    assert obs.proprioception.joint_velocities_rad_s['right_pinky_PIP'] == -13.784337997436523
    previous, velocity = policy.previous.copy(), policy.velocity.copy()
    wrist = policy.public_geometry(obs.proprioception.joint_positions_rad)['wrist']
    wrist['position_m'][2] += .02
    move = action({'wrist': wrist})
    move.observation_sequence = obs.sequence
    policy.begin(move, obs)
    np.testing.assert_array_equal(policy.previous, previous)
    first = executed_step(policy, obs)
    assert first.status == 'running'
    assert_continuous_command(policy, first.command, previous, velocity)
    for name, target in first.command.joint_positions_rad.items():
        joint = policy.geometry.by_name[name]
        assert joint.lower <= target <= joint.upper
    assert obs.model_dump() == raw  # q, dq and FK inputs were never clamped.


def test_recorded_excursion_accepts_an_ordinary_hold(recorded):
    policy, obs = recorded
    policy.begin(timed_hold(obs, .2), obs)
    for tick in range(11):
        result = executed_step(policy, next_observation(obs, tick))
        if tick < 10:
            assert result.status == 'running' and result.diagnostic is None
        else:
            assert result.status == 'completed' and result.command is None


@pytest.mark.parametrize('side', ['lower', 'upper'])
@pytest.mark.parametrize('where', ['begin', 'step'])
def test_excess_beyond_the_safety_band_is_rejected_before_command(recorded, side, where):
    policy, obs = recorded
    if where == 'step':
        policy.begin(timed_hold(obs, .2), obs)
        executed_step(policy, obs)
        obs = next_observation(obs, 1)
    joint = policy.geometry.by_name['right_pinky_PIP']
    obs.proprioception.joint_positions_rad[joint.name] = (
        joint.lower - MEASURED_JOINT_LIMIT_TOLERANCE_RAD - 1e-9 if side == 'lower'
        else joint.upper + MEASURED_JOINT_LIMIT_TOLERANCE_RAD + 1e-9)
    previous = policy.previous.copy()
    if where == 'begin':
        with pytest.raises(GeometryIKError, match='joint-limit safety band') as caught:
            policy.begin(timed_hold(obs, .2), obs)
        assert caught.value.public_diagnostic.code == 'invalid_robot_state'
    else:
        result = executed_step(policy, obs)
        assert result.status == 'failed' and result.command is None
        assert result.diagnostic.code == 'invalid_robot_state'
    np.testing.assert_array_equal(policy.previous, previous)


@pytest.mark.parametrize('bad', [float('nan'), float('inf')])
@pytest.mark.parametrize('where', ['begin', 'step'])
def test_nonfinite_raw_positions_are_rejected(recorded, bad, where):
    policy, obs = recorded
    if where == 'step':
        policy.begin(timed_hold(obs, .2), obs)
        result = executed_step(policy, obs)
        obs = next_observation(obs, 1, positions=result.command.joint_positions_rad)
    obs.proprioception.joint_positions_rad['right_pinky_PIP'] = bad
    if where == 'begin':
        with pytest.raises(GeometryIKError):
            policy.begin(timed_hold(obs, .2), obs)
    else:
        result = executed_step(policy, obs)
        assert result.status == 'failed' and result.command is None
        assert result.diagnostic.code == 'invalid_robot_state'


def test_cancel_and_retry_restore_drive_velocity_and_integral(recorded):
    policy, obs = recorded
    policy.begin(timed_hold(obs, .2), obs)
    previous, velocity, integral = policy.previous.copy(), policy.velocity.copy(), policy.integral.copy()
    candidate = policy.step(obs)
    assert candidate.status == 'running'
    policy.cancel('cancel candidate before native ACK')
    for name, expected in (('previous', previous), ('velocity', velocity), ('integral', integral)):
        np.testing.assert_array_equal(getattr(policy, name), expected)
    policy.begin(timed_hold(obs, .2), obs)
    repeated = policy.step(obs)
    assert repeated.command == candidate.command


def test_requested_postures_stay_strict_inside_the_measured_band(recorded):
    policy, obs = recorded
    q = dict(obs.proprioception.joint_positions_rad)
    with pytest.raises(ValueError, match='outside URDF'):
        policy.geometry.solve(q, hand_posture_rad={'right_pinky_PIP': -.000001})


def _hold_with_pinky(policy, obs, values):
    joint = policy.geometry.by_name['right_pinky_PIP']
    policy.begin(timed_hold(obs, .4), obs)
    result = None
    for tick, value in enumerate(values):
        current = next_observation(obs, tick)
        current.proprioception.joint_positions_rad[joint.name] = joint.lower - value
        result = executed_step(policy, current)
        if result.status != 'running':
            break
    return result


@pytest.mark.parametrize('later,expected', [(.0119, 'running'), (.0125, 'failed')])
def test_excursion_present_at_begin_may_persist_but_not_grow(recorded, later, expected):
    policy, obs = recorded
    obs.proprioception.joint_positions_rad['right_pinky_PIP'] = policy.geometry.by_name['right_pinky_PIP'].lower - .010
    result = _hold_with_pinky(policy, obs, [.010] * 5 + [later])
    assert result.status == expected
    if expected == 'failed':
        assert result.diagnostic.code == 'invalid_robot_state'
        assert result.diagnostic.joint_name == 'right_pinky_PIP'


def test_fresh_excursion_beyond_five_mrad_stops_the_action(recorded):
    policy, obs = recorded
    joint = policy.geometry.by_name['right_pinky_PIP']
    obs.proprioception.joint_positions_rad[joint.name] = joint.lower
    result = _hold_with_pinky(policy, obs, [0., .004, .0051])
    assert result.status == 'failed' and result.diagnostic.code == 'invalid_robot_state'


def test_excursion_at_begin_is_reported_without_blocking(recorded):
    policy, obs = recorded
    policy.begin(timed_hold(obs, .2), obs)
    assert policy._limit_diagnostic.code == 'measured_limit_projection'
    assert policy._limit_diagnostic.joint_error_rad == pytest.approx(.0018969556549564004)
