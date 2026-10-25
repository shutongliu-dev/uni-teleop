"""Diagnostic timing and measured phase completion without a simulator."""
from copy import deepcopy
from types import SimpleNamespace
import math

import pytest

from scripts.validation import mechanical_robot_phase_timing as timing
from scripts.validation.validate_assembly_contacts import _run_robot_phases


def _plan(monkeypatch):
    limits = {'arm_joint': 2.62, 'finger_joint': 12.}
    monkeypatch.setattr(timing, 'velocity_limits', lambda scene: limits)
    initial = {'arm_joint': 0., 'finger_joint': 0.}
    phases = [
        {'id': 'approach', 'joint_positions_rad': {'arm_joint': math.radians(111.93), 'finger_joint': 0.}},
        {'id': 'close', 'joint_positions_rad': {'arm_joint': math.radians(111.93), 'finger_joint': .7}},
    ]
    return initial, timing.time_phases(phases, {}, {'joint_positions_rad': initial}), limits


class RobotOnlyEnv:
    def __init__(self, initial, *, follows=True):
        self.dt = .0005
        self.world = SimpleNamespace(current_time_step_index=0)
        self.observed = dict(initial)
        self.follows = follows
        self.commands = []

    def step(self, action):
        assert set(action) == {'joint_positions_rad', 'duration_s'}
        self.commands.append(deepcopy(action))
        self.world.current_time_step_index += 40
        if self.follows:
            self.observed = dict(action['joint_positions_rad'])
        return {'joint_positions_rad': self.observed, 'metrics': {'state_finite': True}, 'terminated': False}


def test_long_move_respects_quintic_peak_and_has_stationary_settle(monkeypatch):
    initial, phases, limits = _plan(monkeypatch)
    assert phases[0]['timing']['motion_duration_s'] >= 2.8
    env = RobotOnlyEnv(initial)
    result = _run_robot_phases(env, SimpleNamespace(), phases, initial)
    assert result['passed']
    assert result['completed'] == ['approach', 'close']
    previous = initial
    for action in env.commands:
        for name, limit in limits.items():
            rate = abs(action['joint_positions_rad'][name]-previous[name])/.02
            assert rate <= limit*timing.VELOCITY_FRACTION+1.e-9
        previous = action['joint_positions_rad']
    assert all(action['joint_positions_rad'] == phases[-1]['joint_positions_rad'] for action in env.commands[-20:])
    assert all(row['passed'] for row in result['phase_endpoints'])


def test_unreached_phase_cannot_advance_to_grasp(monkeypatch):
    initial, phases, _ = _plan(monkeypatch)
    env = RobotOnlyEnv(initial, follows=False)
    result = _run_robot_phases(env, SimpleNamespace(), phases, initial)
    assert result['passed'] is False
    assert result['failed_phase'] == 'approach'
    assert result['completed'] == []
    assert len(result['phase_endpoints']) == 1
    assert len(env.commands) == round(phases[0]['duration_s']/.02)
    assert all(action['joint_positions_rad']['finger_joint'] == 0 for action in env.commands)


def test_undersized_timing_and_forged_velocity_limits_fail(monkeypatch):
    initial, phases, limits = _plan(monkeypatch)
    phase = deepcopy(phases[0])
    phase['timing']['motion_duration_s'] = .24
    phase['duration_s'] = .24+phase['timing']['settle_duration_s']
    with pytest.raises(ValueError, match='quintic peak'):
        timing.validate_timing(phase, initial, expected_limits=limits)
    phase = deepcopy(phases[0])
    phase['timing']['joint_velocity_limits_rad_s']['arm_joint'] = 20.
    with pytest.raises(ValueError, match='velocity limits differ'):
        timing.validate_timing(phase, initial, expected_limits=limits)
