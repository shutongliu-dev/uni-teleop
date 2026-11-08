"""A successful runner must not hide rejected queries or failed robot actions."""
from copy import deepcopy

import pytest

from scripts.validation.agent_configuration_native import validate_feedback


def ik_feedback():
    return {'ok': True, 'function': 'query_ik', 'result': {
        'solution_found': True, 'hands': {'left': {'solution_found': True}, 'right': {'solution_found': True}},
        'advances_physics': False, 'mutates_controller': False}}


@pytest.mark.parametrize('change', [
    lambda f: f.update(ok=False),
    lambda f: f.update(function='observe'),
    lambda f: f['result'].update(solution_found=False),
    lambda f: f['result']['hands']['left'].update(solution_found=False),
    lambda f: f['result'].update(hands={}),
    lambda f: f['result'].update(advances_physics=True),
    lambda f: f['result'].update(mutates_controller=True),
])
def test_rejected_or_mutating_ik_is_not_qualified(change):
    good = ik_feedback()
    validate_feedback(good, 'query_ik')
    bad = deepcopy(good)
    change(bad)
    with pytest.raises(RuntimeError):
        validate_feedback(bad, 'query_ik')


@pytest.mark.parametrize('function,reason,duration', [
    ('move_geometry', 'controller_converged', .34), ('hold', 'time_hold_completed', .12)])
@pytest.mark.parametrize('change', [
    lambda f: f.update(ok=False),
    lambda f: f['result'].update(status='failed'),
    lambda f: f['result'].update(termination_reason='convergence_timeout'),
    lambda f: f['result'].update(episode_finished=True),
    lambda f: f['result'].update(elapsed_sim_s=0.),
])
def test_unsuccessful_action_cannot_pass_from_partial_movement(function, reason, duration, change):
    good = {'ok': True, 'function': function, 'result': {
        'status': 'completed', 'termination_reason': reason, 'episode_finished': False,
        'elapsed_sim_s': duration}}
    validate_feedback(good, function)
    change(good)
    with pytest.raises(RuntimeError):
        validate_feedback(good, function)


def test_truncated_hold_is_rejected():
    with pytest.raises(RuntimeError, match='duration'):
        validate_feedback({'ok': True, 'function': 'hold', 'result': {
            'status': 'completed', 'termination_reason': 'time_hold_completed',
            'episode_finished': False, 'elapsed_sim_s': .1}}, 'hold')
