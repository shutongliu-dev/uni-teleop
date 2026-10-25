"""The smoke respects public settling, without reading task-private thresholds."""
from copy import deepcopy

import pytest

from scripts.validation.task_runtime_smoke import wait_for_public_ready


def waiting_state():
    return {'sim_time_s': 0., 'current_player_id': 'someone', 'status': 'active',
            'public_progress': {'status': 'reset_pending'}}


def test_readiness_wait_holds_until_public_referee_is_ready():
    state = waiting_state()
    calls = []
    def hold(name, duration):
        calls.append((name, duration))
        state['sim_time_s'] += duration
        if len(calls) == 6:
            state['public_progress']['status'] = 'await_intent'
    result = wait_for_public_ready(lambda: deepcopy(state), hold, max_wait_sim_s=2.)
    assert len(calls) == 6 and len(result['holds']) == 6
    assert result['elapsed_sim_s'] == pytest.approx(1.2)
    assert result['final_public_status'] == 'await_intent'
    assert all(duration == .2 for _, duration in calls)


@pytest.mark.parametrize('failure', ['timeout', 'no_time', 'terminated', 'changed_player'])
def test_readiness_failure_never_continues_to_robot_motion(failure):
    state = waiting_state()
    def hold(name, duration):
        if failure != 'no_time': state['sim_time_s'] += duration
        if failure == 'terminated': state['status'] = 'interrupted'
        if failure == 'changed_player': state['current_player_id'] = 'other'
    with pytest.raises(RuntimeError):
        wait_for_public_ready(lambda: deepcopy(state), hold, max_wait_sim_s=.4)


def test_ready_game_does_not_spend_extra_simulation_time():
    state = waiting_state()
    state['public_progress']['status'] = 'await_intent'
    def forbidden(*args):
        raise AssertionError('Ready game does not need a settling action')
    result = wait_for_public_ready(lambda: deepcopy(state), forbidden, max_wait_sim_s=1.)
    assert result['elapsed_sim_s'] == 0 and result['holds'] == []


@pytest.mark.parametrize('value,expected', [('none', None), ('None', None), ('900', 900.), ('1.25', 1.25)])
def test_optional_episode_wall_timeout_parser(value, expected):
    from scripts.validation.task_runtime_smoke import optional_wall_timeout
    assert optional_wall_timeout(value) == expected


@pytest.mark.parametrize('value', ['0', '-1', 'nan', 'inf', '-inf', 'unlimited', ''])
def test_optional_episode_wall_timeout_rejects_invalid_numbers(value):
    import argparse
    from scripts.validation.task_runtime_smoke import optional_wall_timeout
    with pytest.raises(argparse.ArgumentTypeError, match='positive finite seconds or none'):
        optional_wall_timeout(value)


@pytest.mark.parametrize('wall,action,valid', [('none', '600', True), ('900', '180', True),
    ('none', 'nan', False), ('none', 'inf', False), ('none', '0', False), ('900', '901', False)])
def test_cli_none_preserves_other_budgets_and_serializes_null(monkeypatch, tmp_path, wall, action, valid):
    import json
    import sys
    from scripts.validation import task_runtime_smoke as smoke
    seen = []
    monkeypatch.setattr(smoke, 'run', lambda args: seen.append(args))
    monkeypatch.setenv('ISAAC_BENCH_EXPERIENCE', 'existing-experience.kit')
    monkeypatch.setattr(sys, 'argv', ['task_runtime_smoke', '--release', str(tmp_path), '--task-id', 'generic_task',
        '--gpu-uuid', 'test-uuid', '--output', str(tmp_path/'output'), '--wall-timeout-s', wall,
        '--action-wall-timeout-s', action])
    if valid:
        smoke.main()
        args, = seen
        assert args.ready_timeout_s == 10.
        assert args.action_wall_timeout_s == float(action)
        assert args.wall_timeout_s == (None if wall == 'none' else float(wall))
        if wall == 'none':
            assert json.dumps({'wall_time_budget_s': args.wall_timeout_s}) == '{"wall_time_budget_s": null}'
    else:
        with pytest.raises(RuntimeError, match='Invalid wall-time budgets'):
            smoke.main()
        assert not seen
