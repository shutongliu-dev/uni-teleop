"""Every legacy result source binds frozen episode identity; CPU doubles only."""
import json

import pytest

from agent.model_client import runner
from bench.paths import load_json
from bench.protocol.schemas import EpisodeSpec
from test_model_attempt_accounting import setup_route
from test_online_model_runner import fixtures, setup_loop


SOURCES = ('initial_poll', 'post_model_poll', 'post_decision_poll', 'primitive_poll',
           'session_result', 'session_result_finally', 'cleanup_get', 'cleanup_finish')
DRIFTS = ('implementation_sha256', 'weights_sha256', 'control_period_s', 'missing_spec')


def corrupt(result, drift):
    if drift == 'missing_spec':
        result.pop('spec')
    elif drift is not None:
        result['spec']['system0'][drift] = .04 if drift == 'control_period_s' else 'f' * 64
    return result


def result_source(monkeypatch, tmp_path, source, drift=None):
    action = {'action': 'hold', 'duration_s': .02} if source == 'primitive_poll' else {'action': 'stop'}
    args, bench, model = setup_loop(monkeypatch, tmp_path, [action], fingertip=True)
    calls = {'get': 0, 'finish': 0, 'delivered': False}
    selected_get = {'initial_poll': 1, 'post_model_poll': 2, 'post_decision_poll': 3,
                    'primitive_poll': 3, 'cleanup_get': 4}.get(source)

    def terminal():
        calls['delivered'] = True
        return corrupt(bench.terminal_result('success'), drift)

    def get_result(_):
        calls['get'] += 1
        if calls['get'] == selected_get:
            return terminal()
        # Once bad evidence has been seen, even a valid later result cannot
        # rehabilitate this run. Cleanup must still explicitly finish.
        return bench.terminal_result('success') if calls['delivered'] else None

    def finish(_):
        calls['finish'] += 1
        bench.finished = True
        return terminal() if source == 'cleanup_finish' else bench.terminal_result('success')

    bench.get_result, bench.finish = get_result, finish
    if source.startswith('session_result'):
        def cache_result(self, *args):
            self.result = terminal()
            if source == 'session_result_finally':
                raise runner.BenchHTTPError('Later observation unavailable', category='transport_error')
            return {'status': 'completed'}
        monkeypatch.setattr(runner.InteractionSession, 'move', cache_result)
    return args, bench, model, calls


@pytest.mark.parametrize('source', SOURCES)
@pytest.mark.parametrize('drift', DRIFTS)
def test_every_result_source_rejects_controller_drift_and_missing_spec(monkeypatch, tmp_path, source, drift):
    args, bench, model, calls = result_source(monkeypatch, tmp_path, source, drift)
    report = runner.run(args)
    assert calls['delivered'] and calls['finish'] == 1
    assert bench.finished and bench.closed and model.closed
    assert report['result'] is None and report['result_source'] is None
    assert report['task_success'] is None and report['task_outcome'] == 'unknown'
    assert report['run_status'] == 'infra_error'
    assert report['result_validation_error']['error_category'] == 'invalid_response'
    assert report['error_type'] is not None
    assert load_json(args.output / 'run_report.json') == report
    assert (args.output / 'artifact_hashes.json').is_file()
    if source == 'initial_poll':
        assert not model.requests and not bench.commands


@pytest.mark.parametrize('source', SOURCES)
def test_every_result_source_accepts_the_bound_complete_result(monkeypatch, tmp_path, source):
    args, bench, model, calls = result_source(monkeypatch, tmp_path, source)
    report = runner.run(args)
    assert calls['delivered']
    assert calls['finish'] == (1 if source == 'cleanup_finish' else 0)
    assert report['result']['outcome'] == 'success' and report['task_success'] is True
    assert report['result_validation_error'] is None
    assert report['result_source'] == (
        'session_result' if source in ('primitive_poll', 'session_result', 'session_result_finally')
        else 'cleanup_finish' if source == 'cleanup_finish' else 'server_result')
    if source == 'session_result_finally':
        assert report['run_status'] == 'infra_error'  # Transport failure preserves independent success.
    else:
        assert report['error_type'] is None and report['run_status'] == 'completed'
    assert bench.closed and model.closed


@pytest.mark.parametrize('field,value', [
    ('episode_id', 'other_episode'), ('task_id', 'other_task'), ('seed', 12),
    ('agent_config_sha256', 'f' * 64), ('task_release_sha256', 'f' * 64),
    ('budgets', {'max_actions': 19}), ('robot_id', 'other_robot'),
])
def test_terminal_binding_rejects_other_full_episode_identity(monkeypatch, tmp_path, field, value):
    args, bench, _, _ = result_source(monkeypatch, tmp_path, 'initial_poll')
    original = bench.get_result
    def wrong_identity(episode):
        result = original(episode)
        if result is not None:
            result['spec'][field] = value
        return result
    bench.get_result = wrong_identity
    report = runner.run(args)
    assert report['result'] is None and report['task_success'] is None
    assert report['result_validation_error']['error_category'] == 'invalid_response'
    assert report['run_status'] == 'infra_error'


@pytest.mark.parametrize('change', ['no_evidence', 'no_completion', 'private_extra'])
def test_terminal_binding_rejects_malformed_success(monkeypatch, tmp_path, change):
    args, bench, _, _ = result_source(monkeypatch, tmp_path, 'initial_poll')
    original = bench.get_result
    def malformed(episode):
        result = original(episode)
        if change == 'no_evidence':
            result.pop('evaluator_evidence_sha256')
        elif change == 'no_completion':
            result.pop('completion_sim_s')
        else:
            result['private_state'] = 'private_canary'
        return result
    bench.get_result = malformed
    report = runner.run(args)
    assert report['result'] is None and report['task_success'] is None and bench.finished
    assert report['result_validation_error']['error_category'] == 'invalid_response'
    assert 'private_canary' not in (args.output / 'run_report.json').read_text()


def test_finish_none_is_not_pending_terminal_evidence(monkeypatch, tmp_path):
    args, bench, _ = setup_loop(monkeypatch, tmp_path, [{'action': 'stop'}])
    def empty_finish(_):
        bench.finished = True
        return None
    bench.finish = empty_finish
    report = runner.run(args)
    assert bench.finished and report['result'] is None
    assert report['result_validation_error']['error_category'] == 'invalid_response'
    assert report['run_status'] == 'infra_error' and report['task_success'] is None


@pytest.mark.parametrize('field', ['implementation_sha256', 'weights_sha256', 'control_period_s'])
def test_unadmitted_ready_controller_cannot_gain_success_during_cleanup(monkeypatch, tmp_path, field):
    args, bench, model = setup_loop(monkeypatch, tmp_path, [{'action': 'stop'}], fingertip=True)
    create = bench.create
    def changed_ready(request):
        ready = create(request)
        ready['spec']['system0'][field] = .04 if field == 'control_period_s' else 'f' * 64
        return ready
    bench.create = changed_ready
    bench.get_result = lambda _: bench.terminal_result('success')
    report = runner.run(args)
    assert not model.requests and not bench.commands and bench.finished
    assert report['result'] is None and report['task_success'] is None
    assert report['task_outcome'] == 'unknown' and report['run_status'] == 'infra_error'
    assert report['result_validation_error']['error_category'] == 'invalid_response'


@pytest.mark.parametrize('typed', [False, True])
def test_ready_binding_copies_identity_and_normalizes_omitted_defaults(typed):
    _, raw, _ = fixtures()
    spec = EpisodeSpec.model_validate(raw) if typed else raw
    binding = runner._TerminalResultBinding()
    binding.bind(spec)
    fixed = EpisodeSpec.model_validate(raw).model_dump(mode='json')
    if typed:
        spec.seed = 123
        spec.budgets.max_actions = 99
    else:
        spec['seed'] = 123
        spec['budgets']['max_actions'] = 99
    result = {'spec': fixed, 'outcome': 'cancelled', 'elapsed_sim_s': 0., 'elapsed_wall_s': .1}
    assert binding.validate(result)['spec'] == fixed
    result['spec']['seed'] = 123
    with pytest.raises(runner.BenchHTTPError):
        binding.validate(result)
    result['spec']['seed'] = 0
    with pytest.raises(runner.BenchHTTPError):  # The first identity failure stays latched.
        binding.validate(result)


@pytest.mark.parametrize('initial_error', [
    ValueError('private initial error'),
    runner.ModelGatewayError('private credentials failure', category='http_status', status_code=401),
])
def test_invalid_cleanup_is_infra_error_without_erasing_prior_error(monkeypatch, tmp_path, initial_error):
    args, bench, model = setup_loop(monkeypatch, tmp_path, [initial_error])
    bench.get_result = lambda _: (
        {'outcome': 'success', 'physics_valid': True} if model.requests else None)
    report = runner.run(args)
    assert report['error_type'] == type(initial_error).__name__
    assert report['result_validation_error']['error_category'] == 'invalid_response'
    assert report['run_status'] == 'infra_error' and report['task_success'] is None
    assert report['result'] is None and bench.finished


def test_validated_primitive_success_survives_later_observation_transport_error(monkeypatch, tmp_path):
    args, bench, _ = setup_loop(monkeypatch, tmp_path, [{'action': 'hold', 'duration_s': .02}],
                               fingertip=True)
    accepted = [False]
    original_observation = bench.get_observation
    def result(_):
        if bench.commands:
            accepted[0] = True
            return bench.terminal_result('success')
        return None
    def observation(episode):
        if accepted[0]:
            raise runner.BenchHTTPError('Observation unavailable', category='transport_error')
        return original_observation(episode)
    bench.get_result, bench.get_observation = result, observation
    report = runner.run(args)
    assert accepted[0] and len(bench.commands) == 1 and not bench.finished
    assert report['result_source'] == 'session_result' and report['task_success'] is True
    assert report['run_status'] == 'infra_error' and report['result_validation_error'] is None


@pytest.mark.parametrize('free', [False, True])
@pytest.mark.parametrize('interface', ['fingertip_tools', 'fingertip_code'])
def test_invalid_primitive_result_stops_tools_and_code_before_another_action(
        monkeypatch, tmp_path, free, interface):
    response = ('hold(duration_s=.02)\nhold(duration_s=.02)' if interface.endswith('code')
                else {'tool': 'hold', 'arguments': {'duration_s': .02}})
    args, bench, model = setup_route(monkeypatch, tmp_path, interface, [response, response], free=free)
    calls = {'finish': 0}
    def result(_):
        if bench.commands:
            return corrupt(bench.terminal_result('success'), 'implementation_sha256')
        return None
    def finish(_):
        calls['finish'] += 1
        bench.finished = True
        return bench.terminal_result('success')
    bench.get_result, bench.finish = result, finish
    report = runner.run(args)
    assert len(bench.commands) == 1 and len(model.requests) == 1
    assert calls['finish'] == 1 and bench.finished
    assert report['run_status'] == 'infra_error' and report['task_success'] is None
    assert report['result'] is None
