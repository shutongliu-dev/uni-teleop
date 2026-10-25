"""Task labels must survive API outages without inventing wins or losses."""
import pytest

from agent.model_client import runner
from agent.model_client.outcomes import classify_report
from test_online_model_runner import setup_loop


@pytest.mark.parametrize('status, expected', [(400, 'agent_error'), (401, 'configuration_error'),
    (403, 'configuration_error'), (404, 'agent_error'), (408, 'infra_error'),
    (429, 'infra_error'), (500, 'infra_error'), (503, 'infra_error')])
def test_http_errors_are_typed_and_cleanup_cancellation_is_not_a_task_failure(
        monkeypatch, tmp_path, status, expected):
    error = runner.ModelGatewayError('private provider body', category='http_status', status_code=status)
    args, bench, _ = setup_loop(monkeypatch, tmp_path, [error])
    report = runner.run(args)
    assert report['run_status'] == expected
    assert report['task_success'] is None and report['task_outcome'] == 'unknown'
    assert report['result']['outcome'] == 'cancelled' and report['result_source'] == 'cleanup_finish'
    assert bench.finished


@pytest.mark.parametrize('native, physics_valid, expected', [
    ('success', True, True), ('task_failure', True, False), ('action_budget', True, False),
    ('wall_timeout', True, None), ('success', False, None), ('infrastructure_error', False, None)])
def test_terminal_result_received_after_model_outage_is_preserved(
        monkeypatch, tmp_path, native, physics_valid, expected):
    error = runner.ModelGatewayError('private timeout', category='transport_timeout')
    args, bench, model = setup_loop(monkeypatch, tmp_path, [error])
    original_result = bench.get_result
    bench.get_result = lambda episode: (bench.terminal_result(native, physics_valid=physics_valid)
                                       if model.requests else original_result(episode))
    report = runner.run(args)
    assert report['task_success'] is expected
    assert report['run_status'] == 'infra_error'
    if native == 'success' and not physics_valid:
        assert report['result_source'] is None and report['result'] is None and bench.finished
        assert report['result_validation_error']['error_category'] == 'invalid_response'
    else:
        assert report['result_source'] == 'server_result' and not bench.finished


def test_session_terminal_result_survives_later_failure(monkeypatch, tmp_path):
    args, bench, _ = setup_loop(monkeypatch, tmp_path, [{'action':'hold', 'duration_s':.4}])
    result = {}
    def completed_then_lost_observation(self, *args):
        result.update(bench.terminal_result('success'))
        self.result = result
        raise runner.ModelGatewayError('private', category='transport_error')
    monkeypatch.setattr(runner.InteractionSession, 'move', completed_then_lost_observation)
    report = runner.run(args)
    assert report['result'] == result and report['task_success'] is True
    assert report['result_source'] == 'session_result' and not bench.finished


@pytest.mark.parametrize('responses, ending', [
    ([{'action':'stop'}], 'model_requested_stop'),
    ([{'action':'invalid_tool'}], 'client_decision_budget')])
def test_voluntary_stop_and_exhausted_invalid_decisions_remain_failures(
        monkeypatch, tmp_path, responses, ending):
    args, _, _ = setup_loop(monkeypatch, tmp_path, responses)
    report = runner.run(args)
    assert report['termination'] == ending
    assert report['task_outcome'] == 'failure' and report['task_success'] is False
    assert report['run_status'] == 'completed'


def test_failed_finalization_has_no_task_label(monkeypatch, tmp_path):
    args, bench, _ = setup_loop(monkeypatch, tmp_path, [{'action':'stop'}])
    def failed_finish(*args):
        raise runner.ModelGatewayError('private', category='transport_error')
    bench.finish = failed_finish
    report = runner.run(args)
    assert report['result'] is None and report['task_success'] is None
    assert report['run_status'] == 'infra_error'
    assert report['finalization_error']['error_category'] == 'transport_error'


def test_legacy_error_cancellation_is_unknown_but_real_failure_is_preserved():
    historical = {'task_success':False, 'error_type':'ModelGatewayError',
                  'model_error':{'error_category':'transport_timeout'},
                  'result':{'outcome':'cancelled', 'physics_valid':True}}
    assert classify_report(historical)['task_success'] is None
    assert historical['task_success'] is False  # Never mutate old evidence.
    historical['result']['outcome'] = 'task_failure'
    assert classify_report(historical)['task_success'] is False


@pytest.mark.parametrize('finish_unavailable', [False, True])
def test_output_token_cap_is_agent_budget_not_provider_outage(monkeypatch, tmp_path, finish_unavailable):
    error = runner.ModelGatewayError('output budget', category='output_budget_exhausted')
    args, bench, _ = setup_loop(monkeypatch, tmp_path, [error])
    if finish_unavailable:
        bench.finish = lambda *args: None
    report = runner.run(args)
    assert report['run_status'] == ('infra_error' if finish_unavailable else 'agent_budget')
    assert report['termination'] == 'client_output_budget'
    assert report['task_success'] is (None if finish_unavailable else False)
    if finish_unavailable:
        assert report['result_validation_error']['error_category'] == 'invalid_response'


def test_episode_setup_outage_uses_unknown_label_and_remains_verifiable(monkeypatch, tmp_path):
    from agent.model_client.suite import _setup_failure_candidate
    args, bench, model = setup_loop(monkeypatch, tmp_path, [])
    args.max_decisions = 1
    def unavailable(*args):
        raise runner.BenchHTTPError('unavailable', status_code=503)
    bench.create = unavailable
    report = runner.run(args)
    assert report['episode_id'] is None and report['task_success'] is None
    assert report['run_status'] == 'infra_error' and _setup_failure_candidate(report)
    assert not model.requests and not bench.commands


@pytest.mark.parametrize('category, status', [('http_status', 503), ('transport_timeout', None)])
def test_late_bench_outage_is_not_relabelled_as_an_agent_wall_budget(monkeypatch, tmp_path, category, status):
    from types import SimpleNamespace
    from agent.model_client import interaction
    args, bench, _ = setup_loop(monkeypatch, tmp_path, [{'action':'stop'}])
    now = [100.]
    clock = SimpleNamespace(monotonic=lambda: now[0])
    monkeypatch.setattr(runner, 'time', clock)
    monkeypatch.setattr(interaction, 'time', clock)
    def unavailable(*args):
        now[0] = 160.
        error = runner.BenchHTTPError('unavailable', status_code=status)
        error.category = category
        raise error
    bench.download_cameras = unavailable
    report = runner.run(args)
    assert report['termination'] == 'client_error' and report['run_status'] == 'infra_error'
    assert report['task_success'] is None


def test_cleanup_wall_timeout_after_api_outage_is_not_task_failure(monkeypatch, tmp_path):
    error = runner.ModelGatewayError('private timeout', category='transport_timeout')
    args, bench, _ = setup_loop(monkeypatch, tmp_path, [error])
    bench.finish = lambda *args: bench.terminal_result('wall_timeout')
    report = runner.run(args)
    assert report['run_status'] == 'infra_error' and report['task_success'] is None
    assert report['task_outcome_reason'] == 'cleanup_wall_timeout_after_infrastructure_error'


def test_unresolved_action_prevents_cleanup_from_creating_a_failure_label(monkeypatch, tmp_path):
    from types import SimpleNamespace
    args, _, _ = setup_loop(monkeypatch, tmp_path, [{'action':'hold', 'duration_s':.4}])
    def leave_pending(self, *args):
        self.action_api = SimpleNamespace(pending={'action_id':'unresolved_mock_action'})
        return {'status':'accepted'}
    monkeypatch.setattr(runner.InteractionSession, 'move', leave_pending)
    report = runner.run(args)
    assert report['action_recovery_pending'] is True
    assert report['termination'] == 'client_decision_budget' and report['task_success'] is None
    assert report['task_outcome_reason'] == 'physical_action_execution_unresolved'


@pytest.mark.parametrize('native, expected', [('success', True), ('task_failure', False),
                                            ('cancelled', None), ('wall_timeout', None)])
def test_pending_action_does_not_discard_independent_task_evaluation(native, expected):
    report = {'result':{'outcome':native, 'physics_valid':True}, 'result_source':'server_result',
              'action_recovery_pending':True, 'termination':'server_finished'}
    assert classify_report(report)['task_success'] is expected


def test_unreadable_recovery_journal_still_persists_unknown_result(monkeypatch, tmp_path):
    args, _, _ = setup_loop(monkeypatch, tmp_path, [{'action':'hold', 'duration_s':.4}])
    class UnreadableJournal:
        @property
        def pending(self):
            raise ValueError('private journal detail')
    def leave_unreadable(self, *args):
        self.action_api = UnreadableJournal()
        return {'status':'accepted'}
    monkeypatch.setattr(runner.InteractionSession, 'move', leave_unreadable)
    report = runner.run(args)
    assert report['action_recovery_pending'] is True and report['task_success'] is None
    assert report['action_recovery_error']['error_type'] == 'ValueError'
    assert 'private journal detail' not in (args.output/'run_report.json').read_text()


@pytest.mark.parametrize('api_error', [False, True])
def test_watchdog_wall_timeout_distinguishes_model_outage_from_ordinary_budget(
        monkeypatch, tmp_path, api_error):
    from types import SimpleNamespace
    from agent.model_client import interaction
    response = (runner.ModelGatewayError('retry deadline exhausted', attempts=5,
                    category='request_budget_exhausted') if api_error else {'action':'stop'})
    args, bench, model = setup_loop(monkeypatch, tmp_path, [response])
    now = [100.]
    clock = SimpleNamespace(monotonic=lambda: now[0])
    monkeypatch.setattr(runner, 'time', clock)
    monkeypatch.setattr(interaction, 'time', clock)
    generate = model.generate
    def delayed_model(*args, **kwargs):
        now[0] = 161.
        return generate(*args, **kwargs)
    model.generate = delayed_model
    original_result = bench.get_result
    bench.get_result = lambda episode: (bench.terminal_result('wall_timeout')
                                       if model.requests else original_result(episode))
    report = runner.run(args)
    assert report['result_source'] == 'server_result' and not bench.finished
    assert report['model_call_attempts'] == 1 and not bench.commands
    if api_error:
        assert report['run_status'] == 'infra_error'
        assert report['task_outcome'] == 'unknown' and report['task_success'] is None
        assert report['task_outcome_reason'] == 'wall_timeout_during_model_api_interruption'
    else:
        assert report['run_status'] == 'completed'
        assert report['termination'] == 'client_wall_deadline'
        assert report['task_outcome'] == 'failure' and report['task_success'] is False


@pytest.mark.parametrize('source', ['server_result', 'session_result', 'cleanup_finish'])
@pytest.mark.parametrize('category, status, expected', [
    ('transport_timeout', None, None), ('request_budget_exhausted', None, None),
    ('http_status', 503, None), ('http_status', 400, False), ('unknown', None, False)])
def test_only_typed_model_infrastructure_faults_make_wall_timeout_unassessed(source, category, status, expected):
    report = {'error_type':'ModelGatewayError',
              'run_error':{'error_category':category, 'http_status':status},
              'result':{'outcome':'wall_timeout', 'physics_valid':True}, 'result_source':source}
    assert classify_report(report)['task_success'] is expected


@pytest.mark.parametrize('native, expected', [('success', True), ('task_failure', False)])
def test_model_retry_outage_preserves_independent_authoritative_terminal(native, expected):
    report = {'error_type':'ModelGatewayError',
              'run_error':{'error_category':'request_budget_exhausted'},
              'result':{'outcome':native, 'physics_valid':True}, 'result_source':'server_result'}
    assessed = classify_report(report)
    assert assessed['run_status'] == 'infra_error' and assessed['task_success'] is expected
