"""The fixed-mode runner must bind scores before accepting terminal evidence."""
from copy import deepcopy

import pytest

from agent.model_client import runner
from test_online_model_runner import setup_loop


def contract():
    return {'schema': 'terminal_task_scoring_contract.v1', 'mode': 'continuous',
            'objective': 'maximize', 'score_unit': 'm', 'metric_units': {}}


def scoring(height=.25):
    return {**contract(), 'schema': 'terminal_task_scoring.v1',
            'score': height, 'score_valid': True, 'metrics': {}}


def scored_loop(monkeypatch, tmp_path, *, declared=True):
    args, bench, model = setup_loop(monkeypatch, tmp_path, [{'action': 'stop'}])
    task = {'task_id': 'task', 'instruction': 'Measure released height'}
    if declared:
        task['scoring_contract'] = contract()
    bench.list_tasks = lambda: [deepcopy(task)]
    create = bench.create

    def created(request):
        return create(request) | {'task': deepcopy(task)}

    bench.create = created
    terminal = bench.terminal_result
    bench.terminal_result = lambda *args, **kwargs: terminal(*args, **kwargs) | {
        'scoring': scoring(), 'evaluator_evidence_sha256': 'b' * 64}
    return args, bench, model


@pytest.mark.parametrize('phase', ['poll', 'finish'])
@pytest.mark.parametrize('fault', ['missing', 'unit', 'objective', 'metric_units', 'undeclared'])
def test_terminal_score_must_match_discovery_and_creation(monkeypatch, tmp_path, phase, fault):
    args, bench, _ = scored_loop(monkeypatch, tmp_path, declared=fault != 'undeclared')
    terminal = bench.terminal_result

    def changed():
        value = terminal()
        if fault == 'missing':
            value.pop('scoring')
        elif fault == 'unit':
            value['scoring']['score_unit'] = 'cm'
        elif fault == 'objective':
            value['scoring']['objective'] = 'minimize'
        elif fault == 'metric_units':
            value['scoring'].update(metric_units={'height': 'm'}, metrics={'height': .25})
        return value

    if phase == 'poll':
        bench.get_result = lambda _: changed()
        # A later well-formed finish must not erase the earlier contract failure.
        def corrected_finish(episode):
            bench.finished = True
            return terminal()
        bench.finish = corrected_finish
    else:
        finish = bench.finish
        def changed_finish(episode):
            finish(episode)
            return changed()
        bench.finish = changed_finish
    report = runner.run(args)
    assert bench.finished
    assert report['result_validation_error']['error_category'] == 'invalid_response'
    assert report['result'] is None and report['task_success'] is None
    assert not report.get('task_score_valid') and not report.get('result_valid_for_scoring')


@pytest.mark.parametrize('fault', ['missing_task', 'missing_contract', 'objective', 'undeclared'])
def test_creation_score_contract_cannot_drift_before_model(monkeypatch, tmp_path, fault):
    args, bench, model = scored_loop(monkeypatch, tmp_path, declared=fault != 'undeclared')
    create = bench.create
    def changed(request):
        ready = create(request)
        if fault == 'missing_task':
            ready.pop('task')
        elif fault == 'missing_contract':
            ready['task'].pop('scoring_contract')
        elif fault == 'objective':
            ready['task']['scoring_contract']['objective'] = 'minimize'
        else:
            ready['task']['scoring_contract'] = contract()
        return ready
    bench.create = changed
    report = runner.run(args)
    assert not model.requests and not bench.commands and bench.finished
    assert report['task_success'] is None and not report.get('result_valid_for_scoring')


@pytest.mark.parametrize('height', [0., .25])
def test_valid_continuous_score_remains_nonbinary(monkeypatch, tmp_path, height):
    args, bench, _ = scored_loop(monkeypatch, tmp_path)
    terminal = bench.terminal_result
    bench.terminal_result = lambda *args, **kwargs: terminal(*args, **kwargs) | {'scoring': scoring(height)}
    report = runner.run(args)
    assert report['result_validation_error'] is None
    assert report['task_success'] is None and report['task_outcome'] == 'unknown'
    assert report['task_score'] == height and report['task_score_unit'] == 'm'
    assert report['task_score_valid'] and report['result_valid_for_scoring']
