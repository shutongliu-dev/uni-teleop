"""Agent consumers bind the public scoring declaration and retain earned scores."""
from copy import deepcopy
import json

from fastapi.testclient import TestClient
import httpx
import pytest

from agent.model_client.bench_http import BenchHTTPClient, BenchHTTPError
from agent.model_client.function_runner import RunConfig, run
from bench.isaacsim.release import load_evaluator
from bench.protocol import EpisodeResult
from bench.protocol.task_referee import project_referee_state, project_task_rules
from bench.runtime.server import create_app
from test_function_http_client import success, transport
from test_function_response_bindings import episode_spec
from test_function_runner import ScriptedModel
from test_scene_function_runner import mechanical_service
from test_task_referee_contract import declared_task, score
from test_terminal_scoring import declaration, project, frozen_height, height_receipt


def public_task(*, scored=True):
    task = declared_task('example')
    public = project_task_rules(task, scoring_contract=declaration() if scored else None)
    return public | {'episode': {'terminal': False, 'outcome': None},
                     'referee': project_referee_state(score(), public['public_rules'])}


@pytest.mark.parametrize('function', ['get_result', 'finish_episode'])
@pytest.mark.parametrize('change', ['missing', 'unit', 'objective', 'metric_units', 'undeclared'])
def test_terminal_scoring_cannot_change_or_drop_its_bound_contract(function, change):
    spec = episode_spec('example')
    result = EpisodeResult(spec=spec, outcome='cancelled', elapsed_sim_s=.4, elapsed_wall_s=1.,
                           evaluator_evidence_sha256='a'*64, scoring=project()).model_dump(mode='json')
    if change == 'missing':
        result.pop('scoring')
    elif change == 'unit':
        result['scoring']['score_unit'] = 'm'
    elif change == 'objective':
        result['scoring']['objective'] = 'maximize'
    elif change == 'metric_units':
        result['scoring']['metric_units']['duration_s'] = 'ms'
    client = transport(lambda request: httpx.Response(200, json=success(
        json.loads(request.content), result)))
    try:
        client.bind_episode(spec, task=public_task(scored=change != 'undeclared'))
        with pytest.raises(BenchHTTPError) as rejected:
            client.call({'request_id': 'result', 'function': function, 'arguments': {}})
        assert rejected.value.category == 'invalid_response'
    finally:
        client.close()


@pytest.mark.parametrize('change', ['missing', 'objective', 'unit'])
def test_get_task_scoring_contract_cannot_drift_under_same_rules_hash(change):
    initial = public_task()
    response = deepcopy(initial)
    if change == 'missing':
        response.pop('scoring_contract')
    elif change == 'objective':
        response['scoring_contract']['objective'] = 'maximize'
    else:
        response['scoring_contract']['score_unit'] = 'm'
    client = transport(lambda request: httpx.Response(200, json=success(
        json.loads(request.content), response)))
    try:
        client.bind_episode(episode_spec('example'), task=initial)
        with pytest.raises(BenchHTTPError):
            client.call({'request_id': 'task', 'function': 'get_task', 'arguments': {}})
        with pytest.raises(BenchHTTPError):
            client.bind_episode(episode_spec('example'), task=response)
    finally:
        client.close()


def test_terminal_get_task_cannot_omit_declared_score():
    initial = public_task()
    response = deepcopy(initial)
    response['episode'] = {'terminal': True, 'outcome': 'cancelled'}
    client = transport(lambda request: httpx.Response(200, json=success(
        json.loads(request.content), response)))
    try:
        client.bind_episode(episode_spec('example'), task=initial)
        with pytest.raises(BenchHTTPError):
            client.call({'request_id': 'task', 'function': 'get_task', 'arguments': {}})
    finally:
        client.close()


@pytest.mark.parametrize('fault', ['missing_task', 'changed_scoring'])
def test_declared_task_must_be_present_and_match_discovery_before_model_call(tmp_path, monkeypatch, fault):
    api, workers, _ = mechanical_service(tmp_path, 'tower_building', 'living_room')
    model = ScriptedModel([{'stop': True}])
    with TestClient(create_app(service=api)) as http:
        bench = BenchHTTPClient('http://testserver', client=http)
        original = bench._json
        def corrupt(method, path, **kwargs):
            value = original(method, path, **kwargs)
            if method == 'POST' and path == '/episodes':
                value = deepcopy(value)
                if fault == 'missing_task':
                    value.pop('task')
                else:
                    value['task']['scoring_contract']['objective'] = 'minimize'
            return value
        monkeypatch.setattr(bench, '_json', corrupt)
        report = run(RunConfig('http://testserver', 'task3', tmp_path / 'run',
                              family='living_room', seed=0), bench=bench, models={'single': model})
    assert not model.turns and report['error_type'] is not None
    assert not report['result_valid_for_scoring']
    assert len(workers) == 1 and workers[0].closed


@pytest.mark.parametrize('height', [.025, .3])
def test_function_runner_records_current_frozen_height_without_binary_failure(tmp_path, height):
    api, workers, _ = mechanical_service(tmp_path, 'tower_building', 'living_room')
    root, task = frozen_height()
    evaluator = load_evaluator(root)(task)

    class HeightModel(ScriptedModel):
        def generate(self, *args, **kwargs):
            if not self.turns:
                worker = workers[-1]
                original = worker.on_step
                def stepped(worker):
                    original(worker)
                    worker.score = height_receipt(evaluator, height)
                worker.on_step = stepped
            return super().generate(*args, **kwargs)

    model = HeightModel([{'function': 'hold', 'arguments': {'duration_s': .02}}, {'stop': True}])
    with TestClient(create_app(service=api)) as http:
        report = run(RunConfig('http://testserver', 'task3', tmp_path / 'run',
            family='living_room', seed=0, max_decisions=2, max_status_queries=100000),
            bench=BenchHTTPClient('http://testserver', client=http), models={'single': model})
    assert report['run_status'] == 'completed', report
    assert report['task_outcome'] == 'unknown' and report['task_success'] is None
    assert report['task_score'] == height and report['task_score_unit'] == 'm'
    assert report['task_score_valid'] and report['result_valid_for_scoring']
    assert report['result']['scoring']['metrics']['best_height_m'] == height
    for turns in model.turns:
        context = json.loads(turns[-1]['content'][0]['text'])
        assert context['task']['scoring_contract']['score_unit'] == 'm'
        assert 'scoring' not in context['task']  # Numeric feedback remains terminal-only.
