"""Discovery/controller receipts must agree before model or motion admission."""
from copy import deepcopy
import json
from pathlib import Path

from fastapi.testclient import TestClient
import pytest

from agent.model_client.bench_http import BenchHTTPClient
from agent.model_client import function_runner
from agent.model_client.function_runner import RunConfig, run
from bench.paths import load_json
from bench.runtime.controllers import default_registry
from bench.runtime.server import create_app
from bench.runtime.service import AgentService
from scenes.workrooms.catalog import task_definition
from test_board_game_service import FakeBoardWorker
from test_function_runner import ScriptedModel
from test_living_function_runner import tower_service
from test_scene_function_runner import mechanical_service


def service_case(tmp_path, kind):
    if kind == 'ordinary':
        api, workers, _ = mechanical_service(tmp_path, 'sample_sorting', 'laboratory')
        return api, workers, 'laboratory', 'sample_sorting', ('single',)
    if kind == 'turn':
        api, workers, _ = tower_service(tmp_path)
        return api, workers, 'living_room', 'tower_extraction', ('player_0', 'player_1')
    task_id = 'tic_tac_toe_3x3'
    release = Path(task_definition('card_room', task_id)['source_release']['path'])
    workers = []

    def factory(**kwargs):
        worker = FakeBoardWorker(**kwargs)
        workers.append(worker)
        return worker

    api = AgentService(tmp_path / 'episodes', release=release,
                       registry=default_registry(release), worker_factory=factory)
    return api, workers, 'card_room', task_id, ('player_0', 'player_1')


@pytest.mark.parametrize('kind', ['ordinary', 'board', 'turn'])
@pytest.mark.parametrize('field,value', [
    (None, None),
    ('implementation_sha256', 'a' * 64),
    ('weights_sha256', 'b' * 64),
    ('control_period_s', .04),
])
def test_discovery_controller_is_pinned_before_session_model_and_motion(
        tmp_path, monkeypatch, kind, field, value):
    api, workers, family, task_id, keys = service_case(tmp_path, kind)
    models = {key: ScriptedModel([{'stop': True}]) for key in keys}
    initialized = []
    initialize = function_runner.FunctionSession.initialize

    def track_initialize(session):
        initialized.append(session)
        return initialize(session)

    monkeypatch.setattr(function_runner.FunctionSession, 'initialize', track_initialize)
    with TestClient(create_app(service=api)) as http:
        bench = BenchHTTPClient('http://testserver', client=http)
        list_policies = bench.list_policies

        def stale_discovery(*args, **kwargs):
            policies = deepcopy(list_policies(*args, **kwargs))
            if field is not None:
                for policy in policies:
                    policy[field] = value
            return policies

        monkeypatch.setattr(bench, 'list_policies', stale_discovery)
        report = run(RunConfig('http://testserver', task_id, tmp_path / 'run',
                              family=family, seed=0, max_decisions=1),
                     bench=bench, models=models)
    config = load_json(tmp_path / 'run/agent_config.json')
    episode = report['result'] if kind == 'ordinary' else report['result']['episode']
    actual = episode['spec']['system0']
    assert actual['control_period_s'] == .02  # The real service never runs the stale .04 descriptor.
    assert all(config['controller'][key] == actual[key] for key in ('policy_id', 'version'))
    assert episode['outcome'] == 'cancelled' and report['finalization_error'] is None
    assert len(workers) == 1 and workers[0].closed
    assert not any(operation == 'step' for operation, _ in workers[0].calls)
    assert all(value == 0 for value in report['physical_attempts'].values())
    if field is None:
        assert report['ready_validated'] and report['error_type'] is None
        assert report['run_status'] == 'completed'
        assert len(initialized) == len(keys)
        assert sum(len(model.turns) for model in models.values()) == 1
        assert config['controller'] == actual
    else:
        assert config['controller'][field] == value != actual[field]
        assert not report['ready_validated'] and report['run_status'] == 'infra_error'
        assert report['run_error']['error_category'] == 'invalid_response'
        assert not initialized and all(not model.turns for model in models.values())
        assert report['decisions'] == {} and report['physical_attempts'] == {}
        assert not (tmp_path / 'run/episode_ready.json').exists()
        assert not report['result_valid_for_scoring']
        if kind == 'ordinary':
            assert report['task_success'] is None and report['task_outcome'] == 'unknown'
        else:
            assert report['evaluation_player_outcomes'] == dict.fromkeys(keys, 'unknown')


@pytest.mark.parametrize('continuous', [False, True])
def test_invalid_controller_cannot_score_a_success_received_during_cleanup(
        tmp_path, monkeypatch, continuous):
    family, task_id = (('living_room', 'tower_building') if continuous else
                       ('laboratory', 'sample_sorting'))
    api, workers, _ = mechanical_service(tmp_path, task_id, family)
    model = ScriptedModel([{'stop': True}])
    with TestClient(create_app(service=api)) as http:
        bench = BenchHTTPClient('http://testserver', client=http)
        original_select = function_runner.select_geometry_policy
        selected = None

        def stale_selection(*args, **kwargs):
            nonlocal selected
            selected = original_select(*args, **kwargs)
            selected['weights_sha256'] = 'b' * 64
            return selected

        monkeypatch.setattr(function_runner, 'select_geometry_policy', stale_selection)
        create = bench.create

        def terminal_before_validation(request):
            ready = create(request)
            # Mutating the discovery dict after selection must not rewrite its pin.
            selected.clear()
            selected.update(ready['spec']['system0'])
            session = api.sessions[ready['episode_id']]
            with session.lock:
                # Synthetic terminal evidence tests client classification only.
                session._finish('success')
            return ready

        monkeypatch.setattr(bench, 'create', terminal_before_validation)
        report = run(RunConfig('http://testserver', task_id, tmp_path / 'run',
                              family=family, seed=0, max_decisions=1),
                     bench=bench, models={'single': model})
    config = load_json(tmp_path / 'run/agent_config.json')
    assert config['controller']['weights_sha256'] == 'b' * 64
    assert selected['weights_sha256'] is None
    assert report['result']['outcome'] == 'success'  # Raw cleanup evidence remains auditable.
    assert report['result_source'] == 'server_result'
    assert report['finalization_error'] is None
    assert report['run_status'] == 'infra_error' and not report['ready_validated']
    assert report['task_success'] is None and report['task_outcome'] == 'unknown'
    assert report['task_outcome_reason'] == 'invalid_creation_response'
    assert not report['result_valid_for_scoring']
    if continuous:
        assert 'scoring' in report['result'] and not report['task_score_valid']
    assert not model.turns and report['physical_attempts'] == {}
    assert len(workers) == 1 and workers[0].closed
    assert not any(operation == 'step' for operation, _ in workers[0].calls)
