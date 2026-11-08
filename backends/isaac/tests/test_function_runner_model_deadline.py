"""A run horizon is distinct from provider faults; CPU doubles, no native claim."""
import json
from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient
import pytest

from agent.model_client import function_runner
from agent.model_client.bench_http import BenchHTTPClient
from agent.model_client.function_session import FunctionSession
from agent.model_client.transport import ModelGatewayError
from bench.paths import load_json
from bench.runtime.controllers import default_registry
from bench.runtime.server import create_app
from bench.runtime.service import AgentService
from scenes.workrooms.catalog import task_definition
from test_board_game_service import FakeBoardWorker


@pytest.mark.parametrize('case,expected', [
    ('wall_timeout', 'client_wall_deadline'),
    ('early_timeout', 'run_error'),
    ('separate_model_timeout', 'run_error'),
    ('late_http_error', 'run_error'),
    ('late_response', 'client_wall_deadline'),
    ('uncertain_action', 'run_error'),
    ('retry_deadline', 'client_wall_deadline'),
    ('retry_deadline_with_per_attempt_cap', 'client_wall_deadline'),
    ('retry_deadline_pending', 'run_error'),
])
def test_model_wait_horizon_preserves_raw_error_and_cannot_execute_late(
        tmp_path, monkeypatch, case, expected):
    release = Path(task_definition('card_room', 'tic_tac_toe_3x3')['source_release']['path'])
    workers, pending = [], [False]
    clock = [function_runner.time.monotonic()]
    monkeypatch.setattr(function_runner, 'time', SimpleNamespace(monotonic=lambda: clock[0]))
    # dexbench: the session and HTTP deadlines derive from the runner's clock, so they read the same
    # test clock; otherwise slow real setup (release validation on a network file system) exhausts
    # the 60 s horizon before the first model call.
    import agent.model_client.bench_http as bench_http
    import agent.model_client.function_http as function_http
    import agent.model_client.function_session as function_session
    import time as real_time
    test_clock = SimpleNamespace(monotonic=lambda: clock[0], sleep=real_time.sleep, time=real_time.time)
    for module in (function_session, bench_http, function_http):
        monkeypatch.setattr(module, 'time', test_clock)
    if case in ('uncertain_action', 'retry_deadline_pending'):
        monkeypatch.setattr(FunctionSession, 'pending', property(
            lambda _: {'action_id': 'uncertain'} if pending[0] else None))

    def factory(**kwargs):
        worker = FakeBoardWorker(**kwargs)
        workers.append(worker)
        return worker

    class Model:
        def public_metadata(self):
            return {'model': 'cpu_deadline_test_double'}

        def generate(self, prompt, turns, **kwargs):
            clock[0] += 59. if case == 'early_timeout' else 61.
            pending[0] = case in ('uncertain_action', 'retry_deadline_pending')
            if case == 'late_response':
                return SimpleNamespace(
                    text=json.dumps({'function': 'hold', 'arguments': {'duration_s': .5}}),
                    public_metadata=lambda: {'model': 'cpu_deadline_test_double'})
            raise ModelGatewayError('Injected fault', attempts=1,
                category=('request_budget_exhausted' if case.startswith('retry_deadline') else
                          'http_status' if case == 'late_http_error' else 'transport_timeout'),
                status_code=503 if case == 'late_http_error' else None)

    service = AgentService(tmp_path / 'episodes', release=release,
        registry=default_registry(release), worker_factory=factory)
    with TestClient(create_app(service=service)) as http:
        report = function_runner.run(function_runner.RunConfig(
            'http://testserver', 'task1', tmp_path / 'run', wall_budget=60.,
            model_timeout=.1 if case in ('separate_model_timeout', 'retry_deadline_with_per_attempt_cap') else None),
            bench=BenchHTTPClient('http://testserver', client=http),
            models={'player_0': Model(), 'player_1': Model()})

    assert report['termination'] == expected
    assert report['run_status'] == ('completed' if expected == 'client_wall_deadline' else 'infra_error')
    assert report['decisions'] == {'player_0': 1, 'player_1': 0}
    assert not report['result_valid_for_scoring']
    assert report['evaluation_player_outcomes'] == {'player_0': 'unknown', 'player_1': 'unknown'}
    assert not [call for worker in workers for call in worker.calls if call[0] == 'step']
    folder = tmp_path / 'run/player_0/decisions/0000'
    assert not (folder / 'execution.json').exists()
    if case != 'late_response':
        receipt = load_json(folder / 'model_error.json')
        assert receipt['error_category'] == ('request_budget_exhausted' if case.startswith('retry_deadline') else
                                             'http_status' if case == 'late_http_error' else 'transport_timeout')
    if case in ('uncertain_action', 'retry_deadline_pending'):
        assert report['action_recovery_pending']
        assert report['result'] is None
    else:
        assert report['finalization_error'] is None
        assert report['result']['episode']['outcome'] == 'cancelled'
