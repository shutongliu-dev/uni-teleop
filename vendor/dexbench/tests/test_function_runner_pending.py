"""Unreadable action ledgers retain reports without guessing that motion stopped."""
from copy import deepcopy
import json
from types import SimpleNamespace

from fastapi.testclient import TestClient
import pytest

from agent.model_client import function_runner
from agent.model_client.bench_http import BenchHTTPClient, BenchHTTPError
from agent.model_client.function_http import FunctionHTTPClient
from agent.model_client.function_session import FunctionSession
from agent.model_client.learning.robot_api import ActionRecoveryError
from bench.control.factory import public_robot_geometry, robot_configuration
from bench.isaacsim.release import task_inputs
from bench.runtime.controllers import default_registry
from bench.runtime.placement_tasks import project_placement_task
from bench.runtime.server import create_app
from bench.runtime.service import AgentService
from test_agent_service import FakeWorker, publish_targets
from test_board_game_service import game
from test_card_room_placement_runtime import placement_release


class UncalledModel:
    def public_metadata(self):
        return {'model': 'cpu_test_only'}

    def generate(self, *args, **kwargs):
        pytest.fail('An unreadable action ledger must block model calls')


def exercise(bench, task, output, monkeypatch, phase):
    reads, calls, closed = {}, [], []
    secret = 'private_journal_exception_canary'
    error_type = ActionRecoveryError if phase == 'loop' else OSError
    original_initialize = FunctionSession.initialize

    def unreadable(session):
        if phase == 'loop' and not getattr(session, '_test_initialized', False):
            return None
        key = session.trial_id
        reads[key] = reads.get(key, 0) + 1
        # Once uncertainty was observed, one apparently successful later read
        # must not authorize cleanup of an action whose ledger was unreadable.
        if phase == 'loop' and reads[key] > 1:
            return None
        raise error_type(secret)

    monkeypatch.setattr(FunctionSession, 'pending', property(unreadable))
    if phase == 'loop':
        def initialize(session):
            result = original_initialize(session)
            session._test_initialized = True
            return result
        monkeypatch.setattr(FunctionSession, 'initialize', initialize)
    if phase == 'horizon':
        clock = [function_runner.time.monotonic()]
        monkeypatch.setattr(function_runner, 'time', SimpleNamespace(monotonic=lambda: clock[0]))
        def initialize(session):
            clock[0] = session.deadline + 1
            raise BenchHTTPError('Host deadline expired', category='request_budget_exhausted')
        monkeypatch.setattr(FunctionSession, 'initialize', initialize)

    original_call, original_close = FunctionHTTPClient.call, FunctionHTTPClient.close
    def record_call(client, payload, **kwargs):
        calls.append(payload['function'])
        return original_call(client, payload, **kwargs)
    def record_close(client):
        closed.append(client.binding['player_id'] if client.binding else 'single')
        original_close(client)
    monkeypatch.setattr(FunctionHTTPClient, 'call', record_call)
    monkeypatch.setattr(FunctionHTTPClient, 'close', record_close)
    board = task == 'task1'
    keys = ('player_0', 'player_1') if board else ('single',)
    report = function_runner.run(function_runner.RunConfig('http://testserver', task, output),
                                 bench=bench, models={key: UncalledModel() for key in keys})
    assert report['episode_id']
    assert report['action_recovery_pending'] is True
    assert report['run_status'] == 'infra_error'
    assert report['result'] is None
    assert report['finalization_error']['error_type'] == error_type.__name__
    assert report['pending_read_errors']
    assert all(value['error_type'] == error_type.__name__
               for value in report['pending_read_errors'].values())
    assert all(count >= 2 for count in reads.values())
    assert 'finish_episode' not in calls
    assert set(closed) == set(keys)
    recorded = (output / 'run_report.json').read_text()
    assert json.loads(recorded) == report
    assert secret not in recorded
    return report


@pytest.mark.parametrize('phase', ['initialization', 'loop', 'horizon'])
def test_pending_read_failure_preserves_ordinary_report_and_cleanup_gate(placement_release, tmp_path, monkeypatch, phase):
    def factory(**kwargs):
        worker = FakeWorker(**kwargs, depth=True, acknowledged_targets=True, on_step=publish_targets)
        _, scene, task, initial = task_inputs(kwargs['release'], kwargs['task_id'], seed=kwargs['seed'])
        geometry = public_robot_geometry(robot_configuration(scene))
        worker.arm, worker.hand, worker.active = list(geometry.arm.names), list(geometry.hand.names), list(geometry.names)
        worker.raw['joint_positions_rad'] = deepcopy(initial['joint_positions_rad'])
        worker.raw['joint_velocities_rad_s'] = dict.fromkeys(worker.active, 0.)
        worker.raw['robot'] = {'joint_targets_rad': deepcopy(initial['joint_positions_rad'])}
        worker.initial['observation'] = deepcopy(worker.raw)
        worker.initial['metadata'].update(scene_spec=scene, action_spec={
            'joint_names': worker.active, 'arm_joint_names': worker.arm, 'hand_joint_names': worker.hand,
            'joint_limits_rad': {name: [float(geometry.lower[i]), float(geometry.upper[i])]
                                 for i, name in enumerate(geometry.names)}})
        worker.initial['task'] = project_placement_task(task)
        return worker
    api = AgentService(tmp_path / 'episodes', release=placement_release,
                       registry=default_registry(placement_release), worker_factory=factory)
    with TestClient(create_app(service=api)) as http:
        report = exercise(BenchHTTPClient('http://testserver', client=http),
                          'task0', tmp_path / 'run', monkeypatch, phase)
        assert api.sessions[report['episode_id']].result is None


def test_pending_read_failure_preserves_board_report_without_finishing_unknown_action(game, tmp_path, monkeypatch):
    with game() as g:
        # The service admits one live episode; release the fixture's seed game.
        FunctionHTTPClient('http://testserver', g.created['episode_id'],
            player=g.created['players'][0], client=g.client).finish(g.created['episode_id'])
        report = exercise(BenchHTTPClient('http://testserver', client=g.client),
                          'task1', tmp_path / 'run', monkeypatch, 'initialization')
        assert g.api.sessions[report['episode_id']].result is None
        assert report['evaluation_player_outcomes'] == {'player_0': 'unknown', 'player_1': 'unknown'}
