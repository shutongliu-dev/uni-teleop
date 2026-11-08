"""CPU HTTP tests for operator budgets/targets; no physical manipulation claim."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient
import pytest

from agent.model_client.bench_http import BenchHTTPClient
from agent.model_client.card_games import committed_operation_counts
from agent.model_client.function_runner import RunConfig, run
from bench.paths import load_json
from bench.runtime.controllers import default_registry
from bench.runtime.server import create_app
from bench.runtime.service import AgentService
from scenes.workrooms.catalog import task_definition
from test_board_game_service import FakeBoardWorker
from test_card_game_http_lifecycle import SyntheticCardWorker
from test_function_runner import ScriptedModel


@pytest.mark.parametrize('value', [None, True, 0., -1., float('nan'), float('inf')])
def test_invalid_action_horizon_fails_before_io(tmp_path, value):
    with pytest.raises(ValueError):
        run(RunConfig('https://unused.invalid', 'task1', tmp_path / 'out', max_action_sim_s=value))
    assert not (tmp_path / 'out').exists()


@pytest.mark.parametrize('value', [True, 0, -1, 1.5, '1'])
def test_invalid_commit_target_fails_before_io(tmp_path, value):
    with pytest.raises(ValueError):
        run(RunConfig('https://unused.invalid', 'task1', tmp_path / 'out',
                      stop_after_commits_per_player=value))
    assert not (tmp_path / 'out').exists()


@pytest.mark.parametrize('cap,completed', [(5., False), (20., True)])
def test_action_horizon_is_a_frozen_episode_allocation(tmp_path, cap, completed):
    release = Path(task_definition('card_room', 'tic_tac_toe_3x3')['source_release']['path'])
    service = AgentService(tmp_path / 'episodes', release=release,
        registry=default_registry(release), worker_factory=FakeBoardWorker)
    model = ScriptedModel([{'function': 'hold', 'arguments': {'duration_s': 5.2}}, {'stop': True}])
    with TestClient(create_app(service=service)) as http:
        report = run(RunConfig('http://testserver', 'task1', tmp_path / 'run', max_action_sim_s=cap),
            bench=BenchHTTPClient('http://testserver', client=http),
            models={'player_0': model, 'player_1': ScriptedModel([])})
    assert report['run_error'] is None, report
    request = load_json(tmp_path / 'run/create_request.json')
    assert request['budgets']['max_action_sim_s'] == cap
    feedback = load_json(tmp_path / 'run/player_0/decisions/0000/execution.json')
    assert feedback['ok'] is completed
    if completed:
        assert feedback['result']['status'] == 'completed'
        assert feedback['result']['elapsed_sim_s'] == pytest.approx(5.2)
    else:
        assert feedback['error']['code'] == 'invalid_arguments'
    assert not report['result_valid_for_scoring']


def test_target_counts_both_players_and_excludes_pass_and_reveal_notices():
    rows = [
        {'type': 'operation_committed', 'player_id': 'player_0', 'details': {'operation': 'place_piece'}},
        {'type': 'operation_committed', 'player_id': 'player_0', 'details': {'operation': 'place_piece'}},
        {'type': 'operation_committed', 'player_id': 'player_1', 'details': {'operation': 'pass'}},
        {'type': 'card_revealed', 'player_id': 'player_1', 'details': {}},
    ]
    players = ('player_0', 'player_1')
    delivery = SimpleNamespace(clients=dict.fromkeys(players), events={key: deepcopy(rows) for key in players})
    assert committed_operation_counts(delivery) == {'player_0': 2, 'player_1': 0}
    for values in delivery.events.values():
        values.append({'type': 'operation_committed', 'player_id': 'player_1',
                       'details': {'operation': 'place_piece'}})
    assert committed_operation_counts(delivery) == {'player_0': 2, 'player_1': 1}


def test_target_requires_two_authoritative_commits_and_never_invents_a_winner(tmp_path):
    release = Path(task_definition('card_room', 'barrier_race')['source_release']['path'])
    def factory(**kwargs):
        worker = SyntheticCardWorker(**kwargs)
        worker.mode = 'execute_pending'  # Explicit synthetic evidence, not a robot qualification.
        return worker
    service = AgentService(tmp_path / 'episodes', release=release,
        registry=default_registry(release), worker_factory=factory)
    first = ScriptedModel([
        {'function': 'hold', 'arguments': {'duration_s': .5, 'mode': 'acknowledged_drive'}},
        {'intent': {'type': 'move_pawn', 'destination': [2, 1]}},
        {'function': 'hold', 'arguments': {'duration_s': .5, 'mode': 'acknowledged_drive'}},
    ])
    second = ScriptedModel([
        {'intent': {'type': 'move_pawn', 'destination': [2, 3]}},
        {'function': 'hold', 'arguments': {'duration_s': .5, 'mode': 'acknowledged_drive'}},
    ])
    with TestClient(create_app(service=service)) as http:
        report = run(RunConfig('http://testserver', 'barrier_race', tmp_path / 'run',
            family='card_room', stop_after_commits_per_player=1),
            bench=BenchHTTPClient('http://testserver', client=http),
            models={'player_0': first, 'player_1': second})
    assert report['run_error'] is None and report['finalization_error'] is None, report
    assert report['termination'] == 'requested_commits_completed'
    assert report['decisions'] == {'player_0': 3, 'player_1': 2}
    assert report['validation_target']['reached']
    assert report['validation_target']['observed_commits'] == {'player_0': 1, 'player_1': 1}
    assert report['result']['board_game']['commit_count'] == 2
    assert report['result']['board_game']['winner'] is None
    assert not report['result_valid_for_scoring']
    assert not report['validation_target']['game_completion_claimed']
