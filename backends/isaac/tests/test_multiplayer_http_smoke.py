"""HTTP smoke client against explicit CPU workers; no physical qualification."""
from copy import deepcopy
import io
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from bench.isaacsim.release import validate as validate_frozen_release
from scripts.validation import multiplayer_http_smoke
from scripts.validation.multiplayer_http_smoke import PlayerProbe
from test_agent_service import frozen_inputs
from test_board_game_service import game as board_game
from test_turn_game_service import game as turn_game


@pytest.mark.parametrize('task_id', ['tic_tac_toe_3x3', 'connect_four_5x5', 'hex7x7'])
def test_board_smoke_checks_frozen_release_routes_and_owned_controller(board_game, tmp_path, task_id):
    with board_game(task_id=task_id, track='system0') as game:
        output = tmp_path/'probe'
        output.mkdir()
        report = {}
        probe = PlayerProbe(game.client, output, report, 'board', game.created)
        probe.exercise(expected_player='player_0')
        assert report['result']['episode']['outcome'] == 'cancelled'
        assert report['result']['board_game']['turn_index'] == 0
        assert report['result']['episode']['action_attempts'] == 3
        assert set(report['read_checks']) == {'player_0', 'player_1'}


@pytest.mark.parametrize('starting_player', ['alpha', 'beta'])
def test_named_instance_turn_smoke_uses_current_player_not_fixed_player_id(turn_game, tmp_path, starting_player):
    with turn_game() as game:
        # The CPU double supplies a reset transition, not physical evidence.
        game.worker.state.update(current_player_id=starting_player, public_progress={'status': 'reset_pending'})
        game.session.turn_game.state = game.session.turn_game.state.model_copy(update={
            'current_player_id': starting_player, 'public_progress': {'status': 'reset_pending'}})
        game.created['turn_game'] = game.session.turn_game.state.model_dump(mode='json')
        original = game.worker.request
        def request(operation, **payload):
            reply = original(operation, **payload)
            if operation == 'step':
                if game.worker.raw['sim_time_s'] >= .4:
                    game.worker.state['public_progress']['status'] = 'await_intent'
                reply['turn_game'] = deepcopy(game.worker.state)
            return reply
        game.worker.request = request
        output = tmp_path/'probe'
        output.mkdir()
        report = {}
        probe = PlayerProbe(game.client, output, report, 'turn', game.created)
        probe.exercise(expected_player=starting_player)
        assert report['result']['episode']['outcome'] == 'cancelled'
        assert report['result']['turn_game']['turn_index'] == 0
        assert report['result']['episode']['action_attempts'] == 5
        assert set(report['read_checks']) == {'alpha', 'beta'}
        assert report['ready_state']['public_progress']['status'] == 'await_intent'
        assert [phase['arguments']['mode'] for phase in report['phases'][:2]] == ['acknowledged_drive'] * 2


def test_smoke_refuses_wrong_starting_player_before_robot_commands(turn_game, tmp_path):
    with turn_game() as game:
        output = tmp_path/'probe'
        output.mkdir()
        probe = PlayerProbe(game.client, output, {}, 'turn', game.created)
        with pytest.raises(RuntimeError, match='Unexpected starting player'):
            probe.exercise(expected_player='beta')
        assert not [row for row in game.worker.calls if row[0] == 'step']


def test_smoke_accepts_frozen_scene_cancel_reason_without_changing_it(turn_game, tmp_path, monkeypatch):
    from bench.isaacsim import release as release_api
    from bench.isaacsim.release import load_evaluator, task_inputs
    from scenes.workrooms.catalog import task_definition
    from scenes.workrooms.living_room.runtime import LivingTaskRuntime

    # Use the real adapter and the current immutable referee; no physics or
    # native observation is fabricated for the scene cancellation itself.
    release = Path(task_definition('living_room', 'tower_extraction')['source_release']['path'])
    # The generic turn fixture substitutes its own release validator. Restore
    # the real reader only while loading this independent immutable referee.
    with monkeypatch.context() as actual:
        actual.setattr(release_api, 'validate', validate_frozen_release)
        _, _, task, _ = task_inputs(release, 'tower_extraction')
        evaluator = load_evaluator(release)(task)
    native = LivingTaskRuntime.__new__(LivingTaskRuntime)
    native.task, native.evaluator = task, evaluator
    native.env, native.events = SimpleNamespace(time_s=0.), io.StringIO()
    frozen_reply = native.request({'type': 'end_episode', 'reason': 'cancelled'})
    assert frozen_reply['turn_game']['status'] == 'interrupted'
    assert frozen_reply['turn_game']['terminal_reason'] == 'timeout'
    assert frozen_reply['turn_game']['winner_player_id'] is None

    with turn_game() as game:
        original = game.worker.request
        def request(operation, **payload):
            reply = original(operation, **payload)
            if operation == 'turn_game' and payload['command']['type'] == 'end_episode':
                game.worker.state['terminal_reason'] = frozen_reply['turn_game']['terminal_reason']
                reply['turn_game'] = deepcopy(game.worker.state)
            return reply
        game.worker.request = request
        output = tmp_path/'probe'
        output.mkdir()
        report = {}
        PlayerProbe(game.client, output, report, 'turn', game.created).exercise()
        assert report['result']['episode']['outcome'] == 'cancelled'
        assert report['result']['turn_game']['terminal_reason'] == 'timeout'


def test_smoke_rejects_completed_winner_but_preserves_terminal_evidence(turn_game, tmp_path):
    from bench.protocol.turn_games import TurnGameResult
    with turn_game() as game:
        output = tmp_path/'probe'
        output.mkdir()
        report = {}
        probe = PlayerProbe(game.client, output, report, 'turn', game.created)
        original = probe.request
        def request(method, path, *args, **kwargs):
            reply = original(method, path, *args, **kwargs)
            if method == 'GET' and path.endswith('/result'):
                reply['turn_game'].update(status='completed', winner_player_id='alpha',
                                         loser_player_id='beta', terminal_reason='collapse')
                # This is a valid shared result shape, but outside this
                # cancellation-only smoke's no-game-credit expectation.
                TurnGameResult.model_validate(reply)
            return reply
        probe.request = request
        with pytest.raises(RuntimeError, match='Smoke manufactured a turn or winner'):
            probe.exercise()
        assert report['result']['turn_game']['winner_player_id'] == 'alpha'
        assert report['result']['episode']['outcome'] == 'cancelled'


def cli_arguments(tmp_path):
    return ['multiplayer_http_smoke', 'run', '--output', str(tmp_path/'output'),
            '--release', str(tmp_path/'release'), '--task-id', 'tower_extraction',
            '--kind', 'turn', '--gpu-uuid', 'GPU-test']


@pytest.mark.parametrize('limits', [
    ['--wall-time-s', '2400.1'], ['--wall-time-s', '0'], ['--wall-time-s', 'nan'],
    ['--action-wall-timeout-s', '2401'], ['--action-wall-timeout-s', '0'],
    ['--action-wall-timeout-s', 'inf'],
    ['--wall-time-s', '300', '--action-wall-timeout-s', '600'],
])
def test_cli_rejects_invalid_wall_limits_before_launch(monkeypatch, tmp_path, limits):
    monkeypatch.setattr(sys, 'argv', cli_arguments(tmp_path)+limits)
    def forbidden(args):
        pytest.fail('Invalid budgets must not launch the native harness')
    monkeypatch.setattr(multiplayer_http_smoke, 'run', forbidden)
    with pytest.raises(SystemExit) as error:
        multiplayer_http_smoke.main()
    assert error.value.code == 2


def test_cli_default_wall_limits_cover_recorded_slow_native_smoke(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, 'argv', cli_arguments(tmp_path))
    received = []
    monkeypatch.setattr(multiplayer_http_smoke, 'run', lambda args: received.append(args) or 0)
    assert multiplayer_http_smoke.main() == 0
    assert received[0].wall_time_s == 2400.
    assert received[0].action_wall_timeout_s == 600.
