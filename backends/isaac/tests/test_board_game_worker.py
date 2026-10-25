"""Trusted native adapter tests with native-view doubles, not physics qualification."""
from copy import deepcopy
import json

import numpy as np
import pytest

from bench.paths import load_json
from bench.runtime.board_game_worker import BoardGameWorker
from scenes.board_games.tabletop import BoardGameMatch, GAME_IDS, geometry_for, task_specification
from test_tabletop_native_observer import FakeEnv, contact


@pytest.fixture(params=GAME_IDS)
def runtime(request):
    geometry = geometry_for(request.param)
    env = FakeEnv(BoardGameMatch(geometry))
    env.actuated_joint_names = env.joint_names = ['fr3_left_joint1', 'left_hand', 'fr3_right_joint1', 'right_hand']
    env._targets = np.array([[.1, .2, .3, .4]], dtype=np.float32)

    def validate(command):
        if (set(command) != {'joint_positions_rad', 'duration_s'} or command['duration_s'] != .02
                or set(command['joint_positions_rad']) != set(env.joint_names)
                or any(type(value) not in (int, float) or not np.isfinite(value)
                       for value in command['joint_positions_rad'].values())):
            raise ValueError('Invalid native command')
    env._validate_action = validate
    scene = {'board_game_task': {'schema': 'board_game_native_task.v1', 'board_fixture_id': 'board',
        'players': [{'player_id': 'player_0', 'color': 'black', 'robot_id': 'left'},
                    {'player_id': 'player_1', 'color': 'white', 'robot_id': 'right'}]}}
    game = BoardGameWorker(env, scene, task_specification(geometry))
    yield game
    if not game.closed:
        game.close(reason='cancelled')


def actor(game, player=None):
    return {'player_id': player or game.match.rules.current_player, 'turn_index': game.match.rules.turn_index}


def placement(piece='black_00', q=1, r=1):
    return {'type': 'placement_intent', 'piece_id': piece, 'q': q, 'r': r}


def command(game, **targets):
    return {'joint_positions_rad': {**game._drive_targets, **targets}, 'duration_s': .02}


def native_place(game, name, q, r, steps=32):
    game.env.place(name, game.match.geometry.cell_center(q, r))
    game.env.next_rows = [contact(name) for name in [*game.match.committed, name]]
    for _ in range(steps):
        game.env.advance()


def test_intent_cancel_reselect_never_advance_physics_or_commit(runtime):
    game = runtime
    initial_state = game.env.batch.transforms.copy()
    accepted = game.request(placement(), actor=actor(game))
    assert 'board_game_error' not in accepted
    intent = accepted['board_game']['pending_move']
    assert intent['attempt_index'] == 1
    assert game.env.acquisitions == game.match.rules.turn_index == 0
    assert not game.match.committed
    np.testing.assert_array_equal(game.env.batch.transforms, initial_state)
    same = game.request(placement(), actor=actor(game))
    assert same['board_game']['pending_move'] == intent
    cancelled = game.request({'type': 'cancel_placement'}, actor=actor(game))
    assert cancelled['board_game']['pending_move'] is None
    selected = game.request(placement(q=2), actor=actor(game))
    assert selected['board_game']['pending_move']['attempt_index'] == 2
    assert game.env.acquisitions == game.match.rules.turn_index == 0


@pytest.mark.parametrize('bad', [
    {'type': 'placement_intent', 'piece_id': 'white_00', 'q': 1, 'r': 1},
    {'type': 'placement_intent', 'piece_id': 'black_00', 'q': -1, 'r': 1},
    {'type': 'placement_intent', 'piece_id': 'black_00', 'q': True, 'r': 1},
    {'type': 'placement_intent', 'piece_id': 'black_00', 'q': 1, 'r': 1, 'native_frame': {}},
    {'type': 'cancel_placement', 'object_position_m': [0, 0, 0]},
    {'type': 'joint_targets', 'robot_id': 'left'},
])
def test_invalid_game_requests_return_structured_errors_without_loss(runtime, bad):
    game = runtime
    game.request(placement(), actor=actor(game))
    before = deepcopy(game.public_state())
    response = game.request(bad, actor=actor(game))
    assert response['ok'] is True
    assert response['board_game_error']['code'] == 'invalid_arguments'
    assert response['board_game'] == before
    assert game.env.acquisitions == 0


def test_native_substep_commit_rejects_stale_turn_without_new_physics(runtime):
    game = runtime
    old_actor = actor(game)
    game.request(placement(), actor=old_actor)
    game.env.place('black_00', game.match.geometry.cell_center(1, 1))
    game.env.next_rows = [contact()]
    for _ in range(30):
        game.env.advance()
    assert game.public_state()['settlement_dwell_s'] == pytest.approx(.29)
    assert game.public_state()['commit_count'] == 0
    issued = command(game, fr3_left_joint1=.15)
    admitted = game.authorize_step(issued, old_actor)
    game.env.advance()  # 300 ms stable dwell completes inside this control interval.
    assert game.match.rules.turn_index == 1
    game.env.advance()  # Remainder remains observed; no drive rewrite or new command.
    reply = game.acknowledge_step(issued, admitted)
    assert reply['board_game_turn_changed'] is True
    assert reply['board_game']['current_player'] == 'player_1'
    assert reply['board_game']['commit_count'] == 1
    assert game.observer.samples == game.env.acquisitions == 32
    with pytest.raises(ValueError):
        game.authorize_step(issued, old_actor)
    assert game.env.acquisitions == 32
    assert game.authorize_step(command(game, fr3_right_joint1=.35), actor(game)) == 1
    encoded = json.dumps(reply)
    for private in ('committed', 'position_m', 'board_support', 'legal_cells', 'board":', 'native_substep'):
        assert private not in encoded


@pytest.mark.parametrize('bad_actor', [None, {}, {'player_id': 'player_1', 'turn_index': 0},
    {'player_id': 'player_0', 'turn_index': True}, {'player_id': 'player_0', 'turn_index': 1},
    {'player_id': 'player_0', 'turn_index': 0, 'permission': 'both'}])
def test_native_actions_require_trusted_current_actor(runtime, bad_actor):
    with pytest.raises(ValueError):
        runtime.authorize_step(command(runtime), bad_actor)
    assert runtime.env.acquisitions == 0


def test_idle_hand_keeps_commanded_target_not_measured_state(runtime):
    game = runtime
    # Actual q can lag drives; it is never a license to move the inactive hand.
    with pytest.raises(ValueError, match='other robot'):
        game.authorize_step(command(game, fr3_right_joint1=.2), actor(game))
    issued = command(game, fr3_left_joint1=.15)
    turn = game.authorize_step(issued, actor(game))
    game.acknowledge_step(issued, turn)
    assert game._drive_targets['fr3_left_joint1'] == .15
    assert game._drive_targets['fr3_right_joint1'] == pytest.approx(.3)


def test_recovery_diagnosis_does_not_expose_private_piece_identity(runtime):
    game = runtime
    game.request(placement(), actor=actor(game))
    game.env.place('white_00', game.match.geometry.cell_center(0, 0))
    game.env.next_rows = [contact('white_00')]
    game.env.advance()
    state = game.public_state()
    assert state['status']['recovery_required'] is True
    assert state['status']['reason'] == 'unrequested_piece_on_board'
    assert 'white_00' not in json.dumps(state)
    assert state['status']['retryable'] is True


@pytest.mark.parametrize('reason,outcome', [('time_limit', 'truncated'), ('step_limit', 'truncated'),
    ('runtime_error', 'aborted'), ('cancelled', 'aborted')])
def test_budget_and_runtime_endings_are_not_player_losses(runtime, reason, outcome):
    response = runtime.request({'type': 'end_episode', 'reason': reason})
    assert response['board_game']['status']['outcome'] == outcome
    assert response['board_game']['winner'] is None
    assert response['board_game']['current_player'] is None
    assert response['score']['success'] is False
    assert runtime.env.acquisitions == 0


def test_authenticated_players_cannot_force_runtime_end(runtime):
    response = runtime.request({'type': 'end_episode', 'reason': 'runtime_error'}, actor=actor(runtime))
    assert response['board_game_error']['code'] == 'invalid_arguments'
    assert not response['board_game']['terminal']


def test_native_win_survives_later_shutdown(runtime):
    game = runtime
    if game.match.geometry.game_id != 'tic_tac_toe_3x3':
        pytest.skip('The minimal complete match is L1')
    for piece_id, q, r in [('black_00', 0, 0), ('white_00', 0, 1), ('black_01', 1, 0),
                           ('white_01', 1, 1), ('black_02', 2, 0)]:
        response = game.request(placement(piece_id, q, r), actor=actor(game))
        assert 'board_game_error' not in response
        native_place(game, piece_id, q, r)
    assert game.match.outcome == 'win'
    assert game.score()['success'] is True
    response = game.request({'type': 'end_episode', 'reason': 'runtime_error'})
    assert response['board_game']['status']['outcome'] == 'win'
    assert response['board_game']['winner'] == 'player_0'
    assert response['board_game']['status']['end_reason'] is None


def test_final_skipped_acquisition_fails_closed(runtime):
    game = runtime
    game.env.world.current_time_step_index += 1
    game.env.world.current_time += game.env.dt
    game.env.time_s += game.env.dt
    with pytest.raises(RuntimeError, match='evidence verification'):
        game.close()
    assert game.closed
    assert game.public_state()['status']['outcome'] == 'aborted'


@pytest.mark.parametrize('game_id', GAME_IDS)
def test_current_frozen_releases_partition_all_native_robot_targets(game_id):
    from pathlib import Path
    from scenes.workrooms.catalog import task_definition
    definition = task_definition('card_room', game_id)
    root = Path(definition['source_release']['path'])
    if not root.is_dir():
        pytest.skip('Pinned task release is not installed')
    scene = load_json(root/'scene.json')
    task = load_json(root/'task.json')
    initial = load_json(root/'initial_state.json')
    env = FakeEnv(BoardGameMatch(geometry_for(game_id)))
    targets = initial['joint_positions_rad']
    env.actuated_joint_names = env.joint_names = list(targets)
    env._targets = np.array([list(targets.values())], dtype=np.float32)
    game = BoardGameWorker(env, scene, task)
    try:
        assert len(game.protocol.ownership) == 2
        for owner in game.protocol.ownership:
            assert len(owner.joint_names) == 29
            assert len([name for name in owner.joint_names if name.startswith('fr3_')]) == 7
        assert set(game._drive_targets) == set(targets)
    finally:
        game.close(reason='cancelled')
