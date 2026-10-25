"""Public tabletop schemas preserve referee semantics and reject private input."""
from copy import deepcopy
from dataclasses import asdict
import json

import pytest
from pydantic import ValidationError

from bench.protocol.board_games import (BoardGameResult, BoardGameState, INTENT_ADAPTER,
    PlayerBinding, PlayerSession, board_game_schemas)
from scenes.board_games.tabletop import BoardGameMatch, geometry_for
from test_agent_bench_contracts import result, spec
from test_tabletop_games import frame, settle


def state(match):
    return dict(task_id=match.geometry.game_id, turn_index=match.rules.turn_index,
        current_player=None if match.terminal else match.rules.current_player,
        terminal=match.terminal, winner=match.rules.winner, draw=match.rules.draw,
        pending_move=asdict(match.pending) if match.pending else None,
        status=match.public_status(),
        settlement_dwell_s=match.last_result.dwell_s if match.last_result else 0.,
        commit_count=len(match.commit_log))


@pytest.mark.parametrize('game_id', ['tic_tac_toe_3x3', 'connect_four_5x5', 'hex7x7'])
def test_current_native_match_projection_survives_intent_settlement_and_handoff(game_id):
    match = BoardGameMatch(geometry_for(game_id))
    assert BoardGameState(**state(match)).current_player == 'player_0'
    match.begin_move('player_0', 'black_00', 1, 1)
    match.observe(frame(match, 0))
    pending = BoardGameState(**state(match))
    assert pending.pending_move.piece_id == 'black_00'
    assert pending.status.outcome == 'ongoing' and pending.status.retryable
    for index in range(1, 31):
        match.observe(frame(match, index))
    assert match.last_result.committed
    committed = BoardGameState(**state(match))
    assert committed.commit_count == committed.turn_index == 1
    assert committed.current_player == 'player_1' and committed.pending_move is None
    assert committed.settlement_dwell_s == pytest.approx(.3)


@pytest.mark.parametrize('reason,outcome', [('time_limit', 'truncated'), ('step_limit', 'truncated'),
    ('runtime_error', 'aborted'), ('cancelled', 'aborted')])
def test_interruption_preserves_pending_without_assigning_an_opponent_win(reason, outcome):
    match = BoardGameMatch()
    match.begin_move('player_0', 'black_00', 1, 1)
    match.end_episode(reason)
    projected = BoardGameState(**state(match))
    assert projected.terminal and projected.current_player is None
    assert projected.winner is None and not projected.draw
    assert projected.status.outcome == outcome and projected.status.end_reason == reason
    assert not projected.status.retryable
    assert projected.pending_move.piece_id == 'black_00'


def played_state(moves):
    match = BoardGameMatch()
    next_index = 0
    for turn, cell in enumerate(moves):
        player, color = ('player_0', 'black') if turn % 2 == 0 else ('player_1', 'white')
        match.begin_move(player, f'{color}_{turn // 2:02d}', *cell)
        next_index = settle(match, next_index)
    return BoardGameState(**state(match))


def test_verified_win_and_draw_have_distinct_typed_results():
    win = played_state(((0, 0), (1, 0), (0, 1), (1, 1), (0, 2)))
    assert win.status.outcome == 'win' and win.winner == 'player_0' and not win.draw
    draw = played_state(((0, 0), (1, 0), (2, 0), (1, 1), (0, 1), (2, 1), (1, 2), (0, 2), (2, 2)))
    assert draw.status.outcome == 'draw' and draw.winner is None and draw.draw
    # A fault after the accepted game result is not an opponent win or a draw.
    execution = result(outcome='infrastructure_error', spec=spec(task_id='tic_tac_toe_3x3'))
    combined = BoardGameResult(board_game=win, episode=execution)
    assert combined.board_game.status.outcome == 'win'
    assert combined.episode.outcome == 'infrastructure_error'


@pytest.mark.parametrize('payload', [
    {'type': 'placement_intent', 'piece_id': 'black_00', 'q': 1, 'r': 1},
    {'type': 'cancel_placement'},
])
def test_intents_have_no_actor_identity_and_never_contain_physics(payload):
    assert INTENT_ADAPTER.validate_python(payload).model_dump() == payload
    for field, value in [('player_id', 'player_1'), ('robot_id', 'right'),
                         ('position_m', [0., 0., 0.]), ('board', [])]:
        with pytest.raises(ValidationError):
            INTENT_ADAPTER.validate_python({**payload, field: value})


@pytest.mark.parametrize('value', [True, False, '1', 1., -1, 7])
@pytest.mark.parametrize('axis', ['q', 'r'])
def test_cells_reject_coercions_and_non_board_coordinates(value, axis):
    payload = {'type': 'placement_intent', 'piece_id': 'black_00', 'q': 1, 'r': 1, axis: value}
    with pytest.raises(ValidationError):
        INTENT_ADAPTER.validate_python(payload)


@pytest.mark.parametrize('field,value', [('board', []), ('legal_cells', []), ('pieces', {}),
    ('native_substep_index', 10), ('piece_positions_m', {}), ('commit_log', [])])
def test_public_state_rejects_native_evidence_and_logical_occupancy(field, value):
    with pytest.raises(ValidationError):
        BoardGameState(**{**state(BoardGameMatch()), field: value})


@pytest.mark.parametrize('change', [
    {'terminal': True}, {'winner': 'player_1'}, {'draw': True}, {'current_player': 'player_1'},
    {'turn_index': 1}, {'commit_count': 1}, {'terminal': 0}, {'settlement_dwell_s': float('nan')},
    {'status': {'reason': 'committed_piece_needs_recovery:black_00:piece_not_upright'}},
    {'status': {'outcome': 'aborted', 'end_reason': 'time_limit', 'retryable': False}},
    {'status': {'retryable': False}},
])
def test_state_rejects_contradictory_outcomes_private_reasons_and_coercions(change):
    payload = state(BoardGameMatch())
    changes = deepcopy(change)
    if 'status' in changes:
        changes['status'] = {**payload['status'], **changes['status']}
    with pytest.raises(ValidationError):
        BoardGameState(**{**payload, **changes})


def test_state_rejects_wrong_task_cells_and_pending_ownership():
    match = BoardGameMatch()
    match.begin_move('player_0', 'black_00', 1, 1)
    payload = state(match)
    for change in ({'q': 3}, {'player_id': 'player_1'}, {'piece_id': 'white_00'}, {'turn_index': 1}):
        with pytest.raises(ValidationError):
            BoardGameState(**{**payload, 'pending_move': {**payload['pending_move'], **change}})


def test_session_binding_is_fixed_and_token_not_in_repr():
    binding = dict(player_id='player_0', color='black', hand='left', robot_id='left',
                   joint_names=['left_arm', 'left_finger'])
    token = 'test_token_' + 'a' * 32
    session = PlayerSession(binding=binding, session_token=token)
    assert session.model_dump()['session_token'] == token
    assert token not in repr(session)
    for change in ({'hand': 'right'}, {'color': 'white'}, {'robot_id': 'right'},
                   {'joint_names': ['joint', 'joint']}):
        with pytest.raises(ValidationError):
            PlayerBinding(**{**binding, **change})


def test_final_result_requires_terminal_matching_task_and_schema_is_exportable():
    execution = result(spec=spec(task_id='tic_tac_toe_3x3'))
    with pytest.raises(ValidationError):
        BoardGameResult(board_game=state(BoardGameMatch()), episode=execution)
    match = BoardGameMatch(geometry_for('hex7x7'))
    match.end_episode('cancelled')
    with pytest.raises(ValidationError):
        BoardGameResult(board_game=state(match), episode=execution)
    assert json.loads(json.dumps(board_game_schemas()))['intent']['discriminator']['propertyName'] == 'type'
