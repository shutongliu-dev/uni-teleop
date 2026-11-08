"""Public expansion schemas and event receipts; no physical success is inferred."""
from copy import deepcopy
from threading import RLock
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from bench.paths import load_json
from bench.protocol.card_games import (CardGameEventPage, CardGameState,
    EVENT_ADAPTER, INTENT_ADAPTER, card_game_schemas)
from bench.runtime.card_games import CardGameSession
from bench.runtime.service import ServiceError


def state(**updates):
    value = dict(protocol='bench_card_game_state_v1', task_id='memory_pairs', turn_index=0,
        round_index=0, current_player='player_0', terminal=False, winner=None, draw=False,
        pending_move=None, status=dict(outcome='ongoing', end_reason=None,
        recovery_required=False, evidence_unavailable=False, reason='await_intent', retryable=True),
        settlement_dwell_s=0., commit_count=0, phase='reveal_first', event_cursor=-1)
    value.update(updates)
    return value


def reveal(sequence=0):
    return dict(sequence=sequence, type='card_revealed', task_id='memory_pairs', player_id='player_0',
        turn_index=1, native_substep_index=1000, simulation_time_s=.5,
        details=dict(card_id='card_00', symbol='pair_2', expected=True))


def commit(sequence=1):
    value = reveal(sequence)
    value.update(type='operation_committed', details=dict(operation='reveal_card', piece_ids=['card_00']))
    return value


def receipt(value, events, changed):
    return dict(board_game=value, card_game_events=events, board_game_turn_changed=changed,
        score=dict(success=value['status']['outcome'] in ('win', 'draw'),
            outcome=value['status']['outcome'], winner=value['winner'], draw=value['draw'],
            turn_index=value['turn_index'], end_reason=value['status']['end_reason'], reason=value['status']['reason']))


def facade(pending=False):
    value = CardGameSession.__new__(CardGameSession)
    initial = state()
    if pending:
        initial['pending_move'] = dict(player_id='player_0', type='reveal_card', piece_ids=['card_00'],
                                       parameters={}, turn_index=0, attempt_index=0)
    value.state = CardGameState(**initial)
    value._public_events = []
    value.bindings = dict(player_0=object(), player_1=object())
    value.session = SimpleNamespace(lock=RLock(), score={}, result=None)
    return value


@pytest.mark.parametrize('payload', [
    dict(type='place_piece', piece_id='black_tet_L', anchor=[1, 2], rotation_deg=90),
    dict(type='pass'), dict(type='move_pawn', destination=[2, 3]),
    dict(type='place_barrier', piece_id='white_barrier_0', anchor=[1, 2], orientation='vertical'),
    dict(type='reveal_card', card_id='card_11'), dict(type='collect_pair'), dict(type='flip_back'),
    dict(type='shoot_disk', piece_id='black_disk_0'), dict(type='cancel_intent'),
])
def test_intents_accept_only_public_declarations(payload):
    assert INTENT_ADAPTER.validate_python(payload).model_dump() == payload
    for key, value in dict(player_id='player_1', position_m=[.65, 0., .8],
                           physics_valid=True, private_pairs={}, complete_native_evidence=True).items():
        with pytest.raises(ValidationError):
            INTENT_ADAPTER.validate_python({**payload, key: value})


@pytest.mark.parametrize('value', [True, 1., '1', -1, 6])
def test_cells_do_not_coerce_actor_inputs(value):
    with pytest.raises(ValidationError):
        INTENT_ADAPTER.validate_python(dict(type='place_piece', piece_id='black_mono', anchor=[value, 0]))


@pytest.mark.parametrize('value', [True, '90', 90., 45, -90, 360])
def test_rotations_are_strict_quarter_turns(value):
    with pytest.raises(ValidationError):
        INTENT_ADAPTER.validate_python(dict(type='place_piece', piece_id='black_mono', anchor=[0, 0], rotation_deg=value))


def test_operation_epoch_is_not_player_parity_and_does_not_end_memory_round():
    after = CardGameState(**state(turn_index=1, commit_count=1, phase='reveal_second', event_cursor=0))
    assert after.current_player == 'player_0' and after.round_index == 0
    with pytest.raises(ValidationError):
        CardGameState(**state(turn_index=1, commit_count=0))


@pytest.mark.parametrize('reason', ['missing_physical_evidence', 'physical_failure', 'shot_rule_violation', 'rule_violation'])
def test_observed_faults_never_invent_winners(reason):
    value = state(terminal=True, current_player=None)
    value['status'].update(outcome='aborted', end_reason=reason, reason=reason, retryable=False)
    assert CardGameState(**value).winner is None
    with pytest.raises(ValidationError):
        CardGameState(**{**value, 'winner': 'player_1'})


def test_event_payload_cannot_smuggle_unrevealed_identities_or_native_truth():
    event = reveal()
    assert EVENT_ADAPTER.validate_python(event).details.symbol == 'pair_2'
    for key, value in dict(private_pairs={'card_01': 'pair_1'}, position_m=[0., 0., 1.]).items():
        changed = deepcopy(event); changed['details'][key] = value
        with pytest.raises(ValidationError):
            EVENT_ADAPTER.validate_python(changed)
    with pytest.raises(ValidationError):
        EVENT_ADAPTER.validate_python({**event, 'task_id': 'disk_curling'})
    with pytest.raises(ValidationError):
        CardGameState(**{**state(), 'private_pairs': {}})


def test_same_player_reveal_stops_prior_operation_and_broadcasts_both_players():
    game = facade(pending=True)
    value = state(turn_index=1, commit_count=1, phase='reveal_second', event_cursor=1)
    assert game.acknowledge(receipt(value, [reveal(), commit()], True), SimpleNamespace(joint_positions_rad={}))
    a = game.events('player_0'); b = game.events('player_1')
    assert a == b and a['next_cursor'] == 1 and not a['has_more']
    assert a['events'][0]['details']['symbol'] == 'pair_2'
    a['events'][0]['details']['symbol'] = 'pair_5'
    assert game.events('player_0')['events'][0]['details']['symbol'] == 'pair_2'
    game.session.result = object()  # Worker may already be closed.
    assert game.events('player_1') == b
    assert game.events('player_0', 1) == {'events': [], 'next_cursor': 1, 'has_more': False}
    page = game.events('player_1', limit=1)
    assert page['next_cursor'] == 0 and page['has_more']
    assert game.events('player_1', after=0)['events'] == [commit()]


@pytest.mark.parametrize('mutation', ['missing', 'gap', 'duplicate', 'future_epoch', 'wrong_game'])
def test_bad_event_receipt_rejected_atomically(mutation):
    game = facade()
    value = state(turn_index=1, commit_count=1, event_cursor=0)
    events = [reveal()]
    if mutation == 'missing': events = []
    elif mutation == 'gap': events[0]['sequence'] = 1
    elif mutation == 'duplicate': events = [reveal(), reveal()]
    elif mutation == 'future_epoch': events[0]['turn_index'] = 2
    else: events[0]['task_id'] = 'disk_curling'
    with pytest.raises((RuntimeError, ValidationError)):
        game.acknowledge(receipt(value, events, True), SimpleNamespace(joint_positions_rad={}))
    assert game.state.turn_index == 0 and game._public_events == []


def test_intent_rpc_cannot_emit_a_reveal_and_paging_cannot_skip_history():
    game = facade()
    value = state(event_cursor=0)
    with pytest.raises(RuntimeError, match='Only native physics'):
        game.acknowledge(receipt(value, [reveal()], False))
    for after, limit in [(0, 10), (-2, 10), (True, 10), (-1, 257), (-1, 0)]:
        with pytest.raises(ServiceError): game.events('player_0', after, limit)
    with pytest.raises(ValidationError):
        CardGameEventPage(events=[reveal(0), reveal(2)], next_cursor=2, has_more=False)
    assert card_game_schemas()['intent']['discriminator']['propertyName'] == 'type'


@pytest.mark.parametrize('reason,execution,physical', [
    ('time_limit', 'sim_timeout', True), ('step_limit', 'action_budget', True),
    ('runtime_error', 'infrastructure_error', False), ('cancelled', 'cancelled', True),
    ('physical_failure', 'safety_stop', False),
    ('missing_physical_evidence', 'evaluator_invalid', False),
    ('shot_rule_violation', 'task_failure', True), ('rule_violation', 'task_failure', True),
])
def test_terminal_native_fault_stops_execution_without_a_game_winner(reason, execution, physical):
    game = facade()
    assert game.execution_termination() is None
    value = state(terminal=True, current_player=None)
    value['status'].update(outcome='truncated' if reason.endswith('_limit') else 'aborted',
                           end_reason=reason, reason=reason, retryable=False)
    game.state = CardGameState(**value)
    assert game.execution_termination() == (execution, physical)
    assert game.state.winner is None and game.state.draw is False


@pytest.mark.parametrize('mutation', ['missing_commit', 'wrong_operation', 'wrong_actor', 'backwards_time', 'backwards_step'])
def test_each_native_epoch_binds_its_admitted_operation_and_monotonic_events(mutation):
    game = facade(pending=True)
    events = [reveal(), commit()]
    if mutation == 'missing_commit': events = [reveal()]
    elif mutation == 'wrong_operation': events[-1]['details']['operation'] = 'flip_back'
    elif mutation == 'wrong_actor': events[-1]['player_id'] = 'player_1'
    elif mutation == 'backwards_time': events[-1]['simulation_time_s'] = .4
    else: events[-1]['native_substep_index'] = 999
    value = state(turn_index=1, commit_count=1, event_cursor=len(events)-1)
    with pytest.raises(RuntimeError):
        game.acknowledge(receipt(value, events, True), SimpleNamespace(joint_positions_rad={}))
    assert game.state.turn_index == 0 and not game._public_events


def test_budget_supersedes_late_success_without_erasing_its_private_event_record(tmp_path):
    game = facade(pending=True)
    before = game.state.model_dump(mode='json')
    final = state(turn_index=1, commit_count=1, terminal=True, current_player=None,
                  winner='player_0', event_cursor=1)
    final['status'].update(outcome='win', reason='committed', retryable=False)
    game.acknowledge(receipt(final, [reveal(), commit()], True), SimpleNamespace(joint_positions_rad={}))
    game.session.directory = tmp_path
    (tmp_path/'private').mkdir()
    game.late_terminal_state = lambda outcome: deepcopy(before)
    game.end('sim_timeout')
    assert game.state.status.outcome == 'truncated' and game.state.winner is None
    assert game.state.event_cursor == -1 and game._public_events == []
    assert game.events('player_0') == game.events('player_1') == dict(events=[], next_cursor=-1, has_more=False)
    import json
    late = load_json(tmp_path/'private/late_card_game_events.json')
    assert late['late_native_events'] == [reveal(), commit()]
