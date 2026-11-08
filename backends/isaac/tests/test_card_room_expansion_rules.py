"""Pure rule checks, never physical or manipulation qualification."""
from copy import deepcopy
import json

import pytest

from scenes.workrooms.card_room.game_designs import TASK_IDS, design_for, memory_pairs_for_seed
from scenes.workrooms.card_room.expansion_rules import (
    BarrierRaceRules, DiskCurlingRules, IllegalMove, MemoryPairsRules,
    PhysicalEvidenceRequired, PolyominoTerritoryRules, public_rules, rotated_cells,
)


def memory():
    return MemoryPairsRules({f'card_{i:02d}': f'pair_{i//2}' for i in range(12)})


def assert_rejected_unchanged(game, operation, error=IllegalMove):
    before = deepcopy(game.__dict__)
    with pytest.raises(error):
        operation()
    assert game.__dict__ == before


def test_poly_same_inventory_area_and_real_rotation():
    game = PolyominoTerritoryRules()
    assert sum(len(p['cells']) for p in game.pieces.values() if p['owner'] == 'black') == 17
    assert sorted(tuple(map(tuple, p['cells'])) for p in game.pieces.values() if p['owner'] == 'black') == sorted(
        tuple(map(tuple, p['cells'])) for p in game.pieces.values() if p['owner'] == 'white')
    assert rotated_cells([(0, 0), (1, 0)], 90) == ((0, 0), (0, 1))
    game.place('black', 'black_tet_L', (4, 3))
    assert game.scores() == {'black': 4, 'white': 0}
    game.place('white', 'white_domino', (0, 0), 90)
    assert game.board[(0, 1)] == 'white_domino'


@pytest.mark.parametrize('operation', [
    lambda g: g.pass_turn('black'),
    lambda g: g.place('white', 'white_mono', (0, 0)),
    lambda g: g.place('black', 'white_mono', (0, 0)),
    lambda g: g.place('black', 'black_domino', (5, 0)),
    lambda g: g.place('black', 'black_domino', (0, 0), 45),
    lambda g: g.place('black', 'black_domino', (True, 0)),
])
def test_poly_invalid_requests_do_not_consume_turn(operation):
    game = PolyominoTerritoryRules()
    assert_rejected_unchanged(game, lambda: operation(game))


def test_poly_overlap_and_reuse_rejected():
    game = PolyominoTerritoryRules()
    game.place('black', 'black_mono', (0, 0))
    assert_rejected_unchanged(game, lambda: game.place('white', 'white_mono', (0, 0)))
    game.place('white', 'white_mono', (5, 5))
    assert_rejected_unchanged(game, lambda: game.place('black', 'black_mono', (1, 0)))


def test_poly_finished_by_no_remaining_fit_and_result_latched():
    game = PolyominoTerritoryRules()
    while not game.terminal:
        moves = game.legal_moves()
        if moves:
            piece, anchor, angle = moves[-1]
            game.place(game.current_player, piece, anchor, angle)
        else:
            game.pass_turn(game.current_player)
        assert game.turn_index <= 24
    scores = game.scores()
    assert game.status == 'completed'
    assert sum(scores.values()) == len(game.board) <= 36
    assert game.draw == (scores['black'] == scores['white'])
    before = game.result()
    assert game.interrupt('time_limit') == before


def test_barrier_straight_jump_and_goal():
    game = BarrierRaceRules()
    game.move_pawn('black', (2, 1))
    game.move_pawn('white', (2, 3))
    game.move_pawn('black', (2, 2))
    assert (2, 1) in game.legal_pawn_moves('white')
    assert (1, 2) not in game.legal_pawn_moves('white')
    game.move_pawn('white', (2, 1))
    game.move_pawn('black', (2, 3))
    game.move_pawn('white', (2, 0))
    assert game.winner == 'white'
    assert game.current_player is None


def test_barrier_blocked_jump_side_exits():
    game = BarrierRaceRules()
    game.move_pawn('black', (2, 1))
    game.move_pawn('white', (2, 3))
    game.place_wall('black', 'black_barrier_0', 'horizontal', (1, 0))
    game.move_pawn('white', (2, 2))
    assert set(game.legal_pawn_moves('white')) >= {(1, 1), (3, 1)}
    assert (2, 0) not in game.legal_pawn_moves('white')
    # The side-jump calculation is a query, never a turn or physical mutation.
    assert game.current_player == 'black'


def test_barrier_wall_crossing_overlap_and_boundary_rejected():
    game = BarrierRaceRules()
    game.place_wall('black', 'black_barrier_0', 'horizontal', (1, 1))
    for orientation, anchor in [('vertical', (1, 1)), ('horizontal', (2, 1)), ('horizontal', (4, 1))]:
        assert_rejected_unchanged(game, lambda o=orientation, a=anchor:
                                  game.place_wall('white', 'white_barrier_0', o, a))
    assert_rejected_unchanged(game, lambda: game.place_wall('white', 'black_barrier_1', 'vertical', (0, 0)))


def test_barrier_no_path_walls_are_rejected():
    game = BarrierRaceRules()
    # Enclose black's starting pawn on left/right; a horizontal cap would seal it.
    game.place_wall('black', 'black_barrier_0', 'vertical', (1, 0))
    game.place_wall('white', 'white_barrier_0', 'vertical', (2, 0))
    assert game.has_path('black') and game.has_path('white')
    assert_rejected_unchanged(game, lambda:
                              game.place_wall('black', 'black_barrier_1', 'horizontal', (1, 1)))


def test_memory_hidden_until_each_public_reveal_and_turn_after_return():
    game = memory()
    assert 'pair_' not in json.dumps(game.public_state())
    first = game.reveal('black', 'card_00')
    assert first['sequence'] == 0 and first['symbol'] == 'pair_0'
    assert game.phase == 'reveal_second' and game.current_player == 'black'
    second = game.reveal('black', 'card_02')
    assert second['sequence'] == 1 and second['symbol'] == 'pair_1'
    assert game.phase == 'return_mismatch' and game.current_player == 'black'
    assert_rejected_unchanged(game, lambda: game.reveal('black', 'card_04'))
    assert_rejected_unchanged(game, lambda: game.return_pair('black', ['card_00', 'card_03']))
    game.return_pair('black', ['card_02', 'card_00'])
    assert game.current_player == 'white' and game.turn_index == 1
    assert game.public_state()['face_up'] == []
    assert [e['type'] for e in game.events_after()] == [
        'card_revealed', 'card_revealed', 'pair_returned_face_down']
    assert 'pair_2' not in json.dumps(game.public_state())


def test_memory_match_needs_collection_and_winner_after_six_pairs():
    game = memory()
    for i in range(6):
        cards = [f'card_{2*i:02d}', f'card_{2*i+1:02d}']
        game.reveal('black', cards[0])
        assert_rejected_unchanged(game, lambda: game.reveal('black', cards[0]))
        game.reveal('black', cards[1])
        assert game.scores()['black'] == i
        assert_rejected_unchanged(game, lambda: game.return_pair('black', cards))
        game.collect_pair('black', cards)
    assert game.result()['winner'] == 'black'
    assert game.scores() == {'black': 6, 'white': 0}
    assert game.phase == 'complete'
    event = game.events_after()[0]
    event['symbol'] = 'tampered'
    assert game.events_after()[0]['symbol'] == 'pair_0'


def test_memory_private_map_rejects_wrong_pairs_and_no_map_default():
    with pytest.raises(TypeError):
        MemoryPairsRules()
    with pytest.raises(ValueError):
        MemoryPairsRules({f'card_{i:02d}': 'all_same' for i in range(12)})
    game = MemoryPairsRules(memory_pairs_for_seed(81))
    assert 'pair_' not in json.dumps(game.public_state())
    assert 'private_reset_pairs' not in json.dumps(public_rules('memory_pairs'))


def test_memory_equal_collections_are_a_draw():
    game = memory()
    game.reveal('black', 'card_00')
    game.reveal('black', 'card_02')
    game.return_pair('black', ['card_00', 'card_02'])
    for pair in range(3):
        cards = [f'card_{2*pair:02d}', f'card_{2*pair+1:02d}']
        for card in cards:
            game.reveal('white', card)
        game.collect_pair('white', cards)
    game.reveal('white', 'card_06')
    game.reveal('white', 'card_08')
    game.return_pair('white', ['card_06', 'card_08'])
    for pair in range(3, 6):
        cards = [f'card_{2*pair:02d}', f'card_{2*pair+1:02d}']
        for card in cards:
            game.reveal('black', card)
        game.collect_pair('black', cards)
    assert game.scores() == {'black': 3, 'white': 3}
    assert game.draw and game.winner is None


def shot(game, player, identifier, positions, **overrides):
    board = game.design['board']
    values = dict(contact_episodes=1, complete_native_evidence=True, physics_valid=True,
                  released=True, stable_duration_s=.3, positions_xy_m=positions,
                  launch_position_xy_m=board['launch_centers_xy_m'][player],
                  release_position_xy_m=[.65, board['launch_lines_y_m'][player]],
                  crossed_line_after_release=True, other_disk_robot_contacts=0,
                  reserve_inventory_preserved=True, on_deck={name: True for name in positions})
    values.update(overrides)
    return game.record_shot(player, identifier, **values)


def test_disk_final_scoring_uses_last_positions_not_launch_scores():
    game = DiskCurlingRules()
    center = game.design['board']['target_center_xy_m']
    positions = {}
    for i in range(8):
        owner = game.current_player
        identifier = f'{owner}_disk_{i//2}'
        positions[identifier] = center
        # White's last shot knocks every black disk outside the rings.
        if i == 7:
            positions = {name: (center if name.startswith('white') else (.18, -.45))
                         for name in positions}
        shot(game, owner, identifier, positions)
        if i < 7:
            assert game.status == 'active' and game.final_scores is None
    assert game.final_scores == {'black': 0, 'white': 12}
    assert game.winner == 'white'


def test_disk_boundary_center_scoring_and_offdeck():
    game = DiskCurlingRules()
    x, y = game.design['board']['target_center_xy_m']
    scores = game.score_positions({'black_disk_0': (x+.06, y), 'black_disk_1': (x+.12, y),
                                   'black_disk_2': (x+.18, y), 'black_disk_3': (4, y)},
                                  on_deck={f'black_disk_{i}': i < 3 for i in range(4)})
    assert scores == {'black': 6, 'white': 0}


def test_disk_below_table_center_never_scores_from_xy_alone():
    game = DiskCurlingRules()
    assert game.score_positions({'black_disk_0': (.65, 0.)}, on_deck={'black_disk_0': False}) == {'black': 0, 'white': 0}
    with pytest.raises(PhysicalEvidenceRequired):
        game.score_positions({'black_disk_0': (.65, 0.)}, on_deck={})


@pytest.mark.parametrize('values', [
    {'contact_episodes': 0}, {'complete_native_evidence': False},
    {'physics_valid': None}, {'released': False}, {'stable_duration_s': .299},
    {'positions_xy_m': {}}, {'positions_xy_m': {'black_disk_0': [float('nan'), 0]}},
    {'crossed_line_after_release': False}, {'reserve_inventory_preserved': False},
    {'on_deck': {}}, {'on_deck': {'black_disk_0': None}},
])
def test_disk_missing_evidence_never_commits(values):
    game = DiskCurlingRules()
    assert_rejected_unchanged(game, lambda: shot(game, 'black', 'black_disk_0',
        {'black_disk_0': [.65, 0]}, **values), PhysicalEvidenceRequired)


def test_disk_actual_second_push_is_rule_violation_not_physics_loss():
    game = DiskCurlingRules()
    result = shot(game, 'black', 'black_disk_0', {'black_disk_0': [.65, 0]}, contact_episodes=2)
    assert result['status'] == 'incomplete'
    assert result['winner'] is None and result['physics_valid'] is True
    assert result['termination_reason'] == 'shot_rule_violation'
    assert game.turn_index == 0


@pytest.mark.parametrize('owner', ['black', 'white'])
def test_disk_owner_launch_lines_are_symmetric(owner):
    game = DiskCurlingRules()
    if owner == 'white':
        shot(game, 'black', 'black_disk_0', {'black_disk_0': [.65, 0]})
    positions = {identifier: [.65, 0] for identifier in game.played}
    positions[f'{owner}_disk_0'] = [.65, 0]
    result = shot(game, owner, f'{owner}_disk_0', positions, release_position_xy_m=[.65, 0])
    assert result['termination_reason'] == 'shot_rule_violation'
    assert result['winner'] is None and result['physics_valid'] is True


def test_disk_physical_failure_is_distinct_from_rule_violation():
    game = DiskCurlingRules()
    result = shot(game, 'black', 'black_disk_0', {'black_disk_0': [.65, 0]}, physics_valid=False)
    assert result['termination_reason'] == 'physical_failure'
    assert result['winner'] is None and result['physics_valid'] is False


@pytest.mark.parametrize('factory', [PolyominoTerritoryRules, BarrierRaceRules, memory, DiskCurlingRules])
@pytest.mark.parametrize('reason', ['time_limit', 'step_limit', 'cancelled', 'runtime_error', 'physical_failure'])
def test_interruption_never_assigns_winner(factory, reason):
    game = factory()
    result = game.interrupt(reason, physics_valid=False if reason == 'physical_failure' else None)
    assert result['status'] == 'incomplete' and result['winner'] is None and not result['draw']
    assert result['termination_reason'] == reason
    assert game.current_player is None


def test_expansion_catalog_loads_native_rules_without_claiming_robot_success():
    from scenes.workrooms.catalog import list_tasks, load_task, task_definition
    from scenes.workrooms.card_room.expansion_referee import public_rule_contract
    current = list_tasks()
    assert len(current) == 51
    assert set(TASK_IDS) <= {task['task_id'] for task in current}
    for task_id in TASK_IDS:
        for profile in ('original', 'l_mount'):
            definition = task_definition('card_room', task_id, robot_profile=profile)
            assert definition['status'] == 'implemented' and definition['runnable'] is True
            assert definition['source_release']['manifest_sha256']
            assert definition['evaluator']['implementation'].endswith('.ExpansionMatch')
            assert definition['scene_candidate']['public_game_session_implemented']
            # The L-mount revision publishes its own uninherited qualification record.
            claim = 'native_manipulation' if profile == 'original' else 'robot_manipulation'
            assert not definition['qualification'][claim]
            loaded = load_task('card_room', task_id, robot_profile=profile)
            assert loaded['task']['public_rules'] == public_rule_contract(task_id)
            assert not loaded['release_manifest']['physics_accepted']


@pytest.mark.parametrize('task_id', TASK_IDS)
def test_public_contract_and_candidate_spec_do_not_leak_pairs(task_id):
    from scenes.workrooms.card_room.expansion_tasks import candidate_specification
    value = candidate_specification(task_id)
    assert value['runnable'] is True
    assert value['environment']['version'] == 'card_room_heritage_20261017_v3'
    encoded = json.dumps(value, allow_nan=False)
    assert 'private_reset_pairs' not in encoded
    if task_id == 'memory_pairs':
        assert not any(f'pair_{i}' in encoded for i in range(6))
