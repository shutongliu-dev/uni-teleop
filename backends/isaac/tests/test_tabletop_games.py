"""Rules, metric layout and evidence admission; not simulation qualification."""
from dataclasses import replace
import json
import math

import pytest

from scenes.board_games.tabletop import (ActuatorOwnership, BoardGameGeometry, BoardGameMatch,
    BoardGamePlayerProtocol, BoardGameRules, COLORS, GAME_IDS, IllegalMove, NativeFrame,
    PieceEvidence, RGBD_MODE, SMOKE_MODE, geometry_for, task_specification)


def piece(position, **changes):
    return replace(PieceEvidence(tuple(position), (1., 0., 0., 0.), (0., 0., 0.),
                                 (0., 0., 0.), .001, 0., 0.), **changes)


def frame(match, index, **changes):
    pieces = {name: piece(match.geometry.reserve_position(name.split('_')[0], int(name[-2:])),
                          board_support_impulse_ns=0.) for name in match.inventory}
    for name, cell in match.committed.items():
        pieces[name] = piece(match.geometry.cell_center(*cell))
    if match.pending:
        intent = match.pending
        pieces[intent.piece_id] = piece(match.geometry.cell_center(intent.q, intent.r), **changes)
    return NativeFrame(index, index*.01, .01, pieces, True)


def settle(match, start):
    for index in range(start, start+31):
        result = match.observe(frame(match, index))
    assert result.committed, result
    return start+31


@pytest.mark.parametrize('game_id,total,width,clearance', [
    ('tic_tac_toe_3x3', 9, .46, .042), ('connect_four_5x5', 25, .615, .0295),
    ('hex7x7', 49, .94, .017)])
def test_inventory_metric_cells_and_lateral_reserves(game_id, total, width, clearance):
    geometry = geometry_for(game_id)
    assert len(BoardGameMatch(geometry).inventory) == total
    assert sum(geometry.inventory_counts.values()) == total
    assert geometry.inventory_counts['black'] == geometry.inventory_counts['white']+1
    assert geometry.board_size_xy_m[0] == pytest.approx(width)
    for q in range(geometry.size):
        for r in range(geometry.size):
            center = geometry.cell_center(q, r)
            assert geometry.footprint_clearance_m(q, r, center[:2], .028) == pytest.approx(clearance)
            assert len(geometry.cell_vertices(q, r)) == (6 if geometry.shape == 'hex' else 4)
            for adjacent in geometry.neighbors(q, r):
                assert math.dist(center, geometry.cell_center(*adjacent)) == pytest.approx(geometry.pitch_m)
    reserves = [geometry.reserve_position(color, i) for color, count in geometry.inventory_counts.items()
                for i in range(count)]
    assert len(set(reserves)) == total
    assert min(math.dist(a, b) for i, a in enumerate(reserves) for b in reserves[i+1:]) >= geometry.reserve_pitch_m-1e-12
    assert len(geometry.reserve_stations()) == 4
    for x, y, z in reserves:
        assert 0. <= x <= 1.30
        assert -.80 <= y <= .80
        assert z == pytest.approx(.807)
        # Keep the near-base centre corridor clear at actual pickup height.
        own_base_y = -.60 if y < 0 else .60
        assert math.hypot(x-.65, y-own_base_y) >= .40
        # Reserve footprint must not intersect the board's rectangular collider.
        local_x, local_y = geometry.world_to_local_xy(x, y)
        half_x, half_y = (side/2 for side in geometry.board_size_xy_m)
        dx, dy = max(abs(local_x)-half_x, 0), max(abs(local_y)-half_y, 0)
        assert math.hypot(dx, dy) > geometry.piece_radius_m


@pytest.mark.parametrize('game_id', GAME_IDS)
def test_rotated_geometry_explicit_inventory_and_bounds(game_id):
    geometry = replace(geometry_for(game_id), board_yaw_rad=.73, center_xy_m=(1.2, -.8))
    q, r = geometry.size-1, 0
    center = geometry.cell_center(q, r)
    assert geometry.footprint_clearance_m(q, r, center[:2], .028) == pytest.approx(geometry.pitch_m/2-.028)
    positions = tuple((float(i), 2., 3.) for i in range(geometry.size**2))
    custom = replace(geometry, reserve_positions_m=positions)
    assert custom.reserve_position('white', 0) == positions[custom.inventory_counts['black']]
    for q, r in ((-1, 0), (geometry.size, 0), (True, 0), (0., 0)):
        with pytest.raises(ValueError):
            geometry.cell_center(q, r)
    for color in COLORS:
        with pytest.raises(ValueError):
            geometry.reserve_position(color, geometry.inventory_counts[color])
    with pytest.raises(ValueError):
        replace(geometry, reserve_positions_m=positions[:-1])
    with pytest.raises(ValueError):
        BoardGameGeometry(game_id=game_id, size=9)


@pytest.mark.parametrize('game_id', GAME_IDS[:2])
@pytest.mark.parametrize('direction', [(1, 0), (0, 1), (1, 1), (1, -1)])
def test_square_all_winning_directions_and_terminal(game_id, direction):
    geometry = geometry_for(game_id)
    rules = BoardGameRules(geometry)
    dq, dr = direction
    start = (0, geometry.size-1 if dr < 0 else 0)
    winning = [(start[0]+i*dq, start[1]+i*dr) for i in range(geometry.win_length)]
    # Select fillers outside the winning line; stop before they can reach k.
    filler = [cell for cell in rules.legal_moves() if cell not in winning]
    for i, cell in enumerate(winning):
        rules.play('player_0', *cell)
        if i+1 < len(winning):
            rules.play('player_1', *filler[i])
    assert rules.winner == 'player_0'
    assert len(rules.winning_path) == geometry.win_length
    assert rules.current_player is None
    assert not rules.draw and rules.terminal
    assert rules.legal_moves() == ()
    with pytest.raises(IllegalMove, match='already won or drawn'):
        rules.play('player_1', *filler[-1])


def test_tic_tac_toe_draw_is_terminal_and_has_no_winner():
    rules = BoardGameRules()
    moves = ((0, 0), (1, 0), (2, 0), (1, 1), (0, 1), (2, 1), (1, 2), (0, 2), (2, 2))
    for i, cell in enumerate(moves):
        rules.play('player_'+str(i % 2), *cell)
    assert rules.draw and rules.terminal and rules.winner is None
    assert rules.current_player is None and not rules.legal_moves()
    with pytest.raises(IllegalMove):
        rules.play('player_1', 0, 0)


def test_planar_connect_four_has_no_gravity_and_three_do_not_win():
    rules = BoardGameRules(geometry_for('connect_four_5x5'))
    for q in range(3):
        rules.play('player_0', q, 4)
        rules.play('player_1', q, 2)
    assert rules.winner is None
    rules.play('player_0', 3, 4)
    assert rules.winner == 'player_0'
    assert rules.specification()['gravity'] is False
    assert rules.specification()['variant'] == 'custom_5x5_planar_no_gravity'


def test_planar_full_board_draw_with_minimal_inventory():
    rules = BoardGameRules(geometry_for('connect_four_5x5'))
    pattern = ('BBBWB', 'BWWBW', 'WWWBW', 'BBBWW', 'BBWWB')
    black = [(q, r) for r, row in enumerate(pattern) for q, c in enumerate(row) if c == 'B']
    white = [(q, r) for r, row in enumerate(pattern) for q, c in enumerate(row) if c == 'W']
    assert len(black) == 13 and len(white) == 12
    for i, cell in enumerate(black):
        rules.play('player_0', *cell)
        if i < len(white):
            rules.play('player_1', *white[i])
    assert rules.draw and rules.terminal and rules.winner is None
    assert rules.turn_index == 25 and rules.current_player is None


@pytest.mark.parametrize('color', COLORS)
def test_hex_opposite_edges_and_no_square_diagonal_adjacency(color):
    geometry = geometry_for('hex7x7')
    rules = BoardGameRules(geometry)
    for i in range(7):
        if color == 'black':
            rules.play('player_0', 0, i)
            if i < 6:
                rules.play('player_1', 6, i)
        else:
            rules.play('player_0', i, 6)
            rules.play('player_1', i, 0)
    assert rules.winner == ('player_0' if color == 'black' else 'player_1')
    assert not rules.draw
    assert (2, 2) not in geometry.neighbors(1, 1)
    assert (2, 0) in geometry.neighbors(1, 1)


@pytest.mark.parametrize('game_id', GAME_IDS)
def test_ownership_occupied_cells_and_readonly_board(game_id):
    rules = BoardGameRules(geometry_for(game_id))
    with pytest.raises(IllegalMove, match='current player'):
        rules.play('player_1', 0, 0)
    rules.play('player_0', 0, 0)
    with pytest.raises(IllegalMove, match='occupied'):
        rules.play('player_1', 0, 0)
    with pytest.raises(TypeError):
        rules.board[1, 1] = 'white'


@pytest.mark.parametrize('game_id', GAME_IDS)
def test_full_observed_dwell_required_and_intent_never_commits(game_id):
    match = BoardGameMatch(geometry_for(game_id))
    match.begin_move('player_0', 'black_00', 1, 1)
    assert not match.rules.board and match.rules.current_player == 'player_0'
    for index in range(30):
        assert not match.observe(frame(match, index)).committed
    result = match.observe(frame(match, 30))
    assert result.committed and result.dwell_s == pytest.approx(.300)
    assert match.rules.board[1, 1] == 'black' and match.rules.current_player == 'player_1'


@pytest.mark.parametrize('game_id', GAME_IDS)
@pytest.mark.parametrize('changes,reason', [
    ({'board_support_impulse_ns': 0.}, 'native_board_support_required'),
    ({'all_robot_contact_impulse_ns': .001}, 'robot_must_release_piece'),
    ({'maximum_penetration_m': None}, 'nonfinite_or_invalid_piece_evidence'),
    ({'maximum_penetration_m': .002}, 'excessive_penetration'),
    ({'linear_velocity_m_s': (.1, 0., 0.)}, 'piece_still_translating'),
    ({'quaternion_wxyz': (0., 1., 0., 0.)}, 'piece_not_upright')])
def test_failed_physical_evidence_never_commits(game_id, changes, reason):
    match = BoardGameMatch(geometry_for(game_id))
    match.begin_move('player_0', 'black_00', 1, 1)
    for index in range(40):
        result = match.observe(frame(match, index, **changes))
        assert result.reason == reason and not result.committed
    assert not match.rules.board


@pytest.mark.parametrize('game_id', GAME_IDS)
def test_complete_inventory_and_substep_continuity(game_id):
    match = BoardGameMatch(geometry_for(game_id))
    match.begin_move('player_0', 'black_00', 1, 1)
    for index in range(20):
        match.observe(frame(match, index))
    assert match.observe(frame(match, 19)).reason == 'duplicate_or_out_of_order_frame'
    assert match.observe(frame(match, 21)).reason == 'missing_native_substeps'
    reduced = dict(frame(match, 22).pieces)
    del reduced['white_00']
    assert match.observe(replace(frame(match, 22), pieces=reduced)).reason == 'incomplete_or_invalid_native_frame'
    assert match.observe(frame(match, 23)).reason == 'missing_native_substeps'
    for index in range(24, 54):
        assert not match.observe(frame(match, index)).committed
    assert match.observe(frame(match, 54)).committed


@pytest.mark.parametrize('game_id', GAME_IDS)
def test_wrong_inventory_reuse_and_committed_displacement_requires_recovery(game_id):
    match = BoardGameMatch(geometry_for(game_id))
    with pytest.raises(IllegalMove, match='belong'):
        match.begin_move('player_0', 'white_00', 0, 0)
    match.begin_move('player_0', 'black_00', 0, 0)
    original = match.pending
    assert match.begin_move('player_0', 'black_00', 0, 0) is original
    next_index = settle(match, 0)
    match.begin_move('player_1', 'white_00', 1, 1)
    next_index = settle(match, next_index)
    with pytest.raises(IllegalMove, match='reused'):
        match.begin_move('player_0', 'black_00', 2, 2)
    sample = frame(match, next_index)
    sample.pieces['black_00'] = piece(match.geometry.cell_center(1, 0))
    result = match.observe(sample)
    assert result.recovery_required and not result.invalidated
    assert not match.terminal and match.rules.board[0, 0] == 'black'
    match.begin_move('player_0', 'black_01', 2, 2)


def test_physical_draw_stops_match_actions():
    match = BoardGameMatch()
    moves = ((0, 0), (1, 0), (2, 0), (1, 1), (0, 1), (2, 1), (1, 2), (0, 2), (2, 2))
    next_index = 0
    for i, cell in enumerate(moves):
        match.begin_move('player_'+str(i % 2), f'{COLORS[i % 2]}_{i//2:02d}', *cell)
        next_index = settle(match, next_index)
    assert match.rules.draw and match.terminal and len(match.commit_log) == 9
    assert match.observe(frame(match, next_index)).reason == 'match_terminal'


def protocol(match, mode=RGBD_MODE):
    return BoardGamePlayerProtocol(match, (
        ActuatorOwnership('player_0', 'left', ('left_joint',)),
        ActuatorOwnership('player_1', 'right', ('right_joint',))), observation_mode=mode)


@pytest.mark.parametrize('game_id', GAME_IDS)
def test_equal_rgbd_privacy_and_owned_joint_admission(game_id):
    match = BoardGameMatch(geometry_for(game_id))
    public = {'snapshot_id': 's1', 'simulation_time_s': 0., 'rgb': 'rgb.png', 'depth': 'depth.npy',
              'calibration': {'fx': 500}, 'private_object_poses': 'SECRET', 'opponent_reasoning': 'SECRET'}
    bridge = protocol(match)
    black, white = [bridge.observation(player, public) for player in match.rules.players]
    assert black['snapshot'] == white['snapshot']
    assert 'SECRET' not in json.dumps(black)
    assert 'board' not in black and 'legal_cells' not in black
    black['snapshot']['calibration']['fx'] = 1
    assert public['calibration']['fx'] == 500
    smoke = protocol(match, SMOKE_MODE).observation('player_0', public)
    assert len(smoke['legal_cells']) == match.geometry.size**2
    assert 'not_model_score' in smoke['privilege_label']
    action = {'type': 'joint_targets', 'robot_id': 'left', 'joint_positions': {'left_joint': .1}, 'duration_s': .1}
    assert bridge.admit_action('player_0', action)['physical_step']
    with pytest.raises(IllegalMove):
        bridge.admit_action('player_1', action)
    with pytest.raises(ValueError, match='Another player'):
        bridge.admit_action('player_0', {**action, 'robot_id': 'right'})
    with pytest.raises(ValueError, match='owned'):
        bridge.admit_action('player_0', {**action, 'joint_positions': {'right_joint': .1}})
    with pytest.raises(ValueError, match='Only'):
        bridge.admit_action('player_0', {**action, 'object_pose': [0, 0, 1]})
    response = bridge.admit_action('player_0', {'type': 'placement_intent', 'piece_id': 'black_00', 'q': 1, 'r': 1})
    assert response['physical_step'] is False and not match.rules.board


def test_overlapping_actuators_are_rejected():
    with pytest.raises(ValueError, match='overlaps'):
        BoardGamePlayerProtocol(BoardGameMatch(), (
            ActuatorOwnership('player_0', 'left', ('same_joint',)),
            ActuatorOwnership('player_1', 'right', ('same_joint',))))


@pytest.mark.parametrize('game_id', GAME_IDS)
def test_source_hashed_task_is_explicitly_unqualified(game_id):
    specification = task_specification(geometry_for(game_id))
    assert specification['schema'] == 'robot_board_game_task.v1'
    assert specification['task_id'] == game_id
    assert specification['physics_accepted'] is False
    assert specification['task_policy_success_claimed'] is False
    assert set(specification['sources_sha256']) == {'__init__.py', 'geometry.py', 'rules.py',
        'settlement.py', 'native_observer.py', 'protocol.py', 'release.py'}
    assert all(len(digest) == 64 for digest in specification['sources_sha256'].values())
    json.dumps(specification, allow_nan=False)
