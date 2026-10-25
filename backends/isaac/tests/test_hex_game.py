"""Rules and private native-evidence admission; no simulation acceptance implied."""
from dataclasses import replace
import json
import math

import pytest

from scenes.board_games.hex import (ActuatorOwnership, HexGeometry, HexMatch, HexPlayerProtocol,
                                    HexRules, IllegalMove, NativeFrame, PieceEvidence,
                                    RGBD_MODE, SMOKE_MODE, neighbors, task_specification)


def piece(position, **changes):
    return replace(PieceEvidence(tuple(position), (1., 0., 0., 0.), (0., 0., 0.),
                                 (0., 0., 0.), .001, 0., 0.), **changes)


def frame(match, index, *, target=True, changes=None, contacts_complete=True):
    pieces = {piece_id: piece(match.geometry.reserve_position(piece_id.split('_')[0],
                                                             int(piece_id.split('_')[1])),
                              board_support_impulse_ns=0.) for piece_id in match.inventory}
    for piece_id, cell in match.committed.items():
        pieces[piece_id] = piece(match.geometry.cell_center(*cell))
    if match.pending and target:
        intent = match.pending
        pieces[intent.piece_id] = piece(match.geometry.cell_center(intent.q, intent.r), **(changes or {}))
    return NativeFrame(index, index*.01, .01, pieces, contacts_complete)


def settle(match, start):
    result = None
    for index in range(start, start+31):
        result = match.observe(frame(match, index))
    assert result.committed, result
    return start+31


def protocol(match=None, mode=RGBD_MODE):
    match = match or HexMatch()
    owners = (ActuatorOwnership('player_0', 'left', ('fr3_left_joint1', 'left_thumb')),
              ActuatorOwnership('player_1', 'right', ('fr3_right_joint1', 'right_thumb')))
    return HexPlayerProtocol(match, owners, observation_mode=mode)


def test_axial_neighbors_and_metric_geometry_agree():
    geometry = HexGeometry()
    assert set(neighbors(3, 3)) == {(4, 3), (2, 3), (3, 4), (3, 2), (4, 2), (2, 4)}
    assert set(neighbors(0, 0)) == {(1, 0), (0, 1)}
    assert len(neighbors(0, 6)) == 3
    for q in range(7):
        for r in range(7):
            point = geometry.cell_center(q, r)
            for adjacent in neighbors(q, r):
                assert (q, r) in neighbors(*adjacent)
                assert math.dist(point, geometry.cell_center(*adjacent)) == pytest.approx(geometry.pitch_m)
    assert geometry.cell_label(0, 0) == 'A1'
    assert geometry.cell_label(6, 6) == 'G7'
    assert geometry.board_size_xy_m[0] == pytest.approx(.79)
    assert geometry.footprint_clearance_m(3, 3, geometry.center_xy_m, .028) == pytest.approx(.0095)


def test_rigid_board_transform_and_explicit_reserves():
    geometry = HexGeometry(board_yaw_rad=.73, center_xy_m=(1.2, -.8))
    for q in range(7):
        for r in range(7):
            point = geometry.cell_center(q, r)
            assert geometry.footprint_clearance_m(q, r, point[:2], .028) == pytest.approx(.0095)
    reserves = tuple((float(i), 2., 3.) for i in range(50))
    custom = replace(geometry, reserve_positions_m=reserves)
    assert custom.reserve_position('red', 24) == reserves[24]
    assert custom.reserve_position('blue', 0) == reserves[25]


def test_reserve_pattern_and_handle_dimensions():
    geometry = HexGeometry()
    assert geometry.handle_radius_m == .020
    points = [geometry.reserve_position('blue', i) for i in range(25)]
    assert points[0] == pytest.approx((.65-.435, .300, .787))
    assert points[7] == pytest.approx((.65+.435, .300, .787))
    assert points[8][1] == pytest.approx(.3675)
    assert points[23] == pytest.approx((.65+.435, .435, .787))
    assert points[24] == pytest.approx((.65, .300, .787))
    assert len(set(points)) == 25
    assert min(math.dist(a, b) for i, a in enumerate(points) for b in points[i+1:]) > .056


@pytest.mark.parametrize('q,r', [(-1, 0), (7, 0), (0, 7), (0., 0), (True, 0)])
def test_invalid_coordinates_rejected(q, r):
    with pytest.raises(IllegalMove):
        HexRules().play('player_0', q, r)


def test_turns_occupied_cells_and_board_is_readonly():
    rules = HexRules()
    with pytest.raises(IllegalMove, match='current player'):
        rules.play('player_1', 0, 0)
    rules.play('player_0', 0, 0)
    assert rules.current_player == 'player_1'
    assert len(rules.legal_moves()) == 48
    with pytest.raises(IllegalMove, match='occupied'):
        rules.play('player_1', 0, 0)
    with pytest.raises(TypeError):
        rules.board[2, 2] = 'blue'


def test_red_win_and_no_action_after_win():
    rules = HexRules()
    for r in range(7):
        rules.play('player_0', 0, r)
        if r < 6:
            rules.play('player_1', 6, r)
    assert rules.winner == 'player_0'
    assert rules.winning_path == tuple((0, r) for r in range(7))
    assert rules.current_player is None
    assert rules.legal_moves() == ()
    with pytest.raises(IllegalMove, match='already won'):
        rules.play('player_1', 5, 6)


def test_blue_connects_its_q_edges_and_diagonal_is_not_square_adjacency():
    rules = HexRules()
    for q in range(7):
        rules.play('player_0', q, 6)
        rules.play('player_1', q, 0)
    assert rules.winner == 'player_1'
    assert rules.winning_path == tuple((q, 0) for q in range(7))
    diagonal = HexRules()
    for i in range(7):
        diagonal.play('player_0', i, i)
        if i < 6:
            diagonal.play('player_1', i, 6)
    assert diagonal.winner is None


def test_slanted_axial_chain_wins():
    rules = HexRules()
    for r in range(7):
        rules.play('player_0', 6-r, r)
        if r < 6:
            rules.play('player_1', r+1, 6)
    assert rules.winner == 'player_0'


def test_reservation_is_not_placement_and_requires_full_observed_dwell():
    match = HexMatch()
    match.begin_move('player_0', 'red_00', 3, 3)
    assert len(match.rules.board) == 0
    assert match.rules.current_player == 'player_0'
    for index in range(30):
        result = match.observe(frame(match, index))
        assert not result.committed
    assert len(match.rules.board) == 0
    result = match.observe(frame(match, 30))
    assert result.committed
    assert result.dwell_s == pytest.approx(.3)
    assert match.rules.board[3, 3] == 'red'
    assert match.pending is None
    assert match.rules.current_player == 'player_1'


@pytest.mark.parametrize('changes,reason', [
    ({'position_m': (.65+.025, 0., .799)}, 'piece_footprint_outside_cell'),
    ({'position_m': (.65, 0., 1.)}, 'piece_not_at_board_height'),
    ({'position_m': (float('nan'), 0., .799)}, 'nonfinite_or_invalid_piece_evidence'),
    ({'quaternion_wxyz': (0., 1., 0., 0.)}, 'piece_not_upright'),
    ({'quaternion_wxyz': (0., 0., 0., 0.)}, 'nonfinite_or_invalid_piece_evidence'),
    ({'linear_velocity_m_s': (.1, 0., 0.)}, 'piece_still_translating'),
    ({'angular_velocity_rad_s': (0., 0., 1.)}, 'piece_still_rotating'),
    ({'board_support_impulse_ns': 0.}, 'native_board_support_required'),
    ({'all_robot_contact_impulse_ns': .001}, 'robot_must_release_piece'),
    ({'maximum_penetration_m': .002}, 'excessive_penetration'),
    ({'maximum_penetration_m': None}, 'nonfinite_or_invalid_piece_evidence'),
])
def test_physical_rejection_does_not_commit(changes, reason):
    match = HexMatch()
    match.begin_move('player_0', 'red_00', 3, 3)
    for index in range(50):
        result = match.observe(frame(match, index, changes=changes))
        assert result.reason == reason
        assert not result.committed
    assert len(match.rules.board) == 0
    assert match.rules.current_player == 'player_0'


def test_interruption_gaps_duplicate_and_incomplete_contacts_reset_dwell():
    match = HexMatch()
    match.begin_move('player_0', 'red_00', 3, 3)
    for index in range(20):
        match.observe(frame(match, index))
    assert match.observe(frame(match, 19)).reason == 'duplicate_or_out_of_order_frame'
    assert match.observe(frame(match, 21)).reason == 'missing_native_substeps'
    assert match.observe(frame(match, 22, contacts_complete=False)).reason == 'incomplete_or_invalid_native_frame'
    assert match.observe(frame(match, 23)).reason == 'missing_native_substeps'
    for index in range(24, 54):
        assert not match.observe(frame(match, index)).committed
    assert match.observe(frame(match, 54)).committed


def test_partial_inventory_rejected():
    match = HexMatch()
    match.begin_move('player_0', 'red_00', 3, 3)
    sample = frame(match, 0)
    reduced = dict(sample.pieces)
    del reduced['blue_24']
    assert match.observe(replace(sample, pieces=reduced)).reason == 'incomplete_or_invalid_native_frame'


def test_piece_ownership_reuse_and_multiple_pending_intents():
    match = HexMatch()
    with pytest.raises(IllegalMove, match='belong'):
        match.begin_move('player_0', 'blue_00', 1, 1)
    match.begin_move('player_0', 'red_00', 1, 1)
    with pytest.raises(IllegalMove, match='already pending'):
        match.begin_move('player_0', 'red_01', 2, 2)
    next_index = settle(match, 0)
    match.begin_move('player_1', 'blue_00', 2, 2)
    settle(match, next_index)
    with pytest.raises(IllegalMove, match='reused'):
        match.begin_move('player_0', 'red_00', 3, 3)


def test_committed_piece_displacement_is_irreversible_and_blocks_actions():
    match = HexMatch()
    match.begin_move('player_0', 'red_00', 0, 0)
    next_index = settle(match, 0)
    match.begin_move('player_1', 'blue_00', 2, 2)
    sample = frame(match, next_index)
    sample.pieces['red_00'] = piece(match.geometry.cell_center(1, 0))
    result = match.observe(sample)
    assert result.invalidated
    assert match.terminal
    assert not result.committed
    with pytest.raises(IllegalMove, match='Terminal'):
        protocol(match).admit_action('player_1', {'type': 'joint_targets', 'robot_id': 'right',
                                                 'joint_positions': {'right_thumb': .1}, 'duration_s': .1})


def test_unrequested_board_piece_invalidates_match():
    match = HexMatch()
    match.begin_move('player_0', 'red_00', 0, 0)
    sample = frame(match, 0)
    sample.pieces['blue_01'] = piece(match.geometry.cell_center(5, 5))
    assert match.observe(sample).invalidated


def test_native_settlement_enforces_terminal_win():
    match = HexMatch()
    index = 0
    for r in range(7):
        match.begin_move('player_0', f'red_{r:02d}', 0, r)
        index = settle(match, index)
        if r < 6:
            match.begin_move('player_1', f'blue_{r:02d}', 6, r)
            index = settle(match, index)
    assert match.rules.winner == 'player_0'
    with pytest.raises(IllegalMove, match='Terminal'):
        match.begin_move('player_1', 'blue_07', 4, 4)
    assert match.observe(frame(match, index)).reason == 'match_terminal'


def test_ownership_and_no_object_state_action_routes():
    player_protocol = protocol()
    valid = {'type': 'joint_targets', 'robot_id': 'left',
             'joint_positions': {'fr3_left_joint1': .2}, 'duration_s': .1}
    assert player_protocol.admit_action('player_0', valid)['physical_step']
    with pytest.raises(IllegalMove):
        player_protocol.admit_action('player_1', valid)
    with pytest.raises(ValueError, match='Another player'):
        player_protocol.admit_action('player_0', {**valid, 'robot_id': 'right'})
    with pytest.raises(ValueError, match='owned'):
        player_protocol.admit_action('player_0', {**valid, 'joint_positions': {'right_thumb': .3}})
    with pytest.raises(ValueError, match='Only'):
        player_protocol.admit_action('player_0', {**valid, 'object_pose': [0, 0, 0]})
    with pytest.raises(ValueError, match='Only'):
        player_protocol.admit_action('player_0', {'type': 'set_object_pose'})
    result = player_protocol.admit_action('player_0', {'type': 'placement_intent', 'piece_id': 'red_00', 'q': 0, 'r': 0})
    assert result['physical_step'] is False
    assert not player_protocol.match.rules.board


def test_equal_rgbd_projection_and_explicit_board_privilege_label():
    published = {'snapshot_id': 's1', 'simulation_time_s': 0., 'rgb': 'public/rgb.png',
                 'depth': 'public/depth.npy', 'calibration': {'fx': 500},
                 'private_object_poses': {'secret': 1}, 'opponent_reasoning': 'secret'}
    player_protocol = protocol()
    red = player_protocol.observation('player_0', published)
    blue = player_protocol.observation('player_1', published)
    for field in ('snapshot', 'rules', 'geometry', 'turn_index', 'current_player'):
        assert red[field] == blue[field]
    assert 'board' not in red and 'legal_cells' not in red
    assert 'secret' not in json.dumps(red)
    red['snapshot']['calibration']['fx'] = 1
    assert published['calibration']['fx'] == 500
    public_smoke = protocol(mode=SMOKE_MODE).observation('player_0', published)
    assert public_smoke['observation_mode'] == SMOKE_MODE
    assert 'not_model_score' in public_smoke['privilege_label']
    assert len(public_smoke['legal_cells']) == 49


def test_source_hashed_specification_explicitly_unqualified():
    specification = task_specification()
    assert specification['rules']['swap_rule'] == 'disabled_in_v1_fixed_color_ownership'
    assert specification['physics_accepted'] is False
    assert 'settlement.py' in specification['sources_sha256']
    assert all(len(digest) == 64 for digest in specification['sources_sha256'].values())
    json.dumps(specification, allow_nan=False)
