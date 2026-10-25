"""Recoverable attempts and the public frozen rule contract; CPU doubles only."""
from copy import deepcopy
from dataclasses import asdict, replace
import json

import pytest

from scenes.board_games.tabletop import (BoardGameMatch, GAME_IDS, IllegalMove, SettlementThresholds,
                                       geometry_for, task_specification)
from scenes.board_games.tabletop.rules import contract_sha256
from scenes.board_games.tabletop.release import verify_public_rule_contract, sha256
from test_tabletop_games import frame, piece, settle, protocol


@pytest.mark.parametrize('game_id', GAME_IDS)
def test_duplicate_intent_is_idempotent_but_revision_restarts_dwell(game_id):
    match = BoardGameMatch(geometry_for(game_id))
    intent = match.begin_move('player_0', 'black_00', 0, 0)
    for index in range(20):
        assert not match.observe(frame(match, index)).committed
    assert match.begin_move('player_0', 'black_00', 0, 0) is intent
    for index in range(20,30):
        assert not match.observe(frame(match,index)).committed
    assert match.observe(frame(match,30)).committed
    match.begin_move('player_1','white_00',1,0)
    for index in range(31,51):
        assert not match.observe(frame(match,index)).committed
    revised = match.begin_move('player_1','white_01',1,1)
    assert revised.turn_index == 1 and revised.attempt_index > intent.attempt_index
    for index in range(51,81):
        assert not match.observe(frame(match,index)).committed
    assert match.observe(frame(match,81)).committed
    assert match.committed == {'black_00':(0,0),'white_01':(1,1)}


def test_cancel_and_same_intent_before_next_frame_cannot_reuse_old_dwell():
    match = BoardGameMatch()
    original = match.begin_move('player_0','black_00',1,1)
    for index in range(30):
        match.observe(frame(match,index))
    player = protocol(match)
    result = player.admit_action('player_0',{'type':'cancel_placement'})
    assert result['physical_step'] is False and match.pending is None
    assert match.rules.turn_index == 0 and not match.rules.board
    again = match.begin_move('player_0','black_00',1,1)
    assert again.attempt_index > original.attempt_index
    for index in range(30,60):
        assert not match.observe(frame(match,index)).committed
    assert match.observe(frame(match,60)).committed


@pytest.mark.parametrize('game_id', GAME_IDS)
@pytest.mark.parametrize('disturbance', ['other_cell','held','unsupported','moving'])
def test_committed_piece_disturbance_recovers_without_forfeit(game_id,disturbance):
    match = BoardGameMatch(geometry_for(game_id))
    match.begin_move('player_0','black_00',0,0)
    start = settle(match,0)
    match.begin_move('player_1','white_00',1,1)
    for index in range(start,start+25):
        assert not match.observe(frame(match,index)).committed
    sample = frame(match,start+25)
    change = {'other_cell':{'position_m':match.geometry.cell_center(1,0)},
              'held':{'all_robot_contact_impulse_ns':.001},
              'unsupported':{'board_support_impulse_ns':0.},
              'moving':{'linear_velocity_m_s':(.1,0.,0.)}}[disturbance]
    sample.pieces['black_00'] = replace(sample.pieces['black_00'],**change)
    result = match.observe(sample)
    assert result.recovery_required and not result.committed and not result.invalidated
    assert match.invalid_reason is None and not match.terminal and match.rules.current_player == 'player_1'
    assert match.committed == {'black_00':(0,0)}
    public = match.public_status()
    assert public['recovery_required'] and public['retryable']
    assert public['reason'] == 'committed_piece_needs_recovery'
    assert 'black_00' not in json.dumps(public)
    for index in range(start+26,start+56):
        assert not match.observe(frame(match,index)).committed
    assert match.observe(frame(match,start+56)).committed
    assert match.committed == {'black_00':(0,0),'white_00':(1,1)}


@pytest.mark.parametrize('game_id', GAME_IDS)
def test_small_shift_inside_original_cell_does_not_require_original_centre(game_id):
    match = BoardGameMatch(geometry_for(game_id))
    match.begin_move('player_0','black_00',0,0)
    start = settle(match,0)
    match.begin_move('player_1','white_00',1,1)
    nominal = match.geometry.cell_center(0,0)
    slack = match.geometry.pitch_m/2-match.geometry.piece_radius_m-match.verifier.thresholds.edge_clearance_m
    shifted = (nominal[0]+slack/2,nominal[1],nominal[2])
    for index in range(start,start+31):
        sample = frame(match,index)
        sample.pieces['black_00'] = piece(shifted)
        result = match.observe(sample)
        assert not result.recovery_required
    assert result.committed and match.committed['black_00'] == (0,0)


def test_extra_piece_on_board_blocks_commit_until_removed_without_losing_turn():
    match = BoardGameMatch()
    match.begin_move('player_0','black_00',0,0)
    for index in range(40):
        sample = frame(match,index)
        sample.pieces['white_00'] = piece(match.geometry.cell_center(2,2))
        result = match.observe(sample)
        assert result.recovery_required and not result.invalidated and not result.committed
    assert match.rules.turn_index == 0 and not match.terminal
    settle(match,40)
    assert match.rules.turn_index == 1


def test_unused_reserves_may_move_on_table_without_exact_reset_positions():
    match = BoardGameMatch()
    match.begin_move('player_0','black_00',0,0)
    for index in range(31):
        sample = frame(match,index)
        sample.pieces['white_00'] = piece((.2,.35,.787), board_support_impulse_ns=0.,
            linear_velocity_m_s=(.3,.1,0.), angular_velocity_rad_s=(0.,1.,0.),
            quaternion_wxyz=(0.,1.,0.,0.))
        result = match.observe(sample)
    assert result.committed


def test_retargeted_piece_left_on_board_must_be_physically_cleared():
    match = BoardGameMatch()
    match.begin_move('player_0','black_00',0,0)
    match.observe(frame(match,0))
    match.begin_move('player_0','black_01',1,1)
    sample = frame(match,1)
    sample.pieces['black_00'] = piece(match.geometry.cell_center(0,0))
    assert match.observe(sample).recovery_required
    assert not match.rules.board
    settle(match,2)
    assert match.committed == {'black_01':(1,1)}


@pytest.mark.parametrize('action', [
    {'type':'placement_intent','piece_id':'white_00','q':1,'r':1},
    {'type':'placement_intent','piece_id':'black_00','q':99,'r':1},
    {'type':'cancel_placement','object_pose':[0,0,0]},
    {'type':'set_object_pose'},
    {'type':'joint_targets','robot_id':'right','joint_positions':{'right_joint':0.},'duration_s':.1},
    None])
def test_rejected_action_preserves_pending_and_dwell_without_a_loss(action):
    match = BoardGameMatch()
    intent = match.begin_move('player_0','black_00',1,1)
    for index in range(30):
        match.observe(frame(match,index))
    with pytest.raises((ValueError,IllegalMove)):
        protocol(match).admit_action('player_0',action)
    assert match.pending is intent and not match.rules.board and not match.terminal
    assert match.observe(frame(match,30)).committed


@pytest.mark.parametrize('reason,outcome',[('time_limit','truncated'),('step_limit','truncated'),
                                           ('runtime_error','aborted'),('cancelled','aborted')])
def test_budget_or_api_fault_is_not_a_game_loss(reason,outcome):
    match = BoardGameMatch()
    match.begin_move('player_0','black_00',0,0)
    match.end_episode(reason)
    assert match.terminal and match.outcome == outcome
    assert match.rules.winner is None and match.rules.draw is False
    assert match.invalid_reason is None and match.public_status()['retryable'] is False
    with pytest.raises(IllegalMove):
        match.begin_move('player_1','white_00',1,1)
    assert match.observe(frame(match,0)).reason == 'match_terminal'


def test_native_evidence_fault_pauses_without_winner_and_requires_fresh_dwell():
    match = BoardGameMatch()
    match.begin_move('player_0','black_00',0,0)
    for index in range(30):
        match.observe(frame(match,index))
    result = match.observe(replace(frame(match,30),contacts_complete=False))
    assert result.evidence_unavailable and not match.terminal
    assert match.public_status()['evidence_unavailable']
    assert match.observe(frame(match,31)).reason == 'missing_native_substeps'
    settle(match,32)


@pytest.mark.parametrize('game_id',GAME_IDS)
def test_public_protocol_contract_equals_frozen_task_and_actual_thresholds(game_id):
    thresholds = replace(SettlementThresholds(),dwell_s=.4)
    match = BoardGameMatch(geometry_for(game_id),thresholds=thresholds)
    task = task_specification(match.geometry,thresholds)
    snapshot = {'snapshot_id':'public1','simulation_time_s':0.,'rgb':'rgb.png','depth':'depth.npy','calibration':{}}
    observation = protocol(match).observation('player_0',snapshot)
    assert observation['public_rules'] == task['public_rules'] == match.rule_contract
    assert task['public_rules']['physical_commit']['thresholds'] == asdict(thresholds)
    assert observation['public_rules_sha256'] == task['public_rules_sha256'] == contract_sha256(match.rule_contract)
    assert verify_public_rule_contract(task)['verified_against_evaluator']
    assert task['public_rules']['recovery']['automatic_forfeit'] is False
    assert match.verifier.thresholds.edge_clearance_m == .001


@pytest.mark.parametrize('mutation',[
    lambda task:task['public_rules']['recovery'].update(automatic_forfeit=True),
    lambda task:task['public_rules']['physical_commit']['thresholds'].update(dwell_s=.01),
    lambda task:task.update(public_rules_sha256='0'*64),
    lambda task:task['rules'].update(turn_order='white_first'),
    lambda task:task['settlement'].update(edge_clearance_m=.005),
    lambda task:task.update(public_rules_file='../rules.json'),
])
def test_drifted_public_contract_cannot_validate(mutation):
    task = task_specification()
    mutation(task)
    with pytest.raises(ValueError):
        verify_public_rule_contract(task)


def test_frozen_rule_file_and_manifest_hash_are_required(tmp_path):
    task = task_specification()
    path = tmp_path/'rules.json'
    path.write_text(json.dumps(task['public_rules']))
    manifest = {'public_rules':'rules.json','files_sha256':{'rules.json':sha256(path)}}
    (tmp_path/'release_manifest.json').write_text(json.dumps(manifest))
    assert verify_public_rule_contract(task,release=tmp_path)['frozen_rules_file_verified']
    path.write_text('{}')
    with pytest.raises(ValueError):
        verify_public_rule_contract(task,release=tmp_path)
    manifest['files_sha256']['rules.json'] = sha256(path)
    (tmp_path/'release_manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        verify_public_rule_contract(task,release=tmp_path)


@pytest.mark.parametrize('missing',['public_rules','public_rules_sha256','public_rules_file','rules','geometry','settlement'])
def test_missing_public_rule_fields_are_normalized_value_errors(missing):
    task = task_specification()
    del task[missing]
    with pytest.raises(ValueError):
        verify_public_rule_contract(task)


def test_missing_frozen_public_rule_files_are_normalized_value_errors(tmp_path):
    with pytest.raises(ValueError):
        verify_public_rule_contract(task_specification(),release=tmp_path)
