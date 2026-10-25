"""Independent CPU evidence gates for ready poses and small asset fixtures."""
from copy import deepcopy
import hashlib
import json
import math
from types import SimpleNamespace

import numpy as np
import pytest

from scenes.board_games.tabletop import GAME_IDS, geometry_for
from scenes.workrooms.card_room.game_task import verify_game_assets
from scripts.validation.validate_board_game import ready_pose_check


def ready_fixture():
    names = [f'joint_{i:02d}' for i in range(58)]
    joints = np.linspace(-.3, .3, 58)
    # Native body poses use xyz + xyzw; target rotations are ordinary matrices.
    bodies = np.array([[.65, -.4, 1.5, 0., 0., 0., 1.],
                       [.65, .4, 1.5, 0., 0., math.sqrt(.5), math.sqrt(.5)]])
    state = {'joint_q': joints.copy(), 'body_q': bodies.copy()}
    env = SimpleNamespace(joint_names=names, body_names=['left_hand_flange', 'right_hand_flange'],
                          capture_state=lambda: {name: value.copy() for name, value in state.items()})
    initial = {'joint_positions_rad': dict(zip(names, joints.tolist()))}
    scene = {'board_game_task': {'home_design': {
        'left': {'wrist_position_m': bodies[0, :3].tolist(), 'wrist_rotation_matrix': np.eye(3).tolist()},
        'right': {'wrist_position_m': bodies[1, :3].tolist(),
                  'wrist_rotation_matrix': [[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]]}}}}
    return env, scene, initial, state


def test_ready_gate_uses_actual_native_arrays_and_both_wrists():
    env, scene, initial, state = ready_fixture()
    before = deepcopy(state)
    result = ready_pose_check(env, scene, initial)
    assert result['passed']
    assert result['maximum_joint_error_rad'] == 0.
    assert len(result['joint_errors_rad']) == 58
    assert set(result['wrists']) == {'left', 'right'}
    assert all(row['position_error_m'] == 0. for row in result['wrists'].values())
    assert max(row['rotation_error_rad'] for row in result['wrists'].values()) < 1e-7
    assert all(np.array_equal(state[name], old) for name, old in before.items())


@pytest.mark.parametrize('fault', ['joint', 'left_position', 'right_position', 'left_rotation', 'right_rotation'])
def test_native_pose_error_is_not_accepted(fault):
    env, scene, initial, state = ready_fixture()
    if fault == 'joint':
        state['joint_q'][13] += .021
    else:
        side, kind = fault.split('_')
        index = 0 if side == 'left' else 1
        if kind == 'position':
            state['body_q'][index, 0] += .006
        else:
            angle = (0. if side == 'left' else math.pi/2)+.04
            state['body_q'][index, 3:] = [0., 0., math.sin(angle/2), math.cos(angle/2)]
    result = ready_pose_check(env, scene, initial)
    assert result['passed'] is False
    if fault == 'joint':
        assert result['maximum_joint_error_rad'] == pytest.approx(.021)
    else:
        side, kind = fault.split('_')
        assert result['wrists'][side]['passed'] is False
        key = 'position_error_m' if kind == 'position' else 'rotation_error_rad'
        assert result['wrists'][side][key] == pytest.approx(.006 if kind == 'position' else .04)


@pytest.mark.parametrize('fault', [
    'missing_target_joint', 'both_missing_joint', 'duplicate_native_joint', 'short_native_joint_array',
    'long_native_joint_array', 'missing_left_target', 'missing_right_target', 'missing_left_body',
    'missing_right_body', 'nan_target_joint', 'nan_native_joint', 'nan_target_position',
    'nan_native_position', 'nan_target_rotation', 'nan_native_quaternion', 'zero_native_quaternion',
    'scaled_native_quaternion', 'scaled_target_rotation'])
def test_incomplete_or_invalid_ready_evidence_is_rejected(fault):
    env, scene, initial, state = ready_fixture()
    homes = scene['board_game_task']['home_design']
    if fault == 'missing_target_joint':
        del initial['joint_positions_rad'][env.joint_names[-1]]
    elif fault == 'both_missing_joint':
        del initial['joint_positions_rad'][env.joint_names.pop()]
        state['joint_q'] = state['joint_q'][:-1]
    elif fault == 'duplicate_native_joint':
        env.joint_names.append(env.joint_names[-1])
        state['joint_q'] = np.r_[state['joint_q'], state['joint_q'][-1]]
    elif fault == 'short_native_joint_array':
        state['joint_q'] = state['joint_q'][:-1]
    elif fault == 'long_native_joint_array':
        state['joint_q'] = np.r_[state['joint_q'], 0.]
    elif fault.startswith('missing_') and fault.endswith('_target'):
        del homes[fault.split('_')[1]]
    elif fault.startswith('missing_') and fault.endswith('_body'):
        env.body_names.remove(fault.split('_')[1]+'_hand_flange')
    elif fault == 'nan_target_joint':
        initial['joint_positions_rad'][env.joint_names[0]] = float('nan')
    elif fault == 'nan_native_joint':
        state['joint_q'][0] = np.nan
    elif fault == 'nan_target_position':
        homes['left']['wrist_position_m'][0] = float('nan')
    elif fault == 'nan_native_position':
        state['body_q'][0, 0] = np.nan
    elif fault == 'nan_target_rotation':
        homes['right']['wrist_rotation_matrix'][0][0] = float('nan')
    elif fault == 'nan_native_quaternion':
        state['body_q'][0, 3] = np.nan
    elif fault == 'zero_native_quaternion':
        state['body_q'][0, 3:] = 0.
    elif fault == 'scaled_native_quaternion':
        state['body_q'][0, 3:] *= 2.
    else:
        homes['left']['wrist_rotation_matrix'] = (2*np.eye(3)).tolist()
    with pytest.raises(ValueError):
        ready_pose_check(env, scene, initial)


def tiny_asset_library(path, game_id):
    geometry = geometry_for(game_id)
    catalog = {'task': game_id, 'geometry': json.loads(json.dumps(geometry.specification())), 'assets': {}}
    # The loader checks identity/hashes, so tiny original OBJ fixtures suffice.
    mesh = 'v 0 0 0\nv 0.01 0 0\nv 0 0.01 0\nf 1 2 3\n'
    for asset_id in ('board', 'token_black', 'token_white', 'stand_left', 'stand_right', *geometry.reserve_stations()):
        parts = {}
        for kind in ('combined', 'visual', 'collision'):
            name = asset_id+'_'+kind+'.obj'
            (path/name).write_text(mesh)
            parts[kind] = {'mesh_path': name, 'sha256': hashlib.sha256((path/name).read_bytes()).hexdigest()}
        catalog['assets'][asset_id] = {**parts['combined'],
            'visual_parts': [parts['visual']], 'collision_parts': [parts['collision']]}
    (path/'manifest.json').write_text(json.dumps(catalog))
    return catalog


@pytest.mark.parametrize('game_id', GAME_IDS)
def test_exact_small_asset_library_identity_including_reserve_stations(tmp_path, game_id):
    tiny_asset_library(tmp_path, game_id)
    result = verify_game_assets(tmp_path, geometry_for(game_id))
    assert result['game_id'] == game_id
    assert set(result['asset_ids']) == {'board', 'token_black', 'token_white', 'stand_left', 'stand_right', *geometry_for(game_id).reserve_stations()}
    assert result['manifest_sha256'] == hashlib.sha256((tmp_path/'manifest.json').read_bytes()).hexdigest()


@pytest.mark.parametrize('fault', ['wrong_task', 'wrong_geometry', 'missing_asset', 'extra_asset'])
def test_library_metadata_mismatch_rejected(tmp_path, fault):
    catalog = tiny_asset_library(tmp_path, 'tic_tac_toe_3x3')
    if fault == 'wrong_task':
        catalog['task'] = 'connect_four_5x5'
    elif fault == 'wrong_geometry':
        catalog['geometry']['pitch_m'] = .075
    elif fault == 'missing_asset':
        del catalog['assets']['token_white']
    else:
        catalog['assets']['surprise'] = deepcopy(catalog['assets']['board'])
    (tmp_path/'manifest.json').write_text(json.dumps(catalog))
    with pytest.raises(ValueError):
        verify_game_assets(tmp_path, geometry_for('tic_tac_toe_3x3'))


@pytest.mark.parametrize('kind', ['combined', 'visual', 'collision'])
@pytest.mark.parametrize('mutation', ['changed', 'missing'])
def test_each_actual_mesh_hash_and_presence_are_checked(tmp_path, kind, mutation):
    tiny_asset_library(tmp_path, 'connect_four_5x5')
    path = tmp_path/('token_white_'+kind+'.obj')
    if mutation == 'changed':
        path.write_text(path.read_text()+'# changed after manifest\n')
    else:
        path.unlink()
    with pytest.raises((ValueError, FileNotFoundError)):
        verify_game_assets(tmp_path, geometry_for('connect_four_5x5'))


def test_empty_asset_directory_is_rejected(tmp_path):
    with pytest.raises(FileNotFoundError):
        verify_game_assets(tmp_path, geometry_for('hex7x7'))
