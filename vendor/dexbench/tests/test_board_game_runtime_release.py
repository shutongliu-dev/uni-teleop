"""Canonical loading of the current tabletop releases, without native claims."""
from copy import deepcopy
from pathlib import Path
import shutil

import pytest

from bench.isaacsim.release import digest, dump, load_evaluator, task_inputs, validate
from bench.paths import load_json
from bench.runtime.controllers import default_registry
from scenes.workrooms.catalog import task_definition

GAMES = ('tic_tac_toe_3x3', 'connect_four_5x5', 'hex7x7')


def release_path(game='tic_tac_toe_3x3'):
    return Path(task_definition('card_room', game)['source_release']['path'])


@pytest.fixture
def frozen(tmp_path):
    root = tmp_path / 'release'
    shutil.copytree(release_path(), root)
    return root


def repin(root, relative, value):
    dump(root / relative, value)
    manifest = load_json(root / 'release_manifest.json')
    manifest['files_sha256'][relative] = digest(root / relative)
    dump(root / 'release_manifest.json', manifest)


@pytest.mark.parametrize('game,count,version', [('tic_tac_toe_3x3', 9, 'v4'),
                                               ('connect_four_5x5', 25, 'v4'),
                                               ('hex7x7', 49, 'v11')])
def test_current_releases_load_without_rewriting_or_converting_frozen_inputs(game, count, version):
    root = release_path(game)
    assert root.name == version
    before = digest(root / 'release_manifest.json')
    original = load_json(root / 'release_manifest.json')
    manifest = validate(root)
    entry, scene, task, initial = task_inputs(root, game)
    assert digest(root / 'release_manifest.json') == before
    assert {key: manifest[key] for key in original} == original
    assert manifest['task_kind'] == entry['task_kind'] == 'board_game'
    assert manifest['tasks'] == [entry]
    assert 'files' not in manifest and 'physics_profile' not in manifest
    assert manifest['physics_accepted'] is False
    assert manifest['task_policy_success_claimed'] is False
    identity = manifest['runtime_identity']
    assert len(identity['evaluator_sha256']) == len(identity['physics_profile_sha256']) == 64
    assert scene == load_json(root / 'scene.json')
    assert task == load_json(root / 'task.json')
    assert initial == load_json(root / 'initial_state.json')
    assert len(scene['objects']) == count and len(initial['joint_positions_rad']) == 58
    assert task['public_rules']['revision'] == 'recoverable_physical_moves_v2'
    assert entry['title_zh'] and 'steps' not in task


def test_native_substep_evaluator_is_required_and_goal_reset_is_unavailable():
    root = release_path()
    with pytest.raises(ValueError, match='native-substep'):
        load_evaluator(root)
    with pytest.raises(ValueError, match='reset-at-goal'):
        task_inputs(root, 'tic_tac_toe_3x3', 'qualification_probe')
    with pytest.raises(ValueError, match='Unknown board-game task'):
        task_inputs(root, 'hex7x7')


def test_board_game_default_controller_uses_frozen_dual_robot_positions():
    root = release_path()
    registry = default_registry(root)
    descriptor, = registry.descriptors(task_id='tic_tac_toe_3x3')
    policy = registry.resolve(descriptor.policy_id, descriptor.version, task_id='tic_tac_toe_3x3')
    assert descriptor.version == '1.2.0-board1'
    scene = load_json(root / 'scene.json')
    assert policy.robot_configuration['arm_base_poses'] == scene['robot']['arm_base_poses']
    assert set(policy.geometry.names) == set(load_json(root / 'initial_state.json')['joint_positions_rad'])


@pytest.mark.parametrize('damage', ['scene', 'source', 'missing_source', 'extra_source', 'path_escape', 'symlink'])
def test_changed_or_untrusted_release_files_are_rejected(frozen, tmp_path, damage):
    manifest = load_json(frozen / 'release_manifest.json')
    if damage == 'scene':
        (frozen / 'scene.json').write_text('{}')
    elif damage == 'source':
        path = frozen / 'evaluator/tabletop/settlement.py'
        path.chmod(0o644)
        path.write_text(path.read_text() + '\n# changed\n')
    elif damage == 'missing_source':
        del manifest['files_sha256']['evaluator/tabletop/settlement.py']
        dump(frozen / 'release_manifest.json', manifest)
    elif damage == 'extra_source':
        (frozen / 'evaluator/tabletop/extra.py').write_text('')
    else:
        path = tmp_path / 'outside.json'
        path.write_text('{}')
        if damage == 'path_escape':
            manifest['files_sha256']['../outside.json'] = digest(path)
        else:
            (frozen / 'link.json').symlink_to(path)
            manifest['files_sha256']['link.json'] = digest(path)
        dump(frozen / 'release_manifest.json', manifest)
    with pytest.raises(ValueError):
        validate(frozen, assets=False)


@pytest.mark.parametrize('damage', ['parent', 'geometry', 'ownership', 'piece_missing', 'piece_duplicate',
                                    'reset_object', 'reset_mismatch', 'reset_missing_joint', 'reset_out_of_range',
                                    'robot', 'rules', 'qualification', 'asset_manifest'])
def test_rehashed_inconsistent_board_contract_is_rejected(frozen, damage):
    scene = load_json(frozen / 'scene.json')
    initial = load_json(frozen / 'initial_state.json')
    manifest = load_json(frozen / 'release_manifest.json')
    if damage == 'parent': scene['workroom']['version'] = 'different'
    elif damage == 'geometry': scene['board_game_task']['geometry']['pitch_m'] = .15
    elif damage == 'ownership': scene['board_game_task']['players'][0]['robot_id'] = 'right'
    elif damage == 'piece_missing': scene['objects'].pop()
    elif damage == 'piece_duplicate': scene['objects'][-1] = deepcopy(scene['objects'][0])
    elif damage == 'reset_object':
        initial['objects'] = {'black_00': {'position_m': [0, 0, 0]}}
        scene['board_game_task']['initial_state'] = initial
    elif damage == 'reset_mismatch': initial['joint_positions_rad']['left_index_DIP'] = .3
    elif damage == 'reset_missing_joint':
        initial['joint_positions_rad'].pop('left_index_DIP')
        scene['board_game_task']['initial_state'] = initial
    elif damage == 'reset_out_of_range':
        initial['joint_positions_rad']['left_index_DIP'] = 100.
        scene['board_game_task']['initial_state'] = initial
    elif damage == 'robot': scene['robot']['hand_id'] = 'sharpa_w01_right'
    elif damage == 'rules':
        task = load_json(frozen / 'task.json')
        task['settlement']['dwell_s'] = .1
        repin(frozen, 'task.json', task)
    elif damage == 'qualification':
        manifest['physics_accepted'] = True
        dump(frozen / 'release_manifest.json', manifest)
    elif damage == 'asset_manifest':
        del manifest['asset_files_sha256'][str(Path(scene['objects'][0]['asset_root']) / 'manifest.json')]
        dump(frozen / 'release_manifest.json', manifest)
    repin(frozen, 'scene.json', scene)
    repin(frozen, 'initial_state.json', initial)
    with pytest.raises(ValueError):
        validate(frozen, assets=False)


def test_asset_verification_is_not_silently_skipped(frozen):
    manifest = load_json(frozen / 'release_manifest.json')
    key = next(iter(manifest['asset_files_sha256']))
    manifest['asset_files_sha256'][key] = '0' * 64
    dump(frozen / 'release_manifest.json', manifest)
    validate(frozen, assets=False)
    with pytest.raises(ValueError, match='Frozen input changed'):
        validate(frozen)
