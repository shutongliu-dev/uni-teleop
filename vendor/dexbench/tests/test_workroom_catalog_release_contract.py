"""The scene catalog must reject releases that the native loader cannot use.

Only disposable copies are edited. These are CPU contract checks, not native
physics or manipulation qualification.
"""
from copy import deepcopy
import json
from pathlib import Path
import shutil

import pytest

from bench.paths import load_json
from scenes.workrooms import catalog
from scenes.board_games.runtime_release import validate_board_game_release


@pytest.fixture
def board_release(tmp_path, monkeypatch):
    definition = catalog.task_definition('card_room', 'tic_tac_toe_3x3')
    root = tmp_path / 'release'
    shutil.copytree(definition['source_release']['path'], root)
    definition['source_release']['path'] = str(root)
    monkeypatch.setattr(catalog, 'task_definition', lambda *_: deepcopy(definition))
    return root, definition


def repin(root, definition, updates):
    from scenes.board_games.tabletop.release import sha256

    manifest = load_json(root / 'release_manifest.json')
    for name, value in updates.items():
        path = root / name
        path.write_text(json.dumps(value))
        manifest['files_sha256'][name] = sha256(path)
        if name in manifest.get('files', {}):
            manifest['files'][name]['sha256'] = sha256(path)
    (root / 'release_manifest.json').write_text(json.dumps(manifest))
    definition['source_release']['manifest_sha256'] = sha256(root / 'release_manifest.json')


@pytest.mark.parametrize('damage', [
    'missing_piece', 'duplicate_piece', 'geometry', 'ownership', 'missing_board',
    'reset_mismatch', 'reset_missing_joint', 'reset_out_of_range', 'robot',
    'missing_evaluator_hash', 'extra_evaluator', 'missing_asset_manifest',
    'qualification',
])
def test_catalog_and_runtime_both_reject_inconsistent_frozen_contract(board_release, damage):
    root, definition = board_release
    scene = load_json(root / 'scene.json')
    initial = load_json(root / 'initial_state.json')
    manifest = load_json(root / 'release_manifest.json')
    if damage == 'missing_piece':
        scene['objects'].pop()
    elif damage == 'duplicate_piece':
        scene['objects'][-1] = deepcopy(scene['objects'][0])
    elif damage == 'geometry':
        scene['board_game_task']['geometry']['pitch_m'] = .15
    elif damage == 'ownership':
        scene['board_game_task']['players'][0]['robot_id'] = 'right'
    elif damage == 'missing_board':
        scene['fixtures'] = [row for row in scene['fixtures']
                             if row['id'] != scene['board_game_task']['board_fixture_id']]
    elif damage == 'reset_mismatch':
        initial['joint_positions_rad']['left_index_DIP'] = .3
    elif damage == 'reset_missing_joint':
        initial['joint_positions_rad'].pop('left_index_DIP')
        scene['board_game_task']['initial_state'] = initial
    elif damage == 'reset_out_of_range':
        initial['joint_positions_rad']['left_index_DIP'] = 100.
        scene['board_game_task']['initial_state'] = initial
    elif damage == 'robot':
        scene['robot']['hand_id'] = 'sharpa_w01_right'
    elif damage == 'missing_evaluator_hash':
        del manifest['files_sha256']['evaluator/tabletop/settlement.py']
    elif damage == 'extra_evaluator':
        (root / 'evaluator/tabletop/unrecorded.py').write_text('# Unrecorded source\n')
    elif damage == 'missing_asset_manifest':
        del manifest['asset_files_sha256'][str(Path(scene['objects'][0]['asset_root']) / 'manifest.json')]
    elif damage == 'qualification':
        manifest['physics_accepted'] = True
    (root / 'release_manifest.json').write_text(json.dumps(manifest))
    repin(root, definition, {'scene.json': scene, 'initial_state.json': initial})
    manifest = load_json(root / 'release_manifest.json')
    with pytest.raises(ValueError):
        validate_board_game_release(root, manifest, assets=False)
    with pytest.raises(ValueError):
        catalog.load_task('card_room', 'tic_tac_toe_3x3')


@pytest.mark.parametrize('schema', [None, 'unsupported_task_release.v1', 'mechanical_fastener_task_release.v1'])
def test_catalog_rejects_changed_release_type_before_dispatch(board_release, schema):
    root, definition = board_release
    manifest = load_json(root / 'release_manifest.json')
    if schema is None:
        manifest.pop('schema')
    else:
        manifest['schema'] = schema
    (root / 'release_manifest.json').write_text(json.dumps(manifest))
    repin(root, definition, {})
    with pytest.raises(ValueError, match='schema differs'):
        catalog.load_task('card_room', 'tic_tac_toe_3x3')


@pytest.mark.parametrize('game', ['tic_tac_toe_3x3', 'connect_four_5x5', 'hex7x7'])
def test_current_board_game_catalog_preserves_canonical_frozen_inputs(game):
    from bench.isaacsim.release import digest, task_inputs

    definition = catalog.task_definition('card_room', game)
    root = Path(definition['source_release']['path'])
    before = digest(root / 'release_manifest.json')
    loaded = catalog.load_task('card_room', game)
    _, scene, task, initial = task_inputs(root, game)
    assert (loaded['scene'], loaded['task'], loaded['initial_state']) == (scene, task, initial)
    assert loaded['release_manifest'] == load_json(root / 'release_manifest.json')
    assert digest(root / 'release_manifest.json') == before == definition['source_release']['manifest_sha256']
    assert loaded['release_manifest']['physics_accepted'] is False


@pytest.mark.parametrize('damage', ['asset_catalog', 'asset_dependency'])
def test_mechanical_catalog_checks_canonical_asset_hashes(tmp_path, monkeypatch, damage):
    from bench.isaacsim.release import validate

    definition = catalog.task_definition('mechanical_assembly', 'fastener_assembly')
    root = tmp_path / 'release'
    shutil.copytree(definition['source_release']['path'], root)
    # The published candidate is sealed. Mutate only this disposable copy.
    for path in (root, *root.rglob('*')):
        assert not path.is_symlink()
        path.chmod(path.stat().st_mode | 0o200)
    definition['source_release']['path'] = str(root)
    manifest = load_json(root / 'release_manifest.json')
    updates = {}
    if damage == 'asset_catalog':
        manifest['asset_catalog']['sha256'] = '0' * 64
    else:
        inventory = load_json(root / 'asset_file_hashes.json')
        inventory[next(iter(inventory))]['sha256'] = '0' * 64
        updates['asset_file_hashes.json'] = inventory
    (root / 'release_manifest.json').write_text(json.dumps(manifest))
    repin(root, definition, updates)
    monkeypatch.setattr(catalog, 'task_definition', lambda *_: deepcopy(definition))
    with pytest.raises(ValueError, match='Frozen input changed'):
        validate(root)
    with pytest.raises(ValueError, match='Frozen input changed'):
        catalog.load_task('mechanical_assembly', 'fastener_assembly')
