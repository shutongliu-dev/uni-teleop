"""Independent geometry/split/provenance checks; no simulator is initialized."""
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest

from bench.isaacsim.release import digest, load_evaluator, task_inputs, validate
from bench.paths import load_json
from scenes.registry import DEFAULT_REGISTRY, SIMPLE_TASKS, generate, load_registry, validate_scene
from scenes.release import export_scene, prepare_release


@pytest.fixture(scope='module')
def registry():
    return load_registry()


@pytest.fixture(scope='module')
def clear():
    return generate('clear_table', 'development', 11)


@pytest.mark.parametrize('scene_id', ['clear_table', 'cluttered_table', 'tray_sorting',
                                    'narrow_fixture', 'tool_table', 'cloth_table'])
def test_frozen_splits_have_six_unique_valid_initial_states(scene_id, registry):
    seen = set()
    for split, seeds in registry['split_seeds'].items():
        for seed in seeds:
            row = generate(scene_id, split, seed)
            state = json.dumps({'objects': row['scene']['objects'], 'cloth': row['scene']['cloth']}, sort_keys=True)
            assert state not in seen
            seen.add(state)
            assert row['validation']['passed']
            assert row['identity']['seed'] == seed
            assert row['identity']['split'] == split
            assert not row['identity']['physics_accepted']
            assert not row['validation']['native_stability_verified']
    assert len(seen) == 6


def test_generation_is_repeatable_without_changing_source_or_catalog(clear, registry):
    before = digest(Path(registry['source_release']['path'])/'release_manifest.json')
    repeated = generate('clear_table', 'development', 11)
    assert repeated == clear
    assert digest(Path(registry['source_release']['path'])/'release_manifest.json') == before
    assert all(digest(Path(c['root'])/'manifest.json') == c['manifest_sha256']
               for c in registry['catalogs'].values())


@pytest.mark.parametrize('split,seed', [('development', 101), ('evaluation', 11), ('unknown', 11),
                                     ('development', True), ('development', -1)])
def test_split_selection_cannot_relabel_or_alias_seed(split, seed):
    with pytest.raises(ValueError):
        generate('clear_table', split, seed)


def test_custom_seed_is_explicitly_diagnostic():
    row = generate('clear_table', 'diagnostic', 123456)
    assert row['identity']['split'] == 'diagnostic'
    assert row['identity']['seed'] == 123456


@pytest.mark.parametrize('mutation,match', [
    ('duplicate', 'Duplicate'), ('overlap', 'clearance'), ('off_table', 'beyond'),
    ('airborne', 'support'), ('nan', 'finite'), ('bad_quaternion', 'unit'), ('negative_friction', 'friction'),
])
def test_geometric_validator_rejects_bad_initializations(clear, mutation, match):
    scene = deepcopy(clear['scene'])
    cube = scene['objects'][0]
    if mutation in ('duplicate', 'overlap'):
        second = deepcopy(cube)
        if mutation == 'overlap':
            second['id'] = 'second_cube'
        scene['objects'].append(second)
    elif mutation == 'off_table':
        cube['position_m'][0] = 10.
    elif mutation == 'airborne':
        cube['position_m'][2] += .1
    elif mutation == 'nan':
        cube['position_m'][0] = float('nan')
    elif mutation == 'bad_quaternion':
        cube['quaternion_wxyz'] = [1., 1., 0., 0.]
    else:
        cube['friction'] = -1.
    with pytest.raises(ValueError, match=match):
        validate_scene(scene)


def test_cloth_and_rigid_spacing_is_checked():
    row = generate('cloth_table', 'development', 11)
    scene = deepcopy(row['scene'])
    scene['cloth']['origin_m'][:2] = [.63, .17]
    with pytest.raises(ValueError, match='clearance'):
        validate_scene(scene)


@pytest.mark.parametrize('task_id', SIMPLE_TASKS)
def test_derived_task_initial_and_goal_fields_match_actual_variant(task_id, registry):
    row = generate('clear_table', 'evaluation', 101, task_id=task_id)
    _, original_scene, original_task, _ = task_inputs(registry['source_release']['path'], task_id)
    task, scene = row['task'], row['scene']
    assert task['evaluation']['initial_position_m'] == scene['objects'][0]['position_m']
    assert scene['objects'][0]['asset_id'] == original_scene['objects'][0]['asset_id'] == 'collab_cube_small'
    assert task['public_goal']['target_position_m'] == task['target_position_m'] == task['evaluation']['goal_position_m']
    assert task['public_goal']['success_contract'] == original_task['public_goal']['success_contract']
    assert task['evaluation']['humanlike_grasp'] == original_task['evaluation']['humanlike_grasp']
    if task_id in ('02_cube_lift_hold', '04_cube_grasp_hold'):
        delta = np.array(scene['objects'][0]['position_m']) - original_task['evaluation']['initial_position_m']
        assert np.allclose(task['target_position_m'], np.array(original_task['target_position_m'])+delta)
        assert scene['goal_markers'][0]['position_m'][:2] == task['target_position_m'][:2]
    else:
        assert task['target_position_m'] == original_task['target_position_m']
    assert task['steps'] == original_task['steps']  # Existing wording and target dimensions remain true.


def test_release_is_loadable_by_current_worker_and_keeps_frozen_scorer(tmp_path, registry):
    destination = tmp_path/'new_release'
    manifest = prepare_release('cluttered_table', 'evaluation', 211, destination, task_id='01_cube_push')
    loaded = validate(destination)
    assert loaded == manifest
    _, scene, task, initial = task_inputs(destination, '01_cube_push')
    assert initial == {}
    assert task['evaluation']['initial_position_m'] == scene['objects'][0]['position_m']
    assert scene['scenario']['split'] == 'evaluation'
    assert manifest['episode_request_defaults'] == {'task_id': task['id'], 'seed': scene['scenario']['seed'],
        'scene_variant': 'original_v1', 'robot_id': 'sharpa_w01_right'}
    assert scene['scenario']['seed'] == 211
    assert manifest['task_qualified'] is False
    assert not manifest['physics_accepted']
    assert manifest['status'] == 'prepared_unvalidated'
    scorer = load_evaluator(destination)(task)
    assert scorer is not None
    source = Path(registry['source_release']['path'])
    for relative, item in manifest['files'].items():
        if relative.startswith('evaluator/'):
            assert digest(destination/relative) == digest(source/relative)
    assert manifest['physics_profile_sha256'] == digest(source/manifest['physics_profile'])
    with pytest.raises(FileExistsError):
        prepare_release('cluttered_table', 'evaluation', 211, destination, task_id='01_cube_push')


@pytest.mark.parametrize('scene_id', ['tray_sorting', 'cloth_table'])
def test_unsupported_scoring_fails_before_writing_release(tmp_path, scene_id):
    with pytest.raises(ValueError, match='no supported evaluator'):
        prepare_release(scene_id, 'development', 11, tmp_path/'unsupported', task_id='01_cube_push')
    assert not (tmp_path/'unsupported').exists()
    manifest = export_scene(scene_id, 'development', 11, tmp_path/'scene_only')
    assert manifest['task'] is None and manifest['native_execution_performed'] is False
    assert manifest['episode_request_defaults'] is None and manifest['task_qualified'] is False
    assert all(digest(tmp_path/'scene_only'/name) == expected
               for name, expected in manifest['files_sha256'].items())


def test_registry_rejects_overlapping_splits_and_catalog_tampering(tmp_path):
    registry = load_json(DEFAULT_REGISTRY)
    registry['split_seeds']['evaluation'][0] = registry['split_seeds']['development'][0]
    path = tmp_path/'registry.json'
    path.write_text(json.dumps(registry))
    with pytest.raises(ValueError, match='distinct'):
        load_registry(path)
    registry = load_json(DEFAULT_REGISTRY)
    registry['catalogs']['target_cube']['manifest_sha256'] = '0'*64
    path.write_text(json.dumps(registry))
    with pytest.raises(ValueError, match='catalog changed'):
        generate('clear_table', 'development', 11, registry_path=path)


def test_same_named_target_cannot_switch_asset_catalog(tmp_path):
    registry = load_json(DEFAULT_REGISTRY)
    registry['catalogs']['target_cube']['root'] = str(tmp_path/'same_shape_different_mass')
    path = tmp_path/'registry.json'
    path.write_text(json.dumps(registry))
    with pytest.raises(ValueError, match='Target asset catalog changed'):
        generate('clear_table', 'development', 11, registry_path=path)


@pytest.mark.parametrize('field,value', [('id', 'wrong_target'), ('catalog', 'procedural'),
    ('asset_id', 'other_shape'), ('kind', 'fixtures'), ('friction', 0.)])
def test_target_placement_cannot_replace_scored_object(tmp_path, field, value):
    registry = load_json(DEFAULT_REGISTRY)
    registry['scenes']['clear_table']['target_pose'][field] = value
    path = tmp_path/'registry.json'
    path.write_text(json.dumps(registry))
    with pytest.raises(ValueError, match='target_pose'):
        prepare_release('clear_table', 'development', 11, tmp_path/'invalid_release',
                        task_id='01_cube_push', registry_path=path)
    assert not (tmp_path/'invalid_release').exists()


def test_release_cannot_overwrite_or_nest_in_source(tmp_path, registry):
    source = Path(registry['source_release']['path'])
    with pytest.raises(FileExistsError):
        prepare_release('clear_table', 'development', 11, source, task_id='01_cube_push')
    with pytest.raises(ValueError, match='separate'):
        prepare_release('clear_table', 'development', 11, source/'new_scenario', task_id='01_cube_push')
    assert not (source/'new_scenario').exists()
