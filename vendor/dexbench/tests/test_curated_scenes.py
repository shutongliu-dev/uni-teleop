"""Task inventory and frozen-input integrity for five uncluttered layouts."""
from copy import deepcopy

import pytest

from bench.isaacsim.curated_release import SOURCES, TASK_LAYOUT, prepare
from bench.isaacsim.release import digest, load_evaluator, task_inputs, validate
from bench.paths import load_json
from bench.simulation.isaacsim_backend.scene_variants import CURATED_LAYOUTS, apply_scene_variant, curate_scene


def source_inputs(task_id):
    source = SOURCES['simple' if 'cube_' in task_id else 'sequential']
    manifest = load_json(source / 'release_manifest.json')
    entry = next(row for row in manifest['tasks'] if row['id'] == task_id)
    return source, entry, load_json(source / entry['scene'])


@pytest.mark.parametrize('task_id', list(TASK_LAYOUT))
def test_curated_inventory_preserves_required_assets_and_all_physical_settings(task_id):
    _, _, scene = source_inputs(task_id)
    original = deepcopy(scene)
    selected = curate_scene(scene, TASK_LAYOUT[task_id])
    assert scene == original
    layout = CURATED_LAYOUTS[TASK_LAYOUT[task_id]]
    for kind in ('objects', 'fixtures'):
        assert {row['id']: row['asset_id'] for row in selected[kind]} == layout[kind]
        assert selected[kind] == [row for row in scene.get(kind, []) if row['id'] in layout[kind]]
    for key in ('robot', 'table', 'cloth', 'physics', 'isaacsim', 'goal_markers'):
        assert selected.get(key) == scene.get(key)
    assert selected['workcell_details']['enabled'] is False
    assert selected['workbench']['props'] is False
    inferred = apply_scene_variant(scene, 'curated_v1')
    assert inferred == selected


def test_curating_enriched_scene_removes_unrelated_instances_not_shared_assets():
    _, _, source = source_inputs('01_ordered_kitting')
    decorated = apply_scene_variant(source, 'enriched_v2')
    curated = curate_scene(decorated)
    assert curated['objects'] == source['objects']
    assert curated['fixtures'] == source['fixtures']
    removed = curated['scene_variant']['removed_instance_ids']
    assert len(removed['objects']) == 3
    assert len(removed['fixtures']) == 2
    assert curated['scene_variant']['shared_asset_files_removed'] is False


@pytest.mark.parametrize('mutation', ['missing', 'wrong_asset', 'duplicate', 'cloth'])
def test_curating_rejects_missing_or_substituted_task_geometry(mutation):
    _, _, scene = source_inputs('01_cube_push')
    if mutation == 'missing':
        scene['objects'] = [row for row in scene['objects'] if row['id'] != 'cube']
    elif mutation == 'wrong_asset':
        scene['objects'][0]['asset_id'] = 'unreviewed_cube'
    elif mutation == 'duplicate':
        scene['objects'].append(deepcopy(scene['objects'][0]))
    else:
        scene['cloth']['enabled'] = True
    with pytest.raises(ValueError):
        curate_scene(scene, 'cube_workbench')


@pytest.mark.parametrize('family', ['simple', 'sequential'])
def test_curated_release_freezes_new_scenes_but_preserves_tasks_evaluators_assets_and_source(family, tmp_path):
    source = SOURCES[family]
    original = validate(source)
    original_hash = digest(source / 'release_manifest.json')
    output = tmp_path / 'curated_release'
    result = prepare(source, output)
    assert validate(output) == result
    assert digest(source / 'release_manifest.json') == original_hash
    assert digest(output / 'source_release_manifest.json') == original_hash
    assert result['scene_stratum'] == 'curated_v1'
    for entry in result['tasks']:
        prior = next(row for row in original['tasks'] if row['id'] == entry['id'])
        assert entry['task_sha256'] == prior['task_sha256']
        assert (output / entry['task']).read_bytes() == (source / prior['task']).read_bytes()
        assert entry['scene_sha256'] != prior['scene_sha256']
        _, scene, task, initial = task_inputs(output, entry['id'])
        assert not initial
        assert scene['scene_variant']['retained_asset_specs_unchanged']
        assert task == load_json(source / prior['task'])
    for relative in original['files']:
        if relative.startswith('evaluator/') or relative == 'asset_file_hashes.json':
            assert (output / relative).read_bytes() == (source / relative).read_bytes()
    assert result['physics_profile_sha256'] == original['physics_profile_sha256']
    assert load_evaluator(output)
    assert not list(tmp_path.glob('.curated_release.preparing-*'))
    with pytest.raises(FileExistsError):
        prepare(source, output)
