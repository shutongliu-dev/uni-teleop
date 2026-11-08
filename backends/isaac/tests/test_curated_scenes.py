"""Task inventory and frozen-input integrity for five uncluttered layouts."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from bench.paths import DATA_ROOT, load_json
from bench.simulation.isaacsim_backend.scene_variants import CURATED_LAYOUTS, apply_scene_variant, curate_scene

RELEASES = Path(f'{DATA_ROOT}/datasets/fr3_sharpa_bench/isaacsim/releases')
FROZEN_PAIRS = {
    'simple': ('simple_manipulation/v4_humanlike', 'simple_manipulation/v5_curated'),
    'sequential': ('sequential_workcell/v1', 'sequential_workcell/v2_curated'),
}
TASK_IDS = ('01_cube_push', '02_cube_lift_hold', '03_cube_pick_place', '04_cube_grasp_hold',
            '01_ordered_kitting', '02_loose_gear_assembly', '03_lined_tray_loading', '04_cover_and_secure')


def source_inputs(task_id):
    source_name, published_name = FROZEN_PAIRS['simple' if 'cube_' in task_id else 'sequential']
    source = RELEASES / source_name
    manifest = load_json(source / 'release_manifest.json')
    entry = next(row for row in manifest['tasks'] if row['id'] == task_id)
    published = RELEASES / published_name
    frozen_manifest = load_json(published / 'release_manifest.json')
    frozen_entry = next(row for row in frozen_manifest['tasks'] if row['id'] == task_id)
    frozen_scene = load_json(published / frozen_entry['scene'])
    return frozen_scene['scene_variant']['layout_id'], load_json(source / entry['scene'])


@pytest.mark.parametrize('task_id', TASK_IDS)
def test_curated_inventory_preserves_required_assets_and_all_physical_settings(task_id):
    layout_id, scene = source_inputs(task_id)
    original = deepcopy(scene)
    selected = curate_scene(scene, layout_id)
    assert scene == original
    layout = CURATED_LAYOUTS[layout_id]
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
    _, source = source_inputs('01_ordered_kitting')
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
    _, scene = source_inputs('01_cube_push')
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
