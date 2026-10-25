import json
from copy import deepcopy
from pathlib import Path

import pytest

from bench.paths import DATA_ROOT, load_json
from scripts.assets.prepare_mechanical_ring_sdf import prepare
from scripts.assets.prepare_sorting_ring_sdf import prepare_inputs

SOURCE = Path(f'{DATA_ROOT}/datasets/fr3_sharpa_bench/development/mechanical_task_validation_20261018/sorting/native03/input.json')
VALIDATED_INPUT = SOURCE.parent.parent/'native_sdf1024_01/input.json'


def test_l1_candidate_preserves_all_cases_and_physical_settings(tmp_path):
    pytest.importorskip('pxr')
    original = load_json(SOURCE)
    candidate = prepare_inputs(tmp_path/'ring',SOURCE)
    assert candidate['cases'] == original['cases']
    assert candidate['task'] == original['task']
    assert candidate['initial_state'] == original['initial_state']
    assert candidate['scene']['isaacsim'] == original['scene']['isaacsim']
    assert candidate['scene']['fixtures'] == original['scene']['fixtures']
    assert candidate['scene']['objects'][:2] == original['scene']['objects'][:2]
    ring = deepcopy(candidate['scene']['objects'][2])
    for key in ('asset_root','asset_id','usd_physics_policy','usd_physics_material_fallback'):
        ring.pop(key)
    old = deepcopy(original['scene']['objects'][2])
    for key in ('asset_root','asset_id','friction','collision_mesh_coordinate_scale'):
        old.pop(key)
    assert ring == old
    manifest = load_json(tmp_path/'ring/manifest.json')
    material = manifest['assets']['sorting_ring_sdf1024']['material_prior']
    assert material == {'static_friction':.4,'dynamic_friction':.4,'restitution':0.,'friction_combine':'average'}


def test_missing_physics_fallback_cannot_override_authored_friction(tmp_path):
    pytest.importorskip('pxr')
    with pytest.raises(ValueError,match='source friction'):
        prepare(tmp_path/'bad', f'{DATA_ROOT}/assets/task_parts/mechanical_assembly/parts_sorting/v2',
            'sorting_ring',physics_fallback={'material_prior':{'static_friction':.5,'dynamic_friction':.5,
            'restitution':0.,'friction_combine':'average'},'contact_offset_m':.0001,'rest_offset_m':0.})
    assert not (tmp_path/'bad').exists()


def test_task_release_uses_validated_supply_layout_and_explicit_sdf_ring(tmp_path, monkeypatch):
    from bench.isaacsim.release import task_inputs
    from scenes.workrooms.mechanical_assembly import release
    from scenes.workrooms.mechanical_assembly.tasks.parts_sorting.task import build_task_inputs
    validated = load_json(VALIDATED_INPUT)
    ring_root = next(o['asset_root'] for o in validated['scene']['objects'] if o['id'] == 'ring')
    candidate = build_task_inputs(ring_asset_root=ring_root)
    original = build_task_inputs()
    assert candidate['task'] == original['task']
    assert candidate['initial_state'] == original['initial_state'] == validated['initial_state']
    for key in ('objects', 'fixtures', 'robot', 'isaacsim', 'workroom'):
        assert candidate['scene'][key] == validated['scene'][key]
    assert candidate['task']['goals'] == validated['task']['goals']
    assert candidate['task']['evaluation']['initial_objects'] == validated['cases'][0]['reset_objects']
    monkeypatch.setattr(release, 'DATASETS', tmp_path)
    root = tmp_path/'release'
    manifest = release.prepare_task_release('parts_sorting', root, ring_asset_root=ring_root)
    _, scene, task, initial = task_inputs(root, 'parts_sorting')
    assert (scene, task, initial) == (candidate['scene'], candidate['task'], candidate['initial_state'])
    assert manifest['physics_accepted'] is False
    assert manifest['task_policy_success_claimed'] is False


def test_task_rejects_ring_binding_with_different_material(monkeypatch):
    from scenes.workrooms.mechanical_assembly.tasks.parts_sorting import task
    validated = load_json(VALIDATED_INPUT)
    ring_root = next(o['asset_root'] for o in validated['scene']['objects'] if o['id'] == 'ring')
    load = task.load_task_asset

    def changed_material(asset_id, root, **options):
        result = deepcopy(load(asset_id, root, **options))
        if asset_id == 'sorting_ring_sdf1024':
            result['material_prior']['static_friction'] = .5
        return result

    monkeypatch.setattr(task, 'load_task_asset', changed_material)
    with pytest.raises(ValueError, match='preserve the selected source geometry and physics'):
        task.build_task_inputs(ring_asset_root=ring_root)
