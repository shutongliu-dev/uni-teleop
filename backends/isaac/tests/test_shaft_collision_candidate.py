"""The tested dense rings must remain an explicit canonical-reset option."""
from copy import deepcopy

import pytest

from bench.isaacsim.release import load_evaluator, task_inputs, validate
from bench.paths import load_json
from scenes.workrooms.mechanical_assembly import release
from scenes.workrooms.mechanical_assembly.tasks.shaft_spacer_assembly import (
    collision_candidates as candidates, task)


def dense():
    return task.build_task_inputs(solver_type='PGS', ring_collision_candidate='dense1024_v1')


def test_dense_binding_preserves_supply_rules_fixtures_and_all_other_physics():
    original = task.build_task_inputs(solver_type='PGS')
    selected = dense()
    assert selected['task'] == original['task']
    assert selected['initial_state'] == original['initial_state']
    for key in ('position_m', 'quaternion_wxyz', 'contact_offset_m', 'rest_offset_m'):
        assert [o[key] for o in selected['scene']['objects']] == [o[key] for o in original['scene']['objects']]
    assert [o['position_m'] for o in selected['scene']['objects']] == [
        [.25, -.27, .786], [.25, 0., .786], [.25, .27, .786]]
    for obj in selected['scene']['objects']:
        assert obj['asset_root'] == str(candidates.DENSE_ROOT)
        assert obj['asset_id'] == 'mechanical_l3_'+obj['id']+'_sdf1024'
        assert obj['usd_physics_policy'] == 'preserve_authored_v1'
        assert obj['usd_physics_material_fallback'] == {
            'static_friction': .4, 'dynamic_friction': .4,
            'restitution': 0., 'friction_combine': 'average'}
        assert 'collision_mesh_coordinate_scale' not in obj
        assert 'friction' not in obj
    selected['scene']['objects'] = deepcopy(original['scene']['objects'])
    selected['scene']['mechanical_task_provenance'].pop('ring_collision_candidate')
    assert selected == original
    assert task.build_task_inputs() == task.build_task_inputs(ring_collision_candidate=None)


@pytest.mark.parametrize('solver', [None, 'TGS'])
def test_dense_requires_explicit_tested_solver(solver):
    with pytest.raises(ValueError, match='explicit solver_type=PGS'):
        task.build_task_inputs(solver_type=solver, ring_collision_candidate='dense1024_v1')


@pytest.mark.parametrize('candidate', ['dense2048', 'sparse1024', False, 1])
def test_unknown_collision_candidate_is_rejected(candidate):
    with pytest.raises(ValueError, match='Unknown'):
        task.build_task_inputs(solver_type='PGS', ring_collision_candidate=candidate)


def test_changed_catalog_cannot_replace_pinned_candidate(tmp_path, monkeypatch):
    (tmp_path/'manifest.json').write_text('{"schema": "not_the_reviewed_catalog"}\n')
    monkeypatch.setattr(candidates, 'DENSE_ROOT', tmp_path)
    with pytest.raises(ValueError, match='Pinned dense ring manifest changed'):
        dense()


@pytest.mark.parametrize('key', candidates.PHYSICAL_FIELDS)
def test_source_physical_drift_is_rejected(key, monkeypatch):
    original = candidates.load_task_asset

    def changed(asset_id, *args, **kwargs):
        value = original(asset_id, *args, **kwargs)
        if asset_id.endswith('_sdf1024'):
            value[key] = 'changed'
        return value

    monkeypatch.setattr(candidates, 'load_task_asset', changed)
    with pytest.raises(ValueError, match='physical fields'):
        dense()


def test_dense_frozen_release_loads_original_reset_and_frozen_evaluator(tmp_path, monkeypatch):
    monkeypatch.setattr(release, 'DATASETS', tmp_path)
    path = tmp_path/'dense_release'
    release.prepare_task_release('shaft_spacer_assembly', path,
        solver_type='PGS', ring_collision_candidate='dense1024_v1')
    manifest = validate(path)
    _, scene, frozen, initial = task_inputs(path, 'shaft_spacer_assembly')
    assert frozen == dense()['task'] and initial == dense()['initial_state']
    assert load_evaluator(path)(frozen).task['id'] == 'shaft_spacer_assembly'
    assert manifest['physics_accepted'] is False
    assert manifest['task_policy_success_claimed'] is False
    assert 'engineering_probe' not in frozen
    assert 'engineering_probe' not in scene['mechanical_task_provenance']
    inventory = load_json(path/'asset_file_hashes.json')
    for filename in ('manifest.json', 'SOURCE.json', 'LICENSE.txt',
                     'spacer.usda', 'wide_ring.usda', 'top_ring.usda'):
        assert str(candidates.DENSE_ROOT/filename) in inventory
    with pytest.raises(ValueError, match='original task reset'):
        task_inputs(path, 'shaft_spacer_assembly', phase='qualification_probe')
