"""Frozen laundry candidates retain exact task inputs while binding the table."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from bench.isaacsim.release import digest, dump, task_inputs, validate
from bench.paths import load_json
from scenes.workrooms.laundry_room import composition, release, table_material_release as derived
from scenes.workrooms.laundry_room.robot_configuration_catalog import RELEASES
from scenes.workrooms.laundry_room.table_material import OPTION, table_material_options

CASES = [(task, installation, Path(row['source_release']['path']))
         for task, item in RELEASES.items() for installation, row in item['configurations'].items()]


def read(path):
    return load_json(path)


def inventory(root):
    return {str(path.relative_to(root)): digest(path) for path in root.rglob('*')
            if path.is_file() and '__pycache__' not in path.parts}


@pytest.fixture
def preparation(tmp_path, monkeypatch):
    from scripts.maintenance import freeze_project_sources
    # Fixture bypasses only source freezing and repeated large shared-asset
    # hashing. Release hashes, actual cloth asset loader, reconstruction, frozen
    # evaluator and ordinary task_inputs are exercised unchanged.
    frozen = tmp_path/'frozen_source'
    dump(frozen/'source_snapshot_manifest.json', {'fixture_only': True})
    monkeypatch.setattr(freeze_project_sources, 'verify_if_snapshot',
                        lambda *a, **k: {'fixture_only': True, 'snapshot_root': str(frozen)})
    monkeypatch.setattr(derived, 'DATASETS', Path('/'))
    monkeypatch.setattr(release, 'DATASETS', Path('/'))

    def base_source(source):
        from laundry_release_fixtures import unbound_source
        original_inventory = read(source/'asset_file_hashes.json')
        monkeypatch.setattr(release, 'asset_inventory', lambda scene: deepcopy(original_inventory))
        return unbound_source(source, tmp_path/'unbound', table=False)

    def prepare(source, name='candidate'):
        source = base_source(source)
        original_inventory = read(source/'asset_file_hashes.json')
        monkeypatch.setattr(release, 'asset_inventory', lambda scene: deepcopy(original_inventory))
        output = tmp_path/name
        manifest = derived.prepare(source, output)
        return output, manifest
    prepare.source = base_source
    return prepare


@pytest.mark.parametrize('task,installation,source', CASES, ids=[f'{t}-{i}' for t,i,_ in CASES])
def test_all_fifteen_current_releases_preserve_frozen_files_and_episode_options(
        preparation, task, installation, source):
    source = preparation.source(source)
    before = inventory(source)
    old_manifest = validate(source, assets=False)
    output, manifest = preparation(source)
    assert validate(output) == manifest
    assert manifest['status'] == 'prepared_unvalidated'
    assert manifest['physics_accepted'] is False and manifest['task_policy_success_claimed'] is False
    assert manifest[derived.MARKER]['qualification_inherited'] is False
    assert manifest['configuration_id'] == installation
    assert manifest['physics_profile_sha256'] == digest(output/'physics_profile.json')
    for name, row in old_manifest['files'].items():
        if name not in derived.CHANGED_FILES:
            assert digest(output/name) == row['sha256']
    for seed in (0, 19):
        _, old_scene, old_task, old_initial = task_inputs(source, task, seed=seed)
        _, scene, new_task, initial = task_inputs(output, task, seed=seed)
        assert scene['isaacsim'].pop(OPTION) == table_material_options()
        assert scene == old_scene
        assert new_task == old_task and initial == old_initial
        assert OPTION not in scene['laundry_task_provenance']['cloth_physics_options']
    assert inventory(source) == before


def test_settled_fold_reset_comparison_excludes_explicit_table_option(monkeypatch):
    from scenes.workrooms.laundry_room import assets
    metadata = assets.candidate_metadata('towel_stacking')
    metadata['settled_fold_reset'] = {'physics_options': deepcopy(composition.PHYSICS_PRIORS)}
    monkeypatch.setattr(composition, 'load_asset_manifest', lambda *_: ({'fixture': True}, deepcopy(metadata)))
    plain = composition.build_task_inputs('towel_stacking', asset_manifest='fixture')
    changed = composition.build_task_inputs('towel_stacking', asset_manifest='fixture',
        physics_options={OPTION: table_material_options()})
    assert changed['scene']['isaacsim'].pop(OPTION) == table_material_options()
    assert changed == plain
    with pytest.raises(ValueError, match='successful fold physics'):
        composition.build_task_inputs('towel_stacking', asset_manifest='fixture',
            physics_options={OPTION: table_material_options(), 'cloth_iterations': 32})


@pytest.mark.parametrize('fault', ['table_value', 'profile_dropped', 'scorer', 'qualification', 'source_pin'])
def test_rehashed_candidate_cannot_escape_narrow_derivation(preparation, fault):
    source = CASES[0][2]
    output, manifest = preparation(source)
    if fault == 'table_value':
        for filename in ('scene.json', 'physics_profile.json'):
            value = read(output/filename)
            options = value['isaacsim' if filename == 'scene.json' else 'isaacsim_options']
            options[OPTION]['surfaces']['table']['dynamic_friction'] = .4
            dump(output/filename, value)
    elif fault == 'profile_dropped':
        value = read(output/'physics_profile.json')
        value['isaacsim_options'].pop(OPTION)
        dump(output/'physics_profile.json', value)
    elif fault == 'scorer':
        with (output/'evaluator/sequence_evaluator.py').open('a') as handle:
            handle.write('\n# changed scorer\n')
    elif fault == 'qualification':
        manifest[derived.MARKER]['qualification_inherited'] = True
    else:
        manifest[derived.MARKER]['base_release']['manifest_sha256'] = '0'*64
    for name in manifest['files']:
        manifest['files'][name]['sha256'] = digest(output/name)
        manifest['files_sha256'][name] = manifest['files'][name]['sha256']
    manifest['tasks'][0]['scene_sha256'] = manifest['files']['scene.json']['sha256']
    manifest['physics_profile_sha256'] = manifest['files']['physics_profile.json']['sha256']
    dump(output/'release_manifest.json', manifest)
    with pytest.raises(ValueError):
        validate(output)


def test_original_inputs_and_existing_destinations_are_never_overwritten(preparation):
    source = CASES[0][2]
    source = preparation.source(source)
    before = inventory(source)
    output, _ = preparation(source)
    candidate_before = inventory(output)
    with pytest.raises(ValueError, match='fresh independent'):
        derived.prepare(source, source)
    with pytest.raises(ValueError, match='fresh independent'):
        derived.prepare(source, output)
    with pytest.raises(ValueError, match='original laundry'):
        derived.prepare(output, output.parent/'nested_candidate')
    assert inventory(source) == before and inventory(output) == candidate_before


def test_table_binding_does_not_change_floor_or_static_actor_and_reports_evidence_scope():
    pytest.importorskip('pxr.Usd')
    from pxr import Usd, UsdGeom, UsdPhysics, UsdShade
    from bench.simulation.isaacsim_backend.workroom import SUPPORT_PATHS
    from bench.simulation.isaacsim_backend import workroom_materials as materials
    stage = Usd.Stage.CreateInMemory()
    for path in SUPPORT_PATHS.values():
        UsdPhysics.CollisionAPI.Apply(UsdGeom.Cube.Define(stage, path).GetPrim())
    record = materials.apply_materials(stage, SUPPORT_PATHS, table_material_options())
    table = record['surfaces']['table']
    assert set(record['surfaces']) == {'table'}
    assert table['usd_resolved']['dynamic_friction'] == pytest.approx(.45)
    assert table['usd_resolved']['static_friction'] == pytest.approx(.45)
    assert table['usd_resolved']['restitution'] == 0.
    assert table['usd_resolved']['friction_combine'] == table['usd_resolved']['restitution_combine'] == 'average'
    assert not stage.GetPrimAtPath(SUPPORT_PATHS['table']).HasAPI(UsdPhysics.RigidBodyAPI)
    bound, _ = UsdShade.MaterialBindingAPI(stage.GetPrimAtPath(SUPPORT_PATHS['floor'])).ComputeBoundMaterial('physics')
    assert not bound
    # Deliberately substitute only the native binding query, not its result
    # classification. CPU fixtures cannot claim a native dynamics check.
    report = materials.runtime_material_check(stage, None, {'workbench': {'surface_materials': record}},
        native_material_paths=lambda path: [table['material_path']] if path == SUPPORT_PATHS['table'] else [])
    assert report['surfaces']['table']['native_binding_verified']
    assert report['surfaces']['table']['coefficient_evidence'] == 'usd_only_static_actor'
    assert not report['all_native_coefficients_verified'] and not report['native_combine_verified']
