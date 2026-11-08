"""Versioned source-mesh placement; no solver/controller/native changes."""
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
from pxr import Gf, Usd, UsdGeom, UsdPhysics
import pytest

from bench.isaacsim.release import validate
from bench.paths import DATA_ROOT, load_json
from bench.runtime.worker import public_task_projection
from scenes.workrooms.kitchen import seed_variants as subject


PUBLICATION = Path(f'{DATA_ROOT}/datasets/fr3_sharpa_bench/isaacsim/releases/'
                   'workroom_initial_randomization_v1/publication01/releases')


@pytest.fixture(scope='module')
def parent():
    release = PUBLICATION/'produce_transfer/single_arm'
    manifest = load_json(release/'release_manifest.json')
    original = Path(manifest['base_release']['path'])
    assert subject.digest(original/'release_manifest.json') == manifest['base_release']['manifest_sha256']
    scene, task, asset, metadata, _ = subject._parent(original)
    return original, scene, task, asset, metadata, Usd.Stage.Open(asset['usd_path'])


@pytest.mark.parametrize('task_id', subject.TASK_IDS)
def test_original_v1_release_all_ten_cases_still_strictly_recompose(task_id):
    manifest = validate(PUBLICATION/task_id/'single_arm')
    assert manifest['seed_values'] == list(range(10))
    parent_path = Path(manifest['base_release']['path'])
    metadata = subject._parent(parent_path)[3]
    for record in manifest['cases'].values():
        old = load_json(Path(record['asset_manifest']))
        assert old['generator_version'] == subject.VERSION
        assert subject.case_plan(task_id, old['seed'], metadata) == old['case_plan']


@pytest.mark.parametrize('seed', range(10))
def test_contact_profile_changes_only_nonzero_carrot_initial_z(parent, seed):
    metadata, stage = parent[4:]
    source = stage.GetRootLayer().ExportToString()
    old_plan = subject.case_plan('produce_transfer', seed, metadata)
    plan = subject.case_plan('produce_transfer', seed, metadata, profile=subject.CONTACT_RESET_VERSION)
    assert plan['bodies'] == old_plan['bodies']
    assert plan['reset_profile']['clearance_m'] == 0
    old = subject.derive_stage(stage, metadata, old_plan)
    new = subject.derive_stage(stage, metadata, plan)
    assert stage.GetRootLayer().ExportToString() == source
    assert new['static_geometry'] == old['static_geometry']
    assert new['interaction_targets'] == old['interaction_targets']
    assert new['fixed_support_paths'] == old['fixed_support_paths']
    assert new['collision_recipe_sha256'] == old['collision_recipe_sha256']
    if seed == 0:
        assert new['metadata'] == old['metadata'] == metadata
        assert new['stage'].GetRootLayer().ExportToString() == old['stage'].GetRootLayer().ExportToString()
        assert new['actual_body_poses_m'] == old['actual_body_poses_m']
        return
    receipts = new['metadata']['seed_variant']['reset_support_receipts']
    assert set(receipts) == {'carrot_0', 'carrot_1', 'carrot_2'}
    normalized = deepcopy(new['metadata'])
    normalized['seed_variant'] = old['metadata']['seed_variant']
    previous_rows = {row['id']: row for row in old['metadata']['objects']}
    for row in normalized['objects']:
        name = row['id']; previous = previous_rows[name]
        before, after = [data['actual_body_poses_m'][name] for data in (old, new)]
        if name in receipts:
            assert after['position_m'][:2] == before['position_m'][:2]
            assert after['quaternion_wxyz'] == before['quaternion_wxyz']
            assert before['position_m'][2]-after['position_m'][2] == pytest.approx(.001, abs=2e-9)
            # Use the actual stage again, not the recorded Z receipt.
            board, _ = subject._world_collision_vertices(new['stage'], metadata['fixtures']['source_board'])
            points, _ = subject._world_collision_vertices(new['stage'], row['body_path'])
            assert abs(float(points[:, 2].min()-board[:, 2].max())) < 1e-10
            assert receipts[name]['support_prim_path'] == metadata['fixtures']['source_board']
            assert receipts[name]['support_collision_paths']
            assert receipts[name]['actor_collision_paths'] == row['collider_paths']
            row['initial_pose_xyzw'][2] = previous['initial_pose_xyzw'][2]
            # Restore only the changed transform in the anonymous result.
            # Exact whole-layer equality then checks every mesh/physics/material
            # attribute and all untouched object and fixture transforms.
            prim = new['stage'].GetPrimAtPath(row['body_path'])
            old_prim = old['stage'].GetPrimAtPath(row['body_path'])
            UsdGeom.Xformable(prim).GetOrderedXformOps()[0].Set(
                UsdGeom.Xformable(old_prim).GetOrderedXformOps()[0].Get())
        else:
            assert after == before
    assert normalized == old['metadata']
    assert new['stage'].GetRootLayer().ExportToString() == old['stage'].GetRootLayer().ExportToString()


def test_support_height_uses_actual_mesh_not_parent_height_or_fixed_delta(parent):
    metadata, stage = parent[4:]
    moved = Usd.Stage.Open(stage.Flatten())
    fixture = moved.GetPrimAtPath(metadata['fixtures']['source_board'])
    transform = UsdGeom.Xformable(fixture).GetLocalTransformation()
    position = transform.ExtractTranslation()
    transform.SetTranslateOnly(Gf.Vec3d(position[0], position[1], position[2]+.004))
    UsdGeom.Xformable(fixture).ClearXformOpOrder()
    UsdGeom.Xformable(fixture).AddTransformOp(opSuffix='testSupport').Set(transform)
    plan = subject.case_plan('produce_transfer', 1, metadata, profile=subject.CONTACT_RESET_VERSION)
    actual = subject.derive_stage(moved, metadata, plan)
    canonical = subject.derive_stage(stage, metadata, plan)
    for name in plan['reset_profile']['actors']:
        assert actual['actual_body_poses_m'][name]['position_m'][2] == pytest.approx(
            canonical['actual_body_poses_m'][name]['position_m'][2]+.004, abs=1e-12)


def test_dynamic_source_board_is_not_a_static_reset_root(parent):
    metadata, stage = parent[4:]
    candidate = Usd.Stage.Open(stage.Flatten())
    fixture = candidate.GetPrimAtPath(metadata['fixtures']['source_board'])
    UsdPhysics.RigidBodyAPI.Apply(fixture).CreateRigidBodyEnabledAttr(True)
    plan = subject.case_plan('produce_transfer', 1, metadata, profile=subject.CONTACT_RESET_VERSION)
    with pytest.raises(ValueError, match='fixed support fixture'):
        subject.derive_stage(candidate, metadata, plan)


@pytest.mark.parametrize('field,value', [('clearance_m', .00001), ('support_id', 'carrot_tray'),
                                       ('actors', ['tomato_0'])])
def test_modified_contact_recipe_is_rejected_even_if_self_rehashed(parent, field, value):
    metadata, stage = parent[4:]
    plan = subject.case_plan('produce_transfer', 1, metadata, profile=subject.CONTACT_RESET_VERSION)
    plan['reset_profile'][field] = value
    plan.pop('plan_sha256')
    plan['plan_sha256'] = subject.canonical_hash(plan)
    with pytest.raises(ValueError, match='finite generator'):
        subject.derive_stage(stage, metadata, plan)


@pytest.mark.parametrize('profile', ['unknown', '', None])
def test_unknown_profile_is_rejected(parent, profile):
    with pytest.raises(ValueError, match='profile'):
        subject.case_plan('produce_transfer', 1, parent[4], profile=profile)


@pytest.mark.parametrize('task_id', ['utensil_sorting', 'pantry_stocking', 'table_clearing'])
def test_contact_profile_is_not_available_to_other_kitchen_tasks(task_id):
    manifest = load_json(PUBLICATION/task_id/'single_arm/release_manifest.json')
    metadata = subject._parent(Path(manifest['base_release']['path']))[3]
    with pytest.raises(ValueError, match='profile'):
        subject.case_plan(task_id, 1, metadata, profile=subject.CONTACT_RESET_VERSION)


@pytest.mark.parametrize('seed,profile', [(0, None), (1, None), (1, subject.VERSION)])
def test_prepare_and_compose_explicit_profile_with_unchanged_public_projection(tmp_path, monkeypatch, parent, seed, profile):
    from scripts.maintenance import freeze_project_sources
    monkeypatch.setattr(subject, 'ASSET_ROOT', tmp_path)
    monkeypatch.setattr(freeze_project_sources, 'verify_if_snapshot', lambda *a, **k: {'verified': True, 'test_only': True})
    root, scene, task, asset, metadata, _ = parent
    output = tmp_path/'candidate'
    path = subject.prepare_asset(root, seed, output, profile=profile)
    manifest = load_json(path)
    expected = subject.SUPPORT_LOW_VERTEX_TIP_VERSION if profile is None else profile
    assert manifest['generator_version'] == expected
    assert manifest['case_plan'] == subject.case_plan('produce_transfer', seed, metadata, profile=expected)
    composed = subject.compose_case(root, path, 'single_arm')
    if seed == 0:
        assert (output/'task.usda').read_bytes() == Path(asset['usd_path']).read_bytes()
        assert (output/'metadata.json').read_bytes() == Path(asset['metadata_path']).read_bytes()
        assert load_json(output/'metadata.json') == metadata
    assert composed['scene']['robot'] == scene['robot']
    assert composed['scene']['workroom'] == scene['workroom']
    assert composed['scene']['isaacsim'] == scene['isaacsim']
    assert composed['initial_state'] == task['initial_state']
    assert composed['task']['public_rules'] == task['public_rules']
    public = json.dumps(public_task_projection(composed['task']), sort_keys=True)
    assert 'reset_support_receipts' not in public and 'reset_profile' not in public
    assert 'collision_bottom_before_world_z_m' not in public
    manifest['generator_version'] = 'unknown'
    subject._dump(path, manifest)
    with pytest.raises(ValueError, match='parent/task/generator'):
        subject.compose_case(root, path, 'single_arm')
