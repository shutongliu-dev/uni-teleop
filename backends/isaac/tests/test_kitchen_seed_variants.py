"""Actual selected kitchen USD regressions; no native simulation or model calls."""
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
from pxr import Gf, Usd, UsdGeom, UsdPhysics
import pytest

from bench.paths import load_json
from scenes.workrooms.catalog import list_tasks
from scenes.workrooms.kitchen import seed_variants as subject
from scenes.workrooms.living_room import seed_geometry as geometry


@pytest.fixture(scope='module')
def parents():
    result = {}
    roots = {}
    # Both robot profiles must unwrap to the same kitchen parent: the L-mount
    # profile revises only the robot of the original profile's seed release.
    for profile in ('original', 'l_mount'):
        for row in list_tasks('kitchen', robot_profile=profile):
            if row['task_id'] not in subject.TASK_IDS:
                continue
            root = Path(row['source_release']['path'])
            assert subject.digest(root/'release_manifest.json') == row['source_release']['manifest_sha256']
            manifest = load_json(root/'release_manifest.json')
            for wrapper in ('workroom_robot_revision_release.v1', 'workroom_randomized_release.v1'):
                if manifest.get('schema') == wrapper:
                    parent = manifest['base_release']
                    root = Path(parent['path'])
                    assert subject.digest(root/'release_manifest.json') == parent['manifest_sha256']
                    manifest = load_json(root/'release_manifest.json')
            assert roots.setdefault(row['task_id'], root) == root
    for task_id, root in roots.items():
        scene, task, descriptor, metadata, _ = subject._parent(root)
        result[task_id] = (root, scene, task, descriptor, metadata,
                                  Usd.Stage.Open(descriptor['usd_path']))
    assert set(result) == set(subject.TASK_IDS)
    return result


@pytest.mark.parametrize('task_id', subject.TASK_IDS)
def test_finite_cases_real_geometry_inventory_and_capacity(parents, task_id):
    _, _, _, _, metadata, stage = parents[task_id]
    original = stage.GetRootLayer().ExportToString()
    identities = set()
    for seed in range(10):
        plan = subject.case_plan(task_id, seed, metadata)
        assert plan == subject.case_plan(task_id, seed, deepcopy(metadata))
        identities.add(plan['plan_sha256'])
        result = subject.derive_stage(stage, metadata, plan)
        assert set(result['actual_body_poses_m']) == set(metadata['actors'])
        assert {r['object_id'] for r in result['static_geometry']['bodies']} == set(metadata['actors'])
        alternatives = {'table_clearing': 4, 'pantry_stocking': 4}.get(task_id, 0)
        assert len(result['interaction_targets']) == 2*len(metadata['actors'])+alternatives
        assert all(row['geometry_contained'] for row in result['structural_checks'])
        assert all(row['separating_axis_gap_m'] >= 0 for row in result['footprint_separations'])
        assert result['metadata']['fixtures'] == metadata['fixtures']
        assert result['metadata']['goals'] == metadata['goals']
        assert result['metadata']['fixture_metadata'] == metadata['fixture_metadata']
        assert result['metadata']['physics_prior'] == metadata['physics_prior']
        if seed == 0:
            assert result['metadata'] == metadata
        else:
            assert result['metadata']['objects'][0]['initial_pose_xyzw'] != metadata['objects'][0]['initial_pose_xyzw']
            assert result['metadata']['objects'][0]['collision_bounds_m'] != metadata['objects'][0]['collision_bounds_m']
    assert len(identities) == 10
    assert stage.GetRootLayer().ExportToString() == original


def _tensor(stage, path):
    api = UsdPhysics.MassAPI(stage.GetPrimAtPath(path))
    rotation = geometry.rotation(api.GetPrincipalAxesAttr().Get())
    return float(api.GetMassAttr().Get()), np.asarray(api.GetCenterOfMassAttr().Get()), rotation @ np.diag(api.GetDiagonalInertiaAttr().Get()) @ rotation.T


@pytest.mark.parametrize('task_id', subject.TASK_IDS)
def test_affine_mass_matches_original_tensor_and_world_bounds(parents, task_id):
    _, _, _, _, metadata, stage = parents[task_id]
    plan = subject.case_plan(task_id, 2, metadata)
    result = subject.derive_stage(stage, metadata, plan)
    for row in result['metadata']['objects']:
        path = row['body_path']
        scale = np.array(plan['bodies'][row['id']]['scale_xyz'])
        mass, com, tensor = _tensor(stage, path)
        a = np.diag(scale)
        second = np.linalg.det(a)*a @ (.5*np.trace(tensor)*np.eye(3)-tensor) @ a.T
        actual_mass, actual_com, actual_tensor = _tensor(result['stage'], path)
        assert actual_mass == pytest.approx(mass*np.linalg.det(a), rel=1e-7)
        np.testing.assert_allclose(actual_com, a@com, atol=1e-8)
        expected_tensor = np.trace(second)*np.eye(3)-second
        # Principal axes are stored as Quatf; judge the tensor norm, not the
        # relative error of near-zero off-diagonals in symmetric objects.
        assert np.linalg.norm(actual_tensor-expected_tensor) <= 2e-6*np.linalg.norm(expected_tensor)
        original_points = subject._body_points(stage, path, collision_only=True)
        new_points = subject._body_points(result['stage'], path, collision_only=True)
        np.testing.assert_allclose(new_points, original_points*scale, atol=2e-8)
        assert row['mass_kg'] == actual_mass
        np.testing.assert_allclose(np.array(row['collision_bounds_m']), [new_points.min(0),new_points.max(0)], atol=1e-10)


@pytest.mark.parametrize('task_id', ('utensil_sorting', 'table_clearing'))
def test_sdf_bowl_conditioned_convex_and_thick_grip_retained(parents, task_id):
    _, _, _, _, metadata, stage = parents[task_id]
    result = subject.derive_stage(stage, metadata, subject.case_plan(task_id, 1, metadata))
    rows = {row['id']: row for row in result['metadata']['objects']}
    for old in metadata['objects']:
        if old['role'] not in ('spoon', 'fork', 'knife'):
            continue
        row = rows[old['id']]
        assert row['collider_paths'] == old['collider_paths']
        assert row['handle_grip']['handle_thickness_m'] == old['handle_grip']['handle_thickness_m'] == pytest.approx(.0138)
        if old['role'] == 'spoon':
            path = old['body_path']+'/Collision/spoon_bowl_sdf'
            before, after = stage.GetPrimAtPath(path), result['stage'].GetPrimAtPath(path)
            assert after.GetAppliedSchemas() == before.GetAppliedSchemas()
            assert after.GetAttribute('physxSDFMeshCollision:sdfResolution').Get() == before.GetAttribute('physxSDFMeshCollision:sdfResolution').Get()
            assert UsdPhysics.MeshCollisionAPI(after).GetApproximationAttr().Get() == UsdPhysics.MeshCollisionAPI(before).GetApproximationAttr().Get()
        # The disabled historical collider is not silently enabled by baking.
        for prim in Usd.PrimRange(stage.GetPrimAtPath(old['body_path'])):
            if prim.HasAPI(UsdPhysics.CollisionAPI):
                other = result['stage'].GetPrimAtPath(prim.GetPath())
                assert UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get() == UsdPhysics.CollisionAPI(other).GetCollisionEnabledAttr().Get()


@pytest.mark.parametrize('seed', (True, -1, 10, 1.5, '1'))
def test_invalid_seed_rejected(parents, seed):
    with pytest.raises(ValueError, match='seed'):
        subject.case_plan('produce_transfer', seed, parents['produce_transfer'][4])


def test_tampered_plan_or_old_cutlery_rejected(parents):
    metadata, stage = parents['utensil_sorting'][4:]
    plan = subject.case_plan('utensil_sorting', 1, metadata)
    plan['bodies']['spoon_0']['scale_xyz'][2] = .8
    with pytest.raises(ValueError, match='finite generator'):
        subject.derive_stage(stage, metadata, plan)
    old = deepcopy(metadata)
    old['utensil_grip_profile'] = 'thin'
    with pytest.raises(ValueError, match='thick handles'):
        subject.case_plan('utensil_sorting', 1, old)


def test_goal_capacity_is_not_assumed(parents):
    metadata = deepcopy(parents['produce_transfer'][4])
    metadata['fixture_metadata']['tomato_tray']['interior_bounds_xy_m'] = [[-.001,-.001],[.001,.001]]
    with pytest.raises(ValueError, match='capacity'):
        subject.static_checks(metadata)


def test_initial_overlap_is_rejected(parents):
    metadata = deepcopy(parents['produce_transfer'][4])
    metadata['objects'][1]['initial_pose_xyzw'] = metadata['objects'][0]['initial_pose_xyzw']
    with pytest.raises(ValueError, match='Overlapping initial'):
        subject.footprint_separations(metadata)


@pytest.mark.parametrize('task_id,seed', (('utensil_sorting', 0), ('produce_transfer', 2)))
def test_published_asset_and_composition_bind_parent(tmp_path, monkeypatch, parents, task_id, seed):
    root, original_scene, original_task, descriptor, metadata, _ = parents[task_id]
    from scripts.maintenance import freeze_project_sources
    monkeypatch.setattr(subject, 'ASSET_ROOT', tmp_path)
    monkeypatch.setattr(freeze_project_sources, 'verify_if_snapshot', lambda *a, **kw: {'verified': True, 'test_only': True})
    output = tmp_path/'access02'/task_id
    path = subject.prepare_asset(root, seed, output)
    result = subject.compose_case(root, path, 'single_arm')
    assert {r['reference_state'] for r in result['interaction_targets']} == {'initial'}
    assert {r['type'] for r in result['interaction_targets']} == {'actual_surface', 'goal'}
    assert result['fixed_support_paths']
    assert all(path.startswith(tuple(p+'/' for p in metadata['fixtures'].values()))
               for path in result['fixed_support_paths'])
    if seed == 0:
        assert (output/'task.usda').read_bytes() == Path(descriptor['usd_path']).read_bytes()
        assert result['task']['asset_metadata'] == original_task['asset_metadata']
    else:
        assert (output/'surface_finish.png').is_file()
        assert 'CC0' in (output/'TEXTURE_LICENSE.txt').read_text()
        variant = Usd.Stage.Open(str(output/'task.usda'))
        assert variant.GetPrimAtPath(subject.ROOT+'/SeedVariantMaterials')
        assert not variant.GetPrimAtPath('/World/SeedVariantMaterials')
        paths = [p for row in metadata['objects'] for p in row['collider_paths']]
        assert subject._collision_recipe(variant, paths) == subject._collision_recipe(parents[task_id][5], paths)
    assert result['scene']['robot'] == original_scene['robot']
    assert result['scene']['workroom'] == original_scene['workroom']
    assert result['scene']['isaacsim'] == original_scene['isaacsim']
    assert result['initial_state'] == original_task['initial_state']
    assert result['task']['evaluation']['thresholds'] == original_task['evaluation']['thresholds']
    assert result['task']['public_rules'] == original_task['public_rules']
    assert result['task']['episode_seed'] == seed
    assert result['task']['evaluation']['metadata'] == result['task']['asset_metadata']
    for name, text in result['evaluator_sources'].items():
        assert (root/'evaluator'/name).read_text() == text
    with pytest.raises(ValueError, match='single-arm'):
        subject.compose_case(root, path, 'dual_opposed')
    manifest = load_json(path)
    manifest['parent_references'] = {}
    subject._dump(path, manifest)
    with pytest.raises(ValueError, match='parent/task/generator'):
        subject.compose_case(root, path, 'single_arm')
