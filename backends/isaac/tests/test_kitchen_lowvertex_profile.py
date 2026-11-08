"""Explicit 33-point material solids; no native cooking/stability assertions."""
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from pxr import Sdf, Usd, UsdGeom, UsdPhysics
import pytest
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation

from bench.isaacsim.release import validate
from bench.paths import DATA_ROOT, load_json
from bench.runtime.worker import public_task_projection
from scenes.workrooms.kitchen import produce_support_hull as hull
from scenes.workrooms.kitchen import seed_variants as subject

PUBLICATION = Path(f'{DATA_ROOT}/datasets/fr3_sharpa_bench/isaacsim/releases/'
                   'workroom_initial_randomization_v1/publication01/releases')


@pytest.fixture(scope='module')
def parent():
    manifest = load_json(PUBLICATION/'produce_transfer/single_arm/release_manifest.json')
    root = Path(manifest['base_release']['path'])
    scene, task, asset, metadata, _ = subject._parent(root)
    return root, scene, task, asset, metadata, Usd.Stage.Open(asset['usd_path'])


@pytest.mark.parametrize('seed', range(10))
def test_final_float32_solid_mass_and_worst_case_hull_envelope(parent, seed):
    metadata, original = parent[4:]
    plan = subject.case_plan('produce_transfer', seed, metadata, profile=subject.SUPPORT_LOW_VERTEX_VERSION)
    old_plan = subject.case_plan('produce_transfer', seed, metadata, profile=subject.CONTACT_RESET_VERSION)
    assert plan['bodies'] == old_plan['bodies']
    assert plan['geometry_profile']['parent_axial_ring_indices'] == [0, 2, 4, 5]
    baseline = subject.derive_stage(original, metadata, old_plan)
    result = subject.derive_stage(original, metadata, plan)
    assert result['collision_recipe_sha256'] == baseline['collision_recipe_sha256']
    assert result['fixed_support_paths'] == baseline['fixed_support_paths']
    assert all(row['geometry_contained'] for row in result['structural_checks'])
    assert all(row['separating_axis_gap_m'] >= 0 for row in result['footprint_separations'])
    if not seed:
        assert result['metadata'] == metadata
        assert result['stage'].GetRootLayer().ExportToString() == baseline['stage'].GetRootLayer().ExportToString()
        return
    previous = {r['id']: r for r in baseline['metadata']['objects']}
    for row in result['metadata']['objects']:
        if row['category'] != 'carrot':
            assert row == previous[row['id']]
            continue
        body = result['stage'].GetPrimAtPath(row['body_path'])
        physical, polygons = hull._mesh(result['stage'], row['collider_paths'][0], body)
        visual, visual_polygons = hull._mesh(result['stage'], row['body_path']+'/Visual/body', body)
        np.testing.assert_array_equal(physical.vertices, visual.vertices)
        assert polygons == visual_polygons
        assert len(physical.vertices) == len(polygons) == 33
        assert sorted(map(len, polygons)) == [3]*8+[4]*24+[8]
        assert len(physical.faces) == 62
        assert len(ConvexHull(physical.vertices).simplices) <= 62
        np.testing.assert_array_equal(UsdGeom.Xformable(result['stage'].GetPrimAtPath(
            row['collider_paths'][0])).GetLocalTransformation(), np.eye(4))
        source_body = original.GetPrimAtPath(row['body_path'])
        source, _ = hull._mesh(original, row['collider_paths'][0], source_body)
        source_mass = float(UsdPhysics.MassAPI(source_body).GetMassAttr().Get())
        api = UsdPhysics.MassAPI(body)
        assert float(api.GetMassAttr().Get()) == pytest.approx(source_mass*physical.volume/source.volume, rel=1e-7)
        np.testing.assert_allclose(api.GetCenterOfMassAttr().Get(), physical.center_mass, rtol=0, atol=2e-9)
        q = api.GetPrincipalAxesAttr().Get()
        rotation = Rotation.from_quat([*q.GetImaginary(), q.GetReal()]).as_matrix()
        actual_tensor = rotation@np.diag(api.GetDiagonalInertiaAttr().Get())@rotation.T
        physical.density = source_mass/source.volume
        np.testing.assert_allclose(actual_tensor, physical.moment_inertia, rtol=1e-5, atol=2e-11)
        bottom = physical.vertices[physical.vertices[:, 2] == 0.]
        assert len(bottom) == 4
        support = ConvexHull(bottom[:, :2])
        margin = -(support.equations[:, :2]@physical.center_mass[:2]+support.equations[:, 2]).max()
        assert .0062 < margin < .0067
        profile = row['support_hull_profile']
        assert profile['profile'] == subject.SUPPORT_LOW_VERTEX_VERSION
        assert profile['parent_axial_ring_indices'] == [0, 2, 4, 5]
        assert profile['source_triangulated_faces'] == 62
        assert profile['collision_mesh_coordinate_scale'] == 1.
        assert profile['native_cooked_geometry_verified'] is False
        assert row['initial_pose_xyzw'][:2] == previous[row['id']]['initial_pose_xyzw'][:2]
        assert row['initial_pose_xyzw'][3:] == previous[row['id']]['initial_pose_xyzw'][3:]
        np.testing.assert_array_equal(row['collision_support_points_m'], physical.vertices)
        visible_points = subject._body_points(result['stage'], row['body_path'])
        assert visible_points[:, 2].min() == 0.
        np.testing.assert_array_equal(row['local_bounds_m'], [visible_points.min(0), visible_points.max(0)])
        # Exact rest Z is derived from the actual fixed support, not a constant.
        receipt = result['metadata']['seed_variant']['reset_support_receipts'][row['id']]
        assert receipt['clearance_m'] == 0.
        Sdf.CopySpec(baseline['stage'].GetRootLayer(), row['body_path'],
                     result['stage'].GetRootLayer(), row['body_path'])
    assert result['stage'].GetRootLayer().ExportToString() == baseline['stage'].GetRootLayer().ExportToString()


def test_new_details_follow_selected_four_ring_solid_and_rigid_stem_transform(parent):
    import trimesh
    metadata, original = parent[4:]
    plan = subject.case_plan('produce_transfer', 1, metadata, profile=subject.SUPPORT_LOW_VERTEX_VERSION)
    result = subject.derive_stage(original, metadata, plan)
    row = next(r for r in result['metadata']['objects'] if r['id'] == 'carrot_0')
    transform = np.asarray(row['support_hull_profile']['canonical_extra_rest_transform'])
    scale = np.asarray(plan['bodies']['carrot_0']['scale_xyz'])
    physical, _ = hull._mesh(result['stage'], row['collider_paths'][0], result['stage'].GetPrimAtPath(row['body_path']))
    halfspaces = ConvexHull(physical.vertices).equations
    for i in range(3):
        path = row['body_path']+f'/Visual/stem_{i}'
        before = np.asarray(UsdGeom.Mesh(original.GetPrimAtPath(path)).GetPointsAttr().Get(), float)
        after = np.asarray(UsdGeom.Mesh(result['stage'].GetPrimAtPath(path)).GetPointsAttr().Get(), float)
        expected = (trimesh.transform_points(before, transform).astype(np.float32).astype(float)*scale).astype(np.float32)
        np.testing.assert_array_equal(after, expected)
    for i in range(5):
        prim = result['stage'].GetPrimAtPath(row['body_path']+f'/Visual/root_ring_{i}')
        assert not prim.HasAPI(UsdPhysics.CollisionAPI)
        points = np.asarray(UsdGeom.Mesh(prim).GetPointsAttr().Get(), float)
        signed = (points@halfspaces[:, :3].T+halfspaces[:, 3]).max(1)
        assert signed.min() < 0 < signed.max() < .0008


@pytest.mark.parametrize('field,value', [('radial_segments', 10), ('parent_axial_ring_indices', [0, 1, 4, 5]),
                                       ('collision_mesh_coordinate_scale', 1000.)])
def test_self_rehashed_other_recipe_rejected(parent, field, value):
    metadata, original = parent[4:]
    plan = subject.case_plan('produce_transfer', 1, metadata, profile=subject.SUPPORT_LOW_VERTEX_VERSION)
    plan['geometry_profile'][field] = value
    plan.pop('plan_sha256'); plan['plan_sha256'] = subject.canonical_hash(plan)
    with pytest.raises(ValueError, match='finite generator'):
        subject.derive_stage(original, metadata, plan)


@pytest.mark.parametrize('task_id', subject.TASK_IDS[1:])
def test_profile_rejects_other_kitchen_tasks(task_id):
    manifest = load_json(PUBLICATION/task_id/'single_arm/release_manifest.json')
    metadata = subject._parent(Path(manifest['base_release']['path']))[3]
    with pytest.raises(ValueError, match='profile'):
        subject.case_plan(task_id, 1, metadata, profile=subject.SUPPORT_LOW_VERTEX_VERSION)


def test_v3_current_frozen_release_all_cases_still_strictly_recompose():
    root = PUBLICATION.parents[1]/'convex03/releases/produce_transfer/single_arm'
    manifest = validate(root)
    assert len(manifest['cases']) == 10
    assert all(load_json(Path(row['asset_manifest']))['generator_version'] == subject.SUPPORT_HULL_VERSION
               for row in manifest['cases'].values())


@pytest.mark.parametrize('seed', [0, 1])
def test_fresh_explicit_v4_publish_and_prekit_contract(tmp_path, monkeypatch, parent, seed):
    from scripts.maintenance import freeze_project_sources
    required = []
    def verify(*a, **kw):
        required.append(kw['required_modules']); return {'verified': True, 'test_only': True}
    monkeypatch.setattr(subject, 'ASSET_ROOT', tmp_path)
    monkeypatch.setattr(freeze_project_sources, 'verify_if_snapshot', verify)
    root, scene, task, asset, metadata, _ = parent
    output = tmp_path/'lowpoly04'/f'seed_{seed:02d}'
    path = subject.prepare_asset(root, seed, output, profile=subject.SUPPORT_LOW_VERTEX_VERSION)
    assert 'scenes.workrooms.kitchen.produce_support_hull' in required[0]
    assert load_json(path)['generator_version'] == subject.SUPPORT_LOW_VERTEX_VERSION
    if not seed:
        assert (output/'task.usda').read_bytes() == Path(asset['usd_path']).read_bytes()
        assert (output/'metadata.json').read_bytes() == Path(asset['metadata_path']).read_bytes()
    result = subject.compose_case(root, path, 'single_arm')
    for key in ('robot', 'workroom', 'isaacsim'):
        assert result['scene'][key] == scene[key]
    assert result['initial_state'] == task['initial_state']
    assert result['task']['public_rules'] == task['public_rules']
    public = json.dumps(public_task_projection(result['task']))
    assert 'support_hull_profile' not in public and 'reset_support_receipts' not in public
    script = '''import builtins,sys
original=builtins.__import__
def guarded(name,*args,**kw):
    if name=='pxr' or name.startswith('pxr.'): raise RuntimeError('pre-Kit USD import')
    return original(name,*args,**kw)
builtins.__import__=guarded
from pathlib import Path
from scenes.workrooms.kitchen.seed_variants import compose_case
assert compose_case(Path(sys.argv[1]),Path(sys.argv[2]),'single_arm')['task']['id']=='produce_transfer'
'''
    done = subprocess.run([sys.executable, '-B', '-c', script, str(root), str(path)],
                          capture_output=True, text=True, timeout=60)
    assert done.returncode == 0, done.stdout+done.stderr
