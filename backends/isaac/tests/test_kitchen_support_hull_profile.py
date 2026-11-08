"""Actual closed solids, compatibility and pre-Kit consumer boundaries."""
from copy import deepcopy
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
def test_actual_same_visible_and_collision_polygons_density_support_and_other_bodies(parent, seed):
    metadata, original = parent[4:]
    before = original.GetRootLayer().ExportToString()
    old = subject.derive_stage(original, metadata, subject.case_plan('produce_transfer', seed, metadata,
                               profile=subject.CONTACT_RESET_VERSION))
    plan = subject.case_plan('produce_transfer', seed, metadata, profile=subject.SUPPORT_HULL_VERSION)
    assert plan['bodies'] == subject.case_plan('produce_transfer', seed, metadata)['bodies']
    result = subject.derive_stage(original, metadata, plan)
    assert original.GetRootLayer().ExportToString() == before
    assert result['collision_recipe_sha256'] == old['collision_recipe_sha256']
    assert result['fixed_support_paths'] == old['fixed_support_paths']
    assert len(result['interaction_targets']) == 12
    assert all(row['geometry_contained'] for row in result['structural_checks'])
    assert all(row['separating_axis_gap_m'] >= 0 for row in result['footprint_separations'])
    for key in ('fixtures', 'fixture_metadata', 'goals', 'physics_prior', 'source'):
        assert result['metadata'][key] == old['metadata'][key]
    if not seed:
        assert result['metadata'] == metadata
        assert result['stage'].GetRootLayer().ExportToString() == old['stage'].GetRootLayer().ExportToString()
        return
    previous = {r['id']: r for r in old['metadata']['objects']}
    for row in result['metadata']['objects']:
        path = row['body_path']; body = result['stage'].GetPrimAtPath(path)
        if row['category'] != 'carrot':
            assert row == previous[row['id']]
            continue
        physical, polygons = hull._mesh(result['stage'], row['collider_paths'][0], body)
        visual, visual_polygons = hull._mesh(result['stage'], path+'/Visual/body', body)
        np.testing.assert_array_equal(physical.vertices, visual.vertices)
        assert polygons == visual_polygons
        assert len(physical.vertices) == len(polygons) == 61
        assert sorted(map(len, polygons)) == [3]*10+[4]*50+[10]
        for face in polygons:
            points = physical.vertices[face]
            # Fit the entire float32 polygon. The first three neighboring
            # vertices of the small ten-sided end cap amplify quantization
            # into a misleading plane tilt across its opposite edge.
            center = points.mean(0)
            normal = np.linalg.svd(points-center)[2][-1]
            if normal@(center-physical.center_mass) < 0:
                normal = -normal
            assert np.abs((points-center)@normal).max() < 1e-8
            assert ((physical.vertices-center)@normal).max() < 1e-8
        source_body = original.GetPrimAtPath(path)
        source, _ = hull._mesh(original, row['collider_paths'][0], source_body)
        source_mass = float(UsdPhysics.MassAPI(source_body).GetMassAttr().Get())
        density = source_mass/source.volume
        api = UsdPhysics.MassAPI(body)
        mass = float(api.GetMassAttr().Get())
        assert mass == pytest.approx(density*physical.volume, rel=1e-7)
        np.testing.assert_allclose(api.GetCenterOfMassAttr().Get(), physical.center_mass, rtol=0, atol=2e-9)
        q = api.GetPrincipalAxesAttr().Get()
        rotation = Rotation.from_quat([*q.GetImaginary(), q.GetReal()]).as_matrix()
        tensor = rotation@np.diag(api.GetDiagonalInertiaAttr().Get())@rotation.T
        expected_tensor = physical.moment_inertia*density
        assert np.linalg.norm(tensor-expected_tensor)/np.linalg.norm(expected_tensor) < 2e-6
        assert row['mass_kg'] == mass
        assert row['collision_volume_m3'] == physical.volume
        np.testing.assert_array_equal(row['collision_support_points_m'], physical.vertices)
        np.testing.assert_array_equal(row['collision_bounds_m'], physical.bounds)
        bottom = physical.vertices[physical.vertices[:, 2] == 0.]
        assert len(bottom) == 4
        polygon = ConvexHull(bottom[:, :2])
        margin = -(polygon.equations[:, :2]@physical.center_mass[:2]+polygon.equations[:, 2]).max()
        assert .0039 < margin < .0044
        profile = row['support_hull_profile']
        assert profile['integrated_com_support_margin_m'] == pytest.approx(margin, abs=1e-12)
        assert profile['parent_recipe_reconstruction_max_error_m'] < 2e-8
        np.testing.assert_array_equal(profile['support_vertices_body_m'], bottom)
        assert profile['native_cooked_geometry_verified'] is False
        assert row['initial_pose_xyzw'][:2] == previous[row['id']]['initial_pose_xyzw'][:2]
        assert row['initial_pose_xyzw'][3:] == previous[row['id']]['initial_pose_xyzw'][3:]
        visible_points = subject._body_points(result['stage'], path)
        assert visible_points[:, 2].min() == 0
        np.testing.assert_array_equal(row['local_bounds_m'], [visible_points.min(0), visible_points.max(0)])
        for prim in Usd.PrimRange(result['stage'].GetPrimAtPath(path+'/Visual')):
            assert not prim.HasAPI(UsdPhysics.CollisionAPI)
            if prim.IsA(UsdGeom.Mesh):
                mesh = UsdGeom.Mesh(prim)
                assert len(mesh.GetNormalsAttr().Get()) == sum(mesh.GetFaceVertexCountsAttr().Get())
        # After replacing exactly the three intentionally redesigned bodies,
        # whole-layer equality protects every other object, fixture/material,
        # transform and authored physics parameter.
        Sdf.CopySpec(old['stage'].GetRootLayer(), path, result['stage'].GetRootLayer(), path)
    assert result['stage'].GetRootLayer().ExportToString() == old['stage'].GetRootLayer().ExportToString()


def test_stems_rigid_transform_and_root_ring_relief_is_rebuilt(parent):
    metadata, original = parent[4:]
    import trimesh
    plan = subject.case_plan('produce_transfer', 1, metadata, profile=subject.SUPPORT_HULL_VERSION)
    result = subject.derive_stage(original, metadata, plan)
    row = next(r for r in result['metadata']['objects'] if r['id'] == 'carrot_0')
    transform = np.array(row['support_hull_profile']['canonical_extra_rest_transform'])
    scale = np.array(plan['bodies']['carrot_0']['scale_xyz'])
    for i in range(3):
        path = row['body_path']+f'/Visual/stem_{i}'
        old = np.asarray(UsdGeom.Mesh(original.GetPrimAtPath(path)).GetPointsAttr().Get(), float)
        actual = np.asarray(UsdGeom.Mesh(result['stage'].GetPrimAtPath(path)).GetPointsAttr().Get(), float)
        expected = (trimesh.transform_points(old, transform).astype(np.float32).astype(float)*scale).astype(np.float32)
        np.testing.assert_array_equal(actual, expected)
    for i in range(5):
        path = row['body_path']+f'/Visual/root_ring_{i}'
        old = np.asarray(UsdGeom.Mesh(original.GetPrimAtPath(path)).GetPointsAttr().Get(), float)
        actual = np.asarray(UsdGeom.Mesh(result['stage'].GetPrimAtPath(path)).GetPointsAttr().Get(), float)
        rigid_only = trimesh.transform_points(old, transform)*scale
        assert np.linalg.norm(actual-rigid_only, axis=1).max() > .0001


def test_changed_selected_parent_shape_is_rejected(parent):
    metadata, original = parent[4:]
    stage = Usd.Stage.Open(original.Flatten())
    from pxr import Vt
    for part in ('Visual', 'Collision'):
        usd = UsdGeom.Mesh(stage.GetPrimAtPath('/World/KitchenTask/Actors/carrot_0/'+part+'/body'))
        points = np.asarray(usd.GetPointsAttr().Get(), dtype=np.float32).copy(); points[:, 0] *= 1.01
        usd.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(points))
    with pytest.raises(ValueError, match='reconstructed 64-sector'):
        subject.derive_stage(stage, metadata, subject.case_plan('produce_transfer', 1, metadata,
                             profile=subject.SUPPORT_HULL_VERSION))


@pytest.mark.parametrize('field,value', [('radial_segments', 12), ('source_radial_segments', 32),
                                       ('parent_recipe_reconstruction_required', False)])
def test_self_rehashed_arbitrary_geometry_recipe_is_rejected(parent, field, value):
    metadata, stage = parent[4:]
    plan = subject.case_plan('produce_transfer', 1, metadata, profile=subject.SUPPORT_HULL_VERSION)
    plan['geometry_profile'][field] = value
    plan.pop('plan_sha256'); plan['plan_sha256'] = subject.canonical_hash(plan)
    with pytest.raises(ValueError, match='finite generator'):
        subject.derive_stage(stage, metadata, plan)


@pytest.mark.parametrize('task_id', subject.TASK_IDS[1:])
def test_support_hull_profile_rejects_other_tasks(task_id):
    manifest = load_json(PUBLICATION/task_id/'single_arm/release_manifest.json')
    metadata = subject._parent(Path(manifest['base_release']['path']))[3]
    with pytest.raises(ValueError, match='profile'):
        subject.case_plan(task_id, 1, metadata, profile=subject.SUPPORT_HULL_VERSION)


def test_original_contact_v2_release_all_ten_cases_still_strictly_recompose():
    root = PUBLICATION.parents[1]/'support02/releases/produce_transfer/single_arm'
    manifest = validate(root)
    for record in manifest['cases'].values():
        asset = load_json(Path(record['asset_manifest']))
        assert asset['generator_version'] == subject.CONTACT_RESET_VERSION


@pytest.mark.parametrize('seed', (0, 1, 2))
def test_fresh_v3_publish_compose_source_dependency_and_prekit(tmp_path, monkeypatch, parent, seed):
    from scripts.maintenance import freeze_project_sources
    verified = []
    def verify(*a, **kw):
        verified.append(kw['required_modules']); return {'verified': True, 'test_only': True}
    monkeypatch.setattr(subject, 'ASSET_ROOT', tmp_path)
    monkeypatch.setattr(freeze_project_sources, 'verify_if_snapshot', verify)
    root, scene, task, asset, metadata, _ = parent
    output = tmp_path/'convex03'/f'seed_{seed:02d}'
    path = subject.prepare_asset(root, seed, output, profile=subject.SUPPORT_HULL_VERSION)
    assert 'scenes.workrooms.kitchen.produce_support_hull' in verified[0]
    assert 'scenes.workrooms.kitchen.assets' in verified[0]
    if seed == 0:
        assert (output/'task.usda').read_bytes() == Path(asset['usd_path']).read_bytes()
        assert (output/'metadata.json').read_bytes() == Path(asset['metadata_path']).read_bytes()
    composed = subject.compose_case(root, path, 'single_arm')
    for name in ('robot', 'workroom', 'isaacsim'):
        assert composed['scene'][name] == scene[name]
    assert composed['initial_state'] == task['initial_state']
    assert composed['task']['public_rules'] == task['public_rules']
    public = json.dumps(public_task_projection(composed['task']))
    assert 'support_hull_profile' not in public and 'reset_support_receipts' not in public
    script = '''import builtins,sys
original=builtins.__import__
def guarded(name,*args,**kw):
    if name=='pxr' or name.startswith('pxr.'): raise RuntimeError('pre-Kit USD import')
    return original(name,*args,**kw)
builtins.__import__=guarded
from pathlib import Path
from scenes.workrooms.kitchen.seed_variants import compose_case
result=compose_case(Path(sys.argv[1]),Path(sys.argv[2]),'single_arm')
assert result['task']['id']=='produce_transfer'
'''
    finished = subprocess.run([sys.executable, '-B', '-c', script, str(root), str(path)],
                              capture_output=True, text=True, timeout=60)
    assert finished.returncode == 0, finished.stdout+finished.stderr
