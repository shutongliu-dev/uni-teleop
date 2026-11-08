"""Single end-radius variant with unchanged old profiles and physical scope."""
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
from scenes.workrooms.kitchen import produce_support_hull as hull, seed_variants as subject

PUBLICATION = Path(f'{DATA_ROOT}/datasets/fr3_sharpa_bench/isaacsim/releases/'
                   'workroom_initial_randomization_v1/publication01/releases')


@pytest.fixture(scope='module')
def parent():
    manifest = load_json(PUBLICATION/'produce_transfer/single_arm/release_manifest.json')
    root = Path(manifest['base_release']['path'])
    scene, task, asset, metadata, _ = subject._parent(root)
    return root, scene, task, asset, metadata, Usd.Stage.Open(asset['usd_path'])


@pytest.mark.parametrize('seed', range(10))
def test_only_small_end_ring_changes_and_final_material_is_integrated(parent, seed):
    metadata, original = parent[4:]
    plan = subject.case_plan('produce_transfer', seed, metadata, profile=subject.SUPPORT_LOW_VERTEX_TIP_VERSION)
    baseline = subject.derive_stage(original, metadata, subject.case_plan(
        'produce_transfer', seed, metadata, profile=subject.SUPPORT_LOW_VERTEX_VERSION))
    result = subject.derive_stage(original, metadata, plan)
    assert result['collision_recipe_sha256'] == baseline['collision_recipe_sha256']
    assert result['fixed_support_paths'] == baseline['fixed_support_paths']
    assert all(row['geometry_contained'] for row in result['structural_checks'])
    assert all(row['separating_axis_gap_m'] >= 0 for row in result['footprint_separations'])
    if not seed:
        assert result['metadata'] == metadata
        assert result['stage'].GetRootLayer().ExportToString() == baseline['stage'].GetRootLayer().ExportToString()
        return
    old_rows = {r['id']: r for r in baseline['metadata']['objects']}
    for row in result['metadata']['objects']:
        previous = old_rows[row['id']]
        if row['category'] != 'carrot':
            assert row == previous
            continue
        body = result['stage'].GetPrimAtPath(row['body_path'])
        physical, polygons = hull._mesh(result['stage'], row['collider_paths'][0], body)
        old, old_polygons = hull._mesh(baseline['stage'], row['collider_paths'][0],
                                      baseline['stage'].GetPrimAtPath(row['body_path']))
        visible, visual_polygons = hull._mesh(result['stage'], row['body_path']+'/Visual/body', body)
        np.testing.assert_array_equal(physical.vertices, visible.vertices)
        assert polygons == visual_polygons == old_polygons
        assert len(physical.vertices) == len(polygons) == 33 and len(physical.faces) == 62
        assert len(ConvexHull(physical.vertices).simplices) <= 62
        np.testing.assert_array_equal(physical.vertices[8:], old.vertices[8:])
        assert .00062 < np.linalg.norm(physical.vertices[:8]-old.vertices[:8], axis=1).max() < .00065
        assert row['initial_pose_xyzw'] == previous['initial_pose_xyzw']
        profile, prior = row['support_hull_profile'], previous['support_hull_profile']
        assert profile['canonical_extra_rest_transform'] == prior['canonical_extra_rest_transform']
        assert profile['support_vertices_body_m'] == prior['support_vertices_body_m']
        assert profile['parent_ring_radius_overrides_m'] == {'0': {'original': .002, 'variant': .0025}}
        assert profile['collision_mesh_coordinate_scale'] == 1.
        assert profile['native_cooked_geometry_verified'] is False
        assert physical.volume/old.volume-1 == pytest.approx(.0070249703, abs=3e-8)
        api = UsdPhysics.MassAPI(body)
        density = profile['parent_actual_density_kg_m3']
        assert float(api.GetMassAttr().Get()) == pytest.approx(density*physical.volume, rel=1e-7)
        np.testing.assert_allclose(api.GetCenterOfMassAttr().Get(), physical.center_mass, atol=2e-9, rtol=0)
        q = api.GetPrincipalAxesAttr().Get()
        R = Rotation.from_quat([*q.GetImaginary(), q.GetReal()]).as_matrix()
        actual = R@np.diag(api.GetDiagonalInertiaAttr().Get())@R.T
        physical.density = density
        np.testing.assert_allclose(actual, physical.moment_inertia, atol=2e-11, rtol=1e-5)
        prominences = []
        for i, point in enumerate(physical.vertices):
            planes = ConvexHull(np.delete(physical.vertices, i, axis=0)).equations
            prominences.append((planes[:, :3]@point+planes[:, 3]).max())
        assert min(prominences) > .00086  # CPU surface separation, not a native setting assertion.
        assert .0062 < profile['integrated_com_support_margin_m'] < .0067
        Sdf.CopySpec(baseline['stage'].GetRootLayer(), row['body_path'],
                     result['stage'].GetRootLayer(), row['body_path'])
    assert result['stage'].GetRootLayer().ExportToString() == baseline['stage'].GetRootLayer().ExportToString()


@pytest.mark.parametrize('value', [{'0': {'original': .002, 'variant': .003}}, {},
                                   {'1': {'original': .006, 'variant': .0065}}])
def test_self_rehashed_radius_recipe_is_rejected(parent, value):
    metadata, stage = parent[4:]
    plan = subject.case_plan('produce_transfer', 1, metadata, profile=subject.SUPPORT_LOW_VERTEX_TIP_VERSION)
    plan['geometry_profile']['parent_ring_radius_overrides_m'] = value
    plan.pop('plan_sha256'); plan['plan_sha256'] = subject.canonical_hash(plan)
    with pytest.raises(ValueError, match='finite generator'):
        subject.derive_stage(stage, metadata, plan)


@pytest.mark.parametrize('task_id', subject.TASK_IDS[1:])
def test_tip_profile_rejects_other_tasks(task_id):
    record = load_json(PUBLICATION/task_id/'single_arm/release_manifest.json')
    metadata = subject._parent(Path(record['base_release']['path']))[3]
    with pytest.raises(ValueError, match='profile'):
        subject.case_plan(task_id, 1, metadata, profile=subject.SUPPORT_LOW_VERTEX_TIP_VERSION)


def test_frozen_v4_release_all_ten_cases_still_strictly_consume():
    root = PUBLICATION.parents[1]/'lowpoly04/releases/produce_transfer/single_arm'
    manifest = validate(root)
    assert len(manifest['cases']) == 10
    assert all(load_json(Path(row['asset_manifest']))['generator_version'] == subject.SUPPORT_LOW_VERTEX_VERSION
               for row in manifest['cases'].values())


def test_tip_visual_relief_intersects_new_solid_and_stems_are_unchanged_from_v4(parent):
    metadata, original = parent[4:]
    old = subject.derive_stage(original, metadata, subject.case_plan(
        'produce_transfer', 1, metadata, profile=subject.SUPPORT_LOW_VERTEX_VERSION))
    new = subject.derive_stage(original, metadata, subject.case_plan(
        'produce_transfer', 1, metadata, profile=subject.SUPPORT_LOW_VERTEX_TIP_VERSION))
    path = '/World/KitchenTask/Actors/carrot_0'
    solid, _ = hull._mesh(new['stage'], path+'/Collision/body', new['stage'].GetPrimAtPath(path))
    planes = ConvexHull(solid.vertices).equations
    for i in range(5):
        mesh = UsdGeom.Mesh(new['stage'].GetPrimAtPath(path+f'/Visual/root_ring_{i}'))
        assert not mesh.GetPrim().HasAPI(UsdPhysics.CollisionAPI)
        points = np.asarray(mesh.GetPointsAttr().Get(), float)
        signed = (points@planes[:, :3].T+planes[:, 3]).max(1)
        assert signed.min() < 0 < signed.max() < .0008
    for i in range(3):
        stem = path+f'/Visual/stem_{i}'
        for attr in ('points', 'normals', 'extent', 'faceVertexCounts', 'faceVertexIndices'):
            np.testing.assert_array_equal(old['stage'].GetPrimAtPath(stem).GetAttribute(attr).Get(),
                                          new['stage'].GetPrimAtPath(stem).GetAttribute(attr).Get())


@pytest.mark.parametrize('seed', [0, 1])
def test_fresh_explicit_tip_profile_and_public_prekit_boundaries(tmp_path, monkeypatch, parent, seed):
    from scripts.maintenance import freeze_project_sources
    required = []
    def verify(*a, **kw):
        required.append(kw['required_modules']); return {'verified': True, 'test_only': True}
    monkeypatch.setattr(subject, 'ASSET_ROOT', tmp_path)
    monkeypatch.setattr(freeze_project_sources, 'verify_if_snapshot', verify)
    root, scene, task, asset, metadata, _ = parent
    output = tmp_path/'tip05'/f'seed_{seed:02d}'
    path = subject.prepare_asset(root, seed, output, profile=subject.SUPPORT_LOW_VERTEX_TIP_VERSION)
    assert 'scenes.workrooms.kitchen.produce_support_hull' in required[0]
    assert load_json(path)['generator_version'] == subject.SUPPORT_LOW_VERTEX_TIP_VERSION
    if not seed:
        assert (output/'task.usda').read_bytes() == Path(asset['usd_path']).read_bytes()
        assert (output/'metadata.json').read_bytes() == Path(asset['metadata_path']).read_bytes()
    result = subject.compose_case(root, path, 'single_arm')
    for key in ('robot', 'workroom', 'isaacsim'):
        assert result['scene'][key] == scene[key]
    assert result['initial_state'] == task['initial_state']
    assert result['task']['public_rules'] == task['public_rules']
    public = json.dumps(public_task_projection(result['task']))
    assert all(key not in public for key in ('support_hull_profile', 'reset_support_receipts', 'parent_ring_radius_overrides_m'))
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
    done = subprocess.run([sys.executable, '-B', '-c', script, str(root), str(path)], capture_output=True, text=True, timeout=60)
    assert done.returncode == 0, done.stdout+done.stderr
