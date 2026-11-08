"""Metric geometry, support and density checks for the explicit larger props."""
from itertools import combinations

import numpy as np
import pytest
from pxr import Usd, UsdGeom, UsdPhysics
from scipy.spatial.transform import Rotation

from scenes.workrooms.kitchen.assets import TASK_IDS, author_task


def world_points(row, pose):
    return Rotation.from_quat(pose[3:]).apply(row['collision_support_points_m'])+pose[:3]


@pytest.mark.parametrize('task_id', TASK_IDS)
def test_household_metric_mesh_mass_inertia_and_support_metadata(task_id):
    stage = Usd.Stage.CreateInMemory()
    meta = author_task(stage, task_id, prop_profile='household_v1', dressing_profile='enriched_v1')
    rows = {**meta['fixture_metadata'], **{r['id']: r for r in meta['objects']}}
    for item in meta['objects']:
        change = meta['dimension_changes'][item['id']]
        factor = np.array(change['scale_xyz'])
        assert item['mass_kg'] == pytest.approx(change['before_mass_kg']*np.prod(factor))
        assert min(item['diagonal_inertia_kg_m2']) > 0
        assert max(item['diagonal_inertia_kg_m2']) <= sum(item['diagonal_inertia_kg_m2'])/2+1e-12
        api = UsdPhysics.MassAPI(stage.GetPrimAtPath(item['body_path']))
        assert api.GetMassAttr().Get() == pytest.approx(item['mass_kg'])
        assert not stage.GetPrimAtPath(item['body_path']).HasAttribute('xformOp:scale')
        points = np.concatenate([np.asarray(UsdGeom.Mesh(stage.GetPrimAtPath(p)).GetPointsAttr().Get())
                                 for p in item['collider_paths']])
        np.testing.assert_allclose([points.min(0), points.max(0)], item['collision_bounds_m'], atol=1e-8)
    for mode in ('source', 'goal'):
        poses = {name: (row['initial_pose_xyzw'] if mode == 'source' or name not in meta['actors']
                       else meta['goals']['body_poses_xyzw'][name]) for name, row in rows.items()}
        bounds = {}
        for item in meta['objects']:
            name = item['id']
            points = world_points(item, poses[name])
            bounds[name] = np.array([points.min(0), points.max(0)])
            support = (meta['source_support_by_actor'] if mode == 'source' else meta['goals']['support_by_actor'])[name]
            if support not in rows or (mode == 'goal' and name == 'lid'):
                continue
            parent = rows[support]
            if 'interior_bounds_xy_m' not in parent:
                continue
            local = Rotation.from_quat(poses[support][3:]).inv().apply(points-poses[support][:3])
            interior = np.asarray(parent['interior_bounds_xy_m'])
            assert np.all(local[:, :2].min(0) >= interior[0]-1e-7), (mode, name, support)
            assert np.all(local[:, :2].max(0) <= interior[1]+1e-7), (mode, name, support)
        supports = meta['source_support_by_actor'] if mode == 'source' else meta['goals']['support_by_actor']
        def ancestors(name):
            result = set()
            while name in supports:
                name = supports[name]
                if name in result:
                    raise AssertionError('Cyclic support')
                result.add(name)
            return result
        for a, b in combinations(bounds, 2):
            if a not in ancestors(b) and b not in ancestors(a):
                assert np.any(bounds[a][1]+.001 <= bounds[b][0]) or np.any(bounds[b][1]+.001 <= bounds[a][0]), (a, b)


@pytest.mark.parametrize('task_id', TASK_IDS)
def test_scaled_presentation_preserves_canonical_geometry_and_support_clearance(task_id):
    normal = author_task(Usd.Stage.CreateInMemory(), task_id, prop_profile='household_v1')
    presentation = author_task(Usd.Stage.CreateInMemory(), task_id, prop_profile='household_v1', presentation=True)
    for row, shown in zip(normal['objects'], presentation['objects'], strict=True):
        for key in ('mass_kg', 'diagonal_inertia_kg_m2', 'collision_bounds_m', 'collider_paths'):
            assert row[key] == shown[key]
        np.testing.assert_allclose(shown['initial_pose_xyzw'], normal['goals']['body_poses_xyzw'][row['id']])
    assert normal['physics_prior'] == presentation['physics_prior']
    if task_id == 'pot_lid_placement':
        goals = normal['goals']
        seat = goals['body_poses_xyzw']['lid'][2]+goals['lid_seat_local_z_m']
        rim = goals['body_poses_xyzw']['pot'][2]+goals['pot_rim_local_z_m']
        assert seat-rim == pytest.approx(.0003)


def test_sponge_wiping_face_tracks_real_larger_bottom_without_rule_changes():
    meta = author_task(Usd.Stage.CreateInMemory(), 'surface_wiping', prop_profile='household_v1')
    sponge = meta['objects'][0]
    np.testing.assert_allclose(np.diff(sponge['collision_bounds_m'], axis=0)[0], [.120, .078, .045], atol=1e-8)
    np.testing.assert_allclose(sponge['wiping_face_bounds_xy_m'], [[-.0552, -.0342], [.0552, .0342]])
    assert sponge['collision_count'] == 1
    assert sponge['physical_approximation'].startswith('Rigid dry sponge')
