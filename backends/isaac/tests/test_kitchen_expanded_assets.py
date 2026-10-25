"""Physical inventory/clearance witnesses, separate from native acceptance."""
from itertools import combinations

import numpy as np
import pytest
from pxr import Usd, UsdGeom, UsdPhysics

from scenes.workrooms.kitchen.assets import TASK_IDS, author_task


@pytest.mark.parametrize('task_id', TASK_IDS)
def test_context_is_twelve_real_fixed_objects_outside_workspaces(task_id):
    stage = Usd.Stage.CreateInMemory()
    meta = author_task(stage, task_id, dressing_profile='enriched_v1')
    assert len(meta['contextual_props']) == 12
    assert set(meta['actors']).isdisjoint(meta['contextual_props'])
    assert set(meta['fixtures']).isdisjoint(meta['contextual_props'])
    assert set(meta['goals']['body_poses_xyzw']) == set(meta['actors'])
    assert len(meta['public_goal']['objects']) == len(meta['actors'])
    assert len(meta['public_goal']['context_objects']) == 12
    context_bounds = []
    for item in meta['contextual_prop_metadata'].values():
        assert item['required'] is item['scoring'] is False
        assert item['static_dressing'] is True
        assert not stage.GetPrimAtPath(item['body_path']).HasAPI(UsdPhysics.RigidBodyAPI)
        bounds = np.asarray(item['local_bounds_m']) + item['initial_pose_xyzw'][:3]
        # Context is wholly supported on the existing outer bar strip and
        # physically clear of both the task workspace and the robot aisle.
        assert bounds[0, 0] >= -.95 and bounds[1, 0] <= 1.85
        assert bounds[0, 1] >= -1.35 and bounds[1, 1] < -1.22
        assert bounds[0, 2] == pytest.approx(.9001, abs=1e-7)
        context_bounds.append(bounds)
        for path in item['collider_paths']:
            assert UsdPhysics.CollisionAPI(stage.GetPrimAtPath(path)).GetCollisionEnabledAttr().Get()
    for left, right in combinations(context_bounds, 2):
        assert np.any(left[1] <= right[0]) or np.any(right[1] <= left[0])


@pytest.mark.parametrize('task_id', TASK_IDS[:4])
def test_dressing_opt_in_preserves_existing_contact_geometry_and_reset(task_id):
    legacy_stage, enriched_stage = Usd.Stage.CreateInMemory(), Usd.Stage.CreateInMemory()
    legacy = author_task(legacy_stage, task_id)
    enriched = author_task(enriched_stage, task_id, dressing_profile='enriched_v1')
    assert 'contextual_props' not in legacy and 'public_goal' not in legacy
    for key in ('actors', 'fixtures', 'goals', 'source_support_by_actor', 'physics_prior'):
        assert legacy[key] == enriched[key]
    for item in legacy['objects']:
        newer = next(row for row in enriched['objects'] if row['id'] == item['id'])
        for key in ('mass_kg', 'diagonal_inertia_kg_m2', 'initial_pose_xyzw', 'collision_bounds_m'):
            assert item[key] == newer[key]
        for path in item['collider_paths']:
            before, after = UsdGeom.Mesh(legacy_stage.GetPrimAtPath(path)), UsdGeom.Mesh(enriched_stage.GetPrimAtPath(path))
            assert before.GetPointsAttr().Get() == after.GetPointsAttr().Get()
            assert before.GetFaceVertexIndicesAttr().Get() == after.GetFaceVertexIndicesAttr().Get()


@pytest.mark.parametrize('task_id', TASK_IDS[-2:])
def test_new_inventory_source_and_goal_have_real_geometric_clearance(task_id):
    meta = author_task(Usd.Stage.CreateInMemory(), task_id)
    bounds = []
    for item in meta['objects']:
        local = np.asarray(item['local_bounds_m'])
        source = np.asarray(item['initial_pose_xyzw'][:3])
        source_fixture = meta['fixture_metadata'][meta['source_support_by_actor'][item['id']]]
        source_interior = np.asarray(source_fixture['interior_bounds_xy_m']) + np.asarray(source_fixture['initial_pose_xyzw'][:2])
        world = local + source
        assert np.all(world[0, :2] >= source_interior[0])
        assert np.all(world[1, :2] <= source_interior[1])
        bounds.append(world)
        goal = np.asarray(meta['goals']['body_poses_xyzw'][item['id']][:3])
        target = meta['fixture_metadata'][item['assigned_fixture']]
        interior = np.asarray(target['interior_bounds_xy_m']) + np.asarray(target['initial_pose_xyzw'][:2])
        goal_world = local+goal
        assert np.all(goal_world[0, :2] > interior[0]+.002)
        assert np.all(goal_world[1, :2] < interior[1]-.002)
    for left, right in combinations(bounds, 2):
        assert np.any(left[1]+.001 <= right[0]) or np.any(right[1]+.001 <= left[0])


def test_clearing_stack_fits_dinner_plate_with_actual_floor_support():
    meta = author_task(Usd.Stage.CreateInMemory(), 'table_clearing', presentation=True)
    items = {row['id']: row for row in meta['objects']}
    dinner, side = items['dinner_plate'], items['side_plate']
    assert meta['goals']['support_by_actor']['side_plate'] == 'dinner_plate'
    points = np.asarray(side['collision_support_points_m'])
    assert np.max(np.linalg.norm(points[:, :2], axis=1)) < dinner['interior_radius_m']-.003
    expected = dinner['initial_pose_xyzw'][2]+dinner['floor_top_local_z_m']
    assert side['initial_pose_xyzw'][2]-expected == pytest.approx(.0003)
    assert set(meta['placements'].values()) == {'dishes_tray', 'drinkware_tray', 'cutlery_tray'}
