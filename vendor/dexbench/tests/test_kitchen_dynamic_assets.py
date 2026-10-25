"""CPU geometry/inventory contracts, separate from native scene acceptance."""
import json
import math

import numpy as np
import pytest
import trimesh
from pxr import Usd, UsdGeom, UsdPhysics

from scenes.workrooms.kitchen.assets import TASK_IDS, author_task


def _mesh(stage, path):
    mesh = UsdGeom.Mesh(stage.GetPrimAtPath(path))
    return trimesh.Trimesh(np.asarray(mesh.GetPointsAttr().Get()),
        np.asarray(mesh.GetFaceVertexIndicesAttr().Get()).reshape(-1, 3), process=False)


def _inside_closed_mesh(mesh, point):
    # Solid angle winding avoids optional ray-query acceleration packages.
    triangle = np.asarray(mesh.triangles)-np.asarray(point)
    a, b, c = triangle[:, 0], triangle[:, 1], triangle[:, 2]
    la, lb, lc = (np.linalg.norm(x, axis=1) for x in (a, b, c))
    numerator = np.einsum('ij,ij->i', a, np.cross(b, c))
    denominator = la*lb*lc+np.einsum('ij,ij->i', a, b)*lc
    denominator += np.einsum('ij,ij->i', b, c)*la+np.einsum('ij,ij->i', c, a)*lb
    return abs(float(np.sum(2*np.arctan2(numerator, denominator)))) > 2*math.pi


@pytest.mark.parametrize('task_id,dynamic_count,fixture_count', [
    ('breakfast_serving', 5, 0), ('produce_transfer', 6, 3),
    ('utensil_sorting', 6, 2), ('pot_lid_placement', 2, 1),
    ('pantry_stocking', 8, 4), ('table_clearing', 8, 4)])
def test_passive_inventory_and_mass_are_explicit(task_id, dynamic_count, fixture_count):
    stage = Usd.Stage.CreateInMemory()
    metadata = author_task(stage, task_id)
    assert len(metadata['actors']) == dynamic_count
    assert len(metadata['fixtures']) == fixture_count
    assert set(metadata['source_support_by_actor']) == set(metadata['actors'])
    assert set(metadata['goals']['body_poses_xyzw']) == set(metadata['actors'])
    assert len(json.dumps(metadata, allow_nan=False)) > 100
    for item in metadata['objects']:
        body = stage.GetPrimAtPath(item['body_path'])
        rigid = UsdPhysics.RigidBodyAPI(body)
        assert rigid.GetRigidBodyEnabledAttr().Get()
        assert not rigid.GetKinematicEnabledAttr().Get()
        api = UsdPhysics.MassAPI(body)
        assert api.GetMassAttr().Get() == pytest.approx(item['mass_kg'])
        assert np.all(np.asarray(api.GetDiagonalInertiaAttr().Get()) > 0)
        assert np.linalg.norm(item['principal_axes_xyzw']) == pytest.approx(1.)
        assert item['collision_bounds_m'][0][2] == pytest.approx(0., abs=1e-8)
        assert item['collision_count'] <= 3
        assert item['mass_kg'] > 0
    assert not any(prim.IsA(UsdPhysics.Joint) for prim in stage.Traverse())
    assert not any(prim.IsA(UsdPhysics.Scene) for prim in stage.Traverse())


def test_tapered_produce_rests_on_real_polygon_containing_its_center_of_mass():
    from scipy.spatial import ConvexHull
    stage = Usd.Stage.CreateInMemory()
    meta = author_task(stage, 'produce_transfer')
    for item in meta['objects']:
        if item['role'] != 'carrot':
            continue
        contact = _mesh(stage, item['collider_paths'][0]).convex_hull
        vertices = np.asarray(contact.vertices)
        support = vertices[vertices[:, 2] < vertices[:, 2].min()+1e-7]
        assert len(support) >= 4
        polygon = ConvexHull(support[:, :2])
        center = np.asarray(item['center_of_mass_m'])[:2]
        margin = -(polygon.equations[:, :2]@center+polygon.equations[:, 2]).max()
        assert margin > .0005
        assert polygon.volume > 3e-5
        assert vertices[:, 2].min() == pytest.approx(0., abs=1e-8)
        assert item['natural_rest']['rotation_angle_rad'] < math.radians(10)


@pytest.mark.parametrize('task_id', TASK_IDS)
def test_contact_geometry_is_closed_and_never_nested_rigid_bodies(task_id):
    stage = Usd.Stage.CreateInMemory()
    metadata = author_task(stage, task_id)
    actors = set(metadata['actors'].values())
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            mesh = _mesh(stage, str(prim.GetPath()))
            assert mesh.is_watertight
            assert mesh.is_winding_consistent
            assert mesh.volume > 0
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            assert str(prim.GetPath()) in actors
            assert not any(str(prim.GetPath()).startswith(path+'/') for path in actors)


@pytest.mark.parametrize('task_id,actor,void_point,solid_point', [
    ('breakfast_serving', 'tray', (0, 0, .012), (0, 0, .002)),
    ('breakfast_serving', 'plate', (0, 0, .010), (0, 0, .002)),
    ('breakfast_serving', 'cup', (0, 0, .045), (0, 0, .002)),
    ('pot_lid_placement', 'pot', (0, 0, .050), (0, 0, .003)),
    ('table_clearing', 'dinner_plate', (0, 0, .010), (0, 0, .002)),
    ('table_clearing', 'side_plate', (0, 0, .010), (0, 0, .002)),
    ('table_clearing', 'cereal_bowl', (0, 0, .025), (0, 0, .002)),
    ('table_clearing', 'small_bowl', (0, 0, .020), (0, 0, .002)),
    ('table_clearing', 'mug', (0, 0, .040), (0, 0, .002)),
    ('table_clearing', 'tea_cup', (0, 0, .030), (0, 0, .002)),
])
def test_vessel_cavity_is_physically_open(task_id, actor, void_point, solid_point):
    stage = Usd.Stage.CreateInMemory()
    metadata = author_task(stage, task_id)
    path = metadata['actors'][actor]+'/Collision/shell'
    assert UsdPhysics.MeshCollisionAPI(stage.GetPrimAtPath(path)).GetApproximationAttr().Get() == 'sdf'
    mesh = _mesh(stage, path)
    assert not _inside_closed_mesh(mesh, void_point)
    assert _inside_closed_mesh(mesh, solid_point)


@pytest.mark.parametrize('task_id', TASK_IDS)
def test_goal_reset_remains_dynamic_and_matches_declared_support_poses(task_id):
    stage = Usd.Stage.CreateInMemory()
    metadata = author_task(stage, task_id, presentation=True)
    for item in metadata['objects']:
        np.testing.assert_allclose(item['initial_pose_xyzw'], metadata['goals']['body_poses_xyzw'][item['id']])
        assert UsdPhysics.RigidBodyAPI(stage.GetPrimAtPath(item['body_path'])).GetRigidBodyEnabledAttr().Get()
    assert metadata['presentation'] is True
    assert metadata['physics_accepted'] is False


def test_goal_payloads_fit_their_actual_supports_with_room_to_correct():
    stage = Usd.Stage.CreateInMemory()
    metadata = author_task(stage, 'breakfast_serving', presentation=True)
    items = {item['id']: item for item in metadata['objects']}
    goals = metadata['goals']['body_poses_xyzw']
    for name in ('bread_0', 'bread_1'):
        offset = np.asarray(goals[name][:2])-goals['plate'][:2]
        low, high = np.asarray(items[name]['local_bounds_m'])[:, :2]
        corners = [(x, y) for x in (low[0], high[0]) for y in (low[1], high[1])]
        assert max(np.linalg.norm(np.asarray(corner)+offset) for corner in corners) < items['plate']['interior_radius_m']-.004
    for name in ('plate', 'cup'):
        offset = np.asarray(goals[name][:2])-goals['tray'][:2]
        bounds = np.asarray(items[name]['local_bounds_m'])[:, :2]+offset
        interior = np.asarray(items['tray']['interior_bounds_xy_m'])
        assert np.all(bounds[0] > interior[0]+.004)
        assert np.all(bounds[1] < interior[1]-.004)


def test_pot_lid_has_passive_geometric_clearance():
    stage = Usd.Stage.CreateInMemory()
    metadata = author_task(stage, 'pot_lid_placement')
    items = {item['id']: item for item in metadata['objects']}
    assert items['pot']['interior_radius_m']-items['lid']['locating_outer_radius_m'] == pytest.approx(.002)
    assert items['lid']['seat_radius_range_m'][1] > items['pot']['rim_radius_range_m'][1]
    poses = metadata['goals']['body_poses_xyzw']
    nominal_gap = (poses['lid'][2]+items['lid']['seat_local_z_m'])-(poses['pot'][2]+items['pot']['rim_local_z_m'])
    assert nominal_gap == pytest.approx(.0003)
    # Source locations are separated and both source origin rays are <=1 m
    # horizontally from the fixed base; this does not establish arm IK.
    for item in items.values():
        assert np.linalg.norm(np.asarray(item['initial_pose_xyzw'][:2])-[.65, 0.]) < 1.
