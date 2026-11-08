import numpy as np
import trimesh
import pytest

from scripts.diagnostics.collider_clearance import certify, hull, separated_convex_triangle_distance


def geometry(mesh):
    return {'vertices': mesh.vertices, 'faces': mesh.faces, 'triangles': mesh.triangles}


def test_open_container_cavity_is_not_a_solid_bounding_box():
    ring = trimesh.creation.annulus(r_min=.05, r_max=.10, height=.10, sections=32)
    small = hull(trimesh.creation.box(extents=[.02, .02, .02]).vertices)
    result = certify(small, geometry(ring))
    assert result['passed']
    assert result['obstacle_closed_surface']
    assert abs(result['robot_point_winding_number']) < .01


def test_surface_distance_does_not_accept_solid_containment():
    robot = hull(trimesh.creation.box(extents=[.02, .02, .02]).vertices)
    obstacle = geometry(trimesh.creation.box(extents=[.20, .20, .20]))
    result = certify(robot, obstacle)
    assert not result['passed']
    assert result['reason'] == 'robot_inside_closed_obstacle'
    reverse = certify(hull(obstacle['vertices']), geometry(trimesh.creation.box(extents=[.02, .02, .02])))
    assert not reverse['passed']
    assert reverse['reason'] in ('obstacle_vertex_inside_robot_hull', 'triangle_separation_not_certified')


def test_crossing_triangle_and_subthreshold_gap_do_not_pass():
    robot = hull(trimesh.creation.box(extents=[.02, .02, .02]).vertices)
    for z in (0., .014):
        vertices = np.array([[-.1, -.1, z], [.1, -.1, z], [0., .1, z]])
        result = certify(robot, {'vertices': vertices, 'faces': np.array([[0, 1, 2]]),
                                 'triangles': vertices[None]})
        assert not result['passed']
    vertices[:, 2] = .020
    assert certify(robot, {'vertices': vertices, 'faces': np.array([[0, 1, 2]]),
                           'triangles': vertices[None]})['passed']


def test_diagonal_vertex_distance_exceeds_axis_aligned_gap():
    robot = hull(trimesh.creation.box(extents=[.02, .02, .02]).vertices)
    triangle = np.array([[.014, .014, 0.], [.020, .014, 0.], [.014, .020, 0.]])
    assert np.isclose(separated_convex_triangle_distance(robot, triangle), np.sqrt(2)*.004, atol=1.e-10)


def test_tetmesh_uses_its_authored_closed_surface_including_world_transform():
    pytest.importorskip('pxr')
    from pxr import Gf, Usd, UsdGeom, Vt
    from scripts.diagnostics.collider_clearance import usd_geometry
    stage = Usd.Stage.CreateInMemory()
    body = UsdGeom.TetMesh.Define(stage, '/foam')
    vertices = np.array([[0., 0., 0.], [.1, 0., 0.], [0., .1, 0.], [0., 0., .1]], np.float32)
    faces = np.array([[0, 2, 1], [0, 1, 3], [1, 2, 3], [2, 0, 3]], np.int32)
    body.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(vertices))
    body.CreateTetVertexIndicesAttr([(0, 1, 2, 3)])
    body.CreateSurfaceFaceVertexIndicesAttr(Vt.Vec3iArray.FromNumpy(faces))
    body.AddTranslateOp().Set(Gf.Vec3d(.2, -.8, .9))
    surface = usd_geometry(body.GetPrim(), UsdGeom.XformCache())
    np.testing.assert_allclose(surface['vertices'], vertices+[.2, -.8, .9], atol=1.e-12)
    np.testing.assert_array_equal(surface['faces'], faces)
    robot = trimesh.creation.box(extents=[.005, .005, .005])
    robot.apply_translation([.22, -.78, .92])
    assert certify(hull(robot.vertices), surface)['reason'] == 'robot_inside_closed_obstacle'
    robot.apply_translation([.3, 0., 0.])
    assert certify(hull(robot.vertices), surface)['passed']
    body.GetSurfaceFaceVertexIndicesAttr().Clear()
    with pytest.raises(ValueError, match='explicit physical surface faces'):
        usd_geometry(body.GetPrim(), UsdGeom.XformCache())
