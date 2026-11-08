"""Five-shape backup preserves original body and exact open bowl triangles."""
import numpy as np
import pytest
from pxr import Gf, UsdGeom, UsdPhysics, Vt
import trimesh

from scenes.workrooms.kitchen.utensil_spoon_hybrid import apply_spoon_split_bowl_colliders
from test_kitchen_spoon_convex import _author, _mesh, _snapshot


def _body_mesh(stage, path):
    mesh = _mesh(stage, path)
    transform = UsdGeom.Xformable(stage.GetPrimAtPath(path)).GetLocalTransformation()
    mesh.vertices = np.array([transform.Transform(Gf.Vec3d(*point)) for point in mesh.vertices])
    return mesh


@pytest.mark.parametrize('task', ['utensil_sorting', 'table_clearing'])
def test_five_shapes_keep_body_visual_material_and_original_bowl(task):
    author = _author(task)
    before = _snapshot(author.stage)
    rows = [row for row in author.meta['objects'] if row['role'] == 'spoon']
    originals = {row['id']: _mesh(author.stage, row['collider_paths'][0]) for row in rows}
    old_paths = {row['collider_paths'][0] for row in rows}
    apply_spoon_split_bowl_colliders(author)
    after = _snapshot(author.stage)
    for path, properties in before.items():
        if path in old_paths:
            properties['attributes']['physics:collisionEnabled'] = 'False'
        assert after[path] == properties, path
    for row in rows:
        original = originals[row['id']]
        receipt = row['spoon_split_bowl_collision_partition']
        assert row['collision_count'] == receipt['part_count'] == 5
        assert receipt['native_physics_qualified'] is False
        assert receipt['bowl_triangle_count'] == 1008
        assert receipt['bowl_sdf_resolution'] == 768
        assert 78e-6 < receipt['bowl_requested_grid_pitch_m'] < 79e-6
        assert receipt['convex_coordinate_scale'] == 512.
        original_corners = set(map(tuple, original.vertices))
        original_triangles = {tuple(map(tuple, triangle)) for triangle in original.triangles}
        all_shapes = []
        for index, path in enumerate(row['collider_paths']):
            prim = author.stage.GetPrimAtPath(path)
            assert not prim.HasAPI(UsdPhysics.RigidBodyAPI)
            assert UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get()
            assert prim.GetAttribute('physxCollision:contactOffset').Get() == pytest.approx(.0002)
            assert prim.GetAttribute('physxCollision:restOffset').Get() == 0.
            body_mesh = _body_mesh(author.stage, path)
            assert set(map(tuple, body_mesh.vertices)) <= original_corners
            all_shapes.append(body_mesh)
            if index == 0:
                assert UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get() == 'sdf'
                assert len(body_mesh.faces) == 1008
                assert {tuple(map(tuple, triangle)) for triangle in body_mesh.triangles} <= original_triangles
                assert body_mesh.is_watertight and body_mesh.is_winding_consistent
                for attr, value in receipt['original_sdf_parameters'].items():
                    assert prim.GetAttribute(attr).Get() == (768 if attr.endswith(':sdfResolution') else value)
            else:
                assert UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get() == 'convexHull'
                assert not prim.HasAttribute('physxSDFMeshCollision:sdfResolution')
                np.testing.assert_array_equal(np.array(UsdGeom.Xformable(prim).GetLocalTransformation()),
                                              np.diag([1/512, 1/512, 1/512, 1.]))
        combined = trimesh.util.concatenate(all_shapes)
        np.testing.assert_array_equal(combined.bounds, original.bounds)
        assert combined.volume == pytest.approx(original.volume, rel=1e-7)
        np.testing.assert_allclose(combined.center_mass, original.center_mass, atol=2e-9, rtol=0)
        assert np.linalg.norm(combined.moment_inertia-original.moment_inertia)/np.linalg.norm(original.moment_inertia) < 1e-7
    with pytest.raises(ValueError, match='original enabled|already been authored'):
        apply_spoon_split_bowl_colliders(author)


def test_384_is_an_explicit_separate_resolution_with_original_other_sdf_parameters():
    author = _author('table_clearing')
    apply_spoon_split_bowl_colliders(author, sdf_resolution=384)
    row = next(row for row in author.meta['objects'] if row['role'] == 'spoon')
    prim = author.stage.GetPrimAtPath(row['collider_paths'][0])
    assert prim.GetAttribute('physxSDFMeshCollision:sdfResolution').Get() == 384
    assert prim.GetAttribute('physxSDFMeshCollision:sdfSubgridResolution').Get() == 6
    assert prim.GetAttribute('physxSDFMeshCollision:sdfMargin').Get() == pytest.approx(.01)
    assert prim.GetAttribute('physxSDFMeshCollision:sdfNarrowBandThickness').Get() == pytest.approx(.02)
    assert 156e-6 < row['spoon_split_bowl_collision_partition']['bowl_requested_grid_pitch_m'] < 157e-6


@pytest.mark.parametrize('resolution', [0, 768., True])
def test_invalid_resolution_fails_before_stage_mutation(resolution):
    author = _author('table_clearing')
    before = author.stage.GetRootLayer().ExportToString()
    with pytest.raises(ValueError, match='explicitly 384 or 768'):
        apply_spoon_split_bowl_colliders(author, resolution)
    assert author.stage.GetRootLayer().ExportToString() == before


def test_changed_original_geometry_is_rejected_without_partial_candidates():
    author = _author('utensil_sorting')
    row = next(row for row in author.meta['objects'] if row['id'] == 'spoon_1')
    mesh = UsdGeom.Mesh(author.stage.GetPrimAtPath(row['collider_paths'][0]))
    points = np.asarray(mesh.GetPointsAttr().Get()).copy()
    points[0, 0] += .0001
    mesh.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(points))
    before = author.stage.GetRootLayer().ExportToString()
    with pytest.raises(ValueError, match='recipe|topology'):
        apply_spoon_split_bowl_colliders(author)
    assert author.stage.GetRootLayer().ExportToString() == before
