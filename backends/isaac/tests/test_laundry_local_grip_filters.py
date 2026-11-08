from types import SimpleNamespace

import numpy as np
import pytest

from scripts.validation.laundry_local_grip_filters import (
    LocalGripFilters, local_triangle_region, native_collider_boxes, triangle_box_clearance)


def grid(n=61, spacing=.005):
    points = np.array([[x*spacing, y*spacing, 0.] for y in range(n) for x in range(n)])
    triangles = []
    for y in range(n-1):
        for x in range(n-1):
            a = y*n+x
            triangles.extend(((a, a+1, a+n), (a+1, a+n+1, a+n)))
    return points, np.asarray(triangles), (n//2)*n+n//2


def test_local_region_is_topological_and_union_deduplicates():
    q, triangles, anchor = grid()
    first = local_triangle_region(q, triangles, q, [anchor])
    assert 0 < first['rest_area_fraction'] < .03
    assert first['world_radius_m'] < .065
    assert first['maximum_material_distance_m'] < .03
    assert len(first['triangle_ids']) > np.isin(triangles, anchor).any(axis=1).sum()
    same = local_triangle_region(q, triangles, q, [anchor],
                                 existing_triangle_ids=first['triangle_ids']*2)
    assert same['active_union_rest_area_fraction'] == pytest.approx(first['rest_area_fraction'])
    # A distant sheet brought next to the grasp in world space must stay physical.
    moved = q.copy()
    moved[:30] = q[anchor]
    assert local_triangle_region(q, triangles, moved, [anchor])['triangle_ids'] == first['triangle_ids']


@pytest.mark.parametrize('indices', [[], [-1], [1.2], [True], [999999]])
def test_empty_or_invalid_grasp_never_becomes_whole_cloth_filter(indices):
    q, triangles, _ = grid()
    with pytest.raises(ValueError, match='indices'):
        local_triangle_region(q, triangles, q, indices)


def test_region_area_and_active_union_are_bounded():
    small, faces, anchor = grid(9, .002)
    with pytest.raises(ValueError, match='three percent'):
        local_triangle_region(small, faces, small, [anchor])
    q, faces, anchor = grid()
    with pytest.raises(ValueError, match='five percent'):
        local_triangle_region(q, faces, q, [anchor], existing_triangle_ids=np.arange(len(faces)//10))


def test_world_and_material_locality_are_independent():
    q, faces, anchor = grid()
    world = q.copy()
    world[anchor+1, 2] = .1
    with pytest.raises(ValueError, match='world radius'):
        local_triangle_region(q, faces, world, [anchor])
    material = q.copy()
    material[anchor+1, 0] += .1
    with pytest.raises(ValueError, match='material distance'):
        local_triangle_region(material, faces, q, [anchor])


def scene():
    from pxr import Gf, Usd, UsdGeom, UsdPhysics
    q, triangles, anchor = grid()
    stage = Usd.Stage.CreateInMemory()
    mesh = UsdGeom.Mesh.Define(stage, '/World/Cloth')
    mesh.CreatePointsAttr(q.tolist())
    mesh.CreateFaceVertexCountsAttr([3]*len(triangles))
    mesh.CreateFaceVertexIndicesAttr(triangles.ravel().tolist())
    UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
    for side in ('left', 'right'):
        body = UsdGeom.Xform.Define(stage, f'/World/Robot/{side}/{side}_hand_wrist')
        # Intentionally stale USD pose, far above the native pose in each test.
        body.AddTranslateOp().Set(Gf.Vec3d(q[anchor, 0], q[anchor, 1], 5.))
        UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
        pad = UsdGeom.Cube.Define(stage, str(body.GetPath())+'/pad')
        pad.CreateSizeAttr(.02)
        pad.AddTranslateOp().Set(Gf.Vec3d(.005, 0., 0.))
        UsdPhysics.CollisionAPI.Apply(pad.GetPrim())
    table = UsdGeom.Cube.Define(stage, '/World/Table')
    UsdPhysics.CollisionAPI.Apply(table.GetPrim())
    env = SimpleNamespace(stage=stage, body_names=['left_hand_wrist', 'right_hand_wrist'],
                          scene_info={'cloths': {'towel': {'path': '/World/Cloth'}}})
    return env, q, triangles, anchor


def test_schema_is_explicit_local_and_restores_using_native_not_stale_usd():
    env, q, triangles, anchor = scene()
    flushed = []
    filters = LocalGripFilters(env, q, triangles, flush_changes=lambda: flushed.append(True))
    item = filters.bind('right', [anchor], q, '/World/GripFilter')
    assert item.active and not item.released and filters.active_count == 1
    assert len(item.paths) == 1
    np.testing.assert_allclose(np.mean(item.bindings[0]['body_local_corners_m'], axis=0),
                               [.005, 0., 0.], atol=1.e-12)
    prim = env.stage.GetPrimAtPath(item.paths[0])
    assert prim.GetRelationship('omniphysics:src0').GetTargets() == ['/World/Cloth']
    assert 'right_hand_wrist/pad' in str(prim.GetRelationship('omniphysics:src1').GetTargets()[0])
    assert list(prim.GetAttribute('omniphysics:groupElemCounts0').Get()) == [len(item.triangle_ids)]
    assert list(prim.GetAttribute('omniphysics:groupElemIndices0').Get()) == list(item.triangle_ids)
    assert list(prim.GetAttribute('omniphysics:groupElemCounts1').Get()) == [0]
    poses = np.array([[*q[anchor], 0., 0., 0., 1.]]*2)
    with pytest.raises(ValueError, match='Disable the attachment'):
        item.try_restore(q, poses)
    item.released = True
    assert not item.try_restore(q, poses)  # USD says 5m away; actual body intersects.
    assert item.receipt['min_clearance_m'] == 0.
    poses[1, 2] = .1
    assert item.try_restore(q, poses)
    assert not item.active and filters.active_count == 0
    assert item.receipt['min_clearance_m'] >= .01
    assert prim.GetAttribute('omniphysics:filterEnabled').Get() is False
    assert all(v is False for v in item.receipt['filter_enabled_readback'].values())
    assert not item.receipt['native_filter_effect_independently_verified']
    assert len(flushed) == 2


def test_failed_bind_leaves_no_enabled_partial_filter():
    env, q, triangles, anchor = scene()
    calls = []
    def flush():
        calls.append(True)
        if len(calls) == 1:
            raise RuntimeError('native flush failed')
    filters = LocalGripFilters(env, q, triangles, flush_changes=flush)
    with pytest.raises(ValueError):
        filters.bind('left', [], q, '/World/BadEmpty')
    assert not env.stage.GetPrimAtPath('/World/BadEmpty')
    with pytest.raises(RuntimeError, match='native flush failed'):
        filters.bind('left', [anchor], q, '/World/FailedFilter')
    assert not env.stage.GetPrimAtPath('/World/FailedFilter')
    assert filters.active_count == 0 and not filters.receipt['items']


def test_collision_topology_must_match_real_mesh():
    env, q, triangles, _ = scene()
    with pytest.raises(ValueError, match='actual collision mesh topology'):
        LocalGripFilters(env, q, triangles[:, ::-1], flush_changes=lambda: None)


def test_restore_gap_checks_triangle_interiors_not_only_vertices():
    q = np.array([[-.1, -.1, 0], [.1, -.1, 0], [0, .1, 0]])
    boxes = np.array([[[-.01, -.01, -.01], [.01, .01, .01]]])
    assert triangle_box_clearance(q, [[0, 1, 2]], [0], boxes) == 0.


def test_native_xyzw_rotation_and_missing_pose_rejection():
    binding = {'body_index': 0, 'body_local_corners_m': [[1., 0., 0.], [2., 0., 0.]]}
    poses = np.array([[0, 0, 0, 0, 0, np.sqrt(.5), np.sqrt(.5)]])
    bounds = native_collider_boxes([binding], poses, 1)
    np.testing.assert_allclose(bounds[0, :, :2], [[0, 1], [0, 2]], atol=1.e-12)
    with pytest.raises(ValueError, match='Current native body poses'):
        native_collider_boxes([binding], None, 1)
    poses[0, 3:] = 0.
    with pytest.raises(ValueError, match='normalized xyzw'):
        native_collider_boxes([binding], poses, 1)
