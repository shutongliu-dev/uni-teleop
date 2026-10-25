"""CPU geometry/schema checks, never native button or manipulation acceptance."""
import json

import numpy as np
import pytest
import trimesh
from pxr import Gf, Usd, UsdGeom, UsdPhysics

from scenes.workrooms.living_room.tasks.remote_operation.assets import (
    BUTTON_CENTRES_M, BUTTON_TRAVEL_M, author_task,
)


def built(presentation=False):
    stage = Usd.Stage.CreateInMemory()
    return stage, author_task(stage, presentation=presentation)


def mesh_data(prim, relative=None):
    mesh = UsdGeom.Mesh(prim)
    points = np.asarray(mesh.GetPointsAttr().Get(), dtype=float)
    if relative is not None:
        matrix = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        inverse = UsdGeom.Xformable(relative).ComputeLocalToWorldTransform(Usd.TimeCode.Default()).GetInverse()
        points = np.array([inverse.Transform(matrix.Transform(Gf.Vec3d(*p))) for p in points])
    indices, faces, at = list(mesh.GetFaceVertexIndicesAttr().Get()), [], 0
    for n in mesh.GetFaceVertexCountsAttr().Get():
        polygon = indices[at:at+n]
        faces.extend([[polygon[0], polygon[i], polygon[i+1]] for i in range(1, n-1)])
        at += n
    return trimesh.Trimesh(points, faces, process=False)


def collide_bounds(a, b, epsilon=1.e-8):
    return all(min(a[1][i], b[1][i])-max(a[0][i], b[0][i]) > epsilon for i in range(3))


def test_metadata_and_actual_rigid_topology_without_scene_or_world_anchor():
    stage, meta = built()
    assert json.loads(json.dumps(meta)) == meta
    assert meta['native_qualified'] is False and meta['native_execution_available'] is False
    bodies = [p for p in stage.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI)]
    joints = [p for p in stage.Traverse() if p.IsA(UsdPhysics.PrismaticJoint)]
    assert len(bodies) == 4 and len(joints) == 2
    assert not any(p.IsA(UsdPhysics.Scene) or p.IsA(UsdPhysics.FixedJoint) or p.HasAPI(UsdPhysics.ArticulationRootAPI) for p in stage.Traverse())
    for body in bodies:
        assert not UsdPhysics.RigidBodyAPI(body).GetKinematicEnabledAttr().Get()
        assert UsdPhysics.MassAPI(body).GetMassAttr().Get() > 0
        assert all(i > 0 for i in UsdPhysics.MassAPI(body).GetDiagonalInertiaAttr().Get())
    assert sum(UsdPhysics.MassAPI(p).GetMassAttr().Get() for p in bodies if '/Remote/' in str(p.GetPath())) == pytest.approx(.116)


def test_each_collision_component_is_closed_positive_and_actually_convex():
    stage, _ = built()
    colliders = [p for p in stage.Traverse() if p.HasAPI(UsdPhysics.CollisionAPI)]
    assert len(colliders) == 19
    for prim in colliders:
        mesh = mesh_data(prim)
        assert mesh.is_watertight and mesh.is_winding_consistent, str(prim.GetPath())
        assert mesh.is_convex and mesh.volume > 0, str(prim.GetPath())
        assert UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get() == 'convexHull'
        assert prim.GetAttribute('physxCollision:contactOffset').Get() == pytest.approx(.0001)
        assert prim.GetAttribute('physxCollision:restOffset').Get() == 0


@pytest.mark.parametrize('presentation', [False, True])
def test_joint_rest_anchors_match_and_passive_drive_has_no_animation(presentation):
    stage, meta = built(presentation)
    cache = UsdGeom.XformCache()
    for button in meta['buttons'].values():
        joint = UsdPhysics.PrismaticJoint(stage.GetPrimAtPath(button['joint_path']))
        a = stage.GetPrimAtPath(joint.GetBody0Rel().GetTargets()[0])
        b = stage.GetPrimAtPath(joint.GetBody1Rel().GetTargets()[0])
        p0 = cache.GetLocalToWorldTransform(a).Transform(Gf.Vec3d(joint.GetLocalPos0Attr().Get()))
        p1 = cache.GetLocalToWorldTransform(b).Transform(Gf.Vec3d(joint.GetLocalPos1Attr().Get()))
        assert (p0-p1).GetLength() < 1.e-8
        assert joint.GetAxisAttr().Get() == 'Z'
        assert joint.GetLowerLimitAttr().Get() == pytest.approx(-.002)
        assert joint.GetUpperLimitAttr().Get() == 0
        assert joint.GetCollisionEnabledAttr().Get() is False
        drive = UsdPhysics.DriveAPI(joint.GetPrim(), 'linear')
        assert drive.GetTypeAttr().Get() == 'force'
        assert drive.GetTargetPositionAttr().Get() == drive.GetTargetVelocityAttr().Get() == 0
        assert drive.GetStiffnessAttr().Get() == 600
        assert drive.GetDampingAttr().Get() == pytest.approx(1.2)
        assert drive.GetStiffnessAttr().Get()*BUTTON_TRAVEL_M == pytest.approx(1.2)
    assert all(not attr.GetTimeSamples() for p in stage.Traverse() for attr in p.GetAttributes())


def test_buttons_have_real_apertures_and_full_stroke_without_body_penetration():
    stage, meta = built(True)
    rig = stage.GetPrimAtPath(meta['remote']['assembly_path'])
    body = stage.GetPrimAtPath(meta['remote']['body_path'])
    body_bounds = [mesh_data(p, rig).bounds for p in Usd.PrimRange(body) if p.HasAPI(UsdPhysics.CollisionAPI)]
    for path in meta['remote']['button_paths'].values():
        cap = stage.GetPrimAtPath(path+'/Cap')
        box = mesh_data(cap, rig).bounds
        sweep = box.copy(); sweep[0, 2] -= BUTTON_TRAVEL_M
        assert not any(collide_bounds(sweep, other) for other in body_bounds)
    # Open apertures must stay empty all the way from the core face to the rim.
    for center in BUTTON_CENTRES_M.values():
        for z in (.007, .009):
            probe = np.array([[center[0]-.001, center[1]-.001, z-.0001],
                              [center[0]+.001, center[1]+.001, z+.0001]])
            assert not any(collide_bounds(probe, other) for other in body_bounds)


def test_face_down_guards_prevent_initial_button_contact_with_table():
    stage, meta = built()
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ['default'])
    for path in meta['remote']['button_paths'].values():
        bound = cache.ComputeWorldBound(stage.GetPrimAtPath(path)).ComputeAlignedRange()
        assert bound.GetMin()[2] > .7806
    body = stage.GetPrimAtPath(meta['remote']['body_path'])
    guard_bottoms = [cache.ComputeWorldBound(stage.GetPrimAtPath(str(body.GetPath())+'/'+n)).ComputeAlignedRange().GetMin()[2]
                    for n in ('FrontGuard', 'RearGuard')]
    assert guard_bottoms == pytest.approx([.78015, .78015], abs=1.e-7)
    assert meta['remote']['face_up'] is False
    presented, shown = built(True)
    assert shown['remote']['face_up'] is True and shown['presentation'] is True
    assert shown['native_qualified'] is False
    assert cache.ComputeWorldBound(body).ComputeAlignedRange().GetMin()[2] > .78


def test_holder_opening_really_accepts_full_remote_envelope():
    stage, meta = built()
    holder = stage.GetPrimAtPath(meta['holder']['body_path'])
    walls = [mesh_data(p, holder).bounds for p in Usd.PrimRange(holder)
             if p.HasAPI(UsdPhysics.CollisionAPI) and p.GetName() not in ('Base', 'Liner')]
    remote_bounds = np.array([[-.0275, -.080, .007], [.0275, .080, .007+.0239]])
    assert not any(collide_bounds(remote_bounds, wall) for wall in walls)
    free_interior = np.array([[-.0334, -.0859, .0071], [.0334, .0859, .0259]])
    all_bounds = [mesh_data(p, holder).bounds for p in Usd.PrimRange(holder) if p.HasAPI(UsdPhysics.CollisionAPI)]
    assert not any(collide_bounds(free_interior, box) for box in all_bounds)
    base = mesh_data(stage.GetPrimAtPath(str(holder.GetPath())+'/Base'), holder)
    assert base.bounds[0, 2] == pytest.approx(0., abs=1.e-8)
    assert meta['holder']['opening_xy_m'] == [.067, .172]


def test_parts_stay_inside_original_table_and_separate_at_initialization():
    stage, meta = built()
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ['default'])
    remote = cache.ComputeWorldBound(stage.GetPrimAtPath(meta['remote']['assembly_path'])).ComputeAlignedRange()
    holder = cache.ComputeWorldBound(stage.GetPrimAtPath(meta['holder']['body_path'])).ComputeAlignedRange()
    for box in (remote, holder):
        assert .15 < box.GetMin()[0] < box.GetMax()[0] < 1.15
        assert -.5 < box.GetMin()[1] < box.GetMax()[1] < .5
        assert box.GetMin()[2] >= .78-1.e-7
    assert remote.GetMax()[0] < holder.GetMin()[0]


def test_roundtrip_preserves_topology_and_refuses_overwrite():
    stage, meta = built()
    layer = stage.GetRootLayer().ExportToString()
    target = Usd.Stage.CreateInMemory()
    target.GetRootLayer().ImportFromString(layer)
    assert target.GetPrimAtPath(meta['buttons']['power']['joint_path']).IsA(UsdPhysics.PrismaticJoint)
    assert target.GetPrimAtPath('/World/LivingTask').GetCustomDataByKey('native_qualified') is False
    with pytest.raises(ValueError, match='overwrite'):
        author_task(stage)
    assert stage.GetRootLayer().ExportToString() == layer
    for path in ('/', 'relative/path', '/World.foo'):
        with pytest.raises(ValueError):
            author_task(Usd.Stage.CreateInMemory(), root_path=path)
