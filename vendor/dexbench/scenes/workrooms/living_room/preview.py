"""Compose the original robot meshes in a new static living-room preview layer."""
from __future__ import annotations

import numpy as np

from .robot_layout import (LAYOUT_ID, link_world_transforms, placement_transform,
                          ready_configuration, robot_configuration, stand_geometry)

PREVIEW_CAMERAS = {
    'room': {'eye': [2.40, -1.50, 1.65], 'target': [-.10, .80, 1.25], 'focal_length_mm': 20.},
    'workcell': {'eye': [-1.65, -1.50, 1.70], 'target': [.65, .65, 1.13], 'focal_length_mm': 20.},
}


def add_robot_preview(stage, builder):
    """Reuse the source preview, then relocate its robot and stand as one unit.

    Source assets and the fixed room stay unchanged. The reused composer turns
    off bodies/joints/articulation only in this new preview layer, with no GPU,
    simulator, controller execution, runtime pose playback or PhysicsScene.
    """
    from pxr import Gf, Usd, UsdGeom, UsdPhysics
    from scenes.workrooms.mechanical_assembly.preview import add_robot_preview as add_source_preview

    composition = add_source_preview(stage, builder)
    robot = robot_configuration(composition['robot'])
    initial, ready = ready_configuration(robot, composition['initial_state'])
    expected = link_world_transforms(robot, initial)
    placement = placement_transform()
    cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    stand_names = [item['name'] for item in composition['stand']['boxes']]
    anchor_names = [f'preview_anchor_{i}_{j}' for i in range(2) for j in range(2)]
    roots = ['/World/TaskRobot', *('/World/Geometry/' + name for name in (*stand_names, *anchor_names))]
    for path in roots:
        prim = stage.GetPrimAtPath(path)
        current = cache.GetLocalToWorldTransform(prim)
        parent = cache.GetLocalToWorldTransform(prim.GetParent())
        local = current * Gf.Matrix4d(placement.T.tolist()) * parent.GetInverse()
        xform = UsdGeom.Xformable(prim)
        xform.ClearXformOpOrder()
        xform.AddTransformOp(UsdGeom.XformOp.PrecisionDouble, 'livingPlacement').Set(local)
        cache.Clear()
    readback, bodies = {}, 0
    for prim in Usd.PrimRange(stage.GetPrimAtPath('/World/TaskRobot')):
        if prim.GetName() in expected and prim.IsA(UsdGeom.Xform):
            matrix = np.asarray(cache.GetLocalToWorldTransform(prim)).T
            np.testing.assert_allclose(matrix, expected[prim.GetName()], atol=1e-7, rtol=0.)
            readback[prim.GetName()] = matrix.tolist()
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            bodies += 1
            if UsdPhysics.RigidBodyAPI(prim).GetRigidBodyEnabledAttr().Get() is not False:
                raise ValueError('Preview body is still enabled: ' + str(prim.GetPath()))
        if prim.IsA(UsdPhysics.Joint) and UsdPhysics.Joint(prim).GetJointEnabledAttr().Get() is not False:
            raise ValueError('Preview joint is still enabled: ' + str(prim.GetPath()))
        if (prim.HasAPI(UsdPhysics.ArticulationRootAPI)
                and prim.GetAttribute('physxArticulation:articulationEnabled').Get() is not False):
            raise ValueError('Preview articulation is still enabled')
    if bodies != 43 or not expected.keys() <= readback.keys():
        raise ValueError('Incomplete source robot FK/body readback')
    if any(prim.IsA(UsdPhysics.Scene) for prim in stage.Traverse()):
        raise ValueError('Static preview must not contain a PhysicsScene')
    # Keep Builder's newly authored stand records consistent with the USD.
    for item in builder.records:
        if item.get('name') in stand_names:
            item['center'] = (placement @ [*item['center'], 1.])[:3].tolist()
            item['rotation'] = [0., 0., 90.]
    audit = clearance_audit(stage)
    return {**composition, 'schema': 'living_room_static_composition.v1',
        'layout_id': LAYOUT_ID, 'robot': robot, 'initial_state': initial,
        'ready_design': ready, 'stand': stand_geometry(), 'link_world_transforms': readback,
        'posed_link_count': len(readback), 'source_rigid_body_count': bodies,
        'source_layout_id': composition['ready_design']['layout_id'],
        'source_to_living_world_matrix': placement.tolist(), 'clearance_audit': audit,
        'physics_steps': 0, 'native_contact_qualified': False,
        'scope': 'Static room and original robot meshes only; this preview does not qualify task execution'}


def clearance_audit(stage):
    """Conservative actual-USD bounds checks; not native collision acceptance."""
    from pxr import Usd, UsdGeom
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ['default', 'render', 'proxy'], False, True)

    def bounds(path):
        prim = stage.GetPrimAtPath(path)
        if not prim:
            raise ValueError('Missing preview clearance target: ' + path)
        box = cache.ComputeWorldBound(prim).ComputeAlignedRange()
        result = np.array([box.GetMin(), box.GetMax()])
        if not np.isfinite(result).all():
            raise ValueError('Nonfinite preview bounds: ' + path)
        return result

    def distance(a, b):
        return float(np.linalg.norm(np.maximum(np.maximum(a[0] - b[1], b[0] - a[1]), 0.)))

    arm = bounds('/World/TaskRobot')
    # Link0 contains the entire link hierarchy. Its aggregate AABB encloses
    # empty space under the elevated hand, so clearance uses individual USD
    # geometry shapes rather than treating the whole arm envelope as solid.
    shapes = [bounds(prim.GetPath())
              for prim in Usd.PrimRange(stage.GetPrimAtPath('/World/TaskRobot'), Usd.TraverseInstanceProxies())
              if prim.IsA(UsdGeom.Boundable)]
    if not shapes:
        raise ValueError('Robot preview has no boundable source shapes')
    sofa = bounds('/World/Geometry/living_sofa')
    foot = bounds('/World/Geometry/robot_stand_foot')
    mount = bounds('/World/Geometry/robot_stand_mount')
    table = bounds('/World/Geometry/worktop/Collider')
    gap = float(table[0, 1] - foot[1, 1])
    np.testing.assert_allclose(gap, .20, atol=1e-7, rtol=0.)
    np.testing.assert_allclose(foot[0, 2], 0., atol=1e-7, rtol=0.)
    np.testing.assert_allclose(mount[1, 2], .78, atol=1e-7, rtol=0.)
    sofa_gap = min(distance(shape, sofa) for shape in [*shapes, foot])
    table_gap = min(distance(shape, table) for shape in shapes)
    if sofa_gap <= .02 or table_gap <= .001:
        raise ValueError('Robot preview overlaps or approaches the sofa/table bounds')
    return {'status': 'passed', 'stand_foot_to_table_edge_m': gap,
        'robot_or_foot_to_sofa_shape_aabb_gap_m': sofa_gap,
        'robot_to_table_shape_aabb_gap_m': table_gap, 'robot_shape_count': len(shapes),
        'robot_bounds_m': arm.tolist(), 'sofa_bounds_m': sofa.tolist(),
        'stand_foot_bounds_m': foot.tolist(), 'mount_top_z_m': float(mount[1, 2]),
        'scope': 'CPU conservative USD bounds only; no self-collision, swept-path or native contact qualification'}
