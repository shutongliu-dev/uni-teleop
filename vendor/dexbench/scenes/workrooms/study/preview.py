"""Original-size FR3/SHARPA geometry in a separate static study preview."""
from __future__ import annotations

from copy import deepcopy

import numpy as np


LAYOUT_ID = 'study_single_arm_preview_v1'
TABLE_BOUNDS_M = ((.20, -1.00, .70), (1.10, 1.00, .78))


def add_robot_preview(stage, builder):
    """Reuse the original meshes and ready FK without authoring a room robot.

    The source composer disables the referenced bodies, joints and articulation
    in this preview layer only.  Neither source assets nor control parameters
    change, and this illustration does not introduce an executable study task.
    """
    from pxr import Usd, UsdPhysics
    from scenes.workrooms.mechanical_assembly.preview import add_robot_preview as add_source_preview

    composition = add_source_preview(stage, builder)
    source_layout = composition['ready_design']['layout_id']
    ready, stand = deepcopy(composition['ready_design']), deepcopy(composition['stand'])
    ready['layout_id'] = stand['layout_id'] = LAYOUT_ID
    near = TABLE_BOUNDS_M[0][0]
    base_x = composition['robot']['base_position_m'][0]
    parts = {item['name']: item for item in stand['boxes']}
    front = lambda name: parts[name]['position_m'][0] + parts[name]['size_m'][0] / 2
    stand.update(base_center_to_table_edge_m=near - base_x,
                 foot_to_table_edge_m=near - front('robot_stand_foot'),
                 mount_to_table_edge_m=near - front('robot_stand_mount'))
    bodies = 0
    for prim in Usd.PrimRange(stage.GetPrimAtPath('/World/TaskRobot')):
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            bodies += 1
            if UsdPhysics.RigidBodyAPI(prim).GetRigidBodyEnabledAttr().Get() is not False:
                raise ValueError('Study preview body is still enabled: ' + str(prim.GetPath()))
        if prim.IsA(UsdPhysics.Joint) and UsdPhysics.Joint(prim).GetJointEnabledAttr().Get() is not False:
            raise ValueError('Study preview joint is still enabled: ' + str(prim.GetPath()))
        if (prim.HasAPI(UsdPhysics.ArticulationRootAPI)
                and prim.GetAttribute('physxArticulation:articulationEnabled').Get() is not False):
            raise ValueError('Study preview articulation is still enabled')
    if bodies != 43 or any(prim.IsA(UsdPhysics.Scene) for prim in stage.Traverse()):
        raise ValueError('Study preview must have 43 disabled source bodies and no PhysicsScene')
    audit = clearance_audit(stage)
    return {**composition, 'schema': 'study_room_static_composition.v1',
            'layout_id': LAYOUT_ID, 'source_layout_id': source_layout,
            'ready_design': ready, 'stand': stand,
            'table_bounds_m': [list(point) for point in TABLE_BOUNDS_M],
            'clearance_audit': audit, 'physics_steps': 0,
            'native_contact_qualified': False, 'manipulation_qualified': False,
            'scope': 'Static original robot geometry at the shared ready pose; no study task or rear-shelf manipulation qualification.'}


def clearance_audit(stage):
    """Check individual USD shapes against the desktop, desk and shelf parts.

    Empty cubbies and spaces between robot links are not treated as solids.
    Positive AABB distances lower-bound separation of these authored shapes;
    this check does not evaluate a swept path, self-collision or native contact.
    """
    from pxr import Usd, UsdGeom, UsdPhysics

    # Imported robot collision meshes use purpose=guide.  Include them even
    # though the preview renderer hides guides; omitting them gives empty bounds.
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ['default', 'render', 'proxy', 'guide'], False, True)

    def bounds(prim_or_path):
        prim = stage.GetPrimAtPath(prim_or_path) if isinstance(prim_or_path, str) else prim_or_path
        if not prim:
            raise ValueError('Missing study preview clearance target: ' + str(prim_or_path))
        box = cache.ComputeWorldBound(prim).ComputeAlignedRange()
        value = np.array([box.GetMin(), box.GetMax()], dtype=float)
        if not np.isfinite(value).all() or np.any(value[1] < value[0]):
            raise ValueError('Invalid study preview bounds: ' + str(prim.GetPath()))
        return value

    def distance(first, second):
        return float(np.linalg.norm(np.maximum(np.maximum(first[0] - second[1], second[0] - first[1]), 0.)))

    root = stage.GetPrimAtPath('/World/TaskRobot')
    if not root:
        raise ValueError('Missing study preview robot')
    shapes = [(str(prim.GetPath()), bounds(prim))
              for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies())
              if prim.IsA(UsdGeom.Boundable)]
    if not shapes:
        raise ValueError('Study preview has no boundable source robot shapes')
    furniture_shapes = [(str(prim.GetPath()), bounds(prim))
                        for prim in Usd.PrimRange(stage.GetPseudoRoot(), Usd.TraverseInstanceProxies())
                        if (str(prim.GetPath()).startswith(('/World/Geometry/rear_shelf_',
                                                          '/World/Geometry/desk_'))
                            and prim.IsA(UsdGeom.Boundable)
                            and prim.HasAPI(UsdPhysics.CollisionAPI)
                            and UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get() is not False)]
    shelf_shapes = [(path, box) for path, box in furniture_shapes
                    if path.startswith('/World/Geometry/rear_shelf_')]
    desk_shapes = [(path, box) for path, box in furniture_shapes
                   if path.startswith('/World/Geometry/desk_')]
    if len(shelf_shapes) < 6:
        raise ValueError('Study preview is missing solid rear-shelf board colliders')
    if not desk_shapes:
        raise ValueError('Study preview is missing solid desk colliders')
    table = bounds('/World/Geometry/worktop/Collider')
    np.testing.assert_allclose(table, TABLE_BOUNDS_M, atol=1e-7, rtol=0.)
    stand = [(name, bounds('/World/Geometry/' + name + '/Collider')) for name in
             ('robot_stand_foot', 'robot_stand_column', 'robot_stand_mount')]
    foot, mount = stand[0][1], stand[2][1]
    foot_gap = float(table[0, 0] - foot[1, 0])
    np.testing.assert_allclose(foot_gap, .25, atol=1e-7, rtol=0.)
    np.testing.assert_allclose(foot[0, 2], 0., atol=1e-7, rtol=0.)
    np.testing.assert_allclose(mount[1, 2], .78, atol=1e-7, rtol=0.)
    table_gap, nearest_table_shape = min((distance(box, table), path) for path, box in shapes)
    shelf_checks = []
    for path, box in shelf_shapes:
        gap, nearest = min((distance(other, box), name) for name, other in [*shapes, *stand])
        shelf_checks.append({'shelf_collider': path, 'bounds_m': box.tolist(),
                             'minimum_shape_aabb_gap_m': gap, 'nearest_robot_or_stand_shape': nearest})
    shelf_gap = min(item['minimum_shape_aabb_gap_m'] for item in shelf_checks)
    if table_gap <= .001 or shelf_gap <= .02:
        raise ValueError('Study preview robot approaches or overlaps desktop/rear-shelf shape bounds')
    desk_checks = []
    for path, box in desk_shapes:
        gap, nearest = min((distance(other, box), name) for name, other in [*shapes, *stand])
        desk_checks.append({'desk_collider': path, 'bounds_m': box.tolist(),
                            'minimum_shape_aabb_gap_m': gap, 'nearest_robot_or_stand_shape': nearest})
    desk_gap = min(item['minimum_shape_aabb_gap_m'] for item in desk_checks)
    if desk_gap <= .001:
        raise ValueError('Study preview robot or stand approaches or overlaps desk collider shape bounds')
    return {'status': 'passed', 'stand_foot_to_table_edge_m': foot_gap,
            'robot_to_table_shape_aabb_gap_m': table_gap, 'nearest_table_robot_shape': nearest_table_shape,
            'robot_or_stand_to_shelf_shape_aabb_gap_m': shelf_gap,
            'robot_or_stand_to_desk_shape_aabb_gap_m': desk_gap,
            'robot_shape_count': len(shapes), 'shelf_collider_count': len(shelf_shapes),
            'desk_collider_count': len(desk_shapes), 'desk_checks': desk_checks,
            'shelf_checks': shelf_checks, 'robot_bounds_m': bounds(root).tolist(),
            'table_bounds_m': table.tolist(), 'stand_foot_bounds_m': foot.tolist(),
            'mount_top_z_m': float(mount[1, 2]),
            'scope': 'CPU conservative USD bounds at the ready pose, checked per robot/stand shape and enabled desk/shelf collider; excludes self-collision, swept paths, task objects and native contact.'}
