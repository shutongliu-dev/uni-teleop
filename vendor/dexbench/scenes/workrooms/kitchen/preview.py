"""Static kitchen installation using the original FR3/right-SHARPA meshes."""
from __future__ import annotations

import numpy as np

from .robot_layout import (LAYOUT_ID, link_world_transforms, placement_transform,
                          ready_configuration, robot_configuration, stand_geometry)

PREVIEW_CAMERAS = {
    'room': {'eye': [-1.65, -2.15, 1.95], 'target': [.6, .25, 1.1], 'focal_length_mm': 20.},
    'workcell': {'eye': [2.6, -1.8, 2.05], 'target': [.6, .15, 1.15], 'focal_length_mm': 23.},
    'tabletop': {'eye': [.65, -.3, 2.5], 'target': [.65, .7, .9], 'focal_length_mm': 28.},
}


def _bounds_reader(stage):
    from pxr import Usd, UsdGeom
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ['default', 'render', 'proxy', 'guide'], False, True)

    def bounds(path):
        prim = stage.GetPrimAtPath(path)
        if not prim:
            raise ValueError('Missing kitchen clearance target: ' + str(path))
        extent = cache.ComputeWorldBound(prim).ComputeAlignedRange()
        result = np.array([extent.GetMin(), extent.GetMax()])
        if not np.isfinite(result).all() or (result[1] < result[0]).any():
            raise ValueError('Invalid kitchen bounds: ' + str(path))
        return result
    return bounds


def _support_paths(environment):
    result = {}
    for role, value in environment['support_colliders'].items():
        paths = [value] if isinstance(value, str) else value
        if not isinstance(paths, (list, tuple)) or not paths or not all(isinstance(path, str) for path in paths):
            raise ValueError('Kitchen support paths must be a path or nonempty path list: ' + role)
        result[role] = list(paths)
    if not {'bar', 'preparation', 'cooking', 'floor'} <= result.keys():
        raise ValueError('Declare kitchen bar, preparation, cooking and floor support colliders')
    return result


def _support_bound(bounds, paths):
    shapes = np.asarray([bounds(path) for path in paths])
    return np.asarray([shapes[:, 0].min(axis=0), shapes[:, 1].max(axis=0)])


def audit_environment(stage, environment):
    """Check the declared kitchen's actual static support geometry on the CPU."""
    from pxr import Usd, UsdGeom, UsdPhysics
    if UsdGeom.GetStageMetersPerUnit(stage) != 1. or UsdGeom.GetStageUpAxis(stage) != 'Z':
        raise ValueError('Kitchen must use metres and Z up')
    if environment.get('family') != 'kitchen':
        raise ValueError('Expected kitchen family')
    room = np.asarray(environment['layout']['room']['bounds'], dtype=float)
    if room.shape != (4,) or not np.isfinite(room).all() or not (room[0] < room[1] and room[2] < room[3]):
        raise ValueError('Kitchen room bounds must be [xmin, xmax, ymin, ymax]')
    supports = _support_paths(environment)
    bounds, support_bounds = _bounds_reader(stage), {}
    for role, paths in supports.items():
        for path in paths:
            prim = stage.GetPrimAtPath(path)
            if (not prim or not prim.HasAPI(UsdPhysics.CollisionAPI)
                    or UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get() is False):
                raise ValueError('Missing or disabled kitchen support collider: ' + str(path))
        support_bounds[role] = _support_bound(bounds, paths).tolist()
    floor, bar, preparation = (np.asarray(support_bounds[key]) for key in ('floor', 'bar', 'preparation'))
    np.testing.assert_allclose(floor[1, 2], 0., atol=1.e-6, rtol=0.)
    np.testing.assert_allclose(bar, [[-.95, -1.35, .855], [1.85, -.55, .90]], atol=1.e-6, rtol=0.)
    np.testing.assert_allclose(preparation[:, 1], [.57, 1.31], atol=1.e-6, rtol=0.)
    np.testing.assert_allclose(preparation[1, 2], .90, atol=1.e-6, rtol=0.)
    colliders, cache = [], UsdGeom.XformCache(Usd.TimeCode.Default())
    for prim in Usd.PrimRange.Stage(stage, Usd.TraverseInstanceProxies()):
        if (prim.IsA(UsdPhysics.Scene)
                or (prim.HasAPI(UsdPhysics.ArticulationRootAPI)
                    and prim.GetAttribute('physxArticulation:articulationEnabled').Get() is not False)
                or (prim.IsA(UsdPhysics.Joint)
                    and UsdPhysics.Joint(prim).GetJointEnabledAttr().Get() is not False)
                or (prim.HasAPI(UsdPhysics.RigidBodyAPI)
                    and UsdPhysics.RigidBodyAPI(prim).GetRigidBodyEnabledAttr().Get() is not False)):
            raise ValueError('Kitchen environment owns a runtime physics actor: ' + str(prim.GetPath()))
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            matrix = np.asarray(cache.GetLocalToWorldTransform(prim))
            if not np.isfinite(matrix).all() or np.linalg.det(matrix[:3, :3]) <= 0:
                raise ValueError('Invalid kitchen static collision transform: ' + str(prim.GetPath()))
            colliders.append(str(prim.GetPath()))
    if stage.GetPrimAtPath('/World/TaskRobot'):
        raise ValueError('Parent kitchen must not contain a robot preview installation')
    placeholder = stage.GetPrimAtPath('/World/Robot')
    if not placeholder or placeholder.GetChildren() or placeholder.HasAuthoredReferences():
        raise ValueError('Kitchen robot compatibility placeholder must remain empty')
    return {'status': 'passed', 'units': 'metres', 'up_axis': 'Z',
        'room_bounds_xy_m': room.tolist(), 'support_bounds_m': support_bounds,
        'support_colliders': supports, 'collider_count': len(colliders),
        'robot_articulations': 0, 'physics_scenes': 0,
        'scope': 'CPU USD support and ownership checks; no native contact qualification'}


def add_robot_preview(stage, builder, *, environment):
    """Relocate the original robot and stand in a new static USD layer only."""
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
        xform.AddTransformOp(UsdGeom.XformOp.PrecisionDouble, 'kitchenPlacement').Set(local)
        cache.Clear()
    readback, bodies = {}, 0
    for prim in Usd.PrimRange(stage.GetPrimAtPath('/World/TaskRobot')):
        if prim.GetName() in expected and prim.IsA(UsdGeom.Xform):
            matrix = np.asarray(cache.GetLocalToWorldTransform(prim)).T
            np.testing.assert_allclose(matrix, expected[prim.GetName()], atol=1.e-7, rtol=0.)
            readback[prim.GetName()] = matrix.tolist()
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            bodies += 1
            if UsdPhysics.RigidBodyAPI(prim).GetRigidBodyEnabledAttr().Get() is not False:
                raise ValueError('Kitchen preview body is still enabled: ' + str(prim.GetPath()))
        if prim.IsA(UsdPhysics.Joint) and UsdPhysics.Joint(prim).GetJointEnabledAttr().Get() is not False:
            raise ValueError('Kitchen preview joint is still enabled: ' + str(prim.GetPath()))
        if (prim.HasAPI(UsdPhysics.ArticulationRootAPI)
                and prim.GetAttribute('physxArticulation:articulationEnabled').Get() is not False):
            raise ValueError('Kitchen preview articulation is still enabled')
    if bodies != 43 or not expected.keys() <= readback.keys():
        raise ValueError('Incomplete original robot FK/body readback')
    if any(prim.IsA(UsdPhysics.Scene) for prim in stage.Traverse()):
        raise ValueError('Static preview must not contain a PhysicsScene')
    for item in builder.records:
        if item.get('name') in stand_names:
            item['center'] = (placement @ [*item['center'], 1.])[:3].tolist()
            item['rotation'] = [0., 0., 90.]
    audit = clearance_audit(stage, environment)
    return {**composition, 'schema': 'kitchen_static_composition.v1',
        'layout_id': LAYOUT_ID, 'robot': robot, 'initial_state': initial,
        'ready_design': ready, 'stand': stand_geometry(), 'link_world_transforms': readback,
        'posed_link_count': len(readback), 'source_rigid_body_count': bodies,
        'source_layout_id': composition['ready_design']['layout_id'],
        'source_to_kitchen_world_matrix': placement.tolist(), 'clearance_audit': audit,
        'physics_steps': 0, 'native_contact_qualified': False,
        'scope': 'Static kitchen and original robot meshes; no task execution qualification'}


def clearance_audit(stage, environment):
    """Check each actual USD shape; aggregate arm bounds contain empty space."""
    from pxr import Usd, UsdGeom
    bounds = _bounds_reader(stage)
    robot_root = stage.GetPrimAtPath('/World/TaskRobot')
    shapes = [(str(prim.GetPath()), bounds(prim.GetPath()))
        for prim in Usd.PrimRange(robot_root, Usd.TraverseInstanceProxies())
        if prim.IsA(UsdGeom.Boundable)]
    stand_names = [item['name'] for item in stand_geometry()['boxes']]
    anchor_names = [f'preview_anchor_{i}_{j}' for i in range(2) for j in range(2)]
    stand = [('/World/Geometry/' + name, bounds('/World/Geometry/' + name))
             for name in (*stand_names, *anchor_names)]
    own_roots = {path for path, _ in stand}
    supports = _support_paths(environment)
    floor = _support_bound(bounds, supports['floor'])
    obstacles, floor_shapes = [], []
    for prim in Usd.PrimRange(stage.GetPrimAtPath('/World/Geometry'), Usd.TraverseInstanceProxies()):
        path = str(prim.GetPath())
        if not prim.IsA(UsdGeom.Boundable) or any(path == own or path.startswith(own + '/') for own in own_roots):
            continue
        box = bounds(path)
        if box[1, 2] <= floor[1, 2] + 1.e-6:
            floor_shapes.append(path)
        else:
            obstacles.append((path, box))
    if not shapes or not obstacles or not floor_shapes:
        raise ValueError('Missing original robot shapes, kitchen obstacles or floor')

    def nearest(subjects):
        pairs = []
        for subject_path, subject in subjects:
            for obstacle_path, obstacle in obstacles:
                delta = np.maximum(np.maximum(subject[0] - obstacle[1], obstacle[0] - subject[1]), 0.)
                gap = float(np.linalg.norm(delta))
                pairs.append((gap, subject_path, obstacle_path))
        gap, subject_path, obstacle_path = min(pairs)
        if gap <= .005:
            raise ValueError(f'Kitchen preview shape clearance <= 5 mm: {subject_path}, {obstacle_path}, {gap}')
        return {'gap_m': gap, 'subject': subject_path, 'obstacle': obstacle_path}

    robot_nearest, stand_nearest = nearest(shapes), nearest(stand)
    robot = bounds('/World/TaskRobot')
    foot, mount = (bounds('/World/Geometry/' + name) for name in ('robot_stand_foot', 'robot_stand_mount'))
    np.testing.assert_allclose(foot[0, 2], floor[1, 2], atol=1.e-6, rtol=0.)
    np.testing.assert_allclose(mount[1, 2], .78, atol=1.e-6, rtol=0.)
    xmin, xmax, ymin, ymax = environment['layout']['room']['bounds']
    for name, box in [('/World/TaskRobot', robot), *stand]:
        if not (xmin < box[0, 0] <= box[1, 0] < xmax and ymin < box[0, 1] <= box[1, 1] < ymax):
            raise ValueError('Kitchen installation extends outside room: ' + name)
    prep = _support_bound(bounds, supports['preparation'])
    bar = _support_bound(bounds, supports['bar'])
    front_gap, rear_gap = float(prep[0, 1] - foot[1, 1]), float(foot[0, 1] - bar[1, 1])
    if min(front_gap, rear_gap) < .30:
        raise ValueError('Kitchen stand foot lacks 30 cm clearance to both worktop edges')
    return {'status': 'passed', 'robot_bounds_m': robot.tolist(),
        'stand_foot_bounds_m': foot.tolist(), 'mount_top_z_m': float(mount[1, 2]),
        'foot_to_preparation_edge_m': front_gap, 'foot_to_bar_edge_m': rear_gap,
        'robot_nearest_environment_shape': robot_nearest,
        'stand_nearest_environment_shape': stand_nearest,
        'robot_shape_count': len(shapes), 'obstacle_shape_count': len(obstacles),
        'floor_shapes_excluded_from_obstacle_pairs': floor_shapes,
        'scope': 'CPU conservative actual-USD shape bounds only; no self-collision, swept-path or native contact qualification'}
