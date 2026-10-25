"""Original-size opposed FR3/right-SHARPA meshes in a static preview layer.

The fixed room and source robot assets remain immutable. This composition owns
only presentation poses and stands; it is neither a task nor collision or
manipulation qualification. The two right hands are rotated, never reflected.
"""
from __future__ import annotations

import numpy as np


ROBOT_ROOTS = {'front': '/World/TaskRobots/front', 'rear': '/World/TaskRobots/rear'}
PREVIEW_CAMERAS = {
    'room': {'eye': [2.45, -1.50, 1.80], 'target': [-.10, .55, .65], 'focal_length_mm': 22.},
    'workcell': {'eye': [-1.65, -1.50, 1.80], 'target': [.75, .55, 1.05], 'focal_length_mm': 22.},
}


def _stand_name(side, name):
    return name if name.startswith(side + '_') else side + '_' + name


def add_robot_preview(stage, builder, *, base_half_spacing_m=None, run_clearance_audit=True):
    """Pose two independent pinned asset references, with dynamics disabled.

    ``dual_robot_layout`` owns the bases, same source ready joints and stand
    dimensions. No controller, solver, robot scale or original asset changes.
    ``run_clearance_audit=False`` is only for retaining diagnostic candidates.
    """
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics
    from . import dual_robot_layout as layout

    if stage.GetPrimAtPath('/World/TaskRobot') or stage.GetPrimAtPath('/World/TaskRobots'):
        raise ValueError('Dual preview requires a room without a previous task robot composition')
    configurations = layout.robot_configurations(base_half_spacing_m=base_half_spacing_m)
    stands = layout.stand_geometry(base_half_spacing_m=base_half_spacing_m)
    if set(configurations) != set(ROBOT_ROOTS) or set(stands) != set(ROBOT_ROOTS):
        raise ValueError('Dual layout must provide independent front/rear robots and stands')
    source = layout.robot_asset_reference()
    for key in ('root_usd', 'source_manifest', 'import_manifest'):
        builder.dependency(source[key])
    builder.material('dual_preview_stand_paint', (.105, .16, .20), .50, .18)
    builder.material('dual_preview_stand_steel', (.35, .38, .40), .40, .8)
    UsdGeom.Xform.Define(stage, '/World/TaskRobots')
    records = {}
    for side, root_path in ROBOT_ROOTS.items():
        robot = configurations[side]
        initial, design = layout.ready_configuration(robot)
        world = layout.link_world_transforms(robot, initial)
        root = UsdGeom.Xform.Define(stage, root_path).GetPrim()
        root.GetReferences().AddReference(source['root_usd'])
        targets = {}
        for prim in Usd.PrimRange(root):
            if prim.GetName() in world and prim.IsA(UsdGeom.Xform):
                targets[str(prim.GetPath())] = np.asarray(world[prim.GetName()], dtype=float)
        if not targets:
            raise ValueError('No original source FK link targets for ' + side)
        cache = UsdGeom.XformCache(Usd.TimeCode.Default())
        for path in sorted(targets, key=lambda value: (value.count('/'), value)):
            prim = stage.GetPrimAtPath(path)
            parent = cache.GetLocalToWorldTransform(prim.GetParent())
            transform = UsdGeom.Xformable(prim)
            transform.ClearXformOpOrder()
            transform.AddTransformOp(UsdGeom.XformOp.PrecisionDouble, 'dualPreviewPose').Set(
                Gf.Matrix4d(targets[path].T.tolist()) * parent.GetInverse())
            cache.Clear()
        readback, body_count, joint_count = {}, 0, 0
        for prim in Usd.PrimRange(root):
            path = str(prim.GetPath())
            if path in targets:
                actual = np.asarray(cache.GetLocalToWorldTransform(prim)).T
                np.testing.assert_allclose(actual, targets[path], atol=1.e-7, rtol=0.)
                np.testing.assert_allclose(np.linalg.det(actual[:3, :3]), 1., atol=1.e-7, rtol=0.)
                readback[prim.GetName()] = actual.tolist()
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                if path not in targets:
                    raise ValueError('Source body absent from original FK map: ' + path)
                UsdPhysics.RigidBodyAPI(prim).CreateRigidBodyEnabledAttr(False)
                body_count += 1
            if prim.IsA(UsdPhysics.Joint):
                UsdPhysics.Joint(prim).CreateJointEnabledAttr(False)
                joint_count += 1
            if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
                prim.CreateAttribute('physxArticulation:articulationEnabled', Sdf.ValueTypeNames.Bool).Set(False)
        if body_count != 43 or not world.keys() <= readback.keys():
            raise ValueError('Incomplete original robot FK/body composition for ' + side)
        stand_records = []
        for item in stands[side]['boxes']:
            name = _stand_name(side, item['name'])
            material = ('dual_preview_stand_paint' if item['material'] == 'powder_coated_steel'
                        else 'dual_preview_stand_steel')
            path = builder.box(name, item['position_m'], item['size_m'], material, .004)
            stand_records.append({**item, 'name': name, 'prim_path': path})
        x, y, _ = robot['base_position_m']
        foot = next(item for item in stands[side]['boxes'] if item['name'].endswith('robot_stand_foot'))
        anchor_z = float(foot['position_m'][2]) + float(foot['size_m'][2]) / 2 + .005
        for i, dx in enumerate((-.16, .16)):
            for j, dy in enumerate((-.14, .14)):
                builder.cylinder(f'{side}_preview_anchor_{i}_{j}',
                    (x + dx, y + dy, anchor_z), .012, .010, 'chrome', collision=False)
        records[side] = {'root_path': root_path, 'robot': robot, 'initial_state': initial,
            'ready_design': design, 'stand': {**stands[side], 'boxes': stand_records},
            'source_rigid_body_count': body_count, 'disabled_joint_count': joint_count,
            'posed_link_count': len(readback), 'link_world_transforms': readback}
    if records['front']['initial_state'] != records['rear']['initial_state']:
        raise ValueError('Opposed robots must preserve identical ready joint positions')
    for prim in stage.Traverse():
        if prim.IsA(UsdPhysics.Scene):
            raise ValueError('Static dual preview must not contain a PhysicsScene')
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            if UsdPhysics.RigidBodyAPI(prim).GetRigidBodyEnabledAttr().Get() is not False:
                raise ValueError('Static preview contains an enabled rigid body: ' + str(prim.GetPath()))
    half_spacing = abs(float(configurations['front']['base_position_m'][1]))
    audit = (clearance_audit(stage) if run_clearance_audit else {
        'status': 'not_run', 'scope': 'Diagnostic candidate only; clearance has not been checked'})
    return {'schema': 'living_room_dual_static_composition.v1', 'layout_id': layout.LAYOUT_ID,
        'base_half_spacing_m': half_spacing, 'robots': records, 'robot_asset': source,
        'source_rigid_body_count': sum(r['source_rigid_body_count'] for r in records.values()),
        'posed_link_count': sum(r['posed_link_count'] for r in records.values()),
        'robot_count': 2, 'same_ready_joint_positions': True, 'robot_scale': [1., 1., 1.],
        'hand_sides': ['right', 'right'], 'clearance_audit': audit,
        'pose_source': 'Original CPU URDF FK, independently read back from both new USD roots',
        'physics_steps': 0, 'native_contact_qualified': False,
        'scope': 'Static two-robot room illustration; no self/swept collision, task or native dynamics qualification'}


def clearance_audit(stage, *, raise_on_failure=True):
    """Check actual source shape AABBs at the authored ready pose.

    A positive AABB gap certifies separation of those shapes in this one pose.
    A zero gap is conservative and may be a false positive for concave meshes;
    it is reported for review, never silently called an actual mesh collision.
    """
    from pxr import Usd, UsdGeom, UsdPhysics
    from .dual_robot_layout import GROUND_DATUM_M, TABLE_TOP_Z_M

    # The imported robot's true collision meshes have purpose=guide. Include
    # them explicitly: default/render/proxy alone yields empty guide bounds.
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ['default', 'render', 'proxy', 'guide'], False, True)

    def bounds(prim):
        box = cache.ComputeWorldBound(prim).ComputeAlignedRange()
        result = np.asarray([box.GetMin(), box.GetMax()], dtype=float)
        if not np.isfinite(result).all() or (result[1] < result[0]).any():
            raise ValueError('Invalid shape bounds: ' + str(prim.GetPath()))
        return result

    def geometry_shapes(root):
        if not root:
            raise ValueError('Missing preview root')
        result = []
        for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
            if not prim.IsA(UsdGeom.Boundable):
                continue
            owner = prim
            while owner and not owner.HasAPI(UsdPhysics.RigidBodyAPI):
                owner = owner.GetParent()
            result.append({'path': str(prim.GetPath()), 'bounds': bounds(prim),
                           'body': owner.GetName() if owner else None})
        return result

    def record_pair(a, b):
        delta = np.maximum(np.maximum(a['bounds'][0] - b['bounds'][1],
                                      b['bounds'][0] - a['bounds'][1]), 0.)
        overlap = np.minimum(a['bounds'][1], b['bounds'][1]) - np.maximum(a['bounds'][0], b['bounds'][0])
        return {'a': a['path'], 'b': b['path'], 'gap_m': float(np.linalg.norm(delta)),
                'aabb_overlap_xyz_m': np.maximum(overlap, 0.).tolist()}

    exceptions, problems, checks = [], [], []

    def check_group(name, left, right, allowed=None):
        if not left or not right:
            raise ValueError('Clearance group has no shapes: ' + name)
        closest = None
        tested = exempted = 0
        for a in left:
            for b in right:
                pair = record_pair(a, b)
                reason = allowed(a, b) if allowed is not None else None
                if reason:
                    exempted += 1
                    if pair['gap_m'] <= 1.e-6:
                        exceptions.append({**pair, 'reason': reason})
                    continue
                tested += 1
                if closest is None or pair['gap_m'] < closest['gap_m']:
                    closest = pair
                if pair['gap_m'] <= 1.e-6:
                    problems.append({**pair, 'check': name, 'finding': 'AABB touching/overlap; exact mesh review required'})
        checks.append({'name': name, 'tested_shape_pairs': tested, 'exempted_shape_pairs': exempted,
                       'closest_pair': closest,
                       'status': 'passed' if closest is None or closest['gap_m'] > 1.e-6 else 'review_required'})

    robots = {side: geometry_shapes(stage.GetPrimAtPath(path)) for side, path in ROBOT_ROOTS.items()}
    stands = {side: [] for side in ROBOT_ROOTS}
    environment = {'table': [], 'architecture': [], 'furniture': [], 'floor_finishes': []}
    geometry = stage.GetPrimAtPath('/World/Geometry')
    if not geometry:
        raise ValueError('Missing fixed room geometry')
    for prim in geometry.GetChildren():
        name = prim.GetName()
        side = next((s for s in ROBOT_ROOTS if name.startswith((s + '_robot_stand_', s + '_preview_anchor_'))), None)
        if side:
            stands[side].extend(geometry_shapes(prim))
            continue
        shapes = geometry_shapes(prim)
        if not shapes:
            continue
        if name == 'worktop' or name.startswith(('table_leg_', 'table_apron_')):
            group = 'table'
        elif name in ('room_floor', 'oak_floor_finish') or name.startswith(('oak_board_', 'living_seating_rug_', 'living_rug_')):
            group = 'floor_finishes'
        elif name.startswith(('wall_', 'room_ceiling', 'skirting_', 'door_', 'window_',
                              'sliding_sash_', 'ceiling_', 'cove_', 'lounge_vent_', 'wall_switch_')):
            group = 'architecture'
        else:
            group = 'furniture'
        environment[group].extend(shapes)
    if any(not shapes for shapes in robots.values()) or any(not shapes for shapes in stands.values()):
        raise ValueError('Dual preview is missing original robot or stand shapes')
    check_group('front_robot_vs_rear_robot', robots['front'], robots['rear'])
    check_group('front_stand_vs_rear_stand', stands['front'], stands['rear'])
    layout = {}
    for side in ROBOT_ROOTS:
        for group, shapes in environment.items():
            check_group(side + '_robot_vs_' + group, robots[side], shapes)
        for other in ROBOT_ROOTS:
            def mount_contact(a, b):
                if side == other and a['body'] == 'fr3_link0' and ('/' + side + '_robot_stand_mount/') in b['path']:
                    return 'Original fixed base is bolted to its own mount plate'
                return None
            check_group(side + '_robot_vs_' + other + '_stand', robots[side], stands[other], mount_contact)
        for group, shapes in environment.items():
            def ground_contact(a, b):
                if ('/' + side + '_robot_stand_foot/') in a['path'] and group == 'floor_finishes':
                    if b['bounds'][1, 2] <= .0125 and abs(a['bounds'][0, 2] - b['bounds'][1, 2]) < 1.e-6:
                        return 'Stand foot bottom meets the actual floor/rug finish top; static support datum only'
                return None
            check_group(side + '_stand_vs_' + group, stands[side], shapes, ground_contact)
        foot_prim = stage.GetPrimAtPath('/World/Geometry/' + side + '_robot_stand_foot/Collider')
        mount_prim = stage.GetPrimAtPath('/World/Geometry/' + side + '_robot_stand_mount/Collider')
        table_prim = stage.GetPrimAtPath('/World/Geometry/worktop/Collider')
        if not foot_prim or not mount_prim or not table_prim:
            raise ValueError('Missing stand or table contract collider')
        foot, mount, table = bounds(foot_prim), bounds(mount_prim), bounds(table_prim)
        np.testing.assert_allclose(foot[1] - foot[0], [.40, .36, .05], atol=1.e-7, rtol=0.)
        ground_z = GROUND_DATUM_M[side]
        np.testing.assert_allclose(foot[0, 2], ground_z, atol=1.e-7, rtol=0.)
        np.testing.assert_allclose(mount[1, 2], TABLE_TOP_Z_M, atol=1.e-7, rtol=0.)
        gap = float(table[0, 1] - foot[1, 1] if side == 'front' else foot[0, 1] - table[1, 1])
        if gap <= 0:
            raise ValueError('Stand foot reaches the table footprint: ' + side)
        layout[side] = {'robot_shape_count': len(robots[side]), 'stand_shape_count': len(stands[side]),
            'robot_bounds_m': bounds(stage.GetPrimAtPath(ROBOT_ROOTS[side])).tolist(),
            'stand_foot_bounds_m': foot.tolist(), 'stand_foot_to_table_edge_m': gap,
            'mount_top_z_m': float(mount[1, 2]), 'ground_datum_m': ground_z,
            'stand_support_scope': 'Authored static finish support only; no native contact qualification'}
    report = {'schema': 'living_room_dual_ready_shape_clearance.v1',
        'status': 'review_required' if problems else 'passed', 'layout': layout,
        'checks': checks, 'aabb_findings': problems, 'explicit_support_exceptions': exceptions,
        'environment_shape_counts': {key: len(value) for key, value in environment.items()},
        'method': 'Pairwise world AABBs of every original USD Boundable, including hidden collision shapes',
        'limitations': ['AABB overlap is conservative and is not an exact triangle collision finding',
                       'Only the authored ready pose is checked; no self-collision or swept motion check',
                       'Mount/floor contact exceptions describe assembly support, not native contact validation',
                       'No collision acceptance, reachability, manipulation or task qualification is claimed'],
        'physics_steps': 0, 'native_contact_qualified': False}
    if problems and raise_on_failure:
        sample = '; '.join(p['a'] + ' vs ' + p['b'] for p in problems[:6])
        raise ValueError('Dual preview has conservative AABB findings requiring review: ' + sample)
    return report
