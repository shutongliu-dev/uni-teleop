"""Offline presentation composition for the shared single-arm installation.

This is not a task release, reset implementation or physics rollout. It places
the unmodified robot's source meshes at URDF FK coordinates in a new USD layer.
"""
from __future__ import annotations

import numpy as np


def add_robot_preview(stage, builder):
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics
    from .robot_layout import (robot_configuration, ready_configuration,
                               robot_asset_reference, stand_geometry, link_world_transforms)
    robot = robot_configuration()
    initial, design = ready_configuration(robot)
    source = robot_asset_reference(robot)
    world = link_world_transforms(robot, initial)
    root_path = '/World/TaskRobot'
    root = UsdGeom.Xform.Define(stage, root_path).GetPrim()
    root.GetReferences().AddReference(source['root_usd'])
    targets = {}
    for prim in Usd.PrimRange(root):
        if prim.GetName() in world and prim.IsA(UsdGeom.Xform):
            targets[str(prim.GetPath())] = np.asarray(world[prim.GetName()], dtype=float)
    if not targets:
        raise ValueError('Robot preview found no source link transforms')
    cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    for path in sorted(targets, key=lambda value: (value.count('/'), value)):
        prim = stage.GetPrimAtPath(path)
        parent = cache.GetLocalToWorldTransform(prim.GetParent())
        matrix = Gf.Matrix4d(targets[path].T.tolist())
        local = matrix * parent.GetInverse()
        transform = UsdGeom.Xformable(prim)
        transform.ClearXformOpOrder()
        transform.AddTransformOp(UsdGeom.XformOp.PrecisionDouble, 'previewPose').Set(local)
        cache.Clear()
    readback = {}
    bodies = 0
    for prim in Usd.PrimRange(root):
        path = str(prim.GetPath())
        if path in targets:
            actual = np.asarray(cache.GetLocalToWorldTransform(prim)).T
            np.testing.assert_allclose(actual, targets[path], atol=1.e-7, rtol=0.)
            readback[prim.GetName()] = actual.tolist()
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            bodies += 1
            if path not in targets:
                raise ValueError('Robot body is absent from FK map: ' + path)
            UsdPhysics.RigidBodyAPI(prim).CreateRigidBodyEnabledAttr(False)
        if prim.IsA(UsdPhysics.Joint):
            UsdPhysics.Joint(prim).CreateJointEnabledAttr(False)
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            prim.CreateAttribute('physxArticulation:articulationEnabled', Sdf.ValueTypeNames.Bool).Set(False)
    if bodies != 43:
        raise ValueError('Unexpected source robot link count: ' + str(bodies))
    builder.material('preview_stand_paint', (.105, .16, .20), .50, .18)
    builder.material('preview_stand_steel', (.35, .38, .40), .40, .8)
    stand = stand_geometry()
    for item in stand['boxes']:
        material = 'preview_stand_paint' if item['material'] == 'powder_coated_steel' else 'preview_stand_steel'
        builder.box(item['name'], item['position_m'], item['size_m'], material, .004)
    x, y, _ = robot['base_position_m']
    for i, dx in enumerate((-.14, .14)):
        for j, dy in enumerate((-.16, .16)):
            builder.cylinder(f'preview_anchor_{i}_{j}', (x+dx, y+dy, .055), .012, .010, 'chrome', collision=False)
    builder.dependency(source['root_usd'])
    builder.dependency(source['source_manifest'])
    builder.dependency(source['import_manifest'])
    return {'schema': 'mechanical_workcell_static_composition.v1',
            'robot': robot, 'robot_asset': source, 'initial_state': initial,
            'ready_design': design, 'stand': stand, 'source_rigid_body_count': bodies,
            'posed_link_count': len(targets), 'link_world_transforms': readback,
            'pose_source': 'CPU URDF forward kinematics, read back from the new preview USD',
            'physics_steps': 0, 'native_contact_qualified': False,
            'scope': 'Static robot illustration; no dynamics, tool use or task acceptance'}
