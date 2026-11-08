"""Verified nonphysical robot frames for sensors, separate from rigid bodies."""
from __future__ import annotations

import numpy as np


def _source_rows(robot, source):
    if source is not None:
        return list(source['sides'].values()) if 'sides' in source else [source]
    # Direct USD composition utilities may not pass import_robot's already
    # verified source. Only the explicitly selected new version needs frames.
    from .l_mount import SUPPORTED_VERSIONS, prepared_assembly
    from .robot_asset import DEFAULT_ASSETS
    from .robot_instances import ROBOT_ID, checked_instances
    assets = robot.get('assets_root', DEFAULT_ASSETS)
    if robot.get('hand_id') == ROBOT_ID:
        from .instance_asset import assemble_instance
        return [assemble_instance(name, assets, source_version=row['assembly_version'])[1]
                for name, row in checked_instances(robot).items() if row['assembly_version'] in SUPPORTED_VERSIONS]
    if robot.get('assembly_version') not in SUPPORTED_VERSIONS:
        return []
    sides = ('left', 'right') if robot.get('hand_id') == 'dual_fr3_sharpa_w01' else (None,)
    return [prepared_assembly(assets, side=side, version=robot['assembly_version'])[1] for side in sides]


def nonphysical_frame_paths(stage, robot, rigid_link_paths, source=None):
    """Check the actual composed pure Xforms against their source declarations.

    This does not author anything or add a native tensor/contact sensor. A
    missing, renamed, reparented, reset-stack or physical frame is rejected.
    """
    from pxr import Usd, UsdGeom, UsdPhysics
    result = {}
    for row in _source_rows(robot, source):
        for name, spec in row.get('virtual_frames', {}).items():
            if (name in result or name in rigid_link_paths or spec.get('rigid_body') is not False
                    or spec.get('mass_kg') != 0. or spec.get('parent_body') not in rigid_link_paths):
                raise ValueError('Virtual robot frame conflicts with its physical source: '+name)
            parent_path = rigid_link_paths[spec['parent_body']]
            parent = stage.GetPrimAtPath(parent_path)
            prim = stage.GetPrimAtPath(parent_path+'/'+name)
            if (not parent or not parent.HasAPI(UsdPhysics.RigidBodyAPI)
                    or not prim or not prim.IsActive() or not prim.IsDefined() or not prim.IsLoaded()
                    or not prim.IsA(UsdGeom.Xform)):
                raise ValueError('Virtual robot frame is absent from its declared parent: '+name)
            for item in Usd.PrimRange(prim):
                if any(item.HasAPI(api) for api in (UsdPhysics.RigidBodyAPI, UsdPhysics.MassAPI, UsdPhysics.CollisionAPI)):
                    raise ValueError('Virtual robot frame has physical properties: '+name)
            xform = UsdGeom.Xformable(prim)
            expected = np.asarray(spec['matrix_column_vector'], dtype=float)
            actual = np.asarray(xform.GetLocalTransformation()).T
            if (xform.GetResetXformStack() or expected.shape != (4, 4)
                    or not np.isfinite(expected).all()
                    or not np.allclose(actual, expected, rtol=0., atol=1e-12)):
                raise ValueError('Virtual robot frame transform differs from its source: '+name)
            result[name] = str(prim.GetPath())
    return result
