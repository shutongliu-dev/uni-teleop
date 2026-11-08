"""Condition small convex cooking coordinates while preserving body-space geometry."""
import numpy as np
from bench.simulation.isaacsim_backend.asset_catalog import condition_collision_mesh


def condition_spoon_convex_coordinates(author):
    """Use a binary exact local scale; never rescale a body, visual or mass."""
    from pxr import Gf, UsdGeom, Vt
    for row in author.meta['objects']:
        if row['role'] != 'spoon':
            continue
        if not row.get('spoon_convex_collision_partition') or row['collision_count'] != 220:
            raise ValueError('Expected original-corner full convex spoon partition')
        for path in row['collider_paths']:
            mesh = UsdGeom.Mesh(author.stage.GetPrimAtPath(path))
            if UsdGeom.Xformable(mesh).GetOrderedXformOps():
                raise ValueError('Conditioning requires untransformed body-local spoon mesh')
            original = np.asarray(mesh.GetPointsAttr().Get())
            conditioned, inverse = condition_collision_mesh(original, 512.)
            if not np.array_equal(conditioned*np.asarray(inverse, dtype=np.float32), original):
                raise ValueError('Collision-coordinate round trip changed physical corners')
            mesh.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(conditioned))
            mesh.GetExtentAttr().Set([Gf.Vec3f(*conditioned.min(0).tolist()),
                                     Gf.Vec3f(*conditioned.max(0).tolist())])
            UsdGeom.Xformable(mesh).AddScaleOp().Set(Gf.Vec3f(*inverse))
        row['collision_coordinate_conditioning'] = {
            'factor': 512., 'inverse_prim_scale': [1./512.]*3,
            'physical_float32_corners_roundtrip_exact': True,
            'body_mass_inertia_material_visual_or_offset_changes': False,
            'scope': 'Only convex mesh local coordinates and reciprocal child scale; real body-space geometry is unchanged. Native cooked fidelity and dynamics require separate verification.'}
