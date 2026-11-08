"""Resolution-only spoon SDF candidate; no geometry or mechanical changes."""
import numpy as np


def refine_spoon_sdf(author):
    from pxr import UsdGeom
    for row in author.meta['objects']:
        if row['role'] != 'spoon':
            continue
        path = row['body_path']+'/Collision/utensil'
        prim = author.stage.GetPrimAtPath(path)
        attr = prim.GetAttribute('physxSDFMeshCollision:sdfResolution')
        if prim.GetAttribute('physics:approximation').Get() != 'sdf' or attr.Get() != 384:
            raise ValueError('Expected original384 spoon SDF for controlled resolution comparison')
        extent = float(np.ptp(np.asarray(UsdGeom.Mesh(prim).GetPointsAttr().Get()), axis=0).max())
        attr.Set(768)
        row['sdf_sampling_candidate'] = {
            'collider_path': path, 'previous_resolution': 384, 'resolution': 768,
            'longest_mesh_extent_m': extent,
            'nominal_grid_spacing_before_m': extent/384,
            'nominal_grid_spacing_after_m': extent/768,
            'physical_mesh_mass_COM_inertia_material_unchanged': True,
            'scope': 'Numerical collision sampling only; requires separate native validation.'}
