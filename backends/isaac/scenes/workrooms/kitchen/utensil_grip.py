"""Opt-in thicker cutlery handles, preserving the existing food-contact ends.

The upper grip is part of the original rigid body and is visible contact
geometry. It touches the original upper face without overlapping its volume.
Historical spoon bowl and fork/knife collision recipes remain unchanged.
"""
import numpy as np
import trimesh
from pxr import Gf, Usd, UsdGeom, UsdPhysics, UsdShade, Vt

from bench.simulation.isaacsim_backend.asset_catalog import condition_collision_mesh
from . import geometry as g

PROFILE = 'thick_handle_v1'


def grip_mesh(kind):
    """Metric recipe before household scaling and any spoon rest transform."""
    z = .008 if kind == 'spoon' else .006
    # This entire footprint lies inside the old handle. Its positive-Y end is
    # rounded; the food-contact end at negative Y is never modified.
    xy = np.array([[-.006, .000], [-.007, .040], [-.008, .063],
                   [-.007, .071], [-.004, .076], [.004, .076],
                   [.007, .071], [.008, .063], [.007, .040], [.006, .000]])
    center = np.array([0., .038])
    top = center+(xy-center)*np.array([.82, .95])
    vertices = np.vstack([np.c_[xy, np.full(len(xy), z)],
                          np.c_[xy, np.full(len(xy), z+.0045)],
                          np.c_[top, np.full(len(xy), z+.006)]])
    return trimesh.convex.convex_hull(vertices)


def _body_mesh(stage, path, body):
    prim = stage.GetPrimAtPath(path)
    usd = UsdGeom.Mesh(prim)
    mesh = trimesh.Trimesh(np.asarray(usd.GetPointsAttr().Get(), dtype=float),
        np.asarray(usd.GetFaceVertexIndicesAttr().Get()).reshape(-1, 3), process=False)
    transform, _ = UsdGeom.XformCache().ComputeRelativeTransform(prim, body)
    mesh.apply_transform(np.asarray(transform).T)
    return mesh


def apply_thick_handles(author):
    stage, meta = author.stage, author.meta
    if meta.get('utensil_contact_profile') != 'floor_grid_2mm_splitbowl_sdf768_v7':
        raise ValueError('Thicker handles require the current split-bowl contact profile')
    changed = []
    for row in meta['objects']:
        kind = row['role']
        if kind not in ('fork', 'knife', 'spoon'):
            continue
        body = stage.GetPrimAtPath(row['body_path'])
        old_meshes = [_body_mesh(stage, p, body) for p in row['collider_paths']]
        old_solid = trimesh.util.concatenate(old_meshes)
        rest = np.asarray(row.get('natural_rest', {}).get('source_to_rest_transform', np.eye(4)))
        scale = np.asarray(row['dimension_profile']['linear_scale_xyz'])
        cap = grip_mesh(kind)
        cap.apply_transform(rest)
        cap.apply_scale(scale)
        visual_path = row['body_path']+'/Visual/handle_grip'
        g.write_mesh(stage, visual_path, cap, author.materials['steel'])
        path = row['body_path']+'/Collision/handle_grip_convex'
        points, inverse = condition_collision_mesh(cap.vertices, 512.)
        collision = trimesh.Trimesh(points, cap.faces, process=False)
        prim = g.write_mesh(stage, path, collision, author.materials['steel'], collision='convexHull')
        UsdGeom.Xformable(prim).AddScaleOp().Set(Gf.Vec3f(*inverse))
        UsdGeom.Imageable(prim).CreateVisibilityAttr('invisible')
        UsdShade.MaterialBindingAPI(prim).Bind(author.physical_material, materialPurpose='physics')
        original = stage.GetPrimAtPath(row['collider_paths'][0])
        for attr in original.GetAttributes():
            if attr.GetName().startswith('physxCollision:') and attr.Get() is not None:
                prim.CreateAttribute(attr.GetName(), attr.GetTypeName(), custom=attr.IsCustom(),
                                     variability=attr.GetVariability()).Set(attr.Get())
        # Keep the existing engraved/brushed finish visible on the new top.
        shift = (rest[:3, :3]@np.array([0., 0., .006]))*scale
        for detail in Usd.PrimRange(stage.GetPrimAtPath(row['body_path']+'/Visual')):
            if detail.IsA(UsdGeom.Mesh) and ('handle_engraving_' in detail.GetName()
                                            or 'finish_brushed_handle_' in detail.GetName()):
                usd = UsdGeom.Mesh(detail)
                vertices = np.asarray(usd.GetPointsAttr().Get(), dtype=float)+shift
                usd.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(vertices.astype(np.float32)))
                usd.GetExtentAttr().Set([Gf.Vec3f(*vertices.min(0)), Gf.Vec3f(*vertices.max(0))])
        # Use the actual authored collision mesh, including float32 coordinates.
        new_mesh = _body_mesh(stage, path, body)
        solids = old_meshes+[new_mesh]
        solid = trimesh.util.concatenate(solids)
        old_mass = row['mass_kg']
        mass = old_mass*float(solid.volume/old_solid.volume)
        props = g.mass_properties(solids, mass)
        row.update(props, mass_kg=mass, mass=mass)
        api = UsdPhysics.MassAPI(body)
        api.GetMassAttr().Set(mass)
        api.GetCenterOfMassAttr().Set(Gf.Vec3f(*props['center_of_mass_m']))
        api.GetDiagonalInertiaAttr().Set(Gf.Vec3f(*props['diagonal_inertia_kg_m2']))
        q = props['principal_axes_xyzw']
        api.GetPrincipalAxesAttr().Set(Gf.Quatf(q[3], Gf.Vec3f(*q[:3])))
        row['collider_paths'].append(path)
        row['collision_count'] = len(row['collider_paths'])
        row['collision_bounds_m'] = solid.bounds.tolist()
        row['collision_support_points_m'] = solid.convex_hull.vertices.tolist()
        row['support_shape']['bottom_local_z_m'] = float(solid.bounds[0, 2])
        visible = [_body_mesh(stage, str(p.GetPath()), body)
                   for p in Usd.PrimRange(stage.GetPrimAtPath(row['body_path']+'/Visual'))
                   if p.IsA(UsdGeom.Mesh)]
        row['local_bounds_m'] = trimesh.util.concatenate(solids+visible).bounds.tolist()
        row['handle_grip'] = {'profile': PROFILE, 'visual_path': visual_path,
            'collider_path': path, 'recipe_grip_length_m': .076,
            'grip_length_m': .076*float(scale[1]),
            'original_handle_thickness_m': .006*float(scale[2]),
            'handle_thickness_m': .012*float(scale[2]),
            'added_volume_m3': float(new_mesh.volume), 'source_mass_kg': old_mass,
            'mass_rule': 'Preserve prior effective density from original collision volume; recompute mass, COM and inertia from original solids plus nonoverlapping grip.',
            'mass_calibrated': False,
            'functional_end_and_original_support_geometry_unchanged': True}
        if row.get('natural_rest'):
            from scipy.spatial import ConvexHull
            support = np.asarray(row['natural_rest']['support_vertices_before_transform_m'])
            support = trimesh.transform_points(support, rest)*scale
            polygon = ConvexHull(support[:, :2])
            margin = float(-np.max(polygon.equations[:, :2]@np.asarray(props['center_of_mass_m'])[:2]
                                   +polygon.equations[:, 2]))
            if margin <= 0:
                raise ValueError('Thicker handle center of mass leaves original support polygon')
            row['handle_grip']['original_rest_support_COM_margin_m'] = margin
            row['natural_rest']['center_of_mass_support_edge_margin_m'] = margin/float(scale[0])
            row['natural_rest']['mass_geometry_profile'] = PROFILE
        meta['dimension_changes'][row['id']].update(after_bounds_m=row['collision_bounds_m'],
            after_mass_kg=mass, final_grip_profile=PROFILE)
        changed.append(row['id'])
    if not changed:
        raise ValueError('No cutlery found for thicker handle profile')
    meta['utensil_grip_profile'] = PROFILE
    meta['utensil_grip_revision'] = {'objects': changed,
        'scope': 'Visible upper handle thickening on original bodies; unchanged food-contact ends, lower support faces, XY layout, friction and numerical rules.'}
