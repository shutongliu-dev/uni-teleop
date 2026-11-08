"""Explicit household-size recipe; bake dimensions into geometry, never actors.

The original author remains reproducible. This profile preserves density as a
declared simulation prior, recomputes inertia, and carries support heights into
both reset and goal witnesses. It does not change materials or task tolerances.
"""
from copy import deepcopy

import numpy as np
import trimesh
from pxr import Gf, Usd, UsdGeom, UsdPhysics, Vt

from . import geometry as g

PROFILE = 'household_v1'


def scales(task_id, rows):
    factors = {name: np.ones(3) for name in rows}
    for name, row in rows.items():
        if not row['mass_kg']:
            continue
        factor = 1.2
        if task_id == 'breakfast_serving':
            factor = 1.1 if name == 'tray' else 1.3 if name.startswith('bread') else 1.2
        elif task_id == 'produce_transfer':
            factor = 1.3 if name.startswith('tomato') else 1.25
        elif task_id in ('utensil_sorting', 'pot_lid_placement'):
            factor = 1.15
        elif task_id == 'table_clearing' and name in ('spoon', 'fork'):
            factor = 1.15
        factors[name] = np.full(3, factor)
    if task_id == 'utensil_sorting':
        factors['source_pad'] = np.array([1., 1.04, 1.])
    if task_id == 'pot_lid_placement':
        factors['source_mat'] = np.array([.51/.49, 1., 1.])
    if task_id == 'pantry_stocking':
        factors['source_mat'] = np.array([.46/.42, 1., 1.])
    if task_id == 'table_clearing':
        factors['source_mat'] = np.array([.54/.48, .52/.49, 1.])
    if task_id == 'surface_wiping':
        factors['sponge'] = np.array([1.2, 1.2, 1.])
    return factors


def _extra_dimensions(row, scale):
    """Only declared local lengths are scaled; directions remain unit vectors."""
    xy, z = float(scale[0]), float(scale[2])
    for key in ('interior_radius_m', 'locating_outer_radius_m'):
        if key in row:
            row[key] *= xy
    for key in ('floor_top_local_z_m', 'rim_local_z_m', 'seat_local_z_m', 'wiping_face_local_z_m'):
        if key in row:
            row[key] *= z
    for key in ('interior_bounds_xy_m', 'wiping_face_bounds_xy_m', 'rim_radius_range_m', 'seat_radius_range_m'):
        if key in row:
            row[key] = (np.asarray(row[key])*(scale[:2] if 'bounds_xy' in key else xy)).tolist()
    for key in ('handle_bounds_m', 'knob_bounds_m'):
        if key in row:
            row[key] = (np.asarray(row[key])*scale).tolist()
    if 'handle_opening_xz_m' in row:
        row['handle_opening_xz_m'] = (np.asarray(row['handle_opening_xz_m'])*scale[[0, 2]]).tolist()


def _mesh(prim):
    mesh = UsdGeom.Mesh(prim)
    return trimesh.Trimesh(np.asarray(mesh.GetPointsAttr().Get(), dtype=float),
        np.asarray(mesh.GetFaceVertexIndicesAttr().Get()).reshape(-1, 3), process=False)


def apply_household_profile(author):
    stage, meta = author.stage, author.meta
    rows = {**meta['fixture_metadata'], **{row['id']: row for row in meta['objects']}}
    original = deepcopy(rows)
    factors = scales(meta['task_id'], rows)
    for name, row in rows.items():
        factor = factors[name]
        if np.array_equal(factor, np.ones(3)):
            continue
        physical, visible = [], []
        for prim in Usd.PrimRange(stage.GetPrimAtPath(row['body_path'])):
            if not prim.IsA(UsdGeom.Mesh):
                continue
            mesh = _mesh(prim)
            mesh.apply_scale(factor)
            usd = UsdGeom.Mesh(prim)
            points = np.asarray(mesh.vertices, dtype=np.float32)
            usd.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(points))
            uv = UsdGeom.PrimvarsAPI(usd).GetPrimvar('st')
            if uv and uv.HasValue():
                uv.Set(Vt.Vec2fArray.FromNumpy(np.asarray(g.wood_texcoords(mesh), dtype=np.float32)))
            usd.GetExtentAttr().Set([Gf.Vec3f(*points.min(0).tolist()), Gf.Vec3f(*points.max(0).tolist())])
            normal = np.asarray(usd.GetNormalsAttr().Get(), dtype=float)/factor
            normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-30)
            usd.GetNormalsAttr().Set(Vt.Vec3fArray.FromNumpy(normal.astype(np.float32)))
            if str(prim.GetPath()) in row['collider_paths']:
                physical.append(mesh)
            else:
                visible.append(mesh)
        solid = trimesh.util.concatenate(physical)
        row['collision_bounds_m'] = solid.bounds.tolist()
        row['collision_support_points_m'] = solid.convex_hull.vertices.tolist()
        row['local_bounds_m'] = trimesh.util.concatenate(physical+visible).bounds.tolist()
        row['support_shape']['bottom_local_z_m'] = float(solid.bounds[0, 2])
        _extra_dimensions(row, factor)
        if not row['mass_kg']:
            continue
        old_mass = row['mass_kg']
        mass = old_mass*float(np.prod(factor))
        props = g.mass_properties(physical, mass)
        row.update(props, mass_kg=mass, mass=mass,
            dimension_profile={'name': PROFILE, 'linear_scale_xyz': factor.tolist(),
                'source_mass_kg': old_mass, 'mass_rule': 'preserve declared effective density; recompute inertia from scaled solids'})
        api = UsdPhysics.MassAPI(stage.GetPrimAtPath(row['body_path']))
        api.GetMassAttr().Set(mass)
        api.GetCenterOfMassAttr().Set(Gf.Vec3f(*props['center_of_mass_m']))
        api.GetDiagonalInertiaAttr().Set(Gf.Vec3f(*props['diagonal_inertia_kg_m2']))
        q = props['principal_axes_xyzw']
        api.GetPrincipalAxesAttr().Set(Gf.Quatf(q[3], Gf.Vec3f(*q[:3])))

    goals = meta['goals']
    old_goal = deepcopy(goals['body_poses_xyzw'])
    delta = {}

    def goal_delta(name):
        if name in delta:
            return delta[name]
        support = goals['support_by_actor'][name]
        value = 0.
        if support in rows:
            if support in old_goal:
                value += goal_delta(support)
            key = 'rim_local_z_m' if name == 'lid' else 'floor_top_local_z_m'
            value += rows[support].get(key, 0.)-original[support].get(key, 0.)
        if name == 'lid':
            value -= rows[name]['seat_local_z_m']-original[name]['seat_local_z_m']
        else:
            value -= rows[name]['collision_bounds_m'][0][2]-original[name]['collision_bounds_m'][0][2]
        delta[name] = value
        return value

    for name, pose in goals['body_poses_xyzw'].items():
        pose[2] += goal_delta(name)
    if meta['task_id'] == 'breakfast_serving':
        plate_x = goals['body_poses_xyzw']['plate'][0]
        for index, dx in enumerate((-.034, .034)):
            goals['body_poses_xyzw'][f'bread_{index}'][0] = plate_x+dx
    if meta['task_id'] == 'table_clearing':
        goals['body_poses_xyzw']['cereal_bowl'][0] = .576
        for pose in goals['body_poses_xyzw'].values():
            pose[0] += .06
        for name in ('dishes_tray', 'drinkware_tray', 'cutlery_tray'):
            row = rows[name]
            row['initial_pose_xyzw'][0] += .06
            stage.GetPrimAtPath(row['body_path']).GetAttribute('xformOp:translate').Set(Gf.Vec3d(*row['initial_pose_xyzw'][:3]))
            for prim in Usd.PrimRange(stage.GetPrimAtPath(author.root+'/Markings')):
                if prim.IsA(UsdGeom.Mesh) and prim.GetName().startswith(name+'_rim_mark_'):
                    marking = UsdGeom.Mesh(prim)
                    points = np.asarray(marking.GetPointsAttr().Get(), dtype=np.float32)+[.06, 0., 0.]
                    marking.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(points.astype(np.float32)))
                    marking.GetExtentAttr().Set([Gf.Vec3f(*points.min(0).tolist()), Gf.Vec3f(*points.max(0).tolist())])
    if meta['task_id'] == 'pantry_stocking':
        goals['body_poses_xyzw']['tomato_can'][1] = -1.005
    source_xy = {
        'pantry_stocking': {'olive_oil': (.18, -.70), 'vinegar': (.31, -.70),
            'salt': (.43, -.70), 'pepper': (.18, -.82), 'paprika': (.31, -.82),
            'rice_carton': (.43, -.82), 'pasta_carton': (.31, -.94), 'tomato_can': (.43, -.94)},
        'table_clearing': {'dinner_plate': (.14, -.87), 'side_plate': (.32, -.91),
            'cereal_bowl': (.14, -.68), 'small_bowl': (.29, -.68), 'mug': (.43, -.685),
            'tea_cup': (.444, -.808), 'spoon': (.445, -1.015), 'fork': (.495, -1.015)},
    }.get(meta['task_id'], {})
    if source_xy:
        mat = rows['source_mat']
        if meta['task_id'] == 'table_clearing':
            mat['initial_pose_xyzw'][0] = .27
        mat['initial_pose_xyzw'][1] = -.82 if meta['task_id'] == 'pantry_stocking' else -.86
        stage.GetPrimAtPath(mat['body_path']).GetAttribute('xformOp:translate').Set(Gf.Vec3d(*mat['initial_pose_xyzw'][:3]))
    for name, row in rows.items():
        if name not in meta['actors']:
            continue
        pose = row['initial_pose_xyzw']
        if meta['presentation']:
            pose[:] = goals['body_poses_xyzw'][name]
        else:
            support = meta['source_support_by_actor'][name]
            if support in rows:
                pose[2] += rows[support].get('floor_top_local_z_m', 0.)-original[support].get('floor_top_local_z_m', 0.)
            pose[2] -= row['collision_bounds_m'][0][2]-original[name]['collision_bounds_m'][0][2]
            if name in source_xy:
                pose[:2] = source_xy[name]
                if meta['task_id'] == 'table_clearing' and name in ('spoon', 'fork'):
                    pose[3:] = [0., 0., 0., 1.]
            if meta['task_id'] == 'pot_lid_placement' and name == 'pot':
                pose[1] = .725
        stage.GetPrimAtPath(row['body_path']).GetAttribute('xformOp:translate').Set(Gf.Vec3d(*pose[:3]))
    if meta['task_id'] == 'pot_lid_placement':
        goals['pot_rim_local_z_m'] = rows['pot']['rim_local_z_m']
        goals['lid_seat_local_z_m'] = rows['lid']['seat_local_z_m']
        goals['lid_base_relative_pot_m'] = goals['body_poses_xyzw']['lid'][2]-goals['body_poses_xyzw']['pot'][2]
    meta['prop_profile'] = PROFILE
    meta['source']['dimension_revision'] = 'household_v1: per-object metric geometry enlargement, constant declared effective density'
    meta['dimension_changes'] = {name: {'scale_xyz': factors[name].tolist(),
        'before_bounds_m': original[name]['collision_bounds_m'], 'after_bounds_m': row['collision_bounds_m'],
        'before_mass_kg': original[name]['mass_kg'], 'after_mass_kg': row['mass_kg']}
        for name, row in rows.items() if name in meta['actors']}
