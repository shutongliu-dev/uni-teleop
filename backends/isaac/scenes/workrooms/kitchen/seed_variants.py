"""Immutable, reset-only seed derivatives of the four rigid kitchen inventories.

The selected USD is the source of geometry. In particular this author never
rebuilds cutlery from an earlier recipe, changes collision approximations, or
writes poses in a running simulation. CPU authoring is not native acceptance.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import importlib
import json
import math
from pathlib import Path
import shutil

import numpy as np

from bench.isaacsim.release import checked, digest
from bench.paths import DATA_ROOT, load_json, portable
from scenes.workrooms import seed_geometry as shared_geometry
from .composition import bind_episode_task, load_asset_manifest

TASK_IDS = ('produce_transfer', 'utensil_sorting', 'pantry_stocking', 'table_clearing')
VERSION = 'kitchen_seed_geometry_v1'
CONTACT_RESET_VERSION = 'kitchen_seed_geometry_contact_reset_v2'
SUPPORT_HULL_VERSION = 'kitchen_seed_geometry_support_hull_v3'
SUPPORT_LOW_VERTEX_VERSION = 'kitchen_seed_geometry_support_low_vertex_v4'
SUPPORT_LOW_VERTEX_TIP_VERSION = 'kitchen_seed_geometry_support_low_vertex_tip_v5'
SUPPORT_SOLID_VERSIONS = (SUPPORT_HULL_VERSION, SUPPORT_LOW_VERTEX_VERSION, SUPPORT_LOW_VERTEX_TIP_VERSION)
ROOT = '/World/KitchenTask'
ASSET_ROOT = Path(f'{DATA_ROOT}/assets/task_parts/seed_randomization/v1/kitchen')


def canonical_hash(value):
    # Identities hash data:// references, never this machine's data root.
    return hashlib.sha256(json.dumps(portable(value), sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def _dump(path, value):
    Path(path).write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False)+'\n')


def _unit(*parts):
    raw = hashlib.sha256('|'.join(map(str, parts)).encode()).digest()
    return int.from_bytes(raw[:8], 'big') / (2**64-1) * 2 - 1


def case_plan(task_id, seed, parent_metadata, *, profile=VERSION):
    """Finite deterministic cases; labels cannot select arbitrary geometry."""
    if task_id not in TASK_IDS or type(seed) is not int or seed not in range(10):
        raise ValueError('Kitchen randomization requires a supported task and integer seed 0..9')
    if parent_metadata.get('task_id') != task_id:
        raise ValueError('Parent metadata task differs')
    if profile not in (VERSION, CONTACT_RESET_VERSION, *SUPPORT_SOLID_VERSIONS) or (
            profile != VERSION and task_id != 'produce_transfer'):
        raise ValueError('Unsupported kitchen reset profile for this task')
    actors = parent_metadata['actors']
    rows = {r['id']: r for r in parent_metadata['objects']}
    if set(rows) != set(actors):
        raise ValueError('Exact original kitchen inventory required')
    if task_id in ('utensil_sorting', 'table_clearing') and (
            parent_metadata.get('utensil_grip_profile') != 'thick_handle_v1'
            or parent_metadata.get('utensil_contact_profile') != 'floor_grid_2mm_splitbowl_sdf768_v7'):
        raise ValueError('Current thick handles and split-bowl SDF contact profile are required')
    amplitude = (0., -.03, .03, -.02, .02, -.025, .025, -.01, .01, .015)[seed]
    bodies = {}
    for name, row in sorted(rows.items()):
        cutlery = row['role'] in ('spoon', 'fork', 'knife')
        change = amplitude * (2/3 if cutlery else 1)
        translation = [round(_unit(VERSION, task_id, seed, name, axis)*.004, 8)
                       if seed else 0. for axis in ('x', 'y')]
        yaw = round(_unit(VERSION, task_id, seed, name, 'yaw')*math.radians(2), 9) if seed else 0.
        bodies[name] = {'prim_path': actors[name], 'scale_xyz': [1+change, 1+change, 1.],
                        'translation_xy_m': translation, 'yaw_rad': yaw}
    result = {'schema': 'kitchen_seed_case_plan.v1', 'generator_version': VERSION,
              'family': 'kitchen', 'task_id': task_id, 'seed': seed,
              'parent_metadata_sha256': canonical_hash(parent_metadata), 'bodies': bodies,
              'appearance_variant': 'canonical' if seed == 0 else f'surface_finish_{seed % 3}',
              'fixed_fixtures': sorted(parent_metadata['fixtures']),
              'scope': 'Reset-only bounded actor XY/yaw and body-local XY geometry; fixed Z, fixtures, room, robot, contacts and thresholds'}
    if profile in (CONTACT_RESET_VERSION, *SUPPORT_SOLID_VERSIONS):
        carrots = sorted(name for name, row in rows.items() if row['category'] == 'carrot')
        if (len(carrots) != 3 or 'source_board' not in parent_metadata['fixtures']
                or any(parent_metadata['source_support_by_actor'].get(name) != 'source_board'
                       for name in carrots)):
            raise ValueError('Produce contact reset requires the original carrots and source board')
        result.update(schema='kitchen_seed_case_plan.v2', generator_version=profile,
            reset_profile={'id': 'produce_carrot_actual_support_v2', 'actors': carrots,
                'support_id': 'source_board', 'support_prim_path': parent_metadata['fixtures']['source_board'],
                'method': 'actual_enabled_collision_board_top_minus_rotated_actor_bottom',
                'clearance_m': 0., 'seed_zero': 'original_canonical_bytes'},
            scope='Reset-only original XY/yaw/geometry sampling; nonzero carrot Z aligns actual collision bottom to actual source-board top. Canonical zero, fixtures, room, robot, contacts and thresholds unchanged.')
        if profile == SUPPORT_HULL_VERSION:
            result.update(schema='kitchen_seed_case_plan.v3',
                geometry_profile={'id': 'verified_parent_decagonal_carrot_v1',
                    'source_radial_segments': 64, 'radial_segments': 10,
                    'axial_rings': 6, 'tip_vertices': 1,
                    'main_visible_and_collision_solid': 'identical_closed_planar_polygons',
                    'parent_recipe_reconstruction_required': True,
                    'mass_rule': 'Integrate final float32 material solid at actual selected-parent density'},
                scope='Nonzero carrots use a verified-parent decagonal visible/collision solid, density-preserving integrated mass/COM/inertia and actual floor contact reset. Original seed XY/yaw, canonical zero, other objects, fixtures, room, robots, materials and numerical rules unchanged.')
            result['reset_profile']['id'] = 'produce_carrot_support_hull_v3'
        if profile in (SUPPORT_LOW_VERTEX_VERSION, SUPPORT_LOW_VERTEX_TIP_VERSION):
            result.update(schema='kitchen_seed_case_plan.v4',
                geometry_profile={'id': 'verified_parent_lowvertex_carrot_v1',
                    'source_radial_segments': 64, 'radial_segments': 8,
                    'parent_axial_ring_indices': [0, 2, 4, 5], 'axial_rings': 4,
                    'tip_vertices': 1, 'source_vertices': 33, 'maximum_triangulated_faces': 62,
                    'collision_mesh_coordinate_scale': 1.,
                    'main_visible_and_collision_solid': 'identical_closed_planar_polygons',
                    'parent_recipe_reconstruction_required': True,
                    'mass_rule': 'Integrate final float32 material solid at actual selected-parent density'},
                scope='Nonzero carrots use a verified-parent four-ring octagonal visible/collision solid with 33 vertices, density-preserving integrated mass/COM/inertia and actual floor contact reset. Original seed XY/yaw, canonical zero, other objects, fixtures, room, robots, materials and numerical rules unchanged.')
            result['reset_profile']['id'] = 'produce_carrot_support_low_vertex_v4'
        if profile == SUPPORT_LOW_VERTEX_TIP_VERSION:
            result['schema'] = 'kitchen_seed_case_plan.v5'
            result['geometry_profile'].update(id='verified_parent_lowvertex_tip_carrot_v1',
                parent_ring_radius_overrides_m={'0': {'original': .002, 'variant': .0025}})
            result['reset_profile']['id'] = 'produce_carrot_support_low_vertex_tip_v5'
            result['scope'] = ('Nonzero carrots use the explicit 33-vertex four-ring octagonal solid with only the first ring radius increased from 2 to 2.5 mm before household/seed scaling, identical visible/collision material, density-integrated mass properties and actual floor contact reset. Original seed XY/yaw, canonical zero, other objects, fixtures, room, robots, materials and numerical rules unchanged.')
    result['plan_sha256'] = canonical_hash(result)
    return result


def _body_points(stage, body_path, *, collision_only=False):
    from pxr import Usd, UsdGeom, UsdPhysics
    body = stage.GetPrimAtPath(body_path)
    cache = UsdGeom.XformCache()
    points = []
    for prim in Usd.PrimRange(body):
        if not prim.IsA(UsdGeom.Mesh):
            continue
        if collision_only and (not prim.HasAPI(UsdPhysics.CollisionAPI)
                              or not UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get()):
            continue
        matrix, _ = cache.ComputeRelativeTransform(prim, body)
        values = np.asarray(UsdGeom.Mesh(prim).GetPointsAttr().Get(), dtype=float)
        a = np.asarray(matrix, dtype=float).T
        points.append(values @ a[:3, :3].T + a[:3, 3])
    if not points:
        raise ValueError('No actual mesh geometry at '+body_path)
    return np.concatenate(points)


def _collision_recipe(stage, paths):
    from pxr import UsdPhysics
    result = {}
    for path in paths:
        prim = stage.GetPrimAtPath(path)
        if not prim or not prim.HasAPI(UsdPhysics.CollisionAPI):
            raise ValueError('Declared collider missing: '+path)
        result[path] = {'apis': list(prim.GetAppliedSchemas()),
            'attributes': {a.GetName(): str(a.Get()) for a in prim.GetAttributes()
                if a.GetName().startswith(('physx', 'physics:collision', 'physics:approximation'))},
            'physics_material': [str(p) for p in prim.GetRelationship('material:binding:physics').GetTargets()]}
    return result


def _appearance(stage, metadata, variant, *, texture_path=None, seed=0):
    if variant == 'canonical':
        return {'variant': variant, 'scope': 'Original material/texture bytes', 'changed_materials': []}
    if texture_path is None:
        return {'variant': variant, 'scope': 'Geometry-only anonymous derivation; publisher adds the explicit procedural texture',
                'texture_authored': False}
    return shared_geometry.apply_surface_texture(stage,
        [r['body_path']+'/Visual' for r in metadata['objects']], texture_path, seed,
        material_root=ROOT+'/SeedVariantMaterials')


def _update_geometry_metadata(stage, row, scale):
    from pxr import UsdPhysics
    physical = _body_points(stage, row['body_path'], collision_only=True)
    visible = _body_points(stage, row['body_path'])
    row['collision_bounds_m'] = [physical.min(0).tolist(), physical.max(0).tolist()]
    row['local_bounds_m'] = [visible.min(0).tolist(), visible.max(0).tolist()]
    # The existing hull vertices remain exact support samples under an affine map.
    row['collision_support_points_m'] = (np.asarray(row['collision_support_points_m'])*scale).tolist()
    row['support_shape']['bottom_local_z_m'] = float(physical[:, 2].min())
    api = UsdPhysics.MassAPI(stage.GetPrimAtPath(row['body_path']))
    row.update(mass_kg=float(api.GetMassAttr().Get()), mass=float(api.GetMassAttr().Get()),
               center_of_mass_m=list(api.GetCenterOfMassAttr().Get()),
               diagonal_inertia_kg_m2=list(api.GetDiagonalInertiaAttr().Get()))
    q = api.GetPrincipalAxesAttr().Get()
    row['principal_axes_xyzw'] = [*q.GetImaginary(), q.GetReal()]
    if 'collision_volume_m3' in row:
        row['collision_volume_m3'] *= float(np.prod(scale))
    # XY is isotropic; Z and supporting floor heights are unchanged.
    for key in ('interior_radius_m', 'locating_outer_radius_m'):
        if key in row:
            row[key] *= float(scale[0])
    for key in ('interior_bounds_xy_m', 'rim_radius_range_m', 'seat_radius_range_m'):
        if key in row:
            row[key] = (np.asarray(row[key])*float(scale[0])).tolist()
    for key in ('handle_bounds_m', 'knob_bounds_m'):
        if key in row:
            row[key] = (np.asarray(row[key])*scale).tolist()
    if 'handle_opening_xz_m' in row:
        row['handle_opening_xz_m'] = (np.asarray(row['handle_opening_xz_m'])*scale[[0, 2]]).tolist()
    for key in ('handle_axis_local', 'food_contact_axis_local'):
        if key in row:
            vector = np.asarray(row[key])*scale
            row[key] = (vector/np.linalg.norm(vector)).tolist()
    if 'handle_grip' in row:
        row['handle_grip']['grip_length_m'] *= float(scale[1])
        row['handle_grip']['added_volume_m3'] *= float(np.prod(scale))
    row['seed_dimension_profile'] = {'scale_xyz': scale.tolist(), 'source': 'selected parent USD',
        'mass_rule': 'Density-preserving affine transform of original explicit body mass/COM/inertia; no collider union approximation',
        'native_qualified': False}


def _points_at(points, pose):
    from pxr import Gf
    from scenes.workrooms.living_room import seed_geometry as geometry
    q = Gf.Quatd(float(pose[6]), Gf.Vec3d(*map(float, pose[3:6])))
    return np.asarray(points) @ geometry.rotation(q).T + np.asarray(pose[:3])


def static_checks(metadata):
    """Check reset support footprint and fixed target capacity using real points.

    These are geometric necessary conditions, not a native support certificate.
    Whole target geometry containment is checked; initial support needs only the
    original lower support footprint (handles may overhang a flat source mat).
    """
    from pxr import Gf
    from scenes.workrooms.living_room import seed_geometry as geometry
    rows = {r['id']: r for r in metadata['objects']}
    fixtures = metadata['fixture_metadata']
    checks = []
    for name, row in rows.items():
        points = np.asarray(row['collision_support_points_m'])
        for state, poses, supports in (
                ('reset', {k: r['initial_pose_xyzw'] for k, r in rows.items()}, metadata['source_support_by_actor']),
                ('goal', metadata['goals']['body_poses_xyzw'], metadata['goals']['support_by_actor'])):
            support_id = supports[name]
            support = fixtures.get(support_id, rows.get(support_id))
            if support is None:
                raise ValueError('Unknown declared support '+support_id)
            support_pose = support['initial_pose_xyzw'] if support_id in fixtures else poses[support_id]
            world = _points_at(points, poses[name])
            q = Gf.Quatd(float(support_pose[6]), Gf.Vec3d(*map(float, support_pose[3:6])))
            local = (world-np.asarray(support_pose[:3])) @ geometry.rotation(q)
            if state == 'reset':
                local = local[local[:, 2] <= local[:, 2].min()+.00005]
                bounds = support['collision_bounds_m']
                inside = np.all(local[:, :2] >= np.asarray(bounds)[0, :2]-.0001) and np.all(local[:, :2] <= np.asarray(bounds)[1, :2]+.0001)
            elif 'interior_bounds_xy_m' in support:
                bounds = np.asarray(support['interior_bounds_xy_m'])
                # Cutlery tray has class channels smaller than its outer cavity.
                channels = support.get('channels', {})
                if row.get('assigned_channel') in channels:
                    channel = channels[row['assigned_channel']]
                    bounds = np.asarray(channel.get('interior_bounds_xy_m', bounds))
                inside = np.all(local[:, :2] >= bounds[0]-.0001) and np.all(local[:, :2] <= bounds[1]+.0001)
            elif 'interior_radius_m' in support:
                inside = bool(np.max(np.linalg.norm(local[:, :2], axis=1)) <= support['interior_radius_m']+.0001)
            else:
                raise ValueError('Support lacks a physical capacity description: '+support_id)
            checks.append({'actor_id': name, 'reference_state': state, 'support_id': support_id,
                           'geometry_contained': bool(inside)})
            if not inside:
                raise ValueError(f'{state} support/target capacity exceeded by {name}')
    return checks


def footprint_separations(metadata):
    """Conservative planar convex-envelope check for the separate inventories.

    The explicitly supported optional plate stack is excluded only at the goal.
    A positive separation proves these footprints are disjoint; this does not
    replace native contacts or test a manipulation trajectory.
    """
    from scipy.spatial import ConvexHull
    rows = {row['id']: row for row in metadata['objects']}
    result = []
    for state in ('initial', 'goal'):
        polygons = {}
        for name, row in rows.items():
            pose = row['initial_pose_xyzw'] if state == 'initial' else metadata['goals']['body_poses_xyzw'][name]
            world = _points_at(row['collision_support_points_m'], pose)[:, :2]
            polygons[name] = world[ConvexHull(world).vertices]
        for index, (a, p) in enumerate(polygons.items()):
            for b, q in list(polygons.items())[index+1:]:
                supports = metadata['goals']['support_by_actor']
                if state == 'goal' and (supports[a] == b or supports[b] == a):
                    continue
                edges = np.r_[np.roll(p, -1, axis=0)-p, np.roll(q, -1, axis=0)-q]
                axes = np.c_[-edges[:, 1], edges[:, 0]]
                axes /= np.linalg.norm(axes, axis=1)[:, None]
                pp, qq = p@axes.T, q@axes.T
                gap = float(np.maximum(pp.min(0)-qq.max(0), qq.min(0)-pp.max(0)).max())
                if gap < 0:
                    raise ValueError(f'Overlapping {state} actor footprints: {a}, {b}')
                result.append({'actor_ids': [a, b], 'reference_state': state,
                               'separating_axis_gap_m': gap})
    return result


def _pepper_pickup_alternatives(stage, row, original):
    """Ray exits on the original single convex collider, never a body AABB."""
    from pxr import Usd, UsdGeom, UsdPhysics
    from scipy.spatial import ConvexHull

    body = stage.GetPrimAtPath(row['body_path'])
    colliders = [prim for prim in Usd.PrimRange(body)
                 if prim.HasAPI(UsdPhysics.CollisionAPI)
                 and UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get() is not False]
    if (len(colliders) != 1 or not colliders[0].IsA(UsdGeom.Mesh)
            or UsdPhysics.MeshCollisionAPI(colliders[0]).GetApproximationAttr().Get() != 'convexHull'):
        raise ValueError('Pepper access requires its original single convexHull mesh')
    points = _body_points(stage, row['body_path'], collision_only=True)
    lo, hi = points.min(0), points.max(0)
    hull = ConvexHull(points)
    normals, offsets = hull.equations[:, :3], hull.equations[:, 3]
    recipes = (
        ('pepper_side_mid_surface', .48, [-1., 0., 0.], .08),
        ('pepper_side_high_surface', .85, [-1., 0., 0.], .08),
        ('pepper_diagonal_mid_surface', .48, [-1., 0., 1.], .10),
        ('pepper_diagonal_high_surface', .85, [0., -1., 1.], .08),
    )
    result = []
    for label, fraction, direction, distance in recipes:
        direction = np.asarray(direction)/np.linalg.norm(direction)
        center = np.array([0., 0., lo[2]+fraction*(hi[2]-lo[2])])
        values = normals@center+offsets
        if float(values.max()) >= 1e-8:
            raise ValueError('Pepper access ray origin is outside its actual convex collider')
        slopes = normals@direction
        forward = slopes > 1e-10
        if not forward.any():
            raise ValueError('Pepper access ray has no forward collider boundary')
        travel = float((-values[forward]/slopes[forward]).min())
        point = center+travel*direction
        if travel <= 0 or abs(float((normals@point+offsets).max())) >= 1e-9:
            raise ValueError('Pepper access point is not on its actual convex collider')
        result.append({**original, 'label': label, 'local_position_m': point.tolist(),
                       'approach_direction_local': direction.tolist(),
                       'approach_distance_m': distance, 'approach': 'above'})
    return result


def _projections(stage, metadata):
    from pxr import UsdGeom
    from scenes.workrooms.living_room import seed_geometry as geometry
    rows = {r['id']: r for r in metadata['objects']}
    poses = {name: geometry.pose(stage.GetPrimAtPath(row['body_path'])) for name, row in rows.items()}
    static = {'bodies': [{'object_id': name, 'local_bounds_m': deepcopy(row['collision_bounds_m']),
                          'nominal_mass_kg': float(row.get('mass_kg', 0.))} for name, row in sorted(rows.items())]}
    targets = []
    for row in metadata['objects']:
        path = row['body_path']
        if row.get('handle_grip'):
            path = row['handle_grip']['collider_path']
            prim = stage.GetPrimAtPath(path)
            mesh = UsdGeom.Mesh(prim)
            points = np.asarray(mesh.GetPointsAttr().Get(), dtype=float)
            matrix, _ = UsdGeom.XformCache().ComputeRelativeTransform(prim, stage.GetPrimAtPath(row['body_path']))
            a = np.asarray(matrix).T
            points = points @ a[:3, :3].T+a[:3, 3]
        else:
            points = _body_points(stage, path, collision_only=True)
        top = points[np.abs(points[:, 2]-points[:, 2].max()) < 1e-6]
        point = top[np.argmin(np.linalg.norm(top[:, :2]-top[:, :2].mean(0), axis=1))]
        pickup = {'label': row['id']+'_upper_surface', 'prim_path': row['body_path'],
            'local_position_m': point.tolist(), 'approach_direction_local': [0., 0., 1.],
            'approach_distance_m': .06, 'type': 'actual_surface', 'reference_state': 'initial',
            'operation_group': row['id']+'_pickup', 'required': True}
        targets.append(pickup)
        if metadata['task_id'] == 'table_clearing' and row['id'] in ('spoon', 'fork'):
            # Keep the original point; expose another actual thick-grip vertex
            # near its neck, without moving the utensil or neighboring dishes.
            neck = top[np.lexsort((top[:, 0], top[:, 1]))[0]]
            targets.append({**pickup, 'label': row['id']+'_grip_neck_upper_surface',
                            'local_position_m': neck.tolist()})
        if metadata['task_id'] == 'table_clearing' and row['id'] == 'side_plate':
            # An actual enabled-collider upper rim vertex on the negative-Y
            # side supplements the original pickup point for the same group.
            rim = top[np.lexsort((top[:, 0], top[:, 1]))[0]]
            targets.append({**pickup, 'label': 'side_plate_rim_negative_y_upper_surface',
                            'local_position_m': rim.tolist(), 'approach': 'above'})
        if metadata['task_id'] == 'pantry_stocking' and row['id'] == 'pepper':
            targets.extend(_pepper_pickup_alternatives(stage, row, pickup))
        target = metadata['goals']['support_by_actor'][row['id']]
        # For optional dish stacks, the final tray remains the public destination.
        while target in metadata['actors']:
            target = metadata['goals']['support_by_actor'][target]
        fixture = metadata['fixture_metadata'][target]
        goal = metadata['goals']['body_poses_xyzw'][row['id']]
        local = np.asarray(goal[:3])-np.asarray(fixture['initial_pose_xyzw'][:3])
        local[2] = fixture['floor_top_local_z_m']
        destination = {'label': row['id']+'_destination_surface', 'prim_path': fixture['body_path'],
            'local_position_m': local.tolist(), 'approach_direction_local': [0., 0., 1.],
            'approach_distance_m': .10, 'type': 'goal', 'reference_state': 'initial',
            'operation_group': row['id']+'_destination', 'required': True}
        targets.append(destination)
        if metadata['task_id'] == 'table_clearing' and row['id'] == 'fork':
            # The goal remains on its original tray floor; only the approach
            # direction changes, not the placement reference or scoring input.
            direction = np.asarray([-1., 0., 1.])/np.sqrt(2.)
            targets.append({**destination, 'label': 'fork_destination_diagonal_approach',
                            'approach_direction_local': direction.tolist()})
    return poses, targets, static


def _fixed_support_paths(stage, metadata):
    from pxr import Usd, UsdPhysics
    paths = []
    for root in metadata['fixtures'].values():
        for prim in Usd.PrimRange(stage.GetPrimAtPath(root)):
            if (not prim.HasAPI(UsdPhysics.CollisionAPI)
                    or not UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get()):
                continue
            ancestor = prim
            while ancestor and not ancestor.IsPseudoRoot():
                if (ancestor.HasAPI(UsdPhysics.RigidBodyAPI)
                        and UsdPhysics.RigidBodyAPI(ancestor).GetRigidBodyEnabledAttr().Get()):
                    raise ValueError('Task fixture collider has an enabled rigid-body ancestor')
                ancestor = ancestor.GetParent()
            paths.append(str(prim.GetPath()))
    if not paths:
        raise ValueError('Task has no explicit fixed source/target support colliders')
    return sorted(paths)


def _world_collision_vertices(stage, root_path):
    """Author-only actual enabled task mesh geometry, before any native cooking."""
    from pxr import Usd, UsdGeom, UsdPhysics
    root = stage.GetPrimAtPath(root_path)
    if not root:
        raise ValueError('Missing declared collision root '+root_path)
    cache = UsdGeom.XformCache(); values = []; paths = []
    for prim in Usd.PrimRange(root):
        if (not prim.HasAPI(UsdPhysics.CollisionAPI)
                or UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get() is False):
            continue
        if not prim.IsA(UsdGeom.Mesh):
            raise ValueError('Exact support reset requires actual collision meshes')
        points = np.asarray(UsdGeom.Mesh(prim).GetPointsAttr().Get(), dtype=float)
        transform = np.asarray(cache.GetLocalToWorldTransform(prim)).T
        values.append(points @ transform[:3, :3].T+transform[:3, 3])
        paths.append(str(prim.GetPath()))
    if not values or not np.isfinite(np.concatenate(values)).all():
        raise ValueError('Missing finite enabled collision mesh vertices')
    return np.concatenate(values), paths


def _align_carrot_support(stage, metadata, row, reset_profile):
    """Change only an anonymous asset's reset Z; never an episode object write."""
    from pxr import Gf, UsdGeom, UsdPhysics
    from scipy.spatial import ConvexHull
    support_path = reset_profile['support_prim_path']
    if (metadata['fixtures'][reset_profile['support_id']] != support_path
            or metadata['source_support_by_actor'][row['id']] != reset_profile['support_id']):
        raise ValueError('Carrot reset support identity differs from its declared fixture')
    board, board_paths = _world_collision_vertices(stage, support_path)
    for path in board_paths:
        prim = stage.GetPrimAtPath(path)
        while prim and not prim.IsPseudoRoot():
            if (prim.HasAPI(UsdPhysics.RigidBodyAPI)
                    and UsdPhysics.RigidBodyAPI(prim).GetRigidBodyEnabledAttr().Get()):
                raise ValueError('Carrot reset requires an actual fixed support fixture')
            prim = prim.GetParent()
    top = float(board[:, 2].max())
    face = board[np.abs(board[:, 2]-top) < 1e-9]
    if len(face) < 3:
        raise ValueError('Source board lacks an actual horizontal top face')
    polygon = ConvexHull(face[:, :2])
    points, actor_paths = _world_collision_vertices(stage, row['body_path'])
    bottom = float(points[:, 2].min())
    footprint = points[points[:, 2] <= bottom+1e-8, :2]
    if float((footprint @ polygon.equations[:, :2].T+polygon.equations[:, 2]).max()) > 1e-8:
        raise ValueError('Carrot bottom footprint is outside the actual source-board top')
    body = stage.GetPrimAtPath(row['body_path'])
    cache = UsdGeom.XformCache(); world = cache.GetLocalToWorldTransform(body)
    before = list(world.ExtractTranslation())
    delta = top+reset_profile['clearance_m']-bottom
    moved = Gf.Matrix4d(world)
    moved.SetTranslateOnly(Gf.Vec3d(before[0], before[1], before[2]+delta))
    local = moved * cache.GetParentToWorldTransform(body).GetInverse()
    ops = UsdGeom.Xformable(body).GetOrderedXformOps()
    if len(ops) != 1 or ops[0].GetOpName() != 'xformOp:transform:seedReset':
        raise ValueError('Support alignment requires the original planar seed reset transform')
    ops[0].Set(local)
    actual, _ = _world_collision_vertices(stage, row['body_path'])
    if abs(float(actual[:, 2].min())-top-reset_profile['clearance_m']) > 1e-10:
        raise ValueError('Authored reset does not meet its exact source support plane')
    return {'support_id': reset_profile['support_id'], 'support_prim_path': support_path,
        'support_collision_paths': board_paths, 'actor_collision_paths': actor_paths,
        'support_top_world_z_m': top, 'collision_bottom_before_world_z_m': bottom,
        'position_before_m': before, 'position_after_m': [before[0], before[1], before[2]+delta],
        'translation_z_m': delta, 'clearance_m': reset_profile['clearance_m'],
        'scope': 'Source-mesh reset geometry only; no solver-cooked hull or native stability qualification'}


def derive_stage(parent_stage, original_metadata, plan):
    """Derive an anonymous stage; no writes to the parent or shared assets."""
    from pxr import Usd
    from scenes.workrooms.living_room import seed_geometry as geometry
    if plan != case_plan(plan['task_id'], plan['seed'], original_metadata,
                         profile=plan.get('generator_version')):
        raise ValueError('Kitchen case plan differs from the finite generator or parent')
    stage = Usd.Stage.Open(parent_stage.Flatten())
    metadata = deepcopy(original_metadata)
    paths = [p for row in metadata['objects'] for p in row['collider_paths']]
    recipe = _collision_recipe(stage, paths)
    if plan['seed']:
        reset_receipts = {}
        for row in metadata['objects']:
            entry = plan['bodies'][row['id']]
            scale = np.asarray(entry['scale_xyz'])
            support_hull = plan['generator_version'] in SUPPORT_SOLID_VERSIONS and row['id'] in plan['reset_profile']['actors']
            if support_hull:
                from . import produce_support_hull
                produce_support_hull.apply(stage, metadata, row, profile=plan['generator_version'])
            shared_geometry.bake_body_scale(stage, row['body_path'], scale)
            _update_geometry_metadata(stage, row, scale)
            if support_hull:
                produce_support_hull.finalize(stage, row)
            geometry.transform_group(stage, {'prim_paths': [row['body_path']],
                'pivot_m': row['initial_pose_xyzw'][:3],
                'translation_xy_m': entry['translation_xy_m'], 'yaw_rad': entry['yaw_rad']})
            if (plan['generator_version'] in (CONTACT_RESET_VERSION, *SUPPORT_SOLID_VERSIONS)
                    and row['id'] in plan['reset_profile']['actors']):
                reset_receipts[row['id']] = _align_carrot_support(stage, metadata, row, plan['reset_profile'])
            pose = geometry.pose(stage.GetPrimAtPath(row['body_path']))
            w, x, y, z = pose['quaternion_wxyz']
            row['initial_pose_xyzw'] = [*pose['position_m'], x, y, z, w]
            if row['id'] in metadata.get('dimension_changes', {}):
                metadata['dimension_changes'][row['id']].update(
                    after_bounds_m=row['collision_bounds_m'], after_mass_kg=row['mass_kg'])
        metadata['seed_variant'] = {'generator_version': plan['generator_version'], 'seed': plan['seed'],
            'plan_sha256': plan['plan_sha256'], 'original_collision_representation_preserved': True,
            'native_qualified': False, 'robot_manipulation_qualified': False}
        if plan['generator_version'] in (CONTACT_RESET_VERSION, *SUPPORT_SOLID_VERSIONS):
            metadata['seed_variant'].update(reset_profile=deepcopy(plan['reset_profile']),
                                            reset_support_receipts=reset_receipts)
        if plan['generator_version'] in SUPPORT_SOLID_VERSIONS:
            metadata['seed_variant']['geometry_profile'] = deepcopy(plan['geometry_profile'])
    appearance = _appearance(stage, metadata, plan['appearance_variant'])
    if _collision_recipe(stage, paths) != recipe:
        raise ValueError('Seed derivation changed collision representation, offsets or physics material')
    checks = static_checks(metadata)
    separations = footprint_separations(metadata)
    poses, targets, static = _projections(stage, metadata)
    return {'stage': stage, 'metadata': metadata, 'actual_body_poses_m': poses,
            'interaction_targets': targets, 'static_geometry': static,
            'fixed_support_paths': _fixed_support_paths(stage, metadata),
            'appearance': appearance, 'structural_checks': checks, 'footprint_separations': separations,
            'collision_recipe_sha256': canonical_hash(recipe)}


def _parent(release):
    release = Path(release).resolve()
    manifest_path = release/'release_manifest.json'
    manifest = load_json(manifest_path)
    if manifest.get('schema') != 'kitchen_manipulation_task_release.v1' or manifest.get('task_id') not in TASK_IDS:
        raise ValueError('Selected kitchen parent release required')
    references = {str(manifest_path): digest(manifest_path)}
    for name, value in manifest['files_sha256'].items():
        path = (release/name).resolve()
        if not path.is_relative_to(release):
            raise ValueError('Parent file escapes release')
        checked(path, value); references[str(path)] = value
    scene = load_json(release/'scene.json')
    task = load_json(release/'task.json')
    candidate, metadata = load_asset_manifest(manifest['task_id'], scene['kitchen_task_provenance']['asset_candidate'])
    if task['asset_metadata'] != metadata:
        raise ValueError('Parent physical metadata differs from frozen referee')
    for name, sha in ((candidate['manifest_path'], candidate['manifest_sha256']),
                      (candidate['usd_path'], candidate['usd_sha256']),
                      (candidate['metadata_path'], candidate['metadata_sha256'])):
        references[name] = sha
    references.update(candidate['auxiliary_files_sha256'])
    references.update({p: row['sha256'] for p, row in candidate['dependencies'].items()})
    return scene, task, candidate, metadata, references


def author_variant(parent_asset_reference, plan, output):
    """Publish a fresh compatible asset from a verified full source snapshot."""
    from pxr import Usd
    from scenes.workrooms.living_room import seed_geometry as geometry
    from scripts.maintenance.freeze_project_sources import verify_if_snapshot
    required = (__name__, geometry.__name__, shared_geometry.__name__)
    if plan['generator_version'] in SUPPORT_SOLID_VERSIONS:
        from . import assets, geometry as kitchen_geometry, produce_support_hull
        required += (produce_support_hull.__name__, assets.__name__, kitchen_geometry.__name__)
    snapshot = verify_if_snapshot(Path(__file__).resolve().parents[3],
        required_modules=required, entrypoint_file=Path(__file__))
    if not snapshot or not snapshot.get('verified'):
        raise ValueError('Prepare kitchen seed assets only from a complete frozen source snapshot')
    candidate, metadata = load_asset_manifest(plan['task_id'], parent_asset_reference)
    parent_manifest = load_json(Path(candidate['manifest_path']))
    result = derive_stage(Usd.Stage.Open(candidate['usd_path']), metadata, plan)
    output = Path(output).resolve()
    if not output.is_relative_to(ASSET_ROOT):
        raise ValueError('Kitchen seed assets belong in the versioned shared asset root')
    output.mkdir(parents=True, exist_ok=False)
    dependencies = deepcopy(parent_manifest['dependencies'])
    if plan['seed']:
        texture = shared_geometry.finish_texture(output, plan['seed'])
        result['appearance'] = _appearance(result['stage'], result['metadata'], plan['appearance_variant'],
                                          texture_path=texture, seed=plan['seed'])
        for path in (texture, output/'TEXTURE_LICENSE.txt'):
            dependencies[str(path)] = {'sha256': digest(path)}
    if plan['seed'] == 0:
        shutil.copyfile(candidate['usd_path'], output/'task.usda')
    else:
        result['stage'].GetRootLayer().Export(str(output/'task.usda'))
    if plan['seed'] == 0 and plan['generator_version'] in (CONTACT_RESET_VERSION, *SUPPORT_SOLID_VERSIONS):
        # This profile preserves the canonical input bytes, including the
        # original metadata serialization. Keep historical v1 authoring exact.
        shutil.copyfile(candidate['metadata_path'], output/'metadata.json')
    else:
        _dump(output/'metadata.json', result['metadata'])
    _dump(output/'structural_audit.json', {'schema': 'kitchen_seed_structural_audit.v1',
        'checks': result['structural_checks'], 'collision_recipe_sha256': result['collision_recipe_sha256'],
        'footprint_separations': result['footprint_separations'],
        'scope': 'CPU geometry only; no native acceptance or robot task credit'})
    shutil.copyfile(Path(candidate['manifest_path']).parent/'ATTRIBUTION.txt', output/'ATTRIBUTION.txt')
    _dump(output/'case_geometry.json', {key: result[key] for key in (
        'actual_body_poses_m', 'interaction_targets', 'static_geometry', 'fixed_support_paths', 'appearance')})
    manifest = {'schema': 'kitchen-task-asset.v1', 'task_id': plan['task_id'],
        'root_prim_path': ROOT, 'presentation': False, 'reset_scope': 'finite_seed_reset',
        'usd_sha256': digest(output/'task.usda'), 'metadata_sha256': digest(output/'metadata.json'),
        'structural_audit_sha256': digest(output/'structural_audit.json'),
        'attribution_sha256': digest(output/'ATTRIBUTION.txt'),
        'dependencies': dependencies, 'source_snapshot': snapshot,
        'parent_asset_reference': candidate, 'case_plan': plan,
        'physics_accepted': False, 'task_policy_success_claimed': False}
    for key in ('prop_profile', 'dressing_profile', 'utensil_contact_profile', 'utensil_grip_profile'):
        if key in metadata:
            manifest[key] = metadata[key]
    _dump(output/'manifest.json', manifest)
    return manifest


def prepare_asset(parent_release, seed, output, *, profile=None, attempt=0):
    if profile == "parameterized_physics_v1":
        from scenes.workrooms.parameterized.workroom import prepare_asset as prepare_parameterized
        return prepare_parameterized(parent_release, seed, output, profile=profile, attempt=attempt)
    """Unified publisher adapter; return the immutable seed manifest path."""
    _, task, candidate, metadata, references = _parent(parent_release)
    if profile is None:
        profile = SUPPORT_LOW_VERTEX_TIP_VERSION if task['id'] == 'produce_transfer' else VERSION
    plan = case_plan(task['id'], seed, metadata, profile=profile)
    output = Path(output).resolve()
    native = author_variant(candidate, plan, output)
    dependencies = {p: row['sha256'] for p, row in native['dependencies'].items()}
    dependencies.update({str(p): digest(p) for p in sorted(output.iterdir()) if p.is_file()})
    seed_manifest = {'schema': 'workroom_seed_asset.v1', 'family': 'kitchen',
        'task_id': task['id'], 'seed': seed, 'generator_version': plan['generator_version'],
        'parent_references': references, 'dependencies_sha256': dependencies,
        'asset_manifest': str(output/'manifest.json'), 'asset_manifest_sha256': digest(output/'manifest.json'),
        'case_geometry': str(output/'case_geometry.json'), 'case_geometry_sha256': digest(output/'case_geometry.json'),
        'case_plan': plan, 'license': {'source': deepcopy(metadata['source']),
            'attribution': str(output/'ATTRIBUTION.txt'), 'attribution_sha256': digest(output/'ATTRIBUTION.txt')},
        'physics_accepted': False, 'robot_manipulation_qualified': False}
    _dump(output/'seed_manifest.json', seed_manifest)
    return output/'seed_manifest.json'


def compose_case(parent_release, asset_manifest_path, configuration_id):
    from scenes.workrooms.parameterized.workroom import recognizes, compose_case as compose_parameterized
    if recognizes(asset_manifest_path):
        return compose_parameterized(parent_release, asset_manifest_path, configuration_id)
    """Replace only frozen task assets/geometry; preserve parent installation."""
    if configuration_id != 'single_arm':
        raise ValueError('These kitchen tasks retain their original single-arm installation')
    scene, task, _, _, references = _parent(parent_release)
    path = Path(asset_manifest_path).resolve()
    manifest = load_json(path)
    if (manifest.get('schema') != 'workroom_seed_asset.v1' or manifest.get('family') != 'kitchen'
            or manifest.get('task_id') != task['id'] or manifest.get('parent_references') != references
            or manifest.get('generator_version') not in (VERSION, CONTACT_RESET_VERSION, *SUPPORT_SOLID_VERSIONS)):
        raise ValueError('Seed manifest differs from selected parent/task/generator')
    seed = manifest['seed']
    if type(seed) is not int or seed not in range(10):
        raise ValueError('Unsupported seed')
    if manifest.get('case_plan') != case_plan(task['id'], seed, task['asset_metadata'],
                                             profile=manifest['generator_version']):
        raise ValueError('Seed manifest plan differs from the finite parent-bound generator')
    for filename, sha in manifest['dependencies_sha256'].items():
        checked(filename, sha)
    checked(manifest['asset_manifest'], manifest['asset_manifest_sha256'])
    candidate, metadata = load_asset_manifest(task['id'], manifest['asset_manifest'])
    native_manifest = load_json(Path(candidate['manifest_path']))
    if native_manifest.get('case_plan') != manifest['case_plan']:
        raise ValueError('Native asset plan differs from seed manifest')
    checked(manifest['case_geometry'], manifest['case_geometry_sha256'])
    projections = load_json(Path(manifest['case_geometry']))
    task['asset_metadata'] = metadata
    from bench.isaacsim.release import _frozen_evaluator_module
    module = _frozen_evaluator_module(Path(parent_release).resolve())
    rules = importlib.import_module(module.__package__+'.rules')
    task = bind_episode_task(task, seed, rules)
    scene['kitchen_task_provenance']['asset_candidate'] = candidate
    scene['kitchen_task_provenance'].update(physical_reset_seeded=True,
        seed_scope='Explicit finite seed selects a separately frozen reset/geometry asset; no runtime object writes')
    scene['native_task_bundle'].update(usd_path=candidate['usd_path'], usd_sha256=candidate['usd_sha256'],
                                     rigid_bodies=deepcopy(metadata['actors']))
    scene['native_task_bundle']['observer']['configuration'] = deepcopy(task)
    source_files = {str(p.relative_to(Path(parent_release)/'evaluator')): p.read_text()
                    for p in sorted((Path(parent_release)/'evaluator').rglob('*.py'))}
    asset_files = {**references, **manifest['dependencies_sha256'], str(path): digest(path)}
    return {'scene': scene, 'task': task, 'initial_state': deepcopy(task['initial_state']),
            'evaluator_sources': source_files, 'actual_body_poses_m': projections['actual_body_poses_m'],
            'interaction_targets': projections['interaction_targets'],
            'fixed_support_paths': projections['fixed_support_paths'],
            'static_geometry': projections['static_geometry'], 'asset_files_sha256': asset_files}
