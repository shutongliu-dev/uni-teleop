"""Versioned side-tray layouts and physical static fixtures for tower tasks.

The pure layout helpers can be frozen with task rules without importing USD.
Trays rest on the unchanged table. Their static fixation is an explicit task
fixture assumption; mass estimates describe construction, not dynamic actors.
"""
from __future__ import annotations

from copy import deepcopy
import math


PROFILE = 'trays_v1'
BUILDING_GEOMETRY = {'block_size_m': [.15, .045, .025], 'layers': 6,
                     'in_layer_gap_m': .001}
EXTRACTION_GEOMETRY = {'block_size_m': [.20, .045, .030], 'layers': 12,
                       'in_layer_gap_m': .010}
FLOOR_THICKNESS_M = .006
WALL_THICKNESS_M = .006
WALL_HEIGHT_M = .012
WALL_NAMES = ('neg_x', 'pos_x', 'neg_y', 'pos_y')
TRAY_PHYSICS_PRIOR = {
    'density_kg_m3': 650., 'static_friction': .45, 'dynamic_friction': .35,
    'restitution': 0., 'contact_offset_m': .0001, 'rest_offset_m': 0.,
    'source': 'Uncalibrated homogeneous wooden-tray engineering prior; same contact material as the wooden blocks.',
    'fixture': 'Task-owned static tray resting on the original table; fixed in place by explicit fixture assumption.',
    'mass_semantics': 'Geometric construction estimate only; static colliders have no simulated mass or inertia.',
    'native_accepted': False,
}


def building_supply_trays():
    """Two tray interiors, each accommodating nine two-block supply stacks."""
    trays = []
    for name, xmin in (('supply_left', .245), ('supply_right', .805)):
        actor = 'external/' + name
        trays.append({'id': name, 'interior_bounds_xy_m': [[xmin, -.43], [xmin + .25, .17]],
                      'floor_z_m': .786,
                      'contact_actor_ids': [actor, *[actor + '/wall_' + name for name in WALL_NAMES]]})
    return trays


def building_supply_layout():
    """36 blocks; 35 mm lateral and 50 mm end gaps between stack columns."""
    layout = {}
    q = [math.sqrt(.5), 0., 0., math.sqrt(.5)]
    for bank in range(2):
        for layer in range(2):
            for row in range(3):
                for column in range(3):
                    index = bank * 18 + layer * 9 + row * 3 + column
                    layout[f'block_{index:02d}'] = {
                        'position_m': [(.290 if bank == 0 else .850) + column * .080,
                                       -.330 + row * .200, .786 + (layer + .5) * .025],
                        'quaternion_wxyz': list(q)}
    return layout


def extraction_collection_trays():
    """Symmetric 3 by 5 single-layer collection trays; the tower stays on table."""
    trays = {}
    for side, yrange in (('front', (-.48, -.17)), ('rear', (.17, .48))):
        stem = side + '_collection_tray'
        trays[side] = {'actor_id': stem + '_floor',
                       'interior_bounds_xy_m': [[.29, yrange[0]], [1.01, yrange[1]]],
                       'floor_z_m': .786, 'slot_pitch_xy_m': [.24, .06],
                       'wall_actor_ids': [stem + '_wall_' + name for name in WALL_NAMES]}
    return trays


def _fixture_specs(task_id):
    if task_id == 'tower_building':
        return [{'id': row['id'], 'floor_actor_id': row['contact_actor_ids'][0],
                 'wall_actor_ids': row['contact_actor_ids'][1:],
                 'interior_bounds_xy_m': row['interior_bounds_xy_m'],
                 'floor_z_m': row['floor_z_m'], 'color': 'Supply'}
                for row in building_supply_trays()]
    if task_id == 'tower_extraction':
        return [{'id': side + '_collection_tray', 'floor_actor_id': row['actor_id'],
                 'wall_actor_ids': row['wall_actor_ids'],
                 'interior_bounds_xy_m': row['interior_bounds_xy_m'],
                 'floor_z_m': row['floor_z_m'], 'color': side.title()}
                for side, row in extraction_collection_trays().items()]
    raise ValueError('Unknown tower tray task')


def author_trays(stage, root_path, task_id, materials):
    """Author two real floors and eight rim walls, with exact contact actor paths."""
    from pxr import Gf, Sdf, UsdGeom, UsdPhysics, UsdShade

    UsdGeom.Scope.Define(stage, root_path + '/Trays')
    receipts, actor_paths = [], {}
    for spec in _fixture_specs(task_id):
        path = root_path + '/Trays/' + spec['id']
        UsdGeom.Xform.Define(stage, path)
        UsdGeom.Scope.Define(stage, path + '/Walls')
        (xmin, ymin), (xmax, ymax) = spec['interior_bounds_xy_m']
        cx, cy = (xmin + xmax) / 2, (ymin + ymax) / 2
        sx, sy, z = xmax - xmin, ymax - ymin, spec['floor_z_m']
        t, h = WALL_THICKNESS_M, WALL_HEIGHT_M
        parts = [('Floor', spec['floor_actor_id'], [cx, cy, z - FLOOR_THICKNESS_M / 2],
                  [sx + 2 * t, sy + 2 * t, FLOOR_THICKNESS_M])]
        for name, actor, position, size in zip(WALL_NAMES, spec['wall_actor_ids'],
                ([xmin - t / 2, cy, z + h / 2], [xmax + t / 2, cy, z + h / 2],
                 [cx, ymin - t / 2, z + h / 2], [cx, ymax + t / 2, z + h / 2]),
                ([t, sy, h], [t, sy, h], [sx + 2 * t, t, h], [sx + 2 * t, t, h]), strict=True):
            parts.append(('Walls/' + name, actor, position, size))
        components = []
        for name, actor, position, size in parts:
            prim_path = path + '/' + name
            shape = UsdGeom.Cube.Define(stage, prim_path)
            shape.CreateSizeAttr(1.)
            transform = UsdGeom.Xformable(shape)
            transform.AddTranslateOp().Set(Gf.Vec3d(*position))
            transform.AddScaleOp().Set(Gf.Vec3d(*size))
            prim = shape.GetPrim()
            UsdPhysics.CollisionAPI.Apply(prim).CreateCollisionEnabledAttr(True)
            prim.AddAppliedSchema('PhysxCollisionAPI')
            for field, value in (('contactOffset', TRAY_PHYSICS_PRIOR['contact_offset_m']), ('restOffset', 0.)):
                prim.CreateAttribute('physxCollision:' + field, Sdf.ValueTypeNames.Float).Set(value)
            binding = UsdShade.MaterialBindingAPI.Apply(prim)
            binding.Bind(materials['TrayFloor'] if name == 'Floor' else materials[spec['color']])
            binding.Bind(materials['Physics'], materialPurpose='physics')
            prim.SetCustomDataByKey('living_task_contact_actor', actor)
            prim.SetCustomDataByKey('task_fixture_static', True)
            actor_paths[actor] = prim_path
            components.append({'name': name, 'actor_id': actor, 'prim_path': prim_path,
                'position_m': position, 'size_m': size,
                'bounds_m': [[position[i] + sign * size[i] / 2 for i in range(3)] for sign in (-1, 1)],
                'construction_mass_kg': math.prod(size) * TRAY_PHYSICS_PRIOR['density_kg_m3'],
                'collision': True, 'dynamic': False, 'kinematic': False,
                'support_role': 'floor' if name == 'Floor' else 'wall'})
        receipts.append({**deepcopy(spec), 'prim_path': path, 'components': components,
            'floor_thickness_m': FLOOR_THICKNESS_M, 'wall_thickness_m': t, 'wall_height_m': h,
            'outer_bounds_xy_m': [[xmin - t, ymin - t], [xmax + t, ymax + t]],
            'bottom_z_m': z - FLOOR_THICKNESS_M, 'rim_top_z_m': z + h,
            'construction_mass_kg': sum(row['construction_mass_kg'] for row in components),
            'physics_prior': deepcopy(TRAY_PHYSICS_PRIOR)})
    return {'trays': receipts, 'contact_actor_paths': actor_paths,
            'static_collision_paths': list(actor_paths.values())}
