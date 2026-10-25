"""Create a geometry-bound plan with separated supply, never a native release."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import math

from .rules import canonical_hash, number, quaternion, rotation, rules_for_geometry, vector

TASK_SCHEMA = 'living_tower_building_plan_v1'


def _bounds(row, size):
    p, r = row['position_m'], rotation(row['quaternion_wxyz'])
    extents = [sum(abs(r[i][j])*size[j]/2 for j in range(3)) for i in range(3)]
    return [p[i]-extents[i] for i in range(3)], [p[i]+extents[i] for i in range(3)]


def _overlap_xy(first, second, margin=0.):
    return all(first[0][i] < second[1][i]+margin and second[0][i] < first[1][i]+margin for i in (0, 1))


def _supply(geometry, slots, ids, seed, supplied):
    size = geometry['block_size_m']
    bounds = geometry['table_bounds_xy_m']
    clearance = max(.002, size[1]*.1)
    slot_boxes = [_bounds({'position_m': slot['position_m'],
                          'quaternion_wxyz': [math.cos(slot['yaw_rad']/2), 0., 0., math.sin(slot['yaw_rad']/2)]}, size)
                  for slot in slots]
    if supplied is None:
        alternatives = []
        for yaw in (0., math.pi/2):
            dx, dy = (size[0], size[1]) if yaw == 0. else (size[1], size[0])
            nx = max(0, math.floor((bounds[1][0]-bounds[0][0]-clearance)/(dx+clearance)))
            ny = max(0, math.floor((bounds[1][1]-bounds[0][1]-clearance)/(dy+clearance)))
            rows = []
            for ix in range(nx):
                for iy in range(ny):
                    p = [bounds[0][0]+clearance+dx/2+ix*(dx+clearance),
                         bounds[0][1]+clearance+dy/2+iy*(dy+clearance),
                         geometry['table_top_z_m']+size[2]/2]
                    row = {'position_m': p, 'quaternion_wxyz': [math.cos(yaw/2), 0., 0., math.sin(yaw/2)]}
                    if not any(_overlap_xy(_bounds(row, size), box, clearance) for box in slot_boxes):
                        rows.append(row)
            alternatives.append(rows)
        candidates = max(alternatives, key=len)
        if len(candidates) < len(ids):
            raise ValueError('All supply blocks do not fit the original table outside the tower; choose smaller geometry or separately design an explicit supported supply area')
        def rank(row):
            return hashlib.sha256((str(seed)+':'+canonical_hash(row)).encode()).digest()
        supplied = dict(zip(ids, sorted(candidates, key=rank)[:len(ids)]))
    if not isinstance(supplied, dict) or set(supplied) != set(ids):
        raise ValueError('Supply must contain each physical block exactly once')
    result, boxes = {}, []
    for name in ids:
        row = supplied[name]
        p, q = vector(row['position_m'], 3, 'supply position'), quaternion(row['quaternion_wxyz'])
        r = rotation(q)
        if abs(abs(r[2][2])-1.) > 1.e-9:
            raise ValueError('Supply blocks must lie flat on the original tabletop')
        # Axis-aligned supply is deliberately conservative; no packing solver or
        # assumptions about additional trays are hidden in this plan factory.
        if min(abs(r[0][0]), abs(r[0][1])) > 1.e-9:
            raise ValueError('Supply yaw must be a multiple of 90 degrees')
        value = {'position_m': list(p), 'quaternion_wxyz': list(q)}
        box = _bounds(value, size)
        if (abs(box[0][2]-geometry['table_top_z_m']) > 1.e-9
                or any(box[0][i] < bounds[0][i]+clearance-1.e-9 or box[1][i] > bounds[1][i]-clearance+1.e-9 for i in (0, 1))):
            raise ValueError('Supply block is not fully supported inside the original table')
        if any(_overlap_xy(box, other, clearance-1.e-9) for other in [*slot_boxes, *boxes]):
            raise ValueError('Supply blocks must be separated and outside every target footprint')
        result[name], boxes = value, [*boxes, box]
    return result


def _trays(geometry, slots, supplied):
    """Bind real tray interiors and their observed actors, never a virtual floor."""
    if not isinstance(supplied, list) or not supplied:
        raise ValueError('supply_trays must declare at least one real tray')
    result, names = [], set()
    table = geometry['table_bounds_xy_m']
    size = geometry['block_size_m']
    slot_boxes = [_bounds({'position_m': slot['position_m'],
        'quaternion_wxyz': [math.cos(slot['yaw_rad']/2), 0., 0., math.sin(slot['yaw_rad']/2)]}, size)
        for slot in slots]
    for row in supplied:
        if not isinstance(row, dict) or set(row) != {'id', 'interior_bounds_xy_m', 'floor_z_m', 'contact_actor_ids'}:
            raise ValueError('Tray declarations require id, interior_bounds_xy_m, floor_z_m and contact_actor_ids')
        name = row['id']
        if not isinstance(name, str) or not name or '/' in name or name in names:
            raise ValueError('Tray IDs must be distinct nonempty actor names')
        actors = row['contact_actor_ids']
        floor_actor = 'external/'+name
        if (not isinstance(actors, list) or not actors or any(not isinstance(actor, str) for actor in actors)
                or len(set(actors)) != len(actors) or floor_actor not in actors
                or any(actor != floor_actor and (not actor.startswith(floor_actor+'/')
                    or not actor.rsplit('/', 1)[-1]) for actor in actors)):
            raise ValueError('Tray contact_actor_ids must name its unique floor and optional components')
        bounds = row['interior_bounds_xy_m']
        if not isinstance(bounds, (list, tuple)) or len(bounds) != 2:
            raise ValueError('Tray interior bounds must contain low/high XY corners')
        lo, hi = (vector(value, 2, 'tray interior bounds') for value in bounds)
        z = number(row['floor_z_m'], 'tray floor height')
        if any(not table[0][i] <= lo[i] < hi[i] <= table[1][i] for i in (0, 1)) or z < geometry['table_top_z_m']:
            raise ValueError('Tray interior must be supported above the original tabletop')
        if any(_overlap_xy((lo, hi), box) for box in slot_boxes):
            raise ValueError('Supply trays must be outside every target footprint')
        if any(_overlap_xy((lo, hi), other['interior_bounds_xy_m']) for other in result):
            raise ValueError('Supply tray interiors must not overlap')
        names.add(name)
        result.append({'id': name, 'interior_bounds_xy_m': [list(lo), list(hi)],
                       'floor_z_m': z, 'contact_actor_ids': list(actors)})
    return result


def _tray_supply(geometry, ids, trays, supplied):
    """Validate separated stacks inside declared physical trays at reset."""
    if not isinstance(supplied, dict) or set(supplied) != set(ids):
        raise ValueError('Tray supply_layout must contain each physical block exactly once')
    size = geometry['block_size_m']
    clearance = max(.002, size[1]*.1)
    result, boxes, owners = {}, {}, {}
    for name in ids:
        row = supplied[name]
        p, q = vector(row['position_m'], 3, 'supply position'), quaternion(row['quaternion_wxyz'])
        r = rotation(q)
        if abs(abs(r[2][2])-1.) > 1.e-9 or min(abs(r[0][0]), abs(r[0][1])) > 1.e-9:
            raise ValueError('Tray supply blocks must lie flat with axis-aligned yaw')
        value = {'position_m': list(p), 'quaternion_wxyz': list(q)}
        box = _bounds(value, size)
        inside = [tray for tray in trays if all(
            box[0][i] >= tray['interior_bounds_xy_m'][0][i]+clearance-1.e-9
            and box[1][i] <= tray['interior_bounds_xy_m'][1][i]-clearance+1.e-9 for i in (0, 1))]
        if len(inside) != 1 or box[0][2] < inside[0]['floor_z_m']-1.e-9:
            raise ValueError('Every supply block must be inside one real tray above its floor')
        result[name], boxes[name], owners[name] = value, box, inside[0]
    for i, name in enumerate(ids):
        box = boxes[name]
        for other in ids[i+1:]:
            second = boxes[other]
            if (min(box[1][2], second[1][2])-max(box[0][2], second[0][2]) > 1.e-9
                    and _overlap_xy(box, second, clearance-1.e-9)):
                raise ValueError('Supply stacks must be separated and blocks must not overlap')
        if abs(box[0][2]-owners[name]['floor_z_m']) <= 1.e-9:
            continue
        lower = [other for other in ids if other != name and owners[other]['id'] == owners[name]['id']
            and abs(boxes[other][1][2]-box[0][2]) <= 1.e-9
            and all(box[0][axis] >= boxes[other][0][axis]-1.e-9
                    and box[1][axis] <= boxes[other][1][axis]+1.e-9 for axis in (0, 1))]
        if len(lower) != 1:
            raise ValueError('Every stacked supply block needs full support from a lower block')
    return result


def task_configuration(geometry, *, seed=0, supply_layout=None, inventory_count=None,
                       supply_trays=None, budget=None,
                       stable_duration_s=1., tolerances=None, floor_drop_is_terminal=None,
                       scoring_mode=None):
    from scenes.workrooms.living_room.tower_geometry import block_ids, tower_slots
    from scenes.workrooms.living_room.dual_robot_layout import LAYOUT_ID, placements
    from scenes.workrooms.living_room.scene import environment_reference
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError('Supply seed must be an unsigned 32-bit integer')
    rules = rules_for_geometry(geometry, stable_duration_s=stable_duration_s,
                              tolerances=tolerances, floor_drop_is_terminal=floor_drop_is_terminal,
                              inventory_count=inventory_count, scoring_mode=scoring_mode)
    g = rules['geometry']
    if scoring_mode == 'freeform_height':
        ids = [f'block_{index:02d}' for index in range(inventory_count)]
        slots = []
        if supply_trays is None:
            raise ValueError('Freeform reset requires physical supply trays to exclude reset height credit')
    else:
        ids, slots = block_ids(g), tower_slots(g)
        if len(ids) != g['layers']*3 or len(set(ids)) != len(ids) or len(slots) != len(ids):
            raise ValueError('Shared tower geometry returned an inconsistent inventory')
    trays = None if supply_trays is None else _trays(g, slots, supply_trays)
    initial = (_supply(g, slots, ids, seed, supply_layout) if trays is None
               else _tray_supply(g, ids, trays, supply_layout))
    if trays is not None:
        rules['supply'] = {'tray_contacts': [actor for tray in trays for actor in tray['contact_actor_ids']],
            'surplus': 'All physical blocks remain observed. Tray-supported supplies earn no height credit and do not constrain unconnected scored assemblies.'}
    limits = {'max_sim_time_s': None, 'max_robot_actions': None}
    if budget is not None:
        if not isinstance(budget, dict) or set(budget)-set(limits):
            raise ValueError('Unknown budget field')
        limits.update(budget)
    if limits['max_sim_time_s'] is not None and number(limits['max_sim_time_s'], 'time budget') <= 0:
        raise ValueError('Time budget must be positive')
    if limits['max_robot_actions'] is not None and (type(limits['max_robot_actions']) is not int or limits['max_robot_actions'] <= 0):
        raise ValueError('Action budget must be a positive integer')
    plan = {'schema': TASK_SCHEMA, 'id': 'tower_building', 'task_number': 3,
        'status': 'planned', 'implementation': None, 'native_qualified': False,
        'native_execution_available': False, 'source_release': None,
        'environment': environment_reference(),
        'geometry': deepcopy(g), 'block_ids': list(ids), 'target_slots': deepcopy(slots),
        'rules': rules, 'episode_seed': seed, 'budget': limits,
        'initial_objects': initial,
        'robot_layout': {'arm_count': 1, 'hand_id': 'sharpa_w01_right',
                         'source_layout_id': LAYOUT_ID, 'selected_sides': ['front'],
                         **placements()['front'],
                         'scope': 'Placement, reach and native composition remain to be qualified'},
        'evaluator': {'reference_implementation': 'scenes.workrooms.living_room.tasks.tower_building.evaluator:TowerBuildingEvaluator',
                      'native_adapter': None},
        'scope': 'Geometry-bound CPU rule plan only; no USD assets, object dimensions approval, frozen release or native task is created'}
    if inventory_count is not None:
        plan['inventory_count'] = inventory_count
    if trays is not None:
        plan['supply_trays'] = trays
    if scoring_mode is not None:
        plan['scoring_mode'] = scoring_mode
    plan['configuration_sha256'] = canonical_hash(plan)
    return plan


def validate_configuration(configuration):
    """Reject tampered targets, rules, inventory or an already rewarded reset."""
    if not isinstance(configuration, dict) or configuration.get('schema') != TASK_SCHEMA:
        raise ValueError('Expected a parameterized tower-building plan')
    rules = configuration['rules']
    expected = task_configuration(configuration['geometry'], seed=configuration['episode_seed'],
        supply_layout=configuration['initial_objects'], budget=configuration['budget'],
        inventory_count=configuration.get('inventory_count'), supply_trays=configuration.get('supply_trays'),
        scoring_mode=configuration.get('scoring_mode'),
        stable_duration_s=rules['goal']['stable_duration_s'], tolerances=rules['tolerances'],
        floor_drop_is_terminal=rules['floor_drop_is_terminal'])
    if expected != configuration:
        raise ValueError('Configuration differs from its geometry, rules or source supply contract')
    return deepcopy(expected)
