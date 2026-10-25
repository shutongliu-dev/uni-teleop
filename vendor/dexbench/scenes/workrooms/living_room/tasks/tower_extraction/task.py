"""Parameter-only task4 configuration; no simulator adapter or released assets."""
from __future__ import annotations

import math
import hashlib
import json

from bench.paths import portable
from scenes.workrooms.living_room.scene import environment_reference
from scenes.workrooms.living_room.tower_geometry import initial_tower, validate_geometry
from .rules import public_rules, starting_side


def _tray_collection(geometry, trays, count, thresholds):
    """Bind collection slots to explicit, symmetric physical tray floors."""
    if not isinstance(trays, dict) or set(trays) != {'front', 'rear'}:
        raise ValueError('collection_trays must declare front and rear trays')
    length, width, height = geometry['block_size_m']
    cx, cy = geometry['center_xy_m']
    top = geometry['table_top_z_m']
    table_low, table_high = geometry['table_bounds_xy_m']
    margin = max(height * .25, geometry['in_layer_gap_m'])
    padding = margin / 4
    normalized = {}
    actors = {'table', 'floor', 'hand/front', 'hand/rear', 'robot/front', 'robot/rear'}
    actors.update(item['block_id'] for item in initial_tower(geometry))

    def vector(value):
        if not isinstance(value, (list, tuple)) or len(value) != 2:
            raise ValueError('Tray bounds and pitch must contain two coordinates')
        if any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) for x in value):
            raise ValueError('Tray coordinates must be finite numbers')
        return [float(x) for x in value]

    for side in ('front', 'rear'):
        tray = trays[side]
        required = {'actor_id', 'interior_bounds_xy_m', 'floor_z_m'}
        if (not isinstance(tray, dict) or not required <= set(tray)
                or set(tray) - required - {'wall_actor_ids', 'slot_pitch_xy_m'}):
            raise ValueError('Invalid collection tray fields')
        walls = tray.get('wall_actor_ids', [])
        if not isinstance(walls, (list, tuple)):
            raise ValueError('wall_actor_ids must list distinct physical wall actors')
        for actor in [tray['actor_id'], *walls]:
            if not isinstance(actor, str) or not actor or '/' in actor or actor in actors:
                raise ValueError('Tray actors must be distinct nonreserved actor IDs')
            actors.add(actor)
        bounds = tray['interior_bounds_xy_m']
        if not isinstance(bounds, (list, tuple)) or len(bounds) != 2:
            raise ValueError('Tray interior bounds must be [low_xy, high_xy]')
        low, high = [vector(row) for row in bounds]
        if any(not table_low[i] <= low[i] < high[i] <= table_high[i] for i in (0, 1)):
            raise ValueError('Tray interior must lie within the parent table bounds')
        if ((side == 'front' and high[1] > cy - length / 2 - margin + 1.e-12)
                or (side == 'rear' and low[1] < cy + length / 2 + margin - 1.e-12)):
            raise ValueError('Tray interior must leave the tower footprint clear')
        floor = tray['floor_z_m']
        if (isinstance(floor, bool) or not isinstance(floor, (int, float))
                or not math.isfinite(floor) or not top < floor <= top + height):
            raise ValueError('Tray floor must be above the table and no higher than one block height')
        pitch = vector(tray.get('slot_pitch_xy_m', [length + margin, width + margin]))
        if any(pitch[i] < size + margin - 1.e-12 for i, size in enumerate((length, width))):
            raise ValueError('Tray slot pitch must preserve block separation and placement tolerance')
        normalized[side] = {'actor_id': tray['actor_id'], 'wall_actor_ids': list(walls),
                            'interior_bounds_xy_m': [low, high], 'floor_z_m': float(floor),
                            'slot_pitch_xy_m': pitch}
    front, rear = normalized['front'], normalized['rear']
    if (abs(front['floor_z_m'] - rear['floor_z_m']) > 1.e-12
            or front['slot_pitch_xy_m'] != rear['slot_pitch_xy_m']
            or any(abs(front['interior_bounds_xy_m'][edge][axis]
                       + rear['interior_bounds_xy_m'][1-edge][axis] - 2 * center) > 1.e-12
                   for edge in (0, 1) for axis, center in enumerate((cx, cy)))):
        raise ValueError('Collection trays must be symmetric about the tower center')
    low, high = front['interior_bounds_xy_m']
    pitch = front['slot_pitch_xy_m']
    nx, ny = [math.floor((high[i] - low[i] - size - 2 * padding) / pitch[i] + 1.e-12) + 1
              for i, size in enumerate((length, width))]
    if nx <= 0 or ny <= 0 or nx * ny < count:
        raise ValueError('Trays cannot hold the requested single-layer collection capacity')
    start = [(low[i] + high[i] - (n - 1) * pitch[i]) / 2 for i, n in enumerate((nx, ny))]
    ztol = thresholds['collection_height_tolerance_h'] * height
    slots = {'front': [], 'rear': []}
    for index in range(count):
        row, col = divmod(index, nx)
        x, y = start[0] + col * pitch[0], start[1] + row * pitch[1]
        for side, px, py in (('front', x, y), ('rear', 2 * cx - x, 2 * cy - y)):
            floor = normalized[side]['floor_z_m']
            bounds = [[px - length/2 - padding, py - width/2 - padding, floor - ztol],
                      [px + length/2 + padding, py + width/2 + padding, floor + height + ztol]]
            slots[side].append({'slot_id': f'{side}_collection_{index:02d}',
                'position_m': [px, py, floor + height/2], 'yaw_rad': 0. if side == 'front' else math.pi,
                'support_actor_id': normalized[side]['actor_id'],
                'bounds_h': [[(p[0]-cx)/height, (p[1]-cy)/height, (p[2]-top)/height] for p in bounds]})
    return normalized, slots


def task_configuration(geometry, *, seed=0, max_turns_per_player=None,
                       protected_bottom_layers=1, thresholds=None, collection_trays=None):
    """Reserve symmetric single-layer collection slots, or fail closed.

    Block dimensions are required. Test dimensions never become frozen source
    assets. The future adapter must bind real bodies, contacts and reset IDs.
    Optional collection_trays declares each side's floor actor, interior XY
    bounds, floor height, wall actors and slot pitch. Omitting it preserves the
    historical table-supported collection configuration exactly.
    """
    g = validate_geometry(geometry)
    first = starting_side(seed)
    rules = public_rules(g, max_turns_per_player=max_turns_per_player,
                         protected_bottom_layers=protected_bottom_layers, thresholds=thresholds,
                         collection_surface='tray_floor' if collection_trays is not None else 'table')
    max_turns_per_player = rules['max_turns_per_player']
    length, width, height = g['block_size_m']
    cx, cy = g['center_xy_m']; top = g['table_top_z_m']
    low, high = g['table_bounds_xy_m']
    margin = max(height * .25, g['in_layer_gap_m'])
    # Use only the table area available symmetrically about the tower centre.
    half_x = min(cx - low[0], high[0] - cx) - margin
    half_y = min(cy - low[1], high[1] - cy) - margin
    xmin, xmax = cx - half_x, cx + half_x
    ymin, ymax = cy - half_y, cy - length / 2 - margin
    nx = math.floor((xmax - xmin + margin) / (length + margin) + 1.e-12)
    ny = math.floor((ymax - ymin + margin) / (width + margin) + 1.e-12)
    if collection_trays is None and (nx <= 0 or ny <= 0 or nx * ny < max_turns_per_player):
        raise ValueError('Table cannot hold the requested single-layer collection capacity on both sides')
    slots = {'front': [], 'rear': []}
    padding = margin / 4  # Leave half the inter-slot gap for real placement tolerance.
    ztol = rules['thresholds']['collection_height_tolerance_h'] * height
    for index in range(max_turns_per_player if collection_trays is None else 0):
        row, col = divmod(index, nx)
        x, y = xmin + length / 2 + col * (length + margin), ymin + width / 2 + row * (width + margin)
        for side, px, py in (('front', x, y), ('rear', 2 * cx - x, 2 * cy - y)):
            bounds = [[px - length / 2 - padding, py - width / 2 - padding, top - ztol],
                      [px + length / 2 + padding, py + width / 2 + padding, top + height + ztol]]
            if any(bounds[0][axis] < low[axis] - 1.e-12 or bounds[1][axis] > high[axis] + 1.e-12 for axis in (0, 1)):
                raise ValueError('Collection slot crosses the parent table bounds')
            slots[side].append({'slot_id': f'{side}_collection_{index:02d}',
                'position_m': [px, py, top + height / 2], 'yaw_rad': 0. if side == 'front' else math.pi,
                'bounds_h': [[(p[0] - cx) / height, (p[1] - cy) / height, (p[2] - top) / height] for p in bounds]})
    if collection_trays is not None:
        collection_trays, slots = _tray_collection(g, collection_trays, max_turns_per_player, rules['thresholds'])
    blocks = {}
    for item in initial_tower(g):
        yaw = item['yaw_rad']; p = item['position_m']
        blocks[item['block_id']] = {'layer': item['layer'], 'index': item['index'],
            'center_h': [(p[0] - cx) / height, (p[1] - cy) / height, (p[2] - top) / height],
            'quaternion_wxyz': [math.cos(yaw / 2), 0., 0., math.sin(yaw / 2)],
            'half_extents_h': [length / height / 2, width / height / 2, .5]}
    configuration = {'schema': 'living_tower_extraction_configuration.v1', 'family': 'living_room',
        'task_id': 'tower_extraction', 'task_number': 4, 'status': 'planned',
        'geometry': g, 'seed': seed, 'first_side': first, 'rules': rules,
        'blocks': blocks, 'collection_slots': slots,
        'robot_layout_id': 'living_room_opposed_dual_right_sharpa_v1',
        'parent_environment': environment_reference(),
        'source_release': None, 'native_adapter': None, 'native_execution_available': False,
        'scope': 'Parameter-bound CPU reference configuration only; no reset, collision or native task qualification'}
    if collection_trays is not None:
        configuration['collection_trays'] = collection_trays
    configuration['configuration_sha256'] = hashlib.sha256(json.dumps(portable(configuration),
        sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')).hexdigest()
    return configuration
