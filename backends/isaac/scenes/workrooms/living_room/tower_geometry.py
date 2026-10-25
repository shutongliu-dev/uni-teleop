"""Dimension-parameterized tower design on the pinned living-room table.

No default wooden-block size, authored assets, simulator or native acceptance.
Layers touch vertically; ``in_layer_gap_m`` only separates parallel neighbours.
"""
from __future__ import annotations

from copy import deepcopy
import math

SCHEMA = 'living_room_tower_geometry.v1'
DEFAULTS = {'layers': 6, 'blocks_per_layer': 3, 'in_layer_gap_m': .001,
            'center_xy_m': [.65, 0.], 'table_top_z_m': .78,
            'table_bounds_xy_m': [[.15, -.50], [1.15, .50]]}


def _number(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(name + ' must be a finite number')
    return float(value)


def _vector(value, count, name):
    if not isinstance(value, (list, tuple)) or len(value) != count:
        raise ValueError(name + f' must contain {count} numbers')
    return [_number(x, name) for x in value]


def validate_geometry(geometry):
    """Return independent canonical geometry, rejecting implicit size choices."""
    if not isinstance(geometry, dict):
        raise ValueError('Tower geometry must be a mapping')
    if 'layer_gap_m' in geometry:
        raise ValueError('Use in_layer_gap_m for horizontal gaps; vertical layer gaps are forbidden')
    unknown = set(geometry) - set(DEFAULTS) - {'schema', 'block_size_m'}
    if unknown:
        raise ValueError('Unknown tower geometry fields: ' + ', '.join(sorted(unknown)))
    if geometry.get('schema', SCHEMA) != SCHEMA:
        raise ValueError('Unsupported tower geometry schema')
    if 'block_size_m' not in geometry:
        raise ValueError('block_size_m is required; the user has not frozen wooden-block dimensions')
    result = {**deepcopy(DEFAULTS), **deepcopy(geometry), 'schema': SCHEMA}
    length, width, height = _vector(result['block_size_m'], 3, 'block_size_m')
    if not 0 < height <= width < length:
        raise ValueError('Blocks require 0 < height <= width < length')
    result['block_size_m'] = [length, width, height]
    if type(result['layers']) is not int or not 3 <= result['layers'] <= 12:
        raise ValueError('Tower layers must be an integer in [3, 12]')
    if type(result['blocks_per_layer']) is not int or result['blocks_per_layer'] != 3:
        raise ValueError('This rules version uses exactly three blocks per layer')
    gap = _number(result['in_layer_gap_m'], 'in_layer_gap_m')
    if not 0 <= gap <= width / 4:
        raise ValueError('Horizontal gap must lie between zero and one quarter block width')
    if length + 1e-12 < 3 * width + 2 * gap:
        raise ValueError('Block length must cover all three crossing blocks and horizontal gaps')
    result['in_layer_gap_m'] = gap
    result['center_xy_m'] = _vector(result['center_xy_m'], 2, 'center_xy_m')
    result['table_top_z_m'] = _number(result['table_top_z_m'], 'table_top_z_m')
    bounds = result['table_bounds_xy_m']
    if not isinstance(bounds, (list, tuple)) or len(bounds) != 2:
        raise ValueError('table_bounds_xy_m must be [low_xy, high_xy]')
    result['table_bounds_xy_m'] = [_vector(row, 2, 'table_bounds_xy_m') for row in bounds]
    if (result['table_top_z_m'] != DEFAULTS['table_top_z_m']
            or result['table_bounds_xy_m'] != DEFAULTS['table_bounds_xy_m']):
        raise ValueError('Tower tasks must preserve the pinned parent table dimensions')
    # A 20 mm design border keeps the target footprint inside the actual table.
    for axis, centre in enumerate(result['center_xy_m']):
        if not bounds[0][axis] + .02 <= centre - length/2:
            raise ValueError('Tower footprint crosses the table design border')
        if not centre + length/2 <= bounds[1][axis] - .02:
            raise ValueError('Tower footprint crosses the table design border')
    if result['table_top_z_m'] + result['layers'] * height > 1.95:
        raise ValueError('Tower exceeds the parent reserved task volume height')
    return result


def block_ids(geometry):
    g = validate_geometry(geometry)
    return [f'block_{index:02d}' for index in range(g['layers'] * g['blocks_per_layer'])]


def tower_slots(geometry):
    """Nominal alternating target slots, with no vertical air gap or hidden jig."""
    g = validate_geometry(geometry)
    _, width, height = g['block_size_m']
    cx, cy = g['center_xy_m']
    slots = []
    for layer in range(g['layers']):
        yaw = (layer % 2) * math.pi / 2
        for index in range(3):
            offset = (index - 1) * (width + g['in_layer_gap_m'])
            slots.append({'slot_id': f'layer_{layer:02d}_slot_{index}',
                          'layer': layer, 'index': index,
                          'position_m': [cx - math.sin(yaw) * offset,
                                         cy + math.cos(yaw) * offset,
                                         g['table_top_z_m'] + (layer + .5) * height],
                          'yaw_rad': yaw})
    return slots


def initial_tower(geometry):
    """Reset-only task4 recipe; this is not a settled or physically accepted tower."""
    return [{'block_id': name, **slot}
            for name, slot in zip(block_ids(geometry), tower_slots(geometry), strict=True)]


def geometry_contract():
    return {'schema': SCHEMA, 'required_parameters': ['block_size_m'],
            'block_size_order': ['length', 'width', 'height'],
            'defaults': deepcopy(DEFAULTS),
            'vertical_layer_gap_m': 0.,
            'ownership': 'Task-owned independent wooden rigid bodies, no glue or tower fixture.',
            'dimension_status': 'Unspecified until explicitly supplied; test sizes are not asset selections.',
            'physics_accepted': False}
