"""Parameterized single-arm tower rules; no block dimensions or native release."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math

from bench.paths import portable

RULES_VERSION = 'living_tower_building_rules_v1'
OBSERVATION_SCHEMA = 'living_tower_normalized_observation_v1'


def number(value, name):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(name+' must be a finite number')
    return float(value)


def vector(value, length, name):
    if not isinstance(value, (list, tuple)) or len(value) != length:
        raise ValueError(name+' has the wrong dimension')
    return tuple(number(item, name) for item in value)


def quaternion(value):
    q = vector(value, 4, 'quaternion_wxyz')
    length = math.sqrt(sum(v*v for v in q))
    if abs(length-1.) > 1.e-6:
        raise ValueError('A unit wxyz quaternion is required')
    return tuple(v/length for v in q)


def rotation(q):
    w, x, y, z = quaternion(q)
    return ((1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)),
            (2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)),
            (2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)))


def canonical_hash(value):
    return hashlib.sha256(json.dumps(portable(value), sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def _freeform_geometry(geometry):
    """Keep physical dimensions only; legacy layer and target fields do not apply."""
    from scenes.workrooms.living_room.tower_geometry import DEFAULTS, SCHEMA
    accepted = set(DEFAULTS) | {'schema', 'block_size_m'}
    if set(geometry)-accepted:
        raise ValueError('Unknown freeform physical geometry fields')
    if geometry.get('schema', SCHEMA) not in (SCHEMA, 'living_room_freeform_blocks.v1'):
        raise ValueError('Unsupported freeform geometry schema')
    if 'block_size_m' not in geometry:
        raise ValueError('block_size_m is required')
    size = vector(geometry['block_size_m'], 3, 'block_size_m')
    if min(size) <= 0:
        raise ValueError('Block dimensions must be positive')
    top = number(geometry.get('table_top_z_m', DEFAULTS['table_top_z_m']), 'table_top_z_m')
    bounds = geometry.get('table_bounds_xy_m', DEFAULTS['table_bounds_xy_m'])
    if not isinstance(bounds, (list, tuple)) or len(bounds) != 2:
        raise ValueError('Table bounds must contain low/high XY corners')
    bounds = [list(vector(value, 2, 'table bounds')) for value in bounds]
    if top != DEFAULTS['table_top_z_m'] or bounds != DEFAULTS['table_bounds_xy_m']:
        raise ValueError('Freeform tasks must preserve the pinned parent table dimensions')
    return {'schema': 'living_room_freeform_blocks.v1',
            'block_size_m': list(size), 'table_top_z_m': top, 'table_bounds_xy_m': bounds}


def rules_for_geometry(geometry, *, stable_duration_s=1., tolerances=None,
                       floor_drop_is_terminal=None, inventory_count=None, scoring_mode=None):
    from scenes.workrooms.living_room.tower_geometry import validate_geometry
    if not isinstance(geometry, dict):
        raise ValueError('Tower geometry must be a mapping with explicit block_size_m')
    if 'layer_gap_m' in geometry:
        raise ValueError('Use in_layer_gap_m for horizontal gaps; vertical layer gaps are forbidden')
    if scoring_mode not in (None, 'freeform_height'):
        raise ValueError('Unknown tower scoring mode')
    freeform = scoring_mode == 'freeform_height'
    g = _freeform_geometry(geometry) if freeform else validate_geometry(geometry)
    _, width, height = g['block_size_m']
    dwell = number(stable_duration_s, 'stable_duration_s')
    if dwell < 1.:
        raise ValueError('The full released tower must remain stable for at least one second')
    if floor_drop_is_terminal is None:
        floor_drop_is_terminal = not freeform
    if type(floor_drop_is_terminal) is not bool:
        raise ValueError('floor_drop_is_terminal must be boolean')
    if freeform and floor_drop_is_terminal:
        raise ValueError('Freeform height scoring retains best score and allows recovery after drops')
    t = {'position_xy_m': min(.006, width*.12), 'position_z_m': min(.003, height*.12),
         'yaw_rad': math.radians(5.), 'tilt_rad': math.radians(3.),
         'support_gap_m': min(.002, height*.08), 'penetration_m': min(.001, height*.04),
         'contact_point_m': min(.001, width*.03, height*.1), 'minimum_support_force_n': 1.e-6,
         'linear_speed_m_s': .01, 'angular_speed_rad_s': .05,
         'stable_position_range_m': min(.001, width*.04), 'stable_angle_range_rad': .02,
         'sample_period_s': .02, 'initial_position_m': min(1.e-6, width*1.e-4, height*1.e-4),
         'initial_angle_rad': 1.e-6}
    if tolerances is not None:
        if not isinstance(tolerances, dict) or set(tolerances)-set(t):
            raise ValueError('Unknown tower tolerance')
        t.update(tolerances)
    t = {name: number(value, name) for name, value in t.items()}
    if any(value <= 0 for value in t.values()):
        raise ValueError('All tolerances must be positive')
    if (t['position_xy_m'] >= width/4 or t['position_z_m'] >= height/4
            or t['support_gap_m'] >= height/4 or t['penetration_m'] >= height/4
            or t['contact_point_m'] >= min(width, height)/4
            or t['yaw_rad'] > math.radians(15.) or t['tilt_rad'] > math.radians(10.)
            or t['stable_position_range_m'] > t['position_xy_m']
            or t['stable_angle_range_rad'] > min(t['yaw_rad'], t['tilt_rad'])
            or t['initial_position_m'] > min(1.e-4, t['position_xy_m']/10, t['position_z_m']/10)
            or t['initial_angle_rad'] > min(1.e-4, t['yaw_rad']/10, t['tilt_rad']/10)
            or t['sample_period_s'] > dwell/2):
        raise ValueError('Tolerances must preserve distinct slots, support surfaces and a sampled dwell')
    if freeform:
        if type(inventory_count) is not int or inventory_count <= 0:
            raise ValueError('Freeform inventory_count must be a positive integer')
        return {'schema': 'living_tower_building_freeform_height_rules_v1', 'geometry': deepcopy(g),
            'goal': {'scoring_mode': 'freeform_height', 'stable_duration_s': dwell,
                'reset_credit': False, 'score_units': 'm', 'score_aggregation': 'maximum_stable_height_reached',
                'height_reference': 'Highest full oriented block extent minus the original tabletop height',
                'shape': 'Any arrangement or orientation using any number of supplied blocks; no target slots or required layer count'},
            'inventory_count': inventory_count, 'tolerances': t, 'floor_drop_is_terminal': False,
            'support': 'Only released, stable connected wooden assemblies with complete contact-backed support paths to the original tabletop count. Tray-supported, robot-supported, externally supported and off-table assemblies do not count. Unconnected supplies do not constrain a scored assembly.',
            'recovery': 'Dropped or collapsed blocks may be recovered by robot actions; the best previously established stable height is retained. No object pose writes, welds or automatic rebuilding.',
            'budget_exhaustion': 'Close with terminal incomplete and retain the best stable height score; there is no binary success-height threshold',
            'observation_schema': OBSERVATION_SCHEMA,
            'qualification': 'CPU reference contract only; native reset and manipulation require separate evidence'}
    if inventory_count is not None and inventory_count != g['layers']*3:
        raise ValueError('Extra inventory requires freeform_height scoring')
    result = {'schema': RULES_VERSION, 'geometry': deepcopy(g),
        'goal': {'layers': g['layers'], 'blocks_per_layer': 3,
            'block_count': g['layers']*3, 'alternating_yaw_rad': math.pi/2,
            'assignment': 'Any bijection of physical blocks onto all distinct legal slots',
            'stable_duration_s': dwell, 'reset_credit': False},
        'tolerances': t, 'floor_drop_is_terminal': floor_drop_is_terminal,
        'support': 'Every bottom block has continuous table support; each higher block has a contact-backed chain through the immediately lower layer to the table. Robot and external contacts invalidate the success dwell.',
        'recovery': 'Misplacement, regrasp and collapse may be corrected by real robot actions; all-tower dwell restarts. No object pose writes, welds or automatic rebuilding.',
        'budget_exhaustion': 'Explicit exhaustion or reaching a configured time/action limit takes precedence over same-sample success and returns terminal incomplete; never automatic success or competitive loss',
        'observation_schema': OBSERVATION_SCHEMA,
        'qualification': 'CPU reference contract only; trusted normalization and native task integration are not implemented or qualified'}
    return result
