"""Public, dimension-parameterized rules for task4 tower extraction.

These are reference adjudication rules, not a native task release. Geometry
must be explicitly supplied; no default wooden block dimensions are selected.
"""
from __future__ import annotations

from copy import deepcopy
import math

from scenes.workrooms.living_room.tower_geometry import validate_geometry

SIDES = ('front', 'rear')
RULES_SCHEMA = 'living_tower_extraction_rules.v1'
OBSERVATION_SCHEMA = 'living_tower_extraction_normalized_observation.v1'
DEFAULT_THRESHOLDS = {
    'stable_duration_s': 1., 'max_observation_gap_s': .05,
    'reset_timeout_s': 10., 'slot_position_tolerance_h': .35,
    'slot_rotation_tolerance_rad': .15, 'stable_linear_speed_h_s': .05,
    'stable_angular_speed_rad_s': .08, 'release_clearance_h': .10,
    'collapse_height_drop_h': .80, 'collapse_tilt_rad': math.pi / 4,
    'support_normal_z_min': .50, 'collection_height_tolerance_h': .05,
    'contact_geometry_tolerance_h': .05,
}


def starting_side(seed):
    """Consecutive integer seeds alternate first mover without side bias."""
    if type(seed) is not int or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    return SIDES[seed % 2]


def public_rules(geometry, *, max_turns_per_player=None, protected_bottom_layers=1,
                 thresholds=None, collection_surface='table'):
    g = validate_geometry(geometry)
    if collection_surface not in ('table', 'tray_floor'):
        raise ValueError('Unknown collection support surface')
    if type(protected_bottom_layers) is not int or not 1 <= protected_bottom_layers < g['layers'] - 1:
        raise ValueError('Protect at least the bottom layer and leave an intermediate layer')
    eligible = (g['layers'] - protected_bottom_layers - 1) * 3
    if max_turns_per_player is None:
        max_turns_per_player = eligible // 2
    if type(max_turns_per_player) is not int or not 1 <= max_turns_per_player <= eligible // 2:
        raise ValueError('Equal-turn limit exceeds the initially extractable block inventory')
    t = deepcopy(DEFAULT_THRESHOLDS)
    if thresholds is not None:
        if not isinstance(thresholds, dict) or set(thresholds) - set(t):
            raise ValueError('Unknown physical threshold')
        t.update(thresholds)
    for name, value in t.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError('Threshold must be a positive finite number: ' + name)
        t[name] = float(value)
    if t['stable_duration_s'] < 1.:
        raise ValueError('Handoff requires at least one continuous second')
    if t['max_observation_gap_s'] > .05 or t['max_observation_gap_s'] >= t['stable_duration_s']:
        raise ValueError('Every normalized observation tick is required; intervals may not exceed 50 ms')
    if not 0 < t['support_normal_z_min'] <= 1 or not 0 < t['collapse_tilt_rad'] < math.pi / 2:
        raise ValueError('Invalid support normal or collapse angle threshold')
    if t['slot_position_tolerance_h'] >= t['collapse_height_drop_h']:
        raise ValueError('Collapse drop must exceed the reset/complete-layer position tolerance')
    if (t['slot_position_tolerance_h'] > .5 or t['slot_rotation_tolerance_rad'] > .25
            or t['contact_geometry_tolerance_h'] > .1 or t['collection_height_tolerance_h'] > .1):
        raise ValueError('Geometric tolerances cannot erase layer, contact or placement distinctions')
    if (t['stable_linear_speed_h_s'] * t['stable_duration_s'] > .25
            or t['stable_angular_speed_rad_s'] * t['stable_duration_s'] > .25):
        raise ValueError('Stable interval motion bounds must remain below one quarter block height/radian')
    if t['reset_timeout_s'] < t['stable_duration_s']:
        raise ValueError('Reset timeout cannot be shorter than the required stable interval')
    result = {'schema': RULES_SCHEMA, 'task_id': 'tower_extraction', 'task_number': 4,
        'mode': 'two_player_adversarial', 'sides': list(SIDES),
        'normalization': {'length_unit': 'block_height', 'length_unit_m': g['block_size_m'][2],
                          'origin_m': [*g['center_xy_m'], g['table_top_z_m']]},
        'protected_bottom_layers': protected_bottom_layers,
        'protected_top': 'Highest currently complete original layer and every layer above it',
        'first_mover': 'front for even seed, rear for odd seed',
        'max_turns_per_player': max_turns_per_player, 'thresholds': t,
        'turn': 'Select one legal block; lock selection at its first physical hand contact; push, pull and regrasp are permitted',
        'collection': 'Put the extracted block in one free own-side single-layer collection slot; never return it to the tower top',
        'handoff': 'Own collection support, both hands clear, and all blocks/tower stable for at least one continuous second',
        'loss': 'A nonselected, uncollected tower block drops >= collapse_height_drop_h or tilts >= collapse_tilt_rad before handoff; current player loses',
        'draw': 'Both players complete the configured equal number of turns, or legal targets are exhausted with a stable tower',
        'turn_limit_source': 'Default is half the initial legal inventory; smaller explicit limits are diagnostic/configuration choices',
        'causality': 'After handoff, collapse before current-player physical tower contact aborts without a winner; command execution alone proves no causality',
        'invalid_commands': 'Rejected without execution or changing the active player',
        'abort_without_winner': ['invalid reset', 'missing/inconsistent evidence', 'unannounced reset',
                                 'physical noncurrent-player interference', 'undeclared direct block contact',
                                 'contact with a target that became protected before first touch'],
        'incomplete_without_winner': ['timeout', 'budget'],
        'qualification': {'native_adapter': None, 'native_release': None, 'native_accepted': False,
                          'scope': 'Reference CPU rules and provisional simulation thresholds only'}}
    if collection_surface == 'tray_floor':
        result['collection'] = ('Put the extracted block in one free single-layer slot inside your own tray, '
                                'supported by its physical floor; never return it to the tower top')
        result['handoff'] = ('Own tray-floor support, both hands clear, and all blocks/tower stable '
                             'for at least one continuous second; tray walls and the room floor do not count as support')
    return result
