"""Validate captured-manipulation hand option names before native startup.

Keep provenance beside the options, never inside a hand's command settings.
Numeric bounds remain owned by the existing approach and pinch planners.
"""
from __future__ import annotations

from scripts.validation.laundry_pinch_slot import validate_slot_offset
from scripts.validation.laundry_closure_timing import validate_timing_profile
from scripts.validation.laundry_closure_feed import validate_tangent_profile


HAND_OPTION_KEYS = frozenset({
    'yaw_deg', 'pitch_deg', 'bottom_z_m', 'open_distance_m', 'close_distance_m',
    'closure_lift_m', 'closure_lift_start_fraction', 'capture_rise_m',
    'hover_offset_m', 'hover_arm_joint_positions_rad',
    'pinch_inward_offset_m', 'closure_inward_travel_m',
    'closure_timing_profile',
})


def validate_hand_options(options, *, towel=False):
    """Reject unused hand keys in every phase, including later/unused hands.

    This does not normalize, mutate or interpret joint commands. Towel-relative
    yaw is permitted only by the adapter that actually implements it.
    """
    if not isinstance(options, dict):
        raise ValueError('Capture options must be an object')
    phases = options.get('phases', {})
    if not isinstance(phases, dict):
        raise ValueError('Capture options.phases must be an object')
    allowed = HAND_OPTION_KEYS | ({'yaw_offset_deg', 'corner_tangent_sign'}
                                  if towel else {'pinch_slot_offset_m', 'corner_relative_bottom_m',
                                                 'pinch_tangent_profile'})
    for index, phase in phases.items():
        if (not isinstance(index, str) or not index.isascii() or not index.isdecimal()
                or str(int(index)) != index):
            raise ValueError('Capture phase keys must be nonnegative integer strings')
        location = f'options.phases.{index}'
        if not isinstance(phase, dict):
            raise ValueError(f'{location} must be an object')
        hands = phase.get('hands', {})
        if not isinstance(hands, dict):
            raise ValueError(f'{location}.hands must be an object')
        for side, setting in hands.items():
            if side not in ('left', 'right'):
                raise ValueError(f'Unknown hand {side!r} at {location}.hands; use left or right')
            path = f'{location}.hands.{side}'
            if not isinstance(setting, dict):
                raise ValueError(f'{path} must be a hand-settings object')
            unknown = set(setting) - allowed
            if unknown:
                names = ', '.join(sorted(map(repr, unknown)))
                raise ValueError(f'Unknown hand options at {path}: {names}. '
                                 'Pass the hand settings themselves, without schema/evidence/setting wrappers.')
            if 'corner_tangent_sign' in setting:
                sign = setting['corner_tangent_sign']
                if type(sign) is not int or sign not in (-1, 1):
                    raise ValueError(f'{path}.corner_tangent_sign must be integer -1 or +1')
                if {'yaw_deg', 'yaw_offset_deg'} & set(setting):
                    raise ValueError(f'{path}.corner_tangent_sign is exclusive with yaw_deg/yaw_offset_deg')
            if 'pinch_slot_offset_m' in setting:
                validate_slot_offset(setting['pinch_slot_offset_m'])
            if 'closure_timing_profile' in setting:
                validate_timing_profile(setting['closure_timing_profile'])
            if 'pinch_tangent_profile' in setting:
                validate_tangent_profile(setting['pinch_tangent_profile'])
                if 'pinch_inward_offset_m' not in setting:
                    raise ValueError('A local garment tangent requires the observed edge entry planner')
