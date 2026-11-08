"""Explicit finger timing within the original two-second closure action.

Only original position targets are interpolated. Wrist feed, other fingers,
contact observation, actual state and the independent capture gate are unchanged.
"""
from __future__ import annotations

import math

from scripts.validation.laundry_contact_pinch import interpolate_posture
from scripts.validation.laundry_engineering_motion import minimum_jerk


PROFILES = ('synchronous_v1', 'thumb_early_index_late_v1')


def validate_timing_profile(profile):
    if profile not in PROFILES:
        raise ValueError('Unknown explicit finger closure timing profile')


def bind_closure_timing(plan, setting):
    """Preserve old plans exactly when the optional hand setting is absent."""
    if 'closure_timing_profile' in setting:
        profile = setting['closure_timing_profile']
        validate_timing_profile(profile)
        plan['closure_timing_profile'] = profile
    return plan


def closure_fractions(amount, profile='synchronous_v1'):
    validate_timing_profile(profile)
    if isinstance(amount, bool) or not isinstance(amount, (int, float)) or not math.isfinite(amount) or not 0. <= amount <= 1.:
        raise ValueError('Original closure fraction must lie in [0,1]')
    if profile == 'synchronous_v1' or amount in (0., 1.):
        return {'thumb': amount, 'index': amount}
    # The runner and its existing 101-point preflight pass the original wrist
    # minimum-jerk fraction. Recover that same clock; do not retime the wrist.
    lo, hi = 0., 1.
    for _ in range(60):
        mid = (lo+hi)/2
        if minimum_jerk(mid, 0., 1.) < amount:
            lo = mid
        else:
            hi = mid
    elapsed = lo+hi  # Two-second action: 2*((lo+hi)/2).
    return {'thumb': minimum_jerk(elapsed, 0., 1.4),
            'index': minimum_jerk(elapsed, .6, 2.)}


def closure_hand_posture(plan, amount):
    """Use the original per-finger posture path at explicitly shifted times."""
    profile = plan.get('closure_timing_profile', 'synchronous_v1')
    validate_timing_profile(profile)
    result = interpolate_posture(plan['postures'], amount)
    if profile == 'synchronous_v1':
        return result
    fractions = closure_fractions(amount, profile)
    for finger in ('thumb', 'index'):
        timed = interpolate_posture(plan['postures'], fractions[finger])
        for name in result:
            if name.startswith(('left_'+finger+'_', 'right_'+finger+'_')):
                result[name] = timed[name]
    return result
