"""Explicit measured closure amounts for original narrow-opening fingers."""
from __future__ import annotations

import math
from numbers import Real


PROFILES = ('strict_v1', 'narrow_opening_v1')


def validate_closure_measurement(profile, planned_open_tip_gap_m=None):
    if profile not in PROFILES:
        raise ValueError('Unknown explicit closure measurement profile')
    if profile == 'narrow_opening_v1' and (
            isinstance(planned_open_tip_gap_m, bool)
            or not isinstance(planned_open_tip_gap_m, Real)
            or not math.isfinite(planned_open_tip_gap_m)
            or not .020 <= planned_open_tip_gap_m <= .030):
        raise ValueError('Narrow closure requires an explicit planned opening of20–30mm')


def measured_closure_requirement(profile, measured_open_tip_gap_m, planned_open_tip_gap_m=None):
    validate_closure_measurement(profile, planned_open_tip_gap_m)
    opening = float(measured_open_tip_gap_m)
    minimum = .005 if profile == 'narrow_opening_v1' else .008
    return {'profile': profile, 'planned_open_tip_gap_m': planned_open_tip_gap_m,
            'measured_open_tip_gap_m': opening,
            'open_baseline_valid': math.isfinite(opening) and (
                profile != 'narrow_opening_v1' or .020 <= opening <= .030),
            'minimum_tip_closing_m': minimum, 'minimum_tip_closing_fraction': .20,
            'minimum_required_closure_m': max(minimum, .20*opening)}
