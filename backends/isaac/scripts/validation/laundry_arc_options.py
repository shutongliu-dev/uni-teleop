"""Explicit per-execution-phase arc heights; no change to captured material."""
from __future__ import annotations

import math


def phase_peak_heights(options, count, default):
    """Validate every declared override before startup and resolve used phases.

    Unspecified phases retain the CLI height exactly. In particular, a short
    sleeve need not inherit the larger arc used for the full garment body.
    """
    if type(count) is not int or count < 1:
        raise ValueError('A positive execution-phase count is required')
    if not isinstance(options, dict) or not isinstance(options.get('phases', {}), dict):
        raise ValueError('Arc options and phases must be objects')

    def check(value):
        if (type(value) not in (int, float) or not math.isfinite(value)
                or not .04 <= value <= .15):
            raise ValueError('peak_height_m must be a finite number from .04 to .15 m')
        return value

    check(default)
    phases = options.get('phases', {})
    for index, phase in phases.items():
        if (not isinstance(index, str) or not index.isascii() or not index.isdecimal()
                or str(int(index)) != index or not isinstance(phase, dict)):
            raise ValueError('Arc phases must use nonnegative integer-string object entries')
        if 'peak_height_m' in phase:
            check(phase['peak_height_m'])
    return [phases.get(str(i), {}).get('peak_height_m', default) for i in range(count)]


def arc_height_report(options, heights):
    """Omit the field entirely for legacy options, preserving their reports."""
    phases = options.get('phases', {})
    if not any('peak_height_m' in phases.get(str(i), {}) for i in range(len(heights))):
        return {}
    return {'phase_arc_settings': [
        {'phase_index': i, 'peak_height_m': height,
         'source': 'phase_override' if 'peak_height_m' in phases.get(str(i), {}) else 'cli'}
        for i, height in enumerate(heights)]}
