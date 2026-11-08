"""Explicitly defer a fresh capture plan until released robot parking ends.

Parking still uses the original robot-only commands. No cloth observation is
invented at the later time, and all capture planners keep their original guards.
"""
from __future__ import annotations


BEFORE_PARK = 'before_released_park_v1'
AFTER_PARK = 'after_released_park_v1'


def phase_prepare_timing(options, index):
    selected = options.get('phases', {}).get(str(index), {}).get('prepare_timing', BEFORE_PARK)
    if selected not in (BEFORE_PARK, AFTER_PARK):
        raise ValueError('Unknown explicit phase prepare_timing')
    if selected == AFTER_PARK and index <= 0:
        raise ValueError('Deferred capture preparation requires a later released phase')
    return selected


def prepare_at_stage(timing, stage, prepare, *, previous_assists=(), current=(), capture_sides=()):
    """Run exactly the selected stage, before constructing new capture assists."""
    if timing not in (BEFORE_PARK, AFTER_PARK) or stage not in (BEFORE_PARK, AFTER_PARK):
        raise ValueError('Known preparation timing and execution stage required')
    if timing == AFTER_PARK and (current or capture_sides or any(a.active for a in previous_assists)):
        raise RuntimeError('Deferred preparation requires empty current captures and released prior assists')
    if timing != stage:
        return None
    return prepare()
