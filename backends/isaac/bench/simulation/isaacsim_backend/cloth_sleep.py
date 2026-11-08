"""Explicit surface settling policy and read-only USD receipts; no simulator startup.

The pinned PhysX surface schema uses velocity thresholds, unlike rigid-body
mass-normalized energy thresholds. The optional settling threshold only changes
when the native extra damping engages; it does not write nodal state or put an
actor to sleep. Sleep threshold and settling damping retain schema defaults.
"""
import math


SETTLING_THRESHOLD_OPTION = 'cloth_settling_threshold_m_s'
SLEEP_DEFAULTS = {'settlingDamping': 10., 'settlingThreshold': .10, 'sleepThreshold': .05}


def surface_settling_properties(options):
    """Return only explicitly requested properties, preserving old stage layers."""
    if SETTLING_THRESHOLD_OPTION not in options:
        return {}
    value = options[SETTLING_THRESHOLD_OPTION]
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or value <= SLEEP_DEFAULTS['sleepThreshold']):
        raise ValueError(SETTLING_THRESHOLD_OPTION+' must be finite and exceed the unchanged 0.05 m/s sleep threshold')
    return {'physxDeformableBody:settlingThreshold': float(value)}


def author_surface_settling(prim, options):
    """Author the opt-in threshold without creating unsupported custom attributes."""
    for name, value in surface_settling_properties(options).items():
        attribute = prim.GetAttribute(name)
        if not attribute or not attribute.Set(value):
            raise RuntimeError('Required native surface settling attribute unavailable: '+name)


def read_surface_sleep_settings(prim):
    """Read applied USD values and opinions; this is not native isSleeping()."""
    result = {}
    for suffix in SLEEP_DEFAULTS:
        attribute = prim.GetAttribute('physxDeformableBody:'+suffix)
        result[suffix] = {
            'value': attribute.Get() if attribute else None,
            'authored': bool(attribute and attribute.HasAuthoredValueOpinion()),
            'unit': '1/s' if suffix == 'settlingDamping' else 'stage_distance/s'}
    return result
