"""Explicit native surface material options, without starting a simulator."""
import math


DAMPING_OPTIONS = {
    'cloth_elasticity_damping': 'physxDeformableMaterial:elasticityDamping',
    'cloth_bend_damping': 'physxDeformableMaterial:bendDamping',
}


def surface_damping_properties(options):
    """Return material damping in 1/s, not a fraction of critical damping.

    PhysX deformable schema specifies nonnegative coefficients. Keep the
    historical .005 defaults unless the frozen scene explicitly overrides them.
    """
    result = {}
    for key, attribute in DAMPING_OPTIONS.items():
        value = options.get(key, .005)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(key+' must be a finite nonnegative coefficient in 1/s')
        value = float(value)
        if not math.isfinite(value) or value < 0:
            raise ValueError(key+' must be a finite nonnegative coefficient in 1/s')
        result[attribute] = value
    return result
