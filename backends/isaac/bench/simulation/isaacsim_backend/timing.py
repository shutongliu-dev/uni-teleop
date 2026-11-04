"""Explicit practice timing overrides; benchmark cadence defaults to 20 ms."""
import math


def control_timing(scene):
    options = scene.get('isaacsim', {})
    dt = float(options.get('physics_dt_s', scene['physics']['dt_s']))
    control = float(options.get('control_dt_s', .02))
    if not all(math.isfinite(v) and v > 0 for v in (dt, control)):
        raise ValueError('Physics and control periods must be finite and positive')
    if control != .02 and scene.get('unscored_practice') is not True:
        raise ValueError('Custom control timing requires unscored practice')
    if control > .1 or dt > control or abs(control/dt-round(control/dt)) > 1e-8:
        raise ValueError('Physics period must divide control period; control period must be <= 100 ms')
    return dt, control
