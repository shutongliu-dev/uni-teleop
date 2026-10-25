"""Bound privileged robot target timing by the unchanged robot velocity limits."""
from copy import deepcopy
import math


CONTROL_DT_S = .02
QUINTIC_PEAK_RATE = 1.875
VELOCITY_FRACTION = .5
SETTLE_DURATION_S = .6
ENDPOINT_TOLERANCE_RAD = .02


def velocity_limits(scene):
    from scenes.workrooms.mechanical_assembly.robot_layout import geometry_model
    model = geometry_model(scene['robot'])
    return {joint.name: joint.velocity for joint in model.joints}


def time_phases(phases, scene, initial):
    limits = velocity_limits(scene)
    previous = initial['joint_positions_rad']
    result = []
    for phase in phases:
        target = phase['joint_positions_rad']
        if set(target) != set(previous) or set(limits) != set(previous):
            raise ValueError('Phase timing requires every robot joint and its velocity limit')
        if any(not math.isfinite(value) or value <= 0 for value in limits.values()):
            raise ValueError('Phase timing requires finite positive joint velocity limits')
        durations = {name: QUINTIC_PEAK_RATE*abs(target[name]-previous[name]) /
                     (limit*VELOCITY_FRACTION) for name, limit in limits.items()}
        limiting_joint = max(durations, key=durations.get)
        motion = math.ceil(max(.24, durations[limiting_joint])/CONTROL_DT_S)*CONTROL_DT_S
        total = round(motion+SETTLE_DURATION_S, 8)
        if total > 10.:
            raise ValueError('Velocity-limited diagnostic phase exceeds the ten-second bound')
        updated = deepcopy(phase)
        updated.update(duration_s=total, timing={
            'schema': 'mechanical_robot_phase_timing.v1',
            'profile': 'quintic_smoothstep', 'peak_rate_multiplier': QUINTIC_PEAK_RATE,
            'joint_velocity_limits_rad_s': dict(limits),
            'velocity_fraction': VELOCITY_FRACTION,
            'motion_duration_s': motion, 'settle_duration_s': SETTLE_DURATION_S,
            'endpoint_tolerance_rad': ENDPOINT_TOLERANCE_RAD,
            'stop_on_endpoint_failure': True, 'limiting_joint': limiting_joint,
            'limits_source': 'Selected robot URDF; no actuator gains or limits modified.',
        })
        result.append(updated)
        previous = target
    return result


def validate_timing(phase, previous, *, expected_limits=None):
    timing = phase.get('timing')
    if timing is None:
        return None
    if (not isinstance(timing, dict) or timing.get('schema') != 'mechanical_robot_phase_timing.v1'
            or timing.get('profile') != 'quintic_smoothstep'
            or timing.get('peak_rate_multiplier') != QUINTIC_PEAK_RATE
            or timing.get('stop_on_endpoint_failure') is not True):
        raise ValueError('Invalid robot phase timing contract')
    limits = timing.get('joint_velocity_limits_rad_s')
    if (not isinstance(limits, dict) or set(limits) != set(previous)
            or any(type(value) not in (float, int) or not math.isfinite(value) or value <= 0
                   for value in limits.values())
            or (expected_limits is not None and limits != expected_limits)):
        raise ValueError('Robot phase timing velocity limits differ from the selected robot')
    motion, settle = timing.get('motion_duration_s'), timing.get('settle_duration_s')
    fraction, tolerance = timing.get('velocity_fraction'), timing.get('endpoint_tolerance_rad')
    if (any(type(value) not in (float, int) or not math.isfinite(value)
            or value < CONTROL_DT_S or abs(value/CONTROL_DT_S-round(value/CONTROL_DT_S)) > 1.e-7
            for value in (motion, settle))
            or type(fraction) not in (float, int) or not 0 < fraction <= 1
            or type(tolerance) not in (float, int) or not 0 < tolerance <= ENDPOINT_TOLERANCE_RAD
            or abs(phase['duration_s']-motion-settle) > 1.e-7):
        raise ValueError('Robot phase motion and settle durations must match the frozen timing budget')
    for name, limit in limits.items():
        peak = QUINTIC_PEAK_RATE*abs(phase['joint_positions_rad'][name]-previous[name])/motion
        if peak > limit*fraction+1.e-10:
            raise ValueError('Robot phase quintic peak exceeds its velocity limit: '+name)
    return deepcopy(timing)
