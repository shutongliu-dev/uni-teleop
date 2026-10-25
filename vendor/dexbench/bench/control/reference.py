"""Discrete position-reference governor, independent of task and physics state.

Velocity denotes the difference between *commanded* joint positions / dt. This
does not certify measured joint velocity, collision avoidance or grasp stability.
"""
from __future__ import annotations

import numpy as np


class ReferenceInfeasible(ValueError):
    def __init__(self, message, joint_index=None):
        super().__init__(message)
        self.joint_index = joint_index


def _stoppable_speed(distance, acceleration, dt):
    """Largest next speed whose discrete, maximum-deceleration stop fits.

    For next speed w and n=ceil(w/(a*dt)), the distance through the stop is
    dt*(n*w - a*dt*n*(n-1)/2). Inverting that piecewise linear function avoids
    the continuous w*w/(2*a) formula, which is unsafe for finite control steps.
    """
    distance = np.maximum(distance, 0.)
    n = np.maximum(1., np.ceil((np.sqrt(1. + 8. * distance / (acceleration * dt**2)) - 1.) / 2.))
    speed = (distance / dt + acceleration * dt * n * (n - 1.) / 2.) / n
    return np.nextafter(speed, np.zeros_like(speed))


def bounded_reference(position, velocity, desired, lower, upper, max_velocity,
                      max_acceleration, dt, *, steps_remaining=None):
    """Return a viable next (q, v), or reject without modifying inputs.

    All constraints apply together: position, command speed/acceleration,
    discrete braking distance to both limits, and (optionally) a zero-speed
    final command before a fixed deadline. No physically infeasible state is
    repaired by clipping a command across an acceleration limit.
    """
    values = [np.asarray(x, dtype=float) for x in
              (position, velocity, desired, lower, upper, max_velocity, max_acceleration)]
    q, old_v, desired, lower, upper, vmax, amax = values
    if (not np.isfinite(dt) or dt <= 0 or not all(x.shape == q.shape for x in values)
            or not all(np.isfinite(x).all() for x in values)):
        raise ReferenceInfeasible('Bounded target trajectory is nonfinite')
    if (lower >= upper).any() or (vmax <= 0).any() or (amax <= 0).any():
        raise ReferenceInfeasible('Invalid reference bounds')
    rounding = 8. * np.finfo(float).eps * np.maximum(1., np.maximum(abs(lower), abs(upper)))
    bad = (q < lower - rounding) | (q > upper + rounding) | (abs(old_v) > vmax + rounding / dt)
    if bad.any():
        raise ReferenceInfeasible('Bounded target trajectory would cross a joint limit', int(np.flatnonzero(bad)[0]))
    dv = amax * dt
    lo = np.maximum.reduce([-vmax, old_v - dv, -_stoppable_speed(q - lower, amax, dt)])
    hi = np.minimum.reduce([vmax, old_v + dv, _stoppable_speed(upper - q, amax, dt)])
    if steps_remaining is not None:
        if isinstance(steps_remaining, bool) or int(steps_remaining) != steps_remaining or steps_remaining < 1:
            raise ReferenceInfeasible('No control steps remain')
        terminal_speed = dv * (steps_remaining - 1)
        lo, hi = np.maximum(lo, -terminal_speed), np.minimum(hi, terminal_speed)
    # An intersection inverted by roundoff alone is resolved toward zero speed.
    # Always taking the acceleration endpoint instead accumulates roundoff on
    # every braking step and can eventually cross the position boundary.
    bad = lo > hi + rounding / dt
    if bad.any():
        raise ReferenceInfeasible('Bounded target trajectory would cross a joint limit', int(np.flatnonzero(bad)[0]))
    near_zero = np.where(abs(lo) < abs(hi), lo, hi)
    next_v = np.where(lo > hi, near_zero, np.clip((desired - q) / dt, lo, hi))
    command = q + dt * next_v
    bad = (command < lower - rounding) | (command > upper + rounding)
    if bad.any():
        raise ReferenceInfeasible('Bounded target trajectory would cross a joint limit', int(np.flatnonzero(bad)[0]))
    command = np.clip(command, lower, upper)  # machine roundoff only, not a braking fix
    next_v = (command - q) / dt
    bad = (abs(next_v) > vmax + rounding / dt) | (abs(next_v - old_v) > dv + rounding / dt)
    if bad.any():
        raise ReferenceInfeasible('Bounded target trajectory would cross a joint limit', int(np.flatnonzero(bad)[0]))
    return command, next_v


def stop_reference(position, velocity, lower, upper, max_velocity, max_acceleration, dt):
    """Preview the exact bounded stop without issuing a robot command."""
    q, v = np.asarray(position, float).copy(), np.asarray(velocity, float).copy()
    if not np.isfinite(v).all():
        raise ReferenceInfeasible('Bounded target trajectory is nonfinite')
    steps = int(np.ceil(np.max(abs(v) / (np.asarray(max_acceleration) * dt)) - 1e-10))
    steps = max(steps, 0)
    if steps == 0:
        bounded_reference(q, v, q, lower, upper, max_velocity, max_acceleration,
                          dt, steps_remaining=1)
    for remaining in range(steps, 0, -1):
        q, v = bounded_reference(q, v, q, lower, upper, max_velocity, max_acceleration,
                                 dt, steps_remaining=remaining)
    return q, steps
