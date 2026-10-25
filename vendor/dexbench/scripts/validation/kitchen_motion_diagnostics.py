"""Read-only native motion measurements; never changes or scores the simulation.

Keep native velocity peaks alongside pose-derived peaks and whole-window ranges.
A small 20 ms displacement does not erase genuine substep oscillation or change
the task's stability thresholds. Frames are copied before quaternion normalization.
"""
from copy import deepcopy
import json
import math
from pathlib import Path

import numpy as np


def _finite(value):
    if isinstance(value, bool):
        raise ValueError('Boolean is not a native numeric measurement')
    value = float(value)
    if not math.isfinite(value):
        raise ValueError('Nonfinite native measurement')
    return value


def _vector(value, size):
    if len(value) != size:
        raise ValueError('Native vector dimension differs')
    return [_finite(x) for x in value]


def _angle(a, b):
    """Shortest rotation, robust to quaternion sign and tiny angular changes."""
    dot = np.sum(a*b, axis=-1, keepdims=True)
    b = b*np.where(dot < 0., -1., 1.)
    return 4.*np.arctan2(np.linalg.norm(a-b, axis=-1),
                        np.linalg.norm(a+b, axis=-1))


def _ranges(positions, quaternions):
    position = rotation = 0.
    # Bounded temporary memory, with exact all-pairs ranges inside the window.
    for start in range(0, len(positions), 128):
        block = positions[start:start+128, None, :]-positions[None, :, :]
        position = max(position, float(np.linalg.norm(block, axis=-1).max()))
        rotation = max(rotation, float(_angle(quaternions[start:start+128, None, :],
                                              quaternions[None, :, :]).max()))
    return position, rotation


class NativeMotionDiagnostics:
    """Attach ``observe(frame)`` beside an existing native observer callback.

    ``limits`` uses the four public kitchen speed/range threshold names.
    Invalid data is recorded as a diagnostic error, without raising into or
    replacing the real observer. The caller remains responsible for invoking it.
    """
    def __init__(self, limits, physics_dt):
        self.limits = deepcopy(dict(limits))
        self.physics_dt = _finite(physics_dt)
        if self.physics_dt <= 0.:
            raise ValueError('Positive physics period required')
        for key in ('maximum_linear_speed_m_s', 'maximum_angular_speed_rad_s',
                    'maximum_stable_position_range_m', 'maximum_stable_orientation_range_rad'):
            if _finite(self.limits[key]) <= 0.:
                raise ValueError('Positive declared diagnostic limits required')
        self.frames = []
        self.errors = []

    def observe(self, frame):
        """Copy one native frame; return false for invalid diagnostic data."""
        try:
            step = frame['native_step']
            if type(step) is not int or step < 0:
                raise ValueError('Nonnegative integer native step required')
            row = {'native_step': step, 'time_s': _finite(frame['time_s']),
                   'native_time_s': _finite(frame['native_time_s']),
                   'physics_dt_s': _finite(frame['physics_dt_s']), 'objects': {}}
            if abs(row['physics_dt_s']-self.physics_dt) > 1.e-12:
                raise ValueError('Native physics period changed')
            for name, obj in frame['objects'].items():
                q = _vector(obj['quaternion_wxyz'], 4)
                length = math.sqrt(sum(x*x for x in q))
                if length < 1.e-12:
                    raise ValueError('Zero native quaternion')
                row['objects'][name] = {
                    'position_m': _vector(obj['position_m'], 3),
                    'quaternion_wxyz': [x/length for x in q],
                    'quaternion_norm_before_normalization': length,
                    'linear_velocity_m_s': _vector(obj['linear_velocity_m_s'], 3),
                    'angular_velocity_rad_s': _vector(obj['angular_velocity_rad_s'], 3)}
            if not row['objects']:
                raise ValueError('Empty native object inventory')
            if self.frames:
                previous = self.frames[-1]
                if set(previous['objects']) != set(row['objects']):
                    raise ValueError('Native object inventory changed')
                if row['native_step'] != previous['native_step']+1:
                    self.errors.append('Missing or out-of-order native step '+str(step))
                if row['time_s'] <= previous['time_s']:
                    raise ValueError('Native episode time must increase')
                if abs(row['time_s']-previous['time_s']-self.physics_dt) > 1.e-8:
                    self.errors.append('Noncontiguous episode clock at native step '+str(step))
            self.frames.append(row)
            return True
        except (KeyError, TypeError, ValueError, IndexError, OverflowError, AttributeError) as exc:
            self.errors.append(type(exc).__name__+': '+str(exc))
            return False

    def summary(self, window_s=.4):
        window_s = _finite(window_s)
        if window_s <= 0.:
            raise ValueError('Positive trailing window required')
        end = self.frames[-1]['time_s'] if self.frames else 0.
        rows = [row for row in self.frames if row['time_s'] >= end-window_s-1.e-9]
        span = rows[-1]['time_s']-rows[0]['time_s'] if rows else 0.
        result = {'schema': 'kitchen_native_motion_diagnostics.v1', 'read_only': True,
            'task_success_claimed': False, 'physics_dt_s': self.physics_dt,
            'derivative_clock': 'time_s; native_time_s is preserved separately in raw frames',
            'window_requested_s': window_s, 'window_start_s': rows[0]['time_s'] if rows else None,
            'window_end_s': end if rows else None, 'window_available_s': span,
            'window_complete': span+1.e-8 >= window_s and not self.errors,
            'recorded_frames': len(self.frames), 'window_frames': len(rows),
            'errors': list(self.errors), 'limits': deepcopy(self.limits), 'objects': {},
            'scope': 'Native and pose-derived motion measurements only. 20 ms displacement averages do not waive native peaks or modify referee thresholds.'}
        if not rows:
            return result
        times = np.asarray([r['time_s'] for r in rows])
        steps = np.asarray([r['native_step'] for r in rows])
        differences = np.diff(times)
        contiguous = np.diff(steps) == 1
        lag = max(1, round(.020/self.physics_dt))
        # Only intervals that actually span 20 ms are labelled as such.
        coarse_dt = times[lag:]-times[:-lag]
        coarse_valid = ((steps[lag:]-steps[:-lag]) == lag) & (np.abs(coarse_dt-.020) < 1.e-8)
        for name in rows[0]['objects']:
            states = [r['objects'][name] for r in rows]
            positions = np.asarray([s['position_m'] for s in states])
            quaternions = np.asarray([s['quaternion_wxyz'] for s in states])
            native_linear = np.linalg.norm([s['linear_velocity_m_s'] for s in states], axis=1)
            native_angular = np.linalg.norm([s['angular_velocity_rad_s'] for s in states], axis=1)
            linear_delta = np.linalg.norm(np.diff(positions, axis=0), axis=1)
            angular_delta = _angle(quaternions[1:], quaternions[:-1])
            adjacent_linear = (linear_delta/differences)[contiguous]
            adjacent_angular = (angular_delta/differences)[contiguous]
            coarse_linear = (np.linalg.norm(positions[lag:]-positions[:-lag], axis=1)/coarse_dt)[coarse_valid]
            coarse_angular = (_angle(quaternions[lag:], quaternions[:-lag])/coarse_dt)[coarse_valid]
            p_range, q_range = _ranges(positions, quaternions)
            speeds = {'native': (native_linear, native_angular),
                      'adjacent_pose': (adjacent_linear, adjacent_angular),
                      'pose_20ms': (coarse_linear, coarse_angular)}
            measurements = {}
            for label, (linear, angular) in speeds.items():
                measurements[label] = {'samples': len(linear),
                    'linear_speed_peak_m_s': float(linear.max()) if len(linear) else None,
                    'angular_speed_peak_rad_s': float(angular.max()) if len(angular) else None,
                    'linear_over_limit_count': int(np.count_nonzero(linear > self.limits['maximum_linear_speed_m_s'])),
                    'angular_over_limit_count': int(np.count_nonzero(angular > self.limits['maximum_angular_speed_rad_s']))}
            measurements.update(position_range_m=p_range, orientation_range_rad=q_range,
                position_range_over_limit=p_range > self.limits['maximum_stable_position_range_m'],
                orientation_range_over_limit=q_range > self.limits['maximum_stable_orientation_range_rad'],
                net_displacement_m=float(np.linalg.norm(positions[-1]-positions[0])),
                net_rotation_rad=float(_angle(quaternions[-1], quaternions[0])),
                pose_path_length_m=float(linear_delta[contiguous].sum()),
                pose_rotation_path_rad=float(angular_delta[contiguous].sum()))
            result['objects'][name] = measurements
        return result

    def save_raw(self, path):
        """Write normalized copied samples only; never accesses the simulator."""
        path = Path(path)
        with path.open('w') as stream:
            for row in self.frames:
                stream.write(json.dumps(row, separators=(',', ':'), allow_nan=False)+'\n')
        return str(path)
