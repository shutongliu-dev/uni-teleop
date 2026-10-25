"""Complete native velocity and continuous 20 ms pose checks for fasteners."""
from __future__ import annotations

import numpy as np

from scripts.validation.validate_workrooms import angle


def full_motion_window(rows, dt, config, *, expected_samples, anchor=None):
    """Apply frozen motion limits without resetting observation at phase changes.

    The optional anchor is an actual native state captured before the first
    integration step. Legacy streams without it cannot check the first 20 ms
    pose interval, but still check every recorded native velocity.
    """
    stride = round(config['control_period_s']/dt)
    if stride < 1 or abs(stride*dt-config['control_period_s']) > 1e-9:
        raise ValueError('Frozen control interval must contain whole native steps')
    if type(expected_samples) is not int or expected_samples <= 0:
        raise ValueError('Expected native sample count must be positive')
    count = len(rows)
    checks = {'complete_native_sampling': count == expected_samples and count % stride == 0,
              'finite_native_state': True}
    if not count:
        return {'passed': False, 'checks': checks, 'native_samples': 0, 'objects': {}}
    clock_rows = ([anchor] if anchor is not None else []) + rows
    indices = np.asarray([row['native_step_index'] for row in clock_rows], float)
    times = np.asarray([row['simulation_time_s'] for row in clock_rows], float)
    checks['complete_native_sampling'] &= bool(
        np.isfinite(indices).all() and np.all(indices == np.floor(indices))
        and np.all(np.diff(indices) == 1) and np.isfinite(times).all()
        and np.allclose(np.diff(times), dt, atol=1e-9, rtol=0))
    objects = {}
    for name, angular_key in (('bolt', 'maximum_bolt_angular_speed_rad_s'),
                              ('screwdriver', 'maximum_tool_angular_speed_rad_s')):
        poses = np.asarray([row['object_poses_xyzw'][name] for row in rows], float)
        velocities = np.asarray([row['object_velocities'][name] for row in rows], float)
        valid = (poses.shape == (count, 7) and velocities.shape == (count, 6)
                 and np.isfinite(poses).all() and np.isfinite(velocities).all()
                 and np.allclose(np.linalg.norm(poses[:, 3:], axis=1), 1., atol=1e-5, rtol=0))
        boundary = poses[stride-1::stride]
        if anchor is not None:
            initial = np.asarray(anchor['object_poses_xyzw'][name], float)
            valid &= (initial.shape == (7,) and np.isfinite(initial).all()
                      and np.isclose(np.linalg.norm(initial[3:]), 1., atol=1e-5, rtol=0))
            if valid:
                boundary = np.vstack([initial, boundary])
        checks['finite_native_state'] &= bool(valid)
        if not valid:
            objects[name] = {'passed': False, 'reason': 'Invalid native pose or velocity'}
            continue
        intervals = max(0, len(boundary)-1)
        observed_linear = np.linalg.norm(np.diff(boundary[:, :3], axis=0), axis=1)/config['control_period_s']
        observed_angular = [angle(a[[6, 3, 4, 5]], b[[6, 3, 4, 5]])/config['control_period_s']
                            for a, b in zip(boundary[:-1], boundary[1:])]
        values = {
            'maximum_native_linear_speed_m_s': float(np.linalg.norm(velocities[:, :3], axis=1).max()),
            'maximum_native_angular_speed_rad_s': float(np.linalg.norm(velocities[:, 3:], axis=1).max()),
            'maximum_observed_20ms_linear_speed_m_s': float(observed_linear.max()) if intervals else 0.,
            'maximum_observed_20ms_angular_speed_rad_s': float(max(observed_angular)) if intervals else 0.,
        }
        limits = {'linear_speed_m_s': config['maximum_object_linear_speed_m_s'],
                  'angular_speed_rad_s': config[angular_key]}
        safe = (max(values['maximum_native_linear_speed_m_s'], values['maximum_observed_20ms_linear_speed_m_s']) <= limits['linear_speed_m_s']
                and max(values['maximum_native_angular_speed_rad_s'], values['maximum_observed_20ms_angular_speed_rad_s']) <= limits['angular_speed_rad_s'])
        objects[name] = {'passed': bool(safe), **values, 'limits': limits,
                         'observed_20ms_intervals': intervals}
    checks['object_motion_safety'] = all(row['passed'] for row in objects.values())
    return {'passed': all(checks.values()), 'checks': checks, 'objects': objects,
            'native_samples': count, 'expected_native_samples': expected_samples,
            'actual_reset_anchor_available': anchor is not None,
            'first_20ms_pose_interval_checked': anchor is not None,
            'phase_boundaries_reset_pose_history': False,
            'scope': 'Every native velocity and all available continuous 20 ms pose intervals; frozen limits unchanged.'}
