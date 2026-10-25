"""Pure native-frame checks using the frozen fastener task's original limits."""
from __future__ import annotations

import numpy as np

from scripts.validation.mechanical_tool_checks import phase_window
from scripts.validation.validate_workrooms import angle


def _contact(row):
    """Retain either native scalar sign as contact pressure presence only.

    Signed normal impulses are not a nonnegative force magnitude convention.
    Do not clamp negative records or infer a quantitative force from this sum.
    An all-zero, valid manifold still represents touching for robot release.
    """
    impulses = np.asarray(row['normal_impulses_ns'], dtype=float).reshape(-1)
    separations = np.asarray(row['separations_m'], dtype=float).reshape(-1)
    if (not len(impulses) or impulses.shape != separations.shape
            or not np.isfinite(np.r_[impulses, separations]).all()):
        raise ValueError('Invalid raw native contact impulses or separations')
    return float(np.abs(impulses).sum()), float(separations.min())


def _actor_matches(expected, actual):
    return actual == expected or actual.startswith(expected + '/')


def _pair_matches(actual, expected):
    return all(any(_actor_matches(value, candidate) for candidate in actual) for value in expected)


def task_contact_penetration(rows, paths, dt, config):
    required = set(paths.values()); maximum = 0.; records = 0
    indices = np.asarray([row['native_step_index'] for row in rows], float)
    times = np.asarray([row['simulation_time_s'] for row in rows], float)
    continuous = bool(len(rows) > 0 and np.isfinite(indices).all() and np.isfinite(times).all()
        and np.all(indices == np.floor(indices)) and np.all(np.diff(indices) == 1)
        and np.allclose(np.diff(times), dt, atol=1e-9, rtol=0))
    for row in rows:
        for contact in row['raw_contacts']:
            if required.intersection((contact['actor0'], contact['actor1'])):
                _, separation = _contact(contact)
                maximum = max(maximum, -separation); records += 1
    return {'passed': continuous and records > 0 and maximum <= config['maximum_reported_penetration_m'],
        'complete_native_sequence': continuous, 'native_samples': len(rows),
        'task_contact_records': records, 'maximum_reported_penetration_m': maximum,
        'frozen_limit_m': config['maximum_reported_penetration_m'],
        'scope': 'Every recorded raw pair involving a task actor, including zero-impulse points. Observer retains all tool/bolt contacts; static fixture-room overlap is outside this moving-object gate.'}


def passive_phase_window(rows, phase, paths, dt, config, *, support_pairs=None):
    if phase not in ('settle', 'final_passive'):
        raise ValueError('Passive gate requires settle or final_passive')
    duration = config['settle_s' if phase == 'settle' else 'release_hold_s']
    endpoints = [i for i, row in enumerate(rows) if row['phase'] == phase]
    history = rows[:endpoints[-1]+1] if endpoints else []
    window, sampling = phase_window(history, phase, dt, duration)
    sampling['available_phase_native_samples'] = len(endpoints)
    sampling['available_phase_duration_s'] = len(endpoints)*dt
    thresholds = {k: config[k] for k in ('maximum_stable_linear_speed_m_s',
        'maximum_stable_angular_speed_rad_s', 'maximum_stable_observed_angular_speed_rad_s',
        'maximum_stable_position_range_m', 'maximum_stable_angle_range_rad',
        'maximum_support_gap_s', 'minimum_pair_impulse_ns', 'maximum_reported_penetration_m',
        'maximum_settle_position_drift_m', 'maximum_settle_orientation_drift_rad')}
    if not sampling['passed']:
        return {'passed': False, 'sampling': sampling, 'thresholds': thresholds,
            'reason': 'Incomplete frozen passive dwell window; a shorter diagnostic is not release qualification.'}
    stride = round(config['control_period_s']/dt)
    if stride < 1 or len(window) % stride or abs(stride*dt-config['control_period_s']) > 1e-9:
        raise ValueError('Frozen control interval must divide the passive native window')
    anchor = history[-len(window)-1] if len(history) > len(window) else window[0]
    if support_pairs is not None:
        if (not isinstance(support_pairs, dict) or len(support_pairs) != 2
                or any(not isinstance(k, str) or not k or not isinstance(v, (list, tuple)) or len(v) != 2
                       or any(not isinstance(p, str) or not p for p in v)
                       or v[0] == v[1] for k, v in support_pairs.items())
                or {v[0] for v in support_pairs.values()} != {paths['screwdriver'], paths['bolt']}):
            raise ValueError('Declare one explicit support pair per movable task body')
        expected = {key: frozenset(pair) for key, pair in support_pairs.items()}
    else:
        expected = {}
        if paths.get('tool_supply_rest'):
            expected['tool_supply'] = frozenset((paths['screwdriver'], paths['tool_supply_rest']))
        else:
            expected['tool_bolt'] = frozenset((paths['screwdriver'], paths['bolt']))
        if paths.get('bolt_supply_rest'):
            expected['bolt_supply'] = frozenset((paths['bolt'], paths['bolt_supply_rest']))
        else:
            expected['bolt_fixture'] = frozenset((paths['bolt'], paths['fixture']))
    impulses = {key: [] for key in expected}; unexpected = set(); robot_pairs = set(); penetration = 0.
    robot_prefix = config['robot_actor_prefix']
    if not isinstance(robot_prefix, str) or not robot_prefix:
        raise ValueError('Frozen robot actor prefix is required for passive release')
    measurements = {}
    for name in ('screwdriver', 'bolt'):
        poses = np.asarray([row['object_poses_xyzw'][name] for row in window], float)
        speeds = np.asarray([row['object_velocities'][name] for row in window], float)
        reference = np.asarray(anchor['object_poses_xyzw'][name], float)
        if (poses.shape != (len(window), 7) or speeds.shape != (len(window), 6)
                or not np.isfinite(poses).all() or not np.isfinite(speeds).all()
                or not np.isfinite(reference).all()
                or not np.allclose(np.linalg.norm(poses[:, 3:], axis=1), 1., atol=1e-5, rtol=0)):
            raise ValueError('Invalid native passive pose or velocity')
        boundary = np.vstack([reference, poses[stride-1::stride]])
        observed_linear = np.linalg.norm(np.diff(boundary[:, :3], axis=0), axis=1)/config['control_period_s']
        observed_angular = [angle(a[[6, 3, 4, 5]], b[[6, 3, 4, 5]])/config['control_period_s']
                            for a, b in zip(boundary[:-1], boundary[1:])]
        result = {'maximum_native_linear_speed_m_s': float(np.linalg.norm(speeds[:, :3], axis=1).max()),
            'maximum_native_angular_speed_rad_s': float(np.linalg.norm(speeds[:, 3:], axis=1).max()),
            'maximum_observed_20ms_linear_speed_m_s': float(observed_linear.max()),
            'maximum_observed_20ms_angular_speed_rad_s': float(max(observed_angular)),
            'maximum_position_from_window_anchor_m': float(np.linalg.norm(poses[:, :3]-reference[:3], axis=1).max()),
            'maximum_orientation_from_window_anchor_rad': max(angle(p[[6, 3, 4, 5]], reference[[6, 3, 4, 5]]) for p in poses)}
        if phase == 'settle':
            original = config['initial_objects'][name]
            result['maximum_position_from_frozen_reset_m'] = float(np.linalg.norm(poses[:, :3]-original['position_m'], axis=1).max())
            result['maximum_orientation_from_frozen_reset_rad'] = max(angle(p[[6, 3, 4, 5]], original['quaternion_wxyz']) for p in poses)
        measurements[name] = result
    for row in window:
        values = dict.fromkeys(expected, 0.)
        for contact in row['raw_contacts']:
            pair = frozenset((contact['actor0'], contact['actor1']))
            if not pair.intersection((paths['screwdriver'], paths['bolt'])):
                continue
            impulse, separation = _contact(contact); penetration = max(penetration, -separation)
            if any(actor.startswith(robot_prefix) for actor in pair):
                robot_pairs.add(tuple(sorted(pair)))
            matched = any(_pair_matches(pair, required) for required in expected.values())
            if impulse > config['minimum_pair_impulse_ns'] and not matched:
                unexpected.add(tuple(sorted(pair)))
            for key, required in expected.items():
                if _pair_matches(pair, required):
                    values[key] += impulse
        for key, value in values.items():
            impulses[key].append(value)
    intervals = {key: [sum(values[i:i+stride]) for i in range(0, len(values), stride)]
                 for key, values in impulses.items()}
    bolt_supports = [key for key, pair in expected.items()
                     if paths['bolt'] in pair and paths['screwdriver'] not in pair]
    if len(bolt_supports) != 1:
        raise ValueError('Declare exactly one external support pair for the bolt')
    longest = current = 0
    for value in impulses[bolt_supports[0]]:
        current = 0 if value > config['minimum_pair_impulse_ns'] else current+1
        longest = max(longest, current)
    checks = {'complete_native_dwell': True, 'task_contact_penetration': penetration <= config['maximum_reported_penetration_m'],
        'both_support_pairs_every_20ms': all(min(v) > config['minimum_pair_impulse_ns'] for v in intervals.values()),
        'fixture_support_gap': longest*dt <= config['maximum_support_gap_s']+1e-12,
        'no_unexpected_pressure_object_contacts': not unexpected,
        'no_robot_object_manifolds': not robot_pairs}
    limits = {'maximum_native_linear_speed_m_s': 'maximum_stable_linear_speed_m_s',
        'maximum_native_angular_speed_rad_s': 'maximum_stable_angular_speed_rad_s',
        'maximum_observed_20ms_linear_speed_m_s': 'maximum_stable_linear_speed_m_s',
        'maximum_observed_20ms_angular_speed_rad_s': 'maximum_stable_observed_angular_speed_rad_s',
        'maximum_position_from_window_anchor_m': 'maximum_stable_position_range_m',
        'maximum_orientation_from_window_anchor_rad': 'maximum_stable_angle_range_rad',
        'maximum_position_from_frozen_reset_m': 'maximum_settle_position_drift_m',
        'maximum_orientation_from_frozen_reset_rad': 'maximum_settle_orientation_drift_rad'}
    for name, result in measurements.items():
        checks[name+'_stable_pose_and_velocity'] = all(value <= config[limits[key]] for key, value in result.items())
    return {'passed': all(checks.values()), 'checks': checks, 'sampling': sampling,
        'thresholds': thresholds, 'objects': measurements,
        'maximum_reported_penetration_m': penetration, 'maximum_fixture_support_gap_s': longest*dt,
        'bolt_support_pair': bolt_supports[0], 'maximum_bolt_support_gap_s': longest*dt,
        'support_interval_impulses_ns': intervals,
        'unexpected_pressure_pairs': [list(p) for p in sorted(unexpected)],
        'robot_object_manifold_pairs': [list(p) for p in sorted(robot_pairs)],
        'support_impulse_semantics': 'Sum of absolute raw scalars for pressure presence only, not quantitative normal force.',
        'scope': 'Frozen 0.5s task stability limits applied to complete native pose/velocity/contact windows and 20ms pose differences. No wrench/hand-tool manipulation credit or robot task score.'}
