"""Read-only acceptance gates for the preinserted mechanical tool diagnostic."""
from __future__ import annotations

import math
import numpy as np

from scripts.validation.validate_workrooms import angle


CONTACT_NOISE_NS = 1.e-8
RESET_LIMITS = {'window_s': .2, 'displacement_m': .003, 'rotation_rad': .03,
                'linear_speed_m_s': .02, 'angular_speed_rad_s': .3,
                'penetration_m': .002, 'positive_contact_noise_ns': CONTACT_NOISE_NS,
                'support_interval_s': .02, 'maximum_pressure_gap_s': .001}
RELEASE_LIMITS = {'window_s': .1, 'arm_joint_error_rad': .01,
                  'hand_joint_error_rad': .02, 'positive_contact_noise_ns': CONTACT_NOISE_NS}


def phase_window(rows, phase, dt, duration):
    """Require a complete terminal window from one phase and consecutive steps."""
    if not np.isfinite([dt, duration]).all() or dt <= 0 or duration <= 0:
        raise ValueError('Window duration and native timestep must be positive')
    count = round(duration/dt)
    if count < 1 or abs(count*dt-duration) > 1.e-9:
        raise ValueError('Window duration must be an integer native step count')
    window = rows[-count:]
    complete = len(window) == count and all(row.get('phase') == phase for row in window)
    if complete:
        indices = np.asarray([row['native_step_index'] for row in window], dtype=float)
        times = np.asarray([row['simulation_time_s'] for row in window], dtype=float)
        complete = bool(np.isfinite(indices).all() and np.isfinite(times).all()
            and np.all(indices == np.floor(indices)) and np.all(np.diff(indices) == 1)
            and np.allclose(np.diff(times), dt, atol=1.e-9, rtol=0))
    return window, {'passed': bool(complete), 'phase': phase,
        'duration_s': duration, 'expected_native_samples': count, 'native_samples': len(window),
        'first_native_step_index': window[0].get('native_step_index') if window else None,
        'last_native_step_index': window[-1].get('native_step_index') if window else None}


def _contact(row):
    impulses = np.asarray(row['normal_impulses_ns'], dtype=float).reshape(-1)
    separations = np.asarray(row['separations_m'], dtype=float).reshape(-1)
    if not len(impulses) or impulses.shape != separations.shape or not np.isfinite(np.r_[impulses, separations]).all():
        raise ValueError('Invalid raw native contact impulses or separations')
    return float(np.maximum(impulses, 0.).sum()), float(separations.min())


def preinserted_window(rows, initial_objects, paths, dt):
    """Test stability and both actual contact pairs over the final 0.2 seconds."""
    window, sampling = phase_window(rows, 'settle', dt, RESET_LIMITS['window_s'])
    if not sampling['passed']:
        return {'passed': False, 'sampling': sampling, 'thresholds': dict(RESET_LIMITS)}
    expected = {'tool_bolt': frozenset((paths['screwdriver'], paths['bolt'])),
                'bolt_fixture': frozenset((paths['bolt'], paths['fixture']))}
    contact_samples = dict.fromkeys(expected, 0)
    contact_impulses = dict.fromkeys(expected, 0.)
    history = {name: [] for name in expected}
    unexpected = set()
    measurements = {name: {'maximum_displacement_m': 0., 'maximum_rotation_rad': 0.,
        'maximum_linear_speed_m_s': 0., 'maximum_angular_speed_rad_s': 0.}
        for name in ('screwdriver', 'bolt')}
    penetration = 0.
    for sample in window:
        for name, result in measurements.items():
            pose = np.asarray(sample['object_poses_xyzw'][name], dtype=float)
            velocity = np.asarray(sample['object_velocities'][name], dtype=float)
            if pose.shape != (7,) or velocity.shape != (6,) or not np.isfinite(np.r_[pose, velocity]).all():
                raise ValueError('Invalid native preinserted pose or velocity')
            origin = initial_objects[name]
            result['maximum_displacement_m'] = max(result['maximum_displacement_m'],
                float(np.linalg.norm(pose[:3]-origin['position_m'])))
            result['maximum_rotation_rad'] = max(result['maximum_rotation_rad'],
                angle(pose[[6, 3, 4, 5]], origin['quaternion_wxyz']))
            result['maximum_linear_speed_m_s'] = max(result['maximum_linear_speed_m_s'], float(np.linalg.norm(velocity[:3])))
            result['maximum_angular_speed_rad_s'] = max(result['maximum_angular_speed_rad_s'], float(np.linalg.norm(velocity[3:])))
        impulses = dict.fromkeys(expected, 0.)
        for contact in sample['raw_contacts']:
            pair = frozenset((contact['actor0'], contact['actor1']))
            if not pair.intersection((paths['screwdriver'], paths['bolt'])):
                continue
            impulse, separation = _contact(contact)
            penetration = max(penetration, -separation)
            if impulse > CONTACT_NOISE_NS and pair not in expected.values():
                unexpected.add(tuple(sorted(pair)))
            for label, required_pair in expected.items():
                if pair == required_pair:
                    impulses[label] += impulse
        for label, impulse in impulses.items():
            contact_samples[label] += int(impulse > CONTACT_NOISE_NS)
            contact_impulses[label] += impulse
            history[label].append(impulse)
    stride = round(RESET_LIMITS['support_interval_s']/dt)
    if stride < 1 or len(window) % stride or abs(stride*dt-RESET_LIMITS['support_interval_s']) > 1.e-9:
        raise ValueError('Support interval must divide the measured window into complete native samples')
    intervals, gaps = {}, {}
    for label, values in history.items():
        intervals[label] = [sum(values[i:i+stride]) for i in range(0, len(values), stride)]
        longest = current = 0
        for impulse in values:
            current = 0 if impulse > CONTACT_NOISE_NS else current+1
            longest = max(longest, current)
        gaps[label] = longest*dt
    checks = {'complete_native_window': sampling['passed'],
        'both_support_pairs_positive_every_control_interval': all(
            min(values) > CONTACT_NOISE_NS for values in intervals.values()),
        'pressure_gaps_bounded': all(gap <= RESET_LIMITS['maximum_pressure_gap_s']+1.e-12 for gap in gaps.values()),
        'no_unexpected_positive_object_contacts': not unexpected,
        'relevant_object_contact_penetration': penetration <= RESET_LIMITS['penetration_m']}
    for name, result in measurements.items():
        checks[name+'_stable_pose_and_velocity'] = all(result[measurement] <= RESET_LIMITS[limit]
            for measurement, limit in [('maximum_displacement_m', 'displacement_m'),
                ('maximum_rotation_rad', 'rotation_rad'), ('maximum_linear_speed_m_s', 'linear_speed_m_s'),
                ('maximum_angular_speed_rad_s', 'angular_speed_rad_s')])
    return {'passed': all(checks.values()), 'checks': checks, 'sampling': sampling,
        'objects': measurements, 'positive_contact_samples': contact_samples,
        'support_impulse_ns': contact_impulses, 'maximum_penetration_m': penetration,
        'support_interval_impulses_ns': intervals, 'maximum_pressure_gap_s': gaps,
        'unexpected_positive_actor_pairs': [list(pair) for pair in sorted(unexpected)],
        'expected_actor_pairs': {name: sorted(pair) for name, pair in expected.items()},
        'thresholds': dict(RESET_LIMITS),
        'scope': 'Final 0.2s of fresh passive settle; exact tool-bolt/bolt-fixture positive impulse in every 20ms interval, pressure gaps at most 1ms, plus unchanged pose/velocity/penetration limits. Contact presence only, not quantitative force balance or task success.',
        'gap_rationale': 'Discrete resting contact can have isolated zero-impulse substeps. At 1ms, gravity alone adds 0.00981m/s and 4.905um displacement from rest; actual velocities and poses remain independently bounded on every sample. No physics parameters change.'}


def released_phase_window(rows, phase, targets, hand_paths, tool_path, dt):
    """Verify all actual joints reach the phase endpoint and hand contact ends."""
    if phase not in ('open', 'retreat', 'final_hold'):
        raise ValueError('Release endpoint gate only accepts open, retreat and final_hold')
    arm = {f'fr3_joint{i}' for i in range(1, 8)}
    if (len(targets) != 29 or not arm <= targets.keys()
            or any(not name.startswith('right_') for name in set(targets)-arm)
            or not all(math.isfinite(value) for value in targets.values())):
        raise ValueError('Release endpoints require all 29 finite joint targets')
    window, sampling = phase_window(rows, phase, dt, RELEASE_LIMITS['window_s'])
    if not sampling['passed']:
        return {'passed': False, 'sampling': sampling, 'thresholds': dict(RELEASE_LIMITS)}
    errors = dict.fromkeys(targets, 0.)
    contacting_samples, maximum_impulse = 0, 0.
    for sample in window:
        observed = sample['joint_positions_rad']
        if set(observed) != set(targets) or not all(math.isfinite(value) for value in observed.values()):
            raise ValueError('Native endpoint joint inventory or values are invalid')
        for name in targets:
            errors[name] = max(errors[name], abs(observed[name]-targets[name]))
        impulse = 0.
        for contact in sample['raw_contacts']:
            pair = {contact['actor0'], contact['actor1']}
            if tool_path in pair and pair.intersection(hand_paths):
                impulse += _contact(contact)[0]
        contacting_samples += int(impulse > CONTACT_NOISE_NS)
        maximum_impulse = max(maximum_impulse, impulse)
    checks = {'complete_native_window': sampling['passed'],
        'all_arm_joints_at_target': max(errors[name] for name in arm) <= RELEASE_LIMITS['arm_joint_error_rad'],
        'all_hand_joints_at_target': max(errors[name] for name in set(targets)-arm) <= RELEASE_LIMITS['hand_joint_error_rad'],
        'hand_tool_positive_contact_released': contacting_samples == 0}
    return {'passed': all(checks.values()), 'checks': checks, 'sampling': sampling,
        'maximum_joint_errors_rad': errors, 'hand_tool_contact_samples': contacting_samples,
        'maximum_hand_tool_impulse_ns': maximum_impulse, 'thresholds': dict(RELEASE_LIMITS),
        'scope': 'All 29 actual native joints within endpoint tolerances and no positive hand-tool contact over the final 0.1s; command completion alone is insufficient.'}


def contact_exchange_check(result, native_steps, recorded_samples):
    counts = (result.get('acquisitions'), result.get('deliveries'), native_steps, recorded_samples)
    return {'passed': bool(result.get('all_samples_consumed_once') is True
        and native_steps > 0 and all(value == native_steps for value in counts)),
        'all_samples_consumed_once': result.get('all_samples_consumed_once'),
        'acquisitions': counts[0], 'deliveries': counts[1],
        'native_steps': native_steps, 'recorded_samples': recorded_samples}
