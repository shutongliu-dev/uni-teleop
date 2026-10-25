"""Read-only native workshop diagnostics, separate from manipulation scoring.

Inputs are recorded native observations and actual contact-view path mappings.
These helpers do not author physics, launch a simulator, set state, or accept
planned tasks.  Geometry uses body-local bounds, including offset asset origins.
"""
from __future__ import annotations

from collections.abc import Mapping
import itertools
import math

import numpy as np

from scripts.validation.validate_workrooms import THRESHOLDS, angle, rotation, support_contact, vector


CONTACT_NOISE_NS = 1.e-8
READY_THRESHOLDS = {'arm_joint_error_rad': .01, 'hand_joint_error_rad': .02,
                    'wrist_position_error_m': .005, 'wrist_rotation_error_rad': .03,
                    'rest_window_steps': 20}
READY_PHASES = ('settle', 'post_motion', 'repeat_settle')


def assess_ready_pose(rows, ready_joints, wrist_target):
    """Check every ready joint and native wrist in all three terminal windows.

    Pass the complete record list, the frozen 29-joint initial-state mapping,
    and a world wrist pose with ``position_m`` / ``quaternion_wxyz`` (e.g.
    ``ready_design['wrist_fk']``).  Each of settle, post_motion and repeat_settle
    must contain at least 20 control observations; all last 20 are evaluated.
    """
    arm_names = {f'fr3_joint{i}' for i in range(1, 8)}
    if (not isinstance(ready_joints, dict) or len(ready_joints) != 29
            or not arm_names <= set(ready_joints)
            or any(not math.isfinite(float(value)) for value in ready_joints.values())):
        raise ValueError('Ready pose requires exactly 29 finite frozen joint targets')
    hand_names = set(ready_joints)-arm_names
    if not all(name.startswith('right_') for name in hand_names):
        raise ValueError('Ready pose requires the single right SHARPA hand')
    if wrist_target.get('frame_id', 'world') != 'world':
        raise ValueError('Ready wrist target must be in world coordinates')
    target_position = vector(wrist_target['position_m'], 3)
    target_quaternion = vector(wrist_target['quaternion_wxyz'], 4)
    rotation(target_quaternion)
    result = {}
    for phase in READY_PHASES:
        phase_rows = [row for row in rows if row['phase'] == phase]
        if len(phase_rows) < READY_THRESHOLDS['rest_window_steps']:
            raise ValueError('Missing 20-window ready evidence for phase: '+phase)
        window = phase_rows[-READY_THRESHOLDS['rest_window_steps']:]
        errors = dict.fromkeys(ready_joints, 0.)
        wrist_position_error, wrist_rotation_error = 0., 0.
        finite_active = True
        for row in window:
            observed = row['observation']
            joints = observed['joint_positions_rad']
            if set(joints) != set(ready_joints):
                raise ValueError('Native ready joint inventory differs from frozen 29 targets')
            for name, reference in ready_joints.items():
                measured = float(joints[name])
                if not math.isfinite(measured):
                    raise ValueError('Nonfinite native ready joint')
                errors[name] = max(errors[name], abs(measured-reference))
            wrist = observed['robot']['right_hand_flange']
            position = vector(wrist['position_m'], 3)
            wrist_position_error = max(wrist_position_error, float(np.linalg.norm(position-target_position)))
            wrist_rotation_error = max(wrist_rotation_error, angle(wrist['quaternion_wxyz'], target_quaternion))
            finite_active = finite_active and observed['metrics']['state_finite'] and not observed['terminated']
        arm_error = max(errors[name] for name in arm_names)
        hand_error = max(errors[name] for name in hand_names)
        checks = {'finite_active_state': bool(finite_active),
                  'all_seven_arm_joints_at_ready': arm_error <= READY_THRESHOLDS['arm_joint_error_rad'],
                  'all_twenty_two_hand_joints_at_ready': hand_error <= READY_THRESHOLDS['hand_joint_error_rad'],
                  'native_wrist_position_at_ready': wrist_position_error <= READY_THRESHOLDS['wrist_position_error_m'],
                  'native_wrist_orientation_at_ready': wrist_rotation_error <= READY_THRESHOLDS['wrist_rotation_error_rad']}
        result[phase] = {'passed': all(checks.values()), 'checks': checks,
            'checked_control_windows': len(window), 'maximum_joint_errors_rad': errors,
            'maximum_arm_joint_error_rad': arm_error, 'maximum_hand_joint_error_rad': hand_error,
            'maximum_wrist_position_error_m': wrist_position_error,
            'maximum_wrist_rotation_error_rad': wrist_rotation_error}
    return {'passed': all(row['passed'] for row in result.values()),
            'phases': result, 'thresholds': dict(READY_THRESHOLDS),
            'scope': 'All 29 actual native joints and right_hand_flange pose in each of the last 20 control windows of settle, post_motion and repeat_settle. No manipulation or swept-path acceptance.'}


def _corners(spec):
    low, high = vector(spec['bounds_min_m'], 3), vector(spec['bounds_max_m'], 3)
    if np.any(high <= low):
        raise ValueError('Body-local bounds must have strictly positive extents')
    return np.array(list(itertools.product(*zip(low, high))))


def _bottom(state, corners):
    return float((corners @ rotation(state['quaternion_wxyz']).T
                  + vector(state['position_m'], 3))[:, 2].min())


def _complete_contacts(observation):
    contacts = observation['contacts']
    return not contacts['errors'] and contacts['sampling']['complete'] is True


def assess_support(initial_observation, rows, specs, paths):
    """Check native drop/settle/support for offset-origin diagnostic objects.

    The final 20 control windows use the existing workroom tolerances, including
    at least 18 windows with positive expected-support impulse.  A 30 mm release
    is valid; observed fall is required to reach min(20 mm, initial gap / 2).
    Each row has the shape returned by ``validate_workrooms.record``.
    """
    if not rows or len(rows) < THRESHOLDS['rest_window_steps']:
        raise ValueError('Insufficient settling observations')
    if (not specs or set(initial_observation['objects']) != set(specs)
            or set(paths) != set(specs)):
        raise ValueError('Diagnostic object inventory mismatch')
    if any(set(row['observation']['objects']) != set(specs) for row in rows):
        raise ValueError('Recorded diagnostic object inventory changed')
    window = rows[-THRESHOLDS['rest_window_steps']:]
    complete = all(_complete_contacts(row['observation']) for row in rows)
    results = {}
    for name, spec in specs.items():
        corners = _corners(spec)
        top, release = float(spec['support_top_z_m']), float(spec['release_height_m'])
        if not np.isfinite([top, release]).all() or release <= 0:
            raise ValueError('Diagnostic release height must be finite and positive')
        start = initial_observation['objects'][name]
        p0 = vector(start['position_m'], 3)
        gap0 = _bottom(start, corners)-top
        if not math.isfinite(gap0) or gap0 <= 0:
            raise ValueError('Diagnostic object must begin above its declared support')
        states = [row['observation']['objects'][name] for row in rows]
        rest = [row['observation']['objects'][name] for row in window]
        positions = np.array([vector(state['position_m'], 3) for state in states])
        rest_positions = np.array([vector(state['position_m'], 3) for state in rest])
        velocities = np.array([vector(state['linear_velocity_m_s'], 3) for state in states])
        speeds = [float(np.linalg.norm(vector(state['linear_velocity_m_s'], 3))) for state in rest]
        omegas = [float(np.linalg.norm(vector(state['angular_velocity_rad_s'], 3))) for state in rest]
        gaps = [_bottom(state, corners)-top for state in rest]
        contacts = [support_contact(row['observation'], paths[name], spec['expected_support_path'])
                    for row in rows]
        supported = sum(row['impulse_ns'] > 0 for row in contacts[-THRESHOLDS['rest_window_steps']:])
        separations = [row['min_separation_m'] for row in contacts if row['min_separation_m'] is not None]
        descent, vz = float(p0[2]-positions[:, 2].min()), float(velocities[:, 2].min())
        drift = float(np.linalg.norm(rest_positions-rest_positions[0], axis=1).max())
        impulse = sum(row['impulse_ns'] for row in contacts)
        checks = {
            'complete_native_contacts': complete,
            'declared_release_height': abs(gap0-release) <= 1.e-4,
            'gravity_descent': descent >= min(THRESHOLDS['minimum_fall_m'], gap0/2),
            'observed_downward_velocity': vz <= -THRESHOLDS['minimum_downward_speed_m_s'],
            'positive_expected_support_impulse': impulse > 0,
            'persistent_support': supported >= THRESHOLDS['minimum_supported_rest_windows'],
            'rest_linear_velocity': max(speeds) <= THRESHOLDS['maximum_rest_speed_m_s'],
            'rest_angular_velocity': max(omegas) <= THRESHOLDS['maximum_rest_angular_speed_rad_s'],
            'geometry_penetration': min(gaps) >= -THRESHOLDS['maximum_penetration_m'],
            'geometry_support_gap': max(gaps) <= THRESHOLDS['maximum_support_gap_m'],
            'solver_penetration': bool(separations) and min(separations) >= -THRESHOLDS['maximum_penetration_m'],
            'rest_drift': drift <= THRESHOLDS['maximum_rest_drift_m'],
        }
        results[name] = {'passed': all(checks.values()), 'checks': checks,
            'bounds_min_m': list(spec['bounds_min_m']), 'bounds_max_m': list(spec['bounds_max_m']),
            'initial_support_gap_m': gap0, 'declared_release_height_m': release,
            'maximum_descent_m': descent, 'minimum_vertical_velocity_m_s': vz,
            'maximum_rest_speed_m_s': max(speeds), 'maximum_rest_angular_speed_rad_s': max(omegas),
            'rest_bottom_gap_range_m': [min(gaps), max(gaps)],
            'minimum_solver_separation_m': min(separations) if separations else None,
            'rest_drift_m': drift, 'supported_rest_windows': supported,
            'support_impulse_ns': impulse, 'object_actor_path': paths[name],
            'expected_support_path': spec['expected_support_path']}
    return {'passed': all(row['passed'] for row in results.values()), 'objects': results,
            'rest_window_steps': THRESHOLDS['rest_window_steps'],
            'scope': 'Native drop/support diagnostic only. Bottom uses all eight transformed body-local AABB corners; this bound is conservative for non-box geometry, not a measured contact point.'}


def _paths(values):
    result = set(values.values() if isinstance(values, Mapping) else values)
    if not result or any(not isinstance(path, str) or not path.startswith('/') for path in result):
        raise ValueError('Expected a nonempty set of absolute USD actor paths')
    return result


def assess_robot_clearance(rows, robot_paths, allowed_pairs=()):
    """Reject any observed positive robot contact except exact base/mount pairs.

    Existing native collision filters are preserved.  Allowed pairs must name
    the actual ``fr3_link0`` body and one external installation actor; a broad
    environment prefix or a hand/self-collision exemption is not accepted.
    """
    if not rows:
        raise ValueError('Robot clearance needs recorded native observations')
    robot = _paths(robot_paths)
    allowed = set()
    for pair in allowed_pairs:
        if isinstance(pair, str) or len(pair) != 2:
            raise ValueError('Allowed base contact must be one exact actor pair')
        actors = _paths(pair)
        members = actors & robot
        if (len(actors) != 2 or len(members) != 1
                or next(iter(members)).rsplit('/', 1)[-1] != 'fr3_link0'):
            raise ValueError('Only explicit fixed-base mounting contact may be allowed')
        allowed.add(tuple(sorted(actors)))
    pairs, incomplete = {}, []
    for index, row in enumerate(rows):
        observation = row['observation']
        if not _complete_contacts(observation):
            incomplete.append(index)
        for contact in observation['contacts']['rigid']:
            key = tuple(sorted((contact['actor0'], contact['actor1'])))
            if not robot.intersection(key):
                continue
            impulse, separation = float(contact['impulse_magnitude_sum_ns']), float(contact['min_separation_m'])
            time_s = float(contact['time_s'])
            if not np.isfinite([impulse, separation, time_s]).all() or impulse < 0:
                raise ValueError('Invalid native robot contact evidence')
            item = pairs.setdefault(key, {'actors': list(key),
                'kind': 'robot_self' if set(key) <= robot else 'robot_external',
                'allowed_fixed_base_mount': key in allowed, 'contact_records': 0,
                'positive_substeps': 0, 'impulse_ns': 0., 'maximum_substep_impulse_ns': 0.,
                'minimum_separation_m': separation})
            item['contact_records'] += 1
            item['positive_substeps'] += int(impulse > CONTACT_NOISE_NS)
            item['impulse_ns'] += impulse
            item['maximum_substep_impulse_ns'] = max(item['maximum_substep_impulse_ns'], impulse)
            item['minimum_separation_m'] = min(item['minimum_separation_m'], separation)
    records = [pairs[key] for key in sorted(pairs)]
    allowed_records = [row for row in records if row['allowed_fixed_base_mount']]
    rejected = [row for row in records if not row['allowed_fixed_base_mount'] and row['positive_substeps']]
    return {'passed': not incomplete and not rejected, 'contact_noise_ns': CONTACT_NOISE_NS,
            'allowed_pairs': [list(pair) for pair in sorted(allowed)],
            'allowed_contacts': allowed_records, 'rejected_contacts': rejected,
            'observed_robot_pairs': records, 'incomplete_contact_window_indices': incomplete,
            'scope': 'Every reported native substep robot contact; positive impulse above 1e-8 Ns is rejected except the exact declared fixed-base/mount pair. Requires independent contact-filter coverage; no foundation-reaction inference from absent contacts.'}


def contact_filter_coverage(env):
    """Check actual native sensor/filter paths against every composed collider.

    Collider-to-actor remapping is explicit.  Unobserved environment colliders
    are retained and fail the complete-coverage claim, never silently omitted.
    """
    info = env.scene_info
    robot = _paths(info['robot_link_paths'])
    objects = {row['path'] for row in info['objects'].values()}
    sensors = list(env._contact_sensor_paths)
    filters = [set(row) for row in env._contact_filter_paths]
    if len(filters) != len(sensors):
        raise ValueError('Native contact sensor/filter row counts differ')
    mapping = info.get('collider_actor_paths', {})
    workbench = info.get('workbench', {})
    raw_supports = set(info.get('support_paths', [])) | set(workbench.get('solid_paths', []))
    raw_supports.update(workbench.get('supports', {}).values())
    raw_fixtures = {path for fixture in info.get('fixtures', {}).values()
                    for path in fixture.get('contact_filter_paths', fixture.get('collision_paths', []))}
    support_actors = {mapping.get(path, path) for path in raw_supports}
    fixture_actors = {mapping.get(path, path) for path in raw_fixtures}
    expected_sensors = robot | objects
    expected_filters = expected_sensors | support_actors | fixture_actors
    missing_sensors = sorted(expected_sensors-set(sensors))
    missing_filters = {}
    for sensor in sorted(expected_sensors & set(sensors)):
        missing = sorted((expected_filters-{sensor})-filters[sensors.index(sensor)])
        if missing:
            missing_filters[sensor] = missing
    unobserved = sorted(set(workbench.get('contact_unobserved_paths', [])))
    checks = {'unique_native_sensors': len(sensors) == len(set(sensors)),
              'all_robot_and_object_sensors': not missing_sensors,
              'all_actor_pair_filters': not missing_filters,
              'environment_colliders_observable': not unobserved,
              'support_inventory_nonempty': bool(support_actors)}
    return {'passed': all(checks.values()), 'checks': checks,
            'robot_sensor_count': len(robot), 'object_sensor_count': len(objects),
            'native_sensor_count': len(sensors), 'required_actor_filter_count': len(expected_filters),
            'support_actor_paths': sorted(support_actors), 'fixture_actor_paths': sorted(fixture_actors),
            'missing_sensor_paths': missing_sensors, 'missing_filter_paths_by_sensor': missing_filters,
            'unobserved_environment_collider_paths': unobserved,
            'collider_actor_paths': dict(mapping),
            'scope': 'Actual PhysX contact-view path coverage, not contact forces. Every robot/object sensor must observe other robot/object bodies and all composed supports/fixtures; explicit self-body pairs are unnecessary.'}
