"""Offline scene validation of saved native contacts; no control or pose writes."""
from __future__ import annotations

import math
import numpy as np


def _quaternion_angle(a, b):
    """Sign-invariant rotation distance for already normalized quaternions."""
    b = b*np.where(np.sum(a*b, axis=-1, keepdims=True) < 0., -1., 1.)
    return 4.*np.arctan2(np.linalg.norm(a-b, axis=-1), np.linalg.norm(a+b, axis=-1))


def _saved_pose_review(rows, times, limits):
    """Independent saved-pose check, stricter than the native anchor envelope.

    This checks 20 ms saved endpoints, not every rolling native window. The
    native quiet receipt remains required for all intervening physics steps.
    """
    result = {'passed': False, 'errors': [], 'sample_count': len(rows),
        'sample_period_s': .020, 'envelope': 'maximum_all_pairs_in_saved_terminal_window',
        'scope': 'Additional scene-rest check; native rules use an envelope from the quiet-segment anchor.'}
    try:
        period = float(limits['stability_pose_window_s'])
        if not math.isfinite(period) or abs(period-.020) > 1.e-9:
            raise ValueError('Unsupported saved-pose stability window')
        positions = np.asarray([row['position_m'] for row in rows], dtype=float)
        quaternions = np.asarray([row['quaternion_wxyz'] for row in rows], dtype=float)
        times = np.asarray(times, dtype=float)
        if (len(rows) < 2 or positions.shape != (len(rows), 3)
                or quaternions.shape != (len(rows), 4) or times.shape != (len(rows),)
                or not all(np.isfinite(x).all() for x in (positions, quaternions, times))):
            raise ValueError('Missing or invalid saved native poses')
        durations = np.diff(times)
        if not np.all(np.abs(durations-period) < 1.e-8):
            raise ValueError('Saved native poses do not cover consecutive 20 ms windows')
        lengths = np.linalg.norm(quaternions, axis=1)
        if not np.isfinite(lengths).all() or np.any(lengths < 1.e-12):
            raise ValueError('Invalid saved native quaternion norm')
        quaternions = quaternions/lengths[:, None]
        linear = np.linalg.norm(np.diff(positions, axis=0), axis=1)/durations
        angular = _quaternion_angle(quaternions[1:], quaternions[:-1])/durations
        position_range = float(np.linalg.norm(positions[:, None]-positions[None, :], axis=-1).max())
        rotation_range = float(_quaternion_angle(quaternions[:, None], quaternions[None, :]).max())
        result.update(maximum_linear_speed_m_s=float(linear.max()),
            maximum_angular_speed_rad_s=float(angular.max()), position_range_m=position_range,
            orientation_range_rad=rotation_range,
            passed=bool(linear.max() <= limits['maximum_linear_speed_m_s']
                and angular.max() <= limits['maximum_angular_speed_rad_s']
                and position_range <= limits['maximum_stable_position_range_m']
                and rotation_range <= limits['maximum_stable_orientation_range_rad']))
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        result['errors'].append(str(exc))
    return result


def support_evidence(runtime, records, task, *, goal_reset=False, dwell_s=.4):
    """Check actual upward support and quiet final poses, retaining force totals.

    Native contact points are sampled at every integration step; body poses here
    are the saved 20 ms observations. Sleeping contacts may be absent, therefore
    every body must first have a real loaded support contact and remain stable.
    The separate task observer validates its more precise native contact memory.
    """
    metadata = task['evaluation']['metadata']
    limits = task['evaluation'].get('thresholds', {})
    pose_stability = 'stability_pose_window_s' in limits
    bodies = metadata['actors']
    fixtures = metadata['fixtures']
    objects = {row['id']: row for row in metadata['objects']}
    dt = runtime['physics_dt_s']
    steps = [row['observation'] for row in records if row['kind'] == 'step']
    if not steps:
        return {'passed': False, 'reason': 'no_native_steps'}
    end = steps[-1]['time_s']
    terminal = [row for row in steps if row['time_s'] >= end-dwell_s-1e-9]
    goals = metadata['goals']['support_by_actor']
    if goal_reset:
        expected = goals
    elif 'source_support_by_actor' in metadata:
        expected = metadata['source_support_by_actor']
    elif task['id'] == 'breakfast_serving':
        expected = {name: metadata['work_surface_paths'][0] for name in bodies}
    else:
        source = {'produce_transfer': 'source_board', 'utensil_sorting': 'source_pad',
                  'pot_lid_placement': 'source_mat', 'pantry_stocking': 'source_mat',
                  'table_clearing': 'source_mat'}[task['id']]
        expected = {name: source for name in bodies}
    roots = {**fixtures, **bodies}
    aliases = runtime['scene_info'].get('collider_actor_paths', {})
    def matches(path, root):
        # Fixed-room collisions may be promoted to explicit actors by the
        # shared adapter. Compare the recorded mapping, never guessed names.
        if root.startswith('/World/Geometry/'):
            root = '/World/Room/Geometry/'+root.removeprefix('/World/Geometry/')
        root = aliases.get(root, root)
        return path == root or path.startswith(root+'/') or aliases.get(path, path) == root
    states = {name: {'loaded_steps': set(), 'terminal_loaded_steps': set(),
                     'normal_z_impulse_ns': 0., 'support_paths': set(),
                     'robot_contact_count': 0, 'maximum_penetration_m': 0.} for name in bodies}
    coverage = bool(terminal and terminal[-1]['time_s']-terminal[0]['time_s'] >= dwell_s-.02-1e-8)
    previous = None
    previous_time = None
    for obs in steps:
        contacts = obs['contacts']
        sample = contacts['sampling']
        coverage &= contacts['errors'] == [] and sample['complete'] is True
        coverage &= sample['sampled_substeps'] == round(.02/dt)
        coverage &= sample['native_end_step_index']-sample['native_start_step_index'] == round(.02/dt)
        if previous is not None:
            coverage &= sample['native_start_step_index'] == previous
            coverage &= abs(obs['time_s']-previous_time-.02) < 1.e-8
        previous = sample['native_end_step_index']
        previous_time = obs['time_s']
        for row in contacts['rigid']:
            pair = (row['actor0'], row['actor1'])
            for name, body in bodies.items():
                if not any(matches(path, body) for path in pair):
                    continue
                state = states[name]
                state['maximum_penetration_m'] = max(state['maximum_penetration_m'], -row['min_separation_m'])
                which = 0 if matches(pair[0], body) else 1
                other = pair[1-which]
                if other.startswith('/World/Robot/'):
                    state['robot_contact_count'] += 1
                support = roots.get(expected[name], expected[name])
                if matches(other, support):
                    vector = np.asarray(row['impulse_vector_ns'])* (1 if which == 0 else -1)
                    if vector[2] > 1e-10 and row['min_separation_m'] >= -.004:
                        index = round(row['time_s']/dt)
                        state['loaded_steps'].add(index)
                        state['support_paths'].add(other)
                        if row['time_s'] >= end-dwell_s:
                            state['terminal_loaded_steps'].add(index)
                            state['normal_z_impulse_ns'] += float(vector[2])
    results = {}
    for name, state in states.items():
        rows = [obs['objects'][name] for obs in terminal]
        positions = np.asarray([row['position_m'] for row in rows])
        speeds = [float(np.linalg.norm(row['linear_velocity_m_s'])) for row in rows]
        angular = [float(np.linalg.norm(row['angular_velocity_rad_s'])) for row in rows]
        excursion = float(np.max(np.linalg.norm(positions-positions[-1], axis=1)))
        loaded = bool(state['loaded_steps'])
        expected_root = roots.get(expected[name], expected[name])
        terminal_witnesses = []
        for obs in terminal:
            native = obs['task_state']
            actor = native.get('native_support_by_actor', {}).get(name, {})
            valid = (native.get('complete') is True and native.get('errors') == []
                and actor.get('supported') is True
                and any(matches(w['support_actor_path'], expected_root) for w in actor.get('witnesses', [])
                        if w.get('support_actor_path')))
            terminal_witnesses.append(valid)
        expected_children = {bodies[child] for child, parent in expected.items() if parent == name}
        unexpected = []
        for obs in terminal:
            for contact in obs['contacts']['rigid']:
                pair = (contact['actor0'], contact['actor1'])
                if not any(matches(path, bodies[name]) for path in pair):
                    continue
                other = pair[1] if matches(pair[0], bodies[name]) else pair[0]
                if (contact['impulse_magnitude_sum_ns'] > 1e-10
                        and not matches(other, expected_root)
                        and not any(matches(other, child) for child in expected_children)):
                    unexpected.append(other)
        quiet_s = steps[-1]['task_state']['native_support_by_actor'][name].get('released_quiet_native_s', 0.)
        evidence_valid = all(terminal_witnesses) and not unexpected and quiet_s+1e-8 >= dwell_s

        tensor_gate = max(speeds) <= .025 and max(angular) <= .25 and excursion <= .004
        saved_pose = _saved_pose_review(rows, [obs['time_s'] for obs in terminal], limits) if pose_stability else None
        finite_tensor = all(math.isfinite(value) for value in speeds+angular)
        native_quiet = finite_tensor and (saved_pose['passed'] if pose_stability else tensor_gate)
        passed = loaded and evidence_valid and native_quiet and state['maximum_penetration_m'] <= .004 and not state['robot_contact_count']
        results[name] = {'passed': passed, 'declared_support': expected[name],
            'native_loaded_contact_seen': loaded,
            'expected_native_support_through_terminal_window': all(terminal_witnesses),
            'unexpected_loaded_contacts': sorted(set(unexpected)), 'loaded_contact_substeps': len(state['loaded_steps']),
            'terminal_loaded_contact_substeps': len(state['terminal_loaded_steps']),
            'native_support_paths': sorted(state['support_paths']),
            'maximum_penetration_m': state['maximum_penetration_m'],
            'robot_contact_count': state['robot_contact_count'],
            'released_quiet_native_s': quiet_s, 'terminal_position_excursion_m': excursion, 'terminal_maximum_linear_speed_m_s': max(speeds),
            'terminal_maximum_angular_speed_rad_s': max(angular),
            'tensor_velocity_gate_applied': not pose_stability,
            'tensor_velocity_gate_would_pass': bool(tensor_gate and finite_tensor),
            'saved_pose_stability': saved_pose,
            'terminal_upward_normal_impulse_ns': state['normal_z_impulse_ns'],
            'body_weight_impulse_over_window_ns': objects[name]['mass_kg']*9.81*dwell_s,
            'terminal_position_m': positions[-1].tolist()}
    return {'passed': bool(coverage and end >= dwell_s and all(v['passed'] for v in results.values())),
        'complete_native_contact_coverage': bool(coverage), 'native_dt_s': dt, 'duration_s': end,
        'terminal_dwell_s': dwell_s, 'objects': results, 'goal_reset': goal_reset,
        'pose_sample_period_s': .02, 'force_balance_qualified': False,
        'stability_method': 'native_rolling_pose_and_independent_saved_pose' if pose_stability else 'legacy_native_tensor_and_position',
        'scope': 'Loaded upward contact observed with identified support plus native released quiet duration and saved terminal poses; sleeping contact omissions retained, not invented force.'}
