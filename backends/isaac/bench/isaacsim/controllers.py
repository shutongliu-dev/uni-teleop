"""Pure NumPy/URDF diagnostic targets; no simulator state writes or Newton import."""
from __future__ import annotations

import numpy as np

from bench.geometry import transform
from bench.control.kinematics import URDFKinematics


def smooth(value):
    return value ** 3 * (10 + value * (-15 + 6 * value))


class TipPlanner:
    def __init__(self, scene):
        from pathlib import Path
        assets = Path(scene['robot']['assets_root'])
        self.arm = URDFKinematics(assets / 'robots/franka/fr3/fr3_no_hand.absolute.urdf')
        self.hand = URDFKinematics(assets / 'robots/sharpa/wave_01/derived/right_sharpa_wave_with_flange.urdf')
        self.hand_q = {joint.name: float(np.clip(1.2 if 'index' not in joint.name and 'thumb' not in joint.name
                        and joint.name.endswith(('_FE', '_PIP', '_DIP', '_IP')) else 0., joint.lower, joint.upper))
                       for joint in self.hand.active}
        self.offset = self.hand.forward(self.hand_q)['right_index_fingertip'][:3, 3]
        self.rotation = np.diag([1., -1., -1.])
        self.base = np.asarray(scene['robot']['base_position_m'])
        self.previous = dict(zip(self.arm.names, scene['robot']['arm_home_rad']))
        self.errors = []

    def solve(self, tip):
        target = transform(np.asarray(tip) - self.base - self.rotation @ self.offset, self.rotation)
        result = self.arm.inverse(target, 'fr3_link8', self.previous, iterations=120, multi_starts=8,
                                  position_tolerance=.00003, rotation_tolerance=.0004)
        if not result['converged']:
            raise ValueError(f'Unreachable diagnostic fingertip target: {tip}: {result}')
        self.previous = result['joint_positions_rad']
        self.errors.append({'position_error_m': result['position_error_m'], 'rotation_error_rad': result['rotation_error_rad']})
        return {**self.previous, **self.hand_q}


def push_plan(scene, *, mode='push', tip_z=.778, distance=.10):
    if mode not in ('push', 'hover') or not 0 < distance <= .12:
        raise ValueError('Use push/hover and a positive push distance <= 0.12 m')
    planner = TipPlanner(scene)
    source = np.asarray(next(obj['position_m'] for obj in scene['objects'] if obj['id'] == 'spool'))
    start = np.array([source[0], source[1] - .07, .835])
    low = start.copy(); low[2] = tip_z
    touch = low.copy(); touch[1] = source[1] - .041
    end = touch.copy(); end[1] += distance
    up = end.copy(); up[2] = .865
    away = up.copy(); away[1] -= .14
    if mode == 'hover':
        for point in (start, low, touch, end, up, away):
            point[2] += .08
    phases = [('settle', .6, start, start), ('lower', 1.5, start, low), ('approach', 1.5, low, touch),
              ('push' if mode == 'push' else 'hover', 3.5, touch, end), ('hold', .5, end, end),
              ('lift_clear', 1., end, up), ('withdraw', 1.5, up, away), ('settle_released', 1.5, away, away)]
    initial = planner.solve(start)
    actions, trace = [], []
    for phase, duration, a, b in phases:
        count = round(duration / .02)
        for index in range(count):
            point = a + (b - a) * smooth((index + 1) / count)
            actions.append({'joint_positions_rad': planner.solve(point), 'duration_s': .02})
            trace.append({'phase': phase, 'tip_target_world_m': point.tolist()})
    return initial, actions, trace, {'maximum_position_error_m': max(x['position_error_m'] for x in planner.errors),
                                    'maximum_rotation_error_rad': max(x['rotation_error_rad'] for x in planner.errors),
                                    'collision_free_claimed': False}


class ContactTargets:
    """Joint error integral plus slew limits, expressed entirely as drive targets."""
    def __init__(self, initial, limits, integral_gain=4.):
        self.previous = dict(initial)
        self.velocity = {name: 0. for name in initial}
        self.offset = {name: 0. for name in initial}
        self.limits = limits
        self.integral_gain = integral_gain

    def action(self, planned, observation):
        command = {}
        for name, target in planned['joint_positions_rad'].items():
            if name.startswith('fr3_'):
                error = target - observation['joint_positions_rad'][name]
                self.offset[name] = float(np.clip(self.offset[name] + self.integral_gain * error * .02, -.10, .10))
            low, high = self.limits[name]
            target = float(np.clip(target + self.offset[name], low, high))
            requested = float(np.clip((target - self.previous[name]) / .02, -.65, .65))
            velocity = float(np.clip(requested, self.velocity[name] - .04, self.velocity[name] + .04))
            command[name] = float(np.clip(self.previous[name] + velocity * .02, low, high))
            self.velocity[name] = (command[name] - self.previous[name]) / .02
        self.previous = command
        return {'joint_positions_rad': command, 'duration_s': .02}, {'integral_offset_rad': dict(self.offset),
                    'max_target_velocity_rad_s': max(abs(x) for x in self.velocity.values())}


def contact_sample(observation, object_id='spool'):
    target = '/World/Objects/' + object_id
    impulses, points, support_penetration, self_penetration = 0., 0, 0., 0.
    pairs = []
    for report in observation['contacts'].get('rigid', []):
        actors = [str(report.get(f'actor{i}', '')) for i in (0, 1)]
        colliders = [str(report.get(f'collider{i}', '')) for i in (0, 1)]
        robot = ['/Robot/' in actor or '/Robot/' in collider for actor, collider in zip(actors, colliders)]
        if all(robot) and report.get('min_separation_m') is not None:
            self_penetration = max(self_penetration, -float(report['min_separation_m']), 0.)
        hand = [('/Robot/' in actor or '/Robot/' in collider) and ('/right_' in actor or '/right_' in collider)
                for actor, collider in zip(actors, colliders)]
        at_object = [actor == target or actor.startswith(target + '/') or collider == target or collider.startswith(target + '/')
                     for actor, collider in zip(actors, colliders)]
        if any(hand[i] and at_object[1 - i] for i in (0, 1)):
            impulses += float(report['impulse_magnitude_sum_ns'])
            points += int(report['point_count'])
            pairs.append(report)
        for index in (0, 1):
            other = actors[1 - index] + ' ' + colliders[1 - index]
            if hand[index] and '/Robot/' not in other and '/Objects/' not in other:
                separation = report.get('min_separation_m')
                if separation is not None:
                    support_penetration = max(support_penetration, -float(separation), 0.)
    obj = observation['objects'][object_id]
    fingertip = observation.get('robot', {}).get('fingertips', {}).get('right_index_fingertip', {})
    return {'time_s': observation['time_s'], 'index_fingertip_position_m': fingertip.get('position_m'),
            'object_position_m': obj['position_m'],
            'object_speed_m_s': float(np.linalg.norm(obj['linear_velocity_m_s'])),
            'hand_object_impulse_ns': impulses, 'hand_object_contact_points': points,
            'hand_object_pairs': pairs, 'hand_support_penetration_m': support_penetration,
            'robot_self_penetration_m': self_penetration}


def contact_result(samples, mode):
    p = np.asarray([x['object_position_m'] for x in samples]); t = np.asarray([x['time_s'] for x in samples])
    impulses = np.asarray([x['hand_object_impulse_ns'] for x in samples])
    points = np.asarray([x['hand_object_contact_points'] for x in samples])
    speed = np.asarray([x['object_speed_m_s'] for x in samples])
    xy = np.linalg.norm(p[:, :2] - p[0, :2], axis=1)
    solved = (impulses > 1.e-8) & (points > 0)
    indices = np.flatnonzero(solved)
    prior_drift = float(xy[:indices[0]].max()) if len(indices) and indices[0] else None
    tail = t >= t[-1] - 1.
    pen = max(x['hand_support_penetration_m'] for x in samples)
    self_pen = max(x['robot_self_penetration_m'] for x in samples)
    checks = {'hand_spool_solver_contact': bool(solved.any()), 'final_xy_displacement_at_least_5cm': bool(xy[-1] >= .05),
              'precontact_xy_drift_below_1cm': prior_drift is not None and prior_drift < .01,
              'final_1s_released': not bool((points[tail] > 0).any()),
              'final_1s_speed_below_1cm_s': bool(speed[tail].max() < .01),
              'hand_support_penetration_below_2mm': pen <= .002,
              'reported_self_penetration_below_2mm': self_pen <= .002,
              'final_height_near_initial': bool(abs(p[-1, 2] - p[0, 2]) < .01)}
    return {'mode': mode, 'primitive': 'index finger contact push and withdrawal', 'object': 'spool',
            'final_xy_displacement_m': float(xy[-1]), 'object_delta_xyz_m': (p[-1] - p[0]).tolist(),
            'hand_spool_contact_frames': int(solved.sum()), 'total_hand_spool_impulse_ns': float(impulses.sum()),
            'first_solved_contact_s': float(t[indices[0]]) if len(indices) else None,
            'precontact_max_xy_displacement_m': prior_drift, 'final_1s_max_speed_m_s': float(speed[tail].max()),
            'max_hand_support_penetration_m': pen, 'max_reported_self_penetration_m': self_pen, 'checks': checks,
            'provisional_contact_push_response': mode == 'push' and all(checks.values()),
            'matched_hover_control_required': True, 'independent_geometry_audit_required': True,
            'grasp_claimed': False, 'sequential_task_success_claimed': False,
            'solver_contact_is_hand_spool_pair_only': True}
