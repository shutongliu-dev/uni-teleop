"""Fail-closed streaming reference score for trusted normalized observations.

This module does not authenticate JSON, normalize native contacts, run physics
or move objects. The caller must establish the trusted-input boundary outside
the observation. Even reference success never grants native qualification.
"""
from __future__ import annotations

from copy import deepcopy
import math

from .rules import OBSERVATION_SCHEMA, number, quaternion, rotation, vector
from .task import _bounds, validate_configuration

VERSION = 'living_tower_building_reference_evaluator_v1'
EPS = 1.e-9


def norm(v):
    return math.sqrt(sum(x*x for x in v))


def distance(a, b):
    return norm([x-y for x, y in zip(a, b)])


def angle(a, b):
    dot = min(1., abs(sum(x*y for x, y in zip(a, b))))
    return 0. if dot > 1.-1.e-14 else 2*math.acos(dot)


def boxes_penetrate(a, b, size, tolerance):
    """Separating-axis check for known rectangular blocks, including their tilt."""
    axes_a = [tuple(a['rotation'][i][j] for i in range(3)) for j in range(3)]
    axes_b = [tuple(b['rotation'][i][j] for i in range(3)) for j in range(3)]
    axes = axes_a+axes_b
    for u in axes_a:
        for v in axes_b:
            cross = (u[1]*v[2]-u[2]*v[1], u[2]*v[0]-u[0]*v[2], u[0]*v[1]-u[1]*v[0])
            length = norm(cross)
            if length > 1.e-10:
                axes.append(tuple(x/length for x in cross))
    delta = [x-y for x, y in zip(a['position_m'], b['position_m'])]
    for axis in axes:
        radius_a = sum(size[j]/2*abs(sum(x*y for x, y in zip(axis, axes_a[j]))) for j in range(3))
        radius_b = sum(size[j]/2*abs(sum(x*y for x, y in zip(axis, axes_b[j]))) for j in range(3))
        if radius_a+radius_b-abs(sum(x*y for x, y in zip(axis, delta))) <= tolerance+EPS:
            return False
    return True


class TowerBuildingEvaluator:
    def __init__(self, configuration, *, trusted_normalized_input=False):
        self.task = validate_configuration(configuration)
        if type(trusted_normalized_input) is not bool:
            raise ValueError('Trusted-input boundary must be an explicit boolean')
        self.trusted = trusted_normalized_input
        self.g, self.rules = self.task['geometry'], self.task['rules']
        self.t = self.rules['tolerances']
        self.ids = tuple(self.task['block_ids'])
        self.slots = {row['slot_id']: row for row in self.task['target_slots']}
        self.previous = self.time = self.index = self.episode = None
        self.action_count = 0
        self.stable_start = self.anchor = self.assignment = None
        self.terminal = False
        self.last_result = None
        self.scoring_mode = self.task.get('scoring_mode')
        if self.scoring_mode == 'freeform_height':
            self.height_windows = {}
            self.best_stable_height_m = 0.
            self.best_stable_height_time_s = None
            self.best_stable_block_ids = []

    def _result(self, status, reason, *, blocks=None, stable_s=0., terminal=False):
        self.terminal = terminal
        self.last_result = {'evaluator': VERSION, 'task_id': 'tower_building',
            'status': status, 'reason': reason, 'terminal': terminal,
            'success': status == 'success', 'reference_success': status == 'success',
            'native_qualified': False, 'stable_duration_s': stable_s,
            'required_stable_duration_s': self.rules['goal']['stable_duration_s'],
            'required_block_count': len(self.ids), 'blocks': deepcopy(blocks or {}),
            'evidence_scope': 'Conditional CPU reference result over trusted normalized interval summaries; no authenticated native adapter or manipulation qualification'}
        return deepcopy(self.last_result)

    def _state(self, row):
        p, q = vector(row['position_m'], 3, 'position'), quaternion(row['quaternion_wxyz'])
        v = vector(row['linear_velocity_m_s'], 3, 'linear velocity')
        w = vector(row['angular_velocity_rad_s'], 3, 'angular velocity')
        vmax, wmax = number(row['interval_max_linear_speed_m_s'], 'interval linear speed'), number(row['interval_max_angular_speed_rad_s'], 'interval angular speed')
        if vmax < norm(v)-EPS or wmax < norm(w)-EPS or min(vmax, wmax) < 0:
            raise ValueError('Interval speed maxima must bound the current finite velocities')
        return {'position_m': p, 'quaternion_wxyz': q, 'rotation': rotation(q),
                'vmax': vmax, 'wmax': wmax}

    def _sampling(self, obs):
        if not self.trusted:
            raise ValueError('Caller has not established trusted normalized physical input')
        if obs.get('schema') != OBSERVATION_SCHEMA or obs.get('configuration_sha256') != self.task['configuration_sha256']:
            raise ValueError('Observation schema/configuration identity mismatch')
        if any(obs.get(key) is not True for key in ('sampling_complete', 'contacts_complete', 'physics_valid', 'robot_only_actions')) or obs.get('object_pose_writes') is not False:
            raise ValueError('Complete valid physical sampling and robot-only state evolution are required')
        episode = obs['episode_id']
        if not isinstance(episode, str) or not episode:
            raise ValueError('A nonempty episode identity is required')
        index, actions = obs['sample_index'], obs['robot_action_count']
        if type(index) is not int or index < 0 or type(actions) is not int or actions < self.action_count:
            raise ValueError('Invalid sample/action counters')
        if type(obs['budget_exhausted']) is not bool:
            raise ValueError('budget_exhausted must be boolean')
        time = number(obs['time_s'], 'time')
        start = number(obs['interval_start_s'], 'interval start')
        if self.previous is None:
            if index != 0 or actions != 0 or abs(time) > EPS or abs(start) > EPS:
                raise ValueError('First observation must be the unscored reset baseline')
        elif (episode != self.episode or index != self.index+1 or abs(start-self.time) > EPS
              or abs(time-self.time-self.t['sample_period_s']) > EPS):
            raise ValueError('Observation intervals must be consecutive, fixed-period and episode-bound')
        return time, index, actions, episode

    def _contacts(self, obs):
        rows = obs['contacts']
        if not isinstance(rows, list):
            raise ValueError('Contacts must be a complete normalized list')
        contacts, seen = {name: [] for name in self.ids}, set()
        for row in rows:
            a, b = row['a'], row['b']
            def known(name):
                return isinstance(name, str) and (name in self.ids or name in ('table', 'floor')
                    or (name.startswith(('robot/', 'external/')) and name.rsplit('/', 1)[-1] != ''))
            if not known(a) or not known(b) or a == b or not ({a, b} & set(self.ids)):
                raise ValueError('Contacts must name distinct known actors and at least one task block')
            point = vector(row['point_m'], 3, 'contact point')
            key = (*sorted((a, b)), point)
            if key in seen:
                raise ValueError('Duplicate normalized contact')
            seen.add(key)
            separation = number(row['separation_m'], 'separation')
            force = number(row['minimum_upward_force_n'], 'minimum upward force')
            supported = row['support_for']
            if type(row['continuous']) is not bool or force < 0 or supported not in (None, a, b):
                raise ValueError('Malformed directed interval support evidence')
            if (supported is None and force != 0.) or (supported is not None and supported not in self.ids):
                raise ValueError('Support can only be directed onto a named task block')
            for name, other in ((a, b), (b, a)):
                if name in contacts:
                    contacts[name].append({'other': other, 'point': point, 'separation': separation,
                        'force': force if supported == name else 0., 'continuous': row['continuous']})
        return contacts

    def _slot(self, state):
        r = state['rotation']
        tilt = math.acos(max(-1., min(1., abs(r[2][2]))))
        yaw = math.atan2(r[1][0], r[0][0])
        matches = []
        for name, slot in self.slots.items():
            yaw_error = abs(math.atan2(math.sin(2*(yaw-slot['yaw_rad'])), math.cos(2*(yaw-slot['yaw_rad']))))/2
            if (distance(state['position_m'][:2], slot['position_m'][:2]) <= self.t['position_xy_m']+EPS
                    and abs(state['position_m'][2]-slot['position_m'][2]) <= self.t['position_z_m']+EPS
                    and yaw_error <= self.t['yaw_rad']+EPS and tilt <= self.t['tilt_rad']+EPS):
                matches.append(name)
        return matches[0] if len(matches) == 1 else None

    def _point_inside(self, state, point):
        delta = [p-c for p, c in zip(point, state['position_m'])]
        local = [sum(state['rotation'][i][j]*delta[i] for i in range(3)) for j in range(3)]
        return all(abs(value) <= size/2+self.t['contact_point_m']+EPS for value, size in zip(local, self.g['block_size_m']))

    def _support(self, name, row, states, assignments, supported):
        if (not row['continuous'] or row['force'] <= self.t['minimum_support_force_n']
                or not -self.t['penetration_m'] <= row['separation'] <= self.t['support_gap_m']):
            return False
        state, point = states[name], row['point']
        bottom = _bounds(state, self.g['block_size_m'])[0][2]
        if not self._point_inside(state, point) or abs(point[2]-bottom) > self.t['support_gap_m']+self.t['contact_point_m']:
            return False
        layer, other = self.slots[assignments[name]]['layer'], row['other']
        if other == 'table':
            lo, hi = self.g['table_bounds_xy_m']
            return (layer == 0 and abs(bottom-self.g['table_top_z_m']) <= self.t['support_gap_m']+EPS
                    and abs(point[2]-self.g['table_top_z_m']) <= self.t['contact_point_m']+EPS
                    and all(lo[i] <= point[i] <= hi[i] for i in (0, 1)))
        if other not in self.ids or not assignments[other] or not supported.get(other, False):
            return False
        if self.slots[assignments[other]]['layer'] != layer-1:
            return False
        top = _bounds(states[other], self.g['block_size_m'])[1][2]
        return (abs(bottom-top) <= self.t['support_gap_m']+EPS
                and abs(point[2]-top) <= self.t['support_gap_m']+self.t['contact_point_m']
                and self._point_inside(states[other], point))

    def update(self, observation):
        if self.scoring_mode == 'freeform_height':
            from .freeform import update_height
            return update_height(self, observation)
        if self.terminal:
            return deepcopy(self.last_result)
        try:
            time, index, actions, episode = self._sampling(observation)
            if not isinstance(observation['objects'], dict) or set(observation['objects']) != set(self.ids):
                raise ValueError('Every physical block must be observed exactly once with no extras')
            states = {name: self._state(observation['objects'][name]) for name in self.ids}
            contacts = self._contacts(observation)
            previous_states = self.previous
            first = previous_states is None
            if first:
                for name, state in states.items():
                    initial = self.task['initial_objects'][name]
                    if (distance(state['position_m'], initial['position_m']) > self.t['initial_position_m']
                            or angle(state['quaternion_wxyz'], initial['quaternion_wxyz']) > self.t['initial_angle_rad']):
                        raise ValueError('Reset does not match the separated frozen supply plan')
            else:
                dt = time-self.time
                for name, state in states.items():
                    previous = self.previous[name]
                    if (distance(state['position_m'], previous['position_m'])/dt > state['vmax']+1.e-6
                            or angle(state['quaternion_wxyz'], previous['quaternion_wxyz'])/dt > state['wmax']+1.e-6):
                        raise ValueError('Interval speed bound contradicts observed motion')
            self.previous, self.time, self.index, self.action_count, self.episode = states, time, index, actions, episode
            budget = self.task['budget']
            exhausted = observation['budget_exhausted'] or (budget['max_sim_time_s'] is not None and time >= budget['max_sim_time_s']) or (budget['max_robot_actions'] is not None and actions >= budget['max_robot_actions'])
            if exhausted:
                return self._result('incomplete', 'budget_exhausted', terminal=True)
            if first:
                return self._result('incomplete', 'reset_baseline_not_scored')
            floor = any(row['other'] == 'floor' for rows in contacts.values() for row in rows)
            if floor and self.rules['floor_drop_is_terminal']:
                return self._result('failure', 'block_touched_floor', terminal=True)
            assignments = {name: self._slot(state) for name, state in states.items()}
            blocks = {name: {'slot_id': slot, 'reasons': []} for name, slot in assignments.items()}
            for i, name in enumerate(self.ids):
                for other in self.ids[i+1:]:
                    if boxes_penetrate(states[name], states[other], self.g['block_size_m'], self.t['penetration_m']):
                        for actor in (name, other):
                            if 'geometric_block_penetration' not in blocks[actor]['reasons']:
                                blocks[actor]['reasons'].append('geometric_block_penetration')
            supported = {}
            ordered = sorted(self.ids, key=lambda name: self.slots[assignments[name]]['layer'] if assignments[name] else -1)
            for name in ordered:
                slot, state, result = assignments[name], states[name], blocks[name]
                if slot is None:
                    result['reasons'].append('not_in_one_legal_slot')
                elif list(assignments.values()).count(slot) != 1:
                    result['reasons'].append('slot_has_multiple_blocks')
                forbidden = any(row['other'] == 'floor' or row['other'].startswith(('robot/', 'external/')) for row in contacts[name])
                if forbidden:
                    result['reasons'].append('robot_or_external_contact')
                if any(row['separation'] < -self.t['penetration_m']-EPS for row in contacts[name]):
                    result['reasons'].append('excessive_contact_penetration')
                supported[name] = bool(slot and not result['reasons'] and any(
                    self._support(name, row, states, assignments, supported) for row in contacts[name]))
                result['supported_to_table'] = supported[name]
                if not supported[name]:
                    result['reasons'].append('continuous_contact_support_chain_required')
                if state['vmax'] > self.t['linear_speed_m_s'] or state['wmax'] > self.t['angular_speed_rad_s']:
                    result['reasons'].append('still_moving')
            good = all(not row['reasons'] for row in blocks.values())
            if not good:
                self.stable_start = self.anchor = self.assignment = None
            else:
                drift = self.anchor is not None and any(
                    max(distance(frame[name]['position_m'], self.anchor[name]['position_m']) for frame in (previous_states, states))
                    + states[name]['vmax']*dt/2 > self.t['stable_position_range_m']+EPS
                    or max(angle(frame[name]['quaternion_wxyz'], self.anchor[name]['quaternion_wxyz']) for frame in (previous_states, states))
                    + states[name]['wmax']*dt/2 > self.t['stable_angle_range_rad']+EPS
                    for name in self.ids)
                if self.anchor is None or assignments != self.assignment or drift:
                    self.stable_start, self.anchor, self.assignment = time, deepcopy(states), assignments
            stable = 0. if self.stable_start is None else time-self.stable_start
            if good and stable+EPS >= self.rules['goal']['stable_duration_s']:
                return self._result('success', 'all_slots_released_supported_and_stable', blocks=blocks, stable_s=stable, terminal=True)
            return self._result('incomplete', 'settling' if good else 'build_or_recover_tower',
                                blocks=blocks, stable_s=stable)
        except (KeyError, TypeError, ValueError, IndexError, OverflowError, AttributeError) as exc:
            self.stable_start = self.anchor = self.assignment = None
            return self._result('invalid', str(exc), terminal=True)

    def inspect_conditions(self):
        if self.scoring_mode == 'freeform_height' and self.last_result is None:
            from .freeform import _result
            return _result(self, 'not_sampled')
        return deepcopy(self.last_result or {'success': False, 'status': 'incomplete', 'reason': 'not_sampled', 'native_qualified': False})


def evaluate_tower(observations, configuration, *, trusted_normalized_input=False):
    """Replay a complete ordered sequence; never infer a dwell from one frame."""
    evaluator = TowerBuildingEvaluator(configuration, trusted_normalized_input=trusted_normalized_input)
    for observation in observations:
        result = evaluator.update(observation)
        if result['terminal']:
            return result
    return evaluator.inspect_conditions()
