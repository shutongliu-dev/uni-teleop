"""Pure streaming study referee over trusted complete native substep readback.

This module neither authenticates JSON nor runs physics. Its caller must bind
physical actor paths to frozen IDs and supply every actual native state/contact
sample. A result never grants general manipulation or simulator qualification.
"""
from copy import deepcopy
from itertools import product
import math

from .rules import (OBSERVATION_SCHEMA, THRESHOLDS, number, vector, quaternion,
                    rotation, validate_configuration)

VERSION = 'study_evaluator_v1'
EPS = 1.e-9


def norm(v):
    return math.sqrt(sum(x*x for x in v))


def distance(a, b):
    return norm([x-y for x, y in zip(a, b)])


def angle(a, b):
    dot = min(1., abs(sum(x*y for x, y in zip(a, b))))
    return 0. if dot > 1.-1.e-14 else 2*math.acos(dot)


def world_bounds(state, local_bounds):
    points = [tuple(sum(state['rotation'][i][j]*p[j] for j in range(3))+
                    state['position_m'][i] for i in range(3))
              for p in product(*zip(*local_bounds))]
    return tuple(min(p[i] for p in points) for i in range(3)), tuple(max(p[i] for p in points) for i in range(3))


class SequenceEvaluator:
    def __init__(self, task_spec, *, trusted_normalized_input=False):
        if type(trusted_normalized_input) is not bool:
            raise ValueError('The normalized-input trust boundary must be explicit')
        if 'evaluation' in task_spec:
            if task_spec.get('engineering_probe'):
                raise ValueError('Engineering probes cannot receive task credit')
            self.config = validate_configuration(task_spec['evaluation']['configuration'])
            if task_spec.get('id') != self.config['task_id'] or task_spec.get('public_rules') != self.config['rules']:
                raise ValueError('Task identity/public rules differ from its frozen configuration')
        else:
            self.config = validate_configuration(task_spec)
        self.trusted = trusted_normalized_input
        self.ident = self.config['task_id']
        self.ids = tuple(self.config['objects'])
        self.t = THRESHOLDS
        self.dt = self.config['physics_dt_s']
        self.previous = self.time = self.index = self.episode = None
        self.actions = 0
        self.anchor = self.stable_since = None
        self.terminal = False
        self.last_result = None
        self.status = {}
        initial = {ident: self._state({**obj['initial_pose'], 'linear_velocity_m_s': [0.,0.,0.],
                    'angular_velocity_rad_s': [0.,0.,0.]}) for ident,obj in self.config['objects'].items()}
        if all(not self._geometry(ident, state) for ident,state in initial.items()):
            raise ValueError('A fully solved reset cannot be credited as a manipulation task')

    @staticmethod
    def _state(row):
        p, q = vector(row['position_m'], 3, 'position'), quaternion(row['quaternion_wxyz'])
        return {'position_m': p, 'quaternion_wxyz': q, 'rotation': rotation(q),
                'linear_velocity_m_s': vector(row['linear_velocity_m_s'], 3, 'linear velocity'),
                'angular_velocity_rad_s': vector(row['angular_velocity_rad_s'], 3, 'angular velocity')}

    def _result(self, status, reason, *, detail=None, stable_s=0., terminal=False):
        self.terminal = terminal
        state = {'success': 'success', 'failure': 'task_failure', 'invalid': 'evaluator_invalid'}.get(status, 'ongoing')
        if state == 'success': phase = 'complete'
        elif state == 'task_failure': phase = 'failed'
        elif state == 'evaluator_invalid': phase = 'invalid'
        elif reason == 'not_sampled': phase = 'not_started'
        elif reason == 'reset_baseline_not_scored': phase = 'reset'
        elif terminal: phase = 'stopped'
        elif any('robot_contact_present' in s['reasons'] for s in self.status.values()): phase = 'release'
        elif any(not s['geometry'] for s in self.status.values()): phase = 'placement'
        elif any(not s['supported'] for s in self.status.values()): phase = 'support'
        else: phase = 'settling'
        reasons = [reason] if reason != 'stability_window_incomplete' else list(dict.fromkeys(
            code for row in self.status.values() for code in row['reasons'])) or [reason]
        ready = sum(not s['reasons'] for s in self.status.values())
        self.last_result = {'evaluator': VERSION, 'task_id': self.ident, 'status': status,
            'reason': reason, 'success': status == 'success', 'terminal': terminal,
            'progress': 1. if status == 'success' else 0. if reason == 'reset_baseline_not_scored' else .9*ready/len(self.ids),
            'stable_duration_s': stable_s, 'required_stable_duration_s': self.t['minimum_stable_s'],
            'objects': deepcopy(self.status), 'native_manipulation_qualified': False,
            'referee': {'schema': 'physical_task_referee_state.v1', 'state': state, 'phase': phase,
                'reason_codes': reasons, 'terminal': state != 'ongoing', 'recoverable': state == 'ongoing'},
            'evidence_scope': 'Every native 0.5 ms rigid pose/velocity and signed normal-contact sample; conditional on the service-owned actor mapping and complete native readback, not continuous between-substep trajectories or quantitative force calibration.'}
        maximum = self.config['budget']['max_robot_actions']
        if maximum is not None:
            self.last_result['action_budget'] = {'maximum': maximum, 'admitted': self.actions}
        if detail is not None:
            self.last_result['private_evidence_error'] = detail
        return deepcopy(self.last_result)

    def inspect_conditions(self, observation=None):
        return deepcopy(self.last_result) if self.last_result is not None else self._result('active', 'not_sampled')

    def finish(self, reason='budget_exhausted'):
        if self.terminal:
            return deepcopy(self.last_result)
        self.anchor = self.stable_since = None
        if reason == 'evidence_missing':
            return self._result('invalid', 'invalid_evidence', detail=reason, terminal=True)
        if reason not in ('budget_exhausted', 'cancelled'):
            raise ValueError('Unknown study completion reason')
        return self._result('incomplete', reason, terminal=True)

    def _sampling(self, obs):
        if not self.trusted:
            raise ValueError('Caller has not established the trusted native input boundary')
        if obs.get('schema') != OBSERVATION_SCHEMA or obs.get('configuration_sha256') != self.config['configuration_sha256']:
            raise ValueError('Observation schema or frozen configuration differs')
        if (any(obs.get(k) is not True for k in ('sampling_complete', 'contacts_complete', 'physics_valid', 'robot_only_actions'))
                or obs.get('object_pose_writes') is not False):
            raise ValueError('Complete valid native state and robot-only evolution are required')
        episode, index, actions = obs['episode_id'], obs['step_index'], obs['robot_action_count']
        if (not isinstance(episode, str) or not episode or type(index) is not int or index < 0
                or type(actions) is not int or actions < self.actions or type(obs['budget_exhausted']) is not bool):
            raise ValueError('Invalid episode, native step, action counter or budget state')
        time = number(obs['time_s'], 'native time')
        if number(obs['physics_dt_s'], 'native period') != self.dt:
            raise ValueError('Native period differs from frozen rules')
        if self.previous is None:
            if index != 0 or actions != 0 or abs(time) > EPS:
                raise ValueError('First native observation must be the unscored reset baseline')
        elif episode != self.episode or index != self.index+1 or abs(time-self.time-self.dt) > EPS:
            raise ValueError('Native samples must be consecutive with a fixed episode and clock')
        return time, index, actions, episode

    def _geometry(self, ident, state):
        obj, goal = self.config['objects'][ident], self.config['targets'][ident]
        lo, hi = world_bounds(state, obj['local_bounds_m'])
        minimum = list(goal['bounds_m'][0])
        # Native float32 resting poses can put the envelope just below the
        # support plane. Reuse the frozen contact tolerance without adding it
        # again to an authored lower margin; XY and all upper bounds stay exact.
        minimum[2] = min(minimum[2], goal['support_z_m']-self.t['maximum_penetration_m'])
        reasons = []
        if any(lo[i] < minimum[i]-EPS or hi[i] > goal['bounds_m'][1][i]+EPS for i in range(3)):
            reasons.append('outside_target_slot')
        for axis in ('up', 'spine'):
            if axis+'_axis_local' not in obj:
                continue
            local, target = obj[axis+'_axis_local'], goal[axis+'_direction_world']
            world = [sum(state['rotation'][i][j]*local[j] for j in range(3)) for i in range(3)]
            cosine = max(-1., min(1., sum(x*y for x,y in zip(world, target))))
            if math.acos(cosine) > self.t['maximum_'+axis+'_axis_error_rad']+EPS:
                if 'incorrect_orientation' not in reasons: reasons.append('incorrect_orientation')
        if abs(lo[2]-goal['support_z_m']) > self.t['support_gap_m']+EPS:
            reasons.append('support_height_mismatch')
        return reasons

    def _contacts(self, obs):
        if not isinstance(obs['contacts'], list):
            raise ValueError('A complete native contact list is required')
        contacts = {name: [] for name in self.ids}
        def known(name):
            return isinstance(name, str) and (name in self.ids or name in self.config['supports'] or name == 'floor'
                or name.startswith(('robot/', 'external/')) and name.rsplit('/', 1)[-1] != '')
        for row in obs['contacts']:
            a, b = row['a'], row['b']
            if not known(a) or not known(b) or a == b or not ({a,b} & set(self.ids)):
                raise ValueError('Contact names must map distinct native actors including one task object')
            point = vector(row['point_m'], 3, 'contact point')
            impulse = vector(row['impulse_on_a_n_s'], 3, 'signed normal impulse')
            separation = number(row['separation_m'], 'native separation')
            for ident, other, sign in ((a,b,1.), (b,a,-1.)):
                if ident in contacts:
                    contacts[ident].append({'other': other, 'point': point, 'separation': separation,
                        'upward_impulse': sign*impulse[2]})
        return contacts

    def _inside_object(self, state, ident, point):
        delta = [a-b for a,b in zip(point, state['position_m'])]
        local = [sum(state['rotation'][i][j]*delta[i] for i in range(3)) for j in range(3)]
        lo, hi = self.config['objects'][ident]['local_bounds_m']
        tol = self.t['contact_point_tolerance_m']
        return all(lo[i]-tol-EPS <= local[i] <= hi[i]+tol+EPS for i in range(3))

    def _support(self, ident, contacts, states, supported):
        goal, state = self.config['targets'][ident], states[ident]
        support_id = goal['support_id']
        bottom = world_bounds(state, self.config['objects'][ident]['local_bounds_m'])[0][2]
        if support_id in states:
            if not supported.get(support_id, False):
                return False
            slo, shi = world_bounds(states[support_id], self.config['objects'][support_id]['local_bounds_m'])
        else:
            slo, shi = self.config['supports'][support_id]['bounds_m']
        if abs(bottom-shi[2]) > self.t['support_gap_m']+EPS:
            return False
        total = 0.
        tol = self.t['contact_point_tolerance_m']
        for row in contacts:
            p = row['point']
            if (row['other'] != support_id or not -self.t['maximum_penetration_m']-EPS <= row['separation'] <= self.t['support_gap_m']+EPS
                    or not self._inside_object(state, ident, p) or abs(p[2]-bottom) > tol+self.t['support_gap_m']+EPS
                    or abs(p[2]-shi[2]) > tol+self.t['support_gap_m']+EPS
                    or any(not slo[i]-tol-EPS <= p[i] <= shi[i]+tol+EPS for i in (0,1))):
                continue
            if support_id in states and not self._inside_object(states[support_id], support_id, p):
                continue
            total += row['upward_impulse']
        return total > self.t['minimum_support_impulse_ns']

    def update(self, observation):
        if self.terminal:
            return deepcopy(self.last_result)
        try:
            time, index, actions, episode = self._sampling(observation)
            if not isinstance(observation['objects'], dict) or set(observation['objects']) != set(self.ids):
                raise ValueError('Every frozen physical object must appear exactly once')
            states = {name: self._state(observation['objects'][name]) for name in self.ids}
            contacts = self._contacts(observation)
            first, previous = self.previous is None, self.previous
            if first:
                if observation['contacts']:
                    raise ValueError('Reset baseline has no sampled contact interval')
                for ident,state in states.items():
                    initial = self.config['objects'][ident]['initial_pose']
                    if (distance(state['position_m'], initial['position_m']) > self.t['initial_position_tolerance_m']
                            or angle(state['quaternion_wxyz'], quaternion(initial['quaternion_wxyz'])) > self.t['initial_orientation_tolerance_rad']):
                        raise ValueError('Native reset differs from the frozen object plan')
            self.previous, self.time, self.index, self.episode, self.actions = states, time, index, episode, actions
            budget = self.config['budget']
            if (observation['budget_exhausted'] or budget['max_sim_time_s'] is not None and time+EPS >= budget['max_sim_time_s']
                    or budget['max_robot_actions'] is not None and actions > budget['max_robot_actions']):
                return self.finish('budget_exhausted')
            if first:
                return self._result('active', 'reset_baseline_not_scored')
            if any(row['other'] == 'floor' and row['separation'] <= self.t['support_gap_m']+EPS
                   for rows in contacts.values() for row in rows):
                return self._result('failure', 'object_touched_floor', terminal=True)
            supported, status = {}, {}
            order = self.config.get('stack_order', self.ids)
            for ident in order:
                state, rows = states[ident], contacts[ident]
                reasons = self._geometry(ident, state)
                geometry = not reasons
                support_id = self.config['targets'][ident]['support_id']
                allowed = {support_id} | {name for name,g in self.config['targets'].items() if g['support_id'] == ident}
                released = not any(r['other'].startswith('robot/') for r in rows)
                if not released: reasons.append('robot_contact_present')
                if any(r['other'] not in allowed and not r['other'].startswith('robot/') for r in rows):
                    reasons.append('unexpected_contact')
                if any(r['separation'] < -self.t['maximum_penetration_m']-EPS for r in rows):
                    reasons.append('contact_penetration')
                supported[ident] = self._support(ident, rows, states, supported)
                if not supported[ident]: reasons.append('required_contact_support_missing')
                linear = max(norm(state['linear_velocity_m_s']), distance(state['position_m'], previous[ident]['position_m'])/self.dt)
                angular = max(norm(state['angular_velocity_rad_s']), angle(state['quaternion_wxyz'], previous[ident]['quaternion_wxyz'])/self.dt)
                quiet = linear <= self.t['maximum_linear_speed_m_s']+EPS and angular <= self.t['maximum_angular_speed_rad_s']+EPS
                if not quiet: reasons.append('still_moving')
                status[ident] = {'slot_id': self.config['targets'][ident]['slot_id'], 'geometry': geometry,
                    'supported': supported[ident], 'released': released, 'quiet': quiet,
                    'reasons': reasons, 'bounds_m': world_bounds(state, self.config['objects'][ident]['local_bounds_m'])}
            self.status = status
            good = all(not row['reasons'] for row in status.values()) and actions > 0
            if not good:
                self.stable_since = self.anchor = None
            elif self.anchor is None:
                self.stable_since, self.anchor = time, deepcopy(states)
            elif any(distance(state['position_m'], self.anchor[name]['position_m']) > self.t['maximum_position_drift_m']+EPS
                    or angle(state['quaternion_wxyz'], self.anchor[name]['quaternion_wxyz']) > self.t['maximum_orientation_drift_rad']+EPS
                    for name,state in states.items()):
                self.stable_since, self.anchor = time, deepcopy(states)
                for row in status.values(): row['reasons'].append('stability_drift')
            stable = 0. if self.stable_since is None else time-self.stable_since
            if good and stable+EPS >= self.t['minimum_stable_s']:
                return self._result('success', 'success', stable_s=stable, terminal=True)
            return self._result('active', 'robot_action_required' if actions == 0 else 'stability_window_incomplete', stable_s=stable)
        except (KeyError, ValueError, TypeError, IndexError, OverflowError, AttributeError) as exc:
            self.stable_since = self.anchor = None
            return self._result('invalid', 'invalid_evidence', detail=str(exc), terminal=True)
