"""Fail-closed canonical scoring for L1 sorting, L2 pegs and L3 shaft parts.

Frozen releases copy this module as evaluator/sequence_evaluator.py alongside
rules.py, referee.py and native_evidence.py. No runtime asset loading, controller dependency or state writes occur.
"""
from copy import deepcopy
import math
import numpy as np

from .rules import TASK_OBJECTS, public_rules, thresholds
from .referee import verdict

VERSION = 'mechanical_assembly_evaluator_v2'
from .native_evidence import (EPS, number, vector, rotation, angle, within,
                              read_native_contact_window)


class _SafetyStop(ValueError):
    def __init__(self, message, code):
        super().__init__(message)
        self.code = code


class SequenceEvaluator:
    def __init__(self, task_spec):
        self.task = deepcopy(task_spec)
        self.task_id = self.task.get('id')
        if self.task_id not in TASK_OBJECTS or self.task.get('public_rules') != public_rules(self.task_id):
            raise ValueError('Task identity/public rules differ from frozen evaluator')
        if self.task.get('engineering_probe'):
            raise ValueError('Engineering reset probes cannot earn original task credit')
        self.c = self.task['evaluation']
        for key, expected in thresholds(self.task_id).items():
            if number(self.c.get(key), key) != expected:
                raise ValueError('Task threshold differs from frozen rules: '+key)
        if self.c.get('must_be_released') is not True:
            raise ValueError('Released completion is required')
        self.dt, self.control = self.c['physics_dt_s'], self.c['control_period_s']
        self.substeps = round(self.control/self.dt)
        self.names = TASK_OBJECTS[self.task_id]
        self.actors = self.c['actors']
        if (not isinstance(self.actors, dict) or not set(self.names) <= set(self.actors)
                or any(not isinstance(p, str) or not p.startswith('/World/') or p.endswith('/') for p in self.actors.values())
                or len(set(self.actors.values())) != len(self.actors)):
            raise ValueError('Distinct explicit native actors are required')
        if any(self.actors[n] != '/World/Objects/'+n for n in self.names):
            raise ValueError('Task object actor identity differs')
        self.robot = self.c['robot_actor_prefix']
        if self.robot != '/World/Robot':
            raise ValueError('Expected pinned single robot actor root')
        self.expected = {n: self._state(row, velocities=False) for n, row in self.c['initial_objects'].items()}
        if set(self.expected) != set(self.names):
            raise ValueError('All and only task objects require frozen reset poses')
        raw_pairs = self.c['required_support_pairs']
        if not isinstance(raw_pairs,list) or any(not isinstance(p,(list,tuple)) or len(p)!=2 for p in raw_pairs):
            raise ValueError('Each support pair must contain two actor names')
        self.pairs = [tuple(pair) for pair in raw_pairs]
        if (len(self.pairs) != len(self.names) or {p[0] for p in self.pairs} != set(self.names)
                or any(p[0] == p[1] or p[1] not in self.actors for p in self.pairs)):
            raise ValueError('Each object requires exactly one named support')
        if self.task_id == 'shaft_spacer_assembly':
            if (self.c['part_order'] != list(self.names)
                    or self.pairs != [('spacer','fixture'), ('wide_ring','spacer'), ('top_ring','wide_ring')]):
                raise ValueError('The shaft support graph must preserve layer order')
        elif self.task_id == 'peg_insertion':
            if self.pairs != [('peg_0','fixture'), ('peg_1','fixture')]:
                raise ValueError('Both pegs require socket fixture support')
        elif any(support in self.names for _, support in self.pairs):
            raise ValueError('Sorting requires separate tray floor supports')
        if self.task_id == 'parts_sorting':
            self.goals = self.task['goals']
            if set(self.goals) != set(self.names) or set(self.c['geometry_vertices_local_m']) != set(self.names):
                raise ValueError('Every sorting part requires a goal and complete source geometry')
            self.vertices = {}
            for name in self.names:
                points = self.c['geometry_vertices_local_m'][name]
                if not isinstance(points, list) or len(points) < 4:
                    raise ValueError('Whole-geometry vertices are required')
                self.vertices[name] = np.array([vector(p,3,'local geometry vertex') for p in points])
                goal = self.goals[name]
                vector(goal['center_xy_m'],2,'tray center')
                if min(vector(goal['inner_half_extents_xy_m'],2,'tray extents')) <= 0:
                    raise ValueError('Tray extents must be positive')
                number(goal['floor_top_z_m'],'floor height')
                support = dict(self.pairs)[name]
                if goal['object_actor'] != self.actors[name] or goal['support_collider'] != self.actors[support]:
                    raise ValueError('Tray support identity differs from the declared goal')
        else:
            self.targets = {n: vector(p,3,'bottom target') for n,p in self.c['target_bottom_centers_m'].items()}
            if set(self.targets) != set(self.names):
                raise ValueError('All assembly parts need explicit bottom targets')
        self.reset()

    @staticmethod
    def _state(row, *, velocities=True):
        return {'p': vector(row['position_m'],3,'position'), 'r': rotation(row['quaternion_wxyz']),
                'v': vector(row['linear_velocity_m_s'],3,'linear velocity') if velocities else np.zeros(3),
                'w': vector(row['angular_velocity_rad_s'],3,'angular velocity') if velocities else np.zeros(3)}

    def reset(self):
        self.previous_time = self.native_step = self.native_time = self.previous = None
        self.invalid = self.anchor = self.last_result = None
        self.invalid_state = self.invalid_code = None
        self.stable_s = 0.
        self.success = False
        self.status = {}

    def _contacts(self, obs, time, first):
        rows,step,native_time = read_native_contact_window(obs,time=time,first=first,
            physics_dt_s=self.dt,control_period_s=self.control,previous_time=self.previous_time,
            native_step=self.native_step,native_time=self.native_time)
        support = {name:0. for name in self.names}
        released = {name:True for name in self.names}
        for row in rows:
            a,b = row['actor0'],row['actor1']
            for name,support_name in self.pairs:
                body = self.actors[name]
                if body not in (a,b): continue
                if row['min_separation_m'] < -self.c['maximum_reported_penetration_m']:
                    raise _SafetyStop('Task contact penetration exceeds frozen bound', 'contact_penetration_bound')
                other = b if a==body else a
                if within(other,self.robot): released[name]=False
                support_actor = self.actors[support_name]
                matches = other==support_actor if support_name in self.names else within(other,support_actor)
                if matches:
                    # Native vectors act on actor0. Sum signed z: downward
                    # contacts cannot become evidence by taking an absolute value.
                    support[name] += float(row['impulse_vector_ns'][2])*(1 if a==body else -1)
        return support,released,step,native_time

    def _geometry(self,name,state):
        if self.task_id == 'parts_sorting':
            points = self.vertices[name] @ state['r'].T + state['p']
            goal = self.goals[name]
            excess = np.max(np.abs(points[:,:2]-goal['center_xy_m'])-
                            (np.array(goal['inner_half_extents_xy_m'])-self.c['containment_margin_m']))
            floor_error = float(points[:,2].min()-goal['floor_top_z_m'])
            reasons = []
            if excess > EPS: reasons.append('outside_matching_tray')
            if abs(floor_error)>self.c['maximum_floor_height_error_m']: reasons.append('not_at_tray_floor')
            return {'geometry':bool(excess<=EPS and abs(floor_error)<=self.c['maximum_floor_height_error_m']),
                    'maximum_tray_boundary_excess_m':float(excess),'floor_height_error_m':floor_error,
                    'geometry_reasons':reasons}
        delta = state['p']-self.targets[name]
        tilt = math.acos(float(np.clip(state['r'][2,2],-1.,1.)))
        offset = float(np.linalg.norm(delta[:2]))
        reasons = []
        if offset>self.c['maximum_axis_offset_m']: reasons.append('axis_offset')
        if abs(delta[2])>self.c['maximum_bottom_height_error_m']: reasons.append('bottom_height')
        if tilt>self.c['maximum_axis_tilt_rad']: reasons.append('axis_tilt')
        return {'geometry':bool(offset<=self.c['maximum_axis_offset_m'] and
                    abs(delta[2])<=self.c['maximum_bottom_height_error_m'] and tilt<=self.c['maximum_axis_tilt_rad']),
                'axis_offset_m':offset,'bottom_height_error_m':float(delta[2]),'axis_tilt_rad':tilt,
                'geometry_reasons':reasons}

    def _verdict(self, reason):
        if self.invalid:
            phase = 'safety_stop' if self.invalid_state == 'safety_stop' else 'invalid'
            return verdict(state=self.invalid_state, phase=phase, reasons=[self.invalid_code])
        if self.success:
            return verdict(state='success', phase='complete', reasons=['success'])
        if reason in ('not_sampled', 'reset_baseline_not_scored'):
            return verdict(state='ongoing', phase='not_started' if reason == 'not_sampled' else 'reset',
                           reasons=[reason])
        reasons = list(dict.fromkeys(code for row in self.status.values() for code in row['reasons']))
        if any(not row['geometry'] for row in self.status.values()): phase = 'placement'
        elif any(not row['supported'] for row in self.status.values()): phase = 'support'
        elif any(not row['released'] for row in self.status.values()): phase = 'release'
        else: phase = 'settling'
        return verdict(state='ongoing', phase=phase, reasons=reasons or ['stability_window_incomplete'])

    def _result(self,reason):
        complete = sum(all(s.get(k,False) for k in ('geometry','supported','released','quiet')) for s in self.status.values())
        self.last_result = {'evaluator':VERSION,'task_id':self.task_id,'success':bool(self.success and not self.invalid),
            'progress':1. if self.success and not self.invalid else 0. if self.invalid else .9*complete/len(self.names),
            'reason':reason,'release_stable_s':self.stable_s,'objects':deepcopy(self.status),
            'referee':self._verdict(reason),
            'threshold_status':'engineering_candidates_not_native_accepted',
            'evidence_scope':'20 ms actual object poses/velocities; complete native-substep normal-contact receipts. Not continuous substep object trajectories or controller qualification.'}
        return deepcopy(self.last_result)

    def update(self,observation):
        # EpisodeSession ends on the first accepted success. Freeze that verdict
        # for direct evaluator consumers too; terminal updates consume no data.
        if self.success and self.last_result is not None:
            return deepcopy(self.last_result)
        if self.invalid: return self._result('invalid_observation:'+self.invalid)
        try:
            time = number(observation['time_s'],'time')
            if abs(number(observation['sim_time_s'],'simulation time')-time)>EPS:
                raise ValueError('Observation clocks disagree')
            first = self.previous_time is None
            if not first and abs(time-self.previous_time-self.control)>EPS:
                raise ValueError('Every 20 ms object observation is required')
            if observation.get('metrics',{}).get('state_finite') is not True or observation.get('terminated') is not False:
                raise ValueError('Finite continuing native state is required')
            states = {name:self._state(observation['objects'][name]) for name in self.names}
            # The entire reset must belong to this task before any physical
            # safety limit is classified. A later mismatched body cannot be
            # hidden by an earlier body's otherwise valid speed violation.
            if first and any(np.linalg.norm(state['p']-self.expected[name]['p'])>self.c['initial_position_tolerance_m']
                    or angle(state['r'],self.expected[name]['r'])>self.c['initial_orientation_tolerance_rad']
                    for name,state in states.items()):
                raise ValueError('Initial pose differs from frozen reset')
            supports,released,step,native_time = self._contacts(observation,time,first)
            status = {}
            for name,state in states.items():
                rate = 0. if first else np.linalg.norm(state['p']-self.previous[name]['p'])/self.control
                angular = 0. if first else angle(self.previous[name]['r'],state['r'])/self.control
                if (max(rate,np.linalg.norm(state['v']))>self.c['maximum_object_linear_speed_m_s']+EPS
                        or max(angular,np.linalg.norm(state['w']))>self.c['maximum_object_angular_speed_rad_s']+EPS):
                    raise _SafetyStop('Object motion speed bound exceeded', 'object_speed_bound')
                quiet = (max(rate,np.linalg.norm(state['v']))<=self.c['maximum_stable_linear_speed_m_s']+EPS
                         and max(angular,np.linalg.norm(state['w']))<=self.c['maximum_stable_angular_speed_rad_s']+EPS)
                status[name] = {**self._geometry(name,state),'quiet':bool(quiet),'released':released[name],
                    'supported':supports[name]>self.c['minimum_support_upward_impulse_ns'],
                    'support_upward_impulse_ns':supports[name]}
                status[name]['reasons'] = list(status[name]['geometry_reasons'])
                for key,code in (('supported','required_support_missing'), ('released','robot_contact_present'),
                                 ('quiet','still_moving')):
                    if not status[name][key]: status[name]['reasons'].append(code)
            self.status = status
            if first:
                self.previous,self.previous_time = states,time
                self.native_step,self.native_time = step,native_time
                return self._result('reset_baseline_not_scored')
            qualifying = all(all(row[k] for k in ('geometry','quiet','released','supported')) for row in status.values())
            # Both endpoint poses and native velocities must qualify; the
            # interval that first comes to rest is not credited retrospectively.
            previous_ready = {n: self._geometry(n,s)['geometry']
                and np.linalg.norm(s['v'])<=self.c['maximum_stable_linear_speed_m_s']+EPS
                and np.linalg.norm(s['w'])<=self.c['maximum_stable_angular_speed_rad_s']+EPS
                for n,s in self.previous.items()}
            if qualifying:
                for name,ready in previous_ready.items():
                    if not ready: self.status[name]['reasons'].append('previous_endpoint_not_ready')
            qualifying = qualifying and all(previous_ready.values())
            if qualifying:
                if self.anchor is None: self.anchor=deepcopy(self.previous)
                spread = {n: np.linalg.norm(s['p']-self.anchor[n]['p'])<=self.c['maximum_stable_position_range_m']+EPS
                    and angle(s['r'],self.anchor[n]['r'])<=self.c['maximum_stable_orientation_range_rad']+EPS for n,s in states.items()}
                if all(spread.values()): self.stable_s+=self.control
                else:
                    for name,within_window in spread.items():
                        if not within_window: self.status[name]['reasons'].append('stability_drift')
                    self.stable_s,self.anchor=0.,None
            else: self.stable_s,self.anchor=0.,None
            self.success = self.stable_s+EPS>=self.c['minimum_stable_s']
            self.previous,self.previous_time = states,time
            self.native_step,self.native_time = step,native_time
            return self._result('success' if self.success else 'placement_support_release_stability_required')
        except (KeyError,TypeError,ValueError,IndexError,OverflowError,AttributeError) as exc:
            self.invalid=str(exc);self.success=False;self.stable_s=0.
            self.invalid_state = 'safety_stop' if isinstance(exc,_SafetyStop) else 'evaluator_invalid'
            self.invalid_code = exc.code if isinstance(exc,_SafetyStop) else 'invalid_evidence'
            return self._result('invalid_observation:'+self.invalid)

    def inspect_conditions(self,observation=None):
        """Read the latest score without consuming its contact window twice."""
        return deepcopy(self.last_result or {'success':False,'reason':'not_sampled',
            'referee':self._verdict('not_sampled')})
