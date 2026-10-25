"""Practical streaming pouring score; frozen sibling rules and stdlib only."""
from collections.abc import Mapping
from copy import deepcopy
import math

from .rules import configuration_hash, public_rules, thresholds

VERSION = 'living_water_pouring_evaluator_v2'
EPS = 1.e-7


def number(value, name):
    if type(value) not in (int,float) or not math.isfinite(value):
        raise ValueError(name+' must be finite')
    return float(value)


def integer(value, name, maximum=None):
    if type(value) is not int or value < 0 or (maximum is not None and value > maximum):
        raise ValueError(name+' must be a nonnegative bounded integer')
    return value


def vector(value, size, name):
    if not isinstance(value,(list,tuple)) or len(value) != size:
        raise ValueError(name+' has wrong size')
    return tuple(number(v,name) for v in value)


def norm(v):
    return math.sqrt(sum(x*x for x in v))


def distance(a,b):
    return norm(tuple(x-y for x,y in zip(a,b)))


def angle(a,b):
    return 2*math.acos(min(1.,abs(sum(x*y for x,y in zip(a,b)))))


class SequenceEvaluator:
    def __init__(self, task):
        self.task = deepcopy(task)
        self.c = self.task['evaluation']
        if self.c.get('schema') != 'living_water_evaluation_v1' or self.c.get('contract_sha256') != configuration_hash(self.c):
            raise ValueError('Frozen water configuration/hash mismatch')
        if self.c['thresholds'] != thresholds() or self.task['public_rules'] != public_rules(quantities=self.c['quantities']):
            raise ValueError('Water public rules differ from frozen evaluator')
        self.th, self.goal = self.c['thresholds'], self.task['public_rules']['goal']
        self.n = integer(self.c['particle_count'],'particle count')
        self.ml = number(self.c['particle_mass_kg'],'particle mass')/number(self.c['reference_density_kg_m3'],'density')*1e6
        self.dt = number(self.c['physics_dt_s'],'dt')
        if min(self.n,self.ml,self.dt) <= 0 or abs(self.n*self.ml-self.goal['initial_water_ml']) > 1.e-4:
            raise ValueError('Water mass inventory differs from goal')
        self.reset()

    def reset(self):
        self.previous = self.previous_time = self.previous_native = self.previous_step = self.reset_id = None
        self.stable_start = self.anchor = None
        self.cup_window = []
        self.dwell = 0.
        self.previous_spilled = self.previous_left = self.previous_front = 0
        self.previous_lift = 0.
        self.invalid = self.failure = self.last_result = None
        self.success = False
        self.values = {'kettle_ml':self.n*self.ml,'cup_ml':0.,'water_difference_ml':0.,
            'current_spill_ml':0.,'cumulative_spill_ml':0.,'unresolved_ml':0.,
            'lifted':False,'spout_flow_observed':False}

    def _state(self, row):
        p = vector(row['position_m'],3,'position')
        q = vector(row['quaternion_wxyz'],4,'orientation')
        if abs(norm(q)-1.) > 1.e-4:
            raise ValueError('Vessel orientation must be unit wxyz')
        q = tuple(x/norm(q) for x in q)
        v,w = vector(row['linear_velocity_m_s'],3,'linear velocity'),vector(row['angular_velocity_rad_s'],3,'angular velocity')
        support = number(row['support_force_min_n'],'support force')
        robot = number(row['robot_contact_force_max_n'],'robot contact force')
        if min(support,robot) < 0:
            raise ValueError('Contact force magnitudes cannot be negative')
        return {'p':p,'q':q,'v':v,'w':w,'support':support,'robot':robot,
                'tilt':math.acos(max(-1.,min(1.,1-2*(q[1]*q[1]+q[2]*q[2]))))}

    def _clear_dwell(self):
        self.stable_start = self.anchor = None
        self.cup_window = []
        self.dwell,self.success = 0.,False

    def _result(self, reason):
        progress = .25*int(self.values['lifted'])+.25*int(self.values['spout_flow_observed'])
        low,high = self.goal['target_interval_ml']
        if low <= self.values['cup_ml'] <= high:
            progress += .25
        progress += .25*min(1.,self.dwell/self.th['dwell_s'])
        self.last_result = {'evaluator':VERSION,'task_id':self.task.get('id','water_pouring'),
            'success':bool(self.success and not self.invalid and not self.failure),
            'progress':0. if self.invalid else min(1.,progress),'reason':reason,
            'irrecoverable_reason':self.failure,'evidence_status':'invalid' if self.invalid else 'available',
            'stable_duration_s':self.dwell,'initial_water_ml':self.n*self.ml,
            'kettle_water_ml':self.values['kettle_ml'],'cup_water_ml':self.values['cup_ml'],
            'water_difference_ml':self.values['water_difference_ml'],
            'current_spill_ml':self.values['current_spill_ml'],
            'target_interval_ml':deepcopy(self.goal['target_interval_ml']),
            'cumulative_spill_ml':self.values['cumulative_spill_ml'],'max_spill_ml':self.goal['max_spill_ml'],
            'remaining_spill_budget_ml':max(0.,self.goal['max_spill_ml']-self.values['cumulative_spill_ml']),
            'unresolved_ml':self.values['unresolved_ml'],'water_load_calibrated':False,
            'evidence_scope':'Service-owned native particle inventory and rigid-contact summary; practical nominal-volume scoring, not a quantitative water-load calibration.'}
        return deepcopy(self.last_result)

    def update(self, observation):
        if self.invalid:
            return self._result('invalid_observation:'+self.invalid)
        if self.failure:
            return self._result(self.failure)
        try:
            obs = observation
            time = number(obs['time_s'],'observation time')
            if time < -EPS or abs(number(obs['sim_time_s'],'simulation time')-time) > EPS:
                raise ValueError('Observation clocks disagree')
            state = obs['task_state']
            if not isinstance(state,Mapping) or state.get('schema') != 'living_water_observation_v1' or state.get('errors') != []:
                raise ValueError('Native water observer missing or has errors')
            if state.get('contract_sha256') != self.c['contract_sha256'] or abs(number(state['time_s'],'state time')-time) > EPS:
                raise ValueError('Water observation has wrong configuration or clock')
            reset = state['reset_id']
            if not isinstance(reset,str) or not reset:
                raise ValueError('Native episode identity missing')
            first = self.previous_time is None
            if not first and reset != self.reset_id:
                raise ValueError('Reset changed inside episode')
            if obs.get('metrics',{}).get('state_finite',True) is not True:
                raise ValueError('Native state is nonfinite')
            sampling = state['sampling']
            start,end,count = (integer(sampling[k],k) for k in ('start_step','end_step','sampled_substeps'))
            ns,ne = number(sampling['start_time_s'],'native start'),number(sampling['end_time_s'],'native end')
            if abs(number(sampling['physics_dt_s'],'native dt')-self.dt) > 1.e-12 or end-start != count or abs(ne-ns-count*self.dt) > 1.e-6:
                raise ValueError('Native substep coverage differs from frozen dt')
            if first:
                if count or end != start or abs(ne-ns) > EPS:
                    raise ValueError('First observation must be uncredited native reset baseline')
            else:
                duration = time-self.previous_time
                if (not 0 < duration <= self.th['maximum_observation_gap_s']+EPS or count == 0
                    or start != self.previous_step or abs(ns-self.previous_native) > EPS
                    or abs(duration-count*self.dt) > 1.e-6):
                    raise ValueError('Native observation intervals must be complete and consecutive')
            water = state['water']
            if integer(water['particle_count'],'current particle count') != self.n or water['ids_sha256'] != self.c['particle_ids_sha256']:
                raise ValueError('Particle inventory/identity changed')
            counts = water['counts']
            if set(counts) != {'kettle','cup','spilled_outside','in_flight_or_unresolved'}:
                raise ValueError('Particle compartment keys differ')
            counts = {k:integer(v,k,self.n) for k,v in counts.items()}
            if sum(counts.values()) != self.n:
                raise ValueError('Current particle mass partition does not close')
            spilled = integer(water['cumulative_unique_spilled_particles'],'cumulative spill',self.n)
            left = integer(water['ever_left_kettle_particles'],'ever left kettle',self.n)
            exits = water['first_kettle_exit_regions']
            if set(exits) != {'front_rim','other_rim','below_rim'}:
                raise ValueError('Missing first-exit history')
            exits = {k:integer(v,k,self.n) for k,v in exits.items()}
            if sum(exits.values()) != left or counts['spilled_outside'] > spilled or left < self.n-counts['kettle']:
                raise ValueError('Contradictory cumulative water history')
            front = exits['front_rim']
            lift = number(state['activity']['max_kettle_lift_m'],'maximum lift')
            if min(lift,spilled-self.previous_spilled,left-self.previous_left,front-self.previous_front,lift-self.previous_lift) < -EPS:
                raise ValueError('Cumulative water/activity history moved backwards')
            if set(state['vessels']) != {'kettle','cup'}:
                raise ValueError('Native vessel inventory differs')
            bodies = {name:self._state(row) for name,row in state['vessels'].items()}
            # Subtract integer inventory before converting to mL: airborne water
            # is a pending transfer, not spill. The partition check above closes
            # this difference against the observed outside-spill compartment.
            difference = self.n-counts['kettle']-counts['cup']
            self.values = {'kettle_ml':counts['kettle']*self.ml,'cup_ml':counts['cup']*self.ml,
                'water_difference_ml':difference*self.ml,
                'current_spill_ml':(difference-counts['in_flight_or_unresolved'])*self.ml,
                'cumulative_spill_ml':spilled*self.ml,
                'unresolved_ml':counts['in_flight_or_unresolved']*self.ml,
                'lifted':lift+EPS >= self.th['minimum_lift_m'],
                'spout_flow_observed':front*self.ml+EPS >= self.th['minimum_spout_exit_ml']}
            if first:
                if self.values['cup_ml'] > .2+EPS or spilled or left or lift > .005:
                    raise ValueError('Reset must be unpoured and uncredited')
                self.reset_id = reset
                reason = 'reset_baseline_not_scored'
            else:
                reason = self._evaluate(bodies,time)
            self.previous,self.previous_time,self.previous_native,self.previous_step = bodies,time,ne,end
            self.previous_spilled,self.previous_left,self.previous_front,self.previous_lift = spilled,left,front,lift
            if obs.get('terminated') is True and not self.success and not self.failure:
                self._clear_dwell()
                reason = 'incomplete_terminated'
            return self._result(reason)
        except (KeyError,TypeError,ValueError,IndexError,OverflowError) as error:
            self.invalid = str(error)
            self._clear_dwell()
            return self._result('invalid_observation:'+self.invalid)

    def _evaluate(self, bodies, time):
        if self.values['cumulative_spill_ml'] > self.goal['max_spill_ml']+EPS:
            self.failure = 'spill_limit_exceeded'
        elif any(row['p'][2] < self.c['table_top_z_m']-.15 for row in bodies.values()):
            self.failure = 'vessel_fell_from_table'
        elif bodies['cup']['tilt'] > self.th['failure_tilt_rad']:
            self.failure = 'cup_tipped_over'
        if self.failure:
            self._clear_dwell()
            return self.failure
        low,high = self.goal['target_interval_ml']
        reason = None
        if not self.values['lifted']: reason = 'lift_kettle'
        elif not self.values['spout_flow_observed']: reason = 'pour_through_spout'
        elif not low-EPS <= self.values['cup_ml'] <= high+EPS: reason = 'cup_water_outside_target'
        elif self.values['unresolved_ml'] > self.th['maximum_unresolved_ml']+EPS: reason = 'water_still_in_flight'
        for name,row in bodies.items():
            if reason:break
            if row['tilt'] > self.th['maximum_tilt_rad']:reason = name+'_not_upright'
            elif row['support'] < self.th['minimum_support_force_n']:reason = name+'_not_supported'
            elif row['robot'] > self.th['maximum_robot_contact_force_n']:reason = 'release_'+name
            elif max(norm(row['v']),distance(row['p'],self.previous[name]['p'])/(time-self.previous_time)) > self.th['maximum_linear_speed_m_s']:
                reason = name+'_still_moving'
            elif max(norm(row['w']),angle(row['q'],self.previous[name]['q'])/(time-self.previous_time)) > self.th['maximum_angular_speed_rad_s']:
                reason = name+'_still_rotating'
        target = self.c['containers']['kettle']['initial_position_m']
        p = bodies['kettle']['p']
        if not reason and (distance(p[:2],target[:2]) > self.th['return_xy_radius_m'] or abs(p[2]-target[2]) > self.th['return_z_tolerance_m']):
            reason = 'return_kettle_to_zone'
        if reason:
            self._clear_dwell()
            return reason
        if self.stable_start is None:
            self.stable_start,self.anchor = time,deepcopy(bodies)
            self.cup_window = []
        self.cup_window.append(self.values['cup_ml'])
        if max(self.cup_window)-min(self.cup_window) > self.th['maximum_cup_volume_range_ml']+EPS:
            self._clear_dwell()
            return 'cup_water_still_changing'
        for name,row in bodies.items():
            anchor = self.anchor[name]
            if (distance(row['p'],anchor['p']) > self.th['maximum_stable_position_range_m']
                or angle(row['q'],anchor['q']) > self.th['maximum_stable_orientation_range_rad']):
                self._clear_dwell()
                return 'vessel_drifted_during_settle'
        self.dwell = time-self.stable_start
        self.success = self.dwell+EPS >= self.th['dwell_s']
        return 'success' if self.success else 'release_and_settle'

    def inspect_conditions(self, observation=None):
        return deepcopy(self.last_result or {'success':False,'reason':'not_sampled'})
