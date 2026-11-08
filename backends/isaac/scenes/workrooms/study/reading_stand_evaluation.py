"""Read-only hinged-panel, physical support-pin and book loading referee."""
from copy import deepcopy
from itertools import product
import math
from .evaluation import SequenceEvaluator as Base, norm, distance, angle, EPS
from .rules import rotation, quaternion
from .reading_stand_rules import THRESHOLDS


class SequenceEvaluator(Base):
    def __init__(self, task_spec, *, trusted_normalized_input=False):
        super().__init__(task_spec,trusted_normalized_input=trusted_normalized_input)
        self.t=THRESHOLDS
        self.opened=self.pin_moved=self.book_moved=False

    def _geometry(self, ident, state):
        # The base constructor rejects only a fully solved reset. This task's
        # coupled geometry is checked by physical_conditions below.
        return ['mechanism_not_operated']

    def _loaded_pair(self, name, other, states, contacts):
        total=0.
        for row in contacts[name]:
            if (row['other']==other and -self.t['maximum_penetration_m']<=row['separation']<=self.t['support_gap_m']
                and self._inside_object(states[name],name,row['point'])
                and self._inside_object(states[other],other,row['point'])):
                total+=row['upward_impulse']
        return total>self.t['minimum_support_impulse_ns']

    def physical_conditions(self, states, contacts, previous=None):
        panel,pin,book=(states[n] for n in ('panel','support_pin','book'))
        theta=math.degrees(math.atan2(panel['rotation'][2][0],panel['rotation'][0][0]))
        origin=self.config['mechanism']['hinge_world_m'];pc=self.config['mechanism']['pin_center_world_m']
        panel_geometry=50.<=theta<=60. and distance(panel['position_m'],origin)<=.003
        pin_geometry=(abs(pin['position_m'][1])<=.020 and
            abs(pin['position_m'][0]-pc[0])<=.003 and abs(pin['position_m'][2]-pc[2])<=.003
            and angle(pin['quaternion_wxyz'],[1.,0.,0.,0.])<=.03)
        points=[]
        for corner in product(*zip(*self.config['objects']['book']['local_bounds_m'])):
            world=[book['position_m'][i]+sum(book['rotation'][i][j]*corner[j] for j in range(3)) for i in range(3)]
            delta=[world[i]-panel['position_m'][i] for i in range(3)]
            points.append([sum(panel['rotation'][i][j]*delta[i] for i in range(3)) for j in range(3)])
        bounds=self.config['mechanism']['panel_book_region_local_m']
        book_geometry=(all(bounds[0][i]-.002<=p[i]<=bounds[1][i]+.002 for p in points for i in range(3))
            and angle(book['quaternion_wxyz'],panel['quaternion_wxyz'])<=self.t['maximum_book_panel_angle_rad'])
        supported={'panel':self._loaded_pair('panel','support_pin',states,contacts),
                   'support_pin':pin_geometry,
                   'book':self._loaded_pair('book','panel',states,contacts)}
        allowed={'panel':{'support_pin','book'},'support_pin':{'panel'},'book':{'panel'}}
        geometry={'panel':panel_geometry,'support_pin':pin_geometry,'book':book_geometry}
        failure={'panel':'panel_angle','support_pin':'pin_not_inserted','book':'book_not_on_panel'}
        status={}
        for ident,state in states.items():
            reasons=[]
            if not geometry[ident]:reasons.append(failure[ident])
            if not supported[ident]:reasons.append('panel_not_pin_supported' if ident=='panel' else 'required_contact_support_missing')
            rows=contacts[ident];released=not any(r['other'].startswith('robot/') for r in rows)
            if not released:reasons.append('robot_contact_present')
            if any(r['other'] not in allowed[ident] and not r['other'].startswith('robot/') for r in rows):reasons.append('unexpected_contact')
            if any(r['separation'] < -self.t['maximum_penetration_m'] for r in rows):reasons.append('contact_penetration')
            linear=norm(state['linear_velocity_m_s']);angular=norm(state['angular_velocity_rad_s'])
            if previous:
                linear=max(linear,distance(state['position_m'],previous[ident]['position_m'])/self.dt)
                angular=max(angular,angle(state['quaternion_wxyz'],previous[ident]['quaternion_wxyz'])/self.dt)
            if linear>self.t['maximum_linear_speed_m_s'] or angular>self.t['maximum_angular_speed_rad_s']:reasons.append('still_moving')
            status[ident]={'geometry':geometry[ident],'supported':supported[ident],'released':released,'reasons':reasons,
                'linear_speed_m_s':linear,'angular_speed_rad_s':angular}
        return {'panel_angle_deg':theta,'pin_y_m':pin['position_m'][1],'objects':status,
                'conditions_passed':all(not r['reasons'] for r in status.values())}

    def update(self, observation):
        if self.terminal:return deepcopy(self.last_result)
        try:
            time,index,actions,episode=self._sampling(observation)
            if set(observation['objects'])!=set(self.ids):raise ValueError('Reading stand native object inventory differs')
            states={n:self._state(observation['objects'][n]) for n in self.ids}; contacts=self._contacts(observation)
            first=self.previous is None;previous=self.previous
            if first:
                if observation['contacts']:raise ValueError('Reset baseline must have no sampled contact interval')
                for n,s in states.items():
                    initial=self.config['objects'][n]['initial_pose']
                    if distance(s['position_m'],initial['position_m'])>self.t['initial_position_tolerance_m'] or angle(s['quaternion_wxyz'],quaternion(initial['quaternion_wxyz']))>self.t['initial_orientation_tolerance_rad']:
                        raise ValueError('Reading stand reset differs from frozen plan')
            self.previous,self.time,self.index,self.episode,self.actions=states,time,index,episode,actions
            if observation['budget_exhausted'] or time+EPS>=self.config['budget']['max_sim_time_s']:return self.finish('budget_exhausted')
            if first:return self._result('active','reset_baseline_not_scored')
            if any(r['other']=='floor' and r['separation']<=self.t['support_gap_m'] for rows in contacts.values() for r in rows):return self._result('failure','object_touched_floor',terminal=True)
            measured=self.physical_conditions(states,contacts,previous);self.status=measured['objects']
            self.opened |= measured['panel_angle_deg']>=50.
            self.pin_moved |= abs(states['support_pin']['position_m'][1]-.28)>=.20
            self.book_moved |= distance(states['book']['position_m'],self.config['objects']['book']['initial_pose']['position_m'])>=.08
            if not (self.opened and self.pin_moved and self.book_moved):
                self.status['panel']['reasons'].append('mechanism_not_operated')
            good=actions>0 and all(not row['reasons'] for row in self.status.values())
            if not good:self.anchor=self.stable_since=None
            elif self.anchor is None:self.anchor,self.stable_since=deepcopy(states),time
            elif any(distance(s['position_m'],self.anchor[n]['position_m'])>self.t['maximum_position_drift_m'] or angle(s['quaternion_wxyz'],self.anchor[n]['quaternion_wxyz'])>self.t['maximum_orientation_drift_rad'] for n,s in states.items()):
                self.anchor,self.stable_since=deepcopy(states),time
            stable=0. if self.stable_since is None else time-self.stable_since
            if good and stable+EPS>=self.t['minimum_stable_s']:return self._result('success','success',stable_s=stable,terminal=True)
            return self._result('active','robot_action_required' if actions==0 else 'stability_window_incomplete',stable_s=stable)
        except (KeyError,TypeError,ValueError,IndexError,OverflowError,AttributeError) as exc:
            self.anchor=self.stable_since=None
            return self._result('invalid','invalid_evidence',detail=str(exc),terminal=True)
