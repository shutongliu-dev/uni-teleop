"""Complete native-stream dual-arm referee; no action policy or physical writes."""
from collections import deque
from copy import deepcopy
import math
import numpy as np

if __package__ == 'scenes.workrooms.mechanical_assembly':
    from .cooperative_rules import digest, evaluation_configuration, public_rules
    from .nist_native_evidence import EPS, Contacts, angle, distance, fixed, local, number, points, rotation, state
else:
    from .rules import digest, evaluation_configuration, public_rules
    from .native_evidence import EPS, Contacts, angle, distance, fixed, local, number, points, rotation, state

VERSION='mechanical_cooperative_sequence_evaluator.v1'
STATE_SCHEMA='mechanical_cooperative_native_state.v1'


def configuration(task):
    if task.get('schema_version')!='mechanical-cooperative-task-v1':raise ValueError('Expected cooperative task')
    c=deepcopy(task['evaluation'])
    expected=evaluation_configuration(task['id'],c['metadata'],task['episode_seed'],c['physics_dt_s'],c['room_support_paths'])
    if c!=expected or task['public_rules']!=public_rules(task['id'],task['episode_seed']):raise ValueError('Frozen cooperative rules differ')
    return c,digest(c)


class NativeTaskObserver:
    def __init__(self,task_spec):
        self.c,self.digest=configuration(task_spec)
        self.task_id,self.meta=self.c['task_id'],self.c['metadata']
        self.limits,self.p=self.c['thresholds'],self.meta['task_parameters']
        self.specs={r['id']:r for r in self.meta['objects']}
        self.fixed={n:fixed(r['initial_pose_xyzw']) for n,r in self.meta['fixture_metadata'].items()}
        self.paths={**self.meta['actors'],**self.meta['fixtures']}
        for role,paths_ in self.c['room_support_paths'].items():
            for i,path in enumerate(paths_):self.paths['environment_'+role+'_'+str(i)]=path.replace('/World/Geometry/','/World/Room/Geometry/',1)
        self.reset()

    def reset(self):
        self.previous=self.initial=self.current=None
        self.previous_time=self.previous_native_time=self.previous_step=None
        self.error=self.failure=self.safety=None
        self.stable_s=self.cooperation_s=0.
        self.last_cooperative_depth=None
        self.last_insertion_depth=None
        self.cooperative_progress_m=0.
        self.cooperation_confirmed=False
        self.events=[]
        self.witnesses={}
        self.stable_anchor=None
        self.history=deque(maxlen=round(self.limits['pose_window_s']/self.c['physics_dt_s'])+1)

    def _hand(self,path):
        for side in self.p['robot_instances']:
            scope=self.p.get('robot_stage_scopes',{}).get(side,side)
            root='/World/Robot/'+scope+'/'
            if path.startswith(root):
                # Hand prefixes are instance or anatomical names; fr3_<side>
                # arm links cannot earn hand-contact cooperation credit.
                if any(part.startswith(side+'_') for part in path[len(root):].split('/')):return side
        return None

    def _hands_on(self,rows,t):
        result={name:set() for name in self.specs}
        for row in rows:
            if row['min_separation_m']>self.limits['maximum_mating_gap_m'] or row['impulse_magnitude_sum_ns']<=self.limits['minimum_contact_impulse_ns']:continue
            for objpath,robotpath in ((row['actor0'],row['actor1']),(row['actor1'],row['actor0'])):
                side=self._hand(robotpath)
                if side:
                    for name,path in self.meta['actors'].items():
                        if objpath==path or objpath.startswith(path+'/'):result[name].add(side)
        return result

    def _mate(self,objects):
        sleeve,peg=objects['sleeve'],objects['peg']
        tip=np.asarray(peg['position_m'])+rotation(peg['quaternion_wxyz'])@self.p['peg_tip_local_m']
        rel=local(tip,sleeve)
        depth=self.p['socket_mouth_local_m'][2]-rel[2]
        lateral=float(np.linalg.norm(rel[:2]))
        alignment=float(np.dot(rotation(sleeve['quaternion_wxyz'])@self.p['socket_axis_local'],rotation(peg['quaternion_wxyz'])@self.p['peg_axis_local']))
        angular=math.acos(float(np.clip(alignment,-1.,1.)))
        aligned=lateral<=self.p['maximum_lateral_error_m'] and angular<=self.p['maximum_axis_angle_rad']
        mated=aligned and self.p['minimum_insertion_depth_m']-EPS<=depth<=self.p['maximum_insertion_depth_m']+EPS
        return bool(mated),dict(insertion_depth_m=float(depth),lateral_error_m=lateral,axis_error_rad=angular,aligned=bool(aligned))

    def _supported(self,name,other,objects,contacts,t,geometry=True):
        key=(name,other)
        if not geometry:self.witnesses.pop(key,None);return False
        obj=objects[name];support=objects.get(other,self.fixed.get(other))
        rel=local(obj['position_m'],support)
        if key in contacts.upward:
            self.witnesses[key]=(t,deepcopy(obj),rel.copy());return True
        old=self.witnesses.get(key)
        if old is None:return False
        return bool(name not in contacts.robot and np.linalg.norm(rel-old[2])<=.001
            and distance(obj,old[1])<=.001 and angle(obj,old[1])<=.03)

    def on_frame(self,frame):
        if self.error is not None:return
        try:self._frame(frame)
        except (KeyError,TypeError,ValueError,IndexError,OverflowError) as exc:self.error=str(exc)

    def _frame(self,frame):
        t,nt=number(frame['time_s']),number(frame['native_time_s'])
        step,baseline,dt=frame['native_step'],frame['baseline'],self.c['physics_dt_s']
        first=self.previous is None
        if type(step) is not int or step<0 or type(baseline) is not bool or baseline!=first or t<0 or nt<0 or abs(number(frame['physics_dt_s'])-dt)>EPS:raise ValueError('Invalid native baseline or clock')
        if first:
            if abs(t)>EPS or frame['contacts']:raise ValueError('Unscored contact-free reset required')
        elif step!=self.previous_step+1 or abs(t-self.previous_time-dt)>EPS or abs(nt-self.previous_native_time-dt)>2e-6:raise ValueError('Missing native samples')
        if set(frame['objects'])!=set(self.specs):raise ValueError('Native inventory differs')
        objects={name:state(frame['objects'][name]) for name in self.specs}
        if first:
            for name,obj in objects.items():
                reset=fixed(self.specs[name]['initial_pose_xyzw'])
                if distance(obj,reset)>.003 or angle(obj,reset)>.03:raise ValueError('Native reset differs')
            self.initial=deepcopy(objects)
        contacts=Contacts(frame['contacts'],self.paths,self.specs,t,self.limits['minimum_contact_impulse_ns'],self.limits['maximum_mating_gap_m'])
        if contacts.minimum_separation < -self.limits['maximum_penetration_m']:self.safety='physics_safety_stop'
        if not first and (contacts.floor or any(o['position_m'][2]<self.limits['fallen_height_m'] for o in objects.values())):self.failure='object_fell'
        mated,detail=self._mate(objects)
        hands=self._hands_on(frame['contacts'],t)
        distinct=any(a!=b for a in hands['sleeve'] for b in hands['peg'])
        moved=distance(objects['sleeve'],self.initial['sleeve'])>=self.p['minimum_sleeve_movement_m']
        touch=frozenset(('peg','sleeve')) in contacts.touching
        cooperative=bool(not first and distinct and moved and detail['aligned'] and -EPS<=detail['insertion_depth_m']<=self.p['maximum_insertion_depth_m']+EPS and not self.failure and not self.safety)
        depth=detail['insertion_depth_m']
        # Retraction removes retained progress even across a regrasp. Otherwise
        # repeated tiny cooperative pushes followed by unheld withdrawals can
        # accumulate an insertion that never actually occurred.
        if self.last_insertion_depth is not None:
            retreat=min(0.,depth-self.last_insertion_depth)
            self.cooperative_progress_m=max(0.,self.cooperative_progress_m+retreat)
        self.last_insertion_depth=depth
        if cooperative:
            # Credit only motion between consecutive frames with two real
            # hand contacts. Re-touching after single-arm insertion must not
            # retroactively label the intervening motion cooperative.
            if self.last_cooperative_depth is not None:
                self.cooperative_progress_m+=max(0.,depth-self.last_cooperative_depth)
            self.last_cooperative_depth=depth
            self.cooperation_s+=dt
            if not self.cooperation_confirmed and mated and touch and self.cooperation_s+EPS>=self.limits['cooperation_time_s'] and self.cooperative_progress_m+EPS>=self.p['assembly_motion_m']:
                self.cooperation_confirmed=True
                self.events.append(dict(id=1,kind='cooperative_insertion_confirmed',time_s=t,native_step=step))
        else:
            self.last_cooperative_depth=None
        contained={name:bool(np.all(np.abs(points(obj,self.specs[name]['local_bounds_m'])-self.p['collection_center_m'])<=np.asarray(self.p['collection_half_extents_m'])+self.limits['placement_tolerance_m'])) for name,obj in objects.items()}
        support_sleeve=self._supported('sleeve','collection_tray',objects,contacts,t,contained['sleeve'])
        support_peg=self._supported('peg','sleeve',objects,contacts,t,mated)
        self.history.append(deepcopy(objects))
        quiet=len(self.history)==self.history.maxlen and not contacts.robot and not self.failure and not self.safety
        if quiet:
            quiet=all(distance(obj,self.history[0][name])/self.limits['pose_window_s']<=self.limits['maximum_linear_speed_m_s'] and angle(obj,self.history[0][name])/self.limits['pose_window_s']<=self.limits['maximum_angular_speed_rad_s'] for name,obj in objects.items())
        goal=self.cooperation_confirmed and mated and all(contained.values()) and support_sleeve and support_peg
        if goal and quiet:
            if self.stable_anchor is None:self.stable_anchor=deepcopy(objects)
            steady=all(distance(obj,self.stable_anchor[name])<=self.limits['position_drift_m'] and angle(obj,self.stable_anchor[name])<=self.limits['orientation_drift_rad'] for name,obj in objects.items())
            if steady:self.stable_s+=dt
            else:self.stable_s=0.;self.stable_anchor=None
        else:self.stable_s=0.;self.stable_anchor=None
        detail.update(cooperation_confirmed=self.cooperation_confirmed,cooperation_time_s=self.cooperation_s,
            cooperative_insertion_progress_m=self.cooperative_progress_m,distinct_hands=distinct,
            hand_contacts={k:sorted(v) for k,v in hands.items()},sleeve_moved=moved,actual_mating_contact=touch,
            tray_containment=contained,sleeve_tray_support=support_sleeve,peg_sleeve_support=support_peg,released_quiet=bool(quiet))
        phase='settling' if goal else 'placing' if self.cooperation_confirmed else 'working'
        reason='reset_baseline_not_scored' if first else 'waiting_for_release_and_stability' if goal else 'cooperation_confirmed' if self.cooperation_confirmed else 'task_in_progress'
        self.current=dict(schema=STATE_SCHEMA,configuration_sha256=self.digest,task_id=self.task_id,
            time_s=t,native_time_s=nt,native_step=step,baseline=baseline,phase=phase,reason=reason,
            released_stable_s=self.stable_s,terminal_failure=self.failure,safety_stop=self.safety,detail=detail)
        self.previous,self.previous_time,self.previous_native_time,self.previous_step=objects,t,nt,step

    def on_observe(self):
        result=deepcopy(self.current or dict(schema=STATE_SCHEMA,configuration_sha256=self.digest))
        result.update(complete=self.current is not None and self.error is None,errors=[] if self.error is None else [self.error],events=deepcopy(self.events))
        return result


class SequenceEvaluator:
    """Score authenticated native receipts; missing evidence stays unknown."""
    def __init__(self, task_spec):
        self.c, self.digest = configuration(task_spec)
        self.rules = public_rules(self.c['task_id'], self.c['episode_seed'])['referee']
        self.reset()

    def reset(self):
        self.previous_time = self.previous_step = None
        self.invalid = self.failure = self.safety = None
        self.events, self.detail = [], {}
        self.stable_s = 0.
        self.success = self.incomplete = False
        self.phase = 'working'
        self.last_result = self._result('task_in_progress')

    def _result(self, reason):
        name = ('evaluator_invalid' if self.invalid else 'safety_stop' if self.safety else
                'task_failure' if self.failure else 'incomplete' if self.incomplete else
                'success' if self.success else 'ongoing')
        physical_state = 'ongoing' if self.incomplete else name
        result = {
            'evaluator': VERSION, 'task_id': self.c['task_id'],
            'status': 'unknown' if self.invalid else name, 'success': name == 'success',
            'reason': reason, 'evidence_status': 'unknown' if self.invalid else
                'complete' if self.previous_time is not None else 'pending',
            'terminal': name != 'ongoing',
            'physical_failure': bool(self.failure), 'physics_accepted': False,
            'native_manipulation_qualified': False,
            'irrecoverable_reason': self.failure or self.safety,
            'released_stable_s': self.stable_s, 'detail': deepcopy(self.detail),
            'referee': {'schema': 'physical_task_referee_state.v1', 'state': physical_state,
                'phase': self.phase, 'reason_codes': [reason], 'terminal': physical_state != 'ongoing',
                'recoverable': physical_state == 'ongoing'},
            'evidence_scope': 'Private complete native samples; scene probes and CPU fixtures are not robot success.',
        }
        if self.invalid:
            result['private_evidence_error'] = self.invalid
        self.last_result = deepcopy(result)
        return result

    def update(self, observation):
        if self.invalid or self.failure or self.safety or self.success or self.incomplete:
            return deepcopy(self.last_result)
        try:
            s = observation['task_state']
            if (s.get('schema') != STATE_SCHEMA or s.get('configuration_sha256') != self.digest
                    or s.get('task_id') != self.c['task_id'] or s.get('complete') is not True
                    or s.get('errors') != []):
                raise ValueError('Complete matching private native state required: '+str(s.get('errors')))
            t, step = number(s['time_s']), s['native_step']
            if abs(number(observation['time_s'])-t) > EPS or type(step) is not int or step < 0:
                raise ValueError('Invalid native observation clock')
            first = self.previous_time is None
            if first:
                if s['baseline'] is not True or abs(t) > EPS or s['events'] or s['released_stable_s'] != 0:
                    raise ValueError('Unscored baseline required before scoring')
            elif (s['baseline'] is not False or abs(t-self.previous_time-.02) > EPS
                  or step-self.previous_step != round(.02/self.c['physics_dt_s'])):
                raise ValueError('Consecutive control observations required')
            incoming = s['events']
            if not isinstance(incoming, list) or incoming[:len(self.events)] != self.events:
                raise ValueError('Native event history was rewritten')
            for event in incoming[len(self.events):]:
                if (event['id'] != len(self.events)+1 or event['kind'] != 'cooperative_insertion_confirmed'
                        or self.c['task_id'] != 'cooperative_insertion' or type(event['native_step']) is not int
                        or not self.previous_step < event['native_step'] <= step
                        or not self.previous_time < number(event['time_s']) <= t+EPS):
                    raise ValueError('Native event does not belong to this task interval')
                self.events.append(deepcopy(event))
            if s['phase'] not in self.rules['phases'] or s['reason'] not in self.rules['reason_descriptions']:
                raise ValueError('Undeclared public referee phase or reason')
            stable = number(s['released_stable_s'])
            if stable < 0 or stable > t+EPS:
                raise ValueError('Invalid native released dwell')
            if stable > self.stable_s+(0. if first else .02)+EPS:
                raise ValueError('Native dwell grew faster than complete elapsed observations')
            self.stable_s = stable
            self.phase, self.detail = s['phase'], deepcopy(s['detail'])
            self.previous_time, self.previous_step = t, step
            self.failure, self.safety = s['terminal_failure'], s['safety_stop']
            if self.failure not in (None, 'object_fell') or self.safety not in (None, 'physics_safety_stop'):
                raise ValueError('Unknown physical terminal reason')
            if self.failure or self.safety:
                return self._result(self.safety or self.failure)
            # An expired episode cannot be revived by a late success receipt.
            if (observation.get('budget_exhausted') is True or t > self.c['thresholds']['max_time_s']+EPS
                    or number(observation.get('action_count', 0)) > self.c['thresholds']['max_actions']):
                self.incomplete = True
                return self._result('budget_exhausted')
            self.success = bool(not first and stable+EPS >= self.c['thresholds']['stable_s'])
            if self.success and len(self.events) != 1:
                raise ValueError('Completion requires a prior real dual-arm insertion event')
            return self._result('success' if self.success else s['reason'])
        except (KeyError, TypeError, ValueError, IndexError, OverflowError) as exc:
            self.invalid = str(exc)
            self.failure = self.safety = None
            self.success = self.incomplete = False
            self.stable_s = 0.
            return self._result('native_evidence_unknown')

    def inspect_conditions(self, observation=None):
        return deepcopy(self.last_result)
