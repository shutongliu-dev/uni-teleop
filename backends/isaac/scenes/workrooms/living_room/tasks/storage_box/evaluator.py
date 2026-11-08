"""Private native body/contact readback; no joint drives or scene mutations."""
from copy import deepcopy
from collections import deque
import itertools
import math
from .rules import THRESHOLDS, public_rules, digest

STATE_SCHEMA = 'living_storage_box_native_state.v1'
EPS = 1.e-8


def number(value):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError('Finite scalar required')
    return float(value)


def vector(value, length):
    if len(value) != length: raise ValueError('Wrong vector length')
    return [number(x) for x in value]


def norm(value):
    return math.sqrt(sum(x*x for x in value))


def rotation(q):
    w,x,y,z = vector(q,4)
    if abs(norm(q)-1.) > 1.e-4: raise ValueError('Unit quaternion required')
    return [[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
            [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
            [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]]


def relative_rotation(a,b):
    ra,rb=rotation(a['quaternion_wxyz']),rotation(b['quaternion_wxyz'])
    return [[sum(ra[k][i]*rb[k][j] for k in range(3)) for j in range(3)] for i in range(3)]


def angles(objects):
    lid=relative_rotation(objects['box'],objects['lid'])
    result={'lid':-math.degrees(math.atan2(lid[2][1],lid[1][1]))}
    for name in ('left_catch','right_catch'):
        r=relative_rotation(objects['box'],objects[name])
        result[name]=math.degrees(math.atan2(r[1][0],r[0][0]))*(1 if name=='left_catch' else -1)
    return result


def corners(body,size):
    r=rotation(body['quaternion_wxyz']); p=body['position_m']
    return [[p[i]+sum(r[i][j]*v[j] for j in range(3)) for i in range(3)]
            for v in itertools.product(*[(-s/2,s/2) for s in size])]


def configuration(task):
    c=deepcopy(task['evaluation'])
    if task.get('id')!='storage_box' or c.get('schema')!='living_storage_box_evaluation.v1':
        raise ValueError('Wrong storage task')
    frozen=c.pop('contract_sha256')
    if digest(c)!=frozen or task['public_rules']!=public_rules(asset_metadata=task.get('asset_metadata')) or c['thresholds']!=THRESHOLDS:
        raise ValueError('Frozen storage contract differs')
    if c['public_rules_sha256']!=digest(task['public_rules']) or c['physics_dt_s']!=.0005:
        raise ValueError('Rule/sampling identity differs')
    c['contract_sha256']=frozen
    return c,frozen


class NativeTaskObserver:
    def __init__(self,task):
        self.c,self.digest=configuration(task); self.error=None; self.current=None
        self.previous=None; self.previous_step=None; self.previous_time=None; self.previous_native_time=None
        self.opened=False; self.transferred=False; self.hand_seen=False; self.stable_s=0.; self.anchor=None
        self.failure=None; self.safety=None; self.history=[]

    def on_frame(self,frame):
        if self.error: return
        try: self._frame(frame)
        except (KeyError,TypeError,ValueError,IndexError,OverflowError) as exc: self.error=str(exc)

    def _frame(self,frame):
        c=self.c; th=c['thresholds']; t=number(frame['time_s']); nt=number(frame['native_time_s'])
        step=frame['native_step']; first=self.previous is None
        if type(step) is not int or step<0 or type(frame['baseline']) is not bool or frame['baseline']!=first:
            raise ValueError('A single native reset baseline is required')
        if number(frame['physics_dt_s'])!=c['physics_dt_s']: raise ValueError('Wrong native timestep')
        if first:
            if abs(t)>EPS or frame['contacts']: raise ValueError('Reset precedes stepping and contacts')
        elif (step!=self.previous_step+1 or abs(t-self.previous_time-c['physics_dt_s'])>EPS
              or abs(nt-self.previous_native_time-c['physics_dt_s'])>2.e-6):
            raise ValueError('Missing or repeated native substeps')
        objects={}
        for name in c['actors']:
            row=frame['objects'][name]
            objects[name]={k:vector(row[k],4 if k=='quaternion_wxyz' else 3) for k in
                ('position_m','quaternion_wxyz','linear_velocity_m_s','angular_velocity_rad_s')}
            rotation(objects[name]['quaternion_wxyz'])
        q=angles(objects)
        if first and (any(abs(objects[n]['position_m'][i]-c['initial_positions_m'][n][i])>.003
                         for n in objects for i in range(3)) or any(abs(a)>2. for a in q.values())):
            raise ValueError('Baseline differs from closed canonical storage reset')
        if not -3.<=q['lid']<=104. or any(not -3.<=q[n]<=104. for n in ('left_catch','right_catch')):
            raise ValueError('Mechanism left its physical joint limits')
        hand=False; support=False; seen=set(); own=set(c['actors'].values())
        for row in frame['contacts']:
            a,b=row['actor0'],row['actor1']; pair=tuple(sorted((a,b)))
            if a==b or pair in seen or abs(number(row['time_s'])-t)>EPS: raise ValueError('Invalid contact pair/stamp')
            seen.add(pair)
            if type(row['point_count']) is not int or row['point_count']<1: raise ValueError('Invalid contact count')
            impulse=number(row['impulse_magnitude_sum_ns'])
            if impulse<0: raise ValueError('Negative impulse magnitude')
            if (a in own or b in own) and number(row['min_separation_m']) < -th['maximum_penetration_m']:
                self.safety='physics_safety_stop'
            for actor,other in ((a,b),(b,a)):
                if actor in own and other.startswith(c['robot_actor_prefix']): hand=True
                if actor==c['actors']['item'] and other==c['tray_floor_path'] and impulse>1.e-9: support=True
        self.hand_seen=self.hand_seen or (hand and not first)
        unlocked=all(q[n]>=th['catch_open_deg'] for n in ('left_catch','right_catch'))
        if not first and unlocked and q['lid']>=th['lid_open_deg'] and not self.opened:
            self.opened=True; self.history.append({'event':'opened','time_s':t,'native_step':step})
        pts=corners(objects['item'],c['item_size_m']); lo,hi=c['tray_inside_xy_m']
        inside=all(lo[i]-.001<=p[i]<=hi[i]+.001 for p in pts for i in (0,1))
        bottom=abs(min(p[2] for p in pts)-c['tray_floor_z_m'])<=th['tray_bottom_tolerance_m']
        collected=inside and bottom and support
        if self.opened and collected and not self.transferred:
            self.transferred=True; self.history.append({'event':'transferred','time_s':t,'native_step':step})
        closed=abs(q['lid'])<=th['lid_closed_deg'] and all(abs(q[n])<=th['catch_closed_deg'] for n in ('left_catch','right_catch'))
        stationary=all(norm(b['linear_velocity_m_s'])<=th['maximum_linear_speed_m_s'] and
            norm(b['angular_velocity_rad_s'])<=th['maximum_angular_speed_rad_s'] for b in objects.values())
        if not first:
            for name,b in objects.items():
                stationary &= norm([b['position_m'][i]-self.previous[name]['position_m'][i] for i in range(3)])/c['physics_dt_s'] <= th['maximum_linear_speed_m_s']
        physical_complete=self.opened and self.transferred and collected and closed and stationary and not hand and not first
        if physical_complete:
            if self.anchor is None: self.anchor=deepcopy(objects)
            physical_complete=all(norm([b['position_m'][i]-self.anchor[n]['position_m'][i] for i in range(3)])<=th['stable_position_range_m']
                and 2*math.acos(min(1.,abs(sum(a*v for a,v in zip(b['quaternion_wxyz'],self.anchor[n]['quaternion_wxyz'])))))<=th['stable_orientation_range_rad']
                for n,b in objects.items())
        if physical_complete: self.stable_s+=c['physics_dt_s']
        else: self.stable_s=0.; self.anchor=None
        if min(objects[n]['position_m'][2] for n in objects)<.73: self.failure='object_fell'
        phase=('opening' if not self.opened else 'transfer' if not self.transferred or not collected else
               'closing' if not closed else 'settling')
        self.current={'schema':STATE_SCHEMA,'configuration_sha256':self.digest,'task_id':'storage_box',
            'time_s':t,'native_time_s':nt,'native_step':step,'baseline':first,'phase':phase,
            'angles_deg':q,'opened':self.opened,'transferred':self.transferred,'hand_operation_seen':self.hand_seen,
            'collected':collected,'closed':closed,'stationary':stationary,'hand_contact':hand,
            'physical_released_stable_s':self.stable_s,
            'released_stable_s':self.stable_s if self.hand_seen else 0.,'events':deepcopy(self.history),
            'terminal_failure':self.failure,'safety_stop':self.safety}
        self.previous=objects; self.previous_time=t; self.previous_native_time=nt; self.previous_step=step

    def on_observe(self):
        result=deepcopy(self.current or {'schema':STATE_SCHEMA,'configuration_sha256':self.digest})
        result.update(complete=self.current is not None and self.error is None,errors=[] if self.error is None else [self.error])
        return result


class SequenceEvaluator:
    def __init__(self,task):
        self.c,self.digest=configuration(task); self.reset()

    def reset(self):
        self.previous_time=None; self.previous_step=None; self.last_result=None; self.invalid=None

    def update(self,obs):
        if self.last_result and self.last_result['terminal']: return deepcopy(self.last_result)
        state='ongoing';phase='opening';reason='reset_baseline_not_scored';success=False;stable=0.
        try:
            s=obs['task_state'];t=number(s['time_s']);step=s['native_step'];first=self.previous_time is None
            if s.get('schema')!=STATE_SCHEMA or s.get('configuration_sha256')!=self.digest or not s.get('complete') or s.get('errors')!=[]:
                raise ValueError('Incomplete matching native state required')
            if abs(number(obs['time_s'])-t)>EPS or type(step) is not int: raise ValueError('Observation identity differs')
            if first:
                if not s['baseline'] or abs(t)>EPS or s['events'] or s['released_stable_s']!=0: raise ValueError('Unscored baseline required')
            elif s['baseline'] or abs(t-self.previous_time-.02)>EPS or step-self.previous_step!=40:
                raise ValueError('Consecutive control observations required')
            self.previous_time=t;self.previous_step=step
            stable=number(s['released_stable_s']);phase=s['phase'];reason=phase
            if stable<0 or stable>t+EPS: raise ValueError('Invalid stable dwell')
            if s['safety_stop']: state='safety_stop';phase='failed';reason=s['safety_stop']
            elif s['terminal_failure']: state='task_failure';phase='failed';reason=s['terminal_failure']
            elif stable+EPS>=self.c['thresholds']['stable_s'] and not first:
                if not (s['hand_operation_seen'] and s['opened'] and s['transferred'] and s['collected'] and s['closed'] and not s['hand_contact']):
                    raise ValueError('Stable success differs from actual mechanism conditions')
                state='success';success=True;phase=reason='completed'
            elif obs.get('budget_exhausted') or t>=self.c['thresholds']['max_time_s'] or obs.get('action_count',0)>self.c['thresholds']['max_actions']:
                phase='incomplete';reason='incomplete_budget'
            elif first: reason='reset_baseline_not_scored'
            elif s['physical_released_stable_s']>=.3 and not s['hand_operation_seen']: reason='robot_operation_required'
        except (KeyError,ValueError,TypeError,IndexError,OverflowError) as exc:
            state='evaluator_invalid';phase='evidence_invalid';reason='native_evidence_unknown';self.invalid=str(exc)
        self.last_result={'evaluator':'living_storage_box_evaluator.v1','task_id':'storage_box','success':success,
            'status':'incomplete' if reason=='incomplete_budget' else state,'terminal':state!='ongoing' or reason=='incomplete_budget',
            'reason':'budget_exhausted' if reason=='incomplete_budget' else reason,
            'released_stable_s':stable,'physics_accepted':False,'private_evidence_error':self.invalid,
            'referee':{'schema':'physical_task_referee_state.v1','state':state,'phase':phase,'reason_codes':[reason],
                       'terminal':state!='ongoing','recoverable':state=='ongoing'}}
        return deepcopy(self.last_result)

    def inspect_conditions(self,observation=None): return deepcopy(self.last_result)
