"""Semantic scoring controls using synthetic receipts, not physics qualification."""
from copy import deepcopy
import importlib.util
from itertools import product
import math
from pathlib import Path
import shutil
import sys
import types

import numpy as np
import pytest

from scenes.workrooms.mechanical_assembly.evaluation import SequenceEvaluator, read_native_contact_window
from scenes.workrooms.mechanical_assembly.rules import TASK_OBJECTS, public_rules, thresholds
from bench.protocol.task_referee import PublicContract, project_referee_state


def referee(result, kind, *, state, phase=None, reason=None):
    value=project_referee_state(result,public_rules(kind)['referee'])
    assert value['state']==state
    if phase is not None:assert value['phase']==phase
    if reason is not None:assert reason in value['reason_codes']
    assert value['terminal']==(state!='ongoing')
    assert value['recoverable']==(state=='ongoing')
    return value


def spec(kind):
    names=TASK_OBJECTS[kind]
    initial={n:{'position_m':[.1,i*.1,.8],'quaternion_wxyz':[1.,0.,0.,0.]} for i,n in enumerate(names)}
    actors={n:'/World/Objects/'+n for n in names}
    c={**thresholds(kind),'must_be_released':True,'robot_actor_prefix':'/World/Robot','initial_objects':initial,'actors':actors}
    task={'id':kind,'public_rules':public_rules(kind),'evaluation':c}
    if kind=='parts_sorting':
        task['goals']={n:{'center_xy_m':[.3,i*.1],'inner_half_extents_xy_m':[.065,.065],
            'floor_top_z_m':.8,'object_actor':actors[n],'support_collider':'/World/Fixtures/tray_'+n+'/collision_000'} for i,n in enumerate(names)}
        actors.update({'tray_'+n:task['goals'][n]['support_collider'] for n in names})
        c['required_support_pairs']=[[n,'tray_'+n] for n in names]
        c['geometry_vertices_local_m']={n:[list(x) for x in product((-.03,.03),(-.02,.02),(0.,.04))] for n in names}
    else:
        actors['fixture']='/World/Fixtures/'+('socket_workplate' if kind=='peg_insertion' else 'shaft_fixture')
        c['required_support_pairs']=[[n,'fixture'] for n in names] if kind=='peg_insertion' else [['spacer','fixture'],['wide_ring','spacer'],['top_ring','wide_ring']]
        c['target_bottom_centers_m']={n:[.3,i*.1 if kind=='peg_insertion' else 0.,.8+i*.03 if kind!='peg_insertion' else .8] for i,n in enumerate(names)}
        c['part_order']=list(names)
    return task


class Synthetic:
    def __init__(self,kind):
        self.task=spec(kind);self.e=SequenceEvaluator(self.task);self.index=-1
        self.positions={n:deepcopy(row['position_m']) for n,row in self.task['evaluation']['initial_objects'].items()}
        self.quaternions={n:[1.,0.,0.,0.] for n in self.positions}
        self.goal=(self.task['evaluation'].get('target_bottom_centers_m') or
                   {n:[*g['center_xy_m'],g['floor_top_z_m']] for n,g in self.task['goals'].items()})

    def sample(self,positions=None,*,support=True,held=False,upward=True,pulse=False,wrong_support=False):
        self.index+=1;first=self.index==0;start=max(0,self.index-1)*.02;time=self.index*.02
        positions=deepcopy(self.positions if positions is None else positions)
        objects={n:{'position_m':p,'quaternion_wxyz':self.quaternions[n],
                    'linear_velocity_m_s':((np.array(p)-self.positions[n])/.02).tolist(),
                    'angular_velocity_rad_s':[0.,0.,0.]} for n,p in positions.items()}
        self.positions=positions
        rows=[];actors=self.task['evaluation']['actors']
        def row(a,b,tick,impulse,vec):
            return {'time_s':start+tick*.0005,'actor0':a,'actor1':b,'point_count':1,
                'nonzero_impulse_normal_count':int(impulse>0),'impulse_magnitude_sum_ns':impulse,
                'impulse_vector_ns':vec,'normal_length_max_abs_error':0.,'min_separation_m':-.00001,'impulse_components':'normal_only'}
        if not first:
            for tick in ([40] if pulse else range(1,41)):
                if support:
                    for n,s in self.task['evaluation']['required_support_pairs']:
                        other='/World/Fixtures/unrelated_floor' if wrong_support else actors[s]
                        if s=='fixture' and not wrong_support:other+='/collision_000'
                        rows.append(row(actors[n],other,tick,.001,[0.,0.,.001 if upward else -.001]))
                if held:rows.append(row(actors[next(iter(self.positions))],'/World/Robot/right_index_DP',tick,0.,[0.,0.,0.]))
        receipt={'protocol':'native_rigid_contact_sampling_v1','complete':True,'physics_dt_s':.0005,
            'start_sim_time_s':start,'end_sim_time_s':time,'native_start_time_s':start+10.,'native_end_time_s':time+10.,
            'native_start_step_index':100+round(start/.0005),'native_end_step_index':100+round(time/.0005),'sampled_substeps':0 if first else 40}
        return {'time_s':time,'sim_time_s':time,'last_duration_s':0. if first else .02,
            'metrics':{'state_finite':True},'terminated':False,'objects':objects,
            'contacts':{'source':'PhysX stage-bound rigid contact tensor; normal impulses','errors':[],'sampling':receipt,'rigid':rows}}

    def emit(self,*args,**kwargs):return self.e.update(self.sample(*args,**kwargs))
    def moved(self,goal=None):
        assert self.emit()['reason']=='reset_baseline_not_scored'
        return self.move_to(self.goal if goal is None else goal)
    def move_to(self,target):
        start=deepcopy(self.positions)
        for alpha in np.linspace(0.,1.,61)[1:]:
            result=self.emit({n:((1-alpha)*np.array(start[n])+alpha*np.array(target[n])).tolist() for n in start})
            assert not result['reason'].startswith('invalid_observation:'),result
        assert self.emit()['release_stable_s']==0.  # First quiet endpoint anchors, not an entire interval.
        return self
    def dwell(self,**kwargs):
        for _ in range(25):result=self.emit(**kwargs)
        return result


@pytest.mark.parametrize('kind',list(TASK_OBJECTS))
def test_public_rules_describe_real_conditions_recovery_and_final_acceptance(kind):
    rules=public_rules(kind)
    contract=PublicContract.model_validate(rules['referee'])
    assert contract.task_id==kind
    assert contract.instruction_zh==rules['instruction_zh']
    assert '单臂' in contract.instruction_zh and '0.5秒' in contract.instruction_zh
    assert contract.thresholds==rules['thresholds']==thresholds(kind)
    assert rules['success']['action_order_scored'] is False
    assert rules['success']['reset_credit'] is False
    assert rules['success']['accepted_success_is_final'] is True
    assert rules['recovery']['automatic_forfeit'] is False
    assert rules['success']['final_layer_order']==(list(TASK_OBJECTS[kind]) if kind=='shaft_spacer_assembly' else None)
    # Consumers receive copies; editing a published rule cannot alter future episodes.
    rules['referee']['thresholds']['minimum_stable_s']=0.
    assert public_rules(kind)['referee']['thresholds']['minimum_stable_s']==.5


@pytest.mark.parametrize('kind',list(TASK_OBJECTS))
def test_referee_names_initial_and_first_quiet_endpoint_without_time_credit(kind):
    s=Synthetic(kind)
    referee(s.e.inspect_conditions(),kind,state='ongoing',phase='not_started',reason='not_sampled')
    referee(s.emit(),kind,state='ongoing',phase='reset',reason='reset_baseline_not_scored')
    s.move_to(s.goal)
    result=s.e.inspect_conditions()
    referee(result,kind,state='ongoing',phase='settling',reason='previous_endpoint_not_ready')
    assert result['release_stable_s']==0.
    referee(s.emit(),kind,state='ongoing',phase='settling',reason='stability_window_incomplete')


@pytest.mark.parametrize('kind',list(TASK_OBJECTS))
def test_supported_released_complete_half_second_is_required(kind):
    s=Synthetic(kind).moved()
    for _ in range(24):result=s.emit(pulse=True)
    assert not result['success']
    result=s.emit(pulse=True)
    assert result['success'] and result['release_stable_s']==pytest.approx(.5)
    referee(result,kind,state='success',phase='complete',reason='success')
    assert s.e.inspect_conditions({})==result
    counters=(s.e.previous_time,s.e.native_time,s.e.native_step,s.e.stable_s)
    # Terminal acceptance agrees with EpisodeSession: no later input is consumed.
    assert s.emit(support=False)==result
    assert s.e.update(None)==result
    assert counters==(s.e.previous_time,s.e.native_time,s.e.native_step,s.e.stable_s)
    copy=s.e.update({});copy['referee']['reason_codes'].clear();copy['objects'].clear()
    assert s.e.inspect_conditions()==result


@pytest.mark.parametrize('kind',list(TASK_OBJECTS))
@pytest.mark.parametrize('defect',['held','no_support','downward','wrong_support'])
def test_pose_alone_or_wrong_support_cannot_pass(kind,defect):
    s=Synthetic(kind).moved()
    result=s.dwell(held=defect=='held',support=defect!='no_support',upward=defect!='downward',wrong_support=defect=='wrong_support')
    assert not result['success'] and result['release_stable_s']==0.
    referee(result,kind,state='ongoing',phase='release' if defect=='held' else 'support',
        reason='robot_contact_present' if defect=='held' else 'required_support_missing')
    assert s.dwell()['success']  # Ordinary physical placement errors remain recoverable.


def test_wrong_stack_order_and_outside_socket_do_not_pass():
    s=Synthetic('shaft_spacer_assembly');wrong=deepcopy(s.goal)
    wrong['spacer'],wrong['top_ring']=wrong['top_ring'],wrong['spacer']
    result=s.moved(wrong).dwell()
    referee(result,s.task['id'],state='ongoing',phase='placement',reason='bottom_height')
    assert s.move_to(s.goal).dwell()['success']
    s=Synthetic('peg_insertion');wrong=deepcopy(s.goal);wrong['peg_0'][0]+=.02
    result=s.moved(wrong).dwell()
    referee(result,s.task['id'],state='ongoing',phase='placement',reason='axis_offset')
    assert s.move_to(s.goal).dwell()['success']


def test_sorting_checks_whole_rotated_geometry_and_matching_tray():
    s=Synthetic('parts_sorting');wrong=deepcopy(s.goal);wrong['block'][0]+=.05
    assert not s.moved(wrong).dwell()['success']  # Center is inside; outside corner fails.
    assert s.e.last_result['objects']['block']['maximum_tray_boundary_excess_m']>0
    referee(s.e.last_result,s.task['id'],state='ongoing',phase='placement',reason='outside_matching_tray')
    assert s.move_to(s.goal).dwell()['success']
    s=Synthetic('parts_sorting');wrong=deepcopy(s.goal)
    wrong['block'],wrong['ring']=wrong['ring'],wrong['block']
    assert not s.moved(wrong).dwell()['success']
    s=Synthetic('parts_sorting').moved();s.quaternions['block']=[math.cos(.2),0.,math.sin(.2),0.]
    assert not s.dwell()['success']  # Tilted box penetrates floor at actual extreme vertex.


@pytest.mark.parametrize('damage',['count','duplicate','offgrid','clock','nan','quaternion','impulse','normal','penetration','missing','dropped_pose','source'])
def test_invalid_evidence_fails_closed_and_latches(damage):
    s=Synthetic('peg_insertion');s.emit();obs=s.sample()
    if damage=='count':obs['contacts']['sampling']['sampled_substeps']=39
    elif damage=='duplicate':obs['contacts']['rigid'].append(deepcopy(obs['contacts']['rigid'][0]))
    elif damage=='offgrid':obs['contacts']['rigid'][0]['time_s']+=.0001
    elif damage=='clock':obs['contacts']['sampling']['native_start_step_index']+=1
    elif damage=='nan':obs['objects']['peg_0']['position_m'][0]=float('nan')
    elif damage=='quaternion':obs['objects']['peg_0']['quaternion_wxyz']=[2.,0.,0.,0.]
    elif damage=='impulse':obs['contacts']['rigid'][0]['impulse_magnitude_sum_ns']=float('inf')
    elif damage=='normal':obs['contacts']['rigid'][0]['normal_length_max_abs_error']=.1
    elif damage=='penetration':obs['contacts']['rigid'][0]['min_separation_m']=-.01
    elif damage=='missing':del obs['objects']['peg_0']
    elif damage=='dropped_pose':obs['time_s']+=.02;obs['sim_time_s']+=.02
    else:obs['contacts']['source']='synthetic no receipt'
    result=s.e.update(obs)
    assert result['reason'].startswith('invalid_observation:') and not result['success']
    referee(result,s.task['id'],state='safety_stop' if damage=='penetration' else 'evaluator_invalid',
        phase='safety_stop' if damage=='penetration' else 'invalid',
        reason='contact_penetration_bound' if damage=='penetration' else 'invalid_evidence')
    assert s.emit()==result


def test_actor_order_sign_and_path_boundaries_are_respected():
    s=Synthetic('peg_insertion').moved()
    for _ in range(25):
        obs=s.sample()
        for row in obs['contacts']['rigid']:
            row['actor0'],row['actor1']=row['actor1'],row['actor0']
            row['impulse_vector_ns'][2]*=-1
        result=s.e.update(obs)
    assert result['success']
    s=Synthetic('peg_insertion').moved()
    for _ in range(25):
        obs=s.sample()
        for row in obs['contacts']['rigid']:row['actor1']=row['actor1'].replace('socket_workplate/','socket_workplate_imposter/')
        result=s.e.update(obs)
    assert not result['success']
    referee(result,s.task['id'],state='ongoing',phase='support',reason='required_support_missing')


@pytest.mark.parametrize('kind',list(TASK_OBJECTS))
@pytest.mark.parametrize('magnitude',[0.,1.e-15])
def test_zero_or_tiny_impulses_cannot_accumulate_absolute_vector_slack(kind,magnitude):
    s=Synthetic(kind).moved()
    obs=s.sample()
    for row in obs['contacts']['rigid']:
        row['impulse_magnitude_sum_ns']=magnitude
        row['nonzero_impulse_normal_count']=int(magnitude>0)
        row['impulse_vector_ns']=[0.,0.,9.e-10]
    result=s.e.update(obs)
    assert not result['success'] and result['reason'].startswith('invalid_observation:')
    assert s.emit()==result


def test_zero_magnitude_requires_zero_active_normal_count():
    s=Synthetic('peg_insertion');s.emit();obs=s.sample()
    row=obs['contacts']['rigid'][0]
    row.update(impulse_magnitude_sum_ns=0.,impulse_vector_ns=[0.,0.,0.],nonzero_impulse_normal_count=1)
    result=s.e.update(obs)
    assert not result['success'] and result['reason'].startswith('invalid_observation:')


def test_contact_vector_magnitude_overflow_fails_closed():
    s=Synthetic('peg_insertion');s.emit();obs=s.sample()
    row=obs['contacts']['rigid'][0]
    row.update(impulse_magnitude_sum_ns=sys.float_info.max,impulse_vector_ns=[sys.float_info.max]*3)
    result=s.e.update(obs)
    assert not result['success'] and result['reason'].startswith('invalid_observation:')


@pytest.mark.parametrize('kind',list(TASK_OBJECTS))
@pytest.mark.parametrize('disturbance',[{'support':False},{'held':True}])
def test_pre_success_disturbance_resets_shared_stability_window(kind,disturbance):
    s=Synthetic(kind).moved()
    for _ in range(20):s.emit()
    result=s.emit(**disturbance)
    assert result['release_stable_s']==0.
    referee(result,kind,state='ongoing')
    for _ in range(24):result=s.emit()
    assert not result['success']
    assert result['release_stable_s']==pytest.approx(.48)
    assert s.emit()['success']


def test_drift_clears_stability_then_fixed_pose_can_recover():
    s=Synthetic('peg_insertion').moved()
    saw_drift=False
    for i in range(25):
        p=deepcopy(s.goal);p['peg_0'][0]+=(i+1)*.000015
        result=s.emit(p)
        if 'stability_drift' in result['referee']['reason_codes']:
            saw_drift=True
            referee(result,s.task['id'],state='ongoing',phase='settling',reason='stability_drift')
            assert result['release_stable_s']==0.
    assert saw_drift
    assert not result['success']  # Quiet instantaneous speed, but dwell spread too large.
    assert s.dwell()['success']


@pytest.mark.parametrize('kind',list(TASK_OBJECTS))
def test_motion_before_success_requires_a_fresh_full_quiet_window(kind):
    s=Synthetic(kind).moved()
    for _ in range(20):s.emit()
    positions=deepcopy(s.positions)
    positions[next(iter(positions))][0]+=.0002
    moving=s.emit(positions)
    referee(moving,kind,state='ongoing',phase='settling',reason='still_moving')
    assert moving['release_stable_s']==0.
    # Position is still legal, but the interval starting in motion gets no credit.
    stopped=s.emit()
    referee(stopped,kind,state='ongoing',phase='settling',reason='previous_endpoint_not_ready')
    assert stopped['release_stable_s']==0.
    for _ in range(24):result=s.emit()
    assert not result['success'] and result['release_stable_s']==pytest.approx(.48)
    assert s.emit()['success']


@pytest.mark.parametrize('motion',['pose_jump','native_velocity','angular_jump'])
def test_finite_overspeed_is_a_safety_stop_not_a_placement_loss(motion):
    s=Synthetic('peg_insertion').moved();obs=s.sample()
    row=obs['objects']['peg_0']
    if motion=='pose_jump':row['position_m'][0]+=.03
    elif motion=='native_velocity':row['linear_velocity_m_s'][0]=1.1
    else:row['quaternion_wxyz']=[math.cos(.25),0.,math.sin(.25),0.]
    result=s.e.update(obs)
    referee(result,s.task['id'],state='safety_stop',phase='safety_stop',reason='object_speed_bound')
    assert result['reason']=='invalid_observation:Object motion speed bound exceeded'
    assert s.emit()==result
    s.e.reset()
    referee(s.e.inspect_conditions(),s.task['id'],state='ongoing',phase='not_started',reason='not_sampled')


@pytest.mark.parametrize('kind',list(TASK_OBJECTS))
@pytest.mark.parametrize('bad_reset',[True,False])
def test_complete_reset_identity_precedes_physical_safety_classification(kind,bad_reset):
    s=Synthetic(kind);obs=s.sample()
    first,last=TASK_OBJECTS[kind][0],TASK_OBJECTS[kind][-1]
    obs['objects'][first]['linear_velocity_m_s']=[2.,0.,0.]
    if bad_reset:obs['objects'][last]['position_m'][0]+=.01
    result=s.e.update(obs)
    referee(result,kind,state='evaluator_invalid' if bad_reset else 'safety_stop',
        phase='invalid' if bad_reset else 'safety_stop',
        reason='invalid_evidence' if bad_reset else 'object_speed_bound')
    assert ('Initial pose differs' in result['reason']) is bad_reset
    assert not result['success'] and result['progress']==0.
    assert s.e.update(None)==result


def test_precompleted_reset_probes_and_rule_mutations_are_rejected():
    s=Synthetic('peg_insertion')
    assert s.emit(s.goal)['reason'].startswith('invalid_observation:Initial pose')
    task=spec('shaft_spacer_assembly');task['evaluation']['part_order'].reverse()
    with pytest.raises(ValueError,match='order'):SequenceEvaluator(task)
    task=spec('peg_insertion');task['evaluation']['minimum_stable_s']=0.
    with pytest.raises(ValueError,match='threshold'):SequenceEvaluator(task)
    task=spec('peg_insertion');task['engineering_probe']={'reset_only':True}
    with pytest.raises(ValueError,match='probes'):SequenceEvaluator(task)


def test_contact_helper_is_task_independent_and_frozen_package_loads(tmp_path):
    s=Synthetic('peg_insertion');obs=s.sample()
    rows,step,time=read_native_contact_window(obs,time=0.,first=True,physics_dt_s=.0005,control_period_s=.02)
    assert rows==[] and step==100 and time==10.
    package=types.ModuleType('_mechanical_frozen_test');package.__path__=[str(tmp_path)]
    sys.modules[package.__name__]=package
    source=Path(__file__).parents[1]/'scenes/workrooms/mechanical_assembly'
    for name,target in [('rules.py','rules.py'),('evaluation.py','sequence_evaluator.py'),
                        ('native_evidence.py','native_evidence.py'),('referee.py','referee.py')]:shutil.copy2(source/name,tmp_path/target)
    module_spec=importlib.util.spec_from_file_location(package.__name__+'.sequence_evaluator',tmp_path/'sequence_evaluator.py')
    module=importlib.util.module_from_spec(module_spec);module_spec.loader.exec_module(module)
    for kind in TASK_OBJECTS:assert module.SequenceEvaluator(spec(kind)).inspect_conditions()['success'] is False


def test_authored_task_builders_include_matching_frozen_contracts():
    from scenes.workrooms.mechanical_assembly.tasks.peg_insertion.task import build_task_inputs as peg
    from scenes.workrooms.mechanical_assembly.tasks.shaft_spacer_assembly.task import build_task_inputs as shaft
    from scenes.workrooms.mechanical_assembly.tasks.parts_sorting.task import build_task_inputs as sorting
    for builder in (peg,shaft,sorting):
        inputs=builder();e=SequenceEvaluator(inputs['task'])
        assert set(e.expected)=={obj['id'] for obj in inputs['scene']['objects']}
        assert not inputs['task']['physics_accepted']
    inputs=sorting();e=SequenceEvaluator(inputs['task']);name='cylinder'
    assert len(e.vertices[name])>8
    goal=inputs['task']['goals'][name]
    for yaw in (0.,math.pi/4,math.pi/2):
        row={'position_m':[goal['center_xy_m'][0]+.033,goal['center_xy_m'][1],goal['expected_center_z_m']],
             'quaternion_wxyz':[math.cos(yaw/2),0.,0.,math.sin(yaw/2)]}
        assert e._geometry(name,e._state(row,velocities=False))['geometry']
