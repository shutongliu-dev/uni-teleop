"""Synthetic native-observation positive/negative controls; no physical acceptance."""
from copy import deepcopy
import math

import numpy as np
import pytest

from scenes.workrooms.mechanical_assembly.tasks.fastener_assembly.evaluator import FastenerEvaluator
from scenes.workrooms.mechanical_assembly.tasks.fastener_assembly.task import build_task_inputs


@pytest.fixture(scope='module')
def inputs():
    return build_task_inputs()


def yaw_product(q, theta):
    c,s=math.cos(theta/2),math.sin(theta/2)
    w,x,y,z=q
    return [c*w-s*z, c*x-s*y, c*y+s*x, c*z+s*w]


class Synthetic:
    def __init__(self, task):
        self.task=task
        self.e=FastenerEvaluator(task)
        self.index=0
        self.previous=None
        self.angle=0.

    def sample(self, turn=0., *, advance=None, tool_turn=None, drive=False, support=True,
               tool_bolt=True, direct_bolt=False, axis=0., tilt=0., held=False):
        first=self.previous is None
        start=self.index*.02
        if not first:self.index+=1
        time=self.index*.02
        advance=.003*turn/(2*math.pi) if advance is None else advance
        objects={}
        for name, initial in self.task['evaluation']['initial_objects'].items():
            row=deepcopy(initial)
            yaw=-turn if name=='bolt' or tool_turn is None else -tool_turn
            row['quaternion_wxyz']=yaw_product(row['quaternion_wxyz'],yaw)
            row['position_m'][2]-=advance
            if name=='bolt':
                row['position_m'][0]+=axis
                if tilt:
                    row['quaternion_wxyz']=[math.cos(tilt/2),math.sin(tilt/2),0.,0.]
            previous=row if first else self.previous['objects'][name]
            row['linear_velocity_m_s']=((np.array(row['position_m'])-previous['position_m'])/(.02 if not first else 1.)).tolist()
            row['angular_velocity_rad_s']=[0.,0.,0. if first else -(turn-self.angle)/.02]
            objects[name]=row
        actors=self.task['evaluation']['actors']
        pairs=[]
        if support:pairs.append((actors['bolt'],actors['threaded_fixture']))
        if tool_bolt:pairs.append((actors['screwdriver'],actors['bolt']))
        if drive or held:pairs.append((actors['screwdriver'],'/World/Robot/right_index_DP'))
        if direct_bolt:pairs.append((actors['bolt'],'/World/Robot/right_index_DP'))
        contacts=[]
        if not first:
            for tick in range(1,41):
                for a,b in pairs:
                    contacts.append({'time_s':start+tick*.0005,'actor0':a,'actor1':b,
                        'point_count':1,'nonzero_impulse_normal_count':1,
                        'impulse_magnitude_sum_ns':.001,'impulse_vector_ns':[0.,0.,.001],
                        'normal_length_max_abs_error':0.,'min_separation_m':-.00001,
                        'impulse_components':'normal_only'})
        sampling={'protocol':'native_rigid_contact_sampling_v1','physics_dt_s':.0005,
            'start_sim_time_s':start,'end_sim_time_s':time,'native_start_time_s':start,
            'native_end_time_s':time,'native_start_step_index':round(start/.0005),
            'native_end_step_index':round(time/.0005),'sampled_substeps':0 if first else 40,'complete':True}
        obs={'time_s':time,'sim_time_s':time,'last_duration_s':0. if first else .02,
             'metrics':{'state_finite':True},'terminated':False,'objects':objects,
             'contacts':{'source':'PhysX stage-bound rigid contact tensor; normal impulses',
                         'errors':[],'rigid':contacts,'sampling':sampling}}
        self.previous=deepcopy(obs);self.angle=turn
        return obs

    def emit(self,*args,**kwargs):return self.e.update(self.sample(*args,**kwargs))

    def settled(self):
        self.emit()
        for _ in range(25):result=self.emit()
        assert result['settled_baseline_recorded']
        return self

    def rotate(self, *, final=math.pi/6, advance_scale=1., drive=True, **kwargs):
        for value in np.linspace(0.,final,151)[1:]:
            result=self.emit(float(value),advance=float(.003*value/(2*math.pi)*advance_scale),drive=drive,**kwargs)
        return result


def test_actual_bolt_lead_and_released_stability_succeed(inputs):
    s=Synthetic(inputs['task']).settled()
    result=s.rotate()
    assert not result['success'] and result['bolt_turn_rad']==pytest.approx(math.pi/6)
    assert result['additional_axial_advance_m']==pytest.approx(.000250)
    for _ in range(24):result=s.emit(math.pi/6)
    assert not result['success']
    result=s.emit(math.pi/6)
    assert result['success'] and result['release_stable_s']==pytest.approx(.5)
    assert result['contact_backed_bolt_turn_rad']==pytest.approx(math.pi/6)
    before=deepcopy(result)
    assert s.e.inspect_conditions(s.previous)==before
    assert not inputs['task']['physics_accepted']


@pytest.mark.parametrize('mode',['angle_only','advance_only','wrong_lead','wrong_direction','no_tool_contact','tool_only'])
def test_motion_or_contact_shortcuts_never_complete(inputs,mode):
    s=Synthetic(inputs['task']).settled()
    if mode=='tool_only':
        for value in np.linspace(0.,math.pi/6,151)[1:]:s.emit(0.,tool_turn=float(value),drive=True)
    elif mode=='advance_only':
        for value in np.linspace(0.,.000250,151)[1:]:s.emit(0.,advance=float(value),drive=True)
    else:
        s.rotate(advance_scale=0. if mode=='angle_only' else 2. if mode=='wrong_lead' else 1.,
                 final=-math.pi/6 if mode=='wrong_direction' else math.pi/6,
                 drive=mode!='no_tool_contact')
    for _ in range(30):result=s.emit(s.angle,advance=s.task['evaluation']['initial_objects']['bolt']['position_m'][2]-s.previous['objects']['bolt']['position_m'][2])
    assert not result['success'],mode


def test_contact_only_and_precompleted_reset_get_no_credit(inputs):
    s=Synthetic(inputs['task']).settled()
    for _ in range(30):result=s.emit(drive=True)
    assert result['progress']==0. and not result['success']
    wrong=Synthetic(inputs['task'])
    result=wrong.emit(math.pi/6)
    assert result['reason'].startswith('invalid_observation:Initial pose')


@pytest.mark.parametrize('failure',['held','missing_support','missing_tool_bolt','direct_bolt','axis_escape'])
def test_goal_is_insufficient_without_released_supported_legal_state(inputs,failure):
    s=Synthetic(inputs['task']).settled();s.rotate()
    kwargs={'held':failure=='held','support':failure!='missing_support',
            'tool_bolt':failure!='missing_tool_bolt','direct_bolt':failure=='direct_bolt',
            'axis':.002 if failure=='axis_escape' else 0.}
    for _ in range(30):result=s.emit(math.pi/6,**kwargs)
    assert not result['success']


@pytest.mark.parametrize('damage',['missing_tick','duplicate_pair','off_grid','bad_clock','nan_pose','bad_quaternion','nonfinite_impulse','penetration','missing_object'])
def test_incomplete_or_invalid_native_evidence_stops_without_task_failure(inputs,damage):
    s=Synthetic(inputs['task']);s.emit()
    obs=s.sample()
    if damage=='missing_tick':obs['contacts']['sampling']['sampled_substeps']=39
    elif damage=='duplicate_pair':obs['contacts']['rigid'].append(deepcopy(obs['contacts']['rigid'][0]))
    elif damage=='off_grid':obs['contacts']['rigid'][0]['time_s']+=.0001
    elif damage=='bad_clock':obs['contacts']['sampling']['native_start_step_index']=1
    elif damage=='nan_pose':obs['objects']['bolt']['position_m'][0]=float('nan')
    elif damage=='bad_quaternion':obs['objects']['bolt']['quaternion_wxyz']=[2.,0.,0.,0.]
    elif damage=='nonfinite_impulse':obs['contacts']['rigid'][0]['impulse_magnitude_sum_ns']=float('inf')
    elif damage=='penetration':obs['contacts']['rigid'][0]['min_separation_m']=-.01
    else:del obs['objects']['bolt']
    result=s.e.update(obs)
    assert result['reason'].startswith('invalid_observation:') and not result['success']
    assert result['referee']['state'] == ('safety_stop' if damage == 'penetration' else 'evaluator_invalid')
    assert result['irrecoverable_reason'] is None
    assert s.emit()['reason']==result['reason']


def test_passive_settlement_requires_complete_support_not_one_contact(inputs):
    s=Synthetic(inputs['task']);s.emit()
    for _ in range(30):
        obs=s.sample();obs['contacts']['rigid']=obs['contacts']['rigid'][:2]
        result=s.e.update(obs)
    assert not result['settled_baseline_recorded']


def test_angular_alias_jump_is_not_accepted(inputs):
    s=Synthetic(inputs['task']).settled()
    result=s.emit(math.pi/6,drive=True)
    assert result['reason'].startswith('invalid_observation:object motion speed bound')


def test_passive_twenty_micron_sink_is_excluded_from_scored_advance(inputs):
    s = Synthetic(inputs['task'])
    s.emit()
    for _ in range(26): result = s.emit(advance=.00002)
    assert result['settled_baseline_recorded']
    for _ in range(30): result = s.emit(advance=.00002)
    assert result['additional_axial_advance_m'] == 0.
    assert result['bolt_turn_rad'] == 0. and not result['success']


def test_microincrements_without_transmission_do_not_accumulate_credited_turn(inputs):
    s = Synthetic(inputs['task']).settled()
    # Each increment is below the motion-noise threshold; net motion still needs the tool chain.
    for value in np.linspace(0., math.pi/6, 601)[1:]:
        result = s.emit(float(value))
    for _ in range(30): result = s.emit(math.pi/6)
    assert result['bolt_turn_rad'] == pytest.approx(math.pi/6)
    assert result['contact_backed_bolt_turn_rad'] == 0. and not result['success']


def test_native_angular_jitter_does_not_replace_measured_pose_stability(inputs):
    s = Synthetic(inputs['task']); s.emit()
    for _ in range(25):
        obs = s.sample()
        for row in obs['objects'].values(): row['angular_velocity_rad_s'] = [0., 0., .1]
        result = s.e.update(obs)
    assert result['settled_baseline_recorded'] and not result['success']
    s.rotate()
    for _ in range(25):
        obs = s.sample(math.pi/6)
        for row in obs['objects'].values(): row['angular_velocity_rad_s'] = [0., 0., .1]
        result = s.e.update(obs)
    assert result['success']


@pytest.mark.parametrize('mode', ['native_angular_speed', 'observed_rotation', 'observed_translation', 'slow_cumulative_drift'])
def test_settlement_requires_both_motion_rates_and_half_second_pose_envelope(inputs, mode):
    s = Synthetic(inputs['task']); s.emit()
    for index in range(1, 26):
        turn = index*.0006 if mode == 'observed_rotation' else 0.
        advance = index*(.00005 if mode == 'observed_translation' else .000006) if mode in ('observed_translation', 'slow_cumulative_drift') else None
        obs = s.sample(turn, advance=advance)
        for row in obs['objects'].values():
            row['angular_velocity_rad_s'] = [0., 0., .31 if mode == 'native_angular_speed' else 0.]
            row['linear_velocity_m_s'] = [0., 0., 0.]
        result = s.e.update(obs)
    assert not result['settled_baseline_recorded'] and not result['success']


@pytest.mark.parametrize('mode', ['rotation', 'translation'])
def test_release_drift_is_not_stability_even_when_reported_velocity_is_zero(inputs, mode):
    s = Synthetic(inputs['task']).settled(); s.rotate()
    for index in range(1, 26):
        turn = math.pi/6+(index*.0006 if mode == 'rotation' else 0.)
        # Lateral slow drift keeps screw advance/lead and motion-chain guards valid;
        # only the full release-window position envelope should reject it.
        obs = s.sample(turn, advance=.00025, axis=index*.000006 if mode == 'translation' else 0.)
        for row in obs['objects'].values():
            row['linear_velocity_m_s'] = [0., 0., 0.]
            row['angular_velocity_rad_s'] = [0., 0., 0.]
        result = s.e.update(obs)
    assert not result['success'] and result['release_stable_s'] < .5
    assert result['irrecoverable_reason'] is None


def segment(s, goal, *, drive, advance_offset=0., steps=100):
    for value in np.linspace(s.angle, goal, steps+1)[1:]:
        result = s.emit(float(value), advance=advance_offset+.003*float(value)/(2*math.pi), drive=drive)
    return result


def test_uncredited_micro_settlement_is_recoverable_before_real_tool_advance(inputs):
    s = Synthetic(inputs['task']).settled()
    result = s.emit(advance=.00000137)
    assert result['irrecoverable_reason'] is None and result['progress'] == 0.
    segment(s, math.pi/6, drive=True, advance_offset=.00000137)
    for _ in range(25): result = s.emit(math.pi/6, advance=.00025137)
    assert result['success']


def test_small_uncredited_release_settlement_keeps_actual_retained_progress(inputs):
    s = Synthetic(inputs['task']).settled(); s.rotate()
    for _ in range(25): result = s.emit(math.pi/6, advance=.00027)
    assert result['success'] and result['irrecoverable_reason'] is None
    assert result['contact_backed_bolt_turn_rad'] == pytest.approx(math.pi/6)


def test_contact_forward_uncontacted_reverse_cycles_cannot_farm_progress(inputs):
    s = Synthetic(inputs['task']).settled()
    for _ in range(6):
        segment(s, math.radians(5.), drive=True)
        result = segment(s, 0., drive=False)
        assert result['contact_backed_bolt_turn_rad'] == pytest.approx(0., abs=1.e-12)
    segment(s, math.pi/6, drive=False)
    for _ in range(25): result = s.emit(math.pi/6)
    assert result['progress'] == pytest.approx(0., abs=1.e-12) and not result['success']
    assert result['irrecoverable_reason'] is None


def test_uncontacted_goal_then_contact_micro_oscillation_cannot_farm_progress(inputs):
    s = Synthetic(inputs['task']).settled(); segment(s, math.pi/6, drive=False)
    for _ in range(30):
        segment(s, math.radians(31.), drive=True, steps=10)
        result = segment(s, math.pi/6, drive=False, steps=10)
    for _ in range(25): result = s.emit(math.pi/6)
    assert result['contact_backed_bolt_turn_rad'] == pytest.approx(0., abs=1.e-12)
    assert not result['success'] and result['irrecoverable_reason'] is None


def test_crossing_negative_turn_then_returning_to_zero_never_earns_net_credit(inputs):
    s = Synthetic(inputs['task']).settled()
    segment(s, math.radians(-10.), drive=False)
    result = segment(s, 0., drive=True)
    assert result['bolt_turn_rad'] == pytest.approx(0., abs=1.e-12)
    assert result['contact_backed_bolt_turn_rad'] == pytest.approx(0., abs=1.e-12)
    assert not result['success']


def test_uncontacted_positive_motion_cannot_increase_prior_contact_credit(inputs):
    s = Synthetic(inputs['task']).settled()
    segment(s, math.radians(10.), drive=True)
    result = segment(s, math.pi/6, drive=False)
    assert result['bolt_turn_rad'] == pytest.approx(math.pi/6)
    assert result['contact_backed_bolt_turn_rad'] == pytest.approx(math.radians(10.))
    for _ in range(25): result = s.emit(math.pi/6)
    assert not result['success'] and result['irrecoverable_reason'] is None


def assert_public_verdict(task, result, state):
    public = task['public_rules']['referee']
    verdict = result['referee']
    assert verdict['state'] == state
    assert verdict['phase'] in public['phases']
    assert set(verdict['reason_codes']) <= public['reason_descriptions'].keys()
    assert verdict['terminal'] == (state != 'ongoing')
    assert verdict['recoverable'] == (state == 'ongoing')


def test_public_chinese_rules_bind_exact_thresholds_and_explain_recovery(inputs):
    from scenes.workrooms.mechanical_assembly.tasks.fastener_assembly.rules import thresholds
    rules = deepcopy(inputs['task']['public_rules'])
    public = rules['referee']
    assert rules['schema'] == 'mechanical_fastener_rules_v4'
    assert public['task_id'] == 'fastener_assembly'
    assert public['thresholds'] == thresholds()
    assert all(any('\u4e00' <= char <= '\u9fff' for char in text)
               for text in [*public['phases'].values(), *public['reason_descriptions'].values(),
                            *public['success_conditions'], *public['recovery']])
    assert '零冲量' in ''.join(public['success_conditions'])
    assert '冲量门槛' in public['reason_descriptions']['direct_robot_bolt_contact']
    assert '不计分' in ''.join(public['recovery'])
    public['thresholds']['release_hold_s'] = 99.
    assert inputs['task']['evaluation']['release_hold_s'] == .5


def test_referee_stages_match_complete_observed_sequence(inputs):
    s = Synthetic(inputs['task'])
    unsampled = s.e.inspect_conditions()
    assert_public_verdict(s.task, unsampled, 'ongoing')
    assert unsampled['referee']['reason_codes'] == ['not_sampled']
    assert s.emit()['referee']['phase'] == 'settle'
    for _ in range(25): result = s.emit()
    assert result['referee']['phase'] == 'advance'
    result = s.rotate()
    assert_public_verdict(s.task, result, 'ongoing')
    assert result['referee']['phase'] == 'release'
    assert 'robot_contact_prevents_release' in result['referee']['reason_codes']
    for _ in range(25): result = s.emit(math.pi/6)
    assert_public_verdict(s.task, result, 'success')
    assert result['referee']['phase'] == 'complete'


@pytest.mark.parametrize('robot_contact', ['tool', 'bolt', 'non_hand_tool'])
def test_zero_impulse_robot_manifold_restarts_release_without_forfeit(inputs, robot_contact):
    s = Synthetic(inputs['task']).settled(); s.rotate()
    for _ in range(24): s.emit(math.pi/6)
    obs = s.sample(math.pi/6, held=robot_contact != 'bolt', direct_bolt=robot_contact == 'bolt')
    for row in obs['contacts']['rigid']:
        if any(row[key].startswith('/World/Robot/') for key in ('actor0', 'actor1')):
            row.update(impulse_magnitude_sum_ns=0., impulse_vector_ns=[0., 0., 0.],
                       nonzero_impulse_normal_count=0)
            if robot_contact == 'non_hand_tool': row['actor1'] = '/World/Robot/fr3_link5'
    result = s.e.update(obs)
    assert_public_verdict(s.task, result, 'ongoing')
    assert result['irrecoverable_reason'] is None and result['release_stable_s'] == 0.
    assert 'robot_contact_prevents_release' in result['referee']['reason_codes']
    assert result['contact_backed_bolt_turn_rad'] == pytest.approx(math.pi/6)
    for _ in range(24): result = s.emit(math.pi/6)
    assert not result['success']
    assert s.emit(math.pi/6)['success']


@pytest.mark.parametrize('interruption', ['held', 'tool_disengaged', 'pose_drift'])
def test_regrasp_or_tool_reengagement_requires_fresh_full_release_window(inputs, interruption):
    s = Synthetic(inputs['task']).settled(); s.rotate()
    for _ in range(24): s.emit(math.pi/6)
    for _ in range(10):
        result = s.emit(math.pi/6, held=interruption == 'held', tool_bolt=interruption != 'tool_disengaged',
                        axis=.00015 if interruption == 'pose_drift' else 0.)
    assert_public_verdict(s.task, result, 'ongoing')
    assert not result['success'] and result['irrecoverable_reason'] is None
    for _ in range(24): result = s.emit(math.pi/6)
    assert not result['success']
    # A return from the displaced pose first breaks quietness; its dwell starts
    # at the following boundary, rather than crediting the correction window.
    result = s.emit(math.pi/6)
    if interruption == 'pose_drift':
        assert not result['success']
        result = s.emit(math.pi/6)
    assert result['success']


def test_target_overshoot_can_be_corrected_without_farming_turn_credit(inputs):
    s = Synthetic(inputs['task']).settled()
    result = s.rotate(final=math.radians(34.))
    assert_public_verdict(s.task, result, 'ongoing')
    assert result['referee']['reason_codes'] == ['target_turn_exceeded_recoverably']
    result = segment(s, math.pi/6, drive=False)
    assert result['contact_backed_bolt_turn_rad'] == pytest.approx(math.pi/6)
    for _ in range(25): result = s.emit(math.pi/6)
    assert result['success']


@pytest.mark.parametrize('violation', ['direct_bolt', 'non_hand_tool', 'support', 'axis', 'lead'])
def test_verified_hard_boundary_is_explicit_and_immutable(inputs, violation):
    s = Synthetic(inputs['task']).settled()
    obs = s.sample(direct_bolt=violation == 'direct_bolt', held=violation == 'non_hand_tool',
                   support=violation != 'support', axis=.002 if violation == 'axis' else 0.,
                   advance=.0001 if violation == 'lead' else 0.)
    if violation == 'non_hand_tool':
        for row in obs['contacts']['rigid']:
            if row['actor1'] == '/World/Robot/right_index_DP': row['actor1'] = '/World/Robot/fr3_link5'
    result = s.e.update(obs)
    assert_public_verdict(s.task, result, 'task_failure')
    assert result['irrecoverable_reason'] is not None
    assert s.emit() == result
    assert s.e.update(None) == result


@pytest.mark.parametrize('earlier_hint', ['direct_bolt', 'penetration'])
def test_invalid_late_contact_row_cannot_commit_earlier_violation(inputs, earlier_hint):
    s = Synthetic(inputs['task']).settled()
    obs = s.sample(direct_bolt=earlier_hint == 'direct_bolt')
    if earlier_hint == 'penetration': obs['contacts']['rigid'][0]['min_separation_m'] = -.01
    obs['contacts']['rigid'][-1]['impulse_magnitude_sum_ns'] = float('nan')
    result = s.e.update(obs)
    assert_public_verdict(s.task, result, 'evaluator_invalid')
    assert result['irrecoverable_reason'] is None
    assert s.e.irrecoverable is None


def test_valid_contact_violation_with_invalid_motion_evidence_is_not_task_failure(inputs):
    s = Synthetic(inputs['task']).settled()
    obs = s.sample(direct_bolt=True)
    obs['objects']['bolt']['quaternion_wxyz'] = [2., 0., 0., 0.]
    result = s.e.update(obs)
    assert_public_verdict(s.task, result, 'evaluator_invalid')
    assert result['irrecoverable_reason'] is None


@pytest.mark.parametrize('trigger', ['speed', 'penetration'])
def test_valid_safety_stop_is_distinct_from_task_failure_and_evidence_fault(inputs, trigger):
    s = Synthetic(inputs['task']).settled()
    obs = s.sample()
    if trigger == 'speed': obs['objects']['bolt']['linear_velocity_m_s'] = [1., 0., 0.]
    else: obs['contacts']['rigid'][0]['min_separation_m'] = -.01
    result = s.e.update(obs)
    assert_public_verdict(s.task, result, 'safety_stop')
    assert result['irrecoverable_reason'] is None
    assert s.emit() == result


def test_success_cannot_be_revoked_by_late_bad_observation_or_result_mutation(inputs):
    s = Synthetic(inputs['task']).settled(); s.rotate()
    for _ in range(25): accepted = s.emit(math.pi/6)
    assert_public_verdict(s.task, accepted, 'success')
    result = s.e.update(None)
    assert result == accepted
    result['referee']['state'] = 'task_failure'
    assert s.e.inspect_conditions() == accepted


def test_static_referee_codes_hide_bad_observation_detail(inputs):
    s = Synthetic(inputs['task'])
    result = s.e.update({'time_s': 'private path /World/Objects/bolt'})
    assert_public_verdict(s.task, result, 'evaluator_invalid')
    assert result['referee']['reason_codes'] == ['invalid_native_evidence']


@pytest.mark.parametrize('continuation', ['missing', None, 0])
def test_continuation_requires_explicit_false_not_missing_or_false_like(inputs, continuation):
    s = Synthetic(inputs['task'])
    obs = s.sample()
    if continuation == 'missing': del obs['terminated']
    else: obs['terminated'] = continuation
    result = s.e.update(obs)
    assert_public_verdict(s.task, result, 'evaluator_invalid')
    assert result['irrecoverable_reason'] is None


@pytest.mark.parametrize('fault', ['zero_active', 'zero_vector', 'tiny_vector', 'norm_overflow'])
def test_inconsistent_unrelated_impulse_row_cannot_complete_release(inputs, fault):
    s = Synthetic(inputs['task']).settled(); s.rotate()
    for _ in range(24): result = s.emit(math.pi/6)
    assert not result['success'] and result['release_stable_s'] == pytest.approx(.48)
    obs = s.sample(math.pi/6)
    row = deepcopy(obs['contacts']['rigid'][0])
    row.update(actor0='/World/Unrelated/a', actor1='/World/Unrelated/b',
        impulse_magnitude_sum_ns=0., impulse_vector_ns=[0., 0., 0.], nonzero_impulse_normal_count=0)
    if fault == 'zero_active':
        row['nonzero_impulse_normal_count'] = 1
    elif fault == 'zero_vector':
        row['impulse_vector_ns'] = [0., 0., 1.e-10]
    elif fault == 'tiny_vector':
        row.update(impulse_magnitude_sum_ns=1.e-15, impulse_vector_ns=[0., 0., 1.e-10],
            nonzero_impulse_normal_count=1)
    else:
        maximum = float(np.finfo(float).max)
        row.update(impulse_magnitude_sum_ns=maximum, impulse_vector_ns=[maximum]*3,
            nonzero_impulse_normal_count=1)
    obs['contacts']['rigid'].append(row)
    result = s.e.update(obs)
    assert_public_verdict(s.task, result, 'evaluator_invalid')
    assert not result['success'] and result['irrecoverable_reason'] is None
    assert result['referee']['reason_codes'] == ['invalid_native_evidence']
    assert s.e.update(None) == result


@pytest.mark.parametrize('earlier_hint', ['direct_bolt', 'penetration'])
def test_zero_impulse_active_count_fault_precedes_apparent_physical_violation(inputs, earlier_hint):
    s = Synthetic(inputs['task']).settled()
    obs = s.sample(direct_bolt=earlier_hint == 'direct_bolt')
    if earlier_hint == 'penetration': obs['contacts']['rigid'][0]['min_separation_m'] = -.01
    row = deepcopy(obs['contacts']['rigid'][-1])
    row.update(actor0='/World/Unrelated/a', actor1='/World/Unrelated/b',
        impulse_magnitude_sum_ns=0., impulse_vector_ns=[0., 0., 0.], nonzero_impulse_normal_count=1)
    obs['contacts']['rigid'].append(row)
    result = s.e.update(obs)
    assert_public_verdict(s.task, result, 'evaluator_invalid')
    assert result['irrecoverable_reason'] is None and s.e.irrecoverable is None


@pytest.mark.parametrize('scalars,normals', [([0.], [[0., 0., 0.]]),
    ([1.e-20], [[0., 0., 1.]]), ([.001, .001], [[0., 0., 1.], [0., 0., -1.]])])
def test_consistent_zero_tiny_and_cancelled_native_aggregates_remain_valid(inputs, scalars, normals):
    from bench.simulation.isaacsim_backend.contact_precision import summarize_normal_impulses
    s = Synthetic(inputs['task']).settled()
    obs = s.sample()
    row = deepcopy(obs['contacts']['rigid'][0])
    row.update(actor0='/World/Unrelated/a', actor1='/World/Unrelated/b', point_count=len(scalars))
    row.update(summarize_normal_impulses(scalars, normals))
    obs['contacts']['rigid'].append(row)
    result = s.e.update(obs)
    assert_public_verdict(s.task, result, 'ongoing')
    assert result['irrecoverable_reason'] is None
