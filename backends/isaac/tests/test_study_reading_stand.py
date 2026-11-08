"""CPU rule counterexamples; synthetic contacts are not native dynamics evidence."""
from copy import deepcopy
import pytest
from scenes.workrooms.study.reading_stand_rules import configuration, panel_pose, on_panel
from scenes.workrooms.study.reading_stand_evaluation import SequenceEvaluator


def frame(c, index=0, *, goal=False, actions=0):
    objects={n:{**deepcopy(v['initial_pose']),'linear_velocity_m_s':[0.,0.,0.],
        'angular_velocity_rad_s':[0.,0.,0.]} for n,v in c['objects'].items()}
    rows=[]
    if goal:
        objects['panel'].update(panel_pose())
        objects['support_pin']['position_m'][1]=0.
        objects['book'].update(position_m=on_panel([.10,0.,.014]),quaternion_wxyz=panel_pose()['quaternion_wxyz'])
        pc=c['mechanism']['pin_center_world_m']
        rows=[{'a':'panel','b':'support_pin','point_m':[pc[0]-.00819152,0.,pc[2]+.00573576],
               'impulse_on_a_n_s':[0.,0.,.003],'separation_m':0.},
              {'a':'book','b':'panel','point_m':on_panel([.10,0.,.006]),
               'impulse_on_a_n_s':[0.,0.,.001],'separation_m':0.}]
    return {'schema':'study_native_observation_v1','configuration_sha256':c['configuration_sha256'],
        'episode_id':'synthetic_test','step_index':index,'time_s':index*.0005,'physics_dt_s':.0005,
        'sampling_complete':True,'contacts_complete':True,'physics_valid':True,'robot_only_actions':True,
        'object_pose_writes':False,'budget_exhausted':False,'robot_action_count':actions,'objects':objects,'contacts':rows}


def test_reset_and_zero_action_cannot_score():
    c=configuration();e=SequenceEvaluator(c,trusted_normalized_input=True)
    assert e.update(frame(c))['reason']=='reset_baseline_not_scored'
    for i in range(1,850):r=e.update(frame(c,i,goal=True))
    assert not r['success'] and r['reason']=='robot_action_required'


def test_complete_real_support_predicates_and_released_window_are_required():
    c=configuration();e=SequenceEvaluator(c,trusted_normalized_input=True);e.update(frame(c))
    for i in range(1,804):r=e.update(frame(c,i,goal=True,actions=1))
    assert r['success'] and r['stable_duration_s']>=.4


@pytest.mark.parametrize('change', ['no_pin_contact','wrong_force_sign','pin_withdrawn','book_airborne','held_left','held_right','wrong_angle'])
def test_each_physical_missing_condition_prevents_success(change):
    c=configuration();e=SequenceEvaluator(c,trusted_normalized_input=True);e.update(frame(c))
    for i in range(1,805):
        row=frame(c,i,goal=True,actions=1)
        if change=='no_pin_contact':row['contacts']=row['contacts'][1:]
        if change=='wrong_force_sign':row['contacts'][0]['impulse_on_a_n_s'][2]*=-1
        if change=='pin_withdrawn':row['objects']['support_pin']['position_m'][1]=.28
        if change=='book_airborne':row['objects']['book']['position_m'][2]+=.04;row['contacts']=row['contacts'][:1]
        if change.startswith('held_'):
            row['contacts'].append({'a':'book','b':'robot/'+change[5:]+'_thumb','point_m':row['objects']['book']['position_m'],
                'impulse_on_a_n_s':[0.,0.,.0001],'separation_m':0.})
        if change=='wrong_angle':row['objects']['panel'].update(panel_pose(45.))
        result=e.update(row)
    assert result['status']=='active' and not result['success']


def test_source_contract_mutation_and_missing_native_frame_are_invalid():
    c=configuration();bad=deepcopy(c);bad['mechanism']['panel_angle_deg']=20.
    with pytest.raises(ValueError):SequenceEvaluator(bad,trusted_normalized_input=True)
    e=SequenceEvaluator(c,trusted_normalized_input=True);e.update(frame(c))
    assert e.update(frame(c,2))['status']=='invalid'
