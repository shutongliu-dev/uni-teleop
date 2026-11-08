"""The release endpoint must survive ordinary IK posture regeneration."""
import copy
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.validation.laundry_captured_garment_fold import (
    command_joint_targets, opening_capture_targets, planned_duration)
from scripts.validation.laundry_release_hand import (
    distal_release_target, interpolate_release_target, opening_postures,
    release_hand_profile, release_target, retain_released_targets)


def hand(side):
    opened={side+'_thumb_MCP': .2, side+'_thumb_IP': .3,
            side+'_index_MCP': .4, side+'_index_PIP': .87,
            side+'_index_DIP': 1.13, side+'_middle_PIP': 1.5}
    return [{'hand_posture_rad': opened},
            {'hand_posture_rad': {n: min(v+.2, 1.7) for n,v in opened.items()}}]


@pytest.mark.parametrize('side',['left','right'])
def test_release_keeps_complete_open_map_and_never_mutates_approach(side):
    postures=hand(side);before=copy.deepcopy(postures)
    target=distal_release_target(side,postures)
    assert target[side+'_index_PIP']==target[side+'_index_DIP']==.1
    assert target[side+'_thumb_IP']==.05
    for name in ('thumb_MCP','index_MCP','middle_PIP'):
        assert target[side+'_'+name]==postures[0]['hand_posture_rad'][side+'_'+name]
    assert postures==before and set(target)==set(postures[0]['hand_posture_rad'])


def test_same_61_point_sweep_and_live_targets_release_before_drive_and_hold_endpoint():
    postures=hand('right');start={**postures[-1]['hand_posture_rad'],
        'right_joint1': .4,'left_index_DIP': .9}
    # The held commands are deliberately different from nominal closed posture.
    start['right_index_DIP']=.63;start['right_thumb_IP']=.41
    names=list(start);events=[]
    assist=SimpleNamespace(side='right',active=True,
        held_joint_targets=lambda phase:{} if phase in ('open','release_exit','retreat','free')
        else {'right_index_DIP':.63})
    def release(t,reason):
        if assist.active:events.append(('release',t));assist.active=False
    assist.release=release
    phase={'edges':[{'side':'right','contact_plan':{'postures':postures}}]}
    before=copy.deepcopy(postures)
    sweep=opening_postures(assist,postures,start,'distal_extend_v1')
    assert assist.active and not events and len(sweep)==61
    def write(target,index):
        assert not assist.active
        events.append(('drive',target.copy()))
    env=SimpleNamespace(joint_names=names,actuated_joint_names=names,
        joint_limits={n:(-2.,2.) for n in names},_targets=np.array([list(start.values())],np.float32),
        _expand_mimic_targets=lambda values:values,_tensor=lambda values:values,_one=0,
        _robot_view=SimpleNamespace(set_dof_position_targets=write))
    for i,u in enumerate(np.linspace(0.,1.,61)):
        # IK regenerates the old curved posture; explicit release must override it.
        planned={**start,**postures[0]['hand_posture_rad']}
        result=opening_capture_targets([assist],phase,start,planned,u,14.+2*u,
            hand_profile='distal_extend_v1')
        command_joint_targets(env,result,[assist],'open')
        assert {n:result[n] for n in sweep[i]['hand_posture_rad']}==sweep[i]['hand_posture_rad']
        assert result['left_index_DIP']==.9 and result['right_joint1']==.4
    assert events[0]==('release',14.) and sum(e[0]=='release' for e in events)==1
    endpoint=distal_release_target('right',postures)
    for action in ('release_exit','retreat','free'):
        result=retain_released_targets({**start,**postures[0]['hand_posture_rad']},endpoint)
        command_joint_targets(env,result,[assist],action)
        for n,v in endpoint.items():assert env._targets[0,names.index(n)]==pytest.approx(v)
        assert result['left_index_DIP']==.9
    assert postures==before


def test_old_profile_preserves_original_preview_and_timing():
    postures=hand('left');calls=[]
    assist=SimpleNamespace(opening_postures=lambda p,s:(calls.append((p,s)) or p))
    assert opening_postures(assist,postures,{},'pinch_open_v1') is postures
    assert len(calls)==1 and release_hand_profile({})=='pinch_open_v1'
    options={'phases':{'0':{'capture_order':['right','left']}}}
    before=planned_duration(3,options,2.)
    assert before==58.5
    options.update(release_hand_profile='distal_extend_v1',release_profile='cooked_hand_clearance_v2')
    assert planned_duration(3,options,2.)==before+3.
    options['retreat_profile']='finger_axis_exit_v2'
    assert planned_duration(3,options,2.)==before+6.


def test_new_posture_requires_the_actual_cooked_envelope():
    with pytest.raises(ValueError,match='cooked opening'):
        release_hand_profile({'release_hand_profile':'distal_extend_v1'})
    for value in (None,True,'unmeasured_straight_fingers'):
        with pytest.raises(ValueError,match='Unknown'):
            release_hand_profile({'release_hand_profile':value})


def test_invalid_or_incomplete_targets_cannot_enter_original_drive_commands():
    postures=hand('right');del postures[0]['hand_posture_rad']['right_index_PIP']
    with pytest.raises(ValueError,match='complete original'):distal_release_target('right',postures)
    target=distal_release_target('right',hand('right'))
    for u in (-.01,1.01,float('nan')):
        with pytest.raises(ValueError):interpolate_release_target(target,target,u)
    with pytest.raises(ValueError):interpolate_release_target({},target,0.)
    with pytest.raises(ValueError,match='original target inventory'):
        retain_released_targets(target,{'invented_joint':.1})
    env=SimpleNamespace(joint_names=list(target),actuated_joint_names=list(target),
        joint_limits={n:(0.,.09) for n in target})
    with pytest.raises(ValueError,match='limit'):
        command_joint_targets(env,target,[],'open')


@pytest.mark.parametrize('side', ['left', 'right'])
def test_full_spread_is_independent_of_narrow_capture_and_preserves_other_fingers(monkeypatch, side):
    narrow=hand(side); wide=copy.deepcopy(narrow)
    wide[0]['hand_posture_rad'][side+'_thumb_MCP']=.07
    wide[0]['hand_posture_rad'][side+'_index_MCP']=.09
    wide[0]['hand_posture_rad'][side+'_middle_PIP']=.9
    calls=[];model=SimpleNamespace(hand=SimpleNamespace(active=[
        SimpleNamespace(name=n,lower=-2.,upper=2.) for n in narrow[0]['hand_posture_rad']]),
        hand_posture_prior=SimpleNamespace(violation=lambda finger,posture:None))
    def original_ik(given_model, given_side, **kwargs):
        calls.append((given_model,given_side,kwargs));return np.zeros(3),wide
    monkeypatch.setattr('scripts.validation.laundry_contact_pinch.pinch_postures',original_ik)
    before=copy.deepcopy(narrow)
    target=release_target(side,narrow,'full_spread_extend_v2',model=model)
    assert calls==[(model,side,dict(open_distance=.060,close_distance=.010,samples=2))]
    assert target[side+'_thumb_MCP']==.07 and target[side+'_index_MCP']==.09
    assert target[side+'_middle_PIP']==narrow[0]['hand_posture_rad'][side+'_middle_PIP']
    assert target[side+'_index_PIP']==target[side+'_index_DIP']==.1
    assert target[side+'_thumb_IP']==.05 and narrow==before
    assert distal_release_target(side,narrow)[side+'_thumb_MCP']==.2
    assist=SimpleNamespace(side=side,active=True)
    events=[]
    def release(*args):
        assist.active=False;events.append('disabled')
    assist.release=release
    start=narrow[-1]['hand_posture_rad'];phase={'edges':[{'side':side,'contact_plan':{'postures':narrow}}]}
    sweep=opening_postures(assist,narrow,start,'full_spread_extend_v2',model=model)
    assert assist.active and len(sweep)==61
    for i,u in enumerate(np.linspace(0.,1.,61)):
        got=opening_capture_targets([assist],phase,start,start,u,2*u,
            hand_profile='full_spread_extend_v2',release_targets=target)
        assert got==sweep[i]['hand_posture_rad'] and not assist.active
    assert retain_released_targets(start,target)==target
    model.hand_posture_prior.violation=lambda finger,posture:'invalid intermediate curl'
    with pytest.raises(ValueError,match='posture prior'):
        opening_postures(assist,narrow,start,'full_spread_extend_v2',model=model)
    with pytest.raises(ValueError,match='previewed endpoint'):
        opening_capture_targets([assist],phase,start,start,0.,0.,hand_profile='full_spread_extend_v2')
    with pytest.raises(ValueError,match='original hand model'):
        release_target(side,narrow,'full_spread_extend_v2')


def test_full_spread_requires_cooked_sweep_and_allocates_two_second_opening():
    options={'release_hand_profile':'full_spread_extend_v2'}
    with pytest.raises(ValueError,match='cooked opening'):release_hand_profile(options)
    options['release_profile']='cooked_hand_clearance_v2'
    assert release_hand_profile(options)=='full_spread_extend_v2'
    assert planned_duration(3,options,2.)==planned_duration(3,{},2.)+3.
