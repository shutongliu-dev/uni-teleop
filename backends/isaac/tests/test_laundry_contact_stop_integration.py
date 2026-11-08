"""Exercise original target writes and the unchanged measured capture gate."""
import copy
import hashlib
import json
import sys
from types import SimpleNamespace, MethodType

import numpy as np
import pytest

from scripts.validation.laundry_contact_assist import CapturedPatchAssist
from scripts.validation.laundry_captured_garment_fold import command_joint_targets, opening_capture_targets
from scripts.validation.laundry_release_hand import opening_postures, distal_release_target, retain_released_targets


@pytest.fixture
def setup(tmp_path, monkeypatch):
    tri=np.array([[0,1,2],[1,3,2]])
    rest=np.array([[0,0,0],[.01,0,0],[0,.01,0],[.01,.01,0]])
    q=rest+[0,0,.9]
    names=['right_thumb_IP','right_index_DIP','right_middle_PIP','right_joint1','right_index_follower']
    bodies=['right_hand_wrist','right_thumb_fingertip','right_index_fingertip']
    state={'robot_body_q':np.array([[0,0,.9,0,0,0,1],[0,0,.9,0,0,0,1],[.01,0,.9,0,0,0,1]]),
           'joint_q':np.array([.5,.5,.5,.5,.25]),'joint_qd':np.zeros(5)}
    writes=[];expansions=[]
    def expand(values):
        expansions.append(dict(values))
        return {**values,'right_index_follower':values['right_index_DIP']*.5}
    env=SimpleNamespace(body_names=bodies,joint_names=names,actuated_joint_names=names[:4],
        joint_limits={n:(-1.,1.) for n in names},_expand_mimic_targets=expand,
        _targets=np.array([[.6,.6,.6,.6,.3]],np.float32),_tensor=lambda v:v,_one=0,
        _robot_view=SimpleNamespace(set_dof_position_targets=lambda values,index:writes.append(values.copy())))
    scene={'cloths':[{'id':'towel','rest_vertices_m':rest.tolist(),'triangles':tri.tolist(),
                     'isaacsim':{'cloth_thickness_m':.0018}}]}
    calls=[];geometry_mode={'value':'valid'}
    def diagnose(side,body,cloth,*_):
        calls.append((body.copy(),cloth.copy()))
        result={'side':side,'distal_group_severe_penetration':False,'contact_force_measured':False,
            'surface_pair':{'thumb':'DP','index':'DP'},'multilayer_diagnostic':{'plausible':True,
                'paired_witnesses':[{'thumb':{'triangle_id':0},'index':{'triangle_id':1},'opposing_normal_dot':-.9}]},
            'state_identity':{'body_q_float64_sha256':hashlib.sha256(body.astype('<f8').tobytes()).hexdigest(),
                              'cloth_q_float64_sha256':hashlib.sha256(cloth.astype('<f8').tobytes()).hexdigest()}}
        if geometry_mode['value']=='missing':result['multilayer_diagnostic']['plausible']=False
        if geometry_mode['value']=='stale':result['state_identity']={}
        return result
    monkeypatch.setattr('scripts.validation.laundry_pinch_geometry.PinchGeometry',
                        lambda *_:SimpleNamespace(diagnose=diagnose))
    assistants=[]
    def make(**options):
        a=CapturedPatchAssist(env,scene,'right',{},tmp_path,name='hand_'+str(len(assistants)),
            capture_profile=options.pop('capture_profile','local_assist_v2'),
            allowed_material_vertex_ids=[0,1,2,3],**options)
        assistants.append(a)
        opened=copy.deepcopy(state);opened['robot_body_q'][2,0]=.06
        a.update('descend',-.02,opened,q,env._targets[0])
        return a
    yield SimpleNamespace(env=env,state=state,q=q,make=make,writes=writes,expansions=expansions,
                          calls=calls,mode=geometry_mode)
    for a in assistants:a.close()


def latch(s):
    a=s.make(contact_closure_stop=True)
    assert a.observe_closure('close',0.,s.state,s.q,s.env._targets[0])
    assert not a.active and a.prim is None
    return a


def test_default_does_not_sample_closure_or_modify_old_commands(setup):
    s=setup;a=s.make();values=dict(zip(s.env.joint_names,[.6,.6,.6,.6,.3]))
    assert not a.observe_closure('close',0.,s.state,s.q,s.env._targets[0])
    a.update('close',0.,s.state,s.q,s.env._targets[0])
    assert not s.calls and a.closure_diagnostics_path is None
    assert a.opening_joint_targets([],{},.5)=={}
    for phase in ('close','capture_raise','closed_hold','fold','open','retreat'):
        result=command_joint_targets(s.env,values,[a],phase)
        assert result==values
        np.testing.assert_array_equal(s.env._targets,np.array([list(values.values())],np.float32))


@pytest.mark.parametrize('value',[None,1,'true',[],{}])
def test_option_requires_an_actual_boolean(setup,value):
    with pytest.raises(ValueError,match='explicit boolean'):setup.make(contact_closure_stop=value)


def test_bridge_stop_requires_explicit_consistent_capture_and_local_material_mode(setup):
    options=dict(contact_closure_stop=True,contact_closure_target_policy='hold_command_v2',
                 contact_closure_geometry_profile='adjacent_patch_v3',capture_profile='adjacent_patch_v3')
    for changes in ({'contact_closure_stop':False,'contact_closure_target_policy':'reduce_preload_v1'},
                    {'capture_profile':'local_assist_v2'}, {'attachment_material_mode':'shared_edge'},
                    {'contact_closure_geometry_profile':'unrestricted'}):
        with pytest.raises(ValueError,match='Bridge closure geometry'):
            setup.make(**{**options,**changes})
    a=setup.make(**options)
    assert a.closure_stop.geometry_profile=='adjacent_patch_v3'
    assert a.observe_closure('close',0.,setup.state,setup.q,setup.env._targets[0])
    assert a.closure_stop.receipt['geometry_profile']=='adjacent_patch_v3'
    assert a.prim is None and not a.active


def test_loaded_gate_requires_actual_stop_and_new_full_dwell(setup,monkeypatch):
    s=setup;options=dict(capture_profile='loaded_adjacent_patch_v5',contact_closure_stop=True,
        contact_closure_target_policy='hold_command_v2',contact_closure_geometry_profile='adjacent_patch_v3')
    for change in ({'contact_closure_stop':False,'contact_closure_target_policy':'reduce_preload_v1'},
                   {'contact_closure_target_policy':'reduce_preload_v1'},
                   {'contact_closure_geometry_profile':'shared_edge_v1'},
                   {'attachment_material_mode':'shared_edge'}):
        with pytest.raises(ValueError,match='Loaded capture'):s.make(**{**options,**change})
    a=s.make(**options);s.env._targets[0,:2]=[.566,.51];enabled=[]
    monkeypatch.setattr(a,'_enable',lambda time_s,result,body,q:enabled.append((time_s,result)))
    for t in np.arange(0.,.121,.02):
        a.update('closed_hold',float(t),s.state,s.q,s.env._targets[0])
    assert not enabled and not a.gate.history
    assert a.observe_closure('close',.14,s.state,s.q,s.env._targets[0])
    np.testing.assert_array_equal(list(a.closure_stop.receipt['finger_joint_targets_rad'].values()),s.env._targets[0,:2])
    for t in np.arange(.14,.221,.02):
        a.update('close',float(t),s.state,s.q,s.env._targets[0])
        assert not enabled
    a.update('close',.24,s.state,s.q,s.env._targets[0])
    assert len(enabled)==1 and enabled[0][1]['capture_candidate']
    assert enabled[0][1]['continuous_duration_s']==pytest.approx(.1)
    assert a.geometry_options=={} and a.closure_stop.geometry_profile=='adjacent_patch_v3'
    rows=[json.loads(line) for line in a.diagnostics_path.read_text().splitlines()]
    assert 'loaded_capture_requires_actual_latched_closure_stop' in rows[0]['capture_gate']['reasons']


def test_explicit_hold_observation_freezes_existing_command_then_starts_new_window(setup,monkeypatch):
    s=setup;options=dict(capture_profile='loaded_adjacent_patch_v5',contact_closure_stop=True,
        contact_closure_target_policy='hold_command_v2',contact_closure_geometry_profile='adjacent_patch_v3',
        contact_closure_observation_policy='close_and_hold_v1')
    a=s.make(**options);s.env._targets[0,:2]=[.566,.51];s.env._targets[0,4]=.255
    original=s.env._targets.copy();enabled=[]
    monkeypatch.setattr(a,'_enable',lambda time_s,result,body,q:enabled.append((time_s,result)))
    s.mode['value']='missing'
    for t in np.arange(0,.101,.02):
        assert not a.observe_closure('close',float(t),s.state,s.q,s.env._targets[0])
        a.update('close',float(t),s.state,s.q,s.env._targets[0])
    s.mode['value']='valid'
    assert a.observe_closure('closed_hold',.12,s.state,s.q,s.env._targets[0])
    receipt=a.closure_stop.receipt
    assert receipt['observation_phase']=='closed_hold'
    assert receipt['evidence']['existing_closed_hold_command_frozen_unchanged']
    command_joint_targets(s.env,dict(zip(s.env.joint_names,original[0])),[a],'closed_hold')
    np.testing.assert_array_equal(s.env._targets,original)
    for t in [.12,.14,.16,.18,.20]:
        a.update('closed_hold',t,s.state,s.q,s.env._targets[0]);assert not enabled
    a.update('closed_hold',.22,s.state,s.q,s.env._targets[0])
    assert len(enabled)==1 and enabled[0][1]['evidence']['first_time_s']==.12
    assert not a.observe_closure('open',.24,s.state,s.q,s.env._targets[0])
    assert not a.held_joint_targets('open')


def test_default_stop_still_ignores_late_hold_contacts(setup):
    s=setup;a=s.make(contact_closure_stop=True,contact_closure_target_policy='hold_command_v2')
    assert not a.observe_closure('closed_hold',8.3,s.state,s.q,s.env._targets[0])
    assert not s.calls and not a.closure_stop.latched


def test_nonloaded_hold_latch_cannot_reuse_pre_latch_history(setup):
    s=setup;a=s.make(contact_closure_stop=True,contact_closure_target_policy='hold_command_v2',
                    contact_closure_observation_policy='close_and_hold_v1')
    # Any earlier sampled history belongs to a different command-control epoch.
    a.gate.history=[{'sentinel':'must be discarded'}];a.last_observe_time=8.3
    assert a.observe_closure('closed_hold',8.3,s.state,s.q,s.env._targets[0])
    assert not a.gate.history and a.last_observe_time==-np.inf


def test_stopped_original_actuated_targets_persist_through_every_closed_phase(setup):
    s=setup;before=copy.deepcopy(s.state);cloth_before=s.q.copy();a=latch(s)
    stopped=a.held_joint_targets('close')
    assert set(stopped)=={'right_thumb_IP','right_index_DIP'}
    np.testing.assert_allclose(list(stopped.values()),[.52,.52],atol=1e-8)
    request=dict(zip(s.env.joint_names,[.9,.9,.7,.8,.45]));original=dict(request)
    for phase in ('close','capture_raise','closed_hold','fold','capture_lift','open_hover','descend'):
        command_joint_targets(s.env,request,[a],phase)
        assert s.expansions[-1]['right_thumb_IP']==stopped['right_thumb_IP']
        assert s.expansions[-1]['right_index_DIP']==stopped['right_index_DIP']
        assert s.expansions[-1]['right_middle_PIP']==.7 and s.expansions[-1]['right_joint1']==.8
        assert s.env._targets[0,4]==pytest.approx(stopped['right_index_DIP']*.5)
    assert request==original
    for key in before:np.testing.assert_array_equal(s.state[key],before[key])
    np.testing.assert_array_equal(s.q,cloth_before)
    assert len(s.writes)==7 and not a.active
    for phase in ('open','release_exit','retreat','free'):assert a.held_joint_targets(phase)=={}


def test_explicit_hold_policy_does_not_unload_existing_commands(setup):
    s=setup;previous=s.env._targets[0].copy()
    a=s.make(contact_closure_stop=True,contact_closure_target_policy='hold_command_v2')
    assert a.observe_closure('close',0.,s.state,s.q,previous)
    request=dict(zip(s.env.joint_names,[.9,.9,.7,.8,.45]))
    command_joint_targets(s.env,request,[a],'fold')
    np.testing.assert_array_equal(s.env._targets[0,:2],previous[:2])
    assert s.env._targets[0,2]==pytest.approx(.7)
    assert not a.active
    with pytest.raises(ValueError,match='requires enabled'):
        s.make(contact_closure_target_policy='hold_command_v2')


@pytest.mark.parametrize('invalid',['missing','stale'])
def test_missing_or_stale_geometry_cannot_stop_or_enable_during_close(setup,invalid):
    s=setup;s.mode['value']=invalid;a=s.make(contact_closure_stop=True)
    for t in np.arange(0.,.121,.002):
        assert not a.observe_closure('close',float(t),s.state,s.q,s.env._targets[0])
        a.update('close',float(t),s.state,s.q,s.env._targets[0])
    assert len(s.calls)==7  # Fresh geometry at 20 ms, not at every physics substep.
    assert not a.closure_stop.latched and not a.active and not a.gate.history
    assert a.held_joint_targets('fold')=={}


def test_stop_still_requires_actual_stopped_fingers_and_full_sampled_dwell(setup,monkeypatch):
    s=setup;a=latch(s);request=dict(zip(s.env.joint_names,s.env._targets[0]))
    command_joint_targets(s.env,request,[a],'close')
    enabled=[]
    monkeypatch.setattr(a,'_enable',lambda time_s,result,body,q:enabled.append(copy.deepcopy(result)))
    s.state['joint_qd'][:2]=.2
    a.update('close',0.,s.state,s.q,s.env._targets[0])
    s.state['joint_qd'][:2]=0.
    for t in np.arange(.02,.101,.02):
        a.update('close',float(t),s.state,s.q,s.env._targets[0])
        assert not enabled
    a.update('close',.12,s.state,s.q,s.env._targets[0])
    assert len(enabled)==1 and enabled[0]['capture_candidate']
    assert enabled[0]['continuous_duration_s']==pytest.approx(.1)
    rows=[json.loads(line) for line in a.diagnostics_path.read_text().splitlines()]
    assert 'fingers_not_stopped' in rows[0]['capture_gate']['reasons']
    assert all(row['phase']=='close' for row in rows)


def test_first_open_command_releases_and_interpolates_from_stopped_targets(setup,monkeypatch):
    s=setup;a=latch(s)
    start=command_joint_targets(s.env,dict(zip(s.env.joint_names,s.env._targets[0])),[a],'fold')
    opened={'right_thumb_IP':.1,'right_index_DIP':.2,'right_middle_PIP':.3}
    postures=[{'hand_posture_rad':opened},{'hand_posture_rad':dict.fromkeys(opened,.9)}]
    phase={'edges':[{'side':'right','contact_plan':{'postures':postures}}]}
    a.active=True;events=[]
    def release(time_s,reason):
        if a.active:events.append(('release',time_s));a.active=False
    monkeypatch.setattr(a,'release',release)
    def write(values,index):
        assert not a.active
        events.append(('command',values.copy()))
    s.env._robot_view.set_dof_position_targets=write
    preview=a.opening_postures(postures,start)
    assert len(preview)==61 and a.active  # Preview cannot release or change any state.
    for u in (0.,.01,.5,1.):
        planned={**start,**dict.fromkeys(opened,.9)}  # Original overclosure must never reappear.
        targets=opening_capture_targets([a],phase,start,planned,u,5.+u)
        command_joint_targets(s.env,targets,[a],'open')
        for n,v in opened.items():assert targets[n]==pytest.approx(start[n]+u*(v-start[n]))
    assert events[0]==('release',5.) and len([e for e in events if e[0]=='release'])==1
    assert preview[0]['hand_posture_rad']=={n:start[n] for n in opened}
    assert preview[-1]['hand_posture_rad']==pytest.approx(opened)
    assert preview[30]['hand_posture_rad']==pytest.approx(a.opening_joint_targets(postures,start,.5))


def test_distal_extension_uses_real_assist_release_without_any_closure_stop(setup,monkeypatch):
    s=setup;a=s.make(contact_closure_stop=False)
    assert a.closure_stop is None
    s.env.joint_names.append('right_index_PIP')
    s.env.actuated_joint_names.append('right_index_PIP')
    s.env.joint_limits['right_index_PIP']=(-1.,1.)
    s.env._targets=np.concatenate([s.env._targets,np.array([[.82]],np.float32)],axis=1)
    start=dict(zip(s.env.joint_names,map(float,s.env._targets[0])))
    postures=[{'hand_posture_rad':{'right_thumb_IP':.3,'right_index_PIP':.85,
        'right_index_DIP':.9,'right_middle_PIP':.5}}]
    phase={'edges':[{'side':'right','contact_plan':{'postures':postures}}]}
    # Stub only the native attachment API; execute the real assist.release path.
    native_flag={'value':True};events=[]
    def set_flag(value):native_flag['value']=value;events.append(('native_flag',value))
    attribute=SimpleNamespace(Set=set_flag,Get=lambda:native_flag['value'])
    a.prim=SimpleNamespace(GetAttribute=lambda _:attribute);a.active=True
    physx=SimpleNamespace(get_physx_simulation_interface=lambda:SimpleNamespace(flush_changes=lambda:events.append(('flush',))))
    monkeypatch.setitem(sys.modules,'omni',SimpleNamespace(physx=physx))
    monkeypatch.setitem(sys.modules,'omni.physx',physx)
    def write(values,index):
        assert not native_flag['value'] and not a.active
        events.append(('drive',values.copy()))
    s.env._robot_view.set_dof_position_targets=write
    before=copy.deepcopy(s.state);cloth=s.q.copy()
    sweep=opening_postures(a,postures,start,'distal_extend_v1')
    assert len(sweep)==61 and a.active and native_flag['value'] and not events
    for i,u in enumerate(np.linspace(0.,1.,61)):
        targets=opening_capture_targets([a],phase,start,start,u,10.+2*u,
            hand_profile='distal_extend_v1')
        command_joint_targets(s.env,targets,[a],'open')
        for n,v in sweep[i]['hand_posture_rad'].items():assert targets[n]==v
    assert events[:2]==[('native_flag',False),('flush',)]
    assert a.events==[{'time_s':10.,'event':'disable',
        'reason':'Opening phase releases the local constraint before stepping.',
        'native_attachment_enabled':False}]
    endpoint=distal_release_target('right',postures)
    for phase_name in ('release_exit','retreat','free'):
        request={**start,**postures[0]['hand_posture_rad']}
        command_joint_targets(s.env,retain_released_targets(request,endpoint),[a],phase_name)
        for n,v in endpoint.items():assert s.env._targets[0,s.env.joint_names.index(n)]==pytest.approx(v)
    for n in before:np.testing.assert_array_equal(s.state[n],before[n])
    np.testing.assert_array_equal(s.q,cloth)


def test_override_cannot_write_mimic_or_foreign_actuators(setup):
    s=setup;values=dict(zip(s.env.joint_names,s.env._targets[0]))
    for name,value in [('right_index_follower',.2),('extra_actuator',0.),('right_index_DIP',2.)]:
        fake=SimpleNamespace(held_joint_targets=lambda _,name=name,value=value:{name:value})
        with pytest.raises(ValueError,match='original actuated'):
            command_joint_targets(s.env,values,[fake],'fold')
    assert not s.writes


def test_real_enable_path_records_time_and_actual_wrist_offsets_after_gate(setup,monkeypatch):
    s=setup;a=latch(s);calls=[]
    s.env.stage=object();s.env.scene_info={'cloths':{'towel':{'path':'/World/cloth'}}}
    xform=SimpleNamespace(AddTranslateOp=lambda *_:object(),AddOrientOp=lambda *_:object())
    usdgeom=SimpleNamespace(Xform=SimpleNamespace(Define=lambda *_:xform),
                           XformOp=SimpleNamespace(PrecisionDouble=0))
    physx=SimpleNamespace(get_physx_simulation_interface=lambda:SimpleNamespace(flush_changes=lambda:None))
    monkeypatch.setitem(sys.modules,'pxr',SimpleNamespace(UsdGeom=usdgeom))
    monkeypatch.setitem(sys.modules,'omni',SimpleNamespace(physx=physx))
    monkeypatch.setitem(sys.modules,'omni.physx',physx)
    def attachment(stage,path,cloth,anchor,ids,offsets):
        calls.append((list(ids),offsets.copy()));return object()
    monkeypatch.setitem(sys.modules,'scripts.validation.laundry_cloth_interactions',
                        SimpleNamespace(attachment=attachment))
    monkeypatch.setattr(a,'_pose',lambda pose:setattr(a,'anchor_pose',np.asarray(pose).copy()))
    command_joint_targets(s.env,dict(zip(s.env.joint_names,s.env._targets[0])),[a],'close')
    for t in np.arange(.02,.121,.02):
        a.update('close',float(t),s.state,s.q,s.env._targets[0])
    assert a.active and len(calls)==1 and len(a.events)==1
    event=a.events[0]
    assert event['time_s']==pytest.approx(.12) and event['event']=='enable'
    assert event['captured_gate_evidence']['continuous_duration_s']==pytest.approx(.1)
    assert calls[0][0]==[0,1,2,3]
    np.testing.assert_array_equal(calls[0][1],s.q-s.state['robot_body_q'][0,:3])
    assert not event['friction_grasp_proven']


def test_one_stopped_hand_does_not_overwrite_the_other_hand_opening_path(setup):
    s=setup;right=latch(s)
    left=SimpleNamespace(side='left',closure_stop=SimpleNamespace(latched=False),release=lambda *_:None)
    left.held_joint_targets=MethodType(CapturedPatchAssist.held_joint_targets,left)
    left.opening_joint_targets=MethodType(CapturedPatchAssist.opening_joint_targets,left)
    assert left.held_joint_targets('close')=={}
    start={**dict(zip(s.env.joint_names,s.env._targets[0])),**right.held_joint_targets('fold')}
    right_open={'right_thumb_IP':.1,'right_index_DIP':.2}
    phase={'edges':[{'side':'right','contact_plan':{'postures':[{'hand_posture_rad':right_open}]}},
                    {'side':'left','contact_plan':{'postures':[{'hand_posture_rad':{}}]}}]}
    planned={**start,'left_thumb_IP':.83,'left_index_DIP':.74}
    result=opening_capture_targets([right,left],phase,start,planned,.5,5.)
    assert result['left_thumb_IP']==.83 and result['left_index_DIP']==.74
    assert result['right_thumb_IP']==pytest.approx((start['right_thumb_IP']+.1)/2)
