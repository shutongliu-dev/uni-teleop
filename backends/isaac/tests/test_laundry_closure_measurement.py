from copy import deepcopy
import builtins
import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.validation.laundry_capture_gate import CaptureGate
from scripts.validation.laundry_contact_closure_stop import ContactClosureStop
from scripts.validation.laundry_contact_assist import CapturedPatchAssist
from scripts.validation.laundry_capture_settings import (
    DEFAULTS, phase_capture_settings, planned_capture_openings)


@pytest.fixture
def observed():
    rest=np.array([[0,0,0],[.01,0,0],[0,.01,0],[.01,.01,0]],float)
    q=rest+[0,0,.9]
    body=np.array([[0,0,.9,0,0,0,1],[0,0,.9,0,0,0,1],[.0177,0,.9,0,0,0,1]])
    return {'tri':np.array([[0,1,2],[1,3,2]]),'rest':rest,'q':q,'body':body,
            'bodies':['right_hand_wrist','right_thumb_fingertip','right_index_fingertip'],
            'joints':['right_thumb_IP','right_index_DIP']}


def diagnostic(data, **changes):
    result={'side':'right','distal_group_severe_penetration':False,'contact_force_measured':False,
            'surface_pair':{'thumb':'DP','index':'DP'},
            'multilayer_diagnostic':{'plausible':True,'paired_witnesses':[
                {'thumb':{'triangle_id':0},'index':{'triangle_id':1},'opposing_normal_dot':-.9}]},
            'state_identity':{'body_q_float64_sha256':hashlib.sha256(data['body'].astype('<f8').tobytes()).hexdigest(),
                              'cloth_q_float64_sha256':hashlib.sha256(data['q'].astype('<f8').tobytes()).hexdigest()}}
    result.update(changes)
    return result


def planners(data, profile='narrow_opening_v1', planned=.025):
    args=('right',data['tri'],data['rest'],data['bodies'],data['joints'])
    kwargs={'closure_measurement_profile':profile,'planned_open_tip_gap_m':planned}
    stop=ContactClosureStop(*args,{n:(-1.,1.) for n in data['joints']},
        target_policy='hold_command_v2',geometry_profile='adjacent_patch_v3',**kwargs)
    gate=CaptureGate(*args,profile='adjacent_patch_v3',**kwargs)
    return stop,gate


def stop_once(stop,data,opening=.0246,diag=None):
    return stop.observe(0.,diagnostic(data) if diag is None else diag,data['body'],[.5,.5],
                        [.51,.51],opening,cloth_q=data['q'],allowed_material_vertex_ids=range(4))


def gate_once(gate,data,t,opening=.0246,diag=None,**kw):
    return gate.observe(t,diagnostic(data) if diag is None else diag,data['body'],[.5,.5],
                        kw.get('speed',[0.,0.]),kw.get('target',[.51,.51]),opening,cloth_q=data['q'])


def test_narrow_stop_keeps_real_geometry_and_independent_hundred_ms_gate(observed):
    stop,gate=planners(observed)
    stopped=stop_once(stop,observed)
    assert stopped['latched'] and not stopped['capture_candidate']
    assert stopped['closure_measurement']['minimum_required_closure_m']==.005
    assert stopped['finger_joint_targets_rad']=={'right_thumb_IP':.51,'right_index_DIP':.51}
    for t in np.arange(0,.1,.02):
        assert not gate_once(gate,observed,float(t))['capture_candidate']
    result=gate_once(gate,observed,.1)
    assert result['capture_candidate']
    assert result['thresholds']['minimum_tip_closing_m']==.005
    assert result['thresholds']['minimum_duration_s']==.1
    assert not result['friction_grasp_qualified']


def test_legacy_defaults_remain_exact_and_do_not_inherit_five_mm(observed):
    args=('right',observed['tri'],observed['rest'],observed['bodies'],observed['joints'])
    legacy=ContactClosureStop(*args,{n:(-1.,1.) for n in observed['joints']},
        target_policy='hold_command_v2',geometry_profile='adjacent_patch_v3')
    legacy_gate=CaptureGate(*args,profile='adjacent_patch_v3')
    explicit,explicit_gate=planners(observed,'strict_v1',None)
    assert stop_once(legacy,observed)==stop_once(explicit,observed)
    assert not stop_once(explicit,observed)['latched']
    assert 'closure_measurement' not in stop_once(legacy,observed)
    for t in np.arange(0,.121,.02):
        assert gate_once(legacy_gate,observed,float(t))==gate_once(explicit_gate,observed,float(t))
    assert not gate_once(legacy_gate,observed,.14)['capture_candidate']
    assert legacy_gate.thresholds['minimum_tip_closing_m']==.008
    assert planners(observed)[1].thresholds=={**legacy_gate.thresholds,'minimum_tip_closing_m':.005}
    assert phase_capture_settings({})==DEFAULTS


@pytest.mark.parametrize('planned',[None,.019,.031,.06,True,float('nan')])
def test_wide_or_unbound_command_cannot_claim_narrow_opening(observed,planned):
    with pytest.raises(ValueError,match='planned opening'):
        planners(observed,planned=planned)


@pytest.mark.parametrize('opening',[.019,.031,.06])
def test_measured_opening_must_also_be_narrow(observed,opening):
    stop,gate=planners(observed)
    assert not stop_once(stop,observed,opening)['latched']
    for t in np.arange(0,.121,.02):
        result=gate_once(gate,observed,float(t),opening)
    assert not result['capture_candidate']
    assert 'narrow_opening_baseline_outside_20_to_30mm' in result['reasons']


@pytest.mark.parametrize('opening,closing,expected',[(.025,.0049,False),(.030,.0059,False),(.030,.0061,True)])
def test_five_mm_and_twenty_percent_are_both_required(observed,opening,closing,expected):
    observed['body'][2,0]=opening-closing
    stop,gate=planners(observed)
    assert stop_once(stop,observed,opening)['latched'] is expected
    for t in np.arange(0,.101,.02):
        result=gate_once(gate,observed,float(t),opening)
    assert result['capture_candidate'] is expected


@pytest.mark.parametrize('failure',['missing','deep','stale','normal','remote_material','speed','error'])
def test_narrow_measurement_does_not_weaken_other_capture_guards(observed,failure):
    if failure=='remote_material':observed['rest'][3,0]=.1
    stop,gate=planners(observed);diag=diagnostic(observed);kw={}
    if failure=='missing':diag['multilayer_diagnostic']['paired_witnesses']=[]
    if failure=='deep':diag['distal_group_severe_penetration']=True
    if failure=='stale':diag['state_identity']={}
    if failure=='normal':diag['multilayer_diagnostic']['paired_witnesses'][0]['opposing_normal_dot']=.2
    if failure=='speed':kw['speed']=[.11,0.]
    if failure=='error':kw['target']=[.541,.51]
    if failure not in ('speed','error'):
        assert not stop_once(stop,observed,diag=diag)['latched']
    for t in np.arange(0,.121,.02):
        result=gate_once(gate,observed,float(t),diag=diag,**kw)
    assert not result['capture_candidate']


def test_phase_option_binds_all_declared_hands_before_startup():
    options={'closure_measurement_profile':'narrow_opening_v1','phases':{
        '0':{'capture_order':['right','left'],'hands':{
            'right':{'open_distance_m':.025},'left':{'open_distance_m':.030}}},
        '1':{'capture_settings':{'closure_measurement_profile':'strict_v1'}}}}
    before=deepcopy(options)
    assert planned_capture_openings(options,0,phase_capture_settings(options,0))=={'right':.025,'left':.030}
    assert planned_capture_openings(options,1,phase_capture_settings(options,1))=={}
    assert options==before
    for invalid in ({}, {'hands':{'right':{'open_distance_m':.025}}},
                    {'capture_order':[{}]},
                    {'capture_order':['right'],'hands':{'right':{'open_distance_m':.060}}}):
        bad={**options,'phases':{'0':invalid}}
        with pytest.raises(ValueError):planned_capture_openings(bad,0,phase_capture_settings(bad,0))


def test_wide_command_is_rejected_by_native_entry_before_backend_import(tmp_path,monkeypatch):
    import scripts.maintenance.freeze_project_sources as freeze
    import bench.isaacsim.release as release_module
    from scripts.validation.laundry_captured_garment_fold import native
    release=tmp_path/'release';release.mkdir()
    rows={'task.json':{'id':'shirt_folding'},'initial_state.json':{},
          'scene.json':{'named_cloth_initialization':'author-only','cloths':[{'id':'towel'}],
                        'robot':{'hand_id':'dual_fr3_sharpa_w01'}}}
    for name,row in rows.items():(release/name).write_text(json.dumps(row))
    options=tmp_path/'options.json'
    options.write_text(json.dumps({'closure_measurement_profile':'narrow_opening_v1',
        'phases':{'0':{'capture_order':['left'],'hands':{'left':{'open_distance_m':.060}}}}}))
    monkeypatch.setattr(freeze,'verify_if_snapshot',lambda *a,**k:{'verified':True})
    monkeypatch.setattr(release_module,'validate',lambda _: {'task_id':'shirt_folding'})
    original_import=builtins.__import__;attempted=[]
    def guarded_import(name,*args,**kwargs):
        if name=='bench.simulation.isaacsim_backend.env':
            attempted.append(name);raise AssertionError('Must validate narrow planned opening first')
        return original_import(name,*args,**kwargs)
    monkeypatch.setattr(builtins,'__import__',guarded_import)
    output=tmp_path/'native_output'
    with pytest.raises(ValueError,match='planned opening'):
        native(SimpleNamespace(release=release,options=options,output=output,max_phases=1,peak_height=.12))
    assert not attempted and not output.exists()


def test_real_assist_constructs_stop_and_gate_with_the_same_bound_profile(observed,tmp_path,monkeypatch):
    import scripts.validation.laundry_pinch_geometry as geometry
    monkeypatch.setattr(geometry,'PinchGeometry',lambda *a:SimpleNamespace())
    env=SimpleNamespace(body_names=observed['bodies'],joint_names=observed['joints'],
        actuated_joint_names=observed['joints'],joint_limits={n:(-1.,1.) for n in observed['joints']})
    scene={'cloths':[{'id':'towel','rest_vertices_m':observed['rest'].tolist(),
                     'triangles':observed['tri'].tolist(),'isaacsim':{'cloth_thickness_m':.004}}]}
    assist=CapturedPatchAssist(env,scene,'right',{},tmp_path,capture_profile='adjacent_patch_v3',
        contact_closure_stop=True,contact_closure_geometry_profile='adjacent_patch_v3',
        closure_measurement_profile='narrow_opening_v1',planned_open_tip_gap_m=.025)
    try:
        assert assist.gate.closure_measurement_profile==assist.closure_stop.closure_measurement_profile=='narrow_opening_v1'
        assert assist.gate.planned_open_tip_gap_m==assist.closure_stop.planned_open_tip_gap_m==.025
        assert not assist.active and not assist.events
    finally:assist.close()
