from copy import deepcopy
import hashlib
import io
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.validation.laundry_capture_region import (
    MaterialCaptureRegion, stack_region_descriptor, validate_region_option)
from scripts.validation.laundry_capture_gate import CaptureGate
from scripts.validation.laundry_contact_closure_stop import ContactClosureStop


@pytest.fixture
def data():
    # A true contact face extends6mm beyond the unchanged30mm seed disk.
    rest=np.array([[.02,-.03,0],[.01,-.02,0],[.02,-.02,0],
                   [0,-.01,0],[.01,-.01,0],[0,0,0]])
    tri=np.array([[0,2,1],[1,2,4],[1,4,3]])
    q=rest+[0,0,.9]
    descriptor=stack_region_descriptor({'seed_vertex_ids':[5],'edge_material_m':[0,.2,0]})
    names=['left_hand_wrist','left_thumb_fingertip','left_index_fingertip']
    joints=['left_thumb_IP','left_index_DIP']
    body=np.array([[0,0,.9,0,0,0,1],[0,0,.9,0,0,0,1],[.02,0,.9,0,0,0,1]])
    w={'opposing_normal_dot':-.9}
    for side,face,bary in [('thumb',0,[.2,.6,.2]),('index',2,[.1,.3,.6])]:
        w[side]={'triangle_id':face,'rest_point_m':(bary@rest[tri[face]]).tolist(),
                 'point_m':(bary@q[tri[face]]).tolist()}
    diag={'side':'left','distal_group_severe_penetration':False,'contact_force_measured':False,
        'surface_pair':{'thumb':'DP','index':'DP'},'distal_contact_diagnostics':[],
        'multilayer_diagnostic':{'plausible':True,'paired_witnesses':[w]},
        'state_identity':{'body_q_float64_sha256':hashlib.sha256(body.astype('<f8').tobytes()).hexdigest(),
                          'cloth_q_float64_sha256':hashlib.sha256(q.astype('<f8').tobytes()).hexdigest()}}
    return dict(rest=rest,tri=tri,q=q,descriptor=descriptor,names=names,joints=joints,
                body=body,w=w,diag=diag,allowed=[1,2,3,4,5],
                topology={'relation':'single_bridge_triangle','bridge_triangle_id':1,
                          'support_triangle_ids':[0,1,2]})


def check(d,**changes):
    return MaterialCaptureRegion(d['descriptor'],d['tri'],d['rest']).check(
        changes.get('w',d['w']),changes.get('topology',d['topology']),
        changes.get('q',d['q']),changes.get('allowed',d['allowed']))


def test_actual_contact_point_inside_region_allows_its_original_face_support(data):
    r=check(data)
    assert r['eligible']
    assert r['vertex_ids']==[0,1,2,3,4]
    assert len(r['additional_contact_support_vertices'])==1
    extra=r['additional_contact_support_vertices'][0]
    assert extra['vertex_id']==0 and extra['distance_to_seed_m']>.03
    assert extra['contact_face_bounds'][0]['contact_triangle_id']==0
    assert r['material_diameter_m']<.035


@pytest.mark.parametrize('failure',['outside_radius','outside_x','wrong_world','outside_face','bridge_only','large_patch'])
def test_support_exception_never_expands_actual_witness_or_bridge_region(data,failure):
    d=deepcopy(data)
    if failure=='outside_radius':
        bary=np.array([.7,.2,.1]);d['w']['thumb'].update(
            rest_point_m=(bary@d['rest'][d['tri'][0]]).tolist(),
            point_m=(bary@d['q'][d['tri'][0]]).tolist())
    elif failure=='outside_x':
        # Translating the whole contact patch preserves all barycentric/locality
        # evidence, but puts the actual thumb outside the unchanged crease slab.
        for v in [d['rest'],d['q']]:v[:5,0]+=.01
        for s in ['thumb','index']:
            for key in ['rest_point_m','point_m']:d['w'][s][key][0]+=.01
    elif failure=='wrong_world':d['w']['thumb']['point_m'][0]+=.001
    elif failure=='outside_face':
        d['w']['thumb']['rest_point_m']=[.018,-.019,0]
        d['w']['thumb']['point_m']=[.018,-.019,.9]
    elif failure=='bridge_only':
        d['rest']=np.vstack([d['rest'],[.021,-.025,0]])
        d['q']=np.vstack([d['q'],[.021,-.025,.9]])
        d['tri'][1]=[1,6,4]
    elif failure=='large_patch':
        d['rest'][0,1]=-.05;d['q'][0,1]=-.05
        bary=np.array([.05,.5,.45]);d['w']['thumb'].update(
            rest_point_m=(bary@d['rest'][d['tri'][0]]).tolist(),
            point_m=(bary@d['q'][d['tri'][0]]).tolist())
    assert not check(d)['eligible']


def gate(d,region=True):
    return CaptureGate('left',d['tri'],d['rest'],d['names'],d['joints'],profile='loaded_adjacent_patch_v5',
        **({'material_capture_region':d['descriptor']} if region else {}))


def observe(d,g,t,diag=None):
    return g.observe(t,diag or d['diag'],d['body'],[.5,.5],[0.,0.],[.565,.5],.06,
                     cloth_q=d['q'],allowed_material_vertex_ids=d['allowed'])


def test_region_is_filtered_before_history_and_never_counts_prior_outside_time(data):
    g=gate(data);bad=deepcopy(data['diag']);w=bad['multilayer_diagnostic']['paired_witnesses'][0]
    bary=np.array([.7,.2,.1]);w['thumb']['rest_point_m']=(bary@data['rest'][data['tri'][0]]).tolist()
    w['thumb']['point_m']=(bary@data['q'][data['tri'][0]]).tolist()
    for t in [0,.02,.04,.06,.08]:
        assert not observe(data,g,t,bad)['capture_candidate']
        assert not g.history
    for t in [.10,.12,.14,.16,.18]:assert not observe(data,g,t)['capture_candidate']
    result=observe(data,g,.20)
    assert result['capture_candidate'] and result['continuous_duration_s']==pytest.approx(.1)
    assert result['evidence']['first_time_s']==.10
    assert len(result['evidence']['material_region_window'])==6
    assert all(x['evidence']['eligible'] for x in result['evidence']['material_region_window'])
    assert not observe(data,g,.22,bad)['capture_candidate']
    assert not g.history
    assert observe(data,g,.24)['continuous_duration_s']==0


@pytest.mark.parametrize('failure',['normal','deep','speed','error'])
def test_region_does_not_bypass_geometry_or_loaded_stability(data,failure):
    g=gate(data);d=deepcopy(data['diag']);speed=[0.,0.];target=[.565,.5]
    if failure=='normal':d['multilayer_diagnostic']['paired_witnesses'][0]['opposing_normal_dot']=-.4
    elif failure=='deep':d['distal_group_severe_penetration']=True
    elif failure=='speed':speed=[.101,0]
    else:target=[.581,.5]
    for t in np.arange(0,.101,.02):
        r=g.observe(t,d,data['body'],[.5,.5],speed,target,.06,cloth_q=data['q'],
                    allowed_material_vertex_ids=data['allowed'])
    assert not r['capture_candidate']


def test_stop_and_gate_use_the_same_per_witness_region(data):
    args=('left',data['tri'],data['rest'],data['names'],data['joints'],
          {n:(-1,2) for n in data['joints']})
    default=ContactClosureStop(*args,geometry_profile='adjacent_patch_v3',target_policy='hold_command_v2')
    region=ContactClosureStop(*args,geometry_profile='adjacent_patch_v3',target_policy='hold_command_v2',
                              material_capture_region=data['descriptor'])
    def call(g):return g.observe(1.,data['diag'],data['body'],[.5,.5],[.565,.5],.06,
                                 cloth_q=data['q'],allowed_material_vertex_ids=data['allowed'])
    assert not call(default)['latched']
    new=call(region)
    assert new['latched'] and new['evidence']['material_region']['eligible']
    assert new['finger_joint_targets_rad']=={'left_thumb_IP':.565,'left_index_DIP':.5}
    assert new['capture_candidate'] is False and new['attachment_enabled'] is False


def test_default_gate_output_and_allowed_handling_stay_unchanged(data):
    a=gate(data,False);b=gate(data,False)
    for t in np.arange(0,.101,.02):
        r=observe(data,a,t)
        s=b.observe(t,data['diag'],data['body'],[.5,.5],[0.,0.],[.565,.5],.06,cloth_q=data['q'])
        assert r==s and 'material_region_profile' not in r


@pytest.mark.parametrize('task,towel',[('shirt_folding',False),('trousers_folding',False),
                                      ('towel_folding',True),('towel_flattening',True),('towel_stacking',False)])
def test_explicit_profile_is_not_silently_accepted_for_other_adapters(task,towel):
    with pytest.raises(ValueError,match='restricted'):
        validate_region_option({'capture_region_profile':'region_support_v1'},task,towel_adapter=towel)
    assert validate_region_option({},task,towel_adapter=towel)=='strict_vertices_v1'


def test_only_fixed_original_material_descriptor_is_accepted(data):
    assert validate_region_option({'capture_region_profile':'region_support_v1'},'towel_stacking',towel_adapter=True)=='region_support_v1'
    for key,value in [('radius_m',.04),('relative_material_x_half_width_m',.03),
                      ('seed_vertex_id',True),('seed_canonical_material_m',[.01,.2,0])]:
        d=deepcopy(data['descriptor']);d[key]=value
        with pytest.raises(ValueError):MaterialCaptureRegion(d,data['tri'],data['rest'])


def test_adapter_opt_in_changes_only_region_metadata_and_uses_original_material_seed(monkeypatch):
    from scripts.validation import laundry_captured_towel_tasks as adapter
    x,y=np.meshgrid(np.linspace(-.3,.3,61),np.linspace(-.2,.2,41))
    rest=np.column_stack([x.ravel(),y.ravel(),np.zeros(x.size)])
    task={'id':'towel_stacking','asset_metadata':{'physical_thickness_m':.004,'table_top_z_m':.9,
          'cloths':[{'id':'towel','rest_vertices_m':rest.tolist()}]}}
    scene={'cloths':[{'id':'towel'},{'id':'towel_base'}]};q=rest+[.5,-.6,.91];before=q.copy()
    monkeypatch.setattr(adapter,'pinch_plan',lambda *args,**kw:{})
    old=adapter.prepare_phase(None,task,scene,q,0,{})
    new=adapter.prepare_phase(None,task,scene,q,0,{'capture_region_profile':'region_support_v1'})
    for a,b in zip(old['edges'],new['edges']):
        region=b.pop('material_capture_region')
        assert region['seed_vertex_id']==a['seed_vertex_ids'][0]
        assert region['seed_canonical_material_m']==a['edge_material_m']
        assert b==a
    assert new==old
    np.testing.assert_array_equal(q,before)
    task['id']='towel_folding'
    with pytest.raises(ValueError,match='restricted'):
        adapter.prepare_phase(None,task,{'cloths':[{'id':'towel'}]},q,0,
                              {'capture_region_profile':'region_support_v1'})


def test_assist_handoff_uses_per_sample_region_while_default_final_ids_remain_strict(data):
    from scripts.validation.laundry_contact_assist import CapturedPatchAssist
    for region,expected in [(False,False),(True,True)]:
        a=CapturedPatchAssist.__new__(CapturedPatchAssist)
        a.gate=gate(data,region);a.geometry=SimpleNamespace(diagnose=lambda *args,**kw:deepcopy(data['diag']))
        a.geometry_options={};a.triangles=data['tri'];a.rest=data['rest'];a.thickness=.004
        a.side='left';a.thumb_index=1;a.index_index=2;a.wrist_index=0
        a.active=False;a.prim=None;a.open_tip_gap_m=.06;a.last_observe_time=-np.inf
        a.allowed_ids=set(data['allowed']);a.diagnostics=io.StringIO()
        a.closure_stop=SimpleNamespace(latched=True);a.attachment_material_mode='captured_patch'
        enabled=[];a._enable=lambda t,result,bodies,q:enabled.append((t,result))
        state={'robot_body_q':data['body'],'joint_q':np.array([.5,.5]),'joint_qd':np.zeros(2)}
        for t in np.arange(0,.101,.02):a.update('closed_hold',float(t),state,data['q'],np.array([.565,.5]))
        assert bool(enabled)==expected
        if expected:
            assert enabled[0][0]==pytest.approx(.1)
            assert enabled[0][1]['vertex_ids']==[0,1,2,3,4]
            assert enabled[0][1]['evidence']['material_region']['eligible']
