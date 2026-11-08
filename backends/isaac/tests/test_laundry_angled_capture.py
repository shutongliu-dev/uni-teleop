"""Opt-in curved-pad geometry never removes stable physical-capture prerequisites."""
from copy import deepcopy
import numpy as np
import pytest
import test_laundry_capture_gate as gate_fixtures
import test_laundry_contact_closure_stop as stop_fixtures
from test_laundry_pinch_geometry import fixture
from scripts.validation.laundry_capture_gate import CaptureGate
from scripts.validation.laundry_contact_closure_stop import ContactClosureStop
from scripts.validation.laundry_contact_assist import CapturedPatchAssist
from scripts.validation.laundry_pinch_geometry import PinchGeometry

PROFILE='angled_adjacent_patch_v4'

def observed_fixture():
    f=gate_fixtures.CaptureGateTests();f.setUp();d=deepcopy(f.diag)
    d['engineering_geometry_profile']=PROFILE
    d['multilayer_diagnostic']['paired_witnesses'][0]['opposing_normal_dot']=-.388
    return f,d

def test_angled_profile_changes_only_angle_and_still_waits_one_hundred_ms():
    f,d=observed_fixture();old=CaptureGate('right',f.tri,f.rest,f.names,f.joints,profile='adjacent_patch_v3')
    g=CaptureGate('right',f.tri,f.rest,f.names,f.joints,profile=PROFILE)
    assert {k:v for k,v in old.thresholds.items() if k!='maximum_opposing_normal_dot'}=={k:v for k,v in g.thresholds.items() if k!='maximum_opposing_normal_dot'}
    for t in np.arange(0,.101,.02):
        r=f.observe(g,float(t),diag=d)
        if t<.1-1.e-9:assert not r['capture_candidate']
    assert r['capture_candidate'] and r['schema']=='laundry_capture_candidate.v4'
    assert r['vertex_ids']==[0,1,2,3] and not r['friction_grasp_qualified']
    assert 'engineering_geometry_profile_mismatch' in f.observe(old,0.,diag=d)['reasons']
    d.pop('engineering_geometry_profile')
    for t in np.arange(0,.101,.02):r=f.observe(old,float(t),diag=d)
    assert not r['capture_candidate']

@pytest.mark.parametrize('kind',['same_direction','too_shallow_angle','deep','moving','sliding','open','large','stale','missing_profile'])
def test_angle_never_bypasses_press_open_slip_deep_or_locality_guards(kind):
    f,d=observed_fixture();rest=f.rest.copy();kw={}
    if kind=='same_direction':d['multilayer_diagnostic']['paired_witnesses'][0]['opposing_normal_dot']=.373
    if kind=='too_shallow_angle':d['multilayer_diagnostic']['paired_witnesses'][0]['opposing_normal_dot']=-.249
    if kind=='deep':d['distal_group_severe_penetration']=True
    if kind=='moving':kw['speed']=[.11,0.]
    if kind=='open':kw['open_gap']=.012
    if kind=='large':rest*=4
    if kind=='stale':kw['stale']=True
    if kind=='missing_profile':d.pop('engineering_geometry_profile')
    g=CaptureGate('right',f.tri,rest,f.names,f.joints,profile=PROFILE)
    for t in np.arange(0,.101,.02):
        if kind=='sliding':kw['cloth']=f.q+[0,0,-.021*t]
        r=f.observe(g,float(t),diag=d,**kw)
    assert not r['capture_candidate']

def test_v4_keeps_unique_single_bridge_and_cannot_attach_an_unseen_back_panel():
    f,d=observed_fixture();tri=np.array([[0,1,2],[1,3,2],[2,3,4]])
    rest=np.vstack([f.rest,[0,.02,0]]);d['multilayer_diagnostic']['paired_witnesses'][0]['index']['triangle_id']=2
    for topology,passed in [(tri,True),(np.array([[0,1,2],[0,1,4],[2,3,4]]),False)]:
        g=CaptureGate('right',topology,rest,f.names,f.joints,profile=PROFILE)
        for t in np.arange(0,.101,.02):r=f.observe(g,float(t),diag=d,cloth=rest+[0,0,.9])
        assert r['capture_candidate']==passed
        if passed:assert r['vertex_ids']==list(range(5))

def test_stop_retains_original_command_but_does_not_grant_capture():
    f=stop_fixtures.ContactClosureStopTests();f.setUp();d=deepcopy(f.diag);d['engineering_geometry_profile']=PROFILE
    d['multilayer_diagnostic']['paired_witnesses'][0]['opposing_normal_dot']=-.388
    p=f.planner(target_policy='hold_command_v2',geometry_profile=PROFILE)
    r=f.observe(p,diagnostic=d)
    assert r['stop_requested'] and r['schema']=='laundry_contact_closure_stop.v4'
    assert not r['capture_candidate'] and not r['attachment_enabled']
    assert r['finger_joint_targets_rad']==dict(zip(f.joints[:2],f.target[:2]))
    d['multilayer_diagnostic']['paired_witnesses'][0]['opposing_normal_dot']=.373
    assert not f.observe(f.planner(geometry_profile=PROFILE),diagnostic=d)['stop_requested']

@pytest.mark.parametrize('capture,stop,geometry,mode',[
    (PROFILE,True,'adjacent_patch_v3','captured_patch'),
    ('adjacent_patch_v3',True,PROFILE,'captured_patch'),
    (PROFILE,True,'shared_edge_v1','captured_patch'),
    (PROFILE,False,PROFILE,'captured_patch'),
    (PROFILE,False,'shared_edge_v1','shared_edge'),
])
def test_assistance_requires_matching_explicit_constructor_profiles(capture,stop,geometry,mode):
    with pytest.raises(ValueError):
        CapturedPatchAssist(None,{},'right',{},None,capture_profile=capture,
            contact_closure_stop=stop,contact_closure_geometry_profile=geometry,
            attachment_material_mode=mode)

def test_geometry_default_shape_preserved_and_v4_propagates_to_all_actual_surfaces():
    export,names,body,q,tri=fixture();shells=deepcopy(export['colliders'])
    for row in shells:
        row['body_name']=row['body_name'].replace('elastomer','DP');row['path']=row['path'].replace('elastomer','DP');names.append(row['body_name'])
    export['colliders'].extend(shells);body=np.vstack([body,body.copy()]);g=PinchGeometry(export,names)
    default=g.diagnose('right',body,q,tri,q,.0012)
    assert default==g.diagnose('right',body,q,tri,q,.0012,engineering_profile=None)
    assert 'engineering_geometry_profile' not in default
    angled=g.diagnose('right',body,q,tri,q,.0012,engineering_profile=PROFILE)
    for row in [angled]+angled['distal_contact_diagnostics']:
        assert row['engineering_geometry_profile']==PROFILE
        assert row['thresholds']['maximum_opposing_normal_dot']==-.25
    with pytest.raises(ValueError):g.diagnose('right',body,q,tri,q,.0012,engineering_profile='loose')
