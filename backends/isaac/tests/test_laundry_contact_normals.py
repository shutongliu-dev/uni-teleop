from copy import deepcopy
import hashlib

import numpy as np
import pytest
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation

from scripts.validation.laundry_pinch_geometry import _surface,PinchGeometry
from scripts.validation.laundry_capture_gate import CaptureGate
from scripts.validation.laundry_contact_closure_stop import ContactClosureStop
from scripts.validation.laundry_capture_settings import DEFAULTS,phase_capture_settings
from scripts.validation.laundry_contact_observation import validate_observation_policy


def cube():
    vertices=np.array([[x,y,z] for x in (-1.,1.) for y in (-1.,1.) for z in (-1.,1.)])
    hull=ConvexHull(vertices)
    return vertices,hull.simplices,hull.equations


@pytest.mark.parametrize('point,closest,normal,signed',[
    ([1.2,.2,.3],[1,.2,.3],[1,0,0],.2),
    ([1.1,1.2,.3],[1,1,.3],[1/np.sqrt(5),2/np.sqrt(5),0],np.sqrt(.05)),
    ([1.1,1.2,1.2],[1,1,1],[1/3,2/3,2/3],.3),
    ([.8,.1,.2],[1,.1,.2],[1,0,0],-.2),
])
def test_signed_distance_gradient_face_edge_vertex_and_interior(point,closest,normal,signed):
    vertices,faces,planes=cube();point=np.asarray(point)
    old=_surface(point,vertices,faces,planes)
    actual=_surface(point,vertices,faces,planes,contact_normal_profile='distance_gradient_v1')
    assert actual[0]==old[0]
    np.testing.assert_array_equal(actual[1],old[1])
    assert actual[0]==pytest.approx(signed)
    np.testing.assert_allclose(actual[1],closest,atol=1e-12)
    np.testing.assert_allclose(actual[2],normal,atol=1e-12)
    if signed>0:
        # Every active face normal spans the actual convex normal cone.
        active=planes[abs(planes[:,:3]@actual[1]+planes[:,3])<1e-10,:3]
        assert np.all(active@actual[2]>=-1e-12)


@pytest.mark.parametrize('point',[[1,.2,.3],[1,1,.3],[1,1,1],[0,0,0]])
def test_nondifferentiable_surface_or_internal_medial_axis_has_no_fallback(point):
    vertices,faces,planes=cube()
    _,_,normal=_surface(np.array(point,float),vertices,faces,planes,contact_normal_profile='distance_gradient_v1')
    assert normal is None
    assert _surface(np.array(point,float),vertices,faces,planes)[2] is not None


def test_gradient_is_rotation_translation_and_face_order_equivariant():
    vertices,faces,planes=cube();point=np.array([1.1,1.2,1.3]);R=Rotation.from_rotvec([.2,-.3,.5]).as_matrix();T=np.array([.4,-.8,1.2])
    baseline=_surface(point,vertices,faces,planes,contact_normal_profile='distance_gradient_v1')
    transformed=vertices@R.T+T;h=ConvexHull(transformed)
    for order in [np.arange(len(h.simplices)),np.arange(len(h.simplices))[::-1]]:
        d,p,n=_surface(point@R.T+T,transformed,h.simplices[order],h.equations[order],contact_normal_profile='distance_gradient_v1')
        assert d==pytest.approx(baseline[0]);np.testing.assert_allclose(p,baseline[1]@R.T+T,atol=1e-12)
        np.testing.assert_allclose(n,R@baseline[2],atol=1e-12)


def test_normal_policy_is_explicit_and_invalid_name_fails():
    vertices,faces,planes=cube();p=np.array([1.1,1.2,0.])
    old=_surface(p,vertices,faces,planes);explicit=_surface(p,vertices,faces,planes,contact_normal_profile='facet_v1')
    for a,b in zip(old,explicit):np.testing.assert_array_equal(a,b)
    with pytest.raises(ValueError):_surface(p,vertices,faces,planes,contact_normal_profile='loose')


def test_all_actual_surface_variants_carry_same_profile_and_surface_ambiguity_rejects():
    from test_laundry_pinch_geometry import fixture
    export,names,poses,q,tri=fixture()
    distal=deepcopy(export['colliders'])
    for row in distal:
        row['body_name']=row['body_name'].replace('elastomer','DP')
        row['path']=row['path'].replace('elastomer','DP');names.append(row['body_name'])
    export['colliders'].extend(distal);poses=np.vstack([poses,poses.copy()])
    geometry=PinchGeometry(export,names)
    baseline=geometry.diagnose('right',poses,q,tri,q,.0012)
    assert baseline==geometry.diagnose('right',poses,q,tri,q,.0012,contact_normal_profile='facet_v1')
    gradient=geometry.diagnose('right',poses,q,tri,q,.0012,contact_normal_profile='distance_gradient_v1')
    for d in [gradient]+gradient['distal_contact_diagnostics']:
        assert d['contact_normal_profile']=='distance_gradient_v1' and d['eligible']
        assert not d['undefined_normal_witnesses']
    on_surface=q+[.001,0,0]
    ambiguous=geometry.diagnose('right',poses,on_surface,tri,q,.0012,contact_normal_profile='distance_gradient_v1')
    assert not ambiguous['eligible']
    assert ambiguous['undefined_normal_witnesses']
    assert not ambiguous['multilayer_diagnostic']['paired_witnesses']


@pytest.fixture
def sample():
    tri=np.array([[0,1,2],[1,3,2]]);rest=np.array([[0,0,0],[.01,0,0],[0,.01,0],[.01,.01,0]])
    names=['right_hand_wrist','right_thumb_fingertip','right_index_fingertip'];joints=['right_thumb_IP','right_index_DIP']
    body=np.array([[0,0,.9,0,0,0,1],[0,0,.9,0,0,0,1],[.01,0,.9,0,0,0,1]],float);q=rest+[0,0,.9]
    diag={'side':'right','distal_group_severe_penetration':False,'contact_force_measured':False,
        'surface_pair':{'thumb':'DP','index':'DP'},'multilayer_diagnostic':{'plausible':True,
         'paired_witnesses':[{'thumb':{'triangle_id':0},'index':{'triangle_id':1},'opposing_normal_dot':-.9}]},
        'state_identity':{'body_q_float64_sha256':hashlib.sha256(body.astype('<f8').tobytes()).hexdigest(),
                          'cloth_q_float64_sha256':hashlib.sha256(q.astype('<f8').tobytes()).hexdigest()}}
    return dict(args=('right',tri,rest,names,joints),body=body,q=q,diag=diag)


def test_gate_and_stop_reject_a_mismatched_normal_even_when_geometry_looks_positive(sample):
    for requested,reported in [('facet_v1','distance_gradient_v1'),('distance_gradient_v1',None)]:
        d=deepcopy(sample['diag'])
        if reported:d['contact_normal_profile']=reported
        gate=CaptureGate(*sample['args'],contact_normal_profile=requested)
        stop=ContactClosureStop(*sample['args'],{'right_thumb_IP':(-1,1),'right_index_DIP':(-1,1)},contact_normal_profile=requested)
        for t in np.arange(0,.101,.02):
            g=gate.observe(t,d,sample['body'],[.5,.5],[0,0],[.51,.51],.06,cloth_q=sample['q'])
            s=stop.observe(t,d,sample['body'],[.5,.5],[.51,.51],.06,cloth_q=sample['q'])
        assert not g['capture_candidate'] and not s['latched']
        assert 'contact_normal_profile_mismatch' in g['reasons'] and 'contact_normal_profile_mismatch' in s['reasons']
        assert not gate.history


def test_explicit_matching_gradient_preserves_all_other_capture_thresholds(sample):
    d=deepcopy(sample['diag']);d['contact_normal_profile']='distance_gradient_v1'
    g=CaptureGate(*sample['args'],contact_normal_profile='distance_gradient_v1')
    assert g.thresholds==CaptureGate(*sample['args']).thresholds
    for t in np.arange(0,.101,.02):r=g.observe(t,d,sample['body'],[.5,.5],[0,0],[.51,.51],.06,cloth_q=sample['q'])
    assert r['capture_candidate'] and r['contact_normal_profile']=='distance_gradient_v1'


def test_global_and_phase_settings_do_not_change_legacy_defaults():
    new={'contact_normal_profile':'distance_gradient_v1',
         'contact_closure_observation_policy':'close_and_hold_v1',
         'contact_closure_stop':True,'contact_closure_target_policy':'hold_command_v2'}
    assert phase_capture_settings({})==DEFAULTS
    assert phase_capture_settings(new)=={**DEFAULTS,**new}
    options={'phases':{'1':{'capture_settings':new}}}
    assert phase_capture_settings(options,0)==DEFAULTS
    assert phase_capture_settings(options,1)=={**DEFAULTS,**new}
    for changes in [{'contact_closure_stop':False},{'contact_closure_target_policy':'reduce_preload_v1'}]:
        with pytest.raises(ValueError,match='Hold-period'):phase_capture_settings({**new,**changes})
    with pytest.raises(ValueError):phase_capture_settings({'contact_normal_profile':'unknown'})
    with pytest.raises(ValueError):validate_observation_policy('close_and_hold_v1',False,'hold_command_v2')
