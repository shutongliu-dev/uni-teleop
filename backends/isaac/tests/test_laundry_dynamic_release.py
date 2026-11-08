from types import SimpleNamespace
import numpy as np
import pytest
from scripts.validation.laundry_dynamic_release import (
    SAMPLES, height_envelope, opening_lift_offset, place_arc, rebind_opening)
from scripts.validation.laundry_phase_motions import phase_release_settings
from scripts.validation.laundry_release_hand import release_hand_profile


def fixture():
    class Hand:
        active=[SimpleNamespace(name='left_index_PIP',lower=0.,upper=1.)]
        def forward(self, q):
            u=q['left_index_PIP'];T=np.eye(4);T[2,3]=-4*u*(1-u)*.02
            return {'left_index':T}
    model=SimpleNamespace(hand=Hand(),hand_posture_prior=SimpleNamespace(violation=lambda *_:False))
    F=np.eye(4);F[2,3]=1.
    export={'colliders':[{'body_name':'left_index','mesh_to_body_column_matrix':np.eye(4).tolist(),
        'pieces':[{'conservative_outer_vertices':[[-.01,-.01,0.],[.01,.01,.01]]}]}]}
    return dict(side='left',model=model,actual_joint_positions={'left_index_PIP':0.},
        actual_flange=F,body_q=np.array([[0,0,1,0,0,0,1]],dtype=float),body_names=['left_index'],colliders=export)


def test_intermediate_curled_finger_requires_non_linear_lift_without_high_release():
    kw=fixture();postures=[{'hand_posture_rad':{'left_index_PIP':u}}for u in(0.,1.)]
    arc={'controls_m':[[0,0,1],[.1,0,1.2],[.2,0,.91]]}
    cloth=np.array([[.2,0,.902]])
    before=cloth.copy()
    planned=place_arc(arc,**kw,postures=postures,cloth_q=cloth,table_z=.9,thickness=.002,peak=.06)
    receipt=planned['release_clearance'];z=planned['controls_m'][-1][2]
    assert z==pytest.approx(.910)
    assert max(receipt['monotone_wrist_z_m'])==pytest.approx(.930)
    F=kw['actual_flange'].copy();F[:3,3]=[.2,0,z]
    kw['actual_flange']=F;kw['body_q']=np.array([[.2,0,z,0,0,0,1]])
    lift=rebind_opening(receipt,**kw,start_targets=postures[0]['hand_posture_rad'],opened=postures[1]['hand_posture_rad'])
    assert opening_lift_offset(lift,0)==pytest.approx(0.)
    assert opening_lift_offset(lift,.5)==pytest.approx(.020)
    assert opening_lift_offset(lift,1)==pytest.approx(.020)
    assert opening_lift_offset(lift,.25)>.02*.25
    assert np.array_equal(cloth,before)
    assert arc['controls_m'][-1][2]==.91


def test_upper_bin_envelope_never_loses_sampled_clearance():
    x=np.linspace(0,1,SAMPLES);required,profile=height_envelope(-.02*np.sin(x*np.pi),.9,thickness=.002,model_error_m=.0001)
    plan={'wrist_lift_m':(profile-profile[0]).tolist()}
    for i,u in enumerate(x):assert profile[0]+opening_lift_offset(plan,u)>=required[i]-1e-14
    assert np.all(np.diff(profile)>=0)
    with pytest.raises(ValueError):opening_lift_offset(plan,float('nan'))
    with pytest.raises(ValueError):opening_lift_offset({'wrist_lift_m':[0.]},.5)
    with pytest.raises(ValueError):height_envelope(np.zeros(61),.9,thickness=.002,model_error_m=0.)


def test_requires_original_actual_fk_identity_and_same_held_targets():
    kw=fixture();postures=[{'hand_posture_rad':{'left_index_PIP':u}}for u in(0.,1.)]
    arc={'controls_m':[[0,0,1],[.1,0,1.2],[.2,0,.91]]}
    receipt=place_arc(arc,**kw,postures=postures,cloth_q=np.array([[.2,0,.902]]),table_z=.9,thickness=.002,peak=.06)['release_clearance']
    with pytest.raises(ValueError,match='changed'):
        rebind_opening(receipt,**kw,start_targets={'left_index_PIP':.01},opened={'left_index_PIP':1.})
    kw['body_q'][0,2]+=.003
    with pytest.raises(ValueError,match='FK mismatch'):
        rebind_opening(receipt,**kw,start_targets={'left_index_PIP':0.},opened={'left_index_PIP':1.})


def test_explicit_profile_and_per_phase_scope_only():
    old={'release_profile':'cooked_hand_clearance_v2','release_hand_profile':'full_spread_extend_v2'}
    assert release_hand_profile(old)=='full_spread_extend_v2'
    options={'phases':{'1':{'release_settings':{'release_profile':'cooked_opening_lift_v3','release_hand_profile':'full_spread_extend_v2'}}}}
    assert phase_release_settings(options,0)=={'release_profile':'fixed_v1','release_hand_profile':'pinch_open_v1','retreat_profile':'base_directed_v1'}
    assert phase_release_settings(options,1)['release_profile']=='cooked_opening_lift_v3'
    for hand in('pinch_open_v1','distal_extend_v1'):
        with pytest.raises(ValueError):phase_release_settings({'release_profile':'cooked_opening_lift_v3','release_hand_profile':hand})
    with pytest.raises(ValueError):phase_release_settings({'release_profile':'cooked_opening_lift_v3','release_hand_profile':'full_spread_extend_v2','transport_mode':'capture_lift_probe'})


def test_original_posture_prior_is_checked_inside_dense_opening():
    kw=fixture()
    kw['model'].hand_posture_prior=SimpleNamespace(violation=lambda _,q: .499 < q['left_index_PIP'] < .501)
    with pytest.raises(ValueError,match='posture prior'):
        place_arc({'controls_m':[[0,0,1],[.1,0,1.2],[.2,0,.91]]},**kw,
            postures=[{'hand_posture_rad':{'left_index_PIP':v}}for v in(0.,1.)],
            cloth_q=np.array([[.2,0,.902]]),table_z=.9,thickness=.002,peak=.06)
