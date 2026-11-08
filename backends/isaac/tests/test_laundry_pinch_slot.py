from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from scripts.validation.laundry_capture_options import validate_hand_options
from scripts.validation.laundry_captured_garment_fold import prepare_phase
from scripts.validation.laundry_pinch_slot import pinch_slot_translation,slot_entry_bounds


def model(side, vertical=False):
    thumb=np.eye(4);index=np.eye(4)
    thumb[:3,3]=[.03,-.01,.2]
    index[:3,3]=[.03,.01,.2] if not vertical else [.03,-.01,.23]
    return SimpleNamespace(hand=SimpleNamespace(forward=lambda _: {
        side+'_thumb_fingertip':thumb,side+'_index_fingertip':index}))


def contact(xy, angle=0.):
    a=np.deg2rad(angle);w=np.eye(4)
    w[:2,:2]=[[np.cos(a),-np.sin(a)],[np.sin(a),np.cos(a)]]
    w[:3,3]=[*xy,.95]
    return {'wrist_world_matrix':w.tolist(),
            'postures':[{'hand_posture_rad':{}} for _ in range(61)],
            'arm_ik':[{'xy_solved':list(xy)}]}


@pytest.mark.parametrize('side',['left','right'])
def test_translation_uses_original_fk_axis_in_world_and_never_changes_height(side):
    p=contact([.7,-.4],90.);before=deepcopy(p)
    r=pinch_slot_translation(model(side),side,p,.003)
    np.testing.assert_allclose(r['translation_world_m'],[-.003,0.,0.],atol=1.e-15)
    np.testing.assert_allclose(r['horizontal_thumb_to_index_axis_world'],[-1.,0.,0.],atol=1.e-15)
    assert r['reference_posture_index']==30 and p==before
    reverse=pinch_slot_translation(model(side),side,p,-.003)
    np.testing.assert_allclose(reverse['translation_world_m'],[.003,0.,0.],atol=1.e-15)
    with pytest.raises(ValueError,match='horizontal axis'):
        pinch_slot_translation(model(side,vertical=True),side,p,.003)


def fixture():
    q=np.array([[-.15,0.,.901],[-.15,.02,.901],[-.143,.03,.901],[-.143,.05,.901]])
    task={'id':'trousers_folding','asset_metadata':{'design':{
        'fold_sequence_axis_hinge_sign_radius_m':[[0,0.,-1,.007]]},'cloths':[{
        'rest_vertices_m':q.tolist(),'physical_thickness_m':.0018,
        'boundary_features':{'waistband':[0,1],'left_cuff':[2,3]}}]}}
    scene={'table':{'position_m':[0,0,.8],'half_extents_m':[1,1,.1]}}
    return task,scene,q


def options(offset=None):
    left={'yaw_deg':0.}
    if offset is not None:left['pinch_slot_offset_m']=offset
    return {'phases':{'0':{'sides':['left','right'],'hands':{'left':left,'right':{'yaw_deg':0.}}}}}


def test_prepare_replans_ik_preserves_material_other_hand_and_zero_legacy_result():
    task,scene,q=fixture();before=q.copy();calls=[]
    def planner(release,side,xy,*args):
        calls.append((side,list(xy)));return contact(xy,args[0])
    geometry=SimpleNamespace(sides={s:model(s) for s in ['left','right']})
    with patch('scripts.validation.laundry_captured_garment_fold.pinch_plan',side_effect=planner), \
         patch('bench.control.factory.robot_configuration',return_value={}), \
         patch('bench.control.factory.public_robot_geometry',return_value=geometry):
        legacy=prepare_phase(None,task,scene,q,0,options())
        calls.clear();zero=prepare_phase(None,task,scene,q,0,options(0.))
        assert zero==legacy and len(calls)==2
        calls.clear();changed=prepare_phase(None,task,scene,q,0,options(.003))
    assert [x[0] for x in calls]==['left','left','right']
    np.testing.assert_allclose(np.array(calls[1][1])-calls[0][1],[0.,.003])
    assert changed['edges'][1]==legacy['edges'][1]
    a,b=changed['edges'][0],legacy['edges'][0]
    for key in ['seed_vertex_ids','allowed_material_vertex_ids','edge_world_m','edge_material_m']:
        assert a[key]==b[key]
    assert a['contact_plan']['arm_ik'][0]['xy_solved']==calls[1][1]
    assert a['contact_plan']['slot_centering']['approach_ik_replanned']
    np.testing.assert_array_equal(q,before)


def test_shifted_ik_failure_is_not_replaced_by_unshifted_solution():
    task,scene,q=fixture();calls=[]
    def planner(release,side,xy,*args):
        calls.append(list(xy))
        if len(calls)==2:raise ValueError('shifted IK rejected')
        return contact(xy,args[0])
    with patch('scripts.validation.laundry_captured_garment_fold.pinch_plan',side_effect=planner), \
         patch('bench.control.factory.robot_configuration',return_value={}), \
         patch('bench.control.factory.public_robot_geometry',return_value=SimpleNamespace(sides={'left':model('left')})):
        with pytest.raises(ValueError,match='shifted IK rejected'):
            prepare_phase(None,task,scene,q,0,options(.003))


@pytest.mark.parametrize('value',[True,False,.005001,-.005001,float('nan'),float('inf'),'0.003',None])
def test_invalid_later_unused_hand_offset_fails_before_geometry(value):
    settings={'phases':{'9':{'hands':{'right':{'pinch_slot_offset_m':value}}}}}
    with pytest.raises(ValueError,match='pinch_slot_offset_m'):
        prepare_phase(None,{},None,None,0,settings,only_sides=['left'])


def test_towel_rejects_garment_slot_even_when_zero():
    for value in [0.,.003]:
        with pytest.raises(ValueError,match='Unknown hand options'):
            validate_hand_options(options(value),towel=True)


def test_combined_entry_checks_actual_inward_projection_not_separate_magnitudes():
    pick={'inward_world_xy':[1.,0.],'edge_world_m':[0.,0.,.9],'pinch_tcp_xy_m':[-.003,0.]}
    feed={'translation_world_m':[.009,0.,0.]}
    sideways={'translation_world_m':[0.,.005,0.]}
    result=slot_entry_bounds(pick,sideways,feed)
    assert result['effective_start_inward_offset_m']==pytest.approx(-.003)
    assert result['effective_end_inward_offset_m']==pytest.approx(.006)
    assert result['slot_tangent_component_m']==.005
    for start,shift in [(-.003,.005),(-.006,-.004),(-.008,-.003)]:
        with pytest.raises(ValueError,match='Combined slot/feed'):
            slot_entry_bounds({**pick,'pinch_tcp_xy_m':[start,0.]},
                              {'translation_world_m':[shift,0.,0.]},feed)
    with pytest.raises(ValueError,match='within10mm'):
        slot_entry_bounds({**pick,'pinch_tcp_xy_m':[.006,0.]},
                          {'translation_world_m':[.005,0.,0.]})
