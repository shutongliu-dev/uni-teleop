from copy import deepcopy
from unittest.mock import patch

import numpy as np
import pytest

from scripts.validation.laundry_capture_options import validate_hand_options
from scripts.validation.laundry_captured_towel_tasks import prepare_phase, towel_picks


def fixture(task_id='towel_flattening'):
    x,y=np.meshgrid(np.linspace(-.3,.3,61),np.linspace(-.2,.2,41))
    rest=np.column_stack([x.ravel(),y.ravel(),np.zeros(x.size)])
    task={'id':task_id,'asset_metadata':{'physical_thickness_m':.004,'table_top_z_m':.9,
        'reset_profile':'towel_partial_edge_fold_v1','cloths':[{'id':'towel','rest_vertices_m':rest.tolist()}]}}
    return task,{'cloths':[{'id':'towel'}]},rest


def options(setting,phase='1'):
    return {'phases':{phase:{'hands':{'right':setting}}}}


@pytest.mark.parametrize('task_id,index',[('towel_flattening',1),('towel_folding',0)])
def test_tangent_recomputes_from_each_observed_material_heading(task_id,index):
    task,scene,rest=fixture(task_id)
    settings={'phases':{str(index):{'hands':{
        'right':{'corner_tangent_sign':1},'left':{'corner_tangent_sign':-1}}}}}
    original=deepcopy(settings)
    for theta in (.18,-.37):
        rotation=np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
        q=rest.copy();q[:,:2]=q[:,:2]@rotation.T+[.8,-.55];q[:,2]=.902;before=q.copy()
        with patch('scripts.validation.laundry_captured_towel_tasks.pinch_plan',side_effect=lambda *a,**k:{'called_yaw_deg':a[3]}):
            result=prepare_phase(None,task,scene,q,index,settings)
        original_picks={e['side']:e for e in towel_picks(task,q,index)}
        for edge in result['edges']:
            side=edge['side'];upper=side=='left';chirality=-1 if upper else 1
            inward=rotation@np.array([-1.,-1. if upper else 1.])/np.sqrt(2.)
            normal_yaw=np.rad2deg(np.arctan2(chirality*inward[0],-chirality*inward[1]))%360
            sign=settings['phases'][str(index)]['hands'][side]['corner_tangent_sign']
            expected=(normal_yaw+90*sign)%360
            assert edge['contact_plan']['called_yaw_deg']==pytest.approx(expected)
            assert edge['contact_plan']['yaw_definition']['commanded_yaw_deg']==pytest.approx(expected)
            assert edge['contact_plan']['inside_finger']=='tangent'
            assert edge['seed_vertex_ids']==original_picks[side]['seed_vertex_ids']
            assert edge['allowed_material_vertex_ids']==original_picks[side]['allowed_material_vertex_ids']
            np.testing.assert_allclose(edge['pinch_tcp_xy_m'],np.asarray(edge['edge_world_m'])[:2]+.006*inward)
        np.testing.assert_array_equal(q,before)
    assert settings==original


def test_explicit_tangent_preserves_other_hand_and_absent_legacy_metadata():
    task,scene,rest=fixture('towel_folding');q=rest+[.8,-.55,.902]
    with patch('scripts.validation.laundry_captured_towel_tasks.pinch_plan',side_effect=lambda *a,**k:{'called_yaw_deg':a[3]}):
        legacy=prepare_phase(None,task,scene,q,0,{})
        changed=prepare_phase(None,task,scene,q,0,options({'corner_tangent_sign':1},'0'))
        explicit=prepare_phase(None,task,scene,q,0,options({'yaw_deg':225.},'0'))
    assert legacy['edges'][1]==changed['edges'][1]
    assert legacy['edges'][0]==explicit['edges'][0]
    assert 'yaw_definition' not in legacy['edges'][0]['contact_plan']
    assert 'pinch_direction_basis' not in legacy['edges'][0]
    assert changed['edges'][0]['contact_plan']['called_yaw_deg']==315.


@pytest.mark.parametrize('sign',[True,False,0,2,-2,1.,None,'1',float('nan')])
def test_tangent_sign_is_explicit_integer_and_checked_before_geometry(sign):
    with pytest.raises(ValueError,match='corner_tangent_sign'):
        prepare_phase(None,{},None,None,0,options({'corner_tangent_sign':sign},'9'),only_sides=['left'])


@pytest.mark.parametrize('conflict',[{'yaw_deg':315.},{'yaw_offset_deg':-15.},{'yaw_deg':315.,'yaw_offset_deg':0.}])
def test_tangent_is_exclusive_with_existing_yaw_modes(conflict):
    with pytest.raises(ValueError,match='exclusive'):
        validate_hand_options(options({'corner_tangent_sign':1,**conflict}),towel=True)


def test_tangent_is_towel_only_and_does_not_expand_legacy_relative_range():
    with pytest.raises(ValueError,match='Unknown hand options'):
        validate_hand_options(options({'corner_tangent_sign':1}))
    task,scene,rest=fixture();q=rest+[.8,-.55,.902]
    with pytest.raises(ValueError,match='within30'):
        prepare_phase(None,task,scene,q,1,options({'yaw_offset_deg':90.}))
