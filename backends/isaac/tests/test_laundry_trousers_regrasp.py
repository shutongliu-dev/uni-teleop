import copy
from unittest.mock import patch

import numpy as np
import pytest

from scripts.validation import laundry_captured_trousers_regrasp as adapter


def fixture():
    task={'id':'trousers_folding','asset_metadata':{'design':{
        'fold_sequence_axis_hinge_sign_radius_m':[[0,.75,-1,.007],[1,-.55,-1,.012]]}}}
    return task,{'cloths':[{'id':'towel'}]}


def test_execution_order_regrasps_same_longitudinal_crease_before_transverse_fold():
    task,_=fixture()
    assert [[e['feature'] for e in phase] for phase in adapter.phase_edges(task)]==[
        ['left_cuff'],['waistband'],['paired_cuffs']]
    assert [p[0] for p in adapter.OPERATIONS]==[0,0,1]


def test_actual_q_and_per_execution_hand_options_reach_original_reference_unchanged():
    task,scene=fixture();q=np.array([[.4,-.5,.91],[.42,-.49,.92]])
    options={'capture_profile':'strict_v1','phases':{
        '0':{'sides':['right'],'hands':{'right':{'yaw_deg':250.}}},
        '1':{'sides':['left'],'hands':{'left':{'pitch_deg':75.}}},
        '2':{'sides':['right']}}}
    before=copy.deepcopy((task,scene,options));q_before=q.copy();calls=[]
    def prepare(release,t,s,actual,reference,mapped,*,only_sides=None):
        assert actual is q
        calls.append((reference,mapped,only_sides))
        return {'phase_index':reference,'reference_fold':copy.deepcopy(
            t['asset_metadata']['design']['fold_sequence_axis_hinge_sign_radius_m'][reference])}
    with patch.object(adapter.garment,'prepare_phase',side_effect=prepare):
        results=[adapter.prepare_phase(None,task,scene,q,i,options,only_sides={'left'} if i==1 else None) for i in range(3)]
    assert [r['phase_index'] for r in results]==[0,1,2]
    assert [r['original_reference_phase_index'] for r in results]==[0,0,1]
    assert results[0]['reference_fold']==results[1]['reference_fold']
    for i,(reference,mapped,_) in enumerate(calls):
        cfg=mapped['phases'][str(reference)]
        assert cfg['edge_indices']==[adapter.OPERATIONS[i][1]]
        assert cfg['sides']==options['phases'][str(i)]['sides']
    assert calls[1][2]=={'left'}
    assert (task,scene,options)==before
    np.testing.assert_array_equal(q,q_before)


def test_wrong_scene_and_conflicting_boundary_fail_before_planning():
    task,scene=fixture()
    for invalid in ([],[{'id':'towel'},{'id':'base'}],[{'id':'shirt'}]):
        with pytest.raises(ValueError):adapter.validate_scene(task,{'cloths':invalid})
    for index in (-1,3,True):
        with pytest.raises(ValueError):adapter.prepare_phase(None,task,scene,np.zeros((1,3)),index,{})
    with pytest.raises(ValueError):adapter.prepare_phase(None,task,scene,np.zeros((1,3)),1,
        {'phases':{'1':{'edge_indices':[1]}}})


def test_waist_regrasp_uses_stationary_material_after_the_unpicked_waist_has_slid():
    task,_=fixture()
    rest=np.array([[-.19,.37,.003],[-.18,.37,.003],[-.19,.36,.003],
                   [.19,.37,.003],[.18,.37,.003],[.19,.36,.003]])
    q=rest+[.75,-.55,.9];q[:3,:2]+=[-.025,-.085];q[3:,:2]+=[-.01,.005]
    task['asset_metadata']['cloths']=[{'rest_vertices_m':rest.tolist()}]
    before=copy.deepcopy(task);q_before=q.copy()
    result=adapter.captured_arc(task,{'phase_index':1},q,[0,1,2])
    assert result['counterpart_vertex_ids']==[3,4,5]
    np.testing.assert_allclose(result['controls_m'][-1][:2],q[3:,:2].mean(0))
    assert result['controls_m'][-1][2]==pytest.approx(.928)
    assert result['counterpart_material_errors_m']==[0.,0.,0.]
    assert result['public_fold_goal_changed'] is False
    np.testing.assert_array_equal(q,q_before);assert task==before
    for ids in ([3,4],[0,0],[0,99]):
        with pytest.raises(ValueError):adapter.waist_alignment_arc(task,q,ids)


def test_other_operations_keep_original_fold_arc():
    for index in (0,2):
        phase={'phase_index':index};q=np.zeros((1,3))
        with patch.object(adapter.garment,'captured_arc',return_value={'original':True}) as delegated:
            assert adapter.captured_arc({},phase,q,[0])=={'original':True}
            delegated.assert_called_once_with({},phase,q,[0],release_height=.025,peak=.12)
