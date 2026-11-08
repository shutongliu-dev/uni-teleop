"""A second flap regrasp must never be mistaken for the alignment operation."""
from copy import deepcopy
from unittest.mock import patch

import numpy as np
import pytest

from scripts.validation.laundry_captured_towel_tasks import (
    SEQUENTIAL_FLATTEN, captured_arc, execution_phase_edges, prepare_phase,
    task_phase_index, validate_options,
)


def fixture():
    x,y=np.meshgrid(np.linspace(-.3,.3,61),np.linspace(-.2,.2,41))
    rest=np.column_stack([x.ravel(),y.ravel(),np.zeros(x.size)])
    task={'id':'towel_flattening','asset_metadata':{
        'physical_thickness_m':.004,'table_top_z_m':.9,
        'reset_profile':'towel_partial_edge_fold_v1','target_center_xy_m':[.65,-.55],
        'cloths':[{'id':'towel','rest_vertices_m':rest.tolist()}]}}
    theta=.18
    rotation=np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
    q=rest.copy();q[:,:2]=q[:,:2]@rotation.T+[.56,-.53];q[:,2]=.902
    options={'flatten_phase_sequence':SEQUENTIAL_FLATTEN,'phases':{
        '0':{'edge_indices':[1],'capture_order':['left']},
        '1':{'edge_indices':[0],'capture_order':['right']},'2':{}}}
    return task,rest,q,rotation,options


def test_second_regrasp_is_fresh_original_negative_edge_and_third_is_alignment():
    task,rest,q,rotation,options=fixture();scene={'cloths':[{'id':'towel'}]}
    original=q.copy();q[0,:2]+=[.003,-.004]
    with patch('scripts.validation.laundry_captured_towel_tasks.pinch_plan',return_value={}):
        phases=[prepare_phase(None,task,scene,q,i,options) for i in range(3)]
    assert [p['task_phase_index'] for p in phases]==[0,0,1]
    assert [[e['seed_vertex_ids'][0] for e in p['edges']] for p in phases]==[[2440],[0],[60,2500]]
    np.testing.assert_array_equal(phases[1]['edges'][0]['edge_world_m'],q[0])
    ids=[0,1,61,62]
    arc=captured_arc(task,phases[1],q,ids)
    np.testing.assert_allclose(arc['controls_m'][-1][:2],rest[ids,:2].mean(0)@rotation.T+[.56,-.53])
    assert not arc['free_cloth_state_written'] and not arc['target_shape_written']
    with pytest.raises(ValueError,match='positive-X'):
        captured_arc(task,phases[2],q,ids)
    alignment_ids=[59,60,120,121]
    aligned=captured_arc(task,phases[2],q,alignment_ids)
    np.testing.assert_allclose(aligned['controls_m'][-1][:2],rest[alignment_ids,:2].mean(0)+[.65,-.55])
    np.testing.assert_array_equal(q[1:],original[1:])
    np.testing.assert_array_equal(q[0],original[0]+[.003,-.004,0])


def test_old_phase_count_and_semantic_indices_are_retained():
    task,_,q,_,_=fixture();scene={'cloths':[{'id':'towel'}]}
    assert len(execution_phase_edges(task,{}))==2
    assert [task_phase_index(task,{},i) for i in (0,1)]==[0,1]
    with patch('scripts.validation.laundry_captured_towel_tasks.pinch_plan',return_value={}):
        legacy=prepare_phase(None,task,scene,q,1,{})
    assert [e['seed_vertex_ids'] for e in legacy['edges']]==[[60],[2500]]
    assert 'task_phase_index' not in legacy
    with pytest.raises(ValueError): task_phase_index(task,{},2)


@pytest.mark.parametrize('change',[
    lambda t,o:t.update(id='towel_folding'),
    lambda t,o:o.update(flatten_phase_sequence='unknown'),
    lambda t,o:o.update(transport_mode='capture_lift_probe'),
    lambda t,o:o['phases'].pop('2'),
    lambda t,o:o['phases'].update({'3':{}}),
    lambda t,o:o['phases']['1'].update(edge_indices=[1]),
    lambda t,o:o['phases']['0'].update(edge_indices=[True]),
    lambda t,o:o['phases']['2'].update(edge_indices=[0]),
    lambda t,o:o['phases']['1'].update(capture_order=['left']),
])
def test_incomplete_wrong_corner_or_unsupported_sequence_rejected(change):
    task,_,_,_,options=fixture();change(task,options)
    with pytest.raises(ValueError):validate_options(task,options)


def test_rebind_cannot_add_an_unselected_hand_or_mutate_options():
    task,_,q,_,options=fixture();before=deepcopy(options)
    with pytest.raises(ValueError,match='unselected'):
        prepare_phase(None,task,{'cloths':[{'id':'towel'}]},q,1,options,only_sides=['left'])
    assert options==before
