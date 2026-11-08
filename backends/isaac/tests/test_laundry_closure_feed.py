import copy
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.validation.laundry_closure_feed import (
    edge_feed_plan,closure_displacement,closure_wrists,preflight_edge_feed,observed_garment_offset,ClosureFeedProgress)
from scripts.validation.laundry_engineering_motion import minimum_jerk


def candidate():
    pick={'inward_world_xy':[-1.,0.],
        'pinch_direction_basis':{'seed_vertex_id':60,'neighbour_vertex_ids':[58,182]}}
    settings={'pinch_inward_offset_m':-.003,'closure_inward_travel_m':.009,
              'open_distance_m':.025,'close_distance_m':.010}
    return pick,settings


def plan():
    p,s=candidate()
    return {'closure_feed':edge_feed_plan(p,s),'closure_lift_m':0.,'capture_rise_m':0.}


def test_absent_option_keeps_original_lift_trajectory_exact():
    assert edge_feed_plan({}, {}) is None
    for start in (0.,.5):
        p={'closure_lift_m':.003,'closure_lift_start_fraction':start}
        for u in np.linspace(0.,1.,101):
            f=u if start==0 else minimum_jerk(float(np.clip((u-start)/(1-start),0.,1.)),0.,1.)
            np.testing.assert_array_equal(closure_displacement(p,u),[0.,0.,.003*f])


def test_feed_moves_only_selected_original_wrist_horizontally_without_mutating_plan():
    p=plan();before=copy.deepcopy(p);right=np.eye(4);right[:3,3]=[1.1,-.75,.98]
    left=np.eye(4);left[:3,3]=[1.1,-.35,.98];base={'right':right,'left':left}
    edges=[{'side':'right','contact_plan':p}]
    for u in (0.,.5,1.):
        wrists=closure_wrists(base,edges,u)
        np.testing.assert_array_equal(wrists['left'],left)
        np.testing.assert_array_equal(wrists['right'][:3,:3],right[:3,:3])
        np.testing.assert_allclose(wrists['right'][:3,3],right[:3,3]+[-.009*u,0.,0.])
        assert wrists['right'][2,3]==right[2,3]
    assert p==before and right[0,3]==1.1


def test_preflight_tracks_same101_closure_points_without_advancing_state():
    edges=[{'side':'right','contact_plan':plan()}];base={'right':np.eye(4)}
    seed={'fr3_right_joint1':.2,'right_index_DIP':.8,'left_index_DIP':.9};calls=[]
    def solve(wrists,u,warm):
        calls.append((wrists['right'].copy(),u,dict(warm)))
        return {**warm,'fr3_right_joint1':u,'right_index_DIP':.8-u*.1}
    rows=preflight_edge_feed(base,edges,seed,solve)
    assert len(rows)==len(calls)==101
    assert calls[0][1]==0. and calls[-1][1]==1.
    for i,(matrix,u,warm) in enumerate(calls):
        np.testing.assert_allclose(matrix[:3,3],[-.009*u,0.,0.])
        assert warm['left_index_DIP']==.9
        assert warm==(seed if i==0 else rows[i-1]['joint_positions_rad'])
    assert seed['right_index_DIP']==.8 and base['right'][0,3]==0.


def test_actual_contact_stop_holds_last_sent_feed_and_does_not_restart_during_hold():
    p=plan();edges=[{'side':'right','contact_plan':p},{'side':'left','contact_plan':p}]
    assists=[SimpleNamespace(side='right',closure_stop=SimpleNamespace(latched=False)),
             SimpleNamespace(side='left',closure_stop=None)]
    progress=ClosureFeedProgress();base={'right':np.eye(4),'left':np.eye(4)}
    first=progress.fractions(edges,assists,.31,6.5)
    np.testing.assert_allclose(closure_wrists(base,edges,.31,feed_fractions=first)['right'][:3,3],[-.009*.31,0,0])
    assists[0].closure_stop.latched=True
    for t,u in ((6.52,.32),(7.,.6),(8.,1.),(8.5,1.)):
        fractions=progress.fractions(edges,assists,u,t)
        wrists=closure_wrists(base,edges,u,feed_fractions=fractions)
        assert fractions=={'right':.31,'left':u}
        np.testing.assert_allclose(wrists['right'][:3,3],[-.009*.31,0,0])
        np.testing.assert_allclose(wrists['left'][:3,3],[-.009*u,0,0])
    assert len(progress.events)==1 and progress.events[0]['time_s']==6.52
    assert progress.events[0]['last_submitted_feed_fraction']==.31
    # No stop retains the complete coupled trajectory; a new batch has no stale stop.
    fresh=ClosureFeedProgress();assists[0].closure_stop=None
    assert fresh.fractions(edges,assists,1.,9.)=={'right':1.,'left':1.}
    assert not fresh.events


@pytest.mark.parametrize('patch',[
    {'closure_inward_travel_m':True},{'closure_inward_travel_m':.013},
    {'closure_inward_travel_m':float('nan')},{'pinch_inward_offset_m':.001},
    {'pinch_inward_offset_m':-.01,'closure_inward_travel_m':.003},
    {'open_distance_m':.045},{'close_distance_m':.025},
    {'closure_lift_m':.003},{'capture_rise_m':.002}])
def test_unreviewed_entry_parameters_rejected(patch):
    p,s=candidate();s.update(patch)
    with pytest.raises(ValueError):edge_feed_plan(p,s)


def test_missing_actual_material_direction_and_failed_ik_cannot_pass():
    p,s=candidate();p.pop('pinch_direction_basis')
    with pytest.raises(ValueError):edge_feed_plan(p,s)
    edges=[{'side':'right','contact_plan':plan()}]
    with pytest.raises(ValueError,match='inventory'):
        preflight_edge_feed({'right':np.eye(4)},edges,{'joint':0.},lambda *args:{'invented':0.})
    def failed(*args):raise ValueError('unreachable')
    with pytest.raises(ValueError,match='unreachable'):
        preflight_edge_feed({'right':np.eye(4)},edges,{'joint':0.},failed)


def test_garment_actual_tangent_uses_one_original_panel_and_preserves_local_region():
    x,y=np.meshgrid(np.arange(4)*.01,np.arange(4)*.01)
    panel=np.column_stack([x.ravel(),y.ravel(),np.full(x.size,.0015)])
    rest=np.concatenate([panel,panel-[0.,0.,.003]])
    q=rest.copy()
    for ids,angle in ((slice(0,16),.4),(slice(16,None),-.8)):
        R=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
        q[ids,:2]=rest[ids,:2]@R.T+[.8,-.5]
    q[:,2]+=.902
    pick={'seed_vertex_ids':[0,16],'natural_corner_vertex_id':0,
          'edge_world_m':q[0].tolist(),'allowed_material_vertex_ids':[0,1,4,5,16,17,20,21]}
    before=q.copy();p=observed_garment_offset({'rest_vertices_m':rest.tolist()},q,pick,[1.,1.],-.003)
    theta=.4+np.pi/4
    np.testing.assert_allclose(p['inward_world_xy'],[np.cos(theta),np.sin(theta)])
    np.testing.assert_allclose(p['pinch_tcp_xy_m'],q[0,:2]-.003*np.array([np.cos(theta),np.sin(theta)]))
    assert p['allowed_material_vertex_ids']==pick['allowed_material_vertex_ids']
    assert max(p['pinch_direction_basis']['neighbour_vertex_ids'])<16
    np.testing.assert_array_equal(q,before)
    bad=q.copy();bad[1,2]+=.04
    with pytest.raises(ValueError,match='too folded'):
        observed_garment_offset({'rest_vertices_m':rest.tolist()},bad,pick,[1.,1.],-.003)


def test_local_curved_edge_profile_keeps_old_fit_failure_and_all_material_identity():
    x,y=np.meshgrid(np.arange(4)*.01,np.arange(4)*.01)
    panel=np.column_stack([x.ravel(),y.ravel(),np.full(x.size,.0015)])
    rest=np.concatenate([panel,panel-[0.,0.,.003]])
    q=rest+[.8,-.5,.902]
    radius=np.linalg.norm(rest[:16,:2],axis=1)
    q[np.flatnonzero(radius>.020+1.e-9),2]+=.025
    # A remote facing panel must not influence either local surface estimate.
    q[16:,:2]=rest[16:,:2]@np.array([[0.,-1.],[1.,0.]])+[.8,-.5]
    pick={'seed_vertex_ids':[0,16],'natural_corner_vertex_id':0,
          'edge_world_m':q[0].tolist(),'allowed_material_vertex_ids':[0,1,4,5,16,17,20,21]}
    before=q.copy();spec={'rest_vertices_m':rest.tolist()}
    with pytest.raises(ValueError,match='too folded'):
        observed_garment_offset(spec,q,pick,[1.,1.],-.003)
    result=observed_garment_offset(spec,q,pick,[1.,1.],-.003,profile='local_panel_20mm_v2')
    basis=result['pinch_direction_basis']
    assert basis['maximum_material_radius_m']==.020
    assert not basis['original_30mm_fit']['planning_guard_passed']
    assert basis['original_30mm_fit']['fit_rms_m']>.003
    assert basis['fit_rms_m']<1.e-12 and max(basis['neighbour_vertex_ids'])<16
    np.testing.assert_allclose(result['inward_world_xy'],np.ones(2)/np.sqrt(2))
    assert result['allowed_material_vertex_ids']==pick['allowed_material_vertex_ids']
    np.testing.assert_array_equal(q,before)
    # Genuine curvature within the selected patch still fails the same fit guard.
    bad=q.copy();bad[1,2]+=.04
    with pytest.raises(ValueError,match='too folded'):
        observed_garment_offset(spec,bad,pick,[1.,1.],-.003,profile='local_panel_20mm_v2')


def test_default_tangent_receipt_and_explicit_old_profile_are_identical():
    x,y=np.meshgrid(np.arange(4)*.01,np.arange(4)*.01)
    rest=np.column_stack([x.ravel(),y.ravel(),np.full(x.size,.0015)])
    q=rest+[.8,-.5,.902]
    pick={'seed_vertex_ids':[0],'natural_corner_vertex_id':0,'edge_world_m':q[0].tolist()}
    args=({'rest_vertices_m':rest.tolist()},q,pick,[1.,1.],-.003)
    original=observed_garment_offset(*args)
    assert observed_garment_offset(*args,profile='single_panel_30mm_v1')==original
    assert 'profile' not in original['pinch_direction_basis']
    assert 'original_30mm_fit' not in original['pinch_direction_basis']
    assert original['pinch_direction_basis']['maximum_material_radius_m']==.030


def test_small_patch_cannot_invent_a_tangent_from_sparse_or_vertical_material():
    rest=np.array([[0,0,.0015],[.01,0,.0015],[.02,0,.0015],[.03,0,.0015]],float)
    pick={'seed_vertex_ids':[0],'natural_corner_vertex_id':0,'edge_world_m':[.8,-.5,.9035]}
    with pytest.raises(ValueError,match='resolved local'):
        observed_garment_offset({'rest_vertices_m':rest.tolist()},rest+[.8,-.5,.902],
            pick,[1.,1.],-.003,profile='local_panel_20mm_v2')
    rest=np.array([[0,0,.0015],[.01,0,.0015],[0,.01,.0015],[.01,.01,.0015]],float)
    q=np.column_stack([np.full(4,.8),np.full(4,-.5),.9035+rest[:,0]+rest[:,1]])
    with pytest.raises(ValueError,match='too folded'):
        observed_garment_offset({'rest_vertices_m':rest.tolist()},q,pick,[1.,1.],-.003,
            profile='local_panel_20mm_v2')
