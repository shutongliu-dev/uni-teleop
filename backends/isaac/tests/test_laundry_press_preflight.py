"""Analytic geometry and fail-closed reference/static-stage coverage checks."""
from types import SimpleNamespace
from copy import deepcopy
from itertools import product

import numpy as np
import pytest
from scipy.spatial import ConvexHull

from scripts.validation import laundry_press_preflight as p


def cube(center=(0., 0., 0.), half=.01):
    vertices=np.array(list(product((-half, half), repeat=3)))+center
    return {'vertices':vertices,'planes':ConvexHull(vertices).equations,
            'bounds':np.array([vertices.min(0),vertices.max(0)])}


def test_original_convex_intersection_matches_analytic_overlap():
    a,b=cube(),cube((.015,0,0))
    radius,point,receipt=p.convex_overlap(a,b)
    assert radius==pytest.approx(.0025,abs=1e-10)
    assert receipt is None
    assert np.max(np.vstack([a['planes'],b['planes']])@np.r_[point,1]+radius)<1e-9
    assert p.convex_overlap(a,cube((.03,0,0)))[0]==0


def solver_error(*args,**kwargs):raise RuntimeError('original numeric failure')


def test_numeric_exception_requires_explicit_all_vertex_face_separation(monkeypatch):
    monkeypatch.setattr(p,'original_intersection_radius',solver_error)
    a,b=cube(),cube((.03,0,0))
    radius,point,receipt=p.convex_overlap(a,b)
    assert radius==0 and point is None
    assert receipt['resolution']=='original_face_separation_certificate'
    assert receipt['minimum_all_vertex_separation_m']==pytest.approx(.01)
    plane=np.array(receipt['world_plane'])
    vertices=a['vertices'] if receipt['direction'].startswith('first') else b['vertices']
    assert np.min(vertices@plane[:3]+plane[3])>0


def test_same_geometry_translated_lp_and_unresolved_failure(monkeypatch):
    monkeypatch.setattr(p,'original_intersection_radius',solver_error)
    a,b=cube((1e3,-2e3,3e3)),cube((1e3+.015,-2e3,3e3))
    radius,_,receipt=p.convex_overlap(a,b)
    assert radius==pytest.approx(.0025,abs=1e-9)
    assert receipt['resolution']=='same_planes_translated_interior_point_lp'
    monkeypatch.setattr(p,'linprog',lambda *a,**k:SimpleNamespace(status=4,message='unresolved',success=False))
    with pytest.raises(RuntimeError,match='Unresolved'):p.convex_overlap(a,b)


def test_translated_solution_is_checked_against_world_planes(monkeypatch):
    monkeypatch.setattr(p,'original_intersection_radius',solver_error)
    monkeypatch.setattr(p,'linprog',lambda *a,**k:SimpleNamespace(status=0,message='bad',success=True,x=np.array([0.,0.,0.,1.])))
    with pytest.raises(RuntimeError,match='world-plane residual'):p.convex_overlap(cube(),cube())


def hit(radius=.0005,pair=('left_thumb_MC','left_hand_C_MC')):
    return {'bodies':list(pair),'radius_m':radius}


def refs():return {'recorded_native_start':{'intersections':[hit(.0004)]},
                   'original_release_initial':{'intersections':[hit(.0007)]}}


def test_design_bounds_use_both_independent_states_but_never_candidate_maximum():
    r=refs();before=deepcopy(r);bounds=p.design_contact_bounds(r)
    assert p.bounded_design_contacts([hit(.0007+5e-10)],bounds)
    assert not p.bounded_design_contacts([hit(.0007+2e-9)],bounds)
    assert not p.bounded_design_contacts([hit(.00099)],bounds)
    assert not p.bounded_design_contacts([hit(.000001,('left_index_DP','left_hand_C_MC'))],bounds)
    assert r==before


@pytest.mark.parametrize('change',['missing','newpair','1mm','nan'])
def test_invalid_original_reference_cannot_legalize_path(change):
    r=refs()
    if change=='missing':r.pop('original_release_initial')
    elif change=='newpair':r['recorded_native_start']['intersections']=[hit(.0001,('left_index_DP','left_hand_C_MC'))]
    else:r['recorded_native_start']['intersections']=[hit(.001 if change=='1mm' else float('nan'))]
    with pytest.raises(ValueError):p.design_contact_bounds(r)


def minimal_checker(monkeypatch):
    c=p._PressPreflight.__new__(p._PressPreflight)
    c.models={};c.bounds={};c.references=refs();c.binding={};c.adjacent=set()
    c.body_names=['body'];c.q=np.array([[0.,0,0],[1,0,0],[0,1,0]])
    c.triangles=np.array([[0,1,2]]);c.export={}
    c._joints=lambda j:j
    c._world=lambda j:({}, {'body':np.eye(4)})
    c._environment=lambda j,w:({'passed':True},{'between_robots':{'clear':True}})
    c.arm=SimpleNamespace(check=lambda j:{'clear':True})
    c._internal=lambda w:([],0,[])
    called=[]
    def audit(export,states,tri,bodies,**kw):
        called.append(states)
        return {'export_verified':True,'unresolved_colliders':[],
                'violations':[], 'representation_boundary_uncertainties':[{'frame':1,'triangle_ids':[0]}]}
    monkeypatch.setattr(p,'audit_cloth_robot',audit)
    return c,called


def samples():
    return [{'label':str(i),'time_s':i*.02,'joint_positions_rad':{},
             'intentional_cloth_compression_phase':i==1}for i in range(3)]


def test_all_noncontact_samples_indexed_and_outer_boundary_uncertainty_rejected(monkeypatch):
    c,called=minimal_checker(monkeypatch);rows=samples();before=deepcopy(rows)
    r=c(rows)
    assert len(called)==1 and called[0]['particle_q'].shape==(2,3,3)
    assert r['sample_count']==3 and [x['sample_index']for x in r['samples']]==[0,1,2]
    assert r['static_cloth_sample_indices']==[0,2]
    assert [x['passed']for x in r['samples']]==[True,True,False]
    assert not r['all_clear'] and not r['samples'][1]['static_observed_cloth']['checked']
    assert rows==before


def test_explicit_compression_marker_and_finite_chronological_clock_required(monkeypatch):
    c,_=minimal_checker(monkeypatch)
    for key,value in [('intentional_cloth_compression_phase',1),('time_s',float('nan')),('time_s',-.01)]:
        rows=samples();rows[1][key]=value
        with pytest.raises(ValueError):c(rows)


def test_unchecked_export_rejects_every_noncontact_frame(monkeypatch):
    c,_=minimal_checker(monkeypatch)
    monkeypatch.setattr(p,'audit_cloth_robot',lambda *a,**kw:{'export_verified':False,'unresolved_colliders':['missing'],
                        'violations':[],'representation_boundary_uncertainties':[]})
    result=c(samples())
    assert not result['samples'][0]['passed'] and not result['samples'][2]['passed']
