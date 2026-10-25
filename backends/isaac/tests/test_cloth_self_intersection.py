import numpy as np
import pytest

from bench.evaluation.cloth.audit_cloth_self_intersection import self_intersections


@pytest.mark.parametrize('name,vertices,faces,expected',[
    ('neighbor_edge',[[0,0,0],[1,0,0],[0,1,0],[1,1,0]],[[0,1,2],[1,3,2]],False),
    ('neighbor_fold',[[0,0,0],[1,0,0],[0,1,0],[0,0,1]],[[0,1,2],[0,1,3]],False),
    ('neighbor_overlap',[[0,0,0],[1,0,0],[0,1,0],[.2,.2,0]],[[0,1,2],[0,1,3]],True),
    ('shared_vertex_crossing',[[0,0,0],[1,0,0],[0,1,0],[.3,.2,-1],[.3,.2,1]],[[0,1,2],[0,3,4]],True),
    ('shared_vertex_touch',[[0,0,0],[1,0,0],[0,1,0],[-1,0,0],[0,-1,0]],[[0,1,2],[0,3,4]],False),
    ('coplanar_inside',[[0,0,0],[1,0,0],[0,1,0],[.2,.2,0],[.5,.2,0],[.2,.5,0]],[[0,1,2],[3,4,5]],True),
    ('parallel_separate',[[0,0,0],[1,0,0],[0,1,0],[.2,.2,.00001],[.5,.2,.00001],[.2,.5,.00001]],[[0,1,2],[3,4,5]],False),
    ('small_nearly_parallel_crossing',[[0,0,0],[.01,0,0],[0,.01,0],[.002,.002,-1e-8],[.005,.002,1e-8],[.002,.005,1e-8]],[[0,1,2],[3,4,5]],True),
])
def test_structural_contact_and_real_intersection(name,vertices,faces,expected):
    pairs,_=self_intersections(np.array(vertices,dtype=float),np.array(faces))
    assert bool(pairs)==expected,name
