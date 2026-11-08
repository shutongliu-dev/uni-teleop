import numpy as np
from scipy.spatial.transform import Rotation
from scripts.validation.review_sponge_native import determinant, raw_metrics, contact_metrics


def fixture():
    rest = np.array([[0.,0.,0.],[.01,0.,.045],[0.,.01,.045],[0.,0.,.045]])
    # Positive orientation with three central top nodes and one bottom bond.
    tet = np.array([[0,1,2,3]])
    foam={'rest_points_m':rest.tolist(),'tetrahedra':tet.tolist(),
        'surface_triangles':[[0,1,2],[0,1,3],[0,2,3],[1,2,3]],
        'attachment':{'node_indices':[0],'rigid_local_positions_m':[[0.,0.,0.]]}}
    return rest,foam


def test_raw_review_removes_only_pad_rigid_motion():
    rest,foam=fixture()
    r=Rotation.from_euler('xyz',[.1,.2,.3]); quat=r.as_quat()
    pad=np.array([[.4,.5,.9,*quat],[.4,.5,.9,*quat]])
    q=np.tile(r.apply(rest)+pad[0,:3],(2,1,1))
    m,local,bad=raw_metrics(q,np.zeros_like(q),pad,None,foam,np.array([[0,0,0.]]))
    assert bad is None
    assert np.allclose(local,rest,atol=1e-8)
    assert np.allclose(m['min_j'],1)
    assert np.allclose(m['central_height_m'],.045)
    assert m['bond_error_m'].max()<1e-12


def test_raw_review_catches_inversion_and_true_box_penetration():
    rest,foam=fixture()
    pad=np.array([[0.,0.,0.,0.,0.,0.,1.],[0.,0.,0.,0.,0.,0.,1.]])
    q=np.tile(rest,(2,1,1));q[1,[1,2]]=q[1,[2,1]]
    platen=np.array([[0.,0.,.045,0.,0.,0.,1.],[0.,0.,.045,0.,0.,0.,1.]])
    m,_,bad=raw_metrics(q,np.zeros_like(q),pad,platen,foam,np.array([[0,0,0.]]))
    assert bad['frame']==1 and bad['tetrahedron_indices']==[0]
    assert m['min_j'][1]==-1
    assert np.allclose(m['platen_surface_node_penetration_m'],.004)
    assert np.all(m['platen_bbox_gap_m']==0)


def test_contacts_keep_signed_vectors_not_scalar_magnitudes():
    rows=[{'contacts':[{'point_count':2,'raw_scalar_normal_impulses_ns':[.1,.2],
        'raw_normals':[[0,0,1],[0,0,-1]],'raw_separations_m':[0.,-.001],
        'impulse_vector_ns':[0.,0.,-.1],'impulse_magnitude_sum_ns':.3,
        'actor0':'pad','actor1':'table'}]}]
    f,report=contact_metrics(rows,.01)
    assert np.allclose(f,[[0,0,-10]])
    assert report['point_samples']==2
    assert report['max_online_impulse_aggregate_error_ns']<1e-15
