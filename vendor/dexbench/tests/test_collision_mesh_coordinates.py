"""World-equivalent mesh coordinates and unchanged opt-out stage inputs."""
import numpy as np
import pytest
from bench.simulation.isaacsim_backend.asset_catalog import collision_mesh_coordinate_scale, condition_collision_mesh


def test_default_preserves_exact_points_and_no_new_scale_op():
    points=np.array([[.016,0.,.052],[0.,.024,.070]],dtype=np.float32)
    got,scale=condition_collision_mesh(points,collision_mesh_coordinate_scale({}))
    assert got is points and scale is None
    assert collision_mesh_coordinate_scale({'collision_mesh_coordinate_scale':1})==1.


def test_conditioning_preserves_all_parent_geometry_after_inverse_prim_scale():
    rng=np.random.default_rng(14)
    points=rng.uniform(-.1,.1,(64,3)).astype(np.float32);before=points.copy()
    conditioned,scale=condition_collision_mesh(points,1000)
    np.testing.assert_array_equal(points,before)
    # Match USD's float32 scale and a nonidentity authored part transform.
    rotation=np.array([[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]])
    position=np.array([.04,-.08,.02])
    actual=(conditioned.astype(float)*np.array(scale,dtype=np.float32))@rotation.T+position
    np.testing.assert_allclose(actual,points@rotation.T+position,atol=1e-8,rtol=0)


@pytest.mark.parametrize('factor',[0,-1,.5,1001,100000,True,None,float('inf'),float('nan'),'1000'])
def test_invalid_or_unqualified_coordinate_scales_fail_closed(factor):
    with pytest.raises(ValueError,match='coordinate scale'):
        collision_mesh_coordinate_scale({'collision_mesh_coordinate_scale':factor})


def test_invalid_active_vertices_are_rejected():
    with pytest.raises(ValueError,match='vertices'):
        condition_collision_mesh([[0.,float('nan'),0.]],1000)
