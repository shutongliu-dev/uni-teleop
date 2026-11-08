import numpy as np
from scripts.validation.elastic_sponge_wiping import force,target,STEPS


def test_original_motion_law_and_force_bound_include_real_release():
    meta={'goals':{'soil_cells':[{'center_world_m':[.6,-.8,.9],'patch':'soil_1'}]}}
    for step in (0,800,801,3000,5600,5601,7200,7201,12000,12001,STEPS):
        f,phase,desired,speed=force(step,meta,[0.,0.,1.,0,0,0,1],[0.,0.,0.,0,0,0])
        assert np.max(abs(f))<=float(np.float32(.4)) and f[2]==0.
        if phase in ('settle','release'):assert np.all(f==0)
    assert np.allclose(target(0,meta)[1],[.52,-.819])
    assert np.allclose(target(5600,meta)[1],[.68,-.819])
    assert np.allclose(target(7200,meta)[1],[.68,-.781])
    assert np.allclose(target(STEPS,meta)[1],[.52,-.781])
