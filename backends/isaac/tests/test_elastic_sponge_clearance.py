import numpy as np
from scipy.spatial.transform import Rotation
from scripts.validation.elastic_sponge_clearance import certify_motion,direction_hint,projection_gap


def box(half):
    return np.array([[x,y,z] for x in (-half[0],half[0]) for y in (-half[1],half[1]) for z in (-half[2],half[2])])


def test_rotated_real_shapes_pass_despite_overlapping_aabb():
    shape=Rotation.from_euler('z',45,degrees=True).apply(box([.01,.001,.001]))
    normal=np.array([-1.,1.,0.])/np.sqrt(2);other=shape+.0071*normal
    # World-axis boxes overlap; the actual parallel surfaces are5.1mm apart.
    assert np.all(np.minimum(shape.max(0),other.max(0))>=np.maximum(shape.min(0),other.min(0)))
    result,certs=certify_motion([other[None]],[shape])
    assert result['passed'] and result['global_certified_lower_bound_m']>.00509
    assert len(certs['certificates'])==1


def test_original_5mm_gate_and_roundoff_guard_are_never_relaxed():
    cube=box([.001,.001,.001])
    for gap in (.0049,.0050005):
        result,_=certify_motion([(cube+[gap+.002,0,0])[None]],[cube])
        assert not result['passed']
    result,_=certify_motion([np.stack([cube+[.01,0,0],cube+[.003,0,0]])],[cube])
    assert not result['passed'] and result['failed_certificates'][0]['frame']==1


def test_numerical_direction_is_accepted_only_after_support_extrema_verify_it():
    cube=box([.001,.001,.001]);moving=cube+[.006,.006,0]
    direction=direction_hint(moving,cube);gap,normal=projection_gap(moving,cube,direction)
    assert abs(np.linalg.norm(normal)-1)<1e-12
    assert .00565<gap<.00566
