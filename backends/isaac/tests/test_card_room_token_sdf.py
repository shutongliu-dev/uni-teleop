"""Static union-surface checks; no cooked SDF/native collision qualification."""
from copy import deepcopy
import math

import numpy as np
import pytest

from bench.simulation.isaacsim_backend.asset_catalog import load_task_asset
from scripts.assets.prepare_card_room_token_sdf import (SOURCE, union_surface, geometry_review, validate_original)


@pytest.mark.parametrize('segments', [256, 512])
def test_single_union_boundary_preserves_sharp_shoulder_and_dimensions(segments):
    vertices,faces=union_surface(segments)
    report=geometry_review(vertices,faces,segments)
    assert report['connected_components']==1 and report['euler_number']==2
    assert report['internal_joining_disks']==0
    assert report['maximum_radial_chord_deficit_m'] < (2.2e-6 if segments==256 else .53e-6)
    # Independent midpoint distance on each circular ring attains the chord bound.
    ring=vertices[:segments,:2]
    midpoint=(ring+np.roll(ring,-1,axis=0))*.5
    assert np.allclose(.028-np.linalg.norm(midpoint,axis=1),
                       .028*(1-math.cos(math.pi/segments)),atol=1e-17,rtol=0)
    assert len(vertices)==4*segments+2 and len(faces)==8*segments


def test_reversed_triangle_is_rejected_as_invalid_solid():
    vertices,faces=union_surface()
    faces[0]=faces[0,::-1]
    with pytest.raises(ValueError,match='topology'):
        geometry_review(vertices,faces,512)


def test_original_parameters_are_only_read_and_changed_source_is_rejected():
    original=load_task_asset('token_black',SOURCE,verify_hashes=True)
    before=deepcopy(original)
    validate_original(original)
    assert original==before
    original['collision_parts'][1]['radius_m']+=.0001
    with pytest.raises(ValueError,match='contract changed'):
        validate_original(original)


def test_coarse_circumference_cannot_be_selected_as_high_fidelity_candidate():
    with pytest.raises(ValueError,match='high-resolution'):
        union_surface(30)
