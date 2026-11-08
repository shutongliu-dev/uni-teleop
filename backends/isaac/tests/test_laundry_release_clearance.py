import numpy as np
import pytest

from scripts.validation.laundry_release_clearance import release_envelope


def test_lowest_open_finger_and_off_center_support_set_the_clearance():
    start=np.array([0.,0.,.92]);end=np.array([.2,0.,.93])
    hand=np.array([[0.,0.,.91],[.04,.03,.93],[-.04,-.03,.94]])
    # The tall support lies away from the capture point, inside the opening
    # sweep. A point-only or local percentile calculation would miss it.
    cloth=np.array([[.2,0.,.902],[.235,.025,.915],[.5,0.,1.1]])
    r=release_envelope(start,end,hand,cloth,table_z=.9,thickness=.002,model_error_m=.0005)
    assert r['support_node_count']==2
    assert r['target_support_surface_z_m']==pytest.approx(.916)
    assert r['predicted_minimum_opening_hand_surface_z_m']==pytest.approx(.9235)
    assert r['end_z_m']==pytest.approx(.9335)


def test_bare_table_and_small_hand_offset_allow_gentle_low_placement():
    r=release_envelope([0.,0.,.905],[.2,0.,.93],[[0.,0.,.901]],[[.5,0.,.91]],
        table_z=.9,thickness=.0016)
    assert r['support_node_count']==0
    assert r['end_z_m']==pytest.approx(.9102)
    assert not r['cloth_state_written']


def test_unresolved_kinematic_mismatch_cannot_claim_safe_lowering():
    with pytest.raises(ValueError):
        release_envelope([0.,0.,.905],[.2,0.,.93],[[0.,0.,.901]],[[.5,0.,.91]],
            table_z=.9,thickness=.0016,model_error_m=.003)
