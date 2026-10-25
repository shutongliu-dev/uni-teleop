"""Physical geometry boundary cases; these do not run a simulator."""
from copy import deepcopy
import math

import pytest

from scenes.workrooms.living_room.tower_geometry import block_ids, tower_slots, validate_geometry

# Synthetic test dimensions only, not the selected task asset dimensions.
GEOMETRY = {'block_size_m': [.12, .035, .02]}


def test_dimensions_are_mandatory_and_original_table_is_preserved():
    with pytest.raises(ValueError, match='required'):
        validate_geometry({})
    with pytest.raises(ValueError, match='parent table'):
        validate_geometry({**GEOMETRY, 'table_top_z_m': .9})
    with pytest.raises(ValueError, match='vertical'):
        validate_geometry({**GEOMETRY, 'layer_gap_m': .001})


def test_cross_layer_faces_touch_and_crossing_rows_fit_length():
    slots = tower_slots(GEOMETRY)
    assert len(slots) == len(block_ids(GEOMETRY)) == 18
    assert len({slot['slot_id'] for slot in slots}) == 18
    assert slots[0]['position_m'][2] - .01 == pytest.approx(.78)
    assert slots[3]['position_m'][2] - slots[0]['position_m'][2] == pytest.approx(.02)
    assert slots[3]['yaw_rad'] - slots[0]['yaw_rad'] == pytest.approx(math.pi/2)
    with pytest.raises(ValueError, match='crossing'):
        validate_geometry({'block_size_m': [.12, .04, .02]})


@pytest.mark.parametrize('override', [
    {'block_size_m': [.12, float('nan'), .02]}, {'block_size_m': [.12, .035, -.02]},
    {'layers': True}, {'layers': 2}, {'blocks_per_layer': 2}, {'in_layer_gap_m': -.01},
    {'center_xy_m': [.16, 0.]}, {'block_size_m': [1.2, .35, .1]},
])
def test_invalid_geometry_is_not_silently_repaired(override):
    with pytest.raises(ValueError):
        validate_geometry({**GEOMETRY, **override})


def test_geometry_returns_independent_canonical_data():
    original = deepcopy(GEOMETRY)
    result = validate_geometry(original)
    result['block_size_m'][0] = 10
    assert original == GEOMETRY
    assert validate_geometry(GEOMETRY)['center_xy_m'] == [.65, 0.]
