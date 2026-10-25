"""Room wall colliders must be the slab directly behind the drawn wall.

``room._box_collider`` maps a unit cube onto each wall's collider box. It used to
compose the frame rotation after the translation and use the frame's transpose,
which moved every rotated wall slab somewhere other than its wall -- e.g. the
laboratory's front-wall slab ended up 1.4 m inside the room at z 1.5-2.6 m, and
another one inside arm reach. Rooms that authored their own wall solids only
worked around it. These checks rebuild the collider box with the same numpy
matrix the USD prim receives (``box_collider_matrix``) and compare it with the
slab behind every drawn wall quad. CPU only; no OpenUSD needed.
"""
from __future__ import annotations

import numpy as np
import pytest

from scenes import rooms
from bench.simulation.isaacsim_backend.room import (WALL_FACES, _wall_geometry, _wall_quads, box_collider_matrix,
                                              validate_room)

UNIT_CUBE = np.array([[x, y, z, 1.] for x in (-1., 1.) for y in (-1., 1.) for z in (-1., 1.)])


def _collider_box(center, half, basis):
    corners = UNIT_CUBE @ box_collider_matrix(center, half, basis)
    return corners[:, :3].min(axis=0), corners[:, :3].max(axis=0)


def _wall_slabs(room):
    """(expected slab, collider box) for every drawn quad, built as add_room builds them."""
    by_face = {}
    for opening in room['openings']:
        by_face.setdefault(opening['wall'], []).append(opening)
    thickness = float(room['wall_thickness_m'])
    for face in WALL_FACES:
        wall = _wall_geometry(room, face)
        for lo_u, hi_u, lo_v, hi_v in _wall_quads(wall, by_face.get(face, [])):
            corners = np.array([wall['center'] + u * wall['u_axis'] + v * wall['v_axis']
                                for u, v in ((lo_u, lo_v), (hi_u, lo_v), (hi_u, hi_v), (lo_u, hi_v))])
            slab = np.vstack([corners, corners + wall['normal'] * thickness])
            center = corners.mean(axis=0) + wall['normal'] * (thickness / 2.)
            half = (0.5 * np.linalg.norm(corners[1] - corners[0]), 0.5 * np.linalg.norm(corners[3] - corners[0]),
                    thickness / 2.)
            yield face, (slab.min(axis=0), slab.max(axis=0)), _collider_box(
                center, half, (wall['u_axis'], wall['v_axis'], wall['normal']))


def test_the_collider_matrix_is_a_proper_placement():
    """Unit-cube corners land on the box: axes follow (u, v), translation is last."""
    u, v = np.array([0., -1., 0.]), np.array([0., 0., 1.])
    low, high = _collider_box([1., 2., 3.], [0.5, 0.25, 0.1], (u, v, np.array([1., 0., 0.])))
    np.testing.assert_allclose(low, [0.9, 1.5, 2.75], atol=1e-12)
    np.testing.assert_allclose(high, [1.1, 2.5, 3.25], atol=1e-12)
    rotation = box_collider_matrix([0., 0., 0.], [1., 1., 1.], (u, v, None))[:3, :3]
    assert np.isclose(np.linalg.det(rotation), 1.0), 'a left-handed wall frame must still give a rotation'


@pytest.mark.parametrize('family', rooms.FAMILIES)
def test_every_wall_collider_is_the_slab_behind_its_drawn_wall(family):
    room = validate_room(rooms.scene(family)['room'])
    slabs = list(_wall_slabs(room))
    assert slabs, family
    for face, (want_low, want_high), (got_low, got_high) in slabs:
        np.testing.assert_allclose(got_low, want_low, atol=1e-9, err_msg=f'{family} {face} wall')
        np.testing.assert_allclose(got_high, want_high, atol=1e-9, err_msg=f'{family} {face} wall')


def test_a_wall_opening_stays_open():
    """No wall slab may cover a door or window opening of any room."""
    for family in rooms.FAMILIES:
        room = validate_room(rooms.scene(family)['room'])
        for opening in room['openings']:
            wall = _wall_geometry(room, opening['wall'])
            (cu, cv), (su, sv) = opening['center_m'], opening['size_m']
            middle = wall['center'] + cu * wall['u_axis'] + cv * wall['v_axis'] + wall['normal'] * 0.06
            for face, _, (low, high) in _wall_slabs(room):
                inside = np.all(middle > low + 1e-9) and np.all(middle < high - 1e-9)
                assert not inside, f'{family}: the {face} wall collider fills opening {opening["id"]}'
