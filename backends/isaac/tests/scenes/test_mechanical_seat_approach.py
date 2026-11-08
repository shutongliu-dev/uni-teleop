"""A seat that declares its mating point in the actor frame is approached along its seated axis."""
import numpy as np
import pytest

from scenes.workrooms.mechanical_assembly import placement_authoring as authoring

# Audio plug seat: local +Z of the plug points to world +Y; pin 86.4 mm ahead of the tail origin.
CABLE_SEAT = {'actor': 'cable_end', 'target_position_m': [.40, .309, .8065],
              'target_quaternion_wxyz': [.7071067811865475, -.7071067811865475, 0., 0.],
              'orientation_axis_local': [0., 0., 1.], 'orientation_axis_only': True,
              'actor_local_point_m': [0., 0., .0864]}


def test_declared_mating_point_sets_the_free_approach_side():
    assert authoring._seat_approach_world(CABLE_SEAT) == pytest.approx([0., -1., 0.], abs=1e-12)
    # Seats without a declared mating point keep the fixture's up direction.
    assert authoring._seat_approach_world({k: v for k, v in CABLE_SEAT.items() if k != 'actor_local_point_m'}) is None
    with pytest.raises(ValueError):
        authoring._seat_approach_world({**CABLE_SEAT, 'actor_local_point_m': [0., 0., 0.]})


def test_goal_direction_is_expressed_in_the_receptacle_frame():
    socket = ([.40, .309, .8065], [1., 0., 0., 0.])
    goal = authoring._goal('cable_end_placement', 'cable_end', '/World/NistTask/Fixtures/audio_socket',
                           CABLE_SEAT['target_position_m'], *socket,
                           direction_world=authoring._seat_approach_world(CABLE_SEAT))
    assert goal['approach_direction_local'] == pytest.approx([0., -1., 0.], abs=1e-12)
    assert goal['local_position_m'] == pytest.approx([0., 0., 0.])
    # A yawed receptacle gets the same world direction in its own frame.
    yaw = [np.cos(np.pi/4), 0., 0., np.sin(np.pi/4)]
    rotated = authoring._goal('g', 'cable_end', '/x', CABLE_SEAT['target_position_m'], socket[0], yaw,
                              direction_world=[0., -1., 0.])
    assert authoring._rotation(yaw)@np.asarray(rotated['approach_direction_local']) == pytest.approx([0., -1., 0.])
    # Without a declared direction the legacy fixture-up result is bit-identical.
    legacy = authoring._goal('g', 'b', '/x', [.1, .2, .3], [0., 0., 0.], yaw)
    up = authoring._rotation(yaw).T@np.array([0., 0., 1.])
    assert legacy['approach_direction_local'] == up.tolist()
