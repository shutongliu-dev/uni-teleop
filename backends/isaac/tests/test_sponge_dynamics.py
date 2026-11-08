import numpy as np
import pytest

from scenes.workrooms.kitchen.rules import thresholds
from scenes.workrooms.kitchen.sponge_dynamics import BondedSpongeEvidence, NodeStability, VolumeGeometry


REST = np.array([[0., 0., 0.], [.1, 0., 0.], [0., .1, 0.], [0., 0., .04]])
TETS = [[0, 1, 2, 3]]


def test_compression_and_inversion_are_actual_volume_changes():
    geometry = VolumeGeometry(REST, TETS)
    pressed = REST * [1., 1., .8]
    state = geometry.sample(pressed + [.2, -.7, .9], np.zeros_like(REST))
    assert state['total_volume_ratio'] == pytest.approx(.8)
    assert state['minimum_edge_length_ratio'] == pytest.approx(.8)
    assert state['inverted_tetrahedron_count'] == 0
    pressed[3, 2] *= -1
    assert geometry.sample(pressed, np.zeros_like(REST))['inverted_tetrahedron_count'] == 1


def test_volume_requires_complete_native_arrays():
    geometry = VolumeGeometry(REST, TETS)
    with pytest.raises(ValueError, match='Complete finite'):
        geometry.sample(REST[:-1], np.zeros_like(REST[:-1]))
    broken = REST.copy()
    broken[1, 0] = np.nan
    with pytest.raises(ValueError, match='Complete finite'):
        geometry.sample(broken, np.zeros_like(REST))


def test_rigidly_quiet_base_does_not_hide_one_moving_foam_node():
    stability = NodeStability(thresholds('surface_wiping'), .0005)
    for tick in range(50):
        positions = REST.copy()
        positions[3, 0] += tick*.00005
        result = stability.update(positions, fully_separated=True, baseline=tick == 0)
    assert result['maximum_rolling_node_speed_m_s'] == pytest.approx(.1)
    assert not result['quiet']
    assert result['released_quiet_native_s'] == 0.


def test_near_robot_resets_the_whole_foam_release_window():
    stability = NodeStability(thresholds('surface_wiping'), .0005)
    for tick in range(850):
        result = stability.update(REST, fully_separated=True, baseline=tick == 0)
    assert result['released_quiet_native_s'] > .4
    result = stability.update(REST, fully_separated=False)
    assert result['released_quiet_native_s'] == 0.
    result = stability.update(REST, fully_separated=True)
    assert result['released_quiet_native_s'] == .0005


def test_slow_accumulated_node_drift_cannot_pass_by_local_speed_alone():
    stability = NodeStability(thresholds('surface_wiping'), .0005)
    results = []
    for tick in range(1100):
        positions = REST.copy()
        positions[3, 0] += tick*.000005
        results.append(stability.update(positions, fully_separated=True, baseline=tick == 0))
    assert results[-1]['maximum_rolling_node_speed_m_s'] < .025
    assert any(r['maximum_node_drift_from_quiet_anchor_m'] is not None
               and r['maximum_node_drift_from_quiet_anchor_m'] > .004 and not r['quiet']
               for r in results)


def bonded_fixture():
    spec = {'rest_points_m': REST.tolist(), 'tetrahedra': TETS,
        'parent_rigid_body': 'sponge', 'attachment': {
            'node_indices': [0, 1, 2], 'rigid_local_positions_m': REST[:3].tolist()}}
    evidence = BondedSpongeEvidence({'volume_bodies': {'foam': spec}}, thresholds('surface_wiping'), .0005)
    objects = {'sponge': {'position_m': [0., 0., 0.], 'quaternion_wxyz': [1., 0., 0., 0.]}}
    frame = {'baseline': True, 'volume_bodies': {'foam': {
        'positions_m': REST.copy(), 'velocities_m_s': np.zeros_like(REST)}},
        'volume_robot_geometry': {'complete': True, 'fully_separated': True,
            'method': 'native_cooked_convex_geometry', 'clearance_m': .005}}
    return evidence, objects, frame


def test_missing_native_foam_or_cooked_robot_evidence_fails_closed():
    evidence, objects, frame = bonded_fixture()
    frame.pop('volume_robot_geometry')
    with pytest.raises(ValueError, match='cooked-robot'):
        evidence.sample(frame, objects)
    evidence, objects, frame = bonded_fixture()
    frame.pop('volume_bodies')
    with pytest.raises(ValueError, match='Missing native volume'):
        evidence.sample(frame, objects)


def test_detached_foam_cannot_be_qualified_by_quiet_scouring_layer():
    evidence, objects, frame = bonded_fixture()
    _, result = evidence.sample(frame, objects)
    assert result['geometry_valid']
    frame['volume_bodies']['foam']['positions_m'] += [0., 0., .01]
    _, result = evidence.sample(frame, objects)
    assert not result['geometry_valid']
    assert result['volumes']['foam']['maximum_material_bond_error_m'] == pytest.approx(.01)


def test_positive_but_collapsed_tetrahedron_is_not_valid_material_geometry():
    evidence, objects, frame = bonded_fixture()
    frame['volume_bodies']['foam']['positions_m'][3, 2] *= .0001
    _, result = evidence.sample(frame, objects)
    assert result['volumes']['foam']['inverted_tetrahedron_count'] == 0
    assert not result['geometry_valid']
