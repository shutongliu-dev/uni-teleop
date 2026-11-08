"""Regression checks for workspace evidence coordinate/clearance failures."""
import math

import numpy as np
import pytest

from scripts.diagnostics.check_kitchen_workspace import (
    checked_candidate, composite_volume_poses, nearest, pose_matrix, probe_rotations, wrist_for_tip)


def test_release_xyzw_pose_preserves_identity_and_yaw():
    # Release poses are xyzw while bench.geometry consumes wxyz. Mixing the
    # conventions silently inverts all upright kitchen objects.
    identity = pose_matrix([.2, -.7, .9, 0., 0., 0., 1.])
    np.testing.assert_allclose(identity[:3, :3], np.eye(3), atol=1.e-12)
    np.testing.assert_allclose(identity[:3, 3], [.2, -.7, .9])
    quarter = pose_matrix([0., 0., 0., 0., 0., math.sqrt(.5), math.sqrt(.5)])
    np.testing.assert_allclose(quarter[:3, :3] @ [1., 0., 0.], [0., 1., 0.], atol=1.e-12)
    with pytest.raises(ValueError, match='unit xyzw'):
        pose_matrix([0., 0., 0., 0., 0., 0., 2.])


def test_fingertip_frame_maps_to_surface_for_every_probe_orientation():
    point = np.array([.3, -.8, .96])
    fingertip = np.array([.064, .010, .201])
    rotations = list(probe_rotations(point, [.65, 0., .78]))
    assert len(rotations) == 12
    for _, rotation in rotations:
        transform = wrist_for_tip(point, fingertip, rotation, .015)
        np.testing.assert_allclose((transform @ [*fingertip, 1.])[:3], point+[0., 0., .015], atol=1.e-12)
        np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1.e-12)
        assert np.isclose(np.linalg.det(rotation), 1.)
        assert rotation[2, 2] < 0.


def test_per_shape_bounds_allow_empty_space_between_links_and_reject_overlap():
    robot = [[[-2., 0., 0.], [-1., 1., 1.]], [[1., 0., 0.], [2., 1., 1.]]]
    obstacle = [[[-.1, .2, .2], [.1, .8, .8]]]
    report = nearest(robot, obstacle, ['left', 'right'], ['object'])
    assert report['minimum_aabb_gap_m'] == pytest.approx(.9)
    assert report['clearance_over_5mm']
    obstructed = nearest(robot, [[[-1.1, .2, .2], [-.9, .8, .8]]], ['left', 'right'], ['object'])
    assert not obstructed['clearance_over_5mm']
    assert obstructed['robot_shape'] == 'left'
    assert obstructed['overlap_pair_count'] == 1


def test_expanded_workspace_orientations_preserve_original_attempts_and_fixed_tip():
    point, base = [.6, -.9, .96], [.65, 0., .78]
    original = list(probe_rotations(point, base))
    expanded = list(probe_rotations(point, base, profile='expanded_v1'))
    assert len(expanded) == 48
    assert len({tuple(row[0].values()) for row in expanded}) == 48
    for before, after in zip(original, expanded):
        assert before[0] == after[0]
        np.testing.assert_array_equal(before[1], after[1])
    tip = [.064, .01, .201]
    for _, rotation in expanded:
        wrist = wrist_for_tip(point, tip, rotation, .015)
        np.testing.assert_allclose((wrist @ [*tip, 1.])[:3], np.asarray(point)+[0., 0., .015], atol=1.e-12)
        np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1.e-12)
    with pytest.raises(ValueError, match='Unknown workspace orientation'):
        list(probe_rotations(point, base, profile='unrecorded'))


def test_candidate_rejects_unknown_actors_nonrigid_poses_and_unrecorded_options():
    value = {'schema': 'kitchen_workspace_candidate.v1', 'task_id': 'pot',
             'named_features': {'pot:source': 'handle_right'},
             'pose_overrides': {'goal': {'pot': [0., 0., 1., 0., 0., 0., 1.]}}}
    result = checked_candidate(value, 'pot', ['pot'])
    assert result['named_features'] == {'pot:source': 'handle_right'}
    assert result['pose_overrides'] is not value['pose_overrides']
    for change in ({'task_id': 'other'}, {'unrecorded': True}, {'ik_multistarts': True},
                   {'pose_overrides': {'goal': {'unknown': [0., 0., 1., 0., 0., 0., 1.]}}},
                   {'pose_overrides': {'goal': {'pot': [0., 0., 1., 0., 0., 0., 2.]}}},
                   {'named_features': {'pot:source': '../../fake'}}):
        with pytest.raises(ValueError):
            checked_candidate({**value, **change}, 'pot', ['pot'])


def test_volume_rest_geometry_moves_with_rigid_parent_and_preserves_local_bond_offset():
    original = [.2, -.8, .9, 0., 0., 0., 1.]
    component = [.21, -.8, .904, 0., 0., 0., 1.]
    path = '/World/Task/VolumeBodies/foam'
    meta = {'objects': [{'id': 'sponge', 'initial_pose_xyzw': original}],
            'volume_bodies': {'foam': {'parent_rigid_body': 'sponge', 'prim_path': path,
                                      'initial_pose_xyzw': component}}}
    np.testing.assert_allclose(composite_volume_poses(meta, {'sponge': original})[path], pose_matrix(component), atol=1.e-12)
    new = [.6, -.7, 1., 0., 0., math.sqrt(.5), math.sqrt(.5)]
    actual = composite_volume_poses(meta, {'sponge': new})[path]
    np.testing.assert_allclose(actual[:3, 3], [.6, -.69, 1.004], atol=1.e-12)
    np.testing.assert_allclose(actual[:3, :3], pose_matrix(new)[:3, :3], atol=1.e-12)
    assert composite_volume_poses({'objects': []}, {}) == {}
    with pytest.raises(ValueError, match='declared rigid parent'):
        composite_volume_poses(meta, {})


@pytest.mark.parametrize('bad', [[], [[[0., 0., 0.], [float('nan'), 1., 1.]]], [[[1., 0., 0.], [0., 1., 1.]]]])
def test_invalid_bounds_fail_closed(bad):
    with pytest.raises(ValueError, match='collision bounds'):
        nearest(bad, [[[0., 0., 0.], [1., 1., 1.]]], ['bad'], ['other'])
