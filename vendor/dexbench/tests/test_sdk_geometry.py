"""The reusable SDK preserves optical/world conventions independently of Agents."""
import math

import numpy as np
import pytest

from bench.geometry import quaternion_matrix
from bench.protocol.schemas import Pose
from bench.sdk.geometry import apply_world_pose_delta, camera_z_to_world


def test_camera_z_projection_accounts_for_skew_half_pixels_and_extrinsics():
    pose = Pose(frame_id='world', position_m=[1., 2., 3.],
                quaternion_wxyz=[math.sqrt(.5), 0., 0., math.sqrt(.5)])
    actual = camera_z_to_world(u=5, v=3, depth_m=2.,
        intrinsics_3x3=[2., .5, 3., 0., 4., 2., 0., 0., 1.], world_from_camera=pose)
    np.testing.assert_allclose(actual, [.25, 4.3125, 5.], atol=1e-12)


def test_pose_delta_uses_world_axes_and_does_not_mutate_the_pose():
    pose = Pose(frame_id='world', position_m=[1., 2., 3.],
                quaternion_wxyz=[math.sqrt(.5), math.sqrt(.5), 0., 0.])
    before = pose.model_dump(mode='json')
    actual = apply_world_pose_delta(pose, np.array([.1, .2, .3]), [0., 0., math.pi / 2])
    np.testing.assert_allclose(actual['position_m'], [1.1, 2.2, 3.3], atol=1e-12)
    # Rz(pi/2) Rx(pi/2), not a local Rx(pi/2) Rz(pi/2) rotation.
    np.testing.assert_allclose(quaternion_matrix(actual['quaternion_wxyz']),
                               [[0., 0., 1.], [1., 0., 0.], [0., 1., 0.]], atol=1e-12)
    assert pose.model_dump(mode='json') == before


@pytest.mark.parametrize('depth', [0., -1.])
def test_projection_rejects_an_invalid_depth_without_a_client(depth):
    with pytest.raises(ValueError, match='Selected pixel has invalid depth'):
        camera_z_to_world(u=0, v=0, depth_m=depth, intrinsics_3x3=np.eye(3),
            world_from_camera=Pose(frame_id='world', position_m=[0., 0., 0.], quaternion_wxyz=[1., 0., 0., 0.]))
