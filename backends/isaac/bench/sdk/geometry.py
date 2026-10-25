"""Public coordinate operations; no scene state, model decisions or budgets.

Pose inputs use the validated benchmark protocol. Depth is optical camera z,
pixel indices are zero based, and rigid transforms use SI units and wxyz.
"""
from __future__ import annotations

import numpy as np

from bench.geometry import matrix_quaternion, quaternion_matrix, rotvec_matrix
from bench.protocol.schemas import Pose


def camera_z_to_world(*, u: int, v: int, depth_m: float,
                      intrinsics_3x3, world_from_camera: Pose):
    """Back-project a validated sensor pixel through a calibrated pinhole.

    The caller binds the pixel to a current RGB-D frame and checks its image
    bounds. This function checks its geometric meaning; it does not identify
    an object, choose a grasp, or authorize access to a sensor frame.
    """
    if depth_m <= 0:
        raise ValueError('Selected pixel has invalid depth; choose a valid observed pixel')
    intrinsic = np.asarray(intrinsics_3x3, dtype=float).reshape(3, 3)
    if (intrinsic[0, 0] <= 0 or intrinsic[1, 1] <= 0
            or not np.allclose(intrinsic[2], [0., 0., 1.], atol=1e-12, rtol=0.)
            or abs(intrinsic[1, 0]) > 1e-12):
        raise ValueError('Camera intrinsics do not describe the supported calibrated pinhole model')
    if world_from_camera.frame_id != 'world':
        raise ValueError('Camera extrinsics must describe the world frame')
    try:
        with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
            ray = np.linalg.solve(intrinsic, np.array([u + .5, v + .5, 1.]))
            camera_point = depth_m * ray
            point = (quaternion_matrix(world_from_camera.quaternion_wxyz) @ camera_point
                     + np.asarray(world_from_camera.position_m))
    except np.linalg.LinAlgError:
        raise ValueError('Camera intrinsic matrix cannot be inverted') from None
    if not np.isfinite(point).all():
        raise ValueError('Pixel projection produced nonfinite coordinates')
    return point


def apply_world_pose_delta(pose: Pose, translation_m, rotation_delta_rad):
    """Compose already-validated world translation and rotation-vector deltas.

    Rotation is left-multiplied in world axes. Action admission and limits stay
    with the caller; this calculation alone does not execute or approve motion.
    """
    result = pose.model_dump(mode='json')
    result['position_m'] = (np.array(result['position_m']) + translation_m).tolist()
    result['quaternion_wxyz'] = matrix_quaternion(
        rotvec_matrix(rotation_delta_rad) @ quaternion_matrix(result['quaternion_wxyz'])).tolist()
    return result
