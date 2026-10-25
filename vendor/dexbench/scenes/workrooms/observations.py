"""Explicit workroom observation setup; frozen scene releases stay unchanged.

The wrist extrinsic is a simulation mounting assumption, not a calibrated
commercial camera or an added collision body. Its optical centre is on a short
dorsal wrist bracket, not beyond the fingertips. Camera motion comes entirely
from the parent link; no world target, tracking, or initial-pose aiming is used.
"""
from __future__ import annotations

from copy import deepcopy
import math


OBSERVATION_PROFILE = 'workroom_four_camera_v1'
WRIST_MOUNT_LINK = 'right_hand_flange'
# H: fingers +z, palm +x, thumb +y. The camera looks along its own -z.
# These are explicit extrinsics in H, independent of the robot's world pose.
WRIST_OFFSET_XYZ_M = (-.08, 0., .025)
WRIST_OFFSET_RPY_RAD = (math.pi, .15, 0.)


def expected_camera_world(camera_spec, native_wrist_pose):
    """Independent row-vector camera-to-world reference for RTX validation.

    Read the measured rigid-link pose (``position_m``, ``quaternion_wxyz``) from
    PhysX-backed observations, not a requested controller goal or USD transform.
    Only explicit fixed XYZ/RPY extrinsics are supported. No camera prim, USD
    local transform, backend camera math, or RTX output enters this reference.
    Compare the result with ``inverse(camera_params.world_to_view)``.
    """
    import numpy as np

    if (camera_spec.get('kind') != 'link' or camera_spec.get('aim', False)
            or camera_spec.get('aim_target_m') is not None
            or float(camera_spec.get('standoff_m', 0.)) != 0.):
        raise ValueError('Expected a fixed link camera without aim or standoff')
    offset = np.asarray(camera_spec['offset_xyz_m'], dtype=float)
    rpy = np.asarray(camera_spec['offset_rpy_rad'], dtype=float)
    position = np.asarray(native_wrist_pose['position_m'], dtype=float)
    quaternion = np.asarray(native_wrist_pose['quaternion_wxyz'], dtype=float)
    if (any(v.shape != (3,) or not np.isfinite(v).all() for v in (offset, rpy, position))
            or quaternion.shape != (4,) or not np.isfinite(quaternion).all()
            or np.linalg.norm(quaternion) < 1e-12):
        raise ValueError('Fixed extrinsics and native wrist pose must be finite vectors')
    w, x, y, z = quaternion / np.linalg.norm(quaternion)
    wrist_rotation = np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y)],
        [2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x)],
        [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)]])
    cr, cp, cy = np.cos(rpy)
    sr, sp, sy = np.sin(rpy)
    # Column-vector fixed-axis Rz(yaw) Ry(pitch) Rx(roll).
    local_rotation = np.array([
        [cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
        [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
        [-sp, cp*sr, cp*cr]])
    result = np.eye(4)
    result[:3, :3] = (wrist_rotation @ local_rotation).T
    result[3, :3] = position + wrist_rotation @ offset
    return result


def camera_specs(description, *, resolution=(960, 720), robot=None):
    """Return three source-authored world views plus fixed wrist extrinsics.

    ``description`` is the pinned Astra scene description, including ``cameras``.
    Explicit filmback dimensions preserve square pixels at the requested image
    aspect ratio; captured RTX calibration remains the projection authority.
    Explicit robot instances each receive the same source wrist extrinsic in
    their own namespace. Omitted/single-robot declarations keep the old view IDs.
    """
    if (len(resolution) != 2 or any(isinstance(v, bool) or not isinstance(v, int) or v <= 0
                                    for v in resolution)):
        raise ValueError('resolution must be two positive integers')
    width, height = resolution
    source = description['cameras']
    result = {}
    for name, original, title, far in (
            ('overview', 'workcell', '工作区第三视角', 12.),
            ('task', 'tabletop', '桌面操作视角', 8.),
            ('room', 'room', '房间全景', 16.)):
        result[name] = {**deepcopy(source[original]), 'kind': 'world',
                        'role': 'third_person', 'title_zh': title,
                        'horizontal_aperture_mm': 36.,
                        'vertical_aperture_mm': 36. * height / width,
                        'clipping_m': [.02, far], 'resolution': [width, height]}
    result['wrist'] = {
        'kind': 'link', 'role': 'wrist', 'title_zh': '腕部相机',
        'mount_link': WRIST_MOUNT_LINK, 'aim': False, 'standoff_m': 0.,
        'offset_xyz_m': list(WRIST_OFFSET_XYZ_M),
        'offset_rpy_rad': list(WRIST_OFFSET_RPY_RAD),
        'focal_length_mm': 18., 'horizontal_aperture_mm': 36.,
        'vertical_aperture_mm': 36. * height / width,
        'clipping_m': [.015, 8.], 'resolution': [width, height],
    }
    if robot is not None and 'instances' in robot:
        from bench.simulation.isaacsim_backend.robot_instances import checked_instances, instance_name
        wrist = result.pop('wrist')
        for namespace in checked_instances(robot):
            result[namespace+'_wrist'] = {**deepcopy(wrist),
                'mount_link': instance_name(namespace, wrist['mount_link'])}
    return result


def validation_wrist_poses(target_position_m):
    """Desired H poses for observation diagnostics, requiring IK and physics.

    These are controller goals, never instructions to write a physical pose.
    The 0.32 m lower separation leaves nominal open fingertips (~0.233 m from
    H) above the target. Kinematic reachability alone does not certify collision
    clearance, a valid movement path, or visible pixels in a rendered frame.
    """
    if len(target_position_m) != 3:
        raise ValueError('target_position_m must be a finite triple')
    target = [float(v) for v in target_position_m]
    if not all(math.isfinite(v) for v in target):
        raise ValueError('target_position_m must be a finite triple')
    return {name: {'position_m': [target[0] + .04 + dx, target[1] + .05,
                                target[2] + height],
                   'quaternion_wxyz': [0., 1., 0., 0.]}
            for name, dx, height in (('high', 0., .40), ('low', 0., .32),
                                     ('low_left', -.06, .32), ('low_right', .06, .32))}
