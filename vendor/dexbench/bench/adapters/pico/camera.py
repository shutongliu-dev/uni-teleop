"""Operator-only camera extrinsics for PICO teleoperation.

The released scene cameras remain unchanged.  PICO's wrist feed uses a small
camera frame inside the SHARPA palm, looking forward between the fingers.  It
is rigidly attached to the measured hand link and has no world target.
"""
from __future__ import annotations

import math
from copy import deepcopy

import numpy as np


def rear_camera_pose(scene):
    """Fixed over-shoulder view, oriented from the robot base toward the table."""
    base = np.asarray(scene['robot']['base_position_m'], dtype=float)
    table = scene['table']
    target = np.asarray(table['position_m'], dtype=float).copy()
    target[2] += table['half_extents_m'][2] + .12
    forward = target[:2] - base[:2]
    distance = np.linalg.norm(forward)
    forward = forward / distance if distance > 1e-6 else np.array([1., 0.])
    side = np.array([-forward[1], forward[0]])
    eye = base.copy()
    eye[:2] += -.85 * forward + .30 * side
    eye[2] = max(base[2], target[2]) + 1.10
    return eye, target


def apply_side_overview(scene):
    """Show arm flexion side-on without changing robot or task geometry."""
    result=deepcopy(scene)
    base=np.asarray(scene['robot']['base_position_m'],dtype=float)
    forward=np.asarray(scene['table']['position_m'],dtype=float)[:2]-base[:2]
    distance=np.linalg.norm(forward)
    forward=forward/distance if distance>1e-6 else np.array([1.,0.])
    side=np.array([-forward[1],forward[0]])
    target=base+np.array([.25*forward[0],.25*forward[1],.45])
    eye=target+np.array([1.8*side[0],1.8*side[1],.35])
    result['cameras']['overview'].update(eye=eye.tolist(),target=target.tolist(),
                                        focal_length_mm=24.,horizontal_aperture_mm=36.)
    return result


IN_HAND_WRIST_CAMERA_V1 = {
    'mount_link': 'right_hand_C_MC',
    'aim': False,
    'standoff_m': 0.0,
    # C_MC is the palm center; +z points from the palm toward the fingers.
    # On the palmar surface, behind the finger bases; never inside solid mesh.
    'offset_xyz_m': [0.055, 0.05, 0.045],
    # USD cameras look along local -z; pi roll points the view along hand +z.
    'offset_rpy_rad': [math.pi, 0.2, 0.0],
    'look_axis': '+z',
}


def apply_in_hand_wrist_camera(scene):
    """Return a scene copy whose operator wrist camera is palm-mounted.

    This is a teleoperation presentation override.  It preserves the released
    camera's optics, resolution, clipping and all task data.  Scenes without a
    ``wrist`` camera are returned unchanged so head/overview-only runs remain
    valid.
    """
    result = deepcopy(scene)
    cameras = result.get('cameras', {})
    entries = cameras.get('channels', cameras) if isinstance(cameras, dict) else cameras
    if isinstance(entries, dict):
        entries = [entries['wrist']] if 'wrist' in entries else []
    else:
        entries = [c for c in entries if c.get('id') == 'wrist']
    for camera in entries:
        camera.update(deepcopy(IN_HAND_WRIST_CAMERA_V1))
        camera.pop('aim_target_m', None)
    return result
