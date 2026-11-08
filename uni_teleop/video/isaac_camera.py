"""Operator-only camera extrinsics for PICO teleoperation.

The released scene cameras remain unchanged.  PICO's wrist feed uses a small
camera frame inside the SHARPA palm, looking forward between the fingers.  It
is rigidly attached to the measured hand link and has no world target.
"""
from __future__ import annotations

import math
from copy import deepcopy

import numpy as np


def rear_camera_pose(scene, *, lateral_m=.30, height_m=1.10):
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
    eye[:2] += -.85 * forward + lateral_m * side
    eye[2] = max(base[2], target[2]) + height_m
    return eye, target



# Presentation-only presets for the validated fixed workroom layouts.  These
# include source and destination regions, not just the largest support table.
_WORKROOM_OPERATOR_POSES = {
    ('kitchen', 'pot_lid_placement'): ([.65, -.15, 1.85], [.28, .85, .94]),
    ('card_room', 'task0'): ([.48, -1.15, 1.85], [.48, -.18, .80]),
    ('laboratory', 'sample_sorting'): ([.70, 1.15, 1.48], [.62, .48, .94]),
    ('laboratory', 'vial_racking'): ([.70, 1.15, 1.48], [.65, .48, .94]),
    ('laboratory', 'cap_placement'): ([.70, 1.05, 1.40], [.65, .53, .94]),
    ('study', 'book_retrieval'): ([-.25, -1.65, 2.00], [-.30, .05, .98]),
    ('study', 'book_shelving'): ([-.25, -1.65, 2.00], [-.30, .05, .98]),
}
_WORKROOM_OPERATOR_FAMILY_POSES = {
    'mechanical_assembly': ([-.25, -.05, 1.85], [.35, .10, .80]),
    'study': ([-.25, -.05, 1.75], [.35, .10, .80]),
}


def operator_workspace_pose(scene):
    """Explicit workbench preview profile; never modify frozen scene cameras."""
    variant = scene.get('teleop_scene_variant', {})
    if variant.get('version') != 'teleop_workbench_v1':
        return None
    family, task = variant.get('family'), variant.get('task')
    pose = _WORKROOM_OPERATOR_POSES.get((family, task))
    if pose is None:
        pose = _WORKROOM_OPERATOR_FAMILY_POSES.get(family)
    return None if pose is None else tuple(np.array(v, dtype=float) for v in pose)


def operator_rear_pose(scene):
    """Use a workspace preset, or the existing forward-shifted shoulder view."""
    workspace = operator_workspace_pose(scene)
    if workspace is not None:
        return workspace
    eye, target = rear_camera_pose(scene, lateral_m=.65, height_m=1.40)
    eye[:2] = target[:2] + .70 * (eye[:2] - target[:2])
    eye[2] += .15
    base = np.asarray(scene['robot']['base_position_m'], dtype=float)
    forward = target[:2] - base[:2]
    distance = np.linalg.norm(forward)
    if distance > 1e-6:
        forward /= distance
        # Advance up to 25 cm beyond the base toward the workspace; on small
        # layouts stop at 35% of the base-to-target distance, never cross target.
        advance = min(.25, .35 * distance)
        current_depth = np.dot(target[:2] - eye[:2], forward)
        scale = (distance - advance) / current_depth
        eye = target + scale * (eye - target)
    return eye, target



def table_closeup_pose(scene):
    """Shallow downward tabletop view, keeping the operator view's azimuth."""
    main_eye, main_target = rear_camera_pose(scene, lateral_m=.65, height_m=1.40)
    table = scene['table']
    surface = table['position_m'][2] + table['half_extents_m'][2]
    target = np.asarray(table['position_m'], dtype=float).copy()
    target[2] = surface + .10
    back = main_eye - main_target
    eye = target.copy()
    eye[:2] += .80 * back[:2] / np.linalg.norm(back[:2])
    eye[2] = surface + .26
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
