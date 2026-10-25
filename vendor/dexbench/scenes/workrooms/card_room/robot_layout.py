"""Reset-only opposing robot ready poses for the card-room board games.

A waiting hand is not a pregrasp: both robots use the same arm-local pose,
independent of piece identity, inventory order or current game state.  The
fixed wrist camera looks at a local workspace; it never tracks a world target.
All dimensions/postures are simulation design priors, not manipulation or
native collision qualification.
"""
from __future__ import annotations

import math

import numpy as np

from bench.control.factory import public_robot_geometry, robot_configuration
from bench.geometry import rpy_matrix, transform


# A soft curl with closed lateral spread.  These are explicit reset targets,
# not the single-finger push/grasp prior and not a grasp command.
READY_FINGER_POSTURE_RAD = {
    'thumb_CMC_FE': .35, 'thumb_CMC_AA': .05,
    'thumb_MCP_FE': .35, 'thumb_MCP_AA': -.12, 'thumb_IP': .45,
    'pinky_CMC': .04,
    **{f'{finger}_{joint}': value
       for finger in ('index', 'middle', 'ring', 'pinky')
       for joint, value in (('MCP_FE', .32), ('MCP_AA', 0.), ('PIP', .55), ('DIP', .42))},
}
# Shared by task composition and static stand authoring; metres in the fixed room.
ROBOT_BASE_DISTANCE_M = .60
ROBOT_BASE_HEIGHT_M = .90
READY_WRIST_DISTANCE_M = .40
READY_WRIST_HEIGHT_M = 1.42

READY_LOCAL_RPY_RAD = (math.pi, 0., .75*math.pi)
READY_ARM_SEED_RAD = (0., -.785, 0., -2.356, 0., 1.571, .785)
# H: fingers +z, palm +x.  A virtual dorsal/side bracket projects clear of the
# hand silhouette.  Camera image up points into that robot's work area at ready.
WRIST_OFFSET_XYZ_M = (-.08, .08, .04)
WRIST_OFFSET_RPY_RAD = (-math.pi, 0., -.75*math.pi)


def arm_base_poses(geometry):
    """Reset-time opposing fixed bases, 80 mm farther back and 50 mm lower.

    These transforms are also the static pedestal mount targets.  The foot
    stays on the floor; lowering a mount requires a shorter column, never a
    downward translation of the complete pedestal through the floor.
    """
    center = np.asarray(geometry.center_xy_m, dtype=float)
    if center.shape != (2,) or not np.isfinite(center).all():
        raise ValueError('Robot bases require a finite board centre')
    return {side: {'position_m': [float(center[0]), float(center[1]+sign*ROBOT_BASE_DISTANCE_M),
                                  ROBOT_BASE_HEIGHT_M],
                   'quaternion_wxyz': [math.sqrt(.5), 0., 0., -sign*math.sqrt(.5)]}
            for side, sign in (('left', -1), ('right', 1))}


def _wrist_camera(side):
    return {
        'kind': 'link', 'role': 'wrist',
        'title_zh': ('黑方' if side == 'left' else '白方')+'腕部相机',
        'mount_link': side+'_hand_flange', 'aim': False, 'standoff_m': 0.,
        'offset_xyz_m': list(WRIST_OFFSET_XYZ_M),
        'offset_rpy_rad': list(WRIST_OFFSET_RPY_RAD),
        'focal_length_mm': 14., 'horizontal_aperture_mm': 36.,
        'vertical_aperture_mm': 27., 'clipping_m': [.015, 8.],
        'resolution': [960, 720],
    }


def ready_configuration(robot, geometry):
    """Return ``(initial_state, home_design, wrist_camera_overrides)``.

    The task owns the opposing base layout.  This function accepts the existing
    card-room bases, validates that assumption, and returns joint coordinates
    only; it never edits bases, mesh geometry, dynamics, or a running simulator.
    """
    center = np.asarray(geometry.center_xy_m, dtype=float)
    top = float(geometry.table_top_z_m)
    if (center.shape != (2,) or not np.isfinite(center).all()
            or not math.isfinite(top) or abs(geometry.board_yaw_rad) > 1e-12):
        raise ValueError('Ready layout requires a finite axis-aligned card-room board')
    if robot.get('hand_id') != 'dual_fr3_sharpa_w01':
        raise ValueError('Ready layout requires the pinned dual FR3 + SHARPA robot')
    if not {'assets_root', 'base_position_m', 'assembly_version', 'arm_base_poses'} <= robot.keys():
        raise ValueError('Ready layout requires explicit robot assets, assembly and opposing bases')
    model = public_robot_geometry(robot_configuration({'robot': robot}))
    initial = {'joint_positions_rad': {}}
    homes, cameras = {}, {}
    for side, sign in (('left', -1), ('right', 1)):
        single = model.sides[side]
        expected_base = transform(arm_base_poses(geometry)[side]['position_m'],
                                  rpy_matrix([0., 0., -sign*math.pi/2]))
        if not np.allclose(single.arm_base, expected_base, atol=1e-9, rtol=0):
            raise ValueError('Ready layout requires the reviewed opposing card-room bases')
        # The complete arms retreat with their bases; hands remain outside the
        # expanded board strip and lower by 80 mm versus ready_v2.  Keep a
        # source-mesh hand/arm gap; v1's index/link2 collision is retained as
        # failed history, never repaired by changing collision filtering.
        point = np.array([center[0], center[1]+sign*READY_WRIST_DISTANCE_M, READY_WRIST_HEIGHT_M])
        rotation = single.arm_base[:3, :3] @ rpy_matrix(READY_LOCAL_RPY_RAD)
        wrist_offset = np.eye(4) if single.wrist_offset is None else single.wrist_offset
        target = np.linalg.inv(single.arm_base) @ transform(point, rotation) @ np.linalg.inv(wrist_offset)
        solved = single.arm.inverse(target, single.arm_tip,
            dict(zip(single.arm.names, READY_ARM_SEED_RAD)), iterations=140, multi_starts=0,
            position_tolerance=.0001, rotation_tolerance=.002)
        if not solved['converged']:
            raise ValueError(f'Card-room ready IK failed: {side}')
        joints = {**solved['joint_positions_rad'],
                  **{side+'_'+suffix: value for suffix, value in READY_FINGER_POSTURE_RAD.items()}}
        single.checked_joints(joints, enforce_limits=True)
        arm_margin = min(min(joints[j.name]-j.lower, j.upper-joints[j.name]) for j in single.arm.active)
        if arm_margin < .30:
            raise ValueError(f'Card-room ready arm is too close to a joint limit: {side}')
        for finger in single.finger_joints:
            if single.hand_posture_prior.violation(finger, joints) is not None:
                raise ValueError('Ready posture violates the existing hand posture prior')
        initial['joint_positions_rad'].update(joints)
        fk = single.arm_base @ single.arm.forward(joints)[single.arm_tip] @ wrist_offset
        # Actual board cell coordinates, using surface height rather than the
        # centre-of-token height.  These are audit probes, never camera aims.
        middle = geometry.size//2
        rows = range(0, middle+1) if sign < 0 else range(middle, geometry.size)
        probes = []
        for row in rows:
            for column in sorted({max(0, middle-1), middle, min(geometry.size-1, middle+1)}):
                position = list(geometry.cell_center(column, row))
                position[2] = geometry.board_top_z_m+.001
                probes.append(position)
        homes[side] = {
            'source': 'symmetric arm-local ready pose; independent of reserve identity',
            'profile': 'card_room_ready_v3',
            'wrist_position_m': point.tolist(),
            'wrist_fk_position_m': fk[:3, 3].tolist(),
            'wrist_rotation_matrix': fk[:3, :3].tolist(),
            'position_error_m': solved['position_error_m'],
            'rotation_error_rad': solved['rotation_error_rad'],
            'minimum_arm_joint_margin_rad': arm_margin,
            'finger_posture': 'soft curl; neutral finger lateral spread',
            'base_position_m': arm_base_poses(geometry)[side]['position_m'],
            'base_distance_from_board_center_m': ROBOT_BASE_DISTANCE_M,
            'wrist_probe_points_world': probes,
            'probe_scope': 'board-surface depth probes; not camera tracking targets',
            'collision_qualified': False,
            'scope': 'URDF FK/IK design only; native clearance and camera validation are external evidence',
        }
        cameras['wrist_'+side] = _wrist_camera(side)
    return initial, homes, cameras
