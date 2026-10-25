"""Two unchanged FR3/right-SHARPA robots opposed across the living-room table.

This is task-owned placement and reset geometry. Both hands remain the pinned
right hand: their poses have 180-degree rotational symmetry, not a reflected
left-hand mesh. Endpoint IK is separate from collision or task qualification.
"""
from __future__ import annotations

from copy import deepcopy
import math

import numpy as np

from scenes.workrooms.mechanical_assembly import robot_layout as source_layout

LAYOUT_ID = 'living_room_opposed_dual_right_sharpa_v1'
TABLE_CENTER_XY_M = (.65, 0.)
TABLE_BOUNDS_XY_M = ((.15, -.50), (1.15, .50))
TABLE_TOP_Z_M = .78
BASE_HALF_SPACING_M = .76
GROUND_DATUM_M = {'front': .005, 'rear': .012}
SIDES = ('front', 'rear')


def placements(base_half_spacing_m=None):
    """Unscaled installations; a positive gap separates each stand and table."""
    distance = BASE_HALF_SPACING_M if base_half_spacing_m is None else float(base_half_spacing_m)
    if not math.isfinite(distance) or distance <= .68:
        raise ValueError('Base half-spacing must exceed .68 m for stand/table separation')
    return {side: {
        'base_position_m': [.65, sign*distance, TABLE_TOP_Z_M],
        'base_quaternion_wxyz': [math.sqrt(.5), 0., 0., -sign*math.sqrt(.5)],
        'base_yaw_rad': -sign*math.pi/2,
        'ready_wrist_position_m': [.65, sign*(distance-.33), 1.40],
        'wrist_probe_rpy_rad': [math.pi, 0., .75*math.pi-sign*math.pi/2],
    } for side, sign in (('front', -1), ('rear', 1))}


def placement_transform(side, base_half_spacing_m=None):
    """Column-vector rigid transform from the mechanical source installation."""
    pose = placements(base_half_spacing_m)[side]
    yaw = pose['base_yaw_rad']
    matrix = np.eye(4)
    matrix[:3, :3] = [[0., -math.sin(yaw), 0.], [math.sin(yaw), 0., 0.], [0., 0., 1.]]
    matrix[:3, 3] = np.asarray(pose['base_position_m']) - matrix[:3, :3] @ source_layout.ROBOT_BASE_POSITION_M
    return matrix


def robot_configurations(base_half_spacing_m=None, source_robot=None):
    """Copy the pinned source twice, changing only each base position/rotation."""
    source_robot = source_layout.robot_configuration() if source_robot is None else source_robot
    source_layout.geometry_model(source_robot)
    robots = {}
    for side, pose in placements(base_half_spacing_m).items():
        robot = deepcopy(source_robot)
        for key in ('base_position_m', 'base_quaternion_wxyz'):
            robot[key] = pose[key].copy()
        geometry_model(robot)
        robots[side] = robot
    return robots


def _robot_placement(robot):
    position = np.asarray(robot['base_position_m'], dtype=float)
    if position.shape != (3,) or not np.isfinite(position).all():
        raise ValueError('A finite three-vector robot base is required')
    side = 'front' if position[1] < 0. else 'rear'
    pose = placements(abs(float(position[1])))[side]
    for key in ('base_position_m', 'base_quaternion_wxyz'):
        if not np.allclose(robot[key], pose[key], atol=1e-12, rtol=0.):
            raise ValueError('Robot base does not match the opposed living-room layout')
    return side, pose


def geometry_model(robot):
    """Use the shared public geometry factory; no controller is instantiated."""
    from bench.control.factory import public_robot_geometry, robot_configuration as controller_configuration
    if (robot.get('hand_id') != 'sharpa_w01_right'
            or robot.get('assembly_version') != 'isaacsim_6_1_v2'
            or 'assets_root' not in robot):
        raise ValueError('Dual layout requires the pinned FR3/right-SHARPA assembly')
    _robot_placement(robot)
    return public_robot_geometry(controller_configuration({'robot': robot}))


def ready_configuration(robot, source_initial=None):
    """Reuse the original mechanical ready joints and waiting hand posture."""
    if source_initial is None:
        source_initial, _ = source_layout.ready_configuration(source_layout.robot_configuration())
    initial = deepcopy(source_initial)
    side, pose = _robot_placement(robot)
    model = geometry_model(robot)
    joints = model.checked_joints(initial['joint_positions_rad'], enforce_limits=True)
    poses = link_world_transforms(robot, initial)
    offset = np.eye(4) if model.wrist_offset is None else model.wrist_offset
    wrist = poses[model.arm_tip] @ offset
    if not np.allclose(wrist[:3, 3], pose['ready_wrist_position_m'], atol=1e-4, rtol=0.):
        raise ValueError('Ready joints do not reproduce the relocated source wrist')
    return initial, {
        'layout_id': LAYOUT_ID, 'side': side, **deepcopy(pose),
        'wrist_fk_matrix': wrist.tolist(),
        'minimum_arm_joint_margin_rad': min(min(joints[j.name]-j.lower, j.upper-joints[j.name])
                                           for j in model.arm.active),
        'source_layout_id': source_layout.LAYOUT_ID, 'source_joint_positions_preserved': True,
        'finger_posture': 'Original mechanical soft curl; waiting posture only',
        'native_qualified': False, 'manipulation_qualified': False,
        'scope': 'Reset geometry and source URDF FK; no path, grasp or contact qualification',
    }


def link_world_transforms(robot, initial):
    model = geometry_model(robot)
    joints = model.checked_joints(initial['joint_positions_rad'], enforce_limits=True)
    poses = model.arm.forward(joints, model.arm_base)
    offset = np.eye(4) if model.wrist_offset is None else model.wrist_offset
    poses.update(model.hand.forward(joints, poses[model.arm_tip] @ offset))
    return poses


def stand_geometry(base_half_spacing_m=None):
    """Original stand boxes rotated with each robot; room geometry is untouched."""
    result = {}
    for side, pose in placements(base_half_spacing_m).items():
        distance = abs(pose['base_position_m'][1])
        matrix = placement_transform(side, distance)
        boxes = [{**item,
            'position_m': (matrix @ [*item['position_m'], 1.])[:3].tolist(),
            'size_m': (np.abs(matrix[:3, :3]) @ item['size_m']).tolist()}
            for item in source_layout.stand_geometry()['boxes']]
        ground = GROUND_DATUM_M[side]
        # The room floor/rug has a measured upper surface. Keep the robot mount
        # height fixed while the foot and the continuous column meet that datum.
        for box in boxes:
            if box['name'] == 'robot_stand_foot':
                box['position_m'][2] = ground+box['size_m'][2]/2
            elif box['name'] == 'robot_stand_column':
                bottom, top = ground+.05, TABLE_TOP_Z_M-.025
                box['position_m'][2] = (bottom+top)/2
                box['size_m'][2] = top-bottom
        result[side] = {
            'layout_id': LAYOUT_ID, 'side': side, 'owner': 'task_preview', 'units': 'metres',
            'ground_datum_m': ground,
            'base_center_to_table_edge_m': distance-.50,
            'foot_to_table_edge_m': distance-.68,
            'mount_to_table_edge_m': distance-.64,
            'mount_top_z_m': TABLE_TOP_Z_M, 'boxes': boxes,
            'scope': 'Static floor-standing geometry; support/contact dynamics unqualified',
        }
    return result


def robot_asset_reference():
    """Hash-check the same original assembly used by both unmodified copies."""
    return source_layout.robot_asset_reference(source_layout.robot_configuration())
