"""Front-side FR3 + right SHARPA installation for living-room presentation.

This rigidly relocates the reviewed mechanical waiting posture. It preserves
all robot assets, joint positions, controller values and physical parameters;
it does not implement or qualify any of the planned living-room tasks.
"""
from __future__ import annotations

from copy import deepcopy
import math

import numpy as np

from scenes.workrooms.mechanical_assembly import robot_layout as source_layout

LAYOUT_ID = 'living_room_front_single_arm_v1'
ROBOT_BASE_POSITION_M = (.65, -.88, .78)
ROBOT_BASE_QUATERNION_WXYZ = (math.sqrt(.5), 0., 0., math.sqrt(.5))
READY_WRIST_POSITION_M = (.65, -.55, 1.40)
TABLE_BOUNDS_XY_M = ((.15, -.50), (1.15, .50))


def placement_transform():
    """Column-vector rigid transform from the source installation to this one."""
    matrix = np.eye(4)
    matrix[:3, :3] = [[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]]
    matrix[:3, 3] = np.asarray(ROBOT_BASE_POSITION_M) - matrix[:3, :3] @ source_layout.ROBOT_BASE_POSITION_M
    return matrix


def robot_configuration(source_robot=None):
    source_robot = source_layout.robot_configuration() if source_robot is None else source_robot
    robot = deepcopy(source_robot)
    robot['base_position_m'] = list(ROBOT_BASE_POSITION_M)
    robot['base_quaternion_wxyz'] = list(ROBOT_BASE_QUATERNION_WXYZ)
    geometry_model(robot)
    return robot


def geometry_model(robot):
    from bench.control.factory import public_robot_geometry, robot_configuration as controller_configuration
    if (robot.get('hand_id') != 'sharpa_w01_right'
            or robot.get('assembly_version') != 'isaacsim_6_1_v2'):
        raise ValueError('Living-room preview requires the pinned FR3/right-SHARPA assembly')
    np.testing.assert_allclose(robot['base_position_m'], ROBOT_BASE_POSITION_M, atol=1e-12, rtol=0.)
    np.testing.assert_allclose(robot['base_quaternion_wxyz'], ROBOT_BASE_QUATERNION_WXYZ, atol=1e-12, rtol=0.)
    return public_robot_geometry(controller_configuration({'robot': robot}))


def ready_configuration(robot, source_initial=None):
    if source_initial is None:
        source_initial, _ = source_layout.ready_configuration(source_layout.robot_configuration())
    initial = deepcopy(source_initial)
    model = geometry_model(robot)
    joints = model.checked_joints(initial['joint_positions_rad'], enforce_limits=True)
    poses = link_world_transforms(robot, initial)
    offset = np.eye(4) if model.wrist_offset is None else model.wrist_offset
    wrist = poses[model.arm_tip] @ offset
    np.testing.assert_allclose(wrist[:3, 3], READY_WRIST_POSITION_M, atol=1e-4, rtol=0.)
    return initial, {
        'layout_id': LAYOUT_ID, 'base_position_m': list(ROBOT_BASE_POSITION_M),
        'base_quaternion_wxyz': list(ROBOT_BASE_QUATERNION_WXYZ),
        'wrist_position_m': list(READY_WRIST_POSITION_M), 'wrist_fk_matrix': wrist.tolist(),
        'minimum_arm_joint_margin_rad': min(min(joints[j.name] - j.lower, j.upper - joints[j.name])
                                           for j in model.arm.active),
        'source_layout_id': source_layout.LAYOUT_ID, 'source_joint_positions_preserved': True,
        'finger_posture': 'Original mechanical soft curl; waiting posture only',
        'native_qualified': False, 'manipulation_qualified': False,
        'scope': 'Rigid relocation and CPU source URDF FK only; no task or swept-path qualification',
    }


def link_world_transforms(robot, initial):
    model = geometry_model(robot)
    joints = model.checked_joints(initial['joint_positions_rad'], enforce_limits=True)
    poses = model.arm.forward(joints, model.arm_base)
    offset = np.eye(4) if model.wrist_offset is None else model.wrist_offset
    poses.update(model.hand.forward(joints, poses[model.arm_tip] @ offset))
    return poses


def stand_geometry():
    original = source_layout.stand_geometry()
    matrix = placement_transform()
    boxes = []
    for item in original['boxes']:
        boxes.append({**item,
            'position_m': (matrix @ [*item['position_m'], 1.])[:3].tolist(),
            'size_m': (np.abs(matrix[:3, :3]) @ item['size_m']).tolist()})
    return {'layout_id': LAYOUT_ID, 'owner': 'task_preview', 'units': 'metres',
        'base_center_to_table_edge_m': .38, 'foot_to_table_edge_m': .20,
        'mount_to_table_edge_m': .24, 'mount_top_z_m': .78, 'boxes': boxes,
        'scope': 'Static floor-standing geometry; support/contact dynamics remain unqualified'}
