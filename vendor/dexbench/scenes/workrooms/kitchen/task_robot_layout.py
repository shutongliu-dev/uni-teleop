"""Task-facing central-aisle layouts; preserve the original robot and ready joints."""
from copy import deepcopy
import math

import numpy as np

from scenes.workrooms.mechanical_assembly import robot_layout as source_layout

TASK_IDS = ('breakfast_serving', 'produce_transfer', 'utensil_sorting', 'pot_lid_placement',
            'pantry_stocking', 'table_clearing')
BASE_POSITION_M = (.65, 0., .78)
# Exact original ready values sealed in all four kitchen v1 initial states.
# Re-solving the same source IK in Isaac's NumPy can differ by ~4e-16 rad;
# retain that solve as an independent check, not as a source of release bytes.
READY_ARM_RAD = (-0.0001148324960314788, -0.6928171090342241,
    7.460956668232696e-05, -2.2227211715091095,
    4.7763166686930226e-05, 1.53000628317549, 0.7853426240143195)


def _original_configuration():
    robot = deepcopy(source_layout.robot_configuration())
    if not np.allclose(robot['arm_home_rad'], READY_ARM_RAD, atol=1.e-12, rtol=0.):
        raise ValueError('Original ready arm differs from the frozen kitchen v1 reset')
    robot['arm_home_rad'] = list(READY_ARM_RAD)
    return robot


def yaw_deg(task_id):
    if task_id not in TASK_IDS:
        raise ValueError('Unknown kitchen task: ' + str(task_id))
    return 90. if task_id == 'pot_lid_placement' else -90.


def base_quaternion(task_id):
    angle = math.radians(yaw_deg(task_id)) / 2
    return (math.cos(angle), 0., 0., math.sin(angle))


def placement_transform(task_id):
    angle = math.radians(yaw_deg(task_id))
    c, s = math.cos(angle), math.sin(angle)
    result = np.eye(4)
    result[:3, :3] = [[c, -s, 0.], [s, c, 0.], [0., 0., 1.]]
    result[:3, 3] = np.asarray(BASE_POSITION_M) - result[:3, :3] @ source_layout.ROBOT_BASE_POSITION_M
    return result


def robot_configuration(task_id):
    robot = _original_configuration()
    robot['base_position_m'] = list(BASE_POSITION_M)
    robot['base_quaternion_wxyz'] = list(base_quaternion(task_id))
    return robot


def ready_configuration(task_id, robot=None):
    from bench.control.factory import public_robot_geometry, robot_configuration as controller_configuration
    original = _original_configuration()
    source_initial, source_record = source_layout.ready_configuration(original)
    arm = {f'fr3_joint{i}': value for i, value in enumerate(READY_ARM_RAD, 1)}
    source_joints = source_initial['joint_positions_rad']
    if not np.allclose([source_joints[name] for name in arm], READY_ARM_RAD, atol=1.e-12, rtol=0.):
        raise ValueError('Original ready IK differs from the frozen kitchen v1 reset')
    hand = {name: value for name, value in source_joints.items() if name not in arm}
    if hand != source_layout.READY_FINGER_POSTURE_RAD:
        raise ValueError('Original ready hand differs from its declared posture')
    initial = {'joint_positions_rad': {**arm, **hand}}
    robot = robot_configuration(task_id) if robot is None else robot
    expected = deepcopy(original)
    expected.update(base_position_m=list(BASE_POSITION_M), base_quaternion_wxyz=list(base_quaternion(task_id)))
    if robot != expected:
        raise ValueError('Kitchen task installation must preserve the pinned robot and all original settings')
    model = public_robot_geometry(controller_configuration({'robot': robot}))
    joints = model.checked_joints(initial['joint_positions_rad'], enforce_limits=True)
    poses = model.arm.forward(joints, model.arm_base)
    offset = np.eye(4) if model.wrist_offset is None else model.wrist_offset
    wrist = poses[model.arm_tip] @ offset
    wanted = placement_transform(task_id) @ [*source_layout.READY_WRIST_POSITION_M, 1.]
    np.testing.assert_allclose(wrist[:3, 3], wanted[:3], atol=1.e-4, rtol=0.)
    return initial, {'layout_id': 'kitchen_task_single_arm_v1', 'task_id': task_id,
        'base_position_m': list(BASE_POSITION_M), 'base_yaw_deg': yaw_deg(task_id),
        'base_quaternion_wxyz': list(base_quaternion(task_id)), 'wrist_position_m': wrist[:3, 3].tolist(),
        'source_layout_id': source_layout.LAYOUT_ID, 'source_ready_design': source_record,
        'source_joint_positions_preserved': True, 'controller_settings_preserved': True,
        'native_qualified': False, 'manipulation_qualified': False,
        'scope': 'Task orientation and source URDF FK only; no grasp or swept-path qualification.'}
