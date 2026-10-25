"""One original FR3/right SHARPA installation serving both laboratory benches.

The fixed room keeps ownership of its furniture. This task/preview layout
places the original robot on a stand in the aisle, facing along the aisle in
its unchanged waiting posture. Discrete wrist IK is layout evidence only;
neither a grasp, an obstacle-free path nor a manipulation task is qualified.
"""
from __future__ import annotations

from copy import deepcopy
from itertools import product
import math

import numpy as np

from bench.geometry import rpy_matrix, rotation_distance
from scenes.workrooms.mechanical_assembly import robot_layout as source_layout

LAYOUT_ID = 'laboratory_two_bench_single_arm_v1'
ROBOT_BASE_POSITION_M = (.70, 1.2875, .78)
ROBOT_BASE_QUATERNION_WXYZ = (1., 0., 0., 0.)
READY_WRIST_POSITION_M = (1.03, 1.2875, 1.40)
ISLAND_NEAR_Y_M = .625
COUNTER_NEAR_Y_M = 1.95
WORKTOP_Z_M = .90


def placement_transform():
    """Column-vector rigid relocation of the original waiting installation."""
    matrix = np.eye(4)
    matrix[:3, 3] = np.asarray(ROBOT_BASE_POSITION_M) - source_layout.ROBOT_BASE_POSITION_M
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
        raise ValueError('Laboratory layout requires the pinned FR3/right-SHARPA assembly')
    np.testing.assert_allclose(robot['base_position_m'], ROBOT_BASE_POSITION_M, atol=1e-12, rtol=0.)
    np.testing.assert_allclose(robot['base_quaternion_wxyz'], ROBOT_BASE_QUATERNION_WXYZ, atol=1e-12, rtol=0.)
    return public_robot_geometry(controller_configuration({'robot': robot}))


def link_world_transforms(robot, initial):
    model = geometry_model(robot)
    joints = model.checked_joints(initial['joint_positions_rad'], enforce_limits=True)
    poses = model.arm.forward(joints, model.arm_base)
    offset = np.eye(4) if model.wrist_offset is None else model.wrist_offset
    poses.update(model.hand.forward(joints, poses[model.arm_tip] @ offset))
    return poses


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
        'layout_id': LAYOUT_ID,
        'base_position_m': list(ROBOT_BASE_POSITION_M),
        'base_quaternion_wxyz': list(ROBOT_BASE_QUATERNION_WXYZ),
        'wrist_position_m': wrist[:3, 3].tolist(), 'wrist_fk_matrix': wrist.tolist(),
        'minimum_arm_joint_margin_rad': _joint_margin(model, joints),
        'source_layout_id': source_layout.LAYOUT_ID,
        'source_joint_positions_preserved': True,
        'finger_posture': 'Original mechanical soft curl; waiting posture only',
        'native_qualified': False, 'manipulation_qualified': False,
        'scope': 'Rigid relocation and source URDF FK only; no swept-path qualification',
    }


def stand_geometry():
    matrix = placement_transform()
    boxes = [{**item, 'position_m': (matrix @ [*item['position_m'], 1.])[:3].tolist()}
             for item in source_layout.stand_geometry()['boxes']]
    foot = next(item for item in boxes if item['name'] == 'robot_stand_foot')
    half_y = foot['size_m'][1] / 2
    base_y = ROBOT_BASE_POSITION_M[1]
    return {
        'layout_id': LAYOUT_ID, 'owner': 'task_preview', 'units': 'metres',
        'base_center_to_island_edge_m': base_y - ISLAND_NEAR_Y_M,
        'base_center_to_counter_edge_m': COUNTER_NEAR_Y_M - base_y,
        'foot_to_island_edge_m': base_y - half_y - ISLAND_NEAR_Y_M,
        'foot_to_counter_edge_m': COUNTER_NEAR_Y_M - base_y - half_y,
        'mount_top_z_m': ROBOT_BASE_POSITION_M[2], 'boxes': boxes,
        'scope': 'Static floor-standing geometry; support/contact dynamics remain unqualified',
    }


def workspace_contract():
    """Discrete public-wrist samples and small-object placement guidance.

    A rectangle here guides scene authors; it is not a proof that every grasp
    in that rectangle is possible. The original soft-curl fingertips are
    explicitly transformed and reported by ``workspace_audit``.
    """
    return {
        'layout_id': LAYOUT_ID, 'frame': 'world', 'units': 'metres',
        'base_position_m': list(ROBOT_BASE_POSITION_M),
        'table_top_z_m': WORKTOP_Z_M,
        'surfaces': {
            'island': {
                'surface_bounds_xy_m': [[-.55, -.625], [1.85, .625]],
                'near_edge_y_m': ISLAND_NEAR_Y_M,
                'small_object_center_guidance_xy_m': [[.65, .52], [.85, .59]],
                'wrist_probe_grid_m': {'x': [.60, .70, .80], 'y': [.520, .535, .550],
                                       'z': [1.115, 1.170]},
                'wrist_probe_rpy_rad': [math.pi, 0., math.pi / 4],
            },
            'instrument_counter': {
                'surface_bounds_xy_m': [[-.78, 1.95], [2.02, 2.65]],
                'near_edge_y_m': COUNTER_NEAR_Y_M,
                'small_object_center_guidance_xy_m': [[.55, 1.985], [.75, 2.055]],
                'wrist_probe_grid_m': {'x': [.60, .70, .80], 'y': [2.025, 2.040, 2.055],
                                       'z': [1.115, 1.170]},
                'wrist_probe_rpy_rad': [math.pi, 0., 5 * math.pi / 4],
            },
        },
        'wrist_reference': 'Public right_hand_flange; coordinates are not object or grasp centres',
        'far_surface_policy': 'No whole-table reach claim; place required props near the audited edges.',
        'native_qualified': False, 'manipulation_qualified': False,
        'scope': 'Discrete endpoint IK only; each real grasp, arm/hand clearance and trajectory requires separate checks.',
    }


def _joint_margin(model, joints):
    return min(min(joints[j.name] - j.lower, j.upper - joints[j.name])
               for j in model.arm.active)


def workspace_audit(robot=None):
    """Recompute all 36 endpoint solves using the configured factory geometry."""
    robot = robot_configuration() if robot is None else robot
    model = geometry_model(robot)
    contract = workspace_contract()
    samples = []
    offset = np.eye(4) if model.wrist_offset is None else model.wrist_offset
    for surface_name, surface in contract['surfaces'].items():
        grid = surface['wrist_probe_grid_m']
        rpy = surface['wrist_probe_rpy_rad']
        bounds = np.asarray(surface['surface_bounds_xy_m'])
        for point in product(grid['x'], grid['y'], grid['z']):
            solved = source_layout.solve_wrist(model, point, rpy)
            if not solved['converged']:
                raise ValueError(f'Laboratory wrist sample did not converge: {surface_name} {point}')
            joints = {**solved['joint_positions_rad'], **source_layout.READY_FINGER_POSTURE_RAD}
            joints = model.checked_joints(joints, enforce_limits=True)
            margin = _joint_margin(model, joints)
            if margin < .50:
                raise ValueError(f'Laboratory wrist sample too close to joint limit: {surface_name} {point}')
            wrist = model.arm.forward(joints, model.arm_base)[model.arm_tip] @ offset
            position_error = float(np.linalg.norm(wrist[:3, 3] - point))
            rotation_error = rotation_distance(wrist[:3, :3], rpy_matrix(rpy))
            if position_error > .0001 or rotation_error > .002:
                raise ValueError('Independent FK does not reproduce the laboratory wrist target')
            public = model.public_geometry(joints)
            if public['fingertip_frame_id'] != 'right_hand_flange':
                raise ValueError('Unexpected fingertip coordinate frame')
            fingertips = {name: (wrist @ [*value, 1.])[:3].tolist()
                          for name, value in public['fingertip_positions_m'].items()}
            points = np.asarray(list(fingertips.values()))
            if not (np.all(points[:, :2] >= bounds[0])
                    and np.all(points[:, :2] <= bounds[1])
                    and np.all(points[:, 2] > WORKTOP_Z_M)):
                raise ValueError('Soft-curl fingertip sample leaves the table surface footprint')
            samples.append({
                'surface': surface_name, 'wrist_target_position_m': list(point),
                'wrist_target_rpy_rad': list(rpy), 'wrist_fk_matrix': wrist.tolist(),
                'joint_positions_rad': joints,
                'position_error_m': position_error, 'rotation_error_rad': rotation_error,
                'minimum_arm_joint_margin_rad': margin,
                'soft_curl_fingertip_world_positions_m': fingertips,
                'minimum_fingertip_height_above_table_m': float(points[:, 2].min() - WORKTOP_Z_M),
            })
    return {
        'schema': 'laboratory_discrete_workspace_audit.v1',
        'layout_id': LAYOUT_ID, 'workspace_contract': contract,
        'hand_id': robot['hand_id'], 'assembly_version': robot['assembly_version'],
        'arm_joint_limits_rad': {j.name: [j.lower, j.upper] for j in model.arm.active},
        'sample_count': len(samples), 'all_samples_converged': True,
        'maximum_position_error_m': max(s['position_error_m'] for s in samples),
        'maximum_rotation_error_rad': max(s['rotation_error_rad'] for s in samples),
        'minimum_arm_joint_margin_rad': min(s['minimum_arm_joint_margin_rad'] for s in samples),
        'minimum_fingertip_height_above_table_m': min(
            s['minimum_fingertip_height_above_table_m'] for s in samples),
        'samples': samples, 'physics_steps': 0,
        'collision_free_paths_qualified': False, 'native_qualified': False,
        'manipulation_qualified': False,
        'scope': 'Source URDF endpoint FK/IK and fingertip markers only; no grasp, obstacle clearance, continuous workspace or task-success proof.',
    }
