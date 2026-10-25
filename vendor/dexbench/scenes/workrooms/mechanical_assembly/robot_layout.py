"""Shared, reset-only single-arm installation for the mechanical workshop.

The fixed room owns its empty table.  Every workshop task uses this same robot
and task-owned stand; movable fixtures never become room decorations.  The
workspace is a set of discrete CPU wrist probes, not a collision-free volume
or evidence of successful grasping, tool use, or native physics acceptance.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import math
from pathlib import Path

import numpy as np

from bench.geometry import rpy_matrix, transform
from bench.paths import load_json


LAYOUT_ID = 'mechanical_single_arm_v1'
ROBOT_BASE_POSITION_M = (-.23, 0., .78)
ROBOT_BASE_QUATERNION_WXYZ = (1., 0., 0., 0.)
TABLE_BOUNDS_XY_M = ((.15, -.5), (1.15, .5))
TABLE_TOP_Z_M = .78
READY_WRIST_POSITION_M = (.10, 0., 1.40)
READY_WRIST_RPY_RAD = (math.pi, 0., .75*math.pi)
ARM_SEED_RAD = (0., -.785, 0., -2.356, 0., 1.571, .785)
READY_FINGER_POSTURE_RAD = {
    'right_thumb_CMC_FE': .35, 'right_thumb_CMC_AA': .05,
    'right_thumb_MCP_FE': .35, 'right_thumb_MCP_AA': -.12,
    'right_thumb_IP': .45, 'right_pinky_CMC': .04,
    **{f'right_{finger}_{joint}': value
       for finger in ('index', 'middle', 'ring', 'pinky')
       for joint, value in (('MCP_FE', .32), ('MCP_AA', 0.),
                            ('PIP', .55), ('DIP', .42))},
}


def robot_configuration():
    """Copy the pinned rigid-pad single-arm v12 robot with the shared base pose.

    Gains, hand mount, collision settings and the original assembly version
    are inherited byte-for-byte in value.  Nothing writes to shared assets.
    """
    from bench.isaacsim.release import task_inputs
    from scenes.workrooms.registry import BASE_RELEASE, BASE_MANIFEST_SHA256
    from bench.simulation.isaacsim_backend.workroom import checked

    checked(BASE_RELEASE / 'release_manifest.json', BASE_MANIFEST_SHA256)
    _, scene, _, _ = task_inputs(BASE_RELEASE, '01_cube_push')
    robot = deepcopy(scene['robot'])
    robot['base_position_m'] = list(ROBOT_BASE_POSITION_M)
    robot['base_quaternion_wxyz'] = list(ROBOT_BASE_QUATERNION_WXYZ)
    initial, _ = ready_configuration(robot)
    robot['arm_home_rad'] = [initial['joint_positions_rad'][f'fr3_joint{i}']
                             for i in range(1, 8)]
    return robot


def geometry_model(robot):
    """The original URDF FK model, with the reviewed assembly and base pose."""
    from bench.control.factory import public_robot_geometry, robot_configuration as controller_configuration

    if (robot.get('hand_id') != 'sharpa_w01_right'
            or robot.get('assembly_version') != 'isaacsim_6_1_v2'):
        raise ValueError('Workshop layout requires the pinned rigid single-arm v12 assembly')
    if (not np.allclose(robot.get('base_position_m'), ROBOT_BASE_POSITION_M, atol=1e-12, rtol=0)
            or not np.allclose(robot.get('base_quaternion_wxyz', (1., 0., 0., 0.)),
                               ROBOT_BASE_QUATERNION_WXYZ, atol=1e-12, rtol=0)):
        raise ValueError('Workshop layout requires the reviewed setback base pose')
    if 'assets_root' not in robot:
        raise ValueError('Workshop layout requires explicit robot assets')
    return public_robot_geometry(controller_configuration({'robot': robot}))


def solve_wrist(model, position_m, rpy_rad=READY_WRIST_RPY_RAD):
    """Solve one endpoint only; no object transforms, trajectory or execution."""
    point = np.asarray(position_m, dtype=float)
    angles = np.asarray(rpy_rad, dtype=float)
    if point.shape != (3,) or angles.shape != (3,) or not np.isfinite([point, angles]).all():
        raise ValueError('Wrist targets require finite position and RPY three-vectors')
    offset = np.eye(4) if model.wrist_offset is None else model.wrist_offset
    target = np.linalg.inv(model.arm_base) @ transform(point, rpy_matrix(angles)) @ np.linalg.inv(offset)
    return model.arm.inverse(target, model.arm_tip,
        dict(zip(model.arm.names, ARM_SEED_RAD)), iterations=180, multi_starts=0,
        position_tolerance=.0001, rotation_tolerance=.002)


def ready_configuration(robot):
    """Return ``(initial_state, design_record)`` for every workshop task.

    The soft curl is a waiting posture, not a demonstrated grasp.  The hand
    waits above the gap before the table; all task objects remain visible.
    """
    model = geometry_model(robot)
    solved = solve_wrist(model, READY_WRIST_POSITION_M)
    if not solved['converged']:
        raise ValueError('Workshop ready wrist IK did not converge')
    joints = {**solved['joint_positions_rad'], **READY_FINGER_POSTURE_RAD}
    model.checked_joints(joints, enforce_limits=True)
    margin = min(min(joints[j.name]-j.lower, j.upper-joints[j.name])
                 for j in model.arm.active)
    if margin < .30:
        raise ValueError('Workshop ready arm is too close to a joint limit')
    if any(model.hand_posture_prior.violation(finger, joints) is not None
           for finger in model.finger_joints):
        raise ValueError('Workshop ready hand violates the existing posture prior')
    return {'joint_positions_rad': joints}, {
        'layout_id': LAYOUT_ID, 'base_position_m': list(ROBOT_BASE_POSITION_M),
        'wrist_position_m': list(READY_WRIST_POSITION_M),
        'wrist_rpy_rad': list(READY_WRIST_RPY_RAD),
        'wrist_fk': model.public_geometry(joints)['wrist'],
        'position_error_m': solved['position_error_m'],
        'rotation_error_rad': solved['rotation_error_rad'],
        'minimum_arm_joint_margin_rad': margin,
        'finger_posture': 'soft curl with neutral lateral spread; waiting only',
        'native_qualified': False, 'manipulation_qualified': False,
        'scope': 'Reset configuration and source URDF FK/IK; native checks remain separate.',
    }


def link_world_transforms(robot, initial_state):
    """Return original arm/hand link FK matrices for a static preview only.

    Column-vector 4x4 transforms map each link's local coordinates to world.
    This is reset-pose authoring support, never runtime link pose playback.
    """
    model = geometry_model(robot)
    joints = model.checked_joints(initial_state['joint_positions_rad'], enforce_limits=True)
    poses = model.arm.forward(joints, model.arm_base)
    offset = np.eye(4) if model.wrist_offset is None else model.wrist_offset
    poses.update(model.hand.forward(joints, poses[model.arm_tip] @ offset))
    return poses


def stand_geometry():
    """Task-owned stationary stand boxes; mounting surface equals robot base z.

    Positive dimensions, floor-standing foot and a continuous column.  A task
    composer may author these colliders but must not insert them in the room.
    """
    x, y, top = ROBOT_BASE_POSITION_M
    return {
        'layout_id': LAYOUT_ID, 'owner': 'task', 'units': 'metres',
        'base_center_to_table_edge_m': TABLE_BOUNDS_XY_M[0][0]-x,
        'foot_to_table_edge_m': TABLE_BOUNDS_XY_M[0][0]-(x+.18),
        'mount_to_table_edge_m': TABLE_BOUNDS_XY_M[0][0]-(x+.14),
        'mount_top_z_m': top,
        'boxes': [
            {'name': 'robot_stand_foot', 'position_m': [x, y, .025],
             'size_m': [.36, .40, .05], 'material': 'powder_coated_steel'},
            {'name': 'robot_stand_column', 'position_m': [x, y, (.05+top-.025)/2],
             'size_m': [.18, .22, top-.025-.05], 'material': 'powder_coated_steel'},
            {'name': 'robot_stand_mount', 'position_m': [x, y, top-.0125],
             'size_m': [.28, .30, .025], 'material': 'brushed_steel'},
        ],
        'scope': 'Geometry contract only; stand support reactions are not qualified.',
    }


def workspace_contract():
    """Task layout guidance and explicit points to audit with the current hand.

    Coordinates describe the public wrist frame, not grasp centres.  A new
    object's actual grasp transform and swept arm/hand path need own checks.
    """
    return {
        'layout_id': LAYOUT_ID, 'frame': 'world', 'units': 'metres',
        'table_bounds_xy_m': [list(v) for v in TABLE_BOUNDS_XY_M],
        'table_top_z_m': TABLE_TOP_Z_M,
        'wrist_probe_grid_m': {'x': [.23, .35, .45], 'y': [-.34, 0., .34],
                              'z': [.98, 1.08, 1.18]},
        'wrist_probe_rpy_rad': list(READY_WRIST_RPY_RAD),
        'fixture_center_guidance_xy_m': [.40, 0.],
        'gear_train_axis': 'world_y',
        'tool_and_supply_guidance': 'Near half of table, x=.25..43 and |y|<=.34; verify real grasp offsets.',
        'far_table_policy': 'No required manipulation targets at x>=.55; farther surfaces are optional display only.',
        'screw_wrist_probe_points_m': [[x, 0., z] for x in (.30, .375, .45)
                                      for z in (.95, 1.025, 1.10)],
        'native_qualified': False, 'manipulation_qualified': False,
        'scope': 'Discrete endpoint IK only; neither a reachable-volume proof nor collision-free trajectory evidence.',
    }


def robot_asset_reference(robot):
    """Read and hash-check the original v2 robot reference without modifying it."""
    geometry_model(robot)
    root = Path(robot['assets_root'])/'robots/assemblies/fr3_sharpa/isaacsim_6_1_v2'
    source_path, imported_path = root/'SOURCE.json', root/'USD_IMPORT.json'
    source, imported = load_json(source_path), load_json(imported_path)
    digest = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
    for path, expected in {imported['usd_path']: imported['usd_sha256'],
                           **imported['derived_files']}.items():
        if digest(path) != expected:
            raise ValueError('Pinned shared robot asset changed: '+str(path))
    return {'root_usd': imported['usd_path'], 'root_usd_sha256': imported['usd_sha256'],
            'source_manifest': str(source_path), 'source_manifest_sha256': digest(source_path),
            'import_manifest': str(imported_path), 'import_manifest_sha256': digest(imported_path),
            'derived_files': deepcopy(imported['derived_files']),
            'mount': deepcopy(source['mount']), 'licenses': deepcopy(source['licenses'])}
