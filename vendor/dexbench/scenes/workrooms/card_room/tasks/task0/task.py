"""Single-arm three-piece placement in the unchanged furnished card room."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np

from bench.control.factory import public_robot_geometry, robot_configuration
from bench.geometry import quaternion_matrix, rpy_matrix, transform
from bench.isaacsim.release import task_inputs
from bench.paths import load_json
from bench.simulation.isaacsim_backend.workroom import checked, digest, verify_reference
from scenes.board_games.tabletop import geometry_for
from scenes.workrooms.card_room.game_task import default_assets, verify_game_assets
from scenes.workrooms.card_room.robot_layout import (
    READY_ARM_SEED_RAD, READY_FINGER_POSTURE_RAD, READY_LOCAL_RPY_RAD,
    READY_WRIST_DISTANCE_M, READY_WRIST_HEIGHT_M, WRIST_OFFSET_XYZ_M,
    WRIST_OFFSET_RPY_RAD, arm_base_poses,
)
from scenes.workrooms.card_room.scene import environment_reference
from scenes.workrooms.observations import camera_specs
from scenes.workrooms.registry import BASE_RELEASE, BASE_MANIFEST_SHA256, PHYSICS_OVERRIDES

from .rules import bind_episode_targets, public_rules, thresholds
from .assets import bind_token_collision_asset, bind_board_collision_asset

TASK_ID = 'task0'
TASK_SCHEMA = 'card-room-placement-task-v1'
ASSET_ROOT = default_assets('tic_tac_toe_3x3')
PIECE_IDS = ('black_00', 'black_01', 'black_02')


def _ready(robot, geometry):
    """Reuse the black-side installation with one ordinary right-hand chain."""
    model = public_robot_geometry(robot_configuration({'robot': robot}))
    point = [geometry.center_xy_m[0], -READY_WRIST_DISTANCE_M, READY_WRIST_HEIGHT_M]
    rotation = quaternion_matrix(robot['base_quaternion_wxyz']) @ rpy_matrix(READY_LOCAL_RPY_RAD)
    offset = np.eye(4) if model.wrist_offset is None else model.wrist_offset
    target = np.linalg.inv(model.arm_base) @ transform(point, rotation) @ np.linalg.inv(offset)
    solved = model.arm.inverse(target, model.arm_tip,
        dict(zip(model.arm.names, READY_ARM_SEED_RAD)), iterations=140, multi_starts=0,
        position_tolerance=.0001, rotation_tolerance=.002)
    if not solved['converged']:
        raise ValueError('Task0 single-arm ready IK failed')
    joints = {**solved['joint_positions_rad'],
              **{'right_'+suffix: value for suffix, value in READY_FINGER_POSTURE_RAD.items()}}
    model.checked_joints(joints, enforce_limits=True)
    if any(model.hand_posture_prior.violation(finger, joints) is not None for finger in model.finger_joints):
        raise ValueError('Task0 ready posture violates the hand prior')
    robot['arm_home_rad'] = [joints[f'fr3_joint{i}'] for i in range(1, 8)]
    return {'joint_positions_rad': joints}, {
        'layout_id': 'card_room_single_arm_v1', 'robot_side': 'black_side',
        'base_position_m': deepcopy(robot['base_position_m']),
        'wrist_position_m': point, 'wrist_fk': model.public_geometry(joints)['wrist'],
        'position_error_m': solved['position_error_m'], 'rotation_error_rad': solved['rotation_error_rad'],
        'scope': 'Reset-only URDF design; waiting posture is not a grasp or collision qualification.'}


def build_task_inputs(*, asset_root=ASSET_ROOT, seed=0, token_collision_asset=None, board_collision_asset=None):
    """Prepare robot-only reset and seed-bound targets without advancing physics."""
    checked(BASE_RELEASE/'release_manifest.json', BASE_MANIFEST_SHA256)
    _, scene, _, _ = task_inputs(BASE_RELEASE, '01_cube_push')
    scene = deepcopy(scene)
    geometry = geometry_for('tic_tac_toe_3x3')
    asset_root = Path(asset_root).resolve()
    asset_identity = verify_game_assets(asset_root, geometry)
    reference = environment_reference()
    source_hashes = verify_reference(reference)
    for key in ('room', 'scene_variant', 'static_props', 'surfaces', 'robot_pedestal',
                'soft_bodies', 'scenario'):
        scene.pop(key, None)
    scene.update(description='Task0: one FR3+right SHARPA places three black pieces in three episode-seeded target cells.',
        cloth={'enabled': False}, workbench={'enabled': False}, workcell_details={'enabled': False},
        workroom=reference, goal_markers=[], objects=[], fixtures=[])
    scene['isaacsim'].update(deepcopy(PHYSICS_OVERRIDES))
    scene['isaacsim']['workroom_lighting'] = {'schema': 'workroom-lighting-v1',
        'family': 'card_room', 'profile': 'nominal'}
    scene['physics'] = {'dt_s': .0005, 'control_dt_s': .02, 'gravity_m_s2': [0, 0, -9.81],
                        'backend': 'native_isaacsim_physx_gpu'}
    scene['table'] = {'position_m': [.65, 0, .76], 'half_extents_m': [.5, .5, .02], 'friction': .4}
    base = arm_base_poses(geometry)['left']
    scene['robot'].update(base_position_m=base['position_m'], base_quaternion_wxyz=base['quaternion_wxyz'])
    initial, home = _ready(scene['robot'], geometry)
    for index, name in enumerate(PIECE_IDS):
        scene['objects'].append({'id': name, 'asset_id': 'token_black', 'asset_root': str(asset_root),
            'position_m': list(geometry.reserve_position('black', index)),
            'quaternion_wxyz': [1, 0, 0, 0], 'friction': .65})
    token_changes = (bind_token_collision_asset(scene, token_collision_asset)
                     if token_collision_asset is not None else None)
    scene['fixtures'] = [{'id': 'board', 'asset_id': 'board', 'asset_root': str(asset_root),
        'position_m': [*geometry.center_xy_m, geometry.table_top_z_m], 'friction': .4}] + [
        {'id': name, 'asset_id': name, 'asset_root': str(asset_root),
         'position_m': [0, 0, 0], 'friction': .4} for name in ('stand_left', 'reserve_black_west')]
    board_changes = (bind_board_collision_asset(scene, board_collision_asset)
                     if board_collision_asset is not None else None)
    cameras = camera_specs(load_json(Path(reference['dependency_manifest_path'])))
    cameras['task'].update(eye=[.65, -.02, 2.30], target=[.65, 0., .79], focal_length_mm=23.)
    cameras['wrist'].update(kind='link', role='wrist', title_zh='单臂腕部相机',
        mount_link='right_hand_flange', aim=False, standoff_m=0.,
        offset_xyz_m=list(WRIST_OFFSET_XYZ_M), offset_rpy_rad=list(WRIST_OFFSET_RPY_RAD),
        focal_length_mm=14., horizontal_aperture_mm=36., vertical_aperture_mm=27.,
        clipping_m=[.015, 8.], resolution=[960, 720])
    scene['cameras'] = cameras
    scene['card_room_placement'] = {'task_id': TASK_ID, 'target_randomization': 'episode_seed',
        'home_design': home, 'board_fixture_id': 'board',
        'reserve_supports': {name: {'fixture_id': 'reserve_black_west',
            'surface_z_m': geometry.reserve_surface_z_m} for name in PIECE_IDS}}
    scene['card_room_placement_provenance'] = {
        'base_release': str(BASE_RELEASE), 'base_manifest_sha256': BASE_MANIFEST_SHA256,
        'base_physics_profile_sha256': digest(BASE_RELEASE/'physics_profile.json'),
        'source_file_hashes': source_hashes, 'asset_identity': asset_identity,
        'pad_model': 'Unchanged single-arm v12 rigid pads; no bimanual compliance inheritance.',
        'physics_overrides': deepcopy(PHYSICS_OVERRIDES), 'preparation_status': 'unvalidated',
        'virtual_wrist_mount': 'Fixed local simulation design, not hardware calibration.'}
    if token_changes is not None:
        scene['card_room_placement_provenance']['token_collision_asset'] = {
            'selection': deepcopy(token_collision_asset), 'bindings': token_changes,
            'scope': 'Explicit collision candidate; native qualification is external evidence.'}
    if board_changes is not None:
        scene['card_room_placement_provenance']['board_collision_asset'] = {
            'selection': deepcopy(board_collision_asset), 'bindings': board_changes,
            'scope': 'Explicit board representation candidate; native qualification is external evidence.'}
    evaluation = {**thresholds(), 'geometry': geometry.specification(),
        'target_cells': [[0, 0], [1, 0], [2, 0]], 'piece_ids': list(PIECE_IDS),
        'actors': {name: '/World/Objects/'+name for name in PIECE_IDS},
        'board_contact_paths': ['/World/Fixtures/board/collision_000'],
        'robot_actor_prefix': '/World/Robot/',
        'initial_objects': {row['id']: {key: deepcopy(row[key]) for key in ('position_m', 'quaternion_wxyz')}
                            for row in scene['objects']}}
    task = {'schema_version': TASK_SCHEMA, 'id': TASK_ID, 'task_id': TASK_ID,
        'title_zh': 'Task0 · 单臂随机三子摆放', 'difficulty': 'guided_single_arm', 'cloth': False,
        'steps': [{'id': 'place_three', 'instruction_zh': '将三枚黑子分别放入本回合随机指定的三个目标格，松手并保持稳定。'}],
        'public_rules': public_rules(evaluation['target_cells']), 'evaluation': evaluation,
        'initial_state': deepcopy(initial),
        'recommended_budgets': {'sim_time_s': 120., 'wall_time_s': 1800., 'max_actions': 6000},
        'training_policy_included': False, 'physics_accepted': False, 'task_policy_success_claimed': False,
        'qualification_scope': 'Prepared task; native initialization and full three-piece manipulation require separate evidence.'}
    task = bind_episode_targets(task, seed)
    return {'scene': scene, 'task': task, 'initial_state': initial}
