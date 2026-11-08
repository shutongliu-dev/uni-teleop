"""Task-owned FR3 installations, preserving historical right-hand instances.

Historical same-side dual-right releases use two independent anchored stands
along the workshop table's -X edge. Their ``front`` and ``rear`` instance IDs
name negative/positive Y. That layout and the old opposed layout stay available
for immutable release reproduction.

The left/right layout uses the existing official left and right SHARPA
assemblies, with two FR3 articulations controlled by one Agent. Its left arm
ready joints mirror the original right pose; the hand is a real left asset,
never a rotated or negatively scaled right-hand mesh.
"""
from __future__ import annotations

from copy import deepcopy

import numpy as np

from . import robot_layout as source_layout

OPPOSED_LAYOUT_ID = 'mechanical_opposed_dual_right_sharpa_v1'
SAME_SIDE_LAYOUT_ID = 'mechanical_same_side_dual_right_sharpa_v1'
MIRRORED_LAYOUT_ID = 'mechanical_same_side_left_right_sharpa_v1'
LAYOUT_IDS = (OPPOSED_LAYOUT_ID, SAME_SIDE_LAYOUT_ID, MIRRORED_LAYOUT_ID)
BASE_X_M = -.10
BASE_HALF_SPACING_M = .30
SIDES = ('front', 'rear')
MIRRORED_SIDES = ('left', 'right')


def placements(layout_id=SAME_SIDE_LAYOUT_ID):
    if layout_id == OPPOSED_LAYOUT_ID:
        from scenes.workrooms.living_room.dual_robot_layout import placements as opposed
        return opposed()
    if layout_id not in (SAME_SIDE_LAYOUT_ID, MIRRORED_LAYOUT_ID):
        raise ValueError('Unknown cooperative robot layout: '+str(layout_id))
    sides = (('left', 1), ('right', -1)) if layout_id == MIRRORED_LAYOUT_ID else (('front', -1), ('rear', 1))
    return {side: {
        'base_position_m': [BASE_X_M, sign*BASE_HALF_SPACING_M, .78],
        'base_quaternion_wxyz': [1., 0., 0., 0.],
        'base_yaw_rad': 0.,
        'ready_wrist_position_m': [BASE_X_M+.33, sign*BASE_HALF_SPACING_M, 1.40],
        'wrist_probe_rpy_rad': [*source_layout.READY_WRIST_RPY_RAD[:2],
            source_layout.READY_WRIST_RPY_RAD[2]*(-1 if side == 'left' else 1)],
    } for side, sign in sides}


def robot_configurations(layout_id=SAME_SIDE_LAYOUT_ID):
    if layout_id == MIRRORED_LAYOUT_ID:
        raise ValueError('The real left/right robot is one bimanual configuration; use robot_inputs')
    if layout_id == OPPOSED_LAYOUT_ID:
        from scenes.workrooms.living_room.dual_robot_layout import robot_configurations as opposed
        return opposed()
    source = source_layout.robot_configuration()
    result = {}
    for side, pose in placements(layout_id).items():
        robot = deepcopy(source)
        for key in ('base_position_m', 'base_quaternion_wxyz'):
            robot[key] = deepcopy(pose[key])
        result[side] = robot
    return result


def geometry_model(robot):
    """Use the original public FK/IK model; change neither controls nor meshes."""
    from bench.control.factory import public_robot_geometry, robot_configuration
    if (robot.get('hand_id') != 'sharpa_w01_right'
            or robot.get('assembly_version') != 'isaacsim_6_1_v2'):
        raise ValueError('Cooperative layout requires the original right-hand assembly')
    return public_robot_geometry(robot_configuration({'robot': robot}))


def link_world_transforms(robot, initial):
    model = geometry_model(robot)
    joints = model.checked_joints(initial['joint_positions_rad'], enforce_limits=True)
    poses = model.arm.forward(joints, model.arm_base)
    offset = np.eye(4) if model.wrist_offset is None else model.wrist_offset
    poses.update(model.hand.forward(joints, poses[model.arm_tip] @ offset))
    return poses


def robot_inputs(layout_id=OPPOSED_LAYOUT_ID):
    """Return frozen composite robot, initial state and ready provenance.

    The default is deliberately historical: callers selecting a new layout
    must say so explicitly, and old releases keep their exact robot identity.
    """
    if layout_id == OPPOSED_LAYOUT_ID:
        from scenes.workrooms.living_room.composition import _robots
        robot, initial, ready, _ = _robots('tower_extraction')
        return robot, initial, ready
    if layout_id == MIRRORED_LAYOUT_ID:
        return _mirrored_inputs()
    robots = robot_configurations(layout_id)
    source = source_layout.robot_configuration()
    source_initial, _ = source_layout.ready_configuration(source)
    robot = deepcopy(robots['front'])
    robot.update(hand_id='fr3_sharpa_instances', assembly_version='namespaced_v1',
                 base_position_m=[0., 0., 0.], base_quaternion_wxyz=[1., 0., 0., 0.])
    for key in ('arm_home_rad', 'mount_position_m', 'mount_quaternion_wxyz'):
        robot.pop(key, None)
    robot['instances'] = {side: {key: deepcopy(config[key]) for key in
        ('hand_id', 'assembly_version', 'base_position_m', 'base_quaternion_wxyz')}
        for side, config in sorted(robots.items())}
    robot['arm_home_rad_by_side'] = {side: deepcopy(config['arm_home_rad'])
                                    for side, config in robots.items()}
    initial, ready = {'joint_positions_rad': {}}, {}
    for side, config in sorted(robots.items()):
        model = geometry_model(config)
        poses = link_world_transforms(config, source_initial)
        offset = np.eye(4) if model.wrist_offset is None else model.wrist_offset
        wrist = poses[model.arm_tip] @ offset
        pose = placements(layout_id)[side]
        if not np.allclose(wrist[:3, 3], pose['ready_wrist_position_m'], atol=1e-4, rtol=0.):
            raise ValueError('Relocated original ready pose differs from declared wrist')
        for name, value in source_initial['joint_positions_rad'].items():
            if not name.startswith(('fr3_', 'right_')):
                raise ValueError('Unsupported source joint name: '+name)
            target = 'fr3_'+side+'_'+name[4:] if name.startswith('fr3_') else side+'_'+name[6:]
            initial['joint_positions_rad'][target] = value
        ready[side] = {
            'layout_id': layout_id, 'side': side, **deepcopy(pose),
            'wrist_fk_matrix': wrist.tolist(),
            'source_layout_id': source_layout.LAYOUT_ID,
            'source_joint_positions_preserved': True,
            'minimum_arm_joint_margin_rad': min(
                min(source_initial['joint_positions_rad'][joint.name]-joint.lower,
                    joint.upper-source_initial['joint_positions_rad'][joint.name])
                for joint in model.arm.active),
            'finger_posture': 'Original mechanical soft curl; waiting posture only',
            'native_qualified': False, 'manipulation_qualified': False,
            'scope': 'Task-owned same-side placement; independent original right-hand instances',
        }
    return robot, initial, ready


def _mirrored_inputs():
    """Use the existing dual assembly without changing native contact options.

    Both physical FR3s retain their source geometry and mount convention.
    ``left`` and ``right`` name hand chirality and robot-relative placement:
    looking forward along +X, +Y is left and -Y is right. The source ready
    right wrist remains unchanged and its mirrored left wrist faces inward.
    """
    from bench.control.factory import public_robot_geometry, robot_configuration as configuration
    from bench.simulation.isaacsim_backend.robot_registry import scene_with_hand

    source = source_layout.robot_configuration()
    source_initial, _ = source_layout.ready_configuration(source)
    robot = scene_with_hand({'robot': source}, 'dual_fr3_sharpa_w01',
                            'isaacsim_6_1_v2')['robot']
    robot.update(base_position_m=[BASE_X_M, 0., .78],
                 base_quaternion_wxyz=[1., 0., 0., 0.])
    robot['arm_base_poses']={side:{'position_m':[0.,sign*BASE_HALF_SPACING_M,0.],
        'quaternion_wxyz':[1.,0.,0.,0.]} for side,sign in (('left',1),('right',-1))}
    robot.pop('arm_home_rad', None)
    joints = {}
    for side in MIRRORED_SIDES:
        for name, value in source_initial['joint_positions_rad'].items():
            if name.startswith('fr3_'):
                target = 'fr3_'+side+'_'+name[4:]
                if side == 'left' and name in ('fr3_joint1', 'fr3_joint3', 'fr3_joint5', 'fr3_joint7'):
                    value = -value
            elif name.startswith('right_'):
                target = side+'_'+name[6:]
            else:
                raise ValueError('Unsupported source joint name: '+name)
            joints[target] = value
    robot['arm_home_rad_by_side'] = {
        side: [joints[f'fr3_{side}_joint{i}'] for i in range(1, 8)]
        for side in MIRRORED_SIDES}
    models = public_robot_geometry(configuration({'robot': robot})).sides
    ready = {}
    for side, model in models.items():
        own = model.checked_joints({name: joints[name] for name in model.names}, enforce_limits=True)
        if any(model.hand_posture_prior.violation(finger, own) is not None
               for finger in model.finger_joints):
            raise ValueError('Mirrored ready hand violates the source posture prior')
        wrist = model.arm.forward(own, model.arm_base)[model.arm_tip] @ model.wrist_offset
        placement = placements(MIRRORED_LAYOUT_ID)[side]
        if not np.allclose(wrist[:3, 3], placement['ready_wrist_position_m'], atol=1e-4, rtol=0.):
            raise ValueError('Mirrored original ready pose differs from declared wrist')
        ready[side] = {
            'layout_id': MIRRORED_LAYOUT_ID, 'side': side, **deepcopy(placement),
            'wrist_fk_matrix': wrist.tolist(), 'source_layout_id': source_layout.LAYOUT_ID,
            'source_joint_positions_preserved': side == 'right',
            'source_hand_chirality': side,
            'mirrored_arm_joint_indices': [1, 3, 5, 7] if side == 'left' else [],
            'source_finger_joint_values_preserved': True,
            'minimum_arm_joint_margin_rad': min(min(own[j.name]-j.lower, j.upper-own[j.name])
                                               for j in model.arm.active),
            'finger_posture': 'Original soft curl values in the official respective hand; waiting only',
            'native_qualified': False, 'manipulation_qualified': False,
            'scope': 'One Agent, two original FR3s with official left/right SHARPA, on the same table side',
        }
    return robot, {'joint_positions_rad': joints}, ready


def stand_geometry(layout_id=OPPOSED_LAYOUT_ID):
    """Original three-box stands with continuous floor-to-mount support."""
    if layout_id == OPPOSED_LAYOUT_ID:
        from scenes.workrooms.living_room.dual_robot_layout import stand_geometry as opposed
        result = opposed()
        # Mechanical floor is z=0; historical task authoring already replaced
        # the living-room floor/rug adjustment with these exact dimensions.
        for stand in result.values():
            for item in stand['boxes']:
                if item['name'] == 'robot_stand_foot':
                    item['position_m'][2] = .025
                elif item['name'] == 'robot_stand_column':
                    item['position_m'][2] = .4025
                    item['size_m'][2] = .705
            stand['ground_datum_m'] = 0.
        return result
    result = {}
    for side, pose in placements(layout_id).items():
        delta = np.asarray(pose['base_position_m'])-source_layout.ROBOT_BASE_POSITION_M
        stand = deepcopy(source_layout.stand_geometry())
        for box in stand['boxes']:
            box['position_m'] = (np.asarray(box['position_m'])+delta).tolist()
        stand.update(layout_id=layout_id, side=side, ground_datum_m=0.,
            base_center_to_table_edge_m=.15-BASE_X_M,
            foot_to_table_edge_m=.15-(BASE_X_M+.18),
            mount_to_table_edge_m=.15-(BASE_X_M+.14),
            stand_to_other_stand_foot_gap_m=2*BASE_HALF_SPACING_M-.40)
        result[side] = stand
    return result
