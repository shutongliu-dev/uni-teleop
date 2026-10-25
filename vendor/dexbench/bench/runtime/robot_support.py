"""Public robot facts and read-only IK over a published proprioception snapshot.

No task objects, evaluator state, simulator handle or controller lifecycle enter
this module. A numerical IK result is neither a motion plan nor a grasp claim.
"""
from __future__ import annotations

from bench.control.fingertip import ExecutorLimits, GeometryIKError
from bench.control.hand_presets import hand_preset_catalog
from bench.protocol.robot import IKQuery, IKResult, RobotGuide


def geometry_sides(geometry):
    return geometry.sides if hasattr(geometry, 'sides') else {geometry.embodiment.side: geometry}


def robot_guide(capabilities, geometry):
    """Project only documented robot configuration; never return source paths."""
    hands = {}
    if geometry is not None:
        for side, model in geometry_sides(geometry).items():
            hands[side] = {
                'frame_id': model.frame_id,
                'wrist_pose_frame': 'world',
                'fingertip_position_frame': model.frame_id,
                'mount_convention': model.embodiment.mount,
                'tip_semantics': model.embodiment.controller_tip_semantics,
                'fingers': {finger: {'joint_names': [joint.name for joint in chain],
                                    'tip_link': model.fingertip_links[finger]}
                            for finger, chain in model.finger_joints.items()},
                'posture_constraint': model.hand_posture_prior.describe(),
            }
    value = {
        'protocol': 'bench_robot_guide_v1',
        'robot_id': capabilities.robot_id,
        'robot_asset_sha256': capabilities.robot_asset_sha256,
        'units': {'position': 'm', 'angle': 'rad', 'time': 'simulation_seconds'},
        'quaternion_order': 'wxyz',
        'camera_frame': 'optical_x_right_y_down_z_forward',
        'frame_ids': list(capabilities.frame_ids),
        'arm_joints': list(capabilities.arm_joints),
        'hand_joints': list(capabilities.hand_joints),
        'joint_limits': {name: limit.model_dump(mode='json', exclude_none=True)
                         for name, limit in capabilities.joint_limits.items()},
        'hands': hands,
        'control_period_s': capabilities.control_period_s,
        'direct_joint_execution': capabilities.direct_joint_execution,
        'executor': capabilities.system0.model_dump(mode='json') if capabilities.system0 else None,
        'ik_query': {'available': geometry is not None,
                     'endpoint': 'POST /episodes/{episode_id}/robot/ik',
                     'input_source': 'published_joint_proprioception',
                     'advances_physics': False, 'mutates_controller': False,
                     'collision_planning': False, 'duration_preflight': False},
        'hand_presets': hand_preset_catalog(geometry).model_dump(mode='json'),
        'contact_feedback': {
            'tactile_available': capabilities.tactile_available,
            'mode': capabilities.tactile_mode,
            'grasp_detected_available': False,
            'interpretation': 'Joint closure and robot contact do not establish object capture or load-bearing grasp.'},
    }
    return RobotGuide.model_validate(value).model_dump(mode='json', exclude_unset=True)


def query_ik(geometry, joint_positions, query, *, limits=None):
    """Use the executor's bounded solver, without touching its motion state.

Omitted joints use the measured reference, not the active executor's persistent
goal. Execution still validates its own observation, trajectory and duration.
    """
    query = IKQuery.model_validate(query.model_dump(mode='python') if isinstance(query, IKQuery) else query)
    models = geometry_sides(geometry)
    if hasattr(geometry, 'sides'):
        if query.geometry is not None or not query.hand_geometries:
            raise ValueError('Bimanual IK requires hand_geometries')
        targets = query.hand_geometries
    else:
        if query.geometry is None or query.hand_geometries is not None:
            raise ValueError('Single-arm IK requires geometry')
        targets = {next(iter(models)): query.geometry}
    if not set(targets) <= set(models):
        raise ValueError("Hand namespace is not present in this robot")
    limits = limits or ExecutorLimits()
    for side, target in targets.items():
        if target.fingertip_frame_id != models[side].frame_id:
            raise ValueError('Fingertip frame does not match the selected hand')
    solutions = {}
    for side, target in targets.items():
        model = models[side]
        q = {name: joint_positions[name] for name in model.names}
        try:
            goal, diagnostics = model.solve(q, wrist=target.wrist,
                fingertip_positions_m=target.fingertip_positions_m,
                hand_posture_rad=target.hand_posture_rad,
                position_tolerance_m=min(target.position_tolerance_m, limits.wrist_position_tolerance_m) * limits.ik_verification_scale,
                rotation_tolerance_rad=min(target.orientation_tolerance_rad, limits.wrist_rotation_tolerance_rad) * limits.ik_verification_scale,
                fingertip_tolerance_m=limits.fingertip_tolerance_m * limits.ik_verification_scale,
                fingertip_acceptance_m=limits.fingertip_ik_acceptance_m)
        except GeometryIKError as error:
            solutions[side] = {'solution_found': False,
                'diagnostic': error.public_diagnostic.model_dump(mode='json', exclude_none=True)}
        else:
            solutions[side] = {'solution_found': True, 'joint_positions_rad': goal,
                'verified_geometry': model.public_finger_geometry(goal),
                'residuals': diagnostics}
    value = {'protocol': 'bench_ik_query_v1',
            'observation_sequence': query.observation_sequence,
            'solution_found': all(item['solution_found'] for item in solutions.values()),
            'hands': solutions, 'advances_physics': False, 'mutates_controller': False,
            'collision_checked': False, 'trajectory_checked': False,
            'scope': 'Bounded numerical IK from the published measured joints; absence of a solution is not proof of geometric impossibility. Omitted joints use measured references; execution performs its own stateful planning.'}
    return IKResult.model_validate(value).model_dump(mode='json', exclude_unset=True)
