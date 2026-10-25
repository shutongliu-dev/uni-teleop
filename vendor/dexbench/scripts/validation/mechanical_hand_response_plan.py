"""CPU plan for a privileged hand-to-tool response experiment.

Only the robot reset and subsequent joint references are returned. The caller
keeps the release scene, drives and independent tool/bolt state unchanged.
This bypasses approach and task scoring; it is neither an Agent trajectory nor
a grasp qualification. A failed grasp must remain observable, not skip turning.
"""
from copy import deepcopy
import hashlib
import json
import math

import numpy as np

from bench.control.fingertip import ExecutorLimits
from bench.geometry import quaternion_matrix
from scenes.workrooms.mechanical_assembly.robot_layout import geometry_model
from scenes.workrooms.mechanical_assembly.tasks.fastener_assembly.balanced_motion_plan import balanced_grasp_candidate
from scenes.workrooms.mechanical_assembly.tasks.fastener_assembly.motion_plan import candidate_motion_plan, _solve


CONTROL_PERIOD_S = .02


def _identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def build_hand_response_plan(scene, initial_state, *, turn_degrees=6., tool_pose_xyzw=None):
    """Return ``initial_state`` and full-29-joint ``phases`` for native stepping.

    Each phase interpolates from the preceding commanded target with
    ``10*u**3 - 15*u**4 + 6*u**5`` over its entire ``duration_s``. Initial robot
    joints are the existing balanced candidate's preclose posture. An optional
    measured tool pose supplies only the geometric planning reference; the
    function never modifies the scene or any body pose. The executor must
    record its provenance and must not reset bodies again during the sequence.
    """
    scene_hash, initial_hash = _identity(scene), _identity(initial_state)
    if (type(turn_degrees) not in (int, float) or not math.isfinite(turn_degrees)
            or not 0. < turn_degrees <= 6.):
        raise ValueError('Hand response uses a nonzero turn of at most six degrees')
    if set(initial_state) != {'joint_positions_rad'}:
        raise ValueError('Expected a robot-joint-only frozen initial state')
    objects = scene['objects']
    names = [row['id'] for row in objects]
    if len(names) != len(set(names)) or not {'screwdriver', 'bolt'} <= set(names):
        raise ValueError('Distinct independent screwdriver and bolt bodies are required')
    tool = next(row for row in objects if row['id'] == 'screwdriver')
    if tool_pose_xyzw is None:
        q = tool['quaternion_wxyz']
        pose = np.array([*tool['position_m'], q[1], q[2], q[3], q[0]], dtype=float)
        axis_source = 'frozen_scene_tool_pose_not_native_readback'
    else:
        pose = np.asarray(tool_pose_xyzw, dtype=float)
        axis_source = 'caller_supplied_tool_pose_requires_external_readback_provenance'
    if (pose.shape != (7,) or not np.isfinite(pose).all()
            or abs(np.linalg.norm(pose[3:])-1.) > 1.e-5):
        raise ValueError('Tool planning reference must be finite XYZ and unit XYZW')
    dt = scene['isaacsim']['physics_dt_s']
    if (type(dt) not in (int, float) or not math.isfinite(dt) or dt <= 0.
            or abs(CONTROL_PERIOD_S/dt-round(CONTROL_PERIOD_S/dt)) > 1.e-8):
        raise ValueError('Native timestep must divide the 20 ms command period')
    model = geometry_model(scene['robot'])
    original = model.checked_joints(initial_state['joint_positions_rad'], enforce_limits=True)
    candidate = candidate_motion_plan(scene['robot'], tool_tip_position_m=pose[:3],
        tool_quaternion_xyzw=pose[3:], grasp_candidate=balanced_grasp_candidate())
    waypoints = {row['name']: row['joint_positions_rad'] for row in candidate['phases']}
    preclose = model.checked_joints(waypoints['preclose'], enforce_limits=True)
    closed = model.checked_joints(waypoints['close'], enforce_limits=True)
    limits = ExecutorLimits()
    velocity = {j.name: min(j.velocity, limits.arm_velocity_rad_s if j.name in model.arm.names
                           else limits.hand_velocity_rad_s) for j in model.joints}
    if any(not math.isfinite(v) or v <= 0. for v in velocity.values()):
        raise ValueError('Each original robot joint requires a positive velocity bound')
    joint_limits = {j.name: {'lower_rad': j.lower, 'upper_rad': j.upper,
        'source_max_velocity_rad_s': j.velocity, 'reference_speed_limit_rad_s': velocity[j.name]}
        for j in model.joints}
    phases, previous = [], preclose

    def add(name, target, minimum_s, **evidence):
        nonlocal previous
        target = model.checked_joints(target, enforce_limits=True)
        minimum_s = max(minimum_s, *(1.875*abs(target[n]-previous[n])/velocity[n] for n in model.names))
        duration = CONTROL_PERIOD_S*math.ceil(minimum_s/CONTROL_PERIOD_S-1.e-12)
        if duration > 10.:
            raise ValueError('Engineering motion exceeds the bounded phase duration')
        phases.append({'name': name, 'duration_s': duration,
            'joint_positions_rad': target,
            'maximum_reference_velocity_rad_s': {n: 1.875*abs(target[n]-previous[n])/duration for n in model.names},
            **evidence})
        previous = target

    add('settle', preclose, .3)
    add('close', closed, .8)
    if phases[-1]['duration_s'] > 1.:
        raise ValueError('Original hand velocity limits cannot support the selected one-second close bound')
    add('hold_grasp', closed, .3)
    wrist = model.public_geometry(closed)['wrist']
    wrist_point = np.asarray(wrist['position_m'])
    wrist_rotation = quaternion_matrix(wrist['quaternion_wxyz'])
    tool_rotation = quaternion_matrix(pose[[6, 3, 4, 5]])
    axis, tip = tool_rotation[:, 2], pose[:3]
    skew = np.array([[0., -axis[2], axis[1]], [axis[2], 0., -axis[0]], [-axis[1], axis[0], 0.]])
    # Joint interpolation is not an exact Cartesian orbit. One-degree targets
    # reduce its intermediate axial bow without adding screw-pitch advance.
    count = math.ceil(turn_degrees)
    turn_phase_seconds = .5*(turn_degrees/count)/3.
    turn_targets = [closed]
    for index in range(1, count+1):
        theta = -math.radians(turn_degrees)*index/count
        rotate = np.eye(3)+math.sin(theta)*skew+(1.-math.cos(theta))*(skew @ skew)
        point = tip+rotate @ (wrist_point-tip)
        arm, ik = _solve(model, point, rotate @ wrist_rotation, previous)
        target = {**closed, **arm}
        add(f'turn_{index:02d}', target, turn_phase_seconds, nominal_tool_axis_angle_rad=theta, ik=ik,
            target_wrist_position_m=point.tolist(), target_wrist_rotation=(rotate @ wrist_rotation).tolist())
        turn_targets.append(target)
    add('hold_after_turn', previous, .3)
    for index, target in enumerate(reversed(turn_targets[:-1]), 1):
        theta = -math.radians(turn_degrees)*(count-index)/count
        add(f'return_{index:02d}', target, turn_phase_seconds, nominal_tool_axis_angle_rad=theta)
    add('release_preclose', preclose, .8)
    add('open_thumb_fingers', waypoints['open_thumb_fingers'], .8)
    add('open', waypoints['open'], .8)
    add('final_hold', previous, .3)
    reset_wrist = model.public_geometry(preclose)['wrist']
    reset_rotation = quaternion_matrix(reset_wrist['quaternion_wxyz'])
    offset = np.array([candidate['grasp_candidate']['axis_in_hand_xz_m'][0], 0.,
                       candidate['grasp_candidate']['axis_in_hand_xz_m'][1]])
    center_error = float(np.linalg.norm(np.asarray(reset_wrist['position_m'])+reset_rotation @ offset
                                       -candidate['handle_center_world_m']))
    alignment_error = float(np.linalg.norm(reset_rotation[:, 1]-axis))
    if center_error > .00006 or alignment_error > .001:
        raise ValueError('Preclose wrist does not match the selected tool axis and handle')
    assert _identity(scene) == scene_hash and _identity(initial_state) == initial_hash
    return {'schema': 'mechanical_privileged_hand_response_plan.v1',
        'initial_state': {'joint_positions_rad': deepcopy(preclose)},
        'original_initial_state': deepcopy(initial_state), 'phases': phases,
        'control_period_s': CONTROL_PERIOD_S, 'physics_dt_s': dt,
        'total_duration_s': sum(row['duration_s'] for row in phases),
        'interpolation': 'C2 minimum jerk over each whole phase: 10*u^3-15*u^4+6*u^5.',
        'tool_pose_xyzw': pose.tolist(), 'tool_axis_world': axis.tolist(), 'axis_source': axis_source,
        'turn_degrees': float(turn_degrees), 'turn_direction': 'clockwise_about_tool_local_positive_z',
        'maximum_turn_waypoint_increment_degrees': turn_degrees/count,
        'commanded_axial_advance_m': 0., 'grasp_gate_blocks_motion': False,
        'preclose_geometry': {'handle_center_world_m': candidate['handle_center_world_m'],
            'wrist': reset_wrist, 'handle_center_error_m': center_error,
            'axis_direction_error': alignment_error, 'ik': candidate['wrist_endpoints']['approach']},
        'grasp_candidate': candidate['grasp_candidate'], 'joint_limits': joint_limits,
        'all_target_joint_limits_checked': True,
        'original_reset_joint_limits_checked': original == initial_state['joint_positions_rad'],
        'source_scene_sha256': scene_hash, 'source_initial_state_sha256': initial_hash,
        'robot_geometry_sources_sha256': deepcopy(model.asset_sha256),
        'privileged_reset_only': True, 'agent_calls': 0, 'public_functions_used': False,
        'object_external_wrenches': False, 'object_attachments': False,
        'object_state_writes_after_reset': False, 'native_qualified': False,
        'scope': 'Robot-only engineered reset at preclose, close/hold, bounded six-degree turn and reverse, '
                 'open/hold. Original tool/bolt poses, collision geometry, materials and drive gains stay unchanged. '
                 'Cartesian wrist targets keep constant axial height; joint interpolation and IK have numerical '
                 'path deviations. Reference speed and CPU IK do not establish physical speed, clearance, grip '
                 'or task success.'}
