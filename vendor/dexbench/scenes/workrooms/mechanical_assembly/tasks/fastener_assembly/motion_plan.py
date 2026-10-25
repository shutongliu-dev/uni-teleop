"""CPU-designed joint-target diagnostic for the independent v3 screwdriver.

This is an engineering candidate, not a runnable task qualification.  It emits
only robot joint references; the caller owns reset, physics, contact guards and
the C2 interpolation specified below.  The independent tool is never attached
to the hand and the default diagnostic does not turn the screw.
"""
from __future__ import annotations

import math

import numpy as np

from bench.geometry import quaternion_matrix, transform
from scenes.workrooms.mechanical_assembly.robot_layout import (
    LAYOUT_ID, READY_FINGER_POSTURE_RAD, geometry_model, ready_configuration, robot_configuration,
)


TOOL_TIP_POSITION_M = (.38000191, .0000045621773, .88113297)
TOOL_QUATERNION_XYZW = (-3.6826583e-5, -5.8749396e-5, -.98082989, .19486582)
HANDLE_CENTER_LOCAL_M = (0., 0., .156)
HANDLE_RADIUS_M = .022
HANDLE_LENGTH_M = .110
SIDE_APPROACH_YAW_RAD = 3*math.pi/4
INDEX_PRELOAD_INCREMENT_RAD = .02


def opposing_pinch_candidate():
    """Source-mesh candidate; the small final pad overlap is a drive target only.

    The intermediate thumb-base waypoint matters: direct interpolation from
    open to preclose sweeps the thumb DP shell through the handle.  A previous
    native diagnostic reached its index target but did not maintain opposed
    pressure.  This revision adds only 0.02 rad at index MCP flexion;
    source-cylinder target overlaps are about 0.4 mm thumb / 1.64 mm index.
    Contact must arrest the commanded overclosure.  It is not permission for
    actual penetration, and no body state or physics parameter is changed.
    """
    opened = dict(READY_FINGER_POSTURE_RAD)
    for finger in ('index', 'middle', 'ring', 'pinky'):
        opened.update({f'right_{finger}_{joint}': value for joint, value in
                       (('MCP_FE', .05), ('MCP_AA', 0.), ('PIP', .15), ('DIP', .10))})
    thumb_names = ['right_thumb_'+j for j in ('CMC_FE', 'CMC_AA', 'MCP_FE', 'MCP_AA', 'IP')]
    index_names = ['right_index_'+j for j in ('MCP_FE', 'MCP_AA', 'PIP', 'DIP')]
    preclose = dict(opened)
    preclose.update(zip(thumb_names, (1.0562555448683666, -.2192989270721672,
        .7514473922660801, .09465646442734721, .42013869385818237)))
    preclose.update(zip(index_names, (.20979182706158503, .0045033755863849115,
        .7074808701906168, .8042566540797785)))
    thumb_base = dict(opened)
    thumb_base.update({n: preclose[n] for n in thumb_names[:2]})
    closed = dict(opened)
    closed.update(zip(thumb_names, (1.077166308013878, -.20329423469541918,
        .7425414606563305, .09689366968192807, .3971994206310929)))
    closed.update(zip(index_names, (.21952777326961478, .004547255327636983,
        .7146835668001622, .8071647832132824)))
    closed['right_index_MCP_FE'] += INDEX_PRELOAD_INCREMENT_RAD
    return {'revision': 'opposed_pinch_index_preload_v2',
            'index_preload_increment_rad': INDEX_PRELOAD_INCREMENT_RAD,
            'axis_in_hand_xz_m': [.059555948604, .159473118835],
            'open_hand_rad': opened, 'thumb_base_hand_rad': thumb_base,
            'preclose_hand_rad': preclose, 'closed_hand_rad': closed,
            'geometry': '44 mm diameter, 110 mm long finite cylindrical handle along H+y',
            'preclose_pad_radial_gap_m': {'thumb': .000600289, 'index': .000600007},
            'preclose_pad_radial_direction_dot': -.777085,
            'closed_pad_target_radial_overlap_m': {'thumb': .000400535, 'index': .001639743},
            'minimum_nonpad_shell_gap_in_reviewed_finger_segments_m': .000699999,
            'finger_segment_samples': 21,
            'scope': 'Discrete original-source collision triangle checks against an ideal cylinder; '
                     'thumb/index separation certificates only. The v3 handle source hull has '
                     '64 radial sides; neither source overlap nor a contact manifold proves '
                     'native positive pressure. Native grasp remains unqualified.'}


def c2_fraction(fraction):
    """Zero endpoint velocity/acceleration; peak derivative is exactly 1.875."""
    u = float(fraction)
    if not math.isfinite(u) or not 0. <= u <= 1.:
        raise ValueError('Interpolation fraction must be finite and in [0, 1]')
    return u*u*u*(10.+u*(-15.+6.*u))


def _side_rotation(tool_rotation):
    # H+y follows the actual handle axis, including the released seed's tilt.
    y = tool_rotation[:, 2]
    x = np.array([math.cos(SIDE_APPROACH_YAW_RAD), math.sin(SIDE_APPROACH_YAW_RAD), 0.])
    x -= np.dot(x, y)*y
    x /= np.linalg.norm(x)
    return np.column_stack((x, y, np.cross(x, y)))


def _solve(model, point, rotation, seed):
    offset = np.eye(4) if model.wrist_offset is None else model.wrist_offset
    target = np.linalg.inv(model.arm_base) @ transform(point, rotation) @ np.linalg.inv(offset)
    result = model.arm.inverse(target, model.arm_tip, seed, iterations=240,
        multi_starts=0, position_tolerance=.00005, rotation_tolerance=.001)
    if not result['converged']:
        raise ValueError('Tool diagnostic wrist IK did not converge')
    q = result['joint_positions_rad']
    margin = min(min(q[j.name]-j.lower, j.upper-q[j.name]) for j in model.arm.active)
    if margin < .20:
        raise ValueError('Tool diagnostic wrist has insufficient arm joint margin')
    return q, {'position_m': list(map(float, point)),
               'position_error_m': float(result['position_error_m']),
               'rotation_error_rad': float(result['rotation_error_rad']),
               'minimum_arm_joint_margin_rad': margin}


def candidate_motion_plan(robot=None, *, tool_tip_position_m=TOOL_TIP_POSITION_M,
                          tool_quaternion_xyzw=TOOL_QUATERNION_XYZW,
                          grasp_candidate=None):
    """Build full-29q phases, retaining the common base and reviewed ready pose.

    ``grasp_candidate`` defaults to the reviewed opposing thumb/index candidate.
    Each of its four hand waypoint maps must contain all 22 hand joints.
    A later native contact gate is mandatory before adding any turning phase.
    """
    if grasp_candidate is None:
        grasp_candidate = opposing_pinch_candidate()
    robot = robot_configuration() if robot is None else robot
    model = geometry_model(robot)
    initial, ready = ready_configuration(robot)
    ready_q = initial['joint_positions_rad']
    tip = np.asarray(tool_tip_position_m, dtype=float)
    xyzw = np.asarray(tool_quaternion_xyzw, dtype=float)
    if tip.shape != (3,) or xyzw.shape != (4,) or not np.isfinite(np.r_[tip, xyzw]).all():
        raise ValueError('Tool pose requires finite 3D position and XYZW quaternion')
    if abs(float(np.linalg.norm(xyzw))-1.) > 1e-5:
        raise ValueError('Tool quaternion must be unit XYZW')
    tool_rotation = quaternion_matrix(xyzw[[3, 0, 1, 2]])
    center = tip+tool_rotation @ np.asarray(HANDLE_CENTER_LOCAL_M)
    rotation = _side_rotation(tool_rotation)
    axis = np.asarray(grasp_candidate['axis_in_hand_xz_m'], dtype=float)
    if axis.shape != (2,) or not np.isfinite(axis).all():
        raise ValueError('Grasp axis must supply finite H x,z coordinates')
    hand_names = set(model.names)-set(model.arm.names)
    for key in ('open_hand_rad', 'thumb_base_hand_rad', 'preclose_hand_rad', 'closed_hand_rad'):
        if set(grasp_candidate[key]) != hand_names:
            raise ValueError('Grasp candidate requires exactly 22 hand joint targets')
    opened, closed = grasp_candidate['open_hand_rad'], grasp_candidate['closed_hand_rad']
    thumb_base, preclosed = grasp_candidate['thumb_base_hand_rad'], grasp_candidate['preclose_hand_rad']
    wrist = center-rotation @ np.array([axis[0], 0., axis[1]])
    preapproach = wrist-.08*rotation[:, 0]
    high = preapproach+np.array([0., 0., .22])
    phases, endpoints = [], {}
    previous = ready_q

    def add(name, joints, seconds, *, contact_stage=False):
        nonlocal previous
        q = model.checked_joints(joints, enforce_limits=True)
        arm_delta = max(abs(q[n]-previous[n]) for n in model.arm.names)
        hand_delta = max(abs(q[n]-previous[n]) for n in hand_names)
        # Durations bound peak C2 speed independently of the executor's rate.
        duration = max(float(seconds), 1.875*arm_delta/1.2, 1.875*hand_delta/1.2)
        duration = .02*math.ceil(duration/.02-1e-12)
        if duration > 10.:
            raise ValueError('Candidate phase exceeds the native diagnostic 10-second bound')
        phases.append({'name': name, 'joint_positions_rad': q, 'duration_s': duration,
                       'expected_tool_contact': contact_stage,
                       'maximum_arm_target_speed_rad_s': 1.875*arm_delta/duration,
                       'maximum_hand_target_speed_rad_s': 1.875*hand_delta/duration})
        previous = q

    add('ready_hold', ready_q, .2)
    arm, endpoints['high_approach'] = _solve(model, high, rotation, previous)
    high_q = {**arm, **opened}
    add('high_approach', high_q, 2.76)
    arm, endpoints['preapproach'] = _solve(model, preapproach, rotation, previous)
    pre_q = {**arm, **opened}
    add('preapproach', pre_q, .6)
    arm, endpoints['approach'] = _solve(model, wrist, rotation, previous)
    approach_q = {**arm, **opened}
    add('approach', approach_q, .8)
    add('thumb_base_prepare', {**arm, **thumb_base}, 1.12)
    add('preclose', {**arm, **preclosed}, 1.12)
    add('close', {**arm, **closed}, 1., contact_stage=True)
    add('hold_grasp', previous, 1., contact_stage=True)
    add('release_preclose', {**arm, **preclosed}, 1., contact_stage=True)
    add('open_thumb_fingers', {**arm, **thumb_base}, 1.12, contact_stage=True)
    add('open', approach_q, 1.12, contact_stage=True)
    add('retreat', pre_q, .8)
    add('lift_clear', high_q, .6)
    add('return_ready', ready_q, 2.76)
    add('final_hold', ready_q, .2)
    return {
        'schema': 'mechanical_tool_motion_candidate.v1', 'layout_id': LAYOUT_ID,
        'phases': phases, 'initial_joint_positions_rad': ready_q,
        'interpolation': 'Each previous target to target: 10u^3-15u^4+6u^5, u=t/duration.',
        'maximum_arm_target_speed_rad_s': 1.2, 'maximum_hand_target_speed_rad_s': 1.2,
        'total_duration_s': sum(p['duration_s'] for p in phases),
        'tool_asset_id': 'threaded_sharpa/v3',
        'tool_tip_position_m': tip.tolist(), 'tool_quaternion_xyzw': xyzw.tolist(),
        'handle_center_world_m': center.tolist(),
        'handle_radius_m': HANDLE_RADIUS_M, 'handle_length_m': HANDLE_LENGTH_M,
        'grasp_candidate': grasp_candidate, 'wrist_endpoints': endpoints,
        'ready_wrist_target': ready['wrist_fk'],
        'expected_contact_link_suffixes': [
            'right_thumb_elastomer', 'right_thumb_DP',
            'right_index_elastomer', 'right_index_DP'],
        'required_native_grasp_gate': {
            'minimum_opposed_contact_duration_s': .3,
            'maximum_radial_contact_direction_dot': -.3,
            'tool_contacts_only_on_handle': True,
        },
        'contains_turn': False, 'native_qualified': False, 'manipulation_qualified': False,
        'scope': 'Privileged engineering joint-target diagnostic. Requires independent source-mesh path review, '
                 'native per-substep contacts, held-tool stability and return checks. '
                 'No tool weld, object playback, external force or screw-turning claim.',
    }
