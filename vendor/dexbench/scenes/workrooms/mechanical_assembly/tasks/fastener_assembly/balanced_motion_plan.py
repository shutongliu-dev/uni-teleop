"""Side-balanced source-geometry candidate after the native03 grip failure.

This is task-owned robot motion, not a change to physics or a qualified grasp.
The original four-finger candidate remains available in public_motion_plan.
"""
from __future__ import annotations

from copy import deepcopy

from .motion_plan import candidate_motion_plan
from .public_motion_plan import four_finger_candidate, _capability_values, _retime

_AXIS_IN_HAND_XZ_M = (0.06694558259288776, 0.162537969485566)

_PRECLOSE_RAD = {'right_thumb_CMC_FE': 1.1718199792250428,
 'right_thumb_CMC_AA': -0.18937333990618913,
 'right_thumb_MCP_FE': 0.652705162302515,
 'right_thumb_MCP_AA': 0.07540110924914044,
 'right_thumb_IP': 0.33054990518960214,
 'right_index_MCP_FE': 0.1001033816090613,
 'right_index_MCP_AA': 0.004999999997886636,
 'right_index_PIP': 0.8202544897589384,
 'right_index_DIP': 0.9316490439143543,
 'right_middle_MCP_FE': 0.1097117675619611,
 'right_middle_MCP_AA': 7.996388526727931e-07,
 'right_middle_PIP': 0.8588689466741698,
 'right_middle_DIP': 0.9768202468558865,
 'right_ring_MCP_FE': 0.10873148145122387,
 'right_ring_MCP_AA': 1.9687776885839357e-07,
 'right_ring_PIP': 0.7576002944024873,
 'right_ring_DIP': 0.8938231123916807}

_CLOSED_RAD = {'right_thumb_CMC_FE': 1.258260640343353,
 'right_thumb_CMC_AA': -0.13299474774423015,
 'right_thumb_MCP_FE': 0.6043181314616901,
 'right_thumb_MCP_AA': 0.05407291253755609,
 'right_thumb_IP': 0.2472609175433923,
 'right_index_MCP_FE': 0.1159014952632311,
 'right_index_MCP_AA': 0.004999999999949022,
 'right_index_PIP': 0.8395289470320151,
 'right_index_DIP': 0.9419570224391308,
 'right_middle_MCP_FE': 0.12807664209879693,
 'right_middle_MCP_AA': 6.705251714209863e-07,
 'right_middle_PIP': 0.8757258965106839,
 'right_middle_DIP': 0.9844167156027728,
 'right_ring_MCP_FE': 0.13288736171972423,
 'right_ring_MCP_AA': 2.208899151993775e-05,
 'right_ring_PIP': 0.7623129694623568,
 'right_ring_DIP': 0.9026718863548728}

def balanced_grasp_candidate():
    """Keep the same preload scale while moving contact directions apart.

    Relative to the nominal released tool frame, the hand is shifted +8 mm
    along tool local Y. Finger lateral spread stays within 0.005 rad; the
    earlier unconstrained search caused index/middle shell overlap and was
    rejected. Reoptimized flexion keeps analytic pad targets around 1.4 to
    1.5 mm inside the ideal cylinder; this is neither a native penetration
    allowance nor proof of positive pressure. Nearest source-vertex radial directions
    are proxies for contact geometry, not measured native contact normals.
    """
    previous = four_finger_candidate()
    opened = previous['open_hand_rad']
    preclosed = {**opened, **_PRECLOSE_RAD}
    closed = {**opened, **_CLOSED_RAD}
    thumb_base = dict(opened)
    for name in ('right_thumb_CMC_FE', 'right_thumb_CMC_AA'):
        thumb_base[name] = preclosed[name]
    return {
        'revision': 'side_balanced_four_finger_source_v1',
        'axis_in_hand_xz_m': list(_AXIS_IN_HAND_XZ_M),
        'nominal_hand_shift_tool_local_y_m': .008,
        'open_hand_rad': opened, 'thumb_base_hand_rad': thumb_base,
        'preclose_hand_rad': preclosed, 'closed_hand_rad': closed,
        'intended_contact_fingers': ['thumb', 'index', 'middle', 'ring'],
        'preclose_ideal_pad_gap_m': .0007,
        'closed_ideal_pad_target_overlap_m': .0015,
        'native_penetration_limit_m': previous['native_penetration_limit_m'],
        'native_qualified': False,
        'scope': 'Side-balanced source-geometry candidate with unchanged assets, '
                 'materials, gains, joint limits and task rules. Full source '
                 'path checks are discrete; native opposed pressure, stability '
                 'and the 30-degree screw task remain unqualified.',
    }


def candidate_balanced_motion_plan(capabilities, robot=None, **tool_pose):
    """Full original-ready approach/grasp/release, retimed for public actions.

    Uses the existing public 80% ramp calculation. Nominal move_joints phases
    retain at least 0.20 s at the endpoint to observe tracking before handoff.
    Actual dispatch must still retime from the acknowledged targets and check
    feedback; this plan does not execute or add any tool motion.
    """
    plan = candidate_motion_plan(robot, grasp_candidate=balanced_grasp_candidate(), **tool_pose)
    previous = plan['initial_joint_positions_rad']
    values, _, _, dt = _capability_values(capabilities)
    for phase in plan['phases']:
        phase['arm_speed_cap_rad_s'] = .65
        phase['hand_speed_cap_rad_s'] = .12 if phase['name'] in ('close', 'release_preclose') else 1.
        if phase['name'] in ('hold_grasp', 'final_hold', 'ready_hold'):
            phase['public_function_hint'] = 'hold'
        elif phase['name'] in ('thumb_base_prepare', 'preclose', 'close',
                               'release_preclose', 'open_thumb_fingers', 'open'):
            phase['public_function_hint'] = 'set_hand_targets'
        else:
            phase['public_function_hint'] = 'move_joints'
        duration, minimum, arms, hands, ramp_duration = _retime(phase, previous, values)
        if phase['public_function_hint'] == 'move_joints':
            while duration-ramp_duration < .20-1.e-12:
                phase['duration_s'] = duration+dt
                duration, minimum, arms, hands, ramp_duration = _retime(phase, previous, values)
            phase['minimum_nominal_direct_endpoint_hold_s'] = .20
        phase.update(duration_s=duration, minimum_public_duration_s=minimum,
            direct_ramp_duration_s=ramp_duration,
            direct_endpoint_hold_duration_s=duration-ramp_duration,
            maximum_arm_target_speed_rad_s=max(1.875*abs(phase['joint_positions_rad'][n]-previous[n])/ramp_duration for n in arms),
            maximum_hand_target_speed_rad_s=max(1.875*abs(phase['joint_positions_rad'][n]-previous[n])/ramp_duration for n in hands))
        previous = phase['joint_positions_rad']
    plan.update(schema='mechanical_public_tool_motion_candidate.v2',
        maximum_arm_target_speed_rad_s=.65, maximum_hand_target_speed_rad_s=1.,
        total_duration_s=sum(p['duration_s'] for p in plan['phases']),
        public_execution='move_joints uses actual acknowledged targets and the existing '
                         'min_jerk_velocity_preflight_v2 80% ramp contract',
        direct_ramp_fraction=.8, public_capabilities=deepcopy(values),
        expected_contact_link_suffixes=[f'right_{finger}_{part}'
            for finger in ('thumb', 'index', 'middle', 'ring') for part in ('elastomer', 'DP')],
        scope='Task-owned side-balanced engineering candidate. The original ready, '
              'robot base, task reset and all physics are unchanged. Source geometry '
              'and CPU IK do not qualify cooked collision shapes, opposed pressure, '
              'feedback tracking or screw turning. Contains no object commands.')
    return plan
