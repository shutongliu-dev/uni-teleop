"""Task-owned four-finger candidate for ordinary public bench actions.

Source collision geometry informed these targets; they are not a generic hand
preset or native grasp qualification. Physics, source assets, the robot base,
ready pose and preinserted tool reset are unchanged. No object state is written.
"""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import math

from bench.protocol.functions import MoveJointsArguments
from .motion_plan import candidate_motion_plan, opposing_pinch_candidate


# Timing of the advertised min_jerk_velocity_preflight_v2 executor. Its ramp
# occupies ceil(80% of action ticks), extended by the public speed minimum;
# remaining ticks hold the endpoint. Runtime-contract tests pin this behavior.
_DIRECT_RAMP_FRACTION = .8


_PRECLOSE_RAD = {'right_thumb_CMC_FE': 1.0563284782268856,
 'right_thumb_CMC_AA': -0.22002626537506456,
 'right_thumb_MCP_FE': 0.7512117548757898,
 'right_thumb_MCP_AA': 0.09433435036606638,
 'right_thumb_IP': 0.42006103351626645,
 'right_index_MCP_FE': 0.20881805490340183,
 'right_index_MCP_AA': 0.004499065273336026,
 'right_index_PIP': 0.7067661428883508,
 'right_index_DIP': 0.8039684999641517,
 'right_middle_MCP_FE': 0.17628696535680022,
 'right_middle_MCP_AA': 4.106821275326458e-09,
 'right_middle_PIP': 0.7829422070073412,
 'right_middle_DIP': 0.8935831892898215,
 'right_ring_MCP_FE': 0.11639172844863276,
 'right_ring_MCP_AA': 1.7070033411251497e-08,
 'right_ring_PIP': 0.7411420502592434,
 'right_ring_DIP': 0.8775588421714013}

_CLOSED_RAD = {'right_thumb_CMC_FE': 1.1466019815807418,
 'right_thumb_CMC_AA': -0.14816078566143745,
 'right_thumb_MCP_FE': 0.66903070040104,
 'right_thumb_MCP_AA': 0.08151339805547,
 'right_thumb_IP': 0.3237584480316405,
 'right_index_MCP_FE': 0.23021991903573746,
 'right_index_MCP_AA': 0.004597442594665346,
 'right_index_PIP': 0.7227419696710564,
 'right_index_DIP': 0.8104337058580667,
 'right_middle_MCP_FE': 0.19766372473715604,
 'right_middle_MCP_AA': 3.9684547665252787e-10,
 'right_middle_PIP': 0.7983075426863561,
 'right_middle_DIP': 0.8993673926295945,
 'right_ring_MCP_FE': 0.13764020543352296,
 'right_ring_MCP_AA': 1.2548647515831033e-08,
 'right_ring_PIP': 0.7557869836117458,
 'right_ring_DIP': 0.883227786525132}


def four_finger_candidate():
    """Thumb opposes index, middle and ring at three handle heights.

    Closed targets place original source pad geometry about 1.5 mm inside the
    ideal 22 mm radius cylinder. This is commanded preload, not an accepted
    physical penetration. Original-source shell separation and native contact
    checks remain mandatory. The real handle is a 64-sided source convex hull.
    """
    previous = opposing_pinch_candidate()
    opened = previous['open_hand_rad']
    preclosed = {**opened, **_PRECLOSE_RAD}
    closed = {**opened, **_CLOSED_RAD}
    thumb_base = dict(opened)
    for name in ('right_thumb_CMC_FE', 'right_thumb_CMC_AA'):
        thumb_base[name] = preclosed[name]
    return {
        'revision': 'four_finger_source_v1',
        'axis_in_hand_xz_m': list(previous['axis_in_hand_xz_m']),
        'open_hand_rad': opened, 'thumb_base_hand_rad': thumb_base,
        'preclose_hand_rad': preclosed, 'closed_hand_rad': closed,
        'intended_contact_fingers': ['thumb', 'index', 'middle', 'ring'],
        'preclose_ideal_pad_gap_m': .0007,
        'closed_ideal_pad_target_overlap_m': .0015,
        'native_penetration_limit_m': .002,
        'scope': 'Offline optimized source geometry candidate. Target overlap does not '
                 'prove pressure or permit native penetration. No grasp qualification.',
    }


def _capability_values(capabilities):
    values = capabilities.model_dump(mode='json') if hasattr(capabilities, 'model_dump') else deepcopy(dict(capabilities))
    if values.get('direct_joint_execution') != 'min_jerk_velocity_preflight_v2':
        raise ValueError('Public candidate requires min_jerk_velocity_preflight_v2')
    arms, hands = tuple(values['arm_joints']), tuple(values['hand_joints'])
    names = arms + hands
    if len(set(names)) != len(names) or len(arms) != 7 or len(hands) != 22:
        raise ValueError('Candidate requires the selected 7 arm + 22 hand joints')
    dt = float(values['control_period_s'])
    if not math.isfinite(dt) or dt <= 0:
        raise ValueError('Invalid public control period')
    limits = values['joint_limits']
    if set(limits) != set(names):
        raise ValueError('Capabilities must describe every selected joint')
    for name in names:
        row = limits[name]
        if not isinstance(row, Mapping):
            row = row.model_dump(mode='json')
            limits[name] = row
        speed = row.get('max_velocity_rad_s')
        if isinstance(speed, bool) or not isinstance(speed, (int, float)) or not math.isfinite(speed) or speed <= 0:
            raise ValueError('Every public joint needs an explicit positive velocity limit')
    return values, arms, hands, dt


def _retime(phase, previous, capabilities):
    values, arms, hands, dt = _capability_values(capabilities)
    names = arms + hands
    target = phase['joint_positions_rad']
    if set(previous) != set(names) or set(target) != set(names):
        raise ValueError('Targets and acknowledged start must contain all 29 joints')
    arm_cap = float(phase.get('arm_speed_cap_rad_s', .65))
    hand_cap = float(phase.get('hand_speed_cap_rad_s', 1.))
    if min(arm_cap, hand_cap) <= 0 or not all(map(math.isfinite, (arm_cap, hand_cap))):
        raise ValueError('Phase speed caps must be finite and positive')
    arm_cap, hand_cap = min(.65, arm_cap), min(1., hand_cap)
    public_ramp_minimum = 1.
    task_ramp_minimum = 1.
    for name in names:
        if not all(math.isfinite(float(q[name])) for q in (previous, target)):
            raise ValueError('Joint references must be finite')
        row = values['joint_limits'][name]
        if not row['lower_rad'] <= target[name] <= row['upper_rad']:
            raise ValueError('Candidate target violates advertised joint limits')
        speed = min(row['max_velocity_rad_s'], arm_cap if name in arms else hand_cap)
        peak_distance = 1.875*abs(target[name]-previous[name])
        public_ramp_minimum = max(public_ramp_minimum, peak_distance/row['max_velocity_rad_s']/dt)
        task_ramp_minimum = max(task_ramp_minimum, peak_distance/speed/dt)
    public_steps = math.ceil(public_ramp_minimum-1e-9)
    task_ramp_steps = math.ceil(task_ramp_minimum-1e-12)

    def ramp_steps(total_steps):
        return max(math.ceil(_DIRECT_RAMP_FRACTION*total_steps), public_steps)

    minimum_steps = task_ramp_steps
    while ramp_steps(minimum_steps) < task_ramp_steps:
        minimum_steps += 1
    requested = float(phase['duration_s'])
    if not math.isfinite(requested) or requested <= 0:
        raise ValueError('Phase duration must be finite and positive')
    total_steps = max(minimum_steps, math.ceil(requested/dt-1e-12))
    return dt*total_steps, dt*minimum_steps, arms, hands, dt*ramp_steps(total_steps)


def candidate_public_motion_plan(capabilities, robot=None, **tool_pose):
    """Preserve the reviewed approach, change hand closure and public timing.

    ``capabilities`` comes from the selected episode. Durations bound the C2
    reference speed during the actual 80% ramp of public move_joints, including
    its public-limit extension. A feedback ``set_hand_targets`` run
    has the bench controller's own timing/reference path and must be recorded
    and qualified separately. Function hints are advisory, never auto execution.
    """
    plan = candidate_motion_plan(robot, grasp_candidate=four_finger_candidate(), **tool_pose)
    previous = plan['initial_joint_positions_rad']
    for phase in plan['phases']:
        phase['arm_speed_cap_rad_s'] = .65
        phase['hand_speed_cap_rad_s'] = .12 if phase['name'] in ('close', 'release_preclose') else 1.
        if phase['name'] in ('hold_grasp', 'final_hold', 'ready_hold'):
            phase['public_function_hint'] = 'hold'
        elif phase['name'] in ('thumb_base_prepare', 'preclose', 'close', 'release_preclose', 'open_thumb_fingers', 'open'):
            phase['public_function_hint'] = 'set_hand_targets'
        else:
            phase['public_function_hint'] = 'move_joints'
        duration, minimum, arms, hands, ramp_duration = _retime(phase, previous, capabilities)
        phase['duration_s'] = duration
        phase['minimum_public_duration_s'] = minimum
        phase['direct_ramp_duration_s'] = ramp_duration
        phase['direct_endpoint_hold_duration_s'] = duration-ramp_duration
        phase['maximum_arm_target_speed_rad_s'] = max(1.875*abs(phase['joint_positions_rad'][n]-previous[n])/ramp_duration for n in arms)
        phase['maximum_hand_target_speed_rad_s'] = max(1.875*abs(phase['joint_positions_rad'][n]-previous[n])/ramp_duration for n in hands)
        previous = phase['joint_positions_rad']
    values, _, _, _ = _capability_values(capabilities)
    plan.update(schema='mechanical_public_tool_motion_candidate.v2',
        maximum_arm_target_speed_rad_s=.65, maximum_hand_target_speed_rad_s=1.,
        total_duration_s=sum(p['duration_s'] for p in plan['phases']),
        public_execution='move_joints uses actual acknowledged targets, advertised per-joint speed limits '
                         'and the actual 80% minimum-jerk ramp plus endpoint hold',
        direct_ramp_fraction=_DIRECT_RAMP_FRACTION,
        public_capabilities=deepcopy(values),
        expected_contact_link_suffixes=[f'right_{finger}_{part}'
            for finger in ('thumb', 'index', 'middle', 'ring') for part in ('elastomer', 'DP')],
        scope='Task-owned privileged engineering candidate for ordinary public bench actions; '
              'no object forces, writeback or weld. Only discrete source geometry is reviewed. '
              'Phase speed caps bound the direct ramp, not the entire action duration. '
              'Feedback execution requires separate native evidence. Prior native qualification '
              'does not transfer to this retimed candidate. No screw turn is included.')
    return plan


def public_move_arguments(phase, capabilities, acknowledged_joint_positions_rad, *,
                          observation_sequence, action_id):
    """Retiming at dispatch uses the ACK, never an assumed observed start.

    The caller obtains a fresh idle observation sequence and the public last
    acknowledged drive targets. Bench admission and contact guards remain the
    authority; returned valid arguments are not execution success.
    """
    duration, _, arms, hands, _ = _retime(phase, acknowledged_joint_positions_rad, capabilities)
    goal = phase['joint_positions_rad']
    return MoveJointsArguments(action_id=action_id, observation_sequence=observation_sequence,
        duration_s=duration, arm_joint_positions_rad={n: goal[n] for n in arms},
        hand_joint_positions_rad={n: goal[n] for n in hands}).model_dump(mode='json')
