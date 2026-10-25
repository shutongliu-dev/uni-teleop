"""Frozen scene-side scoring candidates for physical sorting and assembly."""
from copy import deepcopy
import math

from .referee import public_contract

RULES_VERSION = 'mechanical_assembly_rules_v2'
TASK_OBJECTS = {
    'parts_sorting': ('block', 'cylinder', 'ring'),
    'peg_insertion': ('peg_0', 'peg_1'),
    'shaft_spacer_assembly': ('spacer', 'wide_ring', 'top_ring'),
}
OBJECT_NAMES_ZH = {
    'block': '方块', 'cylinder': '圆柱', 'ring': '圆环',
    'peg_0': '插销一', 'peg_1': '插销二',
    'spacer': '垫套', 'wide_ring': '宽环', 'top_ring': '顶环',
}
REASON_LABELS_ZH = {
    'not_sampled': '尚未读取本回合的初始状态。',
    'reset_baseline_not_scored': '已确认初始状态，初始摆放不计为完成。',
    'outside_matching_tray': '零件整体尚未进入对应浅盘。',
    'not_at_tray_floor': '零件底部尚未到达盘底高度。',
    'axis_offset': '零件轴线尚未对准指定孔或轴。',
    'bottom_height': '零件尚未到达指定孔底或装配层的高度。',
    'axis_tilt': '零件倾斜超过当前装配要求。',
    'required_support_missing': '尚未取得指定盘底、孔底或下层零件的承托证据。',
    'robot_contact_present': '机器人仍与零件接触，需要松手并离开。',
    'still_moving': '零件仍在移动或转动，需要重新放稳。',
    'previous_endpoint_not_ready': '刚到位或刚停止运动，从下一段完整合格时间开始计时。',
    'stability_drift': '稳定期间相对起始姿态的漂移过大，需要重新计时。',
    'stability_window_incomplete': '条件已满足，仍需等待完整的共同稳定时间。',
    'success': '所有零件均已完成，成功判决已确认。',
    'invalid_evidence': '观测或原生接触证据不可用，本回合无法继续裁判。',
    'object_speed_bound': '零件运动超过物理审计上限，本回合停止。',
    'contact_penetration_bound': '接触穿透超过物理审计上限，本回合停止。',
}
PHASES_ZH = {
    'not_started': '等待初始状态', 'reset': '确认初始状态', 'placement': '摆放与装配',
    'support': '检查指定承托', 'release': '等待松手', 'settling': '等待共同稳定',
    'complete': '已完成', 'invalid': '证据不可用', 'safety_stop': '物理审计停止',
}
COMMON = {
    'physics_dt_s': .0005, 'control_period_s': .02, 'minimum_stable_s': .5,
    'maximum_stable_linear_speed_m_s': .005,
    'maximum_stable_angular_speed_rad_s': .05,
    'maximum_stable_position_range_m': .0002,
    'maximum_stable_orientation_range_rad': .01,
    'maximum_object_linear_speed_m_s': 1.,
    'maximum_object_angular_speed_rad_s': 20.,
    'initial_position_tolerance_m': .0001,
    'initial_orientation_tolerance_rad': .002,
    'minimum_support_upward_impulse_ns': 1.e-8,
    'maximum_reported_penetration_m': .0005,
}


def thresholds(task_id):
    if task_id not in TASK_OBJECTS:
        raise ValueError('Unsupported mechanical assembly task')
    value = deepcopy(COMMON)
    if task_id == 'parts_sorting':
        value.update(containment_margin_m=0., maximum_floor_height_error_m=.0005)
    else:
        value.update(maximum_axis_offset_m=.00325 if task_id == 'peg_insertion' else .00425,
                     maximum_bottom_height_error_m=.0005,
                     maximum_axis_tilt_rad=math.radians(10. if task_id == 'peg_insertion' else 5.))
    return value


def public_rules(task_id):
    c = thresholds(task_id)
    instruction = {
        'parts_sorting': '用单臂把方块、圆柱和圆环分别放入对应浅盘。零件整体应在盘内并由盘底承托；松手后让三个零件同时稳定至少0.5秒。',
        'peg_insertion': '用单臂把两根插销分别插入指定插孔，到达孔底并取得承托。松手后让两根插销同时稳定至少0.5秒；插入先后不限。',
        'shaft_spacer_assembly': '用单臂将垫套、宽环和顶环装到固定轴上，使最终层序从下到上依次为垫套、宽环、顶环，并由指定下层承托。松手后让三件同时稳定至少0.5秒。',
    }[task_id]
    return {
        'schema': RULES_VERSION, 'task_id': task_id,
        'status': 'engineering_candidates_not_native_accepted',
        'instruction_zh': instruction,
        'objects': list(TASK_OBJECTS[task_id]),
        'object_names_zh': {name: OBJECT_NAMES_ZH[name] for name in TASK_OBJECTS[task_id]},
        'goal': ('Each entire part lies inside its matching open tray, resting on its floor.'
                 if task_id == 'parts_sorting' else
                 'Both pegs reach the bottom of their assigned sockets.'
                 if task_id == 'peg_insertion' else
                 'The shaft carries spacer, wide ring, then top ring, each supported by the layer below.'),
        'success': {'release_required': True, 'stable_duration_s': c['minimum_stable_s'],
                    'require_actual_upward_support_each_20ms_window': True,
                    'require_all_objects_concurrently': True,
                    'require_whole_geometry_containment': task_id == 'parts_sorting',
                    'reset_credit': False, 'exact_centering_required': False,
                    'action_order_scored': False,
                    'final_layer_order': list(TASK_OBJECTS[task_id]) if task_id == 'shaft_spacer_assembly' else None,
                    'accepted_success_is_final': True},
        'recoverable': ['wrong placement', 'wrong stack order', 'unstable or held part'],
        'recovery': {
            'instruction_zh': '放错、偏斜、未到底、层序不对、仍抓持或被扰动都可以通过正常机器人动作调整后继续；裁判不会自动搬动物体。',
            'automatic_forfeit': False, 'placement_intent_required': False,
            'ordinary_placement_errors': 'Correct with robot actions in the same episode; no permanent logical placement is committed.',
            'stability_restart': 'Before accepted success, any part losing geometry, support, release or stability clears the shared dwell for all parts.',
            'invalid_evidence': 'Current episode evidence becomes unavailable and cannot earn success; a new episode is required. This is not a task loss.',
            'physical_validity_stop': 'The existing motion/penetration bounds invalidate the episode; they are not ordinary recoverable placement errors or a competitive loss.',
            'budget_exhaustion': 'Incomplete episode, not success; no automatic forfeit.',
            'reset_is_in_episode_action': False,
        },
        'stability': {
            'instruction_zh': '所有零件必须同时满足位置、指定支撑、松手和稳定阈值要求。刚到位的前一段时间不补计；成功判决前任一件因扰动或其他原因不再满足要求，都清零共同的0.5秒计时。',
            'both_endpoint_geometry_and_velocity_required': True,
            'first_quiet_endpoint_does_not_credit_previous_motion': True,
            'position_orientation_drift_reference': 'The previous endpoint of the first credited interval anchors the shared window; all subsequent sampled poses must remain within the frozen position/orientation limits of that anchor.',
            'accepted_success_is_final': True,
        },
        'referee': public_contract(task_id, instruction_zh=instruction, phases=PHASES_ZH,
            reasons=REASON_LABELS_ZH,
            success_conditions=[
                '所有零件同时到达各自目标位置；分类检查完整零件边界，插装检查轴线、底部高度和倾角。',
                '每个零件都由指定盘底、孔底或下层零件提供实际向上承托。',
                '机器人与所有任务零件脱离接触，并共同稳定至少0.5秒。',
                '初始状态不计分，刚到位或刚停止运动的上一段时间不补计。',
                '成功判决确认后终局锁存，不再消费后续观测。',
            ], recovery=[
                '成功前允许正常动作纠正错放、偏斜、未到底、层序错误和未松手；不自动判输。',
                '成功判决前任一零件因扰动或其他原因不再满足条件，所有零件的共同稳定时间清零后重计。',
                '证据缺失或物理审计失效会停止当前回合，须重新开始；不记为任务判输。',
                '预算耗尽是未完成，不是成功，也不自动判输。',
            ], thresholds=c),
        'reason_labels_zh': deepcopy(REASON_LABELS_ZH),
        'support_evidence': 'Net upward normal impulse on each supported body from its specified support in every complete 20 ms window; all 0.5 ms native samples must be accounted for. Resting impulses may pulse.',
        'release_evidence': 'No reported robot contact manifold on any task object throughout the stable window, including zero-impulse manifolds.',
        'sampling': {'object_pose_period_s': c['control_period_s'], 'native_contact_period_s': c['physics_dt_s'],
                     'continuous_substep_object_poses_claimed': False},
        'prohibited': ['object_pose_playback', 'external_object_wrench', 'grasp_weld'],
        'thresholds': c,
        'qualification': 'Scoring contract only. Synthetic evaluator tests do not establish native task feasibility or manipulation success.',
    }
