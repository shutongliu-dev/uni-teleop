"""Versioned engineering candidates, not native-accepted task tolerances."""
from copy import deepcopy
import math

if __package__.startswith('scenes.'):
    from ...referee import public_contract
else:
    from .referee import public_contract

RULES_VERSION = 'mechanical_fastener_rules_v4'
LOOSE_RULES_VERSION = 'mechanical_fastener_rules_v5'
PHASES = {'settle': '被动稳定与记录基线', 'advance': '通过工具旋拧并保留进度',
          'release': '松手并重新累计完整稳定时间', 'complete': '任务成功', 'stopped': '本次评估已终止'}
REASONS = {
    'not_sampled': '尚未收到原生状态和接触证据。',
    'reset_baseline_not_scored': '重置时已有的螺纹啮合和工具插入不计分。',
    'waiting_for_passive_settlement': '保持待命，等待完整的被动稳定窗口；被触碰或移动后重新计时。',
    'passive_settled': '被动稳定基线已记录，可以开始通过工具旋拧。',
    'additional_thread_advance_required': '仍需通过工具接触链产生足够的实际螺丝净转角和下行量。',
    'contact_backed_advance_required': '实际位置接近目标，但可验证的工具传动净进度仍不足；不能靠空转或抖动刷进度。',
    'uncredited_motion': '本段运动缺少完整工具传动证据，不增加进度；允许重抓后继续。',
    'reverse_motion_debits_credit': '反向运动已扣除进度；可以恢复旋拧，但只计算最终保留的净进度。',
    'target_turn_exceeded_recoverably': '转角暂时超过目标区间但未超过整次操作限制，可以回转纠正并重新稳定。',
    'tool_reengagement_required': '工具与螺丝的有效接触尚未建立，可重新接合后继续。',
    'robot_contact_prevents_release': '机器人仍有接触流，尚未松手；零冲量接触也会重新开始稳定计时。',
    'release_motion_resets_dwell': '物体仍在运动，需重新累计完整稳定时间。',
    'release_envelope_resets_dwell': '稳定窗口内位姿累计漂移过大，需重新累计完整稳定时间。',
    'release_stability_in_progress': '目标进度已达到，正在确认松手后的连续稳定。',
    'success': '实际旋拧、接触传动净进度及松手稳定条件均已满足。',
    'direct_robot_bolt_contact': '出现超过公开冲量门槛的机器人直接接触螺丝，本次任务失败。',
    'non_hand_robot_tool_contact': '出现超过公开冲量门槛的非手部机器人接触工具，本次任务失败。',
    'bolt_left_thread_axis': '螺丝偏离已啮合螺纹允许的轴线或倾角，本次任务失败。',
    'lost_thread_fixture_support': '基线记录后螺丝失去连续螺纹支撑超过允许间隔，本次任务失败。',
    'thread_rotation_advance_inconsistent': '旋转与下行不满足导程约束，或超过整次操作转角上限，本次任务失败。',
    'object_motion_speed_bound': '物体运动超过安全速度边界，停止评估；不作为有效任务失败。',
    'reported_penetration_bound': '原生接触报告的穿透超过安全边界，停止评估；不作为有效任务失败。',
    'invalid_native_evidence': '原生观测、接触回执或时间连续性无效，终止本次评估且不计任务输赢。',
}
THRESHOLDS = {
    'physics_dt_s': .0005, 'control_period_s': .02,
    'settle_s': .5, 'release_hold_s': .5,
    'thread_lead_m_per_revolution': .003, 'tightening_yaw_sign': -1.,
    'minimum_turn_rad': math.radians(40.), 'nominal_turn_rad': math.radians(45.),
    'maximum_target_turn_rad': math.radians(55.),
    'maximum_episode_turn_rad': math.radians(90.),
    'maximum_lead_residual_m': .00010,
    'maximum_axis_offset_m': .0015, 'maximum_bolt_tilt_rad': math.radians(5.),
    'maximum_bolt_angular_speed_rad_s': math.radians(30.),
    'maximum_object_linear_speed_m_s': .20,
    'maximum_tool_angular_speed_rad_s': 2.,
    'maximum_stable_linear_speed_m_s': .002,
    # Native instantaneous angular velocity includes solver oscillation; the
    # separate pose difference and dwell envelope still require actual stability.
    'maximum_stable_angular_speed_rad_s': .3,
    'maximum_stable_observed_angular_speed_rad_s': .02,
    'maximum_stable_position_range_m': .00010,
    'maximum_stable_angle_range_rad': math.radians(1.),
    'initial_position_tolerance_m': .00010,
    'initial_orientation_tolerance_rad': .002,
    'maximum_settle_position_drift_m': .00020,
    'maximum_settle_orientation_drift_rad': .02,
    'minimum_motion_chain_fraction': .5,
    'minimum_tool_interaction_s': .10,
    'maximum_support_gap_s': .001,
    'minimum_pair_impulse_ns': 1.e-8,
    'maximum_reported_penetration_m': .002,
}


def thresholds(target_turn_degrees=30.):
    """Freeze a positive tightening goal; all thresholds remain engineering candidates."""
    if (type(target_turn_degrees) not in (int, float)
            or not math.isfinite(target_turn_degrees) or not 0 < target_turn_degrees <= 45.):
        raise ValueError('Candidate target turn must be finite and in (0, 45] degrees')
    target = float(target_turn_degrees)
    result = deepcopy(THRESHOLDS)
    result.update(nominal_turn_rad=math.radians(target),
                  minimum_turn_rad=math.radians(target-min(5., target/6.)),
                  maximum_target_turn_rad=math.radians(target+min(3., target/10.)),
                  maximum_lead_residual_m=min(.00005, .003*target/360*.2))
    return result


def public_rules(target_turn_degrees=30., *, reset_variant='preinserted'):
    """Static intent only; no initial/current object pose, actor path or score.

    ``preinserted`` is the historical guided task.  ``loose_bolt`` keeps the
    same measured thread/turn contract but adds a pre-engagement phase where
    the independent bolt is picked from the desktop and started in the guide.
    """
    if reset_variant not in ('preinserted', 'loose_bolt'):
        raise ValueError('Unknown fastener reset variant: '+str(reset_variant))
    c = thresholds(target_turn_degrees)
    target = float(target_turn_degrees)
    advance = c['thread_lead_m_per_revolution']*target/360
    if reset_variant == 'loose_bolt':
        instruction = (f'先从桌面取起螺丝并对准导向口，确认螺纹起牙；再取起工具插入六角孔，'
                       f'沿旋入方向缓慢转动约{target:g}度，使螺丝按螺纹下行，最后松手并稳定至少0.5秒。')
        reset_simplifications = [
            'Bolt and screwdriver start as independent gravity-enabled bodies on task-owned desktop rests.',
            'A shallow task-owned entry funnel provides coarse centering only; no snap, weld or thread joint.',
            'Thread engagement, tool insertion and all manipulation remain uncredited until the settled engaged baseline.',
        ]
        success_prefix = [
            '初始螺丝与工具均未啮合；抓取、导向、起牙和工具插入不计分。',
            '只有建立螺丝—夹具和手—工具—螺丝接触后，才允许记录旋拧基线。',
        ]
    else:
        instruction = (f'先保持待命至少0.5秒。握住预插工具，沿旋入方向缓慢转动约{target:g}度，'
                       '使螺丝按螺纹进一步下行，再松手并稳定至少0.5秒。')
        reset_simplifications = ['Threads already engaged before reset.',
            'Independent gravity-enabled tool already inserted; neither achievement is scored.']
        success_prefix = [
            '初始螺纹已啮合、独立工具已插入，均不计分；先在无机器人接触下稳定并记录基线。',
        ]
    result = {
        'schema': RULES_VERSION, 'status': 'engineering_candidates_not_native_accepted',
        'instruction_zh': instruction,
        'referee': public_contract('fastener_assembly', instruction_zh=instruction,
            phases=PHASES, reasons=REASONS, thresholds=c,
            success_conditions=[
                *success_prefix,
                '只计算基线后的实际螺丝旋转和轴向下行，二者必须符合公开导程及目标区间。',
                '旋入进度需同时具有手—工具、工具—螺丝、螺丝—夹具的原生接触证据；全部反转扣除信用，信用不超过实际净转角。',
                '达到目标后机器人须离开工具和螺丝；任何报告的机器人接触流（包括零冲量）都阻止松手稳定计时。',
                '工具可以留在螺丝中，但工具—螺丝及螺丝—夹具的有效支撑必须保留，连续稳定满公开时长后才成功。',
                '接触信用和禁止直接螺丝／非手部工具接触使用 minimum_pair_impulse_ns 门槛；松手判据不忽略零冲量接触流。'],
            recovery=[
                '未达到目标、未计分微动、重抓、工具暂时失去接合以及尚未松手均可继续纠正，不自动判负。',
                '目标区间外的回转可在整次操作上限内纠正；反转扣除已得信用，重新旋入仍需完整接触链。',
                '触碰、运动或累计漂移打断稳定窗口后必须重新累计完整时长，不能拼接先前稳定片段。',
                '直接驱动螺丝、非手部驱动工具、脱离螺纹轴线、丢失螺纹支撑或导程不一致为公开硬边界；本次失败后只能新开一局。',
                '坏观测和缺失回执不判任务失败，也不沿缺失证据恢复计分；本次评估终止且不计分。',
                '动作格式或参数不合法时由公共接口拒绝，不将拒绝本身作为螺丝装配失败。',
                '超速或过度穿透触发安全停止；预算耗尽或取消属于未完成，API／仿真故障属于中止。已接受的终局结果不再被后续观测改写。']),
        'reset_simplifications': reset_simplifications,
        'success': {'measured_body': 'bolt', 'nominal_turn_degrees': target,
            'allowed_final_turn_degrees': [math.degrees(c['minimum_turn_rad']), math.degrees(c['maximum_target_turn_rad'])], 'thread_lead_m_per_revolution': .003,
            'rotation_convention': 'Clockwise looking down the fixture positive Z axis (negative right-handed yaw).',
            'nominal_additional_advance_m': advance, 'maximum_lead_residual_m': c['maximum_lead_residual_m'],
            'release_hold_s': .5, 'tool_may_remain_inserted': True,
            'requirements': ['Actual bolt rotation and axial advance after passive settlement.',
                'Only tightening increments with concurrent hand-tool, tool-bolt and bolt-fixture solver contacts earn retained advance credit.',
                'Every reverse increment removes credit, with or without tool contact; credit cannot exceed actual positive net bolt rotation.',
                'Uncredited micro-motion is recoverable; at least the target lower bound must remain as contact-backed net advance.',
                'No direct robot-bolt contact; release both independent bodies and retain stable progress.']},
        'stability_evidence': {'native_instantaneous_angular_speed_rad_s': .3,
            'observed_20ms_angular_speed_rad_s': .02,
            'observed_20ms_linear_speed_m_s': .002,
            'settle_and_release_position_range_m': .0001,
            'settle_and_release_orientation_range_degrees': 1.,
            'native_speed_bound_provenance': 'Reuse the historically checked preinserted-reset angular-speed upper bound; pose difference and dwell envelopes are separate candidate task conditions.'},
        'evidence_scope': 'Actual object poses/velocities every 20 ms, complete 0.5 ms native contact sampling; not continuous substep object trajectories or measured torque.',
        'prohibited': ['object_pose_playback', 'external_object_wrench', 'grasp_weld', 'screw_joint'],
        'qualification': 'Candidate scoring contract only; CPU examples do not establish physical feasibility or success.',
    }
    # Keep the historical preinserted contract byte-for-byte compatible with
    # already frozen releases.  The new variant carries an explicit marker so
    # canonical loading can select the loose reset behavior.
    if reset_variant == 'loose_bolt':
        result['reset_variant'] = reset_variant
        result['schema'] = LOOSE_RULES_VERSION
        referee = result['referee']
        referee['reason_descriptions']['reset_baseline_not_scored'] = '桌面供料初态不计分；可用手取起螺丝、对准起牙并插入工具。'
        referee['reason_descriptions']['waiting_for_passive_settlement'] = '可继续拾取、对准、起牙和工具插入；接合后须无机器人接触地连续稳定满0.5秒，才记录旋拧基线。'
        referee['reason_descriptions']['direct_robot_bolt_contact'] = '稳定基线后机器人直接接触螺丝，或准备阶段非手部接触螺丝，且超过公开冲量门槛，本次任务失败。'
        referee['success_conditions'][1] = '螺丝与夹具、工具与螺丝均接合后，须无机器人接触地连续稳定满0.5秒，才记录旋拧基线。'
        referee['success_conditions'].append('稳定基线前允许用手抓取螺丝、导向起牙和插入工具，均不计旋入进度；任何机器人接触流都会重置被动稳定计时。')
        referee['recovery'][3] = '准备阶段非手部接触螺丝、稳定基线后直接驱动螺丝、非手部驱动工具、已啮合后的脱轴、基线后丢失螺纹支撑或导程不一致为公开硬边界；本次失败后只能新开一局。'
        result['success']['requirements'][-1] = ('Hand contact with the bolt is allowed only before the settled baseline, '
            'without advance credit. After that baseline, no direct robot-bolt contact; '
            'release both independent bodies and retain stable progress.')
    return result
