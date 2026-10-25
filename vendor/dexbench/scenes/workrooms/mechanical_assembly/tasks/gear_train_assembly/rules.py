"""Contact-backed L5 scoring candidates; scene evidence is separately attached."""
from copy import deepcopy
import math

if __package__.startswith('scenes.workrooms.mechanical_assembly.'):
    from ...referee import public_contract
else:  # Frozen releases carry this helper beside rules.py.
    from .referee import public_contract

GEARS = ('input_gear', 'idler_gear', 'output_gear')
TOOLS = ('socket_crank', 'open_end_wrench', 'straight_hex_screwdriver')
VERSION = 'mechanical_gear_rules_v4'
GEOMETRY_CONTRACT = {
    'gear_geometry_version': 'v3_large_idler_20_32_20',
    'gear_teeth': {'input_gear': 20, 'idler_gear': 32, 'output_gear': 20},
    'shaft_centers_world_xy_m': [[.4, -.13], [.4, 0.], [.4, .13]],
    'signed_angle_ratios': {'idler_per_input': -.625, 'output_per_input': 1.},
    'input_equivalent_turn_weights': [1., 1.6, 1.],
}


def geometry_contract():
    return deepcopy(GEOMETRY_CONTRACT)
INSTRUCTION_ZH = ('把三个齿轮装到各自轴位并松开，静止至少0.3秒；自行选择工具，从输入端缓慢驱动，'
    '让输出标记从装配稳定后的方向逆时针转动约30度，再松开工具，齿轮和本次使用过的工具均稳定至少0.3秒。')
PHASES = {
    'assembly': '装配到各自轴位', 'assembly_settling': '松手并等待装配稳定',
    'transmission': '工具从输入端传动', 'reassembly_required': '重新装配并重建传动基线',
    'release_settling': '松开工具并等待稳定', 'complete': '任务已完成',
    'unavailable': '裁判证据异常，本次结果不可评分', 'safety_stop': '物理安全边界越限，停止本次任务',
}
REASONS = {
    'not_sampled': '尚未收到初始观测。',
    'reset_baseline_not_scored': '初始散件状态已核验；初始状态不计装配或传动信用。',
    'gear_seating_required': '齿轮未全部就座；可以调整或重新装配，原传动信用已清空。',
    'direct_drive_requires_reassembly': '直接接触齿轮或工具接触非输入齿轮，原传动信用已清空；松手并重新静置后再从工具输入端传动。',
    'assembly_support_required': '尚缺三个齿轮各自的向上承托证据，稳定计时重新开始。',
    'assembly_release_required': '仍直接接触齿轮；松手后重新累计装配稳定时间。',
    'assembly_stability_required': '齿轮仍在移动或稳定窗口内漂移过大，装配稳定计时重新开始。',
    'released_assembly_settling': '装配已就座并松手，正在累计0.3秒稳定时间。',
    'assembly_baseline_recorded': '装配稳定基线已记录，可以从工具输入端开始传动。',
    'phase_error_requires_reassembly': '本次传动按齿数换算为输入等效角后的累计相位误差曾超过2度；不能只修正终点来清除历史，须重新装配并静置重建全部基线。',
    'drive_speed_required': '本窗口齿轮转速超过15度每秒，不增加正向传动信用；降低速度后可继续。',
    'contact_chain_required': '缺少同一子步内完整的持工具、工具输入、两处啮合及承托链，本窗口不增加正向信用。',
    'supported_transmission_required': '传动或收尾缺少向上承托证据，不能完成任务。',
    'direction_ratio_required': '传动方向或最终转角比例尚不符合要求；反转会扣除已有信用。',
    'output_turn_required': '输出净转角尚未处于逆时针25至35度范围内。',
    'contact_credit_required': '尚未保留至少25度完整接触链支持的传动信用。',
    'tool_interaction_required': '合格正向工具传动累计时间尚不足0.1秒。',
    'release_required': '仍有机器人接触齿轮或任一工具；零压力接触也不算松手。',
    'used_tool_stability_required': '本次获计分传动所用工具仍在运动或漂移，收尾稳定计时重新开始。',
    'release_stability_required': '齿轮仍在运动或漂移，收尾稳定计时重新开始。',
    'release_settling': '输出与传动证据已达标，正在累计松手后0.3秒稳定时间。',
    'success': '装配、接触传动及松手稳定均已完成，成功结果已锁定。',
    'invalid_observation': '观测、时钟、初始状态或接触回执异常，本次不可评分；修复证据来源后重新开始。',
    'object_speed_limit': '对象速度超过既定安全上限，本次停止；不可作为普通未完成结果继续累计。',
    'penetration_limit': '任务对象接触穿透超过既定安全上限，本次停止。',
}
THRESHOLDS = {
    'physics_dt_s': .0005, 'control_period_s': .02,
    'assembly_settle_s': .3, 'release_hold_s': .3,
    'nominal_output_turn_rad': math.radians(30.),
    'minimum_output_turn_rad': math.radians(25.),
    'maximum_output_turn_rad': math.radians(35.),
    'maximum_axis_offset_m': .0004,
    'maximum_bottom_height_error_m': .0003,
    'maximum_axis_tilt_rad': math.radians(.5),
    'maximum_drive_angular_speed_rad_s': math.radians(15.),
    'maximum_object_angular_speed_rad_s': 2.,
    'maximum_object_linear_speed_m_s': .2,
    'maximum_stable_linear_speed_m_s': .002,
    'maximum_stable_angular_speed_rad_s': .05,
    'maximum_stable_observed_angular_speed_rad_s': .02,
    'maximum_stable_position_range_m': .0001,
    'maximum_stable_angle_range_rad': math.radians(.5),
    'maximum_pair_phase_error_rad': math.radians(2.),
    'maximum_ratio_error': .10,
    'minimum_motion_chain_fraction': .5,
    'minimum_tool_interaction_s': .1,
    'maximum_holding_normal_cosine': -.25,
    'minimum_pair_impulse_ns': 1.e-8,
    'maximum_reported_penetration_m': .0005,
    'initial_position_tolerance_m': .0001,
    'initial_orientation_tolerance_rad': .002,
}


def thresholds():
    return deepcopy(THRESHOLDS)


def public_rules():
    c = thresholds()
    return {
        'schema': VERSION, 'status': 'engineering_candidates_not_native_accepted',
        'instruction_zh': INSTRUCTION_ZH,
        'physical_gear_contract': geometry_contract(),
        'referee': public_contract('gear_train_assembly', instruction_zh=INSTRUCTION_ZH,
            phases=PHASES, reasons=REASONS, thresholds=c,
            success_conditions=[
                '三个齿轮在各自轴位就座，由工作台承托，松手稳定至少0.3秒后建立零转角基线。',
                '工具种类不决定得分；须由两个不同右手连杆以相对法向载荷持住工具，从输入端经两处真实啮合传动。',
                '至少一半原生子步同时满足完整接触链，才增加该20毫秒窗口的正向信用；任何反转均扣除信用。',
                '齿数为20/32/20：惰轮反转角为输入的0.625倍、输出同向等角；终点净转角比例相对理论值的误差不超过10%，允许正常齿隙带来的起转空程；按齿数换算为输入等效角后，每对齿轮的累计相位误差从未超过2度。',
                '输出净转角为25至35度，保留至少25度接触信用，合格正向工具传动累计至少0.1秒。',
                '机器人松开全部齿轮和工具，齿轮与本次获计分的工具共同稳定至少0.3秒；工具可留在啮合位置。',
            ], recovery=[
                '装配未就座、未松手或尚未稳定可继续调整；未达标不自动判负。',
                '缺少接触链或传动过快的窗口不增加正向信用，反转仍扣除信用；可降低速度并恢复接触后继续。',
                '累计相位曾越界时，须真实重新装配或接触齿轮，再松手静置重建基线；转角、信用、相位历史及计时全部清空，不能保留好段。',
                '预算用尽属于未完成；观测异常属于不可评分，安全越界属于安全停止，均不能伪装为正常任务失败。',
                '成功或异常终局锁定后不能由后续观测撤销；新尝试须执行真实场景重置并重新核验初始散件状态。',
            ]),
        'tool_id_affects_score': False,
        'success': {'nominal_output_turn_degrees': 30., 'allowed_output_turn_degrees': [25., 35.],
            'direction': 'Counterclockwise viewed from above the positive world Z axis.',
            'maximum_drive_speed_degrees_per_second': 15., 'assembly_settle_s': .3, 'release_hold_s': .3,
            'requirements': ['All three gears sit on the intended shafts with upward fixture support.',
                'A free tool held by two distinct hand links with opposing normal impulses (cosine at most -0.25) drives the input through simultaneous positive tool/input and both gear-mesh contacts.',
                'For 20/32/20 teeth, idler/input=-0.625 and output/input=+1. Final net-turn ratio errors stay within 10 percent to allow normal backlash takeup; tooth-count-normalized input-equivalent pair phase error stays within 2 degrees.',
                'Only intervals with simultaneous contact-chain evidence retain positive rotation credit; every reverse increment removes credit.',
                'At least 25 degrees of output rotation retain contact-backed credit after release.',
                'All gears and every tool that earned forward credit in this baseline remain stable throughout the final release window.',
                'No direct robot or other tool drive of idler/output during the transmission phase.'],
            'tool_may_remain_engaged_after_release': True},
        'recoverable': ['Tooth-phase adjustment during assembly.', 'Withdraw and reseat parts before transmission.',
            'Changing tools before qualified drive.', 'Backlash takeup with no retained positive output credit.'],
        'prohibited': ['object_pose_playback', 'external_object_wrench', 'gear_ratio_or_hinge_joint', 'grasp_weld'],
        'evidence_scope': '20 ms actual poses/velocities and complete native-substep normal contact receipts; no substep pose or measured torque claim.',
        'qualification': 'Frozen scoring candidates. Private preassembled tool-load experiments do not earn robot task credit.',
    }
