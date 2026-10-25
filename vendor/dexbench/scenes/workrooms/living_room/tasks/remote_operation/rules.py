"""Frozen single-hand rules and practical, asset-bound scoring parameters."""
from copy import deepcopy
import hashlib
import json
import math

LEGACY_RULES_VERSION = 'living_room_remote_operation_rules_v2'
RULES_VERSION = 'living_room_remote_operation_rules_v3'
LEGACY_RULES = {
    'schema': LEGACY_RULES_VERSION, 'status': 'implemented_reference', 'competitive': False,
    'instruction_zh': '把背面朝上的遥控器翻正，用同一只手握持并依次按下公布的两个按键，'
                      '每次按后松键，最后将遥控器按正确朝向放回收纳架并松手。',
    'stages': ['face_up', 'handheld_button_sequence', 'release_buttons', 'return_to_holder'],
    'buttons': {'ids': ['power', 'confirm'], 'mechanism': 'Independent passive spring-return prismatic buttons.',
                'press_evidence': ['Physical finger/button contact.',
                                   'Actual button joint crosses the asset-bound press displacement.',
                                   'Remote is off the table/holder and genuinely held by the same hand.',
                                   'Button physically returns to or above its release position before a new press.'],
                'no_credit': ['Color change without contact', 'One long hold counted repeatedly',
                              'Remote supported by table or holder during credited press',
                              'Agent claims that a button was pressed', 'Simultaneous ambiguous keys']},
    'success': {'all_of': ['Ordered distinct physical press-and-release events match the published sequence.',
                           'Whole remote is within the correctly oriented holder acceptance volume.',
                           'Holder physically supports the remote; all hand contacts have ended.',
                           'Remote and all buttons retain released supported stability for 0.300 seconds.'],
                'reset_credit': False},
    'recovery': {'wrong_button': 'Reset sequence progress; allow retry without resetting objects.',
                 'table_drop': 'Recover by actual regrasp.', 'table_assisted_flip': True,
                 'regrasp': True},
    'failure': ['Remote leaves the permitted workspace or falls to the room floor.'],
    'budget_exhaustion': 'incomplete', 'evidence_fault': 'invalid_no_score',
    'randomization': ['Seed parity selects the public order of the two keys; physical reset poses are fixed in the asset release.'],
    'asset_freeze_required': ['Remote/button collision geometry and masses', 'Button travel, spring and damping',
                              'Press/release hysteresis', 'Holder fit', 'Stable speed/contact/penetration tolerances'],
    'qualification': 'Candidate physical buttons have independent native mechanism evidence. The read-only observer/scorer is implemented; native same-hand robot task execution is not qualified.',
}


RULES = deepcopy(LEGACY_RULES)
RULES.update(
    schema=RULES_VERSION,
    instruction_zh='把背面朝上的遥控器翻正，用同一只手悬空握持遥控器，完整按下并松开电源键开启电视；'
                   '保持电视连续开机至少10秒，再完整按下并松开电源键关闭电视，'
                   '最后将遥控器按正确朝向放回收纳架，松手并保持稳定至少0.300秒。',
    stages=['face_up', 'power_on', 'continuous_power_on', 'power_off', 'return_to_holder'],
    television={'initial_power_on': False, 'required_on_s': 10., 'toggle_button': 'power',
                'toggle_on': 'One valid physical power press-and-release cycle, at its native release time.',
                'time_basis': 'Consecutive native simulation time; wall time and separate short on periods do not count.',
                'confirm_button': 'No effect on television power or task progress.'},
    success={'all_of': ['A valid handheld physical power press-and-release cycle turns the initially off television on.',
                        'The television remains continuously on for at least 10 native simulation seconds.',
                        'A subsequent valid handheld physical power press-and-release cycle turns it off.',
                        'Whole remote is within the correctly oriented holder acceptance volume.',
                        'Holder physically supports the remote; all hand contacts have ended.',
                        'Remote and all buttons retain released supported stability for 0.300 seconds.'],
             'continuous_on_resets_when_off': True, 'automatic_power_off': False,
             'holder_return_required': True, 'stable_duration_s': .300},
    recovery={'early_power_off': 'No success; turn on again and restart the continuous 10-second interval.',
              'table_drop': 'Recover by actual regrasp.', 'table_assisted_flip': True, 'regrasp': True},
    randomization=['Physical reset poses are fixed; the power-on/10-second/power-off/return goal is independent of seed.'])
RULES['buttons']['no_credit'].remove('Simultaneous ambiguous keys')


BUTTON_IDS = ('power', 'confirm')


def button_sequence(seed=0):
    if type(seed) is not int or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    return list(BUTTON_IDS if seed % 2 == 0 else reversed(BUTTON_IDS))


def thresholds():
    return dict(press_position_m=-.0014, release_position_m=-.0003,
        joint_lower_m=-.002, joint_upper_m=0., joint_tolerance_m=.0002,
        minimum_contact_impulse_ns=1.e-10, maximum_penetration_m=.002,
        held_before_press_s=.02, stable_s=.300, maximum_linear_speed_m_s=.02,
        maximum_angular_speed_rad_s=.15, maximum_stable_position_range_m=.001,
        maximum_stable_orientation_range_rad=.025, holder_xy_tolerance_m=.001,
        holder_bottom_tolerance_m=.002, holder_orientation_tolerance_rad=.15,
        minimum_handheld_clearance_m=.005, max_time_s=120., max_actions=6000)


def referee_contract():
    phases = {'await_power_on': '电视已关闭，等待一次有效的电源键完整按压和释放。',
              'holding_power_on': '电视已开启，保持连续开机。',
              'await_power_off': '连续开机时间已满足，等待用遥控器关闭电视。',
              'await_return': '电视已完成连续开机和关机，等待遥控器正确归架、松手并稳定。',
              'completed': '已完成开机、连续开机、关机和遥控器归架松手稳定。',
              'evidence_invalid': '原生物理证据无效。',
              'task_failure': '遥控器落地或离开允许操作区域。',
              'budget_exhausted': '预算结束，未取得本次成功判定。'}
    limits = thresholds()
    return {'schema': 'physical_task_referee_contract.v1', 'task_id': 'remote_operation',
        'instruction_zh': RULES['instruction_zh'], 'phases': phases,
        'reason_descriptions': deepcopy(phases),
        'success_conditions': ['电视初始关闭；每次有效电源键完整按压并真实释放后只切换一次电视电源。',
            '用同一只手悬空握持遥控器，以真实手指接触按压实体电源键，并让按键松开回弹。',
            '电视连续开机至少10秒后，再用遥控器关闭电视；时间按连续原生仿真步计算。',
            '遥控器整体按正确朝向放入收纳架，由架子真实支撑；手与遥控器、按键和架子均无接触。',
            '遥控器及按键全部释放并连续稳定至少0.300秒，才取得成功。',
            '满10秒不会自动关机或自动成功；确认键不影响电视电源。'],
        'recovery': ['允许翻正、重新抓取和重试。',
            '提前关机不成功；下次开机重新计算连续10秒，多次短开机时间不累计。'],
        'thresholds': {**{key: limits[key] for key in ('press_position_m', 'release_position_m',
            'held_before_press_s', 'stable_s', 'max_time_s', 'max_actions')}, 'required_on_s': 10.},
        'outcomes': {'ongoing': '尚未完成，可在预算内继续。',
            'success': '有效物理证据确认开机、连续开机至少10秒、遥控器关机及归架松手稳定。',
            'task_failure': '有效证据确认遥控器落地或离开允许操作区域。',
            'evaluator_invalid': '缺失、矛盾或不连续原生证据，无有效分数。',
            'safety_stop': '物理安全中止单独记录。',
            'budget_exhaustion': '运行器结束回合，未完成不制造成功。',
            'cancelled': '取消或运行故障独立记录，不制造任务成功或物理失败。'},
        'physical_actions': '仅通过机器人真实接触操作遥控器；长按不能重复计数，桌面或收纳架支撑中的按压不计。',
        'observation_scope': '公开静态规则与粗粒度裁判阶段；逐原生子步位姿、接触和连续计时由私有裁判核验。'}


def public_rules(seed=0, *, version=RULES_VERSION):
    sequence = button_sequence(seed)  # Validate the seed for both rule versions.
    if version not in (RULES_VERSION, LEGACY_RULES_VERSION):
        raise ValueError('Unknown remote rules version')
    result = deepcopy(LEGACY_RULES if version == LEGACY_RULES_VERSION else RULES)
    if version == LEGACY_RULES_VERSION:
        result['button_sequence'] = sequence
    else:
        result['referee'] = referee_contract()
    result['thresholds'] = thresholds()
    result['evidence_scope'] = ('Private native body/joint/contact samples only. '
        'CPU replay verifies logic; independent mechanical probes do not establish same-hand robot success.')
    return result


def evaluation_configuration(*, seed=0, physics_dt_s=.0005, asset_metadata=None, rules_version=RULES_VERSION):
    rules = public_rules(seed, version=rules_version)
    if type(physics_dt_s) not in (float, int) or not math.isfinite(physics_dt_s) or not 0 < physics_dt_s <= .002:
        raise ValueError('physics_dt_s must be finite, positive and <= 2 ms')
    ratio = .02 / physics_dt_s
    if abs(ratio-round(ratio)) > 1.e-8:
        raise ValueError('20 ms control period must contain an integer number of physics steps')
    m = deepcopy(asset_metadata) if asset_metadata is not None else {}
    remote, holder = m.get('remote', {}), m.get('holder', {})
    root = m.get('root_path', '/World/LivingTask')
    actors = {'remote': remote.get('body_path', root+'/Remote/Body'),
        'holder': holder.get('body_path', root+'/Holder'),
        **{n: m.get('buttons', {}).get(n, {}).get('body_path', root+'/Remote/'+n.title()+'Button') for n in BUTTON_IDS}}
    config = {'schema': 'living_remote_evaluation.v2' if rules_version == LEGACY_RULES_VERSION else 'living_remote_evaluation.v3',
        'physics_dt_s': float(physics_dt_s), 'control_period_s': .02,
        'actors': actors, 'robot_actor_prefix': '/World/Robot/',
        'hand_link_prefixes': ['right_'],
        'finger_link_prefixes': ['right_thumb_', 'right_index_', 'right_middle_', 'right_ring_', 'right_pinky_'],
        'geometry': {'remote_local_bounds_m': [[-.0275, -.080, -.0104], [.0275, .080, .0135]],
            'holder_opening_xy_m': holder.get('opening_xy_m', [.067, .172]),
            'holder_liner_z_m': .007, 'table_top_z_m': .78,
            'workspace_bounds_m': [[.05, -.7, .72], [1.25, .7, 1.95]]},
        **thresholds()}
    if m and (remote.get('shell_size_xyz_m') != [.055, .16, .02]
              or remote.get('envelope_xyz_m') != [.055, .16, .0239]
              or holder.get('opening_xy_m') != [.067, .172]):
        raise ValueError('Scorer geometry requires the published 160 x 55 mm remote and holder')
    if rules_version == LEGACY_RULES_VERSION:
        config['button_sequence'] = rules['button_sequence']
    else:
        config['required_on_s'] = rules['television']['required_on_s']
        config['public_rules_sha256'] = hashlib.sha256(json.dumps(
            rules, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
    return config
