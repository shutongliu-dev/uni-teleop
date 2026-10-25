"""Practical laboratory tasks: public contracts and frozen observation inputs."""
from copy import deepcopy
import hashlib
import json
import math

TASK_IDS = ('sample_sorting', 'vial_racking', 'cap_placement',
            'solid_weighing', 'pipette_transfer')
TITLES = ('样本分类', '试管扶正入架', '样本瓶开盖与复盖', '药匙定量称取颗粒', '跨台定量移液分装')
INSTRUCTIONS = (
    '把四个密封样本瓶按颜色放入对应分类托盘，保持直立，松手放稳。',
    '将岛台浅槽中的三支横放试管扶正，转运到后台试管架，一孔一支插入任意空孔并松手放稳。',
    '将样本瓶放入后台防转瓶座，旋下盖子并放入盖托，再取回旋合，把瓶子送回岛台归位区。',
    '将称量杯放到后台称量平台，用药匙从原料盒取颗粒加入杯中，达到目标质量后把药匙归位并松手。',
    '使用活塞式移液工具从岛台供液杯吸水，向后台两个接收杯分别分装10±2毫升，工具归架并松手。',
)


def object_properties(task_id):
    """Public design masses, checked against every selected dynamic asset below."""
    rows = {
        'sample_sorting': [(f'sample_{i:02}', f'密封样本瓶 {i}', .025) for i in range(1, 5)],
        'vial_racking': [(f'vial_{i:02}', f'密封试管 {i}', .020) for i in range(1, 4)],
        'cap_placement': [('sample_bottle', '样本瓶身（不含独立盖）', .090), ('cap', '独立螺旋盖', .025)],
        'solid_weighing': [('weighing_cup', '空称量舟', .028), ('scoop', '空药匙', .026)]
            + [(f'pellet_{i:02}', f'模拟颗粒 {i+1}', .0005) for i in range(24)],
        'pipette_transfer': [('source', '空供液杯', .120), ('receiver_a', '空接收杯 A', .060),
            ('receiver_b', '空接收杯 B', .060), ('pipette', '移液工具筒身（不含独立活塞）', .065),
            ('plunger', '独立活塞', .016)],
    }[task_id]
    basis = ('冻结资产声明的模拟刚体质量；不是传感器测量或当前抓取负载。'
             '容器和工具按空载/独立部件列示，液体有限体积模型不向刚体施加水重。'
             if task_id == 'pipette_transfer' else
             '冻结资产声明的模拟刚体质量；不是传感器测量或当前抓取负载。容器和工具按空载/独立部件列示。')
    return [dict(id=name, label=label, nominal_mass_kg=mass, basis=basis)
            for name, label, mass in rows]


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    ensure_ascii=True, allow_nan=False).encode()).hexdigest()


def thresholds(task_id):
    if task_id not in TASK_IDS:
        raise ValueError('Unknown laboratory task')
    return dict(stable_s=.3, pose_window_s=.02, maximum_linear_speed_m_s=.03,
        maximum_angular_speed_rad_s=.3, position_drift_m=.004, orientation_drift_rad=.10,
        upright_tilt_rad=math.radians(15), placement_tolerance_m=.004,
        support_gap_m=.008, minimum_movement_m=.02, minimum_contact_impulse_ns=1.e-10,
        maximum_penetration_m=.005, support_memory_s=.05,
        max_time_s=180., max_actions=9000)


def public_rules(task_id, seed=0):
    if task_id not in TASK_IDS or type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError('Known task and uint32 seed required')
    index = TASK_IDS.index(task_id)
    goals = (
        ['四瓶全部置于正确颜色托盘内、直立且实际受支撑。'],
        ['三支试管分别占据三个不同架孔，管底落在架内支撑面上且基本竖直；不要求对应隐藏身份。'],
        ['瓶身进入防转座后，独立盖实际旋出并在盖托放稳。',
         '盖重新旋合并实际接触瓶口，瓶子最终回到岛台归位区。'],
        ['称量杯实际放在平台上，用药匙装载、运输并倾出独立颗粒。',
         '杯内净样本质量达到资产声明目标（首版6±0.6克）；洒落不超过1克。',
         '名义质量来自已知颗粒质量及实际杯内库存，读数须在杯稳定且不被手或药匙托住时确认。',
         '任务平台屏幕以克显示净颗粒质量（扣除28克空舟），连续稳定0.3秒才出数；无舟、托举、移动或换料时显示横线。读数不要求先达到目标或药匙归位，也不代表操作顺序已合格。',
         '完成窗口内杯内颗粒身份保持，竖直不弹跳；允许杯底缓慢横滚，不要求球的朝向固定。'],
        ['供液必须来自吸嘴实际浸没时的活塞抽拉；空吸不增加水量。',
         '两个接收杯分别含8至12毫升；累计洒漏不超过2毫升，未决飞行水不超过0.5毫升。',
         '水量使用明确声明的活塞有限体积近似：源杯、工具、接收杯、飞行水及洒漏总量守恒。'],
    )[index]
    contract = {
        'schema': 'physical_task_referee_contract.v1', 'task_id': task_id,
        'instruction_zh': INSTRUCTIONS[index],
        'phases': {'working': '完成本任务公开操作。', 'settling': '目标已达到，松手并等待稳定。'},
        'reason_descriptions': {
            'reset_baseline_not_scored': '初始化只建立基准，不得分。',
            'task_in_progress': '操作或目标尚未完成，可继续调整。',
            'waiting_for_release_and_stability': '需松手并保持支撑稳定。',
            'success': '目标、操作历史、释放及稳定条件通过。',
            'spill_limit_exceeded': '累计洒漏超过公开上限。',
            'object_fell': '任务器具落地或离开可恢复台面。',
            'native_evidence_unknown': '原生观测缺失、不连续或不一致，本次不评分。',
            'physics_safety_stop': '明显穿透等物理异常，和任务失败分开记录。',
            'incomplete_budget': '预算耗尽，任务未完成。',
        },
        'success_conditions': deepcopy(goals)+['全部相关器具真实受支撑，机器人释放后稳定0.3秒。'],
        'recovery': ['允许在台面上重新抓取、纠正错放、重新插入和补充未达目标的样本。',
                     '洒落后回收不抵扣累计洒漏；预算耗尽记未完成。'],
        'thresholds': thresholds(task_id),
        'outcomes': {'ongoing': '可继续操作。', 'success': '有效证据确认完成。',
            'task_failure': '物体落地或明显洒漏超限。', 'evaluator_invalid': '证据故障，不评分。',
            'safety_stop': '物理异常停止。', 'budget_exhaustion': '未完成。', 'cancelled': '取消，不制造失败。'},
        'physical_actions': '任务执行仅控制原机器人；场景物物台架的显式工具运动边界单独记录，不计机器人成功。',
        'observation_scope': ('公开任务规则、静态标称物品质量和粗粒度阶段。Task4任务平台数字屏可由RGB-D相机观察；'
            '它显示声明的库存质量，不是测力传感器或机器人持物负载。真实位姿、接触和逐颗物料账本只供私有裁判读取。'
            '额外桌面陈设为固定背景，不是目标或可抓物体，也不是任务计分支撑面。'),
        'object_properties': object_properties(task_id),
    }
    return {'schema': 'laboratory_rules.v2', 'task_id': task_id,
        'instruction_zh': INSTRUCTIONS[index], 'referee': contract, 'reset_credit': False,
        'scope': 'Scene objects and object-object feasibility; hand-object interaction is collected separately.',
        'randomization': 'Frozen physical reset; seed is recorded without silently changing assets.'}


def evaluation_configuration(task_id, metadata, seed=0, physics_dt_s=.0005, room_support_paths=None):
    rules = public_rules(task_id, seed)
    if (type(physics_dt_s) not in (int, float) or not math.isfinite(physics_dt_s)
            or not 0 < physics_dt_s <= .002
            or abs(.02 / physics_dt_s-round(.02/physics_dt_s)) > 1.e-8):
        raise ValueError('Invalid native period')
    if metadata.get('task_id') != task_id or not metadata.get('actors'):
        raise ValueError('Task and physical inventory differ')
    objects = {row['id']: row for row in metadata['objects']}
    if set(objects) != set(metadata['actors']):
        raise ValueError('Every dynamic body must have metadata')
    for name, row in objects.items():
        if (row['body_path'] != metadata['actors'][name] or len(row['local_bounds_m']) != 2
                or len(row['initial_pose_xyzw']) != 7 or row['mass_kg'] <= 0):
            raise ValueError('Incomplete physical body metadata')
    declared = {row['id']: row['nominal_mass_kg'] for row in rules['referee']['object_properties']}
    if (set(declared) != set(objects) or any(not math.isclose(objects[k]['mass_kg'], mass,
            rel_tol=0., abs_tol=1.e-12) for k, mass in declared.items())):
        raise ValueError('Public nominal object masses differ from the frozen dynamic asset')
    return {'schema': 'laboratory_evaluation.v1', 'task_id': task_id,
        'episode_seed': seed, 'physics_dt_s': float(physics_dt_s),
        'metadata': deepcopy(metadata), 'room_support_paths': deepcopy(room_support_paths or {}),
        'thresholds': thresholds(task_id),
        'public_rules_sha256': digest(rules)}


def bind_task_rules(task):
    task['public_rules_sha256'] = digest(task['public_rules'])
    task['public_referee_contract_sha256'] = digest(task['public_rules']['referee'])
    task['public_rules_file'] = 'rules.json'
    return task
