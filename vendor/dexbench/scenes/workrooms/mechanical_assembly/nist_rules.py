"""Public, practical contracts for the deduplicated NIST board adaptations."""
from copy import deepcopy
import hashlib
import json
import math

TASK_IDS = (
    'nist_connector_insertion', 'nist_belt_drive_assembly',
    'nist_cable_routing', 'nist_wire_harness',
)
TASK_NUMBERS = dict(zip(TASK_IDS, range(6, 10)))
TITLES = ('连接器配对插接', '带轮与皮带装配', '单根电缆绕障布线', '多线配对与线束拆除')
INSTRUCTIONS = (
    '将 USB 与 BNC 插头插入各自接口，允许重新对准和重插，最后松手放稳。',
    '装入带轮，将闭环皮带套在两带轮外并经过张紧区域，允许调整松紧，最后松手放稳。',
    '将单根电缆按顺序绕过导向区域，末端插入对应接口，避免抄近路，最后松手放稳。',
    '按颜色将两根导线的两端放入对应端子，沿共同导向区域整理成束并松手确认；'
    '随后从端子和导向处取出整组线束，放入收集托盘并松手放稳。',
)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
        ensure_ascii=True, allow_nan=False).encode()).hexdigest()


def thresholds(task_id):
    if task_id not in TASK_IDS:
        raise ValueError('Unknown NIST adaptation')
    return dict(stable_s=.3, pose_window_s=.02, maximum_linear_speed_m_s=.04,
        maximum_angular_speed_rad_s=.5, position_drift_m=.005,
        orientation_drift_rad=.15, minimum_movement_m=.012,
        minimum_contact_impulse_ns=1.e-10, maximum_penetration_m=.006,
        maximum_mating_gap_m=.0005,
        support_memory_s=.05, placement_tolerance_m=.005,
        fallen_height_m=.10, max_time_s=240., max_actions=12000)


def public_rules(task_id, seed=0):
    if task_id not in TASK_IDS or type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError('Known NIST task and uint32 seed required')
    index = TASK_IDS.index(task_id)
    goals = (
        ['两个不同接口的插头分别进入匹配接口；实际接触接口并受到支撑。',
         '只验证插接位置、朝向、接触和释放稳定，不声称电连接或 BNC 锁扣功能。'],
        ['带轮到位，实际闭环皮带围住两带轮且长度、平面位置及链节连接合理。',
         '皮带分别与两只带轮产生实际配合接触；台面承托不能代替带轮接触。',
         '允许合理松弛；不要求张紧轮持续压紧、精确张力标定或动力传动效率。'],
        ['实际电缆中心线按公开顺序经过各导向区域，末端进入对应接口。',
         '插接位置按插针上的配合点判断，允许端头自然倾斜；端头可由完整线缆及导向部件共同承托。',
         '允许线缆弯曲、少量悬空与重复调整；不能只把末端放对而绕过布线路径。'],
        ['颜色与端子配对正确，两根实际导线分别经过共同导向区域形成线束。',
         '先保持配对、布线和释放稳定 0.3 秒，再将全部导线及端头取出到收集托盘。',
         '不要求扎带锁紧、电气通断或线束内导线的严格毫米间距。'],
    )[index]
    reasons = {
        'reset_baseline_not_scored': '初始化只建立基准，不得分。',
        'task_in_progress': '目标尚未完成，可继续重插、调整和恢复。',
        'waiting_for_release_and_stability': '目标已达到，需释放并保持支撑稳定。',
        'harness_ready_for_removal': '线束配对与布线已稳定确认，可以拆除并放入收集托盘。',
        'success': '实际目标、所需操作历史、支撑与释放稳定均满足。',
        'object_fell': '任务物体实际落地。',
        'physics_safety_stop': '明显穿透等物理异常，与任务失败分别记录。',
        'native_evidence_unknown': '原生观测缺失、不连续或不一致，本次不评分。',
        'budget_exhausted': '预算耗尽，任务未完成。',
    }
    referee = {
        'schema': 'physical_task_referee_contract.v1', 'task_id': task_id,
        'instruction_zh': INSTRUCTIONS[index],
        'phases': {'working': '完成配对、装配或布线。', 'settling': '松手并等待稳定。',
                   'removing': '取出已确认的线束并放入收集托盘。'},
        'reason_descriptions': reasons,
        'success_conditions': deepcopy(goals)+['相关物体释放后保持稳定 0.3 秒；不能用初始化或摆拍得到任务分。'],
        'recovery': ['允许台面上重新抓取、错放纠正、退出重插和重新布线。',
                     '一般错位、暂时松弛和端子配错均可纠正，不立即判输。',
                     '掉落台面上的物体可以捡回；实际落地才记物体掉落。',
                     '预算耗尽记未完成；仿真或证据错误不制造物理失败。'],
        'thresholds': thresholds(task_id),
        'outcomes': {'ongoing': '继续操作。', 'success': '有效证据确认完成。',
            'task_failure': '物体实际落地。', 'evaluator_invalid': '证据故障，不评分。',
            'safety_stop': '物理异常停止。', 'budget_exhaustion': '未完成。',
            'cancelled': '取消，不制造失败。'},
        'physical_actions': '任务动作仅控制原机器人关节；工程物物探针单独记录，不计机器人操作成功。',
        'observation_scope': '公开规则及粗粒度阶段；真实刚体位姿、接触和柔性链几何由私有原生裁判读取。',
    }
    return {'schema': 'mechanical_nist_rules.v1', 'task_id': task_id,
        'task_number': TASK_NUMBERS[task_id], 'source_board': index+1,
        'instruction_zh': INSTRUCTIONS[index], 'referee': referee,
        'reset_credit': False,
        'source_scope': 'Deduplicated task-board adaptation, not the full official NIST benchmark.',
        'simulation_scope': 'Passive connected rigid-link approximation for flexible parts; no electrical or latch qualification.',
        'randomization': 'Frozen physical reset; seed records identity without changing source geometry.'}


def _finite_vector(value, size):
    if (not isinstance(value, (list, tuple)) or len(value) != size
            or any(type(x) not in (int, float) or not math.isfinite(x) for x in value)):
        raise ValueError('Finite metadata vector required')


def evaluation_configuration(task_id, metadata, seed=0, physics_dt_s=.0005, room_support_paths=None):
    rules = public_rules(task_id, seed)
    if (type(physics_dt_s) not in (int, float) or not math.isfinite(physics_dt_s)
            or not 0 < physics_dt_s <= .002
            or abs(.02/physics_dt_s-round(.02/physics_dt_s)) > 1.e-8):
        raise ValueError('Invalid native period')
    if metadata.get('task_id') != task_id or not metadata.get('actors'):
        raise ValueError('Task and native inventory differ')
    objects = {row['id']: row for row in metadata['objects']}
    if len(objects) != len(metadata['objects']) or set(objects) != set(metadata['actors']):
        raise ValueError('Every dynamic body must have unique metadata')
    for name, row in objects.items():
        if row['body_path'] != metadata['actors'][name] or row['mass_kg'] <= 0:
            raise ValueError('Incomplete native body metadata')
        _finite_vector(row['initial_pose_xyzw'], 7)
        if len(row['local_bounds_m']) != 2:
            raise ValueError('Body bounds required')
        for bound in row['local_bounds_m']:
            _finite_vector(bound, 3)
        if any(lo > hi for lo, hi in zip(*row['local_bounds_m'])):
            raise ValueError('Invalid body bounds')
    fixtures = metadata['fixtures']
    if set(fixtures) != set(metadata['fixture_metadata']):
        raise ValueError('Every fixed fixture requires metadata')
    for row in metadata['fixture_metadata'].values():
        _finite_vector(row['initial_pose_xyzw'], 7)
    p = metadata['task_parameters']
    known = set(objects) | set(fixtures)
    for role, paths in (room_support_paths or {}).items():
        known.update('environment_'+role+'_'+str(i) for i in range(len(paths)))
    required = p['required_moved_ids']
    if not required or not set(required) <= set(objects):
        raise ValueError('Moved-body operation evidence required')
    covered = set()
    for goal in p.get('seating_goals', []):
        if goal['actor'] not in objects or not goal['support_ids'] or not goal['contact_ids']:
            raise ValueError('Seating requires an actor, support and mating contact')
        if not set(goal['support_ids']+goal['contact_ids']) <= known:
            raise ValueError('Unknown seating contact or support')
        _finite_vector(goal['target_position_m'], 3)
        _finite_vector(goal.get('actor_local_point_m', [0., 0., 0.]), 3)
        _finite_vector(goal['target_quaternion_wxyz'], 4)
        _finite_vector(goal['position_tolerance_m'], 3)
        if (min(goal['position_tolerance_m']) <= 0 or goal['orientation_tolerance_rad'] <= 0
                or abs(sum(x*x for x in goal['target_quaternion_wxyz'])-1) > 1.e-4):
            raise ValueError('Invalid seating tolerance or quaternion')
        covered.add(goal['actor'])
    for group in p.get('flexible_groups', []):
        ids = group['actor_ids']
        if (len(ids) < 2 or len(set(ids)) != len(ids) or not set(ids) <= set(objects)
                or type(group['closed']) is not bool or group['maximum_neighbor_distance_m'] <= 0
                or not group['support_ids'] or not set(group['support_ids']) <= known
                or not 0 < group.get('minimum_support_fraction', .04) <= 1):
            raise ValueError('Invalid passive flexible group')
        offsets = group.get('centerline_offsets_m', [[0., 0., 0.]]*len(ids))
        if len(offsets) != len(ids):
            raise ValueError('Every centerline body needs one local point')
        for offset in offsets:
            _finite_vector(offset, 3)
        for gate in group.get('route_gates', []):
            _finite_vector(gate['center_m'], 3)
            if gate['radius_m'] <= 0:
                raise ValueError('Route gate radius must be positive')
        for target in group.get('enclosed_targets', []):
            _finite_vector(target['center_m'], 3)
            if target['radius_m'] < 0 or not group['closed']:
                raise ValueError('Enclosed targets require a closed belt')
        mating = group.get('required_mating_contact_ids', [])
        if (not isinstance(mating, (list, tuple)) or len(mating) != len(set(mating))
                or not set(mating) <= known or set(mating) & set(ids)):
            raise ValueError('Invalid flexible mating-contact targets')
        if task_id == 'nist_belt_drive_assembly' and len(mating) < 2:
            raise ValueError('Belt assembly requires two distinct pulley contact targets')
        if group.get('perimeter_range_m') is not None:
            _finite_vector(group['perimeter_range_m'], 2)
            if not 0 < group['perimeter_range_m'][0] <= group['perimeter_range_m'][1]:
                raise ValueError('Invalid belt length interval')
        covered.update(ids)
    group_by_id = {group['id']: group for group in p.get('flexible_groups', [])}
    if len(group_by_id) != len(p.get('flexible_groups', [])):
        raise ValueError('Flexible group identities must be unique')
    for goal in p.get('seating_goals', []):
        via = goal.get('support_via_flexible_group')
        if via is None:
            continue
        group = group_by_id.get(via)
        if (group is None or group['closed'] or goal['actor'] not in
                (group['actor_ids'][0], group['actor_ids'][-1])):
            raise ValueError('Flexible support is only valid for an endpoint of its declared open cable')
    if not covered:
        raise ValueError('At least one physical goal required')
    if task_id == 'nist_wire_harness':
        removal = p['removal']
        if set(removal['actor_ids']) != set(objects) or not removal['support_ids']:
            raise ValueError('Removal must include the complete harness')
        if not set(removal['support_ids']) <= known:
            raise ValueError('Unknown removal support')
        _finite_vector(removal['center_m'], 3)
        _finite_vector(removal['half_extents_m'], 3)
        if min(removal['half_extents_m']) <= 0:
            raise ValueError('Invalid collection tray region')
    return {'schema': 'mechanical_nist_evaluation.v1', 'task_id': task_id,
        'episode_seed': seed, 'physics_dt_s': float(physics_dt_s),
        'metadata': deepcopy(metadata), 'room_support_paths': deepcopy(room_support_paths or {}),
        'thresholds': thresholds(task_id), 'public_rules_sha256': digest(rules)}


def bind_task_rules(task):
    task['public_rules_sha256'] = digest(task['public_rules'])
    task['public_referee_contract_sha256'] = digest(task['public_rules']['referee'])
    task['public_rules_file'] = 'rules.json'
    return task
