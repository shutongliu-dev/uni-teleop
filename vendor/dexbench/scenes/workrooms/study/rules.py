"""Frozen public rules and configuration for the first four study tasks.

Only static, authored task values enter this module. Runtime poses and contact
receipts remain private. The native observer establishes their trust boundary.
"""
from copy import deepcopy
from itertools import product
import hashlib
import json
import math
import re

RULES_VERSION = 'study_rules_v1'
CONFIGURATION_SCHEMA = 'study_task_configuration_v1'
OBSERVATION_SCHEMA = 'study_native_observation_v1'
TASK_IDS = ('stationery_sorting', 'book_stacking', 'book_retrieval', 'book_shelving')
THRESHOLDS = {
    'physics_dt_s': .0005, 'minimum_stable_s': .300,
    'maximum_linear_speed_m_s': .010, 'maximum_angular_speed_rad_s': .050,
    'maximum_position_drift_m': .001, 'maximum_orientation_drift_rad': .02,
    'maximum_up_axis_error_rad': math.radians(5.),
    'maximum_spine_axis_error_rad': math.radians(5.),
    'support_gap_m': .002, 'maximum_penetration_m': .001,
    'contact_point_tolerance_m': .002, 'minimum_support_impulse_ns': 1.e-10,
    'initial_position_tolerance_m': .0001, 'initial_orientation_tolerance_rad': .002,
}
PHASES = {'not_started': '等待初始状态', 'reset': '核对初始状态',
    'placement': '按身份、类别和位置摆放', 'support': '检查真实逐层承托',
    'release': '等待完全松手', 'settling': '等待共同稳定', 'complete': '完成',
    'failed': '物体落到地板', 'invalid': '证据不可用', 'stopped': '回合已停止'}
REASONS = {
    'not_sampled': '尚未收到初始原生状态。',
    'reset_baseline_not_scored': '初始状态已核对，初始摆放不计分。',
    'outside_target_slot': '物体整体尚未进入其身份和类别对应的目标区域。',
    'incorrect_orientation': '物体的朝上方向或书脊方向不符合要求。',
    'support_height_mismatch': '物体底部尚未贴合规定支撑面的高度。',
    'required_contact_support_missing': '缺少指定桌面、层板或下层书本的真实向上承托接触。',
    'robot_contact_present': '机器人仍接触任务物体，需要所有手完全释放。',
    'unexpected_contact': '任务物体仍依赖其他物体、柜壁或非指定支撑。',
    'contact_penetration': '当前接触穿透超过摆放容差，需要重新放置。',
    'still_moving': '任务物体仍在移动或转动。',
    'stability_drift': '共同稳定期间发生过大漂移，稳定时间重新计算。',
    'stability_window_incomplete': '所有物体须共同满足条件并连续稳定300毫秒。',
    'robot_action_required': '尚未有实际机器人动作，初始状态不计为完成。',
    'object_touched_floor': '完整原生证据确认任务物体接触地板，本回合失败。',
    'invalid_evidence': '原生采样、接触、身份或时间证据缺失或不一致，本回合无效。',
    'budget_exhausted': '时间或动作预算已用尽，结果为未完成。',
    'cancelled': '回合已取消，结果为未完成。',
    'success': '全部身份、摆放、支撑、释放及连续稳定条件通过。',
}


def number(value, name):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(name + ' must be a finite number')
    return float(value)


def vector(value, n, name):
    if not isinstance(value, (tuple, list)) or len(value) != n:
        raise ValueError(name + ' has the wrong dimension')
    return tuple(number(x, name) for x in value)


def quaternion(value):
    q = vector(value, 4, 'quaternion_wxyz')
    length = math.sqrt(sum(x*x for x in q))
    if abs(length-1.) > 1.e-4:
        raise ValueError('A unit wxyz quaternion is required')
    return tuple(x/length for x in q)


def rotation(value):
    w, x, y, z = quaternion(value)
    return ((1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)),
            (2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)),
            (2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)))


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
        ensure_ascii=True, allow_nan=False).encode()).hexdigest()


def _bounds(value, name):
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(name + ' must contain low/high 3D bounds')
    lo, hi = (vector(row, 3, name) for row in value)
    if any(a >= b for a, b in zip(lo, hi)):
        raise ValueError(name + ' must have positive volume')
    return lo, hi


def _axis(value, name):
    axis = vector(value, 3, name)
    if abs(sum(x*x for x in axis)-1.) > 1.e-8:
        raise ValueError(name + ' must be a unit direction')
    return axis


def _name(value):
    return isinstance(value, str) and re.fullmatch(r'[a-z][a-z0-9_]*', value)


def _reset_bounds(obj):
    """Transform the immutable local box, including all eight reset corners."""
    lo, hi = obj['local_bounds_m']
    pose = obj['initial_pose']
    r = rotation(pose['quaternion_wxyz'])
    points = [tuple(pose['position_m'][i] + sum(r[i][j]*p[j] for j in range(3))
                    for i in range(3))
              for p in product(*zip(lo, hi))]
    return tuple(min(p[i] for p in points) for i in range(3)), tuple(max(p[i] for p in points) for i in range(3))


def _upright_goal(obj, goal):
    return (_axis(obj['up_axis_local'], 'book up') == (0., 0., 1.) and
            _axis(obj['spine_axis_local'], 'book spine') == (1., 0., 0.) and
            _axis(goal['up_direction_world'], 'goal up') == (0., 0., 1.) and
            _axis(goal['spine_direction_world'], 'goal spine') == (1., 0., 0.))


def public_rules(task_id, configuration=None):
    if task_id not in TASK_IDS:
        raise ValueError('Unknown study task')
    instruction = {
        'stationery_sorting': '将粗铅笔、马克笔和橡皮按其身份与类别分别放入指定桌面区域。',
        'book_stacking': '将三本指定书按底到顶的指定顺序平叠，前封面朝上、书脊同向，逐层真实承托。',
        'book_retrieval': '从书架取出指定的一本书放到指定桌面区域；其余两本最终须保留在各自原书架位置。',
        'book_shelving': '将至少两本、涵盖不同类别的书分别放到各类别指定书架格中的独立区域，竖直放置且书脊朝格口。',
    }[task_id] + ' 所有任务物体均须完全松手，并在真实支撑下共同连续稳定至少300毫秒。'
    conditions = [
        '身份与类别取自冻结资产到原生刚体路径的映射；不能交换标签或用其他物体替代。',
        '每个物体的完整刚体局部包围盒变换后须全部位于指定三维区域；书本须满足指定朝上方向和书脊方向。',
        '每个原生物理步均须取得指定支撑的有符号向上接触冲量，接触点位于物体底部和实际支撑面。叠书必须存在从每本书逐层到桌面的支撑链。',
        '所有手和机器人部位均离开全部任务物体；非指定外部支撑、过深穿透或物体运动均重置共同稳定计时。',
        '连续原生状态、原生接触和稳定速度阈值完整覆盖300毫秒；初始状态不评分。']
    if configuration is not None:
        def label(ident):
            title = configuration['objects'][ident].get('title_zh')
            return f'{title}（{ident}）' if title else ident

        for ident, target in configuration['targets'].items():
            conditions.append(f"{label(ident)} ({target['category']}) → {target['slot_id']}; "
                f"完整范围 bounds_m={target['bounds_m']}; 支撑={target['support_id']}, "
                f"支撑高度={target['support_z_m']}m; 朝上={target.get('up_direction_world')}, "
                f"书脊={target.get('spine_direction_world')}。")
        if task_id == 'book_stacking':
            conditions.append('从底到顶的书本身份顺序：' + ' → '.join(map(label, configuration['stack_order'])) + '。')
        if task_id == 'book_retrieval':
            conditions.append('只取目标书 ' + label(configuration['retrieval_target_id']) +
                '；其余书 ' + ', '.join(map(label, configuration['protected_ids'])) + ' 须最终回到各自原书架区域。')
        if configuration.get('budget'):
            conditions.append('回合预算：仿真时间上限 ' + str(configuration['budget']['max_sim_time_s']) +
                ' 秒；机器人动作上限 ' + str(configuration['budget']['max_robot_actions']) + '；null 表示该项不限。')
            maximum = configuration['budget']['max_robot_actions']
            if maximum is not None:
                conditions.append(f'允许执行完整的前 {maximum} 个公开机器人动作；动作内部的连续物理子步不重复计数。宿主须在后续动作获准执行前阻止其进入物理仿真。')
    recovery = ['在预算内可重抓、纠正错位/类别/顺序、扶正书本；掉到桌上仍可捡回并继续。',
        '取书时误动其他书可放回原位；最终所有书均须满足其冻结目标。所有纠正仅通过真实机器人动作，重新累计完整共同稳定窗口。',
        '任一任务物体真实接触地板为明确不可恢复失败；原生证据缺失、跳步或时钟错误为裁判无效，不能伪判成功或正常任务失败。',
        '时间或动作预算耗尽及主动取消均为未完成；预算在同一步满足时优先于成功。']
    contract = {'schema': 'physical_task_referee_contract.v1', 'task_id': task_id,
        'instruction_zh': instruction, 'phases': deepcopy(PHASES),
        'reason_descriptions': deepcopy(REASONS), 'success_conditions': conditions,
        'recovery': recovery, 'thresholds': deepcopy(THRESHOLDS),
        'outcomes': {'ongoing': '尚未完成，可以在预算内继续恢复或重试。',
            'success': '完整物理证据满足全部条件，接受后不撤销。',
            'task_failure': '真实地板接触触发明确任务失败。',
            'evaluator_invalid': '原生证据无效；不按普通任务失败计分。',
            'safety_stop': '运行器的独立物理安全停止，不伪装成任务成功。',
            'budget_exhaustion': '预算耗尽，记录未完成。', 'cancelled': '用户取消，记录未完成。'},
        'physical_actions': '只控制原始机器人。禁止物体位姿写入、回放、吸附、焊接抓取和直接施加物体外力；诊断台架不计任务成绩。',
        'observation_scope': '规则、冻结目标和粗粒度裁判原因公开；运行时精确物体状态、接触数据和私有评分细节不公开。'}
    return {'schema': RULES_VERSION, 'task_id': task_id, 'referee': contract,
        'thresholds': deepcopy(THRESHOLDS), 'floor_drop_is_terminal': True,
        'observation_schema': OBSERVATION_SCHEMA,
        'qualification': 'Rules and native readback contract; no robot manipulation qualification is implied.'}


def build_configuration(task_id, objects, targets, supports, *, budget=None,
                        stack_order=None, retrieval_target_id=None, protected_ids=None):
    config = {'schema': CONFIGURATION_SCHEMA, 'task_id': task_id,
        'physics_dt_s': THRESHOLDS['physics_dt_s'], 'objects': deepcopy(objects),
        'targets': deepcopy(targets), 'supports': deepcopy(supports),
        'budget': deepcopy(budget) if budget is not None else
            {'max_sim_time_s': 120., 'max_robot_actions': None}}
    if stack_order is not None:
        config['stack_order'] = list(stack_order)
    if retrieval_target_id is not None:
        config['retrieval_target_id'] = retrieval_target_id
    if protected_ids is not None:
        config['protected_ids'] = list(protected_ids)
    config['rules'] = public_rules(task_id, config)
    config['configuration_sha256'] = canonical_hash(config)
    return validate_configuration(config)


def validate_configuration(configuration):
    c = deepcopy(configuration)
    if c.get('schema') != CONFIGURATION_SCHEMA or c.get('task_id') not in TASK_IDS:
        raise ValueError('Unsupported study configuration')
    payload = {k: v for k, v in c.items() if k != 'configuration_sha256'}
    if c.get('configuration_sha256') != canonical_hash(payload):
        raise ValueError('Frozen study configuration hash differs')
    task = c['task_id']
    if c.get('rules') != public_rules(task, c) or c['physics_dt_s'] != THRESHOLDS['physics_dt_s']:
        raise ValueError('Public rules or native physics period differ')
    objects, targets, supports = c['objects'], c['targets'], c['supports']
    count = len(objects) if isinstance(objects, dict) and task == 'book_shelving' else 3
    if (not isinstance(objects, dict) or len(objects) != count or count < 2 or
            not isinstance(targets, dict) or set(objects) != set(targets) or
            not isinstance(supports, dict) or not supports or
            any(not _name(k) for k in (*objects, *supports)) or set(objects) & set(supports)):
        raise ValueError('Distinct complete object, target and support identities are required')
    for name, support in supports.items():
        _bounds(support['bounds_m'], name+' support')
    slots = []
    for ident, obj in objects.items():
        if not _name(obj['category']):
            raise ValueError('Every object requires an immutable category')
        if 'title_zh' in obj and (not isinstance(obj['title_zh'], str) or
                not obj['title_zh'].strip() or len(obj['title_zh']) > 120 or
                any(ord(char) < 32 for char in obj['title_zh'])):
            raise ValueError('An optional frozen Chinese title must be nonempty plain text')
        _bounds(obj['local_bounds_m'], ident+' local bounds')
        vector(obj['initial_pose']['position_m'], 3, 'initial position')
        quaternion(obj['initial_pose']['quaternion_wxyz'])
        goal = targets[ident]
        if not _name(goal['slot_id']) or goal['category'] != obj['category']:
            raise ValueError('Target identity/category/slot must agree with its object')
        slots.append(goal['slot_id'])
        lo, hi = _bounds(goal['bounds_m'], ident+' goal')
        z = number(goal['support_z_m'], 'support height')
        if not lo[2] <= z < hi[2]:
            raise ValueError('Support height must be inside the target vertical bounds')
        supporting = goal['support_id']
        if supporting not in supports and supporting not in objects or supporting == ident:
            raise ValueError('Unknown or self-supporting goal')
        if supporting in supports and abs(supports[supporting]['bounds_m'][1][2]-z) > 1.e-9:
            raise ValueError('Target support height differs from actual fixed support')
        for prefix in ('up', 'spine'):
            local, world = obj.get(prefix+'_axis_local'), goal.get(prefix+'_direction_world')
            if (local is None) != (world is None) or (task != 'stationery_sorting' and local is None):
                raise ValueError('Book goals require both explicit local and world orientation axes')
            if local is not None:
                _axis(local, prefix+' local axis'); _axis(world, prefix+' goal direction')
        if task != 'stationery_sorting':
            if abs(sum(a*b for a,b in zip(obj['up_axis_local'], obj['spine_axis_local']))) > 1.e-8:
                raise ValueError('Book up and spine axes must be orthogonal')
            if abs(sum(a*b for a,b in zip(goal['up_direction_world'], goal['spine_direction_world']))) > 1.e-8:
                raise ValueError('Book target up and spine directions must be orthogonal')
    if len(set(slots)) != count:
        raise ValueError('Each physical object requires a distinct named target slot')
    if task in ('stationery_sorting', 'book_shelving'):
        names = list(objects)
        for i, name in enumerate(names):
            a = targets[name]['bounds_m']
            for other in names[i+1:]:
                b = targets[other]['bounds_m']
                if all(min(a[1][j], b[1][j])-max(a[0][j], b[0][j]) > 1.e-9 for j in range(3)):
                    raise ValueError('Separate physical slots must not overlap in volume')
    if task == 'stationery_sorting':
        if len({o['category'] for o in objects.values()}) != 3 or any(t['support_id'] != 'table' for t in targets.values()):
            raise ValueError('Three stationery categories require separate table slots')
    elif task == 'book_stacking':
        order = c.get('stack_order')
        if not isinstance(order, list) or len(order) != 3 or set(order) != set(objects):
            raise ValueError('The three physical book identities require an explicit order')
        for i, ident in enumerate(order):
            if targets[ident]['support_id'] != ('table' if i == 0 else order[i-1]):
                raise ValueError('Stack support graph must follow bottom-to-top identity order')
            if targets[ident]['up_direction_world'] != [0, 0, 1] or targets[ident]['spine_direction_world'] != targets[order[0]]['spine_direction_world']:
                raise ValueError('All stacked book covers face upward and spines share a direction')
            if (_axis(objects[ident]['up_axis_local'], 'front cover') != (0., 1., 0.) or
                    _axis(objects[ident]['spine_axis_local'], 'book spine') != (1., 0., 0.)):
                raise ValueError('The authored book front cover is local +Y and spine is local +X')
            if i:
                lower = objects[order[i-1]]['local_bounds_m']
                expected_z = targets[order[i-1]]['support_z_m'] + lower[1][1]-lower[0][1]
                if abs(targets[ident]['support_z_m']-expected_z) > 1.e-9:
                    raise ValueError('Stack support heights must follow actual immediately lower book thickness')
    elif task == 'book_retrieval':
        target, protected = c.get('retrieval_target_id'), c.get('protected_ids')
        if (target not in objects or not isinstance(protected, list) or len(protected) != 2 or
                len(set(protected)) != 2 or set(protected) != set(objects)-{target}):
            raise ValueError('One retrieval target and exactly two distinct protected books are required')
        if targets[target]['support_id'] != 'table' or any(not targets[n]['support_id'].startswith('shelf_') for n in protected):
            raise ValueError('Retrieve only the target to the table and retain both neighbors on shelves')
        for ident in protected:
            obj, goal = objects[ident], targets[ident]
            lo, hi = _reset_bounds(obj)
            if (not _upright_goal(obj, goal) or abs(lo[2]-goal['support_z_m']) > THRESHOLDS['support_gap_m'] or
                    any(lo[i] < goal['bounds_m'][0][i]-1.e-9 or hi[i] > goal['bounds_m'][1][i]+1.e-9 for i in range(3))):
                raise ValueError('Protected books must retain their own complete upright original shelf slot')
    elif (len({o['category'] for o in objects.values()}) < 2 or
          any(not t['support_id'].startswith('shelf_') for t in targets.values())):
        raise ValueError('At least two book categories require distinct shelf slots')
    elif any(not _upright_goal(objects[name], targets[name]) for name in objects):
        raise ValueError('Shelved books stand upright with the local +X spine toward the +X opening')
    # Every non-stack object must terminate directly at a fixed support.
    if task != 'book_stacking' and any(t['support_id'] not in supports for t in targets.values()):
        raise ValueError('Only the stacking task permits a movable support chain')
    budget = c['budget']
    if not isinstance(budget, dict) or set(budget) != {'max_sim_time_s', 'max_robot_actions'}:
        raise ValueError('Both budget limits must be declared, using null for unlimited')
    if budget['max_sim_time_s'] is not None and number(budget['max_sim_time_s'], 'time budget') <= 0:
        raise ValueError('Time budget must be positive')
    if budget['max_robot_actions'] is not None and (type(budget['max_robot_actions']) is not int or budget['max_robot_actions'] <= 0):
        raise ValueError('Action budget must be a positive integer')
    return c
