"""Public, practical kitchen contracts and immutable asset-bound configuration.

No observation, action policy, simulator import or controller parameter is here.
Thresholds express useful placement, not a prescribed target grasp or trajectory.
"""
from copy import deepcopy
import hashlib
import json
import math

TASK_IDS = ('breakfast_serving', 'produce_transfer', 'utensil_sorting', 'pot_lid_placement',
            'pantry_stocking', 'table_clearing')
RULES_VERSION = 'kitchen_rules.v2'
GROUPED_RULES_VERSION = 'kitchen_rules.v3'

# Public category identities are checked against the frozen physical asset.
# Each group has a real receiving fixture, not a hidden exact target pose.
GROUPED_INVENTORIES = {
    'pantry_stocking': {
        'bottles': ('bottles_bin', ('olive_oil', 'vinegar')),
        'seasonings': ('seasonings_bin', ('salt', 'pepper', 'paprika')),
        'dry_goods': ('dry_goods_bin', ('rice_carton', 'pasta_carton', 'tomato_can')),
    },
    'table_clearing': {
        'dishware': ('dishes_tray', ('dinner_plate', 'side_plate', 'cereal_bowl', 'small_bowl')),
        'drinkware': ('drinkware_tray', ('mug', 'tea_cup')),
        'cutlery': ('cutlery_tray', ('fork', 'spoon')),
    },
}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
        ensure_ascii=True, allow_nan=False).encode()).hexdigest()


def thresholds(task_id):
    if task_id not in TASK_IDS:
        raise ValueError('Unknown kitchen task: '+str(task_id))
    result = dict(stable_s=.4, stability_pose_window_s=.020, maximum_linear_speed_m_s=.025,
        maximum_angular_speed_rad_s=.25, maximum_stable_position_range_m=.004,
        maximum_stable_orientation_range_rad=.08, minimum_movement_m=.02,
        contact_gap_tolerance_m=.004, maximum_penetration_m=.004,
        support_height_tolerance_m=.008, containment_tolerance_m=.002,
        minimum_contact_impulse_ns=1.e-10, support_memory_s=.04,
        sleep_position_tolerance_m=.0005, sleep_orientation_tolerance_rad=.01,
        reset_position_tolerance_m=.025, reset_orientation_tolerance_rad=.4,
        upright_tilt_max_rad=math.radians(15), utensil_tilt_max_rad=math.radians(25),
        utensil_overlap_height_max_m=.018, source_loading_stable_s=.3,
        carry_clearance_m=.015, carry_net_displacement_min_m=.50,
        carry_payload_contact_gap_s=.06, carry_tilt_max_rad=math.radians(20),
        pot_center_tolerance_m=.035, pot_upright_tilt_max_rad=math.radians(10),
        lid_axis_tilt_max_rad=math.radians(8), lid_radial_offset_max_m=.012,
        lid_seat_height_tolerance_m=.008, pot_before_lid_stable_s=.3,
        max_time_s=120., max_actions=6000)
    if task_id in GROUPED_INVENTORIES:
        result['max_time_s'] = 240.
    return result


_TEXT = {
    'breakfast_serving': {
        'instruction': '先在源区把餐盘、空杯及两块面包装好：面包放盘内，盘和杯放托盘内；端起整盘送至目标区，全部松手放稳。',
        'phases': {'loading': '在源区完成配盘并放手确认。', 'carrying': '将带载托盘实际端离台面，送往目标区域。',
                   'settling': '目标区放下整盘，松手等待稳定。'},
        'success': ['源区配盘完整并释放稳定 0.3 秒。', '同一趟带载搬运实际离台至少 15 毫米，水平净移动至少 0.5 米；不计推滑或在终点重新拼装。',
                    '面包留在盘内，杯和盘留在托盘内，载荷由容器实际承托；允许小幅被动滑动和短暂接触微振动。',
                    '托盘在宽松标记区域内受台面支撑，杯直立，所有物体释放稳定 0.4 秒；无精确杯柄朝向或厘米级摆盘点要求。'],
        'recovery': ['可在源区调整摆放、重抓或放回源区重新端送。', '中途放到安全台面、载荷离盘或手直接扶载荷会清除本趟搬运信用，可回源区重试；不判永久失败。'],
    },
    'produce_transfer': {
        'instruction': '把三个番茄和三根短胡萝卜按类别分进对应浅盘，全部松手后放稳。',
        'phases': {'sorting': '将全部果蔬放入对应类别盘。', 'settling': '确认果蔬在正确盘内受支撑，松手等待稳定。'},
        'success': ['六件果蔬全部容纳于各自类别盘内，并实际接触盘底。', '物体可相互接触；不要求精确摆位或朝向，但不能跨盘沿、悬空或仅由其他果蔬托住。', '六件全部释放稳定 0.4 秒。'],
        'recovery': ['可从错误盘取回、重新摆放，或拾回仍在台面上的果蔬。'],
    },
    'utensil_sorting': {
        'instruction': '把两把勺、两把叉、两把钝餐刀放入各自收纳格，全部松手放稳。',
        'phases': {'sorting': '餐具按类别进入对应格。', 'settling': '确认餐具在格内受支撑，松手等待稳定。'},
        'success': ['六件餐具全部位于正确类别格内；长轴以实际容纳为准，柄朝两端均可。', '同类餐具可并排或轻微搭叠，只要有实际支撑链并留在格内；不能骑隔板、跨格或悬空。', '六件全部释放稳定 0.4 秒。'],
        'recovery': ['可取回错格餐具、调整角度、分开拥挤餐具后重试；没有严格手柄方向惩罚。'],
    },
    'pot_lid_placement': {
        'instruction': '先把空锅放稳在指定冷炉位，再把独立锅盖盖到锅口，锅和盖都松手放稳。',
        'phases': {'placing_pot': '把锅放到冷炉位并释放稳定。', 'seating_lid': '把独立锅盖落在锅口。', 'settling': '锅盖实际承托且基本盖住锅口，松手等待稳定。'},
        'success': ['锅离开源位，落在指定冷炉位中心 35 毫米范围内，直立且由该炉位实际支撑 0.3 秒。',
                    '独立锅盖实际由锅口承托：允许 12 毫米偏心、8 度倾角和 8 毫米正常高度误差。几何接近但无接触不算盖好。',
                    '锅和盖全部释放稳定 0.4 秒；不用严格对齐锅耳或锅盖旋转角。'],
        'recovery': ['可重抓偏心的锅、重新放盖，或从安全操作台与冷灶面取回盖子继续。'],
    },
    'pantry_stocking': {
        'instruction': '将源垫上的八件可移动密封食品分类收好：橄榄油和醋进深蓝色长盒，盐、胡椒、红椒粉进奶油色短盒，米盒、意面盒和番茄罐头进木色盒；全部松手放稳。吧台外缘的固定用品是背景陈设，不计分。',
        'phases': {'sorting': '将八件密封食品按公开类别放入对应收纳盒。', 'settling': '确认全部食品在正确盒内真实受支撑，松手等待稳定。'},
        'success': ['深蓝色长油醋瓶盒 bottles_bin：橄榄油瓶 olive_oil、醋瓶 vinegar；奶油色短调味品盒 seasonings_bin：盐罐 salt、胡椒罐 pepper、红椒粉罐 paprika；木色干货盒 dry_goods_bin：米盒 rice_carton、意面盒 pasta_carton、番茄罐头 tomato_can。只有源垫上这八件可移动物品计分。',
                    '每件物体的完整碰撞外形都在对应盒内，并由盒底或同盒内已承托的食品真实支撑；允许合理堆放，不要求指定槽位、横向位置或标签朝向。',
                    '油醋瓶和调味罐保持基本直立（倾角不超过 15 度）；密封米盒、意面盒和罐头可平放。无需打开包装、倒料或按大小排列。',
                    '八件全部离开各自初始位置至少 20 毫米，并同时释放稳定 0.4 秒；额外固定厨房陈设不参与计分。'],
        'recovery': ['可取回错盒物品、扶正瓶罐、拆开不稳定堆放、重抓或从安全台面拾回后重试。'],
    },
    'table_clearing': {
        'instruction': '收走源垫上的八件可移动餐具：大餐盘、小餐盘、麦片碗、小碗放到奶油色大盘，马克杯、茶杯放到蓝色盘，叉和勺放到木色盘；全部松手放稳。吧台外缘的固定用品是背景陈设，不计分。',
        'phases': {'sorting': '将八件碗盘、杯具和叉勺按类别收进三个接收盘。', 'settling': '确认全部餐具在正确盘内真实受支撑，松手等待稳定。'},
        'success': ['奶油色大碗盘接收盘 dishes_tray：大餐盘 dinner_plate、小餐盘 side_plate、麦片碗 cereal_bowl、小碗 small_bowl；蓝色杯具盘 drinkware_tray：马克杯 mug、茶杯 tea_cup；木色餐具盘 cutlery_tray：叉 fork、勺 spoon。只有源垫上这八件可移动物品计分。',
                    '所有物体完整落在所属接收盘内，不能跨盘沿或靠外部陈设支撑；可在同一接收盘内并排、搭叠或堆放，只要有真实承托链最终连到该盘底。',
                    '碗盘开口朝上，杯具基本直立（倾角不超过 15 度）；杯柄和叉勺朝向自由，不要求叠盘顺序、精确槽位或洗净状态。',
                    '八件全部离开各自初始位置至少 20 毫米，并同时释放稳定 0.4 秒；额外固定厨房陈设不参与计分。'],
        'recovery': ['错盘可取回；可拆开倾斜堆叠、重新摆放或从安全台面取回物品继续，不因一次错放永久失败。'],
    },
}


def public_rules(task_id, seed=0):
    if task_id not in TASK_IDS or type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError('Known task and uint32 seed are required')
    text = _TEXT[task_id]
    reasons = {
        'reset_baseline_not_scored': '初始化只建立基准，不计任务分。',
        'task_in_progress': '尚有物体或阶段未完成，可继续操作。',
        'source_loading_required': '在源区完整配盘并释放后再开始端送。',
        'loaded_carry_required': '需要完整的带载离台搬运证据，可回源区重试。',
        'pot_before_lid_required': '先把锅在冷炉位放稳，再完成盖盖。',
        'waiting_for_release_and_stability': '物体已接近目标，需全部松手并稳定。',
        'success': '所有任务条件和释放稳定窗口通过。',
        'object_left_recoverable_workspace': '物体落地或离开可恢复操作区域。',
        'native_evidence_unknown': '缺失、无效或不连续原生证据；本次无有效任务判定，不记物理失败。',
        'incomplete_budget': '预算耗尽时任务未完成；不等于物体不可恢复。',
        'physics_safety_stop': '有效原生证据显示超过声明穿透限值；单独记录物理停止。',
    }
    stability = '稳定速度由真实位姿相隔 20 毫秒的变化计算，窗口在每个原生子步向前滑动；线速不超过 25 毫米/秒、角速不超过 0.25 弧度/秒。整个连续稳定段仍逐原生子步限制位置漂移 4 毫米、转角漂移 0.08 弧度。求解器瞬时速度脉冲保留为诊断，不单独判为物体移动；任何机器人接触、目标或支撑失效均重新计时，缺失原生证据不作成功判定。'
    contract = {'schema': 'physical_task_referee_contract.v1', 'task_id': task_id,
        'instruction_zh': text['instruction'], 'phases': deepcopy(text['phases']),
        'reason_descriptions': reasons, 'success_conditions': deepcopy(text['success'])+[stability],
        'recovery': deepcopy(text['recovery'])+['物体仍在安全台面时，错放、滑动和重新抓取均可重试。'],
        'thresholds': thresholds(task_id),
        'outcomes': {'ongoing': '尚未完成，可在预算内继续。', 'success': '完整有效证据确认全部任务条件。',
            'task_failure': '完整有效证据确认物体落地或离开可恢复区域。',
            'evaluator_invalid': '证据 unknown，无分数且不记正常物理失败。',
            'safety_stop': '物理异常停止，和任务失败分开记录。',
            'budget_exhaustion': '未完成，由运行器记录预算停止。', 'cancelled': '取消或基础设施故障，不制造任务失败。'},
        'physical_actions': '只控制机器人，允许重抓与重试；禁止物体位姿回放、吸附、焊接、直接外力或隐藏支撑。诊断台架不计机器人任务成绩。',
        'observation_scope': '公开静态规则和粗粒度阶段；真值位姿、接触、支撑记忆及逐原生子步判据只供私有裁判。'}
    return {'schema': GROUPED_RULES_VERSION if task_id in GROUPED_INVENTORIES else RULES_VERSION,
        'task_id': task_id, 'instruction_zh': text['instruction'], 'referee': contract,
        'reset_credit': False, 'randomization': 'This candidate uses the frozen authored reset; the seed does not silently alter physical geometry.',
        'evidence_scope': 'Complete read-only native poses and contact pairs, unscored baseline; geometry alone is insufficient.',
        'qualification': 'Implemented scene candidate. CPU logic tests and scene validation do not establish robot manipulation success.'}


def evaluation_configuration(task_id, metadata, seed=0, physics_dt_s=.0005):
    public_rules(task_id, seed)
    if isinstance(physics_dt_s, bool) or not math.isfinite(physics_dt_s) or not 0 < physics_dt_s <= .002:
        raise ValueError('Native physics period must be finite and no greater than 2 ms')
    if abs(.02/physics_dt_s-round(.02/physics_dt_s)) > 1.e-8:
        raise ValueError('Control interval must contain whole native substeps')
    m = deepcopy(metadata)
    if m.get('task_id') != task_id:
        raise ValueError('Asset metadata belongs to another task')
    actors = m['actors']
    if not actors or len(set(actors.values())) != len(actors) or any(not x.startswith('/World/') for x in actors.values()):
        raise ValueError('Distinct absolute native actor paths are required')
    rows = m['objects']
    objects = rows if isinstance(rows, dict) else {row['id']: row for row in rows}
    if set(objects) != set(actors):
        raise ValueError('Metadata must describe every dynamic task body')
    for name, obj in objects.items():
        if obj['body_path'] != actors[name] or len(obj['local_bounds_m']) != 2 or len(obj['initial_pose_xyzw']) != 7:
            raise ValueError('Incomplete asset-bound body geometry')
    if task_id in GROUPED_INVENTORIES:
        inventory = GROUPED_INVENTORIES[task_id]
        expected_names = {name for _, names in inventory.values() for name in names}
        expected_targets = {kind: target for kind, (target, _) in inventory.items()}
        if set(objects) != expected_names or m['goals'].get('class_to_fixture') != expected_targets:
            raise ValueError('Frozen grouped inventory differs from public category rules')
        for kind, (target, names) in inventory.items():
            fixture = m['fixture_metadata'][target]
            if target not in m['fixtures'] or len(fixture['initial_pose_xyzw']) != 7:
                raise ValueError('Grouped target must have a real fixed fixture')
            bounds = fixture['interior_bounds_xy_m']
            if (len(bounds) != 2 or any(len(row) != 2 for row in bounds)
                    or any(not math.isfinite(float(v)) for row in bounds for v in row)
                    or any(bounds[0][i] >= bounds[1][i] for i in range(2))
                    or not math.isfinite(float(fixture['floor_top_local_z_m']))):
                raise ValueError('Grouped target requires finite nonempty interior geometry')
            orientation = 'upright' if kind in ('bottles', 'seasonings', 'dishware', 'drinkware') else 'any'
            for name in names:
                if (objects[name].get('category') != kind or objects[name].get('assigned_fixture') != target
                        or objects[name].get('placement_orientation') != orientation):
                    raise ValueError('Grouped object category, destination or orientation differs from public rules')
    return {'schema': 'kitchen_evaluation.v1', 'task_id': task_id, 'episode_seed': seed,
        'physics_dt_s': float(physics_dt_s), 'control_period_s': .02, 'actors': deepcopy(actors),
        'robot_actor_prefix': '/World/Robot/', 'metadata': m,
        'thresholds': thresholds(task_id), 'public_rules_sha256': digest(public_rules(task_id, seed))}


def bind_task_rules(task):
    task['public_rules_sha256'] = digest(task['public_rules'])
    task['public_referee_contract_sha256'] = digest(task['public_rules']['referee'])
    task['public_rules_file'] = 'rules.json'
    return task
