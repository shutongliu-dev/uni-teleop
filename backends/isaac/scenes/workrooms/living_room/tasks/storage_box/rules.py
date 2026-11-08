"""Practical storage-box completion contract, independent of robot hand count."""
from copy import deepcopy
import hashlib
import json

INSTRUCTION = '旋开收纳盒前侧的两个金属卡扣，打开盒盖，把盒内蓝色小盒取出放入右侧收集盘，再关盖并把两个卡扣旋回；全部松手放稳。'
THRESHOLDS = {'stable_s': .3, 'max_time_s': 180., 'max_actions': 9000,
    'lid_open_deg': 45., 'catch_open_deg': 85., 'lid_closed_deg': 5., 'catch_closed_deg': 8.,
    'maximum_linear_speed_m_s': .02, 'maximum_angular_speed_rad_s': .15,
    'stable_position_range_m': .003, 'stable_orientation_range_rad': .04,
    'maximum_penetration_m': .003, 'tray_bottom_tolerance_m': .004}
PHASES = {'opening': '旋开卡扣并打开盒盖。', 'transfer': '取出蓝色小盒并放入右侧收集盘。',
    'closing': '关盖并旋回两个卡扣。', 'settling': '松手并等待稳定。',
    'completed': '物件已收集，盒盖与卡扣关闭，松手稳定。', 'evidence_invalid': '原生物理证据不完整。',
    'failed': '物件跌落或出现严重穿透。', 'incomplete': '预算结束，任务尚未完成。'}
REASONS = {**PHASES, 'reset_baseline_not_scored': '初态不计分。',
    'robot_operation_required': '需要机器人实际参与操作。', 'native_evidence_unknown': '原生证据缺失或不连续。',
    'object_fell': '任务物件跌到桌面以下。', 'physics_safety_stop': '任务物件出现严重穿透。',
    'incomplete_budget': '预算结束，任务未完成。'}


def public_rules(seed=0, *, asset_metadata=None):
    contract = {'schema':'physical_task_referee_contract.v1','task_id':'storage_box',
        'instruction_zh':INSTRUCTION,'phases':PHASES,'reason_descriptions':REASONS,
        'success_conditions':[
            '两个卡扣实际旋开且盒盖实际打开，不能穿过仍关闭的卡扣。',
            '蓝色小盒整体放在收集盘内，由真实盘底接触支撑。',
            '盒盖关闭到5度以内，两个卡扣各回到8度以内；不能只把物件移出而留盖或卡扣打开。',
            '机器人实际参与操作；所有任务物体脱离机器人接触并稳定至少0.3秒。'],
        'recovery':['允许重抓、调整、重新开盖或重新放置。','两个卡扣先后及使用哪只手不限，可借助桌面支撑。'],
        'thresholds':THRESHOLDS,
        'outcomes':{k:v for k,v in (
            ('ongoing','可以继续。'),('success','已完成。'),('task_failure','物件跌落。'),
            ('evaluator_invalid','证据不完整，不评分。'),('safety_stop','物理异常停止。'),
            ('budget_exhaustion','未完成。'),('cancelled','取消。'))},
        'physical_actions':'通过普通机器人关节动作操作真实物体；可单手借助桌面，也可左右手配合。不得直接写物体位姿或卡扣状态。',
        'observation_scope':'机构、目标和标称物性公开；物体真值、接触与连续稳定证据为裁判私有。',
        'object_properties':[{'id':name,'label':label,'nominal_mass_kg':mass,'basis':'项目几何与未标定仿真先验。'}
            for name,label,mass in [('box','收纳盒盒体',.8),('lid','盒盖',.12),('left_catch','左旋转卡扣',.025),
                                   ('right_catch','右旋转卡扣',.025),('item','蓝色小盒',.08)]]}
    if asset_metadata is not None and 'geometry_variant' in asset_metadata:
        import math
        if asset_metadata['geometry_variant'].get('schema') != 'living_storage_box_geometry_variant.v1':
            raise ValueError('Unknown storage-box geometry variant')
        masses = asset_metadata['masses_kg']
        if set(masses) != {r['id'] for r in contract['object_properties']} or any(
                type(m) not in (int, float) or not math.isfinite(m) or m <= 0 for m in masses.values()):
            raise ValueError('Finite positive physical masses for every storage actor are required')
        for row in contract['object_properties']:
            row['nominal_mass_kg'] = masses[row['id']]
    return deepcopy({'schema':'living_storage_box_rules.v1','instruction_zh':INSTRUCTION,
                     'competitive':False,'referee':contract,'reset_credit':False})


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def evaluation_configuration(*, seed=0, physics_dt_s=.0005, asset_metadata=None):
    if type(seed) is not int or not 0 <= seed < 2**32 or asset_metadata is None:
        raise ValueError('An unsigned seed and exact asset metadata are required')
    c = {'schema':'living_storage_box_evaluation.v1','seed':seed,'physics_dt_s':float(physics_dt_s),
        'control_period_s':.02,'thresholds':deepcopy(THRESHOLDS),
        'actors':deepcopy(asset_metadata['actors']), 'initial_positions_m':deepcopy(asset_metadata['initial_positions_m']),
        'item_size_m':deepcopy(asset_metadata['item_size_m']),
        'tray_floor_path':asset_metadata['tray_floor_path'], 'tray_floor_z_m':asset_metadata['tray_floor_z_m'],
        'tray_inside_xy_m':deepcopy(asset_metadata['tray_inside_xy_m']), 'robot_actor_prefix':'/World/Robot/',
        'public_rules_sha256':digest(public_rules(asset_metadata=asset_metadata))}
    c['contract_sha256'] = digest(c)
    return c
