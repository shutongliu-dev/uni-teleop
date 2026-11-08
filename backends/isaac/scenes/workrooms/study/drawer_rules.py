"""Practical drawer retrieval rules; separate from the historical book contracts."""
from copy import deepcopy

from .rules import (canonical_hash, number, vector, quaternion, rotation,
                    OBSERVATION_SCHEMA, PHASES, REASONS)

TASK_ID = 'drawer_retrieval'
JOINT_PATH = '/World/StudyTask/DrawerSlide'
THRESHOLDS = {
    'physics_dt_s': .0005, 'minimum_stable_s': .300,
    'minimum_opening_m': .100, 'maximum_closed_gap_m': .012,
    'maximum_linear_speed_m_s': .025, 'maximum_angular_speed_rad_s': .12,
    'maximum_position_drift_m': .003, 'maximum_orientation_drift_rad': .06,
    'support_gap_m': .003, 'maximum_penetration_m': .002,
    'contact_point_tolerance_m': .004, 'minimum_support_impulse_ns': 1.e-10,
    'initial_position_tolerance_m': .0001, 'initial_orientation_tolerance_rad': .002,
}
SUPPORT_PATHS = {
    'table': '/World/Room/Geometry/worktop/Collider',
    'tray': '/World/StudyTask/Fixtures/target_floor/Collider',
    'floor': '/World/Room/Geometry/room_floor/Collider',
}


def public_rules(configuration=None):
    reasons = {**REASONS,
        'drawer_not_opened': '先将抽屉拉出至少10厘米，允许重新调整握持。',
        'notebook_not_extracted': '打开后取出笔记本；柜体内部的物品不计取出。',
        'drawer_not_closed': '笔记本取出后把抽屉推回，允许保留最多12毫米缝隙。',
        'drawer_still_moving': '抽屉仍在滑动，请松手后稍等。'}
    conditions = [
        '初态抽屉关闭，笔记本在抽屉内部；须在本回合真实打开至少100毫米后取出笔记本。',
        '将整本笔记本放到绿色目标托盘内；封面方向和托盘内具体位置不限，不要求中心对齐。',
        '最终抽屉回到关闭端12毫米范围内；笔记本由真实托盘底面承托，机器人完全松开两件动态体。',
        '两个动态体连续稳定至少300毫秒；抽屉由真实滑轨关节承载，不虚构其接触支撑。',
        '允许反复开关、重抓和调整；掉回抽屉或桌面可恢复，不要求固定轨迹或一次完成。',
        '原生每0.5毫秒读回滑轨、刚体与有符号接触；初态和工程台架不计任务成功。']
    if configuration:
        conditions.append('目标托盘完整包络 bounds_m=' + str(configuration['targets']['notebook']['bounds_m']) + '。')
    contract = {'schema': 'physical_task_referee_contract.v1', 'task_id': TASK_ID,
        'instruction_zh': '打开桌上抽屉，取出笔记本放入绿色托盘，再关回抽屉；允许重抓调整，松手稳定300毫秒。',
        'phases': {**PHASES, 'opening': '打开抽屉', 'closing': '关闭抽屉'},
        'reason_descriptions': reasons, 'success_conditions': conditions,
        'thresholds': deepcopy(THRESHOLDS),
        'recovery': ['预算180秒，动作次数不限；安全重抓、调整、桌面掉落均可继续。',
            '笔记本真实落地为任务失败；缺失原生证据为裁判无效；超时或取消为未完成。'],
        'outcomes': {'ongoing': '可继续或重试', 'success': '完成全部顺序和最终支撑释放条件',
            'task_failure': '物品真实落地', 'evaluator_invalid': '原生证据无效',
            'budget_exhaustion': '未完成', 'cancelled': '未完成', 'safety_stop': '独立物理安全停止'},
        'physical_actions': '只控制原始机器人；禁止改写任务物体、吸附、抓取焊接或直接施加任务物体外力。',
        'observation_scope': '公开规则和粗粒度阶段；精确运行时状态与接触留在服务私有裁判。'}
    return {'schema': 'study_drawer_rules.v1', 'task_id': TASK_ID, 'referee': contract,
        'thresholds': deepcopy(THRESHOLDS), 'observation_schema': OBSERVATION_SCHEMA,
        'floor_drop_is_terminal': True, 'qualification': 'Object and scene checks do not qualify robot manipulation.'}


def configuration():
    objects = {
        'drawer': {'category': 'drawer', 'title_zh': '胡桃木桌上抽屉',
            'local_bounds_m': [[-.160, -.135, -.030], [.1225, .135, .035]],
            'initial_pose': {'position_m': [.57,.23,.825], 'quaternion_wxyz': [1.,0.,0.,0.]}},
        'notebook': {'category': 'notebook', 'title_zh': '观物工作笔记本',
            'local_bounds_m': [[-.045,-.055,-.012], [.045,.055,.012]],
            'initial_pose': {'position_m': [.51,.23,.8172], 'quaternion_wxyz': [1.,0.,0.,0.]}},
    }
    c = {'schema': 'study_drawer_configuration.v1', 'task_id': TASK_ID,
        'physics_dt_s': .0005, 'objects': objects,
        'supports': {'table': {'bounds_m': [[.2,-1.,.70],[1.1,1.,.78]]},
                     'tray': {'bounds_m': [[.33,-.415,.78],[.65,-.065,.792]]}},
        'targets': {'notebook': {'slot_id': 'notebook_tray', 'category': 'notebook',
            'bounds_m': [[.342,-.403,.790],[.638,-.077,.890]], 'support_id': 'tray', 'support_z_m': .792}},
        'mechanism': {'joint_path': JOINT_PATH, 'closed_position_m': [.57,.23,.825],
            'opening_direction_world': [-1.,0.,0.], 'travel_m': .16,
            'cabinet_front_x_m': .42, 'notebook_extraction_max_x_m': .421,
            'passive_damping_n_s_m': 5., 'physical_support': 'World-anchored prismatic rail, no spring or pose drive.'},
        'budget': {'max_sim_time_s': 180., 'max_robot_actions': None}}
    c['rules'] = public_rules(c)
    c['configuration_sha256'] = canonical_hash(c)
    return c


def validate_configuration(value):
    # New task is a declared, immutable authored reset. Physical randomization
    # requires a new version; an arbitrary caller cannot relax this referee.
    wanted = configuration()
    if value != wanted:
        raise ValueError('Drawer configuration differs from its frozen authored rules/reset')
    return deepcopy(wanted)
