"""Static free-building rules and coarse feedback for the common task API."""
import hashlib
import json


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    ensure_ascii=True, allow_nan=False).encode()).hexdigest()


def bind_height_rules(task):
    config = task['evaluation']['reference_configuration']
    instruction = ('从两侧托盘取木块，自由搭建积木塔。不限层数、每层根数、朝向或塔形；'
                   '按本局松手后连续稳定至少一秒的最高塔高登记分数。高度从桌面量至塔顶，'
                   '只计真实支撑在桌面上的木块结构，托盘内存料和手持木块不计分。'
                   '可重抓、重搭和继续加高；后续倒塌不撤销已登记的最高高度。')
    contract = {
        'schema': 'physical_task_referee_contract.v1', 'task_id': 'tower_building',
        'instruction_zh': instruction,
        'phases': {'building': '自由搭建或恢复结构。', 'height_recorded': '已有稳定高度纪录，可继续挑战。',
                   'evidence_invalid': '物理证据无效。', 'safety_stop': '物理安全检查中止。'},
        'reason_descriptions': {'building': '尚未形成合格的稳定高度纪录。',
            'height_recorded': '本局最高稳定高度已登记，后续倒塌不撤销纪录。',
            'evidence_invalid': '缺失、矛盾或不连续的物理证据不能产生新纪录。',
            'safety_stop': '物理状态违反安全条件。'},
        'success_conditions': ['本任务按最高稳定高度计分，没有固定塔形或通关高度。',
            '每次新纪录需真实桌面支撑、无手或外物支撑，并连续稳定至少一秒。'],
        'recovery': ['允许自由改变搭法、重抓、重搭和继续加高。', '倒塌后保留此前已经确认的最高稳定高度。'],
        'thresholds': {'stable_duration_s': config['rules']['goal']['stable_duration_s'],
                       'available_block_count': len(config['block_ids']),
                       'height_origin_z_m': config['geometry']['table_top_z_m']},
        'outcomes': {'ongoing': '持续搭建，随实际稳定高度刷新本局纪录。',
            'success': '不设置二元通关高度；按裁判保存的最高稳定高度比较成绩。',
            'task_failure': '只按明确的物理规则记录，不因塔形或高度不足判失败。',
            'evaluator_invalid': '证据无效，停止本次评分；旧纪录保留供审计，不据此宣称有效回合。',
            'safety_stop': '物理安全中止单独记录。',
            'budget_exhaustion': '运行器关闭回合，保存预算内已确认的最高稳定高度；不制造二元成功。',
            'cancelled': '取消与运行故障独立记录；已登记高度保留供审计。'},
        'physical_actions': '只通过机器人动作取放木块，不直接写物体状态或使用粘接、隐藏支架。',
        'observation_scope': '公开任务规则与粗粒度裁判阶段；高度纪录保存在回合裁判结果中。',
    }
    task['public_rules']['referee'] = contract
    task.update(public_rules_file='rules.json', public_rules_sha256=_hash(task['public_rules']),
                public_referee_contract_sha256=_hash(contract), title_zh='自由搭塔：最高稳定高度')
    task['steps'] = [{'id': 'tower_building', 'instruction_zh': instruction}]
    task['public_goal'] = {'operation': 'freeform_tower_height', 'score': 'highest_stable_height',
                           'description': instruction}
    return task


def height_verdict(status, best_height_m):
    if status in ('invalid', 'evaluator_invalid'):
        state = 'evaluator_invalid'
        phase = 'evidence_invalid'
    elif status == 'safety_stop':
        state = phase = 'safety_stop'
    else:
        state = 'ongoing'
        phase = 'height_recorded' if best_height_m > 0 else 'building'
    return {'schema': 'physical_task_referee_state.v1', 'state': state, 'phase': phase,
            'reason_codes': [phase], 'terminal': state != 'ongoing', 'recoverable': state == 'ongoing'}
