"""Static rule contracts and coarse verdicts copied into frozen evaluators."""
from copy import deepcopy
import hashlib
import json
import re

CONTRACT_SCHEMA = 'physical_task_referee_contract.v1'
STATE_SCHEMA = 'physical_task_referee_state.v1'
STATES = ('ongoing', 'success', 'task_failure', 'evaluator_invalid', 'safety_stop')


def contract_sha256(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
        ensure_ascii=True, allow_nan=False).encode()).hexdigest()


def public_contract(task_id, *, instruction_zh, phases, reasons,
                    success_conditions, recovery, thresholds):
    """Only authored static rules enter this value; never observation data."""
    return deepcopy({
        'schema': CONTRACT_SCHEMA, 'task_id': task_id, 'instruction_zh': instruction_zh,
        'phases': phases, 'reason_descriptions': reasons,
        'success_conditions': success_conditions, 'recovery': recovery,
        'thresholds': thresholds,
        'outcomes': {
            'ongoing': '任务尚未完成；在预算内可继续符合规则的操作。',
            'success': '全部物理条件及完整释放稳定窗口通过；接受后结果不可撤销。',
            'task_failure': '完整有效证据确认违反本任务明确公布的不可恢复规则。',
            'evaluator_invalid': '采样、时钟或数值证据无效，终止本次验证；不作为正常任务失败计分。',
            'safety_stop': '有效物理观测触发冻结的速度或穿透上限，停止本回合并单独记录。',
            'budget_exhaustion': '预算耗尽由运行器记录为未完成；不伪造成功，也不宣称零件不可恢复。',
            'cancelled': '用户取消或运行故障由运行器记录；不当作物理任务规则违规。',
        },
        'physical_actions': '只控制机器人；允许按本任务规则重抓、对齐和重试。禁止物体位姿回放、吸附、焊接抓取和直接施加物体外力；诊断台架不计任务成绩。',
        'observation_scope': '公开规则和粗粒度阶段／原因码；精确物体位姿、接触冲量和裁判计算值保持私有。',
    })


def verdict(*, state, phase, reasons):
    if state not in STATES or not isinstance(phase, str) or not re.fullmatch('[a-z][a-z0-9_]*', phase):
        raise ValueError('Invalid referee state or phase')
    if (not isinstance(reasons, (list, tuple)) or not reasons
            or any(not isinstance(code, str) or not re.fullmatch('[a-z][a-z0-9_]*', code) for code in reasons)):
        raise ValueError('Referee reasons must be declared static codes')
    return {'schema': STATE_SCHEMA, 'state': state, 'phase': phase,
            'reason_codes': list(dict.fromkeys(reasons)),
            'terminal': state != 'ongoing', 'recoverable': state == 'ongoing'}


def bind_task_rules(task):
    """Bind the full frozen rule value and its safe public referee contract."""
    rules = task['public_rules']
    if rules.get('referee', {}).get('schema') != CONTRACT_SCHEMA:
        raise ValueError('New mechanical releases require a referee contract')
    task['public_rules_sha256'] = contract_sha256(rules)
    task['public_referee_contract_sha256'] = contract_sha256(rules['referee'])
    task['public_rules_file'] = 'rules.json'
    return task


def verify_task_rules(task):
    rules = task['public_rules']
    if (task.get('public_rules_sha256') != contract_sha256(rules)
            or task.get('public_referee_contract_sha256') != contract_sha256(rules['referee'])
            or task.get('public_rules_file') != 'rules.json'):
        raise ValueError('Frozen public rule identities differ')
