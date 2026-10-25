"""Allowlisted native task observers. The bench owns their execution lifecycle.

Task files cannot select arbitrary Python imports. Each registered scene adapter
owns its rules and evidence mapping, while all motion remains ordinary joints.
"""
from __future__ import annotations


def terminal_scoring_contract(task):
    """Scene-owned compatibility registrations for immutable older releases."""
    if task.get('terminal_scoring_contract') is not None:
        return task['terminal_scoring_contract']
    from .workrooms.living_room.terminal_scoring import legacy_terminal_scoring_contract
    return legacy_terminal_scoring_contract(task)


def build_task_runtime(env, scene, task, output, *, evaluator_factory):
    if task.get('schema_version') == 'study-task-v1':
        from .workrooms.study.runtime import build_runtime
        return build_runtime(env, scene, task, output, evaluator_factory=evaluator_factory)
    if task.get('schema_version') == 'living-room-task-v1':
        from .workrooms.living_room.runtime import build_runtime
        return build_runtime(env, scene, task, output, evaluator_factory=evaluator_factory)
    return None
