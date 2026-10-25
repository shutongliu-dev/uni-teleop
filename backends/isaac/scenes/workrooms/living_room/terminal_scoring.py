"""Public numeric export binding for the immutable free-building v5 release.

This only names existing evaluator result fields. It neither changes scoring nor
rewrites frozen task/rule hashes. Future releases should declare their contract.
"""
import hashlib
import json

from bench.paths import portable


# Canonical portable task.json with the permitted episode seed/phase removed.
_V5_TASK_SHA256 = '3adec4b9ef18511b08e7c159ad0187bb9c235c9035ca0054b6b53fa3036771ea'


def legacy_terminal_scoring_contract(task):
    if task.get('schema_version') != 'living-room-task-v1':
        return None
    static = {key: value for key, value in task.items() if key not in ('seed', 'evaluation_phase')}
    identity = hashlib.sha256(json.dumps(portable(static), sort_keys=True, separators=(',', ':'),
        ensure_ascii=True, allow_nan=False).encode()).hexdigest()
    if identity != _V5_TASK_SHA256:
        return None
    return {'schema': 'terminal_task_scoring_contract.v1', 'mode': 'continuous',
            'objective': 'maximize', 'score_unit': 'm',
            'metric_units': {'current_height_m': 'm', 'best_height_m': 'm', 'best_height_cm': 'cm'}}
