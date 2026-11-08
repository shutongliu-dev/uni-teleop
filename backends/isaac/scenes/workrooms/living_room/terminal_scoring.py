"""Public numeric export bindings for reviewed immutable free-building tasks.

This only names existing evaluator result fields. It neither changes scoring nor
rewrites frozen task/rule hashes. Future releases should declare their contract.
"""
import hashlib
import json

from bench.paths import portable


# Canonical frozen task.json with the permitted episode seed/phase removed.
_FROZEN_TASK_SHA256 = frozenset({
    # Original tower_building/v5.
    'ecce9643126a61394a94bcd61a39f06ee68c56baeaf53e3cc3210752b350defd',
    # robot_configurations_20261023/batch01: dual_same_side/v1 and
    # dual_opposed/v1 share task/reset bytes; installation lives in scene.json.
    '05e73b4df5a79e34be371cef70eabcd209c7feea5f082863b8509cad43755b74',
    '05e73b4df5a79e34be371cef70eabcd209c7feea5f082863b8509cad43755b74',
    # robot_l_mount_update_20261104/releases_candidate03 L-mount revisions of
    # the three releases above (fixed, dual_same_side, dual_opposed); each
    # differs from its frozen parent only in the robot reset joint positions.
    'bdd606b969f6c20bef6c05c3323fff2ea27117566d54be5236a5993335e09b2c',
    '79a5bd129f8d3b96363d93624dffe55181badbad72e75e3c6ea6182ef5f7c953',
    '4079cb99f02bd0d22049181fec433166ad469635ee36f29aa697ad5fdc8d1b62',
})


def legacy_terminal_scoring_contract(task):
    if task.get('schema_version') != 'living-room-task-v1':
        return None
    static = {key: value for key, value in task.items() if key not in ('seed', 'evaluation_phase')}
    identity = hashlib.sha256(json.dumps(portable(static), sort_keys=True, separators=(',', ':'),
        ensure_ascii=True, allow_nan=False).encode()).hexdigest()
    if identity not in _FROZEN_TASK_SHA256:
        return None
    return {'schema': 'terminal_task_scoring_contract.v1', 'mode': 'continuous',
            'objective': 'maximize', 'score_unit': 'm',
            'metric_units': {'current_height_m': 'm', 'best_height_m': 'm', 'best_height_cm': 'cm'}}
