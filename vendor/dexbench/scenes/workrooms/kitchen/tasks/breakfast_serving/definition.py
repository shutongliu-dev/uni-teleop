"""Task1: assemble a breakfast tray and carry it loaded; frozen scene candidate."""
from bench.paths import DATA_ROOT
from scenes.workrooms.kitchen.catalog_definition import implemented_task_definition


TASK = implemented_task_definition(
    'breakfast_serving',
    intended_roles=['handled_serving_tray', 'shallow_plate', 'empty_hollow_cup', 'rigid_bread'],
    inventory_counts={'tray': 1, 'plate': 1, 'cup': 1, 'bread': 2},
    dynamic_count=5, fixed_fixture_count=0,
    contextual_prop_count=12,
    manifest_sha256='62cd44639e545c6b7081052db2e5a3a5536b3d7cfef6b76a75f06cefd118bbff',
    asset_version='v3', release_version='v3',
    evidence_root=f'{DATA_ROOT}/datasets/fr3_sharpa_bench/development/kitchen_six_tasks_20261021',
    canonical_run='native01_canonical', goal_support_run='native01_goal',
    canonical_sha256='e8a563fd39e179fdea5a4f98e88e2960d64fc57dbbfe55d5451d2304c85491c3',
    goal_support_sha256='3b2923b053670da39100d10155893d34c1daff7151646a8e011e45e656f4fc2f',
)
