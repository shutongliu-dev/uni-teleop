"""L2: six rigid produce items into two class trays; frozen scene candidate."""
from bench.paths import DATA_ROOT
from scenes.workrooms.kitchen.catalog_definition import implemented_task_definition
from scenes.workrooms.kitchen.scene_qualification import rigid_scene_qualification

TASK = implemented_task_definition(
    'produce_transfer',
    intended_roles=['rigid_tomato', 'rigid_short_carrot', 'fixed_source_board',
                    'fixed_class_tray'],
    inventory_counts={'tomato': 3, 'short_carrot': 3, 'source_board': 1,
                      'sorting_tray': 2},
    dynamic_count=6, fixed_fixture_count=3,
    contextual_prop_count=12,
    manifest_sha256='98e0815595e337ec55ed9db9f1712fe8a40824308a6826de6a68d6f45feeaa13',
    asset_version='household_access_v2', release_version='household_access_v2',
    evidence_root=f'{DATA_ROOT}/datasets/fr3_sharpa_bench/development/kitchen_dynamics_repair_20261023',
)

TASK['qualification'].update(rigid_scene_qualification('produce_transfer'))
TASK['qualification'].update({'discrete_workspace_evidence': {'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/development/kitchen_dynamics_repair_20261023/formal_workspace01/produce_transfer.json',
                                 'sha256': 'dc7f57f0676fd1853ff6deed40a1020b859f7b60937b970b3d1d0b5ff4b6d5b4'}})
