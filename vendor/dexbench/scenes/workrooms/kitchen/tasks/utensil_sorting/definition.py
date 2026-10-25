"""L3: classify six utensils with practical containment; frozen scene candidate."""
from bench.paths import DATA_ROOT
from scenes.workrooms.kitchen.catalog_definition import implemented_task_definition


TASK = implemented_task_definition(
    'utensil_sorting',
    intended_roles=['spoon', 'fork', 'blunt_table_knife', 'fixed_source_pad',
                    'fixed_three_channel_cutlery_tray'],
    inventory_counts={'spoon': 2, 'fork': 2, 'blunt_table_knife': 2,
                      'source_pad': 1, 'cutlery_tray': 1},
    dynamic_count=6, fixed_fixture_count=2,
    contextual_prop_count=12,
    manifest_sha256='2143e9cc82f59ca94a4337d575da67b401de8b1a81e7f21505d9b47598a5a716',
    asset_version='v3', release_version='v3',
    evidence_root=f'{DATA_ROOT}/datasets/fr3_sharpa_bench/development/kitchen_six_tasks_20261021',
    canonical_run='native01_canonical', goal_support_run='native02_goal',
    canonical_sha256='4878d01b70c1bc609cd55cac3f4e0fdaeca1b51b7bf188c906de42195f75f80e',
    goal_support_sha256='23f1e2bdc1ce1cafe106d18db8bc79ceca8509badd153f6d3426881131e8eda0',
)
