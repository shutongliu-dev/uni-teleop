"""L2: six rigid produce items into two class trays; frozen scene candidate."""
from bench.paths import DATA_ROOT
from scenes.workrooms.kitchen.catalog_definition import implemented_task_definition


TASK = implemented_task_definition(
    'produce_transfer',
    intended_roles=['rigid_tomato', 'rigid_short_carrot', 'fixed_source_board',
                    'fixed_class_tray'],
    inventory_counts={'tomato': 3, 'short_carrot': 3, 'source_board': 1,
                      'sorting_tray': 2},
    dynamic_count=6, fixed_fixture_count=3,
    contextual_prop_count=12,
    manifest_sha256='dea937e736721e7f7af1972f1f325dad1b0af86e2a1fad28a527c0d72d48f1f8',
    asset_version='v3', release_version='v3',
    evidence_root=f'{DATA_ROOT}/datasets/fr3_sharpa_bench/development/kitchen_six_tasks_20261021',
    canonical_run='native01_canonical', goal_support_run='native01_goal',
    canonical_sha256='69d9ef42dbbf9577df12034941adc6ec889af951a874490e2bd136b438925ddc',
    goal_support_sha256='5bac67fb8902a38c3af8abe277e62191aea0e1a059707f487b12a0fe7d3c3210',
)
