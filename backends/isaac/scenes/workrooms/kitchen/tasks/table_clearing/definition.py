"""Task6: clear eight pieces of dishware, drinkware and cutlery by category."""
from bench.paths import DATA_ROOT
from scenes.workrooms.kitchen.catalog_definition import implemented_task_definition


TASK = implemented_task_definition(
    'table_clearing',
    intended_roles=['dinner_plate', 'side_plate', 'cereal_bowl', 'small_bowl',
                    'handled_mug', 'handled_tea_cup', 'fork', 'spoon',
                    'fixed_source_mat', 'fixed_category_collection_tray'],
    inventory_counts={'dinner_plate': 1, 'side_plate': 1, 'cereal_bowl': 1,
                      'small_bowl': 1, 'mug': 1, 'tea_cup': 1, 'fork': 1, 'spoon': 1,
                      'source_mat': 1,
                      'collection_tray': 3},
    dynamic_count=8, fixed_fixture_count=4, contextual_prop_count=12,
    manifest_sha256='d1aeb420ad0f769b9048eda49b2152e04b1e13d35f673a405de2650d19590d57',
    asset_version='v1', release_version='v2',
    evidence_root=f'{DATA_ROOT}/datasets/fr3_sharpa_bench/development/kitchen_six_tasks_20261021',
    canonical_run='native01_canonical', goal_support_run='native02_goal',
    canonical_sha256='8701c112ceafe5a8e5ccf2179c72aa6e89ece9777dee71f56c4cf5e3fd05fd8e',
    goal_support_sha256='13a057ea4a941d61f3453f391d568550b54b6e2e21d33e59286dd770bf15863b',
)
