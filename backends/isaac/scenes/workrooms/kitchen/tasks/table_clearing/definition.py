"""Task6: clear eight pieces of dishware, drinkware and cutlery by category."""
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
    manifest_sha256='775753dd9eea631aa5051b843ce2ec56054288dd7d0896339b9196cdf10dcdff',
    asset_version='household_grip_v1', release_version='household_grip_v1',
)
