"""L3: classify six utensils with practical containment; frozen scene candidate."""
from scenes.workrooms.kitchen.catalog_definition import implemented_task_definition

TASK = implemented_task_definition(
    'utensil_sorting',
    intended_roles=['spoon', 'fork', 'blunt_table_knife', 'fixed_source_pad',
                    'fixed_three_channel_cutlery_tray'],
    inventory_counts={'spoon': 2, 'fork': 2, 'blunt_table_knife': 2,
                      'source_pad': 1, 'cutlery_tray': 1},
    dynamic_count=6, fixed_fixture_count=2,
    contextual_prop_count=12,
    manifest_sha256='e9c8cdd88d0841d269058428706792d4472816809c84e7e6d0528b4935ff3b5b',
    asset_version='household_grip_v1', release_version='household_grip_v1',
)
