"""Task5: sort sealed bottles, seasonings and dry goods into storage bins."""
from bench.paths import DATA_ROOT
from scenes.workrooms.kitchen.catalog_definition import implemented_task_definition
from scenes.workrooms.kitchen.scene_qualification import rigid_scene_qualification

TASK = implemented_task_definition(
    'pantry_stocking',
    intended_roles=['sealed_olive_oil_bottle', 'sealed_vinegar_bottle',
                    'salt_shaker', 'pepper_shaker', 'paprika_jar',
                    'rice_carton', 'pasta_carton', 'tomato_can',
                    'fixed_source_mat', 'fixed_category_storage_bin'],
    inventory_counts={'olive_oil_bottle': 1, 'vinegar_bottle': 1, 'salt_shaker': 1,
                      'pepper_shaker': 1, 'paprika_jar': 1, 'rice_carton': 1,
                      'pasta_carton': 1, 'tomato_can': 1, 'source_mat': 1,
                      'storage_bin': 3},
    dynamic_count=8, fixed_fixture_count=4, contextual_prop_count=12,
    manifest_sha256='4485e32c4e1a5ea0904f11584778050016b479d0c8df77424d85999cb1252f75',
    asset_version='household_v2', release_version='household_v2',
    evidence_root=f'{DATA_ROOT}/datasets/fr3_sharpa_bench/development/kitchen_prop_scale_20261023',
)

TASK['qualification'].update(rigid_scene_qualification('pantry_stocking'))
TASK['qualification'].update({'discrete_workspace_evidence': {'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/development/kitchen_prop_scale_20261023/ik_final01.json',
                                 'sha256': '98f65b16512343b0e6a109d85556100c4d453953182ddc957e600a5649c044e3'}})
