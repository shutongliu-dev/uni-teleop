"""Task5: sort sealed bottles, seasonings and dry goods into storage bins."""
from bench.paths import DATA_ROOT
from scenes.workrooms.kitchen.catalog_definition import implemented_task_definition


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
    manifest_sha256='154090017b2a9d1e05fca06ada9465c9c7367bafaba1979767fe963683729d9f',
    asset_version='v1', release_version='v2',
    evidence_root=f'{DATA_ROOT}/datasets/fr3_sharpa_bench/development/kitchen_six_tasks_20261021',
    canonical_run='native01_canonical', goal_support_run='native02_goal',
    canonical_sha256='5f479dcb8c087875e2936e2c93e2151a0d8c5f8a49b9cfb721663f226f24f428',
    goal_support_sha256='9887b099a6abbe0f5307de2dee670d26e05413047b63e8e7fa2dbe716ef373fc',
)
