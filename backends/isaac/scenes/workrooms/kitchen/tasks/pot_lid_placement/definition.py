"""L4: released pot on a cold burner, then passive lid seating; frozen scene candidate."""
from bench.paths import DATA_ROOT
from scenes.workrooms.kitchen.catalog_definition import implemented_task_definition


TASK = implemented_task_definition(
    'pot_lid_placement',
    intended_roles=['hollow_pot_with_two_ear_handles', 'independent_passive_lid',
                    'fixed_source_mat'],
    inventory_counts={'pot': 1, 'lid': 1, 'source_mat': 1},
    dynamic_count=2, fixed_fixture_count=1,
    contextual_prop_count=12,
    manifest_sha256='0dbcc43adaa15160134d6d3ff9db8302457aa384a082607ed6e1a9507ec1cc0f',
    asset_version='v3', release_version='v4',
    evidence_root=f'{DATA_ROOT}/datasets/fr3_sharpa_bench/development/kitchen_six_tasks_20261021',
    canonical_run='native01_canonical', goal_support_run='native02_goal',
    canonical_sha256='396a7e24b78e2404f15dd3999db6911f6d73a88f4a2e8e8c6ddabd8afb51f0c1',
    goal_support_sha256='cd5bed2096e381ed87fe5bee66c191866572b4ead236e961f95cb58b481e93e6',
)
