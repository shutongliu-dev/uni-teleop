"""L4: released pot on a cold burner, then passive lid seating; frozen scene candidate."""
from bench.paths import DATA_ROOT
from scenes.workrooms.kitchen.catalog_definition import implemented_task_definition
from scenes.workrooms.kitchen.scene_qualification import rigid_scene_qualification

TASK = implemented_task_definition(
    'pot_lid_placement',
    intended_roles=['hollow_pot_with_two_ear_handles', 'independent_passive_lid',
                    'fixed_source_mat'],
    inventory_counts={'pot': 1, 'lid': 1, 'source_mat': 1},
    dynamic_count=2, fixed_fixture_count=1,
    contextual_prop_count=12,
    manifest_sha256='8e62383cd8dcc52aee6d68480d8dbeeb3138103cc29092090401a9624dcbb3be',
    asset_version='household_access_v3', release_version='household_access_v3',
    evidence_root=f'{DATA_ROOT}/datasets/fr3_sharpa_bench/development/kitchen_dynamics_repair_20261023',
)

TASK['qualification'].update(rigid_scene_qualification('pot_lid_placement'))
TASK['qualification'].update({'discrete_workspace_evidence': {'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/development/kitchen_dynamics_repair_20261023/formal_workspace01/pot_lid_placement.json',
                                 'sha256': '4c20537b0853fbf9d4ccb51e1dd51c1024455038515043a417ecc95cfe2a7940'}})
