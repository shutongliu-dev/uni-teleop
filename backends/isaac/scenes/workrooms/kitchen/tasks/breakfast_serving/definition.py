"""Task1: assemble a breakfast tray and carry it loaded; frozen scene candidate."""
from bench.paths import DATA_ROOT
from scenes.workrooms.kitchen.catalog_definition import implemented_task_definition
from scenes.workrooms.kitchen.scene_qualification import rigid_scene_qualification

TASK = implemented_task_definition(
    'breakfast_serving',
    intended_roles=['handled_serving_tray', 'shallow_plate', 'empty_hollow_cup', 'rigid_bread'],
    inventory_counts={'tray': 1, 'plate': 1, 'cup': 1, 'bread': 2},
    dynamic_count=5, fixed_fixture_count=0,
    contextual_prop_count=12,
    manifest_sha256='8752aee85a0c47add1a11a871e39867e0f6ed2301b4fa99eeed874104b3440b8',
    asset_version='household_v2', release_version='household_v2',
    evidence_root=f'{DATA_ROOT}/datasets/fr3_sharpa_bench/development/kitchen_prop_scale_20261023',
)

TASK['qualification'].update(rigid_scene_qualification('breakfast_serving'))
TASK['qualification'].update({'discrete_workspace_evidence': {'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/development/kitchen_prop_scale_20261023/ik_final01.json',
                                 'sha256': '98f65b16512343b0e6a109d85556100c4d453953182ddc957e600a5649c044e3'}})
