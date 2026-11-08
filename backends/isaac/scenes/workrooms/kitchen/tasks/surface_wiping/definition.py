"""Task7: physically elastic sponge wiping and supported return."""
from bench.paths import DATA_ROOT
from scenes.workrooms.kitchen.catalog_definition import implemented_task_definition

TASK = implemented_task_definition(
    'surface_wiping',
    intended_roles=['elastic_cleaning_sponge', 'sponge_return_tray', 'visible_surface_soil'],
    inventory_counts={'sponge': 1, 'return_tray': 1},
    dynamic_count=1, fixed_fixture_count=1, contextual_prop_count=12,
    manifest_sha256='50a6505fd44d71bceb7a37d85ac2f1f642396dfb89cfd9980baf20a0370b1aff',
    asset_version='elastic_cell_lattice_v2', release_version='elastic_cell_lattice_v2',
    evidence_root=f'{DATA_ROOT}/datasets/fr3_sharpa_bench/development/kitchen_flexible_velocity_fix_20261024',
    asset_metadata={'elastic_composites': {'sponge': {'profile': 'elastic_cell_lattice_v1'}}},
)
TASK['objects'].update(dynamic_count_scope='logical_task_objects', native_rigid_body_count=28)
TASK['qualification'].update(
    status='elastic_object_physics_and_agent_interface_checked',
    native_preflight=True, scene_support=True, agent_bench_interface=True,
    discrete_elastic_object_physics=True,
    raw_terminal_native_velocity_limits_passed=True,
    validation_index={
        'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/development/kitchen_flexible_velocity_fix_20261024/sponge_qualification.json',
        'sha256': '724ad4534fcbefa48af4077113d2c8fd7d89bd4aacce23f2ed8654c095553ab0',
    },
    native_source={
        'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/development/kitchen_flexible_velocity_fix_20261024/root/source05',
        'manifest_sha256': 'c123d888a81ab24432efe99c05d92d4ad90fa1e222ce53a5b5cd7e1152714ca9',
    },
    scope='Passive elastic cell sponge: real GPU material springs, 79.2 g total mass and original full-size shape. '
          'Native impulse, calibrated PGS weight, original-tray TGS rest, dynamic compression/recovery and bounded '
          'contact wiping have separate source-bound evidence. The selected release passed ordinary Agent/Bench '
          'HTTP and independent replay of all 3281 native frames, 28 physical parts and 63 passive links. '
          'V1/v2 physical identity and 19 discrete IK/clearance endpoints are explicitly bound in the index. '
          'The original wiping AABB-screen exit1 remains preserved; all-frame actual-convex refinement verifies '
          'the unchanged 5 mm clearance. No absorption, continuum material calibration, robot manipulation '
          'or transfer of whole-source native qualification to later main.',
    continuum_fem_qualified=False, material_parameters_calibrated=False,
    absorbency_simulated=False,
)
