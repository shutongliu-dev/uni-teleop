"""Explicit ordinary-task reset families; independent from RSI curricula."""

LEGACY_TASK_FAMILIES = {
    'stationery_sorting': 'study', 'book_stacking': 'study', 'book_shelving': 'study',
    'produce_transfer': 'kitchen', 'utensil_sorting': 'kitchen',
    'pantry_stocking': 'kitchen', 'table_clearing': 'kitchen',
    'sample_sorting': 'laboratory', 'vial_racking': 'laboratory',
    'parts_sorting': 'mechanical_assembly',
}
PLACEMENT_TASK_FAMILIES = {
    'peg_insertion': 'mechanical_assembly',
    'shaft_spacer_assembly': 'mechanical_assembly',
    'fastener_assembly': 'mechanical_assembly',
    'gear_train_assembly': 'mechanical_assembly',
    'nist_connector_insertion': 'mechanical_assembly',
    'nist_belt_drive_assembly': 'mechanical_assembly',
    'nist_cable_routing': 'mechanical_assembly',
    'nist_wire_harness': 'mechanical_assembly',
    'cooperative_insertion': 'mechanical_assembly',
    'book_retrieval': 'study', 'drawer_retrieval': 'study', 'reading_stand': 'study',
    'breakfast_serving': 'kitchen', 'pot_lid_placement': 'kitchen', 'surface_wiping': 'kitchen',
    'cap_placement': 'laboratory', 'solid_weighing': 'laboratory',
    'pipette_transfer': 'laboratory', 'tube_clamp_transfer': 'laboratory',
}
TASK_FAMILIES = {**LEGACY_TASK_FAMILIES, **PLACEMENT_TASK_FAMILIES}
PLACEMENT_SCOPE = 'initial_xy_translation_and_surface_texture_only'
SEEDS = tuple(range(10))
CONFIGURATIONS = ('single_arm', 'dual_same_side', 'dual_opposed')
SCHEMA = 'workroom_randomized_release.v1'
VERSION = 'isaacsim-workroom-randomized-release-v1'


def configurations(task_id):
    if task_id == 'cooperative_insertion':
        return ('dual_same_side',)
    return ('single_arm',) if TASK_FAMILIES[task_id] == 'kitchen' else CONFIGURATIONS


def selection(task_id, seed, configuration_id=None):
    if task_id not in TASK_FAMILIES or type(seed) is not int or seed not in SEEDS:
        raise ValueError('Known ordinary task and frozen seed 0..9 required')
    if configuration_id is not None and configuration_id not in configurations(task_id):
        raise ValueError('This task does not have the requested original installation')
