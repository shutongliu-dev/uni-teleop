"""Current gear grip inputs; no development outcomes enter task selection."""
from copy import deepcopy

from bench.paths import DATA_ROOT

RELEASE = {
    'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/isaacsim/releases/mechanical_assembly/gear_train_assembly/idler_grip_v2',
    'manifest_sha256': '1445f61c09816c139cf700ff87dae82c6b1ba3745e6a33c6eeb2733770caace8',
}
ASSET_ROOT = f'{DATA_ROOT}/assets/task_parts/mechanical_assembly/gear_train_assembly/idler_grip_v1'


def bind_grip_release(task):
    result = deepcopy(task)
    result['source_release'] = deepcopy(RELEASE)
    result['candidate_referee']['release'] = deepcopy(RELEASE)
    binding = result['objects']['asset_bindings']
    binding['screwdriver'] = {
        'asset_root': f'{DATA_ROOT}/assets/task_parts/mechanical_assembly/gear_train_assembly/screwdriver_input_v1',
        'asset_id': 'mechanical_screwdriver_sharpa_v3',
        'manifest_sha256': '0cfcdbf6b6630d8864a90fc6e02a76dc5ac85168d9e8350e5f310d7e96d27393',
        'scope': 'Same original USD and physical settings; only actual source dependencies and licenses are required.'}
    binding['idler_grip'] = {'asset_root': ASSET_ROOT,
        'manifest_sha256': '6c98f26c35cb1038d163d5259054b076c631b97dc2c5551fbdf38ad04a143996',
        'scope': 'Only the large idler; other gears, plate and tools retain their existing assets.'}
    binding['frozen_scene'] = RELEASE['path']+'/scene.json'
    binding['frozen_inventory'] = RELEASE['path']+'/asset_file_hashes.json'
    geometry = result['design']['candidate_gear_geometry']
    geometry.update(idler_grip_hub_diameter_m=.064, idler_grip_hub_height_above_gear_m=.032)
    result['design']['supply_layout_candidate'] = {
        'revision': 'table_inset_v1', 'release': deepcopy(RELEASE),
        'source_geometry': 'Original teeth, shaft/bore and table reset; only the large idler upper grip is larger.',
        'status': 'prepared_unvalidated',
        'scope': 'Immutable task inputs; independent development checks are not task outcomes.'}
    result['qualification'] = {
        'release_status': 'prepared_unvalidated', 'native_preflight': False,
        'agent_bench_interface': False, 'manipulation': False,
        'physics_accepted': False, 'task_policy_success_claimed': False,
        'scene_contact_evidence': [],
        'scope': 'Larger grasping hub with unchanged task rules; no inherited manipulation or transmission qualification.'}
    return result
