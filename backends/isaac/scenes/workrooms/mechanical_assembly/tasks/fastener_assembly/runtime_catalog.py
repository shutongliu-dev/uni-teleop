"""Current input pins; development test results do not select or admit tasks."""
from copy import deepcopy

from bench.paths import DATA_ROOT

RELEASE = {
    'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/isaacsim/releases/mechanical_assembly/fastener_assembly/runtime_inputs_v1',
    'manifest_sha256': '498d5946278144e17df8af60ba93d65771fab176335f86af68e89d5b52888f42',
}
ASSET_ROOT = f'{DATA_ROOT}/assets/task_parts/mechanical_assembly/fastener_assembly/runtime_inputs_v1'


def bind_runtime_inputs(task):
    result = deepcopy(task)
    result['source_release'] = deepcopy(RELEASE)
    result['objects']['asset_bindings'].update(
        frozen_scene=RELEASE['path']+'/scene.json',
        frozen_inventory=RELEASE['path']+'/asset_file_hashes.json',
        physical_catalog={'path': ASSET_ROOT+'/manifest.json',
                          'sha256': '9247c407d0676ad882a4c6ef5e426a260786cba1225aa6b82dbe39aa05495edd'})
    result['qualification'] = {
        'release_status': 'prepared_unvalidated', 'native_preflight': False,
        'agent_bench_interface': False, 'manipulation': False,
        'physics_accepted': False, 'task_policy_success_claimed': False,
        'scope': 'Input identity and per-episode frozen scoring only; development outcomes are separate.'}
    return result
