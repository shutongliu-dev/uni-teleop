"""Catalog pins for workshop desktop derivatives; original releases remain intact."""
from copy import deepcopy

from bench.paths import DATA_ROOT

RELEASES = {'parts_sorting': {'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/isaacsim/releases/mechanical_assembly/parts_sorting/desktop_v1',
                   'manifest_sha256': '8dbdadd5d0a092f42ac37d9ec6eba55cd48ba726e867bdb3df92db95a784a08f'},
 'peg_insertion': {'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/isaacsim/releases/mechanical_assembly/peg_insertion/desktop_v1',
                   'manifest_sha256': '589cbd190d1e15bad940d1553cdae025f79d57148cfe587ad89e7b686a1f3a3f'},
 'shaft_spacer_assembly': {'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/isaacsim/releases/mechanical_assembly/shaft_spacer_assembly/desktop_v1',
                           'manifest_sha256': '4bda201af19b2701394c1e503455ab62de54584ab75d7829d1999cdc5a8c2532'},
 'fastener_assembly': {'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/isaacsim/releases/mechanical_assembly/fastener_assembly/desktop_v1',
                       'manifest_sha256': '5d02b5db384690b09aa1b9aa2789ac1b9d15b01ecc39c24e4b8f63bdbc98dfa3'},
 'gear_train_assembly': {'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/isaacsim/releases/mechanical_assembly/gear_train_assembly/desktop_v1',
                         'manifest_sha256': '4edad1f71dd67688679613802c025857bf3075764f0f86c315bf7ec01cd3d9e5'},
 'nist_connector_insertion': {'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/isaacsim/releases/mechanical_assembly/nist_connector_insertion/desktop_v1',
                              'manifest_sha256': '873a184db4261640ecac4abb063e44199592eed3439bec6fdcbbc0492214417b'},
 'nist_belt_drive_assembly': {'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/isaacsim/releases/mechanical_assembly/nist_belt_drive_assembly/desktop_v1',
                              'manifest_sha256': '3669d2b9162d1f342e170cef88f76d039d3e6d2a86eec23db52ad33f04141d02'},
 'nist_cable_routing': {'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/isaacsim/releases/mechanical_assembly/nist_cable_routing/desktop_v1',
                        'manifest_sha256': '1bc7b939a15b1a046cf2c3f29da002785dbf6510a1ddca485c05c87e739772c9'},
 'nist_wire_harness': {'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/isaacsim/releases/mechanical_assembly/nist_wire_harness/desktop_v1',
                       'manifest_sha256': '7dc16111e5fb97d7502681a2e743e70c47039ab6d7886e6aab2a5e38775c3106'}}


def bind_desktop_release(task):
    result = deepcopy(task)
    result['previous_source_release'] = deepcopy(result['source_release'])
    result['source_release'] = deepcopy(RELEASES[result['task_id']])
    result['objects']['background_supplies'] = {
        'groups': 31, 'movable': False, 'scoring_support': False,
        'description': 'Fixed edge tool caddy, hex keys, stocked parts bins and caliper; the active task inventory is unchanged.',
    }
    if isinstance(result['objects'].get('asset_bindings'), dict):
        result['objects']['asset_bindings'].update(
            frozen_scene=result['source_release']['path']+'/scene.json',
            frozen_inventory=result['source_release']['path']+'/asset_file_hashes.json')
    previous = deepcopy(result.get('qualification', {}))
    result['qualification'] = {
        'release_status': 'prepared_unvalidated', 'native_preflight': False,
        'manipulation': False, 'physics_accepted': False, 'task_policy_success_claimed': False,
        'previous_release_qualification': previous,
        'scene_contact_evidence': [],
        'scope': 'Static collision-shape clearance checked for all nine desktop releases. Native interface evidence, when present, applies only to the named release. Original task/reset/rules/scorer/physics files are byte-identical; historical evidence stays separate.',
    }
    result['qualification']['scene_geometry_evidence'] = {
        'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/development/mechanical_coop_20261021/desktop_old_tasks_geometry03/report.json',
        'sha256': 'd21204d01cd98c21a33f37cc81eaefe0e6c48f1ddd51c2519370c66930b0c9c3',
        'scope': 'All nine authored task resets: actual USD/OBJ shape bounds show no overlap with the desktop layer. This is not native cooking, swept robot clearance or a full visibility guarantee.',
    }
    if result['task_id'] == 'gear_train_assembly':
        result['qualification'].update(native_preflight=True, agent_bench_interface=True)
        result['qualification']['native_interface_evidence'] = {
            'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/development/three_room_expansion_20261021/mechanical_desktop_http01/qualification.json',
            'sha256': '1568601915a39e80d22559fe6d9423e6e1fbf4f8d70cf71bcb651968f5f21daa',
            'release_manifest_sha256': RELEASES['gear_train_assembly']['manifest_sha256'],
            'passed': True,
            'scope': 'Task5 desktop_v1 only: live HTTP, actual Agent client reads, four RGB-D cameras, IK, joint micro-motion/return/hold and idempotency. No gear manipulation or qualification of the other eight native releases.',
        }
    return result
