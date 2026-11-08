"""Explicit NIST usability revision; frozen evidence is attached separately."""
from bench.paths import DATA_ROOT
from scenes.workrooms.mechanical_assembly.nist_catalog import task_definition

TASK = task_definition(
    'nist_connector_insertion',
    manifest_sha256='76886985c22a43daeac20cbebcab525efc3c340ab1668e8f9867270c3840ba6f',
    asset_version='usability_v1',
    release_version='usability_desktop',
    usability_profile='nist_manipulation_scale_v1',
)

TASK["source_release"] = {'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/isaacsim/releases/nist_usability_20261023/batch01/nist_connector_insertion/desktop_v1',
 'manifest_sha256': '76886985c22a43daeac20cbebcab525efc3c340ab1668e8f9867270c3840ba6f'}
TASK["previous_source_release"] = {'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/isaacsim/releases/mechanical_assembly/nist_connector_insertion/desktop_v1',
 'manifest_sha256': '873a184db4261640ecac4abb063e44199592eed3439bec6fdcbbc0492214417b'}
TASK["objects"]["background_supplies"] = {'groups': 31,
 'movable': False,
 'scoring_support': False,
 'description': 'Fixed edge tool caddy, hex keys, stocked parts bins and caliper; the active task inventory '
                'is unchanged.'}
TASK["qualification"]["workspace_discrete"] = True
TASK["qualification"]["scene_geometry_evidence"] = {'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/development/nist_usability_20261023/root/workspace01/nist_connector_insertion_single_arm.json',
 'sha256': 'a923d55c53450b5aead70f5e89305edb79c13a9af295f952cf6241ee9e37b4ed',
 'passed': True,
 'scope': 'Discrete actual fingertip approach and original USD collision bounds; no continuous path or '
          'manipulation qualification.'}

# Completed physical evidence; immutable preparation flags remain unchanged.
TASK["qualification"].update({'object_interactions': True,
 'scene_contact_evidence': [{'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/development/nist_usability_20261023/scale_audit/native_task6_goal01/qualification.json',
                             'sha256': '0678af29ff6a10e29b4937de3e8826fe76650c541d146c16522bab9cbabff0e6',
                             'passed': True,
                             'scope': 'Asset-specific passive goal fit/support for Task6/8/9; 10mm '
                                      'already-guided idler insertion for Task7. No robot operation credit.',
                             'asset_manifest': {'path': f'{DATA_ROOT}/assets/task_parts/mechanical_assembly/nist_connector_insertion/usability_v1/manifest.json',
                                                'sha256': '066a3c93be4bae05c3765b8054e60688ed3a22226cfd1a8e5f2f989e9283e3ee'},
                             'qualification_index': {'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/development/nist_usability_20261023/scale_audit/qualification02.json',
                                                     'sha256': 'a9ce0d252506a14e28cba028091ad977bab2a77a46b7bb672296fd873f49e771'},
                             'asset_identity_bridge': {'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/development/nist_usability_20261023/scale_audit/root-independent-identity-review02.json',
                                                       'sha256': '0f371c44ac87663055d2332be0c276fef810a9af83fff104c4f1b343c286d2bb'}}],
 'workspace_release_binding': {'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/development/nist_usability_20261023/root/workspace_binding03.json',
                               'sha256': 'b3c6755b71c8dcd1bf6a7ef3e6c082e0aafd9c2629b3731bb01a82d66a9b286c'},
 'manipulation': False,
 'physics_accepted': False,
 'native_preflight': True,
 'agent_bench_interface': True,
 'native_interface_evidence': {'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/development/nist_usability_20261023/root/http_task6_single01/qualification.json',
                               'sha256': '9f2137e38bf043ea2ad4fefc1d8c983eaa09973f7656a55404db5d674afee36a',
                               'passed': True,
                               'configuration_id': 'single_arm',
                               'release_manifest_sha256': '76886985c22a43daeac20cbebcab525efc3c340ab1668e8f9867270c3840ba6f',
                               'scope': 'Exact frozen-source public HTTP/Agent interface, RGB-D, read-only '
                                        'IK, wrist micro-motion/return/hold; no manipulation qualification.'}})
