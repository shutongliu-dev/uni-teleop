"""Explicit NIST usability revision; frozen evidence is attached separately."""
from bench.paths import DATA_ROOT
from scenes.workrooms.mechanical_assembly.nist_catalog import task_definition

TASK = task_definition(
    'nist_wire_harness',
    manifest_sha256='d17ab5072ad1d49245cb715a6bf4a3f1233787483912aaa0ead8d4b06e6f3ba3',
    asset_version='usability_v2',
    release_version='usability_desktop',
    usability_profile='nist_manipulation_scale_v1',
)

TASK["source_release"] = {'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/isaacsim/releases/nist_usability_20261023/batch03/nist_wire_harness/desktop_v1',
 'manifest_sha256': 'd17ab5072ad1d49245cb715a6bf4a3f1233787483912aaa0ead8d4b06e6f3ba3'}
TASK["previous_source_release"] = {'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/isaacsim/releases/mechanical_assembly/nist_wire_harness/desktop_v2',
 'manifest_sha256': 'aabb4765330fba869a7ee7475c03f29a3814fcd041757056dff196e2fb8ef0a4'}
TASK["objects"]["background_supplies"] = {'groups': 31,
 'movable': False,
 'scoring_support': False,
 'description': 'Fixed edge tool caddy, hex keys, stocked parts bins and caliper; the active task inventory '
                'is unchanged.'}
TASK["qualification"]["workspace_discrete"] = True
TASK["qualification"]["scene_geometry_evidence"] = {'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/development/nist_usability_20261023/root/workspace03/nist_wire_harness_single_arm.json',
 'sha256': 'fdb014d7847750c4fbe644e6b82b0d00306ba8793d5bbf1bcd9b317c5178e42f',
 'passed': True,
 'scope': 'Discrete actual fingertip approach and original USD collision bounds; no continuous path or '
          'manipulation qualification.'}

# Completed physical evidence; immutable preparation flags remain unchanged.
TASK["qualification"].update({'object_interactions': True,
 'scene_contact_evidence': [{'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/development/nist_usability_20261023/scale_audit/native_task9_goal03/qualification.json',
                             'sha256': 'eb24f50ceb60d2d122db5befb9c67bf9f81a26944fda4bb39eeafd2c9bbdbaa2',
                             'passed': True,
                             'scope': 'Asset-specific passive goal fit/support for Task6/8/9; 10mm '
                                      'already-guided idler insertion for Task7. No robot operation credit.',
                             'asset_manifest': {'path': f'{DATA_ROOT}/assets/task_parts/mechanical_assembly/nist_wire_harness/usability_v2/manifest.json',
                                                'sha256': '9443ce0e43ed40b114cef9668cc0364f85ff9658eb87aac0c464ec2d57d3f6ed'},
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
 'native_interface_evidence': {'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/development/nist_usability_20261023/root/http_task9_single02/qualification.json',
                               'sha256': '51b4046794b8dbb26c50d64865b67ab08cafec752981f26595d4bb0edd954462',
                               'passed': True,
                               'configuration_id': 'single_arm',
                               'release_manifest_sha256': 'd17ab5072ad1d49245cb715a6bf4a3f1233787483912aaa0ead8d4b06e6f3ba3',
                               'scope': 'Exact frozen-source public HTTP/Agent interface, RGB-D, read-only '
                                        'IK, wrist micro-motion/return/hold; no manipulation qualification.'}})
