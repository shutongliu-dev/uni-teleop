"""Explicit NIST usability revision; frozen evidence is attached separately."""
from bench.paths import DATA_ROOT
from scenes.workrooms.mechanical_assembly.nist_catalog import task_definition

TASK = task_definition(
    'nist_cable_routing',
    manifest_sha256='4106fa53fbadf5d096c11522ada8dc85e6588588305b2141cd3590eddfc404c7',
    asset_version='usability_v2',
    release_version='usability_desktop',
    usability_profile='nist_manipulation_scale_v1',
)

TASK["source_release"] = {'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/isaacsim/releases/nist_usability_20261023/batch02/nist_cable_routing/desktop_v1',
 'manifest_sha256': '4106fa53fbadf5d096c11522ada8dc85e6588588305b2141cd3590eddfc404c7'}
TASK["previous_source_release"] = {'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/isaacsim/releases/mechanical_assembly/nist_cable_routing/desktop_v2',
 'manifest_sha256': '4b765f28ed389f6243d810ec4eed844c30698b1e64761a86ec4a191112523f6e'}
TASK["objects"]["background_supplies"] = {'groups': 31,
 'movable': False,
 'scoring_support': False,
 'description': 'Fixed edge tool caddy, hex keys, stocked parts bins and caliper; the active task inventory '
                'is unchanged.'}
TASK["qualification"]["workspace_discrete"] = True
TASK["qualification"]["scene_geometry_evidence"] = {'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/development/nist_usability_20261023/root/workspace02/nist_cable_routing_single_arm.json',
 'sha256': '9304050f7e0df9ed53493aaaf4e888e97e6790eeaf5e99fe7080da5e0cf22ce5',
 'passed': True,
 'scope': 'Discrete actual fingertip approach and original USD collision bounds; no continuous path or '
          'manipulation qualification.'}

# Completed physical evidence; immutable preparation flags remain unchanged.
TASK["qualification"].update({'object_interactions': True,
 'scene_contact_evidence': [{'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/development/nist_usability_20261023/scale_audit/native_task8_goal02/qualification.json',
                             'sha256': '356c576686245f102ff1e56e3e1e66d649463670e6d23b0fad2e436b18ebcc45',
                             'passed': True,
                             'scope': 'Asset-specific passive goal fit/support for Task6/8/9; 10mm '
                                      'already-guided idler insertion for Task7. No robot operation credit.',
                             'asset_manifest': {'path': f'{DATA_ROOT}/assets/task_parts/mechanical_assembly/nist_cable_routing/usability_v2/manifest.json',
                                                'sha256': 'cd2f5f48258306ae8f0a3b4ff79c598c4c7e652ae371c9d6f3a637b527e29be5'},
                             'qualification_index': {'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/development/nist_usability_20261023/scale_audit/qualification02.json',
                                                     'sha256': 'a9ce0d252506a14e28cba028091ad977bab2a77a46b7bb672296fd873f49e771'},
                             'asset_identity_bridge': {'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/development/nist_usability_20261023/scale_audit/root-independent-identity-review02.json',
                                                       'sha256': '0f371c44ac87663055d2332be0c276fef810a9af83fff104c4f1b343c286d2bb'}}],
 'workspace_release_binding': {'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/development/nist_usability_20261023/root/workspace_binding03.json',
                               'sha256': 'b3c6755b71c8dcd1bf6a7ef3e6c082e0aafd9c2629b3731bb01a82d66a9b286c'},
 'manipulation': False,
 'physics_accepted': False})
