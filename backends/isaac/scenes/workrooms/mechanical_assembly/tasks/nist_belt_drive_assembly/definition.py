"""Explicit NIST usability revision; frozen evidence is attached separately."""
from bench.paths import DATA_ROOT
from scenes.workrooms.mechanical_assembly.nist_catalog import task_definition

TASK = task_definition(
    'nist_belt_drive_assembly',
    manifest_sha256='ec43adbeec31682e74923e0c5220e00c33bfaae8a7411abadea8359ab34b0fde',
    asset_version='usability_v5',
    release_version='usability_desktop',
    usability_profile='nist_manipulation_scale_v1',
)

# usability_v5 moves the closed belt supply loop 1 mm along -y, clear of the
# tensioner support shoulder (nist_assets.BELT_SUPPLY_CENTER_XY_M); every other
# asset value is identical to usability_v3.
TASK["source_release"] = {'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/isaacsim/releases/nist_usability_20261107b/batch01/nist_belt_drive_assembly/desktop_v1',
 'manifest_sha256': 'ec43adbeec31682e74923e0c5220e00c33bfaae8a7411abadea8359ab34b0fde'}
TASK["previous_source_release"] = {'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/isaacsim/releases/nist_usability_20261023/batch02/nist_belt_drive_assembly/desktop_v1',
 'manifest_sha256': '4e1b0371aee0483a9a11a11f3bdb78d040aa45e7293233a5a52d6cc465b53f62'}
TASK["objects"]["background_supplies"] = {'groups': 31,
 'movable': False,
 'scoring_support': False,
 'description': 'Fixed edge tool caddy, hex keys, stocked parts bins and caliper; the active task '
                'inventory is unchanged.'}
TASK["qualification"]["workspace_discrete"] = True
TASK["qualification"]["scene_geometry_evidence"] = {'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/development/parameterized_seeds_20261105/beltbase45_evidence02/workspace01/nist_belt_drive_assembly_single_arm.json',
 'sha256': 'cae2cba5f640d5e52d9ccc4e133da1d3821c5ec0eddad067c0200f9e5547a832',
 'passed': True,
 'scope': 'Discrete actual fingertip approach and original USD collision bounds; no continuous '
          'path or manipulation qualification.'}
TASK["qualification"]["workspace_release_binding"] = {'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/development/parameterized_seeds_20261105/beltbase45_evidence02/workspace_binding01.json',
 'sha256': 'e3928b9b3369c37cc0022eca242da9b09bfa2a7e8787d1d6ad5afa119493285f'}
TASK["qualification"]["historical_usability_v3_evidence"] = {'source_release': {'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/isaacsim/releases/nist_usability_20261023/batch02/nist_belt_drive_assembly/desktop_v1',
                    'manifest_sha256': '4e1b0371aee0483a9a11a11f3bdb78d040aa45e7293233a5a52d6cc465b53f62'},
 'object_interactions': True,
 'scene_contact_evidence': [{'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/development/nist_usability_20261023/task7/guided01/qualification.json',
                             'sha256': 'd6a76fd3267185c88dc7da4af7fc5d2b8ee3d3fb750dd79458a41c49926db69f',
                             'passed': True,
                             'scope': 'Asset-specific passive goal fit/support for Task6/8/9; 10mm '
                                      'already-guided idler insertion for Task7. No robot '
                                      'operation credit.',
                             'asset_manifest': {'path': f'{DATA_ROOT}/assets/task_parts/mechanical_assembly/nist_belt_drive_assembly/usability_v3/manifest.json',
                                                'sha256': '3e30ffadf4ba290c936336774c6bfd4409ef070570119b2fd55c9bd8a2b3a3dd'},
                             'qualification_index': {'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/development/nist_usability_20261023/task7/qualification.json',
                                                     'sha256': 'c92c088b20d45cc4e82fa3e9c7c6866aa602e41e9d81f4cace031eea3ee028a5'},
                             'asset_identity_bridge': {'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/development/nist_usability_20261023/task7/object_identity_bridge01.json',
                                                       'sha256': 'f1d61163062a8b954e9d5e15fa32cef63ca54539b66cbf36395cce07791ef992'}}],
 'workspace_release_binding': {'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/development/nist_usability_20261023/root/workspace_binding03.json',
                               'sha256': 'b3c6755b71c8dcd1bf6a7ef3e6c082e0aafd9c2629b3731bb01a82d66a9b286c'},
 'scope': 'Bound to the superseded usability_v3 base. The usability_v5 base only moves the 64 belt '
          'link supply poses 1 mm along -y, clear of the tensioner shoulder; asset-specific '
          'physical evidence is not inherited and needs a new native run.'}
