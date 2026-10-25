"""Task7 frozen NIST adaptation; native evidence is attached separately."""
from bench.paths import DATA_ROOT
from scenes.workrooms.mechanical_assembly.nist_catalog import task_definition

TASK = task_definition(
    'nist_belt_drive_assembly',
    manifest_sha256='71728cd71e973b5e377289201ce76c812410bafa5a75c40754b2e5c002496b0e',
    asset_version='v5',
    release_version='v2',
    native_preflight=True,
    object_interactions=True,
    evidence=({'kind': 'canonical_native_initialization',
      'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/development/mechanical_nist_20261020/task7/canonical02/qualification.json',
      'sha256': '5f82d764ea8f20d5b5231b22355c1001099c02313f52a3b6f1fbd07ca85eee7a',
      'passed': True,
      'scope': 'Frozen task initialization, public robot hold, read-only adjudication and four actual '
               'RGB-D cameras.'},
     {'kind': 'reset_only_object_goal',
      'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/development/mechanical_nist_20261020/task7/goal01/qualification.json',
      'sha256': '3970748d8ce3b2f847409dcaf7c0eebb9cd44ea827c259a54a4691e822bdfc9a',
      'passed': True,
      'scope': 'Exact task bundle under gravity and passive constraints on the matching table support '
               'plane; fit/support only.'}),
)
