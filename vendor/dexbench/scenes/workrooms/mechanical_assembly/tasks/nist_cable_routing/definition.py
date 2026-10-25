"""Task8 frozen NIST adaptation; native evidence is attached separately."""
from bench.paths import DATA_ROOT
from scenes.workrooms.mechanical_assembly.nist_catalog import task_definition

TASK = task_definition(
    'nist_cable_routing',
    manifest_sha256='b16d8c79bc2e5874d7d57025e6f496c165ab5b124ace5973b3cf743bcc37b7c1',
    asset_version='v5',
    release_version='v3',
    native_preflight=True,
    object_interactions=True,
    evidence=({'kind': 'canonical_native_initialization',
      'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/development/mechanical_nist_20261020/task8/canonical03/qualification.json',
      'sha256': '53d7b64df0c223fc89bb7ed62b4adc5fb076fbc31c592b38e94787f2e8e67690',
      'passed': True,
      'scope': 'Frozen task initialization, public robot hold, read-only adjudication and four actual '
               'RGB-D cameras.'},
     {'kind': 'reset_only_object_goal',
      'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/development/mechanical_nist_20261020/task8/goal02/qualification.json',
      'sha256': 'c1006c14f332f9022d0e0c37da98a9f0a4fad4705b94f5086e9821c4bfa95059',
      'passed': True,
      'scope': 'Exact task bundle under gravity and passive constraints on the matching table support '
               'plane; fit/support only.'}),
)
