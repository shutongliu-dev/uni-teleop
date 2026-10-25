"""Task6 frozen NIST adaptation; native evidence is attached separately."""
from bench.paths import DATA_ROOT
from scenes.workrooms.mechanical_assembly.nist_catalog import task_definition

TASK = task_definition(
    'nist_connector_insertion',
    manifest_sha256='4d969ae536cc57d1ecf19fe5c12d118e85bf4e3362a8e144c3c824be0d852158',
    asset_version='v3',
    release_version='v5',
    native_preflight=True,
    object_interactions=True,
    evidence=({'kind': 'canonical_native_initialization',
      'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/development/mechanical_nist_20261020/task6/canonical04/qualification.json',
      'sha256': 'e03e2d692cc86befe3b54b2ffa80e9dba3ec3d6c602336e193f52a0660d45959',
      'passed': True,
      'scope': 'Frozen task initialization, public robot hold, read-only adjudication and four actual '
               'RGB-D cameras.'},
     {'kind': 'reset_only_object_goal',
      'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/development/mechanical_nist_20261020/task6/goal03/qualification.json',
      'sha256': '479ec023dae65bbeaf6604e10e8b05f31f59d4582d32d90105c7a7fd42ddb607',
      'passed': True,
      'scope': 'Exact task bundle under gravity and passive constraints on the matching table support '
               'plane; fit/support only.'}),
)
