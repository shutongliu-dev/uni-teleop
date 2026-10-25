"""Task9 frozen NIST adaptation; native evidence is attached separately."""
from bench.paths import DATA_ROOT
from scenes.workrooms.mechanical_assembly.nist_catalog import task_definition

TASK = task_definition(
    'nist_wire_harness',
    manifest_sha256='fa7e3265b195db1f266d96e97ddbf32145d15e0bdd5cc6415eeaade11793b766',
    asset_version='v4',
    release_version='v3',
    native_preflight=True,
    object_interactions=True,
    evidence=({'kind': 'canonical_native_initialization',
      'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/development/mechanical_nist_20261020/task9/canonical03/qualification.json',
      'sha256': 'b07fd8e33312c87bd44635551acf753f3506b274468ac819bb3b578068953555',
      'passed': True,
      'scope': 'Frozen task initialization, public robot hold, read-only adjudication and four actual '
               'RGB-D cameras.'},
     {'kind': 'reset_only_object_goal',
      'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/development/mechanical_nist_20261020/task9/goal02/qualification.json',
      'sha256': 'fa69e3bc30c9fcb63a3e923a927fa5c3583824bb3c176d9437fc46cff2fdedaa',
      'passed': True,
      'scope': 'Exact task bundle under gravity and passive constraints on the matching table support '
               'plane; fit/support only.'}),
)
