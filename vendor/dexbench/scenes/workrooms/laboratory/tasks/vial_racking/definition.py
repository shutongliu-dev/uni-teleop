"""Detailed task release with bounded native interface and contact evidence."""
from bench.paths import DATA_ROOT
from scenes.workrooms.laboratory.catalog_definition import task_definition

TASK = task_definition(
    'vial_racking',
    manifest_sha256='ff4421e5dfdd7c1a7681bac4d233fdc267f74b9bc9a1149cb1ea00ad0f8123d4',
    asset_version='v3',
    release_version='v4',
    native_preflight=True,
    object_interactions=False,
    evidence=(
        {'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/development/laboratory_desktop_20261021/http02/vial_racking/qualification.json',
         'sha256': '25ec01a5587b3a1d0359490354bc8281928de0da9fa6b58b976488870ee24279'},
        {'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/development/laboratory_desktop_20261021/native_review02/report.json',
         'sha256': '8f74d0afab6f59b16d10d541e808cc0a96c6cbc33aeeb90fc7c482c225cdd658'},
        {'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/development/laboratory_desktop_20261021/physics_preservation02/vial_racking.json',
         'sha256': '25b8ec067a026ec8e2f2cd7962aa51b19972c741006e01ede63871f7b595cba2'},
    ),
)
