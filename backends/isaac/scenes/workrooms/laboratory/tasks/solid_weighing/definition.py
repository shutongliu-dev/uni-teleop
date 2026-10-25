"""Detailed task release with bounded native interface and contact evidence."""
from bench.paths import DATA_ROOT
from scenes.workrooms.laboratory.catalog_definition import task_definition

TASK = task_definition(
    'solid_weighing',
    manifest_sha256='c822d1458841e0aceaae4caf3522308ac0948b5f67b0861c642e5131ed88bdf1',
    asset_version='v2',
    release_version='v4',
    native_preflight=True,
    object_interactions=False,
    evidence=(
        {'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/development/laboratory_desktop_20261021/http01/solid_weighing/qualification.json',
         'sha256': '4685b5775af7e5c8f42f69058539b78f009102c6f7e2ca9815c2e368e80888fd'},
        {'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/development/laboratory_desktop_20261021/native_review01/report.json',
         'sha256': '8a49196f386aad87624749d9e4a5c05942a2a8ffdac44ae7063169ab7f7ea9ae'},
        {'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/development/laboratory_desktop_20261021/physics_preservation01/solid_weighing.json',
         'sha256': 'eff02d3f91f58d209b0094ed318698d048086797e2a274c79c7f3eab780d9b90'},
    ),
)
