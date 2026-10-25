"""Detailed task release with bounded native interface and contact evidence."""
from bench.paths import DATA_ROOT
from scenes.workrooms.laboratory.catalog_definition import task_definition

TASK = task_definition(
    'sample_sorting',
    manifest_sha256='be45333818e86c78020a730a9286829a5b8f1ba7fae591a6b121eb629a48ebad',
    asset_version='v2',
    release_version='v3',
    native_preflight=True,
    object_interactions=False,
    evidence=(
        {'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/development/laboratory_desktop_20261021/http01/sample_sorting/qualification.json',
         'sha256': '1311450e5258d4fc5e96e674c592a39221bbe8834f14a18782a19c21ff4d76a1'},
        {'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/development/laboratory_desktop_20261021/native_review01/report.json',
         'sha256': '8a49196f386aad87624749d9e4a5c05942a2a8ffdac44ae7063169ab7f7ea9ae'},
        {'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/development/laboratory_desktop_20261021/physics_preservation01/sample_sorting.json',
         'sha256': '68b28236b2acb6b622bf5ef6a94fb052eab9f1e561b709faca415291d239644e'},
    ),
)
