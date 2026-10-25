"""Detailed task release with bounded native interface and contact evidence."""
from bench.paths import DATA_ROOT
from scenes.workrooms.laboratory.catalog_definition import task_definition

TASK = task_definition(
    'pipette_transfer',
    manifest_sha256='864d63fc44c57498f5c22dbb29a935e207eecfd8fb77dfaac89337abb37044c4',
    asset_version='v2',
    release_version='v2',
    native_preflight=True,
    object_interactions=False,
    evidence=(
        {'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/development/laboratory_desktop_20261021/http01/pipette_transfer/qualification.json',
         'sha256': 'f4606c72513ae4c6f9bcfe5d4fbd3b1f21098469bedf9a2bfa04c2db41b5b222'},
        {'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/development/laboratory_desktop_20261021/native_review01/report.json',
         'sha256': '8a49196f386aad87624749d9e4a5c05942a2a8ffdac44ae7063169ab7f7ea9ae'},
        {'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/development/laboratory_desktop_20261021/physics_preservation01/pipette_transfer.json',
         'sha256': '70a98ae5c487cd185daaffd124a560aad732b8feab35256405a4c2ece5983cbb'},
    ),
)
