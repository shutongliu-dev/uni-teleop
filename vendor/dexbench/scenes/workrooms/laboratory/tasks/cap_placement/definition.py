"""Detailed task release with bounded native interface and contact evidence."""
from bench.paths import DATA_ROOT
from scenes.workrooms.laboratory.catalog_definition import task_definition

TASK = task_definition(
    'cap_placement',
    manifest_sha256='5f26dbed4559d0111a56c1fa3b3fd07d0910ad55457df47fef34d1709c68202c',
    asset_version='v3',
    release_version='v4',
    native_preflight=True,
    object_interactions=False,
    evidence=(
        {'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/development/laboratory_desktop_20261021/http01/cap_placement/qualification.json',
         'sha256': '35e2e29fe5b414ed1b50c3bbc2a85930d61a93fd4b24c5f1327ac0927efbdb8b'},
        {'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/development/laboratory_desktop_20261021/native_review01/report.json',
         'sha256': '8a49196f386aad87624749d9e4a5c05942a2a8ffdac44ae7063169ab7f7ea9ae'},
        {'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/development/laboratory_desktop_20261021/physics_preservation01/cap_placement.json',
         'sha256': 'a08c5d5d9962ae3d4a4718bf9585deee4fb95df5ea00db8f6f4919d261ad15a4'},
    ),
)
