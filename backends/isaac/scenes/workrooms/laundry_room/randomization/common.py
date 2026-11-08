"""Shared strict JSON/source checks; no USD or native initialization imports."""
from copy import deepcopy
from pathlib import Path

from bench.paths import DATA_ROOT
from scenes.workrooms.initial_randomization.common import (
    canonical_hash, checked_file, file_hash, local_path, read_json, text_hash, write_json)
from . import CONFIGURATION_IDS, TASK_IDS

ASSET_ROOT = Path(f'{DATA_ROOT}/assets/task_parts/laundry_seed_randomization/v1')
RELEASE_ROOT = Path(f'{DATA_ROOT}/datasets/fr3_sharpa_bench/isaacsim/releases/laundry_initial_randomization_v1')
SOURCE_ROOT = Path(__file__).resolve().parents[4]
SOURCE_FILES = tuple('scenes/workrooms/laundry_room/randomization/'+n+'.py'
                     for n in ('__init__', 'common', 'cases', 'appearance', 'author', 'release'))
PARAMETERIZED_SOURCE_FILES = SOURCE_FILES + (
    'scenes/workrooms/laundry_room/randomization/parameterized.py',
    'scenes/workrooms/laundry_room/randomization/parameterized_appearance.py',
    'scenes/workrooms/parameterized/spec.py', 'scenes/workrooms/parameterized/appearance.py')


def parent_inputs(root, *, assets=True):
    """Only the fifteen existing completion-input releases may be parents."""
    from bench.isaacsim.release import validate
    from ..robot_configuration_catalog import RELEASES
    root = Path(root).resolve()
    manifest = read_json(root/'release_manifest.json')
    task_id, configuration = manifest.get('task_id'), manifest.get('configuration_id')
    if (manifest.get('schema') != 'laundry_cloth_task_release.v1'
            or task_id not in TASK_IDS or configuration not in CONFIGURATION_IDS):
        raise ValueError('Laundry randomization needs an original configured cloth release')
    pin = RELEASES[task_id]['configurations'][configuration]['source_release']
    if root != Path(pin['path']).resolve() or root.name != 'completion_inputs_v1':
        raise ValueError('Parent must be the current immutable completion_inputs_v1 release')
    checked_file(root/'release_manifest.json', pin['manifest_sha256'])
    validate(root, assets=assets)
    scene, task, initial = [read_json(root/(n+'.json')) for n in ('scene', 'task', 'initial_state')]
    if task['robot_configuration_id'] != configuration or task['id'] != task_id:
        raise ValueError('Parent task/installation identity differs')
    references = {str(root/'release_manifest.json'): pin['manifest_sha256'],
                  **{str(root/n): h for n, h in manifest['files_sha256'].items()}}
    return manifest, scene, task, initial, references


def source_provenance(*, parameterized=False):
    from scripts.maintenance.freeze_project_sources import verify_if_snapshot
    source = verify_if_snapshot(SOURCE_ROOT, required_modules=(__name__,),
                                entrypoint_file=Path(__file__))
    if not source or source.get('runtime_imports_verified') is not True:
        raise ValueError('Laundry publication requires complete frozen source imports')
    return {'source_snapshot': source,
            'source_manifest_sha256': file_hash(SOURCE_ROOT/'source_snapshot_manifest.json'),
            'author_sources_sha256': {n: file_hash(SOURCE_ROOT/n) for n in
                                     (PARAMETERIZED_SOURCE_FILES if parameterized else SOURCE_FILES)}}


def verify_provenance(record):
    source = record.get('source_snapshot')
    if not isinstance(source, dict) or source.get('runtime_imports_verified') is not True:
        raise ValueError('Missing frozen author source identity')
    root = Path(source.get('snapshot_root', ''))
    if not root.is_absolute():
        raise ValueError('Frozen author source root must be absolute')
    path = checked_file(root/'source_snapshot_manifest.json', record.get('source_manifest_sha256'))
    manifest = read_json(path)
    hashes = record.get('author_sources_sha256')
    if (manifest.get('schema') != 'complete_benchmark_source_snapshot_v1'
            or not isinstance(hashes, dict)
            or set(hashes) not in (set(SOURCE_FILES), set(PARAMETERIZED_SOURCE_FILES))):
        raise ValueError('Incomplete frozen laundry author source inventory')
    for name, expected in hashes.items():
        if manifest.get('files_sha256', {}).get(name) != expected:
            raise ValueError('Laundry author differs from historical source snapshot')
        checked_file(root/name, expected)
    return deepcopy(record)


def checked_inventory(values, *, verify=True):
    if not isinstance(values, dict) or not values:
        raise ValueError('Nonempty input hash inventory required')
    for path, expected in values.items():
        if (not Path(path).is_absolute() or not isinstance(expected, str)
                or len(expected) != 64 or any(c not in '0123456789abcdef' for c in expected)):
            raise ValueError('Absolute file and lowercase SHA256 required')
        if verify:
            checked_file(path, expected)
    return values
