"""Derive a new laundry release with only an explicit tabletop material binding."""
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import tempfile

from bench.isaacsim.release import checked, digest, dump
from .table_material import OPTION, PROFILE, table_material_options
from bench.paths import DATA_ROOT, load_json

MARKER = 'table_material_derivation'
CHANGED_FILES = frozenset({'scene.json', 'physics_profile.json', 'source_provenance.json'})
DATASETS = Path(f'{DATA_ROOT}/datasets')


def _read(path):
    return load_json(Path(path))


def _expected(source, derivation):
    scene = _read(source/'scene.json')
    if OPTION in scene['isaacsim']:
        raise ValueError('Source already selects workroom materials; choose its original unbound release')
    if scene.get('table', {}).get('friction') != .45:
        raise ValueError('Laundry table profile requires the original declared friction 0.45')
    profile = _read(source/'physics_profile.json')
    if profile['isaacsim_options'] != scene['isaacsim']:
        raise ValueError('Source scene and frozen physics options differ')
    options = table_material_options(derivation['profile'])
    scene['isaacsim'][OPTION] = deepcopy(options)
    profile['isaacsim_options'][OPTION] = deepcopy(options)
    provenance = _read(source/'source_provenance.json')
    provenance[MARKER] = deepcopy(derivation)
    return {'scene.json': scene, 'physics_profile.json': profile, 'source_provenance.json': provenance}


def validate_derivation(root, manifest, *, assets=True):
    """Check the narrow original-to-candidate change, including unchanged scorer bytes."""
    from .release import validate_release, SCHEMA
    derivation = manifest[MARKER]
    expected_keys = {'schema', 'profile', 'base_release', 'source_snapshot', 'source_snapshot_manifest', 'original_release_modified',
                     'qualification_inherited', 'changed_files', 'scope'}
    if (set(derivation) != expected_keys or derivation['schema'] != 'laundry_table_material_derivation.v1'
            or derivation['profile'] != PROFILE or derivation['original_release_modified'] is not False
            or derivation['qualification_inherited'] is not False
            or derivation['changed_files'] != sorted(CHANGED_FILES)):
        raise ValueError('Invalid laundry table material derivation')
    source = Path(derivation['base_release']['path']).resolve()
    root = Path(root).resolve()
    if (source == root or root.is_relative_to(source) or source.is_relative_to(root)
            or not source.is_relative_to(DATASETS)):
        raise ValueError('Table material candidate requires an independent original release')
    checked(source/'release_manifest.json', derivation['base_release']['manifest_sha256'])
    snapshot = derivation['source_snapshot_manifest']
    snapshot_path = Path(snapshot['path']).resolve()
    if not snapshot_path.is_relative_to(DATASETS):
        raise ValueError('Table material provenance requires a frozen source under shared datasets')
    checked(snapshot_path, snapshot['sha256'])
    original = _read(source/'release_manifest.json')
    if original.get('schema') != SCHEMA or MARKER in original:
        raise ValueError('Table material source must be an original laundry release')
    validate_release(source, original, assets=assets)
    if set(manifest['files']) != set(original['files']):
        raise ValueError('Table material derivative changed its frozen file inventory')
    for name, value in _expected(source, derivation).items():
        if _read(root/name) != value:
            raise ValueError('Unexpected table material derivative change: '+name)
    for name, row in original['files'].items():
        if name not in CHANGED_FILES:
            checked(root/name, row['sha256'])
    allowed = {'release_id', 'prepared_at_utc', 'files', 'files_sha256', 'tasks',
               'physics_profile_sha256', MARKER}
    if {k: v for k, v in manifest.items() if k not in allowed} != {
            k: v for k, v in original.items() if k not in allowed}:
        raise ValueError('Table material derivative changed unrelated release metadata')
    entries = deepcopy(original['tasks'])
    entries[0]['scene_sha256'] = manifest['files']['scene.json']['sha256']
    if manifest['tasks'] != entries:
        raise ValueError('Table material derivative changed task entries')


def prepare(source, output, *, profile=PROFILE):
    from .release import validate_release, SCHEMA
    from scripts.maintenance.freeze_project_sources import verify_if_snapshot
    project = Path(__file__).resolve().parents[3]
    snapshot = verify_if_snapshot(project, required_modules=(
        'scenes.workrooms.laundry_room.table_material',
        'scenes.workrooms.laundry_room.table_material_release',
        'scenes.workrooms.laundry_room.composition',
        'scenes.workrooms.laundry_room.release'), entrypoint_file=Path(__file__))
    if snapshot is None:
        raise ValueError('Prepare table material candidates from a complete frozen source snapshot')
    table_material_options(profile)
    source, output = Path(source).resolve(), Path(output).resolve()
    if (not source.is_relative_to(DATASETS) or not output.is_relative_to(DATASETS)
            or output.exists() or output.is_relative_to(source) or source.is_relative_to(output)):
        raise ValueError('Choose a fresh independent candidate under shared datasets')
    original = _read(source/'release_manifest.json')
    if original.get('schema') != SCHEMA or MARKER in original:
        raise ValueError('Table material source must be an original laundry release')
    validate_release(source, original)
    base = {'path': str(source), 'manifest_sha256': digest(source/'release_manifest.json')}
    snapshot_path = Path(snapshot['snapshot_root'])/'source_snapshot_manifest.json'
    derivation = {'schema': 'laundry_table_material_derivation.v1', 'profile': profile,
        'base_release': base, 'source_snapshot': snapshot,
        'source_snapshot_manifest': {'path': str(snapshot_path), 'sha256': digest(snapshot_path)},
        'original_release_modified': False,
        'qualification_inherited': False, 'changed_files': sorted(CHANGED_FILES),
        'scope': 'Explicit tabletop binding only; task, reset, cloth material, robot, frozen evaluator and fixed room assets are unchanged. Native binding/contact qualification is required independently.'}
    values = _expected(source, derivation)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix='.'+output.name+'.preparing-', dir=output.parent))
    try:
        for name, row in original['files'].items():
            target = staging/name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source/name, target)
            checked(target, row['sha256'])
        for name, value in values.items():
            dump(staging/name, value)
        manifest = deepcopy(original)
        manifest.update(release_id='laundry_'+manifest['task_id']+'_'+output.name,
                        prepared_at_utc=datetime.now(timezone.utc).isoformat())
        manifest[MARKER] = derivation
        for name in CHANGED_FILES:
            manifest['files'][name]['sha256'] = digest(staging/name)
            manifest['files_sha256'][name] = manifest['files'][name]['sha256']
        manifest['tasks'][0]['scene_sha256'] = manifest['files']['scene.json']['sha256']
        manifest['physics_profile_sha256'] = manifest['files']['physics_profile.json']['sha256']
        dump(staging/'release_manifest.json', manifest)
        validate_release(staging, manifest)
        checked(source/'release_manifest.json', base['manifest_sha256'])
        if output.exists():
            raise FileExistsError(output)
        staging.rename(output)
        return manifest
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
