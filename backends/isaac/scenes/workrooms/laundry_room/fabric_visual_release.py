"""Derive a render-only laundry release that pins baked fabric visual profiles.

The base release (an original or a table-material candidate) keeps every physical,
reset, rule, robot and evaluator input. Only its scene cloth specifications gain
hash pins of fabric visual profiles, and its shared asset inventory gains the
pinned profile and texture files. Each profile must have been baked for exactly
that cloth's rest mesh and frozen dye. Native rendering needs its own evidence.
"""
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import tempfile

from bench.isaacsim.release import checked, digest, dump
from bench.paths import DATA_ROOT, load_json
from bench.simulation.isaacsim_backend.fabric_visuals import cloth_geometry_digest, load_visual_profile

MARKER = 'fabric_visual_derivation'
CHANGED_FILES = frozenset({'scene.json', 'asset_file_hashes.json', 'source_provenance.json'})
DATASETS = Path(f'{DATA_ROOT}/datasets')
ASSETS = Path(f'{DATA_ROOT}/assets')


def _read(path):
    return load_json(Path(path))


def check_profile_binding(scene, profiles):
    """Every pin names a scene cloth and a profile baked for its rest mesh and dye."""
    from bench.simulation.isaacsim_backend.geometry import cloth_grid
    cloths = {cloth['id']: cloth for cloth in scene['cloths']}
    if not profiles or set(profiles) - set(cloths):
        raise ValueError('Fabric visual profiles must name existing cloth ids')
    for cloth_id, pin in profiles.items():
        if not Path(pin['path']).resolve().is_relative_to(ASSETS):
            raise ValueError('Fabric visual profiles must be immutable shared assets')
        binding = load_visual_profile(pin)['cloth_binding']
        cloth = cloths[cloth_id]
        if (binding['cloth_geometry_sha256'] != cloth_geometry_digest(*cloth_grid(cloth))
                or binding['dye_linear_rgb'] != cloth['isaacsim']['cloth_visual_color']):
            raise ValueError('Fabric visual profile was baked for a different cloth: '+cloth_id)


def profile_inventory(profiles):
    """Shared-asset pins added by the derivation: each profile and its baked maps."""
    hashes = {}
    for pin in profiles.values():
        profile = load_visual_profile(pin)
        hashes[pin['path']] = pin['sha256']
        hashes.update({path: profile['texture_sha256'][role] for role, path in profile['textures'].items()})
    return hashes


def _expected(source, derivation, *, assets=True):
    scene = _read(source/'scene.json')
    if any('visual_profile' in cloth for cloth in scene['cloths']):
        raise ValueError('Source already pins fabric visual profiles')
    for cloth in scene['cloths']:
        if cloth['id'] in derivation['profiles']:
            cloth['visual_profile'] = deepcopy(derivation['profiles'][cloth['id']])
    provenance = _read(source/'source_provenance.json')
    provenance[MARKER] = deepcopy(derivation)
    values = {'scene.json': scene, 'source_provenance.json': provenance}
    if assets:
        check_profile_binding(scene, derivation['profiles'])
        hashes = {path: row['sha256'] for path, row in _read(source/'asset_file_hashes.json').items()}
        hashes.update(profile_inventory(derivation['profiles']))
        values['asset_file_hashes.json'] = {path: {'sha256': sha} for path, sha in sorted(hashes.items())}
    return values


def validate_derivation(root, manifest, *, assets=True):
    """Check the narrow base-to-candidate change, including unchanged physics and scorer bytes."""
    from .release import validate_release, SCHEMA
    derivation = manifest[MARKER]
    expected_keys = {'schema', 'profiles', 'base_release', 'source_snapshot', 'source_snapshot_manifest',
                     'original_release_modified', 'qualification_inherited', 'physics_changed',
                     'changed_files', 'scope'}
    if (set(derivation) != expected_keys or derivation['schema'] != 'laundry_fabric_visual_derivation.v1'
            or derivation['original_release_modified'] is not False
            or derivation['qualification_inherited'] is not False or derivation['physics_changed'] is not False
            or derivation['changed_files'] != sorted(CHANGED_FILES)):
        raise ValueError('Invalid laundry fabric visual derivation')
    source = Path(derivation['base_release']['path']).resolve()
    root = Path(root).resolve()
    if (source == root or root.is_relative_to(source) or source.is_relative_to(root)
            or not source.is_relative_to(DATASETS)):
        raise ValueError('Fabric visual candidate requires an independent base release')
    checked(source/'release_manifest.json', derivation['base_release']['manifest_sha256'])
    snapshot_path = Path(derivation['source_snapshot_manifest']['path']).resolve()
    if not snapshot_path.is_relative_to(DATASETS):
        raise ValueError('Fabric visual provenance requires a frozen source under shared datasets')
    checked(snapshot_path, derivation['source_snapshot_manifest']['sha256'])
    base = _read(source/'release_manifest.json')
    if base.get('schema') != SCHEMA or MARKER in base:
        raise ValueError('Fabric visual base must be a laundry release without fabric profiles')
    validate_release(source, base, assets=assets)
    if set(manifest['files']) != set(base['files']):
        raise ValueError('Fabric visual derivative changed its frozen file inventory')
    for name, value in _expected(source, derivation, assets=assets).items():
        if _read(root/name) != value:
            raise ValueError('Unexpected fabric visual derivative change: '+name)
    if not assets:
        # Without shared assets, the base inventory must still be carried unchanged.
        derived_inventory = _read(root/'asset_file_hashes.json')
        if any(derived_inventory.get(path) != row for path, row in _read(source/'asset_file_hashes.json').items()):
            raise ValueError('Fabric visual derivative changed base asset pins')
    for name, row in base['files'].items():
        if name not in CHANGED_FILES:
            checked(root/name, row['sha256'])
    allowed = {'release_id', 'prepared_at_utc', 'files', 'files_sha256', 'tasks', 'asset_files_sha256', MARKER}
    if {k: v for k, v in manifest.items() if k not in allowed} != {k: v for k, v in base.items() if k not in allowed}:
        raise ValueError('Fabric visual derivative changed unrelated release metadata')
    entries = deepcopy(base['tasks'])
    entries[0]['scene_sha256'] = manifest['files']['scene.json']['sha256']
    if manifest['tasks'] != entries:
        raise ValueError('Fabric visual derivative changed task entries')


def prepare(source, output, profiles):
    """Copy a validated release to a fresh candidate whose cloths pin the given profiles."""
    from .release import validate_release, SCHEMA
    from scripts.maintenance.freeze_project_sources import verify_if_snapshot
    project = Path(__file__).resolve().parents[3]
    snapshot = verify_if_snapshot(project, required_modules=(
        'scenes.workrooms.laundry_room.fabric_visual_release',
        'scenes.workrooms.laundry_room.composition',
        'scenes.workrooms.laundry_room.release',
        'bench.simulation.isaacsim_backend.fabric_visuals'), entrypoint_file=Path(__file__))
    if snapshot is None:
        raise ValueError('Prepare fabric visual candidates from a complete frozen source snapshot')
    source, output = Path(source).resolve(), Path(output).resolve()
    if (not source.is_relative_to(DATASETS) or not output.is_relative_to(DATASETS)
            or output.exists() or output.is_relative_to(source) or source.is_relative_to(output)):
        raise ValueError('Choose a fresh independent candidate under shared datasets')
    base = _read(source/'release_manifest.json')
    if base.get('schema') != SCHEMA or MARKER in base:
        raise ValueError('Fabric visual base must be a laundry release without fabric profiles')
    validate_release(source, base)
    pins = {cloth_id: {'path': str(Path(path).resolve()), 'sha256': digest(path)} for cloth_id, path in profiles.items()}
    snapshot_path = Path(snapshot['snapshot_root'])/'source_snapshot_manifest.json'
    derivation = {'schema': 'laundry_fabric_visual_derivation.v1', 'profiles': pins,
        'base_release': {'path': str(source), 'manifest_sha256': digest(source/'release_manifest.json')},
        'source_snapshot': snapshot,
        'source_snapshot_manifest': {'path': str(snapshot_path), 'sha256': digest(snapshot_path)},
        'original_release_modified': False, 'qualification_inherited': False, 'physics_changed': False,
        'changed_files': sorted(CHANGED_FILES),
        'scope': 'Render-only cloth appearance: hash-pinned baked fabric textures on a read-only skin. Physics, reset, rules, robot, evaluator and room inputs are unchanged; native rendering evidence is separate.'}
    values = _expected(source, derivation)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix='.'+output.name+'.preparing-', dir=output.parent))
    try:
        for name, row in base['files'].items():
            target = staging/name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source/name, target)
            checked(target, row['sha256'])
        for name, value in values.items():
            dump(staging/name, value)
        manifest = deepcopy(base)
        manifest.update(release_id='laundry_'+manifest['task_id']+'_'+output.name,
                        prepared_at_utc=datetime.now(timezone.utc).isoformat())
        manifest[MARKER] = derivation
        for name in CHANGED_FILES:
            manifest['files'][name]['sha256'] = digest(staging/name)
            manifest['files_sha256'][name] = manifest['files'][name]['sha256']
        manifest['tasks'][0]['scene_sha256'] = manifest['files']['scene.json']['sha256']
        manifest['asset_files_sha256'] = {path: row['sha256'] for path, row in values['asset_file_hashes.json'].items()}
        dump(staging/'release_manifest.json', manifest)
        validate_release(staging, manifest)
        checked(source/'release_manifest.json', derivation['base_release']['manifest_sha256'])
        if output.exists():
            raise FileExistsError(output)
        staging.rename(output)
        return manifest
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
