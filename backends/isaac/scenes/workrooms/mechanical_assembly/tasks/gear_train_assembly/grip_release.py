"""Derive the current desktop task with only a larger idler grasping hub."""
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import tempfile

from bench.isaacsim.release import checked, digest, dump, validate
from bench.paths import load_json, OUTPUT_ROOT
from scenes.workrooms.mechanical_assembly.release import asset_inventory
from .assets import GRIP_ASSET_ROOT, IDLER_GRIP


def prepare_release(source, output, *, asset_root=GRIP_ASSET_ROOT, screwdriver_asset_root=None):
    source, output, asset_root = (Path(p).resolve() for p in (source, output, asset_root))
    original = validate(source)
    if original['task_id'] != 'gear_train_assembly':
        raise ValueError('Idler grip revision belongs to the existing three-gear task')
    if output.exists() or not output.is_relative_to(OUTPUT_ROOT):
        raise ValueError('Choose a new immutable shared release directory')
    asset = load_json(asset_root/'manifest.json')['assets']['idler_gear']
    geometry = asset['gear_geometry']
    if any(geometry[k] != v for k, v in IDLER_GRIP.items()):
        raise ValueError('Unexpected idler grip dimensions')
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix='.'+output.name+'.preparing-', dir=output.parent))
    try:
        for relative, row in original['files'].items():
            checked(source/relative, row['sha256'])
            destination = staging/relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source/relative, destination)
        changed = {'scene.json', 'task.json', 'source_provenance.json', 'asset_file_hashes.json'}
        for name in changed:
            (staging/name).chmod(0o644)
        scene = load_json(staging/'scene.json')
        task = load_json(staging/'task.json')
        idler = next(row for row in scene['objects'] if row['id'] == 'idler_gear')
        idler['asset_root'] = str(asset_root)
        if screwdriver_asset_root is not None:
            screwdriver = next(row for row in scene['objects'] if row['id'] == 'straight_hex_screwdriver')
            screwdriver['asset_root'] = str(Path(screwdriver_asset_root).resolve())
            screwdriver.pop('usd_physics_fallbacks_provenance', None)
        task['physical_geometry']['gears']['idler_gear'] = deepcopy(geometry)
        inventory = asset_inventory(scene)
        dump(staging/'scene.json', scene)
        dump(staging/'task.json', task)
        dump(staging/'asset_file_hashes.json', inventory)
        provenance = load_json(staging/'source_provenance.json')
        revision = {'source_release': str(source), 'source_release_sha256': digest(source/'release_manifest.json'),
            'asset_manifest': str(asset_root/'manifest.json'), 'asset_manifest_sha256': digest(asset_root/'manifest.json'),
            'changed': 'Only the idler upper grip hub: diameter 50 to 64 mm, height 22 to 32 mm; integrated mass/COM/inertia follow the actual solids.',
            'preserved': ['teeth', 'bore', 'lower entry', 'robot', 'room', 'reset', 'rules', 'referee', 'physics profile', 'other five objects'],
            'scope': 'Input derivation only; validation records remain external and are not admission or scoring inputs.'}
        provenance['idler_grip_derivation'] = revision
        if screwdriver_asset_root is not None:
            revision['screwdriver_input_catalog'] = {
                'path': str(Path(screwdriver_asset_root).resolve()/'manifest.json'),
                'sha256': digest(Path(screwdriver_asset_root)/'manifest.json'),
                'scope': 'Same USD geometry, material, placement and numerical fallbacks; only historical result-file dependencies and acceptance descriptions removed.'}
        dump(staging/'source_provenance.json', provenance)
        manifest = deepcopy(original)
        manifest.update(release_id='gear_train_assembly_'+output.name,
            prepared_at_utc=datetime.now(timezone.utc).isoformat(), idler_grip_derivation=revision,
            qualification_scope='Larger idler grasping hub; no inherited native transmission or robot-manipulation qualification.')
        manifest['files'] = {name: {**row, 'sha256': digest(staging/name)} for name, row in original['files'].items()}
        manifest['files_sha256'] = {name: row['sha256'] for name, row in manifest['files'].items()}
        manifest['asset_files_sha256'] = {name: row['sha256'] for name, row in inventory.items()}
        for entry in manifest['tasks']:
            entry['scene_sha256'] = digest(staging/entry['scene'])
            entry['task_sha256'] = digest(staging/entry['task'])
        if any(row['sha256'] != manifest['files'][name]['sha256'] for name, row in original['files'].items() if name not in changed):
            raise ValueError('Grip derivation changed a protected frozen input')
        dump(staging/'release_manifest.json', manifest)
        validate(staging)
        if output.exists():
            raise FileExistsError(output)
        staging.rename(output)
        return manifest
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
