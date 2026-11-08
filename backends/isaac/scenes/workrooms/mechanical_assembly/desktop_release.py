"""Derive a new task release by adding fixed, non-scoring workshop supplies."""
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
import json
import shutil
import tempfile

from bench.isaacsim.release import checked, digest, dump, validate
from .desktop_dressing import ASSET_ROOT, scene_dependencies, task_layer
from bench.paths import load_json, OUTPUT_ROOT


def prepare_release(source, output, *, dressing_manifest=ASSET_ROOT/'manifest.json'):
    """Preserve the selected task, scorer, reset and physics bytes exactly."""
    source, output = Path(source).resolve(), Path(output).resolve()
    original = validate(source)
    schema = original.get('schema')
    inventories = {
        'mechanical_assembly_task_release.v1': 'scenes.workrooms.mechanical_assembly.release',
        'mechanical_nist_task_release.v1': 'scenes.workrooms.mechanical_assembly.nist_release',
        'mechanical_fastener_task_release.v1': 'scenes.workrooms.mechanical_assembly.tasks.fastener_assembly.release',
    }
    if schema not in inventories:
        raise ValueError('Desktop derivation requires an existing mechanical task release')
    if not output.is_relative_to(OUTPUT_ROOT) or output.exists():
        raise ValueError('Choose a new release directory under shared datasets')
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix='.'+output.name+'.preparing-', dir=output.parent))
    try:
        for relative, row in original['files'].items():
            checked(source/relative, row['sha256'])
            destination = staging/relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source/relative, destination)
        scene = load_json(staging/'scene.json')
        if scene.get('mechanical_desktop_dressing') is not None:
            raise ValueError('Source task already declares workshop desktop supplies')
        scene['task_layers'] = [*scene.get('task_layers', []), task_layer(dressing_manifest)]
        scene['mechanical_desktop_dressing'] = {
            'manifest_path': str(Path(dressing_manifest).resolve()),
            'manifest_sha256': digest(dressing_manifest),
        }
        # Only newly copied files are made writable; the source stays immutable.
        for relative in ('scene.json', 'source_provenance.json', 'asset_file_hashes.json'):
            (staging/relative).chmod(0o644)
        dump(staging/'scene.json', scene)
        import importlib
        module = importlib.import_module(inventories[schema])
        inventory_fn = module._asset_inventory if schema == 'mechanical_fastener_task_release.v1' else module.asset_inventory
        inventory = inventory_fn(scene)
        if not set(scene_dependencies(scene)) <= set(inventory):
            raise ValueError('Derived release does not inventory all desktop dependencies')
        dump(staging/'asset_file_hashes.json', inventory)
        provenance = load_json(staging/'source_provenance.json')
        provenance['desktop_derivation'] = {
            'source_release': str(source), 'source_release_sha256': digest(source/'release_manifest.json'),
            'dressing': deepcopy(scene['mechanical_desktop_dressing']),
            'changed_scene_fields': ['task_layers', 'mechanical_desktop_dressing'],
            'unchanged': ['task', 'rules', 'reset', 'evaluator', 'physics_profile', 'robot', 'parent_environment'],
            'scope': 'Fixed background supplies, not grasp targets or scoring support; new native checks required.',
        }
        dump(staging/'source_provenance.json', provenance)
        manifest = deepcopy(original)
        manifest['release_id'] = original['task_id']+'_'+output.name
        manifest['prepared_at_utc'] = datetime.now(timezone.utc).isoformat()
        manifest['desktop_derivation'] = deepcopy(provenance['desktop_derivation'])
        manifest['files'] = {relative: {**row, 'sha256': digest(staging/relative)}
                             for relative, row in original['files'].items()}
        manifest['files_sha256'] = {relative: row['sha256'] for relative, row in manifest['files'].items()}
        manifest['asset_files_sha256'] = {path: row['sha256'] for path, row in inventory.items()}
        for entry in manifest['tasks']:
            entry['scene_sha256'] = digest(staging/entry['scene'])
        dump(staging/'release_manifest.json', manifest)
        allowed = {'scene.json', 'source_provenance.json', 'asset_file_hashes.json'}
        if any(row['sha256'] != manifest['files'][relative]['sha256']
               for relative, row in original['files'].items() if relative not in allowed):
            raise ValueError('Desktop derivation changed a task, physics, reset or evaluator dependency')
        validate(staging)
        if output.exists():
            raise FileExistsError(output)
        staging.rename(output)
        return manifest
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
