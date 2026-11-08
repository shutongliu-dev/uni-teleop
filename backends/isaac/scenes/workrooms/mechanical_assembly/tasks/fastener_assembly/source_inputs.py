"""Export current physical inputs independently of historical test outputs.

The source USD, collision meshes, material parameters and licences are retained.
Old catalogs and releases remain unchanged and loadable with their old inputs.
"""
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import tempfile

from bench.isaacsim.release import checked, digest, dump, validate
from bench.paths import DATA_ROOT, load_json, OUTPUT_ROOT

SOURCE_ROOT = Path(f'{DATA_ROOT}/assets/task_parts/threaded_sharpa/sdf1024_convex_partition_candidate02_20261018')
INPUT_ROOT = Path(f'{DATA_ROOT}/assets/task_parts/mechanical_assembly/fastener_assembly/runtime_inputs_v1')
RELEASE_ROOT = Path(f'{DATA_ROOT}/datasets/fr3_sharpa_bench/isaacsim/releases/mechanical_assembly/fastener_assembly/runtime_inputs_v1')
ASSET_IDS = ('mechanical_screwdriver_sharpa_v3', 'mechanical_bolt_sharpa_v3',
             'mechanical_threaded_fixture_sharpa_v3')


def prepare_catalog(output=INPUT_ROOT, *, source_root=SOURCE_ROOT):
    """Resolve real USD dependencies, collision meshes and original licences."""
    from pxr import Sdf, UsdUtils
    output, source_root = Path(output).resolve(), Path(source_root).resolve()
    if output.exists() or not output.is_relative_to(f'{DATA_ROOT}/assets'):
        raise ValueError('Choose a new shared physical asset catalog')
    parent = source_root/'manifest.json'
    original = load_json(parent)
    assets, all_hashes = {}, {}
    for asset_id in ASSET_IDS:
        old = original['assets'][asset_id]
        asset = deepcopy(old)
        layers, paths, unresolved = UsdUtils.ComputeAllDependencies(Sdf.AssetPath(asset['usd_path']))
        if unresolved:
            raise ValueError('Unresolved physical asset dependencies: '+asset_id)
        dependencies = {str(Path(layer.realPath).resolve()) for layer in layers}
        dependencies.update(str(Path(path).resolve()) for path in paths)
        dependencies.update(path for path in old['source_dependencies_sha256']
                            if 'LICENSE' in Path(path).name.upper() or Path(path).name.upper() == 'NOTICE')
        hashes = {}
        for path in sorted(dependencies):
            value = digest(path)
            if old['source_dependencies_sha256'].get(path) != value:
                raise ValueError('Original physical/licence dependency changed: '+path)
            hashes[path] = value
        for part in asset['collision_parts'] + asset['visual_parts']:
            if 'mesh_path' in part:
                part['mesh_path'] = str((source_root/part['mesh_path']).resolve())
                checked(part['mesh_path'], part['sha256'])
                hashes[part['mesh_path']] = part['sha256']
        if any('/development/' in path for path in hashes):
            raise ValueError('A physical asset still depends on an experiment directory')
        asset['source_dependencies_sha256'] = hashes
        asset.pop('qualification', None)
        asset['recommended_spec_options'].pop('usd_physics_fallbacks_provenance', None)
        assets[asset_id] = asset
        all_hashes.update(hashes)
    output.mkdir(parents=True)
    source = {'schema': 'mechanical_fastener_physical_inputs.v1',
        'parent_catalog': {'path': str(parent), 'sha256': digest(parent)},
        'source_versions': ['threaded_sharpa/v3', SOURCE_ROOT.name],
        'source_dependencies_sha256': all_hashes,
        'source_assets_modified': False, 'generator': __name__+'.prepare_catalog',
        'generator_sha256': digest(__file__),
        'scope': 'Physical dependency closure and original licences. Parent provenance is archival; no historical test result is required at runtime.'}
    dump(output/'SOURCE.json', source)
    manifest = {'schema': original['schema'], 'assets': assets,
        'source_provenance': str(output/'SOURCE.json'),
        'source_provenance_sha256': digest(output/'SOURCE.json'),
        'source_assets_modified': False, 'generator': __name__+'.prepare_catalog'}
    dump(output/'manifest.json', manifest)
    (output/'LICENSE.txt').write_text('Reference-only catalog; original USD, meshes and licences are unchanged.\n'
        'Original LICENSE/NOTICE files and their hashes are listed in SOURCE.json.\n')
    return manifest


def prepare_release(source, output=RELEASE_ROOT, *, asset_root=INPUT_ROOT):
    """Rebind exact frozen physical inputs without copying test reports."""
    from .release import _asset_inventory
    source, output, asset_root = map(lambda p: Path(p).resolve(), (source, output, asset_root))
    original = validate(source)
    if (original.get('schema') != 'mechanical_fastener_task_release.v1'
            or original.get('reset_variant') != 'loose_bolt'):
        raise ValueError('Current loose-bolt base release required')
    if output.exists() or not output.is_relative_to(OUTPUT_ROOT):
        raise ValueError('Choose a new immutable release directory')
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix='.'+output.name+'.preparing-', dir=output.parent))
    try:
        for name, row in original['files'].items():
            checked(source/name, row['sha256'])
            dest = staging/name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source/name, dest)
        changed = {'scene.json', 'source_provenance.json', 'asset_file_hashes.json'}
        for name in changed:
            (staging/name).chmod(0o644)
        scene = load_json(staging/'scene.json')
        for spec in scene['objects'] + scene['fixtures']:
            if spec.get('asset_id') in ASSET_IDS:
                spec['asset_root'] = str(asset_root)
                spec.pop('usd_physics_fallbacks_provenance', None)
        scene['mechanical_input_inventory_version'] = 2
        inventory = _asset_inventory(scene)
        if any('/development/' in path for path in inventory):
            raise ValueError('Current fastener inputs still require an experiment file')
        dump(staging/'scene.json', scene)
        dump(staging/'asset_file_hashes.json', inventory)
        provenance = load_json(staging/'source_provenance.json')
        derivation = {'source_release': str(source), 'source_release_sha256': digest(source/'release_manifest.json'),
            'physical_asset_catalog': {'path': str(asset_root/'manifest.json'), 'sha256': digest(asset_root/'manifest.json')},
            'scope': 'Only dependency/provenance rebinding. Explicit object poses, robot reset, physical parameters and frozen rules/evaluator are preserved.',
            'historical_releases_modified': False}
        provenance['physical_input_derivation'] = derivation
        dump(staging/'source_provenance.json', provenance)
        manifest = deepcopy(original)
        manifest.update(release_id='mechanical_fastener_'+output.name,
            prepared_at_utc=datetime.now(timezone.utc).isoformat(),
            physical_input_derivation=derivation,
            asset_catalog=derivation['physical_asset_catalog'],
            qualification_scope='Independent physical inputs; no inherited test outcomes or new manipulation claim.')
        manifest['files'] = {name: {**row, 'sha256': digest(staging/name)} for name, row in original['files'].items()}
        manifest['files_sha256'] = {name: row['sha256'] for name, row in manifest['files'].items()}
        manifest['asset_files_sha256'] = {name: row['sha256'] for name, row in inventory.items()}
        manifest['tasks'][0]['scene_sha256'] = digest(staging/'scene.json')
        assert all(manifest['files'][name]['sha256'] == row['sha256']
                   for name, row in original['files'].items() if name not in changed)
        dump(staging/'release_manifest.json', manifest)
        validate(staging)
        staging.rename(output)
        return manifest
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
