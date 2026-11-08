"""Reference the existing screwdriver without depending on past test results."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

from bench.paths import DATA_ROOT, load_json

SOURCE_ROOT = Path(f'{DATA_ROOT}/assets/task_parts/threaded_sharpa/sdf1024_convex_partition_candidate02_20261018')
INPUT_ROOT = Path(f'{DATA_ROOT}/assets/task_parts/mechanical_assembly/gear_train_assembly/screwdriver_input_v1')
ASSET_ID = 'mechanical_screwdriver_sharpa_v3'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare_catalog(output=INPUT_ROOT):
    """Pin actual USD dependencies and licenses; never edit shared source USD."""
    from pxr import Sdf, UsdUtils
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError('Choose a fresh reference-only asset directory')
    parent = SOURCE_ROOT/'manifest.json'
    original = load_json(parent)['assets'][ASSET_ID]
    asset = deepcopy(original)
    layers, paths, unresolved = UsdUtils.ComputeAllDependencies(Sdf.AssetPath(asset['usd_path']))
    if unresolved:
        raise ValueError('Screwdriver USD has unresolved dependencies')
    dependencies = {str(Path(layer.realPath).resolve()) for layer in layers}
    dependencies.update(str(Path(path).resolve()) for path in paths)
    # Source licenses stay in their original locations and retain their bytes.
    dependencies.update(path for path in original['source_dependencies_sha256']
                        if 'LICENSE' in Path(path).name.upper() or Path(path).name.upper() == 'NOTICE')
    hashes = {}
    for path in sorted(dependencies):
        value = digest(path)
        if original['source_dependencies_sha256'].get(path) != value:
            raise ValueError('Physical/source license dependency differs from the old immutable input')
        hashes[path] = value
    for part in asset['collision_parts']:
        if 'mesh_path' in part:
            part['mesh_path'] = str((SOURCE_ROOT/part['mesh_path']).resolve())
            if digest(part['mesh_path']) != part['sha256']:
                raise ValueError('Screwdriver collision mesh changed')
            hashes[part['mesh_path']] = part['sha256']
    asset['source_dependencies_sha256'] = hashes
    asset.pop('qualification', None)
    asset['recommended_spec_options'].pop('usd_physics_fallbacks_provenance', None)
    asset['physics_assumptions']['friction'] = 'Preserve the same authored 0.6 friction and unchanged explicit numerical fallbacks; not a measured material calibration.'
    source = {'schema': 'mechanical_gear_screwdriver_inputs.v1',
        'source_kind': 'reference_only', 'source_assets_modified': False,
        'parent_catalog': {'path': str(parent), 'sha256': digest(parent)},
        'source_asset_id': ASSET_ID, 'usd_path': asset['usd_path'],
        'source_versions': ['threaded_sharpa/v1', 'threaded_sharpa/v2', 'threaded_sharpa/v3',
                            'sdf1024_convex_partition_candidate02_20261018'],
        'source_dependencies_sha256': hashes,
        'generator': __name__+'.prepare_catalog',
        'generator_sha256': digest(__file__),
        'scope': 'Actual unchanged USD dependency closure and original licenses. Historical validation files and success flags are not startup dependencies.'}
    output.mkdir(parents=True)
    (output/'SOURCE.json').write_text(json.dumps(source, indent=2, allow_nan=False)+'\n')
    catalog = {'schema': 'collaborator_native_usd_catalog.v1', 'assets': {ASSET_ID: asset},
        'source_provenance': str(output/'SOURCE.json'), 'source_provenance_sha256': digest(output/'SOURCE.json'),
        'source_assets_modified': False, 'generator': __name__+'.prepare_catalog'}
    (output/'manifest.json').write_text(json.dumps(catalog, indent=2, allow_nan=False)+'\n')
    (output/'LICENSE.txt').write_text('Reference-only catalog. No upstream USD, mesh, texture or license has been copied or relicensed.\nOriginal LICENSE/NOTICE files are retained and hash-pinned in SOURCE.json.\n')
    return catalog
