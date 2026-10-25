"""Prepare reviewed input-only eight-sector SDF storage contrasts."""
from copy import deepcopy
import argparse
import json
from pathlib import Path
import shutil

from bench.isaacsim.release import digest, dump
from bench.paths import DATA_ROOT, load_json
from bench.simulation.isaacsim_backend.asset_catalog import load_task_asset
from scripts.assets.prepare_gear_dense_sdf import ATTRIBUTE, compare_stages
from scripts.assets.prepare_gear_sector_sdf import ASSET_SUFFIX, PARAMETERS, SECTOR_COUNT

PARENT_ROOT = Path(f'{DATA_ROOT}/assets/task_parts/mechanical_assembly/gear_train_assembly/sdf1250_eight_sector_candidate_20261018')
PARENT_SHA256 = 'e1ce2d9052574fd7be6977f2a1a679b3c6bdcce6f2e7c863f2967926c7f1e85e'
SUBGRID2_PARAMETERS = {**PARAMETERS, 'sdfSubgridResolution': 2}
DENSE_PARAMETERS = {**PARAMETERS, 'sdfSubgridResolution': 0}
DENSE1024_PARAMETERS = {**DENSE_PARAMETERS, 'sdfResolution': 1024}
SUBGRID2_ROOT = PARENT_ROOT.with_name('sdf1250_eight_sector_subgrid2_input_candidate_20261018')
SUBGRID2_SHA256 = 'c8a35e8940923b2fe3479e05f4540477f7af845899fb518373d248c398b3f455'


def validated_parameters(value):
    """Allow only the original contract and explicitly frozen storage contrasts."""
    if (not isinstance(value, dict) or set(value) != set(PARAMETERS)
            or any(type(value[k]) is not type(PARAMETERS[k]) for k in PARAMETERS)
            or value not in (PARAMETERS, SUBGRID2_PARAMETERS, DENSE_PARAMETERS, DENSE1024_PARAMETERS)):
        raise ValueError('Only complete original subgrid6, declared subgrid2, dense1250 or dense1024 parameters are permitted')
    return deepcopy(value)


def contrast_layer(source, before, after):
    if (before, after) not in ((6, 2), (2, 0)):
        raise ValueError('Only reviewed subgrid6-to-2 and subgrid2-to-0 contrasts are permitted')
    old = ('int '+ATTRIBUTE+' = '+str(before)).encode()
    new = ('int '+ATTRIBUTE+' = '+str(after)).encode()
    if source.count(ATTRIBUTE.encode()) != SECTOR_COUNT or source.count(old) != SECTOR_COUNT:
        raise ValueError(f'Exactly eight original subgrid={before} opinions are required')
    result = source.replace(old, new)
    if result.replace(new, old) != source:
        raise ValueError('Unexpected layer edit')
    return result


def subgrid2_layer(source):
    return contrast_layer(source, 6, 2)


def dense_sector_layer(source):
    return contrast_layer(source, 2, 0)


def prepare(output, parent=PARENT_ROOT):
    return _prepare(output, parent, PARENT_SHA256, 6, 2)


def prepare_dense(output, parent=SUBGRID2_ROOT):
    return _prepare(output, parent, SUBGRID2_SHA256, 2, 0)


def _prepare(output, parent, parent_sha256, before, after):
    parameters = validated_parameters({**PARAMETERS, 'sdfSubgridResolution': after})
    label = 'dense0' if after == 0 else 'subgrid2'
    output, parent = Path(output).resolve(), Path(parent).resolve()
    if digest(parent/'manifest.json') != parent_sha256:
        raise ValueError('The sealed parent eight-sector candidate is required')
    manifest = load_json(parent/'manifest.json')
    key = 'input_gear'+ASSET_SUFFIX
    original = load_task_asset(key, parent, verify_hashes=True)
    asset = deepcopy(manifest['assets'][key])
    output.mkdir(parents=True, exist_ok=False)
    (output/'surface').mkdir()
    for source, target in [('LICENSE.txt', 'LICENSE.txt'), ('SOURCE.json', 'PARENT_SOURCE.json'),
                           ('ORIGINAL_SOURCE.json', 'ORIGINAL_SOURCE.json'),
                           ('surface/cpu_geometry.json', 'surface/cpu_geometry.json'),
                           ('cpu_sector_geometry.json', 'cpu_sector_geometry.json')]:
        shutil.copyfile(parent/source, output/target)
    source = Path(original['usd_path'])
    target = output/'input_gear.usda'
    target.write_bytes(contrast_layer(source.read_bytes(), before, after))
    expected = [{'prim': '/Gear/collision_'+str(i).zfill(2), 'attribute': ATTRIBUTE,
                 'before': before, 'after': after} for i in range(SECTOR_COUNT)]
    audit = compare_stages(source, target, expected_changes=expected)
    audit.update(source_usd=str(source), source_usd_sha256=digest(source),
        candidate_usd=str(target), candidate_usd_sha256=digest(target),
        byte_difference_count=sum(a != b for a, b in zip(source.read_bytes(), target.read_bytes(), strict=True)),
        source_geometry_and_partition_proofs_reused=True)
    if audit['byte_difference_count'] != SECTOR_COUNT:
        raise ValueError('Exactly eight byte changes required')
    dump(output/'cpu_subgrid_audit.json', audit)
    dump(output/'SOURCE.json', {'schema': 'gear_input_sector_subgrid_candidate.v1',
        'parent_root': str(parent), 'parent_manifest_sha256': digest(parent/'manifest.json'),
        'generator_sha256': digest(__file__), 'sdf_parameters': parameters,
        'changed_object_ids': ['input_gear'], 'only_physical_change': {
            'attribute': ATTRIBUTE, 'before': before, 'after': after, 'shape_count': SECTOR_COUNT},
        'native_geometry_validated': False, 'physics_accepted': False, 'task_success_claimed': False})
    asset.update(usd_path=str(target), label='input_gear_eight_sector_sdf1250_'+label+'_candidate',
        qualification='Unvalidated input-only '+label+' candidate; no dynamics/task qualification')
    asset['physics_assumptions']['scope'] = f'Only eight subgrid{before}-to-{after} opinions differ from the sealed parent. All source mesh/visual/mass/COM/inertia/material values remain identical.'
    dependencies = asset['source_dependencies_sha256']
    for path in (source, parent/'manifest.json', *[p for p in output.rglob('*') if p.is_file()]):
        dependencies[str(path)] = digest(path)
    dump(output/'manifest.json', {**manifest, 'assets': {key: asset},
        'qualification': 'Independent unvalidated input-gear-only '+label+' candidate'})
    load_task_asset(key, output, verify_hashes=True)
    return {'asset_root': str(output), 'manifest_sha256': digest(output/'manifest.json'),
        'cpu_audit_sha256': digest(output/'cpu_subgrid_audit.json'), 'native_geometry_validated': False}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--dense', action='store_true', help='Copy sealed subgrid2 input with only eight opinions changed to dense0')
    args = parser.parse_args()
    print(json.dumps((prepare_dense if args.dense else prepare)(args.output)))
