"""Copy the sealed exact-union SDF1024 candidate, changing only subgrid 6 to 0."""
from copy import deepcopy
import argparse
import json
from pathlib import Path
import shutil

from bench.isaacsim.release import digest, dump
from bench.paths import DATA_ROOT, load_json
from bench.simulation.isaacsim_backend.asset_catalog import load_task_asset
from scripts.assets.prepare_gear_sdf import GEARS, SDF_PARAMETERS

PARENT_ROOT = Path(f'{DATA_ROOT}/assets/task_parts/mechanical_assembly/gear_train_assembly/sdf1024_exact_union_candidate02_20261018')
DENSE_PARAMETERS = {**SDF_PARAMETERS, 'sdfSubgridResolution': 0}
ATTRIBUTE = 'physxSDFMeshCollision:sdfSubgridResolution'


def dense_layer(source):
    """Exact byte-level single-variable edit; reject missing/duplicate opinions."""
    old = ('int '+ATTRIBUTE+' = 6').encode()
    new = ('int '+ATTRIBUTE+' = 0').encode()
    if source.count(ATTRIBUTE.encode()) != 1 or source.count(old) != 1:
        raise ValueError('Exactly one original subgrid=6 opinion is required')
    result = source.replace(old, new)
    if result.replace(new, old) != source:
        raise ValueError('Unexpected layer edit')
    return result


def compare_stages(source, target, *, expected_changes=None):
    """Require every composed prim, attribute and relationship except subgrid."""
    from pxr import Usd
    first, second = Usd.Stage.Open(str(source)), Usd.Stage.Open(str(target))
    left, right = list(first.Traverse()), list(second.Traverse())
    if [str(p.GetPath()) for p in left] != [str(p.GetPath()) for p in right]:
        raise ValueError('Candidate changed prim inventory')
    changes = []
    for a, b in zip(left, right, strict=True):
        if a.GetAllMetadata() != b.GetAllMetadata():
            raise ValueError('Candidate changed prim metadata')
        aa, bb = a.GetAttributes(), b.GetAttributes()
        if [v.GetName() for v in aa] != [v.GetName() for v in bb]:
            raise ValueError('Candidate changed attribute inventory')
        for x, y in zip(aa, bb, strict=True):
            if x.GetAllMetadata() != y.GetAllMetadata():
                raise ValueError('Candidate changed attribute metadata')
            if x.Get() != y.Get():
                changes.append({'prim': str(a.GetPath()), 'attribute': x.GetName(),
                    'before': x.Get(), 'after': y.Get()})
        if {r.GetName(): r.GetTargets() for r in a.GetRelationships()} != {r.GetName(): r.GetTargets() for r in b.GetRelationships()}:
            raise ValueError('Candidate changed relationship targets')
    expected = expected_changes if expected_changes is not None else [
        {'prim': '/Gear/collision', 'attribute': ATTRIBUTE, 'before': 6, 'after': 0}]
    if changes != expected:
        raise ValueError('Only the single subgrid opinion may change')
    return {'passed': True, 'prim_count': len(left), 'changes': changes,
        'all_other_composed_attributes_metadata_relationships_equal': True}


def prepare(output, parent=PARENT_ROOT):
    output, parent = Path(output).resolve(), Path(parent).resolve()
    source_manifest = load_json(parent/'manifest.json')
    originals = {name: load_task_asset(name+'_sdf1024', parent, verify_hashes=True) for name in GEARS}
    output.mkdir(parents=True, exist_ok=False)
    (output/'surface').mkdir()
    for original, copied in [('LICENSE.txt', 'LICENSE.txt'), ('SOURCE.json', 'PARENT_SOURCE.json'),
                             ('ORIGINAL_SOURCE.json', 'ORIGINAL_SOURCE.json'),
                             ('surface/cpu_geometry.json', 'surface/cpu_geometry.json')]:
        shutil.copyfile(parent/original, output/copied)
    entries, audits = {}, {}
    for name in GEARS:
        asset = deepcopy(source_manifest['assets'][name+'_sdf1024'])
        source = Path(originals[name]['usd_path'])
        target = output/(name+'.usda')
        target.write_bytes(dense_layer(source.read_bytes()))
        audit = compare_stages(source, target)
        audit.update(source_usd=str(source), source_usd_sha256=digest(source),
            candidate_usd=str(target), candidate_usd_sha256=digest(target),
            byte_difference_is_only_subgrid_value=True)
        audits[name] = audit
        asset['usd_path'] = str(target)
        asset['label'] = name+'_single_dense_sdf1024_candidate'
        asset['qualification'] = 'Unvalidated dense single-SDF geometry candidate; no physics/task qualification'
        asset['physics_assumptions']['scope'] = 'Only subgrid 6 to 0 differs from the sealed single-SDF1024 parent. Native geometry and contact dynamics remain unvalidated.'
        dependencies = asset['source_dependencies_sha256']
        for path in (source, target, parent/'manifest.json', output/'surface/cpu_geometry.json'):
            dependencies[str(path)] = digest(path)
        entries[name+'_sdf1024'] = asset
    dump(output/'cpu_dense_audit.json', {'passed': True, 'assets': audits,
        'scope': 'CPU single-variable authoring only; original union proof and mesh bytes reused.'})
    dump(output/'SOURCE.json', {'schema': 'gear_single_dense_sdf_candidate.v1',
        'parent_root': str(parent), 'parent_manifest_sha256': digest(parent/'manifest.json'),
        'generator_sha256': digest(__file__), 'sdf_parameters': DENSE_PARAMETERS,
        'only_physical_change': {'attribute': ATTRIBUTE, 'before': 6, 'after': 0},
        'native_geometry_validated': False, 'physics_accepted': False, 'task_success_claimed': False})
    for asset in entries.values():
        for name in ('SOURCE.json', 'PARENT_SOURCE.json', 'ORIGINAL_SOURCE.json', 'LICENSE.txt', 'cpu_dense_audit.json'):
            asset['source_dependencies_sha256'][str(output/name)] = digest(output/name)
    dump(output/'manifest.json', {**source_manifest, 'assets': entries,
        'qualification': 'Independent unvalidated dense single-SDF1024 candidate'})
    for name in GEARS:
        load_task_asset(name+'_sdf1024', output, verify_hashes=True)
    return {'asset_root': str(output), 'manifest_sha256': digest(output/'manifest.json'),
        'cpu_audit_sha256': digest(output/'cpu_dense_audit.json'), 'native_geometry_validated': False}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.output)))
