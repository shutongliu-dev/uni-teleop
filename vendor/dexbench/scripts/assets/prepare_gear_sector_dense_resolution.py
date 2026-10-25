"""Derive one eight-sector dense1024 contrast from the frozen dense1250 asset."""
from copy import deepcopy
from pathlib import Path
import argparse
import json
import shutil

from bench.isaacsim.release import digest, dump
from bench.paths import DATA_ROOT, load_json
from bench.simulation.isaacsim_backend.asset_catalog import load_task_asset
from scripts.assets.prepare_gear_sector_sdf import ASSET_SUFFIX, SECTOR_COUNT
from scripts.assets.prepare_gear_sector_subgrid import DENSE1024_PARAMETERS, validated_parameters

PARENT_ROOT = Path(f'{DATA_ROOT}/assets/task_parts/mechanical_assembly/gear_train_assembly/sdf1250_eight_sector_dense_input_candidate_20261018')
PARENT_SHA256 = 'f603cfea8c883ae984bb327679edd1208f2fcea21db2505591975a51fd2ed30a'
ATTRIBUTE = 'physxSDFMeshCollision:sdfResolution'
SUBGRID_ATTRIBUTE = 'physxSDFMeshCollision:sdfSubgridResolution'
WELD_ATTRIBUTE = 'physxTriangleMeshCollision:weldTolerance'
WELD_API = 'PhysxTriangleMeshCollisionAPI'


def dense1024_layer(source):
    """Replace exactly eight integer literals, retaining all other layer bytes."""
    old = ('int '+ATTRIBUTE+' = 1250\n').encode()
    new = ('int '+ATTRIBUTE+' = 1024\n').encode()
    dense = ('int '+SUBGRID_ATTRIBUTE+' = 0\n').encode()
    if (source.count(ATTRIBUTE.encode()) != SECTOR_COUNT or source.count(old) != SECTOR_COUNT
            or source.count(SUBGRID_ATTRIBUTE.encode()) != SECTOR_COUNT
            or source.count(dense) != SECTOR_COUNT
            or WELD_ATTRIBUTE.encode() in source or WELD_API.encode() in source):
        raise ValueError('Exactly eight original dense1250 opinions without authored welding required')
    result = source.replace(old, new)
    if result.replace(new, old) != source:
        raise ValueError('Unexpected resolution layer edit')
    return result


def audit_dense1024_stages(source, target):
    """Compare all composed metadata, values, time samples and relationships."""
    from pxr import Sdf, Usd
    first, second = Usd.Stage.Open(str(source)), Usd.Stage.Open(str(target))
    left, right = list(first.TraverseAll()), list(second.TraverseAll())
    if [p.GetPath() for p in left] != [p.GetPath() for p in right]:
        raise ValueError('Dense1024 contrast changed composed prim inventory')
    expected = {f'/Gear/collision_{i:02d}' for i in range(SECTOR_COUNT)}
    changes = []
    for a, b in zip(left, right, strict=True):
        path = str(a.GetPath())
        if a.GetAllMetadata() != b.GetAllMetadata():
            raise ValueError('Dense1024 contrast changed prim metadata')
        aa = {v.GetName(): v for v in a.GetAttributes()}
        bb = {v.GetName(): v for v in b.GetAttributes()}
        if aa.keys() != bb.keys():
            raise ValueError('Dense1024 contrast changed attribute inventory')
        if path in expected:
            x, y = aa[ATTRIBUTE], bb[ATTRIBUTE]
            if (x.Get() != 1250 or y.Get() != 1024 or x.GetTimeSamples() or y.GetTimeSamples()
                    or x.GetTypeName() != Sdf.ValueTypeNames.Int or y.GetTypeName() != Sdf.ValueTypeNames.Int
                    or not x.HasAuthoredValueOpinion() or not y.HasAuthoredValueOpinion()
                    or x.IsCustom() or y.IsCustom()
                    or aa[SUBGRID_ATTRIBUTE].Get() != 0 or bb[SUBGRID_ATTRIBUTE].Get() != 0
                    or a.GetAttribute(WELD_ATTRIBUTE).HasAuthoredValueOpinion()
                    or b.GetAttribute(WELD_ATTRIBUTE).HasAuthoredValueOpinion()
                    or WELD_API in a.GetAppliedSchemas() or WELD_API in b.GetAppliedSchemas()):
                raise ValueError('Exactly eight constant int1250-to-1024 dense opinions without authored welding required')
            changes.append({'prim': path, 'attribute': ATTRIBUTE, 'before': 1250, 'after': 1024})
        for key in aa:
            x, y = aa[key], bb[key]
            x_meta, y_meta = x.GetAllMetadata(), y.GetAllMetadata()
            if key == ATTRIBUTE and path in expected:
                # Registered PhysxSchema exposes the authored default in metadata;
                # an unregistered USD-only reader may omit it. Permit precisely
                # the same declared value change, with all other metadata equal.
                if 'default' in x_meta or 'default' in y_meta:
                    if x_meta.pop('default', None) != 1250 or y_meta.pop('default', None) != 1024:
                        raise ValueError('Unexpected resolution default metadata')
            if x_meta != y_meta or x.GetTimeSamples() != y.GetTimeSamples():
                raise ValueError('Dense1024 contrast changed attribute metadata or time samples')
            if key == ATTRIBUTE and path in expected:
                continue
            if any(x.Get(t) != y.Get(t) for t in [Usd.TimeCode.Default(), *x.GetTimeSamples()]):
                raise ValueError('Dense1024 contrast changed unrelated composed attribute')
        if ({r.GetName(): (r.GetTargets(), r.GetAllMetadata()) for r in a.GetRelationships()}
                != {r.GetName(): (r.GetTargets(), r.GetAllMetadata()) for r in b.GetRelationships()}):
            raise ValueError('Dense1024 contrast changed relationship metadata or targets')
    if len(changes) != SECTOR_COUNT:
        raise ValueError('All eight original collision prims are required')
    old, new = Path(source).read_bytes(), Path(target).read_bytes()
    if new != dense1024_layer(old):
        raise ValueError('Layer bytes differ beyond eight resolution literals')
    return {'passed': True, 'prim_count': len(left), 'changes': changes,
        'source_usd': str(source), 'source_usd_sha256': digest(source),
        'candidate_usd': str(target), 'candidate_usd_sha256': digest(target),
        'byte_difference_count': sum(x != y for x, y in zip(old, new, strict=True)),
        'all_existing_layer_bytes_preserved_except_eight_resolution_literals': True,
        'all_other_composed_attributes_metadata_relationships_equal': True,
        'scope': 'CPU authoring only. Original union, physical fields and automatic welding retained; no native qualification.'}


def prepare(output, parent=PARENT_ROOT):
    output, parent = Path(output).resolve(), Path(parent).resolve()
    if digest(parent/'manifest.json') != PARENT_SHA256:
        raise ValueError('The frozen dense1250 input-only parent is required')
    manifest = load_json(parent/'manifest.json')
    key = 'input_gear'+ASSET_SUFFIX
    original = load_task_asset(key, parent, verify_hashes=True)
    source = Path(original['usd_path'])
    candidate_bytes = dense1024_layer(source.read_bytes())
    output.mkdir(parents=True, exist_ok=False)
    (output/'surface').mkdir()
    for old, new in [('LICENSE.txt', 'LICENSE.txt'), ('SOURCE.json', 'PARENT_SOURCE.json'),
                     ('ORIGINAL_SOURCE.json', 'ORIGINAL_SOURCE.json'),
                     ('surface/cpu_geometry.json', 'surface/cpu_geometry.json'),
                     ('cpu_sector_geometry.json', 'cpu_sector_geometry.json')]:
        shutil.copyfile(parent/old, output/new)
    target = output/'input_gear.usda'
    target.write_bytes(candidate_bytes)
    dump(output/'cpu_resolution_audit.json', audit_dense1024_stages(source, target))
    dump(output/'SOURCE.json', {'schema': 'gear_input_sector_dense_resolution_candidate.v1',
        'parent_root': str(parent), 'parent_manifest_sha256': digest(parent/'manifest.json'),
        'generator_sha256': digest(__file__),
        'sdf_parameters': validated_parameters(DENSE1024_PARAMETERS),
        'changed_object_ids': ['input_gear'], 'only_physical_change': {
            'attribute': ATTRIBUTE, 'before': 1250, 'after': 1024, 'shape_count': SECTOR_COUNT},
        'weld_api_authored': False, 'weld_value_authored': False,
        'source_geometry_and_partition_proofs_reused': True,
        'hypothesis': 'One dense field-integrity contrast with reference per-shape data below4GiB. '
            'Changed spacing and phase prevent attributing any outcome solely to a byte-count defect.',
        'native_grid_dimensions_readback': False, 'native_geometry_validated': False,
        'physics_accepted': False, 'task_success_claimed': False})
    asset = deepcopy(manifest['assets'][key])
    asset.update(usd_path=str(target), label='input_gear_eight_sector_dense1024_candidate',
        qualification='Unvalidated input-only dense1024 contrast; no native or dynamics/task qualification')
    asset['physics_assumptions']['scope'] = ('Only eight dense sdfResolution1250-to-1024 opinions '
        'differ from the frozen dense1250 parent. Source geometry/frames/mass/COM/inertia/material/'
        'offsets and other cooking fields, including original automatic welding, remain identical.')
    for path in (source, parent/'manifest.json', *[p for p in output.rglob('*') if p.is_file()]):
        asset['source_dependencies_sha256'][str(path)] = digest(path)
    dump(output/'manifest.json', {**manifest, 'assets': {key: asset},
        'qualification': 'Independent unvalidated input-gear-only dense1024 contrast'})
    load_task_asset(key, output, verify_hashes=True)
    return {'asset_root': str(output), 'manifest_sha256': digest(output/'manifest.json'),
        'cpu_audit_sha256': digest(output/'cpu_resolution_audit.json'), 'native_geometry_validated': False}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.output)))
