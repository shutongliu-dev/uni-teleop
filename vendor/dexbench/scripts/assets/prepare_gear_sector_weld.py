"""Prepare one input-only contrast: subgrid2 with explicit weldTolerance=0."""
from copy import deepcopy
from pathlib import Path
import argparse
import json
import shutil

from bench.isaacsim.release import digest, dump
from bench.paths import DATA_ROOT, load_json
from bench.simulation.isaacsim_backend.asset_catalog import load_task_asset
from scripts.assets.prepare_gear_sector_sdf import ASSET_SUFFIX, SECTOR_COUNT
from scripts.assets.prepare_gear_sector_subgrid import SUBGRID2_PARAMETERS

PARENT_ROOT = Path(f'{DATA_ROOT}/assets/task_parts/mechanical_assembly/gear_train_assembly/sdf1250_eight_sector_subgrid2_input_candidate_20261018')
PARENT_SHA256 = 'c8a35e8940923b2fe3479e05f4540477f7af845899fb518373d248c398b3f455'
WELD_ATTRIBUTE = 'physxTriangleMeshCollision:weldTolerance'
WELD_API = 'PhysxTriangleMeshCollisionAPI'
SDF_API = 'PhysxSDFMeshCollisionAPI'
WELD0_COOKING_PROFILE = {
    'schema': 'gear_sector_cooking_profile.v1',
    'profile_id': 'input_gear_sdf1250_subgrid2_weld0_v1',
    'sdf_parameters': deepcopy(SUBGRID2_PARAMETERS),
    'weld_api': WELD_API,
    'weld_attribute': WELD_ATTRIBUTE,
    'weld_tolerance': 0.,
    'weld_value_authored': True,
    'shape_paths': [f'/Gear/collision_{i:02d}' for i in range(SECTOR_COUNT)],
    'parent_manifest_sha256': PARENT_SHA256,
}


def weld0_layer(source):
    """Add only the eight schema declarations and their zero float opinions."""
    old_schema = b'"PhysxSDFMeshCollisionAPI", '
    new_schema = b'"PhysxSDFMeshCollisionAPI", "PhysxTriangleMeshCollisionAPI", '
    old_attribute = b'        int physxSDFMeshCollision:sdfSubgridResolution = 2\n'
    new_attribute = old_attribute+b'        float physxTriangleMeshCollision:weldTolerance = 0\n'
    if (b'weldTolerance' in source or WELD_API.encode() in source
            or source.count(old_schema) != SECTOR_COUNT
            or source.count(old_attribute) != SECTOR_COUNT):
        raise ValueError('Exactly eight unmodified subgrid2 SDFs without welding opinions required')
    result = source.replace(old_schema, new_schema).replace(old_attribute, new_attribute)
    if result.replace(new_schema, old_schema).replace(new_attribute, old_attribute) != source:
        raise ValueError('Unexpected weld0 layer change')
    return result


def audit_weld0_stages(source, target):
    """Compare every composed value; allow only weld0 plus its API marker."""
    from pxr import Sdf, Usd
    first, second = Usd.Stage.Open(str(source)), Usd.Stage.Open(str(target))
    a_prims, b_prims = list(first.TraverseAll()), list(second.TraverseAll())
    if [p.GetPath() for p in a_prims] != [p.GetPath() for p in b_prims]:
        raise ValueError('Weld contrast changed composed prim inventory')
    expected = {f'/Gear/collision_{i:02d}' for i in range(SECTOR_COUNT)}
    records = []
    for a, b in zip(a_prims, b_prims, strict=True):
        path = str(a.GetPath())
        a_meta, b_meta = a.GetAllMetadata(), b.GetAllMetadata()
        if path in expected:
            before, after = a_meta.pop('apiSchemas'), b_meta.pop('apiSchemas')
            before_items = list(before.explicitItems)
            after_items = before_items.copy()
            after_items.insert(before_items.index(SDF_API)+1, WELD_API)
            if (not before.isExplicit or not after.isExplicit
                    or list(after.explicitItems) != after_items):
                raise ValueError('Only the weld API marker may change')
            attr = b.GetAttribute(WELD_ATTRIBUTE)
            if (a.GetAttribute(WELD_ATTRIBUTE).HasAuthoredValueOpinion()
                    or not attr.HasAuthoredValueOpinion() or attr.Get() != 0.
                    or attr.GetTypeName() != Sdf.ValueTypeNames.Float
                    or attr.GetTimeSamples() or attr.IsCustom()):
                raise ValueError('Every contrast shape requires an authored constant float weldTolerance=0')
            records.append({'prim': path, 'api_added': WELD_API,
                'attribute': WELD_ATTRIBUTE, 'before_authored': False, 'after': 0.})
        if a_meta != b_meta:
            raise ValueError('Weld contrast changed unrelated prim metadata')
        aa = {attr.GetName(): attr for attr in a.GetAttributes()}
        bb = {attr.GetName(): attr for attr in b.GetAttributes()}
        if path in expected:
            aa.pop(WELD_ATTRIBUTE, None)
            bb.pop(WELD_ATTRIBUTE, None)
        if aa.keys() != bb.keys():
            raise ValueError('Weld contrast changed unrelated attribute inventory')
        for key in aa:
            x, y = aa[key], bb[key]
            if (x.GetAllMetadata() != y.GetAllMetadata() or x.GetTimeSamples() != y.GetTimeSamples()
                    or any(x.Get(time) != y.Get(time) for time in [Usd.TimeCode.Default(), *x.GetTimeSamples()])):
                raise ValueError('Weld contrast changed unrelated composed attribute')
        if ({r.GetName(): (r.GetTargets(), r.GetAllMetadata()) for r in a.GetRelationships()}
                != {r.GetName(): (r.GetTargets(), r.GetAllMetadata()) for r in b.GetRelationships()}):
            raise ValueError('Weld contrast changed relationship metadata or targets')
    if len(records) != SECTOR_COUNT:
        raise ValueError('All eight original colliders must be present')
    if Path(target).read_bytes() != weld0_layer(Path(source).read_bytes()):
        raise ValueError('Layer bytes differ beyond the declared weld opinions and API markers')
    return {'passed': True, 'prim_count': len(a_prims), 'changes': records,
        'source_usd': str(source), 'source_usd_sha256': digest(source),
        'candidate_usd': str(target), 'candidate_usd_sha256': digest(target),
        'all_existing_layer_bytes_preserved_except_declared_insertions': True,
        'all_other_composed_attributes_metadata_relationships_equal': True,
        'scope': 'CPU authoring only. No native cooked topology or effective welding readback.'}


def prepare(output, parent=PARENT_ROOT):
    output, parent = Path(output).resolve(), Path(parent).resolve()
    if digest(parent/'manifest.json') != PARENT_SHA256:
        raise ValueError('The frozen subgrid2 input-only parent is required')
    manifest = load_json(parent/'manifest.json')
    key = 'input_gear'+ASSET_SUFFIX
    original = load_task_asset(key, parent, verify_hashes=True)
    source = Path(original['usd_path'])
    candidate_bytes = weld0_layer(source.read_bytes())
    output.mkdir(parents=True, exist_ok=False)
    (output/'surface').mkdir()
    for old, new in [('LICENSE.txt', 'LICENSE.txt'), ('SOURCE.json', 'PARENT_SOURCE.json'),
                     ('ORIGINAL_SOURCE.json', 'ORIGINAL_SOURCE.json'),
                     ('surface/cpu_geometry.json', 'surface/cpu_geometry.json'),
                     ('cpu_sector_geometry.json', 'cpu_sector_geometry.json')]:
        shutil.copyfile(parent/old, output/new)
    target = output/'input_gear.usda'
    target.write_bytes(candidate_bytes)
    audit = audit_weld0_stages(source, target)
    dump(output/'cpu_weld_audit.json', audit)
    dump(output/'SOURCE.json', {'schema': 'gear_input_sector_weld0_candidate.v1',
        'parent_root': str(parent), 'parent_manifest_sha256': digest(parent/'manifest.json'),
        'generator_sha256': digest(__file__), 'sdf_parameters': SUBGRID2_PARAMETERS,
        'cooking_profile': deepcopy(WELD0_COOKING_PROFILE),
        'changed_object_ids': ['input_gear'], 'only_physical_change': {
            'attribute': WELD_ATTRIBUTE, 'before_authored': False, 'after': 0.,
            'required_schema_added': WELD_API, 'shape_count': SECTOR_COUNT},
        'hypothesis': 'Disable automatic vertex welding as one cooking-path contrast; '
                      'no claim that welding causes the observed SDF error.',
        'native_effective_welding_verified': False, 'native_geometry_validated': False,
        'physics_accepted': False, 'task_success_claimed': False})
    asset = deepcopy(manifest['assets'][key])
    asset.update(usd_path=str(target), label='input_gear_eight_sector_sdf1250_subgrid2_weld0_candidate',
        qualification='Unvalidated input-only weld0 cooking contrast; no native or dynamics/task qualification')
    asset['physics_assumptions']['scope'] = ('Only explicit weldTolerance=0 and its schema marker '
        'are added to eight sealed subgrid2 colliders. Source geometry/frames/mass/inertia/material/'
        'offsets and all SDF parameters remain identical. Other native cleaning is not disabled.')
    for path in (source, parent/'manifest.json', *[p for p in output.rglob('*') if p.is_file()]):
        asset['source_dependencies_sha256'][str(path)] = digest(path)
    dump(output/'manifest.json', {**manifest, 'assets': {key: asset},
        'qualification': 'Independent unvalidated input-gear-only weld0 cooking contrast'})
    load_task_asset(key, output, verify_hashes=True)
    return {'asset_root': str(output), 'manifest_sha256': digest(output/'manifest.json'),
        'cpu_audit_sha256': digest(output/'cpu_weld_audit.json'), 'native_geometry_validated': False}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.output)))
