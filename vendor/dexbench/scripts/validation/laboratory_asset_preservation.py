"""Compare task-relevant USD physics before and after a laboratory detail revision.

This static comparison complements fresh native checks; it does not establish
physical feasibility. Only independently declared fixed desktop dressing may
add colliders. Task mass, inertia, contact geometry, joints and reset stay exact.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from bench.isaacsim.release import digest, dump
from scenes.workrooms.laboratory.composition import load_asset_manifest


def physics_inventory(path):
    from pxr import Usd, UsdGeom, UsdPhysics, UsdShade
    stage = Usd.Stage.Open(str(path))
    cache = UsdGeom.XformCache()
    result = {}

    def value(attribute):
        return {'default': str(attribute.Get()), 'time_samples': {
            format(time, '.17g'): str(attribute.Get(time)) for time in attribute.GetTimeSamples()}}

    for prim in stage.Traverse():
        fields = {attribute.GetName(): value(attribute) for attribute in prim.GetAttributes()
            if attribute.HasAuthoredValueOpinion() and attribute.GetName().startswith(('physics:', 'physx'))}
        geometric = prim.HasAPI(UsdPhysics.CollisionAPI)
        rigid = prim.HasAPI(UsdPhysics.RigidBodyAPI)
        joint = prim.IsA(UsdPhysics.Joint)
        authored = prim.GetMetadata('apiSchemas')
        schemas = sorted(name for name in (authored.GetAppliedItems() if authored else [])
                         if name.startswith(('Physics', 'Physx')))
        relationships = {rel.GetName(): [str(target) for target in rel.GetTargets()]
            for rel in prim.GetRelationships()
            if rel.GetName().startswith('physics:') or rel.GetName().startswith('material:binding:physics')}
        if not (fields or geometric or rigid or joint or schemas or relationships):
            continue
        fields.update(relationships)
        if geometric:
            for name in ('points', 'faceVertexCounts', 'faceVertexIndices', 'size',
                         'radius', 'height', 'axis', 'subdivisionScheme'):
                attribute = prim.GetAttribute(name)
                if attribute and attribute.HasAuthoredValueOpinion():
                    fields[name] = value(attribute)
            material, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial(materialPurpose='physics')
            fields['resolved_physics_material'] = str(material.GetPath()) if material else None
        if geometric or rigid or joint:
            fields['world_transform'] = str(cache.GetLocalToWorldTransform(prim))
            samples = {}
            ancestor = prim
            while ancestor and not ancestor.IsPseudoRoot():
                for attribute in ancestor.GetAttributes():
                    if attribute.GetName().startswith('xformOp:') and attribute.GetTimeSamples():
                        samples[str(attribute.GetPath())] = value(attribute)
                ancestor = ancestor.GetParent()
            fields['ancestor_transform_time_samples'] = samples
        result[str(prim.GetPath())] = {'type': prim.GetTypeName(), 'fields': fields,
            'physics_schemas': schemas}
    return result


def fixed_dressing_issues(path, dressing):
    """Independently reject dynamic or undeclared additions inside the allowlist."""
    from pxr import Usd, UsdPhysics
    if not dressing:
        return []
    root = dressing.get('root_path', '')
    if not root.startswith('/World/LaboratoryTask/') or root.endswith('/'):
        return ['Dressing root is not a dedicated task subtree']
    stage = Usd.Stage.Open(str(path))
    prim = stage.GetPrimAtPath(root)
    if not prim:
        return ['Declared dressing root does not exist']
    issues, colliders = [], set()
    for child in Usd.PrimRange(prim):
        if (child.HasAPI(UsdPhysics.RigidBodyAPI) or child.HasAPI(UsdPhysics.MassAPI)
                or child.HasAPI(UsdPhysics.ArticulationRootAPI)
                or child.IsA(UsdPhysics.Joint) or child.IsA(UsdPhysics.Scene)):
            issues.append('Non-static background physics: '+str(child.GetPath()))
        authored = child.GetMetadata('apiSchemas')
        schemas = authored.GetAppliedItems() if authored else []
        permitted = {'PhysicsCollisionAPI', 'PhysicsMeshCollisionAPI', 'PhysxCollisionAPI', 'PhysicsMaterialAPI'}
        if (any(name.startswith(('Physics', 'Physx')) and name not in permitted for name in schemas)
                or child.GetTypeName().startswith(('Physics', 'Physx'))):
            issues.append('Unsupported background physics schema: '+str(child.GetPath()))
        if any(attribute.GetTimeSamples() for attribute in child.GetAttributes()):
            issues.append('Animated fixed background: '+str(child.GetPath()))
        if child.HasAPI(UsdPhysics.CollisionAPI):
            colliders.add(str(child.GetPath()))
    declared = [path for item in dressing.get('items', []) for path in item.get('collider_paths', [])]
    if len(declared) != len(set(declared)) or set(declared) != colliders:
        issues.append('Dressing collider inventory does not match the authored stage')
    if dressing.get('collider_count') != len(colliders):
        issues.append('Dressing collider count differs')
    if dressing.get('dynamic_actor_count') != 0:
        issues.append('Dressing must declare zero dynamic actors')
    for item in dressing.get('items', []):
        if any(item.get(key) is not False for key in ('dynamic', 'task_object', 'scoring_support')):
            issues.append('Background item is not explicitly excluded from task objects/support: '+str(item.get('id')))
    return issues


def review(task_id, baseline, candidate):
    before, old_meta = load_asset_manifest(task_id, baseline)
    after, new_meta = load_asset_manifest(task_id, candidate)
    previous = physics_inventory(before['usd_path'])
    current = physics_inventory(after['usd_path'])
    differences = [path for path, row in previous.items() if current.get(path) != row]
    added = sorted(set(current)-set(previous))
    dressing = new_meta.get('desktop_dressing', {})
    root = dressing.get('root_path')
    unexpected = [path for path in added if not root or not path.startswith(root+'/')]
    fields = ('task_id', 'actors', 'fixtures', 'objects', 'fixture_metadata',
              'joints', 'particle_sets', 'task_parameters', 'physics_prior')
    metadata_changes = [name for name in fields if old_meta.get(name) != new_meta.get(name)]
    dressing_issues = fixed_dressing_issues(after['usd_path'], dressing)
    return {'schema': 'laboratory_task_physics_preservation.v1', 'task_id': task_id,
        'passed': not differences and not unexpected and not metadata_changes and not dressing_issues,
        'baseline_manifest': str(Path(baseline).resolve()),
        'baseline_manifest_sha256': digest(baseline),
        'candidate_manifest': str(Path(candidate).resolve()),
        'candidate_manifest_sha256': digest(candidate),
        'original_physical_prim_count': len(previous),
        'changed_or_removed_physical_paths': differences,
        'added_dressing_physical_paths': [path for path in added if path not in unexpected],
        'unexpected_added_physical_paths': unexpected,
        'changed_task_metadata_fields': metadata_changes,
        'invalid_dressing_issues': dressing_issues,
        'scope': 'Exact authored physics/reset preservation plus declared new fixed dressing; not native contact acceptance.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', required=True)
    parser.add_argument('--baseline', required=True, type=Path)
    parser.add_argument('--candidate', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Preserve previous review evidence')
    result = review(args.task, args.baseline, args.candidate)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    dump(args.output, result)
    print(json.dumps({'task_id': args.task, 'passed': result['passed'], 'output': str(args.output)}))
    return int(not result['passed'])


if __name__ == '__main__':
    raise SystemExit(main())
