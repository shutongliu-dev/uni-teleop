"""Bind a kitchen goal reset to the exact physical canonical task asset.

Only the initial translate/orient values on declared dynamic actor roots may
differ. This is a read-only identity check, not a physics acceptance test.
Signed infinity is compared exactly: registered PhysX schemas legitimately
expose infinite fallback limits. This helper does not certify property validity.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import numbers
from pathlib import Path

import numpy as np


def _digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _encoded(value):
    """Stable value encoding preserving USD types and external asset bytes."""
    from pxr import Sdf
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, numbers.Integral):
        return int(value)
    if isinstance(value, numbers.Real):
        if np.isnan(value):
            raise ValueError('NaN USD property cannot establish identity')
        if np.isinf(value):
            return {'ieee754': '+infinity' if value > 0 else '-infinity'}
        return float(value)
    if isinstance(value, Sdf.AssetPath):
        if value.path and not value.resolvedPath:
            raise ValueError('Unresolved external asset: '+value.path)
        return {'usd_type': 'Sdf.AssetPath', 'authored_path': value.path,
                'sha256': _digest(value.resolvedPath) if value.path else None}
    if isinstance(value, dict):
        return {str(k): _encoded(v) for k, v in sorted(value.items(), key=lambda row: str(row[0]))}
    if isinstance(value, (list, tuple)):
        return [_encoded(v) for v in value]
    if hasattr(value, 'GetReal') and hasattr(value, 'GetImaginary'):
        return {'usd_type': type(value).__name__, 'real': _encoded(value.GetReal()),
                'imaginary': _encoded(value.GetImaginary())}
    if hasattr(value, 'GetAppliedItems') and hasattr(value, 'isExplicit'):
        return {'usd_type': type(value).__name__, 'is_explicit': bool(value.isExplicit),
                **{field: _encoded(getattr(value, field)) for field in (
                    'explicitItems', 'addedItems', 'prependedItems', 'appendedItems',
                    'deletedItems', 'orderedItems')}}
    try:
        values = list(value)
    except TypeError:
        # Gf quaternions and paths have stable value reprs. Unknown objects
        # with identity-based reprs must fail closed, never hash an address.
        representation = repr(value)
        if ' object at 0x' in representation:
            raise ValueError('Unsupported USD property type: '+type(value).__name__)
        return {'usd_type': type(value).__name__, 'value': representation}
    return {'usd_type': type(value).__name__, 'values': [_encoded(v) for v in values]}


def _json_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def _metadata_without_reset(metadata):
    result = deepcopy(metadata)
    result.pop('presentation')
    for actor in result['objects']:
        actor.pop('initial_pose_xyzw')
    return result


def _property_graph(stage, actors):
    from pxr import Usd
    graph = {'/': {'metadata': _encoded(stage.GetMetadata('customLayerData') or {}),
                   'upAxis': _encoded(stage.GetMetadata('upAxis')),
                   'metersPerUnit': _encoded(stage.GetMetadata('metersPerUnit')),
                   'defaultPrim': _encoded(stage.GetMetadata('defaultPrim')),
                   'timeCodesPerSecond': _encoded(stage.GetMetadata('timeCodesPerSecond')),
                   'framesPerSecond': _encoded(stage.GetMetadata('framesPerSecond'))}}
    for prim in stage.TraverseAll():
        path = str(prim.GetPath())
        entry = {'type': prim.GetTypeName(), 'metadata': _encoded(prim.GetAllMetadata()),
                 'attributes': {}, 'relationships': {}}
        for attr in prim.GetAttributes():
            name = attr.GetName()
            reset = path in actors and name in ('xformOp:translate', 'xformOp:orient')
            metadata = attr.GetAllMetadata()
            if reset:
                metadata.pop('default', None)
                if attr.GetTimeSamples():
                    raise ValueError('Actor reset transform cannot be animated: '+str(attr.GetPath()))
            definition = {'metadata': _encoded(metadata), 'type': str(attr.GetTypeName()),
                'custom': attr.IsCustom(), 'variability': str(attr.GetVariability()),
                'connections': _encoded(attr.GetConnections()), 'authored': attr.HasAuthoredValueOpinion()}
            if not reset:
                definition['default'] = _encoded(attr.Get(Usd.TimeCode.Default()))
                definition['samples'] = {str(t): _encoded(attr.Get(t)) for t in attr.GetTimeSamples()}
            else:
                definition['default'] = 'DECLARED_ACTOR_RESET_POSE'
            entry['attributes'][name] = definition
        for relationship in prim.GetRelationships():
            entry['relationships'][relationship.GetName()] = {
                'metadata': _encoded(relationship.GetAllMetadata()),
                'targets': _encoded(relationship.GetTargets()), 'custom': relationship.IsCustom()}
        graph[path] = entry
    return graph


def _validate_reset_poses(stage, metadata, *, goal):
    from pxr import Gf, UsdGeom, UsdPhysics
    for item in metadata['objects']:
        prim = stage.GetPrimAtPath(item['body_path'])
        if not prim or not prim.HasAPI(UsdPhysics.RigidBodyAPI):
            raise ValueError('Declared actor is not a rigid body: '+item['id'])
        pose = item['initial_pose_xyzw']
        if len(pose) != 7 or not np.isfinite(pose).all() or abs(np.linalg.norm(pose[3:])-1) > 1e-6:
            raise ValueError('Invalid declared reset pose: '+item['id'])
        matrix = Gf.Matrix4d(1.)
        matrix.SetRotate(Gf.Quatd(pose[6], Gf.Vec3d(*pose[3:6])))
        matrix.SetTranslateOnly(Gf.Vec3d(*pose[:3]))
        if not np.allclose(np.asarray(UsdGeom.Xformable(prim).GetLocalTransformation()),
                           np.asarray(matrix), atol=3e-6, rtol=0.):
            raise ValueError('USD reset transform differs from declared metadata: '+item['id'])
        if goal and pose != metadata['goals']['body_poses_xyzw'][item['id']]:
            raise ValueError('Presentation reset differs from declared goal pose: '+item['id'])


def diagnostic_physics_identity(canonical_path, goal_path, canonical_meta, goal_meta):
    """Return a fail-closed identity receipt; paths are never written or modified."""
    report = {'schema': 'kitchen_diagnostic_physics_identity.v1', 'passed': False,
        'allowed_changes': ['metadata.presentation', 'objects[].initial_pose_xyzw',
                            'declared actor root xformOp:translate/orient values'],
        'infinity_semantics': 'Signed infinities compare exactly, including schema fallback limits; this is identity, not property-validity acceptance. NaN fails closed.',
        'native_physics_qualified': False}
    try:
        from pxr import Usd
        canonical_path, goal_path = Path(canonical_path).resolve(), Path(goal_path).resolve()
        report['canonical_usd_sha256'] = _digest(canonical_path)
        report['diagnostic_usd_sha256'] = _digest(goal_path)
        if canonical_meta['presentation'] is not False or goal_meta['presentation'] is not True:
            raise ValueError('Expected canonical reset and explicitly labelled presentation reset')
        if _metadata_without_reset(canonical_meta) != _metadata_without_reset(goal_meta):
            raise ValueError('Metadata changed beyond declared actor reset poses and presentation flag')
        report['metadata_without_reset_sha256'] = _json_digest(_metadata_without_reset(canonical_meta))
        stages = [Usd.Stage.Open(str(path)) for path in (canonical_path, goal_path)]
        if not all(stages):
            raise ValueError('Cannot open task USD')
        actors = set(canonical_meta['actors'].values())
        if not actors or len(actors) != len(canonical_meta['objects']):
            raise ValueError('Invalid actor inventory')
        for stage, meta, goal in zip(stages, (canonical_meta, goal_meta), (False, True)):
            _validate_reset_poses(stage, meta, goal=goal)
        left, right = (_property_graph(stage, actors) for stage in stages)
        different = sorted(key for key in set(left)|set(right) if left.get(key) != right.get(key))
        report['different_prim_paths'] = different
        if different:
            raise ValueError('USD properties differ outside declared reset transforms')
        report['physical_property_graph_sha256'] = _json_digest(left)
        report['compared_prim_count'] = len(left)-1
        report['dynamic_actor_count'] = len(actors)
        report['passed'] = True
    except Exception as exc:
        report['failure'] = {'type': type(exc).__name__, 'message': str(exc)}
    return report
