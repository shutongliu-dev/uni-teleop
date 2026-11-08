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
    if result.get('source_access_profile') == 'source_access_v1' and not result['presentation']:
        import math
        name = {'produce_transfer': 'tomato_0', 'pot_lid_placement': 'pot'}[result['task_id']]
        receipt = result.pop('source_access_changes')
        if set(receipt) != {name}:
            raise ValueError('Unexpected source-pose change inventory')
        before, after = receipt[name]['before_pose_xyzw'], receipt[name]['after_pose_xyzw']
        expected = list(before)
        if name == 'tomato_0':
            expected[0] = .23
        else:
            expected[3:] = [0., 0., math.sin(-math.pi/8), math.cos(-math.pi/8)]
        actor = next(row for row in result['objects'] if row['id'] == name)
        if after != expected or after != actor['initial_pose_xyzw']:
            raise ValueError('Source-pose receipt differs from the explicit profile')
        suffix = (' source_access_v1: tomato_0 source X=0.23 m.' if name == 'tomato_0' else
                  ' source_access_v1: source pot yaw=-45 degrees; right handle faces the robot.')
        if not result['layout_revision'].endswith(suffix):
            raise ValueError('Source-pose description differs from the explicit profile')
        result['layout_revision'] = result['layout_revision'][:-len(suffix)]
    result.pop('presentation')
    for actor in result['objects']:
        actor.pop('initial_pose_xyzw')
    for volume in result.get('volume_bodies', {}).values():
        volume.pop('initial_pose_xyzw')
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
    for name, volume in metadata.get('volume_bodies', {}).items():
        prim = stage.GetPrimAtPath(volume['prim_path'])
        schemas = prim.GetMetadata('apiSchemas') if prim else None
        if not prim or not schemas or 'OmniPhysicsDeformableBodyAPI' not in schemas.GetAppliedItems():
            raise ValueError('Declared volume is not native deformable: '+name)
        pose = volume['initial_pose_xyzw']
        parent = next(item for item in metadata['objects'] if item['id'] == volume['parent_rigid_body'])
        # The sponge FEM vertices share the scouring layer's local rest frame.
        # Allow only the same complete rigid reset of the bonded composite.
        if pose != parent['initial_pose_xyzw']:
            raise ValueError('Bonded volume reset differs from its rigid material layer: '+name)
        matrix = Gf.Matrix4d(1.)
        matrix.SetRotate(Gf.Quatd(pose[6], Gf.Vec3d(*pose[3:6])))
        matrix.SetTranslateOnly(Gf.Vec3d(*pose[:3]))
        if not np.allclose(np.asarray(UsdGeom.Xformable(prim).GetLocalTransformation()),
                           np.asarray(matrix), atol=3e-6, rtol=0.):
            raise ValueError('USD volume reset differs from declared metadata: '+name)


def canonical_physics_identity(canonical_path, diagnostic_path, canonical_meta, diagnostic_meta):
    """Require the exact canonical file, metadata and source reset; no exemptions."""
    report = {'schema': 'kitchen_canonical_physics_identity.v1', 'passed': False,
              'allowed_changes': [], 'native_physics_qualified': False}
    try:
        from pxr import Usd
        canonical_path, diagnostic_path = Path(canonical_path).resolve(), Path(diagnostic_path).resolve()
        report['canonical_usd_sha256'] = _digest(canonical_path)
        report['diagnostic_usd_sha256'] = _digest(diagnostic_path)
        if canonical_path != diagnostic_path:
            raise ValueError('Canonical probe must use the original asset path and dependency namespace')
        if canonical_meta['presentation'] is not False or diagnostic_meta != canonical_meta:
            raise ValueError('Canonical probe metadata or source reset differs')
        stage = Usd.Stage.Open(str(canonical_path))
        if not stage:
            raise ValueError('Cannot open canonical task USD')
        _validate_reset_poses(stage, canonical_meta, goal=False)
        graph = _property_graph(stage, set())
        report.update(metadata_sha256=_json_digest(canonical_meta),
            physical_property_graph_sha256=_json_digest(graph),
            compared_prim_count=len(graph)-1, different_prim_paths=[],
            dynamic_actor_count=len(canonical_meta['actors']),
            native_volume_count=len(canonical_meta.get('volume_bodies', {})), passed=True)
    except Exception as exc:
        report['failure'] = {'type': type(exc).__name__, 'message': str(exc)}
    return report


def diagnostic_physics_identity(canonical_path, goal_path, canonical_meta, goal_meta):
    """Return a fail-closed identity receipt; paths are never written or modified."""
    report = {'schema': 'kitchen_diagnostic_physics_identity.v1', 'passed': False,
        'allowed_changes': ['metadata.presentation', 'objects[].initial_pose_xyzw',
                            'volume_bodies[].initial_pose_xyzw matching the rigid material layer',
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
        volume_roots = {row['prim_path'] for row in canonical_meta.get('volume_bodies', {}).values()}
        for stage, meta, goal in zip(stages, (canonical_meta, goal_meta), (False, True)):
            _validate_reset_poses(stage, meta, goal=goal)
        left, right = (_property_graph(stage, actors | volume_roots) for stage in stages)
        different = sorted(key for key in set(left)|set(right) if left.get(key) != right.get(key))
        report['different_prim_paths'] = different
        if different:
            raise ValueError('USD properties differ outside declared reset transforms')
        report['physical_property_graph_sha256'] = _json_digest(left)
        report['compared_prim_count'] = len(left)-1
        report['dynamic_actor_count'] = len(actors)
        report['native_volume_count'] = len(volume_roots)
        report['passed'] = True
    except Exception as exc:
        report['failure'] = {'type': type(exc).__name__, 'message': str(exc)}
    return report
