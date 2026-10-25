"""Hash-bound physical task subtrees; no scene-name or task scoring branches.

The bench owns the robot and PhysicsScene. A layer supplies already-authored
objects, joints and optional particle sets, with explicit observation names.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
from pathlib import Path
import re

import numpy as np

from .workroom import BUILTIN_MDL


def _name(value):
    if not isinstance(value, str) or not re.fullmatch(r'[A-Za-z][A-Za-z0-9_]*', value):
        raise ValueError('Task layer IDs must be simple nonempty identifiers')
    return value


def _path(value):
    from pxr import Sdf
    path = Sdf.Path(value) if isinstance(value, str) else Sdf.Path.emptyPath
    if not path.IsAbsolutePath() or not path.IsPrimPath() or path == Sdf.Path.absoluteRootPath:
        raise ValueError('Task layer paths must be absolute prim paths')
    return path


def verify_file(path, expected):
    p = Path(path)
    if not p.is_absolute() or not isinstance(expected, str) or not re.fullmatch('[0-9a-f]{64}', expected):
        raise ValueError('Task layer files require absolute paths and SHA256 identities')
    if hashlib.sha256(p.read_bytes()).hexdigest() != expected:
        raise ValueError('Task layer dependency hash mismatch: ' + str(p))
    return p


def _world_pose(stage, prim):
    from pxr import UsdGeom
    matrix = UsdGeom.XformCache().GetLocalToWorldTransform(prim)
    # Body scaling/shear cannot be hidden in a pose-only reset contract.
    basis = np.asarray(matrix)[:3, :3]
    if not np.allclose(basis @ basis.T, np.eye(3), atol=1.e-7, rtol=0) or np.linalg.det(basis) < 0:
        raise ValueError('Task rigid-body root transform must be rigid: ' + str(prim.GetPath()))
    q = matrix.ExtractRotationQuat()
    return {'position_m': list(matrix.ExtractTranslation()),
            'quaternion_wxyz': [q.GetReal(), *q.GetImaginary()]}


def mount_task_layers(stage, declarations, *, physics_scene_path, occupied_objects=()):
    """Reference only declared subtrees and inventory their unchanged physics."""
    from pxr import Sdf, Usd, UsdGeom, UsdPhysics
    if declarations is None:
        declarations = []
    if not isinstance(declarations, list):
        raise ValueError('task_layers must be a list')
    if declarations and not stage.GetPrimAtPath(physics_scene_path).IsA(UsdPhysics.Scene):
        raise ValueError('Task layers require the existing bench-owned PhysicsScene')
    records, objects, static_paths, used_ids, roots = [], {}, [], set(), []
    for raw in declarations:
        spec = deepcopy(raw)
        allowed = {'id', 'usd_path', 'sha256', 'source_prim_path', 'target_prim_path',
                   'dependencies_sha256', 'rigid_bodies', 'particle_sets'}
        if not isinstance(spec, dict) or set(spec) - allowed:
            raise ValueError('Unknown task layer declaration fields')
        layer_id = _name(spec['id'])
        if layer_id in used_ids:
            raise ValueError('Duplicate task layer ID')
        used_ids.add(layer_id)
        source, target = _path(spec['source_prim_path']), _path(spec['target_prim_path'])
        if target == Sdf.Path('/World') or not target.HasPrefix('/World'):
            raise ValueError('Task layer target must be below /World')
        if any(target.HasPrefix(root) or root.HasPrefix(target) for root in roots):
            raise ValueError('Overlapping task layer roots')
        if stage.GetPrimAtPath(target):
            raise ValueError('Task layer must not replace an existing prim: ' + str(target))
        roots.append(target)
        file = verify_file(spec['usd_path'], spec['sha256'])
        dependencies = spec.get('dependencies_sha256', {})
        if not isinstance(dependencies, dict):
            raise ValueError('dependencies_sha256 must be a path/hash mapping')
        for path, digest in dependencies.items():
            verify_file(path, digest)
        original = Usd.Stage.Open(str(file))
        if (original is None or not original.GetPrimAtPath(source)
                or UsdGeom.GetStageMetersPerUnit(original) != 1
                or UsdGeom.GetStageUpAxis(original) != 'Z'):
            raise ValueError('Task layer must contain a metre/Z-up source subtree')
        for layer in original.GetUsedLayers():
            if not layer.anonymous and layer.realPath != str(file) and layer.realPath not in dependencies:
                raise ValueError('Unpinned task layer composition dependency: ' + layer.realPath)
        for prim in Usd.PrimRange(original.GetPrimAtPath(source)):
            for rel in prim.GetRelationships():
                # USD drops references outside the selected subtree. In
                # particular, a lost joint body target becomes a world anchor.
                # The only deliberate remapping is the bench PhysicsScene.
                if rel.GetName() == 'simulationOwner' and prim.GetTypeName() == 'PhysxParticleSystem':
                    continue
                if any(not target_path.HasPrefix(source) for target_path in rel.GetTargets()):
                    raise ValueError('Task subtree has an external relationship target: ' + str(rel.GetPath()))
            for attr in prim.GetAttributes():
                if any(not target_path.HasPrefix(source) for target_path in attr.GetConnections()):
                    raise ValueError('Task subtree has an external attribute connection: ' + str(attr.GetPath()))
                value = attr.Get()
                assets = [value] if isinstance(value, Sdf.AssetPath) else []
                if attr.GetTypeName() == Sdf.ValueTypeNames.AssetArray and value is not None:
                    assets = list(value)
                for asset in assets:
                    if not asset.path or asset.path in BUILTIN_MDL:
                        continue  # Core MDL modules are resolved by the local Isaac Sim.
                    resolved = asset.resolvedPath or Sdf.ComputeAssetPathRelativeToLayer(original.GetRootLayer(), asset.path)
                    if str(Path(resolved)) not in dependencies:
                        raise ValueError('Unpinned task layer asset dependency: ' + resolved)
        stage.DefinePrim(target, 'Xform').GetReferences().AddReference(str(file), source)
        root = stage.GetPrimAtPath(target)
        source_matrix = UsdGeom.XformCache().GetLocalToWorldTransform(original.GetPrimAtPath(source))
        target_matrix = UsdGeom.XformCache().GetLocalToWorldTransform(root)
        if not np.allclose(np.asarray(source_matrix), np.asarray(target_matrix), atol=1.e-9, rtol=0):
            raise ValueError('Task subtree reference changes its authored world placement')
        declared = spec.get('rigid_bodies', {})
        if not isinstance(declared, dict):
            raise ValueError('Task layer must declare its dynamic rigid-body observations')
        body_paths = {}
        for name, path in declared.items():
            _name(name)
            path = _path(path)
            if name in objects or name in occupied_objects or not path.HasPrefix(target):
                raise ValueError('Duplicate or out-of-layer task rigid body: ' + name)
            if str(path) in body_paths:
                raise ValueError('One native body cannot have multiple observation aliases')
            body_paths[str(path)] = name
        actual_bodies, joints, colliders, systems = {}, [], [], []
        for prim in Usd.PrimRange(root):
            path = str(prim.GetPath())
            if prim.IsA(UsdPhysics.Scene) or prim.HasAPI(UsdPhysics.ArticulationRootAPI):
                raise ValueError('Task layer cannot supply a PhysicsScene or robot articulation')
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                body = UsdPhysics.RigidBodyAPI(prim)
                if body.GetRigidBodyEnabledAttr().Get() is False or body.GetKinematicEnabledAttr().Get():
                    raise ValueError('Task rigid body must remain enabled and dynamic: ' + path)
                actual_bodies[path] = prim
            if prim.HasAPI(UsdPhysics.CollisionAPI) and UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get() is not False:
                colliders.append(path)
            if prim.IsA(UsdPhysics.Joint):
                if not prim.IsA(UsdPhysics.PrismaticJoint):
                    raise ValueError('Task joint readback currently supports prismatic joints only: ' + path)
                joint = UsdPhysics.PrismaticJoint(prim)
                record = {'path': path, 'type': 'prismatic', 'axis': str(joint.GetAxisAttr().Get())}
                for n in (0, 1):
                    targets = getattr(joint, f'GetBody{n}Rel')().GetTargets()
                    if len(targets) > 1 or (targets and str(targets[0]) not in body_paths):
                        raise ValueError('Task joint body must be a declared body or world')
                    record[f'body{n}'] = str(targets[0]) if targets else None
                    record[f'local_pos{n}'] = list(getattr(joint, f'GetLocalPos{n}Attr')().Get())
                    quat = getattr(joint, f'GetLocalRot{n}Attr')().Get()
                    record[f'local_rot{n}_xyzw'] = [*quat.GetImaginary(), quat.GetReal()]
                joints.append(record)
            if prim.GetTypeName() == 'PhysxParticleSystem':
                # Relationship targets outside a referenced subtree are not
                # imported by USD. Bind the existing bench-owned physics scene.
                prim.CreateRelationship('simulationOwner').SetTargets([physics_scene_path])
                systems.append(path)
        if set(actual_bodies) != set(body_paths):
            raise ValueError('Declared task body inventory differs from authored dynamic bodies')
        for path, prim in actual_bodies.items():
            owned = [p for p in colliders if Sdf.Path(p).HasPrefix(prim.GetPath())]
            if not owned:
                raise ValueError('Task body has no enabled collision shape: ' + path)
            objects[body_paths[path]] = {'path': path, 'collision_paths': owned,
                'initial_state': _world_pose(stage, prim), 'task_layer_id': layer_id}
        static = [p for p in colliders if not any(Sdf.Path(p).HasPrefix(Sdf.Path(b)) for b in actual_bodies)]
        static_paths.extend(static)
        particles = spec.get('particle_sets', {})
        if not isinstance(particles, dict):
            raise ValueError('particle_sets must be a mapping')
        declared_sets = []
        for name, particle in particles.items():
            _name(name)
            if set(particle) != {'path', 'system_path', 'count', 'total_mass_kg', 'position_attribute'}:
                raise ValueError('Particle set requires explicit path, system, count, mass and raw attribute')
            pp, sp = _path(particle['path']), _path(particle['system_path'])
            if not pp.HasPrefix(target) or str(sp) not in systems or particle['position_attribute'] != 'points':
                raise ValueError('Particle set must use the declared layer and unsmoothed points')
            if isinstance(particle['count'], bool) or not isinstance(particle['count'], int) or particle['count'] <= 0:
                raise ValueError('Particle count must be a positive integer')
            if not np.isfinite(particle['total_mass_kg']) or particle['total_mass_kg'] <= 0:
                raise ValueError('Particle mass must be finite and positive')
            declared_sets.append(str(pp))
        # The composed schema token also exists in CPU USD builds without the
        # optional PhysX schema plugin; do not silently miss authored particles.
        actual_sets = [str(p.GetPath()) for p in Usd.PrimRange(root)
                       if p.GetMetadata('apiSchemas') is not None
                       and 'PhysxParticleSetAPI' in p.GetMetadata('apiSchemas').GetAppliedItems()]
        if set(actual_sets) != set(declared_sets):
            raise ValueError('Unobserved particle sets in task layer')
        records.append({'id': layer_id, 'usd_path': str(file), 'sha256': spec['sha256'],
            'source_prim_path': str(source), 'target_prim_path': str(target),
            'dependencies_sha256': dependencies, 'rigid_bodies': declared,
            'joints': joints, 'particle_sets': particles, 'static_collision_paths': static,
            'physics_scene_path': str(physics_scene_path), 'physical_parameters_modified': False})
    return {'layers': records, 'objects': objects, 'static_collision_paths': static_paths}
