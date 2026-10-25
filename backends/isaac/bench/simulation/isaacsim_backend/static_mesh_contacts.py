"""Explicit fixed-kinematic actors for otherwise invisible static mesh contacts.

Only actor flags are authored into the runtime layer. Meshes, cooking settings,
transforms and materials remain the original collision geometry. No pose targets
are issued: a batched native guard checks every integration step instead.
"""
from __future__ import annotations

import hashlib
import json
import numpy as np

MODE = 'kinematic_mesh_v1'


def _under(path, roots):
    return any(path == root or path.startswith(root + '/') for root in roots)


def geometry_signature(prim):
    """Fingerprint full authored mesh/collision data and evaluated world pose."""
    from pxr import Usd, UsdGeom, UsdShade, UsdPhysics
    values = {}
    for attr in prim.GetAttributes():
        name = attr.GetName()
        if name in UsdPhysics.RigidBodyAPI.GetSchemaAttributeNames():
            continue
        value = attr.Get()
        if value is not None:
            try:
                array = np.asarray(value)
                if array.dtype.kind in 'biufc' and array.ndim:
                    values[name] = [str(array.dtype), list(array.shape),
                                    hashlib.sha256(array.tobytes()).hexdigest()]
                    continue
            except (TypeError, ValueError):
                pass
            values[name] = str(value)
    values['world_transform'] = np.asarray(UsdGeom.XformCache(Usd.TimeCode.Default()).GetLocalToWorldTransform(prim)).tolist()
    material, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial('physics')
    values['physics_material'] = str(material.GetPath()) if material else None
    return hashlib.sha256(json.dumps(values, sort_keys=True).encode()).hexdigest()


def adapt_static_meshes(stage, mode=None, *, exclude_paths=()):
    """Convert only unowned static collision meshes, failing ambiguous ancestry.

A disabled body schema is retained and enabled at its existing non-proxy owner.
With no body schema, the collision mesh itself becomes the actor. Disabled body
schemas are never ignored when determining ownership: they can delimit compounds.
"""
    if mode is None:
        return None
    if mode != MODE:
        raise ValueError(f'Unknown static_mesh_contact_mode: {mode!r}')
    from pxr import Usd, UsdGeom, UsdPhysics
    prims = list(Usd.PrimRange.Stage(stage, Usd.TraverseInstanceProxies()))
    owners = {}
    for prim in prims:
        path = str(prim.GetPath())
        if (_under(path, exclude_paths) or not prim.IsA(UsdGeom.Mesh)
                or not prim.HasAPI(UsdPhysics.CollisionAPI)
                or UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get() is False):
            continue
        chain, current = [], prim
        while current and not current.IsPseudoRoot():
            if current.HasAPI(UsdPhysics.RigidBodyAPI):
                chain.append(current)
            current = current.GetParent()
        if any(UsdPhysics.RigidBodyAPI(p).GetRigidBodyEnabledAttr().Get() is not False for p in chain):
            # Existing dynamic bodies and native kinematic task fixtures already
            # expose actor filters. Their geometry and actor model stay intact.
            continue
        if len(chain) > 1:
            raise ValueError(f'Nested disabled rigid-body ownership is ambiguous: {path}')
        owner = chain[0] if chain else prim
        if owner.IsInstanceProxy():
            raise ValueError(f'Static mesh needs a non-proxy actor owner: {path}')
        owners[str(owner.GetPath())] = owner
    paths = sorted(owners)
    if any(_under(path, paths[:i] + paths[i+1:]) for i, path in enumerate(paths)):
        raise ValueError('Static mesh actor owners must not be nested')
    mapping, signatures, approximations = {}, {}, {}
    # Complete preflight before mutating any actor flags.
    for path, owner in owners.items():
        for prim in Usd.PrimRange(owner, Usd.TraverseInstanceProxies()):
            child = str(prim.GetPath())
            if prim != owner and prim.HasAPI(UsdPhysics.RigidBodyAPI):
                raise ValueError(f'Nested rigid-body schema would truncate collider ownership: {child}')
            if (prim.IsA(UsdPhysics.Scene) or prim.HasAPI(UsdPhysics.ArticulationRootAPI)
                    or (prim.IsA(UsdPhysics.Joint) and UsdPhysics.Joint(prim).GetJointEnabledAttr().Get() is not False)):
                raise ValueError(f'Static mesh owner contains dynamics: {child}')
            if (prim.HasAPI(UsdPhysics.CollisionAPI)
                    and UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get() is not False):
                mapping[child] = path
                signatures[child] = geometry_signature(prim)
                if prim.IsA(UsdGeom.Mesh):
                    approximations[child] = UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get() or 'none'
    for owner in owners.values():
        body = UsdPhysics.RigidBodyAPI.Apply(owner)
        body.CreateRigidBodyEnabledAttr(True)
        body.CreateKinematicEnabledAttr(True)
    for path, before in signatures.items():
        if geometry_signature(stage.GetPrimAtPath(path)) != before:
            raise RuntimeError(f'Static mesh adaptation changed collision geometry: {path}')
    return {'mode': mode, 'kinematic_body_paths': paths, 'collider_actor_paths': mapping,
            'geometry_sha256': signatures, 'mesh_approximations': approximations,
            'geometry_modified': False, 'runtime_pose_targets': False,
            'scope': 'Original collision geometry on fixed kinematic actors; native solver verification required.'}


class FixedActorGuard:
    """CPU reduction of one batched native transform read per physics step."""
    def __init__(self, paths, transforms):
        self.paths = list(paths)
        self.reference = self._checked(transforms).copy()
        self.checks = 0
        self.maximum_translation_m = 0.
        self.maximum_angle_rad = 0.

    def _checked(self, transforms):
        value = np.asarray(transforms, dtype=np.float64)
        if value.shape != (len(self.paths), 7) or not np.isfinite(value).all():
            raise RuntimeError('Invalid native fixed-actor transform batch')
        norms = np.linalg.norm(value[:, 3:], axis=1)
        if np.any(np.abs(norms - 1.) > .001):
            raise RuntimeError('Invalid native fixed-actor quaternion')
        value = value.copy()
        value[:, 3:] /= norms[:, None]
        return value

    def check(self, paths, transforms):
        if list(paths) != self.paths:
            raise RuntimeError('Native fixed-actor path order changed')
        value = self._checked(transforms)
        translation = float(np.max(np.linalg.norm(value[:, :3] - self.reference[:, :3], axis=1), initial=0.))
        # The minimum quaternion chord is stable near zero and handles q == -q.
        chord = np.minimum(np.linalg.norm(value[:, 3:] - self.reference[:, 3:], axis=1),
                           np.linalg.norm(value[:, 3:] + self.reference[:, 3:], axis=1))
        angle = float(np.max(4*np.arcsin(np.clip(chord/2, 0., 1.)), initial=0.))
        self.checks += 1
        self.maximum_translation_m = max(self.maximum_translation_m, translation)
        self.maximum_angle_rad = max(self.maximum_angle_rad, angle)
        if translation > 1.e-6 or angle > 1.e-5:
            raise RuntimeError(f'Fixed kinematic scenery moved: {translation:g} m, {angle:g} rad')

    def report(self):
        return {'actor_paths': self.paths, 'checked_batches': self.checks,
                'maximum_translation_m': self.maximum_translation_m,
                'maximum_angle_rad': self.maximum_angle_rad,
                'translation_tolerance_m': 1.e-6, 'angle_tolerance_rad': 1.e-5,
                'scope': 'One native batched actor pose read at initialization and every integration substep; reference persists across reset.'}
