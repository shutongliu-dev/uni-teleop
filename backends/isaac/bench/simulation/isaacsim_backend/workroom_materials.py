"""Opt-in materials on the original workroom supports, never source-asset edits.

Static supports retain their actor model. Their native collider/material binding
can be queried, but the rigid-body tensor API cannot establish their coefficients.
For existing kinematic actors we also verify every native shape's coefficients.
"""
from __future__ import annotations

import math

SCHEMA = 'workroom-surface-materials-v1'
COMBINE_MODES = ('average', 'min', 'multiply', 'max')
VALUE_KEYS = ('static_friction', 'dynamic_friction', 'restitution')
MODE_KEYS = ('friction_combine', 'restitution_combine')
FIELDS = {*VALUE_KEYS, *MODE_KEYS, 'parameter_source'}
MATERIAL_ROOT = '/World/PhysicsMaterials/Workroom'


def material_options(config):
    """Require explicit coefficients, combine modes and a simulation prior source."""
    if config is None:
        return None
    if (not isinstance(config, dict) or set(config) != {'schema', 'surfaces'}
            or config['schema'] != SCHEMA):
        raise ValueError(f'workroom_surface_materials requires schema {SCHEMA} and surfaces')
    surfaces = config['surfaces']
    if not isinstance(surfaces, dict) or not surfaces or set(surfaces) - {'table', 'floor'}:
        raise ValueError('Workroom materials require a nonempty table/floor surface selection')
    result = {}
    for name, spec in sorted(surfaces.items()):
        if not isinstance(spec, dict) or set(spec) != FIELDS:
            raise ValueError(f'Workroom {name} material needs exactly {sorted(FIELDS)}')
        for key in VALUE_KEYS:
            value = spec[key]
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError(f'Workroom {name}.{key} must be a finite non-negative number')
        if spec['static_friction'] < spec['dynamic_friction']:
            raise ValueError(f'Workroom {name} static_friction must be >= dynamic_friction')
        if spec['restitution'] > 1:
            raise ValueError(f'Workroom {name} restitution must be in [0, 1]')
        if any(spec[key] not in COMBINE_MODES for key in MODE_KEYS):
            raise ValueError(f'Workroom {name} combine modes must be one of {COMBINE_MODES}')
        if not isinstance(spec['parameter_source'], str) or not spec['parameter_source'].strip():
            raise ValueError(f'Workroom {name} parameter_source must be nonempty')
        result[name] = {**spec, **{key: float(spec[key]) for key in VALUE_KEYS}}
    return {'schema': SCHEMA, 'surfaces': result}


def _collider(stage, path):
    from pxr import UsdPhysics
    prim = stage.GetPrimAtPath(path)
    if (not prim or not prim.IsActive() or not prim.HasAPI(UsdPhysics.CollisionAPI)
            or UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get() is False):
        raise ValueError(f'Workroom material target is not an enabled collider: {path}')
    return prim


def _check_values(actual, expected, context):
    if any(actual.get(key) is None or not math.isclose(actual[key], expected[key], rel_tol=1e-6, abs_tol=1e-7)
           for key in VALUE_KEYS):
        raise RuntimeError(f'Workroom material coefficients differ at {context}: {actual}')
    if any(actual.get(key) != expected[key] for key in MODE_KEYS):
        raise RuntimeError(f'Workroom material combine modes differ at {context}: {actual}')


def resolved_material(stage, collider_path):
    """Read USD resolution; this is explicitly not a native coefficient query."""
    from pxr import UsdPhysics, UsdShade
    material, _ = UsdShade.MaterialBindingAPI(_collider(stage, collider_path)).ComputeBoundMaterial('physics')
    if not material or not material.GetPrim().HasAPI(UsdPhysics.MaterialAPI):
        raise RuntimeError(f'Workroom collider lacks a resolved physics material: {collider_path}')
    api = UsdPhysics.MaterialAPI(material.GetPrim())
    return {'material_path': str(material.GetPath()),
            'static_friction': api.GetStaticFrictionAttr().Get(),
            'dynamic_friction': api.GetDynamicFrictionAttr().Get(),
            'restitution': api.GetRestitutionAttr().Get(),
            'friction_combine': material.GetPrim().GetAttribute('physxMaterial:frictionCombineMode').Get(),
            'restitution_combine': material.GetPrim().GetAttribute('physxMaterial:restitutionCombineMode').Get()}


def apply_materials(stage, supports, config=None):
    """Author only runtime materials and exact support bindings; absent means no-op."""
    request = material_options(config)
    if request is None:
        return None
    from pxr import Sdf, UsdPhysics, UsdShade
    targets = {}
    for name in request['surfaces']:
        if name not in supports:
            raise ValueError(f'Workroom has no declared {name} support')
        prim = _collider(stage, supports[name])
        if prim.IsInstanceProxy():
            raise ValueError(f'Workroom support must permit a local collider binding: {prim.GetPath()}')
        path = f'{MATERIAL_ROOT}/{name}'
        if stage.GetPrimAtPath(path):
            raise ValueError(f'Workroom material destination already exists: {path}')
        targets[name] = prim
    records = {}
    for name, spec in request['surfaces'].items():
        material = UsdShade.Material.Define(stage, f'{MATERIAL_ROOT}/{name}')
        prim = material.GetPrim()
        api = UsdPhysics.MaterialAPI.Apply(prim)
        api.CreateStaticFrictionAttr(spec['static_friction'])
        api.CreateDynamicFrictionAttr(spec['dynamic_friction'])
        api.CreateRestitutionAttr(spec['restitution'])
        # Codeless USD authoring also works in CPU-only OpenUSD preflight. The
        # pinned native runtime registers PhysxMaterialAPI and reads these tokens.
        prim.AddAppliedSchema('PhysxMaterialAPI')
        for key, attr in (('friction_combine', 'frictionCombineMode'),
                          ('restitution_combine', 'restitutionCombineMode')):
            prim.CreateAttribute('physxMaterial:' + attr, Sdf.ValueTypeNames.Token, custom=False).Set(spec[key])
        UsdShade.MaterialBindingAPI.Apply(targets[name]).Bind(
            material, UsdShade.Tokens.strongerThanDescendants, 'physics')
        collider_path = str(targets[name].GetPath())
        actual = resolved_material(stage, collider_path)
        if actual['material_path'] != str(material.GetPath()):
            raise RuntimeError(f'Workroom support material was overridden: {collider_path}')
        _check_values(actual, spec, collider_path)
        records[name] = {'collider_path': collider_path, 'material_path': str(material.GetPath()),
                         'requested': dict(spec), 'usd_resolved': actual}
    return {'schema': SCHEMA, 'surfaces': records, 'source_assets_modified': False,
            'physics_accepted': False, 'parameter_basis': 'Explicit simulation priors, not measured material calibration',
            'binding_scope': 'Only named table/floor collision prims; visual materials and other scenery remain unchanged',
            'combine_priority': list(COMBINE_MODES)}


def _actor(stage, collider_path):
    from pxr import UsdPhysics
    prim = stage.GetPrimAtPath(collider_path)
    while prim and not prim.IsPseudoRoot():
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            api = UsdPhysics.RigidBodyAPI(prim)
            if api.GetRigidBodyEnabledAttr().Get() is not False:
                if api.GetKinematicEnabledAttr().Get() is not True:
                    raise RuntimeError(f'Workroom support unexpectedly belongs to a dynamic body: {prim.GetPath()}')
                return str(prim.GetPath())
        prim = prim.GetParent()
    return None


def runtime_material_check(stage, sim_view, scene_info, *, native_material_paths=None):
    """Verify live native binding and, when exposed, native shape coefficients.

    Call after native initialization and after hard resets. No state/material
    writes or actor conversions occur. Static actor coefficients and all combine
    modes are USD readbacks, not numeric native readbacks; the report says so.
    """
    workbench = scene_info.get('workbench', {})
    materials = workbench.get('surface_materials')
    if materials is None:
        return None
    from pxr import Usd, UsdPhysics
    if native_material_paths is None:
        from omni.physx import get_physxunittests_interface
        native_material_paths = get_physxunittests_interface().get_materials_paths
    mapping = workbench.get('collider_actor_paths', {})
    records = {}
    for name, record in materials['surfaces'].items():
        collider_path, expected_path = record['collider_path'], record['material_path']
        expected = record['requested']
        actual = resolved_material(stage, collider_path)
        if actual['material_path'] != expected_path:
            raise RuntimeError(f'Workroom USD material binding changed: {collider_path}')
        _check_values(actual, expected, collider_path)
        paths = [str(path) for path in native_material_paths(collider_path)]
        if not paths or set(paths) != {expected_path}:
            raise RuntimeError(f'Workroom native material binding mismatch: {collider_path}: {paths}')
        owner = _actor(stage, collider_path)
        if collider_path in mapping and mapping[collider_path] != owner:
            raise RuntimeError(f'Workroom material actor mapping differs from USD ownership: {collider_path}')
        shapes = None
        if owner is not None:
            # A one-body tensor has no max_shapes padding. For compounds, all
            # collider materials must agree before assigning body-local rows to
            # this support's material; traversal order never identifies shapes.
            for prim in Usd.PrimRange(stage.GetPrimAtPath(owner), Usd.TraverseInstanceProxies()):
                if (prim.HasAPI(UsdPhysics.CollisionAPI)
                        and UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get() is not False
                        and resolved_material(stage, str(prim.GetPath()))['material_path'] != expected_path):
                    raise RuntimeError(f'Workroom native material attribution is ambiguous on compound actor: {owner}')
            view = sim_view.create_rigid_body_view(owner)
            if view.count != 1 or list(view.prim_paths) != [owner] or view.max_shapes < 1:
                raise RuntimeError(f'Workroom native material query did not resolve one exact actor: {owner}')
            rows = view.get_material_properties()
            if hasattr(rows, 'detach'):
                rows = rows.detach().cpu()
            if hasattr(rows, 'numpy'):
                rows = rows.numpy()
            rows = rows.tolist()
            if len(rows) != 1 or len(rows[0]) != view.max_shapes or any(len(row) != 3 for row in rows[0]):
                raise RuntimeError(f'Workroom native material tensor dimensions differ: {owner}')
            shapes = []
            for index, row in enumerate(rows[0]):
                values = {key: float(value) for key, value in zip(VALUE_KEYS, row)}
                _check_values({**values, **{key: expected[key] for key in MODE_KEYS}}, expected,
                              f'{owner} shape {index}')
                shapes.append({'shape_index': index, 'material_properties': [values[key] for key in VALUE_KEYS]})
        records[name] = {**record, 'usd_resolved': actual, 'actor_path': owner,
                         'native_material_paths': paths, 'native_binding_verified': True,
                         'native_coefficients_verified': shapes is not None, 'native_shapes': shapes,
                         'coefficient_evidence': 'native_rigid_body_tensor' if shapes is not None else 'usd_only_static_actor',
                         'combine_evidence': 'usd_only', 'native_combine_verified': False}
    return {'schema': SCHEMA, 'passed': True, 'surfaces': records,
            'all_native_coefficients_verified': all(row['native_coefficients_verified'] for row in records.values()),
            'native_combine_verified': False,
            'scope': 'Native support material bindings plus coefficients on existing kinematic actors; '
                     'static coefficients and combine modes are USD evidence only; no contact-dynamics qualification'}
