"""Native references for collaborator rigid USDs; import after SimulationApp.

References preserve source composition, texture paths and authored collision
approximations. Unsupported compound articulations are rejected by the catalog
builder.

Two placement forms, chosen per asset from the composed source:

* root form (every asset the loader supported before the anchored form, incl.
  all 203 assets of the frozen target_cube catalog): the source default prim carries RigidBodyAPI
  and an identity transform. It is referenced straight onto /World/<kind>/<id>,
  which therefore IS the rigid body, and the reset-only world transform replaces
  its op order. This branch authors exactly what the loader always authored, so
  frozen release stages and the evaluators that match contact actors against
  '/World/Objects/<id>' are unchanged.
* anchored form (previously unloadable): the rigid body sits on a CHILD prim
  and/or the default prim authors a non-identity transform -- the SHARPA v3 bolt
  and fixture keep their body on /allen_bolt and /platform under a 1.5x
  xformOp:scale. Clearing that op order would shrink them to 0.67x, so the
  source is referenced onto a child anchor and only our own wrapper is placed;
  the returned 'path' is then the rigid-body prim below the wrapper.

Tasks may opt into ``usd_physics_policy='preserve_authored_v1'`` to retain source
physics parameters and material bindings. ``usd_physics_fallbacks`` supplies
contact/rest offsets and solver iterations only where the source has no authored
opinion; the returned record distinguishes source, task and runtime defaults.
This policy reports composed USD evidence, not native contact qualification.
"""
from __future__ import annotations
from copy import deepcopy
import math

from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade


def _is_identity(matrix, tolerance=1e-9):
    return all(abs(matrix[i][j] - (1. if i == j else 0.)) <= tolerance for i in range(4) for j in range(4))


def placement_form(asset):
    """'root' or 'anchored' for this asset, from its composed source (read-only).

    Decided on a scratch stage so that the scene stage only ever receives the
    authoring of the chosen form.
    """
    probe_stage = Usd.Stage.CreateInMemory()
    probe = probe_stage.DefinePrim('/probe')
    probe.GetReferences().AddReference(asset['usd_path'], Sdf.Path(asset['usd_default_prim']))
    local = (UsdGeom.Xformable(probe).GetLocalTransformation(Usd.TimeCode.Default())
             if probe.IsA(UsdGeom.Xformable) else Gf.Matrix4d(1.))
    if probe.HasAPI(UsdPhysics.RigidBodyAPI) and _is_identity(local):
        return 'root'
    if probe.HasAPI(UsdPhysics.RigidBodyAPI) or any(p.HasAPI(UsdPhysics.RigidBodyAPI) for p in Usd.PrimRange(probe)):
        return 'anchored'
    raise ValueError(f'USD catalog root no longer contains a rigid body: {asset["usd_path"]}')


def reference_anchor_path(object_path, usd_default_prim):
    """Where an anchored source is referenced: a child of ``object_path``, named after it."""
    name = str(usd_default_prim).rstrip('/').rsplit('/', 1)[-1]
    if not name:
        raise ValueError(f'{usd_default_prim!r} is not a prim path')
    return str(object_path).rstrip('/') + '/' + name


def rigid_body_prim(anchor, asset):
    """The prim carrying RigidBodyAPI: the anchor itself, or a descendant of it."""
    if anchor.HasAPI(UsdPhysics.RigidBodyAPI):
        return anchor
    for prim in Usd.PrimRange(anchor):
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            return prim
    raise ValueError(f'USD catalog root no longer contains a rigid body: {asset["usd_path"]}')


def require_body_at_wrapper_origin(wrapper, rigid_prim, asset, tolerance=1e-6):
    """Refuse a rigid body that is translated or rotated relative to its wrapper.

    ``env.reset`` writes the task's ``position_m``/``quaternion_wxyz`` straight
    into the rigid-body view, i.e. as the BODY's world pose, while the stage
    places the WRAPPER. The two agree only when the body sits at the wrapper's
    origin with no rotation. A uniform scale is fine -- PhysX folds an ancestor
    scale into the shapes -- which is exactly the SHARPA v3 case (1.5x).
    """
    matrix, _ = UsdGeom.XformCache(Usd.TimeCode.Default()).ComputeRelativeTransform(rigid_prim, wrapper)
    translation = matrix.ExtractTranslation()
    rows = [Gf.Vec3d(matrix[i][0], matrix[i][1], matrix[i][2]) for i in range(3)]
    scales = [row.GetLength() for row in rows]
    if min(scales) <= 0.:
        raise ValueError(f'{asset["usd_path"]}: degenerate transform above {rigid_prim.GetPath()}')
    rotation_error = max(abs(rows[i][j] / scales[i] - (1. if i == j else 0.)) for i in range(3) for j in range(3))
    if translation.GetLength() > tolerance or rotation_error > tolerance or max(scales) - min(scales) > tolerance:
        raise ValueError(
            f'{asset["usd_path"]}: rigid body {rigid_prim.GetPath()} is offset from its placement wrapper '
            f'(translation {tuple(round(v, 6) for v in translation)}, rotation error {rotation_error:.2e}, '
            f'scales {tuple(round(v, 6) for v in scales)}); reset would write the task pose to the body '
            'and disagree with the stage placement')


def _placement_matrix(spec):
    q = spec.get('quaternion_wxyz', [1, 0, 0, 0])
    rotation = Gf.Quatd(float(q[0]), Gf.Vec3d(*map(float, q[1:])))
    rotation.Normalize()
    matrix = Gf.Matrix4d(1.)
    matrix.SetRotateOnly(rotation)
    matrix.SetTranslateOnly(Gf.Vec3d(*map(float, spec['position_m'])))
    return matrix


def _physics_policy(spec):
    policy = spec.get('usd_physics_policy')
    if policy not in (None, 'preserve_authored_v1'):
        raise ValueError(f'Unknown USD physics policy: {policy!r}')
    if policy is not None and 'friction' in spec:
        raise ValueError('preserve_authored_v1 does not accept a task friction override')
    if 'usd_physics_fallbacks' in spec:
        if policy is None:
            raise ValueError('usd_physics_fallbacks requires preserve_authored_v1')
        fallbacks = spec['usd_physics_fallbacks']
        allowed = {'contact_offset_m', 'rest_offset_m', 'solver_position_iterations', 'solver_velocity_iterations',
                   'max_depenetration_velocity_m_s', 'linear_damping', 'angular_damping',
                   'sleep_threshold', 'enable_speculative_ccd'}
        if not isinstance(fallbacks, dict) or set(fallbacks) - allowed:
            raise ValueError('Invalid usd_physics_fallbacks keys')
        for name, value in fallbacks.items():
            if name == 'enable_speculative_ccd':
                if not isinstance(value, bool):
                    raise ValueError('Speculative CCD fallback must be boolean')
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f'Invalid USD physics fallback: {name}')
            if name.endswith('_iterations') and (not isinstance(value, int) or value < (1 if name == 'solver_position_iterations' else 0)):
                raise ValueError(f'Invalid USD solver iteration fallback: {name}')
            if name == 'contact_offset_m' and value <= 0.:
                raise ValueError('USD contact offset fallback must be positive')
            if name in ('max_depenetration_velocity_m_s', 'linear_damping', 'angular_damping', 'sleep_threshold') and value < 0.:
                raise ValueError('USD body fallback cannot be negative: ' + name)
    if 'usd_physics_material_fallback' in spec:
        material = spec['usd_physics_material_fallback']
        if policy is None or not isinstance(material, dict) or set(material) != {
                'static_friction', 'dynamic_friction', 'restitution', 'friction_combine'}:
            raise ValueError('Explicit USD material fallback requires preserve_authored_v1 and all four fields')
        for key in ('static_friction', 'dynamic_friction', 'restitution'):
            value = material[key]
            if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or value < 0:
                raise ValueError('Invalid USD material fallback: ' + key)
        if material['restitution'] > 1 or material['friction_combine'] not in ('average', 'min', 'multiply', 'max'):
            raise ValueError('Invalid USD material restitution or friction combine mode')
    return policy


def _preserve_or_fallback(prim, name, fallback, create, *, task_fallback=False):
    """Keep a composed source opinion; schema defaults are not source opinions.

    Called before this loader authors the corresponding attribute. The source
    layer is never edited, and a missing opinion receives the documented runtime
    fallback even when a native schema has a different default.
    """
    attr = prim.GetAttribute(name)
    authored = bool(attr and attr.HasAuthoredValueOpinion())
    if authored:
        value = attr.Get()
        if value is None:
            raise ValueError(f'Authored USD physics parameter has no default value: {prim.GetPath()}.{name}')
    else:
        value = fallback
        create(fallback)
    return {'value': value, 'origin': 'source_authored' if authored else
            'task_fallback' if task_fallback else 'runtime_fallback'}


def _configure_body(stage, spec, asset, *, dynamic, path, rigid_prim, binding_prim, collider_base):
    """Rigid-body flags, physics material and colliders; shared by both forms."""
    from .scene import rigid_material

    preserve = _physics_policy(spec) is not None
    fallbacks = spec.get('usd_physics_fallbacks', {})
    report = None
    if preserve:
        gravity = rigid_prim.GetAttribute('physxRigidBody:disableGravity')
        if dynamic and gravity and gravity.HasAuthoredValueOpinion() and gravity.Get() is not False:
            raise ValueError(f'Dynamic USD asset must enable gravity: {rigid_prim.GetPath()}')
        report = {'policy': 'preserve_authored_v1', 'body_path': str(rigid_prim.GetPath()),
                  'body_parameters': {}, 'collider_parameters': {}, 'collider_materials': {},
                  'fallbacks': deepcopy(fallbacks),
                  'fallback_provenance': deepcopy(spec.get('usd_physics_fallbacks_provenance')),
                  'body_mode': {'rigid_body_enabled': True, 'kinematic_enabled': not dynamic,
                                'origin': 'task_dynamic_or_fixture_contract'},
                  'source_files_modified': False,
                  'scope': 'Composed USD parameter evidence; native values and contact behavior require separate validation.'}
    rigid = UsdPhysics.RigidBodyAPI(rigid_prim)
    # A kinematic body is an immovable fixture. The rigid root also lets GPU
    # contact tensors identify compound convex-decomposition support contacts;
    # filtering its individual static multi-shape collider is unsupported.
    rigid.CreateRigidBodyEnabledAttr(True)
    rigid.CreateKinematicEnabledAttr(not dynamic)
    if dynamic:
        body = PhysxSchema.PhysxRigidBodyAPI.Apply(rigid_prim)
        if preserve:
            for name, key, fallback, create in (
                ('solverPositionIterationCount', 'solver_position_iterations', 16, body.CreateSolverPositionIterationCountAttr),
                ('solverVelocityIterationCount', 'solver_velocity_iterations', 4, body.CreateSolverVelocityIterationCountAttr),
                ('enableSpeculativeCCD', 'enable_speculative_ccd', True, body.CreateEnableSpeculativeCCDAttr),
                ('disableGravity', None, False, body.CreateDisableGravityAttr),
            ):
                attr_name = 'physxRigidBody:' + name
                report['body_parameters'][attr_name] = _preserve_or_fallback(
                    rigid_prim, attr_name, fallbacks.get(key, fallback), create, task_fallback=key in fallbacks)
            for name, key, create in (
                ('maxDepenetrationVelocity', 'max_depenetration_velocity_m_s', body.CreateMaxDepenetrationVelocityAttr),
                ('linearDamping', 'linear_damping', body.CreateLinearDampingAttr),
                ('angularDamping', 'angular_damping', body.CreateAngularDampingAttr),
                ('sleepThreshold', 'sleep_threshold', body.CreateSleepThresholdAttr),
            ):
                if key in fallbacks:
                    report['body_parameters']['physxRigidBody:' + name] = _preserve_or_fallback(
                        rigid_prim, 'physxRigidBody:' + name, fallbacks[key], create, task_fallback=True)
        else:
            body.CreateSolverPositionIterationCountAttr(16)
            body.CreateSolverVelocityIterationCountAttr(4)
            body.CreateEnableSpeculativeCCDAttr(True)
        contact_report = PhysxSchema.PhysxContactReportAPI.Apply(rigid_prim)
        if preserve:
            report['body_parameters']['physxContactReport:threshold'] = _preserve_or_fallback(
                rigid_prim, 'physxContactReport:threshold', 0., contact_report.CreateThresholdAttr)
        else:
            contact_report.CreateThresholdAttr(0.)
    material_fallback = spec.get('usd_physics_material_fallback')
    material = rigid_material(stage, spec['id'], material_fallback['static_friction']
                              if material_fallback else spec.get('friction', .65))
    if material_fallback:
        fallback_api = UsdPhysics.MaterialAPI(material.GetPrim())
        fallback_api.CreateDynamicFrictionAttr(material_fallback['dynamic_friction'])
        fallback_api.CreateRestitutionAttr(material_fallback['restitution'])
        PhysxSchema.PhysxMaterialAPI.Apply(material.GetPrim()).CreateFrictionCombineModeAttr(material_fallback['friction_combine'])
        report['material_fallback'] = deepcopy(material_fallback)
    # A default material on the root yields to original collider bindings. An
    # explicit task friction override binds every collider and is recorded.
    if not preserve:
        UsdShade.MaterialBindingAPI.Apply(binding_prim).Bind(material, UsdShade.Tokens.weakerThanDescendants, 'physics')
    collision_paths = []
    material_combine_reports = {}

    def configure_preserved_material(collider):
        # Resolve before writing so source physics bindings on either the body
        # or collider retain priority. Disabled source shapes can still occupy
        # native material slots and must have the same explicit fallback audit.
        collider_path = str(collider.GetPath())
        binding = UsdShade.MaterialBindingAPI.Apply(collider)
        resolved, _ = binding.ComputeBoundMaterial('physics')
        source_material = bool(resolved and resolved.GetPrim().HasAPI(UsdPhysics.MaterialAPI))
        if not source_material:
            binding.Bind(material, UsdShade.Tokens.weakerThanDescendants, 'physics')
            resolved, _ = binding.ComputeBoundMaterial('physics')
        if not resolved or not resolved.GetPrim().HasAPI(UsdPhysics.MaterialAPI):
            raise ValueError(f'USD collider has no resolved physics material: {collider_path}')
        report['collider_materials'][collider_path] = {
            'path': str(resolved.GetPath()),
            'origin': 'source_authored' if source_material else 'task_fallback' if material_fallback else 'runtime_fallback',
            'parameters': {name: resolved.GetPrim().GetAttribute(name).Get() for name in (
                'physics:staticFriction', 'physics:dynamicFriction', 'physics:restitution')},
            'scope': 'Composed USD material, not native per-shape verification.'}
        if material_fallback:
            key = str(resolved.GetPath())
            if key not in material_combine_reports:
                api = PhysxSchema.PhysxMaterialAPI.Apply(resolved.GetPrim())
                material_combine_reports[key] = (_preserve_or_fallback(
                    resolved.GetPrim(), 'physxMaterial:frictionCombineMode', material_fallback['friction_combine'],
                    api.CreateFrictionCombineModeAttr, task_fallback=True) if source_material else
                    {'value': material_fallback['friction_combine'], 'origin': 'task_fallback'})
            report['collider_materials'][collider_path]['friction_combine'] = deepcopy(material_combine_reports[key])

    for part in asset['collision_parts']:
        relative = Sdf.Path(part['usd_prim_path']).MakeRelativePath(Sdf.Path(asset['usd_default_prim']))
        collider_path = Sdf.Path(collider_base).AppendPath(relative)
        collider = stage.GetPrimAtPath(collider_path)
        if not collider or not collider.HasAPI(UsdPhysics.CollisionAPI):
            raise ValueError(f'Missing referenced collider {collider_path}')
        physx = PhysxSchema.PhysxCollisionAPI.Apply(collider)
        if preserve:
            report['collider_parameters'][str(collider_path)] = {
                name: _preserve_or_fallback(collider, name, fallbacks.get(key, fallback), create,
                                          task_fallback=key in fallbacks)
                for name, key, fallback, create in (
                    ('physxCollision:contactOffset', 'contact_offset_m', .001, physx.CreateContactOffsetAttr),
                    ('physxCollision:restOffset', 'rest_offset_m', 0., physx.CreateRestOffsetAttr),
                )}
            offsets = report['collider_parameters'][str(collider_path)]
            if offsets['physxCollision:contactOffset']['value'] <= offsets['physxCollision:restOffset']['value']:
                raise ValueError(f'USD contact offset must exceed rest offset: {collider_path}')
            configure_preserved_material(collider)
        else:
            physx.CreateContactOffsetAttr(.001)
            physx.CreateRestOffsetAttr(0.)
        if 'friction' in spec:
            UsdShade.MaterialBindingAPI.Apply(collider).Bind(material, UsdShade.Tokens.strongerThanDescendants, 'physics')
        collision_paths.append(str(collider_path))
    if preserve:
        report['active_collision_paths'] = list(collision_paths)
        report['disabled_collision_paths'] = []
        report['disabled_colliders'] = {}
        for collider in Usd.PrimRange(rigid_prim, Usd.TraverseInstanceProxies()):
            if not collider.HasAPI(UsdPhysics.CollisionAPI):
                continue
            enabled = UsdPhysics.CollisionAPI(collider).GetCollisionEnabledAttr()
            if enabled.Get() is not False:
                continue
            disabled_path = str(collider.GetPath())
            if disabled_path in collision_paths:
                raise ValueError(f'Catalog lists a source-disabled collider as active: {disabled_path}')
            offsets = {}
            for name in ('physxCollision:contactOffset', 'physxCollision:restOffset'):
                attr = collider.GetAttribute(name)
                offsets[name] = {'value': attr.Get() if attr else None,
                                 'source_authored': bool(attr and attr.HasAuthoredValueOpinion())}
            report['disabled_collision_paths'].append(disabled_path)
            report['disabled_colliders'][disabled_path] = {
                'collision_enabled': False,
                'collision_enabled_source_authored': enabled.HasAuthoredValueOpinion(),
                'approximation': UsdPhysics.MeshCollisionAPI(collider).GetApproximationAttr().Get(),
                'preserved_offsets': offsets,
                'scope': 'Composed USD disabled state only; native collision exclusion requires independent evidence.'}
            # Keep the source disabled flag, geometry and offsets untouched.
            # Do not remove this shape or silently discard a native material row.
            configure_preserved_material(collider)
    return collision_paths, report


def add_usd_asset(stage, spec, asset, *, dynamic):
    _physics_policy(spec)
    path = '/World/' + ('Objects/' if dynamic else 'Fixtures/') + spec['id']
    form = placement_form(asset)
    if form == 'root':
        root = UsdGeom.Xform.Define(stage, path).GetPrim()
        root.GetReferences().AddReference(asset['usd_path'], Sdf.Path(asset['usd_default_prim']))
        if not root.HasAPI(UsdPhysics.RigidBodyAPI):
            raise ValueError(f'USD catalog root no longer contains a rigid body: {asset["usd_path"]}')
        # Original roots are verified identity (placement_form). A namespaced matrix
        # avoids changing the type of source xformOp:orient (often quatd, unlike
        # our native quatf).
        xform = UsdGeom.Xformable(root)
        xform.ClearXformOpOrder()
        xform.AddTransformOp(opSuffix='benchPlacement').Set(_placement_matrix(spec))
        rigid_prim, wrapper_path, anchor_path = root, path, path
        collision_paths, physics_report = _configure_body(stage, spec, asset, dynamic=dynamic, path=path, rigid_prim=root,
                                                          binding_prim=root, collider_base=path)
    else:
        # Our wrapper carries only the task placement; the source keeps its own
        # prim (and its own xformOps) on a child anchor.
        wrapper = UsdGeom.Xform.Define(stage, path).GetPrim()
        anchor_path = reference_anchor_path(path, asset['usd_default_prim'])
        anchor = UsdGeom.Xform.Define(stage, anchor_path).GetPrim()
        anchor.GetReferences().AddReference(asset['usd_path'], Sdf.Path(asset['usd_default_prim']))
        rigid_prim = rigid_body_prim(anchor, asset)
        require_body_at_wrapper_origin(wrapper, rigid_prim, asset)
        xform = UsdGeom.Xformable(wrapper)
        xform.ClearXformOpOrder()
        xform.AddTransformOp(opSuffix='benchPlacement').Set(_placement_matrix(spec))
        wrapper_path = path
        # Catalog collider paths are relative to the source default prim, i.e. to
        # the ANCHOR -- never to the rigid body, which may be a descendant.
        collision_paths, physics_report = _configure_body(stage, spec, asset, dynamic=dynamic, path=path, rigid_prim=rigid_prim,
                                                          binding_prim=wrapper, collider_base=anchor_path)
    body_path = str(rigid_prim.GetPath())
    info = {'path': body_path, 'asset_id': spec['asset_id'], 'asset_root': asset['asset_root'],
            'backend': 'usd_reference', 'manifest_sha256': asset['manifest_sha256'],
            'usd_path': asset['usd_path'], 'usd_default_prim': asset['usd_default_prim'],
            'source_dependencies_sha256': deepcopy(asset['source_dependencies_sha256']),
            'runtime_builtin_dependencies': list(asset['runtime_builtin_dependencies']),
            'collision_paths': collision_paths, 'collision_parts': len(collision_paths),
            'contact_filter_paths': [body_path],
            'body_mode': 'dynamic' if dynamic else 'kinematic_fixture',
            'collision_approximations': [x['approximation'] for x in asset['collision_parts']],
            'visual_parts': len(asset['visual_parts']), 'material_prim_paths': asset['material_prim_paths'],
            'mass_kg': asset['mass_kg'] if dynamic else None,
            'center_of_mass_m': asset.get('center_of_mass_m') if dynamic else None,
            'inertia_diag_kg_m2': asset.get('inertia_diag_kg_m2') if dynamic else None,
            'bounds_min_m': asset['bounds_min_m'], 'bounds_max_m': asset['bounds_max_m'],
            'rigid_friction': float(spec.get('usd_physics_material_fallback', {}).get('static_friction', spec.get('friction', .65))),
            'friction_binding': 'explicit task override' if 'friction' in spec else 'fallback; original collider material bindings retain priority',
            'physics_material_purpose': 'physics', 'physics_assumptions': deepcopy(asset['physics_assumptions']),
            'source_modified': False, 'reference_flattened': False}
    if physics_report is not None:
        info['usd_physics_parameters'] = physics_report
    if form == 'anchored':
        # Root-form records stay exactly as they always were (frozen releases
        # store them); only the new form says where its body really is.
        info.update(placement_form=form, wrapper_path=wrapper_path, anchor_path=anchor_path)
    return info
