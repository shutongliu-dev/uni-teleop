"""Task-local USD edits and readbacks for the explicit physical seed profile."""
from copy import deepcopy
from pathlib import Path
import math

import numpy as np


def task_layer_dependencies(spec):
    """Pin actual composition and asset attributes without changing source USD.

    Mirrors the existing native task-layer dependency boundary, including
    AssetArray attributes. Existing pins must still match; they are never
    replaced with a newly observed value.
    """
    from pxr import Sdf, Usd
    from scenes.workrooms.seed_geometry import file_hash
    from bench.simulation.isaacsim_backend.task_layers import verify_file
    file = verify_file(spec['usd_path'], spec['sha256'])
    dependencies = deepcopy(spec.get('dependencies_sha256', {}))
    if not isinstance(dependencies, dict):
        raise ValueError('Task layer dependencies require a path/hash mapping')
    for path, expected in dependencies.items():
        verify_file(path, expected)
    stage = Usd.Stage.Open(str(file))
    if stage is None or not stage.GetPrimAtPath(spec['source_prim_path']):
        raise ValueError('Missing authored task layer source subtree')
    paths = {layer.realPath for layer in stage.GetUsedLayers()
             if not layer.anonymous and layer.realPath != str(file)}
    for prim in Usd.PrimRange(stage.GetPrimAtPath(spec['source_prim_path'])):
        for attr in prim.GetAttributes():
            value = attr.Get()
            assets = [value] if isinstance(value, Sdf.AssetPath) else []
            if attr.GetTypeName() == Sdf.ValueTypeNames.AssetArray and value is not None:
                assets = list(value)
            for asset in assets:
                if asset.path:
                    paths.add(asset.resolvedPath or Sdf.ComputeAssetPathRelativeToLayer(
                        stage.GetRootLayer(), asset.path))
    for name in sorted(paths):
        path = Path(name)
        if not path.is_absolute() or not path.is_file():
            raise ValueError('Task layer references a missing/nonabsolute asset: '+name)
        digest = file_hash(path)
        if str(path) in dependencies and dependencies[str(path)] != digest:
            raise ValueError('Existing task layer dependency changed: '+str(path))
        dependencies[str(path)] = digest
    return dependencies


def check_task_layer_admission(spec):
    """Execute the unchanged native layer admission on a CPU-only USD stage."""
    from pxr import Usd, UsdGeom, UsdPhysics
    from bench.simulation.isaacsim_backend.task_layers import mount_task_layers
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageMetersPerUnit(stage, 1.)
    UsdGeom.SetStageUpAxis(stage, 'Z')
    physics = '/World/ParameterizedAdmissionPhysics'
    UsdPhysics.Scene.Define(stage, physics)
    result = mount_task_layers(stage, [spec], physics_scene_path=physics)
    return {'passed': True, 'observed_bodies': sorted(result['objects']),
            'scope': 'Unchanged native task-layer declaration/dependency checks on a CPU USD stage; no native dynamics or render qualification.'}


def group_layout(parameters, group):
    """Independent group directions retain the sampler's full displacement radius."""
    from .spec import canonical_hash
    value = deepcopy(parameters['layout'])
    if not parameters['canonical']:
        key = [parameters['stream_sha256']['layout'], parameters['seed'], group['id']]
        angle = int(canonical_hash(key)[:13], 16) / (16 ** 13) * 2 * math.pi
        x, y = value['translation_xy_m']; c, s = math.cos(angle), math.sin(angle)
        value['translation_xy_m'] = [c*x-s*y, s*x+c*y]
        if int(canonical_hash([key, 'yaw'])[:2], 16) % 2:
            value['yaw_rad'] *= -1
    if group.get('yaw_policy') == 'fixed_axis_reference':
        value['yaw_rad'] = 0.
        value['yaw_policy'] = 'fixed_axis_reference'
    return value


def world_pose(stage, path):
    from pxr import UsdGeom
    matrix = UsdGeom.XformCache().GetLocalToWorldTransform(stage.GetPrimAtPath(path))
    q = matrix.ExtractRotationQuat()
    return {'prim_path': path, 'position_m': list(matrix.ExtractTranslation()),
            'quaternion_wxyz': [q.GetReal(), *q.GetImaginary()]}


def transform_point(point, layout):
    p = np.asarray(point, dtype=float)
    pivot = np.asarray(layout['pivot_m'], dtype=float)
    yaw = layout['yaw_rad']; c, s = math.cos(yaw), math.sin(yaw)
    rotation = np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])
    return (rotation @ (p-pivot)+pivot+[*layout['translation_xy_m'], 0.]).tolist()


def _rigid(position, quaternion_wxyz):
    from pxr import Gf
    w, x, y, z = (float(v) for v in quaternion_wxyz)
    return Gf.Matrix4d(1).SetRotate(Gf.Quatd(w, Gf.Vec3d(x, y, z)))*Gf.Matrix4d(1).SetTranslate(Gf.Vec3d(*position))


def _pose(path, matrix):
    q = matrix.ExtractRotationQuat()
    return {'prim_path': path, 'position_m': list(matrix.ExtractTranslation()),
            'quaternion_wxyz': [q.GetReal(), *q.GetImaginary()]}


def apply_settled_poses(stage, actors, settled, *, tolerance_m=1e-6):
    """Place every dynamic actor at its recorded robot-free settled world pose.

    Only the rigid part of each body's world transform is replaced (any authored
    scale is kept); body-relative joint frames therefore stay consistent and
    world-anchored joint ends are not moved. Returns before/after receipts.
    """
    from pxr import UsdGeom
    if set(settled) != set(actors):
        raise ValueError('A settled state must cover exactly the dynamic actors')
    cache = UsdGeom.XformCache()
    receipts = {}
    for name in sorted(actors):
        path, row = actors[name], settled[name]
        if row['prim_path'] != path:
            raise ValueError('Settled actor identity differs: '+name)
        q, p = np.asarray(row['quaternion_wxyz'], dtype=float), np.asarray(row['position_m'], dtype=float)
        if q.shape != (4,) or p.shape != (3,) or not np.isfinite([*q, *p]).all() or abs(np.linalg.norm(q)-1) > 1e-5:
            raise ValueError('Invalid settled pose: '+name)
        prim = stage.GetPrimAtPath(path)
        current = cache.GetLocalToWorldTransform(prim)
        before = _pose(path, current.RemoveScaleShear())
        desired = current*current.RemoveScaleShear().GetInverse()*_rigid(p, q/np.linalg.norm(q))
        parent = cache.GetLocalToWorldTransform(prim.GetParent())
        xf = UsdGeom.Xformable(prim); xf.ClearXformOpOrder()
        xf.AddTransformOp(opSuffix='settled').Set(desired*parent.GetInverse())
        cache.Clear()
        after = _pose(path, cache.GetLocalToWorldTransform(prim).RemoveScaleShear())
        error = float(np.linalg.norm(np.asarray(after['position_m'])-p))
        turn = float(min(np.linalg.norm(np.asarray(after['quaternion_wxyz'])-q), np.linalg.norm(np.asarray(after['quaternion_wxyz'])+q)))
        if error > tolerance_m or turn > 1e-5:
            raise ValueError('Settled pose was not applied exactly: '+name)
        receipts[name] = {'prim_path': path, 'before': before, 'after': after}
    return receipts


def apply_layout(stage, groups, parameters):
    """Move a connected group's bodies, supports and world joint anchors together."""
    from pxr import Gf, UsdGeom, UsdPhysics
    receipts = []
    for group in groups:
        paths = group['paths']; poses = {p: world_pose(stage, p) for p in paths}
        pivot = np.mean([p['position_m'] for p in poses.values()], axis=0).tolist()
        layout = {'id': group['id'], 'paths': paths, 'pivot_m': pivot,
                  **group_layout(parameters, group)}
        if not parameters['canonical']:
            yaw = Gf.Matrix4d(1).SetRotate(Gf.Rotation(Gf.Vec3d(0, 0, 1),
                                                       math.degrees(layout['yaw_rad'])))
            matrix = (Gf.Matrix4d(1).SetTranslate(Gf.Vec3d(*(-np.asarray(pivot)))) * yaw *
                      Gf.Matrix4d(1).SetTranslate(Gf.Vec3d(*(np.asarray(pivot)+
                                                  [*layout['translation_xy_m'], 0.]))))
            cache = UsdGeom.XformCache()
            desired = {p: cache.GetLocalToWorldTransform(stage.GetPrimAtPath(p))*matrix for p in paths}
            for path in paths:
                prim = stage.GetPrimAtPath(path)
                parent = cache.GetLocalToWorldTransform(prim.GetParent())
                xf = UsdGeom.Xformable(prim); xf.ClearXformOpOrder()
                xf.AddTransformOp(opSuffix='parameterized').Set(desired[path]*parent.GetInverse())
            for prim in stage.Traverse():
                if not prim.IsA(UsdPhysics.Joint):
                    continue
                joint = UsdPhysics.Joint(prim)
                ends = [list(joint.GetBody0Rel().GetTargets()), list(joint.GetBody1Rel().GetTargets())]
                moving = [bool(e and str(e[0]) in paths) for e in ends]
                if any(moving) and any(e and not m for e, m in zip(ends, moving)):
                    raise ValueError('Layout splits a connected joint: '+str(prim.GetPath()))
                if any(moving):
                    for index, end in enumerate(ends):
                        if not end:
                            position = getattr(joint, 'GetLocalPos'+str(index)+'Attr')()
                            rotation = getattr(joint, 'GetLocalRot'+str(index)+'Attr')()
                            position.Set(Gf.Vec3f(*transform_point(position.Get(), layout)))
                            rotation.Set(Gf.Quatf(yaw.ExtractRotationQuat())*rotation.Get())
        layout['before'] = poses
        layout['after'] = {p: world_pose(stage, p) for p in paths}
        receipts.append(layout)
    return receipts


def apply_physics(stage, actors, root, parameters):
    """Read real body properties; scale mass/tensor once and bind cloned materials."""
    from pxr import Gf, Usd, UsdPhysics, UsdShade
    factor = parameters['physics']['mass_factor']; friction = parameters['physics']['friction_factor']
    masses, materials, colliders, disabled = {}, {}, {}, []
    for name, path in actors.items():
        prim = stage.GetPrimAtPath(path); api = UsdPhysics.MassAPI(prim)
        if not prim or not prim.HasAPI(UsdPhysics.RigidBodyAPI):
            raise ValueError('Missing original rigid actor '+path)
        mass = api.GetMassAttr().Get()
        if mass is None or mass <= 0:
            raise ValueError('Explicit positive parent body mass required: '+path)
        diagonal = api.GetDiagonalInertiaAttr(); old_diagonal = diagonal.Get()
        before = {'mass_kg': float(mass), 'diagonal_inertia_kg_m2':
                  None if old_diagonal is None else list(old_diagonal)}
        if not parameters['canonical']:
            api.CreateMassAttr(float(mass)*factor)
            if old_diagonal is not None and np.all(np.asarray(old_diagonal) > 0):
                diagonal.Set(Gf.Vec3f(*(float(x)*factor for x in old_diagonal)))
            density = api.GetDensityAttr()
            if density.HasAuthoredValue() and density.Get() > 0:
                density.Set(density.Get()*factor)
        masses[name] = {'prim_path': path, 'before': before,
            'mass_kg': float(api.GetMassAttr().Get()), 'mass_factor': factor,
            'diagonal_inertia_kg_m2': None if diagonal.Get() is None else list(diagonal.Get()),
            'inertia_mode': 'explicit_scaled' if old_diagonal is not None and
                np.all(np.asarray(old_diagonal) > 0) else 'native_geometry_computation'}
        for collider in Usd.PrimRange(prim):
            if not collider.HasAPI(UsdPhysics.CollisionAPI):
                continue
            if UsdPhysics.CollisionAPI(collider).GetCollisionEnabledAttr().Get() is False:
                disabled.append(str(collider.GetPath()))
            binding = UsdShade.MaterialBindingAPI(collider)
            original, _ = binding.ComputeBoundMaterial(materialPurpose='physics')
            source = str(original.GetPath()) if original and original.GetPrim().HasAPI(UsdPhysics.MaterialAPI) else None
            key = source or 'chosen_simulator_prior'
            if key not in materials:
                material_api = UsdPhysics.MaterialAPI(original.GetPrim()) if source else None
                baseline = {'static_friction': float(material_api.GetStaticFrictionAttr().Get()) if source else .5,
                            'dynamic_friction': float(material_api.GetDynamicFrictionAttr().Get()) if source else .5,
                            'restitution': float(material_api.GetRestitutionAttr().Get()) if source else 0.}
                dest = source
                actual = deepcopy(baseline) if source else None
                if not parameters['canonical']:
                    dest = root+'/ParameterizedPhysics/Material_'+str(len(materials))
                    material = UsdShade.Material.Define(stage, dest)
                    target = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
                    target.CreateStaticFrictionAttr(baseline['static_friction']*friction)
                    target.CreateDynamicFrictionAttr(baseline['dynamic_friction']*friction)
                    target.CreateRestitutionAttr(baseline['restitution'])
                    actual = {'static_friction': float(target.GetStaticFrictionAttr().Get()),
                              'dynamic_friction': float(target.GetDynamicFrictionAttr().Get()),
                              'restitution': float(target.GetRestitutionAttr().Get())}
                    # Preserve authored PhysX combine modes and other contact-material attributes.
                    if source:
                        for schema in original.GetPrim().GetAppliedSchemas():
                            if schema != 'PhysicsMaterialAPI': material.GetPrim().AddAppliedSchema(schema)
                        for attr in original.GetPrim().GetAttributes():
                            if attr.GetName().startswith('physxMaterial:') and attr.HasAuthoredValue():
                                material.GetPrim().CreateAttribute(attr.GetName(), attr.GetTypeName()).Set(attr.Get())
                materials[key] = {'path': dest, 'parent_material': source, 'baseline': baseline,
                    'applied': actual,
                    'baseline_origin': 'authored_task_material' if source else 'chosen_simulator_prior_not_measured',
                    'baseline_note': None if source else 'Explicit author choice: static/dynamic 0.5, restitution 0; canonical unbound native values remain unknown.'}
            if not parameters['canonical']:
                binding = UsdShade.MaterialBindingAPI.Apply(collider)
                binding.Bind(UsdShade.Material(stage.GetPrimAtPath(materials[key]['path'])),
                             bindingStrength=UsdShade.Tokens.strongerThanDescendants, materialPurpose='physics')
            colliders[str(collider.GetPath())] = materials[key]['path']
    return {'body_physics': masses, 'material_physics': {'materials': materials,
             'collider_material_bindings': colliders, 'collision_disabled_paths_preserved':disabled}, 'mass_factor_applied_once': factor}


def apply_appearance(stage, actors, root, parameters, output):
    """Per-mesh visual clones preserve text, transparent surfaces and original images."""
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade, Vt
    from .appearance import write_appearance
    if parameters['canonical']:
        return {'canonical': True, 'meshes': [], 'artifacts': []}
    changed, artifacts, recipes = [], [], {}
    protected = ('label', 'text', 'glyph', 'number', 'tick', 'mark', 'print', 'ink', 'liquid', 'screen', 'display', 'artwork', 'cover')
    for path in actors.values():
        for prim in list(Usd.PrimRange(stage.GetPrimAtPath(path))):
            if not prim.IsA(UsdGeom.Mesh) or any(s in str(prim.GetPath()).lower() for s in protected):
                continue
            if UsdGeom.Imageable(prim).ComputeVisibility() == 'invisible':
                continue
            binding = UsdShade.MaterialBindingAPI(prim); material, _ = binding.ComputeBoundMaterial()
            shader = material.ComputeSurfaceSource()[0] if material else None
            if not shader or shader.GetIdAttr().Get() != 'UsdPreviewSurface':
                continue
            opacity = shader.GetInput('opacity').Get()
            if opacity is not None and opacity < .99:
                continue
            color_input = shader.GetInput('diffuseColor'); color = color_input.Get()
            connection = color_input.GetConnectedSource(); source_image = None
            if connection:
                tex = UsdShade.Shader(connection[0].GetPrim())
                if tex.GetIdAttr().Get() != 'UsdUVTexture':
                    continue
                filename = tex.GetInput('file').Get()
                source_image = filename.resolvedPath or filename.path
            elif color is None:
                continue
            key = str(material.GetPath())
            if key not in recipes:
                index = len(recipes); target = root+'/ParameterizedAppearance/Material_'+str(index)
                artifact = write_appearance(Path(output)/('appearance_'+str(index)), parameters,
                    source_albedo=source_image, preserve_category_color=True,
                    base_color=None if source_image else list(color))
                new_material = UsdShade.Material.Define(stage, target)
                new_shader = UsdShade.Shader.Define(stage, target+'/Surface'); new_shader.CreateIdAttr('UsdPreviewSurface')
                for original in shader.GetInputs():
                    if original.GetBaseName() not in ('diffuseColor', 'roughness'):
                        entry = new_shader.CreateInput(original.GetBaseName(), original.GetTypeName())
                        if original.Get() is not None: entry.Set(original.Get())
                        if original.HasConnectedSource(): entry.GetAttr().SetConnections(original.GetAttr().GetConnections())
                reader = UsdShade.Shader.Define(stage, target+'/UV'); reader.CreateIdAttr('UsdPrimvarReader_float2')
                reader.CreateInput('varname', Sdf.ValueTypeNames.Token).Set('st')
                for label, filename, channel, field in [('Albedo', artifact['path'], 'rgb', 'diffuseColor'),
                        ('Roughness', artifact['roughness_path'], 'r', 'roughness')]:
                    tex = UsdShade.Shader.Define(stage, target+'/'+label); tex.CreateIdAttr('UsdUVTexture')
                    tex.CreateInput('file', Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(filename))
                    tex.CreateInput('sourceColorSpace', Sdf.ValueTypeNames.Token).Set('sRGB' if channel == 'rgb' else 'raw')
                    tex.CreateInput('st', Sdf.ValueTypeNames.Float2).ConnectToSource(reader.ConnectableAPI(), 'result')
                    for axis in ('S', 'T'): tex.CreateInput('wrap'+axis, Sdf.ValueTypeNames.Token).Set('repeat')
                    new_shader.CreateInput(field, Sdf.ValueTypeNames.Color3f if channel == 'rgb' else Sdf.ValueTypeNames.Float).ConnectToSource(tex.ConnectableAPI(), channel)
                new_material.CreateSurfaceOutput().ConnectToSource(new_shader.ConnectableAPI(), 'surface')
                recipes[key] = target; artifacts.append(artifact)
            # Physics-purpose bindings were explicitly authored before this visual binding.
            UsdShade.MaterialBindingAPI.Apply(prim).Bind(UsdShade.Material(stage.GetPrimAtPath(recipes[key])))
            mesh = UsdGeom.Mesh(prim); points = np.asarray(mesh.GetPointsAttr().Get(), dtype=float)
            linear = np.asarray(UsdGeom.XformCache().GetLocalToWorldTransform(prim))[:3, :3]
            points = points@linear
            indices = list(mesh.GetFaceVertexIndicesAttr().Get()); uv = []; offset = 0
            for count in mesh.GetFaceVertexCountsAttr().Get():
                face = points[indices[offset:offset+count]]; offset += count
                normal = np.cross(face[1]-face[0], face[2]-face[0])
                axes = [i for i in range(3) if i != int(np.argmax(abs(normal)))]
                uv.extend((face[:, axes]/parameters['appearance']['pattern_scale_m']).tolist())
            UsdGeom.PrimvarsAPI(prim).CreatePrimvar('st', Sdf.ValueTypeNames.TexCoord2fArray,
                UsdGeom.Tokens.faceVarying).Set(Vt.Vec2fArray.FromNumpy(np.asarray(uv, dtype=np.float32)))
            changed.append(str(prim.GetPath()))
    if not changed:
        raise ValueError('No unprotected visible mesh received parameterized appearance')
    return {'canonical': False, 'meshes': changed, 'artifacts': artifacts,
            'material_clones': recipes, 'pattern_scale_m': parameters['appearance']['pattern_scale_m']}
