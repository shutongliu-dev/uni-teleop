"""Explicit local material and appearance authoring for parameterized v1.

Only the caller's anonymous task stage changes. Native material readback and
both robot configurations' workspace checks remain separate requirements.
"""
from copy import deepcopy
from pathlib import Path

import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, Vt

VERSION = 'parameterized_physics_v1'
ROOT = '/World/LivingTask'
MATERIALS = {
    'remote_operation': ('Looks/Graphite', 'Looks/Holder'),
    'storage_box': ('Looks/Case', 'Looks/Lid', 'Looks/Tray'),
    'water_pouring': ('Materials/WarmWhite',),
}


def _water_decorative_meshes(stage):
    """Give the original tray cubes explicit UV-capable render-only skins.

    Collision prims/poses remain exact; only their drawn representation changes.
    Cup graduations retain the original unmodified WarmWhite material.
    """
    original = ROOT+'/Materials/WarmWhite'
    selected = ROOT+'/Materials/ParameterizedWarmWhite'
    Sdf.CopySpec(stage.GetRootLayer(), original, stage.GetRootLayer(), selected)
    for prim in Usd.PrimRange(stage.GetPrimAtPath(selected)):
        for attribute in prim.GetAttributes():
            connections = attribute.GetConnections()
            if connections:
                attribute.SetConnections([path.ReplacePrefix(Sdf.Path(original), Sdf.Path(selected))
                                          for path in connections])
    material = UsdShade.Material(stage.GetPrimAtPath(selected))
    for prim in list(stage.Traverse()):
        path = str(prim.GetPath())
        if not (path.startswith(ROOT+'/CatchTray/') or path.startswith(ROOT+'/Kettle/Handle_')):
            continue
        if not prim.IsA(UsdGeom.Gprim):
            continue
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(material)
        if not prim.IsA(UsdGeom.Cube):
            continue
        size = float(UsdGeom.Cube(prim).GetSizeAttr().Get())/2
        points = np.array([[x, y, z] for x in (-size, size) for y in (-size, size) for z in (-size, size)])
        faces = [0,1,3,2, 4,6,7,5, 0,4,5,1, 2,3,7,6, 0,2,6,4, 1,5,7,3]
        mesh = UsdGeom.Mesh.Define(stage, path+'_ParameterizedVisual')
        mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(points.astype(np.float32)))
        mesh.CreateFaceVertexCountsAttr([4]*6); mesh.CreateFaceVertexIndicesAttr(faces)
        mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
        mesh.CreateExtentAttr([Gf.Vec3f(-size), Gf.Vec3f(size)])
        UsdGeom.Xformable(mesh).AddTransformOp().Set(UsdGeom.Xformable(prim).GetLocalTransformation())
        UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)
        UsdGeom.Imageable(prim).CreateVisibilityAttr(UsdGeom.Tokens.invisible)
    return {selected}


def appearance_assets(output, parameters):
    from scenes.workrooms.parameterized.appearance import write_appearance
    return write_appearance(Path(output), parameters, preserve_category_color=False,
                            base_color=[.64, .68, .72])


def apply_appearance(stage, task_id, parameters, artifact):
    """Texture only decorative shell materials; buttons/text/semantic colors stay."""
    from ..seed_variants import file_hash
    if parameters['canonical']:
        return {'id': 'canonical', 'changes': [], 'dependencies_sha256': {}}
    if not artifact or artifact.get('canonical'):
        raise ValueError('Parameterized appearance requires its freshly authored texture')
    files = {str(Path(artifact['path']).resolve()): artifact['sha256'],
             str(Path(artifact['roughness_path']).resolve()): artifact['roughness_sha256']}
    if any(file_hash(path) != digest for path, digest in files.items()):
        raise ValueError('Parameterized texture content differs')
    selected = (_water_decorative_meshes(stage) if task_id == 'water_pouring'
                else {ROOT+'/'+name for name in MATERIALS[task_id]})
    for path in sorted(selected):
        material = UsdShade.Material(stage.GetPrimAtPath(path))
        if not material:
            raise ValueError('Missing explicitly selected decorative material: '+path)
        shader, _, _ = material.ComputeSurfaceSource()
        reader = UsdShade.Shader.Define(stage, path+'/ParameterizedST')
        reader.CreateIdAttr('UsdPrimvarReader_float2')
        reader.CreateInput('varname', Sdf.ValueTypeNames.Token).Set('st')
        reader.CreateOutput('result', Sdf.ValueTypeNames.Float2)
        for label, filename, channel, target in (
                ('Albedo', artifact['path'], 'rgb', 'diffuseColor'),
                ('Roughness', artifact['roughness_path'], 'r', 'roughness')):
            texture = UsdShade.Shader.Define(stage, path+'/Parameterized'+label)
            texture.CreateIdAttr('UsdUVTexture')
            texture.CreateInput('file', Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(str(filename)))
            texture.CreateInput('sourceColorSpace', Sdf.ValueTypeNames.Token).Set('sRGB' if channel == 'rgb' else 'raw')
            for axis in ('S', 'T'):
                texture.CreateInput('wrap'+axis, Sdf.ValueTypeNames.Token).Set('repeat')
            texture.CreateInput('st', Sdf.ValueTypeNames.Float2).ConnectToSource(reader.ConnectableAPI(), 'result')
            texture.CreateOutput(channel, Sdf.ValueTypeNames.Float3 if channel == 'rgb' else Sdf.ValueTypeNames.Float)
            shader.CreateInput(target, Sdf.ValueTypeNames.Color3f if channel == 'rgb' else Sdf.ValueTypeNames.Float).ConnectToSource(texture.ConnectableAPI(), channel)
    scale = float(parameters['appearance']['pattern_scale_m'])
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError('Positive pattern scale required')
    meshes = []
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh):
            continue
        bound, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
        if not bound or str(bound.GetPath()) not in selected:
            continue
        mesh = UsdGeom.Mesh(prim)
        points = np.asarray(mesh.GetPointsAttr().Get())
        # Include baked local scale in the UV metric, including tray cube skins.
        matrix = np.asarray(UsdGeom.Xformable(prim).GetLocalTransformation())
        points = points@matrix[:3, :3]
        indices = np.asarray(mesh.GetFaceVertexIndicesAttr().Get())
        uv, start = [], 0
        for count in mesh.GetFaceVertexCountsAttr().Get():
            face = points[indices[start:start+count]]; start += count
            normal = np.cross(face[1]-face[0], face[2]-face[0])
            axes = [axis for axis in range(3) if axis != int(np.argmax(np.abs(normal)))]
            uv.extend((face[:, axes]/scale).tolist())
        UsdGeom.PrimvarsAPI(prim).CreatePrimvar('st', Sdf.ValueTypeNames.TexCoord2fArray,
            UsdGeom.Tokens.faceVarying).Set(Vt.Vec2fArray.FromNumpy(np.asarray(uv, dtype=np.float32)))
        meshes.append(str(prim.GetPath()))
    return {'id': VERSION, 'changes': sorted(selected), 'meshes': meshes,
            'parameters_sha256': parameters['instance_sha256'], 'artifact': deepcopy(artifact),
            'render_only': True, 'physical_materials_unchanged': True,
            'protected_surfaces': 'All unselected materials, including remote face/buttons, text and semantic item colors.',
            'dependencies_sha256': files}


def apply_physics(stage, metadata, task_id, parameters):
    """Scale mass and inertia once, then bind per-collider local friction."""
    from ..seed_variants import actors
    if parameters['canonical']:
        return {'schema': VERSION, 'canonical': True, 'applied': False}
    factor = float(parameters['physics']['density_factor'])
    friction = float(parameters['physics']['friction_factor'])
    if (factor != parameters['physics']['mass_factor'] or not np.isfinite([factor, friction]).all()
            or factor <= 0 or friction <= 0):
        raise ValueError('Mass/density is one positive factor, not two multiplicative factors')
    masses, colliders, materials = {}, {}, {}
    for name, path in actors(metadata, task_id).items():
        prim = stage.GetPrimAtPath(path)
        mass = UsdPhysics.MassAPI(prim)
        before = float(mass.GetMassAttr().Get())
        mass.CreateMassAttr(before*factor)
        diagonal = mass.GetDiagonalInertiaAttr()
        if diagonal.HasAuthoredValue() and np.all(np.asarray(diagonal.Get()) > 0):
            diagonal.Set(Gf.Vec3f(*(np.asarray(diagonal.Get(), dtype=float)*factor).tolist()))
        density = mass.GetDensityAttr()
        if density.HasAuthoredValue() and float(density.Get()) > 0:
            density.Set(float(density.Get())*factor)
        masses[name] = {'parent_geometry_mass_kg': before, 'mass_kg': float(mass.GetMassAttr().Get()),
                        'density_factor': factor, 'inertia_factor': factor,
                        'inertia_mode': 'explicit_scaled' if diagonal.HasAuthoredValue() else 'native_geometry_computation'}
        for collider in Usd.PrimRange(prim):
            if not collider.HasAPI(UsdPhysics.CollisionAPI):
                continue
            binding = UsdShade.MaterialBindingAPI.Apply(collider)
            original, _ = binding.ComputeBoundMaterial(materialPurpose='physics')
            api = UsdPhysics.MaterialAPI(original.GetPrim()) if original else None
            source = str(original.GetPath()) if original and original.GetPrim().HasAPI(UsdPhysics.MaterialAPI) else None
            key = source or 'unbound_parent_simulation_prior'
            if key not in materials:
                # An explicit new-profile simulation prior for an unbound
                # parent, not USD schema defaults or measured native defaults.
                mu_s = float(api.GetStaticFrictionAttr().Get()) if source else .5
                mu_d = float(api.GetDynamicFrictionAttr().Get()) if source else .5
                restitution = float(api.GetRestitutionAttr().Get()) if source else 0.
                material = UsdShade.Material.Define(stage, ROOT+'/ParameterizedPhysics/Material_'+str(len(materials)))
                target = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
                target.CreateStaticFrictionAttr(mu_s*friction)
                target.CreateDynamicFrictionAttr(mu_d*friction)
                target.CreateRestitutionAttr(restitution)
                materials[key] = {'path': str(material.GetPath()), 'parent_material': source,
                    'baseline': {'static_friction': mu_s, 'dynamic_friction': mu_d, 'restitution': restitution},
                    'applied': {'static_friction': float(target.GetStaticFrictionAttr().Get()),
                                'dynamic_friction': float(target.GetDynamicFrictionAttr().Get()),
                                'restitution': restitution},
                    'baseline_origin': 'authored_task_material' if source else 'declared_uncalibrated_simulation_prior_for_unbound_parent'}
            material = UsdShade.Material(stage.GetPrimAtPath(materials[key]['path']))
            binding.Bind(material, bindingStrength=UsdShade.Tokens.strongerThanDescendants, materialPurpose='physics')
            colliders[str(collider.GetPath())] = materials[key]['path']
    if not masses or not colliders:
        raise ValueError('Parameterized physics must change real task bodies and collision materials')
    if task_id == 'remote_operation':
        metadata['remote']['body_mass_kg'] = masses['remote']['mass_kg']
        metadata['remote']['button_mass_kg'] = masses['power']['mass_kg']
        metadata['remote']['total_mass_kg'] = sum(masses[n]['mass_kg'] for n in ('remote', 'power', 'confirm'))
        metadata['holder']['mass_kg'] = masses['holder']['mass_kg']
    report = {'schema': VERSION, 'canonical': False, 'applied': True,
        'parameters': deepcopy(parameters), 'mass_factor_applied_once': factor,
        'masses': masses, 'materials': materials, 'collider_material_bindings': colliders,
        'unchanged': ['solver', 'controller', 'room', 'robot', 'liquid inventory and mass'],
        'native_readback_verified': False, 'material_calibrated': False}
    metadata['parameterized_physics'] = deepcopy(report)
    return report
