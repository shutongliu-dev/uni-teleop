"""Author a separate single-SDF1024 catalog from the audited L5 solid union.

CPU OpenUSD only; original visuals, mass tensor, material and offsets remain.
This does not alter a task binding, execute physics or publish qualification.
"""
from copy import deepcopy
import argparse
import json
from pathlib import Path
import shutil

import numpy as np

from bench.isaacsim.release import digest, dump
from bench.simulation.isaacsim_backend.asset_catalog import load_task_asset
from scripts.validation.gear_union_geometry import ASSET_ROOT, write_candidate

GEARS = ('input_gear', 'idler_gear', 'output_gear')
MATERIAL = {'static_friction': .3, 'dynamic_friction': .3,
            'restitution': 0., 'friction_combine': 'average'}
SDF_PARAMETERS = {'sdfResolution': 1024, 'sdfSubgridResolution': 6,
    'sdfBitsPerSubgridPixel': 'BitsPerPixel16', 'sdfEnableRemeshing': False,
    'sdfTriangleCountReductionFactor': 1.}


def prepare(output):
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade
    import trimesh
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    surface = write_candidate(output/'surface')
    assets, audits = {}, {}

    def api(prim, name):
        previous = prim.GetMetadata('apiSchemas')
        names = list(previous.GetAppliedItems()) if previous else []
        prim.SetMetadata('apiSchemas', Sdf.TokenListOp.CreateExplicit([*names, name]))

    def attr(prim, name, value):
        kind = {bool: Sdf.ValueTypeNames.Bool, int: Sdf.ValueTypeNames.Int,
            float: Sdf.ValueTypeNames.Float, str: Sdf.ValueTypeNames.Token}[type(value)]
        prim.CreateAttribute(name, kind, custom=False).Set(value)

    def usd_mesh(stage, path, mesh):
        result = UsdGeom.Mesh.Define(stage, path)
        result.CreatePointsAttr(mesh.vertices.tolist())
        result.CreateFaceVertexCountsAttr([3]*len(mesh.faces))
        result.CreateFaceVertexIndicesAttr(mesh.faces.reshape(-1).tolist())
        result.CreateSubdivisionSchemeAttr('none')
        return result

    for name in GEARS:
        original = load_task_asset(name, ASSET_ROOT, verify_hashes=True)
        if original['friction'] != MATERIAL['static_friction']:
            raise ValueError('Candidate material must preserve the original task prior')
        mesh_path = Path(surface['assets'][name]['mesh_path'])
        mesh = trimesh.load(mesh_path, force='mesh', process=False)
        usd_path = output/(name+'.usda')
        stage = Usd.Stage.CreateNew(str(usd_path))
        UsdGeom.SetStageMetersPerUnit(stage, 1.)
        UsdGeom.SetStageUpAxis(stage, 'Z')
        body = UsdGeom.Xform.Define(stage, '/Gear').GetPrim()
        stage.SetDefaultPrim(body)
        UsdPhysics.RigidBodyAPI.Apply(body).CreateRigidBodyEnabledAttr(True)
        mass = UsdPhysics.MassAPI.Apply(body)
        mass.CreateMassAttr(original['mass_kg'])
        mass.CreateCenterOfMassAttr(Gf.Vec3f(*original['center_of_mass_m']))
        mass.CreateDiagonalInertiaAttr(Gf.Vec3f(*original['inertia_diag_kg_m2']))
        quaternion = original['principal_axes_xyzw']
        mass.CreatePrincipalAxesAttr(Gf.Quatf(quaternion[3], Gf.Vec3f(*quaternion[:3])))
        api(body, 'PhysxRigidBodyAPI')
        for key, value in {'solverPositionIterationCount': 16, 'solverVelocityIterationCount': 4,
                'enableSpeculativeCCD': True, 'disableGravity': False}.items():
            attr(body, 'physxRigidBody:'+key, value)
        api(body, 'PhysxContactReportAPI')
        attr(body, 'physxContactReport:threshold', 0.)
        dependencies = deepcopy(surface['assets'][name]['source']['source_files_sha256'])
        visual_meshes = []
        for index, part in enumerate(original['visual_parts']):
            path = Path(part['mesh_path'])
            visual = trimesh.load(path, force='mesh', process=False)
            visual_meshes.append(visual)
            if (part.get('position_m', [0., 0., 0.]) != [0., 0., 0.]
                    or part.get('scale', [1, 1, 1]) != [1, 1, 1]
                    or part.get('quaternion_xyzw', [0., 0., 0., 1.]) != [0., 0., 0., 1.]):
                raise ValueError('Original visual transform must be identity')
            geometry = usd_mesh(stage, '/Gear/visual_'+str(index).zfill(3), visual)
            geometry.CreateDisplayColorAttr([part['color']])
            look = UsdShade.Material.Define(stage, '/Gear/Looks/material_'+str(index).zfill(3))
            shader = UsdShade.Shader.Define(stage, str(look.GetPath())+'/Surface')
            shader.CreateIdAttr('UsdPreviewSurface')
            shader.CreateInput('diffuseColor', Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*part['color']))
            shader.CreateInput('roughness', Sdf.ValueTypeNames.Float).Set(part.get('roughness', .35))
            shader.CreateInput('metallic', Sdf.ValueTypeNames.Float).Set(part.get('metallic', .65))
            look.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), 'surface')
            UsdShade.MaterialBindingAPI.Apply(geometry.GetPrim()).Bind(look)
            dependencies[str(path)] = digest(path)
        geometry = usd_mesh(stage, '/Gear/collision', mesh)
        geometry.CreateVisibilityAttr('invisible')
        collider = geometry.GetPrim()
        UsdPhysics.CollisionAPI.Apply(collider).CreateCollisionEnabledAttr(True)
        UsdPhysics.MeshCollisionAPI.Apply(collider).CreateApproximationAttr('sdf')
        api(collider, 'PhysxCollisionAPI')
        for key, value in {'contactOffset': .0001, 'restOffset': 0.}.items():
            attr(collider, 'physxCollision:'+key, value)
        api(collider, 'PhysxSDFMeshCollisionAPI')
        for key, value in SDF_PARAMETERS.items():
            attr(collider, 'physxSDFMeshCollision:'+key, value)
        material = UsdShade.Material.Define(stage, '/Gear/physics_material')
        physical = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
        physical.CreateStaticFrictionAttr(.3)
        physical.CreateDynamicFrictionAttr(.3)
        physical.CreateRestitutionAttr(0.)
        api(material.GetPrim(), 'PhysxMaterialAPI')
        attr(material.GetPrim(), 'physxMaterial:frictionCombineMode', 'average')
        UsdShade.MaterialBindingAPI.Apply(collider).Bind(material, UsdShade.Tokens.weakerThanDescendants, 'physics')
        stage.GetRootLayer().Save()
        reopened = Usd.Stage.Open(str(usd_path))
        for path, expected in [('/Gear/collision', mesh), *[('/Gear/visual_'+str(i).zfill(3), v) for i, v in enumerate(visual_meshes)]]:
            actual = UsdGeom.Mesh(reopened.GetPrimAtPath(path))
            if (not np.array_equal(actual.GetPointsAttr().Get(), np.asarray(expected.vertices, np.float32))
                    or not np.array_equal(actual.GetFaceVertexIndicesAttr().Get(), expected.faces.reshape(-1))):
                raise ValueError('USD source mesh changed: '+path)
        for i, part in enumerate(original['visual_parts']):
            prim = reopened.GetPrimAtPath('/Gear/visual_'+str(i).zfill(3))
            look, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
            shader = UsdShade.Shader(reopened.GetPrimAtPath(str(look.GetPath())+'/Surface'))
            for key, expected in [('diffuseColor', part['color']), ('roughness', part.get('roughness', .35)),
                                  ('metallic', part.get('metallic', .65))]:
                if not np.allclose(shader.GetInput(key).Get(), expected, rtol=0., atol=3.e-8):
                    raise ValueError('Original visual material changed: '+key)
        for path in (mesh_path, output/'surface/cpu_geometry.json', ASSET_ROOT/'SOURCE.json',
                     ASSET_ROOT/'LICENSE.txt', usd_path):
            dependencies[str(path)] = digest(path)
        dependencies.update(surface['source_license_provenance_sha256'])
        assets[name+'_sdf1024'] = {'backend': 'usd_reference', 'label': name+'_single_sdf1024_candidate',
            'usd_path': str(usd_path), 'usd_default_prim': '/Gear', 'source_rigid_body_prim': '/Gear',
            'placement_form': 'root', 'source_scale': [1., 1., 1.],
            **{key: deepcopy(original[key]) for key in ('mass_kg', 'center_of_mass_m',
                'inertia_diag_kg_m2', 'principal_axes_xyzw', 'inertia_body_frame_kg_m2', 'inertia_method')},
            'material_prior': deepcopy(MATERIAL), 'contact_offset_m': .0001, 'rest_offset_m': 0.,
            'bounds_min_m': mesh.bounds[0].tolist(), 'bounds_max_m': mesh.bounds[1].tolist(),
            'dimensions_m': mesh.extents.tolist(), 'source_dependencies_sha256': dependencies,
            'runtime_builtin_dependencies': [], 'runtime_builtin_files': {},
            'collision_parts': [{'usd_prim_path': '/Gear/collision', 'approximation': 'sdf',
                'geometry': 'usd_sdf', 'position_m': [0., 0., 0.], 'quaternion_xyzw': [0., 0., 0., 1.]}],
            'visual_parts': [{'usd_reference': '/Gear'}], 'material_prim_paths': ['/Gear/physics_material'],
            'source_modified': True, 'qualification': 'Independent unvalidated engineering candidate',
            'physics_assumptions': {'surface': 'Exact original physical solid union with internal seams removed and paint-only marker excluded; source OBJ precision bound recorded.',
                'mass_and_inertia': original['inertia_method'], 'material': deepcopy(MATERIAL),
                'scope': 'Single SDF1024 candidate; finite native geometry and dynamics remain unvalidated.'}}
        audits[name] = {'surface': surface['assets'][name], 'usd_path': str(usd_path),
            'usd_sha256': digest(usd_path), 'source_visual_mesh_count': len(visual_meshes),
            'usd_visual_points_and_triangles_unchanged': True,
            'usd_visual_color_roughness_metallic_unchanged': True,
            'float32_collision_vertex_error_m': float(np.linalg.norm(mesh.vertices-np.asarray(mesh.vertices, np.float32), axis=1).max()),
            'principal_axes_xyzw': deepcopy(original['principal_axes_xyzw'])}
    for source, target in [('LICENSE.txt', 'LICENSE.txt'), ('SOURCE.json', 'ORIGINAL_SOURCE.json')]:
        shutil.copyfile(ASSET_ROOT/source, output/target)
    provenance = {'schema': 'gear_single_sdf_candidate.v1', 'source_root': str(ASSET_ROOT),
        'source_manifest_sha256': digest(ASSET_ROOT/'manifest.json'), 'assets': audits,
        'generator_sha256': digest(__file__), 'sdf_parameters': SDF_PARAMETERS,
        'native_geometry_validated': False, 'physics_accepted': False, 'task_success_claimed': False}
    dump(output/'SOURCE.json', provenance)
    for asset in assets.values():
        for path in (output/'SOURCE.json', output/'LICENSE.txt', output/'ORIGINAL_SOURCE.json'):
            asset['source_dependencies_sha256'][str(path)] = digest(path)
    dump(output/'manifest.json', {'schema': 'collaborator_native_usd_catalog.v1', 'assets': assets,
        'license': 'CC0-1.0', 'qualification': 'Independent unvalidated engineering candidate'})
    return {'asset_root': str(output), 'manifest_sha256': digest(output/'manifest.json'),
        'native_geometry_validated': False, 'physics_accepted': False}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.output)))
