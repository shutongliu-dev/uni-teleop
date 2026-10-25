"""Author independent SDF1024 ring candidates from unchanged annular surfaces.

CPU OpenUSD only. This does not replace a task asset, publish a release, or
qualify contact. Mass properties remain the original polygon integrals.
"""
from copy import deepcopy
import argparse
import json
from pathlib import Path
import shutil

import numpy as np

from bench.isaacsim.release import digest, dump
from bench.paths import DATA_ROOT, load_json
from bench.simulation.isaacsim_backend.asset_catalog import load_task_asset


ASSET_ID = 'mechanical_l3_spacer_sdf1024'
SOURCE_ID = 'mechanical_l3_spacer'
SOURCE_IDS = tuple('mechanical_l3_'+name for name in ('spacer', 'wide_ring', 'top_ring'))
DEFAULT_SOURCE = Path(f'{DATA_ROOT}/assets/task_parts/mechanical_assembly/shaft_spacer_assembly/v3')


def prepare(output, source_root=DEFAULT_SOURCE, source_id=SOURCE_ID, *, physics_fallback=None):
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade
    import trimesh
    output, source_root = Path(output).resolve(), Path(source_root).resolve()
    if output.exists():
        raise FileExistsError('An independent new candidate directory is required')
    original = load_task_asset(source_id, source_root, verify_hashes=True)
    fallback = deepcopy(physics_fallback or {})
    if set(fallback)-{'material_prior', 'contact_offset_m', 'rest_offset_m'}:
        raise ValueError('Only missing source material and collision offsets may be supplied')
    for key, value in fallback.items():
        if key in original and original[key] != value:
            raise ValueError('Fallback cannot change an authored physical parameter: '+key)
        original.setdefault(key,value)
    if (fallback and 'friction' in original
            and any(original['material_prior'][key] != original['friction']
                    for key in ('static_friction','dynamic_friction'))):
        raise ValueError('Fallback material differs from the source friction')
    material_prior = original['material_prior']
    mesh_path = Path(original['mesh_path'])
    mesh = trimesh.load(mesh_path, force='mesh', process=False)
    if not mesh.is_watertight or not mesh.is_winding_consistent or mesh.euler_number != 0:
        raise ValueError('Expected the complete closed annular source surface')
    output.mkdir(parents=True)
    source_copy = output/'source.obj'
    shutil.copyfile(mesh_path, source_copy)
    shutil.copyfile(source_root/'LICENSE.txt', output/'LICENSE.txt')
    usd_path = output/'ring.usda'
    stage = Usd.Stage.CreateNew(str(usd_path))
    UsdGeom.SetStageMetersPerUnit(stage, 1.)
    UsdGeom.SetStageUpAxis(stage, 'Z')
    body = UsdGeom.Xform.Define(stage, '/Ring').GetPrim()
    stage.SetDefaultPrim(body)
    UsdPhysics.RigidBodyAPI.Apply(body).CreateRigidBodyEnabledAttr(True)
    mass = UsdPhysics.MassAPI.Apply(body)
    mass.CreateMassAttr(original['mass_kg'])
    mass.CreateCenterOfMassAttr(Gf.Vec3f(*original['center_of_mass_m']))
    mass.CreateDiagonalInertiaAttr(Gf.Vec3f(*original['inertia_diag_kg_m2']))
    mass.CreatePrincipalAxesAttr(Gf.Quatf(1., Gf.Vec3f(0.)))

    def api(prim, name):
        names = list(prim.GetMetadata('apiSchemas').GetAppliedItems()) if prim.GetMetadata('apiSchemas') else []
        prim.SetMetadata('apiSchemas', Sdf.TokenListOp.CreateExplicit([*names, name]))

    def attr(prim, name, value, kind):
        prim.CreateAttribute(name, kind, custom=False).Set(value)

    api(body, 'PhysxRigidBodyAPI')
    for name, value, kind in [('solverPositionIterationCount', 16, Sdf.ValueTypeNames.Int),
                             ('solverVelocityIterationCount', 4, Sdf.ValueTypeNames.Int),
                             ('enableSpeculativeCCD', True, Sdf.ValueTypeNames.Bool),
                             ('disableGravity', False, Sdf.ValueTypeNames.Bool)]:
        attr(body, 'physxRigidBody:'+name, value, kind)
    geometry = UsdGeom.Mesh.Define(stage, '/Ring/collision')
    geometry.CreatePointsAttr(mesh.vertices.tolist())
    geometry.CreateFaceVertexCountsAttr([3]*len(mesh.faces))
    geometry.CreateFaceVertexIndicesAttr(mesh.faces.reshape(-1).tolist())
    geometry.CreateSubdivisionSchemeAttr('none')
    geometry.CreateDisplayColorAttr([original['visual_parts'][0]['color']])
    collider = geometry.GetPrim()
    UsdPhysics.CollisionAPI.Apply(collider).CreateCollisionEnabledAttr(True)
    UsdPhysics.MeshCollisionAPI.Apply(collider).CreateApproximationAttr('sdf')
    api(collider, 'PhysxCollisionAPI')
    attr(collider, 'physxCollision:contactOffset', original['contact_offset_m'], Sdf.ValueTypeNames.Float)
    attr(collider, 'physxCollision:restOffset', original['rest_offset_m'], Sdf.ValueTypeNames.Float)
    api(collider, 'PhysxSDFMeshCollisionAPI')
    for name, value, kind in [('sdfResolution', 1024, Sdf.ValueTypeNames.Int),
            ('sdfSubgridResolution', 6, Sdf.ValueTypeNames.Int),
            ('sdfBitsPerSubgridPixel', 'BitsPerPixel16', Sdf.ValueTypeNames.Token),
            ('sdfEnableRemeshing', False, Sdf.ValueTypeNames.Bool),
            ('sdfTriangleCountReductionFactor', 1., Sdf.ValueTypeNames.Float)]:
        attr(collider, 'physxSDFMeshCollision:'+name, value, kind)
    material = UsdShade.Material.Define(stage, '/Ring/physics_material')
    physical = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    physical.CreateStaticFrictionAttr(material_prior['static_friction'])
    physical.CreateDynamicFrictionAttr(material_prior['dynamic_friction'])
    physical.CreateRestitutionAttr(material_prior['restitution'])
    api(material.GetPrim(), 'PhysxMaterialAPI')
    attr(material.GetPrim(), 'physxMaterial:frictionCombineMode', material_prior['friction_combine'], Sdf.ValueTypeNames.Token)
    UsdShade.MaterialBindingAPI.Apply(collider).Bind(material, UsdShade.Tokens.weakerThanDescendants, 'physics')
    stage.GetRootLayer().Save()
    reopened = Usd.Stage.Open(str(usd_path))
    actual = UsdGeom.Mesh(reopened.GetPrimAtPath('/Ring/collision'))
    if (not np.array_equal(actual.GetPointsAttr().Get(), np.asarray(mesh.vertices, np.float32))
            or not np.array_equal(actual.GetFaceVertexIndicesAttr().Get(), mesh.faces.reshape(-1))):
        raise ValueError('USD source surface changed beyond its required float32 representation')
    dependencies = {str(p): digest(p) for p in (mesh_path, source_copy, source_root/'manifest.json',
        source_root/'SOURCE.json', source_root/'LICENSE.txt', output/'LICENSE.txt', usd_path)}
    info = {'backend': 'usd_reference', 'label': source_id+'_sdf1024_candidate',
        'usd_path': str(usd_path), 'usd_default_prim': '/Ring', 'source_rigid_body_prim': '/Ring',
        'placement_form': 'root', 'source_scale': [1., 1., 1.],
        **{k: deepcopy(original[k]) for k in ('mass_kg', 'center_of_mass_m', 'inertia_diag_kg_m2')},
        **{k: deepcopy(original[k]) for k in ('material_prior', 'contact_offset_m', 'rest_offset_m',
            'geometry_contract', 'geometry_features', 'geometry_design', 'provenance') if k in original},
        'bounds_min_m': mesh.bounds[0].tolist(), 'bounds_max_m': mesh.bounds[1].tolist(),
        'dimensions_m': mesh.extents.tolist(), 'source_dependencies_sha256': dependencies,
        'runtime_builtin_dependencies': [], 'runtime_builtin_files': {},
        'collision_parts': [{'usd_prim_path': '/Ring/collision', 'approximation': 'sdf',
            'geometry': 'usd_sdf', 'position_m': [0.,0.,0.], 'quaternion_xyzw': [0.,0.,0.,1.]}],
        'visual_parts': [{'usd_reference': '/Ring'}], 'material_prim_paths': ['/Ring/physics_material'],
        'source_modified': False, 'qualification': 'Independent unvalidated engineering candidate',
        'physics_assumptions': {'mass_and_inertia': original['inertia_method'],
            'material': {'scope':'Original simulation prior; not measured material calibration.', **material_prior},
            'surface': 'Exact original complete annular OBJ, float32 USD points; no remeshing or triangle reduction.',
            'scope': 'Independent ring SDF candidate; no task/catalog promotion or force-balance qualification.'}}
    audit = {'source_files_sha256': dependencies, 'source_asset_id': source_id,
        'source_vertex_count': len(mesh.vertices), 'source_triangle_count': len(mesh.faces),
        'source_euler_number': mesh.euler_number, 'float32_vertex_error_m': float(np.max(np.linalg.norm(
            mesh.vertices-np.asarray(mesh.vertices, np.float32),axis=1))),
        'source_mesh_path': str(mesh_path), 'preserved_source_mesh_path': str(source_copy),
        'explicit_missing_physics_fallback': fallback,
        'generator_sha256': digest(__file__),
        'native_geometry_tolerance_m': 20e-6, 'resolution': 1024,
        'native_geometry_verified': False, 'contact_verified': False}
    dump(output/'SOURCE.json', audit)
    info['source_dependencies_sha256'][str(output/'SOURCE.json')] = digest(output/'SOURCE.json')
    manifest = {'schema': 'collaborator_native_usd_catalog.v1', 'assets': {source_id+'_sdf1024': info},
        'license': 'CC0-1.0', 'qualification': info['qualification']}
    dump(output/'manifest.json', manifest)
    return {'asset_root': str(output), 'manifest_sha256': digest(output/'manifest.json'), **audit}


def prepare_assembly(output, source_root=DEFAULT_SOURCE):
    """Create one new three-ring catalog; each ring retains its full source OBJ."""
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    assets, audits = {}, {}
    for source_id in SOURCE_IDS:
        audits[source_id] = prepare(output/source_id, source_root, source_id)
        assets.update(load_json(output/source_id/'manifest.json')['assets'])
    dump(output/'manifest.json', {'schema':'collaborator_native_usd_catalog.v1',
        'assets':assets, 'license':'CC0-1.0', 'qualification':'Independent unvalidated engineering candidate'})
    dump(output/'SOURCE.json', {'source_root':str(Path(source_root).resolve()), 'rings':audits,
        'source_geometry_modified':False, 'native_geometry_verified':False, 'contact_verified':False})
    return {'asset_root':str(output), 'manifest_sha256':digest(output/'manifest.json'), 'rings':audits}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--source-root', type=Path, default=DEFAULT_SOURCE)
    parser.add_argument('--all-rings', action='store_true')
    args = parser.parse_args()
    print(json.dumps((prepare_assembly if args.all_rings else prepare)(args.output, args.source_root), indent=2))
