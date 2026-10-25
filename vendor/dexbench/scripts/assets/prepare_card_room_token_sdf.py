"""Prepare an unqualified, single-surface SDF candidate for the task0 black token.

CPU OpenUSD authoring only. Exact analytic source dimensions and authored mass,
visuals/material priors are retained; finite polygon and cooked-SDF errors are
separate. No release, task binding, native simulation or original asset changes.
"""
from copy import deepcopy
import argparse
import json
import math
from pathlib import Path
import shutil

import numpy as np

from bench.isaacsim.release import digest, dump
from bench.paths import DATA_ROOT
from bench.simulation.isaacsim_backend.asset_catalog import load_task_asset, load_obj_mesh

SOURCE = Path(f'{DATA_ROOT}/assets/task_parts/board_games/tic_tac_toe_3x3/v3')
ASSET_ID = 'token_black_union_sdf1024'
SDF = {'sdfResolution': 1024, 'sdfSubgridResolution': 0,
       'sdfEnableRemeshing': False, 'sdfTriangleCountReductionFactor': 1.}


def union_surface(segments=512, surface_step_m=None):
    """Closed outward union boundary; shared shoulder, no coincident inner disks."""
    if type(segments) is not int or segments not in (256, 512):
        raise ValueError('Only explicit high-resolution 256/512 circumference candidates')
    if surface_step_m not in (None, .0005):
        raise ValueError('Only original or explicit0.5mm surface subdivision')
    angle = np.arange(segments)*2*math.pi/segments
    profile = [(.028,-.007),(.028,.007),(.020,.007),(.020,.047)]
    if surface_step_m:
        profile = []
        corners = np.array([[0.,-.007],[.028,-.007],[.028,.007],[.020,.007],[.020,.047],[0.,.047]])
        for a,b in zip(corners[:-1], corners[1:]):
            count = int(round(np.linalg.norm(b-a)/surface_step_m))
            profile.extend(tuple(value) for value in np.linspace(a,b,count+1)[1:])
        profile.pop()  # Last point is the top axis; caps have unique centers.
    rings = len(profile)
    vertices = np.array([[r*math.cos(a), r*math.sin(a), z]
        for r,z in profile for a in angle]
        + [[0.,0.,-.007],[0.,0.,.047]])
    faces = []
    for ring in range(rings-1):
        for i in range(segments):
            j=(i+1)%segments
            a,b,c,d=ring*segments+i,ring*segments+j,(ring+1)*segments+j,(ring+1)*segments+i
            faces += [[a,b,c],[a,c,d]]
    for i in range(segments):
        j=(i+1)%segments
        faces += [[rings*segments,j,i],[rings*segments+1,(rings-1)*segments+i,(rings-1)*segments+j]]
    return vertices, np.asarray(faces, dtype=np.int32)


def geometry_review(vertices, faces, segments):
    import trimesh
    mesh=trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    expected_volume=math.pi*(.028**2*.014+.020**2*.040)
    polygon_ratio=math.sin(2*math.pi/segments)/(2*math.pi/segments)
    expected_polygon_volume=expected_volume*polygon_ratio
    if (not mesh.is_watertight or not mesh.is_winding_consistent or mesh.euler_number != 2
            or len(mesh.split(only_watertight=False)) != 1 or not mesh.is_volume
            or not np.allclose(mesh.bounds,[[-.028,-.028,-.007],[.028,.028,.047]],atol=1e-15,rtol=0)
            or not math.isclose(mesh.volume,expected_polygon_volume,rel_tol=1e-12)):
        raise ValueError('The single closed union surface failed topology/volume/extents checks')
    edge_counts=np.bincount(mesh.edges_unique_inverse)
    if not np.all(edge_counts==2):raise ValueError('Nonmanifold surface edge')
    triangles=vertices[faces]
    shoulder=np.all(np.isclose(triangles[:,:,2],.007,atol=1e-15),axis=1)
    # Every planar shoulder triangle lies outside the polygonal handle interior.
    if np.any(np.linalg.norm(triangles[shoulder,:,:2],axis=2)<.020-1e-15):
        raise ValueError('Internal joining disk was retained')
    sag=.028*(1-math.cos(math.pi/segments))
    return {'segments':segments,'vertices':len(vertices),'triangles':len(faces),
        'watertight':True,'outward_consistent':True,'connected_components':1,'euler_number':2,
        'edge_incidence':2,'internal_joining_disks':0,'bounds_m':mesh.bounds.tolist(),
        'analytic_union_volume_m3':expected_volume,'mesh_volume_m3':float(mesh.volume),
        'relative_volume_deficit':1-polygon_ratio,
        'maximum_radial_chord_deficit_m':sag,
        'float32_vertex_error_max_m':float(np.linalg.norm(vertices-vertices.astype(np.float32),axis=1).max()),
        'source_shape_scope':'Dimensions unchanged. Inscribed straight faces approximate analytic circle; radial chord bound is explicit, not exact native equivalence.',
        'native_cooked_geometry_verified':False,'sdf_zero_surface_verified':False}


def validate_original(original):
    parts=original['collision_parts']
    expected=[(.028,.007,[0.,0.,0.]),(.020,.020,[0.,0.,.027])]
    if len(parts)!=2:raise ValueError('Expected two source cylinders')
    for part,(radius,half_height,position) in zip(parts,expected):
        if (part['geometry']!='cylinder' or part['radius_m']!=radius
                or part['half_height_m']!=half_height or part['position_m']!=position
                or part['quaternion_xyzw']!=[0.,0.,0.,1.] or part['scale']!=[1,1,1]):
            raise ValueError('Original analytic collision contract changed')
    if original['friction'] != .65:raise ValueError('Source material prior changed')


def sdf_profile(profile):
    if profile not in ('dense1024', 'dense768', 'dense512', 'sparse1024_float32', 'sparse1536_float32', 'sparse1250_float32_subgrid4', 'sparse1250_float32_subgrid3'):
        raise ValueError('Unsupported explicit SDF storage profile')
    parameters = dict(SDF)
    if profile.startswith('sparse'):
        parameters.update(sdfResolution=int(profile.split('_')[0].removeprefix('sparse')),
                          sdfSubgridResolution=int(profile.rsplit('subgrid',1)[1]) if 'subgrid' in profile else 6,
                          sdfBitsPerSubgridPixel='BitsPerPixel32')
    if profile in ('dense512', 'dense768'):
        parameters['sdfResolution'] = int(profile.removeprefix('dense'))
    identity = 'token_black_union_sdf'+str(parameters['sdfResolution'])
    return parameters, identity + ('_sparse32' if profile.startswith('sparse') else '') + ('_subgrid'+str(parameters['sdfSubgridResolution']) if 'subgrid' in profile else '')


def prepare(output, source=SOURCE, segments=512, *, profile='dense1024', coordinate_scale=1., surface_step_m=None):
    from pxr import Gf,Sdf,Usd,UsdGeom,UsdPhysics,UsdShade
    output,source=Path(output).resolve(),Path(source).resolve()
    sdf_parameters, asset_id = sdf_profile(profile)
    if coordinate_scale not in (1., 1000.):
        raise ValueError('Only explicit unchanged or1000x cooking coordinate scale')
    if coordinate_scale != 1.:
        asset_id += '_conditioned1000'
    if surface_step_m is not None:
        asset_id += '_surface05mm'
    if output.exists():raise FileExistsError('Use a new immutable candidate directory')
    original=load_task_asset('token_black',source,verify_hashes=True)
    validate_original(original)
    vertices,faces=union_surface(segments,surface_step_m)
    review=geometry_review(vertices,faces,segments)
    review['surface_subdivision_step_m'] = surface_step_m
    review['surface_subdivision_scope'] = 'Same512 angular facet planes, same sharp axisymmetric polygon profile; additional coplanar vertices/triangles only.' if surface_step_m else 'Original four-ring triangulation.'
    output.mkdir(parents=True)
    mesh_path=output/'union_surface.obj'
    with mesh_path.open('w') as stream:
        stream.write('# Single closed union surface; exact dimensions, explicit circumferential discretization.\n')
        for v in vertices:stream.write('v '+' '.join(format(float(x),'.17g') for x in v)+'\n')
        for f in faces:stream.write('f '+' '.join(str(int(x)+1) for x in f)+'\n')
    usd_path=output/'token_black.usda'
    stage=Usd.Stage.CreateNew(str(usd_path))
    UsdGeom.SetStageMetersPerUnit(stage,1.);UsdGeom.SetStageUpAxis(stage,'Z')
    body=UsdGeom.Xform.Define(stage,'/Token').GetPrim();stage.SetDefaultPrim(body)
    UsdPhysics.RigidBodyAPI.Apply(body).CreateRigidBodyEnabledAttr(True)
    mass=UsdPhysics.MassAPI.Apply(body)
    mass.CreateMassAttr(original['mass_kg'])
    mass.CreateCenterOfMassAttr(Gf.Vec3f(*original['center_of_mass_m']))
    mass.CreateDiagonalInertiaAttr(Gf.Vec3f(*original['inertia_diag_kg_m2']))
    mass.CreatePrincipalAxesAttr(Gf.Quatf(1.,Gf.Vec3f(0.)))
    def api(prim,name):
        previous=prim.GetMetadata('apiSchemas')
        names=list(previous.GetAppliedItems()) if previous else []
        prim.SetMetadata('apiSchemas',Sdf.TokenListOp.CreateExplicit([*names,name]))
    def attr(prim,name,value):
        kinds={bool:Sdf.ValueTypeNames.Bool,int:Sdf.ValueTypeNames.Int,float:Sdf.ValueTypeNames.Float,str:Sdf.ValueTypeNames.Token}
        prim.CreateAttribute(name,kinds[type(value)],custom=False).Set(value)
    def mesh(path,v,f):
        geometry=UsdGeom.Mesh.Define(stage,path)
        geometry.CreatePointsAttr(v.tolist());geometry.CreateFaceVertexCountsAttr([3]*len(f))
        geometry.CreateFaceVertexIndicesAttr(f.reshape(-1).tolist());geometry.CreateSubdivisionSchemeAttr('none')
        return geometry
    api(body,'PhysxRigidBodyAPI')
    for key,value in {'solverPositionIterationCount':16,'solverVelocityIterationCount':4,
                      'enableSpeculativeCCD':True,'disableGravity':False}.items():attr(body,'physxRigidBody:'+key,value)
    api(body,'PhysxContactReportAPI');attr(body,'physxContactReport:threshold',0.)
    dependencies={str(source/'manifest.json'):digest(source/'manifest.json'),str(source/'LICENSE.txt'):digest(source/'LICENSE.txt')}
    visual_meshes=[]
    for i,part in enumerate(original['visual_parts']):
        if (part['position_m']!=[0,0,0] or part['quaternion_xyzw']!=[0.,0.,0.,1.] or part['scale']!=[1,1,1]):
            raise ValueError('Unexpected source visual transform')
        path=Path(part['mesh_path']);v,f=load_obj_mesh(path);visual_meshes.append((v,f))
        geometry=mesh('/Token/visual_'+str(i).zfill(3),v,f)
        geometry.CreateDisplayColorAttr([part['color']])
        look=UsdShade.Material.Define(stage,'/Token/Looks/material_'+str(i).zfill(3))
        shader=UsdShade.Shader.Define(stage,str(look.GetPath())+'/Surface');shader.CreateIdAttr('UsdPreviewSurface')
        shader.CreateInput('diffuseColor',Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*part['color']))
        shader.CreateInput('roughness',Sdf.ValueTypeNames.Float).Set(part.get('roughness',.35))
        shader.CreateInput('metallic',Sdf.ValueTypeNames.Float).Set(part.get('metallic',.65))
        look.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(),'surface')
        UsdShade.MaterialBindingAPI.Apply(geometry.GetPrim()).Bind(look)
        dependencies[str(path)]=digest(path)
    collision_vertices = vertices*coordinate_scale
    collider=mesh('/Token/collision',collision_vertices,faces).GetPrim()
    if coordinate_scale != 1.:
        UsdGeom.Xformable(collider).AddScaleOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(1./coordinate_scale))
    UsdGeom.Imageable(collider).CreateVisibilityAttr('invisible')
    UsdPhysics.CollisionAPI.Apply(collider).CreateCollisionEnabledAttr(True)
    UsdPhysics.MeshCollisionAPI.Apply(collider).CreateApproximationAttr('sdf')
    api(collider,'PhysxCollisionAPI')
    attr(collider,'physxCollision:contactOffset',.001);attr(collider,'physxCollision:restOffset',0.)
    api(collider,'PhysxSDFMeshCollisionAPI')
    for key,value in sdf_parameters.items():attr(collider,'physxSDFMeshCollision:'+key,value)
    material=UsdShade.Material.Define(stage,'/Token/physics_material')
    physical=UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    physical.CreateStaticFrictionAttr(.65);physical.CreateDynamicFrictionAttr(.65);physical.CreateRestitutionAttr(0.)
    api(material.GetPrim(),'PhysxMaterialAPI');attr(material.GetPrim(),'physxMaterial:frictionCombineMode','average')
    UsdShade.MaterialBindingAPI.Apply(collider).Bind(material,UsdShade.Tokens.weakerThanDescendants,'physics')
    stage.GetRootLayer().Save()
    reopened=Usd.Stage.Open(str(usd_path))
    for path,(v,f) in [('/Token/collision',(collision_vertices,faces)),*[(f'/Token/visual_{i:03d}',value) for i,value in enumerate(visual_meshes)]]:
        actual=UsdGeom.Mesh(reopened.GetPrimAtPath(path))
        if (not np.array_equal(actual.GetPointsAttr().Get(),np.asarray(v,np.float32))
                or not np.array_equal(actual.GetFaceVertexIndicesAttr().Get(),f.reshape(-1))):
            raise ValueError('USD points/triangles changed: '+path)
    scale_matrix = UsdGeom.Xformable(reopened.GetPrimAtPath('/Token/collision')).GetLocalTransformation()
    expected_scale = np.diag([1./coordinate_scale]*3+[1.])
    if not np.array_equal(scale_matrix, expected_scale):
        raise ValueError('Collider physical scale changed')
    for i,part in enumerate(original['visual_parts']):
        shader=UsdShade.Shader(reopened.GetPrimAtPath(f'/Token/Looks/material_{i:03d}/Surface'))
        for key,value in [('diffuseColor',part['color']),('roughness',part.get('roughness',.35)),('metallic',part.get('metallic',.65))]:
            if not np.allclose(shader.GetInput(key).Get(),value,rtol=0,atol=3e-8):raise ValueError('Visual material changed')
    authored={key:reopened.GetPrimAtPath('/Token').GetAttribute('physics:'+key).Get()
              for key in ['mass','centerOfMass','diagonalInertia']}
    for key,source_key in [('mass','mass_kg'),('centerOfMass','center_of_mass_m'),('diagonalInertia','inertia_diag_kg_m2')]:
        if not np.array_equal(np.asarray(authored[key]),np.asarray(original[source_key],np.float32)):
            raise ValueError('Original float32 mass properties changed')
    shutil.copyfile(source/'LICENSE.txt',output/'LICENSE.txt')
    dependencies.update({str(path):digest(path) for path in [mesh_path,usd_path,output/'LICENSE.txt']})
    plan={'native_cooking_required':True,'native_runs':0,'model_calls':0,'physics_accepted':False,
        'checks':['Confirm one cooked triangle-mesh SDF shape per token, unchanged native mass/COM/inertia/materials and16/4/speculativeCCD.',
            'Query cooked SDF zero set against analytic cylinder-union exterior, including both sides of shoulder, base and handle rims, all azimuths; separately compare source mesh.',
            'Proposed geometric gate: <=20 micrometres maximum zero-set error; do not claim equivalence before measurement. Increase resolution or reject if failed.',
            'Reproduce the exact recorded MP/piece relative poses with reset-only state writes, original default16/4/dt/CCD/materials, and inspect raw signed normal impulses/points/normals/separations.',
            'Check canonical mass-weight support and released dynamics; only then consider all3-piece original-joint replay, preserved task stability/2mm guard, and fresh release qualification.'],
        'storage_profile':profile,
        'storage_note':{'dense1024':'Dense1024 has a roughly4GiB bounding grid before padding. The first candidate failed native field calibration; do not reuse without acceptance.',
            'dense512':'Dense512 is roughly0.5GiB before padding; precision and field completeness require native acceptance.',
            'dense768':'Dense768 is roughly1.7GiB before padding; actual allocation and precision require native acceptance.',
            'sparse1536_float32':'Sparse1536/subgrid6 uses32-bit pixels; refined voxel spacing does not imply acceptance before native finite witnesses.',
            'sparse1250_float32_subgrid4':'Explicit1250 avoids installed native1536-to1250 clamp; subgrid4 reduces background cell spacing relative to subgrid6;32-bit pixels and original acceptance gates retained.',
            'sparse1250_float32_subgrid3':'Explicit1250/subgrid3 further reduces background cell spacing relative to subgrid4;32-bit pixels and original acceptance gates retained; query quantization must be measured.',
            'sparse1024_float32':'Sparse1024/subgrid6 explicitly uses32-bit pixels; no16-bit encoding. Near-surface precision still requires native finite-witness acceptance.'}[profile],
        'upstream_storage_references':['https://github.com/NVIDIA-Omniverse/PhysX/discussions/352',
            'https://docs.omniverse.nvidia.com/kit/docs/omni_usd_schema_physics/latest/physxschema/class_physx_schema_physx_s_d_f_mesh_collision_a_p_i.html'],
        'no_geometry_shrink_or_rounded_edges':True,'no_native_equal_shape_claim':True}
    provenance={'schema':'card-room-token-single-sdf-candidate.v1','source_root':str(source),
        'source_asset_id':'token_black','source_files_modified':False,'generator_sha256':digest(__file__),
        'cpu_openusd_version':list(Usd.GetVersion()),'source_collision_parts':deepcopy(original['collision_parts']),
        'geometry':review,'sdf_parameters':sdf_parameters,
        'collision_coordinate_conditioning':{'vertex_multiplier':coordinate_scale,'collider_uniform_scale':1./coordinate_scale,
            'source_mesh_obj_units':'m; unconditioned source geometry retained',
            'effective_float32_vertex_error_max_m':float(np.max(np.linalg.norm(collision_vertices.astype(np.float32)/coordinate_scale-vertices,axis=1))),
            'query_units':'Must be independently calibrated; do not infer native tensor input/output units from authored scale.'},
        'original_physical_parameters':{key:deepcopy(original[key]) for key in
            ['mass_kg','center_of_mass_m','inertia_diag_kg_m2','inertia_method','friction','origin','design_dimensions']},
        'visual_parts_preserved':6,'source_visual_points_triangles_materials_preserved':True,
        'native_validation_plan':plan,'dependencies_sha256':dependencies}
    dump(output/'SOURCE.json',provenance)
    dependencies[str(output/'SOURCE.json')]=digest(output/'SOURCE.json')
    info={'backend':'usd_reference','label':'Task0 black token single closed union SDF candidate',
        'usd_path':str(usd_path),'usd_default_prim':'/Token','source_rigid_body_prim':'/Token',
        'placement_form':'root','source_scale':[1.,1.,1.],
        **{key:deepcopy(original[key]) for key in ['mass_kg','center_of_mass_m','inertia_diag_kg_m2','inertia_method','origin']},
        'material_prior':{'static_friction':.65,'dynamic_friction':.65,'restitution':0.,'friction_combine':'average'},
        'contact_offset_m':.001,'rest_offset_m':0.,'friction':.65,
        'bounds_min_m':review['bounds_m'][0],'bounds_max_m':review['bounds_m'][1],'dimensions_m':[.056,.056,.054],
        'source_dependencies_sha256':dependencies,'runtime_builtin_dependencies':[],'runtime_builtin_files':{},
        'collision_mesh_coordinate_scale':coordinate_scale,
        'collision_parts':[{'usd_prim_path':'/Token/collision','approximation':'sdf','geometry':'usd_sdf',
                            'position_m':[0.,0.,0.],'quaternion_xyzw':[0.,0.,0.,1.]}],
        'visual_parts':[{'usd_reference':'/Token'}],'material_prim_paths':['/Token/physics_material'],
        'qualification':'Unvalidated engineering candidate; static geometry only',
        'physics_assumptions':{'surface':review['source_shape_scope'],'sdf':sdf_parameters,
            'mass_and_inertia':'Exact original analytic two-cylinder union values; not recomputed from polygon approximation.',
            'scope':'Candidate only; original asset/task/release not modified.'}}
    dump(output/'manifest.json',{'schema':'collaborator_native_usd_catalog.v1','assets':{asset_id:info},
         'license':'CC0-1.0','qualification':info['qualification']})
    load_task_asset(asset_id,output,verify_hashes=True)
    return {'asset_root':str(output),'asset_id':asset_id,'manifest_sha256':digest(output/'manifest.json'),
            'geometry':review,'native_geometry_validated':False,'physics_accepted':False}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--source',type=Path,default=SOURCE)
    parser.add_argument('--segments',type=int,default=512,choices=(256,512))
    parser.add_argument('--profile',choices=('dense1024','dense768','dense512','sparse1024_float32','sparse1536_float32','sparse1250_float32_subgrid4','sparse1250_float32_subgrid3'),default='dense1024')
    parser.add_argument('--coordinate-scale',type=float,choices=(1.,1000.),default=1.)
    parser.add_argument('--surface-step-m',type=float,choices=(.0005,))
    args=parser.parse_args()
    print(json.dumps(prepare(args.output,args.source,args.segments,profile=args.profile,coordinate_scale=args.coordinate_scale,surface_step_m=args.surface_step_m),indent=2))
