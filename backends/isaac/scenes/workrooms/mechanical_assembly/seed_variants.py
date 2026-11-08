"""Bounded physical seeds for L1 parts sorting, retaining the open SDF ring.

The existing plain OBJ/convex asset catalog and separate native USD ring catalog
remain distinct. No fixture is converted into a different physics backend.
"""
from copy import deepcopy
import json
import math
from pathlib import Path
import shutil

import numpy as np

from bench.paths import load_json, undistributed_bytecode
from scenes.workrooms.seed_geometry import (affine_mass, apply_surface_texture,
    bake_body_scale, check_parent_release, checked_seed, feature, file_hash, finish_texture,
    mass_inventory, planar_offset, write_json)

VERSION='mechanical_parts_sorting_seed_variants.v1'
NAMES=('block','cylinder','ring')


def _read(path):return load_json(Path(path))


def _check(path,sha):
    if undistributed_bytecode(path):return  # pinned bytecode cache, not shipped
    if file_hash(path)!=sha:raise ValueError('Changed source asset '+str(path))


def _scale_obj(path,factor):
    lines=[]
    for line in Path(path).read_text().splitlines():
        if line.startswith('v '):
            xyz=[float(v) for v in line.split()[1:4]];xyz[0]*=factor;xyz[1]*=factor
            line='v '+' '.join(format(v,'.17g') for v in xyz)
        elif line.startswith('vn '):
            xyz=np.asarray([float(v) for v in line.split()[1:4]])/np.array([factor,factor,1.])
            xyz/=np.linalg.norm(xyz);line='vn '+' '.join(format(v,'.17g') for v in xyz)
        lines.append(line)
    Path(path).write_text('\n'.join(lines)+'\n')


def _scale_design(design,factor):
    if not design:return
    for key in ('radius_m','outer_radius_m','inner_radius_m','bore_radius_m'):
        if key in design:design[key]*=factor
    if 'size_m'in design:design['size_m'][:2]=[v*factor for v in design['size_m'][:2]]


def _usd_collision_points(asset):
    from pxr import Gf, Usd, UsdGeom
    stage=Usd.Stage.Open(asset['usd_path']);root=stage.GetPrimAtPath(asset['usd_default_prim'])
    cache=UsdGeom.XformCache();inverse=cache.GetLocalToWorldTransform(root).GetInverse()
    points=[]
    for part in asset['collision_parts']:
        prim=stage.GetPrimAtPath(part['usd_prim_path'])
        matrix=cache.GetLocalToWorldTransform(prim)*inverse
        points.extend(list(matrix.Transform(Gf.Vec3d(*p))) for p in UsdGeom.Mesh(prim).GetPointsAttr().Get())
    return np.unique(points,axis=0)


def _catalog_variant(source,output,factor,seed):
    shutil.copytree(source,output)
    catalog=_read(output/'manifest.json');seen=set()
    scale=np.array([factor,factor,1.])
    for asset in catalog['assets'].values():
        for part in [asset,*asset['visual_parts'],*asset['collision_parts']]:
            if 'mesh_path'in part:
                p=output/part['mesh_path']
                if p not in seen:_scale_obj(p,factor);seen.add(p)
                part['sha256']=file_hash(p)
            if 'aabb_m'in part:part['aabb_m']=(np.asarray(part['aabb_m'])*scale).tolist()
        for part in [*asset['visual_parts'],*asset['collision_parts']]:
            if not np.allclose(part.get('quaternion_xyzw',[0,0,0,1]),[0,0,0,1],atol=0,rtol=0):
                raise ValueError('This authored family expects identity part orientations')
            if 'position_m'in part:part['position_m']=(np.asarray(part['position_m'])*scale).tolist()
        if asset['dynamic']:
            m,c,i=affine_mass(asset['mass_kg'],np.asarray(asset['center_of_mass_m']),np.diag(asset['inertia_diag_kg_m2']),scale)
            if not np.allclose(i,np.diag(np.diag(i)),atol=1e-15,rtol=0):raise ValueError('Unexpected nonprincipal catalog inertia')
            asset.update(mass_kg=m,center_of_mass_m=c.tolist(),inertia_diag_kg_m2=np.diag(i).tolist())
        if 'size_m'in asset:asset['size_m']=(np.asarray(asset['size_m'])*scale).tolist()
        if 'interior_xy_m'in asset:asset['interior_xy_m']=[v*factor for v in asset['interior_xy_m']]
        _scale_design(asset.get('geometry_design'),factor)
        for part in asset['visual_parts']:
            # Preserve the category colours and shape icon; vary only finish.
            part['roughness']=min(.9,max(.1,part.get('roughness',.35)+(-.04, .04, .08)[(seed-1)%3]))
    catalog['seed_geometry']={'schema':'mechanical_parts_sorting_seed_geometry.v1','scale_xyz':scale.tolist(),
        'collision_backend_retained':True,'mass_policy':'Density-preserving affine transform of source explicit moments',
        'native_qualified':False}
    write_json(output/'manifest.json',catalog)
    return catalog


def _ring_variant(source,output,factor,seed):
    from pxr import Usd
    shutil.copytree(source,output)
    catalog=_read(output/'manifest.json');asset=catalog['assets']['sorting_ring_sdf1024']
    stage=Usd.Stage.Open(Usd.Stage.Open(asset['usd_path']).Flatten())
    root=asset['usd_default_prim'];bake_body_scale(stage,root,[factor,factor,1.])
    texture=finish_texture(output,seed)
    appearance=apply_surface_texture(stage,[root],texture,seed,material_root=root+'/SeedVariantMaterials')
    usd=output/'ring.usda';stage.GetRootLayer().Export(str(usd))
    _scale_obj(output/'source.obj',factor)
    scale=np.array([factor,factor,1.])
    mass=mass_inventory(stage,{'ring':root})['ring']
    # Keep exact nominal density-scaled source mass in the catalog; USD uses float32.
    m,c,i=affine_mass(asset['mass_kg'],np.array(asset['center_of_mass_m']),np.diag(asset['inertia_diag_kg_m2']),scale)
    asset.update(usd_path=str(usd),mass_kg=m,center_of_mass_m=c.tolist(),inertia_diag_kg_m2=np.diag(i).tolist(),
        source_modified=True,qualification='New bounded geometry derivative; no native qualification',
        authored_mass_readback=mass)
    for key in ('bounds_min_m','bounds_max_m','dimensions_m'):asset[key]=(np.asarray(asset[key])*scale).tolist()
    _scale_design(asset['geometry_design'],factor)
    asset['source_dependencies_sha256'].update({str(p):file_hash(p) for p in output.rglob('*') if p.is_file() and p.name!='manifest.json'})
    asset['physics_assumptions']['scope']='New geometry derivative retaining original SDF resolution and material/solver opinions; native validation required.'
    write_json(output/'manifest.json',catalog)
    return catalog,appearance


def prepare_asset(parent_release:Path,seed:int,output:Path, *, profile=None, attempt=0)->Path:
    if profile == "parameterized_physics_v1":
        from scenes.workrooms.parameterized.workroom import prepare_asset as prepare_parameterized
        return prepare_parameterized(parent_release, seed, output, profile=profile, attempt=attempt)
    checked_seed(seed);parent=Path(parent_release).resolve();output=Path(output).resolve()
    check_parent_release(parent)
    if output.exists() and any(output.iterdir()):raise FileExistsError('Never overwrite physical seed assets')
    scene=_read(parent/'scene.json');task=_read(parent/'task.json')
    if task['id']!='parts_sorting':raise ValueError('Only mechanical L1 parts_sorting')
    objects={r['id']:r for r in scene['objects']}
    if set(objects)!=set(NAMES):raise ValueError('Unexpected mechanical dynamic inventory')
    mesh_root=Path(objects['block']['asset_root']);ring_root=Path(objects['ring']['asset_root'])
    if objects['ring'].get('usd_physics_policy')!='preserve_authored_v1':raise ValueError('Open ring source-preserving policy required')
    refs={}
    for directory in (parent,mesh_root,ring_root):
        for p in directory.rglob('*'):
            if p.is_file():refs[str(p.resolve())]=file_hash(p)
    ring_original=_read(ring_root/'manifest.json')['assets']['sorting_ring_sdf1024']
    if (ring_original['geometry_design'].get('bore') is not True or
        ring_original['collision_parts'][0]['approximation']!='sdf'):
        raise ValueError('Original open-bore SDF ring required')
    for path,sha in ring_original['source_dependencies_sha256'].items():_check(path,sha);refs[path]=sha
    output.mkdir(parents=True,exist_ok=True);factor=1. if seed==0 else (.96 if seed%2 else 1.04)
    appearance={'variant':'canonical','physical_materials_changed':False}
    mesh_catalog=_read(mesh_root/'manifest.json');ring_catalog=_read(ring_root/'manifest.json')
    for asset in mesh_catalog['assets'].values():
        for part in [asset,*asset['visual_parts'],*asset['collision_parts']]:
            if 'mesh_path' in part:_check(mesh_root/part['mesh_path'],part['sha256'])
    if seed:
        mesh_catalog=_catalog_variant(mesh_root,output/'mesh_catalog',factor,seed);mesh_root=output/'mesh_catalog'
        ring_catalog,appearance=_ring_variant(ring_root,output/'ring_catalog',factor,seed);ring_root=output/'ring_catalog'
    data={'object_updates':{},'fixture_updates':{},'goals':deepcopy(task['goals']),
        'geometry_vertices_local_m':deepcopy(task['evaluation']['geometry_vertices_local_m'])}
    actual=_usd_collision_points(ring_original)
    legacy=np.asarray(task['evaluation']['geometry_vertices_local_m']['ring'])
    data['source_geometry_comparison']={'schema':'canonical_ring_vertex_inventory_comparison.v1',
        'sdf_surface_vertex_count':len(actual),'legacy_evaluator_vertex_count':len(legacy),
        'maximum_sdf_vertex_to_nearest_legacy_vertex_m':float(max(np.linalg.norm(legacy-p,axis=1).min() for p in actual)),
        'maximum_envelope_coordinate_difference_m':float(np.abs(np.array([actual.min(0),actual.max(0)])-np.array([legacy.min(0),legacy.max(0)])).max()),
        'scope':'Vertex-inventory distance, not a surface Hausdorff distance or a native cooking error',
        'seed_zero':'Original evaluator vertices and physical source bytes preserved',
        'nonzero_seeds':'Evaluator ring vertices read from the actual derived SDF mesh'}
    poses={};static={'bodies':[]};targets=[]
    for name,spec in objects.items():
        delta,yaw=planar_offset(seed,'parts_sorting:'+name,xy_m=(.01,.01),yaw_deg=7.)
        if spec['quaternion_wxyz']!=[1.,0.,0.,0.]:raise ValueError('Parent sorting reset orientation changed')
        pos=(np.asarray(spec['position_m'])+[*delta,0.]).tolist();quat=[math.cos(yaw/2),0.,0.,math.sin(yaw/2)]
        update={'position_m':pos,'quaternion_wxyz':quat,'asset_root':str(ring_root if name=='ring' else mesh_root)}
        data['object_updates'][name]=update
        poses[name]={'prim_path':'/World/Objects/'+name,'position_m':pos,'quaternion_wxyz':quat}
        asset=(ring_catalog['assets'][spec['asset_id']] if name=='ring' else mesh_catalog['assets'][spec['asset_id']])
        bounds=([asset['bounds_min_m'],asset['bounds_max_m']] if name=='ring' else asset['aabb_m'])
        static['bodies'].append({'object_id':name,'local_bounds_m':bounds,'nominal_mass_kg':asset['mass_kg']})
        # Annular top point is on material, never the open bore centre.
        point=[.0275*factor if name=='ring' else 0.,0.,bounds[1][2]]
        targets.append(feature(name+'_top',name,poses[name]['prim_path'],point,group=name+'_pickup'))
        for sign in (-1,1):
            targets.append(feature(name+'_side_'+str(sign),name,poses[name]['prim_path'],
                [0.,sign*abs(bounds[sign>0][1]),0.],group=name+'_pickup',direction=(0.,float(sign),0.)))
        goal=data['goals'][name];tray=next(s for s in scene['fixtures'] if s['id']==goal['tray_id'])
        delta,_=planar_offset(seed,'parts_sorting:'+tray['id'],xy_m=(.012,.012),yaw_deg=0.)
        traypos=(np.asarray(tray['position_m'])+[*delta,0.]).tolist()
        data['fixture_updates'][tray['id']]={'asset_root':str(mesh_root),'position_m':traypos}
        goal['center_xy_m']=(np.asarray(goal['center_xy_m'])+delta).tolist()
        goal['inner_half_extents_xy_m']=[v*factor for v in goal['inner_half_extents_xy_m']]
        data['geometry_vertices_local_m'][name]=(np.asarray(data['geometry_vertices_local_m'][name])*[factor,factor,1.]).tolist()
        if name == 'ring' and seed:
            # The legacy evaluator retained vertices from an older convex-sector
            # proxy. New cases freeze the actual selected SDF surface instead.
            data['geometry_vertices_local_m'][name]=_usd_collision_points(asset).tolist()
        targets.append(feature(name+'_place',name,'/World/Fixtures/'+tray['id'],
            [0.,0.,goal['floor_top_z_m']-traypos[2]],kind='goal'))
    data['geometry_variant']={'schema':'mechanical_parts_sorting_seed_geometry.v1','scale_xyz':[factor,factor,1.],
        'ring_sdf_resolution_preserved':1024,'ring_bore_preserved':True,'z_dimensions_unchanged':True,
        'textures_scope':'Original ring render material only; other parts/trays keep semantic colours with varied roughness',
        'mass_policy':'Density-preserving affine transform of source explicit tensor priors','native_qualified':False}
    meta=output/'metadata.json';write_json(meta,data)
    (output/'ATTRIBUTION.txt').write_text('Original mechanical procedural assets and derived geometry/finish: CC0-1.0.\nOriginal source and licenses retained and hash-bound.\nOriginal ring SDF1024 and convex decomposition retained.\n')
    deps={str(p):file_hash(p) for p in output.rglob('*') if p.is_file()}
    manifest={'schema':'workroom_seed_asset.v1','family':'mechanical_assembly','task_id':'parts_sorting',
        'seed':seed,'generator_version':VERSION,'parent_references':refs,'dependencies_sha256':deps,
        'license':'CC0-1.0','metadata':str(meta),'metadata_sha256':file_hash(meta),
        'actual_body_poses_m':poses,'static_geometry':static,'interaction_targets':targets,
        'fixed_support_paths':['/World/Fixtures/tray_'+name for name in NAMES],
        'visual_variant':appearance,'asset_form':'objects_and_fixtures_catalogs','native_qualified':False}
    path=output/'seed_manifest.json';write_json(path,manifest);return path


def compose_case(parent_release,asset_manifest_path,configuration_id):
    from scenes.workrooms.parameterized.workroom import recognizes, compose_case as compose_parameterized
    if recognizes(asset_manifest_path):
        return compose_parameterized(parent_release, asset_manifest_path, configuration_id)
    parent=Path(parent_release).resolve();manifest=_read(asset_manifest_path)
    check_parent_release(parent)
    if manifest['schema']!='workroom_seed_asset.v1' or manifest['family']!='mechanical_assembly':raise ValueError('Wrong seed asset family')
    for path,sha in {**manifest['parent_references'],**manifest['dependencies_sha256']}.items():_check(path,sha)
    scene=_read(parent/'scene.json');task=_read(parent/'task.json');data=_read(manifest['metadata'])
    if task['id']!=manifest['task_id']:raise ValueError('Task and seed asset differ')
    for spec in scene['objects']:
        # Ensure shared seed geometry is derived from the same canonical parent asset.
        path=str((Path(spec['asset_root'])/'manifest.json').resolve())
        if manifest['parent_references'].get(path)!=file_hash(path):raise ValueError('Parent object asset differs')
        spec.update(deepcopy(data['object_updates'][spec['id']]))
    for spec in scene['fixtures']:
        if spec['id'] in data['fixture_updates']:spec.update(deepcopy(data['fixture_updates'][spec['id']]))
    task['episode_seed']=manifest['seed'];scene['scenario']['seed']=manifest['seed']
    task['goals']=deepcopy(data['goals']);task['evaluation']['geometry_vertices_local_m']=deepcopy(data['geometry_vertices_local_m'])
    if manifest['seed']:
        task['evaluation']['geometry_scope']='Actual authored collision vertices: original convex decompositions for block/cylinder and selected complete SDF USD surface for ring; local coordinates with source transforms applied.'
    task['evaluation']['initial_objects']={row['id']:{key:deepcopy(row[key]) for key in ('position_m','quaternion_wxyz')} for row in scene['objects']}
    sources={str(p.relative_to(parent/'evaluator')):p.read_text() for p in (parent/'evaluator').rglob('*.py')}
    return {'scene':scene,'task':task,'initial_state':deepcopy(task['initial_state']),
        'evaluator_sources':sources,'actual_body_poses_m':manifest['actual_body_poses_m'],
        'interaction_targets':manifest['interaction_targets'],'static_geometry':manifest['static_geometry'],
        'fixed_support_paths':manifest['fixed_support_paths'],
        'asset_files_sha256':{**manifest['parent_references'],**manifest['dependencies_sha256'],str(Path(asset_manifest_path).resolve()):file_hash(asset_manifest_path)}}
