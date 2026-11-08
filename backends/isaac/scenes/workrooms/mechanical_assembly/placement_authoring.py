"""CPU USD authoring for the placement-only mechanical seed adapter."""
from copy import deepcopy
from itertools import product
from pathlib import Path
import math

import numpy as np

from scenes.workrooms.seed_geometry import (apply_surface_texture, canonical_hash,
    feature, file_hash, finish_texture, write_json)
from .placement_variants import BELT_SAFE_VERSION, VARIANT_SCOPE, _read, _check, _translate, catalog_body_path


def _author_source_identity():
    import importlib
    from scripts.maintenance.freeze_project_sources import verify_if_snapshot
    required=(__name__,'scenes.workrooms.mechanical_assembly.placement_variants',
        'scenes.workrooms.seed_geometry','bench.simulation.isaacsim_backend.asset_catalog',
        'bench.simulation.isaacsim_backend.presentation')
    modules={name:importlib.import_module(name) for name in required}
    root=Path(__file__).resolve().parents[3]
    proof=verify_if_snapshot(root,required_modules=required,entrypoint_file=Path(__file__))
    if not proof or not proof.get('runtime_imports_verified'):
        raise ValueError('Mechanical seed publication requires a complete frozen source snapshot')
    manifest=root/'source_snapshot_manifest.json'
    return {'snapshot_root':str(root),'manifest_path':str(manifest),'manifest_sha256':file_hash(manifest),
        'source_files_sha256':{str(Path(module.__file__).resolve()):file_hash(module.__file__) for module in modules.values()}}


def _existing_albedo_texture(stage, paths, texture, seed):
    """Retain original machined texture/normal networks; shade RGB pixels only."""
    from PIL import Image
    from pxr import Sdf,Usd,UsdGeom,UsdShade
    changed=[];replacements={}
    with Image.open(texture) as image:pattern=np.asarray(image.convert('RGB'),dtype=float)/255.
    for root in paths:
        for prim in Usd.PrimRange(stage.GetPrimAtPath(root)):
            if not prim.IsA(UsdGeom.Mesh) or UsdGeom.Imageable(prim).ComputeVisibility()=='invisible':continue
            if any(word in str(prim.GetPath()).lower() for word in ('label','number','glyph','print','tick')):continue
            material,_=UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
            shader=material.ComputeSurfaceSource()[0]if material else None
            if not shader or shader.GetIdAttr().Get()!='UsdPreviewSurface':continue
            connection=shader.GetInput('diffuseColor').GetConnectedSource()
            if not connection or connection[1]!='rgb':continue
            source=UsdShade.Shader(connection[0].GetPrim())
            if source.GetIdAttr().Get()!='UsdUVTexture':continue
            key=str(source.GetPath())
            if key not in replacements:
                original=source.GetInput('file').Get()
                path=Path(original.resolvedPath or original.path).resolve()
                with Image.open(path) as image:
                    if image.mode not in ('RGB','RGBA'):raise ValueError('Unsupported original colour texture mode')
                    mode=image.mode;pixels=np.asarray(image).copy()
                h,w=pixels.shape[:2];tile=pattern[np.arange(h)[:,None]%128,np.arange(w)[None,:]%128]
                pixels[...,:3]=np.rint(pixels[...,:3].astype(float)*tile).clip(0,255).astype(np.uint8)
                dest=texture.parent/('albedo_'+file_hash(path)[:16]+'.png')
                Image.fromarray(pixels,mode).save(dest)
                source.GetInput('file').Set(Sdf.AssetPath(str(dest)))
                replacements[key]={'source_path':str(path),'source_sha256':file_hash(path),
                    'path':str(dest),'sha256':file_hash(dest)}
            changed.append(str(prim.GetPath()))
    if not changed:raise ValueError('No existing visible albedo texture was varied')
    return {'schema':'mechanical_existing_albedo_finish.v1','seed':seed,'visual_mesh_paths':changed,
        'only_albedo_changed':True,'original_networks_preserved':True,'albedo_replacements':replacements}


def _rotation(q):
    w,x,y,z = q
    return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
        [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
        [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])


def _primitive_points(kind, size=None, radius=None, height=None, axis='Z'):
    if kind == 'box':
        return np.array(list(product(*[(-v,v) for v in size])),float)
    if kind == 'sphere':
        return np.concatenate([np.eye(3),-np.eye(3)])*radius
    a = 'XYZ'.index(axis); others=[v for v in range(3) if v!=a]
    rows=[]
    for end in (-height/2,height/2):
        for theta in np.linspace(0,2*math.pi,16,endpoint=False):
            p=np.zeros(3);p[a]=end;p[others]=[radius*math.cos(theta),radius*math.sin(theta)];rows.append(p)
    if kind=='capsule':
        for sign in (-1,1):
            p=np.zeros(3);p[a]=sign*(height/2+radius);rows.append(p)
    return np.asarray(rows)


def _usd_points(stage, root_path, *, collision=True):
    from pxr import Usd, UsdGeom, UsdPhysics, Gf
    root=stage.GetPrimAtPath(root_path);cache=UsdGeom.XformCache()
    inverse=cache.GetLocalToWorldTransform(root).GetInverse();points=[]
    for prim in Usd.PrimRange(root):
        if collision and not prim.HasAPI(UsdPhysics.CollisionAPI):continue
        if collision and UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get() is False:continue
        if prim.IsA(UsdGeom.Mesh):p=np.asarray(UsdGeom.Mesh(prim).GetPointsAttr().Get(),float)
        elif prim.IsA(UsdGeom.Cube):p=_primitive_points('box',size=[float(UsdGeom.Cube(prim).GetSizeAttr().Get())/2]*3)
        elif prim.IsA(UsdGeom.Capsule):
            x=UsdGeom.Capsule(prim);p=_primitive_points('capsule',radius=float(x.GetRadiusAttr().Get()),height=float(x.GetHeightAttr().Get()),axis=x.GetAxisAttr().Get())
        elif prim.IsA(UsdGeom.Cylinder):
            x=UsdGeom.Cylinder(prim);p=_primitive_points('cylinder',radius=float(x.GetRadiusAttr().Get()),height=float(x.GetHeightAttr().Get()),axis=x.GetAxisAttr().Get())
        elif prim.IsA(UsdGeom.Sphere):p=_primitive_points('sphere',radius=float(UsdGeom.Sphere(prim).GetRadiusAttr().Get()))
        else:continue
        matrix=cache.GetLocalToWorldTransform(prim)*inverse
        points.extend(list(matrix.Transform(Gf.Vec3d(*v))) for v in p)
    if not points:raise ValueError('No actual collision surface: '+root_path)
    return np.asarray(points,float)


def _world_pose(stage,path):
    from pxr import UsdGeom,Gf
    matrix=UsdGeom.XformCache().GetLocalToWorldTransform(stage.GetPrimAtPath(path))
    transform=Gf.Transform(matrix);q=transform.GetRotation().GetQuat()
    return {'prim_path':path,'position_m':list(transform.GetTranslation()),
            'quaternion_wxyz':[float(q.GetReal()),*map(float,q.GetImaginary())]}


def _surface_texture(stage, paths, texture, seed, *, material_root):
    """Change albedo only; keep every authored non-colour shader input."""
    from pxr import Usd, UsdGeom, UsdShade
    selected=[]; originals={}
    for path in paths:
        for prim in Usd.PrimRange(stage.GetPrimAtPath(path)):
            if not prim.IsA(UsdGeom.Mesh) or UsdGeom.Imageable(prim).ComputeVisibility()=='invisible':
                continue
            material,_=UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
            source=material.ComputeSurfaceSource()[0] if material else None
            if source and (source.GetIdAttr().Get()!='UsdPreviewSurface'
                    or any(i.HasConnectedSource() for i in source.GetInputs())):
                continue  # Existing image finishes/normal networks remain intact.
            if not source and not UsdGeom.Mesh(prim).GetDisplayColorAttr().Get():continue
            inputs=({i.GetBaseName():(i.GetTypeName(),i.Get()) for i in source.GetInputs()
                    if i.GetBaseName()!='diffuseColor'} if source else {})
            selected.append(str(prim.GetPath()));originals[str(prim.GetPath())]=inputs
    if not selected:raise ValueError('No original constant-albedo visible surface')
    result=apply_surface_texture(stage,selected,texture,seed,material_root=material_root)
    if not result['visual_mesh_paths']:raise ValueError('No visible surface received a texture')
    for path in result['visual_mesh_paths']:
        material,_=UsdShade.MaterialBindingAPI(stage.GetPrimAtPath(path)).ComputeBoundMaterial()
        shader=material.ComputeSurfaceSource()[0];inputs=originals[path]
        for current in shader.GetInputs():
            if current.GetBaseName()!='diffuseColor' and current.GetBaseName() not in inputs:
                shader.GetPrim().RemoveProperty(current.GetAttr().GetName())
        for name,(kind,value) in inputs.items():shader.CreateInput(name,kind).Set(value)
        actual={i.GetBaseName():(i.GetTypeName(),i.Get()) for i in shader.GetInputs()
                if i.GetBaseName()!='diffuseColor'}
        if actual!=inputs:raise ValueError('Texture changed a non-albedo shader input')
    result['only_albedo_changed']=True
    return result


def _physics(stage, paths, moving):
    """All source geometry/transform/physics opinions, normalizing allowed XY."""
    from pxr import Usd,UsdGeom,UsdPhysics,UsdShade
    data={}
    for root in paths:
        for prim in Usd.PrimRange(stage.GetPrimAtPath(root)):
            path=str(prim.GetPath());attrs={}
            # Rendering-only shaders/materials contain no geometry or mechanics.
            relevant=(prim.IsA(UsdGeom.Xformable) or prim.IsA(UsdPhysics.Joint)
                      or any('Physics' in s or 'Physx' in s for s in prim.GetAppliedSchemas()))
            if not relevant:continue
            for attr in prim.GetAttributes():
                name=attr.GetName()
                if name.startswith(('physics:','physx','xformOp')) or name in ('points','faceVertexCounts','faceVertexIndices','subdivisionScheme','size','radius','height','axis','extent'):
                    value=attr.Get()
                    if path in moving and name=='xformOp:translate':value=['allowed_xy','allowed_xy',float(value[2])]
                    attrs[name]=repr(value)
            rels={r.GetName():[str(t)for t in r.GetTargets()] for r in prim.GetRelationships() if r.GetName().startswith('physics:')}
            row={'type':prim.GetTypeName(),'schemas':[s for s in prim.GetAppliedSchemas() if 'Physics'in s or 'Physx'in s], 'attrs':attrs,'rels':rels}
            if prim.HasAPI(UsdPhysics.CollisionAPI):
                material,_=UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial('physics')
                row['physical_material']={a.GetName():repr(a.Get()) for a in material.GetPrim().GetAttributes()
                    if a.GetName().startswith(('physics:','physx'))} if material else None
            data[path]=row
    return canonical_hash(data)


def _assert_internal_joints(stage, roots, moved):
    from pxr import Usd,UsdPhysics
    for root in roots:
        for prim in Usd.PrimRange(stage.GetPrimAtPath(root)):
            if not prim.IsA(UsdPhysics.Joint):continue
            j=UsdPhysics.Joint(prim)
            a=[str(x)for x in j.GetBody0Rel().GetTargets()];b=[str(x)for x in j.GetBody1Rel().GetTargets()]
            pair=set(a+b)
            if pair & moved and (len(a)!=1 or len(b)!=1 or not pair<=moved):
                raise ValueError('Translation crosses a fixed/world joint boundary: '+str(prim.GetPath()))


def _catalog_stage(spec):
    """Mirror original visual names for a render-only overlay, without PhysX imports."""
    from pxr import Sdf,Usd,UsdGeom,UsdShade,UsdPhysics,Gf,Vt
    from bench.simulation.isaacsim_backend.asset_catalog import load_task_asset,load_obj_mesh
    asset=load_task_asset(spec['asset_id'],asset_root=spec['asset_root'],verify_hashes=True)
    stage=Usd.Stage.CreateInMemory();root='/World/Objects/'+spec['id']
    prim=UsdGeom.Xform.Define(stage,root).GetPrim()
    if asset.get('backend')=='usd_reference':
        if asset.get('placement_form','root')=='anchored':
            anchor=root+'/'+asset['usd_default_prim'].rsplit('/',1)[-1]
            UsdGeom.Xform.Define(stage,anchor).GetPrim().GetReferences().AddReference(asset['usd_path'],asset['usd_default_prim'])
        else:
            prim.GetReferences().AddReference(asset['usd_path'],asset['usd_default_prim'])
        bodies=[p for p in Usd.PrimRange(prim) if p.HasAPI(UsdPhysics.RigidBodyAPI)]
        if len(bodies)!=1 or str(bodies[0].GetPath())!=catalog_body_path(spec):
            raise ValueError('Pinned catalog body does not match composed original USD')
        points=_usd_points(stage,root)
    else:
        points=[]
        for kind in ('visual','collision'):
            for i,p in enumerate(asset[kind+'_parts']):
                geom=p.get('geometry','convex_mesh');path=f'{root}/{kind}_{i:03d}'
                if geom in ('mesh','convex_mesh','triangle_mesh'):
                    v,f=load_obj_mesh(p['mesh_path']);v=np.asarray(v)*np.asarray(p.get('scale',[1,1,1]))
                elif geom=='box':v=_primitive_points('box',size=p['half_extents_m']);f=[]
                elif geom in ('cylinder','sphere'):v=_primitive_points(geom,radius=p['radius_m'],height=2*p.get('half_height_m',0));f=[]
                else:raise ValueError('Unsupported original geometry '+geom)
                q=p.get('quaternion_xyzw',[0,0,0,1]);v=v@_rotation([q[3],*q[:3]]).T+np.asarray(p.get('position_m',[0,0,0]))
                if kind=='collision':points.extend(v);continue
                # Mesh names match the native loader. Only its material and UV
                # opinions are exported, never these temporary source vertices.
                if geom not in ('mesh','convex_mesh','triangle_mesh'):
                    # Native primitive visuals stay unchanged; avoid replacing
                    # their type merely to manufacture texture coordinates.
                    continue
                mesh=UsdGeom.Mesh.Define(stage,path);mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(v.astype(np.float32)))
                mesh.CreateFaceVertexCountsAttr([3]*len(f));mesh.CreateFaceVertexIndicesAttr(np.asarray(f,dtype=int).reshape(-1).tolist())
                material=UsdShade.Material.Define(stage,root+'/OriginalLooks/m'+str(i))
                shader=UsdShade.Shader.Define(stage,str(material.GetPath())+'/Surface');shader.CreateIdAttr('UsdPreviewSurface')
                shader.CreateInput('diffuseColor',Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*p.get('color',[.55,.60,.65])))
                shader.CreateInput('roughness',Sdf.ValueTypeNames.Float).Set(p.get('roughness',.35))
                shader.CreateInput('metallic',Sdf.ValueTypeNames.Float).Set(p.get('metallic',.65))
                material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(),'surface')
                UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)
        points=np.asarray(points,float)
    return stage,root,asset,points


def _presentation(stage,root,path,texture,seed):
    from pxr import Sdf
    from bench.simulation.isaacsim_backend.presentation import presentation_stage,apply_presentation
    scratch=presentation_stage(stage)
    result=_surface_texture(scratch,[root],texture,seed,material_root=root+'/SeedLooks')
    if not result['visual_mesh_paths']:raise ValueError('No visible surface received a texture: '+root)
    layer=Sdf.Layer.CreateNew(str(path));Sdf.CopySpec(scratch.GetRootLayer(),root,layer,'/Asset');layer.Save()
    descriptor={'schema':'asset_presentation.v1','usd_path':str(path),'sha256':file_hash(path),
        'prim_path':'/Asset','dependencies_sha256':{str(texture):file_hash(texture)}}
    apply_presentation(stage,root,{'presentation':descriptor})
    return descriptor,result


def _pickup(name,path,points,group=None,quaternion=(1.,0.,0.,0.)):
    up=_rotation(quaternion).T@np.array([0.,0.,1.])
    heights=points@up;top=points[heights>=heights.max()-1e-9]
    lateral=top-np.outer(top@up,up)
    p=top[np.argmin(np.sum(lateral*lateral,axis=1))]
    return feature(name+'_pickup',name,path,p.tolist(),group=group or name+'_pickup',direction=up.tolist())


def _goal(label,body,path,world,origin,quaternion=(1.,0.,0.,0.),direction_world=None):
    local=_rotation(quaternion).T@(np.asarray(world)-np.asarray(origin))
    up=_rotation(quaternion).T@np.array([0.,0.,1.] if direction_world is None else direction_world,float)
    return feature(label,body,path,local.tolist(),kind='goal',direction=up.tolist())


def _seat_approach_world(goal):
    """Free approach of a seat whose mating point is declared in the actor frame.

    The target is then that mating point (e.g. the audio pin in its socket), and
    the seated actor's own body lies between it and the actor origin: the actor
    is brought in from that side, along its declared seated axis, never through
    the receptacle wall. Seats without a declared mating point keep the fixture's
    up direction unchanged.
    """
    point=goal.get('actor_local_point_m')
    if not point:return None
    outward=-(_rotation(goal['target_quaternion_wxyz'])@np.asarray(point,float))
    length=float(np.linalg.norm(outward))
    if not math.isfinite(length) or length<=1e-9:raise ValueError('Seat mating point must be off the actor origin')
    return (outward/length).tolist()


def _belt_initial_clearance(stage, root, moving):
    """Verify source solids, not a native cooking/stability qualification.

    Analytic boxes use their eight actual vertices. ConvexHull meshes use their
    authored planes; SDF meshes are accepted only when their AABBs are disjoint.
    The original workplate is the support, not an obstacle to remove or move.
    """
    from pxr import Usd, UsdGeom, UsdPhysics
    from scipy.spatial import ConvexHull
    from scipy.optimize import linprog
    movable=[];fixed=[];support=None
    for prim in Usd.PrimRange(stage.GetPrimAtPath(root)):
        if (not prim.HasAPI(UsdPhysics.CollisionAPI)
                or UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get() is False):
            continue
        path=str(prim.GetPath())
        if prim.IsA(UsdGeom.Mesh):
            points=np.asarray(UsdGeom.Mesh(prim).GetPointsAttr().Get(),float)
            kind=UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get()
            if kind not in ('convexHull','sdf'):raise ValueError('Unsupported belt source collision mesh')
        elif prim.IsA(UsdGeom.Cube):
            points=_primitive_points('box',size=[float(UsdGeom.Cube(prim).GetSizeAttr().Get())/2]*3)
            kind='analytic_box'
        else:raise ValueError('Unsupported belt source collision primitive')
        matrix=np.asarray(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default()),float)
        points=(np.c_[points,np.ones(len(points))]@matrix)[:,:3]
        if path==root+'/Fixtures/workplate/Shape_000':
            if kind!='analytic_box' or support is not None:raise ValueError('Unexpected original belt support')
            support=points;continue
        row=(path,points,kind)
        if any(path.startswith(body+'/')for body in moving):movable.append(row)
        else:fixed.append(row)
    if support is None or not movable or not fixed:raise ValueError('Incomplete belt source collision inventory')
    narrow=[];minimum_aabb_gap=math.inf
    for path,a,kind in movable:
        for other,b,other_kind in fixed:
            gap=float((np.maximum(a.min(0),b.min(0))-np.minimum(a.max(0),b.max(0))).max())
            minimum_aabb_gap=min(minimum_aabb_gap,gap)
            if gap>0:continue
            if 'sdf'in (kind,other_kind):raise ValueError('Belt SDF overlap needs an exact nonconvex check')
            planes=np.r_[ConvexHull(a).equations,ConvexHull(b).equations]
            result=linprog([0,0,0,-1],A_ub=np.c_[planes[:,:3],np.ones(len(planes))],
                b_ub=-planes[:,3],bounds=[(None,None)]*4,method='highs')
            if not result.success or result.x[3]>=-1e-8:
                raise ValueError('Belt initial solid intersects a fixed collider: '+path+' / '+other)
            narrow.append({'moving_collider':path,'fixed_collider':other,
                'common_inscribed_sphere_radius_m':float(result.x[3])})
    points=np.concatenate([a for _,a,_ in movable])
    margin=np.minimum(points[:,:2].min(0)-support[:,:2].min(0),support[:,:2].max(0)-points[:,:2].max(0))
    gap=float(points[:,2].min()-support[:,2].max())
    if np.any(margin<0) or gap < -1e-8:raise ValueError('Belt supplies escape or penetrate the original workplate')
    return {'schema':'mechanical_belt_source_clearance.v1','moving_collider_count':len(movable),
        'fixed_collider_count':len(fixed),'support_collider':root+'/Fixtures/workplate/Shape_000',
        'minimum_pair_aabb_separation_m':minimum_aabb_gap,'narrow_phase':narrow,
        'workplate_xy_margin_m':margin.tolist(),'minimum_original_support_gap_m':gap,
        'scope':'Authored source solid separation only; no native cooking, stability, reachability or task success claim.'}


def author_asset(parent,scene,task,initial,plan,output):
    from pxr import Usd,UsdGeom,Gf
    source_identity=_author_source_identity()
    if output.exists():raise FileExistsError('Never overwrite a seed asset: '+str(output))
    output.mkdir(parents=True)
    seed=plan['seed'];delta=plan['translation_xy_m'];ident=task['id']
    references={str(parent/'release_manifest.json'):file_hash(parent/'release_manifest.json')}
    for name,expected in _read(parent/'release_manifest.json')['files_sha256'].items():references[str(parent/name)]=expected
    references.update(_read(parent/'release_manifest.json').get('asset_files_sha256',{}))
    for p,h in references.items():_check(p,h)
    texture=finish_texture(output,seed) if seed else None
    moved={n for g in plan['groups']for n in g['actor_ids']};rests={n for g in plan['groups']for n in g['fixture_ids']}
    data={'plan':plan,'presentations':{},'actual_body_poses_m':{},'interaction_targets':[],
          'static_geometry':{'bodies':[]},'fixed_support_paths':['/World/Geometry/worktop/Collider']}
    surfaces={};appearance=[]
    if scene.get('native_task_bundle'):
        bundle=scene['native_task_bundle'];source=Usd.Stage.Open(bundle['usd_path'])
        if source is None:raise ValueError('Missing original native bundle')
        stage=Usd.Stage.CreateInMemory();root=bundle['root_prim_path']
        UsdGeom.SetStageMetersPerUnit(stage,UsdGeom.GetStageMetersPerUnit(source))
        UsdGeom.SetStageUpAxis(stage,UsdGeom.GetStageUpAxis(source))
        UsdGeom.Xform.Define(stage,root).GetPrim().GetReferences().AddReference(bundle['usd_path'],root)
        paths=bundle['rigid_bodies'];moving={paths[n]for n in moved}
        before=_physics(stage,[root],moving)
        _assert_internal_joints(stage,[root],moving)
        for name,path in paths.items():
            body=stage.GetPrimAtPath(path);old=_world_pose(stage,path)
            surfaces[name]=_usd_points(stage,path)
            if seed and name in moved:
                attr=body.GetAttribute('xformOp:translate')
                if not attr or len(UsdGeom.Xformable(body).GetOrderedXformOps())!=2:
                    raise ValueError('Expected original translate/orient body ops')
                value=attr.Get();attr.Set(type(value)(value[0]+delta[0],value[1]+delta[1],value[2]))
            actual=_world_pose(stage,path)
            expected=_translate(old['position_m'],delta) if seed and name in moved else old['position_m']
            if not np.allclose(actual['position_m'],expected,atol=1e-14,rtol=0) or actual['quaternion_wxyz']!=old['quaternion_wxyz']:
                raise ValueError('Non-XY authored body change')
            data['actual_body_poses_m'][name]=actual
        if seed and plan['generator_version']==BELT_SAFE_VERSION:
            data['source_initial_clearance']=_belt_initial_clearance(stage,root,moving)
        if seed:
            if ident=='cooperative_insertion':
                appearance.append(_existing_albedo_texture(stage,[paths[n]for n in sorted(moved)],texture,seed))
            else:
                appearance.append(_surface_texture(stage,[paths[n]for n in sorted(moved)],texture,seed,material_root=root+'/SeedLooks'))
        after=_physics(stage,[root],moving)
        if before!=after:raise ValueError('Texture/translation changed protected native physics')
        data.update(protected_physics_before_sha256=before,protected_physics_after_sha256=after)
        if seed:
            usd=output/'task.usda';stage.GetRootLayer().Export(str(usd))
            data['bundle_usd']={'path':str(usd),'sha256':file_hash(usd)}
            data['bundle_dependencies']={str(p):file_hash(p) for p in output.rglob('*')if p.is_file()}
            data['bundle_dependencies'][bundle['usd_path']]=bundle['usd_sha256']
        meta=task['asset_metadata'];objects={o['id']:o for o in meta['objects']}
        fixtures=meta['fixture_metadata']
        data['fixed_support_paths'] += [v['body_path']for n,v in fixtures.items() if 'robot_stand'not in n]
        for name,path in paths.items():
            q=surfaces[name];data['static_geometry']['bodies'].append({'object_id':name,
                'local_bounds_m':[q.min(0).tolist(),q.max(0).tolist()],'nominal_mass_kg':objects[name]['mass_kg']})
        for group in plan['groups']:
            ids=group['actor_ids']
            selected=ids if group['id']=='loose_supply' else (ids[0],ids[len(ids)//2],ids[-1])
            for name in dict.fromkeys(selected):
                data['interaction_targets'].append(_pickup(name,paths[name],surfaces[name],
                    None if group['id']=='loose_supply' else group['id']+'_pickup',
                    data['actual_body_poses_m'][name]['quaternion_wxyz']))
        for goal in meta['task_parameters'].get('seating_goals',[]):
            if goal['actor'] not in moved:continue
            owner=goal['contact_ids'][0] if goal.get('contact_ids') else goal['support_ids'][0]
            fixture=fixtures[owner]
            owner_pose=_world_pose(stage,fixture['body_path'])
            data['interaction_targets'].append(_goal(goal['actor']+'_placement',goal['actor'],fixture['body_path'],goal['target_position_m'],owner_pose['position_m'],owner_pose['quaternion_wxyz'],
                direction_world=_seat_approach_world(goal)))
        for group in meta['task_parameters'].get('flexible_groups',[]):
            workplate=fixtures['workplate'];owner_pose=_world_pose(stage,workplate['body_path'])
            for index,gate in enumerate(group.get('route_gates',group.get('enclosed_targets',[]))):
                target=_goal(group['id']+'_route_'+str(index),group['actor_ids'][0],
                    workplate['body_path'],gate['center_m'],owner_pose['position_m'],owner_pose['quaternion_wxyz'])
                data['interaction_targets'].append(target)
        removal=meta['task_parameters'].get('removal')
        if removal:
            fixture=fixtures[removal['support_ids'][0]];owner_pose=_world_pose(stage,fixture['body_path'])
            data['interaction_targets'].append(_goal('lead_collection',removal['actor_ids'][0],fixture['body_path'],
                removal['center_m'],owner_pose['position_m'],owner_pose['quaternion_wxyz']))
        if ident=='cooperative_insertion':
            fixture=fixtures['collection_tray'];world=meta['task_parameters']['collection_center_m']
            owner_pose=_world_pose(stage,fixture['body_path'])
            data['interaction_targets'].append(_goal('assembly_collection','sleeve',fixture['body_path'],world,owner_pose['position_m'],owner_pose['quaternion_wxyz']))
    else:
        physical={};physical_after={};fixture_map={x['id']:x for x in scene['fixtures']}
        for spec in scene['objects']:
            stage,root,asset,points=_catalog_stage(spec);name=spec['id'];surfaces[name]=points
            physical[name]={'asset':asset,'source_spec':spec,'actual_collision_points':points.tolist(),
                            'usd_physics':_physics(stage,[root],set())}
            if seed:
                descriptor,style=_presentation(stage,root,output/(name+'_surface.usda'),texture,seed)
                data['presentations'][name]=descriptor;appearance.append(style)
            _,_,reread_asset,reread_points=_catalog_stage(spec)
            physical_after[name]={'asset':reread_asset,'source_spec':spec,
                'actual_collision_points':reread_points.tolist(),'usd_physics':_physics(stage,[root],set())}
            p=_translate(spec['position_m'],delta) if seed and name in moved else spec['position_m']
            data['actual_body_poses_m'][name]={'prim_path':catalog_body_path(spec),'position_m':deepcopy(p),'quaternion_wxyz':deepcopy(spec.get('quaternion_wxyz',[1.,0.,0.,0.]))}
            data['static_geometry']['bodies'].append({'object_id':name,'local_bounds_m':[points.min(0).tolist(),points.max(0).tolist()], 'nominal_mass_kg':asset['mass_kg']})
            data['interaction_targets'].append(_pickup(name,root,points,quaternion=spec.get('quaternion_wxyz',[1.,0.,0.,0.])))
        data['protected_physics_before_sha256']=canonical_hash(physical)
        # No catalog, OBJ or source USD bytes are copied or edited. Presentation
        # overlays went through the existing strict runtime preflight above.
        data['protected_physics_after_sha256']=canonical_hash(physical_after)
        if data['protected_physics_before_sha256']!=data['protected_physics_after_sha256']:
            raise ValueError('Presentation changed the actual original geometry/physical opinions')
        data['fixed_support_paths'] += [catalog_body_path(spec,dynamic=False) for n,spec in fixture_map.items() if 'robot_stand'not in n]
        target_name={'peg_insertion':'socket_workplate','shaft_spacer_assembly':'shaft_fixture',
            'fastener_assembly':'threaded_fixture','gear_train_assembly':'gear_workplate'}[ident]
        fixture=fixture_map[target_name]
        goals=task['evaluation'].get('target_bottom_centers_m') or {'bolt':task['evaluation']['fixture_position_m']}
        for name,p in goals.items():
            data['interaction_targets'].append(_goal(name+'_placement',name,'/World/Fixtures/'+target_name,p,fixture['position_m'],fixture.get('quaternion_wxyz',[1.,0.,0.,0.])))
    data['fixed_support_paths']=sorted(set(data['fixed_support_paths']))
    data['appearance']=appearance
    metadata=output/'metadata.json';write_json(metadata,data)
    (output/'ATTRIBUTION.txt').write_text('Original asset geometries and source licenses remain unchanged at their hash-pinned paths.\nNew procedural surface texture: CC0-1.0. No physical material change.\n')
    dependencies={str(p):file_hash(p)for p in output.rglob('*')if p.is_file()}
    manifest={'schema':'workroom_seed_asset.v1','family':'mechanical_assembly','task_id':ident,
        'seed':seed,'generator_version':plan['generator_version'],'variant_scope':VARIANT_SCOPE,'parent_references':references,
        'dependencies_sha256':dependencies,'metadata':str(metadata),'metadata_sha256':file_hash(metadata),
        **{k:deepcopy(data[k])for k in ('actual_body_poses_m','interaction_targets','static_geometry','fixed_support_paths')},
        'license':'Original source licenses retained; new procedural texture CC0-1.0',
        'native_qualified':False,'runtime_object_writes':False}
    manifest['author_source']=source_identity
    path=output/'seed_manifest.json';write_json(path,manifest);return path
