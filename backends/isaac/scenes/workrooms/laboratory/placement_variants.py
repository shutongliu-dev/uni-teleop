"""Frozen world-X placement and render-finish variants for laboratory Task3–6.

Authoring happens before native initialization. No dimensions, collision shapes,
mass properties, joints, liquids, task thresholds or robot controls are changed.
The legacy size-changing Task1/2 author remains separate.
"""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import shutil

import numpy as np

from bench.paths import DATA_ROOT, load_json
from scenes.workrooms.seed_geometry import (canonical_hash, check_parent_release,
    checked_seed, feature, file_hash, mass_inventory, pose, write_json)

TASKS = ('cap_placement', 'solid_weighing', 'pipette_transfer', 'tube_clamp_transfer')
GENERATOR_VERSION = 'laboratory_world_x_surface_texture_v1'
VARIANT_SCOPE = 'initial_xy_translation_and_surface_texture_only'
ROOT = '/World/LaboratoryTask'
ASSET_ROOT = Path(f'{DATA_ROOT}/assets/task_parts/seed_randomization')
SOURCE_FILES = ('scenes/workrooms/laboratory/placement_variants.py',
    'scenes/workrooms/seed_geometry.py', 'scenes/workrooms/living_room/seed_geometry.py')
# A fixed finite design, not a per-seed search for passing outcomes.
OFFSETS_X_M = (0., -.010, -.0075, -.005, -.0025, .0025, .005, .0075, .010, .00125)
GROUPS = {
    'cap_placement': {'closed_bottle': ('sample_bottle', 'cap')},
    'solid_weighing': {
        'source_granules': ('source_tray', *(f'pellet_{i:02}' for i in range(24))),
        'scoop_and_rest': ('scoop', 'scoop_rest'), 'weighing_boat': ('weighing_cup',)},
    'pipette_transfer': {'source_liquid': ('source',),
        'piston_tool_and_cradle': ('pipette', 'plunger', 'pipette_cradle')},
    'tube_clamp_transfer': {'spring_tool_and_rest': ('clamp_frame', 'clamp_jaw', 'tool_rest'),
        'tube_and_supply_holder': ('tube', 'source_holder')},
}


def _read(path):
    return load_json(Path(path))


def _checked(path, expected):
    path = Path(path).resolve(strict=True)
    if file_hash(path) != expected:
        raise ValueError('Frozen laboratory input differs: '+str(path))
    return path


def _seed(task_id, seed):
    checked_seed(seed)
    if task_id not in TASKS:
        raise ValueError('Only laboratory placement Task3 through Task6 are supported')


def plan_case(task_id, seed, metadata):
    _seed(task_id, seed)
    if metadata['task_id'] != task_id:
        raise ValueError('Parent metadata task differs')
    inventory = {**metadata['actors'], **metadata['fixtures']}
    groups = {name: {'object_ids': list(ids), 'prim_paths': [inventory[i] for i in ids]}
              for name, ids in GROUPS[task_id].items()}
    moved = [name for row in groups.values() for name in row['object_ids']]
    if len(set(moved)) != len(moved):
        raise ValueError('Source groups overlap')
    result = {'schema': 'laboratory_placement_plan.v1', 'generator_version': GENERATOR_VERSION,
        'task_id': task_id, 'seed': seed, 'axis': 'world_x',
        'translation_xy_m': [OFFSETS_X_M[seed], 0.], 'yaw_rad': 0.,
        'groups': groups, 'fixed_object_ids': sorted(set(inventory)-set(moved)),
        'parent_metadata_sha256': canonical_hash(metadata),
        'scope': 'Coherent source rows only; Y/Z/yaw, destinations, physical geometry and local joint anchors unchanged'}
    result['plan_sha256'] = canonical_hash(result)
    return result


def shifted_metadata(parent, plan):
    """Update only declared world references; all body-local fields stay exact."""
    expected = plan_case(parent['task_id'], plan['seed'], parent)
    if plan != expected:
        raise ValueError('Unexpected laboratory placement plan')
    result = deepcopy(parent)
    if not plan['seed']:
        return result
    dx = plan['translation_xy_m'][0]
    paths = {p for row in plan['groups'].values() for p in row['prim_paths']}

    def visit(value):
        if isinstance(value, dict):
            # The original evaluator stores some object/fixture rows twice.
            if value.get('body_path') in paths:
                for key in ('initial_position_m', 'initial_pose_xyzw'):
                    if key in value:
                        value[key][0] += dx
            for item in value.values():
                visit(item)
        elif isinstance(value, list):
            for item in value:
                visit(item)
    visit(result)
    p = result['task_parameters']; task = result['task_id']
    if task == 'cap_placement':
        p['bottle_home']['center_m'][0] += dx
        p['thread']['axis_xy_m'][0] += dx
    elif task == 'solid_weighing':
        for corner in p['scoop_return_bounds_m']:
            corner[0] += dx
    elif task == 'pipette_transfer':
        p['tool_return']['initial_pose_xyzw'][0] += dx
    else:
        p['source_position_m'][0] += dx
        p['tool_return_position_m'][0] += dx
        moved_ids = {n for group in GROUPS[task].values() for n in group}
        for row in result.get('interaction_targets', []):
            if row.get('body_id') in moved_ids:
                row['reference_position_m'][0] += dx
    return result


def _stage_values(stage):
    return {str(p.GetPath()): {'type': p.GetTypeName(), 'apis': list(p.GetAppliedSchemas()),
        'attributes': {a.GetName(): (str(a.Get()), tuple(map(str, a.GetConnections())))
                       for a in p.GetAttributes() if a.HasAuthoredValue() or a.GetConnections()},
        'relationships': {r.GetName(): tuple(map(str, r.GetTargets())) for r in p.GetRelationships()}}
        for p in stage.Traverse()}


def assert_translation_only(before, after, plan):
    """Full authored-property guard, including original SDF APIs and all joints."""
    from pxr import UsdGeom
    original, actual = _stage_values(before), _stage_values(after)
    dx = plan['translation_xy_m'][0]
    for path in [p for group in plan['groups'].values() for p in group['prim_paths']]:
        a = UsdGeom.Xformable(before.GetPrimAtPath(path)).GetLocalTransformation()
        b = UsdGeom.Xformable(after.GetPrimAtPath(path)).GetLocalTransformation()
        wanted = np.asarray(a).copy(); wanted[3, 0] += dx
        if not np.allclose(b, wanted, atol=1e-12, rtol=0):
            raise ValueError('Actual source translation differs from its fixed world-X plan')
        for key in ('xformOp:translate', 'xformOpOrder'):
            if key in original[path]['attributes']:
                actual[path]['attributes'][key] = original[path]['attributes'][key]
            else:
                actual[path]['attributes'].pop(key, None)
    if actual != original:
        raise ValueError('Placement changed protected geometry, material, joint, mass or other pose')
    return canonical_hash(original)


def _check_joints(stage, moved_paths):
    from pxr import UsdPhysics
    for prim in stage.Traverse():
        if prim.IsA(UsdPhysics.Joint):
            joint = UsdPhysics.Joint(prim)
            a, b = [list(map(str, rel.GetTargets())) for rel in (joint.GetBody0Rel(), joint.GetBody1Rel())]
            if any(p in moved_paths for p in a+b) and not (len(a) == len(b) == 1 and a[0] in moved_paths and b[0] in moved_paths):
                raise ValueError('Moved mechanism has a fixed/world or unpaired joint endpoint')


def collision_bounds(stage, body_path):
    """Actual enabled collider envelope, in the unchanged body metric frame."""
    from pxr import Gf, Usd, UsdGeom, UsdPhysics
    body = stage.GetPrimAtPath(body_path); cache = UsdGeom.XformCache()
    inverse = cache.GetLocalToWorldTransform(body).GetInverse(); corners = []
    for prim in Usd.PrimRange(body):
        if not prim.HasAPI(UsdPhysics.CollisionAPI) or UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get() is False:
            continue
        matrix = cache.GetLocalToWorldTransform(prim)*inverse
        linear = np.asarray(matrix)[:3, :3].T; center = np.asarray(matrix.ExtractTranslation())
        if prim.IsA(UsdGeom.Mesh):
            vertices = UsdGeom.Mesh(prim).GetPointsAttr().Get()
        elif prim.IsA(UsdGeom.Cube):
            from itertools import product
            h = float(UsdGeom.Cube(prim).GetSizeAttr().Get())/2
            vertices = list(product((-h,h), repeat=3))
        elif prim.IsA(UsdGeom.Sphere):
            extent = float(UsdGeom.Sphere(prim).GetRadiusAttr().Get())*np.linalg.norm(linear, axis=1)
            corners.extend([center-extent, center+extent]); continue
        elif prim.IsA(UsdGeom.Cylinder):
            shape = UsdGeom.Cylinder(prim); axis = {'X':0,'Y':1,'Z':2}[shape.GetAxisAttr().Get()]
            extent = float(shape.GetRadiusAttr().Get())*np.linalg.norm(np.delete(linear,axis,axis=1),axis=1)
            extent += float(shape.GetHeightAttr().Get())/2*np.abs(linear[:,axis])
            corners.extend([center-extent,center+extent]); continue
        else:
            raise ValueError('Unsupported original laboratory collider: '+str(prim.GetPath()))
        corners.extend(np.asarray(matrix.Transform(Gf.Vec3d(*v))) for v in vertices)
    if not corners:
        raise ValueError('Missing enabled body collider')
    return [np.min(corners,axis=0).tolist(), np.max(corners,axis=0).tolist()]


def _surface_witness(stage, body_path, collider_suffix, point):
    """Check the declared point against original source mesh/primitive surface.

    This certifies a static surface reference, not native cooking or a grasp.
    """
    from pxr import Gf, UsdGeom, UsdPhysics
    path = body_path+'/'+collider_suffix; prim = stage.GetPrimAtPath(path)
    if not prim or not prim.HasAPI(UsdPhysics.CollisionAPI) or UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get() is False:
        raise ValueError('Interaction reference must use an original enabled collider')
    cache = UsdGeom.XformCache()
    matrix = cache.GetLocalToWorldTransform(stage.GetPrimAtPath(body_path))*cache.GetLocalToWorldTransform(prim).GetInverse()
    q = np.asarray(matrix.Transform(Gf.Vec3d(*point)))
    tolerance = 2e-6
    if prim.IsA(UsdGeom.Cube):
        h = float(UsdGeom.Cube(prim).GetSizeAttr().Get())/2
        okay = np.all(abs(q)<=h+tolerance) and np.min(abs(abs(q)-h))<=tolerance
    elif prim.IsA(UsdGeom.Cylinder):
        shape = UsdGeom.Cylinder(prim); axis = {'X':0,'Y':1,'Z':2}[shape.GetAxisAttr().Get()]
        h = float(shape.GetHeightAttr().Get())/2; r = float(shape.GetRadiusAttr().Get())
        radius = np.linalg.norm(np.delete(q,axis)); z = abs(q[axis])
        okay = radius<=r+tolerance and z<=h+tolerance and min(abs(radius-r),abs(z-h))<=tolerance
    elif prim.IsA(UsdGeom.Mesh):
        mesh = UsdGeom.Mesh(prim); points = np.asarray(mesh.GetPointsAttr().Get(),dtype=float)
        indices = list(mesh.GetFaceVertexIndicesAttr().Get()); triangles = []; offset = 0
        for count in mesh.GetFaceVertexCountsAttr().Get():
            face = indices[offset:offset+count];offset += count
            triangles.extend([points[[face[0],face[i],face[i+1]]] for i in range(1,count-1)])
        tri = np.asarray(triangles); a = tri[:,1]-tri[:,0]; b = tri[:,2]-tri[:,0]; d = q-tri[:,0]
        aa = np.einsum('ij,ij->i',a,a);bb = np.einsum('ij,ij->i',b,b);ab = np.einsum('ij,ij->i',a,b)
        ad = np.einsum('ij,ij->i',a,d);bd = np.einsum('ij,ij->i',b,d);det = aa*bb-ab*ab
        good = det>1e-24
        u = np.zeros_like(det);v = np.zeros_like(det)
        u[good]=(bb[good]*ad[good]-ab[good]*bd[good])/det[good]
        v[good]=(aa[good]*bd[good]-ab[good]*ad[good])/det[good]
        error = np.linalg.norm(d-u[:,None]*a-v[:,None]*b,axis=1)
        okay = bool(np.any(good & (u>=-1e-5) & (v>=-1e-5) & (u+v<=1+1e-5) & (error<=tolerance)))
    else:
        raise ValueError('Unsupported interaction collider primitive')
    if not okay:
        raise ValueError('Reference is not on its declared original collider: '+path+' '+str(point))
    return path


def interaction_targets(stage, metadata):
    rows = []; actors = metadata['actors']; fixtures = metadata['fixtures']; p = metadata['task_parameters']
    def surface(name,suffix,point,direction,label,group):
        path = actors[name]
        witness = _surface_witness(stage,path,suffix,point)
        row = feature(label,name,path,point,group=group,direction=direction)
        row.update(source_collider_path=witness,semantic_role='direct_robot_surface_reference',
                   source_geometry_scope='Original authored surface; not measured contact or grasp qualification')
        rows.append(row)
    def goal(label,point,group,*,direction=(0.,0.,1.),body=None):
        row = feature(label,body or group,ROOT,point,group=group,direction=direction,kind='goal')
        row['semantic_role'] = 'task_goal_approach_reference'; rows.append(row)
    task = metadata['task_id']
    if task == 'cap_placement':
        for sign in (-1,1):
            surface('sample_bottle','Shape_000',[sign*.022,0.,.03],[sign,0.,0.],
                    'bottle_side_'+str(sign),'bottle_pickup')
        surface('cap','Shape_032',[.025,0.,.014],[0.,0.,1.],'cap_top_edge','cap_grip')
        for sign,part in ((1,'Shape_007'),(-1,'Shape_023')):
            surface('cap',part,[0.,sign*.027,0.],[0.,sign,0.],'cap_side_'+str(sign),'cap_grip')
        goal('bottle_holder_approach',p['holder_center_m'],'bottle_placement')
        goal('cap_tray_approach',p['cap_tray_target']['center_m'],'cap_placement')
        goal('bottle_return',p['bottle_home']['center_m'],'bottle_return')
    elif task == 'solid_weighing':
        for sign,suffix in ((-1,'Collision/side_0'),(1,'Collision/side_2')):
            surface('weighing_cup',suffix,[0.,sign*.0325,.024],[0.,sign,0.],
                    'boat_side_'+str(sign),'boat_pickup')
        surface('scoop','Collision/handle',[-.115,0.,.013],[0.,0.,1.],'scoop_handle_top','scoop_pickup')
        goal('balance_pan',p['weighing_cup_target_position_m'],'boat_placement')
        source = np.asarray(p['source_tray']['initial_position_m']);source[2] += p['source_tray']['floor_top_local_z_m']
        goal('source_granules_tool_access',source.tolist(),'scoop_source_access')
        goal('scoop_return',np.mean(p['scoop_return_bounds_m'],axis=0).tolist(),'scoop_return')
    elif task == 'pipette_transfer':
        for sign in (-1,1):
            surface('pipette','Barrel',[0.,sign*.014,.055],[0.,sign,0.],
                    'pipette_barrel_'+str(sign),'pipette_grip')
        surface('plunger','PullRing',[0.,0.,.1125],[0.,0.,1.],'plunger_pull_ring','plunger_pull')
        surface('plunger','ThumbPad',[0.,.010,.085],[0.,0.,1.],'plunger_press_pad','plunger_press')
        objects = {r['id']:r for r in metadata['objects']}
        for name in ('source','receiver_a','receiver_b'):
            point = np.asarray(objects[name]['initial_position_m']).copy()
            point[2] += p['hydraulics']['containers'][name]['rim_z_m']
            goal(name+'_mouth',point.tolist(),name+'_tool_access',body=name)
        goal('pipette_cradle_return',p['tool_return']['initial_pose_xyzw'][:3],'pipette_return')
    else:
        surface('clamp_frame','Shape_000',[-.08,0.,.024],[0.,0.,1.],'clamp_handle_top','clamp_pickup')
        for sign in (-1,1):
            surface('clamp_frame','Shape_000',[-.08,sign*.018,.012],[0.,sign,0.],
                    'clamp_handle_side_'+str(sign),'clamp_pickup')
        surface('clamp_jaw','Shape_003',[0.,.066,.053],[0.,1.,0.],'jaw_paddle_front','clamp_open')
        surface('clamp_jaw','Shape_003',[0.,.059,.071],[0.,0.,1.],'jaw_paddle_top','clamp_open')
        # Tube is manipulated with the spring tool, never required to be hand-picked.
        source = list(p['source_position_m']);source[2] += p['tube_length_m']/2
        goal('tube_tool_access',source,'tube_tool_access',body='tube')
        goal('receiver_approach',p['receiver_position_m'],'tube_placement',body='tube')
        goal('clamp_return',p['tool_return_position_m'],'clamp_return',body='clamp_frame')
    return rows


def derive_stage(parent_stage, parent_metadata, seed):
    from pxr import Gf, Usd, UsdGeom, UsdPhysics
    plan = plan_case(parent_metadata['task_id'],seed,parent_metadata)
    stage = Usd.Stage.Open(parent_stage.Flatten())
    actors = parent_metadata['actors']
    if {str(p.GetPath()) for p in stage.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI)} != set(actors.values()):
        raise ValueError('Original laboratory rigid actor inventory differs')
    paths = {p for group in plan['groups'].values() for p in group['prim_paths']}
    _check_joints(stage, paths)
    if seed:
        for path in sorted(paths):
            prim = stage.GetPrimAtPath(path); xf = UsdGeom.Xformable(prim)
            parent_matrix = UsdGeom.XformCache().GetLocalToWorldTransform(prim.GetParent())
            if not np.allclose(parent_matrix, np.eye(4), rtol=0, atol=1e-12):
                raise ValueError('World-X author requires original identity task parent')
            ops = xf.GetOrderedXformOps()
            if ops and ops[0].GetOpName() != 'xformOp:translate':
                raise ValueError('Unknown original task transform representation')
            op = ops[0] if ops else xf.AddTranslateOp()
            position = list(op.Get()) if ops else [0.,0.,0.]
            position[0] += plan['translation_xy_m'][0]; op.Set(Gf.Vec3d(*position))
    protected = assert_translation_only(parent_stage,stage,plan)
    metadata = shifted_metadata(parent_metadata,plan)
    poses = {name:pose(stage.GetPrimAtPath(path)) for name,path in actors.items()}
    masses = mass_inventory(stage,actors)
    static = {'bodies':[{'object_id':name, 'local_bounds_m':collision_bounds(stage,path),
        'nominal_mass_kg':next(r['mass_kg'] for r in metadata['objects'] if r['id']==name)} for name,path in actors.items()]}
    supports = []
    for path in metadata['fixtures'].values():
        for prim in Usd.PrimRange(stage.GetPrimAtPath(path)):
            if prim.HasAPI(UsdPhysics.CollisionAPI) and UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get() is not False:
                ancestor = prim
                while ancestor and not ancestor.IsPseudoRoot():
                    if ancestor.HasAPI(UsdPhysics.RigidBodyAPI):
                        raise ValueError('Declared fixed fixture has a dynamic owner')
                    ancestor = ancestor.GetParent()
                supports.append(str(prim.GetPath()))
    return stage, metadata, {'plan':plan, 'protected_parent_properties_sha256':protected,
        'actual_body_poses_m':poses, 'static_geometry':static, 'mass_properties':masses,
        'interaction_targets':interaction_targets(stage,metadata),
        'fixed_support_paths':sorted(supports)}


def _appearance(stage, metadata, seed, output):
    """Clone only render materials; original physics-purpose bindings stay exact."""
    from PIL import Image
    from pxr import Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, Vt
    if not seed:
        return {'id':'canonical','dependencies_sha256':{},'changed_meshes':[]}
    output = Path(output); yy,xx = np.indices((64,64))
    pixels = (172+((xx*17+yy*29+seed*23+(xx*yy)%31)%30)).astype(np.uint8)
    image = output/'surface_roughness.png'; Image.fromarray(pixels).save(image)
    material_root = ROOT+'/PlacementFinish'; UsdGeom.Scope.Define(stage,material_root)
    materials = {}; changed = []; explicit_physics = {}
    for body_path in metadata['actors'].values():
        for prim in list(Usd.PrimRange(stage.GetPrimAtPath(body_path))):
            if not prim.IsA(UsdGeom.Mesh) or any(x in str(prim.GetPath()).lower() for x in
                ('visualdetails','initialliquid','chamberliquid','label','tick','print','mark')):
                continue
            material,_ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
            shader = material.ComputeSurfaceSource()[0] if material else None
            if not shader or shader.GetIdAttr().Get() != 'UsdPreviewSurface':
                continue
            # Preserve transparent liquid/vessel presentation and all diagrams.
            if shader.GetInput('opacity').Get() is not None and shader.GetInput('opacity').Get() < .99:
                continue
            source = str(material.GetPath())
            if source not in materials:
                target = material_root+'/material_'+str(len(materials))
                Sdf.CopySpec(stage.GetRootLayer(),material.GetPath(),stage.GetRootLayer(),target)
                for child in Usd.PrimRange(stage.GetPrimAtPath(target)):
                    # Some original pipette materials serve both render and
                    # physics purposes. The new clone is strictly render-only.
                    schemas = [api for api in child.GetAppliedSchemas() if not api.lower().startswith(('physics','physx'))]
                    if schemas != child.GetAppliedSchemas():
                        child.SetMetadata('apiSchemas',Sdf.TokenListOp.CreateExplicit(schemas))
                    for attr in list(child.GetAttributes()):
                        if attr.GetName().startswith(('physics:','physx')):
                            child.RemoveProperty(attr.GetName())
                    for attr in child.GetAttributes():
                        if attr.GetConnections():
                            attr.SetConnections([p.ReplacePrefix(Sdf.Path(source),Sdf.Path(target)) for p in attr.GetConnections()])
                    for rel in child.GetRelationships():
                        if rel.GetTargets():
                            rel.SetTargets([p.ReplacePrefix(Sdf.Path(source),Sdf.Path(target)) for p in rel.GetTargets()])
                derived = UsdShade.Material(stage.GetPrimAtPath(target)); surface = derived.ComputeSurfaceSource()[0]
                uv = UsdShade.Shader.Define(stage,target+'/FinishUV'); uv.CreateIdAttr('UsdPrimvarReader_float2')
                uv.CreateInput('varname',Sdf.ValueTypeNames.Token).Set('st');uv.CreateOutput('result',Sdf.ValueTypeNames.Float2)
                tex = UsdShade.Shader.Define(stage,target+'/FinishTexture');tex.CreateIdAttr('UsdUVTexture')
                tex.CreateInput('file',Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(str(image)))
                tex.CreateInput('sourceColorSpace',Sdf.ValueTypeNames.Token).Set('raw')
                tex.CreateInput('wrapS',Sdf.ValueTypeNames.Token).Set('repeat');tex.CreateInput('wrapT',Sdf.ValueTypeNames.Token).Set('repeat')
                tex.CreateInput('st',Sdf.ValueTypeNames.Float2).ConnectToSource(uv.ConnectableAPI(),'result')
                tex.CreateOutput('r',Sdf.ValueTypeNames.Float)
                surface.CreateInput('roughness',Sdf.ValueTypeNames.Float).ConnectToSource(tex.ConnectableAPI(),'r')
                materials[source] = target
            points = np.asarray(UsdGeom.Mesh(prim).GetPointsAttr().Get(),dtype=float)
            axes = np.argsort(np.ptp(points,axis=0))[-2:]
            uv = UsdGeom.PrimvarsAPI(prim).CreatePrimvar('st',Sdf.ValueTypeNames.TexCoord2fArray,'vertex')
            uv.Set(Vt.Vec2fArray.FromNumpy((points[:,axes]/.018).astype(np.float32)))
            # Preserve the old all-purpose fallback as explicit physics binding
            # before assigning a different render-only material.
            physical,_ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial(materialPurpose='physics')
            if physical and str(physical.GetPath())==source and physical.GetPrim().HasAPI(UsdPhysics.MaterialAPI):
                UsdShade.MaterialBindingAPI(prim).Bind(physical,materialPurpose='physics')
                explicit_physics[str(prim.GetPath())] = source
            UsdShade.MaterialBindingAPI(prim).Bind(UsdShade.Material(stage.GetPrimAtPath(materials[source])))
            changed.append(str(prim.GetPath()))
    if not changed:
        raise ValueError('No original opaque task surface accepted the texture')
    return {'id':'subtle_roughness_'+str(seed), 'material_root':material_root,
        'material_clones':materials, 'changed_meshes':sorted(changed),
        'explicit_preserved_physics_bindings':explicit_physics,
        'category_colors_and_labels_unchanged':True, 'physical_materials_unchanged':True,
        'dependencies_sha256':{str(image):file_hash(image)}}


def assert_finish_only(before, stage, appearance):
    after = _stage_values(stage)
    new_root = appearance.get('material_root')
    added = set(after)-set(before)
    if set(before)-set(after) or any(not (p==new_root or p.startswith(new_root+'/')) for p in added):
        raise ValueError('Finish changed unrelated prim inventory')
    for path, old in before.items():
        row = deepcopy(after[path])
        if path in appearance['changed_meshes']:
            for key in list(row['attributes']):
                if key.startswith('primvars:st') and key not in old['attributes']:
                    row['attributes'].pop(key)
            for key in old['attributes']:
                if key.startswith('primvars:st'):
                    row['attributes'][key] = old['attributes'][key]
            row['relationships']['material:binding'] = old['relationships']['material:binding']
            if path in appearance.get('explicit_preserved_physics_bindings',{}):
                expected = appearance['explicit_preserved_physics_bindings'][path]
                if (row['relationships'].get('material:binding:physics') != (expected,)
                        or old['relationships'].get('material:binding') != (expected,)):
                    raise ValueError('Finish changed the resolved original physics material')
                if 'material:binding:physics' in old['relationships']:
                    row['relationships']['material:binding:physics'] = old['relationships']['material:binding:physics']
                else:
                    row['relationships'].pop('material:binding:physics',None)
        if row != old:
            raise ValueError('Finish changed protected original geometry, shader, physics or labels')


def _parent(parent_release):
    root = Path(parent_release).resolve(strict=True); manifest = check_parent_release(root)
    if manifest.get('schema') not in ('laboratory_manipulation_task_release.v1','workroom_robot_configuration_release.v1') or len(manifest['tasks']) != 1:
        raise ValueError('Expected one original laboratory release')
    entry = manifest['tasks'][0]; values = {}
    for key in ('scene','task','initial_state'):
        path = (root/entry[key]).resolve(strict=True)
        if not path.is_relative_to(root): raise ValueError('Parent input escapes release')
        values[key] = _read(path)
    _seed(values['task']['id'],0)
    return root, values


def evaluator_sources(parent_release, task_id):
    _seed(task_id,0); source = Path(parent_release)/'evaluator'
    return {str(p.relative_to(source)):p.read_text() for p in sorted(source.rglob('*.py'))}


def _author_provenance(proof):
    root=Path(proof['snapshot_root']).resolve(strict=True)
    manifest=_read(_checked(proof['manifest_path'],proof['manifest_sha256']))
    if (manifest.get('schema')!='complete_benchmark_source_snapshot_v1'
            or Path(manifest['snapshot_root']).resolve()!=root
            or Path(proof['manifest_path']).resolve()!=root/'source_snapshot_manifest.json'):
        raise ValueError('Original laboratory author snapshot identity differs')
    files=proof['source_files_sha256']
    if {str(Path(p).relative_to(root)) for p in files}!=set(SOURCE_FILES):
        raise ValueError('Incomplete laboratory author/helper source identity')
    for path,sha in files.items():
        _checked(path,sha)
        if manifest['files_sha256'].get(str(Path(path).relative_to(root)))!=sha:
            raise ValueError('Author/helper bytes differ from original source snapshot')
    return {**files,proof['manifest_path']:proof['manifest_sha256']}


def _source_provenance(source):
    root=Path(source['snapshot_root']).resolve(strict=True)
    manifest=root/'source_snapshot_manifest.json'
    proof={'snapshot_root':str(root),'manifest_path':str(manifest),'manifest_sha256':file_hash(manifest),
        'source_files_sha256':{str(root/name):file_hash(root/name) for name in SOURCE_FILES}}
    _author_provenance(proof)
    return proof


def prepare_asset(parent_release:Path,seed:int,output:Path, *, profile=None, attempt=0)->Path:
    if profile == "parameterized_physics_v1":
        from scenes.workrooms.parameterized.workroom import prepare_asset as prepare_parameterized
        return prepare_parameterized(parent_release, seed, output, profile=profile, attempt=attempt)
    from pxr import Usd
    from scenes.workrooms import seed_geometry
    from scenes.workrooms.living_room import seed_geometry as usd_geometry
    from scripts.maintenance.freeze_project_sources import verify_if_snapshot
    source = verify_if_snapshot(Path(__file__).resolve().parents[3],
        required_modules=(__name__,seed_geometry.__name__,usd_geometry.__name__),entrypoint_file=Path(__file__))
    if not source or source.get('verified') is not True:
        raise ValueError('Laboratory publication requires complete frozen source')
    author_source=_source_provenance(source)
    parent, values = _parent(parent_release); task = values['task']; _seed(task['id'],seed)
    candidate = values['scene']['laboratory_task_provenance']['asset_candidate']
    refs = {str(parent/'release_manifest.json'):file_hash(parent/'release_manifest.json')}
    for kind in ('manifest','usd','metadata'):
        p = _checked(candidate[kind+'_path'],candidate[kind+'_sha256']);refs[str(p)] = file_hash(p)
    dependencies = _author_provenance(author_source)
    for p, spec in {**candidate.get('dependencies',{}),**candidate.get('auxiliary_files_sha256',{})}.items():
        expected = spec['sha256'] if isinstance(spec,dict) else spec
        dependencies[str(_checked(p,expected))] = expected
    metadata = _read(candidate['metadata_path'])
    if metadata != task['asset_metadata']:
        raise ValueError('Task metadata differs from original asset')
    output = Path(output).resolve()
    if not output.is_relative_to(ASSET_ROOT):
        raise ValueError('Laboratory placement assets require shared versioned storage')
    stage, metadata, observations = derive_stage(Usd.Stage.Open(candidate['usd_path']),metadata,seed)
    output.mkdir(parents=True,exist_ok=False)
    before = _stage_values(stage); appearance = _appearance(stage,metadata,seed,output)
    assert_finish_only(before,stage,appearance)
    usd = output/'task.usda'
    if seed: stage.GetRootLayer().Export(str(usd))
    else: shutil.copyfile(candidate['usd_path'],usd)
    write_json(output/'metadata.json',metadata)
    (output/'ATTRIBUTION.txt').write_text('Original laboratory procedural assets and surface roughness: CC0-1.0.\nOriginal dependency licences and source paths remain hash-bound.\nNo physics, geometry, robot or numerical referee changes.\n')
    write_json(output/'SOURCE.json',{'schema':'laboratory_placement_source.v1','source_snapshot':source,
        'parent_references':refs,'generator_version':GENERATOR_VERSION,'plan':observations['plan'],
        'scope':'Offline authoring only; no native or robot qualification'})
    dependencies.update(refs);dependencies.update(appearance['dependencies_sha256'])
    dependencies.update({str(p):file_hash(p) for p in output.iterdir() if p.is_file()})
    manifest = {'schema':'workroom_seed_asset.v1','family':'laboratory','task_id':task['id'],'seed':seed,
        'generator_version':GENERATOR_VERSION,'variant_scope':VARIANT_SCOPE,'source_snapshot':source,
        'author_source':author_source,
        'parent_references':refs,'dependencies_sha256':dependencies,
        'parent_initial_usd_sha256':candidate['usd_sha256'],'parent_metadata_sha256':candidate['metadata_sha256'],
        'initial_usd':str(usd),'initial_usd_sha256':file_hash(usd),
        'metadata':str(output/'metadata.json'),'metadata_sha256':file_hash(output/'metadata.json'),
        'visual_variant':appearance,**observations,'native_qualified':False,'robot_manipulation_qualified':False}
    write_json(output/'seed_manifest.json',manifest)
    return output/'seed_manifest.json'


def compose_case(parent_release,asset_manifest_path,configuration_id):
    from scenes.workrooms.parameterized.workroom import recognizes, compose_case as compose_parameterized
    if recognizes(asset_manifest_path):
        return compose_parameterized(parent_release, asset_manifest_path, configuration_id)
    parent,values = _parent(parent_release); scene,task = values['scene'],values['task']
    path = Path(asset_manifest_path).resolve(strict=True); manifest = _read(path)
    task_id,seed = manifest['task_id'],manifest['seed']; _seed(task_id,seed)
    if (task_id != task['id'] or manifest.get('schema') != 'workroom_seed_asset.v1'
            or manifest.get('family') != 'laboratory' or manifest.get('generator_version') != GENERATOR_VERSION
            or manifest.get('variant_scope') != VARIANT_SCOPE):
        raise ValueError('Placement asset identity differs')
    if configuration_id not in ('single_arm','dual_same_side','dual_opposed') or scene.get('robot_configuration',{}).get('id','single_arm') != configuration_id:
        raise ValueError('Requested robot configuration differs from parent')
    for p,sha in {**manifest['parent_references'],**manifest['dependencies_sha256']}.items(): _checked(p,sha)
    author_files=_author_provenance(manifest['author_source'])
    if any(manifest['dependencies_sha256'].get(p)!=sha for p,sha in author_files.items()):
        raise ValueError('Asset omitted original author source dependencies')
    _checked(manifest['initial_usd'],manifest['initial_usd_sha256'])
    metadata = _read(_checked(manifest['metadata'],manifest['metadata_sha256']))
    candidate = scene['laboratory_task_provenance']['asset_candidate']; old_metadata = task['asset_metadata']
    plan = plan_case(task_id,seed,old_metadata)
    if (manifest['parent_initial_usd_sha256'] != candidate['usd_sha256']
            or manifest['parent_metadata_sha256'] != candidate['metadata_sha256']
            or manifest['parent_references'].get(candidate['usd_path']) != candidate['usd_sha256']
            or manifest['parent_references'].get(candidate['metadata_path']) != candidate['metadata_sha256']
            or manifest['plan'] != plan or metadata != shifted_metadata(old_metadata,plan)):
        raise ValueError('Placement parent/metadata/plan differs')
    poses = manifest['actual_body_poses_m']; objects = {r['id']:r for r in metadata['objects']}
    if set(poses) != set(metadata['actors']) or {r['object_id'] for r in manifest['static_geometry']['bodies']} != set(poses):
        raise ValueError('Placement actor inventory differs')
    for name,row in poses.items():
        if row['prim_path'] != metadata['actors'][name] or not np.allclose(row['position_m'],objects[name]['initial_position_m'],atol=1e-12,rtol=0):
            raise ValueError('Recorded actor pose differs from frozen metadata')
        q = objects[name]['initial_pose_xyzw'][3:]; expected = [q[3],*q[:3]]
        q_actual=np.asarray(row['quaternion_wxyz']);q_expected=np.asarray(expected)
        if not (np.allclose(q_actual,q_expected,atol=1e-7,rtol=0) or np.allclose(q_actual,-q_expected,atol=1e-7,rtol=0)):
            raise ValueError('Recorded actor orientation differs')
    task['episode_seed'] = seed; task['asset_metadata'] = metadata
    task['evaluation']['episode_seed'] = seed;task['evaluation']['metadata'] = deepcopy(metadata)
    bundle = scene['native_task_bundle'];bundle.update(usd_path=manifest['initial_usd'],usd_sha256=manifest['initial_usd_sha256'])
    bundle['dependencies_sha256'] = {**bundle.get('dependencies_sha256',{}),
        **manifest['dependencies_sha256'],str(path):file_hash(path)}
    bundle['observer']['configuration'] = deepcopy(task)
    if 'reduced_dynamics' in bundle: bundle['reduced_dynamics']['configuration'] = deepcopy(task)
    provenance = scene['laboratory_task_provenance'];provenance['parent_asset_candidate'] = deepcopy(candidate)
    provenance['asset_candidate'] = {'manifest_path':str(path),'manifest_sha256':file_hash(path),
        'usd_path':manifest['initial_usd'],'usd_sha256':manifest['initial_usd_sha256'],
        'metadata_path':manifest['metadata'],'metadata_sha256':manifest['metadata_sha256'],
        'root_prim_path':metadata['root_path'],'dependencies':deepcopy(manifest['dependencies_sha256'])}
    provenance['seed_derivative'] = {'manifest_path':str(path),'manifest_sha256':file_hash(path),
        'seed':seed,'variant_scope':VARIANT_SCOPE,'native_qualified':False}
    return {**values,'evaluator_sources':evaluator_sources(parent,task_id),
        **{key:deepcopy(manifest[key]) for key in ('actual_body_poses_m','static_geometry','interaction_targets','fixed_support_paths')},
        'asset_files_sha256':{**manifest['dependencies_sha256'],str(path):file_hash(path)}}
