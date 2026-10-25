"""Immutable study task props, reset layouts and complete frozen rule inputs.

The fixed room is referenced separately by composition.py. This author owns only
movable props, task fixtures and printed targets, never the room or robot.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from .book_assets import DEFAULT_ASSETS_ROOT, author_book, book_spec
from .rules import TASK_IDS, build_configuration, validate_configuration
from bench.paths import DATA_ROOT, load_json

PROJECT = Path(__file__).resolve().parents[3]
TASK_ROOT = '/World/StudyTask'
OBJECT_ROOT = TASK_ROOT + '/Objects'
DEFAULT_TASK_ROOT = DEFAULT_ASSETS_ROOT / 'task_parts/study'
# Task1 v4 moves supply/organizer into the reviewed near-table workspace.
# Other tasks retain the v3 builds from study_task_v7_migration_20261019/source01.
CANDIDATES = {
    'stationery_sorting': {
        'root': f'{DATA_ROOT}/assets/task_parts/study/stationery_sorting/v4',
        'manifest_sha256': '9d181c155877b786ddf09005ee755a91185fcad375e8d789a044cb77b39ec03a',
    },
    'book_stacking': {
        'root': f'{DATA_ROOT}/assets/task_parts/study/book_stacking/v3',
        'manifest_sha256': '5da60588c37156544d2194c18cbd50433ba667db959d8c61d8598b9fdc40abf4',
    },
    'book_retrieval': {
        'root': f'{DATA_ROOT}/assets/task_parts/study/book_retrieval/v3',
        'manifest_sha256': '067f123860ad9b60cefdd3943c27935a907e76dc433f97f1e7ae536f29790f75',
    },
    'book_shelving': {
        'root': f'{DATA_ROOT}/assets/task_parts/study/book_shelving/v3',
        'manifest_sha256': '28c310659424504be0fd0a94c2cc317cdf2c308a57cf0704372f9928e195fd15',
    },
}
SUPPORTS = {
    'table': {'bounds_m': [[.20, -1., .70], [1.10, 1., .78]]},
    'shelf_lower': {'bounds_m': [[-1.16, -.70, .736], [-.84, .70, .76]]},
    'shelf_upper': {'bounds_m': [[-1.16, -.70, 1.096], [-.84, .70, 1.12]]},
}
PHYSICS = {'static_friction': .50, 'dynamic_friction': .40, 'restitution': .02,
           'contact_offset_m': .001, 'rest_offset_m': 0.,
           'scope': 'Uncalibrated engineering material priors, not native task qualification'}
STATIONERY = {
    'pencil': {'category': 'pencil', 'title': '观物 2B 粗六角铅笔',
        'local_bounds_m': [[-.090, -.0055, -.0055*math.sqrt(3)/2],
                           [.090, .0055, .0055*math.sqrt(3)/2+.000002]],
        'mass_kg': .0081, 'mass_basis': '180mm cedar/graphite pencil engineering prior: cedar 5.9g, graphite 1.8g, lacquer 0.4g'},
    'marker': {'category': 'marker', 'title': '观物 绿色马克笔',
        'local_bounds_m': [[-.0725, -.009, -.009], [.0725, .011, .009]],
        'mass_kg': .023, 'mass_basis': '145mm capped marker engineering prior: PP shell/cap 11g, ink reservoir 10g, nib/clip 2g'},
    'eraser': {'category': 'eraser', 'title': '观物 绘图橡皮',
        'local_bounds_m': [[-.026, -.013, -.008], [.026, .013, .008002]],
        'mass_kg': .052*.026*.016*1300.,
        'mass_basis': '52x26x16mm rubber envelope at engineering density 1300kg/m3; paper sleeve neglected'},
}
# The organizer stays in the original arm's near-table workspace.  Its rails,
# printed labels and rule envelopes share this center; supply objects lie in a
# separate near-edge lane with their long axes along Y.
STATIONERY_LAYOUT = {
    'organizer_center_x_m': .50,
    'source_x_m': .27,
    'source_y_m': (-.30, 0., .30),
    'source_yaw_deg': 90.,
    'compartment_y_m': (-.232, 0., .232),
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _dump(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def _flat(yaw=0.):
    """Book front +Y becomes world +Z, with a world-Z heading."""
    a=math.radians(yaw)/2; h=2**-.5
    return [math.cos(a)*h, math.cos(a)*h, math.sin(a)*h, math.sin(a)*h]


def _yaw(angle):
    a=math.radians(angle)/2
    return [math.cos(a), 0., 0., math.sin(a)]


def _pose(position, q):
    return {'position_m': list(map(float,position)), 'quaternion_wxyz': list(map(float,q))}


def _target(slot, category, bounds, support='table', *, height=None, up=None, spine=None):
    result={'slot_id': slot, 'category': category, 'bounds_m': bounds,
            'support_id': support, 'support_z_m': SUPPORTS[support]['bounds_m'][1][2] if height is None else height}
    if up is not None: result['up_direction_world']=up
    if spine is not None: result['spine_direction_world']=spine
    return result


def configuration(task_id):
    """Return deterministic geometry/rules from the actual authored envelopes."""
    if task_id not in TASK_IDS: raise ValueError('Unknown study task')
    objects={}; targets={}; kwargs={}
    if task_id=='stationery_sorting':
        layout=STATIONERY_LAYOUT;cx=layout['organizer_center_x_m']
        for i,(ident,spec) in enumerate(STATIONERY.items()):
            z=.78-spec['local_bounds_m'][0][2]
            objects[ident]={'title_zh':spec['title'], 'category':spec['category'], 'local_bounds_m':spec['local_bounds_m'],
                'initial_pose':_pose([layout['source_x_m'],layout['source_y_m'][i],z],_yaw(layout['source_yaw_deg']))}
            y=layout['compartment_y_m'][i]
            targets[ident]=_target(ident+'_compartment',spec['category'],[[cx-.140,y-.096,.78],[cx+.140,y+.096,.82]])
    else:
        ids=('robotics','botany','architecture') if task_id=='book_shelving' else ('robotics','astronomy','botany')
        for i,ident in enumerate(ids):
            spec=book_spec(ident);size=np.array(spec['size_xyz_m']);upright=task_id=='book_retrieval'
            objects[ident]={'title_zh':spec['title'],'category':spec['category'],'local_bounds_m':[(-size/2).tolist(),(size/2).tolist()],
                'initial_pose':_pose([-.985,(-.46,-.30,-.14)[i],.76+size[2]/2] if upright else
                    [(.34,.43,.34)[i],(-.32,0.,.32)[i],.78+size[1]/2], [1.,0.,0.,0.] if upright else _flat((-8,17,-12)[i])),
                'up_axis_local':[0,0,1] if task_id=='book_shelving' or upright and ident!='robotics' else [0,1,0],
                'spine_axis_local':[1,0,0]}
        if task_id=='book_stacking':
            height=.78
            for ident in ids:
                d,t,h=book_spec(ident)['size_xyz_m']
                targets[ident]=_target(ident+'_stack_layer','science',
                    [[.4-d/2-.014,-h/2-.014,height-.001],[.4+d/2+.014,h/2+.014,height+t+.008]],
                    support='table',height=height,up=[0,0,1],spine=[1,0,0])
                targets[ident]['support_id']='table' if ident==ids[0] else ids[ids.index(ident)-1]
                height+=t
            kwargs['stack_order']=list(ids)
        elif task_id=='book_retrieval':
            targets['robotics']=_target('retrieved_book','science',[[.295,-.140,.78],[.505,.140,.835]],up=[0,0,1],spine=[1,0,0])
            for ident in ids[1:]:
                d,t,h=book_spec(ident)['size_xyz_m'];y=objects[ident]['initial_pose']['position_m'][1]
                targets[ident]=_target(ident+'_original_shelf','science',
                    [[-1.09,y-t/2-.012,.76],[-.88,y+t/2+.012,.76+h+.015]],'shelf_lower',up=[0,0,1],spine=[1,0,0])
            kwargs.update(retrieval_target_id='robotics',protected_ids=list(ids[1:]))
        else:
            for ident,y in zip(ids,(-.45,-.23,.34)):
                d,t,h=book_spec(ident)['size_xyz_m'];category=book_spec(ident)['category']
                targets[ident]=_target(category+'_'+ident+'_shelf',category,
                    [[-1.095,y-t/2-.018,.76],[-.885,y+t/2+.018,.76+h+.020]],'shelf_lower',up=[0,0,1],spine=[1,0,0])
    used={v['support_id'] for v in targets.values() if v['support_id'] in SUPPORTS}|{'table'}
    return build_configuration(task_id,objects,targets,{k:deepcopy(SUPPORTS[k]) for k in sorted(used)},**kwargs)


def _builder(stage):
    """Reuse room mesh routines while keeping every material under the task root."""
    from pxr import Gf, Sdf, UsdShade
    from bench.simulation.assets.workrooms.builder import Builder, safe
    class Author(Builder):
        def __init__(self):
            self.stage=stage;self.materials={};self.records=[];self._names=set();self.base=TASK_ROOT+'/Fixtures'
        def path(self,name):
            path=self.base+'/'+safe(name)
            if path in self._names:raise ValueError('Duplicate task geometry '+path)
            self._names.add(path);return path
        def material(self,name,colour,roughness=.5,metallic=0.,texture=None):
            path=TASK_ROOT+'/Looks/'+name;mat=UsdShade.Material.Define(stage,path)
            sh=UsdShade.Shader.Define(stage,path+'/Surface');sh.CreateIdAttr('UsdPreviewSurface')
            sh.CreateInput('diffuseColor',Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*colour))
            sh.CreateInput('roughness',Sdf.ValueTypeNames.Float).Set(float(roughness))
            sh.CreateInput('metallic',Sdf.ValueTypeNames.Float).Set(float(metallic))
            if texture:
                uv=UsdShade.Shader.Define(stage,path+'/UV');uv.CreateIdAttr('UsdPrimvarReader_float2')
                uv.CreateInput('varname',Sdf.ValueTypeNames.Token).Set('st');uv.CreateOutput('result',Sdf.ValueTypeNames.Float2)
                tex=UsdShade.Shader.Define(stage,path+'/Texture');tex.CreateIdAttr('UsdUVTexture')
                tex.CreateInput('file',Sdf.ValueTypeNames.Asset).Set(str(texture));tex.CreateInput('sourceColorSpace',Sdf.ValueTypeNames.Token).Set('sRGB')
                tex.CreateInput('st',Sdf.ValueTypeNames.Float2).ConnectToSource(uv.ConnectableAPI(),'result')
                tex.CreateOutput('rgb',Sdf.ValueTypeNames.Float3);sh.GetInput('diffuseColor').ConnectToSource(tex.ConnectableAPI(),'rgb')
            mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(),'surface');self.materials[name]=mat
            return name
    b=Author()
    for name,col,rough,metal in [('oak',(.43,.27,.13),.55,0.),('felt',(.17,.22,.20),.95,0.),
            ('graphite',(.025,.027,.028),.7,0.),('cedar',(.58,.35,.17),.8,0.),('lacquer',(.62,.22,.045),.35,0.),
            ('ink',(.035,.13,.095),.38,0.),('ivory',(.83,.80,.70),.5,0.),('rubber',(.70,.65,.54),.93,0.),
            ('paper',(.78,.75,.66),.85,0.),('brass',(.44,.29,.10),.35,.65)]:
        b.material(name,col,rough,metal)
    return b


def _label(b, output, assets_root, name, text, colour=(228,219,193), background=(46,66,59)):
    from PIL import Image, ImageDraw, ImageFont
    folder=output/'textures';folder.mkdir(exist_ok=True)
    path=folder/(name+'.png');im=Image.new('RGB',(1024,256),background);draw=ImageDraw.Draw(im)
    font=ImageFont.truetype(str(assets_root/'fonts/noto_cjk_sc/v1/NotoSansCJKsc-Regular.otf'),75)
    draw.text((512,128),text,font=font,fill=colour,anchor='mm');im.save(path)
    b.material(name,(1.,1.,1.),.82,0.,texture=path)
    return name


def _quad(b, path, points, material):
    b.mesh(path,points,[(0,1,2,3)],material,[(0,0),(1,0),(1,1),(0,1)])


def _body(stage,path,pose,mass,bounds):
    from pxr import Gf, UsdGeom, UsdPhysics
    root=UsdGeom.Xform.Define(stage,path);xf=UsdGeom.Xformable(root)
    xf.AddTranslateOp().Set(Gf.Vec3d(*pose['position_m']));q=pose['quaternion_wxyz']
    xf.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Quatd(q[0],Gf.Vec3d(*q[1:])))
    body=UsdPhysics.RigidBodyAPI.Apply(root.GetPrim());body.CreateRigidBodyEnabledAttr(True);body.CreateKinematicEnabledAttr(False)
    api=UsdPhysics.MassAPI.Apply(root.GetPrim());api.CreateMassAttr(float(mass));api.CreateCenterOfMassAttr(Gf.Vec3f(0))
    size=np.asarray(bounds[1])-np.asarray(bounds[0]);inertia=[mass*(size[(i+1)%3]**2+size[(i+2)%3]**2)/12 for i in range(3)]
    api.CreateDiagonalInertiaAttr(Gf.Vec3f(*inertia));api.CreatePrincipalAxesAttr(Gf.Quatf(1))
    return inertia


def _physics(stage,root):
    from pxr import Sdf, Usd, UsdPhysics, UsdShade
    material=UsdShade.Material.Define(stage,str(root.GetPath())+'/PhysicsMaterial');api=UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    api.CreateStaticFrictionAttr(PHYSICS['static_friction']);api.CreateDynamicFrictionAttr(PHYSICS['dynamic_friction']);api.CreateRestitutionAttr(PHYSICS['restitution'])
    paths=[]
    for p in Usd.PrimRange(root):
        if p.HasAPI(UsdPhysics.CollisionAPI):
            p.AddAppliedSchema('PhysxCollisionAPI')
            p.CreateAttribute('physxCollision:contactOffset',Sdf.ValueTypeNames.Float).Set(PHYSICS['contact_offset_m'])
            p.CreateAttribute('physxCollision:restOffset',Sdf.ValueTypeNames.Float).Set(PHYSICS['rest_offset_m'])
            UsdShade.MaterialBindingAPI.Apply(p).Bind(material,materialPurpose='physics');paths.append(str(p.GetPath()))
    return paths


def _hex_mesh(b,path,x0,x1,r0,r1,material,collision=False):
    from pxr import UsdPhysics
    points=[(x,r*math.cos(i*math.pi/3),r*math.sin(i*math.pi/3)) for x,r in ((x0,r0),(x1,r1)) for i in range(6)]
    faces=[tuple(reversed(range(6))),tuple(range(6,12))]+[(i,(i+1)%6,(i+1)%6+6,i+6) for i in range(6)]
    # Mesh helper needs uniform face sizes; caps triangulate into degenerate-free quads via a separate mesh.
    side=b.mesh(path,points,faces[2:],material)
    for end,inds in [('left',faces[0]),('right',faces[1])]:
        from pxr import UsdGeom
        cap=UsdGeom.Mesh.Define(b.stage,path+'_'+end);cap.CreatePointsAttr(points)
        cap.CreateFaceVertexCountsAttr([6]);cap.CreateFaceVertexIndicesAttr(list(inds));cap.CreateSubdivisionSchemeAttr('none');b.bind(cap.GetPrim(),material)
    if collision:
        UsdPhysics.CollisionAPI.Apply(side);UsdPhysics.MeshCollisionAPI.Apply(side).CreateApproximationAttr('convexHull')
    return side


def _stationery(stage,b,ident,obj,output,assets_root):
    from pxr import UsdGeom
    path=OBJECT_ROOT+'/'+ident;spec=STATIONERY[ident]
    inertia=_body(stage,path,obj['initial_pose'],spec['mass_kg'],spec['local_bounds_m']);b.base=path+'/Geometry'
    if ident=='pencil':
        _hex_mesh(b,path+'/Geometry/LacquerShaft',-.068,.09,.0055,.0055,'lacquer',True)
        _hex_mesh(b,path+'/Geometry/CedarTip',-.090,-.068,0.,.0055,'cedar',True)
        _hex_mesh(b,path+'/Geometry/GraphiteTip',-.090,-.0865,0.,.00088,'graphite')
        mat=_label(b,output,assets_root,'pencil_print','观物  ·  2B  ·  JUMBO',(212,190,112),(184,96,24))
        z=.0055*math.sqrt(3)/2+.000002
        _quad(b,path+'/Geometry/Print',[(-.043,-.0027,z),(.067,-.0027,z),(.067,.0027,z),(-.043,.0027,z)],mat)
    elif ident=='marker':
        b.cylinder('Barrel',(.020,0.,0.),.0085,.105,'ivory',rotation=(0,90,0),collision=True)
        b.cylinder('Cap',(-.0525,0.,0.),.009,.040,'ink',rotation=(0,90,0),collision=True)
        b.box('CapClip',(-.0495,.0095,0.),(.031,.003,.004),'ink',bevel=.0006,collision=True)
        for k in range(3):b.cylinder('CapRib_'+str(k),(-.070+k*.0035,0.,0.),.009,.0007,'ink',rotation=(0,90,0),collision=False)
        b.cylinder('TailPlug',(.0705,0.,0.),.00852,.004,'ink',rotation=(0,90,0),collision=False)
        mat=_label(b,output,assets_root,'marker_print','观物  |  GREEN  |  04',(34,70,56),(227,221,198))
        # Conforming curved print stays just above the barrel and below cap envelope.
        angles=np.linspace(-.55,.55,17)
        points=[(x,.008504*math.sin(a),.008504*math.cos(a)) for x in (-.022,.060) for a in angles]
        faces=[(i,i+17,i+18,i+1) for i in range(16)]
        uv=[(j,i/16) for j in (0,1) for i in range(17)]
        b.mesh(path+'/Geometry/Print',points,faces,mat,uv)
    else:
        b.box('Rubber',(0.,0.,0.),(.052,.026,.016),'rubber',bevel=.0012,collision=True)
        b.box('PaperSleeve',(0.,0.,.00001),(.031,.02596,.01594),'paper',bevel=.0002,collision=False)
        mat=_label(b,output,assets_root,'eraser_print','观物  |  ART  |  01',(53,71,61),(221,214,193))
        _quad(b,path+'/Geometry/Print',[(-.0149,-.0118,.008002),(.0149,-.0118,.008002),(.0149,.0118,.008002),(-.0149,.0118,.008002)],mat)
    colliders=_physics(stage,stage.GetPrimAtPath(path))
    b.base=TASK_ROOT+'/Fixtures'
    return {**deepcopy(spec),'path':path,'collider_paths':colliders,'diagonal_inertia_kg_m2':inertia,
            'center_of_mass_m':[0.,0.,0.],'inertia_model':'uniform authored-envelope approximation using explicit engineering total mass',
            'physics':deepcopy(PHYSICS),'geometry_scope':'One rigid body; only structural geometry collides. Printed decoration is visual only.'}


def _fixtures(b,task_id,config,output,assets_root):
    """Printed goal marks; no marker is a physical substitute for the room board."""
    if task_id=='stationery_sorting':
        cx=STATIONERY_LAYOUT['organizer_center_x_m']
        # Open bottom means all three objects still touch the actual room table.
        for name,x in [('rear',cx-.160),('front',cx+.160)]:b.box('organizer_'+name,(x,0.,.791),(.012,.720,.022),'oak',bevel=.002)
        for i,y in enumerate((-.354,-.116,.116,.354)):b.box('organizer_cross_'+str(i),(cx,y,.791),(.308,.010,.022),'oak',bevel=.002)
        for ident,text in [('pencil','粗铅笔  /  PENCIL'),('marker','马克笔  /  MARKER'),('eraser','橡皮  /  ERASER')]:
            lo,hi=config['targets'][ident]['bounds_m'];y=(lo[1]+hi[1])/2
            mat=_label(b,output,assets_root,ident+'_slot',text)
            _quad(b,TASK_ROOT+'/Fixtures/'+ident+'_label',[(cx-.049,y-.070,.78008),(cx+.125,y-.070,.78008),(cx+.125,y-.029,.78008),(cx-.049,y-.029,.78008)],mat)
    elif task_id in ('book_stacking','book_retrieval'):
        mat=_label(b,output,assets_root,'book_goal_label','叠书区  /  STACK' if task_id=='book_stacking' else '取书区  /  RETRIEVAL')
        # Front-edge brass corners and paper label remain outside the resting book.
        for x in (.285,.515):
            for y in (-.150,.150):b.box('goal_corner_'+str(x)+'_'+str(y),(x,y,.78010),(.012,.012,.00015),'brass',bevel=0.,collision=False)
        _quad(b,TASK_ROOT+'/Fixtures/goal_label',[(.295,-.197,.78008),(.505,-.197,.78008),(.505,-.160,.78008),(.295,-.160,.78008)],mat)
    else:
        for category,y,label in [('science',-.42,'科学  /  SCIENCE'),('arts',.35,'人文艺术  /  ARTS')]:
            mat=_label(b,output,assets_root,'shelf_'+category,label)
            # Label on shelf fascia, normal +X; full books stay behind the front edge.
            _quad(b,TASK_ROOT+'/Fixtures/shelf_'+category+'_label',[(-.8399,y-.13,.739),(-.8399,y+.13,.739),(-.8399,y+.13,.756),(-.8399,y-.13,.756)],mat)
    return b.records


def build_candidate(task_id,output,assets_root=DEFAULT_ASSETS_ROOT,*,snapshot,version='v1'):
    from pxr import Usd, UsdGeom, UsdPhysics, UsdUtils
    output=Path(output).resolve();assets_root=Path(assets_root).resolve()
    if not version.startswith('v') or not version[1:].isdigit():raise ValueError('Expected a numbered immutable version')
    if output.exists():raise FileExistsError('Preserve immutable task candidate '+str(output))
    output.mkdir(parents=True);initial=output/'initial.usda'
    stage=Usd.Stage.CreateNew(str(initial));world=UsdGeom.Xform.Define(stage,'/World');UsdGeom.Xform.Define(stage,TASK_ROOT)
    stage.SetDefaultPrim(world.GetPrim());UsdGeom.SetStageMetersPerUnit(stage,1.);UsdGeom.SetStageUpAxis(stage,'Z')
    config=configuration(task_id);b=_builder(stage);assets={};dependencies={}
    for ident,obj in config['objects'].items():
        path=OBJECT_ROOT+'/'+ident
        if task_id=='stationery_sorting':assets[ident]=_stationery(stage,b,ident,obj,output,assets_root)
        else:
            pose=obj['initial_pose'];assets[ident]=author_book(stage,path,ident,position=pose['position_m'],quaternion_wxyz=pose['quaternion_wxyz'],assets_root=assets_root)
            dependencies.update(assets[ident]['dependencies'])
    fixtures=_fixtures(b,task_id,config,output,assets_root)
    # Consistent original engineering contact material on task-owned organizer rails.
    _physics(stage,stage.GetPrimAtPath(TASK_ROOT+'/Fixtures'))
    stage.GetRootLayer().Save()
    layers,asset_paths,unresolved=UsdUtils.ComputeAllDependencies(str(initial))
    if unresolved:raise ValueError('Unresolved task assets '+str(unresolved))
    for asset in [*(layer.realPath for layer in layers),*map(str,asset_paths)]:
        if asset and Path(asset).resolve()!=initial:dependencies[str(Path(asset).resolve())]=digest(asset)
    for relative in ('fonts/noto_cjk_sc/v1/NotoSansCJKsc-Regular.otf','fonts/noto_cjk_sc/v1/LICENSE.txt'):
        source=assets_root/relative;dependencies[str(source)]=digest(source)
    source={'schema':'study_task_source.v1','task_id':task_id,'created_utc':datetime.now(timezone.utc).isoformat(),
        'source_snapshot':snapshot,'authorship':'Original task geometry, stationery design, organizer, reset and target layouts by interface project. Book artwork is the separately pinned original study library.',
        'license':'Original task geometry and rendered labels CC0-1.0; Noto font remains OFL-1.1; book/cloth source licenses remain in their dependency inventories.',
        'physics':PHYSICS,'room_contract':'Exact fixed worktop and rear lower/upper shelf collider bounds are recorded in metadata.configuration.supports; the pinned parent room is composed separately and never mutated',
        'qualification':'Authored candidate. CPU dependency and rule checks do not establish native contacts or robot success.'}
    _dump(output/'SOURCE.json',source)
    (output/'LICENSE.txt').write_text('Original task geometry and rendered labels: CC0 1.0 Universal.\nhttps://creativecommons.org/publicdomain/zero/1.0/\nNoto Sans CJK SC source font: SIL OFL 1.1, retained at its dependency path.\nBook and cloth provenance: separately pinned study books/v1 inventory.\n')
    metadata={'schema':'study_task_metadata.v1','task_id':task_id,'configuration':config,'objects':assets,
        'fixtures':fixtures,'source_snapshot':snapshot,'physics_accepted':False,'robot_policy_success_claimed':False}
    _dump(output/'metadata.json',metadata)
    files={str(p.relative_to(output)):{'sha256':digest(p),'bytes':p.stat().st_size} for p in sorted(output.rglob('*')) if p.is_file()}
    # Self-contained manifests list generated files and external dependencies separately.
    for relative,entry in files.items():
        if relative!='initial.usda':dependencies[str(output/relative)]=entry['sha256']
    bodies={ident:OBJECT_ROOT+'/'+ident for ident in config['objects']}
    native_bodies={str(p.GetPath()) for p in stage.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI)}
    if native_bodies!=set(bodies.values()) or any(p.IsA(UsdPhysics.Joint) for p in stage.Traverse()):raise ValueError('Unexpected body/joint in task layer')
    manifest={'schema':'study_task_candidate.v1','task_id':task_id,'version':version,'initial_usd':str(initial),
        'initial_usd_sha256':digest(initial),'metadata_file':'metadata.json','metadata_sha256':digest(output/'metadata.json'),
        'bodies':bodies,'configuration_sha256':config['configuration_sha256'],'files':files,
        'dependencies_sha256':dependencies,'source_snapshot':snapshot,'physics_accepted':False,'robot_policy_success_claimed':False}
    _dump(output/'manifest.json',manifest)
    return manifest


def candidate_inputs(task_id):
    """Load only pinned immutable candidates and verify their complete inventory."""
    if task_id not in CANDIDATES:raise ValueError('No frozen study candidate for '+str(task_id))
    pinned=CANDIDATES[task_id];root=Path(pinned['root']);manifest_path=root/'manifest.json'
    if digest(manifest_path)!=pinned['manifest_sha256']:raise ValueError('Study candidate manifest changed')
    manifest=load_json(manifest_path)
    if manifest['task_id']!=task_id:raise ValueError('Study task identity differs')
    for relative,item in manifest['files'].items():
        path=root/relative
        if path.stat().st_size!=item['bytes'] or digest(path)!=item['sha256']:raise ValueError('Study candidate file changed '+str(path))
    dependencies=deepcopy(manifest['dependencies_sha256'])
    for path,expected in dependencies.items():
        if digest(path)!=expected:raise ValueError('Study dependency changed '+path)
    metadata=load_json(root/'metadata.json');validate_configuration(metadata['configuration'])
    dependencies[str(manifest_path)]=pinned['manifest_sha256']
    return ({'initial_usd':manifest['initial_usd'],'initial_usd_sha256':manifest['initial_usd_sha256'],
             'manifest_path':str(manifest_path),'manifest_sha256':pinned['manifest_sha256']},manifest,metadata,dependencies)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output-root',type=Path,required=True)
    parser.add_argument('--task-id',choices=(*TASK_IDS,'all'),default='all');parser.add_argument('--assets-root',type=Path,default=DEFAULT_ASSETS_ROOT)
    parser.add_argument('--version',default='v1',help='New immutable numbered version, for example v2')
    args=parser.parse_args()
    from scripts.maintenance.freeze_project_sources import verify_if_snapshot
    from bench.simulation.assets.workrooms.builder import Builder
    snapshot=verify_if_snapshot(PROJECT,required_modules=('scenes.workrooms.study.task_scene','scenes.workrooms.study.book_assets',
        'scenes.workrooms.study.rules','bench.simulation.assets.workrooms.builder'),entrypoint_file=Path(__file__))
    if snapshot is None:parser.error('Build task props from a complete frozen project snapshot')
    for task_id in TASK_IDS if args.task_id=='all' else (args.task_id,):
        output=args.output_root/task_id/args.version;manifest=build_candidate(task_id,output,args.assets_root,snapshot=snapshot,version=args.version)
        print(json.dumps({'task_id':task_id,'root':str(output),'manifest_sha256':digest(output/'manifest.json'),
            'initial_usd_sha256':manifest['initial_usd_sha256'],'configuration_sha256':manifest['configuration_sha256']}))


if __name__=='__main__':main()
