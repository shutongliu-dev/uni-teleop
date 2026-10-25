"""Original closed hardback books shared by study tasks and fixed decoration.

Book coordinates are metres, centre origin, X=depth/Y=thickness/Z=height.
The spine faces +X, front cover +Y, back cover -Y and fore-edge -X. A +90°
rotation about X lays the book down with its front cover upward. Printed titles
are visible texture content; they are not a private recognition side channel.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np

from bench.paths import DATA_ROOT, load_json

DEFAULT_ASSETS_ROOT = Path(f'{DATA_ROOT}/assets')
BUNDLE_RELATIVE = Path('task_parts/study/books/v1')
PROJECT = Path(__file__).resolve().parents[3]
PHYSICS = {'static_friction': .50, 'dynamic_friction': .40, 'restitution': .02,
           'contact_offset_m': .001, 'rest_offset_m': 0.,
           'paper_stack_density_kg_m3': 720., 'cover_board_density_kg_m3': 850.,
           'spine_density_kg_m3': 600., 'cloth_areal_density_kg_m2': .30,
           'scope': 'Engineering material prior; no native grasp or contact qualification'}
TASK_BOOK_IDS = ('robotics', 'astronomy', 'botany', 'architecture', 'painting', 'literature')
_TITLES = [
    ('robotics', '机器人学基础', 'science', [.170, .032, .240], 'ROBOTICS / PRINCIPLES & PRACTICE', '林知远', 'robot'),
    ('astronomy', '星空与宇宙', 'science', [.162, .030, .232], 'A FIELD GUIDE TO THE NIGHT SKY', '沈星河', 'orbit'),
    ('botany', '植物观察手册', 'science', [.154, .028, .224], 'BOTANICAL NOTES / FIELD STUDIES', '许青禾', 'leaf'),
    ('architecture', '建筑与空间', 'arts', [.166, .034, .235], 'ARCHITECTURE / FORM AND SPACE', '周见山', 'building'),
    ('painting', '色彩与绘画', 'arts', [.158, .030, .225], 'COLOUR / MATERIAL / EXPRESSION', '程一墨', 'colour'),
    ('literature', '山野随笔', 'arts', [.145, .028, .211], 'ESSAYS FROM THE OPEN COUNTRY', '陆听风', 'mountain'),
    ('geometry', '几何的语言', 'science', [.155, .029, .223], 'GEOMETRY / SHAPES AND PATTERNS', '何方', 'orbit'),
    ('ecology', '森林与河流', 'science', [.150, .030, .218], 'FORESTS / RIVERS / ECOLOGY', '许青禾', 'leaf'),
    ('mechanics', '日常力学', 'science', [.164, .033, .235], 'MECHANICS OF EVERYDAY THINGS', '林知远', 'robot'),
    ('ocean', '海洋笔记', 'science', [.157, .028, .228], 'NOTES FROM THE BLUE PLANET', '沈星河', 'orbit'),
    ('garden', '庭园四季', 'arts', [.146, .030, .218], 'A GARDEN THROUGH THE SEASONS', '陆听风', 'leaf'),
    ('drawing', '素描练习册', 'arts', [.165, .029, .238], 'DRAWING / LINE AND OBSERVATION', '程一墨', 'colour'),
    ('cities', '城市漫游', 'arts', [.153, .031, .228], 'WALKING THROUGH THE CITY', '周见山', 'building'),
    ('mountains', '山川之间', 'arts', [.148, .028, .215], 'BETWEEN MOUNTAINS AND RIVERS', '陆听风', 'mountain'),
    ('materials', '材料的故事', 'science', [.162, .030, .232], 'MATERIALS / STRUCTURE AND TEXTURE', '何方', 'robot'),
    ('weather', '云与天气', 'science', [.151, .029, .221], 'CLOUDS / WEATHER / OBSERVATION', '沈星河', 'orbit'),
    ('typography', '文字与版式', 'arts', [.160, .032, .231], 'TYPE / PAGE / COMPOSITION', '程一墨', 'building'),
    ('craft', '手作之美', 'arts', [.147, .030, .216], 'THE QUIET ART OF MAKING', '周见山', 'colour'),
    ('birdwatching', '飞鸟观察', 'science', [.150, .028, .220], 'BIRDS / A FIELD COMPANION', '许青禾', 'leaf'),
    ('light', '光的旅程', 'science', [.158, .032, .230], 'LIGHT / OPTICS / DISCOVERY', '林知远', 'orbit'),
    ('history', '时间的纹理', 'arts', [.156, .033, .226], 'HISTORY / MEMORY / PLACE', '陆听风', 'mountain'),
    ('photography', '摄影与观看', 'arts', [.163, .030, .233], 'PHOTOGRAPHY / WAYS OF SEEING', '程一墨', 'building'),
    ('mathematics', '数学之美', 'science', [.152, .028, .224], 'MATHEMATICS / ORDER AND WONDER', '何方', 'orbit'),
    ('poetry', '窗边诗集', 'arts', [.144, .028, .210], 'POEMS AT THE WINDOW', '陆听风', 'mountain'),
]
_PALETTE = [(39, 66, 83), (39, 52, 86), (52, 79, 59), (122, 72, 45),
            (132, 94, 42), (101, 49, 44), (66, 76, 72), (54, 85, 84)]


def mass_properties(size):
    """Symmetric paper/board/spine engineering prior; uniform-box inertia."""
    d, t, h = map(float, size)
    paper = (d-.006)*(t-.004)*(h-.006)*PHYSICS['paper_stack_density_kg_m3']
    boards = 2*d*.002*h*PHYSICS['cover_board_density_kg_m3']
    spine = .003*(t-.004)*h*PHYSICS['spine_density_kg_m3']
    cloth = (2*d*h+t*h)*PHYSICS['cloth_areal_density_kg_m2']
    mass = paper+boards+spine+cloth
    inertia = [mass*(size[(i+1)%3]**2+size[(i+2)%3]**2)/12 for i in range(3)]
    return {'mass_kg': mass, 'diagonal_inertia_kg_m2': inertia,
            'mass_components_kg': {'paper_stack': paper, 'cover_boards': boards,
                                   'spine': spine, 'cloth': cloth},
            'center_of_mass_m': [0., 0., 0.],
            'inertia_model': 'uniform box with measured authored envelope and total constituent mass'}


BOOK_SPECS = {row[0]: {'id': row[0], 'title': row[1], 'category': row[2],
    'category_zh': '科学' if row[2]=='science' else '人文艺术', 'size_xyz_m': row[3],
    'subtitle': row[4], 'author': row[5], 'motif': row[6], 'colour_rgb': _PALETTE[i%len(_PALETTE)],
    'library_code': ('SCI' if row[2]=='science' else 'ART')+f'-{i+1:03d}',
    'edition': '初版 · 2026', 'publisher': '观物书房', 'task_book': i<6,
    **mass_properties(row[3])} for i, row in enumerate(_TITLES)}


def _digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _textures(spec, folder, assets_root):
    """Original typography over CC0 cloth, plus procedural paper edge maps."""
    from PIL import Image, ImageDraw, ImageFont, ImageFilter
    folder.mkdir(parents=True)
    font_path = assets_root/'fonts/noto_cjk_sc/v1/NotoSansCJKsc-Regular.otf'
    linen = assets_root/'materials/polyhaven/v1/rough_linen/rough_linen_diff_1k.jpg'
    cloth = Image.open(linen).convert('L')
    rng = np.random.default_rng(int(hashlib.sha256(spec['id'].encode()).hexdigest()[:8], 16))
    def font(n): return ImageFont.truetype(str(font_path), n)
    cream=(230, 222, 195); pale=(200, 189, 156)
    def background(size):
        weave = np.asarray(cloth.resize(size),dtype=np.float32)
        grain = rng.normal(0,1.0,(size[1],size[0]))
        modulation = .90+.16*weave/255.+grain/255.
        base=np.asarray(spec['colour_rgb'])[None,None,:]*modulation[:,:,None]
        return Image.fromarray(np.clip(base,0,255).astype('uint8'))
    def txt(draw, xy, text, size, fill=cream, anchor=None):
        draw.text(xy,text,font=font(size),fill=fill,anchor=anchor)
    def motif(draw, kind):
        # All drawings are original simple line illustrations in cover pixels.
        ink=(192,177,137); x0,y0,x1,y1=150,730,874,1170
        if kind=='orbit':
            for k in range(4):
                pad=k*54;draw.ellipse((x0+pad,y0+pad//2,x1-pad,y1-pad//2),outline=ink,width=3)
            draw.ellipse((486,921,538,973),fill=ink)
            for i in range(24):
                x=int(rng.integers(x0,x1));y=int(rng.integers(y0,y1));draw.ellipse((x,y,x+3,y+3),fill=cream)
        elif kind=='leaf':
            draw.line((500,1170,510,760),fill=ink,width=5)
            for k in range(6):
                y=1120-k*57;sign=-1 if k%2 else 1
                draw.ellipse((min(510,510+sign*175),y-80,max(510,510+sign*175),y),outline=ink,width=3)
                draw.line((510,y-2,510+sign*165,y-68),fill=ink,width=2)
        elif kind=='robot':
            pts=[(260,1110),(340,1010),(455,1080),(620,840),(748,915)]
            draw.line(pts,fill=ink,width=20)
            for x,y in pts:draw.ellipse((x-24,y-24,x+24,y+24),fill=spec['colour_rgb'],outline=cream,width=4)
            draw.line((200,1150,395,1150),fill=cream,width=5)
            draw.line((732,895,785,875,808,903),fill=cream,width=5)
            draw.line((751,932,797,925,812,901),fill=cream,width=5)
        elif kind=='building':
            for k in range(5):
                x=205+k*115;y=930-30*k
                draw.polygon([(x,1160),(x,y),(x+85,y-60),(x+85,1100)],outline=ink,width=3)
            for y in (1040,1080,1120):draw.line((205,y,750,y-200),fill=pale,width=2)
        elif kind=='colour':
            for k,c in enumerate([(196,149,76),(173,88,62),(91,121,125),(204,190,150)]):
                x=200+k*145;draw.rounded_rectangle((x,785+k*48,x+120,1110+k*12),radius=8,fill=c)
            for i in range(10):draw.line((180,800+i*30,855,1110-i*15),fill=cream,width=2)
        else:
            for k in range(5):
                points=[(140,1080+k*20),(280,910+k*25),(430,1020+k*12),(615,805+k*20),(880,1090+k*15)]
                draw.line(points,fill=ink,width=3)
    front=background((1024,1536));draw=ImageDraw.Draw(front)
    draw.rectangle((53,53,970,1482),outline=pale,width=2)
    txt(draw,(86,102),f'{spec["category_zh"]}  /  观物文库',30)
    txt(draw,(86,242),spec['title'],min(91, int(840/max(1,len(spec['title'])))),cream)
    # A thin rule and compact English subtitle give each cover a real hierarchy.
    draw.line((88,398,925,398),fill=pale,width=2)
    words=spec['subtitle'].split(' / ')
    for i,line in enumerate(words):txt(draw,(88,434+i*47),line,27,pale)
    txt(draw,(88,595),spec['author']+' 著',34)
    motif(draw,spec['motif'])
    txt(draw,(88,1365),spec['publisher'],36)
    txt(draw,(920,1407),spec['library_code'],26,pale,'ra')
    front.save(folder/'front.jpg',quality=94,subsampling=0)
    back=background((1024,1536));draw=ImageDraw.Draw(back)
    draw.rectangle((53,53,970,1482),outline=pale,width=2)
    txt(draw,(87,124),'观物文库  /  '+spec['category_zh'],32)
    txt(draw,(87,248),spec['title'],65)
    paragraphs=['从日常观察出发，重新认识身边的世界。',
                '以清晰的图示与细致的文字，记录发现的过程。',
                '在阅读、思考与实践之间，留下属于自己的笔记。',
                '愿每一次翻阅，都成为新的发现与灵感。']
    y=430
    for paragraph in paragraphs:
        for start in range(0,len(paragraph),21):txt(draw,(90,y),paragraph[start:start+21],34);y+=64
        y+=20
    draw.line((90,1040,930,1040),fill=pale,width=2)
    for i,line in enumerate([spec['author']+' 著  |  '+spec['edition'],'装帧：布面精装  ·  插图：观物书房','馆藏编号  '+spec['library_code']]):txt(draw,(90,1094+i*61),line,29)
    draw.rectangle((634,1320,925,1420),fill=cream)
    x=651
    for bit in hashlib.sha256(spec['id'].encode()).digest():
        w=2+bit%4
        draw.rectangle((x,1335,x+w,1403),fill=(28,29,26));x+=w+3
        if x>908:break
    txt(draw,(88,1371),'观物书房',34)
    back.save(folder/'back.jpg',quality=94,subsampling=0)
    spine=background((256,1536));draw=ImageDraw.Draw(spine)
    draw.line((26,57,230,57),fill=pale,width=3);draw.line((26,82,230,82),fill=pale,width=2)
    txt(draw,(128,130),'观物',42,cream,'ma')
    size=83;spacing=99
    for i,char in enumerate(spec['title']):txt(draw,(128,298+i*spacing),char,size,cream,'ma')
    txt(draw,(128,1085),spec['author'],35,pale,'ma')
    draw.rectangle((24,1202,232,1418),fill=cream)
    txt(draw,(128,1242),spec['category_zh'],34,(43,48,43),'ma')
    txt(draw,(128,1320),spec['library_code'],30,(43,48,43),'ma')
    txt(draw,(128,1462),'2026',25,pale,'ma')
    spine.save(folder/'spine.jpg',quality=96,subsampling=0)
    # Physical paper sheets lie in XZ planes; fore-edge lines therefore run up Z.
    for name,size,vertical in [('fore_edge',(256,1536),True),('page_top',(1024,256),False)]:
        w,h=size;noise=rng.normal(0,1.4,(h,w));a=np.empty((h,w,3),dtype=np.float32)
        for k,c in enumerate((225,218,197)):a[:,:,k]=c+noise
        axis=w if vertical else h
        for i in range(axis):
            shade=8 if i%5==0 else (3 if i%2==0 else 0)
            if vertical:a[:,i,:]-=shade
            else:a[i,:,:]-=shade
        paper=Image.fromarray(np.clip(a,0,255).astype('uint8'));paper.save(folder/(name+'.jpg'),quality=95,subsampling=0)
    return {name: str(folder/(name+'.jpg')) for name in ('front','back','spine','fore_edge','page_top')}


def _visual_asset(path, spec, textures, assets_root):
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade
    from bench.simulation.assets.workrooms.builder import Builder, safe
    stage=Usd.Stage.CreateNew(str(path));root=UsdGeom.Xform.Define(stage,'/Book')
    stage.SetDefaultPrim(root.GetPrim());UsdGeom.SetStageMetersPerUnit(stage,1.);UsdGeom.SetStageUpAxis(stage,'Z')
    class VisualAuthor(Builder):
        def __init__(self):
            self.stage=stage;self.materials={};self.records=[];self._names=set()
        def path(self,name):
            name=safe(name)
            if name in self._names:raise ValueError(name)
            self._names.add(name);return '/Book/Geometry/'+name
    b=VisualAuthor()
    linen=assets_root/'materials/polyhaven/v1/rough_linen'
    def material(name, colour=(1.,1.,1.), texture=None, cloth=False):
        p='/Book/Looks/'+name;mat=UsdShade.Material.Define(stage,p)
        sh=UsdShade.Shader.Define(stage,p+'/Surface');sh.CreateIdAttr('UsdPreviewSurface')
        sh.CreateInput('diffuseColor',Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*colour))
        sh.CreateInput('roughness',Sdf.ValueTypeNames.Float).Set(.82 if cloth else .90)
        sh.CreateInput('metallic',Sdf.ValueTypeNames.Float).Set(0.)
        uv=UsdShade.Shader.Define(stage,p+'/UV');uv.CreateIdAttr('UsdPrimvarReader_float2')
        uv.CreateInput('varname',Sdf.ValueTypeNames.Token).Set('st')
        uv.CreateOutput('result',Sdf.ValueTypeNames.Float2)
        for role,file,normal in [('diffuseColor',texture,False),('normal',str(linen/'rough_linen_nor_gl_1k.jpg') if cloth else None,True)]:
            if not file:continue
            tex=UsdShade.Shader.Define(stage,p+'/'+role);tex.CreateIdAttr('UsdUVTexture')
            tex.CreateInput('file',Sdf.ValueTypeNames.Asset).Set(file)
            tex.CreateInput('sourceColorSpace',Sdf.ValueTypeNames.Token).Set('raw' if normal else 'sRGB')
            tex.CreateInput('st',Sdf.ValueTypeNames.Float2).ConnectToSource(uv.ConnectableAPI(),'result')
            if normal:
                tex.CreateInput('scale',Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(.32,.32,2.,1.))
                tex.CreateInput('bias',Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(-.16,-.16,-1.,0.))
            tex.CreateOutput('rgb',Sdf.ValueTypeNames.Float3)
            sh.CreateInput(role,Sdf.ValueTypeNames.Normal3f if normal else Sdf.ValueTypeNames.Color3f).ConnectToSource(tex.ConnectableAPI(),'rgb')
        mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(),'surface');b.materials[name]=mat
    colour=tuple((np.asarray(spec['colour_rgb'])/255.)**2.2)
    material('Cloth',colour,cloth=True);material('Paper',(.76,.72,.62))
    for name in textures:material(name,texture=textures[name],cloth=name in ('front','back','spine'))
    d,t,h=spec['size_xyz_m'];board=.002
    b.box('PaperBlock',(-.0015,0,0),(d-.006-.00002,t-2*board-.00002,h-.006-.00002),'Paper',.0005,collision=False)
    for sign in (-1,1):b.box('Cover_'+str(sign),(0,sign*(t-board-.00002)/2,0),(d,board-.00002,h),'Cloth',.0004,collision=False)
    b.box('ClothSpine',(d/2-.00151,0,0),(.00298,t-.00004,h),'Cloth',.0006,collision=False)
    # Printed quads stop inside the rounded board edges and remain in envelope.
    def face(name,points):b.mesh('/Book/Print/'+name,points,[(0,1,2,3)],name,[(0,0),(1,0),(1,1),(0,1)])
    margin=.00065;xx=d/2-margin;zz=h/2-margin
    face('front',[(xx,t/2, -zz),(-xx,t/2,-zz),(-xx,t/2,zz),(xx,t/2,zz)])
    face('back',[(-xx,-t/2,-zz),(xx,-t/2,-zz),(xx,-t/2,zz),(-xx,-t/2,zz)])
    face('spine',[(d/2,-t/2+margin,-zz),(d/2,t/2-margin,-zz),(d/2,t/2-margin,zz),(d/2,-t/2+margin,zz)])
    edge_x=-d/2+.0015;inner_t=t/2-board;inner_h=h/2-.003
    face('fore_edge',[(edge_x,inner_t,-inner_h),(edge_x,-inner_t,-inner_h),(edge_x,-inner_t,inner_h),(edge_x,inner_t,inner_h)])
    face('page_top',[(-d/2+.0015,-inner_t,inner_h),(d/2-.0045,-inner_t,inner_h),
                     (d/2-.0045,inner_t,inner_h),(-d/2+.0015,inner_t,inner_h)])
    # Bottom face shares the same generated paper map.
    b.mesh('/Book/Print/page_bottom',[(-d/2+.0015,inner_t,-inner_h),(d/2-.0045,inner_t,-inner_h),
               (d/2-.0045,-inner_t,-inner_h),(-d/2+.0015,-inner_t,-inner_h)],[(0,1,2,3)],'page_top',[(0,0),(1,0),(1,1),(0,1)])
    root.GetPrim().SetCustomDataByKey('study:book_id',spec['id'])
    stage.GetRootLayer().Save()


def book_spec(book):
    value=BOOK_SPECS[book] if isinstance(book,str) else book
    if value['id'] not in BOOK_SPECS or value['size_xyz_m']!=BOOK_SPECS[value['id']]['size_xyz_m']:
        raise ValueError('Book ID/geometry must match the pinned library')
    return deepcopy(BOOK_SPECS[value['id']])


def author_book(stage, path, book, *, position=(0.,0.,0.), quaternion_wxyz=(1.,0.,0.,0.),
                assets_root=DEFAULT_ASSETS_ROOT, dynamic=True, collision=True):
    """Reference one book; optionally add exactly one box collider and one body.

    Returns all dimensions, physics priors, source paths and collider/body paths.
    Calling code must include ``dependencies`` in its frozen release inventory.
    Static decoration uses dynamic=False/collision=False and never owns a body.
    """
    from pxr import Gf, Sdf, UsdGeom, UsdPhysics, UsdShade
    spec=book_spec(book);bundle=Path(assets_root)/BUNDLE_RELATIVE
    manifest_path=bundle/'manifest.json';manifest=load_json(manifest_path)
    asset=bundle/spec['id']/'book.usdc'
    published=manifest['books'][spec['id']]
    for field in ('size_xyz_m','mass_kg','title','category'):
        if published[field]!=spec[field]:raise ValueError('Book source differs from frozen library: '+field)
    if manifest['books'][spec['id']]['usd_sha256']!=_digest(asset):raise ValueError('Book USD changed: '+str(asset))
    root=UsdGeom.Xform.Define(stage,path);root.GetPrim().GetReferences().AddReference(str(asset),'/Book')
    xf=UsdGeom.Xformable(root);xf.AddTranslateOp().Set(Gf.Vec3d(*map(float,position)))
    q=np.asarray(quaternion_wxyz,dtype=float)
    if q.shape!=(4,) or not np.isfinite(q).all() or np.linalg.norm(q)<1.e-9:raise ValueError('Invalid book orientation')
    q=q/np.linalg.norm(q);xf.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Quatd(float(q[0]),Gf.Vec3d(*map(float,q[1:]))))
    if dynamic:
        if not collision:raise ValueError('Dynamic books require the declared closed-book collider')
        body=UsdPhysics.RigidBodyAPI.Apply(root.GetPrim());body.CreateRigidBodyEnabledAttr(True);body.CreateKinematicEnabledAttr(False)
        props=UsdPhysics.MassAPI.Apply(root.GetPrim());props.CreateMassAttr(spec['mass_kg'])
        props.CreateCenterOfMassAttr(Gf.Vec3f(0));props.CreateDiagonalInertiaAttr(Gf.Vec3f(*spec['diagonal_inertia_kg_m2']));props.CreatePrincipalAxesAttr(Gf.Quatf(1))
    collider_path=None
    if collision:
        collider_path=path+'/Collider';c=UsdGeom.Cube.Define(stage,collider_path);c.CreateSizeAttr(1.)
        UsdGeom.Xformable(c).AddScaleOp().Set(Gf.Vec3f(*spec['size_xyz_m']));c.CreateVisibilityAttr('invisible')
        UsdPhysics.CollisionAPI.Apply(c.GetPrim()).CreateCollisionEnabledAttr(True)
        c.GetPrim().AddAppliedSchema('PhysxCollisionAPI')
        for key in ('contact_offset_m','rest_offset_m'):
            attr='contactOffset' if key.startswith('contact') else 'restOffset'
            c.GetPrim().CreateAttribute('physxCollision:'+attr,Sdf.ValueTypeNames.Float).Set(PHYSICS[key])
        material=UsdShade.Material.Define(stage,path+'/PhysicsMaterial');api=UsdPhysics.MaterialAPI.Apply(material.GetPrim())
        api.CreateStaticFrictionAttr(PHYSICS['static_friction']);api.CreateDynamicFrictionAttr(PHYSICS['dynamic_friction']);api.CreateRestitutionAttr(PHYSICS['restitution'])
        UsdShade.MaterialBindingAPI.Apply(c.GetPrim()).Bind(material,materialPurpose='physics')
    deps={str(manifest_path):_digest(manifest_path),str(bundle/'SOURCE.json'):_digest(bundle/'SOURCE.json')}
    deps.update({str(bundle/p):v['sha256'] for p,v in manifest['files'].items()
                 if p.startswith(spec['id']+'/') or '/' not in p})
    deps.update({p:v['sha256'] for p,v in manifest['source_dependencies'].items()})
    return {**spec,'path':path,'rigid_body_path':path if dynamic else None,'collider_path':collider_path,
            'position_m':list(position),'quaternion_wxyz':q.tolist(),'asset':str(asset),'physics':deepcopy(PHYSICS),
            'axes':{'depth':'+X','thickness':'+Y','height':'+Z','spine':'+X','front_cover':'+Y'},
            'dependencies':deps}



def author_display_book(builder, name, book, *, position, yaw_deg=0.):
    """Place an unscaled static display book; position is its centre, not its foot.

    Books retain readable covers and source geometry but have no body/collider.
    All source/license/texture paths are registered on the room's Builder.
    """
    from math import cos, sin, radians
    from bench.simulation.assets.workrooms.builder import safe
    path=builder.path(name)
    half=radians(yaw_deg)/2
    result=author_book(builder.stage,path,book,position=position,
        quaternion_wxyz=(cos(half),0.,0.,sin(half)),assets_root=builder.assets_root,
        dynamic=False,collision=False)
    for dependency in result['dependencies']:builder.dependency(dependency)
    builder.stage.GetPrimAtPath(path).SetCustomDataByKey('scence:staticDecoration',True)
    builder.records.append({'name':safe(name),'kind':'asset','source':result['asset'],
        'position':list(position),'rotation':[0.,0.,yaw_deg],'bounds_size':result['size_xyz_m'],
        'static_scenery':True,'book_id':result['id'],'book_title':result['title']})
    return result


def build_library(output, assets_root=DEFAULT_ASSETS_ROOT, *, snapshot):
    from pxr import Usd, UsdGeom, UsdUtils
    output=Path(output).resolve();assets_root=Path(assets_root).resolve()
    if output.exists():raise FileExistsError('Preserve previous book assets: '+str(output))
    output.mkdir(parents=True)
    source_files=[assets_root/'fonts/noto_cjk_sc/v1/NotoSansCJKsc-Regular.otf',assets_root/'fonts/noto_cjk_sc/v1/LICENSE.txt',
        assets_root/'materials/polyhaven/v1/rough_linen/SOURCE.json',
        assets_root/'materials/polyhaven/v1/rough_linen/rough_linen_diff_1k.jpg',assets_root/'materials/polyhaven/v1/rough_linen/rough_linen_nor_gl_1k.jpg']
    sources={str(p):{'sha256':_digest(p),'bytes':p.stat().st_size} for p in source_files}
    books={}
    for name,spec in BOOK_SPECS.items():
        directory=output/name;texture_files=_textures(spec,directory/'textures',assets_root)
        target=directory/'book.usdc';_visual_asset(target,spec,texture_files,assets_root)
        stage=Usd.Stage.Open(str(target));bb=UsdGeom.BBoxCache(0,['default','render']).ComputeWorldBound(stage.GetDefaultPrim()).ComputeAlignedRange()
        np.testing.assert_allclose(bb.GetMin(),-np.asarray(spec['size_xyz_m'])/2,atol=1e-7)
        np.testing.assert_allclose(bb.GetMax(),np.asarray(spec['size_xyz_m'])/2,atol=1e-7)
        _,_,unresolved=UsdUtils.ComputeAllDependencies(str(target))
        if unresolved:raise ValueError(list(unresolved))
        books[name]={**spec,'usd':str(target),'usd_sha256':_digest(target),'textures':texture_files,
                     'bounds_min_max_m':[list(bb.GetMin()),list(bb.GetMax())]}
    provenance={'schema':'study_original_book_library_source.v1','created_utc':datetime.now(timezone.utc).isoformat(),
        'authorship':'Original fictional book titles, author names, illustrations, cover layouts, paper patterns and geometry authored by the interface project.',
        'source_snapshot':snapshot,'source_dependencies':sources,'cloth_source':'https://polyhaven.com/a/rough_linen',
        'cloth_license':'CC0-1.0','cloth_authors':'colormass (photography), Rico Cilliers (processing)',
        'font':'Noto Sans CJK SC Regular','font_source':'https://github.com/notofonts/noto-cjk','font_license':'SIL Open Font License 1.1; rendered documents do not inherit the font license',
        'generated_asset_license':'Original simulation book graphics and geometry: CC0-1.0; source font remains OFL-1.1.',
        'physics_prior':PHYSICS,'scope':'Closed rigid hardbacks; no page articulation, native contact or task qualification'}
    (output/'SOURCE.json').write_text(json.dumps(provenance,ensure_ascii=False,indent=2)+'\n')
    (output/'LICENSE.txt').write_text('Original simulation book artwork and geometry: CC0 1.0 Universal.\nCloth source: Poly Haven rough_linen, CC0-1.0.\nhttps://polyhaven.com/license\nhttps://creativecommons.org/publicdomain/zero/1.0/\nNoto Sans CJK font: SIL Open Font License 1.1; original font/license remain at source_dependencies paths.\n')
    files={str(p.relative_to(output)):{'sha256':_digest(p),'bytes':p.stat().st_size} for p in sorted(output.rglob('*')) if p.is_file()}
    manifest={'schema':'study_book_library.v1','version':'v1','created_utc':datetime.now(timezone.utc).isoformat(),
        'source_snapshot':snapshot,'source_dependencies':sources,'books':books,'task_book_ids':list(TASK_BOOK_IDS),'files':files,
        'frame':{'units':'metres','up_axis':'Z','origin':'closed-book bounding-box centre','spine':'+X','front_cover':'+Y','back_cover':'-Y','fore_edge':'-X'},
        'validation':{'book_count':len(books),'bounds_checked':True,'unresolved_dependencies':[],'native_physics_accepted':False}}
    (output/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    return manifest


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--assets-root',type=Path,default=DEFAULT_ASSETS_ROOT);args=parser.parse_args()
    from scripts.maintenance.freeze_project_sources import verify_if_snapshot
    from bench.simulation.assets.workrooms.builder import Builder
    snapshot=verify_if_snapshot(PROJECT,required_modules=('scenes.workrooms.study.book_assets','bench.simulation.assets.workrooms.builder'),entrypoint_file=Path(__file__))
    if snapshot is None:parser.error('Build books from a complete frozen project source snapshot')
    result=build_library(args.output,args.assets_root,snapshot=snapshot)
    print(json.dumps({'output':str(args.output),'book_count':len(result['books']),'task_books':list(TASK_BOOK_IDS),'manifest_sha256':_digest(args.output/'manifest.json')}))


if __name__=='__main__':main()
