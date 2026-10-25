"""Independent, metre-scale OpenUSD scene authoring for the six workrooms.

Visual meshes have rounded edges and UVs. Primitive colliders use positive
world-axis dimensions. Imported furniture is scenery, never a claimed task asset.
"""
from pathlib import Path
import hashlib
import math
import re
import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade, UsdPhysics, UsdLux

from . import FAMILIES
from bench.paths import load_json


def safe(name):
    return re.sub(r'[^A-Za-z0-9_]', '_', str(name))


class Builder:
    def __init__(self, stage, assets_root, output_root, *, reuse_materials=False):
        self.stage=stage
        self.assets_root=Path(assets_root)
        self.output_root=Path(output_root)
        self.materials={}
        self.records=[]
        self.dependencies={}
        self._names=set()
        # Labels belong to this new build, never the published shared v2 tree.
        self.generated=self.output_root/'generated'/'textures'
        self.generated.mkdir(parents=True,exist_ok=True)
        UsdGeom.Xform.Define(stage,'/World')
        UsdGeom.Xform.Define(stage,'/World/Geometry')
        UsdGeom.Scope.Define(stage,'/World/Looks')
        stage.SetDefaultPrim(stage.GetPrimAtPath('/World'))
        UsdGeom.SetStageUpAxis(stage,UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(stage,1.)
        if reuse_materials:
            # Composition may add task props without reauthoring the fixed
            # environment's materials in the stronger preview layer.
            self.materials = {prim.GetName(): UsdShade.Material(prim)
                              for prim in stage.GetPrimAtPath('/World/Looks').GetChildren()
                              if prim.IsA(UsdShade.Material)}
        self.default_materials()

    def path(self,name):
        name=safe(name)
        if name in self._names:
            raise ValueError('Duplicate scene name: '+name)
        self._names.add(name)
        return '/World/Geometry/'+name

    def dependency(self,path):
        p=Path(path).resolve()
        if p.is_file() and str(p) not in self.dependencies:
            self.dependencies[str(p)]={'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size}

    def material(self,name,color,roughness=.5,metallic=0.,texture=None,normal=None,opacity=1.,roughness_texture=None,metallic_texture=None,texture_tint=None):
        name=safe(name)
        if name in self.materials:
            return name
        mat=UsdShade.Material.Define(self.stage,'/World/Looks/'+name)
        sh=UsdShade.Shader.Define(self.stage,str(mat.GetPath())+'/Surface')
        sh.CreateIdAttr('UsdPreviewSurface')
        sh.CreateInput('diffuseColor',Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color[:3]))
        sh.CreateInput('roughness',Sdf.ValueTypeNames.Float).Set(float(roughness))
        sh.CreateInput('metallic',Sdf.ValueTypeNames.Float).Set(float(metallic))
        sh.CreateInput('opacity',Sdf.ValueTypeNames.Float).Set(float(opacity))
        sh.CreateInput('ior',Sdf.ValueTypeNames.Float).Set(1.46)
        mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(),'surface')
        if any((texture,normal,roughness_texture,metallic_texture)):
            st=UsdShade.Shader.Define(self.stage,str(mat.GetPath())+'/ST')
            st.CreateIdAttr('UsdPrimvarReader_float2')
            st.CreateInput('varname',Sdf.ValueTypeNames.Token).Set('st')
            for role,file,channel,space in [('diffuseColor',texture,'rgb','sRGB'),('normal',normal,'rgb','raw'),('roughness',roughness_texture,'r','raw'),('metallic',metallic_texture,'r','raw')]:
                if not file: continue
                self.dependency(file)
                tex=UsdShade.Shader.Define(self.stage,str(mat.GetPath())+'/'+role+'Texture')
                tex.CreateIdAttr('UsdUVTexture')
                tex.CreateInput('file',Sdf.ValueTypeNames.Asset).Set(str(file))
                tex.CreateInput('sourceColorSpace',Sdf.ValueTypeNames.Token).Set(space)
                tex.CreateInput('st',Sdf.ValueTypeNames.Float2).ConnectToSource(st.ConnectableAPI(),'result')
                tex.CreateInput('wrapS',Sdf.ValueTypeNames.Token).Set('repeat')
                tex.CreateInput('wrapT',Sdf.ValueTypeNames.Token).Set('repeat')
                if role=='diffuseColor' and texture_tint is not None:
                    tex.CreateInput('scale',Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(*texture_tint,1.))
                if role=='normal':
                    tex.CreateInput('scale',Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(2,2,2,1))
                    tex.CreateInput('bias',Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(-1,-1,-1,0))
                sh.CreateInput(role,Sdf.ValueTypeNames.Normal3f if role=='normal' else (Sdf.ValueTypeNames.Color3f if role=='diffuseColor' else Sdf.ValueTypeNames.Float)).ConnectToSource(tex.ConnectableAPI(),channel)
        if name=='glass':
            # Native MDL solid-glass transmission; PreviewSurface remains a portable fallback.
            glass=UsdShade.Shader.Define(self.stage,str(mat.GetPath())+'/GlassMDL')
            glass.SetSourceAsset('OmniGlass.mdl','mdl')
            glass.SetSourceAssetSubIdentifier('OmniGlass','mdl')
            glass.CreateInput('glass_color',Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(.96,.99,1.))
            glass.CreateInput('glass_ior',Sdf.ValueTypeNames.Float).Set(1.47)
            glass.CreateInput('thin_walled',Sdf.ValueTypeNames.Bool).Set(False)
            glass.CreateInput('frosting_roughness',Sdf.ValueTypeNames.Float).Set(.035)
            glass.CreateInput('depth',Sdf.ValueTypeNames.Float).Set(.002)
            glass.CreateOutput('out',Sdf.ValueTypeNames.Token)
            mat.CreateSurfaceOutput('mdl').ConnectToSource(glass.ConnectableAPI(),'out')
            sh.GetInput('opacity').Set(.25)
        self.materials[name]=mat
        return name

    def default_materials(self):
        specs={'oak':([.48,.30,.16],.46,0),'walnut':([.19,.095,.044],.4,0),'steel':([.48,.51,.53],.3,.9),'chrome':([.64,.67,.7],.17,1),'black':([.025,.029,.032],.48,0),'white':([.82,.83,.80],.42,0),'ceramic':([.83,.81,.73],.2,0),'paper':([.89,.87,.80],.8,0),'rubber':([.028,.032,.035],.9,0),'glass':([.68,.82,.86],.11,0),'navy':([.035,.085,.14],.45,0),'green':([.11,.23,.16],.7,0),'red':([.48,.055,.025],.5,0),'fabric':([.48,.45,.39],.96,0),'wall':([.74,.71,.65],.85,0),'floor':([.42,.40,.36],.8,0),'paint':([.12,.18,.21],.43,.08)}
        texmap={'oak':'Wood089','walnut':'Wood089','wall':'PaintedPlaster017','floor':'WoodFloor064','rubber':'Rubber004'}
        for name,(color,r,m) in specs.items():
            extra={}
            if name in texmap:
                aid=texmap[name];root=self.assets_root/'materials'/'ambientcg'/'v1'/aid
                for key,suffix in [('texture','Color'),('normal','NormalGL'),('roughness_texture','Roughness')]:
                    p=root/f'{aid}_1K-JPG_{suffix}.jpg'
                    if p.exists():extra[key]=str(p)
                if name=='walnut':extra['texture_tint']=(.39,.29,.22)
                if name=='oak':extra['texture_tint']=(.91,.81,.65)
            self.material(name,color,r,m,**extra)

    def xform(self,prim,center,rotation=(0,0,0),scale=None):
        x=UsdGeom.Xformable(prim);x.ClearXformOpOrder()
        x.AddTranslateOp().Set(Gf.Vec3d(*map(float,center)))
        if any(rotation):x.AddRotateXYZOp().Set(Gf.Vec3f(*map(float,rotation)))
        if scale is not None:x.AddScaleOp().Set(Gf.Vec3f(*map(float,scale)))

    def bind(self,prim,material):
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(self.materials[material])

    def mesh(self,path,vertices,faces,material,uv=None,normals=None):
        v=np.asarray(vertices,dtype=np.float32);f=np.asarray(faces,dtype=np.int32)
        mesh=UsdGeom.Mesh.Define(self.stage,path)
        mesh.CreatePointsAttr(v.tolist())
        mesh.CreateFaceVertexCountsAttr([len(x) for x in f])
        mesh.CreateFaceVertexIndicesAttr(f.reshape(-1).tolist())
        mesh.CreateSubdivisionSchemeAttr('none')
        mesh.CreateDoubleSidedAttr(True)
        mesh.CreateExtentAttr([Gf.Vec3f(*map(float,v.min(0))),Gf.Vec3f(*map(float,v.max(0)))])
        if normals is not None:
            mesh.CreateNormalsAttr(np.asarray(normals,dtype=float).tolist())
            mesh.SetNormalsInterpolation('vertex')
        if uv is not None:
            UsdGeom.PrimvarsAPI(mesh).CreatePrimvar('st',Sdf.ValueTypeNames.TexCoord2fArray,'vertex').Set(np.asarray(uv,dtype=float).tolist())
        self.bind(mesh.GetPrim(),material)
        return mesh.GetPrim()

    def collider_box(self,path,size):
        c=UsdGeom.Cube.Define(self.stage,path+'/Collider');c.CreateSizeAttr(1.)
        self.xform(c.GetPrim(),(0,0,0),scale=size)
        UsdGeom.Imageable(c).CreateVisibilityAttr().Set('invisible')
        UsdPhysics.CollisionAPI.Apply(c.GetPrim())

    def box(self,name,center,size,material,bevel=.002,rotation=(0,0,0),collision=True):
        size=np.asarray(size,float)
        if len(size)!=3 or min(size)<=0:raise ValueError((name,size))
        path=self.path(name);root=UsdGeom.Xform.Define(self.stage,path)
        self.xform(root.GetPrim(),center,rotation)
        h=size/2;r=min(float(bevel),min(h)*.48)
        verts=[];normals=[];uv=[];faces=[]
        for axis in range(3):
            u=(axis+1)%3;v=(axis+2)%3
            for sign in (-1,1):
                # Sample the bevel arcs while retaining a single planar face centre.
                coords=lambda a: sorted(set([-h[a],-h[a]+r*.293,-h[a]+r,-h[a]+2*r,h[a]-2*r,h[a]-r,h[a]-r*.293,h[a]])) if r else [-h[a],h[a]]
                cu,cv=coords(u),coords(v);start=len(verts)
                for bv in cv:
                    for au in cu:
                        p=np.zeros(3);p[axis]=sign*h[axis];p[u]=au;p[v]=bv
                        q=np.clip(p,-h+r,h-r);d=p-q
                        n=d/max(np.linalg.norm(d),1e-10)
                        if r: p=q+n*r
                        else:n=np.eye(3)[axis]*sign
                        verts.append(p);normals.append(n);uv.append([(au+h[u])*2.,(bv+h[v])*2.])
                for j in range(len(cv)-1):
                    for i in range(len(cu)-1):
                        a=start+j*len(cu)+i;face=[a,a+1,a+len(cu)+1,a+len(cu)]
                        faces.append(face if sign>0 else face[::-1])
        self.mesh(path+'/Visual',verts,faces,material,uv,normals)
        if collision:self.collider_box(path,size)
        self.records.append({'name':name,'kind':'box','center':list(center),'size':size.tolist(),'collision':collision,'rotation':list(rotation)})
        return path

    def cylinder(self,name,center,radius,height,material,rotation=(0,0,0),collision=True):
        path=self.path(name);root=UsdGeom.Xform.Define(self.stage,path);self.xform(root.GetPrim(),center,rotation)
        c=UsdGeom.Cylinder.Define(self.stage,path+'/Visual');c.CreateRadiusAttr(float(radius));c.CreateHeightAttr(float(height));c.CreateAxisAttr('Z');self.bind(c.GetPrim(),material)
        if collision:UsdPhysics.CollisionAPI.Apply(c.GetPrim())
        return path

    def sphere(self,name,center,scale,material,collision=False):
        path=self.path(name);s=UsdGeom.Sphere.Define(self.stage,path);s.CreateRadiusAttr(1.)
        if isinstance(scale,(int,float)):scale=(scale,)*3
        self.xform(s.GetPrim(),center,scale=scale);self.bind(s.GetPrim(),material)
        if collision:UsdPhysics.CollisionAPI.Apply(s.GetPrim())
        return path

    def lathe(self,name,center,profile,material,rotation=(0,0,0),collision=False):
        path=self.path(name);root=UsdGeom.Xform.Define(self.stage,path);self.xform(root.GetPrim(),center,rotation)
        n=96;profile=list(profile);verts=[];uv=[];faces=[]
        for j,(radius,z) in enumerate(profile):
            for i in range(n):
                a=2*math.pi*i/n;verts.append((radius*math.cos(a),radius*math.sin(a),z));uv.append((i/(n-1),j/max(1,len(profile)-1)))
        for j in range(len(profile)):
            k=(j+1)%len(profile)
            for i in range(n):
                a,c,d,e=j*n+i,j*n+(i+1)%n,k*n+(i+1)%n,k*n+i
                for tri in ((a,c,d),(a,d,e)):
                    p0,p1,p2=(np.asarray(verts[t]) for t in tri)
                    if np.linalg.norm(np.cross(p1-p0,p2-p0))>1.e-12: faces.append(tri)
        # Smooth revolved faces; zero-radius centre triangles stay nondegenerate.
        va=np.asarray(verts);norm=np.zeros_like(va)
        for face in faces:
            aa,bb,cc=face;fn=np.cross(va[bb]-va[aa],va[cc]-va[aa])
            for vi in face:norm[vi]+=fn
        norm/=np.maximum(np.linalg.norm(norm,axis=1,keepdims=True),1.e-12)
        p=self.mesh(path+'/Visual',verts,faces,material,uv,norm)
        if collision:
            UsdPhysics.CollisionAPI.Apply(p);UsdPhysics.MeshCollisionAPI.Apply(p).CreateApproximationAttr('none')
        return path

    def tube(self,name,points,radius,material,collision=False):
        points=np.asarray(points,float);n=16;verts=[];faces=[];normals=[]
        for j,p in enumerate(points):
            t=points[min(j+1,len(points)-1)]-points[max(0,j-1)];t/=np.linalg.norm(t)
            ref=np.array((0,0,1.)) if abs(t[2])<.95 else np.array((1.,0,0))
            u=np.cross(t,ref);u/=np.linalg.norm(u);v=np.cross(t,u)
            for k in range(n):
                a=2*math.pi*k/n;d=u*math.cos(a)+v*math.sin(a);verts.append(p+radius*d);normals.append(d)
        for j in range(len(points)-1):
            for k in range(n):faces.append((j*n+k,j*n+(k+1)%n,(j+1)*n+(k+1)%n,(j+1)*n+k))
        path=self.path(name);p=self.mesh(path,verts,faces,material,normals=normals)
        if collision:UsdPhysics.CollisionAPI.Apply(p);UsdPhysics.MeshCollisionAPI.Apply(p).CreateApproximationAttr('none')
        return path

    def torus(self,name,center,major_radius,minor_radius,material,rotation=(0,0,0),collision=False):
        pts=np.array([(major_radius*math.cos(a),major_radius*math.sin(a),0) for a in np.linspace(0,2*math.pi,97)])
        path=self.tube(name,pts,minor_radius,material,collision)
        self.xform(self.stage.GetPrimAtPath(path),center,rotation)
        return path

    def label(self,name,text,center,size,material='paper',rotation=(0,0,0),font_size=80):
        from PIL import Image,ImageDraw,ImageFont
        width,height=size[:2];depth=size[2] if len(size)>2 else .00025
        font=self.assets_root/'fonts/noto_cjk_sc/v1/NotoSansCJKsc-Regular.otf'
        im=Image.new('RGB',(max(128,int(1024*width/max(width,height))),max(128,int(1024*height/max(width,height)))),(235,231,216))
        dr=ImageDraw.Draw(im);ft=ImageFont.truetype(str(font),font_size)
        bbox=dr.multiline_textbbox((0,0),text,font=ft,align='center',spacing=6)
        if bbox[2]>im.width*.9 or bbox[3]>im.height*.9:
            factor=min(im.width*.9/max(1,bbox[2]),im.height*.9/max(1,bbox[3]));ft=ImageFont.truetype(str(font),max(8,int(font_size*factor)))
        dr.multiline_text((im.width/2,im.height/2),text,font=ft,fill=(37,43,43),anchor='mm',align='center',spacing=6)
        key=hashlib.sha256((text+str(size)).encode()).hexdigest()[:16];p=self.generated/(key+'.png')
        if not p.exists():im.save(p)
        mat=self.material('label_'+key,(1,1,1),.75,texture=str(p))
        path=self.path(name);root=UsdGeom.Xform.Define(self.stage,path);self.xform(root.GetPrim(),center,rotation)
        self.mesh(path+'/Print',[(-width/2,-height/2,depth/2),(width/2,-height/2,depth/2),(width/2,height/2,depth/2),(-width/2,height/2,depth/2)],[(0,1,2,3)],mat,[(0,0),(1,0),(1,1),(0,1)])
        return path

    def asset(self,name,path,position,scale=1.,rotation=(0,0,0),ground=True):
        source=Path(path);self.dependency(source)
        if source.suffix.lower() in ('.gltf','.glb','.json'):
            return self.mesh_asset(name,source,position,scale,rotation,ground)
        p=self.path(name);root=UsdGeom.Xform.Define(self.stage,p)
        child=self.stage.DefinePrim(p+'/Asset','Xform');child.GetReferences().AddReference(str(source))
        self.xform(root.GetPrim(),(0,0,0),rotation,scale=(scale,)*3 if isinstance(scale,(int,float)) else scale)
        for prim in Usd.PrimRange(child):
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):UsdPhysics.RigidBodyAPI(prim).CreateRigidBodyEnabledAttr(False)
            if prim.IsA(UsdPhysics.Joint):UsdPhysics.Joint(prim).CreateJointEnabledAttr(False)
            if prim.IsA(UsdGeom.Mesh) and not prim.HasAPI(UsdPhysics.CollisionAPI):
                UsdPhysics.CollisionAPI.Apply(prim);UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr('none')
        cache=UsdGeom.BBoxCache(Usd.TimeCode.Default(),['default','render'],useExtentsHint=False)
        bbox=cache.ComputeWorldBound(root.GetPrim()).ComputeAlignedRange()
        lo=np.array(bbox.GetMin());hi=np.array(bbox.GetMax());pos=np.array(position,float)
        if ground:pos-=np.array(((lo[0]+hi[0])/2,(lo[1]+hi[1])/2,lo[2]))
        self.xform(root.GetPrim(),pos,rotation,scale=(scale,)*3 if isinstance(scale,(int,float)) else scale)
        self.records.append({'name':name,'kind':'asset','source':str(source),'position':list(position),'bounds_size':(hi-lo).tolist(),'static_scenery':True})
        return p

    def mesh_asset(self,name,source,position,scale,rotation,ground):
        # Legacy prepared NPZ mesh manifests; native USD conversions use asset().
        manifest=source if source.suffix=='.json' else source.parent/'converted'/'manifest.json'
        if not manifest.exists():raise FileNotFoundError(f'Missing prepared mesh manifest for {source}: {manifest}')
        data=load_json(manifest);self.dependency(manifest)
        path=self.path(name);root=UsdGeom.Xform.Define(self.stage,path)
        for index,item in enumerate(data['meshes']):
            self.dependency(manifest.parent/item['npz'])
            z=np.load(manifest.parent/item['npz']);m=item['material']
            mat=self.material(safe(name)+'_'+str(index),m['color'],m.get('roughness',.5),m.get('metallic',0),texture=m.get('texture'),normal=m.get('normal'))
            self.mesh(path+'/Part'+str(index),z['vertices'],z['faces'],mat,z['uv'] if 'uv' in z else None,z['normals'] if 'normals' in z else None)
        self.xform(root.GetPrim(),(0,0,0),rotation,scale=(scale,)*3 if isinstance(scale,(int,float)) else scale)
        bbox=UsdGeom.BBoxCache(Usd.TimeCode.Default(),['default','render']).ComputeWorldBound(root.GetPrim()).ComputeAlignedRange();lo=np.array(bbox.GetMin());hi=np.array(bbox.GetMax());pos=np.array(position,float)
        if ground:pos-=np.array(((lo[0]+hi[0])/2,(lo[1]+hi[1])/2,lo[2]))
        self.xform(root.GetPrim(),pos,rotation,scale=(scale,)*3 if isinstance(scale,(int,float)) else scale)
        # Static triangle meshes preserve cavities and outlines of scenery.
        for prim in Usd.PrimRange(root.GetPrim()):
            if prim.IsA(UsdGeom.Mesh):UsdPhysics.CollisionAPI.Apply(prim);UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr('none')
        self.records.append({'name':name,'kind':'mesh_asset','source':str(source),'position':list(position),'bounds_size':(hi-lo).tolist(),'static_scenery':True})
        return path


def build_scene(stage,family,assets_root,output_root,*,lighting_profile='nominal'):
    if family not in FAMILIES:raise ValueError(family)
    from .common import build_room,build_workcell,apply_lighting_profile
    from .interiors import build_interior
    from .props import build_props
    from .layouts import ROOM_SPECS,cameras_for
    b=Builder(stage,assets_root,output_root)
    b.family=family
    build_room(b,family)
    build_workcell(b,family)
    build_interior(b,family)
    objects=build_props(b,family)
    lighting=apply_lighting_profile(stage,family,lighting_profile)
    robot=Path(assets_root)/'robots/assemblies/fr3_sharpa/isaacsim_6_1_v2/usd/fr3_sharpa/fr3_sharpa.usda'
    b.dependency(robot)
    rp=stage.DefinePrim('/World/Robot','Xform');rp.GetReferences().AddReference(str(robot))
    b.xform(rp,(-.13,0,.78))
    return {'family':family,'robot_root':'/World/Robot','robot_base_position':[-.13,0,.78],
            'robot_joint_positions':dict(zip([f'fr3_joint{i}' for i in range(1,8)],[0,-.55,0,-2.15,0,1.65,.75])),
            'cameras':cameras_for(family),
            'layout':{'table_center_xy':[.65,0],'table_size_m':[1,1],'table_top_z':.78,'robot_base_xyz':[-.13,0,.78],'pedestal_footprint_xy':[.36,.4],'pedestal_center_xy':[-.13,0],'table_robot_gap_m':.1,'room':ROOM_SPECS[family]},
            'lighting':lighting,
            'objects':objects,'geometry':b.records,'dependencies':b.dependencies,'scope':'scene construction only; static dressing; no manipulation qualification'}
