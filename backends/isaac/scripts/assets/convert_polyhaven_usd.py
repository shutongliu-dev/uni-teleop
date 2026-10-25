"""Preserve glTF mesh topology, hierarchy, normals, UVs and PBR maps in native USD.

CPU only. Source glTF stays unchanged. World convention is meters, Z up.
"""
from pathlib import Path
import base64, datetime, hashlib, json, sys
import numpy as np

from bench.paths import DATA_ROOT, load_json
sys.path.append(f'{DATA_ROOT}/assets/tools/usd/26.8/venv/lib/python3.12/site-packages')
from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade, Vt

SOURCE = Path(f'{DATA_ROOT}/assets/objects/polyhaven/20261016')
DEST = Path(f'{DATA_ROOT}/assets/objects/polyhaven/derived/v1')
DTYPES = {5120:np.int8,5121:np.uint8,5122:np.int16,5123:np.uint16,5125:np.uint32,5126:np.float32}
COMPS={'SCALAR':1,'VEC2':2,'VEC3':3,'VEC4':4,'MAT4':16}

def convert(path):
    doc=load_json(path);slug=path.stem.removesuffix('_2k')
    folder=DEST/slug;folder.mkdir(parents=True,exist_ok=True)
    target=folder/(slug+'.usdc')
    if target.exists():raise FileExistsError('Do not overwrite versioned derived USD: '+str(target))
    buffers=[(path.parent/v['uri']).read_bytes() for v in doc['buffers']]
    def arr(index):
        a=doc['accessors'][index];v=doc['bufferViews'][a['bufferView']]
        dtype=np.dtype(DTYPES[a['componentType']]).newbyteorder('<');n=COMPS[a['type']]
        offset=v.get('byteOffset',0)+a.get('byteOffset',0)
        stride=v.get('byteStride',dtype.itemsize*n)
        result=np.ndarray((a['count'],n),dtype=dtype,buffer=buffers[v['buffer']],offset=offset,strides=(stride,dtype.itemsize)).copy()
        if a.get('normalized'):
            lim=np.iinfo(dtype);result=np.clip(result.astype(float)/lim.max,-1,1)
        return result
    stage=Usd.Stage.CreateNew(str(target));UsdGeom.SetStageMetersPerUnit(stage,1);UsdGeom.SetStageUpAxis(stage,UsdGeom.Tokens.z)
    root=UsdGeom.Xform.Define(stage,'/Asset');root.AddRotateXOp().Set(90)
    stage.SetDefaultPrim(root.GetPrim())
    root.GetPrim().SetCustomDataByKey('source','https://polyhaven.com/a/'+slug)
    root.GetPrim().SetCustomDataByKey('license','CC0-1.0')
    materials=[];tex_records=[]
    for mi,m in enumerate(doc.get('materials',[])):
        mp='/Asset/Looks/material_'+str(mi);mat=UsdShade.Material.Define(stage,mp)
        sh=UsdShade.Shader.Define(stage,mp+'/Surface');sh.CreateIdAttr('UsdPreviewSurface')
        pbr=m.get('pbrMetallicRoughness',{});color=pbr.get('baseColorFactor',[1,1,1,1])
        sh.CreateInput('diffuseColor',Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color[:3]))
        sh.CreateInput('metallic',Sdf.ValueTypeNames.Float).Set(pbr.get('metallicFactor',1))
        sh.CreateInput('roughness',Sdf.ValueTypeNames.Float).Set(pbr.get('roughnessFactor',1))
        sh.CreateInput('opacity',Sdf.ValueTypeNames.Float).Set(color[3])
        if m.get('alphaMode')=='MASK':sh.CreateInput('opacityThreshold',Sdf.ValueTypeNames.Float).Set(m.get('alphaCutoff',.5))
        reader=UsdShade.Shader.Define(stage,mp+'/UV');reader.CreateIdAttr('UsdPrimvarReader_float2');reader.CreateInput('varname',Sdf.ValueTypeNames.Token).Set('st');reader.CreateOutput('result',Sdf.ValueTypeNames.Float2)
        def tex(info,name,srgb=False,scale=None,bias=None):
            ti=doc['textures'][info['index']];im=doc['images'][ti['source']];file=(path.parent/im['uri']).resolve()
            if not file.is_file():raise FileNotFoundError(file)
            shader=UsdShade.Shader.Define(stage,mp+'/'+name);shader.CreateIdAttr('UsdUVTexture')
            shader.CreateInput('file',Sdf.ValueTypeNames.Asset).Set(str(file));shader.CreateInput('sourceColorSpace',Sdf.ValueTypeNames.Token).Set('sRGB' if srgb else 'raw')
            shader.CreateInput('st',Sdf.ValueTypeNames.Float2).ConnectToSource(reader.ConnectableAPI(),'result')
            sampler=doc.get('samplers',[])[ti['sampler']] if 'sampler' in ti else {}
            for attr,key in [('wrapS','wrapS'),('wrapT','wrapT')]:shader.CreateInput(attr,Sdf.ValueTypeNames.Token).Set({33071:'clamp',33648:'mirror',10497:'repeat'}.get(sampler.get(key,10497),'repeat'))
            if scale is not None:shader.CreateInput('scale',Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(*scale))
            if bias is not None:shader.CreateInput('bias',Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(*bias))
            shader.CreateOutput('rgb',Sdf.ValueTypeNames.Float3)
            for c in 'rgba':shader.CreateOutput(c,Sdf.ValueTypeNames.Float)
            tex_records.append(str(file));return shader
        if 'baseColorTexture' in pbr:
            t=tex(pbr['baseColorTexture'],'BaseColor',True,scale=color)
            sh.CreateInput('diffuseColor',Sdf.ValueTypeNames.Color3f).ConnectToSource(t.ConnectableAPI(),'rgb')
            if m.get('alphaMode','OPAQUE')!='OPAQUE':sh.CreateInput('opacity',Sdf.ValueTypeNames.Float).ConnectToSource(t.ConnectableAPI(),'a')
        if 'metallicRoughnessTexture' in pbr:
            t=tex(pbr['metallicRoughnessTexture'],'MetalRough',scale=[1,pbr.get('roughnessFactor',1),pbr.get('metallicFactor',1),1])
            sh.CreateInput('roughness',Sdf.ValueTypeNames.Float).ConnectToSource(t.ConnectableAPI(),'g')
            sh.CreateInput('metallic',Sdf.ValueTypeNames.Float).ConnectToSource(t.ConnectableAPI(),'b')
        if 'normalTexture' in m:
            strength=m['normalTexture'].get('scale',1);t=tex(m['normalTexture'],'Normal',scale=[2*strength,2*strength,2,1],bias=[-strength,-strength,-1,0])
            sh.CreateInput('normal',Sdf.ValueTypeNames.Normal3f).ConnectToSource(t.ConnectableAPI(),'rgb')
        if 'occlusionTexture' in m:
            t=tex(m['occlusionTexture'],'Occlusion');sh.CreateInput('occlusion',Sdf.ValueTypeNames.Float).ConnectToSource(t.ConnectableAPI(),'r')
        if 'emissiveFactor' in m:sh.CreateInput('emissiveColor',Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*m['emissiveFactor']))
        if 'emissiveTexture' in m:
            t=tex(m['emissiveTexture'],'Emissive',True);sh.CreateInput('emissiveColor',Sdf.ValueTypeNames.Color3f).ConnectToSource(t.ConnectableAPI(),'rgb')
        mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(),'surface');materials.append(mat)
    counts={'meshes':0,'triangles':0,'vertices':0,'material_bindings':0,'normal_meshes':0,'uv_meshes':0}
    def node(index,parent):
        n=doc['nodes'][index];npth=parent+'/node_'+str(index);x=UsdGeom.Xform.Define(stage,npth)
        x.GetPrim().SetDisplayName(n.get('name','node_'+str(index)))
        if 'matrix' in n:matrix=Gf.Matrix4d(*[float(v) for v in n['matrix']])
        else:
            t=Gf.Matrix4d(1).SetTranslate(Gf.Vec3d(*n.get('translation',[0,0,0])))
            q=n.get('rotation',[0,0,0,1]);r=Gf.Matrix4d(1).SetRotate(Gf.Quatd(q[3],Gf.Vec3d(*q[:3])))
            s=Gf.Matrix4d(1).SetScale(Gf.Vec3d(*n.get('scale',[1,1,1])));matrix=s*r*t
        x.AddTransformOp().Set(matrix)
        if 'mesh' in n:
            for pi,p in enumerate(doc['meshes'][n['mesh']]['primitives']):
                if p.get('mode',4)!=4:raise ValueError('Only triangle primitives supported')
                mesh=UsdGeom.Mesh.Define(stage,npth+'/mesh_'+str(pi));pos=arr(p['attributes']['POSITION']).astype(np.float32)
                indices=arr(p['indices']).reshape(-1).astype(np.int32) if 'indices' in p else np.arange(len(pos),dtype=np.int32)
                if len(indices)%3:raise ValueError('Nontriangle index count')
                mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(pos));mesh.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(len(indices)//3,3,np.int32)));mesh.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(indices));mesh.CreateSubdivisionSchemeAttr('none')
                mesh.CreateExtentAttr(Vt.Vec3fArray.FromNumpy(np.asarray([pos.min(axis=0),pos.max(axis=0)],np.float32)))
                if 'NORMAL' in p['attributes']:
                    mesh.CreateNormalsAttr(Vt.Vec3fArray.FromNumpy(arr(p['attributes']['NORMAL']).astype(np.float32)));mesh.SetNormalsInterpolation('vertex');counts['normal_meshes']+=1
                if 'TEXCOORD_0' in p['attributes']:
                    uv=arr(p['attributes']['TEXCOORD_0']).astype(np.float32);uv[:,1]=1-uv[:,1]
                    UsdGeom.PrimvarsAPI(mesh).CreatePrimvar('st',Sdf.ValueTypeNames.TexCoord2fArray,'vertex').Set(Vt.Vec2fArray.FromNumpy(uv));counts['uv_meshes']+=1
                if 'material' in p:
                    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(materials[p['material']]);counts['material_bindings']+=1
                    mesh.CreateDoubleSidedAttr(doc['materials'][p['material']].get('doubleSided',False))
                counts['meshes']+=1;counts['triangles']+=len(indices)//3;counts['vertices']+=len(pos)
        for child in n.get('children',[]):node(child,npth)
    for ni in doc['scenes'][doc.get('scene',0)]['nodes']:node(ni,'/Asset')
    stage.GetRootLayer().Save()
    reopened=Usd.Stage.Open(str(target));bbox=UsdGeom.BBoxCache(Usd.TimeCode.Default(),[UsdGeom.Tokens.default_,UsdGeom.Tokens.render]).ComputeWorldBound(reopened.GetDefaultPrim()).ComputeAlignedBox()
    bounds=[list(bbox.GetMin()),list(bbox.GetMax())]
    result={'schema':'polyhaven_usd_derivation.v1','source':str(path),'source_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'source_provenance':str(path.parent.parent/'SOURCE.json'),'output':str(target),'output_sha256':hashlib.sha256(target.read_bytes()).hexdigest(),'converter':str(Path(__file__).resolve()),'converter_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'axis_conversion':'glTF Y-up meters to USD Z-up meters, +90 degrees around X at root','uv_conversion':'glTF top-left UV to USD bottom-left UV: v -> 1-v','physics':'visual mesh only; static colliders belong to scene integration','license':'CC0-1.0','bounds_min_max_m':bounds,'dimensions_xyz_m':(np.array(bounds[1])-np.array(bounds[0])).tolist(),'counts':counts,'texture_files':sorted(set(tex_records)),'texture_sha256':{v:hashlib.sha256(Path(v).read_bytes()).hexdigest() for v in set(tex_records)}}
    (folder/'DERIVATION.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'asset':slug,'output':str(target),'dimensions':result['dimensions_xyz_m'],'counts':counts}),flush=True)
    return result
if __name__=='__main__':
    DEST.mkdir(parents=True,exist_ok=True)
    results=[convert(p) for p in SOURCE.glob('*/gltf_2k/*.gltf')]
    (DEST/'COLLECTION.json').write_text(json.dumps({'schema':'polyhaven_usd_collection.v1','assets':results},indent=2)+'\n')
