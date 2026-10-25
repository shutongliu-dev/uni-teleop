"""Nonphysical tool finishes for the fixed SHARPA v3 and gear-tool v2 shapes.

Author into a separate overlay layer. Existing working heads, collisions, mass,
body transforms and source assets are never rebuilt. New meshes stay inside the
source envelopes; cylindrical grasp regions retain the original 64-sided radii.
"""
from __future__ import annotations

import math
import numpy as np


MATERIAL_KEYS = ('steel_brushed', 'steel_polished', 'rubber_graphite', 'polymer_teal', 'marking_light')
UV_METERS_PER_REPEAT = .05


def _mesh(stage, path, points, faces, material, *, normals=None, uv=None):
    from pxr import Sdf, UsdGeom, UsdShade
    points = np.asarray(points, dtype=float)
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr(points.tolist())
    mesh.CreateFaceVertexCountsAttr([len(face) for face in faces])
    mesh.CreateFaceVertexIndicesAttr([int(i) for face in faces for i in face])
    mesh.CreateSubdivisionSchemeAttr('none')
    mesh.CreateExtentAttr([points.min(axis=0).tolist(), points.max(axis=0).tolist()])
    if normals is None:
        normals, uv = [], []
        for face in faces:
            p = points[face]
            n = np.cross(p[1]-p[0], p[2]-p[0]); n /= np.linalg.norm(n)
            axes = [i for i in range(3) if i != int(np.argmax(abs(n)))]
            normals.extend([n.tolist()]*len(face))
            uv.extend((p[:, axes]/UV_METERS_PER_REPEAT).tolist())
    mesh.CreateNormalsAttr(normals)
    mesh.SetNormalsInterpolation('faceVarying')
    UsdGeom.PrimvarsAPI(mesh).CreatePrimvar('st', Sdf.ValueTypeNames.TexCoord2fArray,
                                          UsdGeom.Tokens.faceVarying).Set(uv)
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)
    mesh.GetPrim().SetCustomDataByKey('mechanical_visual_only', True)
    return str(mesh.GetPath())


def _lathe(stage, root, profile, band_materials, materials, *, center=(0., 0.), prefix='grip'):
    """64-sided radius envelope, smooth azimuth; hard normals at band boundaries."""
    paths = []
    n = 64
    for band, ((z0, r0), (z1, r1)) in enumerate(zip(profile, profile[1:])):
        points, normals, uv = [], [], []
        faces = []
        for j in range(n):
            a0, a1 = 2*math.pi*j/n, 2*math.pi*(j+1)/n
            face = []
            for angle, radius, z in ((a0,r0,z0),(a1,r0,z0),(a1,r1,z1),(a0,r1,z1)):
                face.append(len(points))
                points.append([center[0]+radius*math.cos(angle), center[1]+radius*math.sin(angle), z])
                normal = np.array([math.cos(angle), math.sin(angle), -(r1-r0)/(z1-z0)])
                normals.append((normal/np.linalg.norm(normal)).tolist())
                uv.append([angle*max(r0,r1)/UV_METERS_PER_REPEAT, z/UV_METERS_PER_REPEAT])
            faces.append(face)
        paths.append(_mesh(stage, f'{root}/{prefix}_{band:02d}', points, faces,
                           materials[band_materials[band]], normals=normals, uv=uv))
    for label, (z, radius), material, reverse in (
            ('bottom', profile[0], band_materials[0], True),
            ('top', profile[-1], band_materials[-1], False)):
        points = [[*center,z]]+[[center[0]+radius*math.cos(j*2*math.pi/n),
                                  center[1]+radius*math.sin(j*2*math.pi/n),z] for j in range(n)]
        faces = [[0,1+j,1+(j+1)%n] for j in range(n)]
        if reverse: faces = [f[::-1] for f in faces]
        paths.append(_mesh(stage,f'{root}/{prefix}_{label}',points,faces,materials[material]))
    return paths


def _bevel_box(stage, path, center, size, bevel, material):
    import trimesh
    half = np.asarray(size)/2
    corners = []
    for x in (-1.,1.):
        for y in (-1.,1.):
            for z in (-1.,1.):
                signs = np.array([x,y,z])
                for axis in range(3):
                    p = half.copy(); p[axis] -= bevel
                    corners.append(np.asarray(center)+signs*p)
    hull = trimesh.convex.convex_hull(corners)
    return _mesh(stage,path,hull.vertices,hull.faces,material)


def _outline(x0,x1,y0,y1,cut):
    return np.array([[x0+cut,y0],[x1-cut,y0],[x1,y0+cut],[x1,y1-cut],
                     [x1-cut,y1],[x0+cut,y1],[x0,y1-cut],[x0,y0+cut]])


def _wrench_handle(stage, root, materials):
    # Original handle envelope: X[-190,-40], Y[-16,16], Z[0,12] mm.
    # The shallow face pocket is entirely inside it; it is not a physical hole.
    outer = _outline(-.190,-.040,-.016,.016,.0012)
    bevel = _outline(-.1893,-.0407,-.0153,.0153,.0008)
    panel = _outline(-.181,-.050,-.010,.010,.0015)
    rings = [(bevel,0.),(outer,.0007),(outer,.0113),(bevel,.012),
             (panel,.012),(panel,.0118)]
    paths=[]
    for i,((low,z0),(high,z1)) in enumerate(zip(rings,rings[1:])):
        points=np.vstack([np.c_[low,np.full(8,z0)],np.c_[high,np.full(8,z1)]])
        faces=[[j,(j+1)%8,(j+1)%8+8,j+8] for j in range(8)]
        paths.append(_mesh(stage,f'{root}/forged_handle_band_{i}',points,faces,
                           materials['steel_brushed' if i==1 else 'steel_polished']))
    for label, polygon, z, reverse in [('base',bevel,0.,True),('face',panel,.0118,False)]:
        points=np.vstack([np.c_[polygon,np.full(8,z)],np.r_[polygon.mean(axis=0),z]])
        faces=[[8,j,(j+1)%8] for j in range(8)]
        if reverse:faces=[f[::-1] for f in faces]
        paths.append(_mesh(stage,f'{root}/forged_handle_{label}',points,faces,materials['steel_brushed']))
    # A small factual 58 size stamp lies within the recessed non-working face.
    segments={'5':(0,2,3,4,6),'8':tuple(range(7))}
    bars=[(.0003,.0042,.0021,.00025),(.00215,.00225,.00025,.0019),
          (0.,.00225,.00025,.0019),(.0003,.0020,.0021,.00025),
          (.00215,0.,.00025,.0019),(0.,0.,.00025,.0019),(.0003,-.00025,.0021,.00025)]
    for digit,ch in enumerate('58'):
        for seg in segments[ch]:
            x,y,w,h=bars[seg];x += -.066+digit*.0032;y -= .002
            points=[[x,y,.011802],[x+w,y,.011802],[x+w,y+h,.011802],[x,y+h,.011802]]
            paths.append(_mesh(stage,f'{root}/size_{digit}_{seg}',points,[[0,1,2,3]],materials['marking_light']))
    return paths


def author_tool_details(stage, object_root, role, *, materials):
    """Add tool visuals in the current edit layer; return an auditable receipt.

    ``object_root`` is the metre-scale body prim (before its world transform).
    ``role`` is screwdriver, straight_hex_screwdriver, socket_crank or
    open_end_wrench. ``materials`` maps MATERIAL_KEYS to UsdShade.Material or
    absolute prim paths. UV coordinates use one texture repeat per 50 mm.
    This function requires the existing source visual paths and rejects reuse;
    call once in a fresh visual overlay. No simulation or controller imports.
    """
    from pxr import Gf, Usd, UsdGeom, UsdPhysics, UsdShade
    if role not in ('screwdriver','straight_hex_screwdriver','socket_crank','open_end_wrench'):
        raise ValueError(f'Unsupported fixed tool role: {role}')
    root = stage.GetPrimAtPath(str(object_root))
    if not root or not root.IsA(UsdGeom.Xform):
        raise ValueError('Tool body must be an existing Xform')
    base=str(root.GetPath())
    detail=base+'/ToolVisualDetails'
    if stage.GetPrimAtPath(detail):
        raise ValueError('Tool visual details already exist; use a fresh overlay')
    checked={}
    for key in MATERIAL_KEYS:
        value=materials[key]
        material=value if isinstance(value,UsdShade.Material) else UsdShade.Material.Get(stage,str(value))
        if not material or material.GetPrim().GetStage()!=stage:
            raise ValueError(f'Missing material in the tool stage: {key}')
        checked[key]=material
    if role in ('screwdriver','straight_hex_screwdriver'):
        anchors=[p for p in Usd.PrimRange(root) if p.GetChild('Visuals')
                 and p.GetChild('Visuals').GetChild('handle')]
    else:
        anchors=[p for p in Usd.PrimRange(root) if p.GetChild('visual_000')]
    if len(anchors)!=1:
        raise ValueError('Expected one unambiguous source visual anchor inside the tool body')
    source=str(anchors[0].GetPath())
    if role in ('screwdriver','straight_hex_screwdriver'):
        replaced=[source+'/Visuals/handle']
        retained=[source+'/Visuals/hex_bit',source+'/Visuals/shaft']
        expected_bounds=([-.022,-.022,.101],[.022,.022,.211])
    elif role=='socket_crank':
        replaced=[source+f'/visual_{i:03d}' for i in range(54,59)]
        retained=[source+f'/visual_{i:03d}' for i in range(54)]
        expected_bounds=([-.087,-.017,.032],[-.030,.017,.154])
    else:
        replaced=[source+'/visual_003']
        retained=[source+f'/visual_{i:03d}' for i in range(3)]
        expected_bounds=([-.190,-.016,0.],[-.040,.016,.012])
    for path in replaced+retained:
        prim=stage.GetPrimAtPath(path)
        if not prim or not prim.IsA(UsdGeom.Mesh) or prim.HasAPI(UsdPhysics.CollisionAPI):
            raise ValueError(f'Expected a separate noncolliding source visual mesh: {path}')
    # A foreign scale or source cannot silently be dressed with these dimensions.
    cache=UsdGeom.XformCache()
    actual=[]
    for path in replaced:
        prim=stage.GetPrimAtPath(path)
        relative,reset=cache.ComputeRelativeTransform(prim,root)
        if reset:
            raise ValueError('Source visual transform resets outside its tool body')
        actual.extend(list(relative.Transform(Gf.Vec3d(*map(float,p))))
                      for p in UsdGeom.Mesh(prim).GetPointsAttr().Get())
    lo=np.min(actual,axis=0);hi=np.max(actual,axis=0)
    if not np.allclose([lo,hi],expected_bounds,rtol=0.,atol=2.e-7):
        raise ValueError('Source tool visual dimensions differ from the fixed v2/v3 contract')
    UsdGeom.Xform.Define(stage,detail)
    for path in replaced:
        UsdGeom.Imageable(stage.GetPrimAtPath(path)).CreateVisibilityAttr('invisible')
    for path in retained:
        UsdShade.MaterialBindingAPI.Apply(stage.GetPrimAtPath(path)).Bind(checked['steel_brushed'])
    paths=[]
    if role in ('screwdriver','straight_hex_screwdriver'):
        profile=[(.101,.0208),(.1022,.022),(.105,.022),(.106,.0217),(.108,.022),
                 (.113,.022),(.195,.022),(.199,.022),(.200,.0217),(.201,.022),
                 (.2085,.022),(.211,.0208)]
        bands=['steel_polished','steel_polished','rubber_graphite','rubber_graphite',
               'polymer_teal','rubber_graphite','polymer_teal','rubber_graphite',
               'rubber_graphite','polymer_teal','rubber_graphite']
        paths += _lathe(stage,detail,profile,bands,checked)
        dimensions={'handle_diameter_m':.044,'handle_z_m':[.101,.211],
                    'unchanged_working_head_z_m':[0.,.101]}
    elif role=='socket_crank':
        paths.append(_bevel_box(stage,detail+'/forged_lever',[-.0585,0.,.038],
                                [.057,.022,.012],.00065,checked['steel_brushed']))
        profile=[(.044,.0158),(.0452,.017),(.048,.017),(.050,.0168),(.053,.017),
                 (.058,.017),(.140,.017),(.145,.017),(.149,.017),(.154,.0158)]
        bands=['steel_polished','steel_polished','rubber_graphite','rubber_graphite',
               'polymer_teal','rubber_graphite','polymer_teal','rubber_graphite','polymer_teal']
        paths += _lathe(stage,detail,profile,bands,checked,center=(-.07,0.))
        dimensions={'handle_diameter_m':.034,'handle_z_m':[.044,.154],
                    'unchanged_socket_inner_across_flats_m':.058,'unchanged_roof_bore_diameter_m':.026}
    else:
        paths += _wrench_handle(stage,detail,checked)
        dimensions={'unchanged_jaw_across_flats_m':.058,'handle_width_m':.032,'thickness_m':.012}
    return {'schema':'mechanical_tool_visual_details.v1','role':role,'root':detail,
            'created_mesh_paths':paths,'hidden_source_visual_paths':replaced,
            'retained_working_visual_paths':retained,'dimensions':dimensions,
            'source_envelope_m':expected_bounds,'uv_meters_per_repeat':UV_METERS_PER_REPEAT,
            'physics_authored':False,'source_assets_modified':False,
            'scope':'Visual overlay only. Inward end chamfers/face recesses are rendering details; '
                    'working heads, grip maximum diameters, collisions and body dynamics remain unchanged.'}
