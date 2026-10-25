"""Detailed, real-scale tabletop dressing for the six scene rooms.

Original geometry is CC0. SimReady models remain in shared storage with their
original CC BY 4.0 licenses. Coordinates are metres; the 1 m table top is z=.78.
This module builds static scene dressing, not manipulation-ready rigid bodies.
Every item rests on the table or on a visible, explicitly modelled support.
"""
from __future__ import annotations

from math import cos, sin, pi
from pathlib import Path

from bench.paths import DATA_ROOT


TABLE_Z = 0.78
SIMREADY = Path(f'{DATA_ROOT}/assets/objects/simready/v1')


def _materials(b):
    specifications = {
        'prop_pages': ((.84, .81, .71), .86, 0),
        'prop_page_edge': ((.54, .52, .46), .91, 0),
        'prop_ivory': ((.82, .77, .64), .28, 0),
        'prop_cobalt': ((.025, .12, .20), .2, 0),
        'prop_sage': ((.22, .32, .22), .35, 0),
        'prop_terracotta': ((.38, .15, .073), .63, 0),
        'prop_brass': ((.56, .35, .12), .27, .8),
        'prop_silver': ((.55, .59, .60), .26, .9),
        'prop_book_red': ((.25, .055, .035), .62, 0),
        'prop_book_blue': ((.035, .085, .13), .62, 0),
        'prop_book_green': ((.09, .16, .10), .62, 0),
        'prop_leather': ((.055, .028, .015), .64, 0),
        'prop_coffee': ((.023, .011, .006), .14, 0),
        'prop_ink': ((.015, .02, .027), .58, 0),
        'prop_chess_dark': ((.045, .022, .012), .3, 0),
        'prop_chess_light': ((.69, .48, .26), .35, 0),
        'prop_felt': ((.035, .14, .095), .95, 0),
        'prop_soft_blue': ((.10, .26, .36), .5, 0),
        'prop_amber': ((.24, .105, .018), .18, 0),
        'prop_stone': ((.56, .57, .51), .83, 0),
        'prop_cork': ((.36, .23, .12), .92, 0),
        'prop_go_board': ((.67, .43, .20), .48, 0),
        'prop_go_black': ((.012, .014, .016), .23, 0),
        'prop_go_white': ((.89, .87, .81), .30, 0),
        'prop_copper': ((.52, .22, .09), .3, .85),
        'prop_tomato': ((.52, .025, .012), .32, 0),
        'prop_tomato_flesh': ((.68, .075, .038), .40, 0),
        'prop_tomato_seed': ((.73, .57, .26), .38, 0),
        'prop_leaf': ((.09, .21, .035), .66, 0),
        'prop_linen': ((.49, .46, .39), .93, 0),
    }
    for name, (color, roughness, metallic) in specifications.items():
        maps={}
        if name=='prop_go_board':
            wood=b.assets_root/'materials/ambientcg/v1/Wood089'
            for key,suffix in [('texture','Color'),('normal','NormalGL'),
                               ('roughness_texture','Roughness')]:
                source=wood/f'Wood089_1K-JPG_{suffix}.jpg'
                if source.is_file():maps[key]=str(source)
        if name=='prop_linen':
            weave=b.assets_root/'scenes/scence_workrooms/v1/textures/cotton_napkin_v2_normal.png'
            if weave.is_file():maps['normal']=str(weave)
        b.material(name, color, roughness=roughness, metallic=metallic,**maps)


def _label(b, name, text, x, y, z, w, h, rotation=0):
    b.label(name, text, (x, y, z), (w, h), rotation=(0, 0, rotation))


def _asset(b, name, rel, position, rotation=(0, 0, 0), scale=1.):
    p = SIMREADY / rel
    if not p.is_file():
        raise FileNotFoundError(f'Required licensed SimReady model is missing: {p}')
    b.asset(name, str(p), position, scale=scale, rotation=rotation, ground=True)
    return {'name': name, 'source': str(p), 'license': 'CC-BY-4.0',
            'license_file': str(p.parent / 'LICENSE.txt'), 'scale': scale,
            'support_z_m': position[2]}


def _cup(b, name, x, y, z=TABLE_Z, radius=.037, height=.085,
         material='prop_cobalt', handle=True):
    """Closed revolved ceramic shell; no filled-in cylinder masquerading as cup."""
    r, h = radius, height
    b.lathe(name + '_body', (x, y, z), [
        (0, .003), (r*.65, .003), (r*.82, 0), (r*.91, .004),
        (r*.95, .010), (r, h-.008), (r-.001, h-.002),
        (r-.003, h), (r-.005, h-.001), (r-.006, h-.007),
        (r-.008, .012), (r*.70, .008), (0, .008),
    ], material, collision=True)
    b.torus(name + '_foot', (x, y, z+.003), r*.73, .002, material)
    if handle:
        # Oval loop in the XZ plane, anchored into the real cup wall.
        points = [(x+r+.009+.023*cos(a), y,
                   z+h*.55+.028*sin(a)) for a in [2*pi*i/40 for i in range(41)]]
        b.tube(name + '_handle', points, .006, material, collision=True)
    return {'name': name, 'type': 'hollow_ceramic_cup', 'support_z_m': z}


def _bowl(b, name, x, y, z=TABLE_Z, radius=.075, height=.045,
          material='prop_ivory'):
    r, h = radius, height
    b.lathe(name, (x, y, z), [
        (0, .002), (r*.40, .002), (r*.45, 0), (r*.51, .003),
        (r*.62, h*.18), (r*.80, h*.48), (r*.95, h*.83),
        (r, h-.002), (r-.001, h), (r-.004, h),
        (r*.90, h*.79), (r*.76, h*.47), (r*.58, h*.22),
        (r*.40, .009), (0, .009),
    ], material, collision=True)
    b.torus(name+'_foot', (x,y,z+.002), r*.43, .002, material)


def _plate(b, name, x, y, z=TABLE_Z, radius=.095, material='prop_ivory'):
    b.lathe(name, (x,y,z), [(0,.002),(radius*.58,.002),(radius*.64,0),
        (radius*.72,.002),(radius*.94,.009),(radius,.012),
        (radius,.015),(radius*.95,.016),(radius*.70,.008),(0,.006)],
        material, collision=True)


def _book(b, name, x, y, z, width=.15, depth=.21, thickness=.026,
          cover='prop_book_blue', title='FIELD NOTES', subtitle='VOLUME 01'):
    # Three actual cover components, inset page block and printed thin page edges.
    cover_thickness = min(.003, thickness*.18)
    page_thickness = thickness-2*cover_thickness
    b.box(name+'_lower_cover',(x,y,z+cover_thickness/2),
          (width,depth,cover_thickness),cover,bevel=.001)
    b.box(name+'_pages',(x+.001,y,z+thickness/2),
          (width-.008,depth-.008,page_thickness),'prop_pages',bevel=.0008)
    b.box(name+'_upper_cover',(x,y,z+thickness-cover_thickness/2),
          (width,depth,cover_thickness),cover,bevel=.001)
    b.box(name+'_spine',(x-width/2+.002,y,z+thickness/2),
          (.004,depth,thickness-.002),cover,bevel=.0015)
    page_edges = min(16, max(3, int(page_thickness/.0009)))
    for i in range(1,page_edges+1):
        height=z+cover_thickness+page_thickness*i/(page_edges+1)
        b.box(f'{name}_foreedge_{i}', (x+width/2-.0038,y,height),
              (.00035,depth-.009,.00015),'prop_page_edge',bevel=0,collision=False)
        b.box(f'{name}_pagebottom_{i}', (x+.001,y-depth/2+.0038,height),
              (width-.009,.00035,.00015),'prop_page_edge',bevel=0,collision=False)
    _label(b,name+'_title',title,x,y+.025,z+thickness+.0002,width*.79,.042)
    _label(b,name+'_subtitle',subtitle,x,y-.04,z+thickness+.0002,width*.65,.013)
    for yy in (-depth*.36,depth*.36):
        b.box(name+('_band_low' if yy<0 else '_band_high'),
              (x,y+yy,z+thickness+.0001),(width*.80,.001,.0002),
              'prop_brass',bevel=0,collision=False)
    return {'name':name,'type':'hardback_book','support_z_m':z,'height_m':thickness}


def _tray(b,name,x,y,z,width,depth,height=.025,material='walnut'):
    b.box(name+'_floor',(x,y,z+.003),(width,depth,.006),material,bevel=.002)
    for side in (-1,1):
        b.box(f'{name}_long_{side}',(x,y+side*(depth/2-.003),z+height/2),
              (width,.006,height),material,bevel=.002)
        b.box(f'{name}_short_{side}',(x+side*(width/2-.003),y,z+height/2),
              (.006,depth-.008,height),material,bevel=.002)
    return z+.006


def _pen(b,name,x,y,z=TABLE_Z,length=.135,material='prop_book_blue'):
    r=.004
    b.cylinder(name+'_body',(x,y,z+r),r,length*.77,material,rotation=(0,90,0))
    b.lathe(name+'_nib',(x+length*.385,y,z+r),
        [(0,0),(.004,0),(.004,.008),(.001,.023),(0,.026)],
        'prop_silver',rotation=(0,90,0))
    b.cylinder(name+'_cap',(x-length*.41,y,z+r),.0045,length*.05,
               'prop_silver',rotation=(0,90,0))
    b.box(name+'_clip',(x-length*.24,y-.003,z+.008),
          (.032,.002,.0014),'prop_silver',bevel=.0006,collision=False)


def _bottle(b,name,x,y,z=TABLE_Z,height=.16,radius=.03,material='prop_amber',text='REAGENT'):
    r,h=radius,height
    b.lathe(name+'_body',(x,y,z),[(0,0),(r*.90,0),(r,.005),
        (r,h*.69),(r*.93,h*.75),(r*.45,h*.83),(r*.42,h*.96),
        (0,h*.96)],material,collision=True)
    b.cylinder(name+'_cap',(x,y,z+h*.96),r*.49,h*.08,'black')
    for i in range(24):
        a=2*pi*i/24
        b.cylinder(f'{name}_cap_grip_{i}',
            (x+r*.49*cos(a),y+r*.49*sin(a),z+h*.96),
            .0008,h*.065,'black',collision=False)
    # Front-facing thin paper label, tangent to the bottle, with flat text panel.
    b.box(name+'_paper_label',(x,y-r+.0002,z+h*.42),
          (r*1.42,.0008,h*.29),'paper',bevel=.0005,collision=False)
    b.label(name+'_label',text,(x,y-r-.0003,z+h*.42),
            (r*1.32,h*.24),rotation=(90,0,0))


def _closed_outline(b, name, center, outline, bottom, top, material,
                    rotation=(0,0,0), collision=True):
    """One continuous convex extruded outline, with independently sloped faces.

    ``bottom`` and ``top`` are heights at the polygon vertices. No collection of
    rectangles is used to fake a shaped blade, tang or handle.
    """
    from pxr import UsdPhysics
    count=len(outline)
    lo=[float(bottom)]*count if isinstance(bottom,(int,float)) else list(bottom)
    hi=[float(top)]*count if isinstance(top,(int,float)) else list(top)
    vertices=[(x,y,z) for zs in (lo,hi) for (x,y),z in zip(outline,zs)]
    faces=[]
    for i in range(1,count-1):
        faces.extend([(0,i+1,i),(count,count+i,count+i+1)])
    for i in range(count):
        j=(i+1)%count
        faces.extend([(i,j,count+j),(i,count+j,count+i)])
    path=b.path(name);root=b.stage.DefinePrim(path,'Xform')
    b.xform(root,center,rotation)
    uv=[((x+.2)/.4,(y+.05)/.10) for x,y,z in vertices]
    prim=b.mesh(path+'/Visual',vertices,faces,material,uv=uv)
    if collision:
        UsdPhysics.CollisionAPI.Apply(prim)
        UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr('none')
    return path


def _chef_knife(b, x, y, board_top, angle=-8.):
    """185 mm chef blade with a curved belly and pointed continuous spine."""
    a=angle*pi/180
    def world(xx,yy,zz):
        return (x+xx*cos(a)-yy*sin(a),y+xx*sin(a)+yy*cos(a),board_top+zz)
    # The edge rises continuously into the tip. Several polygon vertices sample
    # the curved belly; the pointed outline is one watertight stainless mesh.
    outline=[(-.072,-.022),(-.048,-.0235),(-.019,-.0235),(.010,-.0230),
             (.036,-.0214),(.059,-.0183),(.078,-.0134),(.092,-.0062),
             (.104,.0042),(.113,.017),(.097,.0185),(.065,.019),
             (.027,.019),(-.012,.019),(-.046,.0185),(-.072,.018)]
    # Handle underside and knife tip touch the board. A small physical lift at
    # the heel accounts for the thicker wooden handle instead of floating all
    # parts at a common, arbitrary height.
    lower=[.0028*(.113-xx)/.185 for xx,yy in outline]
    upper=[z+(.00025 if i<=9 else .0018) for i,z in enumerate(lower)]
    _closed_outline(b,'chef_knife_blade',(x,y,board_top),outline,lower,upper,
                    'prop_silver',(0,0,angle))
    # Shallow grind bevel is a continuous strip along the actual cutting edge.
    edge=outline[:10]
    verts=[]
    for i,(xx,yy) in enumerate(edge):
        z=lower[i]
        verts.extend([(xx,yy,z+.00026),(xx,yy+(.0035 if i<9 else .0001),z+.00182)])
    triangles=[]
    for i in range(len(edge)-1):
        k=2*i;triangles.extend([(k,k+1,k+3),(k,k+3,k+2)])
    path=b.path('chef_knife_honed_edge');root=b.stage.DefinePrim(path,'Xform')
    b.xform(root,(x,y,board_top),(0,0,angle))
    b.mesh(path+'/Visual',verts,triangles,'chrome')
    # Full tang and gently waisted wooden scales, rather than a square block.
    handle=[(-.181,-.008),(-.177,-.0115),(-.155,-.013),(-.129,-.0118),
            (-.104,-.0105),(-.080,-.011),(-.077,-.008),(-.077,.008),
            (-.082,.011),(-.105,.0105),(-.129,.0118),(-.155,.013),
            (-.177,.0115),(-.181,.008)]
    # The outline is convex enough for a fan, and its scale is kept at true metres.
    _closed_outline(b,'chef_knife_tang',(x,y,board_top),handle,.001,.007,
                    'prop_silver',(0,0,angle))
    inset=[(xx,.91*yy) for xx,yy in handle]
    _closed_outline(b,'chef_knife_wood_scale',(x,y,board_top),inset,0.,.0085,
                    'walnut',(0,0,angle))
    b.box('chef_knife_bolster',world(-.075,0,.0045),(.014,.025,.009),
          'prop_silver',bevel=.002,rotation=(0,0,angle))
    for i,xx in enumerate((-.160,-.130,-.100)):
        b.cylinder(f'knife_rivet_{i}',world(xx,0,.00865),.002,.0004,
                   'prop_silver',collision=False)


def _napkin_material(b):
    """Deterministic woven-cotton PBR maps, original CC0 source geometry."""
    import numpy as np
    from PIL import Image
    directory=b.assets_root/'scenes/scence_workrooms/v1/textures'
    directory.mkdir(parents=True,exist_ok=True)
    color_path=directory/'cotton_napkin_v2_color.png'
    normal_path=directory/'cotton_napkin_v2_normal.png'
    rough_path=directory/'cotton_napkin_v2_roughness.png'
    if not all(p.exists() for p in (color_path,normal_path,rough_path)):
        size=1024;rng=np.random.default_rng(20261016)
        yy,xx=np.mgrid[0:size,0:size].astype(float)
        period=8.
        warp=.5+.5*np.cos(2*pi*xx/period)
        weft=.5+.5*np.cos(2*pi*yy/period)
        over=((np.floor(xx/period)+np.floor(yy/period))%2)==0
        height=np.where(over,.72*warp+.28*weft,.28*warp+.72*weft)
        height+=rng.normal(0,.025,(size,size))
        fibers=.91+.055*height+rng.normal(0,.011,(size,size))
        albedo=np.stack([fibers*.90,fibers*.875,fibers*.81],axis=-1)
        # Two woven blue pinstripes on each side are part of the cloth texture,
        # never hard raised rails on top of the napkin.
        u=xx/size
        stripe=((abs(u-.085)<.005)|(abs(u-.112)<.003)|
                (abs(u-.888)<.003)|(abs(u-.915)<.005))
        albedo[stripe]=fibers[stripe,None]*np.array((.18,.28,.34))
        dy,dx=np.gradient(height)
        normal=np.stack((-dx*1.2,-dy*1.2,np.ones_like(dx)),axis=-1)
        normal/=np.linalg.norm(normal,axis=-1,keepdims=True)
        Image.fromarray(np.clip(albedo*255,0,255).astype('uint8')).save(color_path)
        Image.fromarray(np.clip((normal*.5+.5)*255,0,255).astype('uint8')).save(normal_path)
        Image.fromarray(np.clip((.91+.045*height)*255,0,255).astype('uint8')).save(rough_path)
    b.material('prop_woven_napkin',(1,1,1),roughness=.94,texture=str(color_path),
               normal=str(normal_path),roughness_texture=str(rough_path))


def _folded_napkin(b,x,y,z):
    """One thin, closed cloth mesh folded in a four-layer accordion cross-section.

    This is static room dressing. Curved folds and small edge irregularities are
    modelled surfaces; no invisible solid brick supports the folded textile.
    """
    import numpy as np
    from pxr import UsdPhysics
    _napkin_material(b)
    width=.24;radius=.0012;thickness=.0006
    path_y=[];path_z=[]
    def segment(y0,y1,height):
        for value in np.linspace(y0,y1,45):
            if path_y and abs(value-path_y[-1])<1.e-9 and abs(height-path_z[-1])<1.e-9:
                continue
            path_y.append(float(value));path_z.append(height)
    def fold(edge,height,direction):
        for a in np.linspace(0,pi,17)[1:]:
            path_y.append(edge+direction*radius*sin(a))
            path_z.append(height+radius*(1-cos(a)))
    segment(-.078,.078,.0003)
    fold(.078,.0003,1)
    segment(.078,-.074,.0003+2*radius)
    fold(-.074,.0003+2*radius,-1)
    segment(-.074,.072,.0003+4*radius)
    fold(.072,.0003+4*radius,1)
    segment(.072,-.068,.0003+6*radius)
    ny=len(path_y);nx=65
    vertices=np.zeros((ny,nx,3));uv=np.zeros((ny,nx,2))
    lengths=np.r_[0,np.cumsum(np.hypot(np.diff(path_y),np.diff(path_z)))]
    for j,(yy,zz) in enumerate(zip(path_y,path_z)):
        for i,xx in enumerate(np.linspace(-width/2,width/2,nx)):
            u=i/(nx-1);v=j/(ny-1)
            edge_weight=(abs(xx)/(width/2))**4
            # The cloth is lightly crumpled, with less than 2 mm of variation.
            wrinkle=.00045*(1+sin(5*pi*u+.7))*edge_weight
            wrinkle+=.00055*(1+sin(3*pi*u+1.4))*(zz/.0075)
            edge_shift=.0015*sin(4*pi*u+.6)+.0008*sin(9*pi*u)
            x_offset=.0009*sin(5*pi*v+.3)*edge_weight
            vertices[j,i]=(x+xx+x_offset,y+yy+edge_shift,z+zz+wrinkle)
            uv[j,i]=(u,lengths[j]/width)
    tangent_v=np.gradient(vertices,axis=0)
    tangent_u=np.gradient(vertices,axis=1)
    normals=np.cross(tangent_u,tangent_v)
    normals/=np.linalg.norm(normals,axis=-1,keepdims=True)
    plus=vertices+normals*thickness/2
    minus=vertices-normals*thickness/2
    # Ground the real lowest cloth vertex on the real tabletop, exactly.
    ground=min(float(plus[:,:,2].min()),float(minus[:,:,2].min()))
    plus[:,:,2]+=z-ground;minus[:,:,2]+=z-ground
    count=nx*ny
    mesh_vertices=np.concatenate([plus.reshape(-1,3),minus.reshape(-1,3)])
    mesh_normals=np.concatenate([normals.reshape(-1,3),-normals.reshape(-1,3)])
    mesh_uv=np.concatenate([uv.reshape(-1,2),uv.reshape(-1,2)])
    faces=[]
    for j in range(ny-1):
        for i in range(nx-1):
            a=j*nx+i;quad=(a,a+1,a+nx+1,a+nx)
            faces.append(quad);faces.append(tuple(k+count for k in reversed(quad)))
    perimeter=list(range(nx))+[j*nx+nx-1 for j in range(1,ny)]
    perimeter+=list(range((ny-1)*nx+nx-2,(ny-1)*nx-1,-1))
    perimeter+=[j*nx for j in range(ny-2,0,-1)]
    for i,a in enumerate(perimeter):
        c=perimeter[(i+1)%len(perimeter)]
        faces.append((a,a+count,c+count,c))
    path=b.path('folded_napkin')
    prim=b.mesh(path,mesh_vertices,faces,'prop_woven_napkin',mesh_uv,mesh_normals)
    UsdPhysics.CollisionAPI.Apply(prim)
    UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr('none')
    # The visible free hem has a tiny rolled edge and short sewn stitches.
    hem=[]
    j=ny-1
    for i in range(nx):
        p=vertices[j,i].copy();p[2]+=z-ground+.00045;hem.append(tuple(p))
    b.tube('napkin_free_hem',hem,.0004,'prop_ivory',collision=False)
    for i in range(2,nx-2,2):
        p=np.array(hem[i]);q=np.array(hem[i+1]);p[1]+=.0025;q[1]+=.0025
        p[2]+=.00012;q[2]+=.00012
        b.tube(f'napkin_hem_stitch_{i}',[tuple(p),tuple(q)],.00013,
               'prop_pages',collision=False)


def _mechanical(b):
    inventory=[]
    paths={
        'adjustable_wrench': 'Industrial/Tools/Hand/Monkey_Wrench_A01/sm_handTool_wrench_monkey_a01_01.usd',
        'phillips_screwdriver': 'Industrial/Tools/Hand/Screwdriver_Phillips_Ph1_A01/sm_handTool_screwdriver_phillips_ph1_a01_01.usd',
        'cutting_pliers': 'Industrial/Tools/Hand/Diagonal_Cutting_Pliers_A01/sm_handTool_pliers_diagonal_cutting_a01_01.usd',
        'rubber_mallet': 'Industrial/Tools/Hand/Rubber_Mallet_B01/sm_handTool_hammer_rubberMallet_b01_01.usd',
        'multimeter': 'Industrial/Tools/Multimeter_A01/sm_testingTool_multimeter_a01_01.usd',
        'safety_goggles': 'Industrial/Safety/Goggles_A01/sm_ppe_eyeProtection_goggles_a01_01.usd',
    }
    for name,pos,angle in [
        ('adjustable_wrench',(.49,-.23,TABLE_Z),-12),
        ('phillips_screwdriver',(.50,-.08,TABLE_Z),5),
        ('cutting_pliers',(.52,.09,TABLE_Z),-8),
        ('rubber_mallet',(.78,.26,TABLE_Z),90),
        ('multimeter',(.97,.24,TABLE_Z),0),
        ('safety_goggles',(.43,.31,TABLE_Z),-10),
    ]:
        inventory.append(_asset(b,name,paths[name],pos,(0,0,angle)))
    z=_tray(b,'fastener_tray',.91,-.14,TABLE_Z,.22,.22,.03,'prop_soft_blue')
    b.box('fastener_divider_x',(.91,-.14,z+.012),(.004,.208,.024),
          'prop_soft_blue',bevel=.001)
    b.box('fastener_divider_y',(.91,-.14,z+.012),(.208,.004,.024),
          'prop_soft_blue',bevel=.001)
    # Washers have actual open bores and bevelled edges.
    for i in range(3):
        x=.86+(i%2)*.025; y=-.19+(i//2)*.025
        b.lathe(f'm8_washer_{i}',(x,y,z),[(.0043,0),(.010,0),(.0105,.0005),
            (.0105,.0015),(.010,.002),(.0043,.002)],'prop_silver',collision=True)
    # A visible helical thread, not a pile of cylinders.
    for i in range(3):
        x=.955+(i%2)*.025; y=-.19+(i//2)*.026
        b.cylinder(f'm8_bolt_{i}_shaft',(x,y,z+.012),.0036,.024,'prop_silver')
        thread=[(x+.004*cos(t),y+.004*sin(t),z+.001+.00125*t/(2*pi))
                for t in [2*pi*16*j/320 for j in range(321)]]
        b.tube(f'm8_bolt_{i}_thread',thread,.00045,'prop_silver',collision=False)
        b.cylinder(f'm8_bolt_{i}_head',(x,y,z+.027),.007,.006,'prop_silver')
        b.box(f'm8_bolt_{i}_slot',(x,y,z+.03005),(.007,.0012,.0002),
              'black',bevel=.0003,collision=False)
    for i in range(4):
        x=.955+(i%2)*.026; y=-.085+(i//2)*.024
        b.lathe(f'm8_nut_{i}',(x,y,z),[(.0042,0),(.0074,0),(.008,.001),
            (.008,.005),(.0074,.006),(.0042,.006)],'prop_silver',collision=True)
    _label(b,'parts_tray_label','M8 / STAINLESS',.91,-.235,z+.025,.15,.012)
    inventory.append({'name':'fastener_tray','type':'four_compartment_tray_with_M8_fasteners',
                      'support_z_m':TABLE_Z})
    # A real assembly target: the hollow copper tube rests on the fixture base;
    # two stand-offs support a clamping bridge tangent to the tube's upper wall.
    x,y,z=.70,-.12,TABLE_Z
    b.box('pipe_fixture_base',(x,y,z+.004),(.12,.09,.008),'steel',bevel=.003)
    b.lathe('fixture_copper_tube',(x,y+.037,z+.017),
        [(.006,0),(.009,0),(.009,.074),(.006,.074)],
        'prop_copper',rotation=(90,0,0),collision=True)
    for side in (-1,1):
        xx=x+side*.039
        b.cylinder(f'fixture_standoff_{side}',(xx,y,z+.017),.006,.018,'prop_silver')
        b.cylinder(f'fixture_screw_head_{side}',(xx,y,z+.034),.0065,.004,'prop_silver')
        b.box(f'fixture_screw_slot_{side}',(xx,y,z+.0361),(.008,.0011,.0002),
              'black',bevel=.0003,collision=False)
    b.box('fixture_clamping_bridge',(x,y,z+.029),(.10,.025,.006),'prop_silver',bevel=.002)
    inventory.append({'name':'pipe_clamping_fixture','type':'two_bolt_clamp_with_hollow_copper_tube',
                      'support_z_m':TABLE_Z})
    # Small engineering notebook with fully supported pen.
    inventory.append(_book(b,'assembly_log',.46,-.365,TABLE_Z,.20,.105,.014,
                           'prop_book_green','ASSEMBLY LOG','WORKCELL / 01'))
    _pen(b,'assembly_pen',.74,-.37)
    inventory.append({'name':'assembly_pen','type':'metal_clip_pen','support_z_m':TABLE_Z})
    return inventory


def _laboratory(b):
    inventory=[]
    # Open reagent rack: bottom slab, legs and perimeter rails; real open sockets.
    x,y,z=.85,.235,TABLE_Z
    b.box('tube_rack_base',(x,y,z+.006),(.245,.12,.012),'prop_ivory',bevel=.004)
    for dx in (-.108,.108):
        for dy in (-.045,.045):
            b.cylinder(f'rack_column_{dx}_{dy}',(x+dx,y+dy,z+.042),.004,.072,'prop_silver')
    for dy in (-.047,.047):
        b.box(f'rack_rail_{dy}',(x,y+dy,z+.075),(.23,.009,.008),'prop_ivory',bevel=.003)
    for i in range(5):
        xx=x-.088+i*.044
        b.box(f'rack_crossbar_{i}',(xx,y,z+.075),(.007,.09,.008),'prop_ivory',bevel=.002)
    for i in range(4):
        xx=x-.066+i*.044
        for j in range(2):
            yy=y-.021+j*.042
            b.torus(f'rack_ring_{i}_{j}',(xx,yy,z+.076),.011,.003,'prop_ivory')
            b.lathe(f'sample_tube_{i}_{j}',(xx,yy,z+.012),[
                (0,0),(.006,.002),(.009,.009),(.009,.105),
                (.0075,.105),(.0075,.010),(.005,.004),(0,.003)],'glass',collision=False)
            cap='prop_book_blue' if j==0 else 'prop_book_red'
            b.cylinder(f'sample_cap_{i}_{j}',(xx,yy,z+.121),.0105,.012,cap)
            b.box(f'tube_label_{i}_{j}',(xx,yy-.0087,z+.060),
                  (.010,.0003,.022),'paper',bevel=.0003,collision=False)
    inventory.append({'name':'reagent_tube_rack','type':'eight_capped_glass_tubes_in_open_rack',
                      'support_z_m':TABLE_Z})
    # Two empty, hollow vessels with calibrated printed graduation marks.
    for name,xx,yy,r,h in [('beaker_250ml',.50,.18,.035,.075),
                           ('beaker_100ml',.62,.24,.027,.060)]:
        b.lathe(name,(xx,yy,TABLE_Z),[(0,0),(r-.003,0),(r,.003),(r,h),
            (r-.0015,h+.001),(r-.003,h-.002),(r-.003,.004),(0,.004)],
            'glass',collision=True)
        for i in range(1,5):
            b.box(f'{name}_graduation_{i}',(xx,yy-r-.0002,TABLE_Z+h*i/6),
                  (.013 if i%2==0 else .009,.0004,.0008),'paper',bevel=0,collision=False)
        b.label(name+'_print','250 mL' if '250' in name else '100 mL',
                (xx,yy-r-.0003,TABLE_Z+h*.75),(.023,.009),rotation=(90,0,0))
        inventory.append({'name':name,'type':'empty_glass_beaker','support_z_m':TABLE_Z})
    _bottle(b,'amber_reagent',.99,-.02,height=.14,radius=.032,text='NaCl\n0.9%')
    _bottle(b,'clear_reagent',.87,-.03,height=.12,radius=.028,material='glass',text='DI WATER')
    inventory.extend([
        {'name':'amber_reagent','type':'capped_reagent_bottle','support_z_m':TABLE_Z},
        {'name':'clear_reagent','type':'capped_glass_reagent_bottle','support_z_m':TABLE_Z},
    ])
    # Laboratory wash bottle with bent nozzle and neck ring.
    b.lathe('wash_bottle_body',(.38,.27,TABLE_Z),[(0,0),(.026,0),(.030,.005),
        (.030,.105),(.027,.116),(.012,.13),(.012,.145),(0,.145)],'white',collision=True)
    b.cylinder('wash_bottle_cap',(.38,.27,TABLE_Z+.145),.015,.012,'prop_soft_blue')
    b.tube('wash_bottle_spout',[(.38,.27,TABLE_Z+.151),(.38,.27,TABLE_Z+.18),
        (.42,.27,TABLE_Z+.205),(.45,.27,TABLE_Z+.205),(.46,.27,TABLE_Z+.195)],
        .0025,'white',collision=False)
    inventory.append({'name':'wash_bottle','type':'bent_nozzle_wash_bottle','support_z_m':TABLE_Z})
    inventory.append(_asset(b,'lab_safety_goggles',
        'Industrial/Safety/Goggles_A01/sm_ppe_eyeProtection_goggles_a01_01.usd',
        (.38,-.18,TABLE_Z),(0,0,-12)))
    # Pipetting station, with a shaped adjustable micropipette and two swabs.
    tray_z=_tray(b,'specimen_tray',.76,-.22,TABLE_Z,.23,.13,.018,'prop_silver')
    for i in range(2):
        yy=-.255+.020*i
        b.cylinder(f'swab_{i}_stem',(.76,yy,tray_z+.003),.0013,.14,'oak',rotation=(0,90,0))
        for side in (-1,1):
            b.sphere(f'swab_{i}_tip_{side}',(.76+side*.069,yy,tray_z+.003),(.010,.003,.003),'white')
    px,py,pz=.685,-.188,tray_z+.008
    b.lathe('micropipette_grip',(px,py,pz),[(0,0),(.007,0),(.008,.006),
        (.008,.076),(.006,.086),(.005,.103),(0,.103)],
        'white',rotation=(0,90,0),collision=True)
    b.cylinder('micropipette_plunger',(px-.010,py,pz),.0075,.020,
               'prop_soft_blue',rotation=(0,90,0))
    b.lathe('micropipette_tip',(px+.100,py,pz),[(.0045,0),(.0045,.006),
        (.0010,.062),(.00065,.065),(.0005,.064),(.0031,.006),(.0031,0)],
        'prop_ivory',rotation=(0,90,0),collision=False)
    _label(b,'micropipette_volume','200',px+.044,py,pz+.0082,.027,.009)
    inventory.append({'name':'specimen_tray','type':'stainless_tray_with_adjustable_micropipette_and_two_swabs',
                      'support_z_m':TABLE_Z})
    inventory.append(_book(b,'lab_record',.48,-.36,TABLE_Z,.20,.11,.013,
                           'prop_book_blue','LAB RECORD','EXPERIMENT 024'))
    _pen(b,'lab_pen',.91,-.37)
    inventory.append({'name':'lab_pen','type':'metal_clip_pen','support_z_m':TABLE_Z})
    return inventory


def _tomato(b,name,x,y,z):
    """Slightly lobed produce with a depressed crown, calyx and stem."""
    import numpy as np
    from pxr import UsdPhysics
    sides,rings=64,32
    vertices=[(0,0,0)];uv=[(.5,0)]
    for j in range(1,rings):
        theta=pi*j/rings
        for i in range(sides):
            phi=2*pi*i/sides
            radius=.029*sin(theta)*(1+.035*cos(6*phi)*sin(theta))
            zz=.026*(1-cos(theta))-.004*max(0,-cos(theta))**10
            vertices.append((radius*cos(phi),radius*sin(phi),zz));uv.append((i/sides,j/rings))
    vertices.append((0,0,.048));uv.append((.5,1))
    faces=[]
    for i in range(sides):faces.append((0,1+(i+1)%sides,1+i))
    for j in range(rings-2):
        for i in range(sides):
            a=1+j*sides+i;c=1+j*sides+(i+1)%sides
            faces.extend([(a,c,c+sides),(a,c+sides,a+sides)])
    top=len(vertices)-1
    for i in range(sides):faces.append((top,top-sides+i,top-sides+(i+1)%sides))
    v=np.asarray(vertices);normals=np.zeros_like(v)
    for face in faces:
        a,c,d=face;n=np.cross(v[c]-v[a],v[d]-v[a])
        for index in face:normals[index]+=n
    normals/=np.maximum(np.linalg.norm(normals,axis=1)[:,None],1e-12)
    path=b.path(name);root=b.stage.DefinePrim(path,'Xform');b.xform(root,(x,y,z))
    prim=b.mesh(path+'/Visual',vertices,faces,'prop_tomato',uv=uv,normals=normals)
    UsdPhysics.CollisionAPI.Apply(prim)
    UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr('convexHull')
    for i in range(5):
        a=2*pi*i/5+.3;ca,sa=cos(a),sin(a)
        leaf=[(.003*ca,.003*sa,.0485),(.010*ca-.004*sa,.010*sa+.004*ca,.050),
              (.020*ca,.020*sa,.044),(.010*ca+.004*sa,.010*sa-.004*ca,.050)]
        b.mesh(path+f'/Calyx{i}',leaf,[(0,1,2),(0,2,3)],'prop_leaf')
    b.tube(name+'_stem',[(x,y,z+.048),(x+.001,y,z+.056),(x+.003,y+.001,z+.060)],
           .0015,'prop_leaf',collision=False)


def _kitchen(b):
    inventory=[]
    # End-grain board built from matched timber strips, edged with a carved juice groove.
    x,y=.70,.18
    b.box('chopping_board',(x,y,TABLE_Z+.012),(.34,.235,.024),'oak',bevel=.009)
    for i in range(8):
        b.box(f'board_grain_strip_{i}',(x-.148+i*.042,y,TABLE_Z+.0242),
            (.0012,.214,.00025),'walnut',bevel=0,collision=False)
    groove=[(x-.15,y-.098,TABLE_Z+.0243),(x+.15,y-.098,TABLE_Z+.0243),
            (x+.15,y+.098,TABLE_Z+.0243),(x-.15,y+.098,TABLE_Z+.0243),
            (x-.15,y-.098,TABLE_Z+.0243)]
    b.tube('board_juice_groove',groove,.0008,'walnut')
    _chef_knife(b,.723,.19,TABLE_Z+.024)
    _tomato(b,'whole_tomato',.815,.111,TABLE_Z+.024)
    for i,xx in enumerate((.610,.670)):
        zz=TABLE_Z+.024
        b.lathe(f'tomato_slice_{i}',(xx,.105,zz),[(0,0),(.022,0),(.025,.001),
            (.025,.004),(.023,.005),(0,.005)],'prop_tomato',collision=True)
        b.cylinder(f'tomato_slice_{i}_flesh',(xx,.105,zz+.00505),.0228,.0002,
                   'prop_tomato_flesh',collision=False)
        for j in range(10):
            a=2*pi*j/10+.13*i
            b.sphere(f'tomato_slice_{i}_seed_{j}',
                (xx+.013*cos(a),.105+.013*sin(a),zz+.0053),
                (.0010,.0020,.00025),'prop_tomato_seed')
    inventory.append({'name':'cutting_board_and_knife','type':'wood_board_riveted_chef_knife',
                      'support_z_m':TABLE_Z})
    inventory.append({'name':'tomato_prep','type':'whole_lobed_tomato_and_two_seeded_slices',
                      'support_z_m':TABLE_Z+.024})
    _bowl(b,'mixing_bowl',.90,-.14,radius=.085,height=.063,material='prop_ivory')
    inventory.append({'name':'mixing_bowl','type':'hollow_ceramic_bowl','support_z_m':TABLE_Z})
    inventory.append(_cup(b,'kitchen_mug',.39,.20,material='prop_sage'))
    _plate(b,'small_plate',.48,-.19,radius=.086)
    inventory.append({'name':'small_plate','type':'ceramic_plate','support_z_m':TABLE_Z})
    # A glass oil bottle and two matched spice grinders at the rear edge.
    _bottle(b,'olive_oil',1.00,.28,height=.205,radius=.028,material='prop_sage',text='OLIVE\nOIL')
    for i,mat in enumerate(('prop_chess_dark','prop_ivory')):
        xx=.89+.088*i; yy=.39
        b.lathe(f'spice_grinder_{i}',(xx,yy,TABLE_Z),[(0,0),(.021,0),(.024,.004),
            (.024,.013),(.019,.025),(.013,.061),(.017,.08),(.026,.091),
            (.026,.106),(.021,.117),(0,.121)],mat,collision=True)
        b.sphere(f'spice_grinder_{i}_knob',(xx,yy,TABLE_Z+.123),(.003,.003,.003),'prop_silver')
    inventory.extend([
        {'name':'olive_oil','type':'labelled_oil_bottle','support_z_m':TABLE_Z},
        {'name':'spice_grinder_pair','type':'two_turned_wood_spice_grinders','support_z_m':TABLE_Z},
    ])
    _folded_napkin(b,.73,-.35,TABLE_Z)
    inventory.append({'name':'folded_kitchen_towel','type':'folded_static_textile',
                      'support_z_m':TABLE_Z,'note':'visual static cloth; not a deformable simulation'})
    return inventory


def _study(b):
    inventory=[]
    inventory.append(_book(b,'design_book',.87,.25,TABLE_Z,.185,.24,.035,
                           'prop_book_blue','THE ART\nOF READING','PERSONAL LIBRARY'))
    inventory.append(_book(b,'drawing_book',.87,.25,TABLE_Z+.035,.17,.22,.025,
                           'prop_book_red','SELECTED\nESSAYS','VOLUME TWO'))
    # Open notebook: an actual thin gutter and page ruling, comfortably reachable.
    x,y,z=.64,-.12,TABLE_Z
    b.box('notebook_cover',(x,y,z+.0015),(.29,.205,.003),'prop_leather',bevel=.003)
    for side in (-1,1):
        xx=x+side*.073
        b.box(f'notebook_pages_{side}',(xx,y,z+.005),(.142,.198,.007),
              'prop_pages',bevel=.0012)
        for i in range(15):
            b.box(f'notebook_rule_{side}_{i}',(xx,y-.085+i*.0115,z+.0086),
                  (.122,.0003,.0001),'prop_soft_blue',bevel=0,collision=False)
    b.tube('notebook_bookmark',[(x,y+.103,z+.006),(x,y-.102,z+.006),
        (x+.013,y-.139,z+.001)],.0014,'prop_book_red',collision=False)
    _pen(b,'fountain_pen',.64,-.10,z+.0086,length=.14,material='prop_book_blue')
    inventory.append({'name':'open_notebook','type':'ruled_notebook_with_pen','support_z_m':TABLE_Z})
    # Weighted articulated desk lamp, with a fully connected steel stem.
    x,y=.39,.30
    b.lathe('lamp_base',(x,y,TABLE_Z),[(0,0),(.055,0),(.057,.003),(.057,.012),
        (.051,.018),(0,.018)],'prop_sage',collision=True)
    b.tube('lamp_lower_arm',[(x,y,TABLE_Z+.018),(x-.025,y,TABLE_Z+.19)],.006,'prop_brass')
    b.sphere('lamp_joint',(x-.025,y,TABLE_Z+.19),(.011,.011,.011),'prop_sage')
    b.tube('lamp_upper_arm',[(x-.025,y,TABLE_Z+.19),(x+.065,y,TABLE_Z+.30)],.005,'prop_brass')
    b.lathe('lamp_shade',(x+.065,y,TABLE_Z+.255),[(.058,0),(.061,.005),
        (.028,.052),(.019,.061),(.014,.061),(.024,.050),(.055,.006)],
        'prop_sage',collision=False)
    b.sphere('lamp_bulb',(x+.065,y,TABLE_Z+.275),(.014,.014,.020),'white')
    # Place the emitting surface just below the existing shade's open rim.
    # It illuminates the notebook locally without changing its collision shapes.
    from .common import light
    light(b,'study_desk_task',(x+.065,y,TABLE_Z+.253),(.64,-.12,TABLE_Z+.009),
          (.084,.084),18000.,(1,1,1),shape='disk',role='task',temperature=3200.)
    b.tube('lamp_cord',[(x,y+.02,TABLE_Z+.004),(x-.06,y+.065,TABLE_Z+.003),
        (x-.10,y+.14,TABLE_Z+.003),(x-.12,.48,TABLE_Z+.003),
        (x-.12,.53,.60),(x-.12,.53,.20)],.002,'black')
    inventory.append({'name':'desk_lamp','type':'weighted_base_articulated_shade_lamp',
                      'support_z_m':TABLE_Z})
    # Ribbed pencil pot with physical hollow wall, pencils visible above rim.
    _cup(b,'pencil_pot',.61,.32,radius=.027,height=.082,material='prop_terracotta',handle=False)
    for i in range(7):
        a=2*pi*i/7; xx=.61+.015*cos(a); yy=.32+.015*sin(a); h=.125+.006*(i%3)
        b.cylinder(f'pencil_{i}',(xx,yy,TABLE_Z+.009+h/2),.0026,h,
                   'prop_book_blue' if i%2 else 'oak')
        b.lathe(f'pencil_{i}_tip',(xx,yy,TABLE_Z+.009+h),
                [(0,0),(.0026,0),(.0006,.01),(0,.012)],'oak')
    inventory.append({'name':'pencil_pot','type':'open_ceramic_pot_with_seven_pencils',
                      'support_z_m':TABLE_Z})
    inventory.append(_cup(b,'study_coffee',.97,-.13,material='prop_ivory'))
    b.cylinder('study_coffee_surface',(.97,-.13,TABLE_Z+.062),.0305,.0005,'prop_coffee',collision=False)
    # Reading glasses: two open lens frames and connected temples, resting on pages.
    zz=TABLE_Z+.061
    for side in (-1,1):
        xx=.87+side*.026
        pts=[(xx+.021*cos(a),.25+.015*sin(a),zz) for a in [2*pi*i/40 for i in range(41)]]
        b.tube(f'reading_glasses_rim_{side}',pts,.0011,'prop_brass')
        b.tube(f'reading_glasses_temple_{side}',[(xx+side*.021,.25,zz),
            (xx+side*.028,.29,zz),(xx+side*.023,.315,zz)],.0011,'prop_brass')
    b.tube('reading_glasses_bridge',[(.865,.25,zz),(.87,.254,zz),(.875,.25,zz)],.0011,'prop_brass')
    inventory.append({'name':'reading_glasses','type':'thin_brass_reading_frames',
                      'support_z_m':TABLE_Z+.060})
    return inventory


def _go_stone(b,name,x,y,z,material):
    # Biconvex discs; the lower centre rests exactly on the board or bowl floor.
    b.lathe(name,(x,y,z),[(0,0),(.005,.0004),(.0095,.0015),(.0104,.0026),
        (.0104,.0030),(.0090,.0041),(.005,.0054),(0,.0058)],
        material,collision=True)


def _go_bowl(b,name,x,y,z,stone_material):
    # Turned wood bowl with an open rim and a real flat internal support floor.
    b.lathe(name,(x,y,z),[(0,0),(.034,0),(.045,.004),(.053,.014),
        (.056,.031),(.054,.040),(.051,.042),(.048,.039),(.049,.031),
        (.046,.017),(.037,.008),(0,.008)],'walnut',collision=True)
    for i,(xx,yy) in enumerate([(0,0),(-.022,0),(.022,0),(-.011,-.019),
                               (.011,-.019),(-.011,.019),(.011,.019)]):
        _go_stone(b,f'{name}_stone_{i}',x+xx,y+yy,z+.008,stone_material)


def _tea_pot(b,name,x,y,z):
    scale=.68
    def point(xx,yy,zz):return (x+xx*scale,y+yy*scale,z+zz*scale)
    body=[(0,.001),(.036,0),(.049,.014),(.058,.041),(.055,.067),
          (.043,.083),(.032,.087),(.028,.082),(.039,.078),(.049,.063),
          (.052,.041),(.044,.018),(.032,.008),(0,.008)]
    b.lathe(name+'_body',(x,y,z),[(r*scale,h*scale) for r,h in body],
            'prop_terracotta',collision=True)
    lid=[(0,0),(.035,0),(.037,.003),(.030,.009),(.013,.013),(0,.013)]
    b.lathe(name+'_lid',point(0,0,.085),[(r*scale,h*scale) for r,h in lid],'prop_terracotta')
    b.sphere(name+'_lid_knob',point(0,0,.102),(.0054,.0054,.0048),'prop_terracotta')
    b.tube(name+'_spout',[point(*p) for p in [(.047,0,.032),(.068,0,.040),
        (.086,0,.063),(.102,0,.078),(.112,0,.081)]],.0068,'prop_terracotta',collision=True)
    b.torus(name+'_spout_lip',point(.112,0,.081),.0051,.0017,
            'prop_terracotta',rotation=(0,70,0))
    pts=[point(-.055-.026*cos(a),0,.048+.038*sin(a))
         for a in [2*pi*i/40 for i in range(41)]]
    b.tube(name+'_handle',pts,.0041,'prop_terracotta',collision=True)


def _card_room(b):
    inventory=[]
    # One coherent Chinese tea-and-Go setting, with a playable 13 x 13 grid.
    x,y,z=.78,.19,TABLE_Z
    b.box('go_board_frame',(x,y,z+.009),(.35,.35,.018),'walnut',bevel=.004)
    b.box('go_board_inlay',(x,y,z+.0185),(.328,.328,.001),
          'prop_go_board',bevel=.0005,collision=False)
    surface=z+.0191
    for i in range(13):
        delta=-.144+i*.024
        b.box(f'go_grid_x_{i}',(x+delta,y,z+.01905),(.00065,.288,.0001),
              'prop_ink',bevel=0,collision=False)
        b.box(f'go_grid_y_{i}',(x,y+delta,z+.01905),(.288,.00065,.0001),
              'prop_ink',bevel=0,collision=False)
    for i,j in [(3,3),(3,9),(6,6),(9,3),(9,9)]:
        b.cylinder(f'go_star_{i}_{j}',(x-.144+i*.024,y-.144+j*.024,z+.01906),
                   .0019,.00012,'prop_ink',collision=False)
    # Alternating legal opening positions, with a central region left to play.
    placements=[(3,3),(9,9),(3,9),(9,3),(4,9),(9,4),(4,10),(8,3),
                (3,10),(10,4),(2,9),(8,4),(5,9),(9,5),(4,8),(10,3),
                (2,3),(10,9),(3,4),(9,10),(4,3),(8,9)]
    for i,(column,row) in enumerate(placements):
        _go_stone(b,f'go_board_stone_{i}',x-.144+column*.024,y-.144+row*.024,
                  surface,'prop_go_black' if i%2==0 else 'prop_go_white')
    _go_bowl(b,'black_go_bowl',.44,.265,TABLE_Z,'prop_go_black')
    _go_bowl(b,'white_go_bowl',.44,.065,TABLE_Z,'prop_go_white')
    inventory.append({'name':'go_set','type':'13_line_wood_go_board_22_biconvex_stones_two_open_bowls',
                      'support_z_m':TABLE_Z})
    tray_z=_tray(b,'gongfu_tea_tray',.91,-.24,TABLE_Z,.25,.22,.018,'walnut')
    # Slatted bamboo drain surface rests on the tray floor.
    for i in range(11):
        b.box(f'tea_tray_sl_at_{i}',(.91,-.331+i*.018,tray_z+.0015),
              (.226,.012,.003),'oak',bevel=.001,collision=False)
    tea_z=tray_z+.003
    _tea_pot(b,'yixing_teapot',.90,-.205,tea_z)
    for i,xx in enumerate((.839,.981)):
        _cup(b,f'kungfu_teacup_{i}',xx,-.306,tea_z,radius=.021,height=.031,
             material='prop_ivory',handle=False)
    inventory.append({'name':'gongfu_tea_service','type':'small_yixing_style_pot_two_handleless_cups_slatted_tray',
                      'support_z_m':TABLE_Z})
    return inventory


def _folded_reading_glasses(b,name,x,y,z):
    # Frames and folded arms are connected; the lowest rim rests on the liner.
    zz=z+.0012
    for side in (-1,1):
        xx=x+side*.026
        pts=[(xx+.021*cos(a),y+.015*sin(a),zz)
             for a in [2*pi*i/48 for i in range(49)]]
        b.tube(f'{name}_rim_{side}',pts,.0012,'prop_brass')
        b.tube(f'{name}_folded_arm_{side}',[(xx+side*.021,y,zz),
            (xx+side*.022,y+.009,zz+.002),
            (x-side*.022,y+.011,zz+.003),(x-side*.029,y+.007,zz+.003)],
            .001,'prop_brass')
    b.tube(name+'_bridge',[(x-.005,y,zz),(x,y+.004,zz),(x+.005,y,zz)],
           .0012,'prop_brass')


def _living_room(b):
    inventory=[]
    # A daily-use catchall, with small items organised on an inset felt liner.
    tray_z=_tray(b,'living_catchall',.88,.23,TABLE_Z,.29,.24,.022,'walnut')
    b.box('catchall_felt_liner',(.88,.23,tray_z+.001),(.274,.224,.002),
          'prop_linen',bevel=.003,collision=False)
    support=tray_z+.002
    x,y,z=.805,.23,support
    b.box('remote_body',(x,y,z+.009),(.047,.17,.018),'black',bevel=.009)
    b.box('remote_display',(x,y+.045,z+.0182),(.032,.040,.001),
          'prop_book_blue',bevel=.003,collision=False)
    b.torus('remote_navigation_ring',(x,y-.003,z+.020),.013,.0025,'prop_silver')
    b.cylinder('remote_ok',(x,y-.003,z+.020),.006,.003,'black',collision=False)
    for i in range(3):
        for j in range(3):
            b.box(f'remote_button_{i}_{j}',(x+(i-1)*.011,y-.040-j*.013,z+.019),
                  (.007,.007,.003),'prop_stone',bevel=.002,collision=False)
    b.cylinder('remote_power',(x-.012,y+.071,z+.0195),.0035,.003,
               'prop_book_red',collision=False)
    _folded_reading_glasses(b,'living_glasses',.936,.26,support)
    # A closed leather glasses case is separate from the exposed thin frames.
    b.box('glasses_case_lower',(.936,.169,support+.009),(.12,.041,.018),
          'prop_leather',bevel=.015)
    b.box('glasses_case_lid',(.936,.169,support+.020),(.12,.041,.006),
          'prop_leather',bevel=.003)
    inventory.append({'name':'living_catchall','type':'felt_lined_tray_with_tv_remote_folded_glasses_and_case',
                      'support_z_m':TABLE_Z})
    inventory.append(_book(b,'living_book',.48,-.20,TABLE_Z,.20,.24,.025,
                           'prop_book_green','SPACES\nFOR LIVING','ARCHITECTURE / 2026'))
    inventory.append(_book(b,'living_magazine',.48,-.20,TABLE_Z+.025,.19,.23,.005,
                           'prop_ivory','SLOW\nLIVING','THE AUTUMN ISSUE'))
    # Low tissue carton with a rounded dispensing slot and a creased tissue.
    x,y,z=.44,.245,TABLE_Z
    b.box('tissue_box',(x,y,z+.041),(.19,.12,.082),'prop_linen',bevel=.010)
    b.box('tissue_slot',(x,y,z+.0822),(.105,.013,.0005),'prop_ink',bevel=.006,
          collision=False)
    _label(b,'tissue_box_print','SOFT COTTON',x,y-.039,z+.0822,.13,.012)
    # The lower edge begins in the real slot; alternating heights make the
    # upper paper edge softly folded instead of a rigid rectangular billboard.
    vertices=[]
    for j in range(7):
        t=j/6
        for i in range(11):
            u=i/10
            vertices.append((x+(u-.5)*(.080+.012*t),
                y+.003*sin(pi*u*4)*t+.018*t*t,
                z+.082+.052*t+.008*sin(pi*u*3)*t*t))
    faces=[]
    for j in range(6):
        for i in range(10):
            a=j*11+i;faces.extend([(a,a+1,a+12),(a,a+12,a+11)])
    b.mesh(b.path('dispensed_tissue')+'/Visual',vertices,faces,'paper')
    inventory.append({'name':'tissue_box','type':'rounded_carton_with_open_slot_and_creased_tissue',
                      'support_z_m':TABLE_Z})
    # A small phone occupies the opposite edge, leaving the middle of the
    # table clear for picking, placing and bringing household objects together.
    x,y=.91,-.20
    b.box('phone_frame',(x,y,TABLE_Z+.004),(.072,.145,.008),'prop_silver',bevel=.008)
    b.box('phone_screen',(x,y,TABLE_Z+.0082),(.066,.136,.0006),'black',bevel=.006,
          collision=False)
    b.box('phone_earpiece',(x,y+.058,TABLE_Z+.0087),(.014,.001,.0004),
          'prop_stone',bevel=.0005,collision=False)
    inventory.append({'name':'phone','type':'glass_screen_smartphone','support_z_m':TABLE_Z})
    return inventory


def build_props(b, family):
    """Populate exactly one room's 1 x 1 m workspace; return provenance inventory."""
    _materials(b)
    builders={'mechanical_assembly':_mechanical, 'laboratory':_laboratory,
              'kitchen':_kitchen, 'study':_study, 'card_room':_card_room,
              'living_room':_living_room}
    if family not in builders:
        raise ValueError(f'Unknown room family: {family}')
    items=builders[family](b)
    for item in items:
        item.setdefault('source', 'Original procedural geometry: bench/simulation/assets/workrooms/props.py')
        item.setdefault('license', 'CC0-1.0')
        item['static_scenery']=True
        item['manipulation_validated']=False
    return items
