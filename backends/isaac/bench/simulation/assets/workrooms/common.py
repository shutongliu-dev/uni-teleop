"""Shared full-scale architecture and one-square-metre workcell."""
import math
from pxr import Gf,UsdLux,UsdShade,Sdf,UsdGeom,Usd
from .layouts import FAMILY_LIGHTING,LIGHTING_PROFILES


def _light_baseline(source,role,temperature):
    api=UsdLux.LightAPI(source.GetPrim())
    api.CreateEnableColorTemperatureAttr(True)
    api.CreateColorTemperatureAttr(float(temperature))
    prim=source.GetPrim()
    prim.SetCustomDataByKey('scence:lightRole',role)
    prim.SetCustomDataByKey('scence:nominalIntensity',float(api.GetIntensityAttr().Get()))
    prim.SetCustomDataByKey('scence:nominalTemperatureK',float(temperature))


def light(b,name,position,target,size,intensity,color,shape='rect',*,role='ceiling',temperature=None):
    if shape=='disk':
        light=UsdLux.DiskLight.Define(b.stage,'/World/Lights/'+name)
        light.CreateRadiusAttr(size[0]/2)
    else:
        light=UsdLux.RectLight.Define(b.stage,'/World/Lights/'+name)
        light.CreateWidthAttr(size[0]);light.CreateHeightAttr(size[1])
    family=getattr(b,'family',None)
    if family in FAMILY_LIGHTING:
        spec=FAMILY_LIGHTING[family]
        if role=='ceiling':intensity*=spec['ceiling_scale']
        if temperature is None:temperature=spec['temperature_k']
    light.CreateIntensityAttr(float(intensity))
    light.CreateColorAttr(Gf.Vec3f(*(color if temperature is None else (1.,1.,1.))))
    _light_baseline(light,role,6500. if temperature is None else temperature)
    direction=Gf.Vec3d(*target)-Gf.Vec3d(*position)
    q=Gf.Rotation(Gf.Vec3d(0,0,-1),direction).GetQuat()
    from pxr import UsdGeom
    x=UsdGeom.Xformable(light);x.AddTranslateOp().Set(Gf.Vec3d(*position));x.AddOrientOp().Set(Gf.Quatf(q))
    return light.GetPrim()


def apply_lighting_profile(stage,family,profile='nominal',*,light_root='/World/Lights'):
    """Apply a reproducible variant to a new stage or a referenced room layer.

    Read the authored nominal values on every call, so nominal -> low_light ->
    nominal restores exactly. This authors only lights in the caller's edit
    target and never saves or mutates a referenced source USD.
    """
    if family not in FAMILY_LIGHTING:raise ValueError(f'Unknown room family: {family}')
    if profile not in LIGHTING_PROFILES:raise ValueError(f'Unknown lighting profile: {profile}')
    root=stage.GetPrimAtPath(light_root)
    if not root:raise ValueError(f'Missing light root: {light_root}')
    spec=LIGHTING_PROFILES[profile];entries=[]
    for prim in Usd.PrimRange(root):
        if not prim.HasAPI(UsdLux.LightAPI):continue
        role=prim.GetCustomDataByKey('scence:lightRole')
        nominal=prim.GetCustomDataByKey('scence:nominalIntensity')
        temperature=prim.GetCustomDataByKey('scence:nominalTemperatureK')
        if role not in spec['scales'] or nominal is None or temperature is None:
            raise ValueError(f'Light has no nominal profile metadata: {prim.GetPath()}')
        effective=float(nominal)*spec['scales'][role]
        kelvin=float(temperature)+(spec['temperature_shift_k'] if role not in ('sun','sky') else 0.)
        kelvin=max(2200.,min(10000.,kelvin))
        api=UsdLux.LightAPI(prim)
        api.CreateIntensityAttr(effective)
        api.CreateColorTemperatureAttr(kelvin)
        api.CreateEnableColorTemperatureAttr(True)
        entries.append(dict(path=str(prim.GetPath()),role=role,nominal_intensity=float(nominal),
                            intensity=float(api.GetIntensityAttr().Get()),temperature_k=float(api.GetColorTemperatureAttr().Get()),
                            color=list(api.GetColorAttr().Get())))
    if not entries:raise ValueError(f'No authored lights under {light_root}')
    root.SetCustomDataByKey('scence:lightingProfile',profile)
    root.SetCustomDataByKey('scence:lightingFamily',family)
    return dict(schema='scence.lighting.v1',family=family,profile=profile,light_root=light_root,
                lights=entries,scope='Authored simulation lighting; no measured-lux claim')


# Compact architecture; the fixed metric workcell is defined below.

def _room_materials(b, family):
    root=b.assets_root/'materials/ambientcg/v1'
    plaster=str(root/'PaintedPlaster017/PaintedPlaster017_1K-JPG_NormalGL.jpg')
    specs={
        'mechanical_assembly':((.40,.44,.44),(.52,.54,.53),(.14,.19,.21)),
        'laboratory':((.82,.88,.89),(.59,.73,.76),(.42,.62,.66)),
        'kitchen':((.88,.84,.72),(.74,.71,.61),(.82,.79,.67)),
        'study':((.58,.51,.39),(.43,.29,.15),(.23,.16,.095)),
        'card_room':((.70,.65,.50),(.52,.46,.30),(.16,.105,.065)),
        'living_room':((.66,.67,.62),(.55,.47,.35),(.84,.82,.74)),
    }
    wall,floor,trim=specs[family]
    b.material('room_wall',wall,.87,normal=plaster)
    b.material('room_ceiling',(.87,.86,.82) if family not in ('mechanical_assembly','card_room') else (.50,.51,.48),.93)
    b.material('room_trim',trim,.50)
    b.material('room_floor_finish',floor,.38 if family=='laboratory' else .77)
    b.material('room_grout',(.24,.28,.28),.9)
    b.material('room_warm_wood',(.45,.29,.13),.56,texture=str(root/'Wood089/Wood089_1K-JPG_Color.jpg'),texture_tint=(.70,.53,.34))
    b.material('room_study_floor',(.35,.22,.11),.64,texture=str(root/'WoodFloor064/WoodFloor064_1K-JPG_Color.jpg'),texture_tint=(.52,.48,.42))
    b.material('room_living_floor',(.61,.55,.44),.7,texture=str(root/'Wood089/Wood089_1K-JPG_Color.jpg'),texture_tint=(.85,.86,.80))
    b.material('room_concrete',(.43,.46,.46),.8,texture=str(root/'Concrete048/Concrete048_1K-JPG_Color.jpg'),normal=str(root/'Concrete048/Concrete048_1K-JPG_NormalGL.jpg'),texture_tint=(.53,.60,.65))
    b.material('room_fabric',(.68,.64,.52),.96)
    b.material('room_tatami',(.49,.44,.29),.96,normal=plaster)
    b.material('room_tatami_border',(.11,.15,.095),.98)
    b.material('room_light_glow',(1.,.94,.82),.45)
    sh=UsdShade.Shader(b.stage.GetPrimAtPath('/World/Looks/room_light_glow/Surface'))
    sh.CreateInput('emissiveColor',Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(7.,6.6,5.9))
    # Thin paper transmits the warm internal lamp; the disk light supplies room illumination.
    b.material('room_lantern_paper',(.76,.66,.49),.90)
    sh=UsdShade.Shader(b.stage.GetPrimAtPath('/World/Looks/room_lantern_paper/Surface'))
    sh.CreateInput('emissiveColor',Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(2.0,1.35,.72))
    b.material('room_clinic_white',(.90,.94,.94),.35)
    b.material('room_safety_yellow',(.65,.41,.035),.65)


def _wall_box(b, name, wall, u, z, width, height, s, material='room_wall', thickness=.14, inset=0):
    xmin,xmax,ymin,ymax=s['bounds']
    if wall in ('left','right'):
        x=xmin-thickness/2+inset if wall=='left' else xmax+thickness/2-inset
        b.box(name,(x,u,z),(thickness,width,height),material,0)
    else:
        y=ymax+thickness/2-inset if wall=='back' else ymin-thickness/2+inset
        b.box(name,(u,y,z),(width,thickness,height),material,0)


def _window(b, family, s):
    xmin,xmax,ymin,ymax=s['bounds'];w=s['window'];wall=w['wall'];c=w['center'];a=w['width']/2;lo=w['bottom'];hi=w['top']
    # Local coordinates: u along wall, v height, n toward interior.
    def panel(name,u,z,width,height,depth,mat,offset=.025,bevel=.002):
        if wall=='left':return b.box(name,(xmin+offset,u,z),(depth,width,height),mat,bevel,collision=False)
        return b.box(name,(u,ymax-offset,z),(width,depth,height),mat,bevel,collision=False)
    wood=family in ('study','card_room')
    frame='room_warm_wood' if wood else ('room_clinic_white' if family in ('kitchen','laboratory') else 'black')
    for i,u in enumerate((c-a,c+a)):
        panel('window_jamb_'+str(i),u,(lo+hi)/2,.055,hi-lo+.08,.085,frame)
    for i,z in enumerate((lo,hi)):
        panel('window_rail_'+str(i),c,z,2*a+.08,.055,.085,frame)
    panel('window_glazing',c,(lo+hi)/2,2*a-.05,hi-lo-.05,.005,'glass',offset=0,bevel=0)
    panel('window_sill',c,lo-.035,2*a+.16,.045,.18,frame,offset=.055)
    if family=='mechanical_assembly':
        for j in (-1,1):panel('workshop_mullion'+str(j),c+j*a/3,(lo+hi)/2,.025,hi-lo,.04,'black')
    elif family=='card_room':
        for j in range(7):panel('lattice_u'+str(j),c-a+.09+j*(2*a-.18)/6,(lo+hi)/2,.016,hi-lo-.07,.025,'walnut',offset=.065)
        for j in range(5):panel('lattice_v'+str(j),c,lo+.13+j*(hi-lo-.26)/4,2*a-.07,.016,.025,'walnut',offset=.065)
    elif family=='study':
        for j in range(20):panel('venetian_blind_'+str(j),c,lo+.05+j*(hi-lo-.1)/20,2*a-.07,.028,.045,'room_warm_wood',offset=.07)
        panel('blind_headrail',c,hi-.035,2*a-.03,.05,.08,'room_trim',offset=.07)
    elif family=='living_room':
        panel('window_center_mullion',c,(lo+hi)/2,.035,hi-lo,.045,frame)
        # Two folded textile panels, attached to a real curtain track.
        panel('curtain_track',c,hi+.10,2*a+.17,.028,.038,'white',offset=.15)
        for side in (-1,1):
            for j in range(11):
                u=c+side*(a-.16)+(.028*(j-5)); x=xmin+.12+.035*math.sin(j*math.pi/2)
                b.box(f'curtain_fold_{side}_{j}',(x,u,(hi+.07+.08)/2),(.025,.033,hi-.01),'room_fabric',.010,collision=False)
    else:
        panel('window_center_mullion',c,(lo+hi)/2,.035,hi-lo,.04,frame)
        if family=='laboratory':
            # Waist-high obscuring strip; upper glass still admits daylight.
            panel('lab_privacy_film',c,lo+.22,2*a-.08,.38,.001,'room_clinic_white',offset=.006,bevel=0)


def _floor_and_trims(b,family,s):
    xmin,xmax,ymin,ymax=s['bounds'];W=xmax-xmin;D=ymax-ymin;cx=(xmin+xmax)/2;cy=(ymin+ymax)/2
    floor={'mechanical_assembly':'room_concrete','study':'room_study_floor','living_room':'room_living_floor','card_room':'room_tatami'}.get(family,'room_floor_finish')
    b.box('room_floor',(cx,cy,-.09),(W,D,.18),floor,0)
    if family=='kitchen':
        for axis,lo,hi,other,center in [('x',xmin,xmax,D,cy),('y',ymin,ymax,W,cx)]:
            n=math.ceil((hi-lo)/.45)
            for i in range(1,n):
                q=lo+i*.45
                p=(q,center,.0001) if axis=='x' else (center,q,.0001)
                size=(.003,other,.0002) if axis=='x' else (other,.003,.0002)
                b.box('floor_grout_'+axis+str(i),p,size,'room_grout',0,collision=False)
    elif family=='card_room':
        for i in range(1,5):
            x=xmin+W*i/5
            b.box('tatami_long_border'+str(i),(x,cy,.0002),(.023,D,.0004),'room_tatami_border',0,collision=False)
        for i in range(1,3):
            y=ymin+D*i/3
            b.box('tatami_cross_border'+str(i),(cx,y,.0002),(W,.023,.0004),'room_tatami_border',0,collision=False)
    elif family=='living_room':
        for i in range(1,math.ceil(W/.19)):
            x=xmin+i*.19
            b.box('oak_plank_seam'+str(i),(x,cy,.0001),(.0015,D,.0002),'room_grout',0,collision=False)
    elif family=='mechanical_assembly':
        # ESD work area boundary remains clear of the pedestal and cabinet feet.
        for i,y in enumerate((-.70,.83)):
            b.box('workzone_floor_line'+str(i),(.65,y,.0002),(1.47,.026,.0004),'room_safety_yellow',0,collision=False)
    trim_h=.13 if family in ('study','card_room') else .07
    for wall,u,width in [('left',cy,D),('right',cy,D),('back',cx,W),('front',cx,W)]:
        _wall_box(b,'skirting_'+wall,wall,u,trim_h/2,width,trim_h,s,'room_trim',.025,inset=.025)
    if family in ('study','card_room'):
        # Wood wall panelling gives both rooms a domestic scale; the lattice is unique to tea room.
        h=.78 if family=='card_room' else .68
        _wall_box(b,'wood_wainscot','back',cx,h/2,W,h,s,'walnut',.024,inset=.035)
        for i in range(int(W/.42)):
            x=xmin+.1+i*.42
            b.box('wainscot_stile'+str(i),(x,ymax-.028,h/2),(.024,.015,h-.04),'room_warm_wood',.002,collision=False)
    elif family=='laboratory':
        _wall_box(b,'lab_service_band','back',cx,.94,W,.18,s,'room_trim',.012,inset=.025)
    elif family=='mechanical_assembly':
        _wall_box(b,'workshop_lower_wall','back',cx,.45,W,.90,s,'room_trim',.015,inset=.025)


def _door_and_services(b,family,s):
    xmin,xmax,ymin,ymax=s['bounds'];x=xmax-.025;y=-.85
    mat='steel' if family=='mechanical_assembly' else ('room_clinic_white' if family=='laboratory' else 'room_warm_wood')
    b.box('door_reveal',(x,y,1.03),(.025,.91,2.06),'black',.002)
    b.box('door_leaf',(x-.022,y,1.02),(.04,.85,2.02),mat,.003)
    for j,yy in enumerate((y-.47,y+.47)):b.box('door_jamb'+str(j),(x-.047,yy,1.04),(.065,.055,2.08),'room_trim',.003)
    b.box('door_header',(x-.047,y,2.10),(.065,1.0,.055),'room_trim',.003)
    b.tube('door_lever',[(x-.06,y-.30,1.0),(x-.11,y-.30,1.0),(x-.11,y-.18,1.0)],.008,'chrome')
    if family=='laboratory':
        b.box('lab_door_vision_frame',(x-.048,y,1.52),(.015,.38,.58),'steel',.01,collision=False)
        b.box('lab_door_vision_glass',(x-.059,y,1.52),(.005,.34,.54),'glass',.006,collision=False)
    # Outlets are appropriate to each room and sit on the back wall above skirting.
    for i,xx in enumerate((.20,1.35)):
        b.box('power_outlet'+str(i),(xx,ymax-.012,.30),(.13,.025,.075),'white',.005)
        for j,dx in enumerate((-.025,.025)):
            b.box(f'outlet_slot_{i}_{j}',(xx+dx,ymax-.026,.30),(.004,.002,.014),'black',.001,collision=False)


def _ceiling_lights(b,family,s):
    xmin,xmax,ymin,ymax=s['bounds'];z=s['height']
    cool=family in ('mechanical_assembly','laboratory')
    color=(.90,.96,1.) if cool else {
        'kitchen':(1.,.97,.90), 'study':(1.,.94,.84),
        'card_room':(1.,.93,.80), 'living_room':(1.,.94,.86),
    }[family]
    if family=='mechanical_assembly':
        for i,y in enumerate((-.30,1.2)):
            b.box('linear_light_body'+str(i),(.45,y,z-.10),(1.65,.15,.055),'steel',.008,collision=False)
            b.box('linear_light_diffuser'+str(i),(.45,y,z-.131),(1.57,.09,.012),'room_light_glow',.003,collision=False)
            light(b,'strip'+str(i),(.45,y,z-.14),(.45,y,0),(1.5,.08),42000,color)
        for i,x in enumerate((-.9,1.7)):
            b.box('exposed_ceiling_beam'+str(i),(x,.3,z-.07),(.09,ymax-ymin,.14),'room_trim',.004)
    elif family=='laboratory':
        # Cleanroom ceiling grid and flush medical light panels.
        for i in range(1,7):
            x=xmin+(xmax-xmin)*i/7
            b.box('ceiling_grid_x'+str(i),(x,(ymin+ymax)/2,z-.006),(.012,ymax-ymin,.012),'room_trim',.001,collision=False)
        for i in range(1,6):
            y=ymin+(ymax-ymin)*i/6
            b.box('ceiling_grid_y'+str(i),((xmin+xmax)/2,y,z-.006),(xmax-xmin,.012,.012),'room_trim',.001,collision=False)
        for i,(x,y) in enumerate(((.15,-.30),(.90,1.15))):
            b.box('lab_flush_panel'+str(i),(x,y,z-.015),(1.0,.50,.018),'room_light_glow',.002,collision=False)
            light(b,'lab_panel'+str(i),(x,y,z-.026),(x,y,0),(.95,.45),5600,color)
    elif family=='card_room':
        for i,(x,y) in enumerate(((.50,-.03),(.35,1.35))):
            b.cylinder('pendant_cord'+str(i),(x,y,z-.22),.004,.44,'black',collision=False)
            b.lathe('woven_pendant'+str(i),(x,y,z-.68),[(.21,0),(.24,.09),(.19,.29),(.055,.38),(.051,.38),(.185,.285),(.235,.09),(.205,.006)],'room_lantern_paper')
            for j in range(24):
                a=j*math.pi/12
                b.tube(f'pendant_rib_{i}_{j}',[(x+.21*math.cos(a),y+.21*math.sin(a),z-.68),(x+.24*math.cos(a),y+.24*math.sin(a),z-.59),(x+.19*math.cos(a),y+.19*math.sin(a),z-.39)],.002,'room_warm_wood')
            light(b,'tea_pendant'+str(i),(x,y,z-.69),(x,y,0),(.38,.38),36000,color,shape='disk')
    elif family=='study':
        # A warm ceiling pendant plus library wash, without industrial square panels.
        b.cylinder('study_opal_shade',(.50,-.15,z-.15),.29,.12,'room_light_glow',collision=False)
        light(b,'study_pendant',(.50,-.15,z-.215),(.50,-.15,0),(.5,.5),21000,color,shape='disk')
        light(b,'shelf_wash',(.3,ymax-.45,z-.07),(.3,ymax,1.2),(2.2,.09),11000,(1.,.90,.78),role='accent',temperature=3000.)
    else:
        # Small recessed fixtures suit kitchens and lounges; each has its own visible trim.
        for i,(x,y) in enumerate(((-.80,-.55),(1.35,-.55),(-.80,1.18),(1.35,1.18))):
            b.cylinder('downlight_trim'+str(i),(x,y,z-.012),.085,.025,'room_trim',collision=False)
            b.cylinder('downlight_diffuser'+str(i),(x,y,z-.027),.067,.008,'room_light_glow',collision=False)
            intensity=250000 if family=='kitchen' else 220000
            light(b,'downlight'+str(i),(x,y,z-.033),(x,y,0),(.10,.10),intensity,color,shape='disk')
    sun=UsdLux.DistantLight.Define(b.stage,'/World/Lights/Sun')
    lighting=FAMILY_LIGHTING[family]
    sun.CreateIntensityAttr(lighting['sun']);sun.CreateAngleAttr(.65)
    sun.CreateColorAttr(Gf.Vec3f(1.,1.,1.));_light_baseline(sun,'sun',5600.)
    q=Gf.Rotation(Gf.Vec3d(0,0,-1),Gf.Vec3d(1.,-.3,-.6) if s['window']['wall']=='left' else Gf.Vec3d(.5,-1.,-.5)).GetQuat()
    UsdGeom.Xformable(sun).AddOrientOp().Set(Gf.Quatf(q))
    dome=UsdLux.DomeLight.Define(b.stage,'/World/Lights/Sky')
    hdr=b.assets_root/'skies/polyhaven/v2/kloofendal_48d_partly_cloudy_puresky/kloofendal_48d_partly_cloudy_puresky_2k.hdr'
    dome.CreateTextureFileAttr(str(hdr));dome.CreateIntensityAttr(lighting['sky'])
    _light_baseline(dome,'sky',6500.);b.dependency(hdr)


def build_room(b,family):
    from .layouts import ROOM_SPECS
    s=ROOM_SPECS[family];xmin,xmax,ymin,ymax=s['bounds'];h=s['height'];w=s['window']
    _room_materials(b,family);_floor_and_trims(b,family,s)
    b.box('room_ceiling',((xmin+xmax)/2,(ymin+ymax)/2,h+.07),(xmax-xmin,ymax-ymin,.14),'room_ceiling',0)
    for wall,lo,hi in [('left',ymin,ymax),('right',ymin,ymax),('back',xmin,xmax),('front',xmin,xmax)]:
        if wall!=w['wall']:
            _wall_box(b,'wall_'+wall,wall,(lo+hi)/2,h/2,hi-lo,h,s)
        else:
            a=w['center']-w['width']/2;c=w['center']+w['width']/2
            for name,left,right,z,height in [('before',lo,a,h/2,h),('after',c,hi,h/2,h),('below',a,c,w['bottom']/2,w['bottom']),('above',a,c,(h+w['top'])/2,h-w['top'])]:
                _wall_box(b,'wall_'+wall+'_'+name,wall,(left+right)/2,z,right-left,height,s)
    _window(b,family,s);_door_and_services(b,family,s);_ceiling_lights(b,family,s)


def build_workcell(b,family,*,include_robot_station=True):
    top='steel' if family=='laboratory' else ('oak' if family in ('study','living_room','card_room') else 'white')
    # Exact outer surface bounds x=.15..1.15, y=-.5..+.5, z=.74.. .78.
    b.box('worktop',(.65,0,.76),(1.,1.,.04),top,.004)
    b.box('table_edge_shadow',(.65,0,.736),(.984,.984,.008),'black',.003)
    for i,x in enumerate((.22,1.08)):
        for j,y in enumerate((-.43,.43)):
            b.box(f'table_leg_{i}_{j}',(x,y,.369),(.047,.047,.702),'paint',.003)
            b.cylinder(f'table_foot_{i}_{j}',(x,y,.012),.032,.024,'rubber')
            b.cylinder(f'table_level_screw_{i}_{j}',(x,y,.033),.011,.025,'chrome')
            for k,z in enumerate((.65,.69)):
                b.cylinder(f'table_bolt_{i}_{j}_{k}',(x,y-.025,z),.006,.006,'chrome',(90,0,0),False)
    for i,y in enumerate((-.43,.43)):b.box('table_frame_long_'+str(i),(.65,y,.694),(.86,.032,.068),'paint',.002)
    for i,x in enumerate((.22,1.08)):b.box('table_frame_cross_'+str(i),(x,0,.694),(.032,.86,.068),'paint',.002)
    b.box('table_rear_brace',(.65,.43,.25),(.86,.025,.035),'paint',.002)
    if not include_robot_station:
        # New task-independent rooms keep robot supports in task composition.
        return
    # Robot fixed support is outside tabletop; visible, anchored load path down to floor.
    b.box('robot_anchor_plate',(-.13,0,.018),(.36,.40,.036),'steel',.005)
    b.box('robot_column',(-.13,0,.385),(.22,.25,.698),'paint',.009)
    b.box('robot_mounting_plate',(-.13,0,.755),(.34,.36,.05),'steel',.004)
    for i,x in enumerate((-.27,.01)):
        for j,y in enumerate((-.16,.16)):
            b.cylinder(f'anchor_washer_{i}_{j}',(x,y,.039),.014,.006,'steel',collision=False)
            b.cylinder(f'anchor_bolt_{i}_{j}',(x,y,.049),.009,.016,'chrome',collision=False)
    for i,x in enumerate((-.25,-.01)):
        for j,y in enumerate((-.12,.12)):
            b.cylinder(f'mount_bolt_{i}_{j}',(x,y,.784),.008,.008,'chrome',collision=False)
    # External wiring/cable conduit follows the back of the column down to the floor.
    b.tube('robot_power_conduit',[(-.26,.10,.76),(-.28,.13,.65),(-.28,.13,.10),(-.30,.23,.055),(-.65,.5,.035),(-1.15,1.0,.035)],.014,'black')
    b.box('robot_control_enclosure',(-.69,.53,.245),(.30,.26,.49),'white',.012)
    for i in range(9):b.box('control_vent_'+str(i),(-.69,.397,.32+i*.012),(.20,.004,.004),'black',.001,collision=False)
    b.cylinder('controller_power_button',(-.62,.392,.16),.012,.008,'green',(90,0,0),False)
    b.label('pedestal_identification','FRANKA\nFIXED BASE',(-.13,-.126,.46),(.15,.08),rotation=(90,0,0),font_size=42)
    b.label('table_dimension_label','WORKSPACE  /  1000 × 1000 mm',(.65,-.501,.755),(.30,.022),rotation=(90,0,0),font_size=34)
