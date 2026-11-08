"""A furnished residential laundry: real-scale joinery and static household contents.

The room owns its architecture and dressing. The clear folding surface is
reserved for future task-owned cloth and robots; appliances here are scenery.
"""
from copy import deepcopy
import json
import math
from pathlib import Path

from pxr import Gf, Sdf, UsdGeom, UsdLux, UsdShade

from .builder import Builder
from .common import light, _light_baseline, apply_lighting_profile
from bench.paths import load_json

FAMILY = 'laundry_room'
VARIANT = 'laundry_residential_20261021_v3'
ROOM_SPEC = {'bounds': [-2.6, 2.8, -2.3, 2.2], 'height': 2.75,
             'theme': 'Residential utility room with laundry, drying, ironing and linen storage',
             'window': {'wall': 'left', 'center': .85, 'width': 1.75, 'bottom': 1.12, 'top': 2.40}}
TABLE = {'center_xy': [.65, -.55], 'size_m': [1.8, .9], 'top_z': .90,
         'collider_path': '/World/Geometry/worktop/Collider'}
RESERVED_TASK_VOLUME = {'low': [-.10, -.94, .90001], 'high': [1.40, -.16, 2.10]}
CAMERAS = {
    'room': {'eye': [-.75, -2.10, 1.72], 'target': [.15, .85, 1.08], 'focal_length_mm': 17.5},
    'workcell': {'eye': [2.47, -2.05, 1.87], 'target': [-.45, .85, 1.03], 'focal_length_mm': 18.},
    'tabletop': {'eye': [.22, -2.04, 1.85], 'target': [.65, -.13, .81], 'focal_length_mm': 23.},
    'appliances': {'eye': [-.50, .35, 1.40], 'target': [-1.65, 1.80, .90], 'focal_length_mm': 23.},
    'storage': {'eye': [-.85, .52, 1.70], 'target': [2.05, -.35, 1.14], 'focal_length_mm': 22.},
}


def _materials(b):
    root = b.assets_root / 'materials/polyhaven/v1'
    for name, slug, res, tint, rough in (
            ('laundry_floor', 'terrazzo_tiles', '2k', (1.6, 2.5, 3.4), .67),
            ('laundry_tile', 'interior_tiles', '2k', (.92, .98, .93), .32),
            ('laundry_oak', 'kitchen_wood', '1k', (1.15, 1.08, .96), .66)):
        folder = root / slug
        b.material(name, (.7, .7, .65), rough,
                   texture=str(folder/f'{slug}_diff_{res}.jpg'),
                   normal=str(folder/f'{slug}_nor_gl_{res}.jpg'),
                   roughness_texture=str(folder/f'{slug}_rough_{res}.jpg'), texture_tint=tint)
        b.dependency(folder/'SOURCE.json')
    rough=UsdShade.Shader(b.stage.GetPrimAtPath('/World/Looks/laundry_oak/roughnessTexture'))
    rough.CreateInput('scale',Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(1.35,1.35,1.35,1.))
    normal = b.assets_root/'materials/ambientcg/v1/PaintedPlaster017/PaintedPlaster017_1K-JPG_NormalGL.jpg'
    for name, color, rough, metal in (
            ('laundry_wall', (.78, .78, .71), .9, 0),
            ('laundry_ceiling', (.87, .86, .80), .9, 0),
            ('laundry_sage', (.30, .40, .33), .5, 0),
            ('laundry_cream', (.82, .81, .73), .4, 0),
            ('laundry_quartz', (.78, .79, .74), .3, 0),
            ('laundry_grout', (.40, .42, .39), .9, 0),
            ('laundry_brass', (.45, .33, .17), .33, .85),
            ('laundry_shadow', (.035, .045, .039), .8, 0),
            ('laundry_white_enamel', (.84, .86, .84), .25, .05),
            ('laundry_display', (.009, .022, .024), .16, .05),
            ('laundry_led', (.45, .68, .65), .35, 0),
            ('laundry_diffuser', (.86, .85, .76), .55, 0),
            ('laundry_window_wood', (.71, .67, .54), .6, 0),
            ('laundry_terracotta', (.38, .19, .11), .9, 0),
            ('laundry_leaf', (.12, .23, .10), .8, 0)):
        b.material(name, color, rough, metal,
                   normal=str(normal) if name in ('laundry_wall', 'laundry_ceiling') else None)
    for i,color in enumerate(((.80,.83,.79),(.76,.79,.75),(.83,.84,.79))):
        b.material('laundry_glazed_tile_'+str(i),color,.27)
    for name, emissive in (('laundry_diffuser', (1.8, 1.73, 1.5)), ('laundry_led', (.45, 1.2, 1.05))):
        shader = UsdShade.Shader(b.stage.GetPrimAtPath('/World/Looks/'+name+'/Surface'))
        shader.CreateInput('emissiveColor', Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*emissive))


def _architecture(b):
    xmin, xmax, ymin, ymax = ROOM_SPEC['bounds']; h = ROOM_SPEC['height']
    b.box('room_floor', (.1, -.05, -.09), (5.4, 4.5, .18), 'laundry_floor', 0)
    for axis,lo,hi in ((0,xmin,xmax),(1,ymin,ymax)):
        for i in range(1,math.ceil((hi-lo)/.60)):
            p=[.1,-.05,.00015];p[axis]=lo+i*.60
            size=[5.4,4.5,.0003];size[axis]=.0025
            b.box(f'floor_grout_{axis}_{i}',p,size,'laundry_grout',0,collision=False)
    # Separate real wall segments leave an actual window and entrance opening.
    b.box('wall_back', (.1, 2.28, h/2), (5.72, .16, h), 'laundry_wall', 0)
    b.box('wall_right', (2.88, -.05, h/2), (.16, 4.5, h), 'laundry_wall', 0)
    for name, yc, length, zc, height in (
            ('left_front', -1.1625, 2.275, h/2, h),
            ('left_back', 1.9625, .475, h/2, h),
            ('left_sill', .85, 1.75, .56, 1.12),
            ('left_lintel', .85, 1.75, 2.575, .35)):
        b.box('wall_'+name, (-2.68, yc, zc), (.16, length, height), 'laundry_wall', 0)
    # Closed domestic entrance has joinery, inset panels and brass hardware.
    b.box('wall_front_left', (-2.425, -2.38, h/2), (.35, .16, h), 'laundry_wall', 0)
    b.box('wall_front_right', (.875, -2.38, h/2), (3.85, .16, h), 'laundry_wall', 0)
    b.box('wall_door_lintel', (-1.65, -2.38, 2.435), (1.2, .16, .63), 'laundry_wall', 0)
    b.box('entrance_door', (-1.65, -2.35, 1.05), (1.13, .05, 2.10), 'laundry_cream', .006)
    for x in (-2.245, -1.055):
        b.box('door_jamb_'+str(x), (x, -2.285, 1.075), (.065, .04, 2.15), 'laundry_cream', .003)
    b.box('door_header', (-1.65, -2.285, 2.145), (1.25, .04, .065), 'laundry_cream', .003)
    for i, z in enumerate((.49, 1.43)):
        b.box('door_panel_'+str(i), (-1.65, -2.321, z), (.88, .012, .73), 'laundry_wall', .008, collision=False)
    b.cylinder('door_rose', (-1.22, -2.301, 1.04), .029, .012, 'laundry_brass', rotation=(90,0,0), collision=False)
    b.tube('door_handle', [(-1.22,-2.29,1.04),(-1.22,-2.245,1.04),(-1.36,-2.245,1.04)], .008, 'laundry_brass')
    b.box('ceiling', (.1,-.05,2.79), (5.72,4.82,.08), 'laundry_ceiling', 0)
    for name, center, size in (
            ('back',(.1,2.181,.055),(5.4,.035,.11)),
            ('left',(-2.581,-.05,.055),(.035,4.5,.11)),
            ('right',(2.781,-.05,.055),(.035,4.5,.11)),
            ('front_left',(-2.43,-2.281,.055),(.34,.035,.11)),
            ('front_right',(.89,-2.281,.055),(3.82,.035,.11))):
        b.box('skirting_'+name,center,size,'laundry_cream',.003)
    # Wide double-glazed window with frame, opening handles and outside reveal.
    for i,y in enumerate((-.025,1.725)):
        b.box('window_jamb_'+str(i),(-2.57,y,1.76),(.105,.055,1.34),'laundry_cream',.002)
    for i,z in enumerate((1.12,2.40)):
        b.box('window_frame_'+str(i),(-2.57,.85,z),(.105,1.81,.055),'laundry_cream',.002)
    b.box('window_mullion',(-2.57,.85,1.76),(.08,.045,1.25),'laundry_cream',.002)
    for i,y in enumerate((.423,1.277)):
        b.box('window_glass_'+str(i),(-2.60,y,1.76),(.005,.85,1.24),'glass',0,collision=False)
        b.tube('window_handle_'+str(i),[(-2.508,y+.30,1.65),(-2.47,y+.30,1.65),(-2.47,y+.30,1.76)],.006,'laundry_brass')
    b.box('window_sill',(-2.52,.85,1.088),(.25,1.93,.04),'laundry_quartz',.006)
    b.box('blind_headrail',(-2.47,.85,2.47),(.07,1.91,.065),'laundry_window_wood',.004,collision=False)
    for i in range(9):
        b.box('blind_slat_'+str(i),(-2.471,.85,2.405-i*.027),(.045,1.79,.019),'laundry_window_wood',.002,rotation=(0,16,0),collision=False)
    for side in (-1,1):
        b.tube('blind_ladder_'+str(side),[(-2.443,.85+side*.60,2.46),(-2.443,.85+side*.60,2.17)],.0017,'fabric')
    # Actual outside ledge, brick garden wall, foliage, and distant sky.
    b.box('outside_ledge',(-2.96,.85,1.05),(.64,1.96,.08),'laundry_quartz',.005)
    b.box('courtyard_ground',(-4.3,.5,-.17),(2.8,5.5,.12),'floor',0)
    b.box('courtyard_wall',(-5.5,.5,.48),(.16,5.5,1.10),'laundry_wall',0)
    for j in range(7):
        for i in range(9):
            b.box(f'garden_brick_{j}_{i}',(-5.407,-2.1+i*.60+(j%2)*.30,.04+j*.145),(.028,.57,.124),'laundry_terracotta',.003,collision=False)
    for i,y in enumerate((-.25,.8,1.8)):
        b.lathe('outside_planter_'+str(i),(-3.85,y,0),[(.12,0),(.15,.30),(.14,.32),(.125,.30),(.10,.03)],'laundry_terracotta')
        for j in range(13):
            a=j*2.399; z=.4+j*.036
            b.tube(f'plant_stem_{i}_{j}',[(-3.85,y,.27),(-3.85+.12*math.cos(a),y+.12*math.sin(a),z)],.003,'laundry_leaf')
            b.sphere(f'plant_leaf_{i}_{j}',(-3.85+.12*math.cos(a),y+.12*math.sin(a),z),(.085,.04,.025),'laundry_leaf')


def _cabinet(b, name, x, width, *, y=1.835, depth=.61, top=.88, material='laundry_sage'):
    b.box(name+'_toe',(x,y+.035,.075),(width-.04,depth-.12,.15),'laundry_shadow',.002)
    for side in (-1,1):
        b.box(name+'_side_'+str(side),(x+side*(width/2-.012),y,.50),(.024,depth,.74),material,.004)
    b.box(name+'_back',(x,y+depth/2-.009,.50),(width,.018,.74),material,.002)
    b.box(name+'_bottom',(x,y,.145),(width-.04,depth-.02,.03),material,.002)
    front=y-depth/2-.012
    count=2 if width>.75 else 1
    for i in range(count):
        cx=x+(i-(count-1)/2)*width/count; dw=width/count-.012
        b.box(name+'_door_'+str(i),(cx,front,.505),(dw,.023,.702),material,.003)
        # Shaker frame and recessed centre have distinct surfaces and reveal.
        b.box(name+'_inset_'+str(i),(cx,front-.013,.505),(dw-.10,.004,.59),material,.002,collision=False)
        for side in (-1,1):
            b.box(f'{name}_stile_{i}_{side}',(cx+side*(dw/2-.025),front-.019,.505),(.050,.018,.702),material,.003,collision=False)
        for z in (.179,.831):
            b.box(f'{name}_rail_{i}_{z}',(cx,front-.019,z),(dw-.05,.018,.05),material,.003,collision=False)
        hx=cx+(.12 if count==1 or i==0 else -.12)
        b.tube(name+'_handle_'+str(i),[(hx,front-.025,.64),(hx,front-.055,.64),(hx,front-.055,.77),(hx,front-.025,.77)],.006,'laundry_brass')


def _casework(b):
    # Solid counters are segmented around a real hollow sink opening.
    for name,x,w in (('washers',-1.535,1.68),('sink_left',-.635,.12),('sink_right',.135,.12),('storage',1.36,2.33)):
        b.box('counter_'+name,(x,1.82,.90),(w,.67,.04),'laundry_quartz',.006)
    b.box('sink_counter_back',(-.25,2.104,.90),(.65,.102,.04),'laundry_quartz',.004)
    b.box('sink_counter_front',(-.25,1.535,.90),(.65,.10,.04),'laundry_quartz',.004)
    _cabinet(b,'sink_cabinet',-.25,.90)
    # Replace the apparent centre panel with a dark toe shadow; bowl sits above body.
    _cabinet(b,'supply_cabinet',.78,1.12)
    _cabinet(b,'linen_cabinet',1.94,1.12)
    # Splash tiles have deliberate, independently modelled grout courses.
    b.box('backsplash',(0.08,2.179,1.20),(4.95,.022,.55),'laundry_grout',.002)
    for row in range(4):
        for col in range(20):
            x=-2.263+col*.247
            b.box(f'splash_ceramic_{row}_{col}',(x,2.160,1.005+row*.135),(.244,.012,.132),
                  'laundry_glazed_tile_'+str((row+col*7)%3),.002,collision=False)
    # Shallow upper cupboards over the machines, recessed panel doors.
    for i,x in enumerate((-1.98,-1.26)):
        b.box('upper_body_'+str(i),(x,1.995,2.025),(.69,.37,.81),'laundry_cream',.005)
        b.box('upper_front_'+str(i),(x,1.795,2.025),(.67,.025,.79),'laundry_sage',.003)
        b.box('upper_recess_'+str(i),(x,1.780,2.025),(.55,.005,.65),'laundry_sage',.002,collision=False)
        b.tube('upper_handle_'+str(i),[(x+.22,1.77,1.72),(x+.22,1.73,1.72),(x+.22,1.73,1.84),(x+.22,1.77,1.84)],.006,'laundry_brass')
    shelves=[]
    for col,x in enumerate((.95,1.95)):
        for row,z in enumerate((1.55,1.95,2.30)):
            name=f'laundry_open_shelf_{col}_{row}'
            b.box(name,(x,1.99,z-.016),(.88,.31,.032),'laundry_oak',.005)
            for side in (-1,1):
                xx=x+side*.31
                b.tube(f'{name}_bracket_{side}',[(xx,2.14,z-.15),(xx,2.14,z-.033),(xx,1.88,z-.033)],.007,'laundry_brass')
            shelves.append({'id':name,'top_z':z,'center':[x,1.99],'size':[.88,.31]})
    # Tall side linen cupboard (faces -X); open lower bays add visual depth.
    for y in (.16,1.10):
        b.box('tall_side_'+str(y),(2.52,y,1.17),(.47,.025,2.26),'laundry_cream',.004)
    b.box('tall_back',(2.752,.63,1.17),(.025,.96,2.26),'laundry_sage',.003)
    for i,z in enumerate((.10,.46,.83,1.20,1.63,2.08,2.30)):
        b.box('tall_shelf_'+str(i),(2.52,.63,z),(.47,.96,.028),'laundry_oak',.003)
    for i,y in enumerate((.39,.87)):
        b.box('tall_door_'+str(i),(2.273,y,1.765),(.027,.46,1.04),'laundry_sage',.004)
        b.box('tall_door_panel_'+str(i),(2.255,y,1.765),(.007,.35,.91),'laundry_sage',.003,collision=False)
        b.tube('tall_handle_'+str(i),[(2.25,y,1.36),(2.215,y,1.36),(2.215,y,1.52),(2.25,y,1.52)],.006,'laundry_brass')
    # Island's usable top stays empty. Under-counter cubbies carry future storage.
    b.box('worktop',(.65,-.55,.875),(1.8,.9,.05),'laundry_oak',.009)
    for x in (-.18,1.48):
        b.box('island_end_'+str(x),(x,-.55,.455),(.065,.78,.79),'laundry_cream',.004)
    b.box('island_lower_shelf',(.65,-.55,.18),(1.69,.78,.04),'laundry_oak',.004)
    b.box('island_back_rail',(.65,-.185,.79),(1.68,.045,.11),'laundry_cream',.004)
    b.box('island_mid_partition',(.65,-.55,.49),(.032,.77,.59),'laundry_cream',.003)
    for side,x in enumerate((-.13,1.43)):
        for j,y in enumerate((-.86,-.24)):
            b.cylinder(f'island_leveling_foot_{side}_{j}',(x,y,.038),.035,.076,'rubber')
    b.tube('island_towel_rail',[(1.52,-.85,.76),(1.59,-.85,.76),(1.59,-.27,.76),(1.52,-.27,.76)],.009,'laundry_brass')
    return shelves


def _sink_and_services(b):
    # Bowl is composed of separate bottom and walls, not a filled solid cuboid.
    b.box('sink_bottom',(-.25,1.80,.695),(.62,.486,.025),'laundry_white_enamel',.009)
    for x in (-.55,.05):
        b.box('sink_side_'+str(x),(x,1.80,.80),(.026,.46,.23),'laundry_white_enamel',.012)
    for y in (1.57,2.03):
        b.box('sink_side_'+str(y),(-.25,y,.80),(.62,.026,.23),'laundry_white_enamel',.012)
    for x in (-.57,.07):
        b.box('sink_rim_x_'+str(x),(x,1.80,.928),(.045,.52,.019),'laundry_white_enamel',.008)
    for y in (1.55,2.05):
        b.box('sink_rim_y_'+str(y),(-.25,y,.928),(.68,.045,.019),'laundry_white_enamel',.008)
    b.cylinder('sink_drain',(-.25,1.80,.710),.036,.004,'chrome',collision=False)
    for i in range(8):
        a=i*math.pi/4
        b.cylinder('drain_slot_'+str(i),(-.25+.022*math.cos(a),1.80+.022*math.sin(a),.713),.003,.001,'black',collision=False)
    b.cylinder('faucet_foot',(-.25,2.09,.955),.032,.06,'chrome')
    pts=[(-.25,2.09,.95),(-.25,2.09,1.16)]
    pts += [(-.25,2.01+.08*math.cos(a),1.16+.08*math.sin(a)) for a in [i*math.pi/20 for i in range(21)]]
    pts.append((-.25,1.93,1.12))
    b.tube('gooseneck_faucet',pts,.014,'chrome',collision=True)
    b.tube('faucet_lever',[(-.19,2.09,.96),(-.14,2.09,.97),(-.13,2.09,1.07)],.007,'chrome')
    for i,x in enumerate((-.72,.35,1.68)):
        b.box('counter_socket_'+str(i),(x,2.155,1.125),(.14,.026,.082),'laundry_cream',.005)
        for dx in (-.035,.035):
            b.box(f'socket_recess_{i}_{dx}',(x+dx,2.139,1.125),(.041,.004,.045),'laundry_white_enamel',.006,collision=False)
            for dd in (-.008,.008):
                b.box(f'socket_pin_{i}_{dx}_{dd}',(x+dx+dd,2.135,1.125),(.003,.002,.013),'black',.001,collision=False)
    # Wall vent with individual fins and a visible exhaust grille.
    b.box('vent_frame',(2.774,-.27,2.35),(.035,.34,.29),'laundry_cream',.006)
    b.box('vent_dark',(2.752,-.27,2.35),(.008,.29,.24),'laundry_shadow',.002,collision=False)
    for i in range(10):
        b.box('vent_louver_'+str(i),(2.744,-.27,2.242+i*.024),(.018,.29,.010),'laundry_cream',.001,collision=False)
    # Floor drain, water shutoffs and a utility access hatch are actual geometry.
    b.box('floor_drain',(-.65,1.27,.0015),(.12,.12,.003),'steel',.002,collision=False)
    for i in range(7):
        b.box('floor_drain_slot_'+str(i),(-.698+i*.016,1.27,.0031),(.005,.096,.0004),'black',0,collision=False)
    b.box('service_hatch',(-2.58,1.94,.52),(.03,.29,.38),'laundry_cream',.004)
    for i,x in enumerate((-2.16,-1.90)):
        b.tube('water_pipe_'+str(i),[(x,2.165,.15),(x,2.165,1.12),(x,2.10,1.12)],.009,'chrome')
        b.torus('water_valve_'+str(i),(x,2.06,1.12),.025,.004,'red' if i==0 else 'navy',rotation=(90,0,0))
    b.label('laundry_care_chart','LAUNDRY CARE\nSORT  ·  WASH  ·  DRY\n30° gentle    40° cotton\nAIR DRY DELICATES',(-.36,2.167,1.95),(.57,.48,.0003),rotation=(90,0,0),font_size=78)
    b.box('care_chart_frame',(-.36,2.177,1.95),(.61,.014,.52),'laundry_oak',.003,collision=False)


def _appliances(b):
    # Source conversion is recorded externally and never modifies upstream data.
    base=b.assets_root/'objects/innerscene/20261021/derived_zup_v1'
    sources=[]
    for name,slug,x in (('washing_machine','front-load-washing-machine',-1.95),
                        ('heat_pump_dryer','tumble-dryer',-1.26)):
        converted=base/slug/(slug+'.usdc')
        if not converted.is_file():raise FileNotFoundError(converted)
        source=converted.parent/'DERIVATION.json';b.dependency(source)
        derivation=load_json(source)
        b.dependency(derivation['source']);b.dependency(derivation['source_provenance'])
        b.asset(name,converted,(x,1.81,0.))
        sources.append({'name':name,'source':str(converted),'provenance':derivation['source_provenance'],
                        'role':'static CC0 appliance, not articulated','position_m':[x,1.81,0.],
                        'dimensions_xyz_m':derivation['dimensions_xyz_m']})
    return sources


def _lighting(b):
    for i,(x,y) in enumerate(((-1.15,-.65),(1.30,-.65),(-1.15,1.04),(1.30,1.04))):
        b.lathe('ceiling_light_rim_'+str(i),(x,y,2.72),[(0,-.0175),(.145,-.0175),(.15,-.012),(.15,.012),(.145,.0175),(0,.0175)],'laundry_cream')
        b.lathe('ceiling_light_glow_'+str(i),(x,y,2.699),[(0,-.004),(.133,-.004),(.137,0),(.133,.004),(0,.004)],'laundry_diffuser')
        light(b,'ceiling_'+str(i),(x,y,2.69),(x,y,.2),(.265,.265),22000.,(1,1,1),shape='disk',role='ceiling',temperature=4200.)
    for i,x in enumerate((-1.98,-1.26)):
        b.box('undercabinet_diffuser_'+str(i),(x,1.96,1.611),(.58,.035,.008),'laundry_diffuser',.002,collision=False)
        light(b,'counter_task_'+str(i),(x,1.95,1.60),(x,1.75,.9),(.54,.035),10500.,(1,1,1),role='accent',temperature=4000.)
    daylight = light(b,'window_daylight',(-2.46,.85,1.75),(.5,.35,.75),(1.60,1.10),17000.,(1,1,1),role='accent',temperature=6000.)
    # RTX ghost lights preserve this window fill's diffuse illumination while
    # hiding its helper rectangle in camera, glossy and glass-transmission rays.
    # Primary visibility alone does not hide a light seen through OmniGlass.
    daylight.CreateAttribute('visibleInPrimaryRay', Sdf.ValueTypeNames.Bool).Set(False)
    daylight.CreateAttribute('ghostFactor', Sdf.ValueTypeNames.Float).Set(1.0)
    sun=UsdLux.DistantLight.Define(b.stage,'/World/Lights/Sun')
    sun.CreateIntensityAttr(1600.);sun.CreateAngleAttr(3.);_light_baseline(sun,'sun',5600.)
    q=Gf.Rotation(Gf.Vec3d(0,0,-1),Gf.Vec3d(1.,-.30,-.65)).GetQuat()
    UsdGeom.Xformable(sun).AddOrientOp().Set(Gf.Quatf(q))
    sky=UsdLux.DomeLight.Define(b.stage,'/World/Lights/Sky')
    hdr=b.assets_root/'skies/polyhaven/v2/kloofendal_48d_partly_cloudy_puresky/kloofendal_48d_partly_cloudy_puresky_2k.hdr'
    sky.CreateTextureFileAttr(str(hdr));sky.CreateTextureFormatAttr('latlong');sky.CreateIntensityAttr(650.)
    _light_baseline(sky,'sky',6500.)
    for p in (hdr,hdr.with_name('SOURCE.json'),hdr.with_name('LICENSE.txt')):b.dependency(p)
    return apply_lighting_profile(b.stage,FAMILY,'nominal')


def build_laundry_room(stage,assets_root,output_root):
    from .laundry_details import define_detail_materials,dress_laundry_room
    from .laundry_storage import add_storage_details
    b=Builder(stage,Path(assets_root),Path(output_root));b.family=FAMILY
    UsdGeom.Xform.Define(stage,'/World/Robot')
    _materials(b);define_detail_materials(b);_architecture(b)
    shelves=_casework(b);_sink_and_services(b);appliances=_appliances(b)
    dressing=dress_laundry_room(b);storage=add_storage_details(b);lighting=_lighting(b)
    for name in ('NotoSansCJKsc-Regular.otf','LICENSE.txt','SOURCE.json'):
        b.dependency(b.assets_root/'fonts/noto_cjk_sc/v1'/name)
    for dependency in list(b.dependencies):
        parent=Path(dependency).parent
        for name in ('SOURCE.json','LICENSE.txt'):
            if (parent/name).is_file():b.dependency(parent/name)
    return {'schema':'scence.laundry_environment.v1','family':FAMILY,'variant':VARIANT,
            'task_independent':True,'cameras':deepcopy(CAMERAS),
            'robot_placeholder':{'path':'/World/Robot','contains_robot':False},
            'layout':{'room':deepcopy(ROOM_SPEC),'table_center_xy':TABLE['center_xy'],
                      'table_size_m':TABLE['size_m'],'table_top_z':TABLE['top_z'],
                      'reserved_task_volume':deepcopy(RESERVED_TASK_VOLUME),'shelves':shelves},
            'support_colliders':{'table':[TABLE['collider_path']],'floor':['/World/Geometry/room_floor/Collider']},
            'appliances':appliances,'dressing':dressing,'storage_details':storage,'lighting':lighting,
            'objects':[],'geometry':b.records,'dependencies':b.dependencies,
            'ownership':{'environment':'Architecture, wash/dry appliances, static clothing, furniture, accessories and lighting',
                         'task':'Movable cloth, robot and stand, reset, articulated appliance operation, rules and scoring'},
            'scope':'Static furnished laundry room; no task, cloth dynamics, appliance function or robot manipulation qualification'}
