"""Daylit galley kitchen with a cooking run and a separate breakfast bar.

The room owns fixed cabinetry and scenery. Robots and future movable task
objects are composed separately; there is no generic metre-square table.
"""
from pathlib import Path
import math

from pxr import Gf, Sdf, UsdGeom, UsdLux, UsdShade

from .builder import Builder
from .common import _light_baseline, _wall_box, apply_lighting_profile, light
from .kitchen_furnishings import build_furnishings
from bench.paths import load_json

VARIANT = 'kitchen_galley_20261019_v2'
ROOM_SPEC = {'bounds': [-2.1, 3.1, -2.5, 1.45], 'height': 2.85,
             'theme': 'Bright residential galley kitchen with a central robot aisle',
             'window': {'wall': 'left', 'center': -.20, 'width': 2.30,
                        'bottom': 1.06, 'top': 2.56}}
TABLE = {'center_xy': [.45, -.95], 'size_m': [2.8, .8], 'top_z': .90,
         'collider_path': '/World/Geometry/worktop/Collider', 'role': 'breakfast_bar'}
ROBOT_AISLE = {'low': [.30, -.50, .0], 'high': [1.0, .52, 1.80],
               'counter_inner_edge_y_m': .57, 'bar_inner_edge_y_m': -.55,
               'clear_width_m': 1.12, 'recommended_base_position_m': [.65, 0., .78]}
CAMERAS = {
    'room': {'eye': [-1.73, -2.17, 1.98], 'target': [.55, .25, 1.12], 'focal_length_mm': 17.},
    'workcell': {'eye': [2.73, -2.12, 1.97], 'target': [.45, .20, 1.14], 'focal_length_mm': 18.},
    'tabletop': {'eye': [1.90, -.45, 2.20], 'target': [.50, .70, .96], 'focal_length_mm': 22.},
}
HDR_RELATIVE = 'skies/polyhaven/v2/kloofendal_48d_partly_cloudy_puresky/kloofendal_48d_partly_cloudy_puresky_2k.hdr'


def _materials(b):
    plaster = b.assets_root / 'materials/ambientcg/v1/PaintedPlaster017'
    concrete = b.assets_root / 'materials/ambientcg/v1/Concrete048'
    wood = b.assets_root / 'materials/polyhaven/v1/kitchen_wood'
    for name, color, rough in [
        ('plaster', (.83, .825, .78), .87), ('ceiling', (.91, .91, .88), .90),
        ('window_frame', (.88, .875, .83), .35), ('grout', (.38, .385, .37), .9),
        ('dark_metal', (.045, .055, .052), .37), ('brass', (.52, .35, .15), .28),
        ('diffuser', (.92, .88, .77), .42), ('linen', (.68, .64, .54), .95),
        ('courtyard', (.35, .38, .31), .95), ('exterior_wall', (.70, .68, .61), .89),
    ]:
        b.material('kitchen_arch_' + name, color, rough,
                   metallic=.8 if name in ('dark_metal', 'brass') else 0.,
                   normal=str(plaster / 'PaintedPlaster017_1K-JPG_NormalGL.jpg') if name == 'plaster' else None)
    for i, color in enumerate(((.64,.635,.59), (.67,.659,.615), (.65,.646,.607), (.66,.65,.60))):
        b.material('kitchen_arch_tile_' + str(i), color, .64,
                   normal=str(concrete / 'Concrete048_1K-JPG_NormalGL.jpg'))
    b.material('kitchen_arch_oak', (.60,.48,.32), .55,
               texture=str(wood / 'kitchen_wood_diff_1k.jpg'),
               normal=str(wood / 'kitchen_wood_nor_gl_1k.jpg'),
               roughness_texture=str(wood / 'kitchen_wood_rough_1k.jpg'))
    sh = UsdShade.Shader(b.stage.GetPrimAtPath('/World/Looks/kitchen_arch_diffuser/Surface'))
    sh.CreateInput('emissiveColor', Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(2.6, 2.3, 1.85))
    for folder in (plaster, concrete, wood):
        for filename in ('SOURCE.json', 'LICENSE.txt'):
            b.dependency(folder / filename)


def _architecture(b):
    s = ROOM_SPEC; xmin,xmax,ymin,ymax = s['bounds']; h=s['height']
    cx,cy=(xmin+xmax)/2,(ymin+ymax)/2
    b.box('room_floor', (cx,cy,-.09), (xmax-xmin,ymax-ymin,.18), 'kitchen_arch_grout', 0)
    # Individual porcelain slabs have a visible 2.5 mm joint and rounded edges.
    ix=0; x=xmin
    while x < xmax-.001:
        iy=0; y=ymin; width=min(.60,xmax-x)
        while y < ymax-.001:
            depth=min(.60,ymax-y)
            b.box(f'porcelain_floor_{ix}_{iy}', (x+width/2,y+depth/2,-.005),
                  (width-.0025,depth-.0025,.010), 'kitchen_arch_tile_'+str((ix+3*iy)%4), .0006, collision=False)
            y+=.60; iy+=1
        x+=.60; ix+=1
    b.box('room_ceiling',(cx,cy,h+.07),(xmax-xmin,ymax-ymin,.14),'kitchen_arch_ceiling',0)
    for side,a,z in [('back',xmin,xmax),('front',xmin,xmax),('right',ymin,ymax)]:
        _wall_box(b,'wall_'+side,side,(a+z)/2,h/2,z-a,h,s,'kitchen_arch_plaster')
    w=s['window']; a=w['center']-w['width']/2; z=w['center']+w['width']/2
    for name,u,width,lo,hi in [('front',(ymin+a)/2,a-ymin,0,h),
                              ('rear',(z+ymax)/2,ymax-z,0,h),
                              ('sill',(a+z)/2,z-a,0,w['bottom']),
                              ('header',(a+z)/2,z-a,w['top'],h)]:
        _wall_box(b,'wall_window_'+name,'left',u,(lo+hi)/2,width,hi-lo,s,'kitchen_arch_plaster')
    for side,a0,z0 in [('back',xmin,xmax),('front',xmin,xmax),('left',ymin,ymax),('right',ymin,ymax)]:
        _wall_box(b,'skirting_'+side,side,(a0+z0)/2,.045,z0-a0,.09,s,
                  'kitchen_arch_window_frame',.016,inset=.016)
    for i,yy in enumerate((a,z)):
        b.box('window_jamb_'+str(i),(xmin+.015,yy,1.81),(.085,.048,1.55),'kitchen_arch_window_frame',.003)
    for i,zz in enumerate((1.06,2.56)):
        b.box('window_rail_'+str(i),(xmin+.015,-.20,zz),(.085,2.35,.05),'kitchen_arch_window_frame',.003)
    b.box('window_mullion',(xmin+.015,-.20,1.81),(.065,.034,1.45),'kitchen_arch_window_frame',.003)
    # Sliding panes are open by one half; sunlight travels through a real opening.
    for i,xx in enumerate((xmin-.008,xmin+.025)):
        b.box('window_glazing_'+str(i),(xx,-.767,1.81),(.006,1.09,1.43),'glass',0,collision=False)
    b.box('window_sill',(xmin+.055,-.20,1.0375),(.22,2.40,.045),'kitchen_arch_window_frame',.003)
    b.tube('window_handle',[(xmin+.065,-.25,1.57),(xmin+.10,-.25,1.57),
                           (xmin+.10,-.25,1.70)],.007,'kitchen_arch_dark_metal')
    # A rolled linen blind, mounting brackets and a beaded pull are actual geometry.
    b.cylinder('blind_roll',(xmin+.075,-.20,2.64),.045,2.36,'kitchen_arch_linen',rotation=(90,0,0),collision=False)
    for yy in (-1.40,1.0):
        b.box('blind_bracket_'+str(yy),(xmin+.055,yy,2.65),(.10,.025,.10),'kitchen_arch_window_frame',.003,collision=False)
    for i in range(30):
        b.sphere('blind_pull_'+str(i),(xmin+.10,-1.40,2.59-i*.012),.0032,'kitchen_arch_window_frame')
    # Door trim, separate leaf, hinges and lever on the right-hand end wall.
    y=-1.65
    b.box('door_reveal',(3.087,y,1.08),(.024,.98,2.16),'kitchen_arch_dark_metal',.001)
    b.box('door_leaf',(3.064,y,1.065),(.035,.90,2.11),'kitchen_arch_oak',.003)
    for yy in (y-.49,y+.49):
        b.box('door_jamb_'+str(yy),(3.05,yy,1.09),(.07,.055,2.18),'kitchen_arch_window_frame',.003)
    b.box('door_header',(3.05,y,2.18),(.07,1.035,.055),'kitchen_arch_window_frame',.002)
    b.cylinder('door_escutcheon',(3.039,y-.32,1.02),.026,.008,'kitchen_arch_dark_metal',rotation=(0,90,0),collision=False)
    b.tube('door_handle',[(3.03,y-.32,1.02),(2.98,y-.32,1.02),(2.98,y-.20,1.02)],.008,'kitchen_arch_dark_metal')
    for zz in (.25,1.05,1.86):
        b.cylinder('door_hinge_'+str(zz),(3.043,y+.44,zz),.007,.07,'steel',collision=False)
    for yy in (-1.01,-.89):
        b.box('switch_'+str(yy),(3.083,yy,1.13),(.025,.085,.085),'kitchen_arch_window_frame',.004)
        b.box('switch_rocker_'+str(yy),(3.066,yy,1.13),(.010,.042,.058),'white',.002,collision=False)
    # Ceiling HVAC with inset louvers; above the doorway, outside the robot area.
    b.box('air_vent_frame',(2.38,-1.88,2.836),(.82,.22,.025),'kitchen_arch_window_frame',.003,collision=False)
    for i in range(6):
        b.box('air_vent_slot_'+str(i),(2.38,-1.96+i*.03,2.82),(.73,.012,.005),'kitchen_arch_dark_metal',.001,collision=False)


def _exterior(b):
    # Shallow private garden and a distant residential facade, not a photo plane.
    b.box('garden_ground',(-7.,-.3,-.18),(9.5,14.,.24),'kitchen_arch_courtyard',0,collision=False)
    for i in range(10):
        b.box('garden_paver_'+str(i),(-3.10,-3.7+i*.72,-.043),(.82,.68,.035),'kitchen_arch_tile_1',.005,collision=False)
    b.box('garden_boundary',(-5.4,-.2,.44),(.18,9.,1.0),'kitchen_arch_exterior_wall',.008,collision=False)
    for i in range(50):
        yy=-4.5+i*.175
        b.box('garden_screen_'+str(i),(-5.27,yy,1.35),(.045,.035,.88),'kitchen_arch_oak',.003,collision=False)
    b.box('neighbor_facade',(-12.,.1,3.5),(.4,17.,7.4),'kitchen_arch_exterior_wall',0,collision=False)
    for row in range(3):
        for col in range(7):
            yy=-7.1+col*2.35; zz=.9+row*2.2
            b.box(f'neighbor_reveal_{row}_{col}',(-11.78,yy,zz),(.045,1.25,1.32),'kitchen_arch_dark_metal',.002,collision=False)
            b.box(f'neighbor_window_{row}_{col}',(-11.748,yy,zz),(.018,1.13,1.19),'glass',0,collision=False)
            b.box(f'neighbor_sill_{row}_{col}',(-11.69,yy,zz-.68),(.23,1.38,.055),'kitchen_arch_window_frame',.002,collision=False)
    source=b.assets_root/'objects/polyhaven/derived/v1/potted_plant_01/potted_plant_01.usdc'
    if source.is_file():
        derivation=source.with_name('DERIVATION.json')
        b.dependency(derivation)
        if derivation.is_file():
            import json
            provenance=load_json(derivation).get('source_provenance')
            if provenance:
                b.dependency(provenance)
                b.dependency(Path(provenance).with_name('LICENSE.txt'))
        for i,yy in enumerate((-2.4,-.3,1.6)):
            b.asset('garden_plant_'+str(i),source,(-3.9,yy,-.06),scale=1.7,ground=True)
    return {'type':'procedural garden and residential facade','sky':'pure-sky HDR',
            'scope':'Static exterior scenery; no task objects'}


def _lighting(b):
    fixtures=[]
    for i,(x,y) in enumerate(((-1.0,.30),(1.85,.30),(-1.0,-1.80),(1.85,-1.80))):
        b.cylinder('ceiling_recess_'+str(i),(x,y,2.838),.075,.018,'kitchen_arch_dark_metal',collision=False)
        b.cylinder('ceiling_diffuser_'+str(i),(x,y,2.824),.057,.008,'kitchen_arch_diffuser',collision=False)
        light(b,'downlight_'+str(i),(x,y,2.818),(x,y,.90),(.108,.108),
              450000.,(1,1,1),shape='disk',role='ceiling',temperature=4400.)
        fixtures.append({'name':'downlight_'+str(i),'type':'recessed disk','kelvin':4400.})
    for i,x in enumerate((-.40,1.24)):
        y=-.95
        b.cylinder('pendant_canopy_'+str(i),(x,y,2.823),.075,.042,'kitchen_arch_dark_metal',collision=False)
        b.cylinder('pendant_cord_'+str(i),(x,y,2.51),.003,.59,'kitchen_arch_dark_metal',collision=False)
        b.lathe('pendant_shade_'+str(i),(x,y,1.98),[(.22,0),(.22,.018),(.10,.19),(.035,.23),
                    (.030,.23),(.094,.184),(.212,.015),(.212,0)],'kitchen_arch_window_frame')
        b.cylinder('pendant_rim_'+str(i),(x,y,1.985),.223,.012,'kitchen_arch_brass',collision=False)
        b.cylinder('pendant_diffuser_'+str(i),(x,y,1.976),.201,.006,'kitchen_arch_diffuser',collision=False)
        light(b,'bar_pendant_'+str(i),(x,y,1.971),(x,y,.90),(.39,.39),
              18000.,(1,1,1),shape='disk',role='task',temperature=3500.)
        fixtures.append({'name':'bar_pendant_'+str(i),'type':'suspended shaded disk','kelvin':3500.})
    # Long under-shelf LED strip in a metal channel over the back splash.
    for i,(x,width) in enumerate(((-.80,1.69),)):
        b.box('cabinet_led_channel_'+str(i),(x,1.185,1.674),(width,.026,.014),'steel',.002,collision=False)
        b.box('cabinet_led_diffuser_'+str(i),(x,1.185,1.665),(width-.02,.018,.004),'kitchen_arch_diffuser',.001,collision=False)
        light(b,'cabinet_led_'+str(i),(x,1.185,1.661),(x,.95,.90),(width-.03,.018),
              45000.,(1,1,1),role='accent',temperature=4000.)
    sun=UsdLux.DistantLight.Define(b.stage,'/World/Lights/Sun')
    sun.CreateIntensityAttr(2200.); sun.CreateAngleAttr(3.0); _light_baseline(sun,'sun',5600.)
    q=Gf.Rotation(Gf.Vec3d(0,0,-1),Gf.Vec3d(1.,.18,-.55)).GetQuat()
    UsdGeom.Xformable(sun).AddOrientOp().Set(Gf.Quatf(q))
    sky=UsdLux.DomeLight.Define(b.stage,'/World/Lights/Sky')
    hdr=b.assets_root/HDR_RELATIVE
    sky.CreateTextureFileAttr(str(hdr)); sky.CreateTextureFormatAttr('latlong'); sky.CreateIntensityAttr(1800.)
    _light_baseline(sky,'sky',6500.); b.dependency(hdr); b.dependency(hdr.with_name('SOURCE.json'))
    return fixtures


def build_kitchen(stage, assets_root, output_root, *, lighting_profile='nominal'):
    b=Builder(stage,Path(assets_root),Path(output_root)); b.family='kitchen'
    UsdGeom.Xform.Define(stage,'/World/Robot')
    _materials(b); _architecture(b)
    furnishings=build_furnishings(b)
    exterior=_exterior(b); fixtures=_lighting(b)
    from .kitchen_dressing import build_dressing
    dressing=build_dressing(b)
    from .kitchen_accessories import build_accessories
    accessories=build_accessories(b)
    return {'schema':'scence.kitchen_environment.v1','family':'kitchen','variant':VARIANT,
            'task_independent':True,'cameras':CAMERAS,
            'robot_placeholder':{'path':'/World/Robot','contains_robot':False},
            'layout':{'table_center_xy':TABLE['center_xy'],'table_size_m':TABLE['size_m'],
                      'table_top_z':TABLE['top_z'],'room':ROOM_SPEC,'robot_aisle':ROBOT_AISLE,
                      'work_surfaces':{name: furnishings['layout'][name]
                                       for name in ('bar','back_counter','cooktop','sink')}},
            'support_colliders':{'table':[TABLE['collider_path']],
                                 'floor':['/World/Geometry/room_floor/Collider'],
                                 **furnishings.get('support_colliders',{})},
            'lighting':apply_lighting_profile(stage,'kitchen',lighting_profile),
            'lighting_fixtures':fixtures,'furnishings':furnishings,'dressing':dressing,
            'accessories':accessories,'exterior':exterior,
            'objects':[],'geometry':b.records,'dependencies':b.dependencies,
            'ownership':{'environment':'Architecture, cabinetry, appliances, static kitchen dressing and lights',
                         'task':'Robot, movable objects, resets, rules, scoring and PhysicsScene'},
            'scope':'Static kitchen environment and native rendered appearance; task slots remain planned and physics is unqualified'}
