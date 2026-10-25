"""Detailed sample-preparation laboratory, independent of manipulation tasks."""
from pathlib import Path

from pxr import Gf, Sdf, UsdGeom, UsdLux, UsdShade

from .builder import Builder
from .common import _light_baseline, _wall_box, apply_lighting_profile, light
from .interiors import _Frame, _cabinet

VARIANT = 'laboratory_island_20261020_v3'
ISLAND_YAW_DEG = 180.
NOMINAL_LIGHT_SCALE = .75
ROOM_SPEC = {
    'bounds': [-2.5, 3.5, -2.5, 2.7], 'height': 3.,
    'theme': 'sample preparation laboratory with an island and separate wet/instrument zones',
    'window': {'wall': 'left', 'center': .35, 'width': 2., 'bottom': 1.28, 'top': 2.56},
}
TABLE = {'center_xy': [.65, 0.], 'size_m': [2.4, 1.25], 'thickness_m': .045, 'top_z': .90,
         'collider_path': '/World/Geometry/worktop/Collider',
         'yaw_degrees': ISLAND_YAW_DEG,
         'clear_task_rectangle_xy_m': [-.48, 1.78, -.30, .56]}
CAMERAS = {
    'room': {'eye': [3.12, -2.20, 2.38], 'target': [.30, .65, 1.15], 'focal_length_mm': 19.},
    'workcell': {'eye': [2.70, -1.85, 1.95], 'target': [.55, .55, 1.00], 'focal_length_mm': 24.},
    'tabletop': {'eye': [1.95, -1.52, 1.80], 'target': [.65, .12, .95], 'focal_length_mm': 27.},
    'reverse': {'eye': [-2.12, -1.95, 2.18], 'target': [.65, .80, 1.18], 'focal_length_mm': 20.},
    'safety': {'eye': [2.90, 1.65, 2.22], 'target': [-.40, -1.65, 1.13], 'focal_length_mm': 21.},
}


def _materials(b):
    normal = b.assets_root / 'materials/ambientcg/v1/PaintedPlaster017/PaintedPlaster017_1K-JPG_NormalGL.jpg'
    for name, color, rough, metal in [
        ('wall', (.79, .84, .83), .83, 0), ('ceiling', (.87, .90, .89), .82, 0),
        ('floor', (.46, .58, .59), .55, 0), ('seam', (.33, .45, .46), .67, 0),
        ('white', (.81, .87, .86), .34, 0), ('teal', (.095, .29, .32), .36, .08),
        ('pale', (.50, .68, .68), .40, 0), ('epoxy', (.055, .079, .083), .28, 0),
        ('steel', (.55, .60, .61), .26, .85), ('rubber', (.035, .045, .047), .77, 0),
        ('blue', (.07, .28, .53), .40, 0), ('yellow', (.84, .57, .08), .5, 0),
        ('diffuser', (.90, .97, 1.), .35, 0),
    ]:
        b.material('lab_' + name, color, rough, metal,
                   normal=str(normal) if name == 'wall' else None)
    shader = UsdShade.Shader(b.stage.GetPrimAtPath('/World/Looks/lab_diffuser/Surface'))
    shader.CreateInput('emissiveColor', Sdf.ValueTypeNames.Color3f).Set(
        Gf.Vec3f(3., 3.4, 3.5) * NOMINAL_LIGHT_SCALE)
    b.material('lab_sash', (.75, .89, .91), .12, opacity=.15)
    for filename in ('SOURCE.json', 'LICENSE.txt'):
        b.dependency(normal.parent / filename)
    for filename in ('NotoSansCJKsc-Regular.otf', 'LICENSE.txt', 'SOURCE.json'):
        b.dependency(b.assets_root / 'fonts/noto_cjk_sc/v1' / filename)


def _architecture(b):
    s = ROOM_SPEC
    xmin, xmax, ymin, ymax = s['bounds']
    cx, cy, height = (xmin+xmax)/2, (ymin+ymax)/2, s['height']
    b.box('room_floor', (cx, cy, -.075), (6., 5.2, .15), 'lab_floor', 0)
    # Welded sheet-vinyl seams and perimeter coving, not domestic floor tiles.
    for i, x in enumerate((-1., .5, 2.)):
        b.box('vinyl_weld_'+str(i), (x, cy, .0002), (.002, 5.2, .0004), 'lab_seam', 0, collision=False)
    b.box('room_ceiling', (cx, cy, height+.07), (6., 5.2, .14), 'lab_ceiling', 0)
    for wall, lo, hi in [('front', xmin, xmax), ('back', xmin, xmax)]:
        _wall_box(b, 'wall_'+wall, wall, (lo+hi)/2, height/2, hi-lo, height, s, 'lab_wall')
    for name, lo, hi, z0, z1 in [
        ('before', ymin, -2.25, 0, height), ('after', -1.25, ymax, 0, height),
        ('header', -2.25, -1.25, 2.22, height),
    ]:
        _wall_box(b,'wall_right_'+name,'right',(lo+hi)/2,(z0+z1)/2,hi-lo,z1-z0,s,'lab_wall')
    w = s['window']; start, end = w['center']-1., w['center']+1.
    for name, lo, hi, z0, z1 in [
        ('before', ymin, start, 0, height), ('after', end, ymax, 0, height),
        ('below', start, end, 0, w['bottom']), ('above', start, end, w['top'], height),
    ]:
        _wall_box(b, 'wall_left_'+name, 'left', (lo+hi)/2, (z0+z1)/2, hi-lo, z1-z0, s, 'lab_wall')
    for wall, lo, hi in [('left', ymin, ymax), ('right', ymin, ymax), ('front', xmin, xmax), ('back', xmin, xmax)]:
        _wall_box(b, 'coved_skirting_'+wall, wall, (lo+hi)/2, .055, hi-lo, .11, s, 'lab_pale', .024, .024)
        _wall_box(b, 'wall_bumper_'+wall, wall, (lo+hi)/2, .72, hi-lo, .065, s, 'lab_pale', .018, .018)
    for i, y in enumerate((start, .35, end)):
        b.box('lab_window_jamb_'+str(i), (-2.478, y, 1.92), (.085, .045, 1.35), 'lab_white', .003)
    for i, z in enumerate((1.28, 2.56)):
        b.box('lab_window_rail_'+str(i), (-2.478, .35, z), (.085, 2.08, .055), 'lab_white', .003)
    for i, y in enumerate((-.15, .85)):
        b.box('lab_window_glass_'+str(i), (-2.49, y, 1.92), (.006, .952, 1.23), 'glass', 0, collision=False)
        b.box('lab_window_privacy_'+str(i), (-2.475, y, 1.48), (.002, .952, .33), 'lab_sash', 0, collision=False)
        b.tube('lab_window_handle_'+str(i), [(-2.42,y+.32,1.82),(-2.39,y+.32,1.82),(-2.39,y+.32,1.95)], .007, 'steel')
    b.box('lab_window_sill', (-2.43, .35, 1.252), (.23, 2.12, .045), 'lab_white', .004)
    # A restrained exterior avoids an empty sky immediately outside the glazing.
    b.box('courtyard_ground', (-6., .2, -.18), (7., 11., .20), 'lab_pale', 0, collision=False)
    b.box('courtyard_facade', (-8.5, .2, 2.), (.25, 12., 4.4), 'lab_wall', 0, collision=False)
    for i in range(7):
        b.box('courtyard_window_'+str(i), (-8.36, -4.8+i*1.65, 2.25), (.025, .95, 1.35), 'lab_blue', .005, collision=False)
    # Flush laboratory door with a vision panel, kick plate and closer.
    y = -1.75
    for name, center_y, center_z, width, height in [
        ('left',y-.3025,1.09,.325,2.16),('right',y+.3025,1.09,.325,2.16),
        ('lower',y,.645,.28,1.27),('upper',y,2.025,.28,.29),
    ]:
        b.box('lab_door_leaf_'+name,(3.45,center_y,center_z),(.045,width,height),'lab_white',.003)
    for i, yy in enumerate((y-.50,y+.50)):
        b.box('lab_door_jamb_'+str(i), (3.425,yy,1.12), (.075,.055,2.24), 'lab_teal', .003)
    b.box('lab_door_header', (3.425,y,2.235), (.075,1.055,.055), 'lab_teal', .003)
    for i, yy in enumerate((y-.15,y+.15)):
        b.box('lab_door_vision_stile_'+str(i),(3.423,yy,1.58),(.016,.028,.63),'lab_steel',.003,collision=False)
    for i, zz in enumerate((1.28,1.88)):
        b.box('lab_door_vision_rail_'+str(i),(3.423,y,zz),(.016,.31,.03),'lab_steel',.003,collision=False)
    b.box('lab_door_vision', (3.413,y,1.58), (.005,.27,.59), 'glass', .004, collision=False)
    b.box('corridor_floor',(4.5,y,-.06),(2.,2.2,.12),'lab_floor',0,collision=False)
    b.box('corridor_far_wall',(5.50,y,1.5),(.14,2.2,3.),'lab_wall',0,collision=False)
    b.box('lab_door_kickplate', (3.421,y,.18), (.008,.85,.27), 'lab_steel', .002, collision=False)
    b.tube('lab_door_lever', [(3.42,y-.30,1.),(3.36,y-.30,1.),(3.36,y-.16,1.)], .009, 'chrome')
    b.box('lab_door_closer', (3.40,y+.20,2.08), (.07,.20,.055), 'lab_steel', .005, collision=False)
    b.tube('lab_door_closer_arm', [(3.39,y+.20,2.09),(3.29,y+.10,2.11),(3.43,y-.03,2.18)], .007, 'steel')
    for i, z in enumerate((.25,1.05,1.95)):
        b.cylinder('lab_door_hinge_'+str(i), (3.419,y+.45,z), .009,.09,'steel',collision=False)
    # Suspended ceiling tiles, service grid, and recessed luminaires.
    for i in range(1,10):
        b.box('lab_ceiling_grid_x_'+str(i), (-2.5+i*.6,.1,2.993), (.014,5.2,.014), 'lab_white', .001, collision=False)
    for i in range(1,9):
        b.box('lab_ceiling_grid_y_'+str(i), (.5,-2.5+i*.6,2.993), (6.,.014,.014), 'lab_white', .001, collision=False)


def _island(b):
    # Turn the complete assembly about its own centre, including nested
    # cabinetry frames, labels, sockets and the service shelf.
    b = _Frame(b, (2 * TABLE['center_xy'][0], 0, 0), ISLAND_YAW_DEG)
    b.box('worktop', (.65,0,.8775), (2.4,1.25,.045), 'lab_epoxy', .006)
    b.box('lab_island_shadow', (.65,0,.848), (2.35,1.20,.014), 'lab_rubber', .002)
    # Kitchen-island construction, adapted to washable laboratory casework.
    for side, yaw in [('front',0), ('rear',180)]:
        f = _Frame(b, (.65, -.291 if side=='front' else .291, 0), yaw)
        for i, x in enumerate((-.77,0,.77)):
            _cabinet(f, 'island_'+side+'_'+str(i), x, .755, .838, .55, 'lab_teal', drawers=i!=1)
    for side, x in [('left',-.514),('right',1.814)]:
        b.box('island_end_'+side, (x,0,.48), (.025,1.16,.72), 'lab_pale', .004)
        b.box('island_end_inlay_'+side, (x+(-.014 if side=='left' else .014),0,.48),
              (.004,.98,.55),'lab_white',.002,collision=False)
    b.label('island_front_identity', 'SAMPLE PREPARATION  /  02', (.65,-.627,.878), (.69,.026),
            rotation=(90,0,0), font_size=30)
    # Local +Y services rotate onto world -Y; the clear working side is +Y.
    for i, x in enumerate((-.39,1.69)):
        b.box('island_service_post_'+str(i), (x,.50,1.16), (.035,.035,.52), 'lab_steel', .003)
        b.box('island_service_foot_'+str(i), (x,.50,.909), (.11,.13,.018), 'lab_steel', .003)
    b.box('island_service_spine', (.65,.51,1.04), (2.10,.085,.13), 'lab_white', .005)
    b.box('island_service_shelf', (.65,.50,1.34), (2.18,.24,.027), 'lab_epoxy', .004)
    for i, x in enumerate((-.10,.37,.93,1.40)):
        b.box('island_socket_'+str(i),(x,.464,1.04),(.12,.012,.074),'lab_white',.003,collision=False)
        for j, dx in enumerate((-.025,.025)):
            b.box(f'island_socket_slot_{i}_{j}',(x+dx,.457,1.04),(.006,.002,.019),'lab_rubber',.001,collision=False)
    for i,x in enumerate((-.32,1.62)):
        b.cylinder('island_service_valve_'+str(i),(x,.52,1.14),.017,.033,'lab_blue',collision=False)
        b.tube('island_service_outlet_'+str(i),[(x,.52,1.12),(x,.45,1.12),(x,.45,1.10)],.006,'chrome')
    b.label('island_service_mark', 'POWER  /  DATA', (.65,.464,1.04),(.28,.035),rotation=(90,0,0),font_size=30)
    return {'type':'double-sided laboratory island with epoxy top, 12 drawers and 4 doors',
            'work_surface': TABLE, 'yaw_degrees': ISLAND_YAW_DEG,
            'service_spine_y_m': -.51,
            'task_surface': 'Clear +Y working rectangle; fixed service shelf on the -Y side'}


def _rear_casework(b):
    f = _Frame(b,(0,2.30,0))
    for i,x in enumerate((-2.075,-1.425,-.775,-.125,.525,1.175,1.825)):
        _cabinet(f,'rear_lab_cabinet_'+str(i),x,.635,.855,.62,'lab_white',drawers=i in (2,4,6))
    f.box('rear_lab_counter',(-.125,0,.8775),(4.55,.70,.045),'lab_epoxy',.006)
    f.box('rear_lab_backsplash',(-.125,.329,1.03),(4.55,.027,.26),'lab_pale',.003)
    # Deep fume cupboard with visible baffles, sash tracks and ductwork.
    h = _Frame(b,(-1.65,2.30,0))
    h.box('lab_hood_back',(0,.31,1.60),(1.4,.035,1.40),'lab_white',.003)
    for i,x in enumerate((-.682,.682)):
        h.box('lab_hood_side_'+str(i),(x,0,1.60),(.036,.66,1.40),'lab_white',.003)
        h.box('lab_hood_sash_track_'+str(i),(x,-.343,1.60),(.035,.035,1.12),'lab_steel',.002)
    h.box('lab_hood_header',(0,0,2.24),(1.42,.70,.19),'lab_white',.006)
    h.box('lab_hood_sash_glass',(0,-.35,1.90),(1.29,.007,.47),'lab_sash',0,collision=False)
    for i,z in enumerate((1.655,2.145)):
        h.box('lab_hood_sash_rail_'+str(i),(0,-.35,z),(1.32,.035,.027),'lab_steel',.003)
    h.tube('lab_hood_pull',[(-.19,-.37,1.65),(-.19,-.40,1.65),(.19,-.40,1.65),(.19,-.37,1.65)],.008,'lab_steel')
    for i,z in enumerate((1.01,1.07,1.32,1.38,1.89,1.95)):
        h.box('lab_hood_baffle_'+str(i),(0,.286,z),(1.10,.004,.018),'lab_rubber',.001,collision=False)
    h.box('lab_hood_air_monitor',(.47,-.359,2.24),(.20,.012,.10),'lab_epoxy',.003,collision=False)
    h.label('lab_hood_readout','0.50 m/s',(.47,-.367,2.25),(.17,.035),rotation=(90,0,0),font_size=32)
    h.label('lab_hood_name','FUME CUPBOARD  01',(-.19,-.357,2.24),(.81,.066),rotation=(90,0,0),font_size=44)
    h.cylinder('lab_hood_exhaust',(0,.10,2.65),.145,.62,'lab_steel',collision=False)
    for i,z in enumerate((2.37,2.85)):
        h.cylinder('lab_hood_duct_collar_'+str(i),(0,.10,z),.155,.038,'lab_white',collision=False)
    h.box('lab_hood_light',(0,0,2.137),(1.1,.18,.014),'lab_diffuser',.002,collision=False)
    light(b,'lab_hood_task',(-1.65,2.30,2.12),(-1.65,2.30,.90),(1.0,.15),10000.*NOMINAL_LIGHT_SCALE,(1,1,1),role='task',temperature=5000.)
    h.box('lab_hood_spill_tray',(0,.02,.916),(1.04,.43,.032),'lab_steel',.003)
    # Reagent vessels are secondary scenery inside the extracted area.
    for i,x in enumerate((-.34,-.10,.19,.38)):
        h.lathe('lab_hood_bottle_'+str(i),(x,.10,.932),
                [(0,0),(.035,0),(.038,.015),(.038,.12),(.019,.15),(.019,.18),(0,.18)],'glass')
        h.cylinder('lab_hood_bottle_cap_'+str(i),(x,.10,1.12),.021,.021,'lab_blue',collision=False)
        h.label('lab_hood_bottle_label_'+str(i),'BUFFER\n'+str(i+1),(x,.061,1.01),(.045,.048),rotation=(90,0,0),font_size=28)
    for i,x in enumerate((-.60,.25,1.10,1.85)):
        b.box('instrument_socket_'+str(i),(x,2.637,1.12),(.15,.025,.085),'lab_white',.003,collision=False)
        for j,dx in enumerate((-.032,.032)):
            b.box(f'instrument_socket_slot_{i}_{j}',(x+dx,2.622,1.12),(.006,.002,.017),'lab_rubber',.001,collision=False)
    # Shallow open shelves above the instruments, leaving operating clearance.
    for i,z in enumerate((1.81,2.25)):
        b.box('lab_wall_shelf_'+str(i),(.65,2.48,z),(2.75,.36,.026),'lab_white',.003)
        for j,x in enumerate((-.58,.65,1.88)):
            b.tube(f'lab_wall_shelf_bracket_{i}_{j}',[(x,2.66,z+.02),(x,2.66,z-.20),(x,2.33,z-.02)],.009,'lab_steel')
        for j,x in enumerate((-.47,-.08,.31,.70,1.09,1.48,1.87)):
            b.box(f'lab_stock_box_{i}_{j}',(x,2.48,z+.098),(.30,.25,.17),'lab_pale' if j%2 else 'lab_white',.005)
            b.box(f'lab_stock_lid_{i}_{j}',(x,2.48,z+.189),(.31,.26,.018),'lab_teal',.003,collision=False)
            b.label(f'lab_stock_label_{i}_{j}',('TIPS','TUBES','FILTERS','VIALS','PPE','WIPES','SLIDES')[j],
                    (x,2.353,z+.10),(.22,.052),rotation=(90,0,0),font_size=30)
    return {'counter_top_z_m':.90,'counter_bounds_xy_m':[-2.4,2.15,1.95,2.65],
            'clear_task_rectangle_xy_m':[.40,1.04,1.99,2.20],
            'hood':'ventilated cupboard with raised sash, spill tray, baffles and exhaust',
            'wall_storage_boxes':14}


def _wash_and_storage(b):
    f = _Frame(b,(-2.12,.35,0),90)
    _cabinet(f,'wash_base',0,1.58,.855,.64,'lab_white')
    # Four separate top pieces leave an actual sink opening.
    for name,p,size in [
        ('front',(0,-.2925,.8775),(1.64,.115,.045)),
        ('back',(0,.2925,.8775),(1.64,.115,.045)),
        ('left',(-.59,0,.8775),(.46,.47,.045)),
        ('right',(.59,0,.8775),(.46,.47,.045)),
    ]:
        f.box('wash_counter_'+name,p,size,'lab_epoxy',.004)
    f.box('wash_basin_floor',(0,0,.715),(.72,.47,.015),'lab_steel',.009)
    for i,x in enumerate((-.351,.351)):
        f.box('wash_basin_end_'+str(i),(x,0,.80),(.018,.47,.155),'lab_steel',.003)
    for i,y in enumerate((-.226,.226)):
        f.box('wash_basin_wall_'+str(i),(0,y,.80),(.72,.018,.155),'lab_steel',.003)
    f.cylinder('wash_drain',(0,0,.724),.034,.003,'lab_rubber',collision=False)
    f.cylinder('wash_drain_grid',(0,0,.726),.026,.002,'steel',collision=False)
    f.tube('wash_swan_neck',[(.19,.28,.90),(.19,.28,1.20),(.19,.22,1.26),(.19,.01,1.26),(.19,-.03,1.20)],.014,'chrome')
    f.cylinder('wash_tap_mixer',(.31,.28,.935),.026,.07,'chrome',collision=False)
    f.tube('wash_tap_lever',[(.31,.28,.97),(.31,.22,1.01)],.008,'lab_blue')
    f.tube('wash_eyewash_stem',[(-.28,.28,.9),(-.28,.28,1.12),(-.28,.11,1.12)],.011,'steel')
    for i,x in enumerate((-.35,-.21)):
        f.tube('wash_eyewash_branch_'+str(i),[(-.28,.11,1.12),(x,.11,1.12),(x,.11,1.16)],.007,'steel')
        f.sphere('wash_eyewash_cap_'+str(i),(x,.11,1.17),(.018,.018,.010),'green')
    f.box('wash_drying_board',(.53,.27,1.37),(.38,.04,.63),'lab_white',.004)
    for i in range(3):
        for j in range(4):
            x=.40+i*.13; z=1.15+j*.14
            f.tube(f'wash_drying_peg_{i}_{j}',[(x,.24,z),(x,.11,z+.06)],.008,'lab_pale')
    f.box('wash_drying_drip_tray',(.53,.12,1.035),(.42,.31,.025),'lab_steel',.003)
    f.lathe('wash_soap',(-.61,-.01,.90),[(0,0),(.035,0),(.035,.12),(.018,.145),(0,.145)],'lab_white')
    f.cylinder('wash_soap_pump',(-.61,-.01,1.06),.009,.03,'lab_teal',collision=False)
    f.tube('wash_soap_nozzle',[(-.61,-.01,1.075),(-.61,-.07,1.075)],.006,'lab_teal')
    # Glass-front clean stock cupboard in the back-right corner.
    s = _Frame(b,(2.80,2.31,0))
    s.box('stock_cabinet_toe',(0,.035,.06),(1.00,.56,.12),'lab_rubber',.002)
    for i,x in enumerate((-.52,.52)):
        s.box('stock_cabinet_side_'+str(i),(x,0,1.22),(.028,.66,2.20),'lab_white',.003)
    s.box('stock_cabinet_back',(0,.32,1.22),(1.07,.025,2.20),'lab_white',.003)
    for i,z in enumerate((.13,.76,1.20,1.64,2.08,2.32)):
        s.box('stock_cabinet_shelf_'+str(i),(0,0,z),(1.05,.65,.025),'lab_white',.003)
    for i,x in enumerate((-.26,.26)):
        s.box('stock_cabinet_lower_door_'+str(i),(x,-.345,.44),(.51,.025,.60),'lab_teal',.004)
        s.box('stock_cabinet_glass_'+str(i),(x,-.345,1.54),(.47,.006,1.50),'lab_sash',0,collision=False)
        for j,dx in enumerate((-.249,.249)):
            s.box(f'stock_cabinet_stile_{i}_{j}',(x+dx,-.35,1.54),(.022,.027,1.57),'lab_white',.002)
        for j,z in enumerate((.767,2.313)):
            s.box(f'stock_cabinet_rail_{i}_{j}',(x,-.35,z),(.51,.027,.026),'lab_white',.002)
        s.box('stock_cabinet_handle_'+str(i),((-1 if i==0 else 1)*.045,-.378,1.40),(.011,.027,.20),'chrome',.003,collision=False)
    for row,z in enumerate((.773,1.213,1.653,2.093)):
        for col,x in enumerate((-.33,0,.33)):
            if row in (0,2):
                s.lathe(f'stock_bottle_{row}_{col}',(x,.02,z),
                        [(0,0),(.058,0),(.058,.19),(.03,.23),(.03,.27),(0,.27)],'glass')
                s.cylinder(f'stock_cap_{row}_{col}',(x,.02,z+.28),.032,.025,'lab_blue',collision=False)
                s.label(f'stock_tag_{row}_{col}','CLEAN\nGLASS',(x,-.04,z+.13),(.075,.07),rotation=(90,0,0),font_size=30)
            else:
                s.box(f'stock_pack_{row}_{col}',(x,0,z+.065),(.27,.38,.13),'lab_pale',.003)
                s.label(f'stock_pack_tag_{row}_{col}','STERILE',(x,-.192,z+.065),(.20,.045),rotation=(90,0,0),font_size=30)
    s.box('stock_cabinet_header',(0,-.34,2.345),(1.07,.035,.09),'lab_teal',.003)
    s.label('stock_cabinet_title','CLEAN GLASSWARE',(0,-.359,2.345),(.86,.05),rotation=(90,0,0),font_size=36)
    return {'wet_zone':'recessed sink, mixer, eyewash, drain and drying rack',
            'storage':'four stocked levels behind framed glazing; closed lower cabinets'}


def _lighting(b):
    for i,(x,y) in enumerate(((-1.3,-1.35),(.5,-1.35),(2.3,-1.35),(-1.3,1.15),(.5,1.15),(2.3,1.15))):
        b.box('lab_panel_frame_'+str(i),(x,y,2.974),(1.17,.57,.034),'lab_white',.003,collision=False)
        b.box('lab_panel_diffuser_'+str(i),(x,y,2.953),(1.10,.50,.009),'lab_diffuser',.002,collision=False)
        light(b,'lab_ceiling_'+str(i),(x,y,2.944),(x,y,.90),(1.08,.48),12500.*NOMINAL_LIGHT_SCALE,(1,1,1),role='ceiling',temperature=4800.)
    sun = UsdLux.DistantLight.Define(b.stage,'/World/Lights/Sun')
    sun.CreateIntensityAttr(1600.*NOMINAL_LIGHT_SCALE); sun.CreateAngleAttr(2.5); _light_baseline(sun,'sun',5600.)
    q=Gf.Rotation(Gf.Vec3d(0,0,-1),Gf.Vec3d(1.,.25,-.48)).GetQuat()
    UsdGeom.Xformable(sun).AddOrientOp().Set(Gf.Quatf(q))
    sky=UsdLux.DomeLight.Define(b.stage,'/World/Lights/Sky')
    hdr=b.assets_root/'skies/polyhaven/v2/kloofendal_48d_partly_cloudy_puresky/kloofendal_48d_partly_cloudy_puresky_2k.hdr'
    sky.CreateTextureFileAttr(str(hdr)); sky.CreateTextureFormatAttr('latlong'); sky.CreateIntensityAttr(850.*NOMINAL_LIGHT_SCALE)
    _light_baseline(sky,'sky',6500.); b.dependency(hdr)
    for filename in ('SOURCE.json','LICENSE.txt'): b.dependency(hdr.with_name(filename))


def build_laboratory(stage, assets_root, output_root, *, lighting_profile='nominal'):
    b=Builder(stage,Path(assets_root),Path(output_root)); b.family='laboratory'
    UsdGeom.Xform.Define(stage,'/World/Robot')
    _materials(b); _architecture(b)
    island=_island(b); rear=_rear_casework(b); wet_storage=_wash_and_storage(b)
    from .laboratory_equipment import add_laboratory_equipment
    from .laboratory_dressing import add_laboratory_dressing
    equipment=add_laboratory_equipment(b); dressing=add_laboratory_dressing(b)
    _lighting(b)
    return {'schema':'scence.laboratory_environment.v1','family':'laboratory','variant':VARIANT,
            'task_independent':True,'cameras':CAMERAS,
            'robot_placeholder':{'path':'/World/Robot','contains_robot':False},
            'layout':{'table_center_xy':TABLE['center_xy'],'table_size_m':TABLE['size_m'],
                      'table_top_z':TABLE['top_z'],'room':ROOM_SPEC,
                      'clear_task_rectangle_xy_m':TABLE['clear_task_rectangle_xy_m'],
                      'work_surfaces':{'island':TABLE,'rear_counter':rear}},
            'support_colliders':{'table':[TABLE['collider_path']],
                                 'rear_counter':['/World/Geometry/rear_lab_counter/Collider'],
                                 'floor':['/World/Geometry/room_floor/Collider']},
            'lighting':apply_lighting_profile(stage,'laboratory',lighting_profile),
            'lighting_revision':{'baseline':'laboratory_island_20261020_v1',
                                 'nominal_intensity_scale':NOMINAL_LIGHT_SCALE,
                                 'diffuser_emission_scale':NOMINAL_LIGHT_SCALE},
            'island':island,'rear_casework':rear,'wet_storage':wet_storage,
            'equipment':equipment,'dressing':dressing,
            'objects':[],'geometry':b.records,'dependencies':b.dependencies,
            'ownership':{'environment':'Architecture, cabinets, fixed instruments, static dressing and lights',
                         'task':'Robot, movable objects, reset, rules, scoring and PhysicsScene'},
            'scope':'Static laboratory appearance and USD structure; manipulation tasks remain planned'}
