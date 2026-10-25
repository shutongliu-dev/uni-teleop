"""Task-independent Ming scholar's game room, authored as real USD.

Architecture, garden, furniture, empty table and lights belong to this scene.
Boards, pieces, robot supports, robots and PhysicsScene belong to each task.
"""
from __future__ import annotations

import math
from pathlib import Path
from pxr import Gf, Sdf, UsdGeom, UsdLux, UsdShade
from .builder import Builder
from .common import _light_baseline, _wall_box, apply_lighting_profile, light
from .heritage_details import (bamboo, hanging_scroll, hexagonal_lantern,
                               round_moulding, timber_texture, wall_with_round_opening)
from .heritage_furnishings import build_furnishings

VARIANT = 'card_room_heritage_20261017_v3'
ROOM_SPEC = {
    'bounds': [-1.80, 3.10, -2.20, 2.50], 'height': 2.85,
    'theme': 'Ming scholar game room: walnut, lime plaster, ink scrolls and bamboo courtyard',
    'window': {'wall': 'back', 'shape': 'circular', 'center': .65, 'center_z': 1.83,
               'radius': .82, 'width': 1.64, 'bottom': 1.01, 'top': 2.65,
               'open_to_modeled_courtyard': True},
    'courtyard': {'bounds': [-1.80, 3.10, 2.64, 4.75], 'static_scenery_only': True},
}
TABLE = {'center_xy': [.65, 0.], 'size_m': [1., 1.], 'top_z': .78,
         'thickness_m': .04, 'collider_path': '/World/Geometry/worktop/Collider'}
RESERVED_TASK_VOLUME = {'low': [-.05, -1.10, .78], 'high': [1.35, 1.10, 1.95]}
CAMERAS = {
    'room': {'eye': [2.72, -2.00, 2.02], 'target': [.40, .56, 1.20], 'focal_length_mm': 22.},
    'workcell': {'eye': [2.35, -1.55, 1.95], 'target': [.65, .13, 1.08], 'focal_length_mm': 27.},
    'tabletop': {'eye': [.65, -.85, 1.98], 'target': [.65, 0., .80], 'focal_length_mm': 33.},
}


def _materials(b):
    for filename in ('NotoSansCJKsc-Regular.otf', 'SOURCE.json', 'LICENSE.txt'):
        b.dependency(b.assets_root / 'fonts/noto_cjk_sc/v1' / filename)
    textures = b.assets_root / 'materials/ambientcg/v1'
    plaster = str(textures / 'PaintedPlaster017/PaintedPlaster017_1K-JPG_NormalGL.jpg')
    stone = str(textures / 'Concrete048/Concrete048_1K-JPG_NormalGL.jpg')
    timber = str(timber_texture(b.generated))
    b.material('heritage_plaster', (.77, .755, .69), .94, normal=plaster)
    b.material('heritage_courtyard_plaster', (.48, .50, .45), .95, normal=plaster)
    b.material('heritage_ceiling', (.72, .715, .66), .94)
    b.material('heritage_panel', (.52, .51, .45), .88, normal=plaster)
    b.material('heritage_walnut', (.105, .064, .037), .48, texture=timber, texture_tint=(.46, .40, .32))
    b.material('heritage_table', (.24, .17, .105), .55, texture=timber, texture_tint=(.88, .82, .70))
    b.material('heritage_floor', (.285, .288, .267), .86, normal=stone)
    b.material('heritage_grout', (.18, .19, .175), .96)
    b.material('heritage_stone', (.40, .42, .38), .91, normal=stone)
    b.material('heritage_rock', (.24, .29, .245), .94, normal=stone)
    for name, color, roughness in [
            ('silk', (.50, .51, .405), .94), ('linen', (.61, .59, .48), .96),
            ('paper', (.80, .775, .67), .94), ('ink', (.024, .043, .036), .91),
            ('celadon', (.29, .43, .365), .20), ('tea', (.075, .041, .012), .14),
            ('cinnabar', (.31, .055, .035), .80), ('cord', (.12, .095, .060), .96),
            ('bamboo', (.22, .28, .105), .72), ('bamboo_node', (.32, .355, .16), .80),
            ('leaf', (.08, .16, .075), .74), ('moss', (.14, .19, .095), .98),
            ('lantern_silk', (.79, .725, .55), .87), ('light', (.90, .88, .79), .60)]:
        b.material('heritage_' + name, color, roughness)
    b.material('heritage_brass', (.30, .215, .10), .48, .65)
    for material, emission in [('lantern_silk', (1.15, .78, .40)), ('light', (2.5, 2.35, 2.0))]:
        surface = UsdShade.Shader(b.stage.GetPrimAtPath('/World/Looks/heritage_' + material + '/Surface'))
        surface.CreateInput('emissiveColor', Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*emission))


def _architecture(b):
    s = ROOM_SPEC
    xmin, xmax, ymin, ymax = s['bounds']; h = s['height']
    cx, cy = (xmin + xmax) / 2, (ymin + ymax) / 2
    # The selected support floor retains its original geometry and transform.
    b.box('room_floor', (cx, cy, -.09), (xmax - xmin, ymax - ymin, .18), 'heritage_floor', 0)
    b.box('room_ceiling', (cx, cy, h + .07), (xmax - xmin, ymax - ymin, .14), 'heritage_ceiling', 0)
    for wall, lo, hi in [('left', ymin, ymax), ('right', ymin, ymax), ('front', xmin, xmax)]:
        _wall_box(b, 'wall_' + wall, wall, (lo + hi) / 2, h / 2, hi - lo, h, s, 'heritage_plaster')
    w = s['window']
    wall_with_round_opening(b, center_x=w['center'], center_z=w['center_z'],
                           radius=w['radius'], bounds=s['bounds'], height=h)
    for wall, lo, hi in [('left', ymin, ymax), ('right', ymin, ymax), ('back', xmin, xmax), ('front', xmin, xmax)]:
        _wall_box(b, 'skirting_' + wall, wall, (lo + hi) / 2, .075,
                  hi - lo, .15, s, 'heritage_walnut', .025, inset=.025)
        _wall_box(b, 'cornice_' + wall, wall, (lo + hi) / 2, h - .075,
                  hi - lo, .11, s, 'heritage_walnut', .065, inset=.065)
        _wall_box(b, 'cornice_bead_' + wall, wall, (lo + hi) / 2, h - .147,
                  hi - lo, .018, s, 'heritage_table', .035, inset=.044)
    for wall, lo, hi in [('left', ymin, ymax), ('back', xmin, xmax)]:
        _wall_box(b, 'lower_wall_panel_' + wall, wall, (lo + hi) / 2, .42,
                  hi - lo, .54, s, 'heritage_panel', .014, inset=.021)
        _wall_box(b, 'dado_rail_' + wall, wall, (lo + hi) / 2, .715,
                  hi - lo, .032, s, 'heritage_walnut', .028, inset=.035)
        for i in range(7):
            _wall_box(b, f'panel_stile_{wall}_{i}', wall, lo + (hi - lo) * i / 6,
                      .43, .028, .56, s, 'heritage_walnut', .022, inset=.031)
    for i, xx in enumerate((xmin + .045, xmax - .045)):
        for j, yy in enumerate((ymin + .045, ymax - .045)):
            b.box(f'corner_post_{i}_{j}', (xx, yy, 1.44), (.09, .09, 2.82), 'heritage_walnut', .004, collision=False)
    for i, yy in enumerate((-.99, 1.18)):
        b.box('ceiling_crossbeam_' + str(i), (cx, yy, 2.77),
              (xmax - xmin, .11, .16), 'heritage_walnut', .004, collision=False)
    for i in range(1, 8):
        b.box('floor_joint_long_' + str(i), (xmin + i * (xmax - xmin) / 8, cy, .0001),
              (.0020, ymax - ymin, .0002), 'heritage_grout', 0, collision=False)
    for i in range(8):
        for j in range(1, 6):
            yy = ymin + j * .82 + (i % 2) * .41
            if yy < ymax:
                b.box(f'floor_joint_cross_{i}_{j}', (xmin + (i + .5) * (xmax - xmin) / 8, yy, .0001),
                      ((xmax - xmin) / 8, .0020, .0002), 'heritage_grout', 0, collision=False)
    _moon_window(b); _door(b)


def _moon_window(b):
    w = ROOM_SPEC['window']; x, z, y = w['center'], w['center_z'], 2.475
    round_moulding(b, 'moon_window_outer_moulding', (x, y + .025, z), .810, .061, .13)
    round_moulding(b, 'moon_window_inner_bead', (x, y - .046, z), .783, .026, .023, 'heritage_table')
    round_moulding(b, 'moon_window_reveal', (x, y + .10, z), .797, .028, .10)
    # Sparse stepped fretwork frames a clear central garden view.
    for i, offset in enumerate((-.53, .53)):
        length = math.sqrt(.781 ** 2 - offset ** 2)
        b.box('moon_lattice_u_' + str(i), (x + offset, y - .046, z),
              (.018, .028, length * 2), 'heritage_walnut', .002, collision=False)
        b.box('moon_lattice_v_' + str(i), (x, y - .046, z + offset),
              (length * 2, .028, .018), 'heritage_walnut', .002, collision=False)
    for sx in (-1, 1):
        for sz in (-1, 1):
            points = [(x + sx * .14, y - .046, z + sz * .53), (x + sx * .14, y - .046, z + sz * .39),
                      (x + sx * .39, y - .046, z + sz * .39), (x + sx * .39, y - .046, z + sz * .14),
                      (x + sx * .53, y - .046, z + sz * .14)]
            b.tube(f'moon_lattice_fret_{sx}_{sz}', points, .009, 'heritage_walnut')
    b.box('moon_window_sill', (x, 2.43, .989), (1.85, .28, .045), 'heritage_walnut', .006, collision=False)


def _door(b):
    x, y = 3.065, -1.35
    b.box('door_reveal', (x, y, 1.05), (.025, .96, 2.10), 'heritage_ink', .002)
    b.box('door_leaf', (x - .024, y, 1.035), (.045, .88, 2.07), 'heritage_walnut', .004)
    for i, yy in enumerate((y - .475, y + .475)):
        b.box('door_jamb_' + str(i), (x - .055, yy, 1.07), (.08, .05, 2.14), 'heritage_walnut', .003)
    b.box('door_header', (x - .055, y, 2.16), (.08, 1., .06), 'heritage_walnut', .003)
    for j, zz in enumerate((.50, 1.28, 1.75)):
        b.box('door_inset_' + str(j), (x - .049, y, zz),
              (.009, .66, .52 if j == 0 else .34), 'heritage_panel', .002, collision=False)
        for i in range(5):
            b.box(f'door_lattice_{j}_{i}', (x - .06, y - .28 + i * .14, zz),
                  (.018, .013, .52 if j == 0 else .34), 'heritage_walnut', .001, collision=False)
    b.torus('door_ring_handle', (x - .08, y - .29, .98), .038, .005, 'heritage_brass', rotation=(0, 90, 0))


def _table(b):
    b.box('worktop', (.65, 0., .76), (1., 1., .04), 'heritage_table', .004)
    b.box('table_edge_shadow', (.65, 0., .736), (.984, .984, .008), 'heritage_walnut', .003)
    for i, x in enumerate((.22, 1.08)):
        for j, y in enumerate((-.43, .43)):
            b.box(f'table_leg_{i}_{j}', (x, y, .367), (.055, .055, .706), 'heritage_walnut', .004)
            b.cylinder(f'table_foot_{i}_{j}', (x, y, .012), .032, .024, 'rubber')
            b.box(f'table_brass_foot_{i}_{j}', (x, y, .055), (.057, .057, .05), 'heritage_walnut', .003, collision=False)
            for axis in (0, 1):
                dx, dy = ((.13 * (1 if i == 0 else -1), 0) if axis == 0 else (0, .13 * (1 if j == 0 else -1)))
                b.tube(f'table_spandrel_{i}_{j}_{axis}', [(x, y, .565), (x + dx * .28, y + dy * .28, .585),
                       (x + dx * .45, y + dy * .45, .630), (x + dx, y + dy, .643)], .012, 'heritage_walnut')
    for i, y in enumerate((-.43, .43)):
        b.box('table_apron_long_' + str(i), (.65, y, .684), (.86, .035, .10), 'heritage_walnut', .003)
    for i, x in enumerate((.22, 1.08)):
        b.box('table_apron_cross_' + str(i), (x, 0., .684), (.035, .86, .10), 'heritage_walnut', .003)


def _furniture_asset(b, name, slug, position, *, scale, yaw=0., version='v2'):
    root = b.assets_root / 'objects/polyhaven/derived' / version / slug
    b.dependency(root / 'DERIVATION.json')
    return b.asset(name, root / (slug + '.usdc'), position, scale=scale, rotation=(0., 0., yaw))


def _tea_service(b, z):
    y = 2.12
    b.box('tea_boat_base', (.58, y, z + .012), (.64, .30, .024), 'heritage_walnut', .008)
    for i in range(8):
        b.box('tea_boat_slat_' + str(i), (.30 + .08 * i, y, z + .029),
              (.065, .25, .012), 'heritage_table', .002, collision=False)
    b.lathe('celadon_teapot', (.47, y + .025, z + .035),
            [(0, 0), (.047, 0), (.071, .027), (.076, .063), (.061, .097), (.037, .108),
             (.033, .108), (.033, .092), (.057, .073), (.057, .029), (.035, .008), (0, .008)], 'heritage_celadon')
    b.lathe('teapot_lid', (.47, y + .025, z + .139), [(0, 0), (.039, 0), (.038, .008), (.019, .017), (0, .017)], 'heritage_celadon')
    b.sphere('teapot_finial', (.47, y + .025, z + .163), (.010, .010, .008), 'heritage_celadon')
    b.tube('teapot_handle', [(.422, y + .025, z + .126), (.362, y + .025, z + .136),
           (.350, y + .025, z + .088), (.416, y + .025, z + .064)], .009, 'heritage_celadon')
    b.tube('teapot_spout', [(.526, y + .025, z + .081), (.568, y + .025, z + .100),
           (.590, y + .025, z + .143)], .012, 'heritage_celadon')
    for i, xx in enumerate((.70, .84)):
        b.lathe('tea_cup_' + str(i), (xx, y - .055, z + .036), [(0, 0), (.022, 0), (.031, .014),
                (.036, .043), (.032, .043), (.026, .012), (.018, .006), (0, .006)], 'heritage_celadon')
        b.cylinder('tea_in_cup_' + str(i), (xx, y - .055, z + .065), .030, .001, 'heritage_tea', collision=False)
    b.lathe('tea_caddy', (-.05, y, z), [(0, 0), (.056, 0), (.064, .025), (.062, .142), (.052, .16), (0, .16)], 'heritage_celadon')
    b.cylinder('tea_caddy_lid', (-.05, y, z + .166), .055, .013, 'heritage_walnut', collision=False)
    b.lathe('plum_vase', (1.34, 2.13, z), [(0, 0), (.050, 0), (.081, .067), (.072, .16), (.034, .228),
            (.026, .272), (.022, .272), (.023, .225), (.057, .15), (.060, .066), (.041, .009), (0, .009)], 'heritage_celadon')
    for i in range(3):
        tip = (1.25 + i * .11, 2.13 + (i - 1) * .055, z + .48 + .10 * (i % 2))
        b.tube('plum_branch_' + str(i), [(1.34, 2.13, z + .20), (1.32 + i * .026, 2.13, z + .37), tip], .0027, 'heritage_walnut')
        for k in range(4):
            b.sphere(f'plum_bud_{i}_{k}', (tip[0] + .015 * math.sin(k * 3), tip[1] + .015 * math.cos(k * 2), tip[2] - k * .026),
                     (.007, .006, .007), 'heritage_paper')


def _scholars_corner(b):
    hanging_scroll(b, 'landscape_scroll', .18, 1.77, .65, 1.36)
    hanging_scroll(b, 'bamboo_scroll', 1.05, 1.78, .38, 1.16, kind='bamboo')
    x, y, z = -1.47, .15, .66
    b.box('scholar_console_top', (x, y, z), (.43, 1.20, .035), 'heritage_walnut', .005)
    for i, xx in enumerate((x - .15, x + .15)):
        for j, yy in enumerate((y - .50, y + .50)):
            b.box(f'scholar_console_leg_{i}_{j}', (xx, yy, .326), (.038, .038, .652), 'heritage_walnut', .004)
    for i, yy in enumerate((y - .50, y + .50)):
        b.box('console_stretcher_' + str(i), (x, yy, .18), (.32, .028, .032), 'heritage_walnut', .003, collision=False)
    for i in range(3):
        b.box('stitched_book_' + str(i), (x, y - .32 + i * .005, .692 + i * .029), (.27, .18, .026),
              ['heritage_paper', 'heritage_silk', 'heritage_ink'][i], .001, collision=False)
    b.lathe('brush_pot', (x, y + .29, .68), [(0, 0), (.048, 0), (.047, .12), (.041, .12), (.041, .009), (0, .009)], 'heritage_celadon')
    for i in range(5):
        b.tube('calligraphy_brush_' + str(i), [(x, y + .29, .695),
               (x + .015 * math.cos(i), y + .29 + .019 * math.sin(i), .91 + .018 * (i % 3))], .003, 'heritage_bamboo')
    b.box('scholar_inkstone', (x + .02, y + .02, .693), (.12, .17, .022), 'heritage_ink', .009, collision=False)
    light(b, 'scroll_wash', (-1.42, .20, 2.63), (-1.73, .20, 1.75),
          (.60, .10), 3600., (1., 1., 1.), role='accent', temperature=3600.)


def _perimeter(b):
    _furniture_asset(b, 'tea_chinese_commode', 'chinese_commode', (.65, 2.18, 0.), scale=.42)
    commode = next(r for r in b.records if r['name'] == 'tea_chinese_commode')
    _tea_service(b, commode['bounds_size'][2])
    for name, pos, yaw in [('tea_left_chair', (-.97, 1.22, 0.), 28.), ('tea_right_chair', (2.37, 1.35, 0.), -28.)]:
        _furniture_asset(b, name, 'chinese_armchair', pos, scale=.64, yaw=yaw)
    _scholars_corner(b)
    for i, yy in enumerate((.33, 1.61)):
        b.box('display_upright_' + str(i), (2.91, yy, 1.08), (.045, .045, 2.03), 'heritage_walnut', .003)
    for i, zz in enumerate((.38, .96, 1.54, 2.10)):
        b.box('display_shelf_' + str(i), (2.91, .97, zz), (.29, 1.34, .030), 'heritage_walnut', .004)
    for i, yy in enumerate((.55, .62, .69, .76)):
        b.box('display_book_' + str(i), (2.90, yy, 1.105), (.17, .055, .26),
              ['heritage_silk', 'heritage_paper', 'heritage_ink', 'heritage_cinnabar'][i], .001, collision=False)
    b.lathe('display_celadon_bowl', (2.90, 1.27, .977), [(0, 0), (.048, 0), (.070, .02), (.11, .072),
            (.102, .072), (.057, .019), (.040, .006), (0, .006)], 'heritage_celadon')
    b.lathe('display_celadon_vessel', (2.9, .63, 1.557), [(0, 0), (.052, 0), (.074, .10), (.055, .19),
            (.033, .24), (.028, .24), (.029, .20), (.056, .10), (.040, .009), (0, .009)], 'heritage_celadon')


def _courtyard(b):
    # Real depth, parallax and leaf shadows; there is no background picture.
    b.box('courtyard_ground', (.65, 3.70, .035), (4.9, 2.12, .07), 'heritage_stone', 0, collision=False)
    b.box('courtyard_far_wall', (.65, 4.76, 1.32), (5.10, .12, 2.64), 'heritage_courtyard_plaster', .005, collision=False)
    for i, xx in enumerate((-1.83, 3.13)):
        b.box('courtyard_return_' + str(i), (xx, 3.72, 1.32), (.12, 2.18, 2.64), 'heritage_courtyard_plaster', .003, collision=False)
    b.box('courtyard_wall_cap', (.65, 4.70, 2.67), (5.17, .30, .10), 'heritage_walnut', .005, collision=False)
    for i in range(34):
        b.cylinder('courtyard_roof_tile_' + str(i), (-1.82 + .15 * i, 4.70, 2.742), .040, .32,
                   'heritage_rock', rotation=(90, 0, 0), collision=False)
    b.box('courtyard_planting_bed', (.65, 4.25, .09), (4.4, .73, .11), 'heritage_moss', .025, collision=False)
    for i in range(7):
        b.box('courtyard_step_' + str(i), (-.88 + i * .50, 3.30 + .10 * (i % 2), .092), (.39, .51, .042),
              'heritage_floor', .008, rotation=(0, 0, (i % 3 - 1) * 3), collision=False)
    bamboo(b, 'garden_bamboo_left', (-.46, 4.02, .15), 2.43, seed=31, count=7)
    # The angled interior cameras see a left-shifted slice of the garden.
    # Bring the farther grove into that real window frustum for layered foliage.
    bamboo(b, 'garden_bamboo_right', (.55, 4.31, .15), 2.20, seed=24, count=5)
    for i, (x, y, z, scale) in enumerate(((-.10, 3.96, .30, (.23, .17, .28)),
            (.08, 4.02, .26, (.18, .14, .21)), (.25, 4.07, .36, (.18, .15, .35)), (1.22, 4.14, .23, (.30, .23, .18)))):
        b.sphere('courtyard_rock_' + str(i), (x, y, z), scale, 'heritage_rock')
    b.lathe('courtyard_water_basin', (1.02, 3.95, .13), [(0, 0), (.19, 0), (.255, .14), (.27, .24),
            (.23, .24), (.20, .10), (.15, .045), (0, .045)], 'heritage_stone')
    b.cylinder('courtyard_basin_water', (1.02, 3.95, .335), .23, .002, 'heritage_tea', collision=False)


def _lighting(b):
    for i, y in enumerate((-.36, .52)):
        b.box('task_luminaire_' + str(i), (.65, y, 2.804), (1.02, .31, .055), 'heritage_walnut', .005, collision=False)
        b.box('task_diffuser_' + str(i), (.65, y, 2.772), (.95, .25, .010), 'heritage_lantern_silk', .002, collision=False)
        # Traditional timber-and-silk ceiling cassettes. Their lower face is
        # above the unchanged task emitter, so they do not stripe the worktop.
        for j, xx in enumerate((.65 - .158, .65 + .158)):
            b.box(f'task_lattice_cross_{i}_{j}', (xx, y, 2.765),
                  (.012, .25, .009), 'heritage_walnut', .001, collision=False)
        b.box('task_lattice_long_' + str(i), (.65, y, 2.765),
              (.95, .012, .009), 'heritage_walnut', .001, collision=False)
        light(b, 'card_room_task_' + str(i), (.65, y, 2.758), (.65, y / 2, .78),
              (.92, .24), 19000., (1., 1., 1.), role='task', temperature=4200.)
    for i, (x, y) in enumerate(((-1.07, 1.65), (2.35, 1.67))):
        hexagonal_lantern(b, 'silk_lantern_' + str(i), (x, y, 2.36), radius=.145, height=.35)
        light(b, 'silk_lantern_' + str(i), (x, y, 2.165), (x, y, .70),
              (.26, .26), 10500., (1., 1., 1.), shape='disk', temperature=3100.)
    light(b, 'moon_window_daylight', (.65, 2.64, 1.88), (.20, -.90, .75),
          (1.25, 1.25), 12500., (1., 1., 1.), role='accent', temperature=5700.)
    light(b, 'courtyard_sky_fill', (.65, 3.62, 3.12), (.65, 3.90, .35),
          (3.3, 1.65), 3200., (1., 1., 1.), role='accent', temperature=6000.)
    light(b, 'room_soft_fill', (1.13, -1.60, 2.68), (.45, .80, 1.04),
          (2.6, 1.1), 4400., (1., 1., 1.), role='ceiling', temperature=4400.)
    sun = UsdLux.DistantLight.Define(b.stage, '/World/Lights/Sun')
    sun.CreateIntensityAttr(900.); sun.CreateAngleAttr(2.0)
    _light_baseline(sun, 'sun', 5600.)
    q = Gf.Rotation(Gf.Vec3d(0, 0, -1), Gf.Vec3d(.50, -1., -.78)).GetQuat()
    UsdGeom.Xformable(sun).AddOrientOp().Set(Gf.Quatf(q))
    dome = UsdLux.DomeLight.Define(b.stage, '/World/Lights/Sky')
    hdr = b.assets_root / 'skies/polyhaven/v2/kloofendal_48d_partly_cloudy_puresky/kloofendal_48d_partly_cloudy_puresky_2k.hdr'
    dome.CreateTextureFileAttr(str(hdr)); dome.CreateIntensityAttr(420.)
    _light_baseline(dome, 'sky', 6500.); b.dependency(hdr)


def build_card_room(stage, assets_root, output_root, *, lighting_profile='nominal'):
    """Build a new room version with an empty table and no task/robot actors."""
    b = Builder(stage, Path(assets_root), Path(output_root))
    UsdGeom.Xform.Define(stage, '/World/Robot')
    b.family = 'card_room'
    _materials(b); _architecture(b); _table(b); _perimeter(b); _courtyard(b); _lighting(b)
    furnishings = build_furnishings(b)
    lighting = apply_lighting_profile(stage, 'card_room', lighting_profile)
    return {
        'schema': 'scence.card_room_environment.v1', 'family': 'card_room',
        'variant': VARIANT, 'task_independent': True, 'cameras': CAMERAS,
        'robot_placeholder': {'path': '/World/Robot', 'contains_robot': False},
        'layout': {'table_center_xy': TABLE['center_xy'], 'table_size_m': TABLE['size_m'],
                   'table_top_z': TABLE['top_z'], 'room': ROOM_SPEC, 'reserved_task_volume': RESERVED_TASK_VOLUME},
        'support_colliders': {'table': [TABLE['collider_path']], 'floor': ['/World/Geometry/room_floor/Collider']},
        'lighting': lighting, 'objects': [], 'geometry': b.records, 'dependencies': b.dependencies,
        'furnishings': furnishings,
        'design': {'style': 'Ming scholar interior with modeled bamboo courtyard',
                   'palette': ['warm lime plaster', 'dark walnut', 'grey stone', 'celadon', 'ink'],
                   'features': ['open moon window with stepped wooden fretwork', 'hanging ink scrolls',
                                'silk lanterns', 'celadon tea service', 'scholar console',
                                'asymmetric game library with scrolls and closed Go bowls',
                                'wood and silk floor lantern', 'framed game study',
                                'pine bonsai and folding lattice screen'],
                   'generated_materials': 'original deterministic walnut grain and ink-on-paper studies',
                   'courtyard_is_geometry': True},
        'ownership': {'environment': 'architecture, garden, furniture, empty table and lighting',
                      'task': 'board, pieces, robot supports, articulations and PhysicsScene'},
        'scope': 'Static environment construction only; no native physics or task qualification',
    }
