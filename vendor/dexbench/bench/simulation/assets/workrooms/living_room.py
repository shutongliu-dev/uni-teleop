"""A task-independent residential lounge, with physically housed lighting.

All surfaces, coves, curtain folds and luminaires are actual USD geometry.
The fixed metre-square work surface stays empty; tasks own robots and objects.
"""
from __future__ import annotations

import math
from pathlib import Path

from pxr import Gf, Sdf, UsdGeom, UsdLux, UsdShade

from .builder import Builder
from .common import _light_baseline, _wall_box, apply_lighting_profile, light
from .living_furnishings import build_furnishings
from .living_exterior import HDR_RELATIVE, HDR_ROTATE_XYZ_DEG, build_exterior

VARIANT = 'living_room_residential_20261018_v7'
ROOM_SPEC = {
    'bounds': [-2.0, 2.8, -1.8, 2.7], 'height': 2.85,
    'theme': 'Contemporary lived-in lounge with warm oak and indirect light',
    'window': {'wall': 'left', 'center': .55, 'width': 2.85,
               'bottom': .18, 'top': 2.53, 'configuration': 'half-open sliding glazing'},
}
TABLE = {'center_xy': [.65, 0.], 'size_m': [1., 1.], 'top_z': .78,
         'collider_path': '/World/Geometry/worktop/Collider'}
RESERVED_TASK_VOLUME = {'low': [-.05, -.80, .78], 'high': [1.35, .90, 1.95]}
CAMERAS = {
    'room': {'eye': [-.35, -1.55, 1.66], 'target': [.60, .75, 1.35], 'focal_length_mm': 18.},
    'workcell': {'eye': [2.40, -1.50, 1.75], 'target': [.0, .80, 1.14], 'focal_length_mm': 22.},
    'tabletop': {'eye': [1.85, -1.45, 1.85], 'target': [.65, 0., .80], 'focal_length_mm': 34.},
}


def _materials(b):
    tex = b.assets_root / 'materials/ambientcg/v1'
    plaster = str(tex / 'PaintedPlaster017/PaintedPlaster017_1K-JPG_NormalGL.jpg')
    ph = b.assets_root / 'materials/polyhaven/v1'
    wood = str(ph / 'walnut_veneer/walnut_veneer_diff_1k.jpg')
    rough = str(ph / 'walnut_veneer/walnut_veneer_rough_1k.jpg')
    cloth = str(ph / 'rough_linen/rough_linen_nor_gl_1k.jpg')
    for name, color, r in [
        ('plaster', (.84, .845, .82), .91), ('ceiling', (.90, .91, .89), .92),
        ('reveal', (.045, .042, .035), .85), ('trim', (.73, .71, .65), .65),
        ('linen', (.60, .575, .505), .96), ('stone', (.62, .60, .55), .71),
        ('metal', (.06, .065, .06), .48), ('diffuser', (.84, .81, .72), .62),
        ('lamp_shade', (.65, .60, .49), .94), ('bulb', (.90, .85, .72), .35),
        ('earth', (.16, .17, .12), .97),
        ('wall_panel', (.86, .86, .82), .72), ('moulding', (.88, .88, .85), .62),
    ]:
        normal = cloth if name in ('linen', 'lamp_shade') else plaster if name == 'plaster' else None
        b.material('lounge_' + name, color, r, normal=normal)
    for i, tint in enumerate(((.98, .96, .91), (.94, .92, .88), (1., .98, .93), (.96, .94, .89))):
        b.material('lounge_oak_' + str(i), (.52, .40, .24), .61,
                   texture=wood, roughness_texture=rough, texture_tint=tint,
                   normal=str(ph / 'walnut_veneer/walnut_veneer_nor_gl_1k.jpg'))
    floor = ph / 'laminate_floor_02'
    b.material('lounge_oak_floor', (.6, .5, .4), .70,
               texture=str(floor / 'laminate_floor_02_diff_2k.jpg'),
               roughness_texture=str(floor / 'laminate_floor_02_rough_2k.jpg'),
               normal=str(floor / 'laminate_floor_02_nor_gl_2k.jpg'))
    table = ph / 'kitchen_wood'
    b.material('lounge_table_wood', (.6, .5, .4), .64,
               texture=str(table / 'kitchen_wood_diff_1k.jpg'),
               roughness_texture=str(table / 'kitchen_wood_rough_1k.jpg'),
               normal=str(table / 'kitchen_wood_nor_gl_1k.jpg'))
    for name, emission in [('diffuser', (2.0, 1.66, 1.12)), ('bulb', (3.0, 2.6, 1.9)),
                           ('lamp_shade', (.10, .078, .045))]:
        sh = UsdShade.Shader(b.stage.GetPrimAtPath('/World/Looks/lounge_' + name + '/Surface'))
        sh.CreateInput('emissiveColor', Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*emission))
    for name in ('SOURCE.json', 'LICENSE.txt'):
        for folder in ('PaintedPlaster017',):
            b.dependency(tex / folder / name)
        for folder in ('walnut_veneer', 'laminate_floor_02', 'rough_linen', 'kitchen_wood'):
            b.dependency(ph / folder / name)


def _curtain(b, name, center_y, width=.46):
    """Continuous pleated textile, suspended below the real ceiling track."""
    nu, nv = 48, 36
    verts, uv, faces = [], [], []
    for j in range(nv + 1):
        t = j / nv
        z = .020 + 2.595 * t
        for i in range(nu + 1):
            u = i / nu
            y = center_y + (u - .5) * width
            x = -1.77 + .035 * math.cos(u * 10 * math.pi) + .008 * (1 - t) * math.sin(t * math.pi * 3 + u * 8)
            verts.append((x, y, z)); uv.append((u * 2, t * 5))
    for j in range(nv):
        for i in range(nu):
            a = j * (nu + 1) + i
            faces.append((a, a + 1, a + nu + 2, a + nu + 1))
    path = b.path(name)
    UsdGeom.Xform.Define(b.stage, path)
    b.mesh(path + '/Fabric', verts, faces, 'lounge_linen', uv=uv)
    # The hem follows the lower pleats instead of looking like a rigid panel.
    b.tube(name + '_hem', [(p[0], p[1], p[2] + .015) for p in verts[:nu + 1]], .004, 'lounge_linen')


def _architecture(b):
    s = ROOM_SPEC; xmin, xmax, ymin, ymax = s['bounds']; h = s['height']
    w, d = xmax - xmin, ymax - ymin; cx, cy = (xmin + xmax) / 2, (ymin + ymax) / 2
    b.box('room_floor', (cx, cy, -.09), (w, d, .18), 'lounge_oak_0', 0)
    # Scanned plank grain, joints and roughness share a 2 m UV repeat.
    # Do not stretch a complete multi-board texture across each thin plank.
    floor_path = b.box('oak_floor_finish', (cx, cy, .0025),
                       (w, d, .005), 'lounge_oak_floor', 0, collision=False)
    st = UsdGeom.PrimvarsAPI(b.stage.GetPrimAtPath(floor_path + '/Visual')).GetPrimvar('st')
    st.Set([(float(u) * .25, float(v) * .25) for u, v in st.Get()])
    b.box('room_ceiling', (cx, cy, h + .07), (w, d, .14), 'lounge_ceiling', 0)
    for side, a, z in [('back', xmin, xmax), ('front', xmin, xmax), ('right', ymin, ymax)]:
        _wall_box(b, 'wall_' + side, side, (a + z) / 2, h / 2, z - a, h, s, 'lounge_plaster')
    win = s['window']; a = win['center'] - win['width'] / 2; z = win['center'] + win['width'] / 2
    for name, u, width, low, high in [
        ('left_front', (ymin + a) / 2, a - ymin, 0., h),
        ('left_back', (z + ymax) / 2, ymax - z, 0., h),
        ('left_sill', (a + z) / 2, z - a, 0., win['bottom']),
        ('left_header', (a + z) / 2, z - a, win['top'], h),
    ]:
        _wall_box(b, 'wall_' + name, 'left', u, (low + high) / 2, width, high - low, s, 'lounge_plaster')
    for side, a0, z0 in [('back', xmin, xmax), ('front', xmin, xmax), ('right', ymin, ymax), ('left', ymin, ymax)]:
        _wall_box(b, 'skirting_' + side, side, (a0 + z0) / 2, .045, z0 - a0, .09,
                  s, 'lounge_trim', .016, inset=.016)
    for i, yy in enumerate((a, z)):
        b.box('window_jamb_' + str(i), (-1.987, yy, 1.355), (.085, .05, 2.40), 'lounge_metal', .003)
    for i, zz in enumerate((.18, 2.53)):
        b.box('window_rail_' + str(i), (-1.987, .55, zz), (.085, 2.9, .05), 'lounge_metal', .003)
    b.box('window_mullion', (-1.987, .55, 1.355), (.065, .032, 2.30), 'lounge_metal', .002)
    # The sliding sash is genuinely open: both panes overlap on the front
    # half, leaving an unobstructed aperture for physical daylight transport.
    for i, xx in enumerate((-2.006, -1.972)):
        b.box('window_glazing_' + str(i), (xx, -.1475, 1.355),
              (.006, 1.365, 2.29), 'glass', 0, collision=False)
    for i, yy in enumerate((-.85, .55)):
        b.box('sliding_sash_jamb_' + str(i), (-1.972, yy, 1.355),
              (.028, .025, 2.30), 'lounge_metal', .002)
    for i, zz in enumerate((.208, 2.503)):
        b.box('sliding_sash_rail_' + str(i), (-1.972, -.15, zz),
              (.028, 1.425, .025), 'lounge_metal', .002)
    b.box('window_sill', (-1.945, .55, .164), (.20, 3.0, .032), 'lounge_stone', .003)
    for yy in (-.70, .47):
        b.box('window_handle_' + str(yy), (-1.934, yy, 1.04), (.025, .012, .12), 'lounge_metal', .003, collision=False)
    b.box('curtain_track', (-1.78, .55, 2.635), (.04, 3.35, .026), 'lounge_metal', .003, collision=False)
    _curtain(b, 'curtain_front', -.91); _curtain(b, 'curtain_back', 2.01)
    # A closed joinery door with jambs, latch and skirting returns.
    x, y = 2.76, -1.20
    b.box('door_reveal', (x, y, 1.065), (.035, .97, 2.13), 'lounge_reveal', .001)
    b.box('door_leaf', (x - .025, y, 1.06), (.045, .90, 2.10), 'lounge_oak_1', .003)
    for i, yy in enumerate((y - .49, y + .49)):
        b.box('door_jamb_' + str(i), (x - .045, yy, 1.075), (.075, .055, 2.15), 'lounge_trim', .002)
    b.box('door_head', (x - .045, y, 2.17), (.075, 1.03, .055), 'lounge_trim', .002)
    b.tube('door_lever', [(2.702, -1.52, 1.02), (2.64, -1.52, 1.02), (2.64, -1.40, 1.02)], .008, 'lounge_metal')
    for i, yy in enumerate((-.59, -.45)):
        b.box('wall_switch_' + str(i), (2.776, yy, 1.13), (.025, .080, .080), 'lounge_trim', .003)


def _table(b):
    top = b.box('worktop', (.65, 0., .76), (1., 1., .04), 'lounge_table_wood', .007)
    st = UsdGeom.PrimvarsAPI(b.stage.GetPrimAtPath(top + '/Visual')).GetPrimvar('st')
    st.Set([(float(u) * .5, float(v) * .5) for u, v in st.Get()])
    for i, (x, y) in enumerate(((.23, -.42), (1.07, -.42), (.23, .42), (1.07, .42))):
        bottom = .012 if y > 0 else .005
        b.box('table_leg_' + str(i), (x, y, (.74 + bottom) / 2),
              (.046, .046, .74 - bottom), 'lounge_oak_1', .003)
    for i, yy in enumerate((-.42, .42)):
        b.box('table_apron_' + str(i), (.65, yy, .701), (.84, .032, .078), 'lounge_oak_1', .003)


def _artwork(b):
    # Original tactile plaster relief in an oak frame, mounted to the rear wall.
    x, y, z = .25, 2.663, 1.63
    b.box('artwork_back', (x, y, z), (1.40, .025, .89), 'lounge_oak_1', .002)
    b.box('artwork_canvas', (x, y - .017, z), (1.35, .012, .84), 'lounge_linen', .001)
    for sign in (-1, 1):
        b.box('artwork_frame_side_' + str(sign), (x + sign * .709, y - .015, z),
              (.028, .050, .94), 'lounge_oak_1', .002)
        b.box('artwork_frame_rail_' + str(sign), (x, y - .015, z + sign * .456),
              (1.39, .050, .028), 'lounge_oak_1', .002)
    for i in range(16):
        xx = x - .52 + i * .069
        half = .17 + .13 * math.sin(i * .22 + .3)
        b.box('artwork_relief_' + str(i), (xx, y - .028, z + .045 * math.sin(i * .28)),
              (.033, .014, half * 2), 'lounge_plaster', .010, collision=False)


def _joinery(b):
    # Painted wall panelling and chair rail give the seating wall a human scale.
    for i, x in enumerate((-1.4, -.2, 1.0, 2.2)):
        b.box('wainscot_panel_' + str(i), (x, 2.683, .505),
              (1.10, .022, .79), 'lounge_wall_panel', .002, collision=False)
        for side in (-1, 1):
            b.box(f'wainscot_side_{i}_{side}', (x + side * .49, 2.665, .505),
                  (.018, .018, .65), 'lounge_moulding', .004, collision=False)
            b.box(f'wainscot_rail_{i}_{side}', (x, 2.665, .505 + side * .325),
                  (.998, .018, .018), 'lounge_moulding', .004, collision=False)
    b.box('seating_wall_chair_rail', (.40, 2.670, .950),
          (4.80, .045, .035), 'lounge_moulding', .006, collision=False)
    # A shallow pale media-wall panel; TV mounting hardware remains visible.
    b.box('media_wall_shadow_reveal', (2.795, .85, 1.45),
          (.008, 2.35, 2.37), 'lounge_reveal', .002, collision=False)
    b.box('media_wall_panel', (2.780, .85, 1.45),
          (.028, 2.30, 2.32), 'lounge_wall_panel', .004, collision=False)
    for i, yy in enumerate((-.385, 2.085)):
        b.box('media_wall_edge_' + str(i), (2.774, yy, 1.45),
              (.035, .025, 2.37), 'lounge_oak_1', .003, collision=False)
    # Ceiling supply grille outside the decorative cove and task workspace.
    b.box('ventilation_grille_frame', (.57, -1.58, 2.840),
          (1.15, .18, .018), 'lounge_moulding', .003, collision=False)
    for i in range(5):
        b.box('ventilation_slot_' + str(i), (.57, -1.64 + i * .03, 2.828),
              (1.05, .010, .006), 'lounge_reveal', .001, collision=False)


def _lighting(b):
    fixtures = []
    # Two joined perimeter coves. The source is behind an 85 mm glare lip,
    # mounted in an aluminium channel, and sends light upwards to the ceiling.
    for side in ('back', 'right'):
        if side == 'back':
            shelf = ((.40, 2.59, 2.640), (4.80, .22, .025))
            lip = ((.40, 2.492, 2.6875), (4.80, .024, .085))
            channel = ((.35, 2.63, 2.662), (4.50, .024, .018))
            diffuser = ((.35, 2.63, 2.6725), (4.47, .014, .003))
            emitter = (.35, 2.63, 2.676); size = (4.44, .012)
        else:
            shelf = ((2.69, .37, 2.640), (.22, 4.22, .025))
            lip = ((2.592, .37, 2.6875), (.024, 4.22, .085))
            channel = ((2.73, .37, 2.662), (.024, 4.12, .018))
            diffuser = ((2.73, .37, 2.6725), (.014, 4.09, .003))
            emitter = (2.73, .37, 2.676); size = (.012, 4.06)
        b.box('cove_' + side + '_shelf', *shelf, 'lounge_ceiling', .002)
        b.box('cove_' + side + '_glare_lip', *lip, 'lounge_ceiling', .002)
        b.box('cove_' + side + '_channel', *channel, 'steel', .001, collision=False)
        b.box('cove_' + side + '_diffuser', *diffuser, 'lounge_diffuser', .001, collision=False)
        light(b, 'cove_' + side, emitter, (emitter[0], emitter[1], 2.85), size,
              30000., (1, 1, 1), role='accent', temperature=4200.)
        fixtures.append({'name': side + '_cove', 'type': 'concealed LED strip', 'kelvin': 4200,
                         'source_z_m': emitter[2], 'lip_top_z_m': 2.730,
                         'ceiling_z_m': 2.85, 'direction': 'up', 'aluminium_channel': True})
    # Hollow ceiling-fixed curtain pocket leaves clearance for track and folds.
    b.box('curtain_pelmet_front', (-1.606, .55, 2.73), (.026, 3.45, .24), 'lounge_ceiling', .002)
    for i, yy in enumerate((-1.162, 2.262)):
        b.box('curtain_pelmet_end_' + str(i), (-1.79, yy, 2.73),
              (.38, .026, .24), 'lounge_ceiling', .002)
    # Downlights are disk sources contained inside a matching circular housing.
    for i, (x, y) in enumerate(((-.55, -.95), (1.60, -.95), (.20, 1.65), (1.90, 1.65))):
        b.cylinder('downlight_recess_' + str(i), (x, y, 2.838), .071, .020, 'lounge_metal', collision=False)
        b.cylinder('downlight_diffuser_' + str(i), (x, y, 2.825), .055, .006, 'lounge_diffuser', collision=False)
        light(b, 'downlight_' + str(i), (x, y, 2.820), (x, y, .78), (.105, .105),
              650000., (1, 1, 1), shape='disk', role='task', temperature=5200.)
        fixtures.append({'name': 'downlight_' + str(i), 'type': 'recessed disk', 'kelvin': 5200,
                         'diameter_m': .105, 'source_inside_fixture': True})
    # Reading light sits behind the sofa, outside the television viewing cone.
    x, y = -1.43, 2.40
    b.cylinder('reading_lamp_base', (x, y, .034), .15, .044, 'lounge_metal')
    b.cylinder('reading_lamp_stem', (x, y, .735), .012, 1.38, 'lounge_metal')
    b.lathe('reading_lamp_shade', (x, y, 1.40),
            [(.19, 0), (.125, .29), (.119, .29), (.184, .008)], 'lounge_lamp_shade')
    b.sphere('reading_lamp_bulb', (x, y, 1.52), .027, 'lounge_bulb')
    light(b, 'reading_lamp', (x, y, 1.399), (x, y, .55), (.29, .29),
          11000., (1, 1, 1), shape='disk', role='accent', temperature=2700.)
    b.tube('reading_lamp_cable', [(x, y, .010), (-1.78, y, .010), (-1.88, 2.52, .010),
                               (-1.88, 2.672, .010), (-1.88, 2.672, .23)], .003, 'lounge_metal')
    b.box('reading_lamp_outlet', (-1.88, 2.686, .25), (.085, .028, .085), 'lounge_trim', .003)
    b.box('reading_lamp_plug', (-1.88, 2.660, .25), (.032, .030, .040), 'lounge_metal', .003)
    fixtures.append({'name': 'reading_lamp', 'type': 'shaded floor lamp', 'kelvin': 2700,
                     'floor_supported': True, 'cable_to_wall_outlet': True})
    sun = UsdLux.DistantLight.Define(b.stage, '/World/Lights/Sun')
    sun.CreateIntensityAttr(2700.); sun.CreateAngleAttr(1.8)
    _light_baseline(sun, 'sun', 5600.)
    q = Gf.Rotation(Gf.Vec3d(0, 0, -1), Gf.Vec3d(1., .20, -.60)).GetQuat()
    UsdGeom.Xformable(sun).AddOrientOp().Set(Gf.Quatf(q))
    sky = UsdLux.DomeLight.Define(b.stage, '/World/Lights/Sky')
    hdr = b.assets_root / HDR_RELATIVE
    sky.CreateTextureFileAttr(str(hdr)); sky.CreateTextureFormatAttr('latlong'); sky.CreateIntensityAttr(2200.)
    UsdGeom.Xformable(sky).AddRotateXYZOp().Set(Gf.Vec3f(*HDR_ROTATE_XYZ_DEG))
    _light_baseline(sky, 'sky', 6500.); b.dependency(hdr)
    b.dependency(hdr.with_name('SOURCE.json'))
    return fixtures


def build_living_room(stage, assets_root, output_root, *, lighting_profile='nominal'):
    b = Builder(stage, Path(assets_root), Path(output_root)); b.family = 'living_room'
    UsdGeom.Xform.Define(stage, '/World/Robot')
    _materials(b); _architecture(b); _table(b); _artwork(b); _joinery(b)
    exterior = build_exterior(b)
    furnishings = build_furnishings(b, right_wall_x=ROOM_SPEC['bounds'][1])
    fixtures = _lighting(b)
    lighting = apply_lighting_profile(stage, 'living_room', lighting_profile)
    return {
        'schema': 'scence.living_room_environment.v1', 'family': 'living_room',
        'variant': VARIANT, 'task_independent': True, 'cameras': CAMERAS,
        'robot_placeholder': {'path': '/World/Robot', 'contains_robot': False},
        'layout': {'table_center_xy': TABLE['center_xy'], 'table_size_m': TABLE['size_m'],
                   'table_top_z': TABLE['top_z'], 'room': ROOM_SPEC, 'reserved_task_volume': RESERVED_TASK_VOLUME},
        'support_colliders': {'table': [TABLE['collider_path']], 'floor': ['/World/Geometry/room_floor/Collider']},
        'lighting': lighting, 'lighting_fixtures': fixtures, 'furnishings': furnishings, 'exterior': exterior,
        'objects': [], 'geometry': b.records, 'dependencies': b.dependencies,
        'design': {'style': 'restrained contemporary residential interior',
                   'lighting': 'Residential balcony and sky HDR daylight, concealed 4200 K indirect coves, recessed 5200 K downlights and 2700 K shaded reading light',
                   'no_exposed_floating_light_panels': True,
                   'materials': 'Original procedural joinery with pinned ambientCG PBR and Poly Haven furniture'},
        'ownership': {'environment': 'architecture, furniture, empty table, luminaires and static scenery',
                      'task': 'movable objects, robot layout, reset, scoring and PhysicsScene'},
        'scope': 'Static environment and native rendered appearance only; no task or contact qualification',
    }
