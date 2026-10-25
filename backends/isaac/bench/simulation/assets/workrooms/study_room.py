"""Detailed study with an empty executive desk and an accessible rear shelf.

The task-facing shelf is real open joinery with separate box colliders. Its
empty bays are room geometry, not a task reset or a reachability qualification.
Books and furniture elsewhere are static scenery. Robots and movable books
belong to separate task compositions, never to this fixed environment.
"""
from copy import deepcopy
import math
from pathlib import Path

from pxr import Gf, Sdf, UsdGeom, UsdLux, UsdPhysics, UsdShade

from bench.paths import load_json
from scenes.workrooms.study.book_assets import BOOK_SPECS, author_display_book

from .builder import Builder, safe
from .common import (_ceiling_lights, _door_and_services, _floor_and_trims,
                     _light_baseline, _room_materials, _wall_box, _window,
                     apply_lighting_profile, light)


VARIANT = 'study_library_20261019_v7'
FAMILY = 'study'
TABLE = {'center_xy': [.65, 0.], 'size_m': [.90, 2.00], 'top_z': .78,
         'thickness_m': .08, 'collider_path': '/World/Geometry/worktop/Collider'}
ROOM_SPEC = {
    'bounds': [-1.45, 2.50, -1.65, 1.90], 'height': 2.65,
    'theme': 'executive library with a double-pedestal walnut desk and a reading corner',
    'window': {'wall': 'back', 'center': 1.27, 'width': 1.48,
               'bottom': 1.02, 'top': 2.30},
}
RESERVED_TASK_VOLUME = {'low': [.18, -.72, .78001], 'high': [1.12, .72, 2.05]}
ROBOT_FLOOR_RESERVE = {'low': [-.65, -.70, 0.], 'high': [.18, .70, 2.05]}
CAMERAS = {
    'room': {'eye': [2.25, -1.40, 1.95], 'target': [.12, .46, 1.12], 'focal_length_mm': 19.},
    'workcell': {'eye': [2.24, -1.45, 1.48], 'target': [.17, .06, .85], 'focal_length_mm': 20.},
    'tabletop': {'eye': [2.16, -1.38, 1.78], 'target': [.65, 0., .60], 'focal_length_mm': 25.},
    'shelf': {'eye': [.75, -1.25, 1.62], 'target': [-1., 0., 1.02], 'focal_length_mm': 27.},
}

# Bounds describe the clear interiors, excluding the back, shelf, sides and
# central divider. +X remains entirely open; no front lip or invisible plane.
SHELF_BAYS = [
    {'id': f'rear_{level}_{side}', 'low': [-1.14, yl, support],
     'high': [-.84, yh, support + .336], 'support_z_m': support,
     'collider_path': f'/World/Geometry/rear_shelf_level_{level}/Collider',
     'opening': '+x', 'opening_direction': [1., 0., 0.]}
    for level, support in enumerate((.76, 1.12), start=1)
    for side, yl, yh in (('left', -.672, -.012), ('right', .012, .672))
]


def _materials(b):
    ph = b.assets_root / 'materials/polyhaven/v1'
    ac = b.assets_root / 'materials/ambientcg/v1'
    for name, slug, color, roughness in (
            ('study_walnut', 'walnut_veneer', (.42, .28, .15), .50),
            ('study_desktop', 'walnut_veneer', (.36, .24, .15), .39),
            ('study_linen', 'rough_linen', (.72, .67, .56), .93)):
        folder = ph / slug
        b.material(name, color, roughness,
                   texture=str(folder / f'{slug}_diff_1k.jpg'),
                   normal=str(folder / f'{slug}_nor_gl_1k.jpg'),
                   roughness_texture=str(folder / f'{slug}_rough_1k.jpg'),
                   texture_tint=(.88, .92, .94) if name != 'study_linen' else (.95, .98, 1.))
    floor = ph / 'laminate_floor_02'
    b.material('study_floor', (.56, .42, .28), .67,
               texture=str(floor / 'laminate_floor_02_diff_2k.jpg'),
               normal=str(floor / 'laminate_floor_02_nor_gl_2k.jpg'),
               roughness_texture=str(floor / 'laminate_floor_02_rough_2k.jpg'))
    plaster = str(ac / 'PaintedPlaster017/PaintedPlaster017_1K-JPG_NormalGL.jpg')
    b.material('study_plaster', (.79, .77, .70), .91, normal=plaster)
    b.material('study_sage', (.27, .33, .28), .76, normal=plaster)
    for name, color, roughness, metal in (
            ('study_brass', (.36, .25, .11), .36, .82),
            ('study_dark_metal', (.055, .058, .050), .43, .65),
            ('study_paper', (.82, .79, .69), .92, 0.),
            ('study_cover_ochre', (.43, .28, .11), .81, 0.),
            ('study_cover_blue', (.09, .15, .20), .79, 0.),
            ('study_cover_rust', (.30, .12, .08), .85, 0.),
            ('study_cover_green', (.18, .23, .16), .87, 0.),
            ('study_ceramic', (.62, .57, .46), .32, 0.),
            ('study_ink', (.12, .14, .13), .92, 0.),
            ('study_edge_shadow', (.048, .038, .029), .7, 0.),
            ('study_stitch', (.53, .47, .34), .86, 0.),
            ('study_trim_ivory', (.78, .78, .73), .70, 0.),
            ('study_curtain', (.67, .66, .60), .96, 0.)):
        b.material(name, color, roughness, metal)

    walnut = ph / 'walnut_veneer'
    b.material('study_executive_walnut', (.20, .11, .055), .34,
               texture=str(walnut/'walnut_veneer_diff_1k.jpg'),
               normal=str(walnut/'walnut_veneer_nor_gl_1k.jpg'),
               roughness_texture=str(walnut/'walnut_veneer_rough_1k.jpg'),
               texture_tint=(.58, .54, .49))
    rough = UsdShade.Shader(b.stage.GetPrimAtPath(
        '/World/Looks/study_executive_walnut/roughnessTexture'))
    rough.CreateInput('scale', Sdf.ValueTypeNames.Float4).Set(
        Gf.Vec4f(.72, .72, .72, 1.))

    leather = ph / 'brown_leather'
    # Recolour the source scan while retaining its real grain and wear.
    from PIL import Image, ImageStat
    source = leather/'brown_leather_albedo_1k.jpg'
    b.dependency(source)
    b.dependency(leather/'SOURCE.json')
    grain = Image.open(source).convert('L')
    mean = ImageStat.Stat(grain).mean[0]
    dyed = Image.merge('RGB', [grain.point([
        max(0,min(255,round(base+(value-mean)*.9))) for value in range(256)])
        for base in (61,73,66)])
    dyed_path = b.generated/'sage_leather_albedo.png'
    dyed.save(dyed_path)
    b.material('study_desk_leather', (.14, .23, .20), .53,
               texture=str(dyed_path),
               normal=str(leather/'brown_leather_nor_gl_1k.jpg'),
               roughness_texture=str(leather/'brown_leather_rough_1k.jpg'))
    # Restrained normal amplitude preserves the scale of polished furniture.
    for name in ('study_desktop', 'study_walnut', 'study_executive_walnut', 'study_desk_leather'):
        tex = UsdShade.Shader(b.stage.GetPrimAtPath('/World/Looks/'+name+'/normalTexture'))
        if tex:
            amplitude = .9 if name == 'study_desk_leather' else .5
            tex.GetInput('scale').Set(Gf.Vec4f(amplitude, amplitude, 2., 1.))
            tex.GetInput('bias').Set(Gf.Vec4f(-amplitude/2, -amplitude/2, -1., 0.))


def _uv_scale(b, path, factor):
    st = UsdGeom.PrimvarsAPI(b.stage.GetPrimAtPath(path + '/Visual')).GetPrimvar('st')
    if st and st.Get():
        st.Set([(float(u) * factor, float(v) * factor) for u, v in st.Get()])


def _architecture(b):
    # Use the existing architecture helpers with this room's own window layout;
    # the shared six-room specifications are never changed or monkeypatched.
    _room_materials(b, FAMILY)
    _floor_and_trims(b, FAMILY, ROOM_SPEC)
    xmin, xmax, ymin, ymax = ROOM_SPEC['bounds']
    h, window = ROOM_SPEC['height'], ROOM_SPEC['window']
    b.box('room_ceiling', ((xmin+xmax)/2, (ymin+ymax)/2, h+.07),
          (xmax-xmin, ymax-ymin, .14), 'room_ceiling', 0)
    for wall, low, high in (('left', ymin, ymax), ('right', ymin, ymax),
                            ('back', xmin, xmax), ('front', xmin, xmax)):
        if wall != window['wall']:
            _wall_box(b, 'wall_' + wall, wall, (low+high)/2, h/2, high-low, h, ROOM_SPEC)
            continue
        a, c = window['center']-window['width']/2, window['center']+window['width']/2
        for name, left, right, z, height in (
                ('before', low, a, h/2, h), ('after', c, high, h/2, h),
                ('below', a, c, window['bottom']/2, window['bottom']),
                ('above', a, c, (h+window['top'])/2, h-window['top'])):
            _wall_box(b, 'wall_'+wall+'_'+name, wall, (left+right)/2, z,
                      right-left, height, ROOM_SPEC)
    _window(b, FAMILY, ROOM_SPEC)
    _door_and_services(b, FAMILY, ROOM_SPEC)
    _ceiling_lights(b, FAMILY, ROOM_SPEC)
    for record in b.records:
        name = record['name']
        path = '/World/Geometry/' + name + '/Visual'
        prim = b.stage.GetPrimAtPath(path)
        if prim and name.startswith('wall_'):
            b.bind(prim, 'study_plaster')
    b.bind(b.stage.GetPrimAtPath('/World/Geometry/room_floor/Visual'), 'study_floor')
    b.bind(b.stage.GetPrimAtPath('/World/Geometry/room_ceiling/Visual'), 'study_trim_ivory')
    _uv_scale(b, '/World/Geometry/room_floor', .25)
    # Real housings surround the library wash and the opal ceiling diffuser.
    b.box('library_wash_housing', (.30, ymax-.45, h-.049),
          (2.24, .13, .038), 'study_dark_metal', .006, collision=False)
    b.cylinder('pendant_ceiling_canopy', (.50, -.15, h-.015), .09, .03,
               'study_brass', collision=False)
    b.cylinder('pendant_stem', (.50, -.15, h-.07), .008, .10,
               'study_brass', collision=False)
    # Paired blind lift strings, a fitted sill and opening handle add scale.
    for i, x in enumerate((.68, 1.86)):
        b.tube(f'blind_lift_cord_{i}', [(x, ymax-.075, 2.26), (x, ymax-.075, 1.07)],
               .0015, 'study_linen')
    b.tube('window_handle', [(1.98, ymax-.055, 1.61), (1.98, ymax-.095, 1.61),
                            (1.98, ymax-.095, 1.51)], .007, 'study_brass')
    # A socket near the reading chair, with a real lead confined to that corner.
    b.box('reading_socket', (2.476, 1.18, .27), (.024, .115, .070), 'white', .004)
    for y in (1.155, 1.205):
        b.box('reading_socket_slot_' + str(y), (2.462, y, .27),
              (.002, .005, .018), 'study_dark_metal', .001, collision=False)


def _desk(b):
    """Two-metre executive double-pedestal desk; the owner sits at -X.

    Every furniture collision shape is below the unchanged .78 m task plane.
    The large top is one solid support. Flush leather and decorative veneers
    are visual skins, so book contact never crosses raised trim or a seam.
    """
    wood = 'study_executive_walnut'
    top = b.box('worktop', (.65, 0., .74), (.90, 2.00, .08), wood, .009)
    _uv_scale(b, top, .48)
    # Drop the visual top 2 mm to expose the flush fitted surface, while keeping
    # the exact, unbroken 80 mm collision slab. The edge bands fill its outline.
    UsdGeom.Xformable(b.stage.GetPrimAtPath(top+'/Visual')).AddScaleOp().Set(
        Gf.Vec3f(1., 1., .95))
    for i, x in enumerate((.257, 1.043)):
        b.box(f'desk_top_frame_long_{i}', (x, 0., .776), (.114, 2.00, .008),
              wood, .0015, collision=False)
    for i, y in enumerate((-.875, .875)):
        b.box(f'desk_top_frame_short_{i}', (.65, y, .776), (.672, .250, .008),
              wood, .0015, collision=False)
    # Layered solid lipping gives the visitor-facing edge a broad, substantial
    # profile. The fine metal line is embedded in the vertical edge, not on top.
    b.box('desk_top_shadow_reveal', (.65, 0., .696), (.856, 1.956, .008),
          'study_edge_shadow', .001)
    b.box('desk_top_lower_lipping', (.65, 0., .684), (.838, 1.938, .016),
          wood, .003)
    b.box('desk_front_edge_inlay', (1.0991, 0., .726), (.0008, 1.976, .0015),
          'study_brass', .0002, collision=False)
    for sign in (-1, 1):
        b.box(f'desk_side_edge_inlay_{sign}', (.65, sign*.9991, .726),
              (.878, .0008, .0015), 'study_brass', .0002, collision=False)
    leather = b.box('desk_flush_leather', (.65, 0., .77957), (.664, 1.492, .0005),
                    'study_desk_leather', .0001, collision=False)
    _uv_scale(b, leather, 2.2)
    for x in (.314, .986):
        b.box('desk_leather_inlay_x_'+str(x), (x, 0., .7796), (.0012, 1.502, .0007),
              'study_brass', .0001, collision=False)
    for y in (-.751, .751):
        b.box('desk_leather_inlay_y_'+str(y), (.65, y, .7796), (.672, .0012, .0007),
              'study_brass', .0001, collision=False)
    for sign in (-1, 1):
        for i in range(122):
            b.box(f'desk_stitch_long_{sign}_{i}', (.65+sign*.324, -.724+i*.0119, .77989),
                  (.00065, .0042, .00012), 'study_stitch', 0, collision=False)
        for i in range(53):
            b.box(f'desk_stitch_short_{sign}_{i}', (.34+i*.0119, sign*.734, .77989),
                  (.0042, .00065, .00012), 'study_stitch', 0, collision=False)

    # Full-height cabinet bodies replace all four slender table legs. Recessed
    # plinths meet the floor; the central 1 m knee opening faces the robot.
    for side, y in enumerate((-.745, .745)):
        prefix = f'desk_pedestal_{side}'
        b.box(prefix+'_plinth', (.65, y, .040), (.704, .394, .080),
              'study_edge_shadow', .003)
        b.box(prefix+'_body', (.65, y, .377), (.756, .438, .594), wood, .005)
        b.box(prefix+'_top_bearing', (.65, y, .681), (.756, .438, .014),
              'study_edge_shadow', .001)
        # Broad inset outer side panels and framed front end panels are
        # veneered joinery, with physical carcasses immediately behind them.
        outer = y + (-.220 if side == 0 else .220)
        b.box(prefix+'_side_reveal', (.65, outer, .377), (.644, .003, .488),
              'study_edge_shadow', .0005, collision=False)
        b.box(prefix+'_side_field', (.65, outer+(-.0018 if side==0 else .0018), .377),
              (.626, .003, .470), wood, .002, collision=False)
        # The owner is on -X. The drawer faces, pulls and lock all face -X.
        b.box(prefix+'_drawer_reveal', (.269, y, .377), (.007, .414, .574),
              'study_edge_shadow', .001)
        for row, (z, height) in enumerate(((.565, .180), (.379, .180), (.189, .188))):
            b.box(prefix+f'_drawer_{row}', (.256, y, z), (.022, .402, height), wood, .003)
            b.box(prefix+f'_drawer_border_{row}', (.2445, y, z), (.001, .364, height-.036),
                  'study_edge_shadow', .0002, collision=False)
            b.box(prefix+f'_drawer_field_{row}', (.2437, y, z), (.001, .356, height-.044),
                  wood, .0002, collision=False)
            for sign in (-1, 1):
                b.cylinder(prefix+f'_pull_mount_{row}_{sign}', (.234, y+sign*.067, z+.019),
                           .007, .016, 'study_brass', rotation=(0,90,0), collision=False)
            b.tube(prefix+f'_pull_{row}', [(.232,y-.067,z+.019), (.217,y-.067,z+.019),
                   (.217,y+.067,z+.019), (.232,y+.067,z+.019)], .0038, 'study_brass')
        b.cylinder(prefix+'_lock_bezel', (.2435, y, .620), .008, .002,
                   'study_brass', rotation=(0,90,0), collision=False)
        b.box(prefix+'_key_slot', (.2421, y, .620), (.0006, .0015, .007),
              'study_edge_shadow', 0, collision=False)

    # A large framed visitor-side modesty panel is the defining executive-desk
    # silhouette. It leaves a genuine knee well on -X and an 85 mm floor gap.
    b.box('desk_modesty_backing', (1.011, 0., .377), (.048, 1.816, .584), wood, .004)
    b.box('desk_modesty_reveal', (1.036, 0., .377), (.002, 1.696, .468),
          'study_edge_shadow', .001, collision=False)
    b.box('desk_modesty_raised_field', (1.043, 0., .377), (.014, 1.676, .448),
          wood, .003, collision=False)
    # Inlaid double lines and continuous rails avoid a featureless box facade.
    for z in (.162, .592):
        b.box('desk_modesty_horizontal_inlay_'+str(z), (1.0502, 0., z),
              (.0008, 1.652, .0014), 'study_brass', .0001, collision=False)
    for y in (-.826, .826):
        b.box('desk_modesty_vertical_inlay_'+str(y), (1.0502, y, .377),
              (.0008, .0014, .430), 'study_brass', .0001, collision=False)
    b.box('desk_modesty_lower_rail', (1.014, 0., .104), (.050, 1.834, .038),
          wood, .004)


def _rear_shelf(b):
    b.box('rear_shelf_toe', (-1., 0., .05), (.285, 1.34, .10), 'study_dark_metal', .003)
    b.box('rear_shelf_back', (-1.15, 0., .975), (.020, 1.40, 1.75), 'study_walnut', .002)
    for i, y in enumerate((-.686, .686)):
        b.box(f'rear_shelf_side_{i}', (-1., y, .975), (.32, .028, 1.75),
              'study_walnut', .003)
    for level, top in enumerate((.124, .76, 1.12, 1.48, 1.85)):
        b.box(f'rear_shelf_level_{level}', (-1., 0., top-.012), (.32, 1.40, .024),
              'study_walnut', .002)
        # The edge veneer is wholly inside the board's existing solid envelope.
        b.box(f'rear_shelf_edge_{level}', (-.841, 0., top-.012), (.002, 1.344, .021),
              'study_desktop', .0006, collision=False)
    b.box('rear_shelf_divider', (-.99, 0., 1.108), (.30, .024, .696), 'study_walnut', .002)
    # Lower closed storage is separate from the four open task bays.
    for i, y in enumerate((-.34, .34)):
        b.box(f'rear_shelf_lower_door_{i}', (-.848, y, .430), (.016, .664, .594),
              'study_desktop', .003)
        b.tube(f'rear_shelf_lower_pull_{i}', [(-.834, y-.045, .58), (-.820, y-.045, .58),
                       (-.820, y+.045, .58), (-.834, y+.045, .58)], .003, 'study_brass')
    for side, y in enumerate((-.680, .680)):
        for level, top in enumerate((.76, 1.12, 1.48)):
            for x in (-1.10, -.89):
                b.cylinder(f'rear_shelf_pin_{side}_{level}_{x}', (x, y, top-.029),
                           .0035, .012, 'study_brass', rotation=(90, 0, 0), collision=False)
    display_ids = tuple(BOOK_SPECS)[6:]
    for i in range(9):
        book_id = display_ids[i]
        height = BOOK_SPECS[book_id]['size_xyz_m'][2]
        author_display_book(b, f'rear_shelf_display_book_{i}', book_id,
                            position=(-1.012, -.55+i*.052, 1.48+height/2))
    # Small empty ceramic vase on the upper display tier, away from task bays.
    b.lathe('rear_shelf_display_vase', (-1.005, .43, 1.48),
            [(0, 0), (.043, 0), (.059, .032), (.055, .12), (.024, .19),
             (.019, .19), (.020, .165), (.041, .10), (.034, .012), (0, .012)],
            'study_ceramic')
    for prim in b.stage.GetPrimAtPath('/World/Geometry').GetChildren():
        if prim.GetName().startswith('rear_shelf_display_'):
            prim.SetCustomDataByKey('scence:staticDecoration', True)
    return {'name': 'rear_classification_shelf', 'bounds_m': [[-1.16, -.70, 0.], [-.82, .70, 1.85]],
            'carcass_bounds_m': [[-1.16, -.70, .10], [-.84, .70, 1.85]],
            'opening': '+x', 'empty_bays': deepcopy(SHELF_BAYS),
            'task_bay_contents': [], 'upper_books': 'fixed non-task display geometry',
            'collision': 'Separate positive-scale box panels; open fronts have no collider',
            'scope': 'Room support geometry only; no reachability, grasp or sorting qualification'}


def _source_asset(b, name, slug, position, *, yaw=0., version='v1'):
    root = b.assets_root / 'objects/polyhaven/derived' / version / slug
    derivation = root / 'DERIVATION.json'
    metadata = load_json(derivation)
    for source in (derivation, metadata['source'], metadata['source_provenance'],
                   Path(metadata['source_provenance']).with_name('LICENSE.txt'),
                   *metadata['texture_files']):
        b.dependency(source)
    b.asset(name, root/(slug+'.usdc'), position, rotation=(0, 0, yaw))
    return {'name': name, 'asset': str(root/(slug+'.usdc')), 'derivation': str(derivation),
            'position_m': list(position), 'yaw_deg': yaw, 'scale': 1.,
            'role': 'static room furnishing; not a task object'}


def _background_library(b):
    # This compact back-wall bookcase is scenery. Its front faces -Y and it
    # stays outside the rear-shelf/robot approaches and the writing surface.
    x, y, width, depth = -.72, 1.69, .94, .36
    b.box('background_library_plinth', (x, y, .06), (.90, .31, .12), 'study_walnut', .003)
    b.box('background_library_back', (x, y+.17, 1.16), (.94, .020, 2.08), 'study_sage', .001)
    for sign in (-1, 1):
        b.box('background_library_side_'+str(sign), (x+sign*.456, y, 1.16),
              (.028, depth, 2.08), 'study_walnut', .003)
    for i, z in enumerate((.132, .70, 1.05, 1.40, 1.75, 2.20)):
        b.box(f'background_library_board_{i}', (x, y, z), (width, depth, .024), 'study_walnut', .003)
    for sign in (-1, 1):
        b.box('background_library_door_'+str(sign), (x+sign*.226, y-.185, .414),
              (.443, .024, .536), 'study_desktop', .003)
        b.tube('background_library_pull_'+str(sign), [(x+sign*.04, y-.20, .46),
               (x+sign*.04, y-.225, .46), (x+sign*.04, y-.225, .56)], .0035, 'study_brass')
    display_ids = tuple(BOOK_SPECS)
    for row, support in enumerate((.712, 1.062, 1.412, 1.762)):
        for i in range(11 if row%2 else 8):
            book_id = display_ids[(row*9+i+9)%len(display_ids)]
            height = BOOK_SPECS[book_id]['size_xyz_m'][2]
            author_display_book(b, f'background_book_{row}_{i}', book_id,
                position=(x-.38+i*.065, y, support+height/2), yaw_deg=-90.)


def _wall_art(b):
    """Original botanical plates, with mats, fine frames and printed captions."""
    from PIL import Image, ImageDraw, ImageFont
    import random
    font = b.assets_root/'fonts/noto_cjk_sc/v1/NotoSansCJKsc-Regular.otf'
    b.dependency(font)
    for index,(title,subtitle) in enumerate((('银杏 / GINKGO','GINKGO BILOBA · 植物观察 01'),
                                           ('蕨 / FERN','DRYOPTERIS · 植物观察 02'))):
        rng = random.Random(17+index)
        im = Image.new('RGB',(1200,1560),(229,226,213))
        draw = ImageDraw.Draw(im)
        for _ in range(17000):
            xx,yy=rng.randrange(1200),rng.randrange(1560)
            c=rng.randrange(199,231);draw.point((xx,yy),fill=(c+5,c+3,c-6))
        draw.line((170,1120,1020,1120),fill=(152,151,136),width=2)
        if index==0:
            draw.line([(595,1060),(586,810),(615,575),(600,275)],fill=(79,93,65),width=8)
            for i in range(9):
                cy=390+i*72;side=-1 if i%2 else 1;cx=600+side*(105+(i%3)*27)
                draw.line((590,cy+50,cx,cy),fill=(89,100,66),width=4)
                points=[(cx,cy+7)]
                for j in range(34):
                    angle=math.radians(196+j*4.5)
                    points.append((cx+110*math.cos(angle),cy+107*math.sin(angle)))
                draw.polygon(points,fill=(113+i*4,128+i*2,82+i),outline=(76,91,63),width=3)
                for j in range(1,9):
                    angle=math.radians(202+j*16)
                    draw.line((cx,cy,cx+102*math.cos(angle),cy+99*math.sin(angle)),fill=(153,156,107),width=2)
        else:
            draw.line((590,1060,650,240),fill=(77,100,73),width=7)
            for i in range(22):
                cy=350+i*29;cx=642-i*2.0;length=35+135*math.sin(math.pi*(i+2)/26)
                for side in (-1,1):
                    tip=(cx+side*length,cy-62)
                    draw.polygon([(cx,cy), (cx+side*length*.45,cy-50),tip,
                                  (cx+side*length*.67,cy-10)],fill=(76+i*2,104+i,70+i),outline=(60,87,61))
                    draw.line((cx,cy,*tip),fill=(155,164,119),width=2)
        draw.text((600,1210),title,font=ImageFont.truetype(str(font),51),fill=(45,61,51),anchor='mm')
        draw.text((600,1300),subtitle,font=ImageFont.truetype(str(font),22),fill=(103,109,91),anchor='mm')
        draw.text((600,1435),'FIELD NOTES  /  自然手记',font=ImageFont.truetype(str(font),18),fill=(123,124,109),anchor='mm')
        path=b.generated/f'botanical_plate_{index}.png';im.save(path)
        material=b.material(f'study_art_print_{index}',(1,1,1),.88,texture=str(path))
        # Rear wall above the task shelf; nothing enters the robot's volume.
        x,y,z=-1.407,-.42+index*.82,2.222
        b.box(f'study_art_frame_{index}',(x,y,z),(.027,.414,.524),'study_walnut',.003)
        b.box(f'study_art_mat_{index}',(x+.016,y,z),(.008,.374,.484),'study_paper',.001)
        root=b.path(f'study_art_print_{index}')
        b.mesh(root,[(x+.021,y-.148,z-.195),(x+.021,y+.148,z-.195),
                     (x+.021,y+.148,z+.195),(x+.021,y-.148,z+.195)],
               [(0,1,2,3)],material,[(0,0),(1,0),(1,1),(0,1)])


def _room_details(b):
    # Light painted trim replaces the old all-brown lower wall treatment.
    for record in b.records:
        if record['name'].startswith(('wood_wainscot','wainscot','wall_panel','panel_','baseboard','skirting')):
            prim=b.stage.GetPrimAtPath('/World/Geometry/'+safe(record['name'])+'/Visual')
            if prim:b.bind(prim,'study_trim_ivory')
    for wallx in (-1.414,2.464):
        b.box('picture_rail_'+str(wallx),(wallx,.125,2.52),(.022,3.46,.028),
              'study_trim_ivory',.003,collision=False)
    # Linen curtains with individual folds, hems, rings and a brass pole.
    b.cylinder('curtain_rod',(1.27,1.733,2.405),.010,1.99,'study_brass',
               rotation=(0,90,0),collision=False)
    for sign in (-1,1):
        b.sphere('curtain_finial_'+str(sign),(1.27+sign*1.015,1.733,2.405),.021,'study_brass')
    for side,center in enumerate((.405,2.135)):
        verts=[];uv=[];faces=[]
        for j in range(33):
            z=.30+j*(2.055/32)
            for i in range(57):
                u=i/56;x=center+(u-.5)*.31
                y=1.727+.034*math.cos(u*math.pi*10)+.006*math.sin(z*4+u*9)
                verts.append((x,y,z));uv.append((u*1.5,z*1.5))
        for j in range(32):
            for i in range(56):
                a=j*57+i;faces.append((a,a+1,a+58,a+57))
        b.mesh(b.path('curtain_panel_'+str(side)),verts,faces,'study_linen',uv)
        for i in range(7):
            xx=center-.14+i*.0467
            b.torus(f'curtain_ring_{side}_{i}',(xx,1.733,2.387),.016,.002,'study_brass',rotation=(0,90,0))
        b.tube('curtain_hem_'+str(side),[(center-.15+i*.00535,1.727+.034*math.cos(i/56*math.pi*10),.325)
                                      for i in range(57)],.0012,'study_stitch')
    # Archive boxes and metal label holders on the shelf crown.
    for i,(yy,w) in enumerate(((-.37,.31),(.03,.29))):
        b.box(f'archive_box_{i}',(-1.015,yy,1.936),(.244,w,.167),'study_linen',.003)
        b.box(f'archive_lid_{i}',(-1.015,yy,2.023),(.252,w+.008,.014),'study_cover_green',.003,collision=False)
        b.box(f'archive_label_frame_{i}',(-.891,yy,1.946),(.002,.094,.035),'study_brass',.001,collision=False)
        b.label(f'archive_label_{i}',('笔记 · NOTES','资料 · ARCHIVE')[i],
                (-.889,yy,1.946),(.085,.026,.0002),rotation=(90,0,90),font_size=100)
    # A small analog clock above the reading corner, with all twelve indices.
    c=(.16,1.853,1.83)
    b.cylinder('study_clock_case',c,.174,.031,'study_brass',rotation=(90,0,0),collision=False)
    b.cylinder('study_clock_face',(.16,1.833,1.83),.159,.004,'study_paper',rotation=(90,0,0),collision=False)
    for i in range(60):
        a=2*math.pi*i/60;r0=.131 if i%5==0 else .142
        b.tube(f'clock_tick_{i}',[(.16+r0*math.sin(a),1.829,1.83+r0*math.cos(a)),
               (.16+.149*math.sin(a),1.829,1.83+.149*math.cos(a))],
               .0016 if i%5==0 else .0006,'study_ink')
    for i,(angle,length) in enumerate(((52,.082),(304,.124))):
        a=math.radians(angle)
        b.tube(f'clock_hand_{i}',[(.16,1.822,1.83),(.16+length*math.sin(a),1.822,1.83+length*math.cos(a))],
               .003 if i==0 else .002,'study_dark_metal')
    b.sphere('clock_hub',(.16,1.819,1.83),(.006,.003,.006),'study_brass')
    # A restrained shelf spotlight gives readable spines without tinting the room.
    for i,y in enumerate((-.38,.38)):
        light(b,f'library_spot_{i}',(-.58,y,2.48),(-1.,y,1.55),(.15,.13),
              9000.,(1.,1.,1.),role='accent',temperature=4700.)


def _study_lighting(stage):
    # Room-local baselines preserve every other family's lighting.
    for name,intensity,temp in (('study_pendant',28000.,4800.),('shelf_wash',9000.,4700.),
                                ('reading_lamp',3000.,3200.),('Sun',2400.,5700.),('Sky',650.,6500.)):
        prim=stage.GetPrimAtPath('/World/Lights/'+name)
        if prim:
            source=UsdLux.LightAPI(prim);source.CreateIntensityAttr(intensity)
            _light_baseline(source,prim.GetCustomDataByKey('scence:lightRole'),temp)
    # Reduce the physically visible opal shader's original excessive emission.
    prim=stage.GetPrimAtPath('/World/Geometry/study_opal_shade/Visual')
    bmat=UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()[0]
    if bmat:
        for p in bmat.GetPrim().GetChildren():
            shader=UsdShade.Shader(p)
            if shader:
                for key in ('emissiveColor','emission_color'):
                    inp=shader.GetInput(key)
                    if inp:inp.Set(Gf.Vec3f(.72,.74,.70))


def _reading_corner(b):
    sources = [_source_asset(b, 'reading_armchair', 'modern_arm_chair_01', (1.78, 1.13, .012), yaw=-15.)]
    # A woven rug belongs only to the reading corner, clear of the robot stand.
    rug = b.box('reading_rug', (1.78, 1.14, .006), (1.26, 1.24, .012), 'study_linen', .005, collision=False)
    _uv_scale(b, rug, 1.)
    for sign in (-1, 1):
        b.box('reading_rug_binding_'+str(sign), (1.78+sign*.61, 1.14, .0125),
              (.015, 1.22, .001), 'study_cover_green', .0004, collision=False)
    sources.append(_source_asset(b, 'reading_plant', 'potted_plant_01', (2.06, .10, 0.), yaw=25.))
    table_root = b.assets_root / 'objects/polyhaven/v1/side_table_tall_01'
    provenance = table_root / 'SOURCE.json'
    table_metadata = load_json(provenance)
    b.dependency(provenance)
    for record in table_metadata['files']:
        b.dependency(table_root / record['path'])
    # Flatten the existing corrected layer so strict dependency traversal does
    # not revisit the publisher's superseded absolute texture paths.
    table_derived = b.assets_root / 'objects/polyhaven/derived/study_v1/side_table_tall_01'
    table_source = table_derived / 'side_table_tall_01.usdc'
    b.dependency(table_derived / 'DERIVATION.json')
    b.asset('reading_side_table', table_source, (.94, 1.38, 0.))
    sources.append({'name': 'reading_side_table', 'asset': str(table_source),
                    'provenance': str(provenance), 'position_m': [.94, 1.38, 0.],
                    'yaw_deg': 0., 'scale': 1., 'role': 'static reading-corner furniture'})
    sources.append(_source_asset(b, 'reading_table_lamp', 'industrial_pipe_lamp',
                                (.94, 1.38, .7611515), version='study_v1'))
    # The scanned bulb has an emissive material. A colocated warm source lights
    # the table and shade; its profile metadata follows the other room lights.
    lamp = b.stage.GetPrimAtPath('/World/Geometry/reading_table_lamp')
    bulb_center = UsdGeom.Xformable(lamp).ComputeLocalToWorldTransform(0).Transform(
        Gf.Vec3d(0., -.061919, .299608))
    bulb = UsdLux.SphereLight.Define(b.stage, '/World/Lights/reading_lamp')
    bulb.CreateRadiusAttr(.014)
    bulb.CreateIntensityAttr(8500.)
    bulb.CreateColorAttr(Gf.Vec3f(1., 1., 1.))
    UsdGeom.Xformable(bulb).AddTranslateOp().Set(bulb_center)
    _light_baseline(bulb, 'accent', 2700.)
    b.tube('reading_lamp_lead', [(.94, 1.45, .785), (.965, 1.54, .755),
           (.965, 1.57, .09), (1.10, 1.82, .015), (2.43, 1.82, .015),
           (2.43, 1.32, .015), (2.455, 1.18, .26)], .0028, 'study_dark_metal')
    return sources


def _pin_licenses(b):
    for source in list(b.dependencies):
        parent = Path(source).parent
        for filename in ('LICENSE.txt', 'SOURCE.json'):
            if (parent/filename).is_file():
                b.dependency(parent/filename)


def build_study_room(stage, assets_root, output_root, *, lighting_profile='nominal'):
    b = Builder(stage, Path(assets_root), Path(output_root))
    b.family = FAMILY
    UsdGeom.Xform.Define(stage, '/World/Robot')
    _materials(b)
    _architecture(b)
    _desk(b)
    shelf = _rear_shelf(b)
    _background_library(b)
    _wall_art(b)
    _room_details(b)
    # Broad daylight enters through the actual window opening. Its source is
    # just inside the glazing, beyond the task area, and points into the room.
    light(b,'study_window_daylight',(1.27,1.76,1.72),(.35,-.40,.85),(1.22,1.10),
          14000.,(1.,1.,1.),role='accent',temperature=6000.)
    sources = _reading_corner(b)
    _study_lighting(stage)
    _pin_licenses(b)
    lighting = apply_lighting_profile(stage, FAMILY, lighting_profile)
    return {
        'schema': 'scence.study_environment.v1', 'family': FAMILY,
        'variant': VARIANT, 'task_independent': True, 'cameras': deepcopy(CAMERAS),
        'robot_placeholder': {'path': '/World/Robot', 'contains_robot': False},
        'layout': {'table_center_xy': TABLE['center_xy'], 'table_size_m': TABLE['size_m'],
                   'table_top_z': TABLE['top_z'], 'room': deepcopy(ROOM_SPEC),
                   'reserved_task_volume': deepcopy(RESERVED_TASK_VOLUME),
                   'robot_floor_reserve': deepcopy(ROBOT_FLOOR_RESERVE),
                   'shelf_bays': deepcopy(SHELF_BAYS)},
        'support_colliders': {'table': [TABLE['collider_path']],
            'floor': ['/World/Geometry/room_floor/Collider'],
            'shelf': sorted({bay['collider_path'] for bay in SHELF_BAYS})},
        'lighting': lighting, 'objects': [], 'geometry': b.records, 'dependencies': b.dependencies,
        'furnishings': {'desk': {'type': 'executive_double_pedestal',
                       'width_depth_height_m': [2., .90, .78], 'top_thickness_m': .08,
                       'owner_side': '-x', 'visitor_side': '+x', 'drawer_count': 6,
                       'construction': 'walnut veneer, flush leather inset, twin floor pedestals and framed modesty panel'},
                       'rear_classification_shelf': shelf, 'imported': sources,
                       'display_books': 'static scenery outside the empty task bays'},
        'ownership': {'environment': 'architecture, fixed open shelves, furniture, empty desk and lighting',
                      'task': 'robot, stand, movable books and stationery, resets, rules and evaluator'},
        'scope': 'Static study environment; empty desk and shelf bays do not establish robot sorting qualification',
    }
