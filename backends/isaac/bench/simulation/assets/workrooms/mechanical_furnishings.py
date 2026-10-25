"""Original, static perimeter dressing for the mechanical workshop.

The idle machines, mounted gear samples and measuring instruments are scenery.
They never supply movable tools or fixtures to an assembly task. All dimensions
are metres; the central robot and task volume remains unoccupied.
"""
from __future__ import annotations

import math

import numpy as np
from pxr import Usd, UsdGeom


def _materials(b):
    for name, color, roughness, metallic in (
            ('machining_blue', (.095, .20, .28), .36, .25),
            ('machining_pale', (.64, .70, .70), .41, .15),
            ('machining_iron', (.11, .14, .15), .59, .55),
            ('machining_brass', (.51, .36, .12), .35, .75),
            ('machining_blueprint', (.025, .095, .17), .87, 0.),
            ('machining_drawing_ink', (.65, .80, .83), .8, 0.),
            ('machining_yellow', (.77, .52, .09), .5, .1)):
        b.material(name, color, roughness, metallic)
    b.material('machining_glazing', (.21, .33, .36), .14, opacity=.22)


def _drill_station(b):
    """A cabinet-supported drill press stays below the existing window sill."""
    x, y = -1.055, .32
    for i, xx in enumerate((x - .225, x + .225)):
        for j, yy in enumerate((y - .34, y + .34)):
            b.cylinder(f'machining_drill_foot_{i}_{j}', (xx, yy, .025),
                       .025, .05, 'rubber')
    b.box('machining_drill_cabinet', (x, y, .3975), (.54, .80, .695),
          'machining_blue', .008)
    b.box('machining_drill_counter', (x, y, .7675), (.58, .84, .045),
          'machining_iron', .005)
    for i, yy in enumerate((y - .195, y + .195)):
        b.box('machining_drill_door_' + str(i), (x + .278, yy, .41),
              (.016, .374, .615), 'machining_blue', .004)
        b.box('machining_drill_pull_' + str(i), (x + .293, yy, .63),
              (.025, .115, .013), 'steel', .003, collision=False)
    b.label('machining_drill_cabinet_tag', 'DRILL STATION',
            (x + .288, y, .145), (.40, .045), rotation=(90, 0, 90), font_size=34)
    # The cabinet top is z=.790. A cast base, column and collar carry the head.
    b.box('machining_drill_base', (x, y + .09, .811), (.37, .47, .042),
          'machining_iron', .012)
    b.cylinder('machining_drill_column_socket', (x - .06, y + .24, .858),
               .063, .052, 'machining_blue')
    b.cylinder('machining_drill_column', (x - .06, y + .24, 1.119),
               .032, .522, 'chrome')
    b.box('machining_drill_head', (x, y + .095, 1.366), (.30, .34, .145),
          'machining_blue', .025)
    b.box('machining_drill_belt_cover', (x - .02, y + .16, 1.465),
          (.32, .46, .055), 'machining_pale', .018)
    b.cylinder('machining_drill_motor', (x - .02, y + .28, 1.28),
               .071, .19, 'machining_iron')
    for i in range(5):
        b.torus('machining_drill_motor_fin_' + str(i), (x - .02, y + .28, 1.21 + .03 * i),
                .071, .0035, 'machining_pale')
    b.cylinder('machining_drill_quill', (x + .025, y - .015, 1.26),
               .027, .095, 'chrome')
    b.lathe('machining_drill_chuck', (x + .025, y - .015, 1.17),
            [(0, 0), (.013, 0), (.027, .018), (.027, .05), (.020, .06), (0, .06)],
            'machining_iron')
    b.cylinder('machining_drill_bit', (x + .025, y - .015, 1.143),
               .0035, .054, 'steel', collision=False)
    b.cylinder('machining_drill_table_collar', (x - .06, y + .24, 1.015),
               .048, .065, 'machining_blue')
    b.box('machining_drill_table_arm', (x - .02, y + .115, 1.016),
          (.075, .26, .04), 'machining_blue', .006)
    b.box('machining_drill_table', (x + .015, y - .005, 1.045),
          (.29, .25, .025), 'steel', .006)
    for i, dx in enumerate((-.08, 0., .08)):
        b.box('machining_drill_table_slot_' + str(i), (x + .015 + dx, y - .005, 1.0578),
              (.008, .18, .0006), 'black', .001, collision=False)
    # Feed spokes are fixed to the head, not loose tools or actuated controls.
    hub = (x + .165, y + .10, 1.35)
    b.cylinder('machining_drill_feed_hub', hub, .026, .035, 'chrome',
               rotation=(0, 90, 0), collision=False)
    for i, angle in enumerate((20., 140., 260.)):
        radians = math.radians(angle)
        tip = (hub[0], hub[1] + .105 * math.cos(radians), hub[2] + .105 * math.sin(radians))
        b.tube('machining_drill_feed_spoke_' + str(i), [hub, tip], .005, 'steel')
        b.sphere('machining_drill_feed_grip_' + str(i), tip, .016, 'black')
    b.box('machining_drill_switch_box', (x + .162, y - .035, 1.365),
          (.023, .064, .08), 'machining_yellow', .004, collision=False)
    b.cylinder('machining_drill_stop', (x + .18, y - .035, 1.38),
               .014, .018, 'red', rotation=(0, 90, 0), collision=False)
    for i, xx in enumerate((x - .135, x + .135)):
        for j, yy in enumerate((y - .08, y + .26)):
            b.cylinder(f'machining_drill_anchor_{i}_{j}', (xx, yy, .837),
                       .009, .010, 'steel', collision=False)


def _enclosed_mill(b):
    """Closed compact mill, facing -X, with visible interior and cabinet feet."""
    x, y = 2.025, .36
    for i, xx in enumerate((x - .32, x + .32)):
        for j, yy in enumerate((y - .37, y + .37)):
            b.cylinder(f'machining_mill_foot_{i}_{j}', (xx, yy, .03),
                       .033, .06, 'rubber')
    b.box('machining_mill_base', (x, y, .405), (.79, .87, .69), 'machining_blue', .012)
    b.box('machining_mill_catch_tray', (x, y, .7725), (.83, .91, .045), 'steel', .006)
    b.box('machining_mill_back', (x + .393, y, 1.215), (.034, .91, .84),
          'machining_pale', .006)
    for i, yy in enumerate((y - .438, y + .438)):
        b.box('machining_mill_side_' + str(i), (x, yy, 1.215), (.82, .034, .84),
              'machining_pale', .006)
    b.box('machining_mill_roof', (x, y, 1.6575), (.83, .91, .045),
          'machining_blue', .007)
    # Internal bed and spindle are visible behind the closed glazing.
    b.box('machining_mill_bed', (x, y - .08, .845), (.57, .56, .10),
          'machining_iron', .007)
    b.box('machining_mill_table', (x - .04, y - .08, .92), (.44, .51, .05), 'steel', .004)
    for i in range(5):
        b.box('machining_mill_tslot_' + str(i), (x - .04, y - .28 + .1 * i, .9456),
              (.36, .009, .0012), 'black', .001, collision=False)
    b.box('machining_mill_column', (x + .275, y - .08, 1.17), (.17, .26, .61),
          'machining_iron', .007)
    b.box('machining_mill_head', (x + .125, y - .08, 1.385), (.27, .25, .22),
          'machining_blue', .012)
    b.cylinder('machining_mill_spindle', (x + .045, y - .08, 1.23),
               .035, .09, 'steel')
    # Front window leaves a separate control strip on its right side.
    front = x - .42
    for i, yy in enumerate((y - .42, y + .20)):
        b.box('machining_mill_window_stile_' + str(i), (front, yy, 1.225),
              (.035, .035, .78), 'machining_blue', .004)
    for i, zz in enumerate((.85, 1.60)):
        b.box('machining_mill_window_rail_' + str(i), (front, y - .11, zz),
              (.035, .655, .03), 'machining_blue', .003)
    b.box('machining_mill_window', (front, y - .11, 1.225), (.006, .585, .715),
          'machining_glazing', .001, collision=False)
    b.box('machining_mill_window_handle', (front - .023, y + .145, 1.20),
          (.035, .017, .17), 'steel', .004, collision=False)
    b.box('machining_mill_control_panel', (front, y + .325, 1.225),
          (.035, .22, .78), 'machining_pale', .004)
    b.box('machining_mill_screen_bezel', (front - .024, y + .325, 1.405),
          (.016, .175, .155), 'black', .004, collision=False)
    b.label('machining_mill_screen', 'IDLE\nX  0.00\nY  0.00',
            (front - .033, y + .325, 1.405), (.146, .126),
            rotation=(90, 0, -90), font_size=34)
    for i, yy in enumerate((y + .275, y + .375)):
        b.cylinder('machining_mill_control_' + str(i), (front - .025, yy, 1.265),
                   .017, .018, 'green' if i == 0 else 'black',
                   rotation=(0, 90, 0), collision=False)
    b.cylinder('machining_mill_stop_collar', (front - .025, y + .325, 1.10),
               .033, .016, 'machining_yellow', rotation=(0, 90, 0), collision=False)
    b.cylinder('machining_mill_stop_button', (front - .039, y + .325, 1.10),
               .022, .025, 'red', rotation=(0, 90, 0), collision=False)
    b.label('machining_mill_title', 'COMPACT MILL  /  ENCLOSED',
            (front - .019, y, 1.6575), (.69, .033), rotation=(90, 0, -90), font_size=32)
    for i, yy in enumerate((y - .21, y + .21)):
        b.box('machining_mill_service_door_' + str(i), (front + .018, yy, .405),
              (.016, .405, .58), 'machining_blue', .004)
        b.cylinder('machining_mill_service_lock_' + str(i), (front + .008, yy, .64),
                   .009, .006, 'steel', rotation=(0, 90, 0), collision=False)
    for i in range(6):
        b.box('machining_mill_vent_' + str(i), (front + .008, y - .21, .25 + .028 * i),
              (.001, .25, .008), 'black', .001, collision=False)


def _gear_sample(b, name, x, z, radius, teeth, material):
    """An original extruded spur silhouette with an actual central opening."""
    count = teeth * 4
    vertices = []
    # Each depth layer has its outer contour followed by the bore contour.
    for y in (1.927, 1.915):
        for inner in (False, True):
            for i in range(count):
                angle = math.tau * i / count
                r = .013 if inner else radius * (1. if i % 4 in (1, 2) else .84)
                vertices.append((x + r * math.cos(angle), y, z + r * math.sin(angle)))
    faces = []
    for i in range(count):
        j = (i + 1) % count
        faces.extend(((i, j, count + j, count + i),
                      (2 * count + i, 3 * count + i, 3 * count + j, 2 * count + j),
                      (i, 2 * count + i, 2 * count + j, j),
                      (count + i, count + j, 3 * count + j, 3 * count + i)))
    b.mesh(b.path(name), vertices, [face[::-1] for face in faces], material)
    b.cylinder(name + '_mount_pin', (x, 1.924, z), .009, .033,
               'black', rotation=(90, 0, 0), collision=False)


def _reference_display(b):
    # The sample board sits below the hanging spanners, above the service top.
    b.box('machining_reference_back', (-.47, 1.953, 1.30), (.90, .025, .28),
          'machining_iron', .004)
    for i, (x, radius, teeth) in enumerate(((-.73, .074, 16), (-.47, .058, 12), (-.21, .074, 16))):
        _gear_sample(b, 'machining_reference_gear_' + str(i), x, 1.285, radius,
                     teeth, 'steel' if i != 1 else 'machining_brass')
    b.label('machining_reference_caption', 'GEAR SAMPLES  /  FIXED DISPLAY',
            (-.47, 1.939, 1.405), (.76, .033), rotation=(90, 0, 0), font_size=34)
    for i, xx in enumerate((-.895, -.045)):
        for j, zz in enumerate((1.182, 1.418)):
            b.cylinder(f'machining_reference_spacer_{i}_{j}', (xx, 1.976, zz),
                       .009, .032, 'machining_iron', rotation=(90, 0, 0), collision=False)
            b.cylinder(f'machining_reference_fastener_{i}_{j}', (xx, 1.936, zz),
                       .005, .013, 'steel', rotation=(90, 0, 0), collision=False)
    # This dimensioned section is above the existing conduit, not over a window.
    b.box('machining_drawing_frame', (-.04, 2.025, 2.575), (1.14, .028, .27),
          'steel', .003, collision=False)
    for i, xx in enumerate((-.53, .45)):
        b.cylinder('machining_drawing_wall_mount_' + str(i), (xx, 2.045, 2.575),
                   .010, .025, 'steel', rotation=(90, 0, 0), collision=False)
    b.box('machining_drawing_paper', (-.04, 2.009, 2.575), (1.105, .002, .238),
          'machining_blueprint', .0002, collision=False)
    b.label('machining_drawing_title', 'MACHINING  /  ASSEMBLY & INSPECTION',
            (.13, 2.007, 2.635), (.68, .039), rotation=(90, 0, 0), font_size=34)
    for i, radius in enumerate((.069, .041, .015)):
        b.torus('machining_drawing_section_' + str(i), (-.45, 2.006, 2.573),
                radius, .0009, 'machining_drawing_ink', rotation=(90, 0, 0))
    for axis in (0, 2):
        size = [.001, .0007, .001]
        size[axis] = .18
        b.box('machining_drawing_centerline_' + str(axis), (-.45, 2.005, 2.573),
              size, 'machining_drawing_ink', 0, collision=False)
    for i, zz in enumerate((2.538, 2.585)):
        b.box('machining_drawing_shaft_edge_' + str(i), (-.10, 2.005, zz),
              (.25, .0007, .0012), 'machining_drawing_ink', 0, collision=False)
    for i, xx in enumerate((-.225, .025)):
        b.box('machining_drawing_shaft_end_' + str(i), (xx, 2.005, 2.5615),
              (.0012, .0007, .048), 'machining_drawing_ink', 0, collision=False)
    b.box('machining_drawing_dimension', (-.10, 2.005, 2.515),
          (.29, .0007, .001), 'machining_drawing_ink', 0, collision=False)
    b.label('machining_inspection_caption', 'DATUM A\nCHECK BORE / FACE\nDEBURR BEFORE ASSEMBLY',
            (.295, 2.007, 2.545), (.35, .09), rotation=(90, 0, 0), font_size=30)


def _inspection_tray(b):
    # A shallow fixed tray rests on the perimeter service counter (top z=.916).
    x, y, top = .48, 1.865, .916
    b.box('machining_inspection_tray_base', (x, y, top + .004), (.32, .19, .008),
          'machining_iron', .003)
    for i, yy in enumerate((y - .091, y + .091)):
        b.box('machining_inspection_tray_lip_' + str(i), (x, yy, top + .017),
              (.32, .008, .026), 'machining_iron', .002)
    for i, xx in enumerate((x - .156, x + .156)):
        b.box('machining_inspection_tray_end_' + str(i), (xx, y, top + .017),
              (.008, .174, .026), 'machining_iron', .002)
    # A caliper laid flat in its fitted insert is decorative and non-graspable.
    b.box('machining_caliper_insert', (x, y, top + .011), (.296, .164, .006),
          'rubber', .004, collision=False)
    b.box('machining_caliper_rule', (x - .01, y - .015, top + .018),
          (.245, .014, .008), 'steel', .001, collision=False)
    for i, xx in enumerate((x - .115, x - .055)):
        b.box('machining_caliper_jaw_' + str(i), (xx, y + .009, top + .018),
              (.012, .062, .008), 'steel', .001, collision=False)
    b.box('machining_caliper_slider', (x - .055, y - .018, top + .024),
          (.049, .037, .012), 'machining_iron', .003, collision=False)
    b.box('machining_caliper_display', (x - .055, y - .018, top + .0305),
          (.031, .018, .001), 'machining_pale', .001, collision=False)
    for i in range(12):
        b.box('machining_caliper_tick_' + str(i), (x - .025 + i * .012, y - .015, top + .0223),
              (.0007, .009 if i % 5 == 0 else .005, .0006), 'black', 0, collision=False)
    b.label('machining_inspection_tray_tag', 'METROLOGY / DISPLAY',
            (x, y - .0955, top + .017), (.26, .018), rotation=(90, 0, 0), font_size=28)


def build_furnishings(b):
    """Add static groups and return actual USD geometry paths and metre bounds.

    Call after ``build_room`` and ``build_interior``. All authored names begin
    with ``machining_``. Nothing here adds task objects, robots, or articulations.
    """
    _materials(b)
    for filename in ('NotoSansCJKsc-Regular.otf', 'LICENSE.txt', 'SOURCE.json'):
        b.dependency(b.assets_root / 'fonts/noto_cjk_sc/v1' / filename)
    groups = []
    for name, recipe, purpose, role in (
            ('drill_station', _drill_station, 'idle cabinet-supported drill press', 'floor_furniture'),
            ('enclosed_mill', _enclosed_mill, 'closed idle compact mill with viewing window', 'floor_furniture'),
            ('reference_display', _reference_display, 'wall-mounted gear samples and engineering section', 'wall_fixture'),
            ('inspection_tray', _inspection_tray, 'fixed metrology display on the service counter', 'supported_decor')):
        before = set(b._names)
        recipe(b)
        paths = ['/World/Geometry/' + entry for entry in sorted(set(b._names) - before)]
        cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ['default', 'render'])
        ranges = []
        for path in paths:
            prim = b.stage.GetPrimAtPath(path)
            prim.SetCustomDataByKey('scence:layoutRole', role)
            prim.SetCustomDataByKey('workroom:interactionRole', 'static_scenery')
            ranges.append(cache.ComputeWorldBound(prim).ComputeAlignedRange())
        groups.append({'name': name, 'purpose': purpose, 'role': role,
                       'geometry_paths': paths,
                       'bounds_m': [np.min([list(box.GetMin()) for box in ranges], axis=0).tolist(),
                                    np.max([list(box.GetMax()) for box in ranges], axis=0).tolist()]})
        b.records.append({'kind': 'interior_group', 'name': 'machining_' + name,
                          'role': role, 'prim_paths': paths})
    return {'groups': groups, 'static_perimeter_only': True,
            'protected_work_volume_m': [[-.60, -.70, 0.], [1.30, .70, 1.90]],
            'provenance': 'Original procedural geometry; existing shared materials and font dependencies remain with the builder.',
            'scope': 'CPU-authored static geometry and actual USD bounds; native images and physics require separate validation.'}
