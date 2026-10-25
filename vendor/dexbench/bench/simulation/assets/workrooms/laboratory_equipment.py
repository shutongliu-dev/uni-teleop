"""Static laboratory instruments, reagent stations and mobile storage.

Dimensions are in metres.  These are perimeter room furnishings, not movable
task objects.  Large housings have static collision; labels, handles, controls
and small stored glassware are visual details.
"""

import math

from .interiors import _Frame, _ph


def _materials(b):
    for name, color, roughness, metallic in (
        ("lab_eq_enamel", (.78, .82, .81), .31, .08),
        ("lab_eq_teal", (.055, .25, .28), .34, .12),
        ("lab_eq_pale_teal", (.34, .56, .57), .42, 0),
        ("lab_eq_graphite", (.045, .061, .065), .45, .16),
        ("lab_eq_display", (.009, .035, .045), .16, .1),
        ("lab_eq_led", (.26, .82, .61), .25, 0),
        ("lab_eq_blue_cap", (.035, .18, .43), .37, 0),
        ("lab_eq_lilac_cap", (.39, .26, .50), .42, 0),
        ("lab_eq_yellow_cap", (.83, .64, .15), .43, 0),
        ("lab_eq_stainless", (.53, .59, .60), .26, .88),
        ("lab_eq_cable", (.035, .043, .045), .72, 0),
        ("lab_eq_translucent", (.69, .80, .83), .24, 0),
    ):
        b.material(name, color, roughness, metallic)
    b.material("lab_eq_amber", (.24, .095, .028), .20, opacity=.82)
    b.material("lab_eq_solution", (.36, .70, .75), .18, opacity=.74)


def reagent_bottle(b, name, position, *, radius=.032, height=.14,
                   label="BUFFER\n500 mL", cap="lab_eq_blue_cap",
                   amber=False, collision=False):
    """Closed bottle standing exactly on position.z, with shoulder and neck."""
    _materials(b)
    x, y, z = position
    r, h = radius, height
    neck = r * .59
    wall = min(.0018, r * .07)
    profile = [
        (0, 0), (r * .84, 0), (r, .005), (r, h * .67),
        (r * .88, h * .75), (neck, h * .85), (neck, h * .94),
        (neck - wall, h * .94), (neck - wall, h * .85),
        (r - wall, h * .67), (r - wall, .0035), (0, .0035),
    ]
    b.lathe(name + "_glass", position, profile,
            "lab_eq_amber" if amber else "glass", collision=collision)
    if not amber:
        b.cylinder(name + "_solution", (x, y, z + h * .255),
                   r - wall * 1.5, h * .46, "lab_eq_solution", collision=False)
    b.cylinder(name + "_cap", (x, y, z + h * .965), neck + .003,
               h * .07, cap, collision=False)
    for i in range(16):
        a = i * 2 * math.pi / 16
        b.cylinder(name + "_cap_grip_%02d" % i,
                   (x + (neck + .002) * math.cos(a),
                    y + (neck + .002) * math.sin(a), z + h * .965),
                   .00065, h * .052, cap, collision=False)
    b.label(name + "_label", label, (x, y - r - .0005, z + h * .42),
            (r * 1.45, h * .35, .0003), rotation=(90, 0, 0), font_size=42)
    return dict(name=name, kind="reagent_bottle", base=list(position),
                height=h, radius=r, static=True, collision=collision)


def glass_beaker(b, name, position, *, radius=.036, height=.095,
                 filled=False, collision=False):
    """Hollow open beaker with a real bottom, rim and printed graduations."""
    _materials(b)
    x, y, z = position
    r, h = radius, height
    b.lathe(name + "_glass", position,
            [(0, 0), (r * .90, 0), (r, .005), (r, h - .004),
             (r * 1.035, h), (r - .0018, h),
             (r - .0018, .003), (0, .003)],
            "glass", collision=collision)
    if filled:
        b.cylinder(name + "_liquid", (x, y, z + h * .28 + .003),
                   r - .0025, h * .56, "lab_eq_solution", collision=False)
    for i in range(4):
        zz = z + h * (.25 + i * .15)
        b.box(name + "_graduation_%d" % i, (x, y - r - .0005, zz),
              (r * (.43 if i % 2 else .65), .0005, .0012), "white",
              bevel=0, collision=False)
    b.label(name + "_capacity", "100 mL", (x, y - r - .0008, z + h * .84),
            (r * 1.15, h * .12, .0003), rotation=(90, 0, 0), font_size=32)


def test_tube_rack(b, name, position, *, columns=6, rows=2, pitch=.032):
    """Open two-level rack with individual tube sockets and capped tubes."""
    _materials(b)
    x, y, z = position
    w, d = (columns + .5) * pitch, (rows + .7) * pitch
    b.box(name + "_foot", (x, y, z + .007), (w, d, .014),
          "lab_eq_pale_teal", bevel=.003, collision=False)
    for side in (-1, 1):
        b.box(name + "_end_%d" % side,
              (x + side * (w / 2 - .004), y, z + .043),
              (.008, d, .072), "lab_eq_pale_teal", bevel=.002, collision=False)
    # Individual narrow rails leave the visible rack top genuinely open.
    for row in range(rows + 1):
        yy = y + (row - rows / 2) * pitch
        b.box(name + "_rail_%d" % row, (x, yy, z + .062),
              (w, .006, .008), "lab_eq_pale_teal", bevel=.002, collision=False)
    for col in range(columns + 1):
        xx = x + (col - columns / 2) * pitch
        b.box(name + "_divider_%d" % col, (xx, y, z + .062),
              (.005, rows * pitch, .008), "lab_eq_pale_teal",
              bevel=.001, collision=False)
    for row in range(rows):
        for col in range(columns):
            xx = x + (col - (columns - 1) / 2) * pitch
            yy = y + (row - (rows - 1) / 2) * pitch
            stem = name + "_tube_%d_%d" % (row, col)
            b.lathe(stem, (xx, yy, z + .014),
                    [(0, 0), (.006, .003), (.008, .01),
                     (.008, .091), (.0067, .091), (.0067, .012),
                     (.005, .005), (0, .003)], "glass", collision=False)
            b.cylinder(stem + "_sample", (xx, yy, z + .035),
                       .0059, .022 + .004 * (col % 3),
                       "lab_eq_solution", collision=False)
            b.cylinder(stem + "_cap", (xx, yy, z + .109),
                       .0095, .008,
                       "lab_eq_lilac_cap" if row else "lab_eq_blue_cap",
                       collision=False)
    b.label(name + "_tag", "SAMPLES 01 - %02d" % (rows * columns),
            (x, y - d / 2 - .0005, z + .009), (w * .8, .012, .0003),
            rotation=(90, 0, 0), font_size=30)
    return dict(name=name, kind="test_tube_rack", base=list(position),
                size=[w, d, .113], tubes=rows * columns,
                static=True, collision=False)


def pipette_stand(b, name, position):
    """Four hanging pipettes with coloured plungers and distinct tip shafts."""
    _materials(b)
    x, y, z = position
    b.box(name + "_base", (x, y, z + .009), (.24, .14, .018),
          "lab_eq_enamel", bevel=.006, collision=False)
    for side in (-1, 1):
        b.box(name + "_post_%d" % side,
              (x + side * .108, y + .045, z + .136),
              (.013, .017, .255), "lab_eq_stainless", bevel=.003, collision=False)
    b.box(name + "_beam", (x, y + .045, z + .262), (.24, .038, .018),
          "lab_eq_teal", bevel=.003, collision=False)
    caps = ["lab_eq_yellow_cap", "lab_eq_blue_cap",
            "lab_eq_lilac_cap", "lab_eq_teal"]
    for i in range(4):
        xx = x + (i - 1.5) * .055
        b.box(name + "_hook_%d" % i, (xx, y + .008, z + .258),
              (.035, .067, .008), "lab_eq_teal", bevel=.002, collision=False)
        stem = name + "_pipette_%d" % i
        b.lathe(stem, (xx, y - .012, z + .044),
                [(0, 0), (.0023, 0), (.005, .045), (.005, .093),
                 (.011, .105), (.011, .190), (.0085, .210),
                 (.006, .210), (.006, .233), (0, .233)],
                "lab_eq_enamel", collision=False)
        b.cylinder(stem + "_plunger", (xx, y - .012, z + .284),
                   .0105, .014, caps[i], collision=False)
        b.box(stem + "_window", (xx, y - .0233, z + .179),
              (.010, .0016, .028), "lab_eq_display",
              bevel=.001, collision=False)
        b.box(stem + "_ejector", (xx + .010, y - .012, z + .113),
              (.003, .004, .09), "lab_eq_stainless",
              bevel=.0005, collision=False)
    b.label(name + "_label", "MICROPIPETTES", (x, y - .0705, z + .010),
            (.19, .012, .0003), rotation=(90, 0, 0), font_size=36)
    return dict(name=name, kind="pipette_stand", base=list(position),
                size=[.24, .14, .291], pipettes=4,
                static=True, collision=False)


def _feet(f, name, width, depth):
    for ix in (-1, 1):
        for iy in (-1, 1):
            f.cylinder(name + "_foot_%d_%d" % (ix, iy),
                       (ix * (width / 2 - .037),
                        iy * (depth / 2 - .037), .008),
                       .018, .016, "rubber")


def _centrifuge(b, position):
    name = "lab_centrifuge"
    f = _Frame(b, position)
    _feet(f, name, .50, .44)
    f.box(name + "_body", (0, 0, .141), (.50, .44, .25),
          "lab_eq_enamel", bevel=.026)
    f.box(name + "_dark_base", (0, -.001, .034), (.489, .431, .028),
          "lab_eq_graphite", bevel=.010, collision=False)
    f.box(name + "_lid_seal", (0, .012, .270), (.474, .402, .012),
          "lab_eq_graphite", bevel=.021, collision=False)
    f.box(name + "_lid", (0, .012, .290), (.474, .402, .028),
          "lab_eq_enamel", bevel=.018)
    f.cylinder(name + "_inspection_rim", (0, .012, .305),
               .066, .004, "lab_eq_stainless", collision=False)
    f.cylinder(name + "_inspection_glass", (0, .012, .3075),
               .058, .002, "lab_eq_display", collision=False)
    for x in (-.14, .14):
        f.box(name + "_hinge_" + str(x), (x, .208, .278),
              (.052, .029, .03), "lab_eq_stainless",
              bevel=.005, collision=False)
    f.box(name + "_latch", (0, -.211, .266), (.083, .029, .022),
          "lab_eq_graphite", bevel=.004, collision=False)
    f.box(name + "_panel", (0, -.2208, .120), (.408, .005, .106),
          "lab_eq_teal", bevel=.008, collision=False)
    f.box(name + "_screen", (-.074, -.224, .132), (.174, .003, .052),
          "lab_eq_display", bevel=.004, collision=False)
    f.label(name + "_readout", "4500 rpm    05:00", (-.074, -.226, .133),
            (.158, .026, .0003), rotation=(90, 0, 0), font_size=38)
    for i, x in enumerate((.066, .118, .170)):
        f.cylinder(name + "_button_%d" % i, (x, -.228, .124),
                   .013, .008,
                   "lab_eq_led" if i == 2 else "lab_eq_enamel",
                   rotation=(90, 0, 0), collision=False)
    f.label(name + "_nameplate", "BENCHTOP CENTRIFUGE", (0, -.223, .201),
            (.32, .025, .0003), rotation=(90, 0, 0), font_size=40)
    for i in range(8):
        f.box(name + "_vent_%02d" % i, (.2505, -.10 + i * .029, .111),
              (.001, .015, .070), "lab_eq_graphite",
              bevel=.001, collision=False)
    f.tube(name + "_power_cord",
           [(.19, .22, .09), (.21, .255, .065), (.26, .275, .025),
            (.31, .280, .012)], .0035, "lab_eq_cable")
    return dict(name=name, kind="benchtop_centrifuge", base=list(position),
                housing_size=[.50, .44, .309], static=True,
                support="rear_instrument_counter", collision="housing_and_feet")


def _analytical_balance(b, position):
    name = "lab_analytical_balance"
    f = _Frame(b, position)
    _feet(f, name, .36, .42)
    f.box(name + "_base", (0, 0, .0575), (.36, .42, .083),
          "lab_eq_enamel", bevel=.013)
    f.box(name + "_control_panel", (0, -.212, .059), (.30, .005, .055),
          "lab_eq_teal", bevel=.005, collision=False)
    f.box(name + "_display", (-.036, -.216, .063), (.176, .003, .035),
          "lab_eq_display", bevel=.003, collision=False)
    f.label(name + "_readout", "0.0000 g", (-.036, -.218, .063),
            (.155, .022, .0003), rotation=(90, 0, 0), font_size=44)
    for i, x in enumerate((.090, .123)):
        f.cylinder(name + "_key_%d" % i, (x, -.218, .061),
                   .009, .006, "lab_eq_enamel",
                   rotation=(90, 0, 0), collision=False)
    f.box(name + "_chamber_floor", (0, .030, .103), (.322, .322, .008),
          "lab_eq_stainless", bevel=.003, collision=False)
    f.cylinder(name + "_pan_stem", (0, .030, .114), .021, .014,
               "lab_eq_stainless", collision=False)
    f.cylinder(name + "_weighing_pan", (0, .030, .124), .087, .006,
               "lab_eq_stainless", collision=False)
    for side in (-1, 1):
        f.box(name + "_side_glass_%d" % side,
              (side * .159, .030, .258), (.003, .318, .294),
              "glass", bevel=0)
        for yy in (-.129, .189):
            f.box(name + "_upright_%d_%s" % (side, yy),
                  (side * .159, yy, .257), (.009, .009, .305),
                  "lab_eq_stainless", bevel=.001, collision=False)
    for side in (-1, 1):
        f.box(name + "_face_glass_%d" % side,
              (0, .030 + side * .158, .258), (.310, .003, .294),
              "glass", bevel=0)
    f.box(name + "_roof", (0, .030, .411), (.328, .328, .008),
          "glass", bevel=.001)
    for z in (.109, .408):
        for side in (-1, 1):
            f.box(name + "_rail_%s_%d" % (z, side),
                  (0, .030 + side * .161, z), (.328, .009, .010),
                  "lab_eq_stainless", bevel=.001, collision=False)
    f.box(name + "_slide_handle", (.109, -.134, .243),
          (.016, .012, .055), "lab_eq_graphite", bevel=.003, collision=False)
    f.label(name + "_nameplate", "ANALYTICAL BALANCE", (0, -.211, .093),
            (.236, .013, .0003), rotation=(90, 0, 0), font_size=40)
    return dict(name=name, kind="analytical_balance", base=list(position),
                size=[.36, .431, .415], static=True,
                support="rear_instrument_counter",
                collision="base_feet_and_draft_shield")


def _refrigerator(b, position, yaw=-90):
    name = "lab_refrigerator"
    f = _Frame(b, position, yaw)
    for ix in (-1, 1):
        for iy in (-1, 1):
            f.cylinder(name + "_foot_%d_%d" % (ix, iy),
                       (ix * .28, iy * .27, .035), .025, .070, "rubber")
    f.box(name + "_cabinet", (0, 0, .960), (.72, .70, 1.78),
          "lab_eq_enamel", bevel=.018)
    f.box(name + "_gasket", (0, -.356, 1.120), (.698, .012, 1.438),
          "lab_eq_graphite", bevel=.009, collision=False)
    f.box(name + "_door", (0, -.380, 1.120), (.684, .038, 1.408),
          "lab_eq_enamel", bevel=.012)
    f.box(name + "_lower_vent_frame", (0, -.355, .235), (.668, .020, .225),
          "lab_eq_graphite", bevel=.004, collision=False)
    for i in range(9):
        f.box(name + "_vent_louvre_%02d" % i, (0, -.368, .143 + .023 * i),
              (.632, .013, .011), "lab_eq_stainless",
              bevel=.002, collision=False)
    for z in (.59, 1.58):
        f.box(name + "_hinge_" + str(z), (-.340, -.389, z),
              (.026, .031, .090), "lab_eq_stainless",
              bevel=.004, collision=False)
    f.tube(name + "_door_handle",
           [(.235, -.404, .91), (.235, -.445, .94),
            (.235, -.445, 1.27), (.235, -.404, 1.30)],
           .013, "lab_eq_stainless")
    f.box(name + "_controller", (0, -.403, 1.665), (.298, .008, .082),
          "lab_eq_teal", bevel=.006, collision=False)
    f.box(name + "_display", (-.045, -.409, 1.665), (.156, .004, .049),
          "lab_eq_display", bevel=.003, collision=False)
    f.label(name + "_temperature", "+ 4.0 C", (-.045, -.412, 1.665),
            (.14, .028, .0003), rotation=(90, 0, 0), font_size=48)
    for i, x in enumerate((.063, .102)):
        f.cylinder(name + "_button_%d" % i, (x, -.412, 1.664),
                   .010, .006, "lab_eq_enamel",
                   rotation=(90, 0, 0), collision=False)
    f.cylinder(name + "_lock", (.235, -.403, 1.40), .013, .008,
               "lab_eq_stainless", rotation=(90, 0, 0), collision=False)
    f.box(name + "_keyslot", (.235, -.408, 1.40), (.003, .001, .013),
          "lab_eq_graphite", bevel=0, collision=False)
    f.label(name + "_lab_only", "LABORATORY STORAGE\n2 - 8 C  /  NO FOOD",
            (-.040, -.4005, 1.430), (.32, .090, .0003),
            rotation=(90, 0, 0), font_size=48)
    f.label(name + "_inventory", "SAMPLE INVENTORY\nA   BUFFER STOCK\nB   CONTROL SAMPLES\nC   REFERENCE VIALS",
            (-.075, -.4005, 1.100), (.29, .22, .0003),
            rotation=(90, 0, 0), font_size=46)
    f.label(name + "_unit_id", "COLD STORAGE  01", (0, -.4005, .495),
            (.30, .036, .0003), rotation=(90, 0, 0), font_size=44)
    return dict(name=name, kind="laboratory_refrigerator", base=list(position),
                local_size=[.72, .808, 1.85], yaw_degrees=yaw, static=True,
                support="floor", collision="cabinet_door_and_feet")


def _sample_box(b, name, position, size=(.24, .18, .105)):
    x, y, z = position
    w, d, h = size
    b.box(name + "_base", (x, y, z + h * .46), (w, d, h * .92),
          "lab_eq_pale_teal", bevel=.006, collision=False)
    b.box(name + "_lid", (x, y, z + h * .96), (w + .006, d + .006, h * .08),
          "lab_eq_enamel", bevel=.003, collision=False)
    for side in (-1, 1):
        b.box(name + "_latch_%d" % side,
              (x + side * w * .32, y - d / 2 - .004, z + h * .80),
              (.026, .010, .031), "lab_eq_graphite",
              bevel=.003, collision=False)
    b.label(name + "_label", "PREPARED SAMPLES", (x, y - d / 2 - .0005, z + h * .42),
            (w * .84, h * .27, .0003), rotation=(90, 0, 0), font_size=40)


def _instrument_cart(b, position):
    name = "lab_instrument_cart"
    f = _Frame(b, position)
    for ix in (-1, 1):
        for iy in (-1, 1):
            x, y = ix * .264, iy * .19
            stem = name + "_caster_%d_%d" % (ix, iy)
            f.cylinder(stem + "_wheel", (x, y, .042), .042, .030,
                       "rubber", rotation=(90, 0, 0))
            f.cylinder(stem + "_hub", (x, y - .016, .042), .018, .003,
                       "lab_eq_stainless", rotation=(90, 0, 0), collision=False)
            f.box(stem + "_fork", (x, y, .085), (.046, .038, .032),
                  "lab_eq_stainless", bevel=.004, collision=False)
            f.cylinder(stem + "_swivel", (x, y, .111), .024, .020,
                       "lab_eq_stainless", collision=False)
            f.cylinder(name + "_post_%d_%d" % (ix, iy), (x, y, .47),
                       .013, .70, "lab_eq_stainless")
    for row, top in enumerate((.18, .47, .80)):
        f.box(name + "_shelf_%d" % row, (0, 0, top - .010),
              (.60, .44, .020), "lab_eq_stainless", bevel=.006)
        for side in (-1, 1):
            f.box(name + "_side_lip_%d_%d" % (row, side),
                  (side * .297, 0, top + .017), (.006, .44, .034),
                  "lab_eq_stainless", bevel=.002, collision=False)
        f.box(name + "_rear_lip_%d" % row, (0, .217, top + .017),
              (.60, .006, .034), "lab_eq_stainless", bevel=.002, collision=False)
    f.tube(name + "_push_handle",
           [(-.264, .19, .80), (-.264, .19, .91),
            (.264, .19, .91), (.264, .19, .80)],
           .013, "lab_eq_stainless")
    f.tube(name + "_handle_grip", [(-.15, .19, .91), (.15, .19, .91)],
           .016, "lab_eq_graphite")
    _sample_box(f, name + "_sample_case", (-.135, .010, .80), (.235, .24, .12))
    glass_beaker(f, name + "_beaker_large", (.125, -.095, .80),
                 radius=.045, height=.12)
    glass_beaker(f, name + "_beaker_small", (.220, -.075, .80),
                 radius=.031, height=.085)
    reagent_bottle(f, name + "_wash_bottle", (.155, .105, .80),
                   radius=.035, height=.19, label="DI WATER", cap="lab_eq_teal")
    for i in range(3):
        f.box(name + "_tray_stack_%d" % i, (-.13, .02, .480 + .025 * i),
              (.23, .29, .020), "lab_eq_enamel", bevel=.007, collision=False)
    _sample_box(f, name + "_sterile_case", (.145, .018, .47), (.22, .29, .13))
    f.box(name + "_waste_carboy", (-.11, .015, .300), (.18, .23, .24),
          "lab_eq_translucent", bevel=.022, collision=False)
    f.cylinder(name + "_carboy_neck", (-.11, .015, .428), .027, .016,
               "lab_eq_translucent", collision=False)
    f.cylinder(name + "_carboy_cap", (-.11, .015, .441), .030, .010,
               "lab_eq_blue_cap", collision=False)
    f.label(name + "_carboy_label", "RINSE WATER\n2 L",
            (-.11, -.101, .305), (.12, .075, .0003),
            rotation=(90, 0, 0), font_size=40)
    f.box(name + "_wipe_pack", (.146, .013, .235), (.22, .27, .11),
          "lab_eq_enamel", bevel=.012, collision=False)
    f.label(name + "_wipe_pack_label", "LINT-FREE WIPES",
            (.146, -.123, .239), (.18, .035, .0003),
            rotation=(90, 0, 0), font_size=40)
    return dict(name=name, kind="three_shelf_instrument_cart", base=list(position),
                size=[.62, .48, .99], shelf_tops=[.18, .47, .80],
                static=True, support="floor",
                collision="shelves_posts_and_casters")


def add_laboratory_equipment(b):
    """Dress the laboratory perimeter; the central task island stays empty.

    Rear counter support is z=.90, x=[-.78, 2.02], y=[1.99, 2.57].
    Refrigerator faces -X.  Returned metadata describes room dressing only and
    makes no manipulation, refrigeration or instrument-functionality claim.
    """
    _materials(b)
    equipment = [
        _centrifuge(b, (-.46, 2.28, .90)),
        _analytical_balance(b, (.10, 2.28, .90)),
    ]
    # Keep the original CC0 grant, author record, mesh/texture bytes and v2
    # conversion manifest in this build's dependency hashes.
    microscope_source = (
        b.assets_root / "objects/polyhaven/20261016_compact/industrial_microscope"
    )
    microscope_derived = (
        b.assets_root / "objects/polyhaven/derived/v2/industrial_microscope"
    )
    for root in (microscope_source, microscope_derived):
        for dependency in sorted(root.rglob("*")):
            if dependency.is_file():
                b.dependency(dependency)
    microscope = _ph(b, "lab_industrial_microscope", "industrial_microscope",
                     (1.30, 2.28, .90), version="v2", scale=.90)
    equipment.append(dict(name="lab_industrial_microscope", kind="microscope",
                          prim_path=microscope, base=[1.30, 2.28, .90],
                          scale=.90, static=True,
                          source="polyhaven/derived/v2/industrial_microscope",
                          license="CC0-1.0", author="Lukas Walzer",
                          provenance=str(microscope_source / "SOURCE.json"),
                          derivation=str(microscope_derived / "DERIVATION.json"),
                          support="rear_instrument_counter"))
    equipment.append(pipette_stand(b, "lab_pipette_station", (1.58, 2.32, .90)))
    equipment.append(test_tube_rack(b, "lab_sample_tube_rack", (1.85, 2.12, .90)))
    for i, (x, text) in enumerate(((1.75, "BUFFER\npH 7.0"),
                                   (1.86, "SALINE\n0.9 %"),
                                   (1.97, "REFERENCE\n100 mL"))):
        equipment.append(reagent_bottle(
            b, "lab_reagent_stock_%d" % i, (x, 2.48, .90),
            radius=.032 if i < 2 else .029, height=.165 if i < 2 else .14,
            label=text, amber=i == 2,
            cap="lab_eq_blue_cap" if i < 2 else "lab_eq_yellow_cap"))
    _sample_box(b, "lab_pipette_tip_box", (.98, 2.49, .90), (.17, .13, .065))
    b.label("lab_pipette_tip_box_top", "200 uL\nSTERILE TIPS",
            (.98, 2.49, .9655), (.125, .075, .0003), font_size=42)
    b.box("lab_lens_tissue_box", (.755, 2.43, .932),
          (.17, .14, .064), "lab_eq_enamel", bevel=.006, collision=False)
    b.box("lab_lens_tissue_slot", (.755, 2.43, .9648),
          (.105, .011, .0015), "lab_eq_graphite", bevel=.003, collision=False)
    b.label("lab_lens_tissue_label", "LENS TISSUES",
            (.755, 2.359, .934), (.145, .025, .0003),
            rotation=(90, 0, 0), font_size=38)
    b.box("lab_lens_tissue_fold", (.760, 2.43, .978),
          (.075, .001, .028), "paper", bevel=0, rotation=(0, 10, 0), collision=False)
    equipment.extend((
        _refrigerator(b, (3.08, .80, 0), yaw=-90),
        _instrument_cart(b, (2.65, -.35, 0)),
    ))
    return dict(schema="scence.laboratory_equipment.v1",
                role="static_perimeter_room_dressing", equipment=equipment,
                collision_policy="Static housings, supports, glass shield and cart; "
                                 "small stored containers and control details are visual.",
                task_objects_authored=False)
