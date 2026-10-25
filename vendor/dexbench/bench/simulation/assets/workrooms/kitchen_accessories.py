"""Everyday kitchen storage and utensils, authored as fixed room dressing.

The small items form believable supported groups outside the robot aisle and
the candidate task surfaces.  Procedural label bitmaps belong to the builder's
generated output directory; no generated asset is stored in the source tree.
"""
from __future__ import annotations

import math

from pxr import Usd, UsdGeom


PREFIX = "kitchen_acc_"
KEEP_CLEAR = {
    "robot_aisle": {"low": [.30, -.50, .0], "high": [1.0, .52, 1.80]},
    "bar_task_surface": {"low": [-.10, -1.20, .9001], "high": [1.10, -.60, 1.65]},
    "prep_task_surface": {"low": [-.5025, .57, .90], "high": [.32, 1.31, 1.55]},
    "cooktop_task_surface": {"low": [.32, .645, .90], "high": [.98, 1.235, 1.54]},
}


def _materials(b):
    specifications = {
        "ivory": ((.86, .83, .72), .25, 0.),
        "blue_glaze": ((.12, .30, .39), .19, 0.),
        "sage_glaze": ((.39, .53, .41), .23, 0.),
        "ochre_glaze": ((.70, .43, .13), .27, 0.),
        "terracotta": ((.49, .22, .12), .79, 0.),
        "soil": ((.052, .034, .019), 1., 0.),
        "basil": ((.10, .30, .055), .64, 0.),
        "basil_light": ((.22, .41, .08), .66, 0.),
        "rosemary": ((.19, .31, .20), .80, 0.),
        "stem": ((.21, .30, .095), .86, 0.),
        "oats": ((.64, .45, .23), .89, 0.),
        "rice": ((.86, .80, .62), .81, 0.),
        "lentils": ((.47, .21, .075), .91, 0.),
        "pepper": ((.12, .075, .040), .94, 0.),
        "paprika": ((.51, .10, .035), .94, 0.),
        "turmeric": ((.80, .48, .055), .94, 0.),
        "herbs": ((.25, .30, .10), .94, 0.),
        "salt": ((.88, .87, .80), .79, 0.),
        "cast_iron": ((.035, .043, .040), .70, .32),
        "rack": ((.08, .10, .095), .38, .72),
        "enamel": ((.77, .79, .72), .25, .12),
        "rattan": ((.47, .31, .16), .87, 0.),
        "rattan_light": ((.65, .47, .26), .86, 0.),
        "linen_sage": ((.44, .53, .41), .97, 0.),
        "linen_cream": ((.79, .76, .65), .98, 0.),
        "linen_rust": ((.60, .29, .16), .98, 0.),
        "stitch": ((.83, .80, .67), .99, 0.),
        "paper": ((.90, .87, .77), .92, 0.),
        "potato": ((.43, .30, .16), .96, 0.),
        "potato_eye": ((.18, .12, .060), 1., 0.),
        "garlic": ((.81, .75, .59), .92, 0.),
        "book_red": ((.48, .13, .085), .77, 0.),
        "book_green": ((.15, .32, .25), .80, 0.),
        "book_blue": ((.10, .22, .36), .75, 0.),
        "book_yellow": ((.73, .51, .13), .79, 0.),
        "book_cream": ((.68, .62, .47), .83, 0.),
        "book_orange": ((.66, .29, .11), .79, 0.),
    }
    for name, (color, roughness, metallic) in specifications.items():
        b.material(PREFIX + name, color, roughness, metallic)


def _circle(center, radius, plane="xy", segments=49):
    x, y, z = center
    if plane == "yz":
        return [(x, y + radius * math.cos(t), z + radius * math.sin(t))
                for t in (i * math.tau / (segments - 1) for i in range(segments))]
    return [(x + radius * math.cos(t), y + radius * math.sin(t), z)
            for t in (i * math.tau / (segments - 1) for i in range(segments))]


def _rounded_rect(cx, cy, z, width, depth, radius=.025, samples=7):
    points = []
    for x, y, start in ((cx + width / 2 - radius, cy + depth / 2 - radius, 0),
                        (cx - width / 2 + radius, cy + depth / 2 - radius, 90),
                        (cx - width / 2 + radius, cy - depth / 2 + radius, 180),
                        (cx + width / 2 - radius, cy - depth / 2 + radius, 270)):
        for i in range(samples):
            a = math.radians(start + i * 90 / (samples - 1))
            points.append((x + radius * math.cos(a), y + radius * math.sin(a), z))
    return points + points[:1]


def _leaf(b, name, origin, direction, width, material):
    """A curved, veined leaf with a tapered silhouette, not a flat triangle."""
    x, y, z = origin
    dx, dy, dz = direction
    length_xy = math.hypot(dx, dy)
    ux, uy = (-dy / max(length_xy, 1e-7), dx / max(length_xy, 1e-7))
    vertices = []
    for i in range(9):
        t = i / 8
        half = width * math.sin(math.pi * t) ** .72 / 2
        bend = .010 * math.sin(math.pi * t)
        for side in (-1, 0, 1):
            vertices.append((x + dx * t + ux * half * side,
                             y + dy * t + uy * half * side,
                             z + dz * t + bend * (1 - .62 * abs(side))))
    faces = []
    for i in range(8):
        for j in range(2):
            a = i * 3 + j
            faces.append((a, a + 1, a + 4, a + 3))
    b.mesh(b.path(name), vertices, faces, material)
    vein = [(x + dx * t, y + dy * t,
             z + dz * t + .010 * math.sin(math.pi * t) + .0006)
            for t in (i / 8 for i in range(9))]
    b.tube(name + "_vein", vein, .00065, PREFIX + "stem")


def _book(b, name, x, bottom, width, height, material, title):
    y, depth = 1.181, .181
    # Separate cover boards, rounded spine and inset paper block.
    b.box(name + "_pages", (x, y + .003, bottom + height / 2),
          (width - .005, depth - .012, height - .012), PREFIX + "paper", .001,
          collision=False)
    for side in (-1, 1):
        b.box(name + "_cover_" + str(side),
              (x + side * (width / 2 - .0013), y, bottom + height / 2),
              (.0026, depth, height), material, .001, collision=False)
    b.box(name + "_spine", (x, y - depth / 2 + .001, bottom + height / 2),
          (width, .008, height), material, .003, collision=False)
    for i in range(9):
        b.box(name + "_page_edge_" + str(i),
              (x, y + depth / 2 - .002, bottom + .018 + i * (height - .04) / 9),
              (width - .006, .0006, .0005), PREFIX + "book_cream", .0001,
              collision=False)
    for zz in (bottom + .028, bottom + height - .031):
        b.box(name + "_spine_band_" + str(zz), (x, y - depth / 2 - .0033, zz),
              (width - .004, .001, .003), PREFIX + "paper", .0003,
              collision=False)
    b.label(name + "_title", title, (x, y - depth / 2 - .0038, bottom + height * .52),
            (width - .008, height * .63), rotation=(90, 0, 0), font_size=72)


def _jar(b, name, x, bottom, radius, height, filling, title, seed):
    y = 1.190
    b.lathe(name + "_glass", (x, y, bottom),
            [(0, 0), (radius * .82, 0), (radius, .012), (radius, height - .035),
             (radius * .86, height - .021), (radius * .86, height - .010),
             (radius * .77, height - .010), (radius * .77, height - .025),
             (radius * .91, height - .039), (radius * .91, .014), (0, .014)],
            "glass")
    level = height * .70
    b.lathe(name + "_contents", (x, y, bottom + .014),
            [(0, 0), (radius * .89, 0), (radius * .89, level), (0, level)],
            PREFIX + filling)
    # Individual grains along the visible outside and uneven fill surface.
    for i in range(42):
        a = i * 2.399963 + seed
        zz = .025 + ((i * 17 + seed * 7) % 37) / 37 * (level - .022)
        rr = radius * (.82 if i < 32 else .64 * ((i % 5) / 5) ** .5)
        if i >= 32:
            zz = .014 + level + .001 * math.sin(i)
        scale = (.0043, .0020, .0017) if filling == "rice" else (.0032, .0030, .0018)
        b.sphere(name + "_grain_" + str(i),
                 (x + rr * math.cos(a), y + rr * math.sin(a), bottom + zz),
                 scale, PREFIX + filling)
    b.cylinder(name + "_lid", (x, y, bottom + height - .005), radius * .91, .020,
               "oak", collision=False)
    b.torus(name + "_seal", (x, y, bottom + height - .015), radius * .86, .002,
            PREFIX + "rack")
    b.label(name + "_label", title, (x, y - radius - .001, bottom + height * .48),
            (radius * 1.27, height * .28), rotation=(90, 0, 0), font_size=85)


def _spice(b, name, x, bottom, filling, title, height=.16):
    y, radius = 1.186, .034
    b.lathe(name + "_bottle", (x, y, bottom),
            [(0, 0), (.026, 0), (radius, .007), (radius, height - .037),
             (.023, height - .026), (.023, height - .012), (.018, height - .012),
             (.018, height - .030), (.029, height - .041), (.029, .010), (0, .010)],
            "glass")
    b.cylinder(name + "_seasoning", (x, y, bottom + (height - .045) / 2 + .010),
               .028, height - .045, PREFIX + filling, collision=False)
    b.cylinder(name + "_cap", (x, y, bottom + height - .010), .026, .025,
               PREFIX + "rack", collision=False)
    for i in range(16):
        a = i * math.tau / 16
        b.tube(name + "_cap_rib_" + str(i),
               [(x + .0265 * math.cos(a), y + .0265 * math.sin(a), bottom + height - .020),
                (x + .0265 * math.cos(a), y + .0265 * math.sin(a), bottom + height)],
               .0008, "steel")
    b.label(name + "_label", title, (x, y - radius - .001, bottom + height * .44),
            (.048, .047), rotation=(90, 0, 0), font_size=76)


def _herb(b, name, y, bottom, kind):
    # The pot foot fits wholly on the sill and clears the raised window rail.
    x = -1.985
    material = PREFIX + ("terracotta" if kind == "basil" else "sage_glaze")
    b.lathe(name + "_pot", (x, y, bottom),
            [(0, 0), (.047, 0), (.052, .007), (.070, .094), (.073, .095),
             (.073, .106), (.064, .106), (.062, .091), (.044, .012), (0, .012)],
            material)
    b.cylinder(name + "_soil", (x, y, bottom + .094), .061, .008,
               PREFIX + "soil", collision=False)
    b.torus(name + "_saucer_rim", (x, y, bottom + .004), .052, .0035, material)
    b.cylinder(name + "_saucer", (x, y, bottom + .002), .052, .004,
               material, collision=False)
    if kind == "basil":
        for i in range(7):
            a = i * 2.399963
            base = (x + .017 * math.cos(a), y + .022 * math.sin(a), bottom + .096)
            tip = (x + .015 + .040 * abs(math.cos(a)),
                   y + .065 * math.sin(a), bottom + .235 + .025 * (i % 3))
            b.tube(name + "_stem_" + str(i), [base,
                   ((base[0] + tip[0]) / 2, (base[1] + tip[1]) / 2, bottom + .19), tip],
                   .0017, PREFIX + "stem")
            for j in range(4):
                t = .40 + j * .16
                origin = tuple(base[k] + (tip[k] - base[k]) * t for k in range(3))
                side = -1 if j % 2 else 1
                direction = (.025 + .014 * math.cos(a),
                             side * (.033 + .009 * math.sin(a)), .008 + .013 * (j % 2))
                _leaf(b, name + f"_leaf_{i}_{j}", origin, direction, .026,
                      PREFIX + ("basil" if (i + j) % 2 else "basil_light"))
    else:
        for i in range(10):
            a = i * 2.399963
            base = (x + .015 * math.cos(a), y + .024 * math.sin(a), bottom + .095)
            tip = (x + .026 + .036 * abs(math.cos(a)), y + .075 * math.sin(a),
                   bottom + .28 + .035 * math.sin(i * 1.7))
            b.tube(name + "_stem_" + str(i), [base, tip], .0014, PREFIX + "stem")
            for j in range(13):
                t = .20 + j * .057
                origin = tuple(base[k] + (tip[k] - base[k]) * t for k in range(3))
                for side in (-1, 1):
                    direction = (.014, side * .017, .008)
                    _leaf(b, name + f"_needle_{i}_{j}_{side}", origin, direction, .0034,
                          PREFIX + "rosemary")
    b.label(name + "_plant_marker", "BASIL" if kind == "basil" else "ROSEMARY",
            (x + .015, y - .054, bottom + .112), (.049, .018), rotation=(90, 0, 0),
            font_size=70)


def _hanging_pan(b, name):
    x, y, z = -2.002, -2.225, 1.675
    # Revolved pan has an actual concave cooking bowl, with its open face inward.
    b.lathe(name + "_bowl", (x, y, z),
            [(0, -.018), (.097, -.018), (.123, .006), (.128, .021),
             (.121, .025), (.112, .006), (.092, -.010), (0, -.010)],
            PREFIX + "cast_iron", rotation=(0, 90, 0))
    b.torus(name + "_rim", (x + .021, y, z), .124, .0035,
            PREFIX + "cast_iron", rotation=(0, 90, 0))
    b.tube(name + "_handle", [(x + .012, y, z + .112), (x + .008, y, z + .205),
                             (x + .010, y, z + .305)], .013, PREFIX + "cast_iron")
    b.torus(name + "_handle_eye", (x + .010, y, z + .314), .016, .004,
            PREFIX + "cast_iron", rotation=(0, 90, 0))
    for zz in (z + .088, z + .107):
        b.sphere(name + "_rivet_" + str(zz), (x + .026, y, zz), (.003, .005, .005), "steel")


def _strainer(b, name):
    x, y, z, r = -1.994, -1.940, 1.719, .096
    b.torus(name + "_rim", (x, y, z), r, .004, "steel", rotation=(0, 90, 0))
    # Crossing wires follow a shallow hemispherical sieve, leaving real holes.
    for axis in range(2):
        for i in range(-7, 8):
            offset = i * .0115
            span = math.sqrt(max(.0, (r - .004) ** 2 - offset ** 2))
            points = []
            for j in range(19):
                u = -span + 2 * span * j / 18
                yy, zz = (offset, u) if axis == 0 else (u, offset)
                xx = -.040 * math.sqrt(max(0., 1 - (yy * yy + zz * zz) / r ** 2))
                points.append((x + xx, y + yy, z + zz))
            b.tube(name + f"_mesh_{axis}_{i}", points, .00065, "steel")
    b.tube(name + "_handle", [(x, y - .010, z + .091), (x, y - .013, z + .27),
                              (x, y + .013, z + .27), (x, y + .010, z + .091)],
           .0035, "steel")
    b.torus(name + "_support_loop", (x, y, z - .106), .019, .003, "steel",
            rotation=(0, 90, 0))


def _oven_mitt(b, name):
    x, y, z = -1.994, -1.636, 1.781
    # Front/back quilted fabric panels follow a mitten outline with a thumb.
    outline = [(-.049, 0), (-.055, .105), (-.044, .169), (-.016, .185),
               (.013, .181), (.033, .155), (.035, .104), (.068, .129),
               (.086, .112), (.079, .080), (.040, .033), (.038, 0)]
    vertices = []
    for xx in (-.010, .010):
        vertices.extend((x + xx, y + yy, z + zz) for yy, zz in outline)
    n = len(outline)
    # Triangular fans avoid a single concave polygon on the thumb silhouette.
    vertices.extend(((x - .012, y + .002, z + .087), (x + .012, y + .002, z + .087)))
    faces = []
    for i in range(n):
        j = (i + 1) % n
        faces.extend(((2 * n, j, i), (2 * n + 1, n + i, n + j), (i, j, n + j, n + i)))
    # Builder meshes need uniform face degree; use triangles for the side walls.
    triangles = []
    for face in faces:
        triangles.extend((face,) if len(face) == 3 else ((face[0], face[1], face[2]),
                                                        (face[0], face[2], face[3])))
    b.mesh(b.path(name + "_fabric"), vertices, triangles, PREFIX + "linen_rust")
    seam = [(x + .0125, y + yy, z + zz) for yy, zz in outline] + [(x + .0125, y + outline[0][0], z)]
    b.tube(name + "_binding", seam, .0025, PREFIX + "linen_cream")
    for i in range(5):
        zz = z + .028 + i * .027
        b.tube(name + "_quilt_" + str(i),
               [(x + .013, y - .043, zz), (x + .016, y, zz + .007),
                (x + .013, y + .029, zz + .011)], .00085, PREFIX + "stitch")
    b.tube(name + "_cuff", [(x + .012, y - .049, z + .017),
                            (x + .015, y + .037, z + .017)], .0035, PREFIX + "linen_cream")
    b.torus(name + "_loop", (x, y - .035, z + .205), .013, .003,
            PREFIX + "linen_rust", rotation=(0, 90, 0))


def _paper_roll(b, name):
    x, y, z = -1.980, -1.775, 1.405
    b.lathe(name + "_roll", (x, y, z),
            [(.020, -.122), (.055, -.122), (.056, -.115), (.056, .115),
             (.055, .122), (.020, .122)], PREFIX + "paper", rotation=(90, 0, 0))
    b.cylinder(name + "_cardboard_core", (x, y, z), .0195, .249,
               PREFIX + "rattan", rotation=(90, 0, 0), collision=False)
    b.cylinder(name + "_spindle", (x, y, z), .006, .283,
               PREFIX + "rack", rotation=(90, 0, 0), collision=False)
    for side in (-1, 1):
        yy = y + side * .147
        b.tube(name + "_bracket_" + str(side), [(-2.101, yy, z + .037),
               (-1.981, yy, z + .037), (-1.980, yy, z)], .005, PREFIX + "rack")
    vertices, faces = [], []
    for j in range(8):
        zz = z + .015 - j * .020
        xx = x + .054 + .005 * math.sin(j * .55)
        vertices.extend(((xx, y - .112, zz), (xx, y + .112, zz)))
    for j in range(7):
        a = j * 2
        faces.append((a, a + 1, a + 3, a + 2))
    b.mesh(b.path(name + "_loose_sheet"), vertices, faces, PREFIX + "paper")
    for i in range(11):
        b.tube(name + "_perforation_" + str(i),
               [(x + .055, y - .108 + i * .021, z - .095),
                (x + .055, y - .098 + i * .021, z - .095)], .00035,
               PREFIX + "linen_cream")


def _trolley(b, name):
    x, y = -1.620, .080
    for zz in (.231, .831):
        b.box(name + "_deck_" + str(zz), (x, y, zz), (.560, .500, .026),
              "oak", .006)
        b.tube(name + "_tray_lip_" + str(zz),
               _rounded_rect(x, y, zz + .023, .546, .486), .006, PREFIX + "rack")
    for i, (dx, dy) in enumerate(((-.248, -.218), (.248, -.218),
                                 (-.248, .218), (.248, .218))):
        b.cylinder(name + "_leg_" + str(i), (x + dx, y + dy, .456), .010, .758,
                   PREFIX + "rack", collision=False)
        b.cylinder(name + "_caster_stem_" + str(i), (x + dx, y + dy, .086), .006, .055,
                   "steel", collision=False)
        for side in (-1, 1):
            b.box(name + f"_caster_fork_{i}_{side}", (x + dx, y + dy + side * .015, .057),
                  (.028, .004, .049), "steel", .002, collision=False)
        b.torus(name + "_wheel_" + str(i), (x + dx, y + dy, .047), .033, .014,
                "rubber", rotation=(90, 0, 0))
        b.cylinder(name + "_wheel_hub_" + str(i), (x + dx, y + dy, .047), .018, .028,
                   PREFIX + "rack", rotation=(90, 0, 0), collision=False)
    b.tube(name + "_push_handle", [(x - .248, y - .218, .80),
           (x - .312, y - .218, .944), (x - .312, y + .218, .944),
           (x - .248, y + .218, .80)], .009, PREFIX + "rack")
    # The wheel tyres touch z=0 and the trolley is kept outside the robot lane.


def _drainer(b, name):
    x, y = -1.620, .080
    b.box(name + "_drip_tray", (x, y, .8515), (.528, .450, .015),
          PREFIX + "enamel", .009, collision=False)
    b.tube(name + "_tray_rim", _rounded_rect(x, y, .865, .514, .436), .005,
           PREFIX + "enamel")
    for xx in (-1.845, -1.548):
        b.tube(name + "_base_rail_" + str(xx), [(xx, -.122, .877), (xx, .280, .877)],
               .004, "steel")
        for yy in (-.119, .272):
            b.cylinder(name + f"_foot_{xx}_{yy}", (xx, yy, .866), .004, .014,
                       "steel", collision=False)
    for yy in (-.122, .280):
        b.tube(name + "_end_rail_" + str(yy),
               [(-1.865, yy, .879), (-1.535, yy, .879)], .004, "steel")
    for i in range(5):
        yy = -.098 + i * .066
        b.tube(name + "_plate_slot_" + str(i),
               [(-1.845, yy, .877), (-1.825, yy, .952), (-1.803, yy, .974),
                (-1.590, yy, .974), (-1.566, yy, .950), (-1.548, yy, .877)],
               .003, "steel")
    for xx in (-1.865, -1.535):
        b.tube(name + "_side_" + str(xx), [(xx, -.122, .879), (xx, -.122, .913),
               (xx, .280, .913), (xx, .280, .879)], .0035, "steel")
    b.box(name + "_cutlery_holder_bottom", (-1.435, .247, .863), (.074, .070, .008),
          PREFIX + "enamel", .004, collision=False)
    for zz in (.886, .927, .973):
        b.tube(name + "_cutlery_holder_ring_" + str(zz),
               _rounded_rect(-1.435, .247, zz, .074, .070, .008), .0026, "steel")
    for dx in (-.034, .034):
        for dy in (-.032, .032):
            b.tube(name + f"_cutlery_holder_upright_{dx}_{dy}",
                   [(-1.435 + dx, .247 + dy, .867), (-1.435 + dx, .247 + dy, .975)],
                   .0025, "steel")


def _plate(b, name, y, material):
    b.lathe(name, (-1.705, y, 1.004),
            [(0, -.006), (.080, -.006), (.114, .003), (.123, .012),
             (.124, .016), (.117, .019), (.082, .003), (0, .003)],
            material, rotation=(80, 0, 0))
    b.torus(name + "_glazed_edge", (-1.705, y - .016, 1.007), .119, .0017,
            material, rotation=(80, 0, 0))


def _cup(b, name, y, material):
    x, bottom = -1.438, .859
    b.lathe(name + "_body", (x, y, bottom),
            [(.023, 0), (.032, .003), (.037, .076), (.036, .082),
             (.031, .082), (.027, .012), (0, .012), (0, .002)], material)
    points = [(x + .032 + .024 * math.sin(t), y,
               bottom + .043 + .027 * math.cos(t))
              for t in (i * math.pi / 24 for i in range(25))]
    b.tube(name + "_handle", points, .0045, material)


def _dish_brush(b, name):
    b.tube(name + "_wood_handle", [(-1.438, .243, .875), (-1.438, .243, 1.091),
                                  (-1.433, .243, 1.126)], .008, "oak")
    b.sphere(name + "_head", (-1.433, .243, 1.125), (.023, .015, .025), "oak")
    for i in range(5):
        for j in range(5):
            x, z = -1.449 + i * .008, 1.109 + j * .008
            b.tube(name + f"_bristle_{i}_{j}", [(x, .230, z), (x, .211, z + .001)],
                   .0011, PREFIX + "paper")
    b.torus(name + "_hanging_hole", (-1.438, .243, .904), .0035, .001,
            PREFIX + "rack", rotation=(90, 0, 0))


def _bowl(b, name, bottom, material):
    b.lathe(name, (-1.755, .080, bottom),
            [(0, 0), (.040, 0), (.047, .006), (.083, .048), (.088, .065),
             (.084, .069), (.078, .062), (.039, .012), (0, .012)], material)
    b.torus(name + "_rim", (-1.755, .080, bottom + .066), .085, .002,
            material)


def _teapot(b, name):
    x, y, z = -1.478, .076, .259
    b.lathe(name + "_body", (x, y, z),
            [(0, 0), (.041, 0), (.067, .022), (.075, .063), (.066, .108),
             (.045, .125), (.041, .121), (.060, .102), (.068, .062),
             (.059, .027), (.036, .012), (0, .012)], PREFIX + "blue_glaze")
    b.lathe(name + "_lid", (x, y, z + .119),
            [(0, 0), (.044, 0), (.048, .006), (.035, .016), (0, .021)],
            PREFIX + "blue_glaze")
    b.sphere(name + "_lid_knob", (x, y, z + .150), (.012, .012, .012), "oak")
    # A tapered hollow spout, curved in the XZ plane.
    centers = [(x + .058, y, z + .042), (x + .085, y, z + .055),
               (x + .108, y, z + .091), (x + .133, y, z + .112)]
    vertices = []
    for inner in (False, True):
        for j, c in enumerate(centers):
            radius = (.023 - .004 * j) - (.003 if inner else 0)
            for i in range(32):
                a = i * math.tau / 32
                vertices.append((c[0], c[1] + radius * math.cos(a), c[2] + radius * math.sin(a)))
    faces = []
    for inner in range(2):
        for j in range(3):
            for i in range(32):
                a = inner * 128 + j * 32 + i
                nxt = inner * 128 + j * 32 + (i + 1) % 32
                face = (a, nxt, nxt + 32, a + 32)
                faces.append(face[::-1] if inner else face)
    for i in range(32):
        a, nxt = 96 + i, 96 + (i + 1) % 32
        faces.append((a, nxt, nxt + 128, a + 128))
    b.mesh(b.path(name + "_spout"), vertices, faces, PREFIX + "blue_glaze")
    handle = [(x - .058 - .035 * math.sin(t), y,
               z + .073 + .047 * math.cos(t))
              for t in (i * math.pi / 30 for i in range(31))]
    b.tube(name + "_handle", handle, .008, PREFIX + "blue_glaze")


def _woven_basket(b, name, x):
    y, bottom, width, depth, height = -.965, .315, .302, .432, .224
    b.box(name + "_floor", (x, y, bottom + .006), (width - .030, depth - .030, .012),
          PREFIX + "rattan", .006, collision=False)
    for j in range(17):
        z = bottom + .014 + j * .0125
        w = width * (.89 + .11 * j / 16)
        d = depth * (.89 + .11 * j / 16)
        ring = _rounded_rect(x, y, z, w, d, .024, 9)
        # A real handhold opening in the front and rear woven walls.
        if j >= 12:
            runs, run = [], []
            for p in ring:
                opening = abs(p[0] - x) < .050 and abs(p[1] - y) > d / 2 - .006
                if opening:
                    if len(run) > 1:
                        runs.append(run)
                    run = []
                else:
                    run.append(p)
            if len(run) > 1:
                runs.append(run)
            for k, points in enumerate(runs):
                b.tube(name + f"_weft_{j}_{k}", points, .0025,
                       PREFIX + ("rattan_light" if j % 3 else "rattan"))
        else:
            b.tube(name + "_weft_" + str(j), ring, .0025,
                   PREFIX + ("rattan_light" if j % 3 else "rattan"))
    perimeter = _rounded_rect(x, y, bottom, width, depth, .024, 13)[:-1]
    for i, p in enumerate(perimeter):
        points = []
        for j in range(18):
            t = j / 17
            scale = .89 + .11 * t + .007 * math.sin(j * math.pi + i * math.pi)
            points.append((x + (p[0] - x) * scale, y + (p[1] - y) * scale,
                           bottom + .010 + (height - .010) * t))
        # Shorten the front/rear centre reeds below the handle aperture.
        if abs(p[0] - x) < .048 and abs(p[1] - y) > depth / 2 - .006:
            points = points[:12]
        b.tube(name + "_warp_" + str(i), points, .0023, PREFIX + "rattan")
    for zz in (bottom + .006, bottom + height, bottom + height - .009):
        b.tube(name + "_bound_edge_" + str(zz),
               _rounded_rect(x, y, zz, width, depth, .024, 9), .004,
               PREFIX + "rattan_light")
    for side in (-1, 1):
        yy = y + side * (depth / 2 + .001)
        b.tube(name + "_handle_binding_" + str(side),
               [(x - .052, yy, bottom + .153), (x - .052, yy, bottom + .205),
                (x + .052, yy, bottom + .205), (x + .052, yy, bottom + .153),
                (x - .052, yy, bottom + .153)], .0036, PREFIX + "rattan_light")


def _folded_linen(b, name, bottom, material, index):
    x, y = -.670, -.965
    b.box(name + "_fold", (x + .004 * index, y, bottom + .020), (.232, .337, .040),
          material, .013, collision=False)
    b.tube(name + "_folded_edge", [(x - .108, y - .165, bottom + .015),
           (x, y - .171, bottom + .013), (x + .108, y - .165, bottom + .015)],
           .0026, material)
    for side in (-1, 1):
        b.tube(name + "_hem_" + str(side),
               [(x + side * .102, y - .151, bottom + .037),
                (x + side * .102, y, bottom + .041),
                (x + side * .102, y + .151, bottom + .037)], .00075,
               PREFIX + "stitch")
    for i in range(12):
        xx = x - .104 + i * .019
        b.tube(name + "_fringe_" + str(i), [(xx, y + .163, bottom + .023),
               (xx + .001 * math.sin(i), y + .174, bottom + .020)], .00065,
               PREFIX + "stitch")


def _potato(b, name, center, seed):
    vertices, faces = [], []
    rings, sectors = 16, 24
    sx, sy, sz = (.039, .052, .031)
    for j in range(rings + 1):
        theta = math.pi * j / rings
        for i in range(sectors):
            a = math.tau * i / sectors
            bump = 1 + .07 * math.sin(3 * a + seed) * math.sin(4 * theta + seed)
            vertices.append((center[0] + sx * math.sin(theta) * math.cos(a) * bump,
                             center[1] + sy * math.sin(theta) * math.sin(a) * bump,
                             center[2] + sz * math.cos(theta) * bump))
    for j in range(rings):
        for i in range(sectors):
            a, nxt = j * sectors + i, j * sectors + (i + 1) % sectors
            faces.append((a, nxt, nxt + sectors, a + sectors))
    # Ground each irregular potato on the basket floor, including its bumps.
    correction = .327 - min(p[2] for p in vertices)
    vertices = [(x, y, z + correction) for x, y, z in vertices]
    b.mesh(b.path(name + "_skin"), vertices, faces, PREFIX + "potato")
    for i in range(7):
        a = i * 2.399963 + seed
        theta = .40 + (i % 4) * .59
        b.sphere(name + "_eye_" + str(i),
                 (center[0] + sx * math.sin(theta) * math.cos(a),
                  center[1] + sy * math.sin(theta) * math.sin(a),
                  center[2] + sz * math.cos(theta) + correction), (.0019, .0019, .0006),
                 PREFIX + "potato_eye")


def _garlic(b, name):
    x, y, z = -.273, -.866, .418
    for i in range(9):
        a = i * math.tau / 9
        b.sphere(name + "_clove_" + str(i),
                 (x + .021 * math.cos(a), y + .021 * math.sin(a), z),
                 (.015, .015, .027), PREFIX + "garlic")
    b.lathe(name + "_neck", (x, y, z),
            [(0, 0), (.017, .012), (.009, .035), (.004, .049), (0, .052)],
            PREFIX + "garlic")
    for i in range(13):
        a = i * 2.399963
        b.tube(name + "_root_" + str(i), [(x, y, z - .020),
               (x + .012 * math.cos(a), y + .012 * math.sin(a), z - .035)], .0007,
               PREFIX + "rattan")


def build_accessories(b):
    """Add 49 static semantic objects and report measured protected clearances."""
    _materials(b)
    inventory = []

    def add(name, category, zone, support, author):
        before = set(b._names)
        author(PREFIX + name)
        inventory.append({"name": name, "category": category, "zone": zone,
                          "parts": ["/World/Geometry/" + p for p in sorted(b._names - before)],
                          "support": support, "static_dressing": True})

    def high_shelf(name):
        b.box(name + "_oak_board", (-.80, 1.185, 2.422), (1.69, .250, .028), "oak", .003)
        for i, xx in enumerate((-1.46, -.14)):
            # Continuous wall plates and stand-offs support all three shelves.
            # The original two shelf brackets stop at y=1.3105; the wall is
            # y=1.45, so these solid spacers bridge the actual gap.
            b.box(name + "_wall_plate_" + str(i), (xx, 1.442, 2.005),
                  (.055, .018, 1.178), PREFIX + "rack", .003, collision=False)
            for level, zz in enumerate((1.605, 1.945, 2.485)):
                b.box(name + f"_wall_spacer_{i}_{level}", (xx, 1.377, zz),
                      (.039, .146, .045), PREFIX + "rack", .003, collision=False)
                for dz in (-.035, .035):
                    b.sphere(name + f"_wall_screw_{i}_{level}_{dz}",
                             (xx, 1.429, zz + dz), (.004, .002, .004), "steel")
            b.tube(name + "_bracket_" + str(i), [(xx, 1.316, 2.558),
                   (xx, 1.316, 2.402), (xx, 1.074, 2.402)], .006, PREFIX + "rack")
            b.tube(name + "_brace_" + str(i), [(xx, 1.316, 2.468),
                   (xx, 1.126, 2.402)], .004, PREFIX + "rack")

    add("high_storage_shelf", "shelf", "upper_storage", "rear wall brackets", high_shelf)
    shelf_top = 2.436
    book_specs = [(-1.574, .043, .274, "book_red", "DAILY\nCOOK"),
                  (-1.523, .047, .250, "book_green", "VEG\n&\nHERBS"),
                  (-1.470, .049, .285, "book_blue", "THE\nHOME\nTABLE"),
                  (-1.415, .048, .267, "book_yellow", "GOOD\nBREAD"),
                  (-1.361, .048, .258, "book_cream", "SIMPLE\nSEASONS"),
                  (-1.307, .047, .278, "book_orange", "SOUP\n&\nSTOCK")]
    for i, (xx, ww, hh, color, title) in enumerate(book_specs):
        add("recipe_book_" + str(i + 1), "recipe_book", "upper_storage", "high_storage_shelf",
            lambda n, xx=xx, ww=ww, hh=hh, color=color, title=title:
            _book(b, n, xx, shelf_top, ww, hh, PREFIX + color, title))
    for side, xx in (("left", -1.616), ("right", -1.264)):
        def bookend(name, xx=xx, side=side):
            b.box(name + "_upright", (xx, 1.186, shelf_top + .072), (.006, .154, .144),
                  PREFIX + "rack", .002, collision=False)
            shift = .025 if side == "left" else -.025
            b.box(name + "_foot", (xx + shift, 1.186, shelf_top + .002), (.056, .154, .004),
                  PREFIX + "rack", .001, collision=False)
        add("recipe_bookend_" + side, "bookend", "upper_storage", "high_storage_shelf", bookend)
    for i, (xx, h, fill, title) in enumerate(((-1.095, .258, "oats", "ROLLED\nOATS"),
                                             (-.905, .278, "rice", "BASMATI\nRICE"),
                                             (-.715, .236, "lentils", "RED\nLENTILS"))):
        add("grain_jar_" + fill, "labelled_grain_jar", "upper_storage", "high_storage_shelf",
            lambda n, xx=xx, h=h, fill=fill, title=title, i=i:
            _jar(b, n, xx, shelf_top, .074, h, fill, title, i + 1))
    for i, (xx, fill, title) in enumerate(((-.540, "pepper", "BLACK\nPEPPER"),
                                          (-.425, "paprika", "SMOKED\nPAPRIKA"),
                                          (-.310, "turmeric", "TURMERIC"),
                                          (-.195, "herbs", "MIXED\nHERBS"),
                                          (-.080, "salt", "SEA\nSALT"))):
        add("seasoning_" + fill, "labelled_seasoning_bottle", "upper_storage", "high_storage_shelf",
            lambda n, xx=xx, fill=fill, title=title, i=i:
            _spice(b, n, xx, shelf_top, fill, title, .151 + (i % 3) * .011))

    for kind, yy in (("basil", -1.035), ("rosemary", .620)):
        add("window_" + kind, "potted_herb", "window_sill", "window_sill z=1.06m",
            lambda n, yy=yy, kind=kind: _herb(b, n, yy, 1.060, kind))

    def wall_rack(name):
        # On the opaque front section of the left wall, away from the window.
        b.cylinder(name + "_bar", (-2.002, -1.940, 2.043), .008, .920,
                   PREFIX + "rack", rotation=(90, 0, 0), collision=False)
        for i, yy in enumerate((-2.385, -1.495)):
            b.box(name + "_mount_" + str(i), (-2.095, yy, 2.043), (.014, .045, .074),
                  PREFIX + "rack", .003, collision=False)
            b.cylinder(name + "_standoff_" + str(i), (-2.047, yy, 2.043), .006, .094,
                       PREFIX + "rack", rotation=(0, 90, 0), collision=False)
            for zz in (2.021, 2.065):
                b.sphere(name + f"_screw_{i}_{zz}", (-2.086, yy, zz),
                         (.0016, .003, .003), "steel")
        for i, yy in enumerate((-2.225, -1.940, -1.671)):
            b.tube(name + "_hook_" + str(i), [(-1.995, yy, 2.056),
                   (-1.981, yy, 2.043), (-1.985, yy, 2.014),
                   (-2.004, yy, 2.001), (-2.010, yy, 2.017)], .0035, "steel")
    add("wall_utensil_rack", "wall_rack", "left_wall_storage", "opaque left front wall", wall_rack)
    add("hanging_cast_iron_pan", "cast_iron_skillet", "left_wall_storage", "wall_utensil_rack hook", lambda n: _hanging_pan(b, n))
    add("hanging_mesh_strainer", "wire_mesh_strainer", "left_wall_storage", "wall_utensil_rack hook", lambda n: _strainer(b, n))
    add("quilted_oven_mitt", "oven_mitt", "left_wall_storage", "wall_utensil_rack hook", lambda n: _oven_mitt(b, n))
    add("kitchen_paper_roll", "paper_towel_roll", "left_wall_storage", "left wall spindle brackets", lambda n: _paper_roll(b, n))

    add("dish_trolley", "two_tier_trolley", "left_trolley", "floor z=0m, four caster tyres", lambda n: _trolley(b, n))
    add("dish_drainer", "wire_dish_drainer", "left_trolley", "dish_trolley upper deck z=.844m", lambda n: _drainer(b, n))
    for i, yy in enumerate((-.066, .000, .066, .132)):
        add("draining_plate_" + str(i + 1), "ceramic_plate", "left_trolley", "dish_drainer plate slots",
            lambda n, yy=yy, i=i: _plate(b, n, yy, PREFIX + ("ivory" if i % 2 else "sage_glaze")))
    for i, yy in enumerate((-.075, .085)):
        add("draining_cup_" + str(i + 1), "ceramic_cup", "left_trolley", "dish_drainer tray z=.859m",
            lambda n, yy=yy, i=i: _cup(b, n, yy, PREFIX + ("ochre_glaze" if i else "blue_glaze")))
    add("wooden_dish_brush", "dish_brush", "left_trolley", "dish_drainer cutlery holder", lambda n: _dish_brush(b, n))

    def serving_tray(name):
        b.box(name + "_base", (-1.620, .080, .2515), (.497, .408, .015), "oak", .006,
              collision=False)
        b.tube(name + "_rim", _rounded_rect(-1.620, .080, .267, .485, .396, .025),
               .007, "oak")
    add("lower_serving_tray", "wood_serving_tray", "left_trolley", "dish_trolley lower deck z=.244m", serving_tray)
    for i, bottom in enumerate((.259, .290, .321)):
        add("stacked_bowl_" + str(i + 1), "ceramic_bowl", "left_trolley", "lower_serving_tray / nested bowl",
            lambda n, bottom=bottom, i=i: _bowl(b, n, bottom, PREFIX + ("ivory" if i != 1 else "sage_glaze")))
    add("blue_ceramic_teapot", "teapot", "left_trolley", "lower_serving_tray z=.259m", lambda n: _teapot(b, n))

    def island_shelf(name):
        b.box(name + "_board", (-.495, -.965, .302), (.758, .516, .026), "oak", .004)
        # Two metal brackets transfer the shelf load into the left trestle.
        for yy in (-1.165, -.765):
            b.tube(name + "_bracket_" + str(yy), [(-.866, yy, .404),
                   (-.866, yy, .286), (-.145, yy, .286)], .006, PREFIX + "rack")
    add("island_end_lower_shelf", "lower_storage_shelf", "bar_lower_storage", "island left trestle brackets", island_shelf)
    for name, xx in (("linen", -.670), ("produce", -.322)):
        add("woven_" + name + "_basket", "woven_storage_basket", "bar_lower_storage", "island_end_lower_shelf z=.315m",
            lambda n, xx=xx: _woven_basket(b, n, xx))
    for i, (bottom, color) in enumerate(((.327, "linen_cream"), (.367, "linen_sage"), (.407, "linen_rust"))):
        add("folded_kitchen_linen_" + str(i + 1), "folded_kitchen_towel", "bar_lower_storage", "woven_linen_basket / folded towel",
            lambda n, bottom=bottom, color=color, i=i: _folded_linen(b, n, bottom, PREFIX + color, i))
    for i, center in enumerate(((-.373, -1.064, .367), (-.272, -1.070, .367),
                                (-.371, -.889, .367), (-.270, -.865, .367))):
        add("stored_potato_" + str(i + 1), "potato", "bar_lower_storage", "woven_produce_basket floor",
            lambda n, center=center, i=i: _potato(b, n, center, i + 2))
    add("stored_garlic", "garlic_bulb", "bar_lower_storage", "stored potatoes in woven_produce_basket", lambda n: _garlic(b, n))

    # Measure actual authored world bounds rather than trusting placement hints.
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"], useExtentsHint=False)
    part_ranges = {}
    for obj in inventory:
        ranges = [cache.ComputeWorldBound(b.stage.GetPrimAtPath(path)).ComputeAlignedRange()
                  for path in obj["parts"]]
        part_ranges[obj["name"]] = ranges
        obj["bounds_m"] = {
            "low": [min(float(r.GetMin()[i]) for r in ranges) for i in range(3)],
            "high": [max(float(r.GetMax()[i]) for r in ranges) for i in range(3)],
        }
    clearances = []
    for name, protected in KEEP_CLEAR.items():
        distances = []
        violations = []
        for obj in inventory:
            # A semantic group may span empty space, e.g. a wall plate behind
            # the prep area and a shelf above it.  Check each authored part.
            part_distances = []
            for bounds in part_ranges[obj["name"]]:
                lo, hi = bounds.GetMin(), bounds.GetMax()
                gap = [max(protected["low"][i] - hi[i], lo[i] - protected["high"][i], 0.)
                       for i in range(3)]
                part_distances.append(math.sqrt(sum(g * g for g in gap)))
            distance = min(part_distances)
            distances.append((distance, obj["name"]))
            if distance < 1.e-5:
                violations.append(obj["name"])
        closest = min(distances)
        clearances.append({"name": name, **protected, "clear": not violations,
                           "method": "minimum distance across authored part world AABBs",
                           "minimum_aabb_distance_m": closest[0],
                           "closest_object": closest[1], "violations": violations})
    if any(not x["clear"] for x in clearances):
        raise ValueError("Kitchen accessories encroach on a protected task area: " + str(clearances))
    return {
        "schema": "kitchen_static_accessories.v1", "object_count": len(inventory),
        "objects": inventory,
        "zones": {
            "upper_storage": "Third high shelf, labelled pantry jars, recipe library and bookends",
            "window_sill": "Basil and rosemary in different ceramic pots with saucers",
            "left_wall_storage": "Front opaque wall rail, hanging cookware, quilted mitt and kitchen paper",
            "left_trolley": "Four-caster two-tier trolley with drainer, four plates, two cups, bowls and teapot",
            "bar_lower_storage": "Left-end low shelf with woven linen and produce baskets",
        },
        "clearance_constraints": clearances,
        "procedural_asset_policy": "Geometry is authored from source; labels are generated below output_root/generated/textures",
        "scope": "Fixed static dressing; movable task payloads, robots and controllers are owned separately",
    }
