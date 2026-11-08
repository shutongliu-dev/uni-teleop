"""Detailed, passive dressing for a full-sized domestic laundry room.

The garments are authored wrinkled surface meshes, not deformable task assets.
The room's folding island is deliberately kept empty for future task contents.
Downloaded CC0 models remain in the versioned shared asset store; their source
records and texture dependencies are included in the builder manifest.
"""
from collections import Counter
import json
import math

import numpy as np
from pxr import UsdGeom, UsdPhysics

from bench.paths import load_json


PREFIX = "laundry_"


def define_detail_materials(b):
    """Add washable paint, plastics, woven fabric and paper materials."""
    specs = (
        ("laundry_ivory", (.77, .77, .70), .48, 0),
        ("laundry_sage", (.28, .40, .33), .48, 0),
        ("laundry_deep_sage", (.12, .23, .19), .52, 0),
        ("laundry_chalk", (.88, .88, .82), .62, 0),
        ("laundry_amber", (.40, .20, .065), .28, 0),
        ("laundry_blue", (.14, .27, .36), .46, 0),
        ("laundry_caps", (.07, .12, .11), .42, 0),
        ("laundry_pegs", (.59, .43, .26), .68, 0),
        ("laundry_wicker", (.42, .28, .15), .84, 0),
        ("laundry_pale_wicker", (.62, .49, .31), .81, 0),
        ("laundry_soft_gray", (.43, .46, .45), .84, 0),
        ("laundry_brass", (.43, .32, .15), .29, .8),
        ("laundry_label_green", (.20, .32, .25), .74, 0),
    )
    for name, color, roughness, metallic in specs:
        b.material(name, color, roughness, metallic)
    linen = b.assets_root / "materials/polyhaven/v1/rough_linen"
    pattern = b.assets_root / "materials/polyhaven/v1/fabric_pattern_07"
    fabric_root = b.assets_root / "materials/polyhaven/laundry_20261021"
    terry = fabric_root / "terry_cloth"
    denim = fabric_root / "denim_fabric"
    cotton = fabric_root / "cotton_jersey"
    for source in (terry / "SOURCE.json", denim / "SOURCE.json", cotton / "SOURCE.json"):
        b.dependency(source)
    for source in (linen / "SOURCE.json", pattern / "SOURCE.json"):
        b.dependency(source)
    for name, color in (
        ("laundry_linen_cream", (.82, .79, .68)),
        ("laundry_linen_white", (.90, .90, .85)),
        ("laundry_linen_sage", (.36, .48, .38)),
        ("laundry_linen_blue", (.23, .34, .43)),
        ("laundry_linen_ochre", (.56, .39, .19)),
        ("laundry_linen_blush", (.56, .38, .33)),
        ("laundry_linen_gray", (.46, .47, .44)),
    ):
        b.material(name, color, .96, normal=str(terry / "terry_cloth_nor_gl_2k.jpg"),
                   roughness_texture=str(terry / "terry_cloth_rough_2k.jpg"))
    b.material("laundry_denim", (.20, .30, .40), .94,
               texture=str(denim / "denim_fabric_diff_2k.jpg"),
               normal=str(denim / "denim_fabric_nor_gl_2k.jpg"),
               roughness_texture=str(denim / "denim_fabric_rough_2k.jpg"))
    b.material("laundry_cotton", (.86, .85, .78), .94,
               normal=str(cotton / "cotton_jersey_nor_gl_2k.jpg"),
               roughness_texture=str(cotton / "cotton_jersey_rough_2k.jpg"))
    b.material("laundry_board_fabric", (.62, .65, .61), .95,
               normal=str(pattern / "fabric_pattern_07_nor_gl_1k.jpg"),
               roughness_texture=str(pattern / "fabric_pattern_07_rough_1k.jpg"))
    return [name for name in b.materials if name.startswith(PREFIX)]


def _mesh(b, name, vertices, faces, material, uv=None, collision=False):
    """Smooth normals computed on the actual sculpted cloth surface."""
    points = np.asarray(vertices, dtype=float)
    normals = np.zeros_like(points)
    for face in faces:
        a, c, d = face[:3]
        n = np.cross(points[c] - points[a], points[d] - points[a])
        for index in face:
            normals[index] += n
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1.e-12)
    prim = b.mesh(b.path(name), points, faces, material, uv, normals)
    if collision:
        UsdPhysics.CollisionAPI.Apply(prim)
        UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr("none")
    return prim


def _grid(b, name, fn, nu, nv, material, uv_scale=(1., 1.)):
    vertices, uv, faces = [], [], []
    for j in range(nv + 1):
        for i in range(nu + 1):
            u, v = i / nu, j / nv
            vertices.append(fn(u, v))
            uv.append((u * uv_scale[0], v * uv_scale[1]))
    for j in range(nv):
        for i in range(nu):
            k = j * (nu + 1) + i
            faces.extend(((k, k + 1, k + nu + 2),
                          (k, k + nu + 2, k + nu + 1)))
    return _mesh(b, name, vertices, faces, material, uv)


def _folded_towel(b, name, xyz, size, material, seed=0):
    """A softly rounded folded slab with undulating upper surface and seams."""
    x, y, z = xyz
    w, d, h = size
    # Perimeter rings retain the folded layers and bulging corners, unlike a cube.
    n, rings = 72, 10
    verts, faces, uv = [], [], []
    for j in range(rings + 1):
        v = j / rings
        for i in range(n):
            a = i * math.tau / n
            radius = .94 + .06 * math.sin(math.pi * v)
            px = math.copysign(abs(math.cos(a)) ** .23, math.cos(a)) * w / 2 * radius
            py = math.copysign(abs(math.sin(a)) ** .23, math.sin(a)) * d / 2 * radius
            pz = z + h * v + .0018 * math.sin(5 * a + seed) * math.sin(math.pi * v)
            verts.append((x + px, y + py, pz))
            uv.append((i / n * 2, v * .2))
    for j in range(rings):
        for i in range(n):
            k, q = j * n + i, j * n + (i + 1) % n
            faces.extend(((k, q, q + n), (k, q + n, k + n)))
    _mesh(b, name + "_folded_edge", verts, faces, material, uv)
    def top(u, v):
        px, py = (u - .5) * w * .94, (v - .5) * d * .94
        wrinkle = (.0025 * math.sin(u * 21 + seed) * math.sin(v * 13 + .3)
                   + .003 * math.sin(u * 8 + v * 4)) * math.sin(math.pi * u) * math.sin(math.pi * v)
        return x + px, y + py, z + h + wrinkle
    _grid(b, name + "_top", top, 30, 24, material, (w * 3, d * 3))
    for level in (.28, .61):
        pts = [(x + (u - .5) * w * .91, y - d * .474,
                z + h * level + .0015 * math.sin(u * 12 + seed))
               for u in np.linspace(0, 1, 30)]
        b.tube(name + "_layer_" + str(level), pts, .0007, "laundry_linen_cream")
    for side in (-1, 1):
        pts = [(x + (u - .5) * w * .88, y + side * d * .42,
                z + h + .0015) for u in np.linspace(0, 1, 30)]
        b.tube(name + "_hem_" + str(side), pts, .0010, material)
    # Static folded textile proxy, no rigid-body or deformable simulation.
    path = b.path(name + "_support")
    root = UsdGeom.Xform.Define(b.stage, path)
    b.xform(root.GetPrim(), (x, y, z + h / 2))
    b.collider_box(path, (w * .92, d * .92, h))


def _draped_towel(b, name, xyz, width, drop, material, seed=0):
    x, y, z = xyz
    def cloth(u, v):
        q = (v - .5) * 2
        wave = math.sin(u * math.pi * 8 + seed)
        bend = math.sqrt(q * q + .0018)
        return (x + (u - .5) * width,
                y + .031 * math.tanh(q * 30) + .025 * wave * (.25 + .75 * abs(q))
                + .008 * math.sin(u*19+q*7+seed)*abs(q),
                z - drop * (bend - math.sqrt(.0018))
                + .016 * math.sin(u * 13 + seed) * abs(q))
    _grid(b, name, cloth, 44, 64, material, (width * 3, drop * 6))
    for edge in (0, 1):
        b.tube(name + "_side_seam_" + str(edge),
               [cloth(edge, v) for v in np.linspace(0, 1, 60)], .0013, material)
        b.tube(name + "_end_hem_" + str(edge),
               [cloth(u, edge) for u in np.linspace(0, 1, 44)], .0022, material)


def _hanger(b, name, x, y, z, width=.42):
    # Open wire hook and triangular shoulders; hook genuinely surrounds rail.
    hook = [(x + .031 * math.cos(a), y, z + .045 + .031 * math.sin(a))
            for a in np.linspace(-.8, math.pi * 1.35, 22)]
    hook += [(x, y, z + .005), (x, y, z - .025)]
    b.tube(name + "_hook", hook, .0026, "chrome")
    b.tube(name + "_wood", [(x, y, z - .022), (x - width / 2, y, z - .13),
                            (x - width / 2 + .02, y, z - .155),
                            (x + width / 2 - .02, y, z - .155),
                            (x + width / 2, y, z - .13), (x, y, z - .022)],
           .008, "laundry_pegs")


def _shirt(b, name, xyz, material, seed=0):
    x, y, z = xyz
    # Broad shoulder / short-sleeve silhouette narrows into the hanging torso.
    row_v = (0, .12, .28, .40, .44, 1)
    row_w = (.075, .22, .31, .30, .22, .225)
    def panel(u, v, side):
        w = np.interp(v, row_v, row_w)
        ripple = (.030 * math.sin(u * 17 + .9*v + seed) * (.18 + .82*v)
                  + .010 * math.sin(v*23+u*9+seed)*math.sin(math.pi*u)*v)
        px = x + (2 * u - 1) * w + .008*math.sin(v*8+seed)*v
        pz = z - .68 * v + .019 * math.sin(u * 13 + seed) * v
        # Curved neckline and gravity folds lead away from the shoulders.
        pz -= .024 * math.sin(u * math.pi) * (1 - v) ** 12
        sleeve = .020*math.exp(-((v-.31)/.10)**2)*abs(2*u-1)**3
        py = y + side * (.013 + .014 * math.sin(u * math.pi)) + ripple + sleeve
        return px, py, pz
    for side in (-1, 1):
        fn = lambda u, v, side=side: panel(u, v, side)
        _grid(b, name + "_panel_" + str(side), fn, 42, 48, material, (1.6, 2.))
        b.tube(name + "_hem_" + str(side), [fn(u, 1) for u in np.linspace(0, 1, 40)],
               .0018, material)
        b.tube(name + "_neck_" + str(side), [fn(u, 0) for u in np.linspace(0, 1, 24)],
               .003, material)
        for edge in (0, 1):
            b.tube(name + f"_seam_{side}_{edge}", [fn(edge, v) for v in np.linspace(0, 1, 50)],
                   .0012, material)
    # Small sewn chest pocket and visible button placket on one cotton shirt.
    if seed == 1:
        for i in range(6):
            b.sphere(name + "_button_" + str(i), (x, y - .035, z - .12 - i * .078),
                     (.0025, .0015, .0025), "laundry_ivory")
        b.tube(name + "_placket", [(x, y - .033, z - .08), (x, y - .033, z - .64)],
               .001, "laundry_linen_cream")


def _bottle(b, name, xyz, color, label, radius=.047, height=.23, pump=False):
    x, y, z = xyz
    b.lathe(name + "_bottle", xyz,
            [(0, 0), (radius * .82, 0), (radius, .012),
             (radius, height * .70), (radius * .88, height * .80),
             (radius * .43, height * .88), (radius * .43, height), (0, height)],
            color, collision=True)
    b.cylinder(name + "_cap", (x, y, z + height + .012), radius * .48, .024,
               "laundry_caps", collision=False)
    for k in range(18):
        a = math.tau * k / 18
        b.tube(name + "_cap_ridge_" + str(k),
               [(x + radius * .485 * math.cos(a), y + radius * .485 * math.sin(a), z + height + .003),
                (x + radius * .485 * math.cos(a), y + radius * .485 * math.sin(a), z + height + .021)],
               .0007, color)
    b.label(name + "_label", label, (x, y - radius - .0008, z + height * .49),
            (radius * 1.56, height * .44), rotation=(90, 0, 0), font_size=48)
    if pump:
        b.cylinder(name + "_pump_stem", (x, y, z + height + .045), .006, .047,
                   "laundry_ivory", collision=False)
        b.tube(name + "_pump", [(x, y, z + height + .07),
                                (x, y - .05, z + height + .07),
                                (x, y - .055, z + height + .062)], .008, "laundry_caps")


def _jug(b, name, xyz, color, label, height=.265):
    x, y, z = xyz
    # Soft-shouldered detergent jug with a separate open loop handle.
    b.lathe(name + "_body", xyz,
            [(0, 0), (.061, 0), (.073, .012), (.078, height * .55),
             (.068, height * .75), (.035, height * .88), (.027, height), (0, height)],
            color, collision=True)
    b.tube(name + "_handle", [(x + .054, y, z + height * .77),
                              (x + .105, y, z + height * .73),
                              (x + .115, y, z + height * .50),
                              (x + .092, y, z + height * .38),
                              (x + .067, y, z + height * .40)], .012, color)
    b.cylinder(name + "_cap", (x, y, z + height + .018), .029, .036,
               "laundry_caps", collision=False)
    b.label(name + "_label", label, (x, y - .0785, z + height * .45),
            (.112, .11), rotation=(90, 0, 0), font_size=60)


def _jar(b, name, xyz, label, material="laundry_ivory", radius=.065, height=.16):
    x, y, z = xyz
    b.lathe(name + "_body", xyz, [(0, 0), (radius * .92, 0), (radius, .009),
                                  (radius, height - .012), (radius * .92, height), (0, height)],
            material, collision=True)
    b.cylinder(name + "_lid", (x, y, z + height + .006), radius * 1.03, .019,
               "oak", collision=False)
    b.label(name + "_tag", label, (x, y - radius - .001, z + height * .53),
            (radius * 1.6, height * .42), rotation=(90, 0, 0), font_size=42)


def _source_asset(b, name, slug, xyz, *, scale=1., yaw=0):
    root = b.assets_root / "objects/polyhaven/v1" / slug
    record = root / "SOURCE.json"
    b.dependency(record)
    metadata = load_json(record)
    for file in metadata.get("files", []):
        b.dependency(root / file["path"])
    return b.asset(name, root / f"{slug}_1k.usdc", xyz, scale=scale,
                   rotation=(0, 0, yaw))


def _innerscene_asset(b, name, slug, xyz):
    source = b.assets_root / "objects/innerscene/20261021" / slug
    derived = b.assets_root / "objects/innerscene/20261021/derived_zup_v1" / slug
    record = derived / "DERIVATION.json"
    b.dependency(record)
    metadata = load_json(record)
    for path in (source / "SOURCE.json", source / "LICENSE.txt", metadata["source"]):
        b.dependency(path)
    for path in metadata["texture_files"]:
        b.dependency(path)
    return b.asset(name, derived / f"{slug}.usdc", xyz)


def _laundry_pile(b, name, xyz, material, seed=0, radius=.22):
    x, y, z = xyz
    def pile(u, v):
        angle = u * math.tau
        r = v * radius
        ridge = .025 * math.sin(angle * 4 + seed) * math.sin(v * 5)
        return (x + r * math.cos(angle), y + r * .77 * math.sin(angle),
                z + .035 + .10 * (1 - v * v) + ridge + .012 * math.sin(v * 23 + angle * 3))
    _grid(b, name, pile, 60, 32, material, (2, 1))
    b.tube(name + "_hem", [pile(u, 1) for u in np.linspace(0, 1, 60)], .002, material)


def _ironing_board(b):
    name, x, y, top = "laundry_ironing", 2.02, -1.29, .88
    # Actual rounded board outline; long axis Y, tapered curved nose toward -Y.
    outline = []
    for a in np.linspace(math.pi, math.tau, 32):
        outline.append((x + .19 * math.cos(a), y - .35 + .25 * math.sin(a)))
    for a in np.linspace(0, math.pi, 32):
        outline.append((x + .23 * math.cos(a), y + .44 + .08 * math.sin(a)))
    vertices = [(px, py, z) for z in (top - .028, top) for px, py in outline]
    n = len(outline)
    faces = [(i, (i + 1) % n, (i + 1) % n + n, i + n) for i in range(n)]
    faces += [(n, n + i, n + i + 1) for i in range(1, n - 1)]
    # Use triangles throughout so Builder.mesh receives a rectangular array.
    triangles = []
    for face in faces:
        triangles.extend((face[0], face[i], face[i + 1]) for i in range(1, len(face) - 1))
    uv = [(px * 4, py * 4) for px, py, _ in vertices]
    _mesh(b, name + "_board", vertices, triangles, "laundry_board_fabric", uv, collision=True)
    b.tube(name + "_edge", [(px, py, top - .009) for px, py in outline + outline[:1]],
           .007, "laundry_linen_white")
    for sx in (-.155, .155):
        for s in (-1, 1):
            b.tube(name + f"_crossleg_{sx}_{s}",
                   [(x + sx, y + s * .40, .045), (x + sx, y - s * .30, top - .06)],
                   .012, "laundry_ivory", collision=True)
        b.cylinder(name + "_hinge_" + str(sx), (x + sx, y, .47), .023, .036,
                   "chrome", rotation=(0, 90, 0), collision=False)
    for s in (-1, 1):
        b.tube(name + "_footbar_" + str(s), [(x - .23, y + s * .40, .04),
                                             (x + .23, y + s * .40, .04)], .015,
               "laundry_ivory", collision=True)
        for sx in (-.235, .235):
            b.sphere(name + f"_foot_{s}_{sx}", (x + sx, y + s * .40, .025),
                     (.024, .023, .025), "rubber")
    # Iron has a curved pointed sole, smaller sculpted reservoir and open grip.
    ix, iy = x, y + .19
    points = [(ix - .062, iy + .10), (ix + .062, iy + .10),
              (ix + .071, iy + .025), (ix + .045, iy - .075), (ix, iy - .145),
              (ix - .045, iy - .075), (ix - .071, iy + .025)]
    vv = [(px, py, z) for z in (top + .004, top + .018) for px, py in points]
    ff = []
    for i in range(7):
        q = (i + 1) % 7
        ff.extend(((i, q, q + 7), (i, q + 7, i + 7)))
    ff.extend((7, 7 + i, 7 + i + 1) for i in range(1, 6))
    _mesh(b, name + "_iron_sole", vv, ff, "chrome", collision=True)
    b.sphere(name + "_iron_reservoir", (ix, iy, top + .050),
             (.062, .105, .040), "laundry_sage")
    b.tube(name + "_iron_grip", [(ix, iy + .068, top + .035),
                                  (ix, iy + .055, top + .125),
                                  (ix, iy - .055, top + .135),
                                  (ix, iy - .087, top + .055)], .014, "laundry_ivory")
    b.cylinder(name + "_iron_dial", (ix + .032, iy + .01, top + .091), .016, .009,
               "laundry_caps", collision=False)
    b.sphere(name + "_iron_button", (ix, iy - .047, top + .150),
             (.008, .014, .004), "laundry_sage")
    cable = [(ix, iy + .105, top + .07), (ix + .09, iy + .17, top + .06),
             (ix + .21, iy + .10, top - .03), (ix + .28, iy + .02, .43),
             (ix + .25, iy - .10, .09)]
    cable += [(ix + .17 + .10 * math.cos(a), iy - .16 + .12 * math.sin(a), .025)
              for a in np.linspace(.7, math.tau * 1.8, 44)]
    b.tube(name + "_braided_cable", cable, .004, "laundry_soft_gray")
    b.box(name + "_unplugged_plug", (cable[-1][0], cable[-1][1], .025),
          (.032, .045, .028), "laundry_caps", .006, collision=False)
    return {"name": name, "class": "ironing_board_and_iron", "count": 2,
            "bounds": [[1.76, -1.91, .01], [2.34, -.72, 1.05]]}


def dress_laundry_room(b):
    """Compose everyday supplies, drying textiles, ironing and sorted storage."""
    define_detail_materials(b)
    inventory = []
    def item(name, kind, xyz, count=1):
        inventory.append({"name": name, "class": kind, "position": list(xyz), "count": count})

    # Open shelves: six separate, deliberately uneven folded towel stacks.
    palette = ["laundry_linen_white", "laundry_linen_sage", "laundry_linen_cream",
               "laundry_linen_blue", "laundry_linen_gray"]
    for stack, (x, z, layers) in enumerate(((.64, 1.55, 4), (1.06, 1.55, 3),
                                          (1.82, 1.55, 3), (2.18, 1.95, 3),
                                          (.70, 2.30, 2), (1.08, 2.30, 2))):
        for j in range(layers):
            name = f"laundry_shelf_towel_{stack}_{j}"
            xyz = (x + .005 * math.sin(j + stack), 1.99 + .004 * math.cos(j), z + j * .035)
            _folded_towel(b, name, xyz, (.29 if stack != 2 else .31, .235, .035),
                          palette[(stack + j // 3) % len(palette)], seed=stack + j)
            item(name, "folded_towel", xyz)
    for i, (x, slug) in enumerate(((1.70, "wicker_basket_01"), (2.15, "wicker_basket_02"))):
        name = f"laundry_upper_wicker_{i}"
        _source_asset(b, name, slug, (x, 1.99, 2.30), scale=.70)
        item(name, "woven_storage_basket", (x, 1.99, 2.30))
    for j, (x, label, color, h) in enumerate(((.56, "WOOL\nDELICATES", "laundry_ivory", .23),
                                            (.75, "LINEN\nRINSE", "laundry_sage", .20),
                                            (1.0, "DAILY\nWASH", "laundry_blue", .25),
                                            (1.26, "COTTON\nCARE", "laundry_ivory", .22))):
        name = f"laundry_shelf_bottle_{j}"
        _bottle(b, name, (x, 1.99, 1.95), color, label, height=h)
        item(name, "laundry_product", (x, 1.99, 1.95))
    for j, (x, label) in enumerate(((1.62, "SOAP\nFLAKES"), (1.80, "DRYER\nBALLS"))):
        name = f"laundry_shelf_jar_{j}"
        _jar(b, name, (x, 1.99, 1.95), label, height=.17, radius=.07)
        item(name, "lidded_supply_jar", (x, 1.99, 1.95))

    # Countertop essentials: assembled in a tray, with practical empty gaps.
    b.box("laundry_products_tray", (.82, 1.79, .935), (.86, .36, .028),
          "laundry_ivory", .020)
    for side in (-1, 1):
        b.box("laundry_products_tray_rail_" + str(side),
              (.82, 1.79 + side * .17, .958), (.83, .012, .045), "laundry_ivory", .004)
    item("laundry_products_tray", "countertop_tray", (.82, 1.79, .92))
    for j, (x, color, text) in enumerate(((.53, "laundry_sage", "EVERYDAY\nLAUNDRY"),
                                         (.78, "laundry_ivory", "FABRIC\nSOFTENER"))):
        name = f"laundry_counter_detergent_{j}"
        _jug(b, name, (x, 1.83, .95), color, text)
        item(name, "detergent_jug", (x, 1.83, .95))
    _bottle(b, "laundry_counter_delicate", (1.10, 1.84, .95), "laundry_amber",
            "WOOL\nWASH", height=.23, pump=True)
    item("laundry_counter_delicate", "pump_detergent", (1.10, 1.84, .95))
    _source_asset(b, "laundry_counter_spray", "multi_cleaner_bottle", (1.38, 1.90, .92), yaw=-10)
    item("laundry_counter_spray", "cleaner_spray", (1.38, 1.90, .92))
    _jar(b, "laundry_counter_powder", (1.60, 1.91, .92), "LAUNDRY\nPOWDER", radius=.084, height=.20)
    item("laundry_counter_powder", "powder_canister", (1.60, 1.91, .92))
    _source_asset(b, "laundry_counter_basket", "wicker_basket_01", (2.08, 1.84, .92), scale=.95)
    item("laundry_counter_basket", "cloth_storage_basket", (2.08, 1.84, .92))
    for j in range(3):
        name = f"laundry_counter_washcloth_{j}"
        _folded_towel(b, name, (2.08, 1.84, .946 + j * .025), (.24, .19, .025),
                      palette[j], seed=6 + j)
        item(name, "folded_washcloth", (2.08, 1.84, .946 + j * .025))
    # Brush and stain-soap dish in front of the product grouping.
    b.box("laundry_soap_dish", (1.53, 1.65, .937), (.19, .11, .03), "ceramic", .016)
    b.box("laundry_stain_soap", (1.53, 1.65, .963), (.12, .066, .022), "laundry_chalk", .01)
    item("laundry_stain_soap", "stain_soap_and_dish", (1.53, 1.65, .92), 2)
    b.box("laundry_scrub_brush_back", (1.79, 1.65, .968), (.15, .062, .023), "oak", .014)
    for ix in range(12):
        for iy in range(4):
            b.tube(f"laundry_brush_bristle_{ix}_{iy}",
                   [(1.731 + ix * .0105, 1.626 + iy * .016, .922),
                    (1.731 + ix * .0105, 1.626 + iy * .016, .956)],
                   .0015, "laundry_linen_cream")
    item("laundry_scrub_brush", "wood_scrub_brush", (1.79, 1.65, .92))

    # A full-size clothes rail on the left, with four separate textile pieces.
    for y in (-1.10, .42):
        b.tube("laundry_rail_leg_" + str(y), [(-1.79, y, .05), (-1.79, y, 1.66)],
               .016, "laundry_ivory", collision=True)
        b.tube("laundry_rail_foot_" + str(y), [(-2.20, y, .04), (-1.38, y, .04)],
               .016, "laundry_ivory", collision=True)
        for x in (-2.20, -1.38):
            b.sphere("laundry_rail_footpad_" + str((x, y)), (x, y, .029),
                     (.026, .032, .027), "rubber")
    b.tube("laundry_rail_top", [(-1.79, -1.10, 1.66), (-1.79, .42, 1.66)],
           .015, "chrome", collision=True)
    b.tube("laundry_rail_lower", [(-1.79, -1.10, .20), (-1.79, .42, .20)],
           .011, "laundry_ivory", collision=True)
    item("laundry_drying_rail", "freestanding_drying_rail", (-1.79, -.34, 0))
    for j, y in enumerate((-.91, -.60, -.22, .16)):
        name = f"laundry_wood_hanger_{j}"
        _hanger(b, name, -1.79, y, 1.63)
        item(name, "wood_hanger", (-1.79, y, 1.63))
        if j < 2:
            _shirt(b, f"laundry_hanging_shirt_{j}", (-1.79, y, 1.595),
                   ("laundry_cotton", "laundry_denim")[j], seed=j)
            item(f"laundry_hanging_shirt_{j}", "hanging_cotton_shirt", (-1.79, y, 1.595))
        else:
            _draped_towel(b, f"laundry_hanging_towel_{j}", (-1.79, y, 1.482), .40,
                           .61 if j == 2 else .50, palette[j - 1], seed=j)
            item(f"laundry_hanging_towel_{j}", "hanging_towel", (-1.79, y, 1.482))
    for j, y in enumerate((.28, .32, .36)):
        _hanger(b, f"laundry_spare_hanger_{j}", -1.79, y, 1.63)
        item(f"laundry_spare_hanger_{j}", "spare_hanger", (-1.79, y, 1.63))
    # A row of wooden clothespins and their steel torsion springs.
    for j in range(12):
        y = -.95 + j * .112
        name = f"laundry_clothespin_{j}"
        for side in (-1, 1):
            b.box(name + "_wood_" + str(side), (-1.79 + side * .006, y, 1.686),
                  (.008, .014, .062), "laundry_pegs", .002, rotation=(0, side * 7, 0),
                  collision=False)
        b.torus(name + "_spring", (-1.79, y, 1.681), .006, .0011,
                 "chrome", rotation=(90, 0, 0))
        item(name, "clothespin", (-1.79, y, 1.686))

    _innerscene_asset(b, "laundry_sorting_hamper", "air_hamper", (-2.10, -.30, 0))
    item("laundry_sorting_hamper", "wood_and_fabric_hamper", (-2.10, -.30, 0))
    for j in range(2):
        xyz = (-2.10, -.30, .6355 + j * .035)
        _folded_towel(b, f"laundry_hamper_textile_{j}", xyz, (.29, .23, .035),
                      ("laundry_linen_white", "laundry_linen_blush")[j], seed=j)
        item(f"laundry_hamper_textile_{j}", "folded_towel", xyz)
    # Wicker floor bin sits behind the rack and clear of the folding island.
    _source_asset(b, "laundry_floor_wicker", "wicker_basket_02", (-1.91, .66, 0.), scale=1.38, yaw=8)
    item("laundry_floor_wicker", "woven_laundry_basket", (-1.91, .66, 0.))
    for j in range(3):
        xyz = (-1.93 + (j - 1) * .08, .66, .18 + j * .024)
        _laundry_pile(b, f"laundry_floor_textile_{j}", xyz, palette[j], seed=j + 3, radius=.15)
        item(f"laundry_floor_textile_{j}", "gathered_laundry", xyz)

    inventory.append(_ironing_board(b))
    _source_asset(b, "laundry_ironing_spray", "multi_cleaner_bottle", (2.31, -.79, 0.), scale=1.0, yaw=20)
    item("laundry_ironing_spray", "ironing_spray", (2.31, -.79, 0.))
    _bottle(b, "laundry_starch", (2.34, -1.02, 0.), "laundry_ivory", "SPRAY\nSTARCH", height=.25)
    item("laundry_starch", "starch_bottle", (2.34, -1.02, 0.))

    # Wall-mounted care kit at the rear left, small practical items off counters.
    b.box("laundry_lint_caddy", (-.86, 2.11, 1.30), (.27, .15, .20),
          "laundry_sage", .009)
    b.label("laundry_lint_caddy_label", "LINT", (-.86, 2.031, 1.31), (.16, .045),
            rotation=(90, 0, 0), font_size=60)
    item("laundry_lint_caddy", "wall_lint_bin", (-.86, 2.11, 1.30))
    b.box("laundry_lint_roller_handle", (-.91, 2.05, 1.51), (.026, .026, .15),
          "laundry_caps", .008, collision=False)
    b.cylinder("laundry_lint_roller", (-.91, 2.05, 1.62), .034, .12,
               "paper", collision=False)
    item("laundry_lint_roller", "lint_roller", (-.91, 2.05, 1.51))
    b.box("laundry_care_brush", (-.79, 2.05, 1.54), (.035, .028, .17),
          "oak", .01, collision=False)
    item("laundry_care_brush", "garment_brush", (-.79, 2.05, 1.54))
    # Rubber-mat weave at washing machines supplies a lived-in surface detail.
    b.box("laundry_machine_mat", (-1.64, 1.12, .008), (1.32, .47, .016),
          "laundry_soft_gray", .013, collision=False)
    for i in range(31):
        b.tube(f"laundry_machine_mat_rib_{i}", [(-2.26 + i * .042, .90, .017),
                                                (-2.26 + i * .042, 1.34, .017)],
               .0022, "laundry_linen_gray")
    item("laundry_machine_mat", "woven_utility_mat", (-1.64, 1.12, .008))
    groups = Counter()
    for record in inventory:
        groups[record["class"]] += record["count"]
    return {"schema": "laundry_static_dressing.v1", "items": inventory,
            "semantic_object_count": sum(groups.values()), "groups": dict(groups),
            "bounds_m": [[-2.34, -1.91, 0], [2.40, 2.19, 2.56]],
            "reserved_empty_worktop_m": {"x": [-.1, 1.4], "y": [-.94, -.16], "z": .90},
            "scope": "fixed passive room dressing; no robot, joints or simulated cloth; no task qualification"}
