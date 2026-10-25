"""Grounded, source-backed furniture for the contemporary living-room release.

The window-side sofa faces the media wall. The one-metre work surface and its
approach are deliberately free of dressing; these are fixed room furnishings,
not movable task objects. Referenced Poly Haven sources are never rewritten.
"""
import math
from pathlib import Path

import numpy as np
from pxr import Gf, Usd, UsdGeom

from bench.paths import load_json


def _asset(b, name, slug, position, *, yaw=0.0, scale=1.0):
    root = b.assets_root / "objects/polyhaven/derived/v1" / slug
    derivation = root / "DERIVATION.json"
    metadata = load_json(derivation)
    # Include provenance and every referenced texture in the release inventory,
    # in addition to Builder.asset's hash of the exact frozen geometry layer.
    for source in (derivation, metadata["source"],
                   metadata["source_provenance"],
                   Path(metadata["source_provenance"]).with_name("LICENSE.txt"),
                   *metadata["texture_files"]):
        b.dependency(source)
    return b.asset(name, root / (slug + ".usdc"), position,
                   rotation=(0, 0, yaw), scale=scale)


def _materials(b):
    cloth = b.assets_root / "materials/polyhaven/architecture_v1/terlenka"
    b.dependency(cloth / "SOURCE.json")
    # Preserve the CC0 source maps, including their photographed thread colour
    # variation. UVs on the soft goods use the measured source tile dimensions.
    normal = str(cloth / "terlenka_nor_gl_1k.jpg")
    rough = str(cloth / "terlenka_rough_1k.jpg")
    b.material("living_cushion_flax", (.69, .63, .51), .93,
               texture=str(cloth / "terlenka_diff_1k.jpg"),
               normal=normal, roughness_texture=rough, texture_tint=(.97, .94, .85))
    linen = b.assets_root / "materials/polyhaven/v1/rough_linen"
    wool = b.assets_root / "materials/polyhaven/v1/wool_boucle"
    for source in (linen, wool):
        b.dependency(source / "SOURCE.json")
    b.material("living_cushion_sage", (.23, .30, .24), .94,
               texture=str(linen / "rough_linen_diff_1k.jpg"),
               normal=str(linen / "rough_linen_nor_gl_1k.jpg"),
               roughness_texture=str(linen / "rough_linen_rough_1k.jpg"),
               texture_tint=(.80, .94, .83))
    b.material("living_throw_wool", (.52, .49, .43), .98,
               texture=str(wool / "wool_boucle_diff_1k.jpg"),
               normal=str(wool / "wool_boucle_nor_gl_1k.jpg"),
               roughness_texture=str(wool / "wool_boucle_rough_1k.jpg"))
    b.material("living_rug_wool", (.51, .48, .41), .98,
               texture=str(cloth / "terlenka_diff_1k.jpg"), normal=normal,
               texture_tint=(.87, .85, .78))
    b.material("living_rug_binding", (.30, .29, .25), .97, normal=normal)
    b.material("living_rug_stripe", (.22, .25, .23), .98, normal=normal)
    b.material("living_basket_felt", (.20, .22, .20), .97,
               normal=str(linen / "rough_linen_nor_gl_1k.jpg"))
    b.material("living_seam", (.50, .45, .36), .95)
    b.material("living_black_anodized", (.016, .019, .021), .34, .65)
    b.material("living_screen_off", (.008, .012, .015), .11, .12)
    b.material("living_speaker_grille", (.028, .030, .029), .88,
               normal=normal)
    b.material("living_book_cover", (.20, .28, .26), .72)
    b.material("living_book_clay", (.38, .23, .16), .83)
    b.material("living_book_linen", (.61, .59, .51), .89, normal=normal)
    b.material("living_page_edge", (.66, .63, .54), .96)
    b.material("living_ceramic_sand", (.57, .51, .40), .33)
    b.material("living_ceramic_chalk", (.68, .67, .59), .73,
               normal=str(b.assets_root / "materials/ambientcg/v1/PaintedPlaster017/PaintedPlaster017_1K-JPG_NormalGL.jpg"))
    b.material("living_dried_stem", (.22, .17, .10), .91)


def _smooth_mesh(b, path, vertices, faces, material, uv):
    vertices = np.asarray(vertices, dtype=float)
    normals = np.zeros_like(vertices)
    for face in faces:
        p, q, r = vertices[list(face[:3])]
        normal = np.cross(q - p, r - p)
        for index in face:
            normals[index] += normal
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1.e-12)
    return b.mesh(path, vertices, faces, material, uv, normals)


def _cushion(b, name, center, size, material, *, yaw, lean=-12.0):
    """Two stuffed textile panels with pinched seams and asymmetric creases."""
    a, d, c = (float(value) / 2 for value in size)
    nlon, nr = 96, 22
    signed = lambda value: math.copysign(abs(value) ** .34, value)
    vertices, uv, faces = [], [], []
    for side in (-1, 1):
        start = len(vertices)
        vertices.append((0., side * d, 0.)); uv.append((a / .266, c / .266))
        for j in range(1, nr + 1):
            radius = j / nr
            for i in range(nlon):
                t = 2 * math.pi * i / nlon
                x = a * radius * signed(math.cos(t))
                z = c * radius * signed(math.sin(t))
                puff = d * max(0., 1 - radius ** 2) ** .66
                crease = .007 * math.sin(15 * t + radius * 13) * radius ** 4 * (1 - radius)
                depth = .004 + puff * (1 - .16 * radius * math.sin(3 * t + .6)) + crease
                vertices.append((x, side * depth, z + .004 * radius * math.cos(3 * t)))
                uv.append(((x + a) / .266, (z + c) / .266))
        for i in range(nlon):
            f = (start, start + 1 + i, start + 1 + (i + 1) % nlon)
            faces.append(f if side < 0 else f[::-1])
        for j in range(nr - 1):
            for i in range(nlon):
                k = start + 1 + j * nlon + i
                nxt = start + 1 + j * nlon + (i + 1) % nlon
                for f in ((k, k + nlon, nxt), (nxt, k + nlon, nxt + nlon)):
                    faces.append(f if side < 0 else f[::-1])
    ring1, ring2 = 1 + (nr - 1) * nlon, 2 + (2 * nr - 1) * nlon
    for i in range(nlon):
        j = (i + 1) % nlon
        faces.extend(((ring1 + i, ring2 + i, ring1 + j),
                      (ring1 + j, ring2 + i, ring2 + j)))
    path = b.path(name)
    root = UsdGeom.Xform.Define(b.stage, path)
    b.xform(root.GetPrim(), center, rotation=(lean, 0, yaw))
    _smooth_mesh(b, path + "/Visual", vertices, faces, material, uv)
    points = [(a * signed(math.cos(t)), 0,
               c * signed(math.sin(t)) + .004 * math.cos(3 * t))
              for t in np.linspace(0, 2 * math.pi, 97)]
    piping = b.tube(name + "_piping", points, .0012, "living_seam")
    b.xform(b.stage.GetPrimAtPath(piping), center, rotation=(lean, 0, yaw))
    return path


def _rug(b):
    # Thin woven pile rather than a floating slab. The seats stand on its 12 mm
    # surface; the room's task table still owns its original top elevation.
    b.box("living_seating_rug_binding", (.20, 1.22, .004),
          (3.64, 2.62, .008), "living_rug_binding", .004,
          collision=False)
    # Real-scale UVs keep the woven yarn visible; shallow surface undulations
    # and inset border bands avoid the appearance of a single beige board.
    vertices, uv, faces = [], [], []
    nx, ny = 96, 72
    for j in range(ny + 1):
        for i in range(nx + 1):
            x, y = -1.59 + i / nx * 3.58, -.06 + j / ny * 2.56
            z = .0113 + .0007 * math.sin(x * 33) * math.sin(y * 27)
            vertices.append((x, y, z)); uv.append((x / .266, y / .266))
    for j in range(ny):
        for i in range(nx):
            k = j * (nx + 1) + i
            faces.extend(((k, k + 1, k + nx + 2), (k, k + nx + 2, k + nx + 1)))
    _smooth_mesh(b, b.path("living_seating_rug_pile"), vertices, faces, "living_rug_wool", uv)
    for side in (-1, 1):
        b.box("living_rug_long_seam_" + str(side),
              (.20 + side * 1.775, 1.22, .0122),
              (.002, 2.50, .0004), "living_seam", 0,
              collision=False)
        for index, offset in enumerate((.07, .105, .17)):
            b.box(f"living_rug_border_{side}_{index}",
                  (.20, 1.22 + side * (1.28 - offset), .0121),
                  (3.48, .012 if index < 2 else .003, .00035),
                  "living_rug_stripe", .0001, collision=False)
        for index in range(94):
            x = -1.53 + index * .037
            y = 1.22 + side * 1.306
            b.tube(f"living_rug_fringe_{side}_{index}",
                   [(x, y, .007), (x + .003, y + side * .023, .006),
                    (x + .007 * math.sin(index), y + side * .044, .0055)],
                   .0013, "living_seam")


def _surface_sampler(stage, root_path):
    """Sample the real source upholstery rather than assume a flat armrest."""
    triangles = []
    cache = UsdGeom.XformCache()
    for prim in Usd.PrimRange(stage.GetPrimAtPath(root_path)):
        if not prim.IsA(UsdGeom.Mesh):
            continue
        mesh = UsdGeom.Mesh(prim)
        matrix = cache.GetLocalToWorldTransform(prim)
        points = np.asarray([matrix.Transform(Gf.Vec3d(*p))
                             for p in mesh.GetPointsAttr().Get()])
        indices = mesh.GetFaceVertexIndicesAttr().Get()
        offset = 0
        for count in mesh.GetFaceVertexCountsAttr().Get():
            face = indices[offset:offset + count]
            triangles.extend(points[[face[0], face[i], face[i + 1]]]
                             for i in range(1, count - 1))
            offset += count
    triangles = np.asarray(triangles)
    a = triangles[:, 0]; u = triangles[:, 1] - a; v = triangles[:, 2] - a
    det = u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]
    usable = np.abs(det) > 1.e-12
    a, u, v, det = a[usable], u[usable], v[usable], det[usable]

    def height(x, y):
        dx, dy = x - a[:, 0], y - a[:, 1]
        p = (dx * v[:, 1] - dy * v[:, 0]) / det
        q = (u[:, 0] * dy - u[:, 1] * dx) / det
        inside = (p >= -1.e-7) & (q >= -1.e-7) & (p + q <= 1 + 1.e-7)
        if not inside.any():
            raise ValueError(f"Throw has no sofa support at {(x, y)}")
        return float((a[:, 2] + p * u[:, 2] + q * v[:, 2])[inside].max())
    return height


def _throw(b):
    # This is static dressing, not cloth dynamics. Its supported section is
    # derived from the frozen sofa mesh; the rest hangs outside its near arm.
    surface = _surface_sampler(b.stage, "/World/Geometry/living_sofa")
    nx, ny = 48, 88
    vertices, uv, faces = [], [], []
    # The source's rolled arm has a curved plan edge, not a straight box edge.
    # Find the supporting crest separately for every longitudinal fabric row.
    edges = []
    for i in range(nx + 1):
        x = -1.15 + .48 * i / nx
        for y in np.linspace(.065, .20, 136):
            try:
                if surface(x, y) > .58:
                    edge = float(y) + .002
                    edges.append((edge, surface(x, edge)))
                    break
            except ValueError:
                continue
        else:
            raise ValueError("No rolled-arm edge for throw at x=" + str(x))
    for j in range(ny + 1):
        t = j / ny
        for i in range(nx + 1):
            s = i / nx; x = -1.15 + .48 * s
            edge, edge_z = edges[i]
            if t <= .60:
                y = .39 - t / .60 * (.39 - edge)
                z = surface(x, y) + .005
            else:
                tail = (t - .60) / .40
                y = edge - .165 * math.sin(tail * math.pi / 2)
                z = edge_z + .005 - .43 * tail
            # Small soft ridges vary down the hanging panel and along its hem.
            fold = .003 * (1 + math.sin(s * 8 * math.pi + t * 3))
            z += fold
            if t > .60:
                y -= .006 * math.sin(s * 8 * math.pi) * (t - .60) / .40
            vertices.append((x, y, z))
            uv.append((s * .48 / .3148, t * .91 / .3901))
    for j in range(ny):
        for i in range(nx):
            k = j * (nx + 1) + i
            faces.extend(((k, k + 1, k + nx + 2), (k, k + nx + 2, k + nx + 1)))
    _smooth_mesh(b, b.path("living_sofa_wool_throw"), vertices, [f[::-1] for f in faces],
                 "living_throw_wool", uv)
    for edge, indices in (("left", range(0, len(vertices), nx + 1)),
                          ("right", range(nx, len(vertices), nx + 1)),
                          ("hem", range(ny * (nx + 1), len(vertices)))):
        b.tube("living_throw_" + edge, [vertices[i] for i in indices],
               .0015, "living_seam")
    for index in range(25):
        x, y, z = vertices[ny * (nx + 1) + 2 * index]
        b.tube("living_throw_fringe_" + str(index),
               [(x, y, z), (x + .002, y - .002, z - .020),
                (x + .004 * math.sin(index), y - .004, z - .037)],
               .0012, "living_seam")


def _seating(b):
    _rug(b)
    # The original sofa's front is -Y. A +90 degree yaw makes it face +X,
    # directly toward the television instead of facing across it.
    _asset(b, "living_sofa", "sofa_02", (-.97, 1.15, .012),
           yaw=90, scale=1.20)
    _asset(b, "living_armchair", "modern_arm_chair_01", (1.38, 2.00, .012),
           yaw=45, scale=.88)
    # Source seat top is z=0.493 m after scaling. The relaxed pillow bottoms
    # settle into that seat by a few millimetres instead of hovering over it.
    _cushion(b, "living_sofa_flax_cushion", (-1.035, .53, .700),
             (.39, .15, .39), "living_cushion_flax", yaw=90, lean=-10)
    _cushion(b, "living_sofa_sage_cushion", (-1.030, 1.76, .690),
             (.36, .15, .37), "living_cushion_sage", yaw=90, lean=-14)
    _throw(b)


def _book(b, name, center, size, cover, *, yaw=0.):
    """A hardback has a recessed page block, overhanging covers and a spine."""
    x, y, z = center; width, depth, thick = size
    angle = math.radians(yaw)
    def position(dx, dy, dz):
        return (x + dx * math.cos(angle) - dy * math.sin(angle),
                y + dx * math.sin(angle) + dy * math.cos(angle), z + dz)
    b.box(name + "_pages", center, (width - .007, depth - .010, thick - .004),
          "paper", .001, rotation=(0, 0, yaw), collision=False)
    for side in (-1, 1):
        b.box(name + "_cover_" + str(side), position(0, 0, side * (thick / 2 - .001)),
              (width, depth, .002), cover, .0008,
              rotation=(0, 0, yaw), collision=False)
    b.box(name + "_spine", position(0, depth / 2 - .002, 0),
          (width, .004, thick), cover, .001,
          rotation=(0, 0, yaw), collision=False)
    for i in range(1, 5):
        b.box(name + "_page_edge_" + str(i),
              position(-width / 2 + .0033, -.002, -thick / 2 + .003 + i * (thick - .006) / 6),
              (.0005, depth - .013, .0002), "living_page_edge", 0,
              rotation=(0, 0, yaw), collision=False)


def _media(b, *, right_wall_x=3.2):
    wall_shift_x = right_wall_x - 3.2
    _asset(b, "living_console", "modern_wooden_cabinet", (2.86 + wall_shift_x, .85, 0),
           yaw=-90, scale=.82)
    top = .6801425069570542 * .82
    # A wall mount and cable route explain the screen's support. The picture is
    # intentionally off: it should reflect the room, never act as a fake light.
    b.box("living_tv_wall_plate", (3.173 + wall_shift_x, .85, 1.22),
          (.024, .34, .25), "living_black_anodized", .003)
    for index, y in enumerate((.60, 1.10)):
        b.box("living_tv_mount_arm_" + str(index), (3.144 + wall_shift_x, y, 1.22),
              (.060, .032, .17), "living_black_anodized", .003)
    b.box("television_frame", (3.10 + wall_shift_x, .85, 1.20),
          (.048, 1.36, .78), "living_black_anodized", .008)
    b.box("television_screen", (3.072 + wall_shift_x, .85, 1.20),
          (.006, 1.322, .741), "living_screen_off", .002,
          collision=False)
    b.box("living_tv_lower_sensor", (3.068 + wall_shift_x, .85, .814),
          (.002, .024, .006), "black", .001, collision=False)
    b.box("living_tv_cable_raceway", (3.174 + wall_shift_x, .85, .725),
          (.025, .033, .20), "wall", .003)
    b.box("living_media_soundbar", (2.803 + wall_shift_x, .85, top + .032),
          (.105, .74, .064), "living_black_anodized", .015)
    b.box("living_soundbar_grille", (2.748 + wall_shift_x, .85, top + .032),
          (.003, .704, .040), "living_speaker_grille", .001,
          collision=False)
    b.box("living_media_remote", (2.73 + wall_shift_x, .08, top + .008),
          (.045, .14, .016), "black", .006, rotation=(0, 0, -9))
    for index in range(4):
        b.sphere("living_remote_key_" + str(index),
                 (2.73 + wall_shift_x, .040 + index * .018, top + .017),
                 (.006, .006, .0015), "living_black_anodized")
    # Three slightly offset hardbacks and one low vase, clear of ventilation.
    height = top
    for index, (width, depth, thick, mat, yaw) in enumerate((
            (.24, .29, .026, "living_book_cover", -3),
            (.22, .28, .023, "living_book_clay", 2),
            (.21, .265, .020, "living_book_linen", -5))):
        _book(b, "living_console_book_" + str(index),
              (2.86 + wall_shift_x + index * .004, 1.66 - index * .005, height + thick / 2),
              (width, depth, thick), mat, yaw=yaw)
        height += thick
    b.lathe("living_console_vase", (2.905 + wall_shift_x, -.015, top),
            [(0, 0), (.047, 0), (.061, .021), (.067, .083),
             (.055, .125), (.028, .150), (.026, .185),
             (.021, .185), (.022, .151), (.049, .122),
             (.060, .081), (.054, .024), (.040, .009), (0, .009)],
            "living_ceramic_chalk", collision=False)
    for i in range(3):
        x = 2.905 + wall_shift_x + (i - 1) * .009; y = -.015
        dx, dy = (.012, -.025, .022)[i], (-.045, -.055, .012)[i]
        end = (x + dx, y + dy, top + .34 + i * .035)
        b.tube("living_vase_stem_" + str(i),
               [(x, y, top + .09), (x + dx * .35, y + dy * .35, top + .24), end],
               .0017, "living_dried_stem")
        for j in range(3):
            base_z = top + .25 + j * .035
            sign = -1 if j % 2 else 1
            b.tube(f"living_vase_twig_{i}_{j}",
                   [(x + dx * .5, y + dy * .5, base_z),
                    (x + dx + sign * .025, y + dy - .017, base_z + .026)],
                   .0009, "living_dried_stem")


def _reading_storage(b):
    # A low open felt book basket under the wall artwork is separate from the
    # robot approach, and does not add another large piece of room furniture.
    x, y, bottom = .05, 2.42, .005
    b.box("living_reading_basket_base", (x, y, bottom + .008),
          (.52, .29, .016), "living_basket_felt", .005, collision=False)
    for side in (-1, 1):
        b.box("living_reading_basket_end_" + str(side),
              (x + side * .263, y, bottom + .155), (.012, .31, .31),
              "living_basket_felt", .005, rotation=(0, side * 4, 0), collision=False)
        b.box("living_reading_basket_side_" + str(side),
              (x, y + side * .152, bottom + .155), (.53, .012, .31),
              "living_basket_felt", .005, rotation=(-side * 4, 0, 0), collision=False)
        # Loop handles are actual openings, rather than black painted slots.
        b.tube("living_reading_basket_handle_" + str(side),
               [(x + side * .278, y - .060, .315),
                (x + side * .285, y - .050, .355),
                (x + side * .285, y + .050, .355),
                (x + side * .278, y + .060, .315)],
               .005, "living_seam")
    # Upright page blocks, covers and spines sit on the basket's real base.
    for i, (height, mat) in enumerate(((.315, "living_book_linen"),
                                      (.280, "living_book_cover"),
                                      (.330, "living_book_clay"))):
        xx = x - .15 + i * .13
        b.box("living_basket_book_pages_" + str(i),
              (xx, y, .021 + height / 2), (.020, .205, height),
              "paper", .001, collision=False)
        for side in (-1, 1):
            b.box(f"living_basket_book_cover_{i}_{side}",
                  (xx + side * .0115, y, .021 + height / 2),
                  (.003, .213, height + .003), mat, .001, collision=False)
        b.box("living_basket_book_spine_" + str(i),
              (xx, y - .105, .021 + height / 2), (.025, .004, height + .003),
              mat, .001, collision=False)


def _wall_shelves(b):
    """Slim oak shelves with restrained reading-room dressing above the chair."""
    for index, z in enumerate((1.48, 1.96)):
        b.box("living_wall_shelf_" + str(index), (1.83, 2.58, z),
              (1.05, .20, .032), "oak", .004)
        # Concealed rear steel shoes carry the shelf into the plaster wall.
        for j, x in enumerate((1.47, 2.19)):
            b.box(f"living_shelf_wall_fixing_{index}_{j}", (x, 2.688, z),
                  (.055, .024, .025), "living_black_anodized", .001,
                  collision=False)
    # A short group of books on the lower shelf, not a fully stocked bookcase.
    for i, (width, height, material) in enumerate((
            (.032, .245, "living_book_linen"), (.025, .215, "living_book_cover"),
            (.034, .235, "living_book_clay"))):
        x = 1.405 + i * .040; z = 1.496 + height / 2
        b.box("living_shelf_book_pages_" + str(i), (x, 2.578, z),
              (width - .004, .143, height - .004), "paper", .001, collision=False)
        for side in (-1, 1):
            b.box(f"living_shelf_book_cover_{i}_{side}",
                  (x + side * (width / 2 - .001), 2.58, z),
                  (.002, .152, height), material, .0007, collision=False)
        b.box("living_shelf_book_spine_" + str(i), (x, 2.503, z),
              (width, .004, height), material, .001, collision=False)
        for j in range(2):
            b.box(f"living_shelf_spine_rule_{i}_{j}", (x, 2.5005, z + height * (.27 - j * .54)),
                  (width * .63, .0005, .0015), "living_seam", 0, collision=False)
    b.lathe("living_shelf_shallow_bowl", (2.13, 2.58, 1.496),
            [(0, 0), (.060, 0), (.081, .013), (.086, .042),
             (.082, .045), (.076, .016), (.056, .008), (0, .008)],
            "living_ceramic_sand")
    _book(b, "living_upper_shelf_book", (1.53, 2.58, 1.986),
          (.26, .16, .020), "living_book_cover", yaw=-3)
    b.lathe("living_shelf_small_vessel", (2.07, 2.58, 1.976),
            [(0, 0), (.049, 0), (.063, .047), (.058, .105),
             (.027, .151), (.024, .173), (.019, .173),
             (.022, .150), (.052, .102), (.056, .047), (.043, .009), (0, .009)],
            "living_ceramic_chalk")


def _side_table(b):
    x, y, top = -1.15, -.32, .530
    # A slender pedestal keeps three solid support surfaces visibly connected.
    b.cylinder("living_side_table_foot", (x, y, .011), .175, .022,
               "living_black_anodized")
    b.cylinder("living_side_table_stem", (x, y, .266), .025, .488,
               "living_black_anodized")
    b.lathe("living_side_table_top", (x, y, top - .025),
            [(0, 0), (.236, 0), (.245, .008), (.245, .017),
             (.236, .025), (0, .025)], "walnut", collision=True)
    b.box("living_side_book_pages", (x - .040, y + .025, top + .010),
          (.165, .215, .020), "paper", .001)
    for index, z in enumerate((top + .001, top + .021)):
        b.box("living_side_book_cover_" + str(index),
              (x - .040, y + .025, z), (.172, .222, .002),
              "living_book_cover", .001)
    b.label("living_side_book_title", "SLOW\nLIVING",
            (x - .040, y + .025, top + .0222), (.118, .105),
            font_size=82)
    # The ceramic cup has a modeled open cavity and a continuous handle.
    mx, my = x + .118, y - .095
    b.lathe("living_side_cup", (mx, my, top),
            [(0, 0), (.028, 0), (.032, .007), (.036, .082),
             (.035, .087), (.031, .087), (.028, .012), (0, .012)],
            "living_ceramic_sand", collision=True)
    b.tube("living_side_cup_handle",
           [(mx + .031 + .030 * math.sin(t), my,
             top + .045 + .025 * math.cos(t))
            for t in np.linspace(0, math.pi, 25)],
           .004, "living_ceramic_sand", collision=True)


def build_furnishings(b, *, right_wall_x=3.2):
    """Add fixed furnishings without architecture, light rigs, or task props."""
    _materials(b)
    _seating(b)
    _media(b, right_wall_x=right_wall_x)
    _side_table(b)
    _reading_storage(b)
    _wall_shelves(b)
    plant = _asset(b, "living_plant", "potted_plant_01", (right_wall_x - .54, 2.27, 0),
                   yaw=-15, scale=.85)
    # This published source has one pebble below the ceramic base. Rather than
    # lowering that frozen mesh through the floor, a real saucer supports the
    # pot at its measured base elevation (about 11.65 mm above the floor).
    pot = b.stage.GetPrimAtPath(plant + "/Asset/node_2/mesh_0")
    bounds = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"])
    extent = bounds.ComputeWorldBound(pot).ComputeAlignedRange()
    low, high = extent.GetMin(), extent.GetMax()
    base = float(low[2])
    b.lathe("living_plant_saucer",
            ((low[0] + high[0]) / 2, (low[1] + high[1]) / 2, 0),
            [(0, 0), (.192, 0), (.205, .005), (.205, .023),
             (.197, .023), (.190, base), (0, base)],
            "living_ceramic_sand", collision=True)
    placements = {
        "living_sofa": (90., 1.20),
        "living_armchair": (45., .88),
        "living_console": (-90., .82),
        "living_plant": (-15., .85),
    }
    furniture = []
    for record in b.records:
        if record["name"] not in placements:
            continue
        yaw, scale = placements[record["name"]]
        source = record["source"]
        furniture.append({
            "name": record["name"],
            "prim_path": "/World/Geometry/" + record["name"],
            "position_m": record["position"], "yaw_deg": yaw,
            "scale": scale, "bounds_size_m": record["bounds_size"],
            "source_usd": source,
            "source_sha256": b.dependencies[str(Path(source).resolve())]["sha256"],
            "derivation_manifest": str(Path(source).with_name("DERIVATION.json")),
            "license": "CC0-1.0", "source_version": "polyhaven/derived/v1",
        })
    return {
        "schema": "scence.living_furnishings.v3",
        "layout": furniture,
        "sofa_facing_world": [1., 0., 0.],
        "television_center_m": [right_wall_x - .10, .85, 1.20],
        "television_screen_faces_world": [-1., 0., 0.],
        "support_heights_m": {
            "bare_floor_authoring_datum": 0., "seating_rug_top": .012,
            "side_table_top": .530, "media_console_top": .6801425069570542 * .82,
            "plant_saucer_contact": base,
        },
        "soft_goods": {
            "cushions": "Paired stuffed panels with pinched 8 mm seam, creases and real-scale fabric UVs",
            "throw": {"source_texture": "polyhaven/v1/wool_boucle", "license": "CC0-1.0",
                      "tile_size_m": [.3148, .3901],
                      "placement": "Supported section samples frozen sofa triangles; hanging near-arm hem",
                      "physics": "Authored static drape; no cloth simulation claim"},
            "rug": {"surface_z_range_m": [.0106, .012],
                    "weave_tile_size_m": [.2657, .2663],
                    "details": ["inset triple border", "188 individual fringe strands", "shallow woven surface"]},
        },
        "dressing": ["three layered hardbacks with recessed pages", "open ceramic vase and dried twigs",
                     "low felt reading basket with loop handles and upright books",
                     "two thin wall-mounted oak shelves with upright books and open ceramics"],
        "reserved_robot_approach": {"x_m": [.45, .85], "y_m": [-1.06, -.70], "added_dressing": False},
        "scope": "Fixed scenery; geometric placement only, not dynamic task qualification",
    }
