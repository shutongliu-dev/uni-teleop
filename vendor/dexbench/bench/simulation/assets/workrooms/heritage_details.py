"""Deterministic craft details for a furnished, task-independent Chinese room.

The garden is actual geometry, not a background image. Generated raster assets
are limited to material grain and ink on the hanging paper scrolls. They are
written beside each new room build and are part of its dependency inventory.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from pxr import UsdGeom, UsdPhysics


def timber_texture(output: Path) -> Path:
    """Quiet, fine walnut grain; no photographic furniture or scene content."""
    from PIL import Image

    output = Path(output) / 'heritage_walnut.png'
    yy, xx = np.mgrid[0:1024, 0:1024].astype(float) / 1024.
    warp = xx + .014 * np.sin(yy * math.tau) + .006 * np.sin(yy * math.tau * 3)
    grain = (2.8 * np.sin(warp * math.tau * 64)
             + 1.6 * np.sin(warp * math.tau * 137)
             + 3.2 * np.sin(warp * math.tau * 7)
             + 1.4 * np.sin(yy * math.tau * 2 + xx * 12))
    pores = np.random.default_rng(2217).normal(0., .8, xx.shape)
    color = np.array([133., 111., 87.]) + (grain + pores)[..., None]
    Image.fromarray(np.clip(color, 0, 255).astype('uint8')).save(output)
    return output


def wall_with_round_opening(b, *, center_x, center_z, radius, bounds, height):
    """A closed wall mesh around a genuine circular opening, with no backdrop."""
    xmin, xmax, _, back = bounds
    for suffix, x0, x1, z0, z1 in (
            ('left', xmin, center_x - radius, 0., height),
            ('right', center_x + radius, xmax, 0., height),
            ('below', center_x - radius, center_x + radius, 0., center_z - radius),
            ('above', center_x - radius, center_x + radius, center_z + radius, height)):
        b.box('wall_back_' + suffix, ((x0 + x1) / 2, back + .07, (z0 + z1) / 2),
              (x1 - x0, .14, z1 - z0), 'heritage_plaster', 0)
    # Fill the four corners of the square enclosing the aperture. Each ray
    # meets the square exactly; all inner vertices lie on the same true circle.
    vertices, faces = [], []
    n = 128
    for i in range(n):
        a = i * math.tau / n
        c, s = math.cos(a), math.sin(a)
        outer = radius / max(abs(c), abs(s))
        for y in (back, back + .14):
            for r in (radius, outer):
                vertices.append((center_x + r * c, y, center_z + r * s))
    for i in range(n):
        j, k = i * 4, ((i + 1) % n) * 4
        faces.extend(((j, k, k + 1, j + 1), (j + 2, j + 3, k + 3, k + 2),
                      (j, j + 2, k + 2, k), (j + 1, k + 1, k + 3, j + 3)))
    # On cardinal rays the circle touches its enclosing square. Merge those
    # coincident vertices and triangulate nonzero faces before PhysX can cook
    # this static triangle mesh; a convex approximation would close the hole.
    merged, index, remap = [], {}, []
    for vertex in vertices:
        key = tuple(round(float(value), 10) for value in vertex)
        if key not in index:
            index[key] = len(merged)
            merged.append(vertex)
        remap.append(index[key])
    triangles = []
    for face in faces:
        unique = list(dict.fromkeys(remap[i] for i in face))
        for i in range(1, len(unique) - 1):
            triangle = (unique[0], unique[i], unique[i + 1])
            a, c, d = (np.asarray(merged[j]) for j in triangle)
            if np.linalg.norm(np.cross(c - a, d - a)) > 1.e-12:
                triangles.append(triangle)
    path = b.path('wall_back_round_opening')
    prim = b.mesh(path, merged, triangles, 'heritage_plaster')
    UsdPhysics.CollisionAPI.Apply(prim)
    UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr('none')
    prim.SetCustomDataByKey('scence:architecturalFeature', 'open circular moon window')


def round_moulding(b, name, center, radius, width, depth, material='heritage_walnut'):
    r, w, d = radius, width, depth
    # Raised outer bead, flat face and a tiny sloping inner bevel catch daylight.
    return b.lathe(name, center, [(r, -d / 2), (r + w * .22, -d / 2 - .004),
                   (r + w * .82, -d / 2 - .004), (r + w, -d / 2 + .005),
                   (r + w, d / 2 - .005), (r + w * .80, d / 2),
                   (r + .006, d / 2), (r, d / 2 - .008)],
                   material, rotation=(90, 0, 0))


def ink_scroll_texture(b, kind='landscape') -> Path:
    """An original deterministic ink study on fibrous paper, for a real scroll."""
    from PIL import Image, ImageDraw, ImageFilter, ImageFont

    w, h = 768, 1536
    rng = np.random.default_rng(144 if kind == 'landscape' else 145)
    noise = rng.normal(0, 1.35, (h, w, 1))
    fibers = np.sin(np.arange(h)[:, None, None] * .71) * .5
    pixels = np.clip(np.array([231., 226., 208.]) + noise + fibers, 0, 255).astype('uint8')
    paper = Image.fromarray(pixels)
    ink = Image.new('RGBA', (w, h))
    draw = ImageDraw.Draw(ink)
    if kind == 'landscape':
        # Each wash disperses into the paper: no rectangular polygon bottoms
        # or hard framing edges. Fibre noise gives the ink a dry-brush edge.
        xx, yy = np.meshgrid(np.arange(w), np.arange(h))
        edge = np.clip(np.minimum(xx - 28, w - 28 - xx) / 43., 0, 1)
        for ridge, alpha, base in ((0, 39, 694), (1, 61, 872), (2, 81, 1068)):
            curve = (base - 135 * np.sin(np.arange(w) / 147 + ridge * 1.5)
                     - 55 * np.sin(np.arange(w) / 61 + ridge * .6)
                     - 15 * np.sin(np.arange(w) / 19 + ridge)
                     - 4 * np.sin(np.arange(w) / 6.3 + ridge))
            distance = yy - curve[None, :]
            coverage = np.clip((distance + 3.) / 6., 0, 1)
            wash = coverage * np.exp(-np.maximum(distance, 0) / (96 + 18 * ridge))
            wash *= edge * np.clip(1 + rng.normal(0, .10, (h, w)), .5, 1.3)
            layer = np.zeros((h, w, 4), dtype='uint8')
            layer[..., :3] = (45, 63, 53)
            layer[..., 3] = np.clip(wash * alpha, 0, 255).astype('uint8')
            ink = Image.alpha_composite(ink, Image.fromarray(layer))
        draw = ImageDraw.Draw(ink)
        # A tapered, gently curved foreground pine with loose needle clusters.
        trunk = [(153, 1260), (158, 1190), (147, 1124), (150, 1053), (166, 980)]
        for j in range(len(trunk) - 1):
            draw.line(trunk[j:j + 2], fill=(40, 54, 44, 175), width=9 - j * 2)
        for bx, by, ex, ey in ((153, 1191, 80, 1131), (148, 1122, 241, 1072),
                               (151, 1070, 99, 1032), (160, 1012, 213, 985)):
            draw.line([(bx, by), ((bx + ex) / 2, ey + 11), (ex, ey)],
                      fill=(40, 54, 44, 164), width=3)
            for j in range(58):
                px, py = ex + rng.normal(0, 25), ey + rng.normal(0, 8)
                a, length = rng.uniform(-2.8, -.3), rng.uniform(5, 14)
                draw.line([(px, py), (px + math.cos(a) * length, py + math.sin(a) * length)],
                          fill=(39, 57, 44, int(rng.integers(65, 150))), width=2)
        # Broken shore strokes leave a broad, misty river between the ridges.
        for j in range(23):
            x = 60 + j * 20
            y = 1270 - 24 * math.sin(x / 130)
            draw.line([(x, y), (x + rng.uniform(7, 19), y + rng.uniform(-2, 3))],
                      fill=(48, 63, 51, int(rng.integers(25, 70))), width=1)
        ink = ink.filter(ImageFilter.GaussianBlur(.7))
    else:
        # Bamboo study: thin curved culms, nodes, and grouped brush leaves.
        for stem in range(3):
            base = (235 + stem * 137, 1320)
            end = (base[0] + (-40, 35, 90)[stem], 320 + stem * 95)
            draw.line([base, end], fill=(42, 63, 48, 177), width=8 - stem)
            for t in np.linspace(.16, .85, 6):
                x = base[0] + (end[0] - base[0]) * t
                y = base[1] + (end[1] - base[1]) * t
                draw.line([(x - 7, y), (x + 8, y - 2)], fill=(33, 52, 38, 210), width=3)
                side = -1 if (int(t * 20) + stem) % 2 else 1
                ex, ey = x + side * 130, y - 70
                draw.line([(x, y), (ex, ey)], fill=(46, 65, 50, 163), width=2)
                for j in range(8):
                    q = .20 + .10 * j
                    bx, by = x + (ex - x) * q, y + (ey - y) * q
                    dx, dy = side * (30 + rng.random() * 24), -38 + rng.random() * 68
                    draw.polygon([(bx, by), (bx + dx * .55 - 7, by + dy * .4),
                                  (bx + dx, by + dy), (bx + dx * .45 + 6, by + dy * .7)],
                                 fill=(33, 59, 41, int(rng.integers(140, 215))))
    paper = Image.alpha_composite(paper.convert('RGBA'), ink)
    draw = ImageDraw.Draw(paper)
    font_path = b.assets_root / 'fonts/noto_cjk_sc/v1/NotoSansCJKsc-Regular.otf'
    font = ImageFont.truetype(str(font_path), 36)
    title = '山静日长' if kind == 'landscape' else '竹影清风'
    for i, letter in enumerate(title):
        draw.text((w - 119, 134 + i * 47), letter, font=font, fill=(55, 65, 55, 220))
    seal_x, seal_y = w - 109, 345
    draw.rounded_rectangle((seal_x, seal_y, seal_x + 30, seal_y + 31), radius=2,
                           outline=(143, 60, 44, 190), width=3)
    for xx, yy in ((4, 6), (14, 5), (6, 18), (17, 16)):
        draw.line([(seal_x + xx, seal_y + yy), (seal_x + xx + 7, seal_y + yy),
                   (seal_x + xx + 7, seal_y + yy + 7)], fill=(143, 60, 44, 190), width=2)
    path = b.generated / ('heritage_' + kind + '.png')
    paper.convert('RGB').save(path)
    return path


def hanging_scroll(b, name, y, z, width, height, *, kind='landscape', x=-1.746):
    """Paper and woven mounting with turned rollers, facing into the room."""
    b.box(name + '_silk_mount', (x, y, z), (.008, width, height),
          'heritage_silk', .001, collision=False)
    b.box(name + '_paper', (x + .005, y, z + .018), (.003, width * .81, height * .76),
          'heritage_paper', .0005, collision=False)
    texture = ink_scroll_texture(b, kind)
    material = b.material(name + '_ink', (1, 1, 1), .93, texture=str(texture))
    iw, ih = width * .81, height * .76
    path = b.path(name + '_painting')
    prim = b.mesh(path, [(x + .007, y - iw / 2, z - ih / 2 + .018),
                        (x + .007, y + iw / 2, z - ih / 2 + .018),
                        (x + .007, y + iw / 2, z + ih / 2 + .018),
                        (x + .007, y - iw / 2, z + ih / 2 + .018)],
                  [(0, 1, 2, 3)], material, [(0, 0), (1, 0), (1, 1), (0, 1)])
    prim.SetCustomDataByKey('scence:artProvenance', 'original procedural ink study on paper')
    for i, zz in enumerate((z - height / 2, z + height / 2)):
        b.cylinder(name + '_roller_' + str(i), (x + .008, y, zz), .014,
                   width + .045, 'heritage_walnut', rotation=(90, 0, 0), collision=False)
        for side in (-1, 1):
            b.sphere(name + f'_roller_cap_{i}_{side}',
                     (x + .008, y + side * (width / 2 + .027), zz),
                     (.016, .010, .016), 'heritage_walnut')
    b.tube(name + '_hanging_cord', [(x, y - width * .30, z + height / 2),
           (x, y, z + height / 2 + .14), (x, y + width * .30, z + height / 2)],
           .0014, 'heritage_cord')


def bamboo(b, name, center, height=2.3, *, seed=8, count=5):
    """Tapered segmented stems and a batched canopy of curved lance leaves."""
    rng = np.random.default_rng(seed)
    vertices, faces, normals = [], [], []
    for stem in range(count):
        x, y, z = np.array(center) + (rng.uniform(-.28, .28), rng.uniform(-.12, .12), 0)
        h = height * rng.uniform(.77, 1.04)
        lean_x, lean_y = rng.uniform(-.22, .22), rng.uniform(-.13, .13)
        points = [(x + lean_x * t * t, y + lean_y * t * t, z + h * t)
                  for t in np.linspace(0, 1, 8)]
        b.tube(name + '_culm_' + str(stem), points, rng.uniform(.011, .018), 'heritage_bamboo')
        for k, t in enumerate(np.linspace(.12, .95, 8)):
            bx, by, bz = x + lean_x * t * t, y + lean_y * t * t, z + h * t
            b.cylinder(name + f'_node_{stem}_{k}', (bx, by, bz), .019, .011,
                       'heritage_bamboo_node', collision=False)
            if t < .35:
                continue
            angle = rng.uniform(0, math.tau)
            reach = rng.uniform(.20, .41)
            endpoint = np.array((bx + reach * math.cos(angle), by + reach * math.sin(angle),
                                 bz + rng.uniform(.08, .23)))
            base = np.array((bx, by, bz))
            b.tube(name + f'_twig_{stem}_{k}', [base, (base + endpoint) / 2 + (0, 0, .025),
                   endpoint], .0022, 'heritage_bamboo')
            for leaf in range(9):
                tleaf = .28 + leaf * .079
                origin = base * (1 - tleaf) + endpoint * tleaf
                a = angle + (-1 if leaf % 2 else 1) * rng.uniform(.48, 1.14)
                length = rng.uniform(.085, .155)
                direction = np.array((math.cos(a), math.sin(a), rng.uniform(-.60, .25)))
                side = np.array((-math.sin(a), math.cos(a), .14)) * rng.uniform(.009, .015)
                mid = origin + direction * length * .43 + (0, 0, .012)
                tip = origin + direction * length
                v = [origin, mid + side, tip, mid - side, mid + (0, 0, .003)]
                start = len(vertices); vertices.extend(v)
                faces.extend((start + a0, start + a1, start + a2)
                             for a0, a1, a2 in ((0, 1, 4), (1, 2, 4), (2, 3, 4), (3, 0, 4)))
                normals.extend([(0, 0, 1)] * 5)
    b.mesh(b.path(name + '_leaf_canopy'), vertices, faces, 'heritage_leaf', normals=normals)


def hexagonal_lantern(b, name, center, *, radius=.16, height=.40):
    """A small six-sided silk lantern with real timber ribs and brass joints."""
    x, y, z = center
    vertices, faces = [], []
    for i in range(6):
        a0, a1 = math.tau * i / 6, math.tau * (i + 1) / 6
        p0 = (x + radius * math.cos(a0), y + radius * math.sin(a0))
        p1 = (x + radius * math.cos(a1), y + radius * math.sin(a1))
        start = len(vertices)
        vertices.extend([(p0[0], p0[1], z - height / 2), (p1[0], p1[1], z - height / 2),
                         (p1[0], p1[1], z + height / 2), (p0[0], p0[1], z + height / 2)])
        faces.append(tuple(range(start, start + 4)))
        b.tube(name + '_rib_' + str(i), [(p0[0], p0[1], z - height / 2 - .014),
               (p0[0], p0[1], z + height / 2 + .014)], .008, 'heritage_walnut')
    b.mesh(b.path(name + '_silk_shade'), vertices, faces, 'heritage_lantern_silk')
    for k, zz in enumerate((z - height / 2, z + height / 2)):
        points = [(x + radius * math.cos(a), y + radius * math.sin(a), zz)
                  for a in np.linspace(0, math.tau, 7)]
        b.tube(name + '_rim_' + str(k), points, .012, 'heritage_walnut')
    b.cylinder(name + '_cap', (x, y, z + height / 2 + .024), .038, .025,
               'heritage_brass', collision=False)
    b.tube(name + '_suspension', [(x, y, z + height / 2 + .036), (x, y, 2.79)],
           .0028, 'heritage_cord')
    b.tube(name + '_silk_tassel', [(x, y, z - height / 2), (x, y, z - height / 2 - .11)],
           .006, 'heritage_cinnabar')
