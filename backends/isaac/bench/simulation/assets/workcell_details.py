"""Original visual meshes for a detailed, cloth-safe robotics workcell.

Recipes and provenance live under shared assets. No dynamic bodies, particles,
collision shapes or state-update hooks are created by this module.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from bench.paths import DATA_ROOT

DEFAULT_ASSET_ROOT = Path(f"{DATA_ROOT}/assets/scenes/workcell_details/v1")


def _normals(vertices, faces):
    vertices, faces = np.asarray(vertices), np.asarray(faces, dtype=np.int32)
    result = np.zeros_like(vertices, dtype=float)
    tri = vertices[faces]
    face_normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    for column in range(3):
        np.add.at(result, faces[:, column], face_normals)
    lengths = np.linalg.norm(result, axis=1)
    result /= np.maximum(lengths[:, None], 1e-15)
    return result


def _rotation(axis, angle):
    axis = np.asarray(axis, dtype=float)
    axis /= np.linalg.norm(axis)
    x, y, z = axis
    cross = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return np.eye(3) * math.cos(angle) + (1 - math.cos(angle)) * np.outer(axis, axis) + math.sin(angle) * cross


def _ring_points(hx, hy, radius, count=5):
    radius = min(radius, hx, hy)
    result = []
    for cx, cy, start in ((hx-radius, hy-radius, 0), (-hx+radius, hy-radius, 90),
                          (-hx+radius, -hy+radius, 180), (hx-radius, -hy+radius, 270)):
        for angle in np.linspace(start, start + 90, count, endpoint=False):
            a = math.radians(angle)
            result.append([cx + radius * math.cos(a), cy + radius * math.sin(a)])
    return np.asarray(result)


def _loft(rings, caps=True):
    """Closed consistent CCW ring loft, with correctly oriented cap triangles."""
    n = len(rings[0])
    vertices = np.concatenate(rings).tolist()
    faces = []
    for k in range(len(rings)-1):
        for j in range(n):
            a, b = k*n+j, k*n+(j+1) % n
            faces.extend([[a, b, b+n], [a, b+n, a+n]])
    if caps:
        for k, reverse in ((0, True), (len(rings)-1, False)):
            center = len(vertices)
            vertices.append(np.mean(rings[k], axis=0).tolist())
            for j in range(n):
                a, b = k*n+j, k*n+(j+1) % n
                faces.append([center, b, a] if reverse else [center, a, b])
    return np.asarray(vertices), np.asarray(faces, dtype=np.int32)


class _Assembly:
    """Keep curved surface normals, while merging material-compatible meshes."""
    def __init__(self, recipe, palette):
        self.recipe = recipe
        self.name = recipe["name"]
        self.origin = np.asarray(recipe.get("position_m", [0, 0, 0]), dtype=float)
        self.rotation = _rotation([0, 0, 1], math.radians(recipe.get("yaw_deg", 0)))
        self.palette = palette
        self.parts = {}

    def mesh(self, vertices, faces, color, center=(0, 0, 0), rotation=None):
        vertices, faces = np.asarray(vertices, dtype=float), np.asarray(faces, dtype=np.int32)
        normals = _normals(vertices, faces)
        if rotation is not None:
            vertices, normals = vertices @ rotation.T, normals @ rotation.T
        vertices = (vertices + center) @ self.rotation.T + self.origin
        normals = normals @ self.rotation.T
        part = self.parts.setdefault(color, {"vertices": [], "faces": [], "normals": [], "count": 0})
        part["vertices"].append(vertices)
        part["faces"].append(faces + part["count"])
        part["normals"].append(normals)
        part["count"] += len(vertices)

    def box(self, center, size, color, bevel=.002, rotation=None):
        hx, hy, hz = np.asarray(size, dtype=float) / 2
        b = min(bevel, hx*.45, hy*.45, hz*.45)
        rings = []
        for z, inset in ((-hz, b), (-hz+b, 0), (hz-b, 0), (hz, b)):
            ring = _ring_points(hx-inset, hy-inset, max(.00005, bevel-inset))
            rings.append(np.c_[ring, np.full(len(ring), z)])
        vertices, faces = _loft(rings)
        self.mesh(vertices, faces, color, center, rotation)

    def lathe(self, center, profile, color, segments=48, rotation=None, hex_outer=False):
        """Profile is (radius, height); may enclose an open vessel or ring."""
        angles = np.arange(segments) * 2*math.pi/segments
        rings = []
        outer = max(p[0] for p in profile)
        for r, z in profile:
            radii = np.full(segments, max(r, 1e-8))
            if hex_outer and r > outer*.78:
                radii *= math.cos(math.pi/6) / np.cos((angles % (math.pi/3)) - math.pi/6)
            rings.append(np.c_[radii*np.cos(angles), radii*np.sin(angles), np.full(segments, z)])
        # Revolving a closed boundary explicitly also closes the inner wall.
        vertices, faces = _loft(rings + [rings[0]], caps=False)
        self.mesh(vertices, faces, color, center, rotation)

    def cylinder(self, center, radius, height, color, rotation=None, segments=48):
        h, b = height/2, min(.0015, radius*.12, height*.15)
        self.lathe(center, [(0, -h), (radius-b, -h), (radius, -h+b), (radius, h-b),
                            (radius-b, h), (0, h)], color, segments, rotation)

    def tube(self, points, radius, color, closed=False, segments=12):
        points = np.asarray(points, dtype=float)
        tangents = np.roll(points, -1, axis=0) - np.roll(points, 1, axis=0)
        if not closed:
            tangents[0], tangents[-1] = points[1]-points[0], points[-1]-points[-2]
        tangents /= np.maximum(np.linalg.norm(tangents, axis=1)[:, None], 1e-15)
        rings = []
        # Parallel transport prevents flips along the curved cable.
        reference = np.array([0., 0., 1.])
        if abs(np.dot(reference, tangents[0])) > .9:
            reference = np.array([0., 1., 0.])
        basis = np.cross(tangents[0], reference)
        for point, tangent in zip(points, tangents):
            basis = basis - np.dot(basis, tangent)*tangent
            basis /= max(np.linalg.norm(basis), 1e-15)
            other = np.cross(tangent, basis)
            angles = np.arange(segments) * 2*math.pi/segments
            rings.append(point + radius*(np.cos(angles)[:, None]*basis + np.sin(angles)[:, None]*other))
        vertices, faces = _loft(rings + [rings[0]] if closed else rings, caps=not closed)
        self.mesh(vertices, faces, color)

    def torus(self, center, radius, tube_radius, color, rotation=None, arc=(0, 2*math.pi)):
        angles = np.linspace(*arc, 64, endpoint=arc != (0, 2*math.pi))
        points = np.c_[radius*np.cos(angles), radius*np.sin(angles), np.zeros(len(angles))]
        if rotation is not None:
            points = points @ rotation.T
        self.tube(points + center, tube_radius, color, closed=arc == (0, 2*math.pi), segments=10)

    def prism(self, points, center, height, color):
        # The tool polygons are simple, star-shaped silhouettes with a fan-safe
        # center selected by the recipe, or convex rectangular jaws.
        points = np.asarray(points, dtype=float)
        rings = [np.c_[points, np.full(len(points), z)] for z in (-height/2, height/2)]
        vertices, faces = _loft(rings)
        self.mesh(vertices, faces, color, center)


# Original compact glyphs: each 3x5 pattern is authored as mesh pixels.
_GLYPHS = {
    'A': ['010','101','111','101','101'], 'B': ['110','101','110','101','110'],
    'C': ['011','100','100','100','011'], 'D': ['110','101','101','101','110'],
    'E': ['111','100','110','100','111'], 'F': ['111','100','110','100','100'],
    'G': ['011','100','101','101','011'], 'H': ['101','101','111','101','101'],
    'I': ['111','010','010','010','111'], 'K': ['101','101','110','101','101'],
    'L': ['100','100','100','100','111'], 'M': ['101','111','111','101','101'],
    'N': ['101','111','111','111','101'], 'O': ['010','101','101','101','010'],
    'P': ['110','101','110','100','100'], 'R': ['110','101','110','101','101'],
    'S': ['011','100','010','001','110'], 'T': ['111','010','010','010','010'],
    'U': ['101','101','101','101','111'], 'V': ['101','101','101','101','010'],
    'W': ['101','101','111','111','101'], 'Y': ['101','101','010','010','010'],
    '0': ['111','101','101','101','111'], '1': ['010','110','010','010','111'],
    '2': ['110','001','010','100','111'], '3': ['110','001','010','001','110'],
    '4': ['101','101','111','001','001'], '5': ['111','100','110','001','110'],
    '6': ['011','100','111','101','111'], '7': ['111','001','010','010','010'],
    '8': ['111','101','111','101','111'], '9': ['111','101','111','001','110'],
    '-': ['000','000','111','000','000'], ' ': ['000']*5,
}


def _text(a, text, center, height, color="dark", face="front"):
    cell = height/5
    width = max(0, len(text)*4-1)*cell
    cx, cy, cz = center
    for k, char in enumerate(text.upper()):
        for row, line in enumerate(_GLYPHS.get(char, _GLYPHS[' '])):
            for column, bit in enumerate(line):
                if bit == '1':
                    x = cx-width/2+(k*4+column+.5)*cell
                    v = (2-row)*cell
                    if face == "front":
                        a.box([x, cy, cz+v], [cell*.86, .00035, cell*.86], color, bevel=.00004)
                    else:
                        a.box([x, cy+v, cz], [cell*.86, cell*.86, .00035], color, bevel=.00004)


def _bin(a, center, size, color, label):
    x, y, z = center
    sx, sy, sz = size
    wall = .006
    a.box([x, y, z+wall/2], [sx, sy, wall], color, .003)
    for sign in (-1, 1):
        a.box([x+sign*(sx-wall)/2, y, z+sz/2], [wall, sy, sz], color, .003)
        # Front/rear walls have a real hand-grip opening, bridged above.
        yy = y+sign*(sy-wall)/2
        a.box([x, yy, z+sz*.33], [sx, wall, sz*.66], color, .003)
        a.box([x, yy, z+sz-.008], [sx, wall+.002, .016], color, .003)
        for side in (-1, 1):
            a.box([x+side*sx*.37, yy, z+sz*.80], [sx*.26, wall, sz*.28], color, .003)
        # Raised rim follows the rectangular container mouth.
        a.box([x+sign*(sx-.004)/2, y, z+sz], [.010, sy+.008, .007], color, .002)
        a.box([x, y+sign*(sy-.004)/2, z+sz], [sx+.008, .010, .007], color, .002)
    a.box([x, y-sy/2-.001, z+sz*.38], [sx*.68, .002, sz*.24], "paper", .001)
    _text(a, label, [x, y-sy/2-.0022, z+sz*.38], min(sz*.15, sx*.52/max(1,len(label))*.9))
    # Four molded vertical ribs on the visible face.
    for xx in (x-sx*.43, x+sx*.43):
        a.box([xx, y-sy/2-.002, z+sz*.36], [.005, .004, sz*.58], color, .001)


def _washer(a, center, outer=.010, inner=.0045, height=.0018, color="silver", hexagonal=False):
    b = min(.0006, height*.2)
    a.lathe(center, [(inner, 0), (outer-b, 0), (outer, b), (outer, height-b),
                     (outer-b, height), (inner, height)], color, 48 if not hexagonal else 60,
            hex_outer=hexagonal)


def _mug(a):
    a.lathe([0, 0, 0], [(0, 0), (.027, 0), (.032, .004), (.035, .084),
                        (.035, .095), (.033, .098), (.0305, .095), (.030, .087),
                        (.027, .013), (0, .013)], "ceramic", 72)
    a.torus([0, 0, .096], .0325, .002, "cream")
    t = np.linspace(0, math.pi, 40)
    a.tube(np.c_[.030+.035*np.sin(t), np.zeros(len(t)), .052+.032*np.cos(t)], .006, "ceramic", segments=16)
    a.lathe([0, 0, 0], [(.032, .030), (.033, .030), (.0335, .045), (.0325, .045)], "blue", 72)
    _text(a, "LAB", [0, -.0349, .063], .015, "blue")
    # An empty cup exposes its bottom and inner wall; no flat top cap is used.


def _bottle(a):
    a.lathe([0, 0, 0], [(0, 0), (.029, 0), (.034, .006), (.035, .016),
                        (.035, .154), (.033, .164), (.022, .178), (.020, .184),
                        (.020, .198), (0, .198)], "silver", 72)
    a.lathe([0, 0, 0], [(.035, .025), (.0358, .028), (.0358, .099), (.035, .103)], "blue", 72)
    for z in (.027, .036, .092, .101):
        a.torus([0, 0, z], .0355, .0012, "blue_light")
    a.cylinder([0, 0, .203], .023, .019, "dark", segments=64)
    a.cylinder([0, 0, .215], .0215, .005, "steel", segments=64)
    for angle in np.linspace(0, 2*math.pi, 28, endpoint=False):
        a.box([.023*math.cos(angle), .023*math.sin(angle), .204], [.0018, .0018, .012], "steel", .0003)
    _text(a, "01", [0, -.0363, .064], .020, "paper")


def _parts(a):
    sx, sy, height = a.recipe["size_m"]
    a.box([0, 0, .003], [sx, sy, .006], "blue", .003)
    for sign in (-1, 1):
        a.box([sign*(sx-.005)/2, 0, height/2], [.005, sy, height], "blue", .002)
        a.box([0, sign*(sy-.005)/2, height/2], [sx, .005, height], "blue", .002)
    for x in (-sx/6, sx/6):
        a.box([x, 0, height*.43], [.003, sy-.01, height*.80], "blue_light", .001)
    a.box([0, 0, height*.43], [sx-.008, .003, height*.80], "blue_light", .001)
    # Clearly open rings and six-sided fasteners, arranged in separate cells.
    for i, x in enumerate((-sx/3, 0, sx/3)):
        for j in range(3):
            xx, yy = x + (j-1)*.022, -sy/4
            _washer(a, [xx, yy, .006], .009 if i == 0 else .008, .004,
                    .0018 if i == 0 else .007, hexagonal=i > 0)
            if i == 0:
                _washer(a, [xx+.001, yy, .008], .009, .004, .0018)
        for j in range(3):
            xx, yy = x + (j-1)*.023, sy/4
            if i == 0:
                a.cylinder([xx, yy, .009], .007, .006, 'steel', segments=6)
                a.cylinder([xx, yy, .018], .0035, .014, 'silver', segments=20)
            elif i == 1:
                _washer(a, [xx, yy, .006], .007, .0034, .006, hexagonal=True)
            else:
                a.cylinder([xx, yy, .012], .006, .012, 'orange', segments=32)
    a.box([0, -sy/2-.001, .022], [.135, .0015, .016], 'paper', .0005)
    _text(a, 'M4  M6  M8', [0, -sy/2-.002, .022], .010)


def _tools(a):
    sx, sy, h = a.recipe["size_m"]
    a.box([0, 0, h/2], [sx, sy, h], "dark", .004)
    # Perimeter ribs and ruler marks make the silicone mat readable.
    for sign in (-1, 1):
        a.box([0, sign*(sy/2-.004), h+.001], [sx-.006, .003, .002], "blue_light", .001)
    for i in range(24):
        a.box([-.12+i*.010, -.070, h+.0005], [.0008, .006 if i%5 else .011, .001], "steel", .0001)
    rx = _rotation([0, 1, 0], math.pi/2)
    y, z = -.041, h+.015
    a.lathe([-.085, y, z], [(0, -.036), (.009, -.036), (.012, -.029), (.012, .028),
                            (.008, .037), (0, .037)], "orange", 48, rx)
    for x in (-.107, -.093, -.079, -.065):
        a.cylinder([x, y, z], .0122, .0025, "orange_dark", rx, 40)
    a.cylinder([-.044, y, z], .007, .008, "dark", rx)
    a.cylinder([.008, y, z], .0032, .100, "silver", rx, 32)
    a.box([.064, y, z], [.016, .007, .0025], "steel", .001)
    # Combination spanner: a long chamfered shank, open jaws and box ring.
    y, z = .008, h+.006
    a.box([-.012, y, z], [.166, .014, .007], "silver", .003)
    _washer(a, [.090, y, z-.004], .017, .010, .008, "silver")
    a.box([-.105, y, z], [.027, .024, .008], "silver", .005)
    for sign in (-1, 1):
        a.box([-.118, y+sign*.014, z], [.031, .010, .008], "silver", .003,
              _rotation([0, 0, 1], sign*.25))
    _text(a, "10", [.018, y, z+.0045], .005, "steel", face="top")
    # Vernier caliper with separate fixed/moving jaws, depth rod and scale.
    y, z = .061, h+.005
    a.box([-.010, y, z], [.218, .012, .005], "silver", .001)
    a.box([.103, y, z], [.045, .002, .002], "steel", .0005)
    for xx, direction in ((-.110, 1), (-.031, -1)):
        a.box([xx, y-.014, z+.002], [.006, .039, .008], "silver", .001)
        a.box([xx+direction*.005, y-.034, z+.002], [.012, .010, .008], "silver", .001)
        a.box([xx, y+.013, z+.002], [.005, .020, .008], "silver", .001)
    a.box([-.030, y, z+.006], [.032, .024, .010], "steel", .002)
    a.box([-.030, y, z+.012], [.025, .013, .002], "dark", .001)
    _text(a, "24", [-.030, y, z+.0135], .006, "screen_light", face="top")
    for i in range(27):
        a.box([-.096+i*.007, y+.003, z+.003], [.00055, .004 if i%5 else .007, .0006], "dark", .0001)
    a.cylinder([-.030, y-.014, z+.004], .005, .009, "dark", _rotation([1, 0, 0], math.pi/2), 24)


def _panel(a):
    sx, sy, height = a.recipe["size_m"]
    a.box([0, 0, height/2], [sx, sy, height], "frame", .012)
    a.box([0, 0, height+.002], [sx-.012, sy-.012, .004], "steel", .006)
    # A recessed screen and mesh glyphs lie on the actual top surface.
    a.box([-.041, .014, height+.006], [.111, .088, .008], "dark", .007)
    a.box([-.041, .014, height+.0105], [.094, .068, .002], "screen", .003)
    _text(a, "READY", [-.041, .031, height+.012], .009, "screen_light", face="top")
    _text(a, "CELL 01", [-.041, .012, height+.012], .006, "paper", face="top")
    for row in range(3):
        a.box([-.058+row*.013, -.006, height+.012], [.010, .002, .001], "screen_light", .0003)
    # A mushroom emergency stop, its yellow-orange collar and ring detail.
    a.cylinder([.064, .031, height+.006], .023, .009, "orange", segments=64)
    a.cylinder([.064, .031, height+.017], .011, .015, "dark")
    a.lathe([.064, .031, height+.016], [(0, 0), (.017, 0), (.021, .004),
                (.022, .013), (.019, .018), (0, .018)], "red", 64)
    a.torus([.064, .031, height+.0335], .014, .0007, "orange_dark")
    for i, color in enumerate(("green", "orange", "blue_light")):
        xx = -.064 + i*.045
        a.cylinder([xx, -.055, height+.006], .011, .008, "dark", segments=32)
        a.cylinder([xx, -.055, height+.011], .0078, .004, color, segments=32)
    _text(a, "CONTROL", [0, -sy/2-.001, height*.5], .012, "paper")
    for x in (-sx/2+.014, sx/2-.014):
        for y in (-sy/2+.014, sy/2-.014):
            a.cylinder([x, y, height+.0045], .003, .0015, "silver", segments=24)
            a.box([x, y, height+.0055], [.004, .00065, .0005], "dark", .0001)


def _rack(a):
    sx, sy, height = a.recipe["size_m"]
    for ix in (-1, 1):
        for iy in (-1, 1):
            x, y = ix*(sx/2-.028), iy*(sy/2-.025)
            a.box([x, y, height/2], [.038, .038, height], "frame", .003)
            a.box([x, y, .009], [.08, .07, .018], "dark", .004)
            a.box([x, y-.020, height/2], [.005, .002, height-.06], "steel", .0008)
            for z in np.arange(.10, height-.04, .080):
                a.box([x, y-.0215, z], [.009, .001, .016], "dark", .002)
    for level, z in enumerate((.20, .67, 1.15)):
        a.box([0, 0, z], [sx, sy, .023], "shelf", .006)
        for sign in (-1, 1):
            a.box([0, sign*(sy/2-.010), z+.026], [sx, .014, .035], "frame", .002)
        a.box([0, -sy/2-.001, z-.007], [sx-.05, .004, .024], "steel", .002)
        _text(a, f"0{level+1}", [-sx/2+.075, -sy/2-.0035, z-.007], .012, "paper")
    # Cross braces are swept rectangular section rails, tilted in XZ.
    for sign in (-1, 1):
        a.box([0, sy/2-.015, height*.49], [.022, .018, 1.55], "steel", .001,
              _rotation([0, 1, 0], sign*.60))
    # Bottom: large totes with real handles and inset paper labels.
    for i, color in enumerate(("blue", "cream", "blue")):
        _bin(a, [(i-1)*.46, 0, .212], [.40, .29, .265], color, ("CABLES", "PARTS", "TOOLS")[i])
    # Middle: smaller open hardware bins plus a compact case.
    for i, color in enumerate(("blue", "orange", "blue_light", "cream")):
        _bin(a, [-.54+i*.35, -.015, .682], [.29, .27, .19], color, ("M4", "M6", "M8", "PINS")[i])
        # Objects visible through the open tops add content, not sealed cubes.
        for j in range(3):
            _washer(a, [-.61+i*.35+j*.067, -.005, .698], .025, .012, .018, "silver", hexagonal=True)
    # Upper shelf: lidded instrument case, two archive boxes, and tape rolls.
    a.box([-.40, 0, 1.257], [.43, .28, .19], "dark", .015)
    a.box([-.40, 0, 1.358], [.445, .29, .019], "frame", .007)
    for x in (-.54, -.26):
        a.box([x, -.143, 1.292], [.025, .010, .038], "silver", .003)
    a.tube([[-.465, -.140, 1.269], [-.465, -.180, 1.299], [-.335, -.180, 1.299], [-.335, -.140, 1.269]], .007, "dark")
    a.box([-.40, -.141, 1.221], [.19, .002, .035], "paper", .001)
    _text(a, "SENSORS", [-.40, -.1425, 1.221], .018)
    for i, color in enumerate(("cream", "blue_light")):
        x = -.018+i*.255
        a.box([x, 0, 1.251], [.22, .275, .176], color, .009)
        a.box([x, 0, 1.347], [.229, .284, .018], color, .004)
        a.box([x, -.139, 1.255], [.12, .002, .031], "paper", .001)
        _text(a, ("SPARES", "LABELS")[i], [x, -.141, 1.255], .015)
    for j in range(2):
        a.lathe([.535, .02, 1.162+j*.040], [(.021,0),(.063,0),(.063,.035),(.021,.035)], "orange" if j else "blue", 64)
        a.lathe([.535, .02, 1.162+j*.040], [(.020,0),(.023,0),(.023,.035),(.020,.035)], "cream", 64)


def _cabinet(a):
    sx, sy, height = a.recipe["size_m"]
    wheel_height = .079
    for ix in (-1, 1):
        for iy in (-1, 1):
            x, y = ix*(sx/2-.060), iy*(sy/2-.065)
            a.cylinder([x, y, .040], .040, .029, "dark", _rotation([0, 1, 0], math.pi/2), 48)
            a.cylinder([x-.016, y, .040], .018, .003, "silver", _rotation([0, 1, 0], math.pi/2), 32)
            a.box([x, y, .075], [.045, .048, .015], "steel", .004)
    a.box([0, 0, (height+wheel_height)/2], [sx, sy, height-wheel_height], "blue", .016)
    a.box([0, 0, height+.007], [sx+.025, sy+.025, .020], "dark", .007)
    a.box([0, -sy/2-.004, .303], [sx-.036, .012, .370], "frame", .008)
    for z in (.521, .612):
        a.box([0, -sy/2-.007, z], [sx-.03, .018, .075], "steel", .004)
        a.tube([[-.074, -sy/2-.018, z], [-.065, -sy/2-.036, z], [.065, -sy/2-.036, z], [.074, -sy/2-.018, z]], .006, "dark")
    # Open black slits and their raised louvers distinguish the door face.
    for z in np.arange(.17, .34, .019):
        a.box([-.025, -sy/2-.011, z], [.21, .003, .006], "dark", .0015)
        a.box([-.025, -sy/2-.013, z+.004], [.21, .004, .004], "steel", .001)
    a.cylinder([.12, -sy/2-.016, .411], .011, .010, "silver", _rotation([1, 0, 0], math.pi/2), 40)
    a.box([.12, -sy/2-.022, .411], [.002, .001, .009], "dark", .0003)
    a.box([-.038, -sy/2-.012, .431], [.19, .002, .037], "paper", .001)
    _text(a, "CELL 01", [-.038, -sy/2-.0135, .431], .018)
    for y in (-.11, .11):
        a.tube([[sx/2, y, height-.045], [sx/2+.040, y, height-.045]], .012, "frame")
    a.tube([[sx/2+.040, -.11, height-.045], [sx/2+.040, .11, height-.045]], .013, "frame")
    # Blue status LED and power connector on the right-hand side.
    a.cylinder([sx/2+.002, .060, .40], .012, .010, "dark", _rotation([0, 1, 0], math.pi/2), 32)


def _bezier(points, count=40):
    p = np.asarray(points, dtype=float)
    t = np.linspace(0, 1, count)[:, None]
    return (1-t)**3*p[0]+3*(1-t)**2*t*p[1]+3*(1-t)*t**2*p[2]+t**3*p[3]


def _cables(a):
    # Cable runs remain behind the tabletop and outside the XY reserved area.
    for i, color in enumerate(("dark", "blue")):
        offset = i*.022
        # First clear the rear edge above the top; only then descend.
        lead = _bezier([[-.10+offset, .72, .781], [-.10+offset, .76, .785],
                       [-.10+offset, .81, .785], [-.10+offset, .835, .775]], 24)
        drop = _bezier([[-.10+offset, .835, .775], [-.10+offset, .93, .73],
                       [.98+offset, .92, .28], [1.13+offset, .89, .45]], 64)
        a.tube(np.concatenate([lead, drop[1:]]), .007 if i == 0 else .0045, color)
    # Small restrained coil secured to the right cabinet side, not on the floor.
    angles = np.linspace(0, 5*math.pi, 120)
    points = np.c_[1.466+.004*angles/(2*math.pi), .68+.080*np.cos(angles), .30+.080*np.sin(angles)]
    a.tube(points, .006, "dark")
    for y in (.615, .745):
        a.box([1.47, y, .35], [.016, .008, .024], "steel", .002)


_RECIPES = {"mug": _mug, "bottle": _bottle, "parts_tray": _parts, "tools": _tools,
            "panel": _panel, "rack": _rack, "cabinet": _cabinet, "cables": _cables}

