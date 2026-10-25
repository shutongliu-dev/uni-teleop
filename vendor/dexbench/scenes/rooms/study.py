"""The ``study`` scene family: a furnished study with a writing desk.

Contents. A 1.2 m pale-wood writing desk stands in front of the robot. Its work
end carries a leather desk pad, and on the pad lie three hardback books, a
notebook, a wooden organiser tray, a ceramic pen cup, a roll of masking tape, a
ballpoint pen and a hexagonal pencil, every one a free dynamic body. The desk's
far end holds an open laptop and an arm desk lamp whose bulb is a real light.
Both are immovable *fixtures*. A drawer pedestal and a modesty panel sit under
the desk. Against the window wall stand a low bookcase full of books and
storage boxes, with a potted plant on top and a framed picture above it, a
mullioned window with a roller blind and two floor-length curtains, and a wall
clock. A chair with a stack of papers on its seat stands on a rug, and a waste
basket sits by the desk. The door is on the opposite wall.

Physical discipline (see ``bench.simulation.isaacsim_backend.scene_family``):
  * every dynamic object is an authored OBJ asset in ``assets/study/v8`` with a
    positive mass, a uniform-density inertia from its own bounding box, and
    analytic or convex collision pieces (rule 1);
  * the tape roll is the family's through hole: 24 convex annulus sectors declare
    ``bore: true`` around its 38 mm core, and no piece seals it (rule 2). The pen
    cup is a *blind* cavity, not a through hole, so it declares no bore (the
    laboratory convention). Its 24 wall sectors are tangent to the cavity and a
    solid floor closes it below;
  * nothing is stacked. Every object lies directly on the desk pad, the family's
    one declared support. No dynamic body rests on another dynamic body, and no
    static geometry rests on, or lies over, a dynamic body (rule 3). Version v6
    stacked its books on 1 mm static pads that pinned three of them from above.
    That design is withdrawn;
  * every collider, dynamic or static, is at least 3 mm thick (rule 4); masses
    are design values recorded in each asset (rule 5);
  * object centres lie inside x in [0.30, 0.80], y in [-0.10, 0.45] and within
    0.80 m of the robot base (rule 6). The desk starts at x = 0.26 m, clear of
    the 0.22 m pedestal (rule 7), and the pedestal stands on a 0.65 m skirt that
    reaches the floor;
  * the room is closed: floor, four walls with a window and a door, ceiling,
    two ceiling panels, the desk-lamp bulb and a dome (rule 8). The third-person
    camera sits 2.06 m from the desk centre and 0.70 m above the desk top, and it
    frames the desk, the bookcase, the window and curtains, the clock, the
    picture, the plant, the laptop, the lamp and the basket (rule 9).

The desk pad replaces v6's duplicated desk collider. ``family.support_surfaces``
reads only ``surfaces`` and ``static_props``, so a work table must be repeated
there to count as a support. v6 repeated the desk slab, which built two
coincident desk colliders. Here the slab is built once, from ``table``. The
declared support is a separate 4 mm leather pad lying on top of the slab, and
every object rests on that pad.

Wall-local openings. ``room.openings`` use (u, v) measured from the wall centre:
world z = ceiling_height / 2 + v, and u runs along +x on the front (+y) wall and
along -x on the back (-y) wall. The window frame, blind and door leaf sit in
the wall's reveal (0 to 0.12 m behind the interior plane), inside their openings.

Collision frames. ``scene.add_task_asset`` loads an OBJ verbatim, then applies
``position_m`` and ``quaternion_xyzw`` as the prim transform. So every convex
piece here is written with its offset baked in and ``position_m`` zero, and
every analytic piece is placed only by its own ``position_m``/``quaternion_xyzw``.

Textures. The floor, walls, desk and bookcase use CC0 ambientCG sets from the
shared library. Each binds its colour, roughness and normal maps through the
``<slot>_map`` keys of ``visuals.TEXTURE_SLOTS``. No scalar ``roughness`` is
set on a textured material, because an older ``visuals.validate_material`` read
that key as a map path. Everything else is a flat PBR colour, with the finish of
its real material.

Masses are simulation design values, not manufacturer measurements. That is
recorded in the asset manifest and in each asset's ``mass_design_note``.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from bench.paths import DATA_ROOT, load_json
from scenes.rooms import DEFAULT_LIBRARY_ROOT
from bench.simulation.assets.scene_parts import (Asset, annulus_mesh, annulus_sectors, box_mesh, cylinder_mesh,
                                lathe_mesh, write_library)

FAMILY = 'study'
ASSET_VERSION = 'v8'
# v7 and v6 are kept beside v8 as evidence (history in the library's VERSIONS.md).
ASSET_ROOT = str(DEFAULT_LIBRARY_ROOT / FAMILY / ASSET_VERSION)
SUPERSEDED_ASSET_ROOTS = (str(DEFAULT_LIBRARY_ROOT / FAMILY / 'v7'),
                          str(DEFAULT_LIBRARY_ROOT / FAMILY / 'v6'))
GENERATOR = 'scenes/rooms/study.py'
LICENSE_ID = 'CC0-1.0'

MATERIAL_SOURCE_ROOT = f'{DATA_ROOT}/assets/materials/ambientcg/v1'
TEXTURE_SOURCES = {
    'WoodFloor064': 'https://ambientcg.com/a/WoodFloor064',
    'Wood089': 'https://ambientcg.com/a/Wood089',
    'Plaster001': 'https://ambientcg.com/a/Plaster001',
}

# --- room ---------------------------------------------------------------------

ROOM_SIZE = (5.0, 4.0, 2.7)
ROOM_ORIGIN = (0.0, 0.0, 0.0)
ROOM_HEIGHT = ROOM_SIZE[2]
WALL_T = 0.12
FRONT_WALL_Y = ROOM_ORIGIN[1] + ROOM_SIZE[1] / 2.        # +y wall: window, bookcase
BACK_WALL_Y = ROOM_ORIGIN[1] - ROOM_SIZE[1] / 2.         # -y wall: door

WINDOW_X = (-1.25, -0.15)                                # front wall
WINDOW_Z = (0.80, 1.90)
DOOR_X = (-1.65, -0.75)                                  # back wall
DOOR_Z = (0.0, 2.10)

# --- robot and desk -----------------------------------------------------------

PEDESTAL_CENTER_Z = 0.70
PEDESTAL_HALF = (0.22, 0.22, 0.05)
PEDESTAL_SKIRT = PEDESTAL_CENTER_Z - PEDESTAL_HALF[2]    # 0.65 m: the skirt reaches the floor
ROBOT_BASE_Z = PEDESTAL_CENTER_Z + PEDESTAL_HALF[2]      # 0.750 m

DESK_X = (0.26, 0.92)
DESK_Y = (-0.20, 1.00)
DESK_TOP_Z = 0.750
DESK_SLAB_T = 0.035
DESK_CENTER_XY = ((DESK_X[0] + DESK_X[1]) / 2., (DESK_Y[0] + DESK_Y[1]) / 2.)   # (0.59, 0.40)
DESK_HALF_XY = ((DESK_X[1] - DESK_X[0]) / 2., (DESK_Y[1] - DESK_Y[0]) / 2.)     # (0.33, 0.60)
DESK_SLAB_CENTER_Z = DESK_TOP_Z - DESK_SLAB_T / 2.
DESK_UNDERSIDE_Z = DESK_TOP_Z - DESK_SLAB_T              # 0.715 m
DESK_LEG = 0.05

# The declared support: a 4 mm leather desk pad on the slab, under every object.
PAD_X = (0.275, 0.835)
PAD_Y = (-0.165, 0.505)
PAD_T = 0.004
PAD_TOP_Z = DESK_TOP_Z + PAD_T                           # 0.754 m
WORK_CENTER = ((PAD_X[0] + PAD_X[1]) / 2., (PAD_Y[0] + PAD_Y[1]) / 2., PAD_TOP_Z)

# --- object geometry ----------------------------------------------------------

BOOK_SPECS = (
    # asset_id, width (x), depth (y), thickness (z), cover colour, mass_kg
    ('book_blue', 0.165, 0.235, 0.038, [0.16, 0.26, 0.46], 0.62),
    ('book_red', 0.140, 0.180, 0.030, [0.55, 0.16, 0.15], 0.44),
    ('book_green', 0.150, 0.150, 0.026, [0.18, 0.35, 0.24], 0.36),
)
BOOK_BOARD = 0.0025                                      # cover board, visual only
BOOK_SQUARE = 0.003                                      # page block inset at the open edges
PAGE_COLOR = [0.91, 0.88, 0.80]

NOTEBOOK_SIZE = (0.210, 0.148, 0.014)
NOTEBOOK_MASS = 0.220
NOTEBOOK_COVER = [0.66, 0.50, 0.32]                    # kraft board: reads against the black pad

TRAY_SIZE = (0.200, 0.160, 0.026)
TRAY_WALL = 0.010
TRAY_BASE = 0.006
TRAY_MASS = 0.180

CUP_OUTER_R = 0.040
CUP_INNER_R = 0.034                                      # 6 mm wall
CUP_HEIGHT = 0.094
CUP_FLOOR = 0.008
CUP_SECTORS = 24
CUP_MASS = 0.26
CUP_GLAZE = [0.22, 0.38, 0.47]

TAPE_OUTER_R = 0.048
TAPE_CORE_R = 0.0405                                     # cardboard core outside radius (visual)
TAPE_INNER_R = 0.038                                     # the through bore
TAPE_WIDTH = 0.025
TAPE_SECTORS = 24
TAPE_MASS = 0.10

# A hexagonal barrel (as on the common BIC-Cristal-style ballpoint): it rests on a
# flat face and cannot roll. A round barrel rolls on the desk; v7's did, natively.
PEN_APOTHEM = 0.0035                                     # 7 mm across the flats
PEN_CIRCUMRADIUS = PEN_APOTHEM / math.cos(math.pi / 6.)
PEN_LENGTH = 0.128
PEN_MASS = 0.008
PENCIL_APOTHEM = 0.0035                                  # 7 mm across the flats
PENCIL_CIRCUMRADIUS = PENCIL_APOTHEM / math.cos(math.pi / 6.)
PENCIL_LENGTH = 0.150
PENCIL_MASS = 0.005

SECTOR_BEVEL = 0.0003

# --- fixture geometry ---------------------------------------------------------

LAMP_BASE_R, LAMP_BASE_H = 0.070, 0.020
LAMP_STEM_R, LAMP_STEM_TOP = 0.009, 0.420
LAMP_ARM_X = (-0.150, 0.009)
LAMP_ARM_HALF = 0.008
LAMP_ARM_Z = (0.402, 0.418)
LAMP_SHADE_X = -0.140
LAMP_SHADE_Z = (0.290, 0.400)
LAMP_SHADE_R = (0.075, 0.035)                            # bottom, top
LAMP_SHADE_WALL = 0.003
LAMP_HOLDER_R, LAMP_HOLDER_Z = 0.012, (0.370, 0.402)
LAMP_BULB_R = 0.020
LAMP_BULB_LOCAL = (LAMP_SHADE_X, 0.0, 0.350)

LAPTOP_BASE = (0.220, 0.320, 0.016)                      # x (depth), y (width), z
LAPTOP_DECK_T = 0.001
LAPTOP_LID = (0.005, 0.320, 0.210)                       # thickness, width, height
LAPTOP_TILT_DEG = 20.0                                   # lid leans back from vertical
LAPTOP_HINGE = (0.1075, 0.0, 0.0180)

CLOCK_R, CLOCK_FACE_R, CLOCK_DEPTH = 0.150, 0.1345, 0.045
CLOCK_FACE_TOP = 0.034

POT_PROFILE = ((0.0, 0.0), (0.055, 0.0), (0.068, 0.115), (0.074, 0.117), (0.074, 0.135),
               (0.065, 0.135), (0.062, 0.118), (0.0, 0.118))
POT_R, POT_H = 0.074, 0.135

BASKET_R, BASKET_H, BASKET_WALL, BASKET_FLOOR = 0.130, 0.300, 0.006, 0.010
BASKET_SIDES = 12

# --- layout -------------------------------------------------------------------

def _yaw(degrees):
    """Quaternion (w, x, y, z) for a rotation about +z."""
    half = math.radians(degrees) / 2.
    return [math.cos(half), 0., 0., math.sin(half)]


# Pens are authored upright (axis +z); this lays the axis along +y with the
# pencil's +x flat face down: first +90 deg about y, then +90 deg about z.
LIE_ALONG_Y = [0.5, -0.5, 0.5, 0.5]
CLOCK_ON_FRONT_WALL = [math.cos(math.pi / 4.), math.sin(math.pi / 4.), 0., 0.]  # local +z -> world -y

OBJECT_LAYOUT = (
    # object_id, asset_id, x, y, quaternion_wxyz, friction
    ('organiser_tray', 'organiser_tray', 0.390, -0.030, _yaw(-4.0), 0.55),
    ('book_blue', 'book_blue', 0.390, 0.195, _yaw(-4.0), 0.52),
    ('pen_cup', 'pen_cup', 0.390, 0.405, _yaw(0.0), 0.58),
    ('pen', 'pen', 0.535, -0.020, LIE_ALONG_Y, 0.40),
    ('tape_roll', 'tape_roll', 0.545, 0.175, _yaw(0.0), 0.60),
    ('pencil', 'pencil', 0.535, 0.370, LIE_ALONG_Y, 0.45),
    ('notebook', 'notebook', 0.690, -0.050, _yaw(8.0), 0.60),
    ('book_red', 'book_red', 0.695, 0.160, _yaw(3.0), 0.52),
    ('book_green', 'book_green', 0.695, 0.370, _yaw(-3.0), 0.52),
)

FIXTURE_LAYOUT = (
    # fixture_id, asset_id, position of the asset origin (base-bottom centre), quaternion, friction
    # The arm points back-left so the open shade hangs beside the pad, not over it.
    ('desk_lamp', 'desk_lamp', (0.845, 0.720, DESK_TOP_Z), _yaw(40.0), 0.55),
    # Screen turned toward the work area (and the room camera); lid inside the desk edge.
    ('laptop', 'laptop', (0.560, 0.780, DESK_TOP_Z), _yaw(75.0), 0.50),
    # Far enough left that the left curtain does not hide it from the third-person camera.
    ('wall_clock', 'wall_clock', (-2.080, FRONT_WALL_Y, 1.620), CLOCK_ON_FRONT_WALL, 0.50),
    # Unrotated so the foliage box stays 3 cm off the wall it stands against.
    ('potted_plant', 'potted_plant', (1.030, 1.835, 1.150), _yaw(0.0), 0.60),
    ('waste_basket', 'waste_basket', (1.100, 1.280, 0.0), _yaw(0.0), 0.60),
)

BOOKCASE_X = (0.30, 1.20)
BOOKCASE_Y = (1.70, FRONT_WALL_Y)
BOOKCASE_H = 1.150
BOOKCASE_PANEL = 0.018
BOOKCASE_BACK = 0.010
BOOKCASE_PLINTH = 0.060
BOOKCASE_BOARD_BOTTOMS = (BOOKCASE_PLINTH, 0.420, 0.780)
BOOK_SPINE_Y = BOOKCASE_Y[0] + 0.006                     # spines just behind the front edge

CHAIR_CENTER_XY = (-0.950, 1.050)                # pushed back, clear of the curtains in view
CHAIR_SEAT_TOP = 0.450
RUG_X, RUG_Y, RUG_T = (-1.35, 0.20), (0.45, 1.55), 0.008

ART_X, ART_Z = (0.36, 0.88), (1.31, 1.71)

# Static props and fixtures that hang on a wall. Everything else is floor
# furniture and must rest its lowest part on the floor or on a static top.
WALL_MOUNTED = ('wall_art', 'window_frame', 'curtains', 'door', 'wall_clock')


def _r(value):
    return round(float(value), 6)


def _r3(values):
    return [round(float(v), 6) for v in values]


def _rotation(wxyz):
    w, x, y, z = np.asarray(wxyz, dtype=float) / np.linalg.norm(wxyz)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def _rot_z(degrees):
    return _rotation(_yaw(degrees))


def _rot_y(degrees):
    half = math.radians(degrees) / 2.
    return _rotation([math.cos(half), 0., math.sin(half), 0.])


def _quat_xyzw_z(degrees):
    half = math.radians(degrees) / 2.
    return [0., 0., math.sin(half), math.cos(half)]


def _quat_xyzw_y(degrees):
    half = math.radians(degrees) / 2.
    return [0., math.sin(half), 0., math.cos(half)]


# --- mesh helpers ---------------------------------------------------------------

def _box_at(lo, hi):
    lo, hi = np.asarray(lo, dtype=float), np.asarray(hi, dtype=float)
    mesh = box_mesh(hi - lo)
    mesh.apply_translation((lo + hi) / 2.)
    return mesh


def _moved(mesh, offset=(0., 0., 0.), rotation=None):
    mesh = mesh.copy()
    if rotation is not None:
        transform = np.eye(4)
        transform[:3, :3] = rotation
        mesh.apply_transform(transform)
    mesh.apply_translation(np.asarray(offset, dtype=float))
    return mesh


def _hull(points):
    import trimesh
    return trimesh.convex.convex_hull(np.asarray(points, dtype=float))


def _hex_prism_points(apothem, z_lo, z_hi):
    """Six-sided prism, flat faces at +-x, about the local z axis."""
    radius = apothem / math.cos(math.pi / 6.)
    angles = [math.pi / 6. + k * math.pi / 3. for k in range(6)]
    ring = [(radius * math.cos(a), radius * math.sin(a)) for a in angles]
    return [(x, y, z) for z in (z_lo, z_hi) for x, y in ring]


def _ellipsoid_points(centre, axes, frame):
    """A fixed 26-direction sample of an ellipsoid: deterministic, hull-friendly."""
    directions = []
    for x in (-1, 0, 1):
        for y in (-1, 0, 1):
            for z in (-1, 0, 1):
                if (x, y, z) != (0, 0, 0):
                    vector = np.array([x, y, z], dtype=float)
                    directions.append(vector / np.linalg.norm(vector))
    local = np.asarray(directions) * np.asarray(axes, dtype=float)
    return local @ np.asarray(frame, dtype=float).T + np.asarray(centre, dtype=float)


# --- dynamic assets -------------------------------------------------------------

def _bound_block(asset, size, cover, board, square, *, label='block'):
    """Covers, spine and an inset page block inside one closed box collider.

    The spine runs along the long edge. The page block is inset ``square`` at
    the three open edges, so the covers overhang as on a real hardback. All
    visual pieces lie inside the collider box.
    """
    w, d, t = size
    hx, hy, hz = w / 2., d / 2., t / 2.
    finish = dict(roughness=0.70, metallic=0.0)
    asset.add_display(_box_at((-hx, -hy, hz - board), (hx, hy, hz)), cover, 'front_board', **finish)
    asset.add_display(_box_at((-hx, -hy, -hz), (hx, hy, -hz + board)), cover, 'back_board', **finish)
    if d >= w:          # spine along y at -x
        asset.add_display(_box_at((-hx, -hy, -hz + board), (-hx + board, hy, hz - board)), cover,
                          'spine', **finish)
        pages = ((-hx + board, -hy + square, -hz + board), (hx - square, hy - square, hz - board))
    else:               # spine along x at -y
        asset.add_display(_box_at((-hx, -hy, -hz + board), (hx, -hy + board, hz - board)), cover,
                          'spine', **finish)
        pages = ((-hx + square, -hy + board, -hz + board), (hx - square, hy - square, hz - board))
    asset.add_display(_box_at(*pages), PAGE_COLOR, 'page_block', roughness=0.85, metallic=0.0)
    asset.add_collision(None, label, geometry='box', half_extents_m=[hx, hy, hz], position_m=[0., 0., 0.])


def _book(asset_id, width, depth, thickness, cover, mass):
    """A hardback lying flat: one analytic box collider over the closed book."""
    asset = Asset(asset_id, dynamic=True, mass_kg=mass, friction=0.52)
    _bound_block(asset, (width, depth, thickness), cover, BOOK_BOARD, BOOK_SQUARE)
    asset.data.update(
        nominal_size_m=[width, depth, thickness],
        size_note=(f'{width * 1000:.0f} x {depth * 1000:.0f} mm hardback, {thickness * 1000:.0f} mm '
                   'thick; the closed book is the collider'),
        mass_design_note=f'Design value: a hardback of this size is about {mass:.2f} kg',
        collision_note=('One analytic box over the closed book. The 2.5 mm boards and the page '
                        'block are visual only, so no thin board is ever a collider'),
        rest_note='Lies flat on the desk pad; nothing rests on it and nothing is stacked under it')
    return asset


def _notebook():
    asset = Asset('notebook', dynamic=True, mass_kg=NOTEBOOK_MASS, friction=0.60)
    _bound_block(asset, NOTEBOOK_SIZE, NOTEBOOK_COVER, 0.0015, 0.002)
    asset.data.update(
        nominal_size_m=list(NOTEBOOK_SIZE),
        size_note='A5 notebook, 210 x 148 x 14 mm, bound on its long edge',
        mass_design_note=f'Design value: a 200-page A5 notebook is about {NOTEBOOK_MASS * 1000:.0f} g',
        collision_note='One analytic box; a closed notebook behaves as a solid block',
        rest_note='Lies flat on the desk pad')
    return asset


def _tray_parts():
    length, width, height = TRAY_SIZE
    floor_z = -height / 2. + TRAY_BASE / 2.
    wall_h = height - TRAY_BASE
    wall_z = -height / 2. + TRAY_BASE + wall_h / 2.
    parts = [('floor', (length, width, TRAY_BASE), (0., 0., floor_z))]
    for name, size, (x, y) in (
            ('wall_xpos', (TRAY_WALL, width, wall_h), (length / 2. - TRAY_WALL / 2., 0.)),
            ('wall_xneg', (TRAY_WALL, width, wall_h), (-(length / 2. - TRAY_WALL / 2.), 0.)),
            ('wall_ypos', (length - 2 * TRAY_WALL, TRAY_WALL, wall_h), (0., width / 2. - TRAY_WALL / 2.)),
            ('wall_yneg', (length - 2 * TRAY_WALL, TRAY_WALL, wall_h), (0., -(width / 2. - TRAY_WALL / 2.)))):
        parts.append((name, size, (x, y, wall_z)))
    return parts


def _uniform_com(pieces):
    """Centre of mass of solid pieces given as (volume, centroid)."""
    volume = sum(v for v, _ in pieces)
    return np.sum([v * np.asarray(c, dtype=float) for v, c in pieces], axis=0) / volume


def _organiser_tray():
    """An open desk tray: a base box plus four wall boxes, all analytic colliders."""
    parts = _tray_parts()
    com = _uniform_com([(float(np.prod(size)), centre) for _, size, centre in parts])
    asset = Asset('organiser_tray', dynamic=True, mass_kg=TRAY_MASS, friction=0.55,
                  center_of_mass_m=[0., 0., _r(com[2])])
    for name, size, centre in parts:
        half = np.asarray(size) / 2.
        asset.add_display(_box_at(np.asarray(centre) - half, np.asarray(centre) + half),
                          [0.34, 0.23, 0.14], name, roughness=0.55, metallic=0.0)
        asset.add_collision(None, name, geometry='box', half_extents_m=[float(v) for v in half],
                            position_m=[float(v) for v in centre])
    asset.data.update(
        nominal_size_m=list(TRAY_SIZE),
        size_note='200 x 160 x 26 mm, 10 mm walls, 6 mm base; open on top',
        mass_design_note=f'Design value: a small wooden desk tray is about {TRAY_MASS * 1000:.0f} g',
        center_of_mass_note='Uniform-density centre of mass of the five boxes (below the box centre)',
        collision_note='A solid base box plus four wall boxes; an open tray, not a bored ring',
        rest_note='Lies flat on the desk pad, empty')
    return asset


def _pen_cup():
    """A glazed ceramic pen cup: a dynamic, empty, blind cavity.

    The wall is 24 convex annulus sectors over the full height, with inner chords
    tangent to the 34 mm cavity radius, and a solid 8 mm floor disc closes it
    below. This is a blind cavity, not a through hole, so no bore is declared;
    ``cavity_radius_m`` and ``cavity_floor_z_m`` record what the CPU test checks.
    """
    half = CUP_HEIGHT / 2.
    wall_volume = math.pi * (CUP_OUTER_R ** 2 - CUP_INNER_R ** 2) * CUP_HEIGHT
    floor_volume = math.pi * CUP_INNER_R ** 2 * CUP_FLOOR
    com = _uniform_com([(wall_volume, (0., 0., 0.)), (floor_volume, (0., 0., -half + CUP_FLOOR / 2.))])
    asset = Asset('pen_cup', dynamic=True, mass_kg=CUP_MASS, friction=0.58,
                  center_of_mass_m=[0., 0., _r(com[2])])
    profile = [(0., -half), (CUP_OUTER_R, -half), (CUP_OUTER_R, half), (CUP_INNER_R, half),
               (CUP_INNER_R, -half + CUP_FLOOR), (0., -half + CUP_FLOOR)]
    asset.add_display(lathe_mesh(profile, segments=96), CUP_GLAZE, 'body', roughness=0.25, metallic=0.0)
    for index, piece in enumerate(annulus_sectors(CUP_OUTER_R, CUP_INNER_R, CUP_HEIGHT, CUP_SECTORS,
                                                  SECTOR_BEVEL)):
        asset.add_collision(piece, f'wall_{index:02d}', geometry='convex_mesh', position_m=[0., 0., 0.],
                            sector=f'{index:02d}/{CUP_SECTORS}', role='wall',
                            clearance_radius_m=CUP_INNER_R)
    asset.add_collision(None, 'floor', geometry='cylinder', radius_m=CUP_INNER_R,
                        half_height_m=CUP_FLOOR / 2., position_m=[0., 0., -half + CUP_FLOOR / 2.],
                        role='floor')
    asset.data.update(
        nominal_size_m=[2 * CUP_OUTER_R, 2 * CUP_OUTER_R, CUP_HEIGHT],
        size_note=(f'{2 * CUP_OUTER_R * 1000:.0f} mm outside, {2 * CUP_INNER_R * 1000:.0f} mm cavity, '
                   f'{CUP_HEIGHT * 1000:.0f} mm tall, {CUP_FLOOR * 1000:.0f} mm floor'),
        mass_design_note=f'Design value: a glazed ceramic pen cup of this size is about {CUP_MASS:.2f} kg',
        center_of_mass_note='Uniform-density centre of mass of wall and floor (below the box centre)',
        open_top=True, closed_bottom=True, declares_bore=False,
        cavity_radius_m=CUP_INNER_R, cavity_floor_z_m=-half + CUP_FLOOR, cavity_top_z_m=half,
        collision_note=('Blind cavity, not a through hole: 24 convex wall sectors tangent to the '
                        f'{CUP_INNER_R * 1000:.0f} mm cavity plus a solid floor disc below it. No bore is '
                        'declared (laboratory convention); no piece enters the cavity'),
        rest_note='Stands upright and empty on the desk pad; a free object the robot may move')
    return asset


def _tape_roll():
    """A roll of masking tape: the family's one through hole."""
    asset = Asset('tape_roll', dynamic=True, mass_kg=TAPE_MASS, friction=0.60)
    asset.add_display(annulus_mesh(TAPE_OUTER_R, TAPE_CORE_R, TAPE_WIDTH), [0.86, 0.80, 0.62], 'tape',
                      roughness=0.75, metallic=0.0)
    asset.add_display(annulus_mesh(TAPE_CORE_R, TAPE_INNER_R, TAPE_WIDTH), [0.55, 0.41, 0.26], 'core',
                      roughness=0.88, metallic=0.0)
    for index, piece in enumerate(annulus_sectors(TAPE_OUTER_R, TAPE_INNER_R, TAPE_WIDTH, TAPE_SECTORS,
                                                  SECTOR_BEVEL)):
        asset.add_collision(piece, f'ring_{index:02d}', geometry='convex_mesh', position_m=[0., 0., 0.],
                            sector=f'{index:02d}/{TAPE_SECTORS}', bore=True, bore_radius_m=TAPE_INNER_R,
                            clearance_radius_m=TAPE_INNER_R, bore_axis_xy_m=[0., 0.])
    asset.data.update(
        nominal_size_m=[2 * TAPE_OUTER_R, 2 * TAPE_OUTER_R, TAPE_WIDTH],
        size_note=(f'{2 * TAPE_OUTER_R * 1000:.0f} mm roll on a {2 * TAPE_INNER_R * 1000:.0f} mm core, '
                   f'{TAPE_WIDTH * 1000:.0f} mm wide'),
        mass_design_note=f'Design value: a 25 mm x 50 m masking tape roll is about {TAPE_MASS * 1000:.0f} g',
        collision_note=(f'{TAPE_SECTORS} convex annulus sectors declared bore: true; every inner chord '
                        f'is tangent to the {TAPE_INNER_R * 1000:.0f} mm bore, so the core stays open'),
        rest_note='Lies flat on the desk pad, bore axis vertical')
    return asset


def _pen():
    """A capped hexagonal ballpoint, authored upright with flats at +-x; a convex hex collider."""
    half = PEN_LENGTH / 2.
    asset = Asset('pen', dynamic=True, mass_kg=PEN_MASS, friction=0.40)
    tip = lathe_mesh([(0., -half), (0.0007, -half), (PEN_APOTHEM, -half + 0.008), (0., -half + 0.008)],
                     segments=48)
    asset.add_display(tip, [0.72, 0.72, 0.70], 'tip', roughness=0.30, metallic=0.85)
    asset.add_display(_hull(_hex_prism_points(PEN_APOTHEM, -half + 0.008, half - 0.034)),
                      [0.10, 0.17, 0.42], 'barrel', roughness=0.38, metallic=0.0)
    # The round cap's radius equals the barrel's apothem, so on a flat face it is
    # tangent to the desk exactly where the hex collider is.
    asset.add_display(_moved(cylinder_mesh(PEN_APOTHEM, 0.034, sections=48), (0., 0., half - 0.017)),
                      [0.06, 0.06, 0.07], 'cap', roughness=0.40, metallic=0.0)
    asset.add_collision(_hull(_hex_prism_points(PEN_APOTHEM, -half, half)), 'body',
                        geometry='convex_mesh', position_m=[0., 0., 0.])
    asset.data.update(
        nominal_size_m=[2 * PEN_APOTHEM, 2 * PEN_CIRCUMRADIUS, PEN_LENGTH],
        size_note=f'{2 * PEN_APOTHEM * 1000:.0f} mm across the flats, {PEN_LENGTH * 1000:.0f} mm long',
        mass_design_note=f'Design value: a capped ballpoint is about {PEN_MASS * 1000:.0f} g',
        collision_note=('The convex hull of its own hexagonal prism over the full length; it lies on a '
                        'flat face and cannot roll (the v7 round cylinder rolled 6.7 mm natively)'),
        rest_note='Lies on the desk pad along +y, one flat face down')
    return asset


def _pencil():
    """A hexagonal pencil, authored upright with flats at +-x; a convex hex collider."""
    half = PENCIL_LENGTH / 2.
    asset = Asset('pencil', dynamic=True, mass_kg=PENCIL_MASS, friction=0.45)
    asset.add_display(_hull(_hex_prism_points(PENCIL_APOTHEM, -half + 0.0125, half - 0.015)),
                      [0.88, 0.68, 0.14], 'body', roughness=0.45, metallic=0.0)
    cone = _hex_prism_points(PENCIL_APOTHEM, -half + 0.0125, -half + 0.0125)[:6]
    cone += [(0.0012 * math.cos(a), 0.0012 * math.sin(a), -half + 0.0025)
             for a in np.linspace(0., 2 * math.pi, 8, endpoint=False)]
    asset.add_display(_hull(cone), [0.80, 0.64, 0.45], 'sharpened_wood', roughness=0.70, metallic=0.0)
    asset.add_display(_hull([(0.0012 * math.cos(a), 0.0012 * math.sin(a), -half + 0.0025)
                             for a in np.linspace(0., 2 * math.pi, 8, endpoint=False)] + [(0., 0., -half)]),
                      [0.12, 0.12, 0.13], 'lead', roughness=0.50, metallic=0.0)
    asset.add_display(_moved(cylinder_mesh(PENCIL_APOTHEM, 0.008, sections=32), (0., 0., half - 0.011)),
                      [0.70, 0.70, 0.68], 'ferrule', roughness=0.30, metallic=0.85)
    asset.add_display(_moved(cylinder_mesh(0.0033, 0.007, sections=32), (0., 0., half - 0.0035)),
                      [0.86, 0.50, 0.52], 'eraser', roughness=0.85, metallic=0.0)
    asset.add_collision(_hull(_hex_prism_points(PENCIL_APOTHEM, -half, half)), 'body',
                        geometry='convex_mesh', position_m=[0., 0., 0.])
    asset.data.update(
        nominal_size_m=[2 * PENCIL_APOTHEM, 2 * PENCIL_CIRCUMRADIUS, PENCIL_LENGTH],
        size_note=f'7 mm across the flats, {PENCIL_LENGTH * 1000:.0f} mm long',
        mass_design_note=f'Design value: a wooden pencil of this length is about {PENCIL_MASS * 1000:.0f} g',
        collision_note='The convex hull of its own hexagonal prism; it lies on a flat face and cannot roll',
        rest_note='Lies on the desk pad along +y, one flat face down')
    return asset


# --- fixtures (origin at the base bottom centre) --------------------------------

def _desk_lamp():
    """Arm desk lamp: base disc, stem, arm and an open conical shade over a bulb."""
    asset = Asset('desk_lamp', dynamic=False, mass_kg=0.0, friction=0.55)
    metal = dict(roughness=0.35, metallic=0.85)
    asset.add_display(_moved(cylinder_mesh(LAMP_BASE_R, LAMP_BASE_H), (0., 0., LAMP_BASE_H / 2.)),
                      [0.16, 0.17, 0.18], 'base', **metal)
    stem_h = LAMP_STEM_TOP - LAMP_BASE_H
    asset.add_display(_moved(cylinder_mesh(LAMP_STEM_R, stem_h, sections=32),
                             (0., 0., LAMP_BASE_H + stem_h / 2.)), [0.70, 0.53, 0.25], 'stem', **metal)
    asset.add_display(_box_at((LAMP_ARM_X[0], -LAMP_ARM_HALF, LAMP_ARM_Z[0]),
                              (LAMP_ARM_X[1], LAMP_ARM_HALF, LAMP_ARM_Z[1])), [0.70, 0.53, 0.25], 'arm',
                      **metal)
    asset.add_display(_moved(cylinder_mesh(LAMP_HOLDER_R, LAMP_HOLDER_Z[1] - LAMP_HOLDER_Z[0], sections=32),
                             (LAMP_SHADE_X, 0., sum(LAMP_HOLDER_Z) / 2.)), [0.16, 0.17, 0.18], 'holder',
                      **metal)
    (r_bottom, r_top), (z0, z1), wall = LAMP_SHADE_R, LAMP_SHADE_Z, LAMP_SHADE_WALL
    shade = lathe_mesh([(0., z1), (r_top, z1), (r_bottom, z0), (r_bottom - wall, z0),
                        (r_top - wall, z1 - wall), (0., z1 - wall)], segments=64)
    asset.add_display(_moved(shade, (LAMP_SHADE_X, 0., 0.)), [0.17, 0.32, 0.27], 'shade',
                      roughness=0.45, metallic=0.0)
    import trimesh
    bulb = trimesh.creation.icosphere(subdivisions=2, radius=LAMP_BULB_R)
    asset.add_display(_moved(bulb, LAMP_BULB_LOCAL), [0.98, 0.95, 0.86], 'bulb', roughness=0.20, metallic=0.0)
    asset.add_collision(None, 'base', geometry='cylinder', radius_m=LAMP_BASE_R,
                        half_height_m=LAMP_BASE_H / 2., position_m=[0., 0., LAMP_BASE_H / 2.])
    asset.add_collision(None, 'stem', geometry='cylinder', radius_m=LAMP_STEM_R, half_height_m=stem_h / 2.,
                        position_m=[0., 0., LAMP_BASE_H + stem_h / 2.])
    asset.add_collision(None, 'arm', geometry='box',
                        half_extents_m=[(LAMP_ARM_X[1] - LAMP_ARM_X[0]) / 2., LAMP_ARM_HALF,
                                        (LAMP_ARM_Z[1] - LAMP_ARM_Z[0]) / 2.],
                        position_m=[sum(LAMP_ARM_X) / 2., 0., sum(LAMP_ARM_Z) / 2.])
    asset.add_collision(None, 'shade', geometry='cylinder', radius_m=r_bottom, half_height_m=(z1 - z0) / 2.,
                        position_m=[LAMP_SHADE_X, 0., (z0 + z1) / 2.])
    asset.data.update(
        nominal_size_m=DESIGN_SIZES['desk_lamp'],
        size_note='140 mm base, 420 mm stem, 150 mm arm, conical shade 150 mm across the mouth',
        static_note='Fixture: furniture the robot works around, never a manipulation target',
        light_note=('The bulb is a real light: room.ceiling_lights carries a sphere light of the '
                    'bulb radius at the bulb centre, inside the open shade'),
        bulb_local_m=list(LAMP_BULB_LOCAL), bulb_radius_m=LAMP_BULB_R,
        collision_note=('Analytic pieces only (two cylinders, a box, and a closed cylinder over '
                        'the open shade: nothing is meant to enter the shade)'),
        rest_note='The base bottom stands on the desk top at z = 0.750 m')
    return asset


def _laptop_lid_transform():
    return _rot_y(LAPTOP_TILT_DEG), np.asarray(LAPTOP_HINGE, dtype=float)


def _laptop_lid_boxes():
    """The lid and its screen panel as (lo, hi) corners in the lid frame (hinge at the origin)."""
    t, w, h = LAPTOP_LID
    return {'lid': ((-t / 2., -w / 2., 0.), (t / 2., w / 2., h)),
            'screen': ((-t / 2. - 0.001, -w / 2. + 0.010, 0.012), (-t / 2., w / 2. - 0.010, h - 0.010))}


def _laptop_pieces():
    """Base body, top deck tiles and the tilted lid; returns (label, lo, hi, colour, finish)."""
    bx, by, bz = LAPTOP_BASE[0] / 2., LAPTOP_BASE[1] / 2., LAPTOP_BASE[2]
    top = (bz, bz + LAPTOP_DECK_T)
    alu, dark, pad = [0.68, 0.69, 0.71], [0.07, 0.07, 0.08], [0.55, 0.56, 0.58]
    finish_alu, finish_plastic = (0.35, 0.80), (0.55, 0.0)
    tiles = [('body', (-bx, -by, 0.), (bx, by, bz), alu, finish_alu)]
    for label, (x0, x1), (y0, y1), colour, finish in (
            ('keyboard', (-0.035, 0.100), (-0.145, 0.145), dark, finish_plastic),
            ('trackpad', (-0.100, -0.045), (-0.055, 0.055), pad, (0.30, 0.0)),
            ('deck_far', (0.100, bx), (-by, by), alu, finish_alu),
            ('deck_left', (-0.035, 0.100), (0.145, by), alu, finish_alu),
            ('deck_right', (-0.035, 0.100), (-by, -0.145), alu, finish_alu),
            ('deck_mid', (-0.045, -0.035), (-by, by), alu, finish_alu),
            ('deck_near', (-bx, -0.100), (-by, by), alu, finish_alu),
            ('palm_left', (-0.100, -0.045), (0.055, by), alu, finish_alu),
            ('palm_right', (-0.100, -0.045), (-by, -0.055), alu, finish_alu)):
        tiles.append((label, (x0, y0, top[0]), (x1, y1, top[1]), colour, finish))
    return tiles


def _laptop():
    """An open laptop: base box and a lid tilted 20 deg back, both analytic boxes."""
    asset = Asset('laptop', dynamic=False, mass_kg=0.0, friction=0.50)
    for label, lo, hi, colour, (roughness, metallic) in _laptop_pieces():
        asset.add_display(_box_at(lo, hi), colour, label, roughness=roughness, metallic=metallic)
    rotation, hinge = _laptop_lid_transform()
    t, w, h = LAPTOP_LID
    boxes = _laptop_lid_boxes()
    asset.add_display(_moved(_box_at(*boxes['lid']), hinge, rotation), [0.68, 0.69, 0.71], 'lid',
                      roughness=0.35, metallic=0.80)
    asset.add_display(_moved(_box_at(*boxes['screen']), hinge, rotation), [0.16, 0.30, 0.46], 'screen',
                      roughness=0.12, metallic=0.0)
    base_h = LAPTOP_BASE[2] + LAPTOP_DECK_T
    asset.add_collision(None, 'base', geometry='box',
                        half_extents_m=[LAPTOP_BASE[0] / 2., LAPTOP_BASE[1] / 2., base_h / 2.],
                        position_m=[0., 0., base_h / 2.])
    lid_centre = hinge + rotation @ np.array([0., 0., h / 2.])
    asset.add_collision(None, 'lid', geometry='box', half_extents_m=[t / 2., w / 2., h / 2.],
                        position_m=[float(v) for v in lid_centre],
                        quaternion_xyzw=_quat_xyzw_y(LAPTOP_TILT_DEG))
    asset.data.update(
        nominal_size_m=DESIGN_SIZES['laptop'],
        size_note='14-inch class laptop, 320 x 220 mm, open; the screen faces local -x',
        static_note='Fixture: furniture the robot works around, never a manipulation target',
        collision_note='Two analytic boxes: the 17 mm base and the 5 mm lid at its 20 deg tilt',
        rest_note='The base bottom stands on the desk top at z = 0.750 m')
    return asset


def _clock_hand(length, width, tail, angle_deg, z0, z1):
    """A clock hand from ``tail`` behind the centre to ``length`` in front; clockwise from 12."""
    hand = _box_at((-width / 2., -tail, z0), (width / 2., length, z1))
    return _moved(hand, (0., 0., 0.), _rot_z(-angle_deg))


def _wall_clock():
    """Round wall clock, authored facing local +z with 12 o'clock at local +y."""
    asset = Asset('wall_clock', dynamic=False, mass_kg=0.0, friction=0.50)
    asset.add_display(_moved(annulus_mesh(CLOCK_R, CLOCK_FACE_R + 0.0005, CLOCK_DEPTH), (0., 0., CLOCK_DEPTH / 2.)),
                      [0.20, 0.13, 0.09], 'case', roughness=0.45, metallic=0.0)
    asset.add_display(_moved(cylinder_mesh(CLOCK_FACE_R, CLOCK_FACE_TOP), (0., 0., CLOCK_FACE_TOP / 2.)),
                      [0.95, 0.94, 0.90], 'face', roughness=0.60, metallic=0.0)
    z0, z1 = CLOCK_FACE_TOP, CLOCK_FACE_TOP + 0.0015
    for hour in range(12):
        size = 0.020 if hour % 3 == 0 else 0.012
        tick = _box_at((-0.003, 0.118 - size, z0), (0.003, 0.118, z1))
        asset.add_display(_moved(tick, (0., 0., 0.), _rot_z(-30. * hour)), [0.08, 0.08, 0.09],
                          f'tick_{hour:02d}', roughness=0.50, metallic=0.0)
    asset.add_display(_clock_hand(0.070, 0.008, 0.012, 305., z1, z1 + 0.0015), [0.08, 0.08, 0.09],
                      'hour_hand', roughness=0.50, metallic=0.0)
    asset.add_display(_clock_hand(0.105, 0.005, 0.015, 60., z1 + 0.0015, z1 + 0.003), [0.08, 0.08, 0.09],
                      'minute_hand', roughness=0.50, metallic=0.0)
    asset.add_display(_moved(cylinder_mesh(0.006, 0.0025, sections=24), (0., 0., z1 + 0.003 + 0.00125)),
                      [0.70, 0.53, 0.25], 'centre_cap', roughness=0.30, metallic=0.85)
    asset.add_collision(None, 'body', geometry='cylinder', radius_m=CLOCK_R, half_height_m=CLOCK_DEPTH / 2.,
                        position_m=[0., 0., CLOCK_DEPTH / 2.])
    asset.data.update(
        nominal_size_m=DESIGN_SIZES['wall_clock'],
        size_note='300 mm wall clock, 45 mm deep, hands at 10:10',
        static_note='Fixture: hung on the window wall, out of the robot reach',
        collision_note='One analytic cylinder over the case',
        rest_note='The back face (local z = 0) lies on the front wall plane')
    return asset


def _leaves():
    """Deterministic leaf clumps: (label, centre, axes, frame, colour)."""
    leaves = []
    greens = ([0.20, 0.42, 0.18], [0.27, 0.50, 0.22], [0.16, 0.36, 0.15])
    for index in range(8):
        yaw = math.radians(index * 45. + (12. if index % 2 else 0.))
        pitch = math.radians(38. if index % 2 else 24.)
        radial = np.array([math.cos(yaw), math.sin(yaw), 0.])
        axis = np.array([math.cos(yaw) * math.cos(pitch), math.sin(yaw) * math.cos(pitch), math.sin(pitch)])
        side = np.array([-math.sin(yaw), math.cos(yaw), 0.])
        normal = np.cross(axis, side)
        centre = np.array([0., 0., 0.20 + 0.03 * (index % 3)]) + radial * 0.055 + axis * 0.02
        leaves.append((f'leaf_{index:02d}', centre, (0.068, 0.028, 0.006),
                       np.column_stack([axis, side, normal]), greens[index % 3]))
    for index in range(3):
        yaw = math.radians(60. + index * 120.)
        axis = np.array([0.35 * math.cos(yaw), 0.35 * math.sin(yaw), 0.94])
        axis /= np.linalg.norm(axis)
        side = np.array([-math.sin(yaw), math.cos(yaw), 0.])
        normal = np.cross(axis, side)
        centre = np.array([0., 0., 0.31]) + axis * 0.035
        leaves.append((f'shoot_{index:02d}', centre, (0.055, 0.022, 0.006),
                       np.column_stack([axis, side, normal]), greens[(index + 1) % 3]))
    return leaves


def _leaf_bounds():
    points = np.vstack([_ellipsoid_points(c, a, f) for _, c, a, f, _ in _leaves()])
    return points.min(axis=0), points.max(axis=0)


def _potted_plant():
    """A small potted plant: terracotta lathe pot, soil, stem and convex leaf clumps."""
    asset = Asset('potted_plant', dynamic=False, mass_kg=0.0, friction=0.60)
    asset.add_display(lathe_mesh(POT_PROFILE, segments=64), [0.66, 0.36, 0.24], 'pot', roughness=0.80,
                      metallic=0.0)
    asset.add_display(_moved(cylinder_mesh(0.061, 0.006, sections=48), (0., 0., 0.121)), [0.18, 0.12, 0.08],
                      'soil', roughness=0.95, metallic=0.0)
    asset.add_display(_moved(cylinder_mesh(0.005, 0.20, sections=16), (0., 0., 0.124 + 0.10)),
                      [0.30, 0.38, 0.16], 'stem', roughness=0.70, metallic=0.0)
    for label, centre, axes, frame, colour in _leaves():
        asset.add_display(_hull(_ellipsoid_points(centre, axes, frame)), colour, label, roughness=0.65,
                          metallic=0.0)
    low, high = _leaf_bounds()
    asset.add_collision(None, 'pot', geometry='cylinder', radius_m=POT_R, half_height_m=POT_H / 2.,
                        position_m=[0., 0., POT_H / 2.])
    asset.add_collision(None, 'foliage', geometry='box',
                        half_extents_m=[float(v) for v in (high - low) / 2.],
                        position_m=[float(v) for v in (high + low) / 2.])
    asset.data.update(
        nominal_size_m=DESIGN_SIZES['potted_plant'],
        size_note='148 mm terracotta pot, foliage to about 0.38 m',
        static_note='Fixture: stands on the bookcase top, out of the robot reach',
        collision_note='An analytic cylinder over the pot and a box over the leaf clumps',
        rest_note='The pot bottom stands on the bookcase top at z = 1.150 m')
    return asset


def _basket_faces():
    """The 12 flat wall faces of the basket: (angle_deg, centre_radius, half_thickness, half_width)."""
    step = 360. / BASKET_SIDES
    flat = math.cos(math.pi / BASKET_SIDES)
    outer, inner = BASKET_R * flat, (BASKET_R - BASKET_WALL) * flat
    return [((k + 0.5) * step, (outer + inner) / 2., (outer - inner) / 2.,
             BASKET_R * math.sin(math.pi / BASKET_SIDES)) for k in range(BASKET_SIDES)]


def _waste_basket():
    """A twelve-sided waste basket: lathe display, twelve wall boxes and a floor disc."""
    asset = Asset('waste_basket', dynamic=False, mass_kg=0.0, friction=0.60)
    profile = [(0., 0.), (BASKET_R, 0.), (BASKET_R, BASKET_H), (BASKET_R - BASKET_WALL, BASKET_H),
               (BASKET_R - BASKET_WALL, BASKET_FLOOR), (0., BASKET_FLOOR)]
    asset.add_display(lathe_mesh(profile, segments=BASKET_SIDES), [0.17, 0.18, 0.19], 'body',
                      roughness=0.60, metallic=0.0)
    for index, (angle, radius, half_t, half_w) in enumerate(_basket_faces()):
        rad = math.radians(angle)
        asset.add_collision(None, f'wall_{index:02d}', geometry='box',
                            half_extents_m=[half_t, half_w, BASKET_H / 2.],
                            position_m=[radius * math.cos(rad), radius * math.sin(rad), BASKET_H / 2.],
                            quaternion_xyzw=_quat_xyzw_z(angle))
    asset.add_collision(None, 'floor', geometry='cylinder',
                        radius_m=(BASKET_R - BASKET_WALL) * math.cos(math.pi / BASKET_SIDES),
                        half_height_m=BASKET_FLOOR / 2., position_m=[0., 0., BASKET_FLOOR / 2.])
    asset.data.update(
        nominal_size_m=DESIGN_SIZES['waste_basket'],
        size_note='260 mm across the corners, 300 mm tall, 6 mm wall; open top',
        static_note='Fixture: stands on the floor by the desk, out of the robot reach',
        collision_note='Twelve analytic wall boxes on the twelve flat faces and a floor disc; open on top',
        rest_note='Stands on the floor at z = 0')
    return asset


# --- design sizes (import-time truth; the test checks them against the manifest) -----

def _laptop_bounds():
    rotation, hinge = _laptop_lid_transform()
    corners = np.array([[x, y, z] for lo, hi in _laptop_lid_boxes().values()
                        for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
    lid = corners @ rotation.T + hinge
    low = np.minimum(lid.min(axis=0), [-LAPTOP_BASE[0] / 2., -LAPTOP_BASE[1] / 2., 0.])
    high = np.maximum(lid.max(axis=0), [LAPTOP_BASE[0] / 2., LAPTOP_BASE[1] / 2.,
                                        LAPTOP_BASE[2] + LAPTOP_DECK_T])
    return low, high


def _design_sizes():
    sizes = {asset_id: [w, d, t] for asset_id, w, d, t, _, _ in BOOK_SPECS}
    sizes.update({
        'notebook': list(NOTEBOOK_SIZE),
        'organiser_tray': list(TRAY_SIZE),
        'pen_cup': [2 * CUP_OUTER_R, 2 * CUP_OUTER_R, CUP_HEIGHT],
        'tape_roll': [2 * TAPE_OUTER_R, 2 * TAPE_OUTER_R, TAPE_WIDTH],
        'pen': [2 * PEN_APOTHEM, 2 * PEN_CIRCUMRADIUS, PEN_LENGTH],
        'pencil': [2 * PENCIL_APOTHEM, 2 * PENCIL_CIRCUMRADIUS, PENCIL_LENGTH],
    })
    shade_x = LAMP_SHADE_X - LAMP_SHADE_R[0]
    sizes['desk_lamp'] = [LAMP_BASE_R - min(shade_x, -LAMP_BASE_R),
                          2 * max(LAMP_BASE_R, LAMP_SHADE_R[0]), LAMP_STEM_TOP]
    low, high = _laptop_bounds()
    sizes['laptop'] = [float(v) for v in high - low]
    sizes['wall_clock'] = [2 * CLOCK_R, 2 * CLOCK_R, CLOCK_DEPTH]
    leaf_low, leaf_high = _leaf_bounds()
    sizes['potted_plant'] = [float(max(POT_R, leaf_high[0]) - min(-POT_R, leaf_low[0])),
                             float(max(POT_R, leaf_high[1]) - min(-POT_R, leaf_low[1])),
                             float(max(POT_H, leaf_high[2]))]
    sizes['waste_basket'] = [2 * BASKET_R, 2 * BASKET_R, BASKET_H]
    return sizes


DESIGN_SIZES = _design_sizes()

ASSET_BUILDERS = (
    *((spec[0], (lambda spec=spec: _book(*spec))) for spec in BOOK_SPECS),
    ('notebook', _notebook),
    ('organiser_tray', _organiser_tray),
    ('pen_cup', _pen_cup),
    ('tape_roll', _tape_roll),
    ('pen', _pen),
    ('pencil', _pencil),
    ('desk_lamp', _desk_lamp),
    ('laptop', _laptop),
    ('wall_clock', _wall_clock),
    ('potted_plant', _potted_plant),
    ('waste_basket', _waste_basket),
)
DYNAMIC_ASSETS = tuple(spec[0] for spec in BOOK_SPECS) + ('notebook', 'organiser_tray', 'pen_cup',
                                                          'tape_roll', 'pen', 'pencil')
FIXTURE_ASSETS = ('desk_lamp', 'laptop', 'wall_clock', 'potted_plant', 'waste_basket')
METAL_VISUALS = {('pen', 'tip'), ('pencil', 'ferrule'), ('desk_lamp', 'base'), ('desk_lamp', 'stem'),
                 ('desk_lamp', 'arm'), ('desk_lamp', 'holder'), ('wall_clock', 'centre_cap'),
                 ('laptop', 'lid'), ('laptop', 'body')} | {('laptop', label) for label in
                                                           ('deck_far', 'deck_left', 'deck_right', 'deck_mid',
                                                            'deck_near', 'palm_left', 'palm_right')}


def build_library(asset_root=ASSET_ROOT):
    """Write the authored OBJ library and return the manifest record.

    ``write_library`` refuses a non-empty target, so this is one-shot: rebuild into
    a new version directory and repoint ``SCENE`` rather than overwrite.
    """
    assets = [(asset_id, builder()) for asset_id, builder in ASSET_BUILDERS]
    return write_library(Path(asset_root), assets, generator=GENERATOR, license_id=LICENSE_ID,
                         extra={'family': FAMILY, 'units': 'metres',
                                'source': 'Procedural geometry authored for the study scene family',
                                'dimension_status': 'Simulation task design values; not manufacturer CAD',
                                'licence_note': 'Procedural geometry; no third-party mesh or texture copied',
                                'supersedes': list(SUPERSEDED_ASSET_ROOTS),
                                'collision_frame_note': 'Convex pieces have their offset baked in and '
                                                        'position_m zero; analytic pieces are placed '
                                                        'only by position_m / quaternion_xyzw'})


def load_manifest(asset_root=ASSET_ROOT):
    """The authored manifest, or ``None`` if the library is not built yet."""
    path = Path(asset_root) / 'manifest.json'
    if not path.is_file():
        return None
    import json
    return load_json(path)


def asset_sizes(asset_root=ASSET_ROOT):
    """AABB size of every asset: measured from the manifest when it exists."""
    manifest = load_manifest(asset_root)
    if not manifest:
        return {key: list(value) for key, value in DESIGN_SIZES.items()}
    return {key: list(record['size_m']) for key, record in manifest['assets'].items()}


# --- the scene layout -------------------------------------------------------------

def rest_extent(size, quaternion):
    """World-axis extent of a centred asset box in its rest orientation."""
    return np.abs(_rotation(quaternion)) @ np.asarray(size, dtype=float)


def _objects(sizes):
    objects = []
    for object_id, asset_id, x, y, quaternion, friction in OBJECT_LAYOUT:
        extent = rest_extent(sizes[asset_id], quaternion)
        objects.append({
            'id': object_id, 'asset_id': asset_id, 'asset_root': ASSET_ROOT,
            'position_m': [_r(x), _r(y), _r(PAD_TOP_Z + extent[2] / 2.)],
            'quaternion_wxyz': [round(float(v), 9) for v in quaternion],
            'friction': friction,
            'rest_geometry': {'size_m': _r3(extent), 'origin': 'centre'},
            'rest_note': f'Lies on desk_pad at z = {PAD_TOP_Z:.3f} m; nothing above or below it',
        })
    return objects


def lamp_bulb_world():
    """World centre of the desk-lamp bulb (the sphere light sits here)."""
    _, _, position, quaternion, _ = FIXTURE_LAYOUT[0]
    return np.asarray(position, dtype=float) + _rotation(quaternion) @ np.asarray(LAMP_BULB_LOCAL)


def _fixtures():
    return [{'id': fixture_id, 'asset_id': asset_id, 'asset_root': ASSET_ROOT,
             'position_m': _r3(position), 'quaternion_wxyz': [round(float(v), 9) for v in quaternion],
             'friction': friction}
            for fixture_id, asset_id, position, quaternion, friction in FIXTURE_LAYOUT]


def _box(part_id, lo, hi, material, **extra):
    """One static box part from its two corners."""
    lo, hi = np.asarray(lo, dtype=float), np.asarray(hi, dtype=float)
    part = {'id': part_id, 'geometry': 'box', 'half_extents_m': _r3((hi - lo) / 2.),
            'position_m': _r3((hi + lo) / 2.), 'material': material}
    part.update(extra)
    return part


def _supports():
    return [{'id': 'desk_pad', 'title_zh': '皮质桌垫',
             'position_m': [_r((PAD_X[0] + PAD_X[1]) / 2.), _r((PAD_Y[0] + PAD_Y[1]) / 2.),
                            _r(DESK_TOP_Z + PAD_T / 2.)],
             'half_extents_m': [_r((PAD_X[1] - PAD_X[0]) / 2.), _r((PAD_Y[1] - PAD_Y[0]) / 2.), _r(PAD_T / 2.)],
             'material': 'leather_pad', 'friction': 0.60,
             'note': ('A 4 mm leather desk pad lying on the desk slab: the only declared support. '
                      'It is not a copy of the table collider')}]


def _textured(material_id, uv_scale, color, note):
    """A CC0 ambientCG set from the shared library: colour, roughness and OpenGL normal maps."""
    root = Path(MATERIAL_SOURCE_ROOT) / material_id
    return {'kind': 'textured', 'color': list(color), 'metallic': 0.0,
            'uv_scale_m': list(uv_scale), 'uv_scale_semantics': 'tiles_per_meter',
            'base_color': str(root / f'{material_id}_1K-JPG_Color.jpg'),
            'roughness_map': str(root / f'{material_id}_1K-JPG_Roughness.jpg'),
            'normal_map': str(root / f'{material_id}_1K-JPG_NormalGL.jpg'),
            'source': TEXTURE_SOURCES[material_id], 'license': LICENSE_ID, 'note': note}


def _materials():
    def color(rgb, roughness, metallic=0.0):
        return {'kind': 'color', 'color': list(rgb), 'roughness': roughness, 'metallic': metallic}
    return {
        # room shell
        'oak_floor': _textured('WoodFloor064', (0.8, 0.8), [0.58, 0.40, 0.24], 'honey oak plank floor'),
        'plaster_wall': _textured('Plaster001', (0.5, 0.5), [0.86, 0.85, 0.82], 'white plaster walls'),
        'ceiling_white': color([0.92, 0.92, 0.90], 0.92),
        'baseboard_white': color([0.90, 0.89, 0.86], 0.50),
        # furniture
        'desk_wood': _textured('Wood089', (1.2, 1.2), [0.66, 0.55, 0.42], 'pale ash desk and drawers'),
        'shelf_wood': _textured('Wood089', (1.6, 1.6), [0.66, 0.55, 0.42], 'pale ash bookcase'),
        'walnut': color([0.30, 0.20, 0.13], 0.45),
        'leather_pad': color([0.12, 0.12, 0.13], 0.55),
        'brass': color([0.70, 0.53, 0.25], 0.30, 0.90),
        'seat_fabric': color([0.36, 0.42, 0.33], 0.95),
        'paper': color([0.93, 0.92, 0.88], 0.82),
        'cardboard': color([0.60, 0.46, 0.30], 0.88),
        'rug_wool': color([0.42, 0.18, 0.16], 0.97),
        'curtain_linen': color([0.74, 0.70, 0.60], 0.95),
        'blind_fabric': color([0.88, 0.86, 0.80], 0.90),
        'window_white': color([0.93, 0.93, 0.91], 0.40),
        'door_paint': color([0.90, 0.89, 0.86], 0.45),
        'frame_black': color([0.06, 0.06, 0.07], 0.40),
        'canvas_white': color([0.94, 0.93, 0.89], 0.85),
        'paint_red': color([0.70, 0.12, 0.10], 0.70),
        'paint_blue': color([0.10, 0.22, 0.52], 0.70),
        'paint_yellow': color([0.92, 0.75, 0.16], 0.70),
        'paint_black': color([0.05, 0.05, 0.05], 0.70),
        'cover_navy': color([0.13, 0.18, 0.32], 0.60),
        'cover_maroon': color([0.42, 0.11, 0.12], 0.60),
        'cover_olive': color([0.33, 0.36, 0.18], 0.60),
        'cover_ochre': color([0.70, 0.52, 0.22], 0.60),
        'cover_grey': color([0.42, 0.43, 0.45], 0.60),
        'cover_teal': color([0.12, 0.36, 0.38], 0.60),
        'cover_cream': color([0.84, 0.80, 0.70], 0.60),
        'cover_black': color([0.08, 0.08, 0.09], 0.60),
        # add_pedestal binds exactly these two names: the slab ('top') and its skirt ('side')
        'top': color([0.25, 0.26, 0.28], 0.45, 0.35),
        'side': color([0.33, 0.34, 0.36], 0.60, 0.10),
    }


def _desk_frame():
    """Three legs, a modesty panel and a drawer pedestal under the desk slab."""
    x0, x1 = DESK_X
    y0, y1 = DESK_Y
    top = DESK_UNDERSIDE_Z
    leg = DESK_LEG
    wood = {'uv_scale': [1.2, 1.2]}
    parts = [
        # Named from the robot, which faces +x: its right is -y. The near-left corner is
        # carried by the drawer pedestal instead of a leg.
        _box('leg_near_right', (x0 + 0.02, y0 + 0.02, 0.), (x0 + 0.02 + leg, y0 + 0.02 + leg, top), 'walnut'),
        _box('leg_far_right', (x1 - 0.02 - leg, y0 + 0.02, 0.), (x1 - 0.02, y0 + 0.02 + leg, top), 'walnut'),
        _box('leg_far_left', (x1 - 0.02 - leg, y1 - 0.02 - leg, 0.), (x1 - 0.02, y1 - 0.02, top), 'walnut'),
        # Between the two far legs, under the slab: hides the knee space from the room.
        _box('modesty_panel', (x1 - 0.035, y0 + 0.02 + leg, 0.40), (x1 - 0.02, y1 - 0.02 - leg, top),
             'desk_wood', **wood),
        _box('drawer_carcass', (0.30, 0.56, 0.), (0.72, 0.98, top), 'desk_wood', **wood),
    ]
    for index, (z0, z1) in enumerate(((0.020, 0.240), (0.250, 0.470), (0.480, 0.700))):
        parts.append(_box(f'drawer_front_{index}', (0.285, 0.57, z0), (0.30, 0.97, z1), 'desk_wood', **wood))
        middle = (z0 + z1) / 2.
        parts.append(_box(f'drawer_handle_{index}', (0.272, 0.72, middle - 0.008), (0.285, 0.82, middle + 0.008),
                          'brass'))
    return {'id': 'desk_frame', 'title_zh': '书桌框架与抽屉柜', 'material': 'desk_wood', 'friction': 0.65,
            'parts': parts}


def _bookcase():
    """The bookcase carcass: two sides, a back, a plinth (the bottom shelf), two shelves and a top."""
    x0, x1 = BOOKCASE_X
    y0, y1 = BOOKCASE_Y
    p, back, h = BOOKCASE_PANEL, BOOKCASE_BACK, BOOKCASE_H
    wood = {'uv_scale': [1.6, 1.6]}
    parts = [
        _box('side_left', (x0, y0, 0.), (x0 + p, y1, h), 'shelf_wood', **wood),
        _box('side_right', (x1 - p, y0, 0.), (x1, y1, h), 'shelf_wood', **wood),
        _box('back_panel', (x0 + p, y1 - back, 0.), (x1 - p, y1, h - p), 'shelf_wood', **wood),
        _box('top_panel', (x0 + p, y0, h - p), (x1 - p, y1, h), 'shelf_wood', **wood),
    ]
    for name, bottom in zip(('plinth', 'shelf_low', 'shelf_mid'), BOOKCASE_BOARD_BOTTOMS):
        lo_z = 0. if name == 'plinth' else bottom
        parts.append(_box(name, (x0 + p, y0, lo_z), (x1 - p, y1 - back, bottom + p), 'shelf_wood', **wood))
    return {'id': 'bookcase', 'title_zh': '矮书柜', 'material': 'shelf_wood', 'friction': 0.60, 'parts': parts}


BOOKCASE_RUNS = (
    # compartment, [(width, height, depth, material), ...] from the left side panel
    (0, ((0.040, 0.300, 0.230, 'cover_navy'), (0.034, 0.285, 0.215, 'cover_maroon'),
         (0.046, 0.310, 0.235, 'cover_olive'), (0.030, 0.260, 0.200, 'cover_ochre'),
         (0.038, 0.295, 0.225, 'cover_grey'), (0.042, 0.300, 0.220, 'cover_teal'),
         (0.036, 0.275, 0.210, 'cover_black'))),
    (1, ((0.045, 0.290, 0.230, 'cover_maroon'), (0.040, 0.300, 0.215, 'cover_navy'),
         (0.036, 0.270, 0.205, 'cover_ochre'), (0.032, 0.250, 0.190, 'cover_cream'),
         (0.044, 0.280, 0.220, 'cover_teal'))),
    (2, ((0.050, 0.290, 0.240, 'cover_olive'), (0.042, 0.280, 0.220, 'cover_navy'),
         (0.040, 0.250, 0.200, 'cover_maroon'), (0.038, 0.270, 0.215, 'cover_grey'),
         (0.034, 0.240, 0.190, 'cover_black'), (0.046, 0.300, 0.230, 'cover_ochre'))),
)


def _bookcase_contents():
    """Static books and storage boxes in the three compartments, plus a stack on top."""
    p = BOOKCASE_PANEL
    floors = [bottom + p for bottom in BOOKCASE_BOARD_BOTTOMS]
    left = BOOKCASE_X[0] + p
    parts = []
    for compartment, run in BOOKCASE_RUNS:
        x = left + 0.004
        for index, (width, height, depth, material) in enumerate(run):
            parts.append(_box(f'book_{compartment}_{index}', (x, BOOK_SPINE_Y, floors[compartment]),
                              (x + width, BOOK_SPINE_Y + depth, floors[compartment] + height), material))
            x += width + 0.001
    right = BOOKCASE_X[1] - p
    parts.append(_box('storage_box_low', (right - 0.004 - 0.300, BOOK_SPINE_Y, floors[0]),
                      (right - 0.004, BOOK_SPINE_Y + 0.270, floors[0] + 0.230), 'cardboard'))
    # A flat stack on the middle shelf, and a magazine box on the top shelf.
    stack_x = right - 0.004 - 0.250
    z = floors[1]
    for index, (thick, material) in enumerate(((0.030, 'cover_grey'), (0.026, 'cover_olive'),
                                               (0.022, 'cover_maroon'))):
        parts.append(_box(f'flat_book_{index}', (stack_x, BOOK_SPINE_Y, z), (stack_x + 0.240, BOOK_SPINE_Y + 0.175,
                                                                         z + thick), material))
        z += thick
    parts.append(_box('magazine_box', (right - 0.004 - 0.100, BOOK_SPINE_Y, floors[2]),
                      (right - 0.004, BOOK_SPINE_Y + 0.240, floors[2] + 0.280), 'cardboard'))
    # Two books lying on the bookcase top, left of the plant.
    z = BOOKCASE_H
    for index, (thick, material) in enumerate(((0.032, 'cover_navy'), (0.024, 'cover_cream'))):
        parts.append(_box(f'top_book_{index}', (0.40, 1.760, z), (0.64, 1.930, z + thick), material))
        z += thick
    return {'id': 'bookcase_books', 'title_zh': '书柜里的书与收纳盒', 'material': 'cover_navy',
            'friction': 0.60, 'parts': parts}


def _wall_art():
    """A framed abstract print above the bookcase, hung flat on the front wall."""
    (x0, x1), (z0, z1) = ART_X, ART_Z
    wall, bar = FRONT_WALL_Y, 0.025
    canvas_y = wall - 0.015
    paint_y = canvas_y - 0.004
    parts = [
        _box('frame_top', (x0, wall - 0.030, z1 - bar), (x1, wall, z1), 'frame_black'),
        _box('frame_bottom', (x0, wall - 0.030, z0), (x1, wall, z0 + bar), 'frame_black'),
        _box('frame_left', (x0, wall - 0.030, z0 + bar), (x0 + bar, wall, z1 - bar), 'frame_black'),
        _box('frame_right', (x1 - bar, wall - 0.030, z0 + bar), (x1, wall, z1 - bar), 'frame_black'),
        _box('canvas', (x0 + bar, canvas_y, z0 + bar), (x1 - bar, wall, z1 - bar), 'canvas_white'),
    ]
    cx0, cx1, cz0, cz1 = x0 + bar, x1 - bar, z0 + bar, z1 - bar
    for part_id, (u0, u1), (v0, v1), material in (
            ('paint_red', (0.06, 0.52), (0.40, 0.94), 'paint_red'),
            ('paint_blue', (0.60, 0.94), (0.06, 0.34), 'paint_blue'),
            ('paint_yellow', (0.06, 0.30), (0.06, 0.32), 'paint_yellow'),
            ('line_vertical', (0.54, 0.57), (0.06, 0.94), 'paint_black'),
            ('line_horizontal', (0.06, 0.94), (0.35, 0.38), 'paint_black')):
        lo = (cx0 + u0 * (cx1 - cx0), paint_y, cz0 + v0 * (cz1 - cz0))
        hi = (cx0 + u1 * (cx1 - cx0), canvas_y, cz0 + v1 * (cz1 - cz0))
        if part_id == 'line_horizontal':
            # Split around the vertical line so no two paint boxes overlap.
            split = (cx0 + 0.54 * (cx1 - cx0), cx0 + 0.57 * (cx1 - cx0))
            parts.append(_box('line_horizontal_a', lo, (split[0], hi[1], hi[2]), material))
            parts.append(_box('line_horizontal_b', (split[1], lo[1], lo[2]), hi, material))
            continue
        parts.append(_box(part_id, lo, hi, material))
    return {'id': 'wall_art', 'title_zh': '装饰画', 'material': 'frame_black', 'friction': 0.60,
            'parts': parts}


def _window_frame():
    """Frame, mullion, transom and roller blind in the reveal; stool, apron and casing in the room."""
    (x0, x1), (z0, z1) = WINDOW_X, WINDOW_Z
    wall = FRONT_WALL_Y
    ry = (wall + 0.020, wall + 0.070)                    # frame depth inside the wall reveal
    s, mid = 0.050, (x0 + x1) / 2.
    transom = ((z0 + z1) / 2. - 0.02, (z0 + z1) / 2. + 0.02)
    parts = [
        _box('stile_left', (x0, ry[0], z0), (x0 + s, ry[1], z1), 'window_white'),
        _box('stile_right', (x1 - s, ry[0], z0), (x1, ry[1], z1), 'window_white'),
        _box('rail_bottom', (x0 + s, ry[0], z0), (x1 - s, ry[1], z0 + s), 'window_white'),
        _box('rail_top', (x0 + s, ry[0], z1 - s), (x1 - s, ry[1], z1), 'window_white'),
        _box('transom', (x0 + s, ry[0], transom[0]), (x1 - s, ry[1], transom[1]), 'window_white'),
        _box('mullion_low', (mid - 0.02, ry[0], z0 + s), (mid + 0.02, ry[1], transom[0]), 'window_white'),
        _box('mullion_high', (mid - 0.02, ry[0], transom[1]), (mid + 0.02, ry[1], z1 - s), 'window_white'),
        # Roller blind: a tube across the head of the reveal, the blind pulled a third of the way down.
        _box('blind_roller', (x0, wall, z1 - 0.050), (x1, wall + 0.020, z1), 'window_white'),
        _box('blind', (x0 + 0.010, wall + 0.006, 1.55), (x1 - 0.010, wall + 0.014, z1 - 0.050), 'blind_fabric'),
        # Interior trim on the wall face.
        _box('stool', (x0 - 0.04, wall - 0.14, z0 - 0.03), (x1 + 0.04, wall, z0), 'window_white'),
        _box('apron', (x0 - 0.02, wall - 0.015, z0 - 0.10), (x1 + 0.02, wall, z0 - 0.03), 'window_white'),
        _box('casing_left', (x0 - 0.07, wall - 0.015, z0), (x0, wall, z1 + 0.07), 'window_white'),
        _box('casing_right', (x1, wall - 0.015, z0), (x1 + 0.07, wall, z1 + 0.07), 'window_white'),
        _box('casing_head', (x0, wall - 0.015, z1), (x1, wall, z1 + 0.07), 'window_white'),
    ]
    return {'id': 'window_frame', 'title_zh': '窗框与卷帘', 'material': 'window_white', 'friction': 0.60,
            'parts': parts}


def _curtains():
    """Floor-length linen curtains on a rod with two wall brackets."""
    wall = FRONT_WALL_Y
    rod_y, rod_z = (wall - 0.105, wall - 0.085), (2.18, 2.20)
    parts = [
        _box('rod', (-1.76, rod_y[0], rod_z[0]), (0.30, rod_y[1], rod_z[1]), 'brass'),
        _box('bracket_left', (-1.72, rod_y[1], rod_z[0] - 0.01), (-1.69, wall, rod_z[1] + 0.01), 'brass'),
        _box('bracket_right', (0.25, rod_y[1], rod_z[0] - 0.01), (0.28, wall, rod_z[1] + 0.01), 'brass'),
    ]
    # Pleats hang in front of the rod (as if on rings), clear of the window stool.
    for side, start, pleat in (('left', -1.72, 0.10), ('right', -0.10, 0.09)):
        for index in range(4):
            depth = 0.040 if index % 2 == 0 else 0.030
            x = start + pleat * index
            parts.append(_box(f'curtain_{side}_{index}', (x, rod_y[0] - depth, 0.03), (x + pleat, rod_y[0], rod_z[1]),
                              'curtain_linen'))
    return {'id': 'curtains', 'title_zh': '窗帘', 'material': 'curtain_linen', 'friction': 0.70, 'parts': parts}


def _chair_and_papers():
    cx, cy = CHAIR_CENTER_XY
    seat_t, back_t = 0.050, 0.040
    seat_lo, seat_hi = (cx - 0.23, cy - 0.22), (cx + 0.23, cy + 0.22)
    legs = []
    for name, (sx, sy) in (('a', (-1, -1)), ('b', (1, -1)), ('c', (-1, 1)), ('d', (1, 1))):
        lx = cx + sx * 0.19
        ly = cy + sy * 0.18
        legs.append(_box(f'leg_{name}', (lx - 0.02, ly - 0.02, RUG_T), (lx + 0.02, ly + 0.02,
                                                                          CHAIR_SEAT_TOP - seat_t), 'walnut'))
    chair = {'id': 'chair', 'title_zh': '椅子', 'material': 'walnut', 'friction': 0.70, 'parts': [
        _box('seat', (*seat_lo, CHAIR_SEAT_TOP - seat_t), (*seat_hi, CHAIR_SEAT_TOP), 'seat_fabric'),
        _box('backrest', (seat_lo[0], seat_lo[1] + 0.02, CHAIR_SEAT_TOP),
             (seat_lo[0] + back_t, seat_hi[1] - 0.02, CHAIR_SEAT_TOP + 0.47), 'walnut'),
        *legs]}
    papers = {'id': 'paper_stack', 'title_zh': '椅面上的一叠纸', 'material': 'paper', 'friction': 0.60, 'parts': [
        _box('papers', (cx - 0.13, cy - 0.115, CHAIR_SEAT_TOP), (cx + 0.167, cy + 0.095, CHAIR_SEAT_TOP + 0.012),
             'paper')]}
    return chair, papers


def _rug():
    return {'id': 'rug', 'title_zh': '地毯', 'material': 'rug_wool', 'friction': 0.75, 'parts': [
        _box('rug', (RUG_X[0], RUG_Y[0], 0.), (RUG_X[1], RUG_Y[1], RUG_T), 'rug_wool')]}


def _door():
    """A closed panel door in the back-wall reveal, with casing and a lever handle."""
    (x0, x1), (z0, z1) = DOOR_X, DOOR_Z
    wall = BACK_WALL_Y
    leaf_y = (wall - 0.080, wall - 0.040)
    room_face = leaf_y[1]
    parts = [_box('leaf', (x0, leaf_y[0], z0), (x1, leaf_y[1], z1), 'door_paint')]
    for index, (u0, u1, v0, v1) in enumerate(((0.12, 0.88, 0.56, 0.92), (0.12, 0.88, 0.08, 0.46))):
        parts.append(_box(f'panel_{index}', (x0 + u0 * (x1 - x0), room_face, z0 + v0 * (z1 - z0)),
                          (x0 + u1 * (x1 - x0), room_face + 0.006, z0 + v1 * (z1 - z0)), 'door_paint'))
    parts += [
        _box('handle_plate', (x1 - 0.105, room_face, 0.95), (x1 - 0.075, room_face + 0.008, 1.12), 'brass'),
        _box('handle_lever', (x1 - 0.175, room_face + 0.008, 1.02), (x1 - 0.075, room_face + 0.020, 1.036), 'brass'),
        _box('casing_left', (x0 - 0.07, wall, z0), (x0, wall + 0.015, z1 + 0.07), 'window_white'),
        _box('casing_right', (x1, wall, z0), (x1 + 0.07, wall + 0.015, z1 + 0.07), 'window_white'),
        _box('casing_head', (x0, wall, z1), (x1, wall + 0.015, z1 + 0.07), 'window_white'),
    ]
    return {'id': 'door', 'title_zh': '房门', 'material': 'door_paint', 'friction': 0.60, 'parts': parts}


def _static_props():
    chair, papers = _chair_and_papers()
    return [_desk_frame(), _bookcase(), _bookcase_contents(), _wall_art(), _window_frame(), _curtains(),
            chair, papers, _rug(), _door()]


def _opening(opening_id, wall, x_range, z_range, kind):
    """A room opening from world x and z ranges (front: u = x; back: u = -x)."""
    centre_x = (x_range[0] + x_range[1]) / 2. - ROOM_ORIGIN[0]
    u = centre_x if wall == 'front' else -centre_x
    v = (z_range[0] + z_range[1]) / 2. - ROOM_HEIGHT / 2.
    return {'id': opening_id, 'wall': wall, 'center_m': [_r(u), _r(v)],
            'size_m': [_r(x_range[1] - x_range[0]), _r(z_range[1] - z_range[0])], 'kind': kind}


_OBJECTS = _objects(asset_sizes())
LAMP_BULB_WORLD = lamp_bulb_world()

THIRD_PERSON_EYE = (1.750, -1.150, 1.450)
DESK_VIEW_EYE = (1.550, -0.550, 1.400)

SCENE = {
    'schema_version': 'isaacsim-scene-family-v1',
    'family': FAMILY,
    'title_zh': '书房',
    'description_zh': ('书房场景：浅色实木书桌上铺着皮质桌垫，垫上平放着三本厚度与封面色各不相同的硬皮书、'
                       '一本笔记本、一个小木托盘、一只空的陶瓷笔筒、一卷美纹纸胶带、一支圆珠笔和一支六棱铅笔，'
                       '全部是可自由移动的物体，没有任何叠放；书桌另一端是打开的笔记本电脑和会发光的摇臂台灯'
                       '（固定物），桌下有抽屉柜。窗墙前立着装满书和收纳盒的矮书柜，柜上有一盆绿植，'
                       '上方挂着装饰画；旁边是带卷帘与落地窗帘的窗户和一只挂钟。椅子放在地毯上，'
                       '座面上有一叠纸，书桌旁有废纸篓，房门在对面墙上。'),
    'room': {
        'enabled': True,
        'size_m': list(ROOM_SIZE),
        'origin_m': list(ROOM_ORIGIN),
        'wall_thickness_m': WALL_T,
        'baseboard_m': 0.09,
        'uv_scale': {'floor': [0.8, 0.8], 'wall': [0.5, 0.5], 'ceiling': [1.0, 1.0], 'baseboard': [1.0, 1.0]},
        'surfaces': {'floor': 'oak_floor', 'wall': 'plaster_wall',
                     'ceiling': 'ceiling_white', 'baseboard': 'baseboard_white'},
        'openings': [
            _opening('window', 'front', WINDOW_X, WINDOW_Z, 'window'),
            _opening('door', 'back', DOOR_X, DOOR_Z, 'door'),
        ],
        'ceiling_lights': [
            {'id': 'panel_desk', 'position_m': [0.55, 0.30, 2.58], 'extent_m': [0.80, 0.60],
             'intensity': 8000.0, 'color': [1.0, 0.97, 0.91]},
            {'id': 'panel_room', 'position_m': [-0.95, 0.85, 2.58], 'extent_m': [0.60, 0.60],
             'intensity': 4500.0, 'color': [1.0, 0.98, 0.94]},
            # The desk-lamp bulb, inside the open shade. Uncalibrated design value: by the
            # inverse-square law its pool beside the pad is about a third of the desk panel's light.
            {'id': 'desk_lamp_bulb', 'shape': 'sphere', 'position_m': _r3(LAMP_BULB_WORLD),
             'extent_m': [LAMP_BULB_R, LAMP_BULB_R], 'intensity': 40000.0, 'color': [1.0, 0.84, 0.62]},
        ],
        'dome': {'enabled': True, 'intensity': 350.0, 'color': [0.85, 0.90, 1.0], 'hdri_visible': False},
        'collide': {'floor': True, 'walls': True, 'ceiling': False},
    },
    'materials': _materials(),
    # 0.5 ms steps give 40 physics steps per 20 ms control window (bench convention).
    'physics': {'dt_s': 0.0005, 'control_dt_s': 0.02, 'gravity_m_s2': [0.0, 0.0, -9.81]},
    'cameras': [
        # The room view: 16 mm on the USD default 20.955 mm aperture (about a 28 mm
        # full-frame lens), low enough to see the window wall above the desk.
        {'id': 'third_person', 'kind': 'world', 'role': 'third_person',
         'eye': list(THIRD_PERSON_EYE), 'target': [DESK_CENTER_XY[0], DESK_CENTER_XY[1], DESK_TOP_Z],
         'focal_length_mm': 16.0, 'resolution': [960, 720], 'title_zh': '第三人称视角'},
        {'id': 'desk_view', 'kind': 'world', 'role': 'table',
         'eye': list(DESK_VIEW_EYE), 'target': [0.60, 0.30, 0.76],
         'focal_length_mm': 22.0, 'resolution': [960, 720], 'title_zh': '桌面视角'},
        {'id': 'wrist', 'kind': 'link', 'role': 'wrist', 'mount_link': 'right_hand_C_MC',
         'offset_xyz_m': [0.0, 0.0, 0.06], 'offset_rpy_rad': [0.0, 0.0, 0.0],
         # Aimed at the middle of the desk pad and stood off along the view axis so the
         # hand the camera rides does not fill its own frame.
         'aim_target_m': _r3(WORK_CENTER), 'standoff_m': 0.06,
         'look_axis': '+z', 'clipping_m': [0.02, 20.0], 'resolution': [640, 480], 'title_zh': '手腕相机'},
    ],
    # The mount transform is deliberately omitted: it is inherited from the
    # isaacsim_6_1_v2 SHARPA assembly, and the validator rejects a drifting copy.
    'robot': {'base_position_m': [0.0, 0.0, ROBOT_BASE_Z], 'hand_id': 'sharpa_w01_right',
              'assembly_version': 'isaacsim_6_1_v2', 'assets_root': f'{DATA_ROOT}/assets',
              'arm_home_rad': [0.4, -0.5, 0.0, -2.3, 0.0, 1.8, 0.8],
              'arm_kp': 2000.0, 'arm_kd': 140.0, 'hand_kp': 3.0, 'hand_kd': 0.1},
    # The slab carries the robot; its skirt (bound to 'side') carries the slab to the floor.
    'pedestal': {'position_m': [0.0, 0.0, PEDESTAL_CENTER_Z], 'half_extents_m': list(PEDESTAL_HALF),
                 'skirt_height_m': _r(PEDESTAL_SKIRT)},
    # The desk slab is the work table; scene.py builds it once as a solid static prop.
    'table': {'position_m': [_r(DESK_CENTER_XY[0]), _r(DESK_CENTER_XY[1]), _r(DESK_SLAB_CENTER_Z)],
              'half_extents_m': [_r(DESK_HALF_XY[0]), _r(DESK_HALF_XY[1]), _r(DESK_SLAB_T / 2.)],
              'friction': 0.55, 'material': 'desk_wood', 'uv_scale': [1.2, 1.2]},
    'surfaces': _supports(),
    'static_props': _static_props(),
    'fixtures': _fixtures(),
    'objects': _OBJECTS,
    'cloth': {'enabled': False},
    'isaacsim': {'physics_dt_s': 0.0005, 'robot_self_collisions': True,
                 'filter_thumb_cmc_structure': True},
}
