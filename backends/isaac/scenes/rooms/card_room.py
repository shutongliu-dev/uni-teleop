"""The ``card_room`` scene family: a Chinese chess-and-card room (棋牌室).

What is in the room
-------------------
An 0.86 m square mahjong / card table stands in front of the robot: a dark-wood
slab on four legs and an apron, with a green felt play surface inset 50 mm inside
a raised rim. On the felt lie the pieces of the three games played in such a
room:

* xiangqi (Chinese chess): a wooden board with its grid, river and palaces, and
  eight pieces standing on its intersections in a mid-game position;
* mahjong: the robot's own hand of eight tiles standing on edge in a row facing
  it, three discards lying face up, and two dice;
* poker: a chip tray with a chip in each of its three wells, two chip stacks, two
  loose chips, a deck and a loose card.

Two handleless teacups and a resin dealer ring complete the table. Three chairs
stand at the other three sides of the table (the robot takes the fourth seat), a
square mahjong lamp hangs over it, and against the front wall stands a tea
cabinet carrying a tea set, a mahjong case and boxed card decks, under a framed
calligraphy panel. The right wall has a trimmed door and the left wall a trimmed window.

Conventions
-----------
* ``room.openings`` are wall-local (u, v) measured from the wall *centre*
  (``room._wall_geometry``), so world z = ROOM_HEIGHT / 2 + v. The door frame and
  window trim are derived from the same constants through ``opening_world_rect``.
* Every box is declared exactly once: the slab is ``SCENE['table']`` (dark wood),
  the felt and the xiangqi board are ``SCENE['surfaces']`` (the solid plates the
  pieces stand on), and no static prop repeats either.
* The layout is literal data (``LAYOUT``), not solved at import. A piece's z is
  its support's top plus the half height of its collider at the declared
  orientation (``COLLIDER_HALF_M``), so importing the module never reads the
  asset library and the initial state is identical everywhere.
  ``tests/test_room_card_room.py`` re-derives the clearances and the
  collider sizes from the authored manifest, whose digest is pinned below.

Physical discipline
-------------------
* Every dynamic object is an authored OBJ asset (``assets/card_room/v2``) with a
  positive design mass, a uniform-density bounding-box inertia and convex
  collision pieces that agree with its display mesh within 0.5 mm.
* Chips are 3.3 mm, cards 3 mm (the contract's floor for a thin dynamic part; a
  real card is ~0.3 mm), and every thin part has an analytic box or cylinder
  collider, never a hull over a thin plate. Stacks (chips, the deck) are one
  rigid body each with a single collider spanning the stack.
* The dealer ring's 41 mm bore is a genuine through hole: 24 convex sectors
  whose inner chords are tangent to the 20.5 mm bore radius, so a 39 mm chip
  passes with 1 mm radial clearance. The teacups are open vessels with a closed
  base: their walls are sectorised (the interior stays hollow), their base is a
  solid disc, and the display mesh is closed at r = 0 the same way.
* Every object's collider bottom sits on a declared support within 1 mm: the
  felt at z = 0.765 m or the xiangqi board at 0.780 m. The chip tray's wells are
  cut clean through, so a chip in a well stands on the felt.
* The FR3 stands on its own pedestal at the origin (base z = 0.75 m); the table
  starts at x = 0.27 m. Every object centre lies in x 0.30-0.80 m, y -0.10-0.45 m
  and within 0.80 m of the base.
* Anything the robot must not move is static: the table, felt, board, rim, chip
  tray, chairs, lamp, cabinet, its contents and the trim. Static props are boxes;
  the tea set is an authored, immovable fixture.

Room shell and materials
------------------------
* Textured materials bind ``base_color`` only. The normal and roughness maps are
  recorded under ``unbound_textures``: the shared material path can bind them
  (``normal_map`` / ``roughness_map``), but no render of this room with them has
  been checked yet.
* The floor collides; the walls do not (``room.collide.walls = False``). No wall
  is within 2 m of the robot base or 1.4 m of any object, so the walls have no
  physical role in a task. The shared wall colliders are correct
  (``room.box_collider_matrix``, ``tests/test_room_wall_colliders.py``); switching
  them on changes the scene and needs a new native validation.

Masses are design values for a simulation task, not manufacturer measurements;
every asset records that in the manifest.
"""
from __future__ import annotations

import hashlib
import io
import math
from pathlib import Path

import numpy as np

from bench.paths import DATA_ROOT, load_json
from scenes.rooms import DEFAULT_LIBRARY_ROOT
from bench.simulation.assets.scene_parts import (Asset, annulus_mesh, annulus_sectors, box_mesh, cone_mesh,
                                cylinder_mesh, lathe_mesh, write_library)

FAMILY = 'card_room'
ASSET_VERSION = 'v2'
ASSET_BASE = str(DEFAULT_LIBRARY_ROOT / FAMILY)
ASSET_ROOT = f'{ASSET_BASE}/{ASSET_VERSION}'
# sha256 of ASSET_ROOT/manifest.json as written by build_library(). v1 (manifest
# df1ab67146718e4af7eed7f7cf0cfa16599818a20dd75d4c9bc7deb72cd05793) is kept
# unchanged as the record of the earlier build; nothing here references it.
MANIFEST_SHA256 = 'd69cddd574241bc37fb5401c86b93259fe85cfce968b17fbbbd479e022dab820'
GENERATOR = 'scenes/rooms/card_room.py'
LICENSE_ID = 'CC0-1.0'

# The xiangqi board's albedo is generated by build_library into the library. A
# 'surfaces' plate gets metric UVs (uv_box, scale 1), so one texture tile spans
# exactly 1 m x 1 m and the board's grid is drawn at its true size in the corner
# that the board's UVs (0..W, 0..D metres from its low corner) sample.
BOARD_TEXTURE = 'textures/xiangqi_board_albedo.png'
BOARD_TEXTURE_PATH = f'{ASSET_ROOT}/{BOARD_TEXTURE}'
TEXTURE_PX_PER_M = 2048

# --- shared room textures (ambientCG, CC0 1.0) ------------------------------
MATERIAL_SOURCE_ROOT = f'{DATA_ROOT}/assets/materials/ambientcg/v1'
SOURCES = {
    'WoodFloor064': 'https://ambientcg.com/a/WoodFloor064',
    'Wood089': 'https://ambientcg.com/a/Wood089',
}

# --- the room ----------------------------------------------------------------
ROOM_SIZE = (5.0, 4.0, 2.7)
ROOM_ORIGIN = (0.0, 0.0, 0.0)
ROOM_HALF_X, ROOM_HALF_Y, ROOM_HEIGHT = ROOM_SIZE[0] / 2., ROOM_SIZE[1] / 2., ROOM_SIZE[2]
WALL_THICKNESS = 0.12

# Openings in wall-local (u, v) from the wall CENTRE: world z = ROOM_HEIGHT/2 + v.
# Right (+x) wall: u runs along -y. Left (-x) wall: u runs along +y.
DOOR_U, DOOR_V, DOOR_SIZE = -1.05, -0.30, (0.90, 2.10)        # world y 0.60-1.50, z 0.00-2.10
WINDOW_U, WINDOW_V, WINDOW_SIZE = 0.80, 0.45, (1.20, 1.00)    # world y 0.20-1.40, z 1.30-2.30
OPENINGS = (
    {'id': 'door', 'wall': 'right', 'center_m': [DOOR_U, DOOR_V], 'size_m': list(DOOR_SIZE),
     'kind': 'door'},
    {'id': 'window', 'wall': 'left', 'center_m': [WINDOW_U, WINDOW_V], 'size_m': list(WINDOW_SIZE),
     'kind': 'window'},
)
TRIM_W = 0.070          # casing face width
TRIM_D = 0.025          # casing depth, standing proud of the wall into the room
SILL_D = 0.100          # window sill depth into the room


def opening_world_rect(opening):
    """World rectangle of a left/right-wall opening.

    Returns the wall plane's x, the inward direction (+1 or -1 along x) and the
    opening's y and z ranges, using the same wall-centre (u, v) convention as
    ``room._wall_geometry``.
    """
    u, v = (float(c) for c in opening['center_m'])
    su, sv = (float(s) for s in opening['size_m'])
    z = (ROOM_ORIGIN[2] + ROOM_HEIGHT / 2. + v - sv / 2.,
         ROOM_ORIGIN[2] + ROOM_HEIGHT / 2. + v + sv / 2.)
    if opening['wall'] == 'right':
        return {'wall': 'right', 'plane_x': ROOM_ORIGIN[0] + ROOM_HALF_X, 'inward': -1.0,
                'y': (ROOM_ORIGIN[1] - (u + su / 2.), ROOM_ORIGIN[1] - (u - su / 2.)), 'z': z}
    if opening['wall'] == 'left':
        return {'wall': 'left', 'plane_x': ROOM_ORIGIN[0] - ROOM_HALF_X, 'inward': 1.0,
                'y': (ROOM_ORIGIN[1] + u - su / 2., ROOM_ORIGIN[1] + u + su / 2.), 'z': z}
    raise ValueError(f'{FAMILY} frames left/right-wall openings only, not {opening["wall"]!r}')


# --- robot stand -----------------------------------------------------------------
ROBOT_BASE_Z = 0.75
PEDESTAL_CENTER = (0.0, 0.0, 0.70)
PEDESTAL_HALF = (0.22, 0.22, 0.05)
PEDESTAL_SKIRT_M = 0.65               # the stand reaches the floor instead of hovering

# --- the table -------------------------------------------------------------------
TABLE_CENTER_XY = (0.70, 0.18)
TABLE_HALF_XY = (0.43, 0.43)          # 0.86 m square: x 0.27-1.13, y -0.25-0.61
TABLE_TOP_Z = 0.745                   # top face of the wooden slab
TABLE_SLAB_HALF_Z = 0.020
TABLE_SLAB_CENTER_Z = TABLE_TOP_Z - TABLE_SLAB_HALF_Z          # 0.725
TABLE_UNDERSIDE_Z = TABLE_TOP_Z - 2. * TABLE_SLAB_HALF_Z       # 0.705
RIM_W = 0.050
RIM_PROUD = 0.015                     # the rim stands 15 mm proud of the felt
FELT_HALF_XY = (TABLE_HALF_XY[0] - RIM_W, TABLE_HALF_XY[1] - RIM_W)   # x 0.32-1.08, y -0.20-0.56
FELT_HALF_Z = 0.010
FELT_CENTER_Z = TABLE_TOP_Z + FELT_HALF_Z                       # 0.755
FELT_TOP_Z = TABLE_TOP_Z + 2. * FELT_HALF_Z                     # 0.765
RIM_TOP_Z = FELT_TOP_Z + RIM_PROUD                              # 0.780
LEG_W = 0.070
LEG_INSET = 0.030
APRON_H = 0.100
APRON_T = 0.025

# --- the xiangqi board (a solid support plate on the felt) ---------------------
# Ranks run along +x (the robot plays from the near side), files along +y.
XQ_SPACING = 0.029
XQ_MARGIN = 0.022
BOARD_SIZE_XY = (9 * XQ_SPACING + 2 * XQ_MARGIN, 8 * XQ_SPACING + 2 * XQ_MARGIN)   # 0.305 x 0.276
BOARD_LO_XY = (0.470, -0.097)
BOARD_THICKNESS = 0.015
BOARD_TOP_Z = FELT_TOP_Z + BOARD_THICKNESS                      # 0.780


def xiangqi_point(rank, file):
    """World (x, y) of a board intersection: rank 0-9 along +x, file 0-8 along +y."""
    return (BOARD_LO_XY[0] + XQ_MARGIN + rank * XQ_SPACING,
            BOARD_LO_XY[1] + XQ_MARGIN + file * XQ_SPACING)


# --- the chip tray (static frame on the felt) ------------------------------------
TRAY_CENTER_XY = (0.445, 0.262)
TRAY_H = 0.025
TRAY_WALL_T = 0.008
TRAY_DIVIDER_T = 0.005
TRAY_WELL_PITCH = 0.046
TRAY_WELL_INNER = TRAY_WELL_PITCH - TRAY_DIVIDER_T               # 41 mm: 1 mm a side for a chip
TRAY_OUTER_HALF_X = (3 * TRAY_WELL_INNER + 2 * TRAY_DIVIDER_T) / 2. + TRAY_WALL_T    # 0.0745
TRAY_OUTER_HALF_Y = 0.030                                        # 44 mm inside: 2.5 mm a side
TRAY_WELL_OFFSETS = (-TRAY_WELL_PITCH, 0.0, TRAY_WELL_PITCH)

# --- the pieces ------------------------------------------------------------------
CHIP_RADIUS, CHIP_THICKNESS, CHIP_MASS = 0.0195, 0.0033, 0.010
STACK_COUNT = 3
TILE_SIZE = (0.020, 0.026, 0.016)                                # x, y, z; face plate on +z
TILE_MASS = 0.018
TILE_FACE_T = 0.003
DIE_SIZE, DIE_MASS = 0.018, 0.008
# A real card is ~0.3 mm, but a dynamic body thinner than the 0.5 mm PhysX contact
# offset can tunnel, so a loose card is authored at the contract's 3 mm floor.
CARD_THICKNESS_M = 0.003
CARD_SIZE = (0.064, 0.089, CARD_THICKNESS_M)
CARD_MASS = 0.0018
DECK_CARDS, DECK_PITCH_M = 20, 0.0003                            # a 6 mm deck of twenty cards
CUP_RADIUS, CUP_INNER_RADIUS, CUP_HEIGHT, CUP_FLOOR, CUP_MASS = 0.033, 0.0285, 0.076, 0.0045, 0.19
RING_OUTER, RING_INNER, RING_HEIGHT, RING_MASS = 0.029, 0.0205, 0.018, 0.030
RING_SECTORS = 24
XQ_RADIUS, XQ_THICKNESS, XQ_MASS = 0.0125, 0.011, 0.005

# Body-frame half extents of each dynamic asset's collider (the union of its
# collision parts). The test re-derives these from the manifest. ROUND assets are
# solids of revolution about their z axis, so a yaw does not change their extent.
COLLIDER_HALF_M = {
    'poker_chip': (CHIP_RADIUS, CHIP_RADIUS, CHIP_THICKNESS / 2.),
    'chip_stack': (CHIP_RADIUS, CHIP_RADIUS, STACK_COUNT * CHIP_THICKNESS / 2.),
    'mahjong_tile': (TILE_SIZE[0] / 2., TILE_SIZE[1] / 2., TILE_SIZE[2] / 2.),
    'mahjong_die': (DIE_SIZE / 2., DIE_SIZE / 2., DIE_SIZE / 2.),
    'playing_card': (CARD_SIZE[0] / 2., CARD_SIZE[1] / 2., CARD_SIZE[2] / 2.),
    'card_stack': (CARD_SIZE[0] / 2., CARD_SIZE[1] / 2., DECK_CARDS * DECK_PITCH_M / 2.),
    'teacup': (CUP_RADIUS, CUP_RADIUS, CUP_HEIGHT / 2.),
    'marker_ring': (RING_OUTER, RING_OUTER, RING_HEIGHT / 2.),
    'xiangqi_red': (XQ_RADIUS, XQ_RADIUS, XQ_THICKNESS / 2.),
    'xiangqi_black': (XQ_RADIUS, XQ_RADIUS, XQ_THICKNESS / 2.),
}
ROUND_ASSETS = frozenset({'poker_chip', 'chip_stack', 'teacup', 'marker_ring',
                          'xiangqi_red', 'xiangqi_black'})
FRICTION = {'poker_chip': 0.55, 'chip_stack': 0.55, 'mahjong_tile': 0.45, 'mahjong_die': 0.45,
            'playing_card': 0.40, 'card_stack': 0.40, 'teacup': 0.55, 'marker_ring': 0.60,
            'xiangqi_red': 0.50, 'xiangqi_black': 0.50}

# --- chairs, lamp, cabinet -------------------------------------------------------
CHAIR_SEAT_TOP = 0.46
CHAIR_GAP = 0.08                       # seat front edge to the table edge
CHAIRS = (                             # id, seat centre (x, y), yaw: the sitter faces the table
    ('chair_far', (TABLE_CENTER_XY[0] + TABLE_HALF_XY[0] + CHAIR_GAP + 0.20, TABLE_CENTER_XY[1]), 180),
    ('chair_left', (TABLE_CENTER_XY[0], TABLE_CENTER_XY[1] - TABLE_HALF_XY[1] - CHAIR_GAP - 0.20), 90),
    ('chair_right', (TABLE_CENTER_XY[0], TABLE_CENTER_XY[1] + TABLE_HALF_XY[1] + CHAIR_GAP + 0.20), -90),
)
LAMP_HALF_XY = 0.25
LAMP_BOTTOM_Z = 1.62                   # 0.86 m above the felt, clear of the arm's reach
LAMP_SHADE_H = 0.14
LAMP_PANEL_T = 0.012
LAMP_TOP_T = 0.015
CANOPY_H = 0.030
CABINET_X = (-1.55, -0.35)             # a 1.2 m tea cabinet against the front wall
CABINET_DEPTH = 0.42
CABINET_TOP_Z = 0.860
TEA_SET_XY = (-1.21, 1.79)             # the tea set fixture on the cabinet top

# --- the frozen layout -----------------------------------------------------------
# (object_id, asset_id, support_id, x_m, y_m, yaw_deg, pose). 'standing' turns a
# mahjong tile onto its 20 x 16 mm end with its face plate toward the robot (-x).
STANDING_WXYZ = (0.5, 0.5, -0.5, -0.5)
SUPPORT_TOP_Z = {'felt_top': FELT_TOP_Z, 'xiangqi_board': BOARD_TOP_Z}
LAYOUT = (
    # xiangqi, mid-game: red (the robot's side, near ranks) and black (far ranks)
    ('xq_red_general', 'xiangqi_red', 'xiangqi_board', 0.492, 0.041, 0.0, 'flat'),     # rank 0 file 4
    ('xq_red_cannon', 'xiangqi_red', 'xiangqi_board', 0.550, -0.046, 0.0, 'flat'),     # rank 2 file 1
    ('xq_red_horse', 'xiangqi_red', 'xiangqi_board', 0.550, 0.099, 0.0, 'flat'),       # rank 2 file 6
    ('xq_red_soldier', 'xiangqi_red', 'xiangqi_board', 0.579, 0.041, 0.0, 'flat'),     # rank 3 file 4
    ('xq_black_general', 'xiangqi_black', 'xiangqi_board', 0.753, 0.041, 0.0, 'flat'),  # rank 9 file 4
    ('xq_black_cannon', 'xiangqi_black', 'xiangqi_board', 0.695, 0.128, 0.0, 'flat'),   # rank 7 file 7
    ('xq_black_soldier', 'xiangqi_black', 'xiangqi_board', 0.666, -0.017, 0.0, 'flat'),  # rank 6 file 2
    ('xq_black_chariot', 'xiangqi_black', 'xiangqi_board', 0.666, 0.099, 0.0, 'flat'),   # rank 6 file 6
    # mahjong: the robot's hand standing in a row facing it, three discards, two dice
    ('tile_hand_00', 'mahjong_tile', 'felt_top', 0.365, -0.085, 0.0, 'standing'),
    ('tile_hand_01', 'mahjong_tile', 'felt_top', 0.365, -0.063, 0.0, 'standing'),
    ('tile_hand_02', 'mahjong_tile', 'felt_top', 0.365, -0.041, 0.0, 'standing'),
    ('tile_hand_03', 'mahjong_tile', 'felt_top', 0.365, -0.019, 0.0, 'standing'),
    ('tile_hand_04', 'mahjong_tile', 'felt_top', 0.365, 0.003, 0.0, 'standing'),
    ('tile_hand_05', 'mahjong_tile', 'felt_top', 0.365, 0.025, 0.0, 'standing'),
    ('tile_hand_06', 'mahjong_tile', 'felt_top', 0.365, 0.047, 0.0, 'standing'),
    ('tile_hand_07', 'mahjong_tile', 'felt_top', 0.365, 0.069, 0.0, 'standing'),
    ('tile_discard_00', 'mahjong_tile', 'felt_top', 0.425, -0.065, 8.0, 'flat'),
    ('tile_discard_01', 'mahjong_tile', 'felt_top', 0.428, -0.022, -14.0, 'flat'),
    ('tile_discard_02', 'mahjong_tile', 'felt_top', 0.424, 0.020, 21.0, 'flat'),
    ('die_00', 'mahjong_die', 'felt_top', 0.427, 0.072, 15.0, 'flat'),
    ('die_01', 'mahjong_die', 'felt_top', 0.440, 0.118, -25.0, 'flat'),
    # poker: a chip in each tray well (on the felt: the wells are cut through)
    ('chip_well_00', 'poker_chip', 'felt_top', 0.399, 0.262, 0.0, 'flat'),
    ('chip_well_01', 'poker_chip', 'felt_top', 0.445, 0.262, 0.0, 'flat'),
    ('chip_well_02', 'poker_chip', 'felt_top', 0.491, 0.262, 0.0, 'flat'),
    ('chip_stack_00', 'chip_stack', 'felt_top', 0.385, 0.345, 0.0, 'flat'),
    ('chip_stack_01', 'chip_stack', 'felt_top', 0.432, 0.360, 0.0, 'flat'),
    ('chip_loose_00', 'poker_chip', 'felt_top', 0.488, 0.333, 10.0, 'flat'),
    ('chip_loose_01', 'poker_chip', 'felt_top', 0.482, 0.420, -20.0, 'flat'),
    ('card_deck_00', 'card_stack', 'felt_top', 0.585, 0.243, 4.0, 'flat'),
    ('card_loose_00', 'playing_card', 'felt_top', 0.690, 0.262, -24.0, 'flat'),
    # tea and the dealer ring
    ('teacup_00', 'teacup', 'felt_top', 0.378, 0.425, 0.0, 'flat'),
    ('teacup_01', 'teacup', 'felt_top', 0.655, 0.372, 0.0, 'flat'),
    ('dealer_ring_00', 'marker_ring', 'felt_top', 0.565, 0.395, 0.0, 'flat'),
)


def _yaw(degrees):
    """Quaternion (w, x, y, z) for a rotation about +z."""
    half = math.radians(degrees) / 2.
    return [math.cos(half), 0., 0., math.sin(half)]


def _quat_matrix(q):
    w, x, y, z = (float(v) for v in q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def object_quaternion(yaw_deg, pose):
    """The declared orientation (w, x, y, z) of a layout entry."""
    if pose == 'flat':
        return _yaw(yaw_deg)
    if pose == 'standing':
        if yaw_deg:
            raise ValueError('a standing tile faces the robot; it takes no extra yaw')
        return list(STANDING_WXYZ)
    raise ValueError(f'unknown pose {pose!r}')


def collider_world_half(asset_id, quaternion):
    """World-axis half extents of an asset's collider at an orientation."""
    half = np.asarray(COLLIDER_HALF_M[asset_id], dtype=float)
    rotation = _quat_matrix(quaternion)
    if asset_id in ROUND_ASSETS and abs(abs(rotation[2, 2]) - 1.) < 1e-12:
        return half                                      # a yawed solid of revolution
    return np.abs(rotation) @ half


def _build_objects():
    """Every dynamic object, its collider's bottom exactly on its support."""
    objects = []
    for object_id, asset_id, support, x, y, yaw_deg, pose in LAYOUT:
        quaternion = object_quaternion(yaw_deg, pose)
        half = collider_world_half(asset_id, quaternion)
        top = SUPPORT_TOP_Z[support]
        objects.append({
            'id': object_id, 'asset_id': asset_id, 'asset_root': ASSET_ROOT,
            'position_m': [round(x, 6), round(y, 6), round(top + float(half[2]), 6)],
            'quaternion_wxyz': [round(float(v), 9) for v in quaternion],
            'friction': FRICTION[asset_id],
            # World-axis extent of the collider at the declared orientation, centred
            # on the body origin, so family.validate_support sees the true bottom.
            'rest_geometry': {'size_m': [round(2. * float(v), 6) for v in half], 'origin': 'centre'},
            'support': support,
            'rest_note': f'{pose} on {support} at z = {top:.4f} m',
        })
    return objects


# --- authored dynamic pieces ------------------------------------------------------

def _off(mesh, offset):
    mesh = mesh.copy()
    mesh.apply_translation(np.asarray(offset, dtype=float))
    return mesh


def _ring_band(inner, outer, z_lo, z_hi, segments=96):
    """A thin display annulus (an inlay or a glaze band) as a closed lathe ring."""
    return lathe_mesh([(inner, z_lo), (outer, z_lo), (outer, z_hi), (inner, z_hi)], segments=segments)


def _add_sector_collision(asset, label, outer, inner, height, center, *, count, declare_bore=False):
    """Convex annular sectors whose inner chords are tangent to ``inner``.

    The union of the pieces is a ring, never a disc, and each piece records the
    ``clearance_radius_m`` that ``write_library`` and the family validator read.
    ``declare_bore`` is set only for a genuine through hole: a vessel's wall is
    authored the same way but its base closes it, so it is not a bore.
    """
    for index, piece in enumerate(annulus_sectors(outer, inner, height, count)):
        component = {'geometry': 'convex_mesh', 'position_m': [0., 0., 0.],
                     'sector': f'{index:02d}/{count}', 'clearance_radius_m': float(inner),
                     'bore_axis_xy_m': [float(center[0]), float(center[1])]}
        if declare_bore:
            component.update(bore=True, bore_radius_m=float(inner))
        # The offset is baked into the vertices (position_m stays zero), exactly as
        # Asset.add_part does, so the loader does not translate the piece twice.
        asset.add_collision(_off(piece, center), f'{label}_{index:02d}', **component)


def _poker_chip():
    """A 39 mm clay poker chip with a cream ring inlaid in each face."""
    asset = Asset('poker_chip', dynamic=True, mass_kg=CHIP_MASS, friction=FRICTION['poker_chip'])
    half = CHIP_THICKNESS / 2.
    asset.add_display(lathe_mesh([(0., -half), (CHIP_RADIUS, -half), (CHIP_RADIUS, half), (0., half)],
                                 segments=96), 'plastic_red', 'body')
    # 0.1 mm proud of each face: display only, inside the 0.5 mm display/collider tolerance.
    asset.add_display(_ring_band(0.0118, 0.0150, half - 0.0001, half + 0.0001), 'cream', 'face_ring_top')
    asset.add_display(_ring_band(0.0118, 0.0150, -half - 0.0001, -half + 0.0001), 'cream',
                      'face_ring_bottom')
    asset.add_collision(None, 'disc', geometry='cylinder', radius_m=CHIP_RADIUS,
                        half_height_m=half, position_m=[0., 0., 0.])
    asset.data.update(
        nominal_size_m=[2 * CHIP_RADIUS, 2 * CHIP_RADIUS, CHIP_THICKNESS],
        mass_design_note='Design value: a 39 mm clay chip is about 10 g',
        collision_note='One analytic cylinder covering the body exactly; the face inlays are '
                       '0.1 mm display-only rings, never a hull over a thin plate',
        rest_note='Flat, face up, on the felt or in a chip-tray well (the wells are cut through)')
    return asset


def _chip_stack():
    """Three chips as one rigid body: one collider spans the stack."""
    height = STACK_COUNT * CHIP_THICKNESS
    asset = Asset('chip_stack', dynamic=True, mass_kg=CHIP_MASS * STACK_COUNT,
                  friction=FRICTION['chip_stack'])
    jitter = ((-0.0003, 0.0002), (0.0003, -0.0002), (0.0, 0.0003))     # at most 0.3 mm
    for index in range(STACK_COUNT):
        disc = cylinder_mesh(CHIP_RADIUS, CHIP_THICKNESS * 0.94, sections=96)
        disc.apply_translation([jitter[index][0], jitter[index][1],
                                -height / 2. + (index + 0.5) * CHIP_THICKNESS])
        asset.add_display(disc, 'plastic_red' if index % 2 else 'plastic_blue', f'chip_{index:02d}')
    asset.add_collision(None, 'stack', geometry='cylinder', radius_m=CHIP_RADIUS,
                        half_height_m=height / 2., position_m=[0., 0., 0.])
    asset.data.update(
        nominal_size_m=[2 * CHIP_RADIUS, 2 * CHIP_RADIUS, height], chip_count=STACK_COUNT,
        mass_design_note=f'{STACK_COUNT} x {CHIP_MASS} kg clay chips (design value)',
        collision_note='One convex cylinder spanning the whole stack; the chips are display '
                       'meshes offset by at most 0.3 mm, so no thin plate is hulled',
        rest_note='Flat on the felt, lowest chip face down')
    return asset


def _mahjong_tile():
    """A 20 x 26 x 16 mm mahjong tile: a dark back and a 3 mm ivory face plate."""
    asset = Asset('mahjong_tile', dynamic=True, mass_kg=TILE_MASS, friction=FRICTION['mahjong_tile'])
    width, length, height = TILE_SIZE
    body_height = height - TILE_FACE_T
    body_z = -height / 2. + body_height / 2.
    asset.add_display(_off(box_mesh((width, length, body_height)), (0., 0., body_z)),
                      'plastic_green', 'tile_back')
    asset.add_collision(None, 'body', geometry='box',
                        half_extents_m=[width / 2., length / 2., body_height / 2.],
                        position_m=[0., 0., body_z])
    face_z = height / 2. - TILE_FACE_T / 2.
    asset.add_display(_off(box_mesh((width - 0.001, length - 0.001, TILE_FACE_T)), (0., 0., face_z)),
                      'cream', 'face_plate')
    asset.add_collision(None, 'face', geometry='box',
                        half_extents_m=[(width - 0.001) / 2., (length - 0.001) / 2., TILE_FACE_T / 2.],
                        position_m=[0., 0., face_z])
    asset.data.update(
        nominal_size_m=list(TILE_SIZE),
        mass_design_note='Design value: about 18 g for a 20 x 26 x 16 mm melamine tile',
        collision_note='Two analytic boxes (back plus a 3 mm face plate, 0.5 mm inset); 3 mm is six '
                       'times the 0.5 mm PhysX contact offset, so a box cannot tunnel',
        rest_note='Face up on the felt, or standing on its 20 x 16 mm end facing the robot')
    return asset


_DIE_FACES = {          # (axis, sign) -> pip count; opposite faces sum to seven
    (2, 1): 1, (2, -1): 6, (0, 1): 2, (0, -1): 5, (1, 1): 3, (1, -1): 4}
_PIP_LAYOUT = {1: [(0, 0)], 2: [(-1, -1), (1, 1)], 3: [(-1, -1), (0, 0), (1, 1)],
               4: [(-1, -1), (-1, 1), (1, -1), (1, 1)],
               5: [(-1, -1), (-1, 1), (0, 0), (1, -1), (1, 1)],
               6: [(-1, -1), (-1, 0), (-1, 1), (1, -1), (1, 0), (1, 1)]}


def _mahjong_die():
    """An 18 mm die; the one and four pips are red, as on Chinese dice."""
    import trimesh
    asset = Asset('mahjong_die', dynamic=True, mass_kg=DIE_MASS, friction=FRICTION['mahjong_die'])
    s = DIE_SIZE
    asset.add_display(box_mesh((s, s, s)), 'cream', 'body')
    red, dark = [], []
    for (axis, sign), count in _DIE_FACES.items():
        others = [a for a in range(3) if a != axis]
        size = 0.0045 if count == 1 else 0.0030
        for a, b in _PIP_LAYOUT[count]:
            # 0.1 mm sunk into the face, 0.2 mm proud of it: no coplanar faces.
            extent = [0.0003, 0.0003, 0.0003]
            extent[others[0]] = extent[others[1]] = size
            center = [0., 0., 0.]
            center[axis] = sign * (s / 2. + 0.00005)
            center[others[0]], center[others[1]] = a * 0.26 * s, b * 0.26 * s
            (red if count in (1, 4) else dark).append(_off(box_mesh(extent), center))
    asset.add_display(trimesh.util.concatenate(red), 'plastic_red', 'pips_red')
    asset.add_display(trimesh.util.concatenate(dark), 'rubber', 'pips_dark')
    asset.add_collision(None, 'cube', geometry='box', half_extents_m=[s / 2.] * 3,
                        position_m=[0., 0., 0.])
    asset.data.update(
        nominal_size_m=[s, s, s],
        mass_design_note='Design value: 8 g for an 18 mm acrylic die',
        collision_note='One analytic box; the pips are display-only and stand 0.2 mm proud',
        rest_note='Flat on the felt')
    return asset


def _playing_card():
    """One 64 x 89 mm playing card with an analytic box collider."""
    asset = Asset('playing_card', dynamic=True, mass_kg=CARD_MASS, friction=FRICTION['playing_card'])
    width, length, thickness = CARD_SIZE
    asset.add_display(box_mesh((width, length, thickness)), 'paper', 'card')
    asset.add_collision(None, 'card', geometry='box',
                        half_extents_m=[width / 2., length / 2., thickness / 2.],
                        position_m=[0., 0., 0.])
    asset.data.update(
        nominal_size_m=list(CARD_SIZE),
        nominal_thickness_note='3 mm is the contract minimum for a thin dynamic part, chosen so a '
                               '0.5 mm PhysX contact offset cannot tunnel; a real card is ~0.3 mm',
        mass_design_note='Design value: 1.8 g of 300 g/m^2 card stock',
        collision_note='The collider is an analytic box, never a hull of a thin plate',
        rest_note='Flat on the felt, face down')
    return asset


def _card_stack():
    """A squared deck of twenty cards with one box collider over the deck."""
    thickness = DECK_CARDS * DECK_PITCH_M
    asset = Asset('card_stack', dynamic=True, mass_kg=CARD_MASS * DECK_CARDS,
                  friction=FRICTION['card_stack'])
    width, length = CARD_SIZE[0], CARD_SIZE[1]
    for index in range(DECK_CARDS):
        card = box_mesh((width, length, DECK_PITCH_M * 0.94))
        card.apply_translation([0.0002 * (index % 3) - 0.0002, -0.0002 * ((index + 1) % 3) + 0.0002,
                                -thickness / 2. + (index + 0.5) * DECK_PITCH_M])
        asset.add_display(card, 'white' if index % 2 else 'paper', f'card_{index:02d}')
    asset.add_collision(None, 'deck', geometry='box',
                        half_extents_m=[width / 2., length / 2., thickness / 2.],
                        position_m=[0., 0., 0.])
    asset.data.update(
        nominal_size_m=[width, length, thickness], card_count=DECK_CARDS, card_pitch_m=DECK_PITCH_M,
        mass_design_note=f'Design value: {DECK_CARDS} x {CARD_MASS} kg of card stock',
        collision_note=f'One {thickness * 1000:.1f} mm box spans the deck; the {DECK_PITCH_M * 1000:.1f} mm '
                       'cards are display meshes offset by at most 0.2 mm, so no thin plate is hulled',
        rest_note='Flat on the felt')
    return asset


def _vessel_profile(outer, inner, height, floor, bevel=0.0004):
    """A closed lathe profile of an open-mouthed vessel with a solid base.

    It runs from the axis across the underside, up the outer wall, over the lip,
    down the inner wall and back to the axis across the inner floor, so the
    revolved mesh is closed at r = 0 top and bottom: the vessel has a floor.
    """
    return [(0., -height / 2.), (outer - bevel, -height / 2.), (outer, -height / 2. + bevel),
            (outer, height / 2. - bevel), (outer - bevel, height / 2.), (inner + bevel, height / 2.),
            (inner, height / 2. - bevel), (inner, -height / 2. + floor), (0., -height / 2. + floor)]


def _add_vessel_collision(asset, label, outer, inner, height, floor, center=(0., 0., 0.), count=24):
    """Sectorised wall from the inner floor to the lip, plus a solid base disc."""
    cx, cy, cz = (float(v) for v in center)
    wall_h = height - floor
    _add_sector_collision(asset, f'{label}_wall', outer, inner, wall_h,
                          (cx, cy, cz - height / 2. + floor + wall_h / 2.), count=count)
    asset.add_collision(None, f'{label}_base', geometry='cylinder', radius_m=float(outer),
                        half_height_m=floor / 2., position_m=[cx, cy, cz - height / 2. + floor / 2.])


def _teacup():
    """A 66 mm handleless porcelain teacup with a blue band below the lip."""
    asset = Asset('teacup', dynamic=True, mass_kg=CUP_MASS, friction=FRICTION['teacup'])
    outer, inner, height, floor = CUP_RADIUS, CUP_INNER_RADIUS, CUP_HEIGHT, CUP_FLOOR
    asset.add_display(lathe_mesh(_vessel_profile(outer, inner, height, floor), segments=96),
                      'porcelain', 'vessel')
    # A cobalt band 0.1 mm proud of the outer wall: display only.
    asset.add_display(_ring_band(outer - 0.0003, outer + 0.0001, height / 2. - 0.014,
                                 height / 2. - 0.008), [0.16, 0.27, 0.56], 'blue_band',
                      roughness=0.22, metallic=0.0)
    _add_vessel_collision(asset, 'cup', outer, inner, height, floor)
    asset.data.update(
        nominal_size_m=[2 * outer, 2 * outer, height],
        outer_radius_m=outer, inner_radius_m=inner, floor_thickness_m=floor,
        mass_design_note='Design value: 0.19 kg for a 66 mm porcelain teacup (2.4 g/cm^3 x 78 cm^3)',
        collision_note='24 convex annular sectors for the wall (outer chords inscribed in the 33 mm '
                       'outer radius, inner chords tangent to the 28.5 mm inner radius) plus one '
                       'analytic 33 mm disc for the 4.5 mm base; the display uses the same radii '
                       'and the same floor, so the two agree within 0.3 mm',
        openness_note='The mouth is open and the wall is sectorised, so no hull fills the '
                      'interior. It is not declared a bore: the base closes the vessel',
        rest_note='Upright on the felt, mouth up')
    return asset


def _marker_ring():
    """A 58 mm resin dealer ring with a 41 mm through bore."""
    asset = Asset('marker_ring', dynamic=True, mass_kg=RING_MASS, friction=FRICTION['marker_ring'])
    asset.add_display(annulus_mesh(RING_OUTER, RING_INNER, RING_HEIGHT, segments=96),
                      [0.72, 0.46, 0.13], 'ring', roughness=0.35, metallic=0.0)
    _add_sector_collision(asset, 'ring', RING_OUTER, RING_INNER, RING_HEIGHT, (0., 0., 0.),
                          count=RING_SECTORS, declare_bore=True)
    asset.data.update(
        nominal_size_m=[2 * RING_OUTER, 2 * RING_OUTER, RING_HEIGHT], bore_radius_m=RING_INNER,
        mass_design_note='Design value: 0.03 kg for a 58 mm cast-resin ring (1.3 g/cm^3)',
        collision_note=f'{RING_SECTORS} convex annular sectors whose inner chords are tangent to '
                       'the bore, so the union is a ring and not a disc',
        bore_note=f'A {2 * RING_INNER * 1000:.0f} mm through bore: a 39 mm poker chip passes with '
                  f'{(RING_INNER - CHIP_RADIUS) * 1000:.1f} mm radial clearance',
        rest_note='Flat on the felt, bore vertical')
    return asset


def _xiangqi_piece(side):
    """A 25 mm boxwood xiangqi piece with an engraved ring and glyph in red or black."""
    asset_id = f'xiangqi_{side}'
    ink = [0.62, 0.08, 0.06] if side == 'red' else [0.06, 0.06, 0.07]
    asset = Asset(asset_id, dynamic=True, mass_kg=XQ_MASS, friction=FRICTION[asset_id])
    radius, half, chamfer = XQ_RADIUS, XQ_THICKNESS / 2., 0.0006
    asset.add_display(lathe_mesh([(0., -half), (radius - chamfer, -half), (radius, -half + chamfer),
                                  (radius, half - chamfer), (radius - chamfer, half), (0., half)],
                                 segments=96), [0.80, 0.64, 0.42], 'body', roughness=0.55, metallic=0.0)
    z_lo, z_hi = half - 0.0001, half + 0.00015          # 0.15 mm proud: display only
    asset.add_display(_ring_band(0.0088, 0.0102, z_lo, z_hi), ink, 'ring_inlay',
                      roughness=0.5, metallic=0.0)
    # A stylised character: three bars and a stem, inlaid like the ring.
    import trimesh
    strokes = [_off(box_mesh((0.0012, 0.0110, z_hi - z_lo)), (dx, 0., (z_lo + z_hi) / 2.))
               for dx in (-0.0040, 0.0, 0.0040)]
    strokes.append(_off(box_mesh((0.0092, 0.0012, z_hi - z_lo)), (0., 0., (z_lo + z_hi) / 2.)))
    asset.add_display(trimesh.util.concatenate(strokes), ink, 'glyph_inlay', roughness=0.5, metallic=0.0)
    asset.add_collision(None, 'disc', geometry='cylinder', radius_m=radius, half_height_m=half,
                        position_m=[0., 0., 0.])
    asset.data.update(
        nominal_size_m=[2 * radius, 2 * radius, XQ_THICKNESS], side=side,
        mass_design_note='Design value: 5 g for a 25 x 11 mm boxwood piece',
        collision_note='One analytic cylinder; the 0.6 mm edge chamfer and the 0.15 mm inlays are '
                       'display-only and stay within 0.5 mm of it',
        rest_note='Flat on a xiangqi board intersection')
    return asset


# --- authored fixture -------------------------------------------------------------

def _tea_set():
    """A tea tray with a side-handled clay teapot and three cups (immovable).

    A fixture's local origin is the bottom centre of its base (the tray), which is
    the convention ``scene.py`` uses for immovable assets. Every collider matches
    its display: the pot body, knob, spout and side handle are convex solids, and
    the cups are open vessels authored like the teacup (sectorised wall, solid base).
    """
    import trimesh
    asset = Asset('tea_set', dynamic=False, mass_kg=0.0, friction=0.60)
    tray = (0.360, 0.240, 0.025)
    asset.add_display(_off(box_mesh(tray), (0., 0., tray[2] / 2.)), 'dark_wood', 'tray')
    asset.add_collision(None, 'tray', geometry='box', half_extents_m=[v / 2. for v in tray],
                        position_m=[0., 0., tray[2] / 2.])
    z0 = tray[2]
    pot_xy = (-0.080, 0.0)
    body = lathe_mesh([(0., 0.), (0.040, 0.), (0.051, 0.012), (0.056, 0.035), (0.053, 0.062),
                       (0.040, 0.083), (0.030, 0.090), (0.016, 0.096), (0., 0.098)], segments=96)
    body.apply_translation([pot_xy[0], pot_xy[1], z0])
    clay = [0.40, 0.22, 0.15]
    asset.add_display(body, clay, 'pot_body', roughness=0.6, metallic=0.0)
    asset.add_collision(body.convex_hull, 'pot_body', geometry='convex_mesh', position_m=[0., 0., 0.])
    knob = _off(cylinder_mesh(0.008, 0.012, sections=48), (pot_xy[0], pot_xy[1], z0 + 0.098 + 0.006))
    asset.add_display(knob, clay, 'pot_knob', roughness=0.6, metallic=0.0)
    asset.add_collision(knob.convex_hull, 'pot_knob', geometry='convex_mesh', position_m=[0., 0., 0.])
    # Spout: a closed cone rising at 40 degrees toward +x, its base inside the body.
    spout = cone_mesh(0.010, 0.070, tip_radius=0.004, sections=48)
    spout.apply_transform(trimesh.transformations.rotation_matrix(math.radians(50.), [0., 1., 0.]))
    axis = np.array([math.sin(math.radians(50.)), 0., math.cos(math.radians(50.))])
    spout.apply_translation(np.array([pot_xy[0] + 0.040, pot_xy[1], z0 + 0.035]) + axis * 0.035)
    asset.add_display(spout, clay, 'pot_spout', roughness=0.6, metallic=0.0)
    asset.add_collision(spout.convex_hull, 'pot_spout', geometry='convex_mesh', position_m=[0., 0., 0.])
    # Side handle (a Chinese side-handled pot): a straight rod at 90 degrees to the spout.
    handle = cylinder_mesh(0.008, 0.075, sections=48)
    handle.apply_transform(trimesh.transformations.rotation_matrix(math.radians(75.), [1., 0., 0.]))
    direction = np.array([0., -math.sin(math.radians(75.)), math.cos(math.radians(75.))])
    handle.apply_translation(np.array([pot_xy[0], pot_xy[1] - 0.045, z0 + 0.045]) + direction * 0.0375)
    asset.add_display(handle, [0.30, 0.16, 0.10], 'pot_handle', roughness=0.6, metallic=0.0)
    asset.add_collision(handle.convex_hull, 'pot_handle', geometry='convex_mesh',
                        position_m=[0., 0., 0.])
    cup = (0.025, 0.0215, 0.032, 0.004)
    for index, y in enumerate((-0.065, 0.0, 0.065)):
        center = (0.085, y, z0 + cup[2] / 2.)
        mesh = lathe_mesh(_vessel_profile(*cup), segments=64)
        mesh.apply_translation(center)
        asset.add_display(mesh, 'porcelain', f'cup_{index:02d}')
        _add_vessel_collision(asset, f'cup_{index:02d}', *cup, center=center, count=12)
    asset.data.update(
        nominal_size_m=[tray[0], tray[1], z0 + 0.110],
        static_note='Fixture: solid but immovable, out of the robot\'s reach on the tea cabinet',
        origin_note='Local origin at the bottom centre of the tray, as scene.py expects')
    return asset


ASSET_BUILDERS = (
    ('poker_chip', _poker_chip),
    ('chip_stack', _chip_stack),
    ('mahjong_tile', _mahjong_tile),
    ('mahjong_die', _mahjong_die),
    ('playing_card', _playing_card),
    ('card_stack', _card_stack),
    ('teacup', _teacup),
    ('marker_ring', _marker_ring),
    ('xiangqi_red', lambda: _xiangqi_piece('red')),
    ('xiangqi_black', lambda: _xiangqi_piece('black')),
    ('tea_set', _tea_set),
)


# --- the xiangqi board texture ----------------------------------------------------

def _board_pattern(scale):
    """Board-local line segments (metres): grid, river gap, palaces, frame, marks."""
    s, m = XQ_SPACING, XQ_MARGIN
    lines = []
    for rank in range(10):
        lines.append(((m + rank * s, m), (m + rank * s, m + 8 * s), 0.0009))
    for file in range(9):
        v = m + file * s
        if file in (0, 8):
            lines.append(((m, v), (m + 9 * s, v), 0.0009))
        else:                                           # the river interrupts inner files
            lines.append(((m, v), (m + 4 * s, v), 0.0009))
            lines.append(((m + 5 * s, v), (m + 9 * s, v), 0.0009))
    for rank_lo in (0, 7):                              # the two palaces
        lines.append(((m + rank_lo * s, m + 3 * s), (m + (rank_lo + 2) * s, m + 5 * s), 0.0009))
        lines.append(((m + rank_lo * s, m + 5 * s), (m + (rank_lo + 2) * s, m + 3 * s), 0.0009))
    pad = 0.004                                         # a heavier frame round the grid
    corners = [(m - pad, m - pad), (m + 9 * s + pad, m - pad), (m + 9 * s + pad, m + 8 * s + pad),
               (m - pad, m + 8 * s + pad)]
    for index in range(4):
        lines.append((corners[index], corners[(index + 1) % 4], 0.0016))
    marks = [(2, 1), (2, 7), (7, 1), (7, 7)] + [(rank, file) for rank in (3, 6) for file in (0, 2, 4, 6, 8)]
    gap, arm = 0.0022, 0.0040
    for rank, file in marks:
        cu, cv = m + rank * s, m + file * s
        for du in (-1, 1):
            for dv in (-1, 1):
                if (file == 0 and dv < 0) or (file == 8 and dv > 0):
                    continue
                corner = (cu + du * gap, cv + dv * gap)
                lines.append((corner, (corner[0] + du * arm, corner[1]), 0.0007))
                lines.append((corner, (corner[0], corner[1] + dv * arm), 0.0007))
    return [((a[0] * scale, a[1] * scale), (b[0] * scale, b[1] * scale), w * scale) for a, b, w in lines]


def board_texture_png():
    """Deterministic PNG bytes of the board albedo (procedural grain plus grid)."""
    from PIL import Image, ImageDraw
    supersample = 4
    px = TEXTURE_PX_PER_M
    width_px = int(round(BOARD_SIZE_XY[0] * px))
    depth_px = int(round(BOARD_SIZE_XY[1] * px))
    rng = np.random.default_rng(20261016)
    base = np.array([201., 160., 106.])
    # Board-local grain: soft bands along the board's long (rank) axis.
    v = (np.arange(depth_px) + 0.5)[:, None] / px
    u = (np.arange(width_px) + 0.5)[None, :] / px
    phase = rng.uniform(0., 2. * np.pi, size=3)
    grain = (0.050 * np.sin(2. * np.pi * 140. * v + 0.9 * np.sin(2. * np.pi * 7. * u + phase[0]))
             + 0.025 * np.sin(2. * np.pi * 37. * v + phase[1])
             + 0.012 * np.sin(2. * np.pi * 410. * v + 3. * np.sin(2. * np.pi * 2. * u + phase[2])))
    board = np.clip(base[None, None, :] * (1. + grain[..., None]), 0, 255).astype(np.uint8)
    image = Image.fromarray(board, 'RGB').resize((width_px * supersample, depth_px * supersample),
                                                 Image.NEAREST)
    draw = ImageDraw.Draw(image)
    ink = (38, 24, 16)
    for (u0, v0), (u1, v1), stroke in _board_pattern(px * supersample):
        draw.line([(u0, v0), (u1, v1)], fill=ink, width=max(1, int(round(stroke))))
    board_image = image.resize((width_px, depth_px), Image.LANCZOS)
    # Row r of board_image is v = r / px from the board's low corner ("v down").
    texture = Image.new('RGB', (px, px), tuple(int(c) for c in base))
    # Top-left copy serves a v-down texture convention; the flipped bottom-left copy
    # serves the v-up (OpenGL / UsdUVTexture) convention. The grid is symmetric
    # about its centre file, so both copies are the same board.
    texture.paste(board_image, (0, 0))
    texture.paste(board_image.transpose(Image.FLIP_TOP_BOTTOM), (0, px - depth_px))
    buffer = io.BytesIO()
    texture.save(buffer, format='PNG', optimize=False, compress_level=9)
    return buffer.getvalue()


# --- the library ------------------------------------------------------------------

def build_library(asset_root=ASSET_ROOT):
    """Write the authored OBJ library and the board texture into ``asset_root``.

    Returns the manifest record. ``write_library`` refuses a non-empty target, so
    this is a one-shot authoring step: to rebuild, write the next version and
    repoint ``ASSET_VERSION`` and ``MANIFEST_SHA256`` rather than overwriting.
    """
    texture = board_texture_png()
    texture_sha = hashlib.sha256(texture).hexdigest()
    assets = [(asset_id, builder()) for asset_id, builder in ASSET_BUILDERS]
    record = write_library(Path(asset_root), assets, generator=GENERATOR, license_id=LICENSE_ID,
                           extra={'family': FAMILY, 'units': 'metres',
                                  'source': 'Procedural geometry authored for the card_room scene family',
                                  'dimension_status': 'Simulation task design values; not manufacturer CAD',
                                  'collision_note': 'Convex collision pieces are the physical shape; '
                                                    'their union, not one global hull, is the collider',
                                  'textures': {BOARD_TEXTURE: {
                                      'sha256': texture_sha, 'bytes': len(texture),
                                      'pixels_per_metre': TEXTURE_PX_PER_M,
                                      'board_size_m': list(BOARD_SIZE_XY),
                                      'source': 'Procedural: grain and xiangqi grid drawn by '
                                                'card_room.board_texture_png (no third-party image)',
                                      'license': LICENSE_ID}}})
    path = Path(asset_root) / BOARD_TEXTURE
    path.parent.mkdir(parents=True, exist_ok=False)
    path.write_bytes(texture)
    if hashlib.sha256(path.read_bytes()).hexdigest() != texture_sha:
        raise RuntimeError(f'{path} did not round-trip')
    record['textures'] = {BOARD_TEXTURE: texture_sha}
    return record


def load_manifest(asset_root=ASSET_ROOT):
    """Read the authored manifest; a missing library is an error, not a fallback."""
    path = Path(asset_root) / 'manifest.json'
    if not path.is_file():
        raise FileNotFoundError(f'{FAMILY}: the authored asset library is missing: {path}; '
                                f'run build_library({str(asset_root)!r})')
    return load_json(path)


def verify_library(asset_root=ASSET_ROOT, expected_sha256=None):
    """Fail loudly unless the library on disk is the one this scene pins."""
    path = Path(asset_root) / 'manifest.json'
    if not path.is_file():
        raise FileNotFoundError(f'{FAMILY}: the authored asset library is missing: {path}')
    expected = expected_sha256 or MANIFEST_SHA256
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if expected is None or digest != expected:
        raise ValueError(f'{FAMILY}: {path} has sha256 {digest}, the scene pins {expected}')
    return digest


# --- room appearance ----------------------------------------------------------------

def _textured(name, material_id, color):
    """A textured material bound to a CC0 set in the shared asset library.

    Only ``base_color`` is bound. The normal and roughness maps are recorded under
    ``unbound_textures`` (no ``*_map`` key, so nothing binds them) until a render
    of this room with them has been checked. No scalar roughness is set, so the
    shared default (0.5) applies.
    """
    root = Path(MATERIAL_SOURCE_ROOT) / material_id
    return {'kind': 'textured', 'name': name, 'color': list(color), 'metallic': 0.0,
            'uv_scale_m': [1.0, 1.0], 'uv_scale_semantics': 'tiles_per_meter',
            'base_color': str(root / f'{material_id}_1K-JPG_Color.jpg'),
            'unbound_textures': {'normal_gl': str(root / f'{material_id}_1K-JPG_NormalGL.jpg'),
                                 'roughness': str(root / f'{material_id}_1K-JPG_Roughness.jpg')},
            'source': SOURCES[material_id], 'license': LICENSE_ID}


def _materials():
    """Every material the room, the furniture and the pedestal bind; none is unused."""
    return {
        'plank_floor': _textured('plank_floor', 'WoodFloor064', [0.58, 0.44, 0.32]),
        'oak_wood': _textured('oak_wood', 'Wood089', [0.62, 0.46, 0.30]),
        'xiangqi_board': {'kind': 'textured', 'name': 'xiangqi_board', 'color': [0.79, 0.63, 0.42],
                          'metallic': 0.0, 'uv_scale_m': [1.0, 1.0],
                          'uv_scale_semantics': 'tiles_per_meter', 'base_color': BOARD_TEXTURE_PATH,
                          'source': 'Procedural: generated by card_room.board_texture_png',
                          'license': LICENSE_ID},
        'wall_paint': {'kind': 'color', 'color': [0.76, 0.69, 0.58], 'roughness': 0.90},
        'ceiling_cream': {'kind': 'color', 'color': [0.88, 0.87, 0.83], 'roughness': 0.92},
        'wainscot_dark': {'kind': 'color', 'color': [0.20, 0.13, 0.09], 'roughness': 0.45},
        'dark_wood': {'kind': 'color', 'color': [0.27, 0.17, 0.10], 'roughness': 0.42},
        'felt_green': {'kind': 'color', 'color': [0.06, 0.29, 0.18], 'roughness': 0.97},
        'chair_cushion': {'kind': 'color', 'color': [0.46, 0.11, 0.10], 'roughness': 0.90},
        'lamp_fabric': {'kind': 'color', 'color': [0.86, 0.73, 0.50], 'roughness': 0.85},
        'cord_black': {'kind': 'color', 'color': [0.04, 0.04, 0.04], 'roughness': 0.60},
        'brass': {'kind': 'color', 'color': [0.70, 0.52, 0.22], 'roughness': 0.35, 'metallic': 0.9},
        'paper_cream': {'kind': 'color', 'color': [0.90, 0.87, 0.78], 'roughness': 0.85},
        'leather_brown': {'kind': 'color', 'color': [0.36, 0.20, 0.12], 'roughness': 0.55},
        'card_box_red': {'kind': 'color', 'color': [0.58, 0.10, 0.10], 'roughness': 0.50},
        # room.add_pedestal binds materials named 'top' and 'side' to the robot stand.
        'top': {'kind': 'color', 'color': [0.23, 0.24, 0.26], 'roughness': 0.45, 'metallic': 0.3},
        'side': {'kind': 'color', 'color': [0.17, 0.18, 0.20], 'roughness': 0.50, 'metallic': 0.3},
    }


# --- furniture (static boxes) --------------------------------------------------------

def _box(part_id, half, position, material):
    return {'id': part_id, 'geometry': 'box', 'half_extents_m': [round(float(v), 6) for v in half],
            'position_m': [round(float(v), 6) for v in position], 'material': material}


def _span(part_id, lo, hi, material):
    """A box given by its world corners."""
    lo, hi = np.asarray(lo, dtype=float), np.asarray(hi, dtype=float)
    return _box(part_id, (hi - lo) / 2., (hi + lo) / 2., material)


def _table_props():
    tx, ty = TABLE_CENTER_XY
    hx, hy = TABLE_HALF_XY
    x0, x1, y0, y1 = tx - hx, tx + hx, ty - hy, ty + hy
    fx0, fx1, fy0, fy1 = x0 + RIM_W, x1 - RIM_W, y0 + RIM_W, y1 - RIM_W
    rim = {'id': 'table_rim', 'material': 'dark_wood', 'friction': 0.62, 'title_zh': '桌沿',
           'parts': [_span('rim_near', (x0, y0, TABLE_TOP_Z), (fx0, y1, RIM_TOP_Z), 'dark_wood'),
                     _span('rim_far', (fx1, y0, TABLE_TOP_Z), (x1, y1, RIM_TOP_Z), 'dark_wood'),
                     _span('rim_left', (fx0, y0, TABLE_TOP_Z), (fx1, fy0, RIM_TOP_Z), 'dark_wood'),
                     _span('rim_right', (fx0, fy1, TABLE_TOP_Z), (fx1, y1, RIM_TOP_Z), 'dark_wood')]}
    lx0, lx1 = x0 + LEG_INSET, x1 - LEG_INSET                 # outer faces of the legs
    ly0, ly1 = y0 + LEG_INSET, y1 - LEG_INSET
    legs = {'id': 'table_legs', 'material': 'dark_wood', 'friction': 0.70, 'title_zh': '桌腿',
            'parts': [_span(f'leg_{index:02d}', (ax, ay, 0.), (ax + LEG_W, ay + LEG_W, TABLE_UNDERSIDE_Z),
                            'dark_wood')
                      for index, (ax, ay) in enumerate(((lx0, ly0), (lx0, ly1 - LEG_W),
                                                        (lx1 - LEG_W, ly0), (lx1 - LEG_W, ly1 - LEG_W)))]}
    z_lo = TABLE_UNDERSIDE_Z - APRON_H
    apron = {'id': 'table_apron', 'material': 'oak_wood', 'friction': 0.70, 'title_zh': '桌裙',
             'parts': [_span('apron_near', (lx0, ly0 + LEG_W, z_lo), (lx0 + APRON_T, ly1 - LEG_W, TABLE_UNDERSIDE_Z),
                             'oak_wood'),
                       _span('apron_far', (lx1 - APRON_T, ly0 + LEG_W, z_lo), (lx1, ly1 - LEG_W, TABLE_UNDERSIDE_Z),
                             'oak_wood'),
                       _span('apron_left', (lx0 + LEG_W, ly0, z_lo), (lx1 - LEG_W, ly0 + APRON_T, TABLE_UNDERSIDE_Z),
                             'oak_wood'),
                       _span('apron_right', (lx0 + LEG_W, ly1 - APRON_T, z_lo), (lx1 - LEG_W, ly1, TABLE_UNDERSIDE_Z),
                             'oak_wood')]}
    # The tray is a frame of boxes with its three wells cut clean through, so a chip
    # in a well stands on the felt and the tray declares no bore.
    cx, cy = TRAY_CENTER_XY
    z0, z1 = FELT_TOP_Z, FELT_TOP_Z + TRAY_H
    inner_x = TRAY_OUTER_HALF_X - TRAY_WALL_T
    tray = [_span('tray_wall_xneg', (cx - TRAY_OUTER_HALF_X, cy - TRAY_OUTER_HALF_Y, z0),
                  (cx - inner_x, cy + TRAY_OUTER_HALF_Y, z1), 'dark_wood'),
            _span('tray_wall_xpos', (cx + inner_x, cy - TRAY_OUTER_HALF_Y, z0),
                  (cx + TRAY_OUTER_HALF_X, cy + TRAY_OUTER_HALF_Y, z1), 'dark_wood'),
            _span('tray_wall_yneg', (cx - inner_x, cy - TRAY_OUTER_HALF_Y, z0),
                  (cx + inner_x, cy - TRAY_OUTER_HALF_Y + TRAY_WALL_T, z1), 'dark_wood'),
            _span('tray_wall_ypos', (cx - inner_x, cy + TRAY_OUTER_HALF_Y - TRAY_WALL_T, z0),
                  (cx + inner_x, cy + TRAY_OUTER_HALF_Y, z1), 'dark_wood')]
    for index in range(2):
        centre = cx + (TRAY_WELL_OFFSETS[index] + TRAY_WELL_OFFSETS[index + 1]) / 2.
        tray.append(_span(f'tray_divider_{index:02d}',
                          (centre - TRAY_DIVIDER_T / 2., cy - TRAY_OUTER_HALF_Y + TRAY_WALL_T, z0),
                          (centre + TRAY_DIVIDER_T / 2., cy + TRAY_OUTER_HALF_Y - TRAY_WALL_T, z1),
                          'dark_wood'))
    chip_tray = {'id': 'chip_tray', 'material': 'dark_wood', 'friction': 0.60, 'title_zh': '筹码盘',
                 'parts': tray,
                 'static_note': 'A frame of boxes with three wells cut clean through, so a chip in '
                                'a well rests on the felt and no bore is needed'}
    return [rim, legs, apron, chip_tray]


def _chair(prop_id, centre, yaw_deg):
    """A dining chair of boxes; local +x is the way the sitter faces."""
    local = [  # (id, centre (x', y', z), half, material)
        ('seat', (0., 0., CHAIR_SEAT_TOP - 0.02), (0.20, 0.21, 0.02), 'chair_cushion'),
        ('leg_front_a', (0.18, 0.19, (CHAIR_SEAT_TOP - 0.04) / 2.), (0.02, 0.02, (CHAIR_SEAT_TOP - 0.04) / 2.),
         'dark_wood'),
        ('leg_front_b', (0.18, -0.19, (CHAIR_SEAT_TOP - 0.04) / 2.), (0.02, 0.02, (CHAIR_SEAT_TOP - 0.04) / 2.),
         'dark_wood'),
        ('post_back_a', (-0.22, 0.19, 0.46), (0.02, 0.02, 0.46), 'dark_wood'),
        ('post_back_b', (-0.22, -0.19, 0.46), (0.02, 0.02, 0.46), 'dark_wood'),
        ('back_panel', (-0.22, 0., 0.715), (0.012, 0.17, 0.165), 'dark_wood'),
    ]
    yaw = math.radians(yaw_deg)
    c, s = round(math.cos(yaw)), round(math.sin(yaw))
    parts = []
    for part_id, (lx, ly, lz), (hx, hy, hz), material in local:
        position = (centre[0] + c * lx - s * ly, centre[1] + s * lx + c * ly, lz)
        half = (hx, hy, hz) if c else (hy, hx, hz)
        parts.append(_box(part_id, half, position, material))
    return {'id': prop_id, 'material': 'dark_wood', 'friction': 0.70, 'title_zh': '椅子', 'parts': parts}


def _lamp():
    """A square mahjong lamp hung over the table centre from a ceiling canopy."""
    cx, cy = TABLE_CENTER_XY
    h, t, b = LAMP_HALF_XY, LAMP_PANEL_T, LAMP_BOTTOM_Z
    top = b + LAMP_SHADE_H
    parts = [
        _span('panel_xneg', (cx - h, cy - h, b), (cx - h + t, cy + h, top), 'lamp_fabric'),
        _span('panel_xpos', (cx + h - t, cy - h, b), (cx + h, cy + h, top), 'lamp_fabric'),
        _span('panel_yneg', (cx - h + t, cy - h, b), (cx + h - t, cy - h + t, top), 'lamp_fabric'),
        _span('panel_ypos', (cx - h + t, cy + h - t, b), (cx + h - t, cy + h, top), 'lamp_fabric'),
        _span('lid', (cx - h, cy - h, top), (cx + h, cy + h, top + LAMP_TOP_T), 'dark_wood'),
        _span('cord', (cx - 0.006, cy - 0.006, top + LAMP_TOP_T), (cx + 0.006, cy + 0.006, ROOM_HEIGHT - CANOPY_H),
              'cord_black'),
        _span('canopy', (cx - 0.06, cy - 0.06, ROOM_HEIGHT - CANOPY_H), (cx + 0.06, cy + 0.06, ROOM_HEIGHT),
              'brass'),
    ]
    return {'id': 'pendant_lamp', 'material': 'lamp_fabric', 'friction': 0.60, 'title_zh': '麻将吊灯',
            'parts': parts}


def _cabinet_props():
    x0, x1 = CABINET_X
    xm = (x0 + x1) / 2.
    back = ROOM_HALF_Y                                  # the front wall's face, y = +2.0
    front = back - CABINET_DEPTH
    cabinet = {'id': 'tea_cabinet', 'material': 'dark_wood', 'friction': 0.70, 'title_zh': '茶水柜',
               'parts': [
                   _span('plinth', (x0 + 0.03, front + 0.04, 0.), (x1 - 0.03, back, 0.06), 'dark_wood'),
                   _span('carcass', (x0, front + 0.02, 0.06), (x1, back, CABINET_TOP_Z - 0.025), 'dark_wood'),
                   _span('door_a', (x0 + 0.01, front, 0.08), (xm - 0.002, front + 0.02, CABINET_TOP_Z - 0.045),
                         'oak_wood'),
                   _span('door_b', (xm + 0.002, front, 0.08), (x1 - 0.01, front + 0.02, CABINET_TOP_Z - 0.045),
                         'oak_wood'),
                   _span('top', (x0 - 0.015, front - 0.015, CABINET_TOP_Z - 0.025), (x1 + 0.015, back, CABINET_TOP_Z),
                         'dark_wood'),
                   _span('handle_a', (xm - 0.032, front - 0.012, 0.50), (xm - 0.020, front, 0.62), 'brass'),
                   _span('handle_b', (xm + 0.020, front - 0.012, 0.50), (xm + 0.032, front, 0.62), 'brass')]}
    on_top = {'id': 'cabinet_clutter', 'material': 'leather_brown', 'friction': 0.60,
              'title_zh': '麻将盒与扑克牌盒',
              'parts': [
                  _span('mahjong_case', (x1 - 0.42, back - 0.30, CABINET_TOP_Z), (x1 - 0.08, back - 0.09,
                                                                                  CABINET_TOP_Z + 0.11),
                        'leather_brown'),
                  _span('card_box_a', (x1 - 0.50, back - 0.28, CABINET_TOP_Z),
                        (x1 - 0.434, back - 0.188, CABINET_TOP_Z + 0.022), 'card_box_red'),
                  _span('card_box_b', (x1 - 0.50, back - 0.28, CABINET_TOP_Z + 0.022),
                        (x1 - 0.434, back - 0.188, CABINET_TOP_Z + 0.044), 'card_box_red')]}
    scroll = {'id': 'wall_scroll', 'material': 'dark_wood', 'friction': 0.60, 'title_zh': '装框书法横幅',
              # Hung flat on the front wall: the frame's back face is the wall plane.
              'parts': [_span('frame', (xm - 0.45, back - 0.020, 1.20), (xm + 0.45, back, 1.62), 'dark_wood'),
                        _span('paper', (xm - 0.40, back - 0.025, 1.24), (xm + 0.40, back - 0.020, 1.58),
                              'paper_cream')]}
    return [cabinet, on_top, scroll]


def _trim_props():
    """Door casing and window trim, derived from the opening constants."""
    door = opening_world_rect(OPENINGS[0])
    plane, inward = door['plane_x'], door['inward']
    x_in = plane + inward * TRIM_D
    xs = (min(plane, x_in), max(plane, x_in))
    (y0, y1), (z0, z1) = door['y'], door['z']
    door_frame = {'id': 'door_frame', 'material': 'dark_wood', 'friction': 0.70, 'title_zh': '门框',
                  'parts': [_span('jamb_a', (xs[0], y0 - TRIM_W, z0), (xs[1], y0, z1), 'dark_wood'),
                            _span('jamb_b', (xs[0], y1, z0), (xs[1], y1 + TRIM_W, z1), 'dark_wood'),
                            _span('head', (xs[0], y0 - TRIM_W, z1), (xs[1], y1 + TRIM_W, z1 + TRIM_W),
                                  'dark_wood')]}
    window = opening_world_rect(OPENINGS[1])
    plane, inward = window['plane_x'], window['inward']
    x_in, x_sill = plane + inward * TRIM_D, plane + inward * SILL_D
    xs = (min(plane, x_in), max(plane, x_in))
    xsill = (min(plane, x_sill), max(plane, x_sill))
    (y0, y1), (z0, z1) = window['y'], window['z']
    window_trim = {'id': 'window_trim', 'material': 'dark_wood', 'friction': 0.70, 'title_zh': '窗套与窗台',
                   'parts': [_span('sill', (xsill[0], y0 - TRIM_W - 0.03, z0 - 0.04), (xsill[1], y1 + TRIM_W + 0.03, z0),
                                   'dark_wood'),
                             _span('jamb_a', (xs[0], y0 - TRIM_W, z0), (xs[1], y0, z1), 'dark_wood'),
                             _span('jamb_b', (xs[0], y1, z0), (xs[1], y1 + TRIM_W, z1), 'dark_wood'),
                             _span('head', (xs[0], y0 - TRIM_W, z1), (xs[1], y1 + TRIM_W, z1 + TRIM_W),
                                   'dark_wood')]}
    return [door_frame, window_trim]


def _static_props():
    """Every immovable box in the room; none repeats the table slab or a support plate."""
    return (_table_props() + [_chair(*chair) for chair in CHAIRS] + [_lamp()] + _cabinet_props()
            + _trim_props())


def _support_plates():
    """The solid plates a dynamic object may stand on (built as static boxes by scene.py)."""
    board_half = (BOARD_SIZE_XY[0] / 2., BOARD_SIZE_XY[1] / 2., BOARD_THICKNESS / 2.)
    return [
        {'id': 'felt_top', 'position_m': [TABLE_CENTER_XY[0], TABLE_CENTER_XY[1], FELT_CENTER_Z],
         'half_extents_m': [FELT_HALF_XY[0], FELT_HALF_XY[1], FELT_HALF_Z],
         'material': 'felt_green', 'friction': 0.55, 'title_zh': '绿色毛毡台面'},
        {'id': 'xiangqi_board',
         'position_m': [round(BOARD_LO_XY[0] + board_half[0], 6), round(BOARD_LO_XY[1] + board_half[1], 6),
                        round(FELT_TOP_Z + board_half[2], 6)],
         'half_extents_m': [round(v, 6) for v in board_half],
         'material': 'xiangqi_board', 'friction': 0.50, 'title_zh': '象棋棋盘'},
    ]


def _fixtures():
    """Immovable authored assets; a fixture's origin is the bottom centre of its base."""
    return [{'id': 'tea_set', 'asset_id': 'tea_set', 'asset_root': ASSET_ROOT,
             'position_m': [TEA_SET_XY[0], TEA_SET_XY[1], CABINET_TOP_Z],
             'quaternion_wxyz': [1., 0., 0., 0.], 'friction': 0.60, 'title_zh': '茶具'}]


WORK_TARGET = [0.570, 0.170, FELT_TOP_Z]               # centre of the pieces the arm works on


def _cameras():
    """Third-person and table views from inside the room, plus an aimed wrist camera."""
    return [
        # 2.18 m from the table centre and 1.0 m above the felt, from the corner
        # between the far and left chairs: the table, all three chairs, the robot,
        # the tea cabinet and the lamp's lower edge are in frame (the CPU test
        # projects them).
        {'id': 'third_person', 'kind': 'world', 'role': 'third_person',
         'eye': [2.09, -1.17, 1.77],
         'target': [TABLE_CENTER_XY[0], TABLE_CENTER_XY[1], FELT_TOP_Z + 0.015],
         'focal_length_mm': 18.0, 'resolution': [960, 720], 'title_zh': '第三人称视角'},
        {'id': 'table_view', 'kind': 'world', 'role': 'table',
         'eye': [1.20, -0.62, 1.38], 'target': [WORK_TARGET[0], WORK_TARGET[1], WORK_TARGET[2] + 0.015],
         'focal_length_mm': 24.0, 'resolution': [960, 720], 'title_zh': '台面视角'},
        {'id': 'wrist', 'kind': 'link', 'role': 'wrist', 'mount_link': 'right_hand_C_MC',
         'offset_xyz_m': [0.0, 0.0, 0.06], 'offset_rpy_rad': [0.0, 0.0, 0.0],
         'aim_target_m': list(WORK_TARGET), 'standoff_m': 0.12, 'focal_length_mm': 18.0,
         'look_axis': '+z', 'clipping_m': [0.02, 20.0], 'resolution': [640, 480],
         'title_zh': '手腕相机'},
    ]


SCENE = {
    'schema_version': 'isaacsim-scene-family-v1',
    'family': FAMILY,
    'title_zh': '棋牌室',
    'description_zh': ('棋牌室场景：机器人立于一张0.86米见方的深色木质麻将桌一侧，桌面嵌绿色毛毡、四周有凸起的桌沿。'
                       '毡面上有一副木质象棋棋盘（棋盘上八枚棋子处于中局），机器人面前立着一排八张麻将牌，'
                       '旁边有三张打出的明牌和两枚骰子；筹码盘三个格内各放一枚筹码，另有两叠筹码、两枚散放筹码、'
                       '一副扑克牌和一张散牌，以及两只无柄瓷茶杯和一枚庄家标记环。桌子另外三边各摆一把椅子，'
                       '桌子上方悬挂方形麻将吊灯；前墙边的茶水柜上放着茶盘（紫砂侧把壶和三只小茶杯）、麻将盒和扑克牌盒，'
                       '柜子上方墙上挂着装框的书法横幅。房间铺木地板、浅米色墙面配深色踢脚线，右墙有门，左墙有带窗台的窗户。'),
    'room': {
        'enabled': True,
        'size_m': list(ROOM_SIZE),
        'origin_m': list(ROOM_ORIGIN),
        'wall_thickness_m': WALL_THICKNESS,
        'baseboard_m': 0.09,
        'uv_scale': {'floor': [1.0, 1.0], 'wall': [1.0, 1.0]},
        'surfaces': {'floor': 'plank_floor', 'wall': 'wall_paint',
                     'ceiling': 'ceiling_cream', 'baseboard': 'wainscot_dark'},
        # (u, v) from the wall centre: world z = 1.35 + v (see opening_world_rect).
        'openings': [dict(opening, center_m=list(opening['center_m']), size_m=list(opening['size_m']))
                     for opening in OPENINGS],
        'ceiling_lights': [
            # Inside the pendant shade, emitting down onto the table.
            {'id': 'lamp_table', 'position_m': [TABLE_CENTER_XY[0], TABLE_CENTER_XY[1],
                                                LAMP_BOTTOM_Z + LAMP_SHADE_H - 0.015],
             'extent_m': [2 * (LAMP_HALF_XY - LAMP_PANEL_T) - 0.03] * 2, 'intensity': 6000.0,
             'color': [1.0, 0.95, 0.85]},
            {'id': 'lamp_room_a', 'position_m': [-1.10, -0.90, 2.62], 'extent_m': [0.60, 0.60],
             'intensity': 4500.0, 'color': [1.0, 0.98, 0.94]},
            {'id': 'lamp_room_b', 'position_m': [-0.80, 1.10, 2.62], 'extent_m': [0.60, 0.60],
             'intensity': 4500.0, 'color': [1.0, 0.98, 0.94]},
        ],
        'dome': {'enabled': True, 'intensity': 350.0, 'color': [0.85, 0.90, 1.0],
                 'hdri_visible': False},
        # Wall colliders stay off until room._box_collider places rotated boxes
        # correctly (see the module docstring); the floor collider is correct.
        'collide': {'floor': True, 'walls': False, 'ceiling': False},
    },
    'materials': _materials(),
    'cameras': _cameras(),
    # 0.5 ms physics steps give 40 steps per 20 ms control window (bench convention).
    'physics': {'dt_s': 0.0005, 'control_dt_s': 0.02, 'gravity_m_s2': [0.0, 0.0, -9.81]},
    # The mount transform is omitted so it is inherited from the isaacsim_6_1_v2
    # SHARPA assembly; the validator rejects a hand-copied value that has drifted.
    'robot': {'base_position_m': [0.0, 0.0, ROBOT_BASE_Z], 'hand_id': 'sharpa_w01_right',
              'assembly_version': 'isaacsim_6_1_v2', 'assets_root': f'{DATA_ROOT}/assets',
              'arm_home_rad': [0.4, -0.5, 0.0, -2.3, 0.0, 1.8, 0.8],
              'arm_kp': 2000.0, 'arm_kd': 140.0, 'hand_kp': 3.0, 'hand_kd': 0.1},
    'pedestal': {'position_m': list(PEDESTAL_CENTER), 'half_extents_m': list(PEDESTAL_HALF),
                 'skirt_height_m': PEDESTAL_SKIRT_M},
    # The table slab: built once, by scene.py, as the work_table static prop.
    'table': {'position_m': [TABLE_CENTER_XY[0], TABLE_CENTER_XY[1], TABLE_SLAB_CENTER_Z],
              'half_extents_m': [TABLE_HALF_XY[0], TABLE_HALF_XY[1], TABLE_SLAB_HALF_Z],
              'friction': 0.5, 'material': 'dark_wood'},
    # The resting planes, each a solid plate: the felt and the xiangqi board on it.
    'surfaces': _support_plates(),
    'static_props': _static_props(),
    'fixtures': _fixtures(),
    'objects': _build_objects(),
    'cloth': {'enabled': False},
    'isaacsim': {'physics_dt_s': 0.0005, 'robot_self_collisions': True,
                 'filter_thumb_cmc_structure': True},
}
