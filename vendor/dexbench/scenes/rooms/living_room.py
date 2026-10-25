"""Living-room scene family: a furnished lounge with loose toys on a coffee table.

The room is a 5.0 x 4.0 x 2.7 m lounge. The FR3 stands on its own floor-standing
pedestal at the origin; in front of it (+x) is a coffee table whose top (z =
0.45 m) is the only manipulation surface. On it lie ten loose toddler toys -
seven studded building blocks, a small plush bear, a banded 60 mm ball and a
board picture book - and a shallow toy tray, the reachable receptacle for a
tidy-up task. The rest of the room is immovable furniture: a sofa with cushions
under a curtained window, a pouf on a rug, a TV console with a television, a
side table with a lamp, a floor lamp, a bookshelf, a plant, a picture and a
closed door.

Design notes:

* Wall solids. ``room.collide.walls`` is False: the room authors its wall solids
  itself as the ``wall_solids`` static prop - world-aligned boxes, one behind
  every drawn wall quad (the quads ``room._wall_quads`` cuts around the
  openings), whose inner face sits 1 mm behind the drawn wall and which extend one
  wall thickness outward. The door opening is closed by a door leaf and the
  window opening by a glass pane, so the shell is closed. (The shared wall
  colliders are also correct now, ``room.box_collider_matrix``; switching to them
  is a scene change and needs a new native validation.)
* Finishes. Every visual part records roughness and metallic; without them the
  task-asset loader falls back to metallic 0.65 (chrome toys).
* The plush rests on a flat 25 mm base disc instead of a single apex vertex.
* The pedestal reaches the floor (``skirt_height_m``), the home pose is the
  registry's right-arm default, the coffee-table top is authored once (by the
  ``table`` block; the ``coffee_table`` prop only *declares* it as the support),
  the table is 0.45 m high and every toy lies within 0.62 m of the robot axis so
  the palm can reach down to it, and the third-person camera frames the table,
  the robot, the sofa and the TV together.

Asset geometry is built by :func:`build_library` with ``bench.simulation.assets.scene_parts`` and
written as a versioned OBJ library. ``SCENE`` derives every object's rest height
from the *same* meshes, so each piece rests exactly on the table top. Masses are
real-world design values for toddler toys, not measured; the inertia is the
uniform-density bounding-box approximation the writer records.

CPU only: nothing here imports OpenUSD, PhysX or Isaac Sim.
"""
from __future__ import annotations

import math
import os

import numpy as np

from bench.paths import DATA_ROOT
from scenes.rooms import DEFAULT_LIBRARY_ROOT
from bench.simulation.assets import scene_parts as parts
from bench.simulation.isaacsim_backend.room import WALL_FACES, _wall_geometry, _wall_quads, validate_room


FAMILY = 'living_room'
SCHEMA_VERSION = 'isaacsim-scene-family-v1'
# v3; v1 and v2 are kept beside it as evidence (history in the library's VERSIONS.md).
ASSET_VERSION = 'v3'
ASSET_LIBRARY_BASE = str(DEFAULT_LIBRARY_ROOT / FAMILY)
ASSET_ROOT = f'{ASSET_LIBRARY_BASE}/{ASSET_VERSION}'

# --- room, robot and work surface -------------------------------------------

ROOM_SIZE_M = (5.0, 4.0, 2.7)
ROOM_ORIGIN_M = (0.0, 0.0, 0.0)
WALL_THICKNESS_M = 0.12
# The wall solids' inner face sits this far behind the drawn wall, so the two
# never share a plane (no z-fighting) while the robot still meets the wall
# within a millimetre of where it is drawn.
WALL_SOLID_GAP_M = 0.001

PEDESTAL_POSITION_M = (0.0, 0.0, 0.70)
PEDESTAL_HALF_M = (0.22, 0.22, 0.05)
# The skirt carries the pedestal top down to the floor (0.65 m = top underside).
PEDESTAL_SKIRT_M = PEDESTAL_POSITION_M[2] - PEDESTAL_HALF_M[2]
ROBOT_BASE_M = (0.0, 0.0, 0.75)
# The registry's right-arm home (robot_registry arm_home_rad_by_side['right'])
# and the sibling families' home. The system0-system1 bridge pins its own v12
# release scene (base z 0.77, q1 1.5) by sha256 and never loads a scene family,
# so nothing requires the q1 = 1.5 pose the previous revision had copied.
ARM_HOME_RAD = (0.4, -0.5, 0.0, -2.3, 0.0, 1.8, 0.8)

# The coffee table. Its top is the manipulation surface at z = 0.45 m, the high
# end of the usual 0.40-0.46 m range: the FR3 base is 0.30 m above it and the
# palm can still reach down to a toy 0.62 m from the robot axis. Near edge
# x = 0.265 m clears the pedestal (rule 7) by 45 mm.
COFFEE_TABLE_CENTER_M = (0.545, 0.14, 0.43)
COFFEE_TABLE_HALF_M = (0.28, 0.30, 0.02)
COFFEE_TABLE_TOP_Z = COFFEE_TABLE_CENTER_M[2] + COFFEE_TABLE_HALF_M[2]
SUPPORT_ID = 'coffee_table'

# Reach guidance from the frozen contract (rule 6), plus this family's tighter
# radius: every toy centre within 0.62 m of the robot's vertical axis.
REACH_ENVELOPE_X = (0.30, 0.80)
REACH_ENVELOPE_Y = (-0.10, 0.45)
REACH_RADIUS_M = 0.80
POWER_GRASP_RADIUS_M = 0.62

# --- authored toy geometry --------------------------------------------------

STUD_HEIGHT_M = 0.004
STUD_RADIUS_M = 0.0075
STUD_SECTIONS = 20
PLASTIC_FRICTION = 0.45

BLOCK_SPECS = (
    {'asset_id': 'block_cube_40_red', 'kind': 'cube', 'unit_m': 0.040,
     'mass_kg': 0.022, 'color': 'plastic_red'},
    {'asset_id': 'block_cube_44_blue', 'kind': 'cube', 'unit_m': 0.044,
     'mass_kg': 0.026, 'color': 'plastic_blue'},
    {'asset_id': 'block_cube_48_yellow', 'kind': 'cube', 'unit_m': 0.048,
     'mass_kg': 0.032, 'color': 'plastic_yellow'},
    {'asset_id': 'block_brick_2x4_green', 'kind': 'brick', 'unit_m': 0.042,
     'mass_kg': 0.040, 'color': 'plastic_green'},
    {'asset_id': 'block_brick_2x4_red', 'kind': 'brick', 'unit_m': 0.042,
     'mass_kg': 0.040, 'color': 'plastic_red'},
    {'asset_id': 'block_brick_2x4_blue', 'kind': 'brick', 'unit_m': 0.042,
     'mass_kg': 0.040, 'color': 'plastic_blue'},
)

# Plush bear: a truncated-egg body standing on a flat disc, a head, ears, arms
# and a face. The collision piece is one convex hull of body, head and arms.
PLUSH_MASS_KG = 0.060
PLUSH_BODY_RADIUS_M = 0.040
PLUSH_BODY_HALF_HEIGHT_M = 0.040
PLUSH_BODY_TAPER = 0.15
PLUSH_BASE_RADIUS_M = 0.025          # the flat disc the bear stands on
PLUSH_FABRIC = (0.95, 0.0)            # (roughness, metallic): matte fabric
PLUSH_COLOR = [0.78, 0.60, 0.42]      # honey-brown plush
PLUSH_MUZZLE_COLOR = [0.93, 0.86, 0.74]
PLUSH_EYE_COLOR = [0.05, 0.04, 0.04]

BALL_RADIUS_M = 0.030
BALL_MASS_KG = 0.030
BALL_FINISH = (0.40, 0.0)             # glossy toy plastic

BOOK_SIZE_M = (0.200, 0.200, 0.016)
BOOK_MASS_KG = 0.400
BOOK_COVER_M = 0.0025
BOOK_COVER_COLOR = [0.93, 0.50, 0.17]  # orange board cover
BOOK_COVER_FINISH = (0.55, 0.0)

MASS_BASIS = ('Design value for a real toddler toy of this size; not a measured '
              'mass. Inertia is the uniform-density bounding-box approximation '
              'recorded by write_library.')


def _yaw_quaternion_wxyz(theta_rad):
    """Planar yaw as a wxyz quaternion (the scene schema's order)."""
    return [math.cos(theta_rad / 2.0), 0.0, 0.0, math.sin(theta_rad / 2.0)]


def _sphere_profile(radius, rings=17, lo=-math.pi / 2.0, hi=math.pi / 2.0):
    """A closed (r, z) lathe profile of a sphere segment between two latitudes."""
    points = [(radius * math.cos(lo + (hi - lo) * k / (rings - 1)),
               radius * math.sin(lo + (hi - lo) * k / (rings - 1))) for k in range(rings)]
    profile = [(max(r, 0.0), z) for r, z in points]
    # Close the segment through the axis so a band is a solid, not a strip.
    if profile[-1][0] > 1e-12:
        profile.append((0.0, profile[-1][1]))
    if profile[0][0] > 1e-12:
        profile.insert(0, (0.0, profile[0][1]))
    return profile


def _studded_block(asset_id, *, length_m, width_m, height_m, stud_centers, mass_kg, color):
    """One studded toy block; returns the accumulated :class:`parts.Asset`."""
    asset = parts.Asset(asset_id, dynamic=True, mass_kg=mass_kg, friction=PLASTIC_FRICTION)
    total_height = height_m + STUD_HEIGHT_M
    asset.add_part({'geometry': 'box', 'size_m': [length_m, width_m, height_m],
                    'collision': False, 'color': color},
                   label='body', offset=(0.0, 0.0, -STUD_HEIGHT_M / 2.0))
    for index, (x_m, y_m) in enumerate(stud_centers):
        asset.add_part({'geometry': 'cylinder', 'radius_m': STUD_RADIUS_M,
                        'height_m': STUD_HEIGHT_M, 'sections': STUD_SECTIONS,
                        'collision': False, 'color': color},
                       label=f'stud_{index:02d}',
                       offset=(x_m, y_m, height_m / 2.0))
    # One box collider over the full extent: convex, and the block rests on a
    # flat bottom face (the studs are decorative, they do not interlock).
    asset.add_box_collision([length_m / 2.0, width_m / 2.0, total_height / 2.0],
                            label='collider')
    asset.data.update(
        geometry_class='studded_plastic_block',
        visual_note='Studs are display-only; the collider is one box over the full extent.',
        finish_note='Glossy dielectric toy plastic (parts.FINISHES).',
        mass_kg_design=mass_kg, mass_basis=MASS_BASIS)
    return asset


def _block_asset(spec):
    unit = float(spec['unit_m'])
    if spec['kind'] == 'cube':
        length = width = unit
        studs = [(sx * unit / 4.0, sy * unit / 4.0) for sx in (-1, 1) for sy in (-1, 1)]
    else:                                     # 2 studs by 4 studs, 21 mm pitch
        length = 2.0 * unit
        width = unit
        pitch = unit / 2.0
        studs = [(ix * pitch, iy * pitch)
                 for ix in (-1.5, -0.5, 0.5, 1.5) for iy in (-0.5, 0.5)]
    return _studded_block(spec['asset_id'], length_m=length, width_m=width,
                          height_m=unit, stud_centers=studs,
                          mass_kg=float(spec['mass_kg']), color=spec['color'])


def _plush_body_profile():
    """Truncated egg: a flat base disc of PLUSH_BASE_RADIUS_M, then the egg curve."""
    radius, half, taper = PLUSH_BODY_RADIUS_M, PLUSH_BODY_HALF_HEIGHT_M, PLUSH_BODY_TAPER

    def r_of(t):
        return radius * math.sqrt(max(0.0, 1.0 - t * t)) * (1.0 - taper * t)

    # Solve r(t_base) = base radius on the lower half (r rises monotonically there).
    lo, hi = -1.0, -0.3
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if r_of(mid) < PLUSH_BASE_RADIUS_M:
            lo = mid
        else:
            hi = mid
    t_base = 0.5 * (lo + hi)
    profile = [(0.0, t_base * half), (PLUSH_BASE_RADIUS_M, t_base * half)]
    steps = 16
    for k in range(1, steps + 1):
        t = t_base + (1.0 - t_base) * k / steps
        profile.append((max(r_of(t), 0.0), t * half))
    return profile


def _plush_layout():
    """``(label, part, offset, color, finish, in_hull)`` for every bear piece,
    in the bear's own frame (base disc at z = body bottom)."""
    body = {'geometry': 'lathe', 'profile': _plush_body_profile(), 'segments': 48}
    body_top = PLUSH_BODY_HALF_HEIGHT_M
    head_radius = 0.028
    head_z = body_top + head_radius - 0.012          # the head sits into the shoulders
    sphere = lambda r, rings=13: {'geometry': 'lathe', 'profile': _sphere_profile(r, rings),
                                  'segments': 32}
    fabric, eyes = PLUSH_FABRIC, (0.25, 0.0)
    return [
        ('body', body, (0.0, 0.0, 0.0), PLUSH_COLOR, fabric, True),
        ('head', sphere(head_radius, 17), (0.0, 0.0, head_z), PLUSH_COLOR, fabric, True),
        ('ear_left', sphere(0.011), (0.0, 0.020, head_z + 0.022), PLUSH_COLOR, fabric, True),
        ('ear_right', sphere(0.011), (0.0, -0.020, head_z + 0.022), PLUSH_COLOR, fabric, True),
        ('arm_left', sphere(0.013), (0.004, 0.034, 0.006), PLUSH_COLOR, fabric, True),
        ('arm_right', sphere(0.013), (0.004, -0.034, 0.006), PLUSH_COLOR, fabric, True),
        # Face toward +x; tiny display-only pieces, inside the hull's reach.
        ('muzzle', sphere(0.011), (0.022, 0.0, head_z - 0.006), PLUSH_MUZZLE_COLOR, fabric, False),
        ('nose', sphere(0.004, 9), (0.032, 0.0, head_z - 0.003), PLUSH_EYE_COLOR, eyes, False),
        ('eye_left', sphere(0.0035, 9), (0.022, 0.010, head_z + 0.008), PLUSH_EYE_COLOR, eyes, False),
        ('eye_right', sphere(0.0035, 9), (0.022, -0.010, head_z + 0.008), PLUSH_EYE_COLOR, eyes, False),
    ]


def _plush_asset():
    """A small plush bear that stands on a flat base disc.

    The previous body of revolution started its profile at radius 0, so it stood
    on one apex vertex and any touch toppled it. The body is now truncated at a
    25 mm disc. Every piece is shifted so the bounding box is centred on the
    asset origin (the ``centre`` rest-geometry convention), and the declared
    centre of mass is the hull's own centroid.
    """
    import trimesh

    layout = _plush_layout()
    meshes = []
    for index, (label, part, offset, *_rest) in enumerate(layout):
        _, display = parts.part_mesh(dict(part, collision=False), name='plush_toy', index=index)
        mesh = display.copy()
        mesh.apply_translation(offset)
        meshes.append(mesh)
    bounds = np.asarray(trimesh.util.concatenate(meshes).bounds, dtype=float)
    shift = -0.5 * (bounds[0] + bounds[1])

    asset = parts.Asset('plush_toy', dynamic=True, mass_kg=PLUSH_MASS_KG, friction=0.70)
    hull_points = []
    for (label, part, offset, color, finish, in_hull), mesh in zip(layout, meshes):
        placed = np.asarray(offset, dtype=float) + shift
        asset.add_part(dict(part, collision=False, roughness=finish[0], metallic=finish[1]),
                       label=label, offset=tuple(placed.tolist()), color=color)
        if in_hull:
            hull_points.append(np.asarray(mesh.vertices, dtype=float) + shift)
    points = _plush_hull_points(np.concatenate(hull_points))
    asset.add_part({'geometry': 'convex', 'points_m': points.tolist(), 'visual': False},
                   label='hull')
    hull = trimesh.convex.convex_hull(points)
    asset.center_of_mass_m = [round(float(v), 6) for v in hull.center_mass]
    base_z = float(points[:, 2].min())
    base = points[np.abs(points[:, 2] - base_z) < 1e-9]
    asset.data.update(
        geometry_class='plush_bear_flat_base',
        base_disc_radius_m=PLUSH_BASE_RADIUS_M,
        base_support_radius_m=round(float(np.max(np.linalg.norm(
            base[:, :2] - base[:, :2].mean(axis=0), axis=1))), 6),
        hull_vertex_count=int(len(hull.vertices)),
        visual_note=('Truncated-egg body on a flat 25 mm disc, head, ears, arms and a face '
                     'toward +x; matte fabric finish.'),
        collision_note=('One convex hull over body, head, ears and arms, kept within '
                        f'{PHYSX_HULL_VERTEX_LIMIT} vertices so PhysX cooking (default '
                        'hullVertexLimit 64) simulates exactly this hull; it stands on a '
                        f'{PLUSH_HULL_BASE_POINTS}-gon inscribed in the base disc.'),
        center_of_mass_note='Centroid of the collision hull (uniform density), not measured.',
        mass_kg_design=PLUSH_MASS_KG, mass_basis=MASS_BASIS)
    return asset


# PhysX cooks a ``convexHull`` collider down to at most hullVertexLimit vertices
# (64 by default, and scene.py sets no other limit). A hull authored within the
# limit is simulated as authored; a larger one would be silently re-approximated.
PHYSX_HULL_VERTEX_LIMIT = 64
PLUSH_HULL_BASE_POINTS = 12


def _fibonacci_directions(count):
    golden = math.pi * (3.0 - math.sqrt(5.0))
    directions = []
    for k in range(count):
        z = 1.0 - 2.0 * (k + 0.5) / count
        r = math.sqrt(max(0.0, 1.0 - z * z))
        directions.append((r * math.cos(golden * k), r * math.sin(golden * k), z))
    return np.asarray(directions)


def _plush_hull_points(points):
    """Reduce the bear's display vertices to a hull PhysX will not re-cook.

    Keeps a regular 12-gon of the flat base disc (actual base vertices, so the
    bear still stands flat on its full disc) plus the support points of the
    display vertices in a spread of directions; every kept point is a display
    vertex, so the hull is inscribed in the visual shape.
    """
    import trimesh

    points = np.round(np.asarray(points, dtype=float), 9)
    base_z = float(points[:, 2].min())
    base = points[np.abs(points[:, 2] - base_z) < 1e-9]
    centre = base[:, :2].mean(axis=0)
    rim = base[np.linalg.norm(base[:, :2] - centre, axis=1) > 0.9 * PLUSH_BASE_RADIUS_M]
    angles = np.arctan2(rim[:, 1] - centre[1], rim[:, 0] - centre[0])
    keep = [rim[int(np.argmin(np.abs(np.angle(np.exp(1j * (angles - 2.0 * math.pi * k
                                                              / PLUSH_HULL_BASE_POINTS))))))]
            for k in range(PLUSH_HULL_BASE_POINTS)]
    for count in range(96, 8, -4):
        upper = [d for d in _fibonacci_directions(count) if d[2] > -0.35]
        support = [points[int(np.argmax(points @ d))] for d in upper]
        chosen = np.unique(np.round(np.asarray(keep + support), 9), axis=0)
        if len(trimesh.convex.convex_hull(chosen).vertices) <= PHYSX_HULL_VERTEX_LIMIT:
            return chosen
    raise AssertionError('could not reduce the plush hull below the PhysX vertex limit')


def _ball_asset():
    """A 60 mm hollow toy ball in three colour bands, on an analytic sphere.

    The collision body is an analytic sphere, not the convex hull of the display
    mesh: a faceted hull has flat facets, so a ball at rest creeps to one (the v1
    ball drifted 9 mm and turned 0.295 rad natively). A true sphere is isotropic.
    """
    asset = parts.Asset('ball_60', dynamic=True, mass_kg=BALL_MASS_KG, friction=0.35)
    band = math.radians(30.0)
    pieces = (('cap_low', -math.pi / 2.0, -band, 'plastic_blue'),
              ('band_mid', -band, band, 'plastic_yellow'),
              ('cap_high', band, math.pi / 2.0, 'plastic_red'))
    for label, lo, hi, color in pieces:
        asset.add_part({'geometry': 'lathe', 'profile': _sphere_profile(BALL_RADIUS_M, 13, lo, hi),
                        'segments': 64, 'collision': False, 'color': color,
                        'roughness': BALL_FINISH[0], 'metallic': BALL_FINISH[1]}, label=label)
    asset.add_sphere_collision(BALL_RADIUS_M, label='sphere')
    asset.data.update(
        geometry_class='sphere_60mm', nominal_diameter_m=2.0 * BALL_RADIUS_M,
        visual_note='Three coloured bands of a faceted sphere; glossy dielectric plastic.',
        collision_note='Analytic sphere (UsdGeom.Sphere): isotropic, so it cannot rest on a facet.',
        mass_kg_design=BALL_MASS_KG, mass_basis=MASS_BASIS)
    return asset


def _book_asset():
    """A board picture book: covers, a page block and a spine; one box collider."""
    asset = parts.Asset('picture_book', dynamic=True, mass_kg=BOOK_MASS_KG, friction=0.55)
    length, width, thickness = BOOK_SIZE_M
    cover = BOOK_COVER_M
    spine = 0.0055
    # Covers stop 0.5 mm short of the spine's outer face so no two faces share a plane.
    cover_len = length - 0.0005
    cover_x = 0.0005 / 2.0
    for label, z in (('cover_top', thickness / 2.0 - cover / 2.0),
                     ('cover_bottom', -thickness / 2.0 + cover / 2.0)):
        asset.add_part({'geometry': 'box', 'size_m': [cover_len, width, cover],
                        'collision': False, 'color': BOOK_COVER_COLOR,
                        'roughness': BOOK_COVER_FINISH[0], 'metallic': BOOK_COVER_FINISH[1]},
                       label=label, offset=(cover_x, 0.0, z))
    pages_len = length - spine - 0.003
    asset.add_part({'geometry': 'box', 'size_m': [pages_len, width - 0.006, thickness - 2 * cover],
                    'collision': False, 'color': 'paper'},
                   label='pages', offset=(-length / 2.0 + spine + pages_len / 2.0, 0.0, 0.0))
    asset.add_part({'geometry': 'box', 'size_m': [spine, width - 0.0004, thickness - 0.001],
                    'collision': False, 'color': 'plastic_blue'},
                   label='spine', offset=(-length / 2.0 + spine / 2.0, 0.0, 0.0))
    asset.add_box_collision([length / 2.0, width / 2.0, thickness / 2.0], label='collider')
    asset.data.update(
        geometry_class='hardcover_picture_book', thickness_m=thickness,
        visual_note='Orange board covers, cream page block, blue spine; matte dielectric.',
        collision_note='Box collision over the full extent, not a hull of a thin plate (rule 4).',
        mass_kg_design=BOOK_MASS_KG, mass_basis=MASS_BASIS)
    return asset


def _assets():
    """Build every authored asset of this family, in a stable order."""
    built = [(spec['asset_id'], _block_asset(spec)) for spec in BLOCK_SPECS]
    built.append(('plush_toy', _plush_asset()))
    built.append(('ball_60', _ball_asset()))
    built.append(('picture_book', _book_asset()))
    return built


_CACHE = {'assets': None, 'geometry': None}


def assets():
    """The authored assets, memoised (CPU geometry only)."""
    if _CACHE['assets'] is None:
        _CACHE['assets'] = _assets()
    return _CACHE['assets']


def asset_bounds(asset):
    """The combined-visual bounding box the writer records for ``asset``."""
    import trimesh
    combined = trimesh.util.concatenate([mesh for _, mesh, _ in asset.visual])
    return np.asarray(combined.bounds, dtype=float)


def asset_geometry():
    """``{asset_id: {'bounds', 'min_m', 'max_m', 'size_m', 'mass_kg'}}``.

    Derived from the same meshes :func:`build_library` writes, so the rest poses
    in ``SCENE`` cannot drift from the authored assets.
    """
    if _CACHE['geometry'] is None:
        geometry = {}
        for asset_id, asset in assets():
            bounds = asset_bounds(asset)
            geometry[asset_id] = {'bounds': bounds.tolist(),
                                  'min_m': bounds[0].tolist(),
                                  'max_m': bounds[1].tolist(),
                                  'size_m': (bounds[1] - bounds[0]).tolist(),
                                  'mass_kg': float(asset.mass_kg)}
        _CACHE['geometry'] = geometry
    return _CACHE['geometry']


def build_library(asset_root=ASSET_ROOT):
    """Write the OBJ asset library into ``asset_root``; returns the manifest record.

    ``write_library`` refuses a non-empty target, so rebuilding requires a new
    version directory; ``SCENE`` pins :data:`ASSET_VERSION`.
    """
    from pathlib import Path

    root = Path(asset_root)
    expected = asset_geometry()
    for asset_id, asset in assets():
        bounds = asset_bounds(asset)
        if not np.allclose(bounds[0], -bounds[1], atol=1e-9):
            raise AssertionError(f'{asset_id}: bounds {bounds.tolist()} are not centred on the '
                                 "origin, which the 'centre' rest geometry assumes")
        size = (bounds[1] - bounds[0]).tolist()
        if size != expected[asset_id]['size_m']:
            raise AssertionError(f'{asset_id}: geometry drifted from the declared size '
                                 f'{size} vs {expected[asset_id]["size_m"]}')
    record = parts.write_library(
        root, assets(), generator=f'scenes.rooms.{FAMILY}',
        license_id='CC0-1.0',
        extra={'family': FAMILY, 'asset_version': root.name,
               'room_size_m': list(ROOM_SIZE_M),
               'mass_basis': MASS_BASIS,
               'support_surface_top_z_m': COFFEE_TABLE_TOP_Z,
               'supersedes': {'v2': 'no visual finishes recorded (toys rendered as chrome); '
                                    'plush stood on an apex vertex',
                              'v1': 'faceted-hull ball crept 9 mm natively'},
               'note': 'Procedurally authored toddler-toy assets for the living_room family.'})
    record['asset_versions'] = {asset_id: dict(geometry, dynamic=True)
                                for asset_id, geometry in expected.items()}
    return record


def manifest_path(asset_root=ASSET_ROOT):
    from pathlib import Path
    return Path(asset_root) / 'manifest.json'


# --- objects on the coffee table --------------------------------------------

# (object id, asset id, x, y, yaw). Three loose rows of blocks with the plush in
# the middle, the book on the robot's left and the ball beyond it. Every centre
# is inside x [0.30, 0.60], y [-0.10, 0.45] and within POWER_GRASP_RADIUS_M of
# the robot axis; footprints keep >= 15 mm apart and >= 20 mm inside the edge.
OBJECT_LAYOUT = (
    ('block_cube_40_red_a', 'block_cube_40_red', 0.320, -0.090, 0.0),
    ('block_brick_2x4_red_a', 'block_brick_2x4_red', 0.430, -0.090, 0.0),
    ('block_cube_44_blue_a', 'block_cube_44_blue', 0.550, -0.090, 0.0),
    ('block_cube_48_yellow_a', 'block_cube_48_yellow', 0.330, 0.045, 0.5),
    ('plush_toy_a', 'plush_toy', 0.445, 0.050, 0.0),
    ('block_brick_2x4_blue_a', 'block_brick_2x4_blue', 0.565, 0.040, 0.0),
    ('block_brick_2x4_green_a', 'block_brick_2x4_green', 0.400, 0.145, 0.25),
    ('block_cube_40_red_b', 'block_cube_40_red', 0.545, 0.165, 0.35),
    ('picture_book_a', 'picture_book', 0.395, 0.305, -0.08),
    ('ball_60_a', 'ball_60', 0.555, 0.250, 0.0),
)

OBJECT_FRICTION = {
    'block_cube_40_red': 0.45, 'block_cube_44_blue': 0.45, 'block_cube_48_yellow': 0.45,
    'block_brick_2x4_green': 0.45, 'block_brick_2x4_red': 0.45, 'block_brick_2x4_blue': 0.45,
    'plush_toy': 0.70, 'ball_60': 0.35, 'picture_book': 0.55,
}


def _objects():
    geometry = asset_geometry()
    objects = []
    for object_id, asset_id, x_m, y_m, yaw in OBJECT_LAYOUT:
        size = geometry[asset_id]['size_m']
        low = geometry[asset_id]['min_m'][2]
        objects.append({
            'id': object_id,
            'asset_id': asset_id,
            'asset_root': ASSET_ROOT,
            # The lowest point of the piece lands exactly on the table top.
            'position_m': [x_m, y_m, COFFEE_TABLE_TOP_Z - low],
            'quaternion_wxyz': _yaw_quaternion_wxyz(yaw),
            'friction': OBJECT_FRICTION[asset_id],
            'rest_geometry': {'size_m': list(size), 'origin': 'centre'},
            'support_surface': SUPPORT_ID,
        })
    return objects


# --- materials --------------------------------------------------------------

MATERIALS = {
    # Room shell.
    'oak_floor':       {'kind': 'color', 'color': [0.55, 0.40, 0.25], 'roughness': 0.55},
    'wall_paint':      {'kind': 'color', 'color': [0.88, 0.85, 0.79], 'roughness': 0.90},
    'ceiling_paint':   {'kind': 'color', 'color': [0.95, 0.95, 0.93], 'roughness': 0.95},
    'baseboard_paint': {'kind': 'color', 'color': [0.93, 0.92, 0.89], 'roughness': 0.60},
    # The pedestal: add_pedestal binds the materials named 'top' and 'side'.
    'top':             {'kind': 'color', 'color': [0.30, 0.31, 0.33], 'roughness': 0.50},
    'side':            {'kind': 'color', 'color': [0.36, 0.37, 0.39], 'roughness': 0.55},
    # Coffee table and the toy tray on it.
    'veneer_top':      {'kind': 'color', 'color': [0.66, 0.50, 0.32], 'roughness': 0.42},
    'table_frame':     {'kind': 'color', 'color': [0.36, 0.27, 0.18], 'roughness': 0.60},
    'tray_plastic':    {'kind': 'color', 'color': [0.20, 0.56, 0.62], 'roughness': 0.45},
    # Seating.
    'fabric_sofa':     {'kind': 'color', 'color': [0.30, 0.40, 0.52], 'roughness': 0.96},
    'fabric_cushion':  {'kind': 'color', 'color': [0.38, 0.48, 0.60], 'roughness': 0.96},
    'fabric_accent':   {'kind': 'color', 'color': [0.86, 0.62, 0.22], 'roughness': 0.95},
    'pouf_fabric':     {'kind': 'color', 'color': [0.62, 0.38, 0.30], 'roughness': 0.95},
    'rug_wool':        {'kind': 'color', 'color': [0.74, 0.68, 0.60], 'roughness': 0.98},
    # TV wall.
    'console_wood':    {'kind': 'color', 'color': [0.42, 0.30, 0.20], 'roughness': 0.50},
    'console_door':    {'kind': 'color', 'color': [0.88, 0.86, 0.82], 'roughness': 0.45},
    'tv_bezel':        {'kind': 'color', 'color': [0.05, 0.05, 0.06], 'roughness': 0.35},
    'screen_glass':    {'kind': 'color', 'color': [0.04, 0.06, 0.09], 'roughness': 0.08},
    'paper_mag':       {'kind': 'color', 'color': [0.88, 0.86, 0.78], 'roughness': 0.85},
    'pot_ceramic':     {'kind': 'color', 'color': [0.82, 0.80, 0.76], 'roughness': 0.40},
    'leaf_green':      {'kind': 'color', 'color': [0.22, 0.46, 0.24], 'roughness': 0.85},
    'frame_wood':      {'kind': 'color', 'color': [0.25, 0.18, 0.12], 'roughness': 0.55},
    'canvas_art':      {'kind': 'color', 'color': [0.35, 0.55, 0.62], 'roughness': 0.80},
    # Lamps.
    'lamp_metal':      {'kind': 'color', 'color': [0.20, 0.20, 0.21], 'roughness': 0.40,
                        'metallic': 0.80},
    'lamp_shade':      {'kind': 'color', 'color': [0.95, 0.92, 0.84], 'roughness': 0.85},
    # Bookshelf.
    'shelf_wood':      {'kind': 'color', 'color': [0.72, 0.60, 0.44], 'roughness': 0.55},
    'book_red':        {'kind': 'color', 'color': [0.62, 0.17, 0.15], 'roughness': 0.70},
    'book_blue':       {'kind': 'color', 'color': [0.16, 0.29, 0.50], 'roughness': 0.70},
    'book_cream':      {'kind': 'color', 'color': [0.85, 0.80, 0.68], 'roughness': 0.75},
    'storage_box':     {'kind': 'color', 'color': [0.56, 0.63, 0.51], 'roughness': 0.80},
    # Door and window.
    'door_paint':      {'kind': 'color', 'color': [0.93, 0.92, 0.89], 'roughness': 0.50},
    'brushed_steel':   {'kind': 'color', 'color': [0.70, 0.70, 0.72], 'roughness': 0.30,
                        'metallic': 0.90},
    'window_glass':    {'kind': 'color', 'color': [0.66, 0.78, 0.88], 'roughness': 0.05},
    'window_frame':    {'kind': 'color', 'color': [0.94, 0.94, 0.93], 'roughness': 0.50},
    'curtain_linen':   {'kind': 'color', 'color': [0.86, 0.80, 0.70], 'roughness': 0.95},
}

# Materials consumed by name by shared code rather than referenced by a part.
IMPLICIT_MATERIALS = {'top': 'add_pedestal top slab', 'side': 'add_pedestal skirt'}


# --- static props -----------------------------------------------------------

def _box(part_id, center, half, material):
    return {'id': part_id, 'geometry': 'box', 'position_m': [float(v) for v in center],
            'half_extents_m': [float(v) for v in half], 'material': material}


def _box_lohi(part_id, lo, hi, material):
    lo, hi = np.asarray(lo, dtype=float), np.asarray(hi, dtype=float)
    return _box(part_id, (lo + hi) / 2.0, (hi - lo) / 2.0, material)


def wall_solid_boxes(room_spec=None):
    """World-aligned solid boxes behind every drawn wall quad.

    Uses the room's own ``_wall_geometry``/``_wall_quads``, so each box backs
    exactly the surface ``add_room`` draws (cut around the same openings). The
    inner face is WALL_SOLID_GAP_M behind the drawn plane and the box extends one
    wall thickness outward. Boxes reaching a wall end are lengthened on the
    front/back walls so the four walls close the corners.
    """
    room = validate_room(room_spec if room_spec is not None else ROOM)
    thickness = float(room['wall_thickness_m'])
    by_face = {}
    for opening in room['openings']:
        by_face.setdefault(opening['wall'], []).append(opening)
    origin = np.asarray(room['origin_m'], dtype=float)
    half_x = room['size_m'][0] / 2.0
    boxes = []
    for face in WALL_FACES:
        wall = _wall_geometry(room, face)
        for index, (lo_u, hi_u, lo_v, hi_v) in enumerate(_wall_quads(wall, by_face.get(face, []))):
            corners = np.array([wall['center'] + u * wall['u_axis'] + v * wall['v_axis']
                                for u, v in ((lo_u, lo_v), (hi_u, lo_v), (hi_u, hi_v), (lo_u, hi_v))])
            normal = np.asarray(wall['normal'], dtype=float)
            inner = corners + normal * WALL_SOLID_GAP_M
            outer = corners + normal * (WALL_SOLID_GAP_M + thickness)
            lo = np.minimum(inner.min(axis=0), outer.min(axis=0))
            hi = np.maximum(inner.max(axis=0), outer.max(axis=0))
            if face in ('front', 'back'):
                reach = half_x + WALL_SOLID_GAP_M + thickness
                if lo[0] <= origin[0] - half_x + 1e-9:
                    lo[0] = origin[0] - reach
                if hi[0] >= origin[0] + half_x - 1e-9:
                    hi[0] = origin[0] + reach
            boxes.append({'face': face, 'index': index, 'quad_uv': [lo_u, hi_u, lo_v, hi_v],
                          'lo': lo.tolist(), 'hi': hi.tolist()})
    return boxes


def _wall_solids():
    return {'id': 'wall_solids', 'material': 'wall_paint', 'friction': 0.6,
            'role': 'room_shell',
            'parts': [_box_lohi(f'{box["face"]}_{box["index"]:02d}', box['lo'], box['hi'],
                                'wall_paint') for box in wall_solid_boxes()]}


def _opening(kind):
    """The world box of the declared door or window opening (through the wall)."""
    room = validate_room(ROOM)
    opening = next(o for o in room['openings'] if o['kind'] == kind)
    wall = _wall_geometry(room, opening['wall'])
    (cu, cv), (su, sv) = opening['center_m'], opening['size_m']
    corners = np.array([wall['center'] + u * wall['u_axis'] + v * wall['v_axis']
                        for u in (cu - su / 2.0, cu + su / 2.0) for v in (cv - sv / 2.0, cv + sv / 2.0)])
    return opening, wall, corners.min(axis=0), corners.max(axis=0)


def _door():
    """A closed door leaf in the left-wall opening, with a casing and a handle."""
    _, wall, lo, hi = _opening('door')
    plane_x = float(lo[0])                        # the drawn wall plane, x = -2.5
    thickness = WALL_THICKNESS_M
    leaf_x = (plane_x - thickness / 2.0 - 0.02, plane_x - thickness / 2.0 + 0.02)
    y0, y1, top = float(lo[1]), float(hi[1]), float(hi[2])
    casing = 0.08
    front = (plane_x + 0.001, plane_x + 0.021)    # casing stands proud of the wall
    return {'id': 'door', 'material': 'door_paint', 'friction': 0.6, 'role': 'opening_closure',
            'parts': [
                _box_lohi('leaf', (leaf_x[0], y0 + 0.005, 0.005), (leaf_x[1], y1 - 0.005, top - 0.005),
                          'door_paint'),
                _box_lohi('casing_a', (front[0], y0 - casing, 0.0), (front[1], y0, top + casing),
                          'baseboard_paint'),
                _box_lohi('casing_b', (front[0], y1, 0.0), (front[1], y1 + casing, top + casing),
                          'baseboard_paint'),
                _box_lohi('casing_head', (front[0], y0, top), (front[1], y1, top + casing),
                          'baseboard_paint'),
                _box_lohi('handle', (leaf_x[1], y1 - 0.12, 0.985), (plane_x + 0.035, y1 - 0.05, 1.015),
                          'brushed_steel'),
            ]}


def _window():
    """Glass, a sash frame, an interior sill and curtains for the back-wall window."""
    _, wall, lo, hi = _opening('window')
    plane_y = float(hi[1])                        # the drawn wall plane, y = -2.0
    x0, x1, z0, z1 = float(lo[0]), float(hi[0]), float(lo[2]), float(hi[2])
    mid = plane_y - WALL_THICKNESS_M / 2.0
    glass = (mid - 0.005, mid + 0.005)
    frame = (glass[1], glass[1] + 0.040)
    bar = 0.04
    return {'id': 'window', 'material': 'window_frame', 'friction': 0.6, 'role': 'opening_closure',
            'parts': [
                _box_lohi('glass', (x0, glass[0], z0), (x1, glass[1], z1), 'window_glass'),
                _box_lohi('frame_bottom', (x0, frame[0], z0), (x1, frame[1], z0 + bar), 'window_frame'),
                _box_lohi('frame_top', (x0, frame[0], z1 - bar), (x1, frame[1], z1), 'window_frame'),
                _box_lohi('frame_left', (x0, frame[0], z0 + bar), (x0 + bar, frame[1], z1 - bar),
                          'window_frame'),
                _box_lohi('frame_right', (x1 - bar, frame[0], z0 + bar), (x1, frame[1], z1 - bar),
                          'window_frame'),
                _box_lohi('sill', (x0 - 0.08, plane_y + 0.001, z0 - 0.04), (x1 + 0.08, plane_y + 0.05, z0),
                          'window_frame'),
                _box_lohi('curtain_a', (x0 - 0.10, plane_y + 0.055, 0.95), (x0 + 0.14, plane_y + 0.095, 2.36),
                          'curtain_linen'),
                _box_lohi('curtain_b', (x1 - 0.14, plane_y + 0.055, 0.95), (x1 + 0.10, plane_y + 0.095, 2.36),
                          'curtain_linen'),
                _box_lohi('curtain_rod', (x0 - 0.20, plane_y + 0.070, 2.36), (x1 + 0.20, plane_y + 0.090, 2.38),
                          'lamp_metal'),
            ]}


def _coffee_table():
    """The coffee table's understructure, and the declaration of its top as the support.

    The top slab itself is authored once, by the scene's ``table`` block (which is
    also what makes ``family`` check rule 7). This prop's own ``position_m`` /
    ``half_extents_m`` re-state that slab for ``family.support_surfaces`` - they
    declare the support, they do not author a second collider - and its parts are
    only what is *below* the top: apron, shelf and four legs.
    """
    cx, cy, cz = COFFEE_TABLE_CENTER_M
    underside = cz - COFFEE_TABLE_HALF_M[2]
    leg_half = underside / 2.0
    leg_x, leg_y = 0.25, 0.27
    parts_list = [
        _box('apron', (cx, cy, underside - 0.03), (0.225, 0.245, 0.03), 'table_frame'),
        _box('shelf', (cx, cy, 0.115), (0.225, 0.245, 0.015), 'table_frame'),
    ]
    for index, (sx, sy) in enumerate(((1, 1), (1, -1), (-1, 1), (-1, -1))):
        parts_list.append(_box(f'leg_{index}', (cx + sx * leg_x, cy + sy * leg_y, leg_half),
                               (0.025, 0.025, leg_half), 'table_frame'))
    return {'id': SUPPORT_ID, 'material': 'table_frame', 'friction': 0.55, 'role': 'work_surface',
            'position_m': list(COFFEE_TABLE_CENTER_M), 'half_extents_m': list(COFFEE_TABLE_HALF_M),
            'support_declared_by': 'table', 'parts': parts_list}


# The receptacle: a shallow open tray standing on the far edge of the table top,
# within the palm's drop-in reach and clear of every toy footprint.
TOY_TRAY_CENTER_XY = (0.715, -0.010)
TOY_TRAY_HALF_XY = (0.085, 0.120)
TOY_TRAY_HEIGHT_M = 0.040
TOY_TRAY_WALL_M = 0.012
TOY_TRAY_FLOOR_M = 0.006


def _toy_tray():
    (cx, cy), (hx, hy) = TOY_TRAY_CENTER_XY, TOY_TRAY_HALF_XY
    z0, wall, floor = COFFEE_TABLE_TOP_Z, TOY_TRAY_WALL_M, TOY_TRAY_FLOOR_M
    z_floor, z_top = z0 + floor, z0 + TOY_TRAY_HEIGHT_M
    return {'id': 'toy_tray', 'material': 'tray_plastic', 'friction': 0.6, 'role': 'receptacle',
            'parts': [
                _box_lohi('floor', (cx - hx, cy - hy, z0), (cx + hx, cy + hy, z_floor), 'tray_plastic'),
                _box_lohi('wall_near', (cx - hx, cy - hy, z_floor), (cx - hx + wall, cy + hy, z_top),
                          'tray_plastic'),
                _box_lohi('wall_far', (cx + hx - wall, cy - hy, z_floor), (cx + hx, cy + hy, z_top),
                          'tray_plastic'),
                _box_lohi('wall_left', (cx - hx + wall, cy + hy - wall, z_floor),
                          (cx + hx - wall, cy + hy, z_top), 'tray_plastic'),
                _box_lohi('wall_right', (cx - hx + wall, cy - hy, z_floor),
                          (cx + hx - wall, cy - hy + wall, z_top), 'tray_plastic'),
            ]}


RUG_LO_XY, RUG_HI_XY, RUG_THICKNESS_M = (-1.95, -1.00), (-0.50, 1.30), 0.008


def _rug():
    return {'id': 'rug', 'material': 'rug_wool', 'friction': 0.9, 'role': 'floor_covering',
            'parts': [_box_lohi('pile', (*RUG_LO_XY, 0.0), (*RUG_HI_XY, RUG_THICKNESS_M), 'rug_wool')]}


def _sofa():
    """A three-seat sofa against the back wall, under the window, facing +y."""
    y0, y1 = -1.95, -1.07
    return {'id': 'sofa', 'material': 'fabric_sofa', 'friction': 0.9, 'role': 'furniture',
            'parts': [
                _box_lohi('base', (-1.95, y0, 0.0), (-0.45, y1, 0.36), 'fabric_sofa'),
                _box_lohi('arm_left', (-2.15, y0, 0.0), (-1.95, y1, 0.62), 'fabric_sofa'),
                _box_lohi('arm_right', (-0.45, y0, 0.0), (-0.25, y1, 0.62), 'fabric_sofa'),
                _box_lohi('back', (-1.95, y0, 0.36), (-0.45, -1.75, 0.88), 'fabric_sofa'),
                _box_lohi('seat_cushion_a', (-1.94, -1.74, 0.36), (-1.21, -1.09, 0.49), 'fabric_cushion'),
                _box_lohi('seat_cushion_b', (-1.19, -1.74, 0.36), (-0.46, -1.09, 0.49), 'fabric_cushion'),
                _box_lohi('back_cushion_a', (-1.92, -1.75, 0.49), (-1.22, -1.61, 0.83), 'fabric_cushion'),
                _box_lohi('back_cushion_b', (-1.18, -1.75, 0.49), (-0.48, -1.61, 0.83), 'fabric_cushion'),
                _box_lohi('pillow_a', (-1.90, -1.60, 0.49), (-1.64, -1.48, 0.73), 'fabric_accent'),
                _box_lohi('pillow_b', (-0.74, -1.60, 0.49), (-0.48, -1.48, 0.73), 'fabric_accent'),
            ]}


def _side_table():
    """A pedestal side table with a reading lamp, beside the sofa's right arm."""
    return {'id': 'side_table', 'material': 'table_frame', 'friction': 0.6, 'role': 'furniture',
            'parts': [
                _box_lohi('foot', (-0.16, -1.65, 0.0), (0.14, -1.35, 0.02), 'table_frame'),
                _box_lohi('column', (-0.04, -1.53, 0.02), (0.02, -1.47, 0.52), 'table_frame'),
                _box_lohi('top', (-0.21, -1.70, 0.52), (0.19, -1.30, 0.55), 'veneer_top'),
                _box_lohi('lamp_base', (-0.07, -1.56, 0.55), (0.05, -1.44, 0.58), 'lamp_metal'),
                _box_lohi('lamp_stem', (-0.02, -1.51, 0.58), (0.00, -1.49, 0.82), 'lamp_metal'),
                _box_lohi('lamp_shade', (-0.13, -1.62, 0.82), (0.11, -1.38, 1.00), 'lamp_shade'),
            ]}


def _pouf():
    """A fabric pouf standing on the rug in front of the sofa."""
    return {'id': 'pouf', 'material': 'pouf_fabric', 'friction': 0.85, 'role': 'furniture',
            'rests_on': 'rug',
            'parts': [_box_lohi('cushion', (-1.30, -0.95, RUG_THICKNESS_M), (-0.90, -0.55, 0.40),
                                'pouf_fabric')]}


def _tv_console():
    """A low media console on the front wall with a 55-inch television, facing the sofa."""
    x0, x1, y0, y1 = -2.05, -0.35, 1.52, 1.98
    return {'id': 'tv_console', 'material': 'console_wood', 'friction': 0.6, 'role': 'furniture',
            'parts': [
                _box_lohi('plinth', (x0 + 0.05, y0 + 0.04, 0.0), (x1 - 0.05, y1 - 0.04, 0.06), 'console_wood'),
                _box_lohi('carcass', (x0, y0, 0.06), (x1, y1, 0.50), 'console_wood'),
                _box_lohi('door_a', (x0 + 0.05, y0 - 0.008, 0.10), (-1.22, y0, 0.46), 'console_door'),
                _box_lohi('door_b', (-1.18, y0 - 0.008, 0.10), (x1 - 0.05, y0, 0.46), 'console_door'),
                _box_lohi('tv_stand', (-1.40, 1.66, 0.50), (-1.00, 1.84, 0.53), 'tv_bezel'),
                _box_lohi('tv_neck', (-1.24, 1.73, 0.53), (-1.16, 1.79, 0.62), 'tv_bezel'),
                _box_lohi('tv_body', (-1.81, 1.69, 0.60), (-0.59, 1.73, 1.30), 'tv_bezel'),
                _box_lohi('tv_screen', (-1.79, 1.687, 0.62), (-0.61, 1.69, 1.28), 'screen_glass'),
                _box_lohi('magazines', (-2.00, 1.60, 0.50), (-1.84, 1.84, 0.53), 'paper_mag'),
                _box_lohi('vase', (-0.56, 1.62, 0.50), (-0.42, 1.76, 0.64), 'pot_ceramic'),
                _box_lohi('vase_foliage', (-0.58, 1.60, 0.64), (-0.40, 1.78, 0.84), 'leaf_green'),
            ]}


def _wall_picture():
    """A framed landscape canvas above the television on the front wall."""
    return {'id': 'wall_picture', 'material': 'frame_wood', 'friction': 0.5, 'role': 'wall_hung',
            'parts': [
                _box_lohi('frame', (-1.65, 1.960, 1.42), (-0.75, 1.999, 1.86), 'frame_wood'),
                _box_lohi('canvas', (-1.61, 1.950, 1.46), (-0.79, 1.960, 1.82), 'canvas_art'),
            ]}


def _floor_lamp():
    """A floor lamp in the corner beside the sofa's left arm."""
    return {'id': 'floor_lamp', 'material': 'lamp_metal', 'friction': 0.6, 'role': 'furniture',
            'parts': [
                _box_lohi('base', (-2.42, -1.92, 0.0), (-2.18, -1.68, 0.024), 'lamp_metal'),
                _box_lohi('pole', (-2.315, -1.815, 0.024), (-2.285, -1.785, 1.424), 'lamp_metal'),
                _box_lohi('shade', (-2.47, -1.97, 1.424), (-2.13, -1.63, 1.624), 'lamp_shade'),
            ]}


def _bookshelf():
    """An open bookshelf against the left wall, with books and storage boxes."""
    x0, x1, y0, y1, height = -2.48, -2.16, -0.95, 0.15, 1.80
    inner_y0, inner_y1 = y0 + 0.02, y1 - 0.02
    board = []
    for index, z in enumerate((0.50, 0.95, 1.40)):
        board.append(_box_lohi(f'shelf_{index}', (x0 + 0.01, inner_y0, z), (x1, inner_y1, z + 0.02),
                               'shelf_wood'))
    return {'id': 'bookshelf', 'material': 'shelf_wood', 'friction': 0.6, 'role': 'furniture',
            'parts': [
                _box_lohi('side_a', (x0, y0, 0.0), (x1, inner_y0, height), 'shelf_wood'),
                _box_lohi('side_b', (x0, inner_y1, 0.0), (x1, y1, height), 'shelf_wood'),
                _box_lohi('back_panel', (x0, inner_y0, 0.0), (x0 + 0.01, inner_y1, height), 'shelf_wood'),
                _box_lohi('plinth', (x0 + 0.01, inner_y0, 0.0), (x1, inner_y1, 0.08), 'shelf_wood'),
                *board,
                _box_lohi('top', (x0 + 0.01, inner_y0, height - 0.02), (x1, inner_y1, height), 'shelf_wood'),
                _box_lohi('boxes', (x0 + 0.02, -0.90, 0.08), (x1 - 0.02, -0.42, 0.36), 'storage_box'),
                _box_lohi('books_low', (x0 + 0.02, -0.88, 0.52), (x1 - 0.06, -0.45, 0.78), 'book_red'),
                _box_lohi('books_mid', (x0 + 0.02, -0.30, 0.97), (x1 - 0.08, 0.08, 1.21), 'book_blue'),
                _box_lohi('books_high', (x0 + 0.02, -0.85, 1.42), (x1 - 0.10, -0.55, 1.62), 'book_cream'),
            ]}


def _plant():
    """A potted floor plant at the front wall, right of the TV console."""
    return {'id': 'plant', 'material': 'pot_ceramic', 'friction': 0.6, 'role': 'furniture',
            'parts': [
                _box_lohi('pot', (-0.21, 1.56, 0.0), (0.11, 1.88, 0.36), 'pot_ceramic'),
                _box_lohi('foliage_low', (-0.30, 1.47, 0.36), (0.20, 1.97, 0.76), 'leaf_green'),
                _box_lohi('foliage_high', (-0.23, 1.54, 0.76), (0.13, 1.90, 1.16), 'leaf_green'),
            ]}


def _static_props():
    return [_wall_solids(), _door(), _window(), _coffee_table(), _toy_tray(), _rug(), _sofa(),
            _side_table(), _pouf(), _tv_console(), _wall_picture(), _floor_lamp(), _bookshelf(),
            _plant()]


# --- room, cameras, scene ---------------------------------------------------

ROOM = {
    'enabled': True,
    'size_m': list(ROOM_SIZE_M),
    'origin_m': list(ROOM_ORIGIN_M),
    'wall_thickness_m': WALL_THICKNESS_M,
    'baseboard_m': 0.09,
    'ceiling_height_m': ROOM_SIZE_M[2],
    'surfaces': {'floor': 'oak_floor', 'wall': 'wall_paint',
                 'ceiling': 'ceiling_paint', 'baseboard': 'baseboard_paint'},
    # Wall-local (u, v) frames: u runs along the wall (left wall: +y, back wall:
    # -x), v is measured from the wall's mid-height. The door (left wall, y 0.62
    # to 1.52, floor to 2.10 m) is in view behind the robot; the window (back
    # wall, x -2.0 to -0.4, z 1.2 to 2.2) is above the sofa.
    'openings': [
        {'id': 'door', 'wall': 'left', 'center_m': [1.07, -0.30], 'size_m': [0.90, 2.10],
         'kind': 'door'},
        {'id': 'window', 'wall': 'back', 'center_m': [1.20, 0.35], 'size_m': [1.60, 1.00],
         'kind': 'window'},
    ],
    # Brighter than the first revision (two 0.7 m panels at 9000, dome 350,
    # which rendered at rgb mean 57/255): one panel over the work area, one over
    # the lounge and one by the camera side.
    'ceiling_lights': [
        {'id': 'lamp_table', 'position_m': [0.50, 0.10, 2.58], 'extent_m': [1.00, 0.80],
         'intensity': 12000.0, 'color': [1.0, 0.97, 0.92]},
        {'id': 'lamp_lounge', 'position_m': [-1.25, -0.10, 2.58], 'extent_m': [1.00, 1.00],
         'intensity': 11000.0, 'color': [1.0, 0.96, 0.90]},
        {'id': 'lamp_entry', 'position_m': [1.70, 0.10, 2.58], 'extent_m': [0.70, 0.70],
         'intensity': 8000.0, 'color': [1.0, 0.98, 0.94]},
    ],
    'dome': {'enabled': True, 'intensity': 650.0, 'color': [0.85, 0.90, 1.00],
             'hdri_visible': False},
    # Walls are NOT built by add_room: its rotated wall colliders are misplaced
    # (shared room._box_collider composes the basis transposed and after the
    # translation). The ``wall_solids`` static prop closes the room instead.
    'collide': {'floor': True, 'walls': False, 'ceiling': False},
    'title_zh': '客厅房间外壳',
    'description_zh': '木地板、乳白墙面、白色踢脚线、三盏顶灯与天光；左墙一扇木门，后墙沙发上方一扇带窗帘的玻璃窗。',
}

# Projection model used by the tests: RTX keeps the USD default horizontal
# aperture (20.955 mm) and square pixels, so the vertical field follows the
# image aspect. 14 mm on 960 x 720 gives 73.6 x 58.6 degrees.
HORIZONTAL_APERTURE_MM = 20.955

CAMERAS = [
    # Rule 9: aimed at the table-top centre from 1.99 m, 0.70 m above the top,
    # from 0.10 m inside the right wall. It looks back at the robot across the
    # table: the sofa and window on the left, the TV wall and plant on the right,
    # the door, bookshelf and floor lamp behind the robot, and the whole arm at
    # home inside the frame, so one view reads as a living room. A 14 mm lens
    # (about 24 mm full-frame equivalent) is a normal interior wide angle.
    {'id': 'third_person', 'kind': 'world', 'role': 'third_person',
     'eye': [2.40, 0.25, 1.15], 'target': [COFFEE_TABLE_CENTER_M[0], COFFEE_TABLE_CENTER_M[1],
                                          COFFEE_TABLE_TOP_Z],
     'focal_length_mm': 14.0, 'resolution': [960, 720], 'title_zh': '第三人称视角'},
    # A close view of the toys and the tray (the table nearly fills the frame).
    {'id': 'table_view', 'kind': 'world', 'role': 'table',
     'eye': [1.20, 0.62, 1.00], 'target': [0.48, 0.10, COFFEE_TABLE_TOP_Z],
     'focal_length_mm': 22.0, 'resolution': [960, 720], 'title_zh': '茶几视角'},
    # Aimed at the table top (cameras.py default) and stood off 0.12 m along the
    # view axis so the hand it rides on does not fill the frame.
    {'id': 'wrist', 'kind': 'link', 'role': 'wrist', 'mount_link': 'right_hand_C_MC',
     'offset_xyz_m': [0.0, 0.0, 0.06], 'offset_rpy_rad': [0.0, 0.0, 0.0],
     'look_axis': '+z', 'clipping_m': [0.02, 20.0],
     'focal_length_mm': 18.0, 'resolution': [640, 480], 'title_zh': '腕部视角'},
]

SCENE = {
    'schema_version': SCHEMA_VERSION,
    'family': FAMILY,
    'title_zh': '客厅玩具',
    'description_zh': (
        '客厅：机械臂站在落地底座上，面前茶几上散落着塑料积木、毛绒小熊、皮球和图画书，'
        '还有一个收纳玩具的浅托盘；四周是沙发与靠垫、地毯、坐墩、电视柜与电视、边几台灯、'
        '落地灯、书架、绿植、挂画、木门和带窗帘的窗户，均为不可移动的家具。'),
    'room': ROOM,
    'materials': MATERIALS,
    'cameras': CAMERAS,
    'robot': {
        'base_position_m': list(ROBOT_BASE_M),
        'hand_id': 'sharpa_w01_right',
        'assembly_version': 'isaacsim_6_1_v2',
        'assets_root': f'{DATA_ROOT}/assets',
        'arm_home_rad': list(ARM_HOME_RAD),
        # Joint-drive gains of the released SHARPA scenes (v12_mount_v2_r1).
        'arm_kp': 2000.0, 'arm_kd': 140.0, 'hand_kp': 3.0, 'hand_kd': 0.1,
        'self_collisions': True,
    },
    # Floor-standing: the skirt carries the 0.10 m top slab down to the floor.
    'pedestal': {'position_m': list(PEDESTAL_POSITION_M),
                 'half_extents_m': list(PEDESTAL_HALF_M),
                 'skirt_height_m': PEDESTAL_SKIRT_M},
    # The coffee-table top: the one authored slab of the work surface.
    'table': {'position_m': list(COFFEE_TABLE_CENTER_M),
              'half_extents_m': list(COFFEE_TABLE_HALF_M),
              'material': 'veneer_top', 'friction': 0.55},
    # No extra slabs: the support is declared by the ``coffee_table`` prop.
    'surfaces': [],
    'static_props': _static_props(),
    'fixtures': [],
    'objects': _objects(),
    # The bench's native PhysX cadence: 2 kHz physics, 50 Hz control.
    'physics': {'engine': 'Isaac Sim native PhysX GPU',
                'dt_s': 0.0005, 'control_dt_s': 0.02, 'gravity_m_s2': [0.0, 0.0, -9.81]},
}


if __name__ == '__main__':                                  # explicit, opt-in write
    print(build_library(os.environ.get('LIVING_ROOM_ASSET_ROOT', ASSET_ROOT)))
