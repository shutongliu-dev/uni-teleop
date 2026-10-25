"""The ``kitchen`` scene family: a galley kitchen run with a hob, a sink and a fridge.

The FR3 stands on its own plinth at the origin and works at the counter of a
galley kitchen that runs along the +x wall. Walking the run from the camera end
(-y) to the far end (+y):

  * a cooking section: a black-glass hob on the worktop, a built-in oven (door
    and handle bar, with a tea towel over it) underneath, and a chimney range
    hood above;
  * the work section in front of the robot - the scene's work table - with the
    seven loose objects on it;
  * a sink section: a steel basin in a real cut-out, a swan-neck tap behind it,
    two storage canisters and an upper wall-cabinet run above;
  * a tall fridge-freezer against the far wall at the end of the galley, facing
    back down the room, with a door-swing clearance in front of it.

A tiled splashback, a wall socket, a utensil crock, a waste bin and a stool
complete the room. Everything immovable is a static prop made of boxes; the
seven manipulation objects (sponge, mug, bowl, cutting board, paring knife,
capped bottle, storage jar) are authored OBJ assets.

Physical discipline (see ``bench.simulation.isaacsim_backend.scene_family``):
  * every dynamic object is an authored OBJ asset in ``assets/kitchen/v8`` with a
    positive mass, a uniform-density bounding-box inertia and at least one convex
    collision piece;
  * collision pieces carrying a mesh are authored in *asset-local* coordinates and
    placed at the asset origin (``position_m = [0, 0, 0]``, identity quaternion),
    which is the convention ``write_library`` uses for visual parts. Analytic
    ``box``/``cylinder`` pieces are unit primitives and carry their real
    ``position_m``. ``scene.add_task_asset`` applies ``position_m`` on top of the
    OBJ vertices, so a meshed piece whose vertices were moved must NOT also carry
    the move in ``position_m`` - that double shift is what displaced the v6 mug's
    collision 26.5 mm off its own visual. ``_recentre`` now moves the vertices of a
    meshed piece and the ``position_m`` of an analytic one, never both, and
    ``_verify_as_built`` rebuilds every piece exactly as ``add_task_asset`` does
    and refuses to write an asset whose collision union and visual union disagree
    by more than 1 mm, or whose declared bore or vessel interior is not open;
  * the mug's handle is a genuine through hole: twelve convex sectors of the ring
    that circumscribes the display torus, with the inner chords tangent to the
    17 mm finger hole; the mug and bowl walls are annular sectors so their
    interiors stay open;
  * thin parts take analytic box colliders, never a hull of a plate, and every
    one is at least 3 mm thick (the knife blade is 3.0 mm, the board 14 mm);
  * rest poses are computed from the manifest's own AABBs, so the lowest point of
    every object - collision and visual - is on the 0.90 m work-table top;
  * the work table (``SCENE['table']``) is the one worktop box under the objects.
    ``validate_family`` reads supports only from ``surfaces``/``static_props``,
    so ``SCENE['surfaces']`` must restate the same box; the two entries are kept
    identical (same box, material and friction) and the counter run itself has no
    top piece over the work section, so no third box coincides with them;
  * the FR3's home pose leaves its hand over the work section, so the third-person
    camera stands at the camera end of the galley at x = 0.75 m: every line of
    sight from it to an object passes on the +x side of the hand, and the arm's
    silhouette falls towards -x, off the work strip.

Masses are design values for a simulation task, not manufacturer measurements;
each asset records that explicitly in the manifest.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from bench.paths import DATA_ROOT, load_json
from scenes.rooms import DEFAULT_LIBRARY_ROOT
from bench.simulation.assets.scene_parts import (Asset, annulus_sectors, box_mesh, cylinder_mesh, lathe_mesh,
                                torus_ring, write_library)

FAMILY = 'kitchen'
# ``write_library`` refuses a non-empty target, so every rebuild is a new version;
# the history of v6-v8 is in the library's VERSIONS.md.
ASSET_VERSION = 'v8'
ASSET_ROOT = str(DEFAULT_LIBRARY_ROOT / FAMILY / ASSET_VERSION)
GENERATOR = 'scenes/rooms/kitchen.py'
LICENSE_ID = 'CC0-1.0'
# The as-built collision union must reproduce the visual union to this tolerance.
AS_BUILT_TOLERANCE_M = 1e-3

# --- shared room texture set (ambientCG, CC0 1.0) ---------------------------
MATERIAL_SOURCE_ROOT = f'{DATA_ROOT}/assets/materials/ambientcg/v1'
SOURCES = {'PaintedPlaster017': 'https://ambientcg.com/a/PaintedPlaster017'}

# --- the galley run along the +x wall (metres) ------------------------------
# One source of truth: the static props, the work table, the rest poses, the
# reach checks and the camera targets are all derived from these numbers.
FRONT_FACE_X = 0.380
CARCASS_X0, CARCASS_X1 = 0.38, 0.98
PLINTH_X0, PLINTH_X1 = 0.42, 0.98
CARCASS_Z0, CARCASS_Z1 = 0.10, 0.86
TOP_Z0, TOP_Z1 = 0.86, 0.90
COUNTER_TOP_Z = TOP_Z1
RUN_Y0, RUN_Y1 = -1.05, 1.14

# Cooking section: hob on the worktop, oven below, hood above.
HOB_SECTION_Y0, HOB_SECTION_Y1 = RUN_Y0, -0.45
HOB_X0, HOB_X1 = 0.42, 0.93
HOB_Y0, HOB_Y1 = -1.04, -0.46
HOB_GLASS_T = 0.006
HOOD_X0, HOOD_X1 = 0.50, 0.98
HOOD_Z0, HOOD_Z1 = 1.60, 1.70                 # canopy 0.70 m above the hob glass
CHIMNEY_X0, CHIMNEY_X1 = 0.76, 0.98
CHIMNEY_Y0, CHIMNEY_Y1 = -0.86, -0.64

# Work section: the work table the robot works on. The counter run has no top
# piece here; SCENE['table'] is this worktop.
WORK_Y0, WORK_Y1 = HOB_SECTION_Y1, 0.52

# Sink section: a basin hung in a cut-out of the worktop, tap behind it.
SINK_SECTION_Y0, SINK_SECTION_Y1 = WORK_Y1, RUN_Y1
SINK_X0, SINK_X1 = 0.47, 0.89
SINK_Y0, SINK_Y1 = 0.62, 1.04
SINK_FLOOR_BOTTOM_Z, SINK_FLOOR_TOP_Z = 0.66, 0.72

SPLASH_X0, SPLASH_X1 = 0.955, 1.00
SPLASH_Z1 = 1.45

# Upper wall cabinets over the sink section only: nothing hangs over the work
# section or the hob, so no overhead prop shares a footprint with an object.
UPPER_X0, UPPER_X1 = 0.63, 1.00
UPPER_Z0, UPPER_Z1 = 1.45, 2.15
UPPER_Y0, UPPER_Y1 = SINK_SECTION_Y0, SINK_SECTION_Y1

# The fridge-freezer stands against the far wall facing back down the galley,
# 0.66 m beyond the end of the run so its doors can swing open.
FRIDGE_X0, FRIDGE_X1 = 0.32, 0.98
FRIDGE_Y0, FRIDGE_Y1 = 1.82, 2.48             # body; the doors stand 20 mm proud
FRIDGE_DOOR_T = 0.020
FRIDGE_Z1 = 1.85

# --- room ------------------------------------------------------------------
ROOM_SIZE = [3.0, 4.8, 2.7]
ROOM_ORIGIN = [-0.5, 0.1, 0.0]           # interior x in [-2.0, 1.0], y in [-2.3, 2.5]

# --- authored dimensions and design masses ----------------------------------
SPONGE_SIZE = (0.090, 0.055, 0.025)
SPONGE_MASS = 0.008

MUG_OUTER_R, MUG_INNER_R, MUG_HEIGHT = 0.040, 0.0335, 0.095
MUG_FLOOR_T = 0.007
MUG_HANDLE_MAJOR, MUG_HANDLE_MINOR = 0.022, 0.005
# The ring's centre puts the tube's nearest point (centre_y - major - minor =
# 0.0390 m) 1.0 mm inside the 0.040 m outer wall, so the handle welds to the wall,
# and 5.5 mm clear of the 0.0335 m inner wall, so the interior stays open. The
# ring lies in the y-z plane with its axis along x.
MUG_HANDLE_CENTRE = (0.0, 0.066, 0.005)
MUG_HANDLE_SECTORS = 12
MUG_WALL_SECTORS = 12
MUG_HANDLE_BORE_R = MUG_HANDLE_MAJOR - MUG_HANDLE_MINOR          # 0.017 m
MUG_HANDLE_REACH_Y = MUG_HANDLE_CENTRE[1] + MUG_HANDLE_MAJOR + MUG_HANDLE_MINOR   # 0.093 m
MUG_MASS = 0.28
MUG_SIZE = (2 * MUG_OUTER_R, MUG_HANDLE_REACH_Y + MUG_OUTER_R, MUG_HEIGHT)  # (0.080, 0.133, 0.095)

BOWL_OUTER_R, BOWL_INNER_R, BOWL_HEIGHT = 0.065, 0.057, 0.070
BOWL_FLOOR_T = 0.010
BOWL_WALL_SECTORS = 12
BOWL_MASS = 0.25

BOARD_SIZE = (0.160, 0.220, 0.014)
BOARD_MASS = 0.16

# Paring knife. y = 0 is the blade/bolster/handle junction: the handle runs into
# -y (butt at -KNIFE_HANDLE_L) and the blade into +y (tip at +KNIFE_BLADE_L). The
# blade is 3.0 mm thick in both display and collision - the contract's thin-part
# floor, 6x the 0.5 mm PhysX contact offset - and shares its bottom plane with the
# 20 mm handle, which is how a knife lies on a board.
KNIFE_BLADE_L = 0.085
KNIFE_HANDLE_L = 0.095
KNIFE_BLADE_T = 0.003
KNIFE_HANDLE_T = 0.020
KNIFE_WIDTH = 0.018
KNIFE_BOLSTER_L = 0.008
KNIFE_MASS = 0.080
KNIFE_SIZE = (KNIFE_WIDTH, KNIFE_BLADE_L + KNIFE_HANDLE_L, KNIFE_HANDLE_T)  # (0.018, 0.18, 0.02)

BOTTLE_BODY_R, BOTTLE_NECK_R, BOTTLE_CAP_R = 0.032, 0.0145, 0.0155
BOTTLE_HEIGHT = 0.190
BOTTLE_BODY_TOP_Z = -0.032
BOTTLE_NECK_TOP_Z = 0.070
BOTTLE_MASS = 0.30

JAR_BODY_R, JAR_LID_R = 0.045, 0.042
JAR_HEIGHT = 0.130
JAR_BODY_H = 0.110
JAR_MASS = 0.30

# Nominal AABBs, used only if the authored manifest is not present yet. The CPU
# test asserts they agree with the built library, so a drift fails loudly.
DESIGN_SIZES = {
    'sponge': [SPONGE_SIZE[0], SPONGE_SIZE[1], SPONGE_SIZE[2]],
    'mug': [round(v, 6) for v in MUG_SIZE],
    'ceramic_bowl': [2 * BOWL_OUTER_R, 2 * BOWL_OUTER_R, BOWL_HEIGHT],
    'cutting_board': list(BOARD_SIZE),
    'paring_knife': [round(v, 6) for v in KNIFE_SIZE],
    'capped_bottle': [2 * BOTTLE_BODY_R, 2 * BOTTLE_BODY_R, BOTTLE_HEIGHT],
    'storage_jar': [2 * JAR_BODY_R, 2 * JAR_BODY_R, JAR_HEIGHT],
}


def _yaw(radians):
    """Quaternion (w, x, y, z) for a rotation about +z."""
    return [math.cos(radians / 2.), 0., 0., math.sin(radians / 2.)]


def _off(mesh, offset):
    mesh = mesh.copy()
    mesh.apply_translation(np.asarray(offset, dtype=float))
    return mesh


# Rotation taking a part built about +z (an x-y-plane ring) onto a ring in the
# y-z plane: (x, y, z) -> (z, y, -x). Shared by the handle's display and its
# collision sectors so the two cannot drift apart.
_HANDLE_ROTATION = np.array([[0., 0., 1.], [0., 1., 0.], [-1., 0., 0.]])


def _to_ring_plane(mesh, centre):
    """Move an x-y-plane ring onto the y-z plane and shift it to ``centre``."""
    mesh = mesh.copy()
    transform = np.eye(4)
    transform[:3, :3] = _HANDLE_ROTATION
    mesh.apply_transform(transform)
    mesh.apply_translation(np.asarray(centre, dtype=float))
    return mesh


def _handle_torus(centre, major, minor, *, psi_segments=64, phi_segments=20):
    """A smooth swept torus handle whose ring axis runs along x."""
    return _to_ring_plane(torus_ring(major, minor, 2 * math.pi, psi_segments, phi_segments), centre)


# --- the as-built frame -------------------------------------------------------
# ``scene.add_task_asset`` places every part's OBJ vertices (scaled), then applies
# the part's own quaternion_xyzw and position_m. These helpers reproduce exactly
# that composition so the module checks the geometry Isaac builds, not the
# geometry the authoring code happens to hold.

def _quat_xyzw_matrix(q):
    x, y, z, w = (float(v) for v in q)
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def _primitive_points(component):
    """Support points of an analytic piece in its own frame (exact for the AABB)."""
    geometry = component.get('geometry')
    if geometry == 'box':
        half = np.asarray(component['half_extents_m'], dtype=float)
        return np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]) * half
    if geometry == 'cylinder':
        radius, half_h = float(component['radius_m']), float(component['half_height_m'])
        angles = np.arange(256) * 2 * math.pi / 256
        ring = np.c_[radius * np.cos(angles), radius * np.sin(angles)]
        return np.r_[np.c_[ring, np.full(256, -half_h)], np.c_[ring, np.full(256, half_h)]]
    if geometry == 'sphere':
        radius = float(component['radius_m'])
        return np.array([[sx * radius, 0, 0] for sx in (-1, 1)] + [[0, sy * radius, 0] for sy in (-1, 1)]
                        + [[0, 0, sz * radius] for sz in (-1, 1)])
    raise ValueError(f'unknown analytic collision geometry {geometry!r}')


def _as_built_points(mesh, component):
    """One collision piece's points in the asset frame, as add_task_asset builds it."""
    if mesh is not None:
        points = np.asarray(mesh.vertices, dtype=float) * np.asarray(component.get('scale', [1, 1, 1]))
    else:
        points = _primitive_points(component)
    rotation = _quat_xyzw_matrix(component.get('quaternion_xyzw', [0., 0., 0., 1.]))
    return points @ rotation.T + np.asarray(component.get('position_m', [0., 0., 0.]), dtype=float)


def _union_bounds(point_sets):
    stacked = np.vstack(list(point_sets))
    return np.stack([stacked.min(axis=0), stacked.max(axis=0)])


def _convex_polygon(points_2d):
    """Andrew's monotone chain: the convex hull of 2-D points, counter-clockwise."""
    pts = sorted(set(map(tuple, np.round(np.asarray(points_2d, dtype=float), 12))))
    if len(pts) <= 2:
        return np.asarray(pts, dtype=float)

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower, upper = [], []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return np.asarray(lower[:-1] + upper[:-1], dtype=float)


def projected_clearance(points, axis_point, axis_direction):
    """Exact distance from an infinite axis to a convex piece (0 if the axis pierces it).

    A convex piece projects onto the plane normal to the axis as the convex
    polygon of its projected vertices, so the distance from the axis's footprint
    to that polygon is the true minimum over the whole piece - not a vertex
    sample, which would miss a face or an edge crossing the bore.
    """
    direction = np.asarray(axis_direction, dtype=float)
    direction = direction / np.linalg.norm(direction)
    helper = np.array([1., 0., 0.]) if abs(direction[0]) < 0.9 else np.array([0., 1., 0.])
    u = np.cross(direction, helper)
    u /= np.linalg.norm(u)
    v = np.cross(direction, u)
    rel = np.asarray(points, dtype=float) - np.asarray(axis_point, dtype=float)
    polygon = _convex_polygon(np.c_[rel @ u, rel @ v])
    if len(polygon) < 3:
        return float(np.min(np.linalg.norm(polygon, axis=1)))
    inside = True
    best = np.inf
    for a, b in zip(polygon, np.roll(polygon, -1, axis=0)):
        edge = b - a
        if edge[0] * (0 - a[1]) - edge[1] * (0 - a[0]) < 0:
            inside = False
        t = float(np.clip(-(a @ edge) / max(edge @ edge, 1e-300), 0.0, 1.0))
        best = min(best, float(np.linalg.norm(a + t * edge)))
    return 0.0 if inside else best


def _verify_as_built(asset_id, asset):
    """Refuse to write an asset whose built collision is not its real geometry."""
    visual = _union_bounds(np.asarray(mesh.vertices, dtype=float) for _, mesh, _ in asset.visual)
    pieces = [(label, component, _as_built_points(mesh, component))
              for label, mesh, component in asset.collision]
    collision = _union_bounds(points for _, _, points in pieces)
    deviation = float(np.max(np.abs(collision - visual)))
    if deviation > AS_BUILT_TOLERANCE_M:
        raise ValueError(f'{asset_id}: as built, the collision union {collision.tolist()} deviates '
                         f'{deviation * 1000:.2f} mm from the visual union {visual.tolist()}')
    handle = asset.data.get('handle')
    if isinstance(handle, dict) and 'axis_direction' in handle:
        bore = float(handle['hole_radius_m'])
        for label, _, points in pieces:
            clearance = projected_clearance(points, handle['centre_m'], handle['axis_direction'])
            if clearance < bore - 1e-9:
                raise ValueError(f'{asset_id}/{label}: as built, the piece reaches '
                                 f'{clearance * 1000:.3f} mm from the {bore * 1000:.1f} mm bore axis')
    vessel = asset.data.get('vessel')
    if isinstance(vessel, dict):
        inner = float(vessel['inner_radius_m'])
        for label, component, points in pieces:
            if component.get('role') == 'vessel_floor':
                if points[:, 2].max() > float(vessel['floor_top_z_m']) + 1e-9:
                    raise ValueError(f'{asset_id}/{label}: the floor rises into the vessel interior')
                continue
            clearance = projected_clearance(points, vessel['axis_point_m'], vessel['axis_direction'])
            if clearance < inner - 1e-9:
                raise ValueError(f'{asset_id}/{label}: as built, the piece reaches '
                                 f'{clearance * 1000:.3f} mm from the vessel axis, inside the '
                                 f'{inner * 1000:.1f} mm interior')
    asset.data['as_built_check'] = {
        'convention': 'mesh pieces: asset-local OBJ vertices at position_m 0; analytic pieces: '
                      'unit primitive at position_m (scene.add_task_asset composition)',
        'collision_vs_visual_aabb_max_deviation_m': round(deviation, 9),
        'tolerance_m': AS_BUILT_TOLERANCE_M}
    return asset


def _recentre(asset):
    """Translate every authored piece so the asset AABB is centred on the origin.

    ``validate_support`` compares ``position_m[2] - size_m[2] / 2`` against a
    support top, while ``write_library`` reports ``size_m`` as the *visual* AABB
    size; the two only agree when the AABB is centred on the local origin, so an
    asymmetric asset (the mug's handle, the knife's bolster) is shifted here.

    The shift is applied exactly once per piece, in the frame ``add_task_asset``
    uses: a meshed piece moves its vertices and keeps ``position_m = [0, 0, 0]``;
    an analytic piece (a unit primitive with no vertices) moves its
    ``position_m``. Moving both would place a meshed collider twice as far as its
    display, which is the v6 defect this replaces.
    """
    visual = _union_bounds(np.asarray(mesh.vertices, dtype=float) for _, mesh, _ in asset.visual)
    collision = _union_bounds(_as_built_points(mesh, component) for _, mesh, component in asset.collision)
    low, high = np.minimum(visual[0], collision[0]), np.maximum(visual[1], collision[1])
    shift = -(low + high) / 2.0
    if not np.any(np.abs(shift) > 1e-12):
        return asset
    # Each entry gets its own copy: a mesh shared between a display and a
    # collision entry would otherwise be translated twice.
    asset.visual = [(label, _off(mesh, shift), colour) for label, mesh, colour in asset.visual]
    rebuilt = []
    for label, mesh, component in asset.collision:
        component = dict(component)
        if mesh is not None:
            placed = [float(v) for v in component.get('position_m', [0., 0., 0.])]
            if any(abs(v) > 1e-12 for v in placed):
                raise ValueError(f'{asset.asset_id}/{label}: a meshed collision piece must be authored '
                                 f'in asset-local coordinates at position_m = 0, not {placed}')
            mesh = _off(mesh, shift)
        else:
            component['position_m'] = [float(v + s) for v, s in
                                       zip(component.get('position_m', [0., 0., 0.]), shift)]
        rebuilt.append((label, mesh, component))
    asset.collision = rebuilt
    # Recorded reference points must move with the geometry, or the manifest
    # would describe an axis the authored meshes no longer sit on.
    handle = asset.data.get('handle')
    if isinstance(handle, dict) and 'centre_m' in handle:
        handle['centre_m'] = [round(float(v + s), 9) for v, s in zip(handle['centre_m'], shift)]
        handle['centre_note'] = ('Ring centre in the as-built asset frame (after recentring); the bore '
                                 'axis is the line through this point along axis_direction.')
    vessel = asset.data.get('vessel')
    if isinstance(vessel, dict):
        vessel['axis_point_m'] = [round(float(v + s), 9) for v, s in zip(vessel['axis_point_m'], shift)]
        vessel['floor_top_z_m'] = round(float(vessel['floor_top_z_m'] + shift[2]), 9)
    return asset


def _add_ring(asset, label, outer, inner, height, centre, *, count=16,
              rotate_to_ring_plane=False, bore_radius=None):
    """Add a ring as bore-preserving convex collision sectors (asset-local, position 0).

    ``annulus_sectors`` circumscribes each sector's inner chords on the bore, so
    every point of every sector keeps at least ``inner`` of radial clearance from
    the ring axis. The same value is declared as the piece's clearance and, when
    ``bore_radius`` is given, as the bore the piece must not seal.

    The sectors are collision only. The caller authors the smooth display (lathe
    shell, swept torus); copying the faceted sectors into the display as well (v6,
    v7) buried the torus handle inside a square-section 12-gon and z-fought the
    lathe walls.
    """
    for index, piece in enumerate(annulus_sectors(outer, inner, height, count)):
        mesh = _to_ring_plane(piece, centre) if rotate_to_ring_plane else _off(piece, centre)
        component = {'geometry': 'convex_mesh', 'position_m': [0., 0., 0.],
                     'sector': f'{index:02d}/{count}', 'clearance_radius_m': float(inner)}
        if bore_radius is not None:
            component.update(bore=True, bore_radius_m=float(bore_radius))
        asset.add_collision(mesh, f'{label}_{index:02d}', **component)


# --- authored assets --------------------------------------------------------

def _sponge():
    """90 x 55 x 25 mm cellulose sponge: 8 g, a yellow block with a scour pad."""
    asset = Asset('sponge', dynamic=True, mass_kg=SPONGE_MASS, friction=0.90)
    length, width, height = SPONGE_SIZE
    body_h = height - 0.004
    asset.add_display(_off(box_mesh((length, width, body_h)), (0., 0., -0.002)),
                      'plastic_yellow', 'foam', roughness=0.92)
    asset.add_display(_off(box_mesh((length - 0.001, width - 0.001, 0.004)), (0., 0., 0.0105)),
                      'plastic_green', 'scour_pad', roughness=0.95)
    asset.add_collision(None, 'block', geometry='box',
                        half_extents_m=[length / 2., width / 2., height / 2.],
                        position_m=[0., 0., 0.])
    asset.data.update(
        nominal_size_m=list(SPONGE_SIZE),
        mass_design_note='Design value: 8 g for a 90 x 55 x 25 mm domestic sponge',
        collision_note='One analytic box exactly enclosing the 25 mm block; the foam and the scour '
                       'pad are display meshes, so thin detail is never hulled',
        rest_note='Flat on the counter, scour pad up')
    return asset


def _mug():
    """A 80 mm mug with a swept torus handle whose finger hole stays open."""
    asset = Asset('mug', dynamic=True, mass_kg=MUG_MASS, friction=0.55)
    outer, inner, height = MUG_OUTER_R, MUG_INNER_R, MUG_HEIGHT
    half = height / 2.
    floor_top = -half + MUG_FLOOR_T
    # Display: a thin-walled shell - outer wall, rim, rolled lip, inner wall.
    shell = lathe_mesh([(inner, floor_top), (outer, -half), (outer, half),
                        (inner, half), (inner - 0.0018, half - 0.0015),
                        (inner, floor_top)], segments=64)
    asset.add_display(shell, 'ceramic', 'vessel')
    asset.add_display(_off(cylinder_mesh(inner, MUG_FLOOR_T), (0., 0., -half + MUG_FLOOR_T / 2.)),
                      'ceramic', 'floor')
    asset.add_display(_handle_torus(MUG_HANDLE_CENTRE, MUG_HANDLE_MAJOR, MUG_HANDLE_MINOR),
                      'ceramic', 'handle')
    # Collision: the wall as annular sectors, the floor as a solid disc, and the
    # handle as twelve convex sectors of the ring that circumscribes its torus.
    _add_ring(asset, 'wall', outer, inner, height - MUG_FLOOR_T, (0., 0., MUG_FLOOR_T / 2.),
              count=MUG_WALL_SECTORS)
    asset.add_collision(None, 'base', geometry='cylinder', radius_m=outer,
                        half_height_m=MUG_FLOOR_T / 2.,
                        position_m=[0., 0., -half + MUG_FLOOR_T / 2.], clearance_radius_m=outer,
                        role='vessel_floor')
    _add_ring(asset, 'handle', MUG_HANDLE_MAJOR + MUG_HANDLE_MINOR, MUG_HANDLE_BORE_R,
              2 * MUG_HANDLE_MINOR, MUG_HANDLE_CENTRE,
              count=MUG_HANDLE_SECTORS, rotate_to_ring_plane=True,
              bore_radius=MUG_HANDLE_BORE_R)
    asset.data.update(
        nominal_size_m=[round(v, 6) for v in MUG_SIZE], capacity_ml=250.0,
        inner_radius_m=inner, wall_m=outer - inner, floor_thickness_m=MUG_FLOOR_T,
        floor_is_solid='The 7 mm base disc is a solid cylinder: a mug floor has no through hole.',
        handle={'major_radius_m': MUG_HANDLE_MAJOR, 'minor_radius_m': MUG_HANDLE_MINOR,
                'hole_radius_m': MUG_HANDLE_BORE_R, 'collision_sectors': MUG_HANDLE_SECTORS,
                'plane': 'y-z (ring axis along x)', 'centre_m': list(MUG_HANDLE_CENTRE),
                'axis_direction': [1.0, 0.0, 0.0],
                'collision_note': 'Twelve convex sectors of the annulus that circumscribes the '
                                  'display torus; the inner chords are tangent to the hole, so the '
                                  '17 mm finger hole is preserved exactly rather than approximated'},
        vessel={'axis_point_m': [0.0, 0.0, 0.0], 'axis_direction': [0.0, 0.0, 1.0],
                'inner_radius_m': inner, 'floor_top_z_m': floor_top,
                'note': 'Open vessel: no collision piece other than the floor disc (role '
                        'vessel_floor) may come within inner_radius_m of this axis'},
        mass_design_note='Design value: 0.28 kg mug, the bench reference mass',
        collision_note='Straight-wall approximation: the tapered wall becomes a uniform annulus of 12 '
                       'convex sectors over the height above the floor plus a solid 7 mm base disc, '
                       'so the vessel interior is never bridged. Meshed sectors are asset-local '
                       '(position_m 0); the base disc is an analytic cylinder at its position_m',
        bore_note='The handle is the one declared through bore. Its collision pieces were built by '
                  'annulus_sectors, which places the inner vertices at inner / cos(pi / 12), so the '
                  'tangent chord keeps the full 17 mm hole open. The module and the CPU test both '
                  'rebuild the pieces as scene.add_task_asset places them and measure the exact '
                  'projected distance from the handle axis.',
        rest_note='Upright on the counter, mouth up, handle towards +y')
    return asset


def _ceramic_bowl():
    """A 130 mm ceramic bowl: an open vessel, wall and floor authored separately."""
    asset = Asset('ceramic_bowl', dynamic=True, mass_kg=BOWL_MASS, friction=0.55)
    outer, inner, height = BOWL_OUTER_R, BOWL_INNER_R, BOWL_HEIGHT
    half = height / 2.
    floor_top = -half + BOWL_FLOOR_T
    asset.add_display(lathe_mesh([(inner, floor_top), (outer, -half), (outer, half),
                                  (inner, half), (inner, floor_top)], segments=64),
                      'porcelain', 'vessel')
    asset.add_display(_off(cylinder_mesh(inner, BOWL_FLOOR_T), (0., 0., -half + BOWL_FLOOR_T / 2.)),
                      'porcelain', 'floor')
    _add_ring(asset, 'wall', outer, inner, height - BOWL_FLOOR_T, (0., 0., BOWL_FLOOR_T / 2.),
              count=BOWL_WALL_SECTORS)
    asset.add_collision(None, 'base', geometry='cylinder', radius_m=outer,
                        half_height_m=BOWL_FLOOR_T / 2.,
                        position_m=[0., 0., -half + BOWL_FLOOR_T / 2.], clearance_radius_m=outer,
                        role='vessel_floor')
    asset.data.update(
        nominal_size_m=list(DESIGN_SIZES['ceramic_bowl']),
        inner_radius_m=inner, wall_m=outer - inner, floor_thickness_m=BOWL_FLOOR_T,
        vessel={'axis_point_m': [0.0, 0.0, 0.0], 'axis_direction': [0.0, 0.0, 1.0],
                'inner_radius_m': inner, 'floor_top_z_m': floor_top,
                'note': 'Open vessel: no collision piece other than the floor disc may come within '
                        'inner_radius_m of this axis'},
        mass_design_note='Design value: 0.25 kg for a 130 mm stoneware bowl',
        collision_note='The shallow taper becomes a uniform 8 mm annulus of 12 convex sectors plus a '
                       'solid 10 mm base disc; no piece spans the mouth',
        rest_note='Upright on the counter, mouth up')
    return asset


def _cutting_board():
    """A small 160 x 220 x 14 mm board: an analytic box collider, never a plate hull."""
    asset = Asset('cutting_board', dynamic=True, mass_kg=BOARD_MASS, friction=0.60)
    length, width, thickness = BOARD_SIZE
    asset.add_display(box_mesh((length, width, thickness)), 'wood', 'board')
    asset.add_display(_off(box_mesh((length - 0.004, width - 0.004, 0.002)),
                           (0., 0., thickness / 2. - 0.001)), 'dark_wood', 'work_face')
    for index, sign in enumerate((-1, 1)):
        asset.add_display(_off(box_mesh((0.004, width - 0.004, thickness - 0.002)),
                               (sign * (length / 2. - 0.002), 0., 0.)),
                          'dark_wood', f'end_grain_{index:02d}')
    asset.add_collision(None, 'board_box', geometry='box',
                        half_extents_m=[length / 2., width / 2., thickness / 2.],
                        position_m=[0., 0., 0.])
    asset.data.update(
        nominal_size_m=list(BOARD_SIZE),
        mass_design_note='Design value: 0.16 kg of 14 mm hardwood for a small prep board',
        collision_note='One analytic box; the 14 mm slab is well above the 3 mm thin-part floor and '
                       'is never replaced by a hull',
        rest_note='Flat on the counter, work face up')
    return asset


def _paring_knife():
    """A paring knife: a 3 mm blade and a 20 mm handle, two box collision pieces.

    The knife is laid out as three abutting segments along +y - handle, bolster,
    blade - so the body is continuous and the two collision boxes (blade, handle)
    share one bottom plane, which is how a knife lies on a board. The bolster is
    display only; it sits inside the handle box's footprint.
    """
    asset = Asset('paring_knife', dynamic=True, mass_kg=KNIFE_MASS, friction=0.45)
    blade_l, handle_l, blade_t, handle_t = KNIFE_BLADE_L, KNIFE_HANDLE_L, KNIFE_BLADE_T, KNIFE_HANDLE_T
    width = KNIFE_WIDTH
    half_t = handle_t / 2.
    # Blade lies on its flat face: its centre sits half its own thickness above
    # the shared bottom plane at z = -half_t.
    blade_z = -half_t + blade_t / 2.
    handle_y = -handle_l / 2.                      # handle occupies [-handle_l, 0]
    bolster_y = KNIFE_BOLSTER_L / 2. - 0.004       # bolster straddles the junction
    blade_y = blade_l / 2.                         # blade occupies [0, blade_l]
    asset.add_display(_off(box_mesh((width, handle_l, handle_t)), (0., handle_y, 0.)),
                      'dark_wood', 'handle')
    asset.add_display(_off(box_mesh((width - 0.001, KNIFE_BOLSTER_L, handle_t)),
                           (0., bolster_y, 0.)), 'steel', 'bolster')
    asset.add_display(_off(box_mesh((width, blade_l, blade_t)), (0., blade_y, blade_z)),
                      'steel', 'blade')
    # A ground edge: a thin witness strip along the blade's lower face.
    asset.add_display(_off(box_mesh((width - 0.004, blade_l - 0.006, 0.0004)),
                           (0., blade_y, blade_z - blade_t / 2. + 0.0002)), 'white', 'edge')
    asset.add_collision(None, 'handle_box', geometry='box',
                        half_extents_m=[width / 2., handle_l / 2., handle_t / 2.],
                        position_m=[0., handle_y, 0.])
    asset.add_collision(None, 'blade_box', geometry='box',
                        half_extents_m=[width / 2., blade_l / 2., blade_t / 2.],
                        position_m=[0., blade_y, blade_z])
    asset.data.update(
        nominal_size_m=[round(v, 6) for v in KNIFE_SIZE],
        blade_thickness_m=blade_t, handle_thickness_m=handle_t,
        blade_length_m=blade_l, handle_length_m=handle_l,
        thin_part_floor_m=0.003,
        mass_design_note='Design value: 0.080 kg for a paring knife',
        collision_note='Two analytic boxes (blade 3.0 mm, handle 20 mm) that abut at y = 0, so the '
                       'physical body is continuous. The blade box is exactly the 3 mm display blade: '
                       'the contract thin-part floor and 6x the 0.5 mm PhysX contact offset. No hull is '
                       'taken over either plate',
        rest_note='Flat on the counter; the blade and handle bottom faces are coplanar with the '
                  'surface and with each other, so no part of the body hangs above it')
    return asset


def _capped_bottle():
    """A capped bottle: body, neck and a separate cap, three convex cylinders."""
    asset = Asset('capped_bottle', dynamic=True, mass_kg=BOTTLE_MASS, friction=0.45)
    half = BOTTLE_HEIGHT / 2.
    body_r, neck_r, cap_r = BOTTLE_BODY_R, BOTTLE_NECK_R, BOTTLE_CAP_R
    body_top, neck_top, cap_base = BOTTLE_BODY_TOP_Z, BOTTLE_NECK_TOP_Z, 0.070
    shoulder_z = body_top + 0.028
    asset.add_display(lathe_mesh([(0., -half), (body_r - 0.002, -half), (body_r, -half + 0.008),
                                  (body_r, body_top), (neck_r, shoulder_z), (neck_r, neck_top),
                                  (0., neck_top)], segments=64), 'glass', 'bottle')
    asset.add_display(_off(cylinder_mesh(cap_r, half - cap_base), (0., 0., (cap_base + half) / 2.)),
                      'plastic_green', 'cap')
    # A paper label, inset a fraction of a millimetre so it never grows the
    # bottle's rated 64 mm diameter (the AABB the scene places against).
    asset.add_display(lathe_mesh([(body_r - 0.0002, -0.008), (body_r - 0.0002, 0.014),
                                  (body_r - 0.0012, 0.014), (body_r - 0.0012, -0.008)],
                                 segments=64), 'paper', 'label')
    asset.add_collision(None, 'body', geometry='cylinder', radius_m=body_r,
                        half_height_m=(body_top + half) / 2.,
                        position_m=[0., 0., (body_top - half) / 2.], clearance_radius_m=body_r)
    asset.add_collision(None, 'neck', geometry='cylinder', radius_m=neck_r + 0.002,
                        half_height_m=(cap_base - body_top) / 2.,
                        position_m=[0., 0., (cap_base + body_top) / 2.],
                        clearance_radius_m=neck_r + 0.002)
    asset.add_collision(None, 'cap', geometry='cylinder', radius_m=cap_r,
                        half_height_m=(half - cap_base) / 2.,
                        position_m=[0., 0., (half + cap_base) / 2.], clearance_radius_m=cap_r)
    asset.data.update(
        nominal_size_m=[2 * body_r, 2 * body_r, BOTTLE_HEIGHT],
        mass_design_note='Design value: 0.30 kg for a part-full 190 mm capped bottle',
        collision_note='Three convex cylinders (body, neck, cap) that tile the profile with no gap; '
                       'the label and the shoulder taper are display meshes. Closed on purpose: a '
                       'capped bottle has no through bore',
        rest_note='Upright on the counter, cap up')
    return asset


def _storage_jar():
    """A lidded storage jar: a straight-sided crock with its own lid.

    The display is an open crock (a real wall and floor) with a lid on top; the
    collision is the two convex cylinders of the closed jar, which is what the
    object is when the robot meets it.
    """
    asset = Asset('storage_jar', dynamic=True, mass_kg=JAR_MASS, friction=0.55)
    half = JAR_HEIGHT / 2.
    body_bottom = -half
    body_top = body_bottom + JAR_BODY_H
    asset.add_display(lathe_mesh([(0., body_bottom), (JAR_BODY_R - 0.002, body_bottom),
                                  (JAR_BODY_R, body_bottom + 0.008), (JAR_BODY_R, body_top),
                                  (JAR_BODY_R - 0.003, body_top),
                                  (JAR_BODY_R - 0.003, body_bottom + 0.012),
                                  (0., body_bottom + 0.012)], segments=64), 'ceramic', 'body')
    asset.add_display(_off(cylinder_mesh(JAR_LID_R, half - body_top), (0., 0., (half + body_top) / 2.)),
                      'dark_wood', 'lid')
    asset.add_display(_off(cylinder_mesh(0.012, 0.010), (0., 0., half - 0.005)), 'dark_wood', 'knob')
    asset.add_collision(None, 'body', geometry='cylinder', radius_m=JAR_BODY_R,
                        half_height_m=JAR_BODY_H / 2.,
                        position_m=[0., 0., (body_bottom + body_top) / 2.],
                        clearance_radius_m=JAR_BODY_R)
    asset.add_collision(None, 'lid', geometry='cylinder', radius_m=JAR_LID_R,
                        half_height_m=(half - body_top) / 2.,
                        position_m=[0., 0., (half + body_top) / 2.], clearance_radius_m=JAR_LID_R)
    asset.data.update(
        nominal_size_m=[2 * JAR_BODY_R, 2 * JAR_BODY_R, JAR_HEIGHT],
        mass_design_note='Design value: 0.30 kg for a 90 mm lidded ceramic jar',
        collision_note='Two convex cylinders (crock body and lid) as a closed jar; the knob is a '
                       'display mesh within the lid collider. Modelled lidded, so the asset declares '
                       'no through bore',
        rest_note='Upright on the counter, lid up')
    return asset


ASSET_BUILDERS = (
    ('sponge', _sponge),
    ('mug', _mug),
    ('ceramic_bowl', _ceramic_bowl),
    ('cutting_board', _cutting_board),
    ('paring_knife', _paring_knife),
    ('capped_bottle', _capped_bottle),
    ('storage_jar', _storage_jar),
)


def authored_assets():
    """Every asset, recentred and verified in the frame ``add_task_asset`` builds."""
    return [(asset_id, _verify_as_built(asset_id, _recentre(builder())))
            for asset_id, builder in ASSET_BUILDERS]


def build_library(asset_root=ASSET_ROOT):
    """Write the authored OBJ library; returns the manifest record.

    ``write_library`` refuses a non-empty target, so this is a one-shot authoring
    step: a rebuild goes to the next version directory (v9, ...) and
    ``ASSET_ROOT`` is repointed, never an overwrite.
    """
    return write_library(Path(asset_root), authored_assets(), generator=GENERATOR,
                         license_id=LICENSE_ID,
                         extra={'family': FAMILY, 'units': 'metres',
                                'source': 'Procedural geometry authored for the kitchen scene family',
                                'dimension_status': 'Simulation task design values; not manufacturer CAD',
                                'collision_note': 'Convex collision pieces are the physical shape; '
                                                  'their union, not one global hull, is the collider',
                                'collision_frame': 'Meshed pieces are asset-local OBJs at position_m 0 '
                                                   'and identity rotation; analytic pieces are unit '
                                                   'primitives placed by position_m. scene.add_task_asset '
                                                   'applies position_m/quaternion_xyzw on top of the '
                                                   'vertices, so this is the frame Isaac builds.',
                                'supersedes': [
                                    {'version': 'v6', 'kept_on_disk': True,
                                     'defects': ['mug collision double-shifted 26.5 mm in -y '
                                                 '(mesh recentred and position_m shifted)',
                                                 'paring_knife blade collider 2.0 mm < 3 mm',
                                                 'visual parts without a finish (rendered metallic)']},
                                    {'version': 'v7', 'kept_on_disk': True,
                                     'defects': ['collision sectors duplicated into the display: the '
                                                 'torus handle hidden in a square-section 12-gon, '
                                                 'sector walls z-fighting the lathe shells']}]})


def load_manifest(asset_root=ASSET_ROOT):
    """Read the authored manifest, or ``None`` if the library is not built yet."""
    path = Path(asset_root) / 'manifest.json'
    if not path.is_file():
        return None
    return load_json(path)


def asset_sizes(asset_root=ASSET_ROOT):
    """AABB size of every asset: measured from the manifest when it exists.

    Falls back to the design values only so that importing the module before the
    library is authored still yields a self-consistent scene; the CPU test asserts
    the two agree once the library exists.
    """
    manifest = load_manifest(asset_root)
    if not manifest:
        return dict(DESIGN_SIZES)
    return {key: list(record['size_m']) for key, record in manifest['assets'].items()}


# --- the scene --------------------------------------------------------------

# Work-strip layout on the work table. Low objects (knife, bowl, sponge, board)
# sit in front, the tall ones (bottle, mug, jar) behind them, so from the camera
# end of the galley nothing tall hides anything low. Objects beyond the robot's
# hand (y > 0.11 m) keep their near edge at x >= 0.45 m, so every sight line from
# the third-person camera passes on the +x side of the hand. The CPU test re-derives
# footprints, reach and visibility from the built manifest rather than trusting
# these numbers.
OBJECT_LAYOUT = (
    ('paring_knife', 'paring_knife', (0.50, -0.07), math.radians(-90.), 0.45),
    ('bowl', 'ceramic_bowl', (0.48, 0.04), 0.0, 0.55),
    ('sponge', 'sponge', (0.70, -0.06), 0.0, 0.90),
    ('cutting_board', 'cutting_board', (0.68, 0.09), 0.0, 0.60),
    ('bottle', 'capped_bottle', (0.49, 0.36), 0.0, 0.45),
    ('mug', 'mug', (0.60, 0.34), 0.0, 0.55),
    ('storage_jar', 'storage_jar', (0.70, 0.26), 0.0, 0.55),
)


def _build_objects():
    """Every dynamic object, resting exactly on the work-table top."""
    sizes = asset_sizes()
    objects = []
    for object_id, asset_id, xy, yaw_rad, friction in OBJECT_LAYOUT:
        size = sizes[asset_id]
        z = COUNTER_TOP_Z + size[2] / 2.
        objects.append({
            'id': object_id, 'asset_id': asset_id, 'asset_root': ASSET_ROOT,
            'position_m': [round(float(xy[0]), 6), round(float(xy[1]), 6), round(float(z), 6)],
            'quaternion_wxyz': [round(v, 9) for v in _yaw(yaw_rad)],
            'friction': friction,
            'rest_geometry': {'size_m': [round(float(v), 6) for v in size], 'origin': 'centre'},
        })
    return objects


def _textured(name, material_id, uv_scale, color):
    """A textured material bound to a CC0 set in the shared asset library.

    Maps use the ``<slot>_map`` keys of ``visuals.validate_material`` (the albedo
    map is ``base_color``), so the roughness map is bound under ``roughness_map``
    and no scalar roughness is needed. ``color`` is only the fallback the
    albedo connection overrides.
    """
    root = Path(MATERIAL_SOURCE_ROOT) / material_id
    return {'kind': 'textured', 'name': name, 'color': list(color), 'metallic': 0.0,
            'uv_scale_m': list(uv_scale), 'uv_scale_semantics': 'tiles_per_meter',
            'base_color': str(root / f'{material_id}_1K-JPG_Color.jpg'),
            'normal': str(root / f'{material_id}_1K-JPG_NormalGL.jpg'),
            'roughness_map': str(root / f'{material_id}_1K-JPG_Roughness.jpg'),
            'source': SOURCES[material_id], 'license': LICENSE_ID}


def _materials():
    """A CC0 texture where the shared library has one, flat colour otherwise.

    The shared ambientCG library holds no ceramic tile, so the floor and the
    splashback are flat colours with the roughness of glazed tile rather than
    textures pretending to be something they are not.
    """
    return {
        'tile_floor': {'kind': 'color', 'color': [0.76, 0.77, 0.75], 'roughness': 0.30},
        'plaster_wall': _textured('plaster_wall', 'PaintedPlaster017', uv_scale=(1.0, 1.0),
                                  color=[0.86, 0.85, 0.81]),
        'ceiling_white': {'kind': 'color', 'color': [0.93, 0.93, 0.92], 'roughness': 0.95},
        'baseboard_white': {'kind': 'color', 'color': [0.90, 0.90, 0.87], 'roughness': 0.55},
        'splash_tile': {'kind': 'color', 'color': [0.79, 0.83, 0.84], 'roughness': 0.18},
        'counter_quartz': {'kind': 'color', 'color': [0.30, 0.31, 0.33], 'roughness': 0.28},
        'cabinet_door': {'kind': 'color', 'color': [0.23, 0.34, 0.31], 'roughness': 0.45},
        'cabinet_carcass': {'kind': 'color', 'color': [0.88, 0.87, 0.84], 'roughness': 0.60},
        'steel_sink': {'kind': 'color', 'color': [0.70, 0.72, 0.74], 'roughness': 0.28,
                       'metallic': 0.85},
        'hob_glass': {'kind': 'color', 'color': [0.04, 0.04, 0.05], 'roughness': 0.08},
        'hob_ring': {'kind': 'color', 'color': [0.30, 0.31, 0.33], 'roughness': 0.35},
        'oven_glass': {'kind': 'color', 'color': [0.08, 0.08, 0.09], 'roughness': 0.12},
        'brushed_steel': {'kind': 'color', 'color': [0.66, 0.68, 0.70], 'roughness': 0.35,
                          'metallic': 0.80},
        'fridge_white': {'kind': 'color', 'color': [0.92, 0.92, 0.91], 'roughness': 0.30},
        'fridge_gasket': {'kind': 'color', 'color': [0.20, 0.21, 0.22], 'roughness': 0.70},
        'socket_plate': {'kind': 'color', 'color': [0.95, 0.95, 0.94], 'roughness': 0.35},
        'cloth_teal': {'kind': 'color', 'color': [0.36, 0.52, 0.55], 'roughness': 0.92},
        'ceramic_crock': {'kind': 'color', 'color': [0.84, 0.80, 0.73], 'roughness': 0.40},
        'wood_utensil': {'kind': 'color', 'color': [0.52, 0.37, 0.22], 'roughness': 0.55},
        'bin_grey': {'kind': 'color', 'color': [0.32, 0.34, 0.36], 'roughness': 0.50},
    }


def _box(part_id, centre, half, material):
    return {'id': part_id, 'geometry': 'box',
            'half_extents_m': [round(float(v), 6) for v in half],
            'position_m': [round(float(v), 6) for v in centre], 'material': material}


def _span(part_id, low, high, material):
    """A box part from its two corners."""
    low, high = np.asarray(low, dtype=float), np.asarray(high, dtype=float)
    return _box(part_id, (low + high) / 2., (high - low) / 2., material)


def _counter_run():
    """Base units of the galley: plinth, carcasses split around the sink pocket,
    door and drawer fronts, and the worktop pieces *other than* the work section
    (which is SCENE['table'])."""
    x0, x1 = CARCASS_X0, CARCASS_X1
    z0, z1 = CARCASS_Z0, CARCASS_Z1
    parts = [
        _span('plinth', (PLINTH_X0, RUN_Y0, 0.0), (PLINTH_X1, RUN_Y1, CARCASS_Z0), 'cabinet_carcass'),
        _span('carcass_hob', (x0, HOB_SECTION_Y0, z0), (x1, HOB_SECTION_Y1, z1), 'cabinet_carcass'),
        _span('carcass_work', (x0, WORK_Y0, z0), (x1, WORK_Y1, z1), 'cabinet_carcass'),
        # The sink cabinet is a real pocket: carcass walls around an empty middle.
        _span('carcass_sink_lo', (x0, SINK_SECTION_Y0, z0), (x1, SINK_Y0, z1), 'cabinet_carcass'),
        _span('carcass_sink_hi', (x0, SINK_Y1, z0), (x1, SINK_SECTION_Y1, z1), 'cabinet_carcass'),
        _span('carcass_sink_front', (x0, SINK_Y0, z0), (SINK_X0, SINK_Y1, z1), 'cabinet_carcass'),
        _span('carcass_sink_back', (SINK_X1, SINK_Y0, z0), (x1, SINK_Y1, z1), 'cabinet_carcass'),
        # Worktop pieces around the hob section and the sink cut-out.
        _span('top_hob', (x0, HOB_SECTION_Y0, TOP_Z0), (x1, HOB_SECTION_Y1, TOP_Z1), 'counter_quartz'),
        _span('top_sink_lo', (x0, SINK_SECTION_Y0, TOP_Z0), (x1, SINK_Y0, TOP_Z1), 'counter_quartz'),
        _span('top_sink_hi', (x0, SINK_Y1, TOP_Z0), (x1, SINK_SECTION_Y1, TOP_Z1), 'counter_quartz'),
        _span('top_sink_front', (x0, SINK_Y0, TOP_Z0), (SINK_X0, SINK_Y1, TOP_Z1), 'counter_quartz'),
        _span('top_sink_back', (SINK_X1, SINK_Y0, TOP_Z0), (x1, SINK_Y1, TOP_Z1), 'counter_quartz'),
        # A drawer above the oven; the oven door itself is the ``oven`` prop.
        _span('drawer_hob', (FRONT_FACE_X - 0.010, -1.02, 0.735), (FRONT_FACE_X, -0.48, 0.845),
              'cabinet_door'),
    ]
    for index, (y_lo, y_hi) in enumerate(((-0.43, 0.025), (0.035, 0.50), (0.54, 0.825), (0.835, 1.12))):
        parts.append(_span(f'door_{index:02d}', (FRONT_FACE_X - 0.010, y_lo, 0.12),
                           (FRONT_FACE_X, y_hi, 0.845), 'cabinet_door'))
    return {'id': 'counter_run', 'material': 'counter_quartz', 'friction': 0.55,
            'title_zh': '橱柜地柜与台面', 'parts': parts}


def _static_props():
    """Immovable boxes: the kitchen, and everything the robot must not move."""
    props = [_counter_run()]

    props.append({'id': 'backsplash', 'material': 'splash_tile', 'friction': 0.40,
                  'title_zh': '墙面瓷砖挡水板',
                  'parts': [_span('splash_field', (SPLASH_X0, RUN_Y0, TOP_Z1),
                                  (SPLASH_X1, RUN_Y1, SPLASH_Z1), 'splash_tile')]})

    # Cooking: a black-glass hob with four cooking zones and a touch strip.
    hob_top = COUNTER_TOP_Z + HOB_GLASS_T
    hob_parts = [_span('glass', (HOB_X0, HOB_Y0, COUNTER_TOP_Z), (HOB_X1, HOB_Y1, hob_top), 'hob_glass')]
    for index, (cx, cy, half) in enumerate(((0.58, -0.895, 0.085), (0.58, -0.605, 0.065),
                                            (0.81, -0.895, 0.065), (0.81, -0.605, 0.085))):
        hob_parts.append(_span(f'zone_{index:02d}', (cx - half, cy - half, hob_top),
                               (cx + half, cy + half, hob_top + 0.001), 'hob_ring'))
    hob_parts.append(_span('touch_strip', (HOB_X0 + 0.010, -0.82, hob_top),
                           (HOB_X0 + 0.050, -0.68, hob_top + 0.001), 'hob_ring'))
    props.append({'id': 'hob', 'material': 'hob_glass', 'friction': 0.30,
                  'title_zh': '电磁炉灶台', 'parts': hob_parts})

    # The built-in oven under the hob: glass door, handle bar on two standoffs.
    props.append({'id': 'oven', 'material': 'oven_glass', 'friction': 0.40,
                  'title_zh': '嵌入式烤箱', 'parts': [
                      _span('door', (FRONT_FACE_X - 0.010, -1.00, 0.16), (FRONT_FACE_X, -0.50, 0.71),
                            'oven_glass'),
                      _span('standoff_lo', (FRONT_FACE_X - 0.030, -0.94, 0.648),
                            (FRONT_FACE_X - 0.010, -0.92, 0.662), 'brushed_steel'),
                      _span('standoff_hi', (FRONT_FACE_X - 0.030, -0.58, 0.648),
                            (FRONT_FACE_X - 0.010, -0.56, 0.662), 'brushed_steel'),
                      _span('handle_bar', (FRONT_FACE_X - 0.042, -0.96, 0.645),
                            (FRONT_FACE_X - 0.030, -0.54, 0.665), 'brushed_steel')]})

    # A chimney range hood over the hob, 0.70 m above the glass.
    props.append({'id': 'range_hood', 'material': 'brushed_steel', 'friction': 0.40,
                  'title_zh': '抽油烟机', 'parts': [
                      _span('canopy', (HOOD_X0, HOB_Y0 + 0.01, HOOD_Z0), (HOOD_X1, HOB_Y1 - 0.01, HOOD_Z1),
                            'brushed_steel'),
                      _span('chimney', (CHIMNEY_X0, CHIMNEY_Y0, HOOD_Z1), (CHIMNEY_X1, CHIMNEY_Y1, 2.70),
                            'brushed_steel')]})

    # The sink basin hangs in the carcass pocket under the cut-out; the tap stands
    # on the strip of worktop behind the bowl, with its spout over the bowl.
    sx0, sx1, sy0, sy1 = SINK_X0, SINK_X1, SINK_Y0, SINK_Y1
    wall_t = 0.006
    props.append({'id': 'sink_basin', 'material': 'steel_sink', 'friction': 0.30,
                  'title_zh': '水槽与龙头', 'parts': [
                      _span('floor', (sx0, sy0, SINK_FLOOR_BOTTOM_Z), (sx1, sy1, SINK_FLOOR_TOP_Z),
                            'steel_sink'),
                      _span('wall_x_lo', (sx0, sy0, SINK_FLOOR_TOP_Z), (sx0 + wall_t, sy1, CARCASS_Z1),
                            'steel_sink'),
                      _span('wall_x_hi', (sx1 - wall_t, sy0, SINK_FLOOR_TOP_Z), (sx1, sy1, CARCASS_Z1),
                            'steel_sink'),
                      _span('wall_y_lo', (sx0 + wall_t, sy0, SINK_FLOOR_TOP_Z),
                            (sx1 - wall_t, sy0 + wall_t, CARCASS_Z1), 'steel_sink'),
                      _span('wall_y_hi', (sx0 + wall_t, sy1 - wall_t, SINK_FLOOR_TOP_Z),
                            (sx1 - wall_t, sy1, CARCASS_Z1), 'steel_sink'),
                      _span('tap_column', (0.915, 0.815, COUNTER_TOP_Z), (0.945, 0.845, 1.24),
                            'brushed_steel'),
                      _span('tap_spout', (0.78, 0.815, 1.205), (0.915, 0.845, 1.235), 'brushed_steel'),
                      _span('tap_nozzle', (0.78, 0.818, 1.175), (0.80, 0.842, 1.205), 'brushed_steel'),
                      _span('tap_lever', (0.925, 0.845, 1.15), (0.935, 0.905, 1.165), 'brushed_steel')]})

    # Upper wall cabinets over the sink section only.
    upper_parts = [_span('carcass', (UPPER_X0, UPPER_Y0, UPPER_Z0), (UPPER_X1, UPPER_Y1, UPPER_Z1),
                         'cabinet_carcass')]
    for index, (y_lo, y_hi) in enumerate(((UPPER_Y0 + 0.005, 0.825), (0.835, UPPER_Y1 - 0.005))):
        upper_parts.append(_span(f'door_{index:02d}', (UPPER_X0 - 0.012, y_lo, UPPER_Z0 + 0.005),
                                 (UPPER_X0, y_hi, UPPER_Z1 - 0.005), 'cabinet_door'))
    props.append({'id': 'upper_cabinets', 'material': 'cabinet_carcass', 'friction': 0.50,
                  'title_zh': '吊柜', 'parts': upper_parts})

    # The fridge-freezer at the far end of the galley, facing back down the room.
    door_y = FRIDGE_Y0 - FRIDGE_DOOR_T
    props.append({'id': 'fridge', 'material': 'fridge_white', 'friction': 0.50,
                  'title_zh': '冰箱', 'parts': [
                      _span('body', (FRIDGE_X0, FRIDGE_Y0, 0.0), (FRIDGE_X1, FRIDGE_Y1, FRIDGE_Z1),
                            'fridge_white'),
                      _span('freezer_door', (FRIDGE_X0 + 0.005, door_y, 0.02),
                            (FRIDGE_X1 - 0.005, FRIDGE_Y0, 0.725), 'fridge_white'),
                      _span('door_gap', (FRIDGE_X0 + 0.005, door_y, 0.725),
                            (FRIDGE_X1 - 0.005, FRIDGE_Y0, 0.745), 'fridge_gasket'),
                      _span('fridge_door', (FRIDGE_X0 + 0.005, door_y, 0.745),
                            (FRIDGE_X1 - 0.005, FRIDGE_Y0, FRIDGE_Z1 - 0.01), 'fridge_white'),
                      _span('fridge_handle', (FRIDGE_X0 + 0.030, door_y - 0.025, 0.95),
                            (FRIDGE_X0 + 0.050, door_y, 1.55), 'brushed_steel'),
                      _span('freezer_handle', (FRIDGE_X0 + 0.10, door_y - 0.025, 0.64),
                            (FRIDGE_X1 - 0.10, door_y, 0.66), 'brushed_steel')]})

    # A wall socket on the splashback beside the hob.
    props.append({'id': 'wall_socket', 'material': 'socket_plate', 'friction': 0.40,
                  'title_zh': '墙面插座',
                  'parts': [_span('plate', (SPLASH_X0 - 0.013, -0.37, 1.10), (SPLASH_X0, -0.28, 1.19),
                                  'socket_plate')]})

    # Counter clutter, all of it clear of the work strip at y in [-0.10, 0.45].
    props.append({'id': 'utensil_crock', 'material': 'ceramic_crock', 'friction': 0.60,
                  'title_zh': '厨具筒',
                  'parts': [_span('crock', (0.82, -0.38, COUNTER_TOP_Z), (0.92, -0.28, COUNTER_TOP_Z + 0.15),
                                  'ceramic_crock'),
                            _span('crock_utensils', (0.845, -0.355, COUNTER_TOP_Z + 0.15),
                                  (0.895, -0.305, COUNTER_TOP_Z + 0.25), 'wood_utensil')]})
    props.append({'id': 'counter_canisters', 'material': 'ceramic_crock', 'friction': 0.60,
                  'title_zh': '储物罐',
                  'parts': [_span('canister_a', (0.84, 0.535, COUNTER_TOP_Z), (0.92, 0.605, COUNTER_TOP_Z + 0.16),
                                  'ceramic_crock'),
                            _span('canister_b', (0.74, 0.535, COUNTER_TOP_Z), (0.82, 0.605, COUNTER_TOP_Z + 0.12),
                                  'ceramic_crock')]})
    props.append({'id': 'tea_towel', 'material': 'cloth_teal', 'friction': 0.85,
                  'title_zh': '挂在烤箱把手上的抹布',
                  'parts': [_span('towel', (FRONT_FACE_X - 0.050, -0.84, 0.43),
                                  (FRONT_FACE_X - 0.042, -0.66, 0.665), 'cloth_teal')]})

    # Floor clutter in the open floor, clear of the run, the plinth and the fridge
    # door swing.
    props.append({'id': 'waste_bin', 'material': 'bin_grey', 'friction': 0.60,
                  'title_zh': '垃圾桶',
                  'parts': [_span('bin_body', (0.50, -1.50, 0.0), (0.80, -1.20, 0.52), 'bin_grey'),
                            _span('bin_rim', (0.495, -1.505, 0.52), (0.805, -1.195, 0.55), 'bin_grey')]})
    stool_parts = [_span('seat', (-1.18, 0.62, 0.575), (-0.82, 0.98, 0.625), 'wood_utensil')]
    for index, (sx, sy) in enumerate(((-1, -1), (-1, 1), (1, -1), (1, 1))):
        cx, cy = -1.0 + sx * 0.14, 0.8 + sy * 0.14
        stool_parts.append(_span(f'leg_{index:02d}', (cx - 0.02, cy - 0.02, 0.0), (cx + 0.02, cy + 0.02, 0.575),
                                 'wood_utensil'))
    props.append({'id': 'stool', 'material': 'wood_utensil', 'friction': 0.80,
                  'title_zh': '厨房凳', 'parts': stool_parts})
    return props


# The work table: the worktop of the work section, in front of the robot.
WORK_TABLE = {
    'position_m': [(CARCASS_X0 + CARCASS_X1) / 2., (WORK_Y0 + WORK_Y1) / 2., (TOP_Z0 + TOP_Z1) / 2.],
    'half_extents_m': [(CARCASS_X1 - CARCASS_X0) / 2., (WORK_Y1 - WORK_Y0) / 2., (TOP_Z1 - TOP_Z0) / 2.],
    'material': 'counter_quartz',
    'friction': 0.55,
}
WORK_TABLE_TOP_CENTRE = [WORK_TABLE['position_m'][0], WORK_TABLE['position_m'][1], COUNTER_TOP_Z]


def _cameras():
    return [
        # At the camera end of the galley, looking down the run at the work-table
        # centre: hob and hood in the near right, the work strip in the middle, the
        # sink, tap, upper cabinets and fridge beyond. Standing at x = 0.75 m keeps
        # every sight line to an object on the +x side of the robot's hand.
        {'id': 'third_person', 'kind': 'world', 'role': 'third_person',
         'eye': [0.75, -1.90, 1.62], 'target': list(WORK_TABLE_TOP_CENTRE), 'focal_length_mm': 20.0,
         'resolution': [960, 720], 'title_zh': '第三人称视角'},
        # A closer, lower task view from just past the end of the run (beyond the
        # hood's edge, over the floor): the work strip with the sink and fridge
        # behind it.
        {'id': 'counter_view', 'kind': 'world', 'role': 'table',
         'eye': [0.72, -1.20, 1.48], 'target': [0.62, 0.18, 0.92], 'focal_length_mm': 24.0,
         'resolution': [960, 720], 'title_zh': '台面视角'},
        {'id': 'wrist', 'kind': 'link', 'role': 'wrist', 'mount_link': 'right_hand_C_MC',
         'offset_xyz_m': [0.0, 0.0, 0.06], 'offset_rpy_rad': [0.0, 0.0, 0.0],
         'look_axis': '+z', 'clipping_m': [0.02, 20.0], 'resolution': [640, 480],
         'title_zh': '手腕相机'},
    ]


SCENE = {
    'schema_version': 'isaacsim-scene-family-v1',
    'family': FAMILY,
    'title_zh': '厨房台面',
    'description_zh': ('家庭厨房场景：沿墙布置的一字型橱柜，近端为黑色玻璃电磁炉灶台、嵌入式烤箱和'
                       '烟囱式抽油烟机，中段是机械臂面前的工作台面，远端为带龙头的嵌入式水槽与吊柜，'
                       '走廊尽头立着一台双门冰箱；台面上有海绵、带真实环形把手的马克杯、陶瓷碗、'
                       '小砧板、削皮刀、带盖瓶子和储物罐，还有厨具筒、储物罐、挂在烤箱把手上的抹布'
                       '与墙面插座；地面铺瓷砖、墙面刷乳胶漆、白色天花板带吸顶灯，'
                       '房间里还有垃圾桶和厨房凳。'),
    'room': {
        'enabled': True,
        'size_m': list(ROOM_SIZE),
        'origin_m': list(ROOM_ORIGIN),
        'wall_thickness_m': 0.12,
        'baseboard_m': 0.09,
        'uv_scale': {'floor': [1.0, 1.0], 'wall': [1.0, 1.0]},
        'surfaces': {'floor': 'tile_floor', 'wall': 'plaster_wall',
                     'ceiling': 'ceiling_white', 'baseboard': 'baseboard_white'},
        'openings': [
            # Left (-x) wall: u runs along +y from the wall centre (y = 0.1).
            {'id': 'door', 'wall': 'left', 'center_m': [-1.20, -0.30], 'size_m': [0.90, 2.10],
             'kind': 'door'},
            # Front (+y, far) wall: u runs along +x from the wall centre (x = -0.5).
            {'id': 'window', 'wall': 'front', 'center_m': [-0.55, 0.20], 'size_m': [1.10, 1.00],
             'kind': 'window'},
        ],
        'ceiling_lights': [
            {'id': 'lamp_counter', 'position_m': [0.45, -0.20, 2.62], 'extent_m': [0.90, 0.70],
             'intensity': 9000.0, 'color': [1.0, 0.97, 0.91]},
            {'id': 'lamp_galley_end', 'position_m': [0.45, 1.55, 2.62], 'extent_m': [0.60, 0.60],
             'intensity': 6000.0, 'color': [1.0, 0.97, 0.91]},
            {'id': 'lamp_room', 'position_m': [-1.10, -0.90, 2.62], 'extent_m': [0.60, 0.60],
             'intensity': 4000.0, 'color': [1.0, 0.98, 0.94]},
        ],
        'dome': {'enabled': True, 'intensity': 350.0, 'color': [0.85, 0.90, 1.0],
                 'hdri_visible': False},
        'collide': {'floor': True, 'walls': True, 'ceiling': False},
    },
    'materials': _materials(),
    'cameras': _cameras(),
    'physics': {'dt_s': 0.0005, 'control_dt_s': 0.02, 'gravity_m_s2': [0.0, 0.0, -9.81]},
    'robot': {'base_position_m': [0.0, 0.0, 0.75], 'hand_id': 'sharpa_w01_right',
              'assembly_version': 'isaacsim_6_1_v2', 'assets_root': f'{DATA_ROOT}/assets',
              'arm_home_rad': [0.4, -0.5, 0.0, -2.3, 0.0, 1.8, 0.8],
              'arm_kp': 2000.0, 'arm_kd': 140.0, 'hand_kp': 3.0, 'hand_kd': 0.1},
    'pedestal': {'position_m': [0.0, 0.0, 0.70], 'half_extents_m': [0.22, 0.22, 0.05],
                 'skirt_height_m': 0.65},
    # The work table is the worktop of the work section; ``surfaces`` restates the
    # identical box because validate_family reads supports only from surfaces and
    # static props (a limitation of the shared scene schema).
    'table': dict(WORK_TABLE, position_m=list(WORK_TABLE['position_m']),
                  half_extents_m=list(WORK_TABLE['half_extents_m'])),
    'surfaces': [{'id': 'counter_top', 'position_m': list(WORK_TABLE['position_m']),
                  'half_extents_m': list(WORK_TABLE['half_extents_m']),
                  'material': WORK_TABLE['material'], 'friction': WORK_TABLE['friction'],
                  'title_zh': '台面工作区',
                  'note': 'Same box as SCENE.table, restated so validate_family can see the support'}],
    'static_props': _static_props(),
    'fixtures': [],
    'objects': _build_objects(),
    'cloth': {'enabled': False},
    'isaacsim': {'physics_dt_s': 0.0005, 'robot_self_collisions': True,
                 'filter_thumb_cmc_structure': True},
}
