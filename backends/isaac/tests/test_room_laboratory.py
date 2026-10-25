"""CPU contracts for the ``laboratory`` scene family (no OpenUSD, PhysX, Isaac Sim or GPU).

The physical promises are re-derived from the authored files, never read back from
the numbers the module wrote down:

* the scene passes ``family.validate_family`` and pins a NEW library version (v2),
  with the superseded v1 kept byte-identical as evidence;
* every object and fixture asset resolves with verified hashes;
* colliders are rebuilt exactly as ``scene.add_task_asset`` builds them (OBJ
  vertices as written, then the part's quaternion and ``position_m``): their box
  must equal the visual box, every vertex must stay inside the visible outer
  surface, and the lowest collider point must sit on its declared support;
* the bore rule is checked with the EXACT distance from the axis to each convex
  piece (2-D hull of the piece, polygon-to-axis distance), not the nearest vertex;
  blind vessels carry a real collision floor and declare no bore;
* every convex piece is >= 3 mm across in every direction (exact minimum width,
  or a rigorous inscribed-ball lower bound for large hulls);
* the racks hold what they carry (closed holes on all four sides), tips stand
  point down with the collar on the deck;
* no static collider duplicates another, every declared support is a real
  collider top, every material that is used is defined, and nothing sits in a
  wall opening except the window's own glazing;
* cameras: world cameras 1.5 - 2.2 m from the bench centre and 0.6 - 1.4 m above
  it, inside the room, seeing every object unobstructed; the wrist camera is
  checked with forward kinematics at the home pose and must look at the bench;
* the robot at its home pose clears every prop and object;
* the same checks, run on the v1 library, detect each defect v2 fixes.
"""
from __future__ import annotations

import copy
import functools
import hashlib
import math
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest

from scenes.rooms import laboratory as lab
from bench.simulation.assets.task_parts import load_obj_mesh, load_task_asset
from bench.simulation.isaacsim_backend.cameras import resolve_cameras
from bench.simulation.isaacsim_backend.scene_family import (FamilyError, support_surfaces, uniform_box_inertia,
                                               validate_family, validate_support)
from bench.simulation.isaacsim_backend.room import WALL_FACES, _wall_geometry, validate_room
from bench.simulation.isaacsim_backend.visuals import validate_material

SCENE = lab.SCENE
MANIFEST = lab.load_manifest()
SUPPORTS = support_surfaces(SCENE)
V1_ROOT = Path(lab.SUPERSEDED_ASSET_ROOTS['v1'])
MIN_WIDTH = 0.003
TOL = 1e-7          # OBJ vertices load as float32 (as USD points do): 0.1 um
VESSELS = ('beaker_50ml', 'beaker_150ml', 'beaker_250ml', 'erlenmeyer_flask', 'graduated_cylinder')
DYNAMIC_ASSETS = sorted({spec['asset_id'] for spec in SCENE['objects']})
FIXTURE_ASSETS = sorted({spec['asset_id'] for spec in SCENE['fixtures']})

pytestmark = pytest.mark.skipif(MANIFEST is None, reason=f'library not built under {lab.ASSET_ROOT}')


# --- geometry helpers ---------------------------------------------------------

def _quat_xyzw_matrix(q):
    x, y, z, w = (float(v) for v in q)
    n = math.sqrt(x * x + y * y + z * z + w * w)
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def _quat_wxyz_matrix(q):
    w, x, y, z = q
    return _quat_xyzw_matrix([x, y, z, w])


def part_points(part, segments=64):
    """A part's vertices in its asset frame, placed exactly as scene.add_task_asset does.

    Meshes are loaded with ``load_obj_mesh`` (which does not recentre), scaled,
    then rotated by ``quaternion_xyzw`` and moved by ``position_m``. Analytic
    shapes are sampled on their true surface (inscribed polytopes).
    """
    geometry = part.get('geometry', 'convex_mesh')
    if geometry in ('mesh', 'convex_mesh'):
        vertices, _ = load_obj_mesh(part['mesh_path'])
        vertices = vertices.astype(float) * np.asarray(part.get('scale', [1, 1, 1]), dtype=float)
    elif geometry == 'box':
        h = np.asarray(part['half_extents_m'], dtype=float)
        vertices = np.array([[a * h[0], b * h[1], c * h[2]] for a in (-1, 1) for b in (-1, 1) for c in (-1, 1)])
    elif geometry == 'cylinder':
        radius, half = float(part['radius_m']), float(part['half_height_m'])
        angles = np.arange(segments) * 2 * math.pi / segments
        ring = np.c_[radius * np.cos(angles), radius * np.sin(angles)]
        vertices = np.vstack([np.c_[ring, np.full(segments, z)] for z in (-half, half)])
    elif geometry == 'sphere':
        radius = float(part['radius_m'])
        vertices = np.array([[radius * math.sin(t) * math.cos(p), radius * math.sin(t) * math.sin(p),
                              radius * math.cos(t)]
                             for t in np.linspace(0., math.pi, 33)
                             for p in np.arange(segments) * 2 * math.pi / segments])
    else:
        raise ValueError(f'unknown collision geometry {geometry!r}')
    rotation = _quat_xyzw_matrix(part.get('quaternion_xyzw', [0, 0, 0, 1]))
    return vertices @ rotation.T + np.asarray(part.get('position_m', [0, 0, 0]), dtype=float)


def _hull(points):
    import trimesh
    return trimesh.convex.convex_hull(np.asarray(points, dtype=float))


def _hull_2d(points):
    pts = sorted(set(map(tuple, np.round(np.asarray(points, dtype=float)[:, :2], 12))))

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
    return np.asarray(lower[:-1] + upper[:-1])


def axis_distance(points):
    """EXACT distance from the asset's z axis to the convex hull of ``points``.

    The closest point of a convex body to a vertical line projects onto the
    closest point of its 2-D shadow to the origin, so this is the shadow
    polygon's distance to the origin (0 if the shadow contains it). The nearest
    vertex radius - which the v1 test used - can only over-estimate it.
    """
    poly = _hull_2d(points)
    count = len(poly)
    inside = all((poly[(i + 1) % count][0] - poly[i][0]) * (-poly[i][1]) -
                 (poly[(i + 1) % count][1] - poly[i][1]) * (-poly[i][0]) >= -1e-15 for i in range(count))
    if inside:
        return 0.0
    best = math.inf
    for i in range(count):
        a, b = poly[i], poly[(i + 1) % count]
        d = b - a
        t = float(np.clip(-(a @ d) / (d @ d), 0., 1.))
        best = min(best, float(np.linalg.norm(a + t * d)))
    return best


def exact_min_width(points):
    """Exact minimum width of a convex polytope: over face normals and edge-pair normals."""
    hull = _hull(points)
    v = np.asarray(hull.vertices, dtype=float)
    e = v[hull.edges_unique[:, 1]] - v[hull.edges_unique[:, 0]]
    e = e / np.linalg.norm(e, axis=1)[:, None]
    e = np.where((e[:, [0]] < 0) | ((e[:, [0]] == 0) & (e[:, [1]] < 0)), -e, e)
    e = np.unique(np.round(e, 9), axis=0)
    candidates = [np.asarray(hull.face_normals)]
    c = np.cross(e[:, None, :], e[None, :, :]).reshape(-1, 3)
    n = np.linalg.norm(c, axis=1)
    candidates.append(c[n > 1e-9] / n[n > 1e-9][:, None])
    best = math.inf
    for d in candidates:
        for start in range(0, len(d), 20000):
            proj = v @ d[start:start + 20000].T
            best = min(best, float((proj.max(0) - proj.min(0)).min()))
    return best


def inscribed_diameter(points):
    """Diameter of the largest ball inside the hull (Chebyshev centre LP): a width lower bound."""
    from scipy.optimize import linprog
    hull = _hull(points)
    normals = np.asarray(hull.face_normals, dtype=float)
    offsets = np.einsum('ij,ij->i', normals, np.asarray(hull.vertices)[hull.faces[:, 0]])
    a_ub = np.c_[normals, np.ones(len(normals))]
    result = linprog([0, 0, 0, -1], A_ub=a_ub, b_ub=offsets, bounds=[(None, None)] * 3 + [(0, None)],
                     method='highs')
    assert result.success, result.message
    return 2. * float(result.x[3])


def piece_min_width(part):
    """Smallest width of one collision piece (exact, or a rigorous lower bound for big hulls)."""
    geometry = part.get('geometry', 'convex_mesh')
    if geometry == 'box':
        return 2. * min(part['half_extents_m'])
    if geometry == 'cylinder':
        return min(2. * part['radius_m'], 2. * part['half_height_m'])
    if geometry == 'sphere':
        return 2. * part['radius_m']
    points = part_points(part)
    if len(_hull(points).vertices) <= 64:
        return exact_min_width(points)
    bound = inscribed_diameter(points)
    return bound if bound >= MIN_WIDTH else exact_min_width(points)


def _box_corners(lo, hi):
    return np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])


def box_penetration(points, lo, hi):
    """Separating-axis overlap depth of a convex point set and an axis-aligned box.

    <= 0 means separated or touching. Axes: the box normals, the hull's face
    normals and the cross products of the hull's edges with the box axes.
    """
    p = np.asarray(points, dtype=float)
    if np.any(p.max(0) <= lo + 1e-9) or np.any(p.min(0) >= hi - 1e-9):
        return 0.0
    hull = _hull(p)
    v = np.asarray(hull.vertices)
    edges = v[hull.edges_unique[:, 1]] - v[hull.edges_unique[:, 0]]
    edges = np.unique(np.round(edges / np.linalg.norm(edges, axis=1)[:, None], 9), axis=0)
    axes = [np.eye(3), np.asarray(hull.face_normals)]
    for k in range(3):
        c = np.cross(edges, np.eye(3)[k])
        n = np.linalg.norm(c, axis=1)
        axes.append(c[n > 1e-9] / n[n > 1e-9][:, None])
    corners = _box_corners(lo, hi)
    best = math.inf
    for d in axes:
        pa, pb = v @ d.T, corners @ d.T
        best = min(best, float((np.minimum(pa.max(0), pb.max(0)) - np.maximum(pa.min(0), pb.min(0))).min()))
    return best


def _outside_revolved_profile(profile, points):
    """How far each point lies OUTSIDE the solid of revolution of an outer (r, z) profile.

    The region is the profile closed along the axis; a point inside scores 0,
    otherwise its distance to the region's boundary in the (r, z) plane. This is
    well conditioned near a round bottom, unlike comparing radii at equal z.
    """
    poly = [(0., profile[0][1])] + [tuple(v) for v in profile] + [(0., profile[-1][1])]
    poly = np.asarray(poly, dtype=float)
    rz = np.c_[np.linalg.norm(points[:, :2], axis=1), points[:, 2]]
    a, b = poly, np.roll(poly, -1, axis=0)
    inside = np.zeros(len(rz), dtype=bool)
    for (r0, z0), (r1, z1) in zip(a, b):           # even-odd ray casting along +r
        crosses = (z0 > rz[:, 1]) != (z1 > rz[:, 1])
        with np.errstate(divide='ignore', invalid='ignore'):
            r_hit = r0 + (rz[:, 1] - z0) * (r1 - r0) / (z1 - z0)
        inside ^= crosses & (rz[:, 0] < r_hit)
    d = b - a
    t = np.clip(((rz[:, None, :] - a[None]) * d[None]).sum(2) / np.maximum((d * d).sum(1), 1e-30)[None], 0., 1.)
    nearest = a[None] + t[..., None] * d[None]
    distance = np.linalg.norm(rz[:, None, :] - nearest, axis=2).min(1)
    return np.where(inside, 0., distance)


# --- scene helpers --------------------------------------------------------------

@functools.lru_cache(maxsize=None)
def _asset(asset_id, root=lab.ASSET_ROOT):
    return load_task_asset(asset_id, root, verify_hashes=True)


def world_pieces(spec):
    asset = _asset(spec['asset_id'], spec['asset_root'])
    rotation = _quat_wxyz_matrix(spec.get('quaternion_wxyz', [1, 0, 0, 0]))
    origin = np.asarray(spec['position_m'], dtype=float)
    return [(part, part_points(part) @ rotation.T + origin) for part in asset['collision_parts']]


@functools.lru_cache(maxsize=None)
def _world_objects():
    return {spec['id']: world_pieces(spec) for spec in SCENE['objects']}


@functools.lru_cache(maxsize=None)
def _world_fixtures():
    return {spec['id']: world_pieces(spec) for spec in SCENE['fixtures']}


def static_boxes(scene=SCENE):
    """Every box collider add_structured_scene builds: table, plates, prop parts, pedestal."""
    rows = []
    table = scene['table']
    c, h = np.asarray(table['position_m'], dtype=float), np.asarray(table['half_extents_m'], dtype=float)
    rows.append(('table', 'top', c - h, c + h, {'material': table.get('material', 'table')}))
    for plate in scene.get('surfaces', []) or []:
        c, h = np.asarray(plate['position_m'], dtype=float), np.asarray(plate['half_extents_m'], dtype=float)
        rows.append((plate['id'], 'plate', c - h, c + h, plate))
    for prop in scene.get('static_props', []):
        for part in prop['parts']:
            c, h = np.asarray(part['position_m'], dtype=float), np.asarray(part['half_extents_m'], dtype=float)
            rows.append((prop['id'], part['id'], c - h, c + h, part))
    pedestal = scene['pedestal']
    c, h = np.asarray(pedestal['position_m'], dtype=float), np.asarray(pedestal['half_extents_m'], dtype=float)
    rows.append(('pedestal', 'top', c - h, c + h, {}))
    skirt = float(pedestal.get('skirt_height_m', 0.))
    if skirt > 0:
        rows.append(('pedestal', 'skirt', np.r_[c[:2] - h[:2], c[2] - h[2] - skirt],
                     np.r_[c[:2] + h[:2], c[2] - h[2]], {}))
    return rows


def _support_under(x, y):
    over = [s for s in SUPPORTS if s['xy_min'][0] - 1e-6 <= x <= s['xy_max'][0] + 1e-6
            and s['xy_min'][1] - 1e-6 <= y <= s['xy_max'][1] + 1e-6]
    return max(over, key=lambda s: s['top_z_m']) if over else None


def _prop(prop_id):
    return next(prop for prop in SCENE['static_props'] if prop['id'] == prop_id)


def _opening_boxes(scene=SCENE, depth_into_room=0.10):
    """World box of each opening: its wall-plane rectangle through the wall and 10 cm into the room."""
    room = validate_room(scene['room'])
    boxes = []
    for opening in room['openings']:
        wall = _wall_geometry(room, opening['wall'])
        u, v = opening['center_m']
        su, sv = (s / 2. for s in opening['size_m'])
        centre = wall['center'] + u * wall['u_axis'] + v * wall['v_axis']
        half = np.abs(wall['u_axis']) * su + np.abs(wall['v_axis']) * sv
        normal = wall['normal']
        lo = centre - half + np.minimum(normal * room['wall_thickness_m'], -normal * depth_into_room)
        hi = centre + half + np.maximum(normal * room['wall_thickness_m'], -normal * depth_into_room)
        boxes.append((opening, lo, hi))
    return boxes


def _overlap(lo_a, hi_a, lo_b, hi_b):
    return np.minimum(hi_a, hi_b) - np.maximum(lo_a, lo_b)


def _hole_clearance(parts, x, y, radius):
    """Gap from a round body (centre x, y) to the nearest plate part on +x, -x, +y, -y.

    ``None`` for a side means no part bounds it there: a slot, not a hole.
    """
    sides = []
    for axis, sign in ((0, 1), (0, -1), (1, 1), (1, -1)):
        gaps = []
        for part in parts:
            c, h = np.asarray(part['position_m']), np.asarray(part['half_extents_m'])
            other = 1 - axis
            if abs(c[other] - (x, y)[other]) > h[other] + radius:      # not beside the body
                continue
            gap = sign * (c[axis] - sign * h[axis] - (x, y)[axis])
            if gap > 0:
                gaps.append(gap - radius)
        sides.append(min(gaps) if gaps else None)
    return sides


# --- forward kinematics (URDF only, CPU) -------------------------------------------

def _rpy_matrix(r, p, y):
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


def _axis_angle(axis, angle):
    a = np.asarray(axis, dtype=float)
    a = a / np.linalg.norm(a)
    k = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + math.sin(angle) * k + (1 - math.cos(angle)) * k @ k


def _floats(text, default='0 0 0'):
    return [float(v) for v in (text or default).split()]


@functools.lru_cache(maxsize=None)
def _robot_urdf():
    from bench.simulation.isaacsim_backend.robot_asset import sharpa_assembly_dir
    robot = SCENE['robot']
    return sharpa_assembly_dir(robot['assets_root'], robot['assembly_version']) / 'fr3_sharpa.urdf'


def _robot():
    """URDF of the pinned assembly and world transforms of every link at the home pose."""
    robot = SCENE['robot']
    urdf = _robot_urdf()
    if not urdf.is_file():
        pytest.skip(f'assembled URDF missing: {urdf}')
    root = ET.parse(urdf).getroot()
    joints, parent = {}, {}
    for joint in root.findall('joint'):
        origin, axis, limit = joint.find('origin'), joint.find('axis'), joint.find('limit')
        joints[joint.get('name')] = {
            'type': joint.get('type'), 'parent': joint.find('parent').get('link'),
            'xyz': _floats(origin.get('xyz') if origin is not None else None),
            'rpy': _floats(origin.get('rpy') if origin is not None else None),
            'axis': _floats(axis.get('xyz') if axis is not None else None, '1 0 0'),
            'lower': float(limit.get('lower', 0.)) if limit is not None else 0.,
            'upper': float(limit.get('upper', 0.)) if limit is not None else 0.}
        parent[joint.find('child').get('link')] = joint.get('name')
    home = dict(zip([f'fr3_joint{i}' for i in range(1, 8)], robot['arm_home_rad']))
    base = np.asarray(robot['base_position_m'], dtype=float)
    cache = {}

    def world(link):
        if link not in cache:
            if link not in parent:
                matrix = np.eye(4)
                matrix[:3, 3] = base
            else:
                joint = joints[parent[link]]
                local = np.eye(4)
                local[:3, :3] = _rpy_matrix(*joint['rpy'])
                local[:3, 3] = joint['xyz']
                if joint['type'] in ('revolute', 'continuous'):
                    angle = home.get(parent[link], float(np.clip(0., joint['lower'], joint['upper'])))
                    turn = np.eye(4)
                    turn[:3, :3] = _axis_angle(joint['axis'], angle)
                    local = local @ turn
                matrix = world(joint['parent']) @ local
            cache[link] = matrix
        return cache[link]

    links = [link.get('name') for link in root.findall('link')]
    return root, {name: world(name) for name in links}


@functools.lru_cache(maxsize=None)
def _robot_collision_points():
    import trimesh
    root, world = _robot()
    points = {}
    for link in root.findall('link'):
        chunks = []
        for collision in link.findall('collision'):
            mesh = collision.find('geometry').find('mesh')
            if mesh is None:
                continue
            origin = collision.find('origin')
            local = np.eye(4)
            local[:3, :3] = _rpy_matrix(*_floats(origin.get('rpy') if origin is not None else None))
            local[:3, 3] = _floats(origin.get('xyz') if origin is not None else None)
            matrix = world[link.get('name')] @ local
            vertices = np.asarray(trimesh.load(_robot_urdf().parent / mesh.get('filename'), force='mesh').vertices)
            vertices = vertices * _floats(mesh.get('scale'), '1 1 1')
            chunks.append(vertices @ matrix[:3, :3].T + matrix[:3, 3])
        if chunks:
            points[link.get('name')] = np.vstack(chunks)
    return points


# --- camera helpers --------------------------------------------------------------

def _project(eye, target, point, focal_mm, width, height, h_aperture=20.955, v_aperture=15.2908):
    """Pinhole projection matching cameras.project_point for a look-at camera (z up)."""
    eye = np.asarray(eye, dtype=float)
    forward = np.asarray(target, dtype=float) - eye
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, [0., 0., 1.])
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    delta = np.asarray(point, dtype=float) - eye
    depth = float(delta @ forward)
    if depth <= 1e-6:
        return None
    return (width / 2. + (delta @ right) / depth * focal_mm / h_aperture * width,
            height / 2. - (delta @ up) / depth * focal_mm / v_aperture * height)


def _segment_hits_box(a, b, lo, hi):
    d = b - a
    t0, t1 = 0., 1.
    for k in range(3):
        if abs(d[k]) < 1e-12:
            if a[k] < lo[k] or a[k] > hi[k]:
                return False
            continue
        ta, tb = (lo[k] - a[k]) / d[k], (hi[k] - a[k]) / d[k]
        t0, t1 = max(t0, min(ta, tb)), min(t1, max(ta, tb))
        if t0 > t1:
            return False
    return True


def _segment_hits_hull(a, b, hull):
    """Does segment a-b pass through a convex hull? (clip against every face plane)"""
    normals = np.asarray(hull.face_normals)
    offsets = np.einsum('ij,ij->i', normals, np.asarray(hull.vertices)[hull.faces[:, 0]])
    t0, t1 = 0., 1.
    for da, db in zip(normals @ a - offsets, normals @ b - offsets):
        if da > 0 and db > 0:
            return False
        if da > 0 or db > 0:
            s = da / (da - db)
            t0, t1 = (max(t0, s), t1) if da > 0 else (t0, min(t1, s))
            if t0 > t1:
                return False
    return True


def _table_top_centre():
    table = SCENE['table']
    return np.array([table['position_m'][0], table['position_m'][1],
                     table['position_m'][2] + table['half_extents_m'][2]])


# =================================================================================
# 1. the family validates and pins a new library version
# =================================================================================

def test_scene_passes_the_family_validator():
    record = validate_family(SCENE)
    assert record['family'] == 'laboratory'
    assert record['objects'] == len(SCENE['objects']) == 27
    assert record['static_props'] == len(SCENE['static_props'])
    assert SCENE['schema_version'] == 'isaacsim-scene-family-v1'
    physics = SCENE['physics']
    assert physics['control_dt_s'] >= 2 * physics['dt_s'] > 0
    assert physics['gravity_m_s2'] == [0.0, 0.0, -9.81]


def test_scene_pins_library_v2_and_keeps_v1_as_evidence():
    assert lab.ASSET_VERSION == 'v2' and lab.ASSET_ROOT.endswith('/laboratory/v2')
    assert MANIFEST['asset_version'] == 'v2' and MANIFEST['family'] == 'laboratory'
    assert MANIFEST['schema_version'] == 'scene_family_asset_library_v1'
    assert MANIFEST['supersedes']['manifest_sha256'] == lab.V1_MANIFEST_SHA256
    for spec in SCENE['objects'] + SCENE['fixtures']:
        assert spec['asset_root'] == lab.ASSET_ROOT, spec['id']
    v1 = V1_ROOT / 'manifest.json'
    if v1.is_file():           # the superseded library must never be rewritten
        assert hashlib.sha256(v1.read_bytes()).hexdigest() == lab.V1_MANIFEST_SHA256


def test_every_object_and_fixture_asset_resolves_with_verified_hashes():
    for asset_id in DYNAMIC_ASSETS + FIXTURE_ASSETS:
        asset = _asset(asset_id)
        assert asset['visual_parts'] and asset['collision_parts'], asset_id
        for part in asset['visual_parts'] + asset['collision_parts']:
            if 'mesh_path' in part:
                assert Path(part['mesh_path']).is_file(), part['mesh_path']
        assert asset['dynamic'] is (asset_id in DYNAMIC_ASSETS), asset_id
        assert np.allclose(asset['size_m'], lab.DESIGN_SIZES[asset_id], atol=1e-6), asset_id


def test_rest_geometry_is_the_manifest_box():
    for spec in SCENE['objects']:
        assert spec['rest_geometry']['origin'] == 'centre'
        assert np.allclose(spec['rest_geometry']['size_m'], MANIFEST['assets'][spec['asset_id']]['size_m'],
                           atol=1e-6), spec['id']


# =================================================================================
# 2. colliders as scene.py builds them (the v1 flask had its offset applied twice)
# =================================================================================

def test_every_convex_piece_is_written_centred_and_placed_only_by_position_m():
    for asset_id in DYNAMIC_ASSETS + FIXTURE_ASSETS:
        for part in _asset(asset_id)['collision_parts']:
            if part.get('geometry') == 'convex_mesh':
                vertices, _ = load_obj_mesh(part['mesh_path'])
                centre = (vertices.min(0) + vertices.max(0)) / 2.
                assert np.abs(centre).max() <= 1e-7, (asset_id, part['label'], centre)


def test_rebuilt_collider_box_equals_the_visual_box():
    for asset_id in DYNAMIC_ASSETS + FIXTURE_ASSETS:
        asset = _asset(asset_id)
        visual = np.vstack([part_points(p) for p in asset['visual_parts']])
        collider = np.vstack([part_points(p) for p in asset['collision_parts']])
        error = np.abs(np.r_[visual.min(0) - collider.min(0), visual.max(0) - collider.max(0)]).max()
        assert error <= 1e-5, (asset_id, error)


def test_colliders_stay_inside_the_visible_outer_surface():
    """No collision vertex may stick out of the visible glass (v1: 35.085 vs 35.0 mm).

    0.03 mm covers the 96/48-gon visual facets and the sampled heel arc; the v1
    sectors stuck out by 0.085 mm (beaker) and 0.52 mm (flask neck).
    """
    checked = 0
    for asset_id in DYNAMIC_ASSETS:
        profile = MANIFEST['assets'][asset_id].get('outer_profile_m')
        if not profile:
            continue
        checked += 1
        for part in _asset(asset_id)['collision_parts']:
            outside = _outside_revolved_profile(profile, part_points(part))
            assert outside.max() <= 3e-5, (asset_id, part['label'], float(outside.max()))
    assert checked >= 8


def test_lowest_collider_point_rests_exactly_on_its_support():
    for spec in SCENE['objects']:
        points = np.vstack([w for _, w in _world_objects()[spec['id']]])
        support = _support_under(*spec['position_m'][:2])
        assert support is not None, spec['id']
        assert abs(points[:, 2].min() - support['top_z_m']) <= 1e-6, (spec['id'], points[:, 2].min(), support)
        lowest = points[points[:, 2] <= points[:, 2].min() + 1e-6]
        assert np.all(lowest[:, :2] >= np.asarray(support['xy_min']) - 1e-6), spec['id']
        assert np.all(lowest[:, :2] <= np.asarray(support['xy_max']) + 1e-6), spec['id']


def test_nothing_floats_or_sinks_per_the_family_validator():
    validate_support(SCENE['objects'], SUPPORTS, family='laboratory')


# =================================================================================
# 3. bore rule and blind cavities, by exact distance (v1: pieces inside the bore)
# =================================================================================

def test_declared_bores_stay_open_by_exact_polygon_to_axis_distance():
    bored = [asset_id for asset_id in DYNAMIC_ASSETS
             if any(p.get('bore') for p in _asset(asset_id)['collision_parts'])]
    assert bored == ['cork_ring'], bored
    for asset_id in bored:
        parts = _asset(asset_id)['collision_parts']
        radius = min(float(p['bore_radius_m']) for p in parts if p.get('bore'))
        assert len(parts) >= 3 and all(p.get('geometry') == 'convex_mesh' and p.get('sector') for p in parts)
        for part in parts:
            exact = axis_distance(part_points(part))
            assert exact >= radius - TOL, (asset_id, part['label'], exact, radius)
            assert float(part['clearance_radius_m']) <= exact + TOL, (asset_id, part['label'])


def test_closed_vessels_have_a_floor_an_open_cavity_and_no_bore_claim():
    for asset_id in VESSELS:
        record, parts = MANIFEST['assets'][asset_id], _asset(asset_id)['collision_parts']
        assert not any(p.get('bore') for p in parts), (asset_id, 'a blind vessel is not a through hole')
        floors = [p for p in parts if p.get('role') == 'floor']
        walls = [p for p in parts if p.get('role') == 'wall']
        assert len(floors) == 1 and len(walls) >= 16, asset_id
        floor = part_points(floors[0])
        bottom = min(part_points(p)[:, 2].min() for p in parts)
        assert axis_distance(floor) == 0.0, asset_id                   # the floor crosses the axis
        assert abs(floor[:, 2].min() - bottom) <= TOL, asset_id          # and is what the vessel stands on
        assert abs(floor[:, 2].max() - record['cavity_floor_z_m']) <= 1e-6, asset_id
        assert np.linalg.norm(floor[:, :2], axis=1).max() >= record['cavity_radius_m'], asset_id
        for part in walls:
            points = part_points(part)
            exact = axis_distance(points)
            assert exact >= record['cavity_radius_m'] - TOL, (asset_id, part['label'], exact)
            assert float(part['clearance_radius_m']) <= exact + TOL, (asset_id, part['label'])
            assert points[:, 2].min() >= record['cavity_floor_z_m'] - 1e-6, (asset_id, part['label'])
        # the collision mouth is real: no piece above the floor reaches into the cavity
        assert record['cavity_radius_m'] >= 0.008, asset_id


def test_the_flask_collider_is_a_hollow_cone_not_a_constant_bore_tube():
    """v1 was a thick tube with one 9.9 mm bore all the way down; the cavity must follow the cone."""
    parts = _asset('erlenmeyer_flask')['collision_parts']
    stages = [f'cone_{index:02d}' for index in range(4)] + ['neck']
    clearance = [min(axis_distance(part_points(p)) for p in parts if p['label'].startswith(stage + '_'))
                 for stage in stages]
    assert all(a > b for a, b in zip(clearance, clearance[1:])), clearance
    assert clearance[0] >= 2.0 * clearance[-1], clearance


# =================================================================================
# 4. thin parts (v1: 0.37 - 0.97 mm convex wall plates)
# =================================================================================

def test_every_dynamic_collision_piece_is_at_least_3_mm_across():
    for asset_id in DYNAMIC_ASSETS:
        for part in _asset(asset_id)['collision_parts']:
            width = piece_min_width(part)
            assert width >= MIN_WIDTH - 1e-9, (asset_id, part['label'], width)


def test_every_static_box_is_at_least_3_mm_thick():
    for prop, part_id, lo, hi, _ in static_boxes():
        assert (hi - lo).min() >= MIN_WIDTH - 1e-9, (prop, part_id, hi - lo)


# =================================================================================
# 5. penetration, racks and the pieces they hold
# =================================================================================

def test_no_object_penetrates_a_static_collider_or_another_object():
    boxes = static_boxes()
    for object_id, pieces in _world_objects().items():
        for part, points in pieces:
            for prop, part_id, lo, hi, _ in boxes:
                depth = box_penetration(points, lo, hi)
                assert depth <= 1e-5, (object_id, part['label'], prop, part_id, depth)
    rows = [(oid, np.vstack([w for _, w in pieces])) for oid, pieces in _world_objects().items()]
    for i, (id_a, a) in enumerate(rows):
        for id_b, b in rows[i + 1:]:
            gap = np.maximum(a.min(0) - b.max(0), b.min(0) - a.max(0))
            assert np.any(gap >= -1e-9), (id_a, id_b, gap)


def test_the_tube_rack_holds_every_tube_in_a_closed_hole():
    rack = _prop('tube_rack')
    plate = [p for p in rack['parts'] if p['id'].startswith('upper_')]
    z0, z1 = rack['holds']['plate_z_m']
    tubes = [spec for spec in SCENE['objects'] if spec['asset_id'] == 'test_tube']
    assert len(tubes) == 12
    radius = _asset('test_tube')['size_m'][0] / 2.
    for spec in tubes:
        points = np.vstack([w for _, w in _world_objects()[spec['id']]])
        bottom, height = points[:, 2].min(), np.ptp(points[:, 2])
        assert abs(bottom - rack['position_m'][2] - rack['half_extents_m'][2]) <= 1e-6, spec['id']
        assert bottom < z0 < z1 < bottom + height, spec['id']
        assert z0 - bottom >= 0.55 * height, spec['id']
        clearance = _hole_clearance(plate, *spec['position_m'][:2], radius)
        assert None not in clearance, (spec['id'], clearance, 'the hole is open on a side')
        assert 0. < max(clearance) <= 0.0015, (spec['id'], clearance)
        tilt = math.degrees(math.atan(max(clearance) / (z0 - bottom)))
        assert tilt <= 2.0, (spec['id'], tilt)


def test_the_tips_stand_point_down_with_the_collar_on_the_deck_around_closed_holes():
    rack = _prop('tip_rack')
    deck = [p for p in rack['parts'] if p['id'].startswith('deck_')]
    deck_bottom, deck_top = rack['holds']['deck_z_m']
    hole = rack['holds']['hole_half_m']
    floor_top = rack['position_m'][2] + rack['half_extents_m'][2]
    record = MANIFEST['assets']['pipette_tip']
    profile = record['outer_profile_m']
    assert profile[0][0] < profile[-1][0], 'the wide collar end must be up'
    collar_r = _asset('pipette_tip')['size_m'][0] / 2.
    tips = [spec for spec in SCENE['objects'] if spec['asset_id'] == 'pipette_tip']
    assert len(tips) == 8
    for spec in tips:
        pieces = dict((p['label'], w) for p, w in _world_objects()[spec['id']])
        points = np.vstack(list(pieces.values()))
        assert abs(points[:, 2].min() - floor_top) <= 1e-6, (spec['id'], 'point on the floor plate')
        assert abs(pieces['collar'][:, 2].min() - deck_top) <= 1e-6, (spec['id'], 'collar on the deck')
        body = pieces['body']
        at_deck = body[body[:, 2] >= deck_bottom - 1e-6]
        body_r = np.linalg.norm(at_deck[:, :2] - spec['position_m'][:2], axis=1).max()
        assert body_r <= hole - 0.0002, (spec['id'], body_r)
        sides = _hole_clearance(deck, *spec['position_m'][:2], 0.)
        assert None not in sides, (spec['id'], sides, 'slot open to the rack edge')
        assert max(sides) <= hole + 1e-9 < collar_r, (spec['id'], sides, 'the collar must bear on the deck')


def test_tips_have_a_real_tip_mass_for_their_length():
    tip = MANIFEST['assets']['pipette_tip']
    assert tip['size_m'][2] == pytest.approx(0.072) and tip['mass_kg'] == pytest.approx(0.0008)


def test_the_stirring_rod_rests_on_a_flat_face_and_its_collider_is_its_mesh():
    asset = _asset('stirring_rod')
    collider = part_points(asset['collision_parts'][0])
    visual = part_points(asset['visual_parts'][0])
    gaps = np.linalg.norm(collider[:, None, :] - visual[None, :, :], axis=2)
    assert gaps.min(1).max() <= TOL and gaps.min(0).max() <= TOL, 'collider differs from the mesh'
    lowest = collider[collider[:, 2] <= collider[:, 2].min() + TOL]
    assert len(lowest) >= 4 and np.ptp(lowest[:, 1]) >= 0.003, 'rests on an edge, not a face'


# =================================================================================
# 6. static colliders, supports, materials
# =================================================================================

def test_no_two_static_colliders_overlap_so_nothing_is_built_twice():
    boxes = static_boxes()
    for i, (prop_a, id_a, lo_a, hi_a, _) in enumerate(boxes):
        for prop_b, id_b, lo_b, hi_b, _ in boxes[i + 1:]:
            assert not np.all(_overlap(lo_a, hi_a, lo_b, hi_b) > 1e-6), (prop_a, id_a, prop_b, id_b)


def test_every_declared_support_is_a_real_collider_top():
    boxes = static_boxes()
    assert {s['id'] for s in SUPPORTS} == {'lab_bench', 'tube_rack', 'tip_rack'}
    for support in SUPPORTS:
        matches = [(prop, part) for prop, part, lo, hi, _ in boxes
                   if abs(hi[2] - support['top_z_m']) <= 1e-9
                   and np.allclose(lo[:2], support['xy_min'], atol=1e-9)
                   and np.allclose(hi[:2], support['xy_max'], atol=1e-9)]
        assert len(matches) == 1, (support['id'], matches)
    assert _prop('lab_bench')['support_of'] == 'table'


def test_every_material_used_is_defined_and_valid():
    defined = SCENE['materials']
    for name, spec in defined.items():
        validate_material(spec, name)
    used = {SCENE['table']['material']}
    used |= {plate['material'] for plate in SCENE.get('surfaces', []) or []}
    for prop in SCENE['static_props']:
        used.add(prop['material'])
        used |= {part['material'] for part in prop['parts']}
    missing = sorted(name for name in used if name not in defined)
    assert not missing, missing
    # room.add_room binds by surface KEY and add_pedestal by 'top'/'side'.
    for key in ('floor', 'wall', 'ceiling', 'baseboard', 'top', 'side'):
        assert key in defined, key
    assert SCENE['room']['surfaces'] == {key: key for key in ('floor', 'wall', 'ceiling', 'baseboard')}


def test_object_visuals_carry_their_own_roughness_and_metallic():
    """scene.add_task_asset falls back to roughness 0.35 / metallic 0.65 without these keys."""
    for asset_id in DYNAMIC_ASSETS + FIXTURE_ASSETS:
        for part in _asset(asset_id)['visual_parts']:
            assert 'roughness' in part and 'metallic' in part, (asset_id, part['label'])
            appearance = lab.APPEARANCE[part['appearance']]
            assert part['roughness'] == appearance['roughness'] and part['metallic'] == appearance['metallic']
            assert part['color'] == pytest.approx(appearance['color'])
    for asset_id in lab.GLASSWARE:
        for part in _asset(asset_id)['visual_parts']:
            assert part['metallic'] == 0.0 and part['roughness'] >= 0.2, (asset_id, part['label'])


# =================================================================================
# 7. room, openings, furniture
# =================================================================================

def test_the_room_shell_is_complete_and_closed():
    room = validate_room(SCENE['room'])
    assert room['enabled'] and len(room['size_m']) == 3 and min(room['size_m']) > 0
    assert room['wall_thickness_m'] > 0 and room['baseboard_m'] > 0
    assert room['collide'] == {'floor': True, 'walls': True, 'ceiling': False}
    assert set(room['surfaces']) == {'floor', 'wall', 'ceiling', 'baseboard'}
    assert len(room['ceiling_lights']) >= 2 and room['dome']['enabled']
    kinds = sorted(opening['kind'] for opening in room['openings'])
    assert kinds == ['door', 'window']
    for opening in room['openings']:
        assert opening['wall'] in WALL_FACES
    for face in WALL_FACES:                        # four walls, each still mostly solid
        wall = _wall_geometry(room, face)
        area = 4 * wall['half'][0] * wall['half'][1]
        cut = sum(np.prod(o['size_m']) for o in room['openings'] if o['wall'] == face)
        assert cut <= 0.35 * area, face


def test_openings_fit_their_walls_doors_reach_the_floor_windows_clear_the_corners():
    room = validate_room(SCENE['room'])
    height = room['ceiling_height_m']
    for opening in room['openings']:
        u, v = opening['center_m']
        su, sv = (s / 2. for s in opening['size_m'])
        half_u = room['size_m'][0] / 2. if opening['wall'] in ('front', 'back') else room['size_m'][1] / 2.
        z0, z1 = height / 2. + v - sv, height / 2. + v + sv
        if opening['kind'] == 'door':
            assert abs(z0) <= 1e-6 and 1.95 <= z1 <= 2.3, opening['id']
            assert -half_u + 0.15 <= u - su and u + su <= half_u - 0.15, opening['id']
        else:
            assert 0.8 <= z0 and z1 <= 2.5, (opening['id'], z0, z1)
            assert -half_u + 0.15 <= u - su and u + su <= half_u - 0.15, (opening['id'], 'too close to a corner')


def test_nothing_sits_in_a_wall_opening_except_the_window_glazing():
    fixtures = {fid: np.vstack([w for _, w in pieces]) for fid, pieces in _world_fixtures().items()}
    objects = {oid: np.vstack([w for _, w in pieces]) for oid, pieces in _world_objects().items()}
    for opening, lo, hi in _opening_boxes():
        for prop, part_id, b_lo, b_hi, part in static_boxes():
            if np.all(_overlap(lo, hi, b_lo, b_hi) > 1e-6):
                assert opening['kind'] == 'window' and part.get('glazing_of') == opening['id'], \
                    (opening['id'], prop, part_id)
        for name, points in {**fixtures, **objects}.items():
            assert not np.all(_overlap(lo, hi, points.min(0), points.max(0)) > 1e-6), (opening['id'], name)


def test_the_door_casing_frames_the_opening_without_a_gap():
    opening = next(o for o in SCENE['room']['openings'] if o['id'] == 'door')
    (x0, x1), (_, z1) = opening['world_along_m'], opening['world_z_m']
    parts = {p['id']: p for p in _prop('door_casing')['parts']}
    head = parts['head']
    assert abs(head['position_m'][2] - head['half_extents_m'][2] - z1) <= 1e-9
    assert abs(parts['stile_a']['position_m'][0] + parts['stile_a']['half_extents_m'][0] - x0) <= 1e-9
    assert abs(parts['stile_b']['position_m'][0] - parts['stile_b']['half_extents_m'][0] - x1) <= 1e-9


def test_the_room_is_furnished_as_a_laboratory():
    props = {prop['id'] for prop in SCENE['static_props']}
    assert {'lab_bench', 'bench_lip', 'reagent_riser', 'tube_rack', 'tip_rack', 'hot_plate_stirrer',
            'fume_hood', 'sink_unit', 'window_frame', 'door_casing', 'safety_shower', 'wall_shelf',
            'flammables_cabinet', 'first_aid_box', 'safety_signs'} <= props
    sink = {p['id'] for p in _prop('sink_unit')['parts']}
    assert {'eyewash_body', 'faucet_post', 'basin_bottom'} <= sink
    bench = {p['id'] for p in _prop('lab_bench')['parts']}
    assert 'carcass' in bench and sum(p.startswith('door_') for p in bench) == 4
    fixtures = {spec['asset_id'] for spec in SCENE['fixtures']}
    assert {'micropipette_stand', 'fire_extinguisher', 'reagent_bottle_amber_500ml'} <= fixtures
    ids = [spec['asset_id'] for spec in SCENE['objects']]
    assert ids.count('test_tube') == 12 and ids.count('pipette_tip') == 8
    assert {'beaker_50ml', 'beaker_150ml', 'beaker_250ml', 'erlenmeyer_flask', 'graduated_cylinder',
            'stirring_rod', 'cork_ring'} <= set(ids)


def test_fixtures_rest_on_a_real_surface_and_penetrate_nothing():
    boxes = static_boxes()
    for fixture_id, pieces in _world_fixtures().items():
        points = np.vstack([w for _, w in pieces])
        bottom = points[:, 2].min()
        x, y = points[points[:, 2] <= bottom + 1e-6][:, :2].mean(0)
        tops = [hi[2] for _, _, lo, hi, _ in boxes if lo[0] <= x <= hi[0] and lo[1] <= y <= hi[1]
                and hi[2] <= bottom + 1e-6]
        support = max(tops + [0.0])               # the floor is z = 0
        assert abs(bottom - support) <= 1e-6, (fixture_id, bottom, support)
        for part, w in pieces:
            for prop, part_id, lo, hi, _ in boxes:
                assert box_penetration(w, lo, hi) <= 1e-5, (fixture_id, part['label'], prop, part_id)


# =================================================================================
# 8. robot placement and reach
# =================================================================================

def test_the_pedestal_stands_on_the_floor_beside_the_bench():
    pedestal, table = SCENE['pedestal'], SCENE['table']
    bottom = pedestal['position_m'][2] - pedestal['half_extents_m'][2] - pedestal.get('skirt_height_m', 0.)
    assert abs(bottom) <= 1e-9, 'the pedestal floats above the floor'
    top = pedestal['position_m'][2] + pedestal['half_extents_m'][2]
    assert SCENE['robot']['base_position_m'][2] == pytest.approx(top)
    gap = (np.abs(np.asarray(pedestal['position_m'][:2]) - table['position_m'][:2])
           - np.asarray(pedestal['half_extents_m'][:2]) - table['half_extents_m'][:2])
    assert np.any(gap >= 0.0)
    assert table['position_m'][0] - table['half_extents_m'][0] >= 0.25 - 1e-9


def test_every_object_is_inside_the_reach_envelope():
    base = np.asarray(SCENE['robot']['base_position_m'], dtype=float)
    for spec in SCENE['objects']:
        centre = np.asarray(spec['position_m'], dtype=float)
        assert 0.30 <= centre[0] <= 0.80 and -0.10 <= centre[1] <= 0.45, spec['id']
        assert np.linalg.norm(centre - base) <= 0.80, spec['id']


def test_the_robot_at_home_clears_every_prop_fixture_and_object():
    links = _robot_collision_points()
    assert len(links) >= 30
    objects = {oid: np.vstack([w for _, w in pieces]) for oid, pieces in _world_objects().items()}
    fixtures = {fid: np.vstack([w for _, w in pieces]) for fid, pieces in _world_fixtures().items()}
    for link, points in links.items():
        lo, hi = points.min(0), points.max(0)
        for prop, part_id, b_lo, b_hi, _ in static_boxes():
            if link == 'fr3_link0' and prop == 'pedestal':
                continue                           # the base plate stands on it
            assert np.max(np.maximum(b_lo - hi, lo - b_hi)) >= 0.02, (link, prop, part_id)
        for name, q in {**objects, **fixtures}.items():
            assert np.max(np.maximum(q.min(0) - hi, lo - q.max(0))) >= 0.05, (link, name)


# =================================================================================
# 9. cameras (v1: 2.34 m third-person, wrist camera looking at the ceiling)
# =================================================================================

def test_cameras_resolve():
    cameras = resolve_cameras(SCENE)
    assert set(cameras) == {'third_person', 'bench_view', 'wrist'}
    assert cameras['third_person']['kind'] == 'world' and cameras['wrist']['kind'] == 'link'


def test_world_cameras_are_inside_the_room_and_frame_the_bench_by_rule_9():
    room = validate_room(SCENE['room'])
    centre = _table_top_centre()
    table = SCENE['table']
    fixture_boxes = [(fid, *(lambda p: (p.min(0), p.max(0)))(np.vstack([w for _, w in pieces])))
                     for fid, pieces in _world_fixtures().items()]
    for camera in resolve_cameras(SCENE).values():
        if camera['kind'] != 'world':
            continue
        eye, target = np.asarray(camera['eye']), np.asarray(camera['target'])
        assert np.all(np.abs(eye[:2]) <= np.asarray(room['size_m'][:2]) / 2. - 0.10), camera['id']
        assert 0.10 <= eye[2] <= room['ceiling_height_m'] - 0.10, camera['id']
        for prop, part_id, lo, hi, _ in static_boxes():
            assert not (np.all(eye > lo - 0.10) and np.all(eye < hi + 0.10)), (camera['id'], prop, part_id)
        distance = float(np.linalg.norm(eye - centre))
        assert 1.5 <= distance <= 2.2, (camera['id'], distance)
        assert 0.6 <= eye[2] - centre[2] <= 1.4, (camera['id'], eye[2] - centre[2])
        assert (abs(target[0] - table['position_m'][0]) <= table['half_extents_m'][0] and
                abs(target[1] - table['position_m'][1]) <= table['half_extents_m'][1]), camera['id']
        width, height = camera['resolution']
        for object_id, pieces in _world_objects().items():
            points = np.vstack([w for _, w in pieces])
            top = np.r_[(points.min(0)[:2] + points.max(0)[:2]) / 2., points[:, 2].max() + 0.001]
            pixel = _project(eye, target, top, camera['focal_length_mm'], width, height)
            assert pixel is not None and 0.03 * width <= pixel[0] <= 0.97 * width \
                and 0.03 * height <= pixel[1] <= 0.97 * height, (camera['id'], object_id, pixel)
            for prop, part_id, lo, hi, _ in static_boxes():
                assert not _segment_hits_box(eye, top, lo, hi), (camera['id'], object_id, prop, part_id)
            for fid, lo, hi in fixture_boxes:
                assert not _segment_hits_box(eye, top, lo, hi), (camera['id'], object_id, fid)
        for sx in (-1, 1):
            for sy in (-1, 1):
                corner = [table['position_m'][0] + sx * table['half_extents_m'][0],
                          table['position_m'][1] + sy * table['half_extents_m'][1], centre[2]]
                pixel = _project(eye, target, corner, camera['focal_length_mm'], width, height)
                assert pixel is not None and 0 <= pixel[0] <= width and 0 <= pixel[1] <= height, \
                    (camera['id'], 'bench corner out of frame', corner, pixel)
    third = resolve_cameras(SCENE)['third_person']
    assert np.allclose(third['target'], centre, atol=1e-9), 'the third-person camera aims at the bench centre'


def test_the_robot_at_home_hides_no_object_from_the_world_cameras():
    hulls = {name: _hull(points) for name, points in _robot_collision_points().items()}
    for camera in resolve_cameras(SCENE).values():
        if camera['kind'] != 'world':
            continue
        eye = np.asarray(camera['eye'], dtype=float)
        for object_id, pieces in _world_objects().items():
            points = np.vstack([w for _, w in pieces])
            top = np.r_[(points.min(0)[:2] + points.max(0)[:2]) / 2., points[:, 2].max() + 0.001]
            hidden = [name for name, hull in hulls.items() if _segment_hits_hull(eye, top, hull)]
            assert not hidden, (camera['id'], object_id, hidden)


def test_the_wrist_camera_looks_at_the_bench_from_the_home_pose():
    """FK at the home pose: USD cameras look along local -Z; v1's camera looked straight up."""
    camera = resolve_cameras(SCENE)['wrist']
    root, world = _robot()
    assert camera['mount_link'] in world, 'mount link missing from the assembled URDF'
    assert camera['aim_source'] == 'declared'
    aim = np.asarray(camera['aim_target_m'], dtype=float)
    table = SCENE['table']
    assert abs(aim[2] - _table_top_centre()[2]) <= 1e-9
    assert np.all(np.abs(aim[:2] - table['position_m'][:2]) <= np.asarray(table['half_extents_m'][:2]))
    link = world[camera['mount_link']]
    mount = link[:3, :3] @ np.asarray(camera['offset_xyz_m']) + link[:3, 3]
    eye = mount + (aim - mount) / np.linalg.norm(aim - mount) * camera['standoff_m']
    view = (aim - eye) / np.linalg.norm(aim - eye)
    assert view[2] <= -0.7, ('aimed wrist camera must look down at the bench', view)
    assert math.degrees(math.acos(-view[2])) >= 5.0, 'look-at with up=z would be degenerate'
    assert eye[2] >= _table_top_centre()[2] + 0.25, eye
    # The un-aimed fallback (offset_rpy_rad) must also look along the fingers, i.e. down.
    r, p, y = camera['offset_rpy_rad']
    rotation = _axis_angle([1, 0, 0], r) @ _axis_angle([0, 1, 0], p) @ _axis_angle([0, 0, 1], y)
    fallback = link[:3, :3] @ rotation @ np.array([0., 0., -1.])
    assert fallback[2] <= -0.9, ('fallback wrist camera looks away from the bench', fallback)


# =================================================================================
# 10. masses and inertia
# =================================================================================

def test_dynamic_assets_have_mass_inertia_com_and_design_mass_notes():
    for asset_id in DYNAMIC_ASSETS:
        record = MANIFEST['assets'][asset_id]
        assert record['dynamic'] and record['mass_kg'] > 0, asset_id
        assert np.allclose(record['inertia_diag_kg_m2'], uniform_box_inertia(record['mass_kg'], record['size_m']),
                           rtol=1e-9), asset_id
        assert 'bounding-box approximation' in record['inertia_method']
        assert 'design value' in record['mass_source'].lower(), asset_id
        com = np.asarray(record['center_of_mass_m'])
        assert np.all(np.abs(com) <= np.asarray(record['size_m']) / 2.), asset_id
        assert np.allclose(com[:2], 0.0), asset_id
    for asset_id in FIXTURE_ASSETS:
        assert MANIFEST['assets'][asset_id]['dynamic'] is False


def test_masses_are_plausible_real_world_design_values():
    mass = {key: record['mass_kg'] for key, record in MANIFEST['assets'].items()}
    assert mass['test_tube'] == pytest.approx(0.012)           # brief: a test tube 0.012 kg
    assert mass['beaker_250ml'] == pytest.approx(0.10)         # brief: a beaker 0.10 kg
    assert mass['beaker_50ml'] < mass['beaker_150ml'] < mass['beaker_250ml']
    assert 0.06 <= mass['erlenmeyer_flask'] <= 0.12
    assert 0.07 <= mass['graduated_cylinder'] <= 0.15
    assert 0.012 <= mass['stirring_rod'] <= 0.025
    assert 0.008 <= mass['cork_ring'] <= 0.03
    assert 0.0006 <= mass['pipette_tip'] <= 0.0010


# =================================================================================
# 11. negative controls: the checks are not vacuous, and they catch the v1 defects
# =================================================================================

def test_a_broken_scene_is_rejected():
    with pytest.raises(FamilyError):
        validate_family({**copy.deepcopy(SCENE), 'schema_version': 'wrong'})
    floating = copy.deepcopy(SCENE)
    floating['objects'][0]['position_m'][2] += 0.05
    with pytest.raises(FamilyError):
        validate_family(floating)
    sealed = copy.deepcopy(SCENE)
    sealed['objects'].append(
        {'id': 'fake_ring', 'asset_id': 'cork_ring', 'asset_root': lab.ASSET_ROOT,
         'position_m': [0.6, 0.2, 0.7625], 'quaternion_wxyz': [1., 0., 0., 0.], 'friction': 0.5,
         'body': {'origin': 'centre', 'mass_kg': 0.015, 'size_m': [0.08, 0.08, 0.025],
                  'collision_parts': [{'geometry': 'box', 'half_extents_m': [0.04, 0.04, 0.0125],
                                       'bore': True, 'bore_radius_m': 0.025}]},
         'rest_geometry': {'size_m': [0.08, 0.08, 0.025], 'origin': 'centre'}})
    with pytest.raises(FamilyError):
        validate_family(sealed)


def test_the_collider_rebuild_catches_a_doubly_applied_offset():
    part = dict(next(p for p in _asset('beaker_50ml')['collision_parts'] if p['label'] == 'wall_00'))
    honest = part_points(part)
    part['position_m'] = [0., 0., part['position_m'][2] + 0.02]     # shift the placement again
    assert part_points(part)[:, 2].max() - honest[:, 2].max() == pytest.approx(0.02)


def test_the_checks_detect_the_slot_duplicate_and_thin_plate_defects():
    # remove the rib beside the first tube: its hole becomes a slot and the check must say so
    rack = _prop('tube_rack')
    plate = [p for p in rack['parts'] if p['id'].startswith('upper_')]
    x, y = rack['holds']['slots_xy_m'][0]
    radius = _asset('test_tube')['size_m'][0] / 2.
    assert None not in _hole_clearance(plate, x, y, radius)
    rib = min((p for p in plate if p['id'].startswith('upper_rib') and p['position_m'][0] > x),
              key=lambda p: p['position_m'][0] - x)
    opened = _hole_clearance([p for p in plate if p is not rib], x, y, radius)
    assert opened[0] is None or opened[0] > 0.0015, opened
    # identical boxes (the v1 bench top was built three times) are flagged as an overlap
    lo, hi = np.zeros(3), np.ones(3)
    assert np.all(_overlap(lo, hi, lo, hi) > 1e-6)
    # a thin convex plate is measured as thin, by both width routines
    thin = np.array([[x, y, z] for x in (0, 0.05) for y in (0, 0.05) for z in (0, 0.0004)])
    assert exact_min_width(thin) == pytest.approx(0.0004)
    assert inscribed_diameter(thin) <= 0.0004 + 1e-9


@pytest.mark.skipif(not (V1_ROOT / 'manifest.json').is_file(), reason='v1 evidence not present')
def test_the_same_checks_flag_every_v1_collider_defect():
    def v1(asset_id):
        return load_task_asset(asset_id, str(V1_ROOT), verify_hashes=True)
    # the flask collider is 0.201 m tall around a 0.115 m visual
    flask = v1('erlenmeyer_flask')
    visual = np.vstack([part_points(p) for p in flask['visual_parts']])
    collider = np.vstack([part_points(p) for p in flask['collision_parts']])
    assert np.ptp(collider[:, 2]) - np.ptp(visual[:, 2]) > 0.08
    # bore intrusion: exact distance below the declared 33.6 mm bore
    beaker = v1('beaker_250ml')
    sectors = [p for p in beaker['collision_parts'] if p.get('bore')]
    declared = float(sectors[0]['bore_radius_m'])
    assert min(axis_distance(part_points(p)) for p in sectors) < declared - 5e-5
    # thin walls: well under 3 mm
    assert piece_min_width(sectors[0]) < 0.001
    # collider outside the visible glass (the v1 beaker is 70 mm across)
    radius = max(np.linalg.norm(part_points(p)[:, :2], axis=1).max() for p in sectors)
    assert radius > 0.035 + 5e-5
    profile = [[0., -0.0475], [0.035, -0.0475], [0.035, 0.0475]]
    assert max(_outside_revolved_profile(profile, part_points(p)).max() for p in sectors) > 5e-5
