"""CPU contract test for the living_room scene family.

Everything here is offline: it imports the family module, resolves every object
through the real task-asset loader against the written OBJ library, and applies
the scene-family physical rules. No GPU and no Isaac Sim. Where a check
depends on how the shared code turns the spec into a stage (collider transforms,
camera projection, robot pose), the test reproduces that step explicitly:

* collider transforms: ``room._box_collider`` composes ``Gf`` matrices in USD's
  row-vector convention; the test composes the same matrices with numpy;
* camera projection: RTX keeps the USD default horizontal aperture (20.955 mm)
  with square pixels, so the vertical field follows the image aspect;
* robot pose: forward kinematics on the fr3_sharpa isaacsim_6_1_v2 URDF that the
  robot USD was imported from (its sha256 is checked against USD_IMPORT.json).

The native ``scripts/validation/validate_scene_family.py`` run remains the
acceptance gate; these checks are what can be decided before it.
"""
from __future__ import annotations

import copy
import hashlib
import math
import sys
import types
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest

from bench.paths import load_json
from bench.simulation.assets import scene_parts as parts
from scenes.rooms import living_room as family
from bench.simulation.isaacsim_backend.cameras import MAX_CAMERAS, resolve_cameras
from bench.simulation.isaacsim_backend.scene_family import (FamilyError, support_surfaces,
                                                uniform_box_inertia, validate_family,
                                                validate_object, validate_support)
from bench.simulation.isaacsim_backend.room import WALL_FACES, _wall_geometry, _wall_quads, validate_room

SCENE = family.SCENE
ROOM = validate_room(SCENE['room'])
MANIFEST_PATH = Path(family.ASSET_ROOT) / 'manifest.json'
TABLE_TOP_Z = family.COFFEE_TABLE_TOP_Z
USD_HORIZONTAL_APERTURE_MM = 20.955          # UsdGeom.Camera schema default
PHYSX_DEFAULT_HULL_VERTEX_LIMIT = 64         # PhysxConvexHullCollisionAPI default

# The first revision's (v2) values, kept to show each check discriminates.
V2_CEILING_LIGHT_POWER = 2 * 0.70 * 0.70 * 9000.0
V2_DOME_INTENSITY = 350.0
V2_THIRD_PERSON = {'id': 'third_person', 'kind': 'world', 'eye': [1.28, -0.78, 1.50],
                   'target': [0.52, 0.11, 0.42], 'focal_length_mm': 24.0,
                   'resolution': [1280, 720]}


@pytest.fixture(scope='module')
def manifest():
    """The pinned library's manifest. A clean checkout builds the pinned version
    only when it is genuinely absent; ``write_library`` never overwrites one."""
    if not MANIFEST_PATH.is_file():
        family.build_library()
    assert MANIFEST_PATH.is_file(), f'the authored library could not be written: {MANIFEST_PATH}'
    return load_json(MANIFEST_PATH)


@pytest.fixture(scope='module')
def loaded(manifest):
    """``{asset_id: asset}`` resolved through the real catalogue loader."""
    from bench.simulation.isaacsim_backend.asset_catalog import load_task_asset
    return {asset_id: load_task_asset(asset_id, family.ASSET_ROOT, verify_hashes=True)
            for asset_id in manifest['assets']}


def _prop(prop_id):
    return next(prop for prop in SCENE['static_props'] if prop['id'] == prop_id)


def _part(prop_id, part_id):
    return next(part for part in _prop(prop_id)['parts'] if part['id'] == part_id)


def _box_of(item):
    center = np.asarray(item['position_m'], dtype=float)
    half = np.asarray(item['half_extents_m'], dtype=float)
    return center - half, center + half


def _quat_matrix(wxyz):
    w, x, y, z = np.asarray(wxyz, dtype=float) / np.linalg.norm(wxyz)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def _footprint(spec, record):
    """World xy bounds of an object's recorded AABB after its yaw (no centring assumed)."""
    lo, hi = np.asarray(record['aabb_m'][0]), np.asarray(record['aabb_m'][1])
    corners = np.array([[a, b, 0.0] for a in (lo[0], hi[0]) for b in (lo[1], hi[1])])
    world = corners @ _quat_matrix(spec['quaternion_wxyz']).T + np.asarray(spec['position_m'])
    return world[:, :2].min(axis=0), world[:, :2].max(axis=0)


def _authored_static_boxes():
    """Every static box collider scene.add_structured_scene authors, by USD path."""
    boxes = {}
    for prop in SCENE['static_props']:
        for part in prop['parts']:
            boxes[f'/World/Props/{prop["id"]}/{part["id"]}'] = _box_of(part)
    boxes['/World/Props/work_table/top'] = _box_of(SCENE['table'])
    for surface in SCENE.get('surfaces', []) or []:
        boxes[f'/World/Props/{surface["id"]}/plate'] = _box_of(surface)
    pedestal = SCENE['pedestal']
    lo, hi = _box_of(pedestal)
    boxes['/World/Room/RobotPedestal'] = (lo, hi)
    skirt = float(pedestal.get('skirt_height_m', 0.0))
    if skirt > 0:
        boxes['/World/Room/RobotPedestalSkirt'] = (np.array([lo[0], lo[1], lo[2] - skirt]),
                                                   np.array([hi[0], hi[1], lo[2]]))
    return boxes


# --- robot forward kinematics -------------------------------------------------

def _urdf_path():
    robot = SCENE['robot']
    return (Path(robot['assets_root']) / 'robots/assemblies/fr3_sharpa' / robot['assembly_version']
            / 'fr3_sharpa.urdf')


def _rpy(r, p, y):
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


def _axis_angle(axis, angle):
    x, y, z = np.asarray(axis, dtype=float) / np.linalg.norm(axis)
    c, s = math.cos(angle), math.sin(angle)
    t = 1.0 - c
    return np.array([[c + x * x * t, x * y * t - z * s, x * z * t + y * s],
                     [y * x * t + z * s, c + y * y * t, y * z * t - x * s],
                     [z * x * t - y * s, z * y * t + x * s, c + z * z * t]])


@pytest.fixture(scope='module')
def urdf():
    path = _urdf_path()
    root = ET.parse(path).getroot()
    joints = {}
    for joint in root.findall('joint'):
        origin = joint.find('origin')
        limit = joint.find('limit')
        axis = joint.find('axis')
        joints[joint.get('name')] = {
            'type': joint.get('type'), 'parent': joint.find('parent').get('link'),
            'child': joint.find('child').get('link'),
            'xyz': [float(v) for v in (origin.get('xyz') if origin is not None else '0 0 0').split()],
            'rpy': [float(v) for v in (origin.get('rpy') if origin is not None else '0 0 0').split()],
            'axis': [float(v) for v in axis.get('xyz').split()] if axis is not None else None,
            'limit': (float(limit.get('lower')), float(limit.get('upper'))) if limit is not None else None}
    return {'path': path, 'joints': joints}


def _fk(urdf, arm_q, base=None):
    """World frames of every link for the seven arm angles (hand joints at 0)."""
    q = {f'fr3_joint{i + 1}': float(v) for i, v in enumerate(arm_q)}
    children = {}
    for name, joint in urdf['joints'].items():
        children.setdefault(joint['parent'], []).append((name, joint))
    frames = {'base': np.eye(4)}
    frames['base'][:3, 3] = SCENE['robot']['base_position_m'] if base is None else base
    stack = ['base']
    while stack:
        link = stack.pop()
        for name, joint in children.get(link, []):
            local = np.eye(4)
            local[:3, :3] = _rpy(*joint['rpy'])
            local[:3, 3] = joint['xyz']
            if joint['type'] == 'revolute':
                rotate = np.eye(4)
                rotate[:3, :3] = _axis_angle(joint['axis'], q.get(name, 0.0))
                local = local @ rotate
            frames[joint['child']] = frames[link] @ local
            stack.append(joint['child'])
    return frames


ARM_CHAIN = ('fr3_link0', 'fr3_link1', 'fr3_link2', 'fr3_link3', 'fr3_link4', 'fr3_link5',
             'fr3_link6', 'fr3_link7', 'fr3_link8')


def _robot_capsules(frames, *, hand=True):
    """Conservative capsules for occlusion: the arm chain and the hand."""
    points = [frames[name][:3, 3] for name in ARM_CHAIN + ('right_hand_C_MC',)]
    capsules = [(points[i], points[i + 1], 0.07) for i in range(len(points) - 1)]
    if hand:
        capsules.append((frames['right_hand_C_MC'][:3, 3], frames['right_middle_fingertip'][:3, 3], 0.06))
    return capsules


def _segment_distance(p0, p1, q0, q1):
    """Closest distance between segments p0p1 and q0q1 (Ericson 5.1.9)."""
    d1, d2, r = p1 - p0, q1 - q0, p0 - q0
    a, e, f = float(d1 @ d1), float(d2 @ d2), float(d2 @ r)
    if a <= 1e-12 and e <= 1e-12:
        return float(np.linalg.norm(r)), 0.0
    if a <= 1e-12:
        s, t = 0.0, min(max(f / e, 0.0), 1.0)
    else:
        c = float(d1 @ r)
        if e <= 1e-12:
            s, t = min(max(-c / a, 0.0), 1.0), 0.0
        else:
            b = float(d1 @ d2)
            denom = a * e - b * b
            s = min(max((b * f - c * e) / denom, 0.0), 1.0) if denom > 1e-12 else 0.0
            t = (b * s + f) / e
            if t < 0.0:
                s, t = min(max(-c / a, 0.0), 1.0), 0.0
            elif t > 1.0:
                s, t = min(max((b - c) / a, 0.0), 1.0), 1.0
    return float(np.linalg.norm(p0 + d1 * s - (q0 + d2 * t))), s


def _box_distance_to_segment(lo, hi, p0, p1, samples=64):
    """Minimum distance from a segment to an axis-aligned box (sampled)."""
    points = p0 + (p1 - p0) * np.linspace(0.0, 1.0, samples)[:, None]
    clamped = np.minimum(np.maximum(points, lo), hi)
    return float(np.min(np.linalg.norm(points - clamped, axis=1)))


# --- camera projection and occlusion -----------------------------------------

def _project(spec, points):
    eye = np.asarray(spec['eye'], dtype=float)
    forward = np.asarray(spec['target'], dtype=float) - eye
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, [0.0, 0.0, 1.0])
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    width, height = spec['resolution']
    pixels = spec['focal_length_mm'] / USD_HORIZONTAL_APERTURE_MM * width
    delta = np.atleast_2d(np.asarray(points, dtype=float)) - eye
    depth = delta @ forward
    with np.errstate(divide='ignore', invalid='ignore'):
        u = width / 2.0 + pixels * (delta @ right) / depth
        v = height / 2.0 - pixels * (delta @ up) / depth
    inside = (depth > 1e-6) & (u >= 0) & (u < width) & (v >= 0) & (v < height)
    return u, v, depth, inside


def _segment_hits_box(p0, p1, lo, hi):
    d = p1 - p0
    t0, t1 = 0.0, 1.0
    for axis in range(3):
        if abs(d[axis]) < 1e-12:
            if p0[axis] < lo[axis] or p0[axis] > hi[axis]:
                return False
            continue
        ta, tb = (lo[axis] - p0[axis]) / d[axis], (hi[axis] - p0[axis]) / d[axis]
        t0, t1 = max(t0, min(ta, tb)), min(t1, max(ta, tb))
        if t0 > t1:
            return False
    return t0 < 1.0 - 1e-6 and t1 > 1e-6


def _occluder(spec, point, *, own=(), capsules=()):
    """The first static box or robot capsule on the sightline to ``point``."""
    eye = np.asarray(spec['eye'], dtype=float)
    point = np.asarray(point, dtype=float)
    for path, (lo, hi) in _authored_static_boxes().items():
        if any(f'/{name}/' in path + '/' or path.endswith('/' + name) for name in own):
            continue
        if _segment_hits_box(eye, point, lo, hi):
            return path
    for q0, q1, radius in capsules:
        distance, s = _segment_distance(eye, point, q0, q1)
        if distance < radius and s < 1.0 - 1e-3:
            return 'robot'
    return None


def _face_point(prop_id, part_id, axis, sign, offset=0.001):
    lo, hi = _box_of(_part(prop_id, part_id))
    point = (lo + hi) / 2.0
    point[axis] = (hi if sign > 0 else lo)[axis] + sign * offset
    return point


def _top_point(prop_id, part_id):
    return _face_point(prop_id, part_id, 2, +1)


# --- 1. the family validates --------------------------------------------------

def test_validate_family_succeeds_and_reports_the_declared_counts():
    report = validate_family(SCENE)
    assert report['family'] == 'living_room'
    assert report['objects'] == len(SCENE['objects']) == 10
    assert report['static_props'] == len(SCENE['static_props']) == 14
    assert report['room']['enabled'] is True
    assert [s['id'] for s in report['supports']] == [family.SUPPORT_ID]
    assert report['supports'][0]['top_z_m'] == pytest.approx(TABLE_TOP_Z, abs=1e-12)


def test_schema_version_and_family_are_pinned():
    assert SCENE['schema_version'] == 'isaacsim-scene-family-v1'
    assert SCENE['family'] == family.FAMILY == 'living_room'
    assert SCENE['title_zh'] and SCENE['description_zh']


def test_the_scene_is_not_mutated_by_validation():
    clone = copy.deepcopy(SCENE)
    validate_family(SCENE)
    assert SCENE == clone, 'validate_family must not mutate the scene it checks'


def test_physics_block_matches_the_bench_cadence():
    physics = SCENE['physics']
    assert physics['dt_s'] == 0.0005 and physics['control_dt_s'] == 0.02
    assert abs(physics['control_dt_s'] / physics['dt_s'] - 40) < 1e-9
    assert physics['gravity_m_s2'] == [0.0, 0.0, -9.81]


# --- 2. the room is a closed solid shell --------------------------------------

def test_room_has_floor_walls_ceiling_lights_and_a_dome():
    assert set(ROOM['surfaces']) == {'floor', 'wall', 'ceiling', 'baseboard'}
    assert ROOM['ceiling_height_m'] == pytest.approx(ROOM['size_m'][2])
    assert ROOM['baseboard_m'] > 0
    assert len(ROOM['ceiling_lights']) >= 3
    assert ROOM['dome']['enabled'] is True
    assert ROOM['dome']['hdri'] is None and ROOM['dome']['hdri_visible'] is False
    for face in WALL_FACES:                          # four drawn walls, cut around openings
        assert _wall_quads(_wall_geometry(ROOM, face),
                           [o for o in ROOM['openings'] if o['wall'] == face])


def test_add_room_is_not_asked_to_build_its_misplaced_wall_colliders():
    """room._box_collider composes a rotated wall collider with the basis as
    columns and after the translation, which lands every wall slab away from its
    wall (one inside arm reach). This family therefore builds no room wall
    collider and closes the shell with its own world-aligned ``wall_solids``."""
    assert SCENE['room']['collide'] == {'floor': True, 'walls': False, 'ceiling': False}
    walls = _prop('wall_solids')
    assert walls['role'] == 'room_shell' and walls['parts']


def _drawn_quads():
    quads = []
    for face in WALL_FACES:
        wall = _wall_geometry(ROOM, face)
        for index, quad in enumerate(_wall_quads(wall, [o for o in ROOM['openings'] if o['wall'] == face])):
            lo_u, hi_u, lo_v, hi_v = quad
            corners = np.array([wall['center'] + u * wall['u_axis'] + v * wall['v_axis']
                                for u, v in ((lo_u, lo_v), (hi_u, lo_v), (hi_u, hi_v), (lo_u, hi_v))])
            quads.append({'face': face, 'index': index, 'wall': wall, 'corners': corners})
    return quads


def test_every_drawn_wall_quad_is_backed_by_a_solid_box_just_behind_it():
    thickness, gap = ROOM['wall_thickness_m'], family.WALL_SOLID_GAP_M
    assert 0.0 < gap <= 0.001 + 1e-12, 'the solid must meet the robot within 1 mm of the drawn wall'
    solids = {part['id']: _box_of(part) for part in _prop('wall_solids')['parts']}
    quads = _drawn_quads()
    assert len(solids) == len(quads) == 9
    for quad in quads:
        lo, hi = solids[f'{quad["face"]}_{quad["index"]:02d}']
        normal = np.asarray(quad['wall']['normal'], dtype=float)
        axis = int(np.argmax(np.abs(normal)))
        plane = float(quad['corners'][0][axis])
        inner = hi[axis] if normal[axis] < 0 else lo[axis]
        outer = lo[axis] if normal[axis] < 0 else hi[axis]
        assert abs(inner - (plane + normal[axis] * gap)) < 1e-9, f'{quad["face"]} inner face'
        assert abs(outer - (plane + normal[axis] * (gap + thickness))) < 1e-9, f'{quad["face"]} outer face'
        # The solid covers the drawn quad's whole in-plane extent.
        qlo, qhi = quad['corners'].min(axis=0), quad['corners'].max(axis=0)
        for other in range(3):
            if other != axis:
                assert lo[other] <= qlo[other] + 1e-9 and hi[other] >= qhi[other] - 1e-9
        # Probe through the wall at 49 points: the solid spans 1 mm to 121 mm out.
        c0, c1, c3 = quad['corners'][0], quad['corners'][1], quad['corners'][3]
        for s in np.linspace(0.02, 0.98, 7):
            for t in np.linspace(0.02, 0.98, 7):
                for depth in (gap + 1e-4, gap + thickness / 2.0, gap + thickness - 1e-4):
                    probe = c0 + s * (c1 - c0) + t * (c3 - c0) + normal * depth
                    assert np.all(probe >= lo) and np.all(probe <= hi), (quad['face'], probe)


def test_no_wall_solid_intrudes_into_the_room_interior():
    origin, size = np.asarray(ROOM['origin_m']), np.asarray(ROOM['size_m'])
    inner_lo = np.array([origin[0] - size[0] / 2, origin[1] - size[1] / 2, origin[2]])
    inner_hi = np.array([origin[0] + size[0] / 2, origin[1] + size[1] / 2, origin[2] + ROOM['ceiling_height_m']])
    for part in _prop('wall_solids')['parts']:
        lo, hi = _box_of(part)
        overlap = np.minimum(hi, inner_hi) - np.maximum(lo, inner_lo)
        assert not np.all(overlap > 1e-9), f'{part["id"]} intrudes {overlap} into the room'


def test_the_wall_solids_close_the_corners():
    solids = [_box_of(part) for part in _prop('wall_solids')['parts']]
    reach = ROOM['size_m'][0] / 2.0 + family.WALL_SOLID_GAP_M + ROOM['wall_thickness_m']
    xs = [lo[0] for lo, _ in solids] + [hi[0] for _, hi in solids]
    assert min(xs) == pytest.approx(-reach) and max(xs) == pytest.approx(reach)
    # Each outer corner column is covered by a front/back solid.
    for sx in (-1, 1):
        for sy in (-1, 1):
            corner = np.array([sx * (reach - 0.01), sy * (ROOM['size_m'][1] / 2 + 0.05), 1.0])
            assert any(np.all(corner >= lo) and np.all(corner <= hi) for lo, hi in solids), corner


def _gf_scale(half):
    matrix = np.eye(4)
    matrix[0, 0], matrix[1, 1], matrix[2, 2] = half
    return matrix


def _gf_translate(center):
    matrix = np.eye(4)
    matrix[3, :3] = center                      # Gf row-vector convention: translation row
    return matrix


def _gf_cube_bounds(matrix):
    """World AABB of the size-2 cube ``_box_collider`` defines, under ``matrix``."""
    corners = np.array([[sx, sy, sz, 1.0] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
    world = corners @ matrix
    return world[:, :3].min(axis=0), world[:, :3].max(axis=0)


def test_every_static_collider_composes_exactly_where_it_is_declared():
    """The axis-aligned ``_box_collider`` path (``scale * translation``, row
    vectors) puts each collider on its declared box: every static-prop part (the
    wall solids, door, window and furniture), the work table, the pedestal and
    its skirt, and the room floor. Exactly one collider exists per box."""
    boxes = _authored_static_boxes()
    thickness = ROOM['wall_thickness_m']
    half_floor = (ROOM['size_m'][0] / 2 + thickness, ROOM['size_m'][1] / 2 + thickness, thickness / 2)
    floor_center = (ROOM['origin_m'][0], ROOM['origin_m'][1], ROOM['origin_m'][2] - thickness / 2)
    boxes['/World/Room'] = (np.subtract(floor_center, half_floor), np.add(floor_center, half_floor))
    for path, (lo, hi) in boxes.items():
        center, half = (lo + hi) / 2.0, (hi - lo) / 2.0
        got_lo, got_hi = _gf_cube_bounds(_gf_scale(half) @ _gf_translate(center))
        assert np.allclose(got_lo, lo, atol=1e-12) and np.allclose(got_hi, hi, atol=1e-12), path
    # No two authored colliders are the same box (the old duplicate table top).
    keys = [tuple(np.round(np.concatenate(box), 9)) for box in boxes.values()]
    assert len(keys) == len(set(keys)), 'two colliders are authored on the same box'


def test_the_wall_solids_equal_what_a_corrected_rotated_collider_would_author():
    """With the basis as *rows* and ``scale * rotation * translation`` (as
    room.box_collider_matrix composes it), each drawn quad's collider is the
    wall slab behind it; the family's wall solids are that slab moved 1 mm out
    (and lengthened at the corners on the front and back walls)."""
    thickness, gap = ROOM['wall_thickness_m'], family.WALL_SOLID_GAP_M
    solids = {part['id']: _box_of(part) for part in _prop('wall_solids')['parts']}
    for quad in _drawn_quads():
        wall, corners = quad['wall'], quad['corners']
        normal = np.asarray(wall['normal'], dtype=float)
        center = corners.mean(axis=0) + normal * (thickness / 2.0)
        half = (0.5 * np.linalg.norm(corners[1] - corners[0]), 0.5 * np.linalg.norm(corners[3] - corners[0]),
                thickness / 2.0)
        rotation = np.eye(4)
        rotation[0, :3], rotation[1, :3], rotation[2, :3] = wall['u_axis'], wall['v_axis'], normal
        lo, hi = _gf_cube_bounds(_gf_scale(half) @ rotation @ _gf_translate(center))
        solid_lo, solid_hi = solids[f'{quad["face"]}_{quad["index"]:02d}']
        shift = normal * gap
        axis = int(np.argmax(np.abs(normal)))
        for k in range(3):
            if k == axis:
                assert solid_lo[k] == pytest.approx(lo[k] + shift[k], abs=1e-9)
                assert solid_hi[k] == pytest.approx(hi[k] + shift[k], abs=1e-9)
            else:
                assert solid_lo[k] <= lo[k] + 1e-9 and solid_hi[k] >= hi[k] - 1e-9
                # Only a corner extension (gap + thickness) may lengthen a solid.
                assert lo[k] - solid_lo[k] <= gap + thickness + 1e-9
                assert solid_hi[k] - hi[k] <= gap + thickness + 1e-9


def test_built_stage_colliders_match_the_declared_boxes_with_real_openusd(monkeypatch):
    """Where OpenUSD is importable, build the room exactly as scene.py does (PhysX
    schema stubbed: it only sets contact offsets) and read every collider back."""
    pxr = pytest.importorskip('pxr')
    import importlib.util
    from pxr import Gf, Usd, UsdGeom, UsdPhysics
    if not hasattr(pxr, 'PhysxSchema'):
        class _Chain:
            def __getattr__(self, name):
                return lambda *args, **kwargs: self

        class _API:
            @staticmethod
            def Apply(*args, **kwargs):
                return _Chain()
        stub = types.ModuleType('pxr.PhysxSchema')
        for name in ('PhysxCollisionAPI', 'PhysxMaterialAPI', 'PhysxRigidBodyAPI',
                     'PhysxContactReportAPI', 'PhysxArticulationAPI', 'PhysxMimicJointAPI'):
            setattr(stub, name, _API)
        monkeypatch.setitem(sys.modules, 'pxr.PhysxSchema', stub)
        monkeypatch.setattr(pxr, 'PhysxSchema', stub, raising=False)
    import bench.simulation.isaacsim_backend as backend
    name = 'bench.simulation.isaacsim_backend._living_room_scene_probe'
    spec = importlib.util.spec_from_file_location(name, Path(backend.__file__).parent / 'scene.py')
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(stage, '/World')
    module.add_structured_scene(stage, SCENE, ROOM)
    cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    built = {}
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            matrix = cache.GetLocalToWorldTransform(prim)
            corners = np.array([list(matrix.Transform(Gf.Vec3d(sx, sy, sz)))
                                for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
            built[str(prim.GetPath())] = (corners.min(axis=0), corners.max(axis=0))
    assert not [path for path in built if '/World/Room/Wall_' in path]
    expected = {path + '/collider': box for path, box in _authored_static_boxes().items()}
    assert set(expected) <= set(built)
    for path, (lo, hi) in expected.items():
        assert np.allclose(built[path][0], lo, atol=1e-9) and np.allclose(built[path][1], hi, atol=1e-9), path


def test_the_door_and_window_openings_are_closed_by_solid_parts():
    door = next(o for o in ROOM['openings'] if o['kind'] == 'door')
    window = next(o for o in ROOM['openings'] if o['kind'] == 'window')
    # The door is person-sized and reaches the floor; the window sits above the sofa back.
    assert door['size_m'][1] >= 1.90
    assert abs(door['center_m'][1] - door['size_m'][1] / 2.0 + ROOM['ceiling_height_m'] / 2.0) < 1e-9
    assert window['center_m'][1] - window['size_m'][1] / 2.0 + ROOM['ceiling_height_m'] / 2.0 >= 0.9
    thickness = ROOM['wall_thickness_m']
    for opening, prop_id, part_id, clearance, min_depth in ((door, 'door', 'leaf', 0.006, 0.03),
                                                           (window, 'window', 'glass', 1e-9, 0.003)):
        wall = _wall_geometry(ROOM, opening['wall'])
        (cu, cv), (su, sv) = opening['center_m'], opening['size_m']
        corners = np.array([wall['center'] + u * wall['u_axis'] + v * wall['v_axis']
                            for u in (cu - su / 2, cu + su / 2) for v in (cv - sv / 2, cv + sv / 2)])
        olo, ohi = corners.min(axis=0), corners.max(axis=0)
        lo, hi = _box_of(_part(prop_id, part_id))
        normal = np.asarray(wall['normal'], dtype=float)
        axis = int(np.argmax(np.abs(normal)))
        plane = float(corners[0][axis])
        # Inside the wall's thickness, and filling the opening to within the clearance.
        depth_lo, depth_hi = sorted([plane, plane + normal[axis] * thickness])
        assert depth_lo <= lo[axis] and hi[axis] <= depth_hi
        assert hi[axis] - lo[axis] >= min_depth
        for k in range(3):
            if k != axis:
                assert olo[k] - 1e-9 <= lo[k] <= olo[k] + clearance
                assert ohi[k] - clearance <= hi[k] <= ohi[k] + 1e-9
    assert {'casing_a', 'casing_b', 'casing_head', 'handle'} <= {p['id'] for p in _prop('door')['parts']}
    assert {'frame_bottom', 'frame_top', 'sill', 'curtain_a', 'curtain_b'} <= {p['id'] for p in _prop('window')['parts']}


def test_the_lighting_is_brighter_than_the_dim_first_revision():
    """v2 rendered at rgb mean 57/255 under two 0.7 m panels at 9000 and a 350 dome."""
    power = sum(light['intensity'] * light['extent_m'][0] * light['extent_m'][1]
                for light in ROOM['ceiling_lights'])
    assert power >= 2.0 * V2_CEILING_LIGHT_POWER
    assert ROOM['dome']['intensity'] >= 1.5 * V2_DOME_INTENSITY
    half = np.asarray(ROOM['size_m']) / 2.0
    table_xy = np.asarray(SCENE['table']['position_m'][:2])
    for light in ROOM['ceiling_lights']:
        x, y, z = light['position_m']
        assert abs(x) + light['extent_m'][0] / 2 < half[0] and abs(y) + light['extent_m'][1] / 2 < half[1]
        assert TABLE_TOP_Z + 1.5 < z < ROOM['ceiling_height_m']
    assert min(np.linalg.norm(np.asarray(light['position_m'][:2]) - table_xy)
               for light in ROOM['ceiling_lights']) <= 0.30, 'a light hangs over the work table'


# --- 3. cameras ----------------------------------------------------------------

def test_every_camera_resolves_with_its_declared_id():
    cameras = resolve_cameras(SCENE)
    assert list(cameras) == ['third_person', 'table_view', 'wrist'] == [c['id'] for c in SCENE['cameras']]
    assert len(cameras) <= MAX_CAMERAS
    assert cameras['third_person']['kind'] == cameras['table_view']['kind'] == 'world'
    assert cameras['wrist']['kind'] == 'link' and cameras['wrist']['mount_link'] == 'right_hand_C_MC'
    assert family.HORIZONTAL_APERTURE_MM == USD_HORIZONTAL_APERTURE_MM


def test_the_third_person_camera_obeys_rule_9_inside_the_room():
    spec = resolve_cameras(SCENE)['third_person']
    eye, target = np.asarray(spec['eye']), np.asarray(spec['target'])
    origin, size = np.asarray(ROOM['origin_m']), np.asarray(ROOM['size_m'])
    assert np.all(np.abs(eye[:2] - origin[:2]) <= size[:2] / 2.0 - 0.05), 'eye must be inside the walls'
    assert 0.0 < eye[2] < ROOM['ceiling_height_m']
    assert 1.5 <= float(np.linalg.norm(eye - target)) <= 2.2
    assert 0.6 <= eye[2] - TABLE_TOP_Z <= 1.4
    table = SCENE['table']['position_m']
    assert np.allclose(target, [table[0], table[1], TABLE_TOP_Z]), 'aimed at the table-top centre'


def _third_person_targets(frames):
    table = np.asarray(SCENE['table']['position_m'])
    half = np.asarray(SCENE['table']['half_extents_m'])
    top = TABLE_TOP_Z + 0.001
    groups = {
        'table': [([table[0], table[1], top], ('work_table',))] + [
            ([table[0] + sx * (half[0] - 0.04), table[1] + sy * (half[1] - 0.04), top], ('work_table',))
            for sx in (-1, 1) for sy in (-1, 1)],
        'toys': [([o['position_m'][0], o['position_m'][1],
                   o['position_m'][2] + o['rest_geometry']['size_m'][2] / 2 + 0.001], ())
                 for o in SCENE['objects']],
        'tray': [(_top_point('toy_tray', 'floor'), ('toy_tray',))],
        'sofa': [(_top_point('sofa', name), ('sofa',)) for name in
                 ('seat_cushion_a', 'seat_cushion_b', 'back_cushion_a', 'back_cushion_b', 'arm_right')]
                + [(_face_point('sofa', 'base', 1, +1), ('sofa',))],
        'tv': [(_face_point('tv_console', 'tv_screen', 1, -1), ('tv_console',)),
               (_face_point('tv_console', 'door_b', 1, -1), ('tv_console',)),
               (_top_point('tv_console', 'carcass'), ('tv_console',))],
        'door': [(_face_point('door', 'leaf', 0, +1), ('door',))],
        # The front face of the pedestal top, right under the robot base.
        'robot_base': [([SCENE['pedestal']['position_m'][0] + SCENE['pedestal']['half_extents_m'][0] + 0.001,
                         SCENE['pedestal']['position_m'][1], SCENE['pedestal']['position_m'][2]],
                        ('RobotPedestal',))],
    }
    context = {'window': (_face_point('window', 'glass', 1, +1), ('window',)),
               'floor_lamp': (_face_point('floor_lamp', 'shade', 0, +1), ('floor_lamp',)),
               'plant': (_face_point('plant', 'foliage_low', 0, +1), ('plant',)),
               'picture': (_face_point('wall_picture', 'canvas', 1, -1), ('wall_picture',)),
               'side_lamp': (_face_point('side_table', 'lamp_shade', 0, +1), ('side_table',)),
               'pouf': (_top_point('pouf', 'cushion'), ('pouf',)),
               'bookshelf': (_face_point('bookshelf', 'books_low', 0, +1), ('bookshelf',))}
    robot = {name: frames[name][:3, 3] for name in
             ('fr3_link0', 'fr3_link1', 'fr3_link3', 'fr3_link4', 'fr3_link5', 'right_hand_C_MC',
              'right_middle_fingertip')}
    return groups, context, robot


def _visible(spec, point, own, capsules):
    _, _, _, inside = _project(spec, [point])
    return bool(inside[0]) and _occluder(spec, point, own=own, capsules=capsules) is None


def _framing(spec, urdf):
    frames = _fk(urdf, SCENE['robot']['arm_home_rad'])
    capsules = _robot_capsules(frames)
    groups, context, robot = _third_person_targets(frames)
    seen = {name: [_visible(spec, point, own, capsules) for point, own in items]
            for name, items in groups.items()}
    seen_context = {name: _visible(spec, point, own, capsules) for name, (point, own) in context.items()}
    robot_in_frame = {name: bool(_project(spec, [point])[3][0])
                      and _occluder(spec, point, own=('RobotPedestal',)) is None
                      for name, point in robot.items()}
    return seen, seen_context, robot_in_frame


def test_the_third_person_view_frames_table_sofa_tv_and_robot_unoccluded(urdf):
    """The view must read as a living room: the table and every toy, the tray,
    the sofa, the TV console, the door and the robot base all in frame and on a
    clear sightline (static furniture and the robot's own links as occluders),
    and the whole arm at home inside the frame."""
    seen, context, robot = _framing(resolve_cameras(SCENE)['third_person'], urdf)
    assert seen['table'][0] and sum(seen['table']) >= 4, seen['table']
    assert all(seen['toys']), seen['toys']
    assert all(seen['tray'])
    assert sum(seen['sofa']) >= 5, seen['sofa']
    assert seen['tv'][0] and sum(seen['tv']) >= 2, seen['tv']
    assert all(seen['door']) and all(seen['robot_base'])
    assert all(robot.values()), robot
    assert sum(context.values()) >= 5, context


def test_the_framing_check_rejects_the_old_table_only_view(urdf):
    """Not vacuous: the v2 camera (1280 x 720, 24 mm, from beside the sofa)
    misses the sofa, the TV and the robot under the same check."""
    seen, _, robot = _framing(V2_THIRD_PERSON, urdf)
    assert sum(seen['sofa']) < 5 or not seen['tv'][0] or not all(robot.values())
    assert not all(robot.values()), 'the old view did not show the arm'


def test_the_table_view_frames_every_toy_and_the_tray(urdf):
    spec = resolve_cameras(SCENE)['table_view']
    frames = _fk(urdf, SCENE['robot']['arm_home_rad'])
    capsules = _robot_capsules(frames)
    groups, _, _ = _third_person_targets(frames)
    for point, own in groups['toys'] + groups['tray'] + groups['table'][:1]:
        assert _visible(spec, point, own, capsules), point
    assert 0.8 <= float(np.linalg.norm(np.subtract(spec['eye'], spec['target']))) <= 1.5


def test_the_wrist_camera_is_aimed_at_the_table_and_stood_off_the_hand():
    spec = resolve_cameras(SCENE)['wrist']
    assert spec['aim_source'] == 'default_work_table_top'
    table = SCENE['table']['position_m']
    assert spec['aim_target_m'] == pytest.approx([table[0], table[1], TABLE_TOP_Z])
    assert spec['standoff_m'] == pytest.approx(0.12)
    assert 0.0 < spec['offset_xyz_m'][2] <= 0.12 and spec['offset_rpy_rad'] == [0.0, 0.0, 0.0]


# --- 4. robot, pedestal and home pose -------------------------------------------

def test_the_fk_urdf_is_the_one_the_robot_usd_was_imported_from(urdf):
    record = load_json(urdf['path'].parent / 'USD_IMPORT.json')
    assert hashlib.sha256(urdf['path'].read_bytes()).hexdigest() == record['urdf_sha256']


def test_the_pedestal_reaches_the_floor_and_carries_the_base():
    pedestal = SCENE['pedestal']
    lo, hi = _box_of(pedestal)
    assert pedestal['skirt_height_m'] == pytest.approx(lo[2], abs=1e-12), 'skirt must reach z = 0'
    assert lo[2] - pedestal['skirt_height_m'] == pytest.approx(0.0, abs=1e-12)
    assert SCENE['robot']['base_position_m'][2] == pytest.approx(hi[2], abs=1e-12)
    assert pedestal['half_extents_m'] == [0.22, 0.22, 0.05] and pedestal['position_m'][:2] == [0.0, 0.0]
    # add_pedestal binds the materials named 'top' and 'side'.
    assert {'top', 'side'} <= set(SCENE['materials'])


def test_the_pedestal_does_not_overlap_the_table():
    pedestal, table = SCENE['pedestal'], SCENE['table']
    p_lo, p_hi = _box_of(pedestal)
    t_lo, _ = _box_of(table)
    assert t_lo[0] - p_hi[0] >= 0.02, 'the table must start clear of the pedestal (rule 7)'
    assert t_lo[0] >= 0.25


def test_the_home_pose_is_the_registry_right_arm_default(urdf):
    from bench.simulation.isaacsim_backend.robot_registry import HANDS
    registry = HANDS['dual_fr3_sharpa_w01']['arm_home_rad_by_side']['right']
    assert SCENE['robot']['arm_home_rad'] == list(registry) == list(family.ARM_HOME_RAD)
    for index, value in enumerate(SCENE['robot']['arm_home_rad'], start=1):
        lower, upper = urdf['joints'][f'fr3_joint{index}']['limit']
        assert lower < value < upper


def test_the_hand_starts_over_the_table_clear_of_all_furniture(urdf):
    frames = _fk(urdf, SCENE['robot']['arm_home_rad'])
    palm = frames['right_hand_C_MC'][:3, 3]
    t_lo, t_hi = _box_of(SCENE['table'])
    assert t_lo[0] <= palm[0] <= t_hi[0] and t_lo[1] <= palm[1] <= t_hi[1], 'palm starts over the table'
    assert TABLE_TOP_Z + 0.30 <= palm[2] <= 1.40
    assert frames['right_hand_C_MC'][2, 2] < -0.9, 'fingers point down at the toys'
    for q0, q1, radius in _robot_capsules(frames):
        for path, (lo, hi) in _authored_static_boxes().items():
            if 'RobotPedestal' in path:
                continue
            assert _box_distance_to_segment(lo, hi, q0, q1) > radius + 0.10, (path, q0, q1)


# IK witnesses, solved offline (scipy SLSQP on this URDF): the palm 20 mm above
# each object's top at its xy, fingers within 40 deg of straight down, joints
# 5 mrad inside their limits, nearest the home pose. The test re-checks each by
# forward kinematics, so a moved object or a changed robot invalidates it.
REACH_WITNESSES = {
    'block_cube_40_red_a': [-0.0301, 1.3787, -0.1279, -1.9281, 0.0908, 2.6069, 0.8],
    'block_brick_2x4_red_a': [0.0085, 1.4314, -0.1493, -1.614, -0.0074, 2.3914, 0.8],
    'block_cube_44_blue_a': [-0.0153, 1.536, -0.1388, -1.3008, -0.0025, 2.506, 0.8],
    'block_cube_48_yellow_a': [0.2326, 1.3423, -0.0517, -1.9458, 0.0348, 2.5896, 0.8],
    'plush_toy_a': [0.2199, 1.2035, -0.0776, -1.7889, -0.0056, 2.313, 0.8],
    'block_brick_2x4_blue_a': [0.1539, 1.5504, -0.0812, -1.2622, -0.0012, 2.5191, 0.8],
    'block_brick_2x4_green_a': [0.3669, 1.4088, -0.0129, -1.6429, -0.0007, 2.3777, 0.8],
    'block_cube_40_red_b': [0.3204, 1.5593, -0.0261, -1.2442, -0.0003, 2.5253, 0.8],
    'picture_book_a': [0.5814, 1.5588, 0.0628, -1.3724, 0.0016, 2.4805, 0.8],
    'ball_60_a': [0.4181, 1.5787, 0.0057, -1.1501, 0.0, 2.5593, 0.8],
    'toy_tray': [0.0369, 1.6994, -0.091, -0.7911, 0.0037, 2.6852, 0.8],
}


def _check_witness(urdf, q, target):
    for index, value in enumerate(q, start=1):
        lower, upper = urdf['joints'][f'fr3_joint{index}']['limit']
        assert lower < value < upper, f'fr3_joint{index}={value} outside [{lower}, {upper}]'
    frames = _fk(urdf, q)
    palm = frames['right_hand_C_MC']
    assert np.linalg.norm(palm[:3, 3] - target) <= 0.002, (palm[:3, 3], target)
    assert -palm[2, 2] >= math.cos(math.radians(45.0)), 'fingers within 45 deg of vertical'
    # The arm from the shoulder to the wrist (link1..link7; not the flange and hand,
    # which must come down to the toy) clears all furniture.
    capsules = _robot_capsules(frames, hand=False)
    for q0, q1, radius in capsules[1:7]:
        for path, (lo, hi) in _authored_static_boxes().items():
            assert _box_distance_to_segment(lo, hi, q0, q1) > radius, (path, q0, q1)


def test_the_palm_can_reach_down_to_every_toy(urdf):
    """The previous 0.40 m table and x up to 0.68 m left the palm >= 137 mm above
    the far toys (no enveloping grasp). Every toy now has an IK witness with the
    palm 20 mm above its top, inside joint limits, and the arm clear of furniture."""
    assert set(REACH_WITNESSES) == {o['id'] for o in SCENE['objects']} | {'toy_tray'}
    for spec in SCENE['objects']:
        x, y, z = spec['position_m']
        top = z + spec['rest_geometry']['size_m'][2] / 2.0
        _check_witness(urdf, REACH_WITNESSES[spec['id']], np.array([x, y, top + 0.02]))


def test_every_object_is_inside_the_reach_envelope_and_the_grasp_radius():
    for spec in SCENE['objects']:
        x, y, z = spec['position_m']
        assert family.REACH_ENVELOPE_X[0] <= x <= 0.60, f'{spec["id"]} x={x}'
        assert family.REACH_ENVELOPE_Y[0] <= y <= family.REACH_ENVELOPE_Y[1], f'{spec["id"]} y={y}'
        assert math.hypot(x, y) <= family.POWER_GRASP_RADIUS_M
        base = SCENE['robot']['base_position_m']
        assert float(np.linalg.norm([x - base[0], y - base[1], z - base[2]])) <= family.REACH_RADIUS_M


def test_the_toy_tray_is_a_reachable_receptacle_on_the_table(urdf, manifest):
    tray = _prop('toy_tray')
    assert tray['role'] == 'receptacle'
    lo = np.min([_box_of(p)[0] for p in tray['parts']], axis=0)
    hi = np.max([_box_of(p)[1] for p in tray['parts']], axis=0)
    t_lo, t_hi = _box_of(SCENE['table'])
    assert lo[2] == pytest.approx(TABLE_TOP_Z, abs=1e-12), 'the tray stands on the table top'
    assert np.all(lo[:2] >= t_lo[:2] + 0.02) and np.all(hi[:2] <= t_hi[:2] - 0.02)
    assert hi[2] - lo[2] <= 0.045, 'a shallow tray, not an obstacle'
    assert min(min(p['half_extents_m']) for p in tray['parts']) * 2 >= 0.003
    centre = (lo + hi) / 2.0
    _check_witness(urdf, REACH_WITNESSES['toy_tray'], np.array([centre[0], centre[1], hi[2] + 0.10]))
    for spec in SCENE['objects']:                       # every toy starts outside it
        f_lo, f_hi = _footprint(spec, manifest['assets'][spec['asset_id']])
        assert np.any(f_lo >= hi[:2] + 0.01) or np.any(f_hi <= lo[:2] - 0.01), spec['id']


# --- 5. assets resolve, are physically real and look like toys ----------------

def test_every_object_asset_resolves_through_the_real_loader(loaded):
    digest = hashlib.sha256(MANIFEST_PATH.read_bytes()).hexdigest()
    for spec in SCENE['objects']:
        assert spec['asset_id'] in loaded, f'{spec["id"]} does not resolve'
        assert Path(spec['asset_root']) == Path(family.ASSET_ROOT)
        asset = loaded[spec['asset_id']]
        assert Path(asset['manifest_path']) == MANIFEST_PATH
        assert asset['manifest_sha256'] == digest
        assert Path(asset['mesh_path']).is_file()


def test_every_asset_has_positive_mass_convex_collision_and_a_recorded_inertia(manifest):
    for asset_id, record in manifest['assets'].items():
        assert record['dynamic'] is True and record['mass_kg'] > 0
        assert record['collision_parts'], f'{asset_id} needs a collision part'
        for part in record['collision_parts']:
            if part.get('geometry') == 'box':
                assert all(v > 0 for v in part['half_extents_m'])
            elif part.get('geometry') == 'sphere':
                assert part['radius_m'] > 0
            else:
                assert part['geometry'] == 'convex_mesh' and part['triangles'] > 0
        assert all(v > 0 for v in record['inertia_diag_kg_m2'])
        assert 'approximation' in record['inertia_method'].lower()
        assert 'not measured' in record['inertia_method']
        expected = uniform_box_inertia(record['mass_kg'], record['size_m'])
        assert np.allclose(record['inertia_diag_kg_m2'], expected, rtol=1e-9)


def test_the_declared_masses_are_plausible_toddler_toy_values(manifest):
    expected = {'block_cube_40_red': (0.015, 0.045), 'block_cube_44_blue': (0.018, 0.055),
                'block_cube_48_yellow': (0.022, 0.065), 'block_brick_2x4_green': (0.028, 0.080),
                'block_brick_2x4_red': (0.028, 0.080), 'block_brick_2x4_blue': (0.028, 0.080),
                'plush_toy': (0.030, 0.120), 'ball_60': (0.015, 0.060), 'picture_book': (0.250, 0.600)}
    assert set(expected) == set(manifest['assets'])
    for asset_id, (low, high) in expected.items():
        assert low <= manifest['assets'][asset_id]['mass_kg'] <= high, asset_id
    for record in manifest['assets'].values():
        assert record['mass_basis'].startswith('Design value')
        assert record['provenance']['source_kind'] == 'original_procedural'
        assert record['provenance']['license'] == 'CC0-1.0'


def test_every_visual_part_records_a_finish_and_no_toy_renders_as_metal(manifest, loaded):
    """v2 recorded no finish, so add_task_asset fell back to metallic 0.65 and
    the plush, ball and book rendered as chrome. Every visual part must now carry
    its own roughness and metallic, and nothing in this family is metal."""
    for asset_id, record in manifest['assets'].items():
        for part in record['visual_parts']:
            assert isinstance(part.get('roughness'), float) and 0.0 <= part['roughness'] <= 1.0, (asset_id, part)
            assert isinstance(part.get('metallic'), float) and part['metallic'] == 0.0, (asset_id, part)
    for asset_id, asset in loaded.items():                  # what scene.py actually binds
        for part in asset['visual_parts']:
            assert part.get('metallic', 0.65) == 0.0 and part.get('roughness', 0.35) >= 0.2
    for asset_id in ('plush_toy', 'ball_60', 'picture_book'):
        assert all(p['metallic'] == 0.0 for p in manifest['assets'][asset_id]['visual_parts'])
    fabric = [p for p in manifest['assets']['plush_toy']['visual_parts'] if not p['label'].startswith(('eye', 'nose'))]
    assert fabric and all(p['roughness'] >= 0.9 for p in fabric), 'plush fabric is matte'
    assert all(p['roughness'] >= 0.5 for p in manifest['assets']['picture_book']['visual_parts']
               if p['label'] != 'spine')


def test_the_superseded_libraries_are_kept_and_explain_the_rebuild(manifest):
    base = Path(family.ASSET_LIBRARY_BASE)
    v1, v2 = (load_json(base / name / 'manifest.json') for name in ('v1', 'v2'))
    assert v1['asset_version'] == 'v1' and v2['asset_version'] == 'v2'
    assert not any('metallic' in part for record in v2['assets'].values() for part in record['visual_parts'])
    assert set(manifest['supersedes']) == {'v1', 'v2'}
    assert manifest['asset_version'] == family.ASSET_VERSION == 'v3'


def test_blocks_read_as_blocks_cubes_are_cubic_and_bricks_are_twice_as_long(manifest):
    assets = manifest['assets']
    for asset_id in ('block_cube_40_red', 'block_cube_44_blue', 'block_cube_48_yellow'):
        size = assets[asset_id]['size_m']
        nominal = round(size[0], 6)
        assert nominal in (0.040, 0.044, 0.048)
        assert size[0] == pytest.approx(size[1], abs=1e-9)
        assert size[2] == pytest.approx(nominal + family.STUD_HEIGHT_M, abs=1e-9)
        assert len(assets[asset_id]['visual_parts']) == 5
        assert [p['geometry'] for p in assets[asset_id]['collision_parts']] == ['box']
    for asset_id in ('block_brick_2x4_green', 'block_brick_2x4_red', 'block_brick_2x4_blue'):
        size = assets[asset_id]['size_m']
        assert size[0] == pytest.approx(2.0 * size[1], abs=1e-9)
        assert size[2] == pytest.approx(size[1] + family.STUD_HEIGHT_M, abs=1e-9)


def _obj_vertices(record_part):
    from bench.simulation.isaacsim_backend.asset_catalog import load_obj_mesh
    vertices, _ = load_obj_mesh(Path(family.ASSET_ROOT) / record_part['mesh_path'])
    return np.asarray(vertices, dtype=float)


def test_the_plush_stands_on_a_flat_base_not_an_apex(manifest):
    """v2's plush stood on one apex vertex (support area 0). The bear's hull now
    stands on a 12-gon inscribed in a 25 mm disc, its centre of mass well inside."""
    from scipy.spatial import ConvexHull
    record = manifest['assets']['plush_toy']
    assert record['geometry_class'] == 'plush_bear_flat_base'
    (hull_part,) = record['collision_parts']
    assert hull_part['geometry'] == 'convex_mesh'
    vertices = _obj_vertices(hull_part)
    base_z = vertices[:, 2].min()
    assert base_z == pytest.approx(record['aabb_m'][0][2], abs=1e-7), 'the hull bottom is the rest plane'
    base = vertices[np.abs(vertices[:, 2] - base_z) < 1e-6][:, :2]
    assert len(base) >= 12
    polygon = ConvexHull(base)
    assert polygon.volume >= math.pi * 0.020 ** 2, f'support area {polygon.volume * 1e4:.1f} cm2'
    com = np.asarray(record['center_of_mass_m'])
    inset = -(polygon.equations[:, :2] @ com[:2] + polygon.equations[:, 2])
    assert inset.min() >= 0.020, 'the centre of mass projects well inside the base'
    assert math.degrees(math.atan2(inset.min(), com[2] - base_z)) >= 20.0


def test_every_convex_collision_piece_is_simulated_as_authored(manifest):
    """PhysX cooks a convex hull to at most hullVertexLimit vertices (64 by
    default, scene.py sets no other); a piece within the limit is not re-cooked."""
    import trimesh
    for asset_id, record in manifest['assets'].items():
        for part in record['collision_parts']:
            if part['geometry'] != 'convex_mesh':
                continue
            vertices = _obj_vertices(part)
            assert len(vertices) <= PHYSX_DEFAULT_HULL_VERTEX_LIMIT, (asset_id, len(vertices))
            hull = trimesh.convex.convex_hull(vertices)
            assert len(hull.vertices) == len(vertices), f'{asset_id} piece is not its own hull'


def test_the_ball_rests_on_an_analytic_sphere_not_a_faceted_hull(manifest):
    pieces = manifest['assets']['ball_60']['collision_parts']
    assert [p['geometry'] for p in pieces] == ['sphere']
    assert pieces[0]['radius_m'] == pytest.approx(family.BALL_RADIUS_M, abs=1e-12)
    assert pieces[0]['position_m'] == [0.0, 0.0, 0.0]
    ball = manifest['assets']['ball_60']
    assert ball['center_of_mass_m'] == [0.0, 0.0, 0.0]
    for axis in range(3):
        assert ball['size_m'][axis] == pytest.approx(2.0 * family.BALL_RADIUS_M, abs=1e-9)


def test_every_asset_is_centred_so_the_centre_rest_convention_holds(manifest):
    for asset_id, record in manifest['assets'].items():
        lo, hi = np.asarray(record['aabb_m'][0]), np.asarray(record['aabb_m'][1])
        assert np.allclose(lo, -hi, atol=1e-9), f'{asset_id} bounds {lo}..{hi} are not centred'


def test_no_collision_piece_declares_or_seals_a_bore(manifest):
    for asset_id, record in manifest['assets'].items():
        for part in record['collision_parts']:
            assert part.get('bore') is not True, f'{asset_id} declares a bore; sectors would apply'


def test_a_declared_bore_with_an_intruding_piece_is_rejected():
    bored = {'id': 'ring', 'body': {'mass_kg': 0.01, 'size_m': [0.06, 0.06, 0.01], 'collision_parts': [
        {'geometry': 'cylinder', 'radius_m': 0.03, 'bore': True, 'bore_radius_m': 0.02}]}}
    validate_object(bored, family='living_room')
    sealed = {'id': 'ring', 'body': {'mass_kg': 0.01, 'size_m': [0.06, 0.06, 0.01], 'collision_parts': [
        {'geometry': 'cylinder', 'radius_m': 0.03, 'bore': True, 'bore_radius_m': 0.02},
        {'geometry': 'box', 'half_extents_m': [0.01, 0.03, 0.005]}]}}
    with pytest.raises(FamilyError, match='bore'):
        validate_object(sealed, family='living_room')


def test_thin_parts_are_thick_enough_for_a_half_millimetre_contact_offset(manifest):
    for asset_id, record in manifest['assets'].items():
        for part in record['collision_parts']:
            if part.get('geometry') == 'box':
                assert min(part['half_extents_m']) * 2.0 >= 0.003, f'{asset_id} collider < 3 mm'
    book = manifest['assets']['picture_book']
    assert [p['geometry'] for p in book['collision_parts']] == ['box']
    assert book['collision_parts'][0]['half_extents_m'] == pytest.approx([0.1, 0.1, 0.008])


def test_visual_geometry_is_never_used_as_collision(manifest):
    for asset_id, record in manifest['assets'].items():
        for part in record['visual_parts']:
            assert part['classification'] == 'visual_only'
            assert 'collision' not in Path(part['mesh_path']).parts
        for part in record['collision_parts']:
            assert part['classification'] == 'physics'
            if 'mesh_path' in part:
                assert 'collision' in Path(part['mesh_path']).parts
        combined = Path(family.ASSET_ROOT) / record['mesh_path']
        assert hashlib.sha256(combined.read_bytes()).hexdigest() == record['sha256']


def test_the_library_is_versioned_and_pins_a_real_directory(manifest):
    assert manifest['schema_version'] == parts.SCHEMA_VERSION
    assert manifest['asset_version'] == family.ASSET_VERSION == 'v3'
    assert manifest['units'] == 'metres' and manifest['quaternion_order'] == 'xyzw'
    assert Path(family.ASSET_ROOT).name == family.ASSET_VERSION
    assert manifest['license'] == 'CC0-1.0'
    assert all(spec['asset_root'] == family.ASSET_ROOT for spec in SCENE['objects'])


def test_build_library_refuses_to_overwrite_the_pinned_version(tmp_path):
    with pytest.raises(FileExistsError):
        parts.write_library(family.ASSET_ROOT, [], generator='test')
    record = parts.write_library(tmp_path / 'v3', family.assets(), generator='tests.test_room_living_room')
    assert Path(record['manifest']).is_file() and record['assets'] == len(family.assets())


def test_build_library_is_deterministic(tmp_path):
    """Two builds are byte-identical; tests/test_room_libraries.py checks that a
    build reproduces the pinned library."""
    first = family.build_library(tmp_path / 'a' / family.ASSET_VERSION)
    second = family.build_library(tmp_path / 'b' / family.ASSET_VERSION)
    assert first['manifest_sha256'] == second['manifest_sha256']


# --- 6. rest pose, stability and the single work surface ----------------------

def test_no_object_floats_or_sinks(manifest):
    supports = support_surfaces(SCENE)
    assert [s['id'] for s in supports] == [family.SUPPORT_ID]
    for spec in SCENE['objects']:
        record = manifest['assets'][spec['asset_id']]
        assert spec['rest_geometry'] == {'size_m': record['size_m'], 'origin': 'centre'}
        z_min = spec['position_m'][2] + record['aabb_m'][0][2]
        assert z_min == pytest.approx(TABLE_TOP_Z, abs=1e-9), spec['id']
    validate_support(SCENE['objects'], supports, family='living_room')


def _collision_points(record, spec):
    """World points of an object's collision geometry (box corners, sphere
    bottom, hull vertices) at its spawn pose."""
    rotation = _quat_matrix(spec['quaternion_wxyz'])
    position = np.asarray(spec['position_m'], dtype=float)
    points = []
    for part in record['collision_parts']:
        offset = np.asarray(part.get('position_m', [0, 0, 0]), dtype=float)
        if part['geometry'] == 'box':
            half = np.asarray(part['half_extents_m'])
            local = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]) * half + offset
        elif part['geometry'] == 'sphere':
            centre = rotation @ offset + position
            points.append(centre - [0.0, 0.0, part['radius_m']])
            continue
        else:
            local = _obj_vertices(part) + offset
        points.extend(local @ rotation.T + position)
    return np.asarray(points)


def test_the_collision_geometry_itself_rests_on_the_table_top(manifest):
    """Independent of rest_geometry: the lowest point of each object's actual
    collision pieces at its spawn pose lies on the table top, and its rotated
    footprint lies inside the table."""
    t_lo, t_hi = _box_of(SCENE['table'])
    for spec in SCENE['objects']:
        points = _collision_points(manifest['assets'][spec['asset_id']], spec)
        assert points[:, 2].min() == pytest.approx(TABLE_TOP_Z, abs=1e-6), spec['id']
        assert np.all(points[:, :2] >= t_lo[:2] + 0.02) and np.all(points[:, :2] <= t_hi[:2] - 0.02), spec['id']


def test_every_object_is_statically_stable_where_it_rests(manifest):
    """The centre of mass projects inside the contact polygon with a tipping
    angle of at least 20 deg (the ball rests on its contact point directly
    beneath its centre)."""
    from scipy.spatial import ConvexHull
    for spec in SCENE['objects']:
        record = manifest['assets'][spec['asset_id']]
        rotation = _quat_matrix(spec['quaternion_wxyz'])
        com = rotation @ np.asarray(record['center_of_mass_m']) + np.asarray(spec['position_m'])
        if [p['geometry'] for p in record['collision_parts']] == ['sphere']:
            centre = rotation @ np.asarray(record['collision_parts'][0]['position_m']) + spec['position_m']
            assert np.allclose(centre[:2], com[:2], atol=1e-9)
            continue
        points = _collision_points(record, spec)
        contact = points[np.abs(points[:, 2] - TABLE_TOP_Z) < 1e-6][:, :2]
        polygon = ConvexHull(contact)
        inset = -(polygon.equations[:, :2] @ com[:2] + polygon.equations[:, 2])
        assert inset.min() > 0, f'{spec["id"]} centre of mass is outside its base'
        angle = math.degrees(math.atan2(inset.min(), com[2] - TABLE_TOP_Z))
        assert angle >= 20.0, f'{spec["id"]} tips at {angle:.1f} deg'


def test_spawn_poses_are_upright_yaw_only():
    for spec in SCENE['objects']:
        w, x, y, z = spec['quaternion_wxyz']
        assert (x, y) == (0.0, 0.0) and w * w + z * z == pytest.approx(1.0)


def test_every_object_names_the_support_it_rests_on():
    declared = {s['id'] for s in support_surfaces(SCENE)}
    assert all(spec['support_surface'] in declared for spec in SCENE['objects'])


def test_the_work_surface_is_authored_once():
    """v2 authored the table top twice (``table`` and an identical ``surfaces``
    slab). The slab is now authored only by ``table``; the coffee_table prop
    declares it as the support without authoring a part on it."""
    assert SCENE['surfaces'] == []
    table_lo, table_hi = _box_of(SCENE['table'])
    prop = _prop(family.SUPPORT_ID)
    assert prop['support_declared_by'] == 'table'
    assert prop['position_m'] == SCENE['table']['position_m']
    assert prop['half_extents_m'] == SCENE['table']['half_extents_m']
    for part in prop['parts']:
        assert _box_of(part)[1][2] <= table_lo[2] + 1e-9, f'{part["id"]} is not below the top'
    same = [path for path, (lo, hi) in _authored_static_boxes().items()
            if np.allclose(lo, table_lo) and np.allclose(hi, table_hi)]
    assert same == ['/World/Props/work_table/top']
    assert SCENE['table']['material'] in SCENE['materials']


def test_objects_do_not_interpenetrate_each_other(manifest):
    boxes = [(spec['id'], *_footprint(spec, manifest['assets'][spec['asset_id']])) for spec in SCENE['objects']]
    for index, (name_a, lo_a, hi_a) in enumerate(boxes):
        for name_b, lo_b, hi_b in boxes[index + 1:]:
            gap = max(float(np.max(lo_b - hi_a)), float(np.max(lo_a - hi_b)))
            assert gap >= 0.010, f'{name_a} and {name_b} are {gap * 1000:.1f} mm apart'


# --- 7. furniture --------------------------------------------------------------

ROOM_SHELL_ROLES = ('room_shell', 'opening_closure')


def test_furniture_is_context_not_an_obstacle():
    """Inside the reach envelope only the table's own understructure (below the
    top, derived from the prop) and the shallow receptacle on the top may exist."""
    x_lo, x_hi = family.REACH_ENVELOPE_X
    y_lo, y_hi = family.REACH_ENVELOPE_Y
    underside = SCENE['table']['position_m'][2] - SCENE['table']['half_extents_m'][2]
    for prop in SCENE['static_props']:
        for part in prop['parts']:
            lo, hi = _box_of(part)
            if not (lo[0] < x_hi and hi[0] > x_lo and lo[1] < y_hi and hi[1] > y_lo):
                continue
            if prop['id'] == family.SUPPORT_ID:
                assert hi[2] <= underside + 1e-9, f'{part["id"]} rises above the table underside'
            elif prop['role'] == 'receptacle':
                assert lo[2] >= TABLE_TOP_Z - 1e-9 and hi[2] <= TABLE_TOP_Z + 0.045
            else:
                raise AssertionError(f'{prop["id"]}/{part["id"]} intrudes into the reach envelope')


def test_static_props_are_all_box_parts_with_sensible_sizes():
    for prop in SCENE['static_props']:
        assert prop['id'].isidentifier() and prop['parts']
        limit = max(ROOM['size_m']) if prop['role'] == 'room_shell' else 2.0
        for part in prop['parts']:
            assert part['geometry'] == 'box' and part['id'].isidentifier()
            half = np.asarray(part['half_extents_m'], dtype=float)
            assert np.all(half > 0) and np.all(half < limit), (prop['id'], part['id'])
            assert part['material'] in SCENE['materials']
        ids = [part['id'] for part in prop['parts']]
        assert len(ids) == len(set(ids)), f'{prop["id"]} repeats a part id'


def test_every_declared_material_is_valid_and_used():
    from bench.simulation.isaacsim_backend.visuals import validate_material
    for name, raw in SCENE['materials'].items():
        spec = validate_material(raw, name)
        assert spec['kind'] == 'color' and all(0.0 <= c <= 1.0 for c in spec['color'])
        assert 0.0 <= spec['roughness'] <= 1.0 and 0.0 <= spec['metallic'] <= 1.0
    referenced = set(SCENE['room']['surfaces'].values()) | {SCENE['table']['material']}
    referenced |= set(family.IMPLICIT_MATERIALS)
    for prop in SCENE['static_props']:
        referenced.add(prop['material'])
        referenced.update(part['material'] for part in prop['parts'])
    assert referenced == set(SCENE['materials']), (sorted(referenced ^ set(SCENE['materials'])))


def test_furniture_stands_in_the_room_on_the_floor_or_the_rug():
    origin, size = np.asarray(ROOM['origin_m']), np.asarray(ROOM['size_m'])
    lim_lo, lim_hi = origin - size / 2.0, origin + size / 2.0
    rug_top = _box_of(_part('rug', 'pile'))[1][2]
    for prop in SCENE['static_props']:
        if prop['role'] in ROOM_SHELL_ROLES:
            continue
        lo = np.min([_box_of(p)[0] for p in prop['parts']], axis=0)
        hi = np.max([_box_of(p)[1] for p in prop['parts']], axis=0)
        assert np.all(lo[:2] >= lim_lo[:2] - 1e-9) and np.all(hi[:2] <= lim_hi[:2] + 1e-9), prop['id']
        assert lo[2] >= -1e-9 and hi[2] <= ROOM['ceiling_height_m']
        if prop['role'] == 'wall_hung':
            assert lo[2] >= 0.9
        elif prop['role'] == 'receptacle':
            assert lo[2] == pytest.approx(TABLE_TOP_Z, abs=1e-12)
        elif prop.get('rests_on') == 'rug':
            assert lo[2] == pytest.approx(rug_top, abs=1e-12), prop['id']
        else:
            assert lo[2] == pytest.approx(0.0, abs=1e-9), f'{prop["id"]} floats {lo[2]:.4f} m'


def test_the_wall_picture_hangs_just_proud_of_its_wall():
    front_inner_y = ROOM['origin_m'][1] + ROOM['size_m'][1] / 2.0
    hi = max(_box_of(part)[1][1] for part in _prop('wall_picture')['parts'])
    assert front_inner_y - 0.005 <= hi < front_inner_y


def test_static_prop_parts_do_not_overlap_each_other():
    for prop in SCENE['static_props']:
        boxes = [(part['id'], *_box_of(part)) for part in prop['parts']]
        for index, (name_a, lo_a, hi_a) in enumerate(boxes):
            for name_b, lo_b, hi_b in boxes[index + 1:]:
                overlap = np.minimum(hi_a, hi_b) - np.maximum(lo_a, lo_b)
                assert not np.all(overlap > 1e-6), f'{prop["id"]}: {name_a} and {name_b} overlap {overlap}'


def test_different_props_and_the_table_do_not_interpenetrate():
    boxes = list(_authored_static_boxes().items())
    for index, (name_a, (lo_a, hi_a)) in enumerate(boxes):
        for name_b, (lo_b, hi_b) in boxes[index + 1:]:
            if name_a.rsplit('/', 1)[0] == name_b.rsplit('/', 1)[0]:
                continue
            overlap = np.minimum(hi_a, hi_b) - np.maximum(lo_a, lo_b)
            assert not np.all(overlap > 1e-6), f'{name_a} and {name_b} interpenetrate by {overlap}'


def test_the_rug_lies_on_the_floor():
    lo, hi = _box_of(_part('rug', 'pile'))
    assert lo[2] == pytest.approx(0.0, abs=1e-12) and hi[2] - lo[2] <= 0.010


def test_the_furniture_covers_a_recognisable_living_room():
    ids = {prop['id'] for prop in SCENE['static_props']}
    assert {'sofa', 'coffee_table', 'rug', 'tv_console', 'floor_lamp', 'wall_picture', 'bookshelf',
            'side_table', 'pouf', 'plant', 'door', 'window', 'toy_tray', 'wall_solids'} == ids
    sofa = {part['id'] for part in _prop('sofa')['parts']}
    assert {'base', 'back', 'arm_left', 'arm_right'} <= sofa and sum('cushion' in p for p in sofa) >= 4
    assert {'tv_body', 'tv_screen'} <= {part['id'] for part in _prop('tv_console')['parts']}


def test_the_object_mix_matches_the_brief(manifest):
    kinds = {}
    for spec in SCENE['objects']:
        kinds.setdefault(manifest['assets'][spec['asset_id']]['geometry_class'], []).append(spec['id'])
    blocks = kinds['studded_plastic_block']
    assert 6 <= len(blocks) <= 10
    assert len([b for b in blocks if 'cube' in b]) >= 3 and len([b for b in blocks if 'brick' in b]) >= 2
    assert len(kinds['plush_bear_flat_base']) == len(kinds['sphere_60mm']) == 1
    assert len(kinds['hardcover_picture_book']) == 1
    assert len(SCENE['objects']) == len({spec['id'] for spec in SCENE['objects']})


def test_an_object_identifier_is_safe_for_the_runtime():
    import re
    pattern = re.compile(r'[a-z][a-z0-9_]{1,63}')
    for spec in SCENE['objects']:
        assert pattern.fullmatch(spec['id']) and 0.0 <= spec['friction'] <= 5.0
