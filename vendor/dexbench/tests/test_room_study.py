"""CPU validation of the ``study`` scene family (no GPU, no Isaac Sim).

Every check reads the authored module and its pinned asset library. Where it
matters, it reads the *real collider geometry*: OBJ vertices and analytic
parts placed exactly as ``scene.add_task_asset`` places them, and static boxes
exactly as ``scene.add_structured_scene``, ``room.add_room`` and
``room.add_pedestal`` build them. What a scene says about itself is not taken
on trust.

  contract   validate_family succeeds and rejects broken copies; v7 is pinned, v6 kept
  rule 1     assets resolve with hashes; positive mass, bounding-box inertia, colliders
  rule 2     the tape roll's declared bore passes the shared strict rule and is open in
             the meshes; the pen cup is a blind cavity that no collider enters
  rule 3     each object's lowest collider point is on the desk pad; nothing interpenetrates;
             nothing is stacked; no static geometry rests on or hangs low over an object
  rule 4     every collider, static boxes and fixtures included, is at least 3 mm thick
  rules 5-7  design masses; reach; the pedestal is clear of the desk, its skirt
             reaches the floor, and it binds the 'top'/'side' materials
  rule 8     the room is a closed shell with openings, lights (the lamp bulb inside its
             shade) and a dome
  rule 9     both world cameras frame the whole desk top; every object is visible with
             the arm at home; the signature furniture is inside the third-person frustum
             and not hidden
  rule 10    the cup is a free object, the furniture is static, and nothing static
             occupies the work volume
  support    every static box is connected to the floor or a wall by face contact

The last section mutates copies of the scene and asserts that each check rejects
the defect it guards against: the v6 book pads, a floating pedestal, lifted
props, a thin plate, a duplicated desk collider, an unframed bookcase, a desk
cut off by desk_view, and a static pen cup.

Not claimed: anything that needs PhysX or a renderer. Visibility is a ray-cast
against collider boxes and the arm's link capsules, not a rendered image.
"""
from __future__ import annotations

import copy
import hashlib
import math
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial import ConvexHull

from bench.paths import load_json
from scenes.rooms import study
from bench.simulation.isaacsim_backend import cameras as cameras_module
from bench.simulation.isaacsim_backend.asset_catalog import load_obj_mesh, load_task_asset
from bench.simulation.isaacsim_backend.scene_family import (FamilyError, support_surfaces, uniform_box_inertia,
                                                validate_family, validate_object)
from bench.simulation.isaacsim_backend.room import WALL_FACES, _wall_geometry, _wall_quads, validate_room
from bench.simulation.isaacsim_backend.visuals import validate_material

SCENE = study.SCENE
ASSET_ROOT = Path(study.ASSET_ROOT)
MANIFEST_PATH = ASSET_ROOT / 'manifest.json'
V7_ROOT = Path(study.SUPERSEDED_ASSET_ROOTS[0])
V7_MANIFEST_SHA256 = 'ad470636cf9f7ceb23429ccfb573089852fc50381b4b97345a2da4e4f03a82eb'
V6_ROOT = Path(study.SUPERSEDED_ASSET_ROOTS[1])
V6_MANIFEST_SHA256 = 'db677c8bb554b8fa9c9a81fdad6e8f719fc3b841aba8be8e4fdcc47eb23a33a6'
TOUCH = 1e-6                     # faces closer than this touch; deeper overlap interpenetrates
MIN_THICKNESS_M = 0.003          # rule 4
CAGE_CLEARANCE_M = 0.10          # static geometry over an object must leave the hand this much room
H_APERTURE_MM = 20.955           # the USD default; env.py never sets an aperture
V_APERTURES_MM = (15.2908, 20.955 * 720 / 960)   # USD default, and the 4:3 render-product fit


@pytest.fixture(scope='module')
def manifest():
    assert MANIFEST_PATH.is_file(), f'the pinned study library is missing: {MANIFEST_PATH}'
    return load_json(MANIFEST_PATH)


@pytest.fixture(scope='module')
def arm_capsules():
    return _home_pose_capsules(SCENE)


# --- rigid-body geometry ---------------------------------------------------------

def _rotation(wxyz):
    w, x, y, z = np.asarray(wxyz, dtype=float) / np.linalg.norm(wxyz)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def _xyzw_rotation(xyzw):
    x, y, z, w = xyzw
    return _rotation([w, x, y, z])


_MESH_CACHE = {}


def _mesh_vertices(part):
    path = str(ASSET_ROOT / part['mesh_path'])
    if path not in _MESH_CACHE:
        _MESH_CACHE[path] = np.asarray(load_obj_mesh(path)[0], dtype=float)
    return _MESH_CACHE[path]


def _part_points(part):
    """Collider points of one part in the asset frame, as add_task_asset places them."""
    position = np.asarray(part.get('position_m', [0., 0., 0.]), dtype=float)
    rotation = _xyzw_rotation(part.get('quaternion_xyzw', [0., 0., 0., 1.]))
    geometry = part.get('geometry', 'convex_mesh')
    if geometry == 'box':
        half = np.asarray(part['half_extents_m'], dtype=float)
        local = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]) * half
    elif geometry == 'cylinder':
        angles = np.linspace(0., 2 * math.pi, 96, endpoint=False)
        ring = np.c_[part['radius_m'] * np.cos(angles), part['radius_m'] * np.sin(angles)]
        hh = part['half_height_m']
        local = np.vstack([np.c_[ring, np.full(len(ring), -hh)], np.c_[ring, np.full(len(ring), hh)]])
    else:
        local = _mesh_vertices(part)
    return local @ rotation.T + position


def _part_obb(part):
    """(centre, rotation, half) of a box bounding one collider part, in the asset frame."""
    position = np.asarray(part.get('position_m', [0., 0., 0.]), dtype=float)
    rotation = _xyzw_rotation(part.get('quaternion_xyzw', [0., 0., 0., 1.]))
    geometry = part.get('geometry', 'convex_mesh')
    if geometry == 'box':
        return position, rotation, np.asarray(part['half_extents_m'], dtype=float)
    if geometry == 'cylinder':
        return position, rotation, np.array([part['radius_m'], part['radius_m'], part['half_height_m']])
    points = _mesh_vertices(part) @ rotation.T + position
    low, high = points.min(axis=0), points.max(axis=0)
    return (low + high) / 2., np.eye(3), (high - low) / 2.


def _body_obbs(manifest, spec):
    """World collider boxes of one object or fixture."""
    rotation = _rotation(spec.get('quaternion_wxyz', [1., 0., 0., 0.]))
    position = np.asarray(spec['position_m'], dtype=float)
    boxes = []
    for part in manifest['assets'][spec['asset_id']]['collision_parts']:
        centre, local, half = _part_obb(part)
        boxes.append({'centre': rotation @ centre + position, 'rotation': rotation @ local, 'half': half,
                      'name': f'{spec["id"]}/{part["label"]}'})
    return boxes


def _body_points(manifest, spec):
    rotation = _rotation(spec.get('quaternion_wxyz', [1., 0., 0., 0.]))
    position = np.asarray(spec['position_m'], dtype=float)
    return np.vstack([_part_points(part) @ rotation.T + position
                      for part in manifest['assets'][spec['asset_id']]['collision_parts']])


def _aabb_of(obb):
    half = np.abs(obb['rotation']) @ obb['half']
    return obb['centre'] - half, obb['centre'] + half


def _obbs_overlap(a, b, touch=TOUCH):
    """Separating-axis test; faces closer than ``touch`` count as touching, not overlap."""
    ra, rb = a['rotation'], b['rotation']
    axes = [ra[:, i] for i in range(3)] + [rb[:, i] for i in range(3)]
    for i in range(3):
        for j in range(3):
            cross = np.cross(ra[:, i], rb[:, j])
            if np.linalg.norm(cross) > 1e-9:
                axes.append(cross / np.linalg.norm(cross))
    delta = b['centre'] - a['centre']
    for axis in axes:
        reach = (sum(a['half'][k] * abs(ra[:, k] @ axis) for k in range(3)) +
                 sum(b['half'][k] * abs(rb[:, k] @ axis) for k in range(3)))
        if abs(delta @ axis) >= reach - touch:
            return False
    return True


def _footprints_overlap(a, b):
    """Do two boxes overlap in plan (their z extents ignored)?"""
    tall = np.array([1., 1., 1e3])
    a = dict(a, half=a['half'] * tall if abs(a['rotation'][2, 2]) > 0.999 else a['half'])
    b = dict(b, half=b['half'] * tall if abs(b['rotation'][2, 2]) > 0.999 else b['half'])
    a = dict(a, centre=a['centre'] * np.array([1., 1., 0.]))
    b = dict(b, centre=b['centre'] * np.array([1., 1., 0.]))
    return _obbs_overlap(a, b)


# --- static geometry, exactly as scene.py and room.py build it ------------------

def _box(name, kind, group, centre, half):
    centre, half = np.asarray(centre, dtype=float), np.asarray(half, dtype=float)
    return {'name': name, 'kind': kind, 'group': group, 'centre': centre, 'half': half,
            'rotation': np.eye(3), 'lo': centre - half, 'hi': centre + half}


def _static_boxes(scene):
    """Every solid static box scene.py builds: table, surfaces, prop parts, pedestal, skirt."""
    boxes = []
    table = scene.get('table')
    if table:
        boxes.append(_box('work_table/top', 'table', 'table', table['position_m'], table['half_extents_m']))
    for spec in scene.get('surfaces', []) or []:
        boxes.append(_box(f'{spec["id"]}/plate', 'surface', spec['id'], spec['position_m'],
                          spec['half_extents_m']))
    for prop in scene.get('static_props', []) or []:
        for part in prop['parts']:
            boxes.append(_box(f'{prop["id"]}/{part["id"]}', 'prop', prop['id'], part.get('position_m', [0, 0, 0]),
                              part['half_extents_m']))
    pedestal = scene.get('pedestal')
    if pedestal:
        centre = np.asarray(pedestal['position_m'], dtype=float)
        half = np.asarray(pedestal['half_extents_m'], dtype=float)
        boxes.append(_box('pedestal/top', 'pedestal', 'pedestal', centre, half))
        skirt = float(pedestal.get('skirt_height_m', 0.))
        if skirt > 0:
            boxes.append(_box('pedestal/skirt', 'pedestal', 'pedestal',
                              centre - np.array([0., 0., half[2] + skirt / 2.]), [half[0], half[1], skirt / 2.]))
    return boxes


def _room_colliders(scene):
    """The floor and the wall colliders room.add_room cuts around the openings."""
    room = validate_room(scene['room'])
    sx, sy, _ = room['size_m']
    ox, oy, oz = room['origin_m']
    t = room['wall_thickness_m']
    boxes = [_box('room/floor', 'floor', 'room', (ox, oy, oz - t / 2.), (sx / 2. + t, sy / 2. + t, t / 2.))]
    for face in WALL_FACES:
        wall = _wall_geometry(room, face)
        for index, (lo_u, hi_u, lo_v, hi_v) in enumerate(_wall_quads(wall, [o for o in room['openings']
                                                                              if o['wall'] == face])):
            centre = (wall['center'] + (lo_u + hi_u) / 2. * wall['u_axis'] + (lo_v + hi_v) / 2. * wall['v_axis']
                      + wall['normal'] * t / 2.)
            half = (np.abs(wall['u_axis']) * (hi_u - lo_u) / 2. + np.abs(wall['v_axis']) * (hi_v - lo_v) / 2.
                    + np.abs(wall['normal']) * t / 2.)
            boxes.append(_box(f'room/wall_{face}_{index:02d}', 'wall', 'room', centre, half))
    return boxes


def _fixture_boxes(scene, manifest):
    """Each fixture as one world box (the AABB of its colliders), for support and occlusion."""
    boxes = []
    for spec in scene.get('fixtures', []) or []:
        points = _body_points(manifest, spec)
        low, high = points.min(axis=0), points.max(axis=0)
        boxes.append(_box(f'fixture/{spec["id"]}', 'fixture', spec['id'], (low + high) / 2., (high - low) / 2.))
    return boxes


def _face_contact(a, b, tol=TOUCH, min_overlap=1e-4):
    """Do two axis-aligned boxes share a face patch of positive area?"""
    for k in range(3):
        if abs(a['hi'][k] - b['lo'][k]) <= tol or abs(b['hi'][k] - a['lo'][k]) <= tol:
            others = [j for j in range(3) if j != k]
            if all(min(a['hi'][j], b['hi'][j]) - max(a['lo'][j], b['lo'][j]) > min_overlap for j in others):
                return True
    return False


def _interior(scene):
    room = validate_room(scene['room'])
    size, origin = np.asarray(room['size_m'], dtype=float), np.asarray(room['origin_m'], dtype=float)
    low = np.array([origin[0] - size[0] / 2., origin[1] - size[1] / 2., origin[2]])
    high = np.array([origin[0] + size[0] / 2., origin[1] + size[1] / 2., origin[2] + room['ceiling_height_m']])
    return low, high


# --- the checks, as functions of a scene so they can be mutation-tested ----------

def _support_graph_problems(scene, manifest):
    """Every static box and fixture must be held: by the floor, a wall, or a held box.

    Contact is a shared face patch of positive area. The floor holds a box whose
    bottom is on the floor plane inside the room; a wall collider holds a box
    that touches it. Anything else must touch something already held.
    """
    nodes = _static_boxes(scene) + _fixture_boxes(scene, manifest)
    walls = [b for b in _room_colliders(scene) if b['kind'] == 'wall']
    low, high = _interior(scene)
    held = set()
    for index, node in enumerate(nodes):
        on_floor = (abs(node['lo'][2] - low[2]) <= TOUCH and np.all(node['lo'][:2] >= low[:2] - TOUCH)
                    and np.all(node['hi'][:2] <= high[:2] + TOUCH))
        if on_floor or any(_face_contact(node, wall) for wall in walls):
            held.add(index)
    changed = True
    while changed:
        changed = False
        for index, node in enumerate(nodes):
            if index not in held and any(_face_contact(node, nodes[j]) for j in held):
                held.add(index)
                changed = True
    problems = [f'{node["name"]} is held by nothing (no face contact with the floor, a wall or a held box)'
                for index, node in enumerate(nodes) if index not in held]
    # Gravity: floor furniture cannot hang on a wall. Unless a group is declared
    # wall-mounted, every box at its lowest level must rest on the floor or on the
    # top face of another group's box.
    groups = {}
    for node in nodes:
        groups.setdefault(node['group'], []).append(node)
    for group, members in groups.items():
        if group in study.WALL_MOUNTED:
            continue
        bottom = min(node['lo'][2] for node in members)
        for node in members:
            if node['lo'][2] > bottom + TOUCH:
                continue
            on_floor = abs(node['lo'][2] - low[2]) <= TOUCH
            on_top = any(other['group'] != group and abs(node['lo'][2] - other['hi'][2]) <= TOUCH
                         and all(min(node['hi'][k], other['hi'][k]) - max(node['lo'][k], other['lo'][k]) > 1e-4
                                 for k in (0, 1)) for other in nodes)
            if not (on_floor or on_top):
                problems.append(f'{node["name"]} is floor furniture whose base rests on nothing '
                                f'(bottom z {node["lo"][2]:.4f})')
    return problems


def _caging_problems(scene, manifest):
    """No static box or fixture may rest on, pin, or hang low over a dynamic object.

    Whatever static geometry overlaps an object's plan footprint must lie wholly
    below the object's lowest collider point (its support), or at least
    CAGE_CLEARANCE_M above its highest one. A static plate whose bottom face is
    on a book's top face (v6) fails both.
    """
    statics = [dict(b) for b in _static_boxes(scene)]
    for spec in scene.get('fixtures', []) or []:
        statics += [dict(obb, kind='fixture') for obb in _body_obbs(manifest, spec)]
    problems = []
    for spec in scene['objects']:
        points = _body_points(manifest, spec)
        bottom, top = points[:, 2].min(), points[:, 2].max()
        for body in _body_obbs(manifest, spec):
            for static in statics:
                if not _footprints_overlap(body, static):
                    continue
                low, high = _aabb_of(static)
                if high[2] <= bottom + TOUCH or low[2] >= top + CAGE_CLEARANCE_M:
                    continue
                problems.append(f'{static["name"]} spans z {low[2]:.4f}-{high[2]:.4f} over '
                                f'{spec["id"]} (z {bottom:.4f}-{top:.4f})')
    return sorted(set(problems))


def _stacking_problems(scene, manifest):
    """No dynamic object may lie over another (no dynamic-on-dynamic support)."""
    problems = []
    objects = scene['objects']
    for i, first in enumerate(objects):
        for second in objects[i + 1:]:
            if any(_footprints_overlap(a, b) for a in _body_obbs(manifest, first)
                   for b in _body_obbs(manifest, second)):
                problems.append(f'{first["id"]} and {second["id"]} overlap in plan')
    return problems


def _convex_min_width(points):
    """Minimum width of a convex point set over its own face normals."""
    hull = ConvexHull(points)
    widths = []
    for normal in hull.equations[:, :3]:
        projection = points @ normal
        widths.append(projection.max() - projection.min())
    return float(min(widths))


def _part_min_width(part):
    geometry = part.get('geometry', 'convex_mesh')
    if geometry == 'box':
        return 2. * float(min(part['half_extents_m']))
    if geometry == 'cylinder':
        return 2. * float(min(part['radius_m'], part['half_height_m']))
    return _convex_min_width(_mesh_vertices(part))


def _thin_problems(scene, manifest):
    """Every collider - static box, fixture piece or dynamic piece - is >= 3 mm in every direction."""
    problems = [f'{box["name"]} is {2 * box["half"].min() * 1000:.2f} mm thick'
                for box in _static_boxes(scene) if 2 * box['half'].min() < MIN_THICKNESS_M - 1e-9]
    for spec in list(scene['objects']) + list(scene.get('fixtures', []) or []):
        for part in manifest['assets'][spec['asset_id']]['collision_parts']:
            width = _part_min_width(part)
            if width < MIN_THICKNESS_M - 1e-9:
                problems.append(f'{spec["id"]}/{part["label"]} is {width * 1000:.2f} mm thick')
    return problems


def _pedestal_problems(scene):
    """The pedestal stands on the floor through its skirt and is bound to furniture materials."""
    pedestal = scene.get('pedestal') or {}
    problems = []
    centre = np.asarray(pedestal.get('position_m', [0, 0, 0]), dtype=float)
    half = np.asarray(pedestal.get('half_extents_m', [0, 0, 0]), dtype=float)
    bottom = centre[2] - half[2]
    skirt = float(pedestal.get('skirt_height_m', 0.))
    floor = _interior(scene)[0][2]
    if abs(bottom - skirt - floor) > 1e-9:
        problems.append(f'the pedestal slab bottom is at z {bottom:.3f} but its skirt reaches only '
                        f'{bottom - skirt:.3f} (floor {floor:.3f})')
    for name in ('top', 'side'):
        if name not in (scene.get('materials') or {}):
            problems.append(f"add_pedestal binds a material named {name!r}; none is defined")
    return problems


def _duplicate_static_problems(scene):
    """No two solid static boxes coincide (v6 built the desk slab twice)."""
    boxes = _static_boxes(scene)
    problems = []
    for i, first in enumerate(boxes):
        for second in boxes[i + 1:]:
            if np.allclose(first['lo'], second['lo'], atol=1e-6) and np.allclose(first['hi'], second['hi'], atol=1e-6):
                problems.append(f'{first["name"]} and {second["name"]} are the same box')
    return problems


def _role_problems(scene):
    """The pen cup is a free object; only furniture assets are fixtures."""
    fixtures = {spec['asset_id'] for spec in scene.get('fixtures', []) or []}
    objects = {spec['asset_id'] for spec in scene['objects']}
    problems = []
    if 'pen_cup' not in objects or 'pen_cup' in fixtures:
        problems.append('the pen cup must be a dynamic object, not a fixture')
    if fixtures - set(study.FIXTURE_ASSETS):
        problems.append(f'unexpected fixtures {sorted(fixtures - set(study.FIXTURE_ASSETS))}')
    if objects & fixtures:
        problems.append(f'assets used as both object and fixture: {sorted(objects & fixtures)}')
    return problems


# --- cameras --------------------------------------------------------------------

def _camera_basis(camera):
    eye = np.asarray(camera['eye'], dtype=float)
    forward = np.asarray(camera['target'], dtype=float) - eye
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, [0., 0., 1.])
    right /= np.linalg.norm(right)
    return eye, right, np.cross(right, forward), forward


def _project(camera, points, v_aperture):
    """Pixel coordinates and in-frame flags, as cameras.project_point computes them."""
    eye, right, up, forward = _camera_basis(camera)
    width, height = camera['resolution']
    delta = np.asarray(points, dtype=float).reshape(-1, 3) - eye
    depth = delta @ forward
    safe = np.maximum(depth, 1e-9)
    px = width / 2. + (delta @ right) / safe * camera['focal_length_mm'] / H_APERTURE_MM * width
    py = height / 2. - (delta @ up) / safe * camera['focal_length_mm'] / v_aperture * height
    inside = (depth > 1e-6) & (px >= 0) & (px <= width) & (py >= 0) & (py <= height)
    return px, py, inside


def _in_frame(camera, points):
    return np.logical_and.reduce([_project(camera, points, v)[2] for v in V_APERTURES_MM])


def _face_samples(box, n=6):
    """Points on the six faces of a box, with outward normals."""
    grid = (np.arange(n) + 0.5) / n * 2. - 1.
    a, b = (g.ravel() for g in np.meshgrid(grid, grid))
    points, normals = [], []
    for axis in range(3):
        others = [k for k in range(3) if k != axis]
        for sign in (-1., 1.):
            local = np.zeros((len(a), 3))
            local[:, axis] = sign * box['half'][axis]
            local[:, others[0]] = a * box['half'][others[0]]
            local[:, others[1]] = b * box['half'][others[1]]
            normal = np.zeros((len(a), 3))
            normal[:, axis] = sign
            points.append(local @ box['rotation'].T + box['centre'])
            normals.append(normal @ box['rotation'].T)
    return np.vstack(points), np.vstack(normals)


def _segments_hit_box(eye, points, box, eps=1e-6):
    """Vectorised slab test: does the open segment eye -> point pass through the box?"""
    start = (np.asarray(eye, dtype=float) - box['centre']) @ box['rotation']
    end = (points - box['centre']) @ box['rotation']
    direction = end - start
    t0, t1 = np.zeros(len(points)), np.ones(len(points))
    hit = np.ones(len(points), dtype=bool)
    for k in range(3):
        flat = np.abs(direction[:, k]) < 1e-12
        hit &= ~(flat & ((start[k] <= -box['half'][k]) | (start[k] >= box['half'][k])))
        with np.errstate(divide='ignore', invalid='ignore'):
            a = (-box['half'][k] - start[k]) / direction[:, k]
            b = (box['half'][k] - start[k]) / direction[:, k]
        t0 = np.maximum(t0, np.where(flat, -np.inf, np.minimum(a, b)))
        t1 = np.minimum(t1, np.where(flat, np.inf, np.maximum(a, b)))
    return hit & (t0 < t1) & (t0 < 1. - eps) & (t1 > eps)


def _segments_near_capsule(eye, points, capsule, samples=40):
    start, end, radius = capsule
    t = np.linspace(0., 1., samples)
    along = eye[None, None, :] + t[None, :, None] * (points[:, None, :] - eye[None, None, :])
    axis = end - start
    s = np.clip(((along - start) @ axis) / max(float(axis @ axis), 1e-12), 0., 1.)
    nearest = start + s[..., None] * axis
    return (np.linalg.norm(along - nearest, axis=-1) < radius).any(axis=1)


def _home_pose_capsules(scene):
    """The arm and hand at the scene's home pose, as capsules from CPU forward kinematics."""
    from bench.control.fingertip import Fr3SharpaGeometry
    from bench.geometry import quaternion_matrix
    robot = scene['robot']
    geometry = Fr3SharpaGeometry(robot['assets_root'], robot['base_position_m'], hand_id=robot['hand_id'],
                                 assembly_version=robot['assembly_version'])
    joints = {joint.name: 0.0 for joint in geometry.hand.active}
    joints.update({f'fr3_joint{i}': float(v) for i, v in enumerate(robot['arm_home_rad'], start=1)})
    arm = {name: (geometry.arm_base @ np.asarray(transform))[:3, 3]
           for name, transform in geometry.arm.forward({f'fr3_joint{i}': float(v) for i, v in
                                                        enumerate(robot['arm_home_rad'], start=1)}).items()}
    chain = [('fr3_link0', 'fr3_link2', .080), ('fr3_link2', 'fr3_link3', .070), ('fr3_link3', 'fr3_link4', .070),
             ('fr3_link4', 'fr3_link5', .065), ('fr3_link5', 'fr3_link7', .065), ('fr3_link7', 'fr3_link8', .060)]
    capsules = [(arm[a], arm[b], radius) for a, b, radius in chain]
    pose = geometry.public_geometry(joints)
    wrist = np.eye(4)
    wrist[:3, :3] = quaternion_matrix(pose['wrist']['quaternion_wxyz'])
    wrist[:3, 3] = pose['wrist']['position_m']
    links = [(wrist @ transform)[:3, 3] for transform in geometry.hand.forward(joints).values()]
    tips = [(wrist @ np.r_[np.asarray(tip, dtype=float), 1.])[:3] for tip in pose['fingertip_positions_m'].values()]
    palm = np.mean(links, axis=0)
    capsules.append((np.asarray(pose['wrist']['position_m'], dtype=float), palm, .060))
    capsules += [(palm, tip, .025) for tip in tips]
    return capsules


def _occluders(scene, manifest):
    """Everything solid a camera ray can meet: static boxes, fixtures and objects."""
    boxes = [dict(b) for b in _static_boxes(scene)]
    for spec in scene.get('fixtures', []) or []:
        boxes += [dict(obb, group=spec['id']) for obb in _body_obbs(manifest, spec)]
    for spec in scene['objects']:
        boxes += [dict(obb, group=spec['id']) for obb in _body_obbs(manifest, spec)]
    return boxes


def _visibility(camera, points, normals, own_groups, occluders, capsules):
    """(fraction in frame, fraction in frame and unoccluded) of camera-facing sample points."""
    eye = np.asarray(camera['eye'], dtype=float)
    facing = ((eye - points) * normals).sum(axis=1) > 1e-9
    points = points[facing]
    if not len(points):
        return 0., 0.
    inside = _in_frame(camera, points)
    toward = eye - points
    lifted = points + 1e-4 * toward / np.linalg.norm(toward, axis=1, keepdims=True)
    blocked = np.zeros(len(points), dtype=bool)
    for box in occluders:
        if box.get('group') not in own_groups:
            blocked |= _segments_hit_box(eye, lifted, box)
    for capsule in capsules:
        blocked |= _segments_near_capsule(eye, lifted, capsule)
    return float(inside.mean()), float((inside & ~blocked).mean())


def _group_samples(scene, manifest, item):
    """Sample points for one signature item: static props by id, fixtures, or an opening."""
    points, normals = [], []
    for box in _static_boxes(scene):
        if box['group'] in item['props']:
            p, n = _face_samples(box)
            points.append(p)
            normals.append(n)
    for spec in scene.get('fixtures', []) or []:
        if spec['id'] in item['fixtures']:
            for obb in _body_obbs(manifest, spec):
                p, n = _face_samples(obb)
                points.append(p)
                normals.append(n)
    for opening in item.get('openings', ()):
        spec = next(o for o in validate_room(scene['room'])['openings'] if o['id'] == opening)
        wall = _wall_geometry(validate_room(scene['room']), spec['wall'])
        grid = (np.arange(10) + 0.5) / 10. * 2. - 1.
        uu, vv = (g.ravel() for g in np.meshgrid(grid, grid))
        u = spec['center_m'][0] + uu * spec['size_m'][0] / 2.
        v = spec['center_m'][1] + vv * spec['size_m'][1] / 2.
        plane = wall['center'][None, :] + u[:, None] * wall['u_axis'] + v[:, None] * wall['v_axis']
        points.append(plane)
        normals.append(np.repeat(-wall['normal'][None, :], len(plane), axis=0))
    if not points:
        return np.zeros((0, 3)), np.zeros((0, 3))
    return np.vstack(points), np.vstack(normals)


# The furniture that makes the room read as a study, and the least of it that must
# show in the third-person view: in frame (both aperture conventions) and unoccluded.
SIGNATURE = {
    'bookcase': {'props': ('bookcase', 'bookcase_books'), 'fixtures': (), 'in_frame': .90, 'visible': .55},
    'wall_art': {'props': ('wall_art',), 'fixtures': (), 'in_frame': .90, 'visible': .60},
    'window': {'props': ('window_frame',), 'fixtures': (), 'openings': ('window',), 'in_frame': .70,
               'visible': .40},
    # Floor-length: the frame cuts their heads off and the desk hides their feet.
    'curtains': {'props': ('curtains',), 'fixtures': (), 'in_frame': .45, 'visible': .20},
    'wall_clock': {'props': (), 'fixtures': ('wall_clock',), 'in_frame': .90, 'visible': .80},
    'potted_plant': {'props': (), 'fixtures': ('potted_plant',), 'in_frame': .90, 'visible': .80},
    'laptop': {'props': (), 'fixtures': ('laptop',), 'in_frame': .95, 'visible': .70},
    'desk_lamp': {'props': (), 'fixtures': ('desk_lamp',), 'in_frame': .95, 'visible': .80},
    'waste_basket': {'props': (), 'fixtures': ('waste_basket',), 'in_frame': .90, 'visible': .80},
    'chair': {'props': ('chair', 'paper_stack'), 'fixtures': (), 'in_frame': .90, 'visible': .35},
}


def _signature_report(scene, manifest, capsules, camera_id='third_person'):
    camera = cameras_module.resolve_cameras(scene)[camera_id]
    occluders = _occluders(scene, manifest)
    report = {}
    for name, item in SIGNATURE.items():
        points, normals = _group_samples(scene, manifest, item)
        own = set(item['props']) | set(item['fixtures'])
        report[name] = _visibility(camera, points, normals, own, occluders, capsules) if len(points) else (0., 0.)
    return report


def _frustum_problems(scene, manifest, capsules):
    problems = []
    for name, (in_frame, visible) in _signature_report(scene, manifest, capsules).items():
        item = SIGNATURE[name]
        if in_frame < item['in_frame'] or visible < item['visible']:
            problems.append(f'{name}: {in_frame:.2f} in frame, {visible:.2f} visible '
                            f'(needs {item["in_frame"]:.2f} / {item["visible"]:.2f})')
    return problems


def _desk_corners(scene):
    table = scene['table']
    centre, half = np.asarray(table['position_m'], dtype=float), np.asarray(table['half_extents_m'], dtype=float)
    top = centre[2] + half[2]
    return np.array([[centre[0] + sx * half[0], centre[1] + sy * half[1], top] for sx in (-1, 1) for sy in (-1, 1)])


def _desk_framing_problems(scene, margin_px=10.):
    """Both world cameras frame the whole desk top, with a margin, under both apertures."""
    problems = []
    for name, camera in cameras_module.resolve_cameras(scene).items():
        if camera['kind'] != 'world':
            continue
        width, height = camera['resolution']
        for v_aperture in V_APERTURES_MM:
            px, py, inside = _project(camera, _desk_corners(scene), v_aperture)
            low = min(px.min(), py.min(), width - px.max(), height - py.max())
            if not inside.all() or low < margin_px:
                problems.append(f'{name}: desk-top corners at {np.c_[px, py].round().tolist()} '
                                f'(aperture {v_aperture:.3f} mm) are not all {margin_px:.0f} px inside')
    return problems


# --- the family contract --------------------------------------------------------

def test_validate_family_succeeds():
    record = validate_family(SCENE)
    assert record['family'] == 'study'
    assert record['objects'] == len(SCENE['objects']) == len(study.OBJECT_LAYOUT)
    assert record['static_props'] == len(SCENE['static_props'])
    assert [s['id'] for s in record['supports']] == ['desk_pad']
    assert SCENE['schema_version'] == 'isaacsim-scene-family-v1'


def test_family_rejects_broken_copies():
    """The validator is really exercised: floating and sunk objects are refused."""
    floating = copy.deepcopy(SCENE)
    floating['objects'][0]['position_m'][2] += 0.005
    with pytest.raises(FamilyError, match='floats'):
        validate_family(floating)
    sunk = copy.deepcopy(SCENE)
    sunk['objects'][-1]['position_m'][2] -= 0.005
    with pytest.raises(FamilyError, match='sinks'):
        validate_family(sunk)


def test_scene_declares_the_required_blocks():
    assert SCENE['physics']['control_dt_s'] / SCENE['physics']['dt_s'] >= 2
    assert SCENE['physics']['gravity_m_s2'] == [0.0, 0.0, -9.81]
    assert 'mount_position_m' not in SCENE['robot'] and 'mount_quaternion_wxyz' not in SCENE['robot']
    assert len(SCENE['robot']['arm_home_rad']) == 7
    assert SCENE['robot']['assembly_version'] == 'isaacsim_6_1_v2'


def test_module_import_needs_no_gpu_stack():
    import subprocess
    import sys
    code = ('import sys, scenes.rooms.study as m; '
            'bad = [k for k in ("pxr", "omni", "isaacsim", "torch", "warp", "trimesh") if k in sys.modules]; '
            'print(bad); sys.exit(1 if bad else 0)')
    result = subprocess.run([sys.executable, '-B', '-c', code], capture_output=True, text=True,
                            cwd=str(Path(__file__).resolve().parents[1]))
    assert result.returncode == 0, result.stdout + result.stderr


def test_scene_pins_v8_and_keeps_v7_and_v6_as_evidence(manifest):
    assert study.ASSET_VERSION == 'v8' == ASSET_ROOT.name == manifest['asset_version']
    v7 = V7_ROOT / 'manifest.json'
    assert v7.is_file(), 'v7 (whose round pen rolled natively) must stay on disk as evidence'
    assert hashlib.sha256(v7.read_bytes()).hexdigest() == V7_MANIFEST_SHA256
    for spec in SCENE['objects'] + SCENE['fixtures']:
        assert spec['asset_root'] == study.ASSET_ROOT, spec['id']
    v6 = V6_ROOT / 'manifest.json'
    assert v6.is_file(), 'the withdrawn v6 library must stay on disk as evidence'
    assert hashlib.sha256(v6.read_bytes()).hexdigest() == V6_MANIFEST_SHA256


# --- rule 1: assets resolve and are real bodies --------------------------------

def test_every_object_and_fixture_resolves_through_the_runtime_loader(manifest):
    """The exact loader scene.add_task_asset uses, with hash verification."""
    for spec in SCENE['objects'] + SCENE['fixtures']:
        asset = load_task_asset(spec['asset_id'], asset_root=spec['asset_root'], verify_hashes=True)
        assert asset['collision_parts'] and asset['visual_parts'], spec['id']
        for part in asset['visual_parts'] + asset['collision_parts']:
            assert part.get('geometry', 'convex_mesh') in ('box', 'cylinder', 'convex_mesh'), spec['id']
            if 'mesh_path' in part:
                vertices, faces = load_obj_mesh(part['mesh_path'])
                assert len(vertices) >= 4 and len(faces) >= 4, part['mesh_path']


def test_dynamic_objects_are_real_bodies(manifest):
    for spec in SCENE['objects']:
        record = manifest['assets'][spec['asset_id']]
        assert record['dynamic'] is True and record['mass_kg'] > 0, spec['id']
        assert record['collision_parts'], spec['id']
        assert all(part['classification'] == 'physics' for part in record['collision_parts'])
        size = np.asarray(record['size_m'], dtype=float)
        assert np.allclose(record['inertia_diag_kg_m2'], uniform_box_inertia(record['mass_kg'], size), rtol=1e-9)
        assert 'bounding-box' in record['inertia_method'], spec['id']
        low, high = np.asarray(record['aabb_m'][0]), np.asarray(record['aabb_m'][1])
        com = np.asarray(record['center_of_mass_m'], dtype=float)
        assert np.all(com > low) and np.all(com < high), spec['id']


def test_centres_of_mass_are_the_uniform_density_values(manifest):
    """The tray and the cup are not solid boxes: their centre of mass sits low."""
    assert manifest['assets']['pen_cup']['center_of_mass_m'][2] < -0.005
    assert manifest['assets']['organiser_tray']['center_of_mass_m'][2] < -0.004
    for asset_id in ('book_blue', 'book_red', 'book_green', 'notebook', 'tape_roll', 'pen', 'pencil'):
        assert manifest['assets'][asset_id]['center_of_mass_m'] == [0.0, 0.0, 0.0], asset_id


def test_fixture_assets_are_static_with_base_bottom_origins(manifest):
    for spec in SCENE['fixtures']:
        record = manifest['assets'][spec['asset_id']]
        assert record['dynamic'] is False and record['mass_kg'] == 0.0, spec['id']
        assert np.isclose(record['aabb_m'][0][2], 0.0, atol=1e-9), spec['id']


def test_design_sizes_match_the_authored_library(manifest):
    """The import-time fallback can never silently drift from the real geometry."""
    assert set(manifest['assets']) == set(study.DESIGN_SIZES)
    assert set(study.DYNAMIC_ASSETS) | set(study.FIXTURE_ASSETS) == set(manifest['assets'])
    for asset_id, record in manifest['assets'].items():
        assert np.allclose(study.DESIGN_SIZES[asset_id], record['size_m'], atol=1e-9), asset_id


def test_collision_pieces_are_placed_once(manifest):
    """A piece pre-offset *and* given position_m would be translated twice."""
    for asset_id, record in manifest['assets'].items():
        low = np.asarray(record['aabb_m'][0], dtype=float)
        high = np.asarray(record['aabb_m'][1], dtype=float)
        for part in record['collision_parts']:
            if part['geometry'] == 'convex_mesh':
                assert part['position_m'] == [0., 0., 0.], (asset_id, part['label'])
            points = _part_points(part)
            assert np.all(points.min(axis=0) >= low - 1e-6), (asset_id, part['label'])
            assert np.all(points.max(axis=0) <= high + 1e-6), (asset_id, part['label'])


def test_dynamic_colliders_fill_their_bounding_box(manifest):
    """Each dynamic asset's colliders reach every face of its box: nothing visual-only sticks out."""
    for asset_id in study.DYNAMIC_ASSETS:
        record = manifest['assets'][asset_id]
        points = np.vstack([_part_points(part) for part in record['collision_parts']])
        assert np.allclose(points.min(axis=0), record['aabb_m'][0], atol=6e-4), asset_id
        assert np.allclose(points.max(axis=0), record['aabb_m'][1], atol=6e-4), asset_id


def test_every_visual_records_a_physical_surface_finish(manifest):
    """The loader's metallic 0.65 fallback would render paper, wood and ceramic as chrome."""
    for asset_id, record in manifest['assets'].items():
        for part in record['visual_parts']:
            assert 'roughness' in part and 'metallic' in part, (asset_id, part['label'])
            assert 0. < part['roughness'] <= 1., (asset_id, part['label'])
            if (asset_id, part['label']) in study.METAL_VISUALS:
                assert part['metallic'] >= 0.5, (asset_id, part['label'])
            else:
                assert part['metallic'] == 0.0, (asset_id, part['label'])


def test_the_pinned_library_is_never_overwritten():
    with pytest.raises(FileExistsError):
        study.build_library()


# --- rule 2: the tape roll's through bore and the cup's blind cavity ----------------

def _inside_any(points, parts):
    """Which points lie strictly inside at least one convex or analytic collider part."""
    inside = np.zeros(len(points), dtype=bool)
    for part in parts:
        geometry = part.get('geometry', 'convex_mesh')
        position = np.asarray(part.get('position_m', [0., 0., 0.]), dtype=float)
        rotation = _xyzw_rotation(part.get('quaternion_xyzw', [0., 0., 0., 1.]))
        local = (points - position) @ rotation
        if geometry == 'box':
            inside |= np.all(np.abs(local) < np.asarray(part['half_extents_m']) - 1e-9, axis=1)
        elif geometry == 'cylinder':
            inside |= ((np.hypot(local[:, 0], local[:, 1]) < part['radius_m'] - 1e-9)
                       & (np.abs(local[:, 2]) < part['half_height_m'] - 1e-9))
        else:
            equations = ConvexHull(_mesh_vertices(part)).equations
            inside |= np.all(points @ equations[:, :3].T + equations[:, 3] < -1e-9, axis=1)
    return inside


def _cylinder_samples(radius, z0, z1, count=20000, seed=7):
    rng = np.random.default_rng(seed)
    r = radius * np.sqrt(rng.random(count))
    a = rng.random(count) * 2 * math.pi
    return np.c_[r * np.cos(a), r * np.sin(a), z0 + (z1 - z0) * rng.random(count)]


def _distance_from_axis(points_xy):
    """Distance from the z axis to a convex piece's plan footprint (0 if it covers the axis)."""
    hull = points_xy[ConvexHull(points_xy).vertices]
    edges = list(zip(hull, np.roll(hull, -1, axis=0)))
    if all((b[0] - a[0]) * (-a[1]) - (b[1] - a[1]) * (-a[0]) >= 0 for a, b in edges):
        return 0.
    best = math.inf
    for a, b in edges:
        segment = b - a
        t = np.clip(-(a @ segment) / max(segment @ segment, 1e-18), 0., 1.)
        best = min(best, float(np.linalg.norm(a + t * segment)))
    return best


def test_only_the_tape_roll_declares_a_bore(manifest):
    declared = {asset_id for asset_id, record in manifest['assets'].items()
                if any(part.get('bore') for part in record['collision_parts'])}
    assert declared == {'tape_roll'}
    parts = manifest['assets']['tape_roll']['collision_parts']
    assert len(parts) == study.TAPE_SECTORS >= 8
    for part in parts:
        assert part['geometry'] == 'convex_mesh' and part['bore'] is True
        assert part['bore_radius_m'] == pytest.approx(study.TAPE_INNER_R)
        assert part['clearance_radius_m'] == pytest.approx(study.TAPE_INNER_R)


def test_the_tape_bore_passes_the_shared_strict_rule(manifest):
    """family.validate_object, run on an inline copy of the real record, accepts the ring
    and refuses it once any piece (here a centre box) seals the bore."""
    record = manifest['assets']['tape_roll']
    body = {'origin': 'centre', 'mass_kg': record['mass_kg'], 'size_m': record['size_m'],
            'inertia_diag_kg_m2': record['inertia_diag_kg_m2'],
            'collision_parts': [dict(part) for part in record['collision_parts']]}
    validate_object({'id': 'tape_roll', 'body': body}, family='study')
    sealed = copy.deepcopy(body)
    sealed['collision_parts'].append({'geometry': 'box', 'half_extents_m': [0.01, 0.01, 0.01]})
    with pytest.raises(FamilyError, match='intrudes'):
        validate_object({'id': 'tape_roll', 'body': sealed}, family='study')


def test_the_tape_bore_is_open_in_the_real_meshes(manifest):
    parts = manifest['assets']['tape_roll']['collision_parts']
    radius, half = study.TAPE_INNER_R, study.TAPE_WIDTH / 2.
    for part in parts:
        points = _part_points(part)
        assert _distance_from_axis(points[:, :2]) >= radius - 1e-6, part['label']
    probe = _cylinder_samples(radius - 1e-4, -half - 0.01, half + 0.01)
    assert not _inside_any(probe, parts).any(), 'a collider enters the bore'
    spans = []
    for part in parts:
        xy = _part_points(part)[:, :2]
        angles = np.unwrap(np.sort(np.arctan2(xy[:, 1], xy[:, 0])))
        spans.append(angles.max() - angles.min())
    assert sum(spans) == pytest.approx(2 * math.pi, rel=1e-6), 'the sectors do not close the ring'


def test_the_pen_cup_is_an_open_blind_cavity(manifest):
    """Wall sectors stay out of the cavity, a solid floor closes it, nothing else enters it."""
    record = manifest['assets']['pen_cup']
    assert record['declares_bore'] is False and record['closed_bottom'] and record['open_top']
    radius, floor_z, top_z = record['cavity_radius_m'], record['cavity_floor_z_m'], record['cavity_top_z_m']
    assert radius == pytest.approx(study.CUP_INNER_R)
    walls = [p for p in record['collision_parts'] if p['label'].startswith('wall_')]
    floors = [p for p in record['collision_parts'] if p['label'] == 'floor']
    assert len(walls) == study.CUP_SECTORS and len(floors) == 1
    for part in walls:
        points = _part_points(part)
        assert _distance_from_axis(points[:, :2]) >= radius - 1e-6, part['label']
        assert part['clearance_radius_m'] == pytest.approx(radius)
    floor = floors[0]
    assert floor['geometry'] == 'cylinder' and floor['radius_m'] <= radius + 1e-9
    assert floor['position_m'][2] + floor['half_height_m'] == pytest.approx(floor_z, abs=1e-9)
    probe = _cylinder_samples(radius - 1e-4, floor_z + 1e-4, top_z + 0.01)
    assert not _inside_any(probe, record['collision_parts']).any(), 'a collider enters the cavity'
    below = _cylinder_samples(radius * 0.9, -study.CUP_HEIGHT / 2. + 1e-4, floor_z - 1e-4, count=2000)
    assert _inside_any(below, record['collision_parts']).all(), 'the floor does not close the cup'


# --- rule 3: rest poses, interpenetration, stacking, caging -------------------------

def test_no_object_floats_or_sinks_by_the_validator_rule():
    supports = support_surfaces(SCENE)
    for spec in SCENE['objects']:
        position = np.asarray(spec['position_m'], dtype=float)
        bottom = position[2] - spec['rest_geometry']['size_m'][2] / 2.
        tops = [s['top_z_m'] for s in supports if s['xy_min'][0] <= position[0] <= s['xy_max'][0]
                and s['xy_min'][1] <= position[1] <= s['xy_max'][1]]
        assert tops, spec['id']
        assert abs(bottom - max(tops)) <= 1e-3, (spec['id'], bottom, max(tops))


def test_lowest_collider_point_is_on_the_pad_and_the_footprint_inside_it(manifest):
    """Rule 3 against the real colliders, in each object's own pose; no overhang."""
    pad = next(s for s in support_surfaces(SCENE) if s['id'] == 'desk_pad')
    for spec in SCENE['objects']:
        points = _body_points(manifest, spec)
        assert points[:, 2].min() == pytest.approx(pad['top_z_m'], abs=1e-6), spec['id']
        declared = spec['position_m'][2] - spec['rest_geometry']['size_m'][2] / 2.
        assert declared == pytest.approx(points[:, 2].min(), abs=1e-6), spec['id']
        assert np.all(points[:, :2] >= np.asarray(pad['xy_min']) + 0.002), spec['id']
        assert np.all(points[:, :2] <= np.asarray(pad['xy_max']) - 0.002), spec['id']


def test_nothing_interpenetrates_at_spawn(manifest):
    """Objects, fixtures, static boxes and the room shell touch but never overlap."""
    bodies = {spec['id']: _body_obbs(manifest, spec) for spec in SCENE['objects'] + SCENE['fixtures']}
    ids = list(bodies)
    for i, first in enumerate(ids):
        for second in ids[i + 1:]:
            for a in bodies[first]:
                for b in bodies[second]:
                    assert not _obbs_overlap(a, b), (a['name'], b['name'])
    solids = _static_boxes(SCENE) + _room_colliders(SCENE)
    for boxes in bodies.values():
        for body in boxes:
            for solid in solids:
                assert not _obbs_overlap(body, solid), (body['name'], solid['name'])


def test_no_two_static_boxes_interpenetrate():
    solids = _static_boxes(SCENE) + _room_colliders(SCENE)
    for i, first in enumerate(solids):
        for second in solids[i + 1:]:
            if first['kind'] in ('wall', 'floor') and second['kind'] in ('wall', 'floor'):
                continue          # room.py's own shell boxes meet at the corners
            assert not _obbs_overlap(first, second), (first['name'], second['name'])


def test_nothing_is_stacked_on_a_dynamic_object(manifest):
    assert _stacking_problems(SCENE, manifest) == []


def test_no_static_geometry_rests_on_or_hangs_over_an_object(manifest):
    """The v6 defect: 1 mm static pads lay on three books and pinned them from above."""
    assert _caging_problems(SCENE, manifest) == []
    assert not [s for s in SCENE['surfaces'] if 'contact_pad' in s['id']]


# --- rule 4: thin parts, every collider ------------------------------------------

def test_every_collider_is_at_least_3_mm_thick(manifest):
    assert _thin_problems(SCENE, manifest) == []


def test_thin_objects_use_box_colliders(manifest):
    for asset_id in ('book_blue', 'book_red', 'book_green', 'notebook', 'organiser_tray'):
        assert {p['geometry'] for p in manifest['assets'][asset_id]['collision_parts']} == {'box'}, asset_id


# --- rule 5: masses ---------------------------------------------------------------

def test_masses_are_recorded_design_values(manifest):
    expected = {'pen': 0.008, 'pencil': 0.005, 'notebook': 0.220, 'organiser_tray': 0.180,
                'pen_cup': 0.26, 'tape_roll': 0.10, 'book_blue': 0.62, 'book_red': 0.44, 'book_green': 0.36}
    assert set(expected) == set(study.DYNAMIC_ASSETS)
    for asset_id, mass in expected.items():
        record = manifest['assets'][asset_id]
        assert record['mass_kg'] == pytest.approx(mass), asset_id
        assert record['mass_design_note'].startswith('Design value'), asset_id
        assert record['provenance']['dimension_status'].startswith('simulation task design')


# --- rules 6 and 7: reach, desk and pedestal --------------------------------------

def test_objects_are_inside_the_reach_envelope():
    base = np.asarray(SCENE['robot']['base_position_m'], dtype=float)
    for spec in SCENE['objects']:
        x, y, _ = spec['position_m']
        assert 0.30 <= x <= 0.80 and -0.10 <= y <= 0.45, spec['id']
        assert np.linalg.norm(np.asarray(spec['position_m']) - base) <= 0.80, spec['id']


def test_pedestal_is_clear_of_the_desk_and_carries_the_robot():
    p_half = np.asarray(SCENE['pedestal']['half_extents_m'], dtype=float)
    t_half = np.asarray(SCENE['table']['half_extents_m'], dtype=float)
    t_xy = np.asarray(SCENE['table']['position_m'][:2], dtype=float)
    assert t_xy[0] - t_half[0] >= 0.25
    assert abs(t_xy[0]) >= t_half[0] + p_half[0]
    top = SCENE['pedestal']['position_m'][2] + p_half[2]
    assert SCENE['robot']['base_position_m'][2] == pytest.approx(top)


def test_the_pedestal_reaches_the_floor_and_renders_as_furniture():
    assert _pedestal_problems(SCENE) == []
    skirt = [b for b in _static_boxes(SCENE) if b['name'] == 'pedestal/skirt']
    assert len(skirt) == 1 and skirt[0]['lo'][2] == pytest.approx(0.0, abs=1e-9)
    for name in ('top', 'side'):
        validate_material(SCENE['materials'][name], name)


def test_the_desk_slab_is_built_once():
    """v6 declared the slab as `table` and again as a surface; the support is now a separate pad."""
    assert _duplicate_static_problems(SCENE) == []
    pad = SCENE['surfaces'][0]
    table = SCENE['table']
    assert pad['position_m'][2] - pad['half_extents_m'][2] == pytest.approx(
        table['position_m'][2] + table['half_extents_m'][2], abs=1e-9), 'the pad lies on the slab'
    for axis in (0, 1):
        assert (abs(pad['position_m'][axis] - table['position_m'][axis]) + pad['half_extents_m'][axis]
                <= table['half_extents_m'][axis] + 1e-9)


# --- rule 10: roles, and nothing static in the work volume -------------------------

def test_the_pen_cup_is_a_free_object_and_furniture_is_static():
    assert _role_problems(SCENE) == []
    assert {spec['asset_id'] for spec in SCENE['fixtures']} == set(study.FIXTURE_ASSETS)


def test_nothing_static_occupies_the_work_volume(manifest):
    """The volume over the desk pad, 0.6 m high, is the robot's alone."""
    pad = SCENE['surfaces'][0]
    centre = np.asarray(pad['position_m'], dtype=float) + np.array([0., 0., pad['half_extents_m'][2] + 0.30])
    band = {'centre': centre, 'rotation': np.eye(3),
            'half': np.array([pad['half_extents_m'][0], pad['half_extents_m'][1], 0.30 - 1e-4])}
    for box in _static_boxes(SCENE):
        if box['kind'] in ('prop', 'pedestal'):
            assert not _obbs_overlap(band, box), box['name']
    for spec in SCENE['fixtures']:
        for obb in _body_obbs(manifest, spec):
            assert not _obbs_overlap(band, obb), obb['name']


def test_static_props_sit_inside_the_room_or_inside_an_opening():
    """A part may enter the wall slab only inside a window or door opening."""
    low, high = _interior(SCENE)
    room = validate_room(SCENE['room'])
    t = room['wall_thickness_m']
    openings = []
    for opening in room['openings']:
        wall = _wall_geometry(room, opening['wall'])
        centre = wall['center'] + opening['center_m'][0] * wall['u_axis'] + opening['center_m'][1] * wall['v_axis']
        half = (np.abs(wall['u_axis']) * opening['size_m'][0] / 2. + np.abs(wall['v_axis']) * opening['size_m'][1] / 2.
                + np.abs(wall['normal']) * t / 2.)
        openings.append((centre + wall['normal'] * t / 2. - half, centre + wall['normal'] * t / 2. + half))
    for box in _static_boxes(SCENE):
        inside = np.all(box['lo'] >= low - 1e-9) and np.all(box['hi'] <= high + 1e-9)
        in_opening = any(np.all(box['lo'] >= lo - 1e-9) and np.all(box['hi'] <= hi + 1e-9) for lo, hi in openings)
        assert inside or in_opening, box['name']


# --- support: no floating static geometry ------------------------------------------

def test_every_static_box_and_fixture_is_held(manifest):
    """Each solid box rests on the floor, on another static top, or against a wall or a side face."""
    assert _support_graph_problems(SCENE, manifest) == []


def test_fixtures_stand_on_a_static_top_or_hang_on_a_wall(manifest):
    tops = [box['hi'][2] for box in _static_boxes(SCENE)] + [0.0]
    for spec in SCENE['fixtures']:
        points = _body_points(manifest, spec)
        if spec['asset_id'] == 'wall_clock':
            assert points[:, 1].max() == pytest.approx(study.FRONT_WALL_Y, abs=1e-6)
        else:
            assert any(abs(points[:, 2].min() - top) <= 1e-6 for top in tops), spec['id']


# --- cameras ------------------------------------------------------------------------

def test_cameras_resolve_inside_the_room():
    resolved = cameras_module.resolve_cameras(SCENE)
    assert {c['kind'] for c in resolved.values()} == {'world', 'link'}
    low, high = _interior(SCENE)
    for name, camera in resolved.items():
        if camera['kind'] == 'world':
            eye = np.asarray(camera['eye'], dtype=float)
            assert np.all(eye > low) and np.all(eye < high), (name, eye.tolist())
        else:
            assert camera['mount_link'] == 'right_hand_C_MC'
            assert camera['aim_target_m'] == pytest.approx(list(study.WORK_CENTER))
            assert camera['standoff_m'] > 0


def test_third_person_camera_aims_at_the_desk_centre_from_the_required_place():
    third = cameras_module.resolve_cameras(SCENE)['third_person']
    eye = np.asarray(third['eye'], dtype=float)
    table = SCENE['table']
    centre = np.array([table['position_m'][0], table['position_m'][1],
                       table['position_m'][2] + table['half_extents_m'][2]])
    assert np.allclose(third['target'], centre, atol=1e-9)
    assert 1.5 <= float(np.linalg.norm(eye - centre)) <= 2.2
    assert 0.6 <= eye[2] - centre[2] <= 1.4


def test_both_world_cameras_frame_the_whole_desk_top():
    assert _desk_framing_problems(SCENE) == []


def test_every_object_is_in_frame_and_visible_with_the_arm_at_home(manifest, arm_capsules):
    occluders = _occluders(SCENE, manifest)
    for name, camera in cameras_module.resolve_cameras(SCENE).items():
        if camera['kind'] != 'world':
            continue
        for spec in SCENE['objects']:
            samples = [_face_samples(obb) for obb in _body_obbs(manifest, spec)]
            points = np.vstack([p for p, _ in samples])
            normals = np.vstack([n for _, n in samples])
            in_frame, visible = _visibility(camera, points, normals, {spec['id']}, occluders, arm_capsules)
            assert in_frame >= 0.99, (name, spec['id'], in_frame)
            assert visible >= 0.50, (name, spec['id'], visible)


def test_the_signature_furniture_is_inside_the_third_person_frustum(manifest, arm_capsules):
    assert _frustum_problems(SCENE, manifest, arm_capsules) == []


def test_no_static_prop_blocks_a_camera_from_the_work_area():
    resolved = cameras_module.resolve_cameras(SCENE)
    targets = [np.asarray(spec['position_m'], dtype=float) for spec in SCENE['objects']]
    targets.append(np.asarray(study.WORK_CENTER, dtype=float))
    props = [box for box in _static_boxes(SCENE) if box['kind'] == 'prop']
    for name, camera in resolved.items():
        if camera['kind'] != 'world':
            continue
        for box in props:
            blocked = _segments_hit_box(np.asarray(camera['eye'], dtype=float), np.asarray(targets), box)
            assert not blocked.any(), (name, box['name'])


def test_wrist_mount_link_exists_and_home_clears_the_objects():
    """CPU forward kinematics of the pinned assembly at the scene's home pose."""
    from bench.control.fingertip import Fr3SharpaGeometry
    from bench.geometry import quaternion_matrix
    robot = SCENE['robot']
    geometry = Fr3SharpaGeometry(robot['assets_root'], robot['base_position_m'], hand_id=robot['hand_id'],
                                 assembly_version=robot['assembly_version'])
    joints = {joint.name: 0.0 for joint in geometry.hand.active}
    joints.update({f'fr3_joint{i}': float(v) for i, v in enumerate(robot['arm_home_rad'], start=1)})
    links = geometry.hand.forward(joints)
    assert cameras_module.resolve_cameras(SCENE)['wrist']['mount_link'] in links
    pose = geometry.public_geometry(joints)
    wrist = np.eye(4)
    wrist[:3, :3] = quaternion_matrix(pose['wrist']['quaternion_wxyz'])
    wrist[:3, 3] = pose['wrist']['position_m']
    lowest = min([float((wrist @ transform)[2, 3]) for transform in links.values()] +
                 [float((wrist @ np.r_[np.asarray(tip, dtype=float), 1.])[2])
                  for tip in pose['fingertip_positions_m'].values()])
    tallest = max(spec['position_m'][2] + spec['rest_geometry']['size_m'][2] / 2. for spec in SCENE['objects'])
    assert lowest - tallest >= 0.10, (lowest, tallest)


# --- the room ---------------------------------------------------------------------

def test_the_room_is_a_closed_shell_with_openings():
    room = validate_room(SCENE['room'])
    assert {'floor', 'wall', 'ceiling'} <= set(room['surfaces'])
    assert room['collide']['floor'] and room['collide']['walls']
    assert {o['kind'] for o in room['openings']} == {'door', 'window'}
    for face in WALL_FACES:
        wall = _wall_geometry(room, face)
        holes = [o for o in room['openings'] if o['wall'] == face]
        quads = _wall_quads(wall, holes)
        assert quads, face
        area = sum((hu - lu) * (hv - lv) for lu, hu, lv, hv in quads)
        full = 4. * wall['half'][0] * wall['half'][1]
        assert area == pytest.approx(full - sum(o['size_m'][0] * o['size_m'][1] for o in holes)), face
        for opening in holes:
            u, v = opening['center_m']
            assert abs(u) + opening['size_m'][0] / 2. <= wall['half'][0] + 1e-9, opening['id']
            assert -1e-9 <= room['ceiling_height_m'] / 2. + v - opening['size_m'][1] / 2.
            assert room['ceiling_height_m'] / 2. + v + opening['size_m'][1] / 2. <= room['ceiling_height_m'] + 1e-9


def test_openings_are_dressed_where_they_are():
    """The window frame and the door leaf sit in their own openings' reveals."""
    room = validate_room(SCENE['room'])
    by_id = {o['id']: o for o in room['openings']}
    boxes = {box['name']: box for box in _static_boxes(SCENE)}
    window, door = by_id['window'], by_id['door']
    assert window['center_m'][0] == pytest.approx(sum(study.WINDOW_X) / 2.)
    assert door['center_m'][0] == pytest.approx(-sum(study.DOOR_X) / 2.)
    stile = boxes['window_frame/stile_left']
    assert stile['lo'][0] == pytest.approx(study.WINDOW_X[0]) and stile['lo'][1] > study.FRONT_WALL_Y
    leaf = boxes['door/leaf']
    assert leaf['lo'][0] == pytest.approx(study.DOOR_X[0]) and leaf['hi'][0] == pytest.approx(study.DOOR_X[1])
    assert leaf['hi'][1] < study.BACK_WALL_Y and leaf['lo'][2] == pytest.approx(0.0)


def test_ceiling_lights_dome_and_the_desk_lamp_bulb(manifest):
    room = validate_room(SCENE['room'])
    assert room['dome']['enabled'] and room['dome']['intensity'] > 0
    low, high = _interior(SCENE)
    lights = {light['id']: light for light in room['ceiling_lights']}
    for light in lights.values():
        assert np.all(np.asarray(light['position_m']) > low) and np.all(np.asarray(light['position_m']) < high)
        assert light['intensity'] > 0
    assert sum(light['shape'] == 'rect' for light in lights.values()) >= 2
    bulb = lights['desk_lamp_bulb']
    assert bulb['shape'] == 'sphere'
    lamp = next(spec for spec in SCENE['fixtures'] if spec['id'] == 'desk_lamp')
    record = manifest['assets']['desk_lamp']
    expected = _rotation(lamp['quaternion_wxyz']) @ np.asarray(record['bulb_local_m']) + np.asarray(lamp['position_m'])
    assert np.allclose(bulb['position_m'], expected, atol=1e-6)
    assert bulb['extent_m'][0] == pytest.approx(record['bulb_radius_m'])
    # Inside the open shade: under its top, above its mouth, and inside its cone.
    local = np.asarray(record['bulb_local_m'])
    radius = record['bulb_radius_m']
    (r_bottom, r_top), (z0, z1) = study.LAMP_SHADE_R, study.LAMP_SHADE_Z
    assert z0 < local[2] - radius and local[2] + radius < z1
    cone = r_bottom + (r_top - r_bottom) * (local[2] + radius - z0) / (z1 - z0) - study.LAMP_SHADE_WALL
    assert math.hypot(local[0] - study.LAMP_SHADE_X, local[1]) + radius < cone


def test_materials_are_valid_and_textures_are_real_cc0_files():
    for name, spec in SCENE['materials'].items():
        assert name.isidentifier(), name
        normalised = validate_material(spec, name)
        if spec['kind'] == 'textured':
            # The colour map is always bound; the shared validator also binds the
            # '<slot>_map' roughness and normal maps. No scalar 'roughness' is set,
            # because an older validate_material read that key as a map path.
            assert 'base_color' in normalised['textures'], name
            assert set(normalised['textures']) <= {'base_color', 'roughness', 'normal'}, name
            assert 'roughness' not in spec, name
            for key in ('base_color', 'roughness_map', 'normal_map'):
                assert Path(spec[key]).is_file(), (name, key)
            for path in normalised['textures'].values():
                assert Path(path).is_file(), (name, path)
            assert spec['license'] == 'CC0-1.0' and spec['source'].startswith('https://ambientcg.com/')
    textured = {name for name, spec in SCENE['materials'].items() if spec['kind'] == 'textured'}
    assert {SCENE['room']['surfaces']['floor'], SCENE['room']['surfaces']['wall'], SCENE['table']['material']} <= textured


def test_every_referenced_material_is_defined():
    defined = set(SCENE['materials'])
    referenced = set(SCENE['room']['surfaces'].values()) | {SCENE['table']['material'], 'top', 'side'}
    for key in ('surfaces', 'static_props'):
        for spec in SCENE[key]:
            referenced.add(spec['material'])
            referenced.update(part['material'] for part in spec.get('parts', []))
    assert referenced <= defined, sorted(referenced - defined)


def test_every_prim_name_is_a_valid_identifier():
    for key in ('objects', 'fixtures', 'surfaces', 'static_props'):
        for spec in SCENE[key]:
            assert spec['id'].isidentifier(), (key, spec['id'])
            parts = [part['id'] for part in spec.get('parts', [])]
            assert all(p.isidentifier() for p in parts) and len(parts) == len(set(parts)), (key, spec['id'])
    for light in SCENE['room']['ceiling_lights']:
        assert light['id'].isidentifier(), light['id']
    ids = [spec['id'] for key in ('objects', 'fixtures', 'surfaces', 'static_props') for spec in SCENE[key]]
    assert len(ids) == len(set(ids))


# --- mutation tests: each check rejects the defect it guards against ----------------

def test_caging_check_rejects_the_v6_book_pad(manifest):
    """Re-create v6's 1 mm static pad on top of book_blue: the caging check must fail."""
    broken = copy.deepcopy(SCENE)
    book = next(spec for spec in broken['objects'] if spec['id'] == 'book_blue')
    top = book['position_m'][2] + book['rest_geometry']['size_m'][2] / 2.
    broken['surfaces'].append({'id': 'book_blue_contact_pad', 'material': 'paper',
                               'position_m': [book['position_m'][0], book['position_m'][1], top + 0.0005],
                               'half_extents_m': [0.02, 0.03, 0.0005]})
    problems = _caging_problems(broken, manifest)
    assert any('book_blue_contact_pad' in p and 'book_blue' in p for p in problems), problems
    assert any('book_blue_contact_pad' in p for p in _thin_problems(broken, manifest))


def test_support_check_rejects_a_floating_pedestal_and_lifted_props(manifest):
    no_skirt = copy.deepcopy(SCENE)
    del no_skirt['pedestal']['skirt_height_m']
    assert 'pedestal/top is held by nothing' in ' '.join(_support_graph_problems(no_skirt, manifest))
    assert _pedestal_problems(no_skirt)
    for prop_ids, lift in ((('chair', 'paper_stack'), 0.6), (('rug',), 1.0),
                           (('bookcase', 'bookcase_books'), 0.3)):
        lifted = copy.deepcopy(SCENE)
        for prop in lifted['static_props']:
            if prop['id'] in prop_ids:
                for part in prop['parts']:
                    part['position_m'][2] += lift
        for spec in lifted['fixtures']:
            if 'bookcase' in prop_ids and spec['id'] == 'potted_plant':
                spec['position_m'][2] += lift
        problems = _support_graph_problems(lifted, manifest)
        assert any(p.startswith(f'{prop_ids[0]}/') for p in problems), (prop_ids, problems)
    # A floor bookcase pushed up against the wall is still floating: the wall does not hold it.
    against_wall = copy.deepcopy(SCENE)
    for prop in against_wall['static_props']:
        if prop['id'] in ('bookcase', 'bookcase_books'):
            for part in prop['parts']:
                part['position_m'][2] += 0.3
    for spec in against_wall['fixtures']:
        if spec['id'] == 'potted_plant':
            spec['position_m'][2] += 0.3
    assert any('bookcase/' in p and 'rests on nothing' in p
               for p in _support_graph_problems(against_wall, manifest))


def test_pedestal_check_rejects_missing_furniture_materials():
    broken = copy.deepcopy(SCENE)
    del broken['materials']['top']
    del broken['materials']['side']
    assert len(_pedestal_problems(broken)) == 2


def test_duplicate_check_rejects_a_second_desk_slab():
    broken = copy.deepcopy(SCENE)
    broken['surfaces'].append({'id': 'desk_work_top', 'material': 'desk_wood',
                               'position_m': list(SCENE['table']['position_m']),
                               'half_extents_m': list(SCENE['table']['half_extents_m'])})
    assert _duplicate_static_problems(broken)


def test_frustum_check_rejects_a_missing_or_unframed_bookcase(manifest, arm_capsules):
    removed = copy.deepcopy(SCENE)
    removed['static_props'] = [p for p in removed['static_props'] if not p['id'].startswith('bookcase')]
    assert any(p.startswith('bookcase:') for p in _frustum_problems(removed, manifest, arm_capsules))
    moved = copy.deepcopy(SCENE)
    for prop in moved['static_props']:
        if prop['id'].startswith('bookcase'):
            for part in prop['parts']:
                part['position_m'][1] -= 3.6          # against the back wall, behind the camera
    assert any(p.startswith('bookcase:') for p in _frustum_problems(moved, manifest, arm_capsules))


def test_framing_check_rejects_a_desk_view_that_cuts_off_the_desk():
    broken = copy.deepcopy(SCENE)
    camera = next(c for c in broken['cameras'] if c['id'] == 'desk_view')
    camera['focal_length_mm'] = 30.0
    assert any(p.startswith('desk_view') for p in _desk_framing_problems(broken))


def test_role_check_rejects_a_static_pen_cup():
    broken = copy.deepcopy(SCENE)
    cup = next(spec for spec in broken['objects'] if spec['asset_id'] == 'pen_cup')
    broken['objects'].remove(cup)
    broken['fixtures'].append(cup)
    assert _role_problems(broken)


def test_stacking_check_rejects_a_book_on_a_book(manifest):
    broken = copy.deepcopy(SCENE)
    red = next(spec for spec in broken['objects'] if spec['id'] == 'book_red')
    blue = next(spec for spec in broken['objects'] if spec['id'] == 'book_blue')
    red['position_m'][:2] = blue['position_m'][:2]
    assert _stacking_problems(broken, manifest)


def test_the_pen_rests_on_a_flat_face_and_cannot_roll(manifest):
    """v7's pen was a round analytic cylinder: natively it rolled 6.7 mm (1.92 rad).

    In its spawn pose the pen's collider must touch the desk with a FACE (four
    or more vertices on the lowest plane, spread across the barrel), not a line,
    and the centre of mass must sit over that face, so it cannot start rolling.
    """
    record = manifest['assets']['pen']
    pieces = record['collision_parts']
    assert [p['geometry'] for p in pieces] == ['convex_mesh'], 'the pen must not be a round analytic cylinder'
    spec = next(s for s in SCENE['objects'] if s['id'] == 'pen')
    w, x, y, z = spec['quaternion_wxyz']
    rotation = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                         [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                         [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])
    points = np.vstack([_part_points(piece) for piece in pieces]) @ rotation.T
    lowest = points[:, 2].min()
    face = points[points[:, 2] < lowest + 1e-6]
    assert len(face) >= 4, f'the pen touches the desk on {len(face)} vertices (a line), not a face'
    across = np.ptp(face[:, 0])
    assert across > 0.003, f'contact face only {across * 1e3:.2f} mm wide across the barrel'
    com = rotation @ np.asarray(record['center_of_mass_m'], dtype=float)
    assert face[:, 0].min() < com[0] < face[:, 0].max(), 'the centre of mass is not over the contact face'
