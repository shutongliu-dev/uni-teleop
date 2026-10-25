"""CPU contract tests for the ``card_room`` scene family.

No GPU and no Isaac Sim. Besides the data checks, a recording stand-in for
OpenUSD (``_stub_pxr``: a numpy ``Gf.Matrix4d`` with Gf's row-vector
conventions, every other call logged) drives the *real* shared build path -
``visuals.textured_material``, ``room.add_room``, ``scene.add_structured_scene``
and ``scene.add_task_asset`` - so the materials, room colliders, static colliders
and object poses that the GPU build would author are checked here too.

Regression checks cover the defects found in the earlier build: textured
materials that crash the GPU build, openings measured from the floor instead of
the wall centre, trim that frames nothing, duplicated boxes, a bottomless cup
with a 3 mm collision skin, empty camera views, floating furniture, a layout
re-solved at import, a wrong manifest digest, and a press-fit ring bore.
"""
from __future__ import annotations

import hashlib
import itertools
import math
import re
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from bench.paths import load_json
from scenes.rooms import card_room as family
from bench.simulation.isaacsim_backend import room as room_module
from bench.simulation.isaacsim_backend.cameras import resolve_cameras
from bench.simulation.isaacsim_backend.scene_family import support_surfaces, uniform_box_inertia, validate_family
from bench.simulation.isaacsim_backend.room import validate_room
from bench.simulation.isaacsim_backend.visuals import TEXTURE_KEYS, validate_material

SCENE = family.SCENE
ROOT = Path(family.ASSET_ROOT)
MANIFEST_PATH = ROOT / 'manifest.json'
INTERIOR_LO = np.array([-family.ROOM_HALF_X, -family.ROOM_HALF_Y, 0.0])
INTERIOR_HI = np.array([family.ROOM_HALF_X, family.ROOM_HALF_Y, family.ROOM_HEIGHT])
APERTURE_H_MM = 20.955                  # USD camera default; env.py does not override it
DISPLAY_TOLERANCE_M = 0.0005


# --- a recording stand-in for OpenUSD -------------------------------------------

class _Matrix4d:
    """Gf.Matrix4d: row-major storage, row vectors (p' = p M), A * B = A then B."""

    def __init__(self, *args):
        if not args:
            self.m = np.eye(4)
        elif len(args) == 1 and np.isscalar(args[0]):
            self.m = np.eye(4) * float(args[0])
        elif len(args) == 16:
            self.m = np.asarray(args, dtype=float).reshape(4, 4)
        else:
            raise TypeError(args)

    def SetTranslateOnly(self, v):
        self.m[3, :3] = np.asarray(v, dtype=float)
        return self

    def SetScale(self, v):
        self.m = np.diag([*np.asarray(v, dtype=float), 1.0])
        return self

    def __mul__(self, other):
        out = _Matrix4d()
        out.m = self.m @ other.m
        return out


class _Tokens:
    def __getattr__(self, name):
        return name


class _Prim:
    def __init__(self, log, path, kind):
        self.log, self.path, self.kind = log, path, kind

    def GetPrim(self):
        return self

    def GetPath(self):
        return self.path

    def __getattr__(self, name):
        def call(*args, **kwargs):
            self.log.append((self.path, self.kind, name, args))
            return _Prim(self.log, self.path, name)
        return call


class _Schema:
    def __init__(self, log, kind):
        self.log, self.kind = log, kind

    def Define(self, stage, path):
        self.log.append((str(path), self.kind, 'Define', ()))
        return _Prim(self.log, str(path), self.kind)

    def Apply(self, prim, *args):
        self.log.append((prim.path, self.kind, 'Apply', args))
        return _Prim(self.log, prim.path, self.kind)

    def __call__(self, prim=None, *args):
        return _Prim(self.log, getattr(prim, 'path', None), self.kind)


class _SchemaModule(types.ModuleType):
    def __init__(self, name, log):
        super().__init__(name)
        self._log = log

    def __getattr__(self, name):
        if name.startswith('__'):
            raise AttributeError(name)
        return _Tokens() if name == 'Tokens' else _Schema(self._log, name)


def _stub_pxr(log):
    gf = types.ModuleType('pxr.Gf')
    gf.Matrix4d = _Matrix4d
    for name in ('Vec2f', 'Vec3d', 'Vec3f', 'Vec4f'):
        setattr(gf, name, lambda *a: np.asarray(a, dtype=float))
    gf.Quatf = gf.Quatd = lambda w, v=(0., 0., 0.): np.asarray([w, *np.asarray(v, dtype=float)])
    sdf = types.ModuleType('pxr.Sdf')
    sdf.ValueTypeNames, sdf.Path = _Tokens(), str
    vt = types.ModuleType('pxr.Vt')
    array = type('Array', (), {'FromNumpy': staticmethod(lambda a: np.asarray(a))})
    for name in ('Vec3fArray', 'IntArray', 'Vec2fArray', 'FloatArray'):
        setattr(vt, name, array)
    pxr = types.ModuleType('pxr')
    pxr.Gf, pxr.Sdf, pxr.Vt = gf, sdf, vt
    for name in ('Usd', 'UsdGeom', 'UsdShade', 'UsdLux', 'UsdPhysics', 'PhysxSchema'):
        setattr(pxr, name, _SchemaModule('pxr.' + name, log))
    return pxr


class _StubbedUsd:
    """Install the stub as ``pxr`` (and a fresh ``scene`` module) for one block."""

    SCENE_MODULE = 'bench.simulation.isaacsim_backend.scene'

    def __enter__(self):
        self.log = []
        self.saved = {name: sys.modules.get(name) for name in ('pxr', self.SCENE_MODULE)}
        sys.modules['pxr'] = _stub_pxr(self.log)
        sys.modules.pop(self.SCENE_MODULE, None)
        return self

    def scene(self):
        import importlib
        return importlib.import_module(self.SCENE_MODULE)

    def __exit__(self, *exc):
        for name, module in self.saved.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module
        return False


def _quat_matrix(q):
    w, x, y, z = (float(v) for v in q)
    n = math.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def _prim_records(log):
    """path -> kind, composed local matrix (row vectors) and authored geometry."""
    records = {}
    for path, kind, name, args in log:
        record = records.setdefault(path, {'kind': None, 'ops': [], 'collider': False})
        if name == 'Define':
            record['kind'] = kind
        elif name == 'ClearXformOpOrder':
            record['ops'] = []
        elif name == 'Set' and kind in ('AddTranslateOp', 'AddOrientOp', 'AddScaleOp', 'AddTransformOp'):
            record['ops'].append((kind, args[0]))
        elif name == 'CreatePointsAttr':
            record['points'] = np.asarray(args[0], dtype=float)
        elif name == 'CreateSizeAttr':
            record['size'] = float(args[0])
        elif name == 'CreateRadiusAttr':
            record['radius'] = float(args[0])
        elif name == 'CreateHeightAttr':
            record['height'] = float(args[0])
        elif name == 'Apply' and kind == 'CollisionAPI':
            record['collider'] = True
    for record in records.values():
        local = np.eye(4)
        for kind, value in record['ops']:     # xformOpOrder: the first op is outermost
            op = np.eye(4)
            if kind == 'AddTranslateOp':
                op[3, :3] = value
            elif kind == 'AddOrientOp':
                op[:3, :3] = _quat_matrix(value).T
            elif kind == 'AddScaleOp':
                op[:3, :3] = np.diag(value)
            else:
                op = value.m
            local = op @ local
        record['local'] = local
    return records


def _world_points(records, path):
    record = records[path]
    if record['kind'] == 'Cube':
        s = record.get('size', 2.) / 2.
        points = np.array([[x, y, z] for x in (-s, s) for y in (-s, s) for z in (-s, s)])
    elif record['kind'] == 'Cylinder':
        angles = np.linspace(0., 2. * np.pi, 96, endpoint=False)
        ring = np.c_[record['radius'] * np.cos(angles), record['radius'] * np.sin(angles)]
        points = np.vstack([np.c_[ring, np.full(96, -record['height'] / 2.)],
                            np.c_[ring, np.full(96, record['height'] / 2.)]])
    else:
        points = record['points']
    matrix = record['local']
    parent = path.rsplit('/', 1)[0]
    while parent:
        if parent in records:
            matrix = matrix @ records[parent]['local']
        parent = parent.rsplit('/', 1)[0]
    return (np.c_[points, np.ones(len(points))] @ matrix)[:, :3]


def _aabb(points):
    return points.min(axis=0), points.max(axis=0)


def _collider_boxes(records):
    return {path: _aabb(_world_points(records, path)) for path, record in records.items()
            if path.endswith('/collider') and record['kind'] == 'Cube'}


def _overlap_volume(a, b, tol=1e-6):
    (alo, ahi), (blo, bhi) = a, b
    return bool(np.all(np.minimum(ahi, bhi) - np.maximum(alo, blo) > tol))


# --- fixtures ------------------------------------------------------------------------

@pytest.fixture(scope='module')
def manifest():
    if not MANIFEST_PATH.is_file():
        pytest.fail(f'the authored asset library is missing: {MANIFEST_PATH}; run '
                    f'card_room.build_library() first')
    return load_json(MANIFEST_PATH)


@pytest.fixture(scope='module')
def built(manifest):
    """The stage that scene.py would author for this family, recorded on CPU."""
    with _StubbedUsd() as usd:
        scene = usd.scene()
        room_spec = validate_room(SCENE['room'])
        structured = scene.add_structured_scene(None, SCENE, room_spec)
        objects = {spec['id']: scene.add_task_asset(None, spec, dynamic=True) for spec in SCENE['objects']}
        fixtures = {spec['id']: scene.add_task_asset(None, spec, dynamic=False)
                    for spec in SCENE['fixtures']}
        log = list(usd.log)
    return {'structured': structured, 'objects': objects, 'fixtures': fixtures,
            'records': _prim_records(log)}


def _static_boxes():
    """Every immovable box the family declares: (name, centre, half)."""
    boxes = []
    table = SCENE['table']
    boxes.append(('table', np.array(table['position_m']), np.array(table['half_extents_m'])))
    for spec in SCENE['surfaces']:
        boxes.append((f'surface/{spec["id"]}', np.array(spec['position_m']), np.array(spec['half_extents_m'])))
    for prop in SCENE['static_props']:
        for part in prop['parts']:
            boxes.append((f'{prop["id"]}/{part["id"]}', np.array(part['position_m']),
                          np.array(part['half_extents_m'])))
    pedestal = SCENE['pedestal']
    centre, half = np.array(pedestal['position_m']), np.array(pedestal['half_extents_m'])
    boxes.append(('pedestal/top', centre, half))
    skirt = float(pedestal.get('skirt_height_m', 0.))
    if skirt:
        boxes.append(('pedestal/skirt', centre - [0., 0., half[2] + skirt / 2.],
                      np.array([half[0], half[1], skirt / 2.])))
    return boxes


def _collision_aabb(record):
    """Body-frame AABB of an asset's collider union, from its manifest record."""
    lo, hi = np.full(3, np.inf), np.full(3, -np.inf)
    for part in record['collision_parts']:
        offset = np.asarray(part.get('position_m', [0., 0., 0.]), dtype=float)
        if 'aabb_m' in part:
            box = np.asarray(part['aabb_m'], dtype=float) + offset
        elif part['geometry'] == 'box':
            half = np.asarray(part['half_extents_m'], dtype=float)
            box = np.array([offset - half, offset + half])
        elif part['geometry'] == 'cylinder':
            r, h = part['radius_m'], part['half_height_m']
            box = np.array([offset - [r, r, h], offset + [r, r, h]])
        else:
            raise AssertionError(f'unexpected collision geometry {part["geometry"]}')
        lo, hi = np.minimum(lo, box[0]), np.maximum(hi, box[1])
    return lo, hi


def _visual_aabb(record):
    boxes = np.array([part['aabb_m'] for part in record['visual_parts']], dtype=float)
    return boxes[:, 0].min(axis=0), boxes[:, 1].max(axis=0)


def _load_obj(relative):
    from bench.simulation.assets.task_parts import load_obj_mesh
    vertices, faces = load_obj_mesh(ROOT / relative)
    return np.asarray(vertices, dtype=float), np.asarray(faces)


def _ray_hits(vertices, faces, origin, direction):
    """Sorted distances at which a ray meets a triangle mesh (Moller-Trumbore)."""
    tri = vertices[faces]
    v0, e1, e2 = tri[:, 0], tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]
    d, o = np.asarray(direction, dtype=float), np.asarray(origin, dtype=float)
    p = np.cross(d, e2)
    det = np.einsum('ij,ij->i', e1, p)
    ok = np.abs(det) > 1e-14
    inv = np.where(ok, 1. / np.where(ok, det, 1.), 0.)
    s = o - v0
    u = np.einsum('ij,ij->i', s, p) * inv
    q = np.cross(s, e1)
    v = (q @ d) * inv
    t = np.einsum('ij,ij->i', e2, q) * inv
    good = ok & (u >= -1e-12) & (v >= -1e-12) & (u + v <= 1 + 1e-12) & (t > 1e-12)
    return np.unique(np.round(t[good], 9))


def _footprint(spec):
    """World xy footprint of an object's collider: a disc or a convex polygon."""
    asset_id = spec['asset_id']
    rotation = _quat_matrix(spec['quaternion_wxyz'])
    centre = np.asarray(spec['position_m'][:2], dtype=float)
    half = np.asarray(family.COLLIDER_HALF_M[asset_id], dtype=float)
    if asset_id in family.ROUND_ASSETS and abs(abs(rotation[2, 2]) - 1.) < 1e-9:
        return ('disc', centre, float(half[0]))
    corners = np.array([[sx * half[0], sy * half[1], sz * half[2]]
                        for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]) @ rotation.T
    return ('polygon', _hull(corners[:, :2] + centre), None)


def _hull(points):
    points = sorted(set(map(tuple, np.round(points, 12))))
    if len(points) <= 2:
        return np.array(points)
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    lower, upper = [], []
    for p in points:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    for p in reversed(points):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return np.array(lower[:-1] + upper[:-1])


def _rect(centre, half):
    cx, cy = centre
    hx, hy = half
    return ('polygon', np.array([[cx - hx, cy - hy], [cx + hx, cy - hy], [cx + hx, cy + hy],
                                 [cx - hx, cy + hy]]), None)


def _polygon_gap(a, b):
    """Separating distance of two convex polygons (negative when they overlap)."""
    best = -np.inf
    for p, q in ((a, b), (b, a)):
        for i in range(len(p)):
            edge = p[(i + 1) % len(p)] - p[i]
            normal = np.array([edge[1], -edge[0]]) / np.linalg.norm(edge)
            best = max(best, (q @ normal).min() - (p @ normal).max())
    return best


def _cross2(a, b):
    return float(a[0] * b[1] - a[1] * b[0])


def _disc_polygon_gap(centre, radius, polygon):
    inside = all(_cross2(polygon[(i + 1) % len(polygon)] - polygon[i], centre - polygon[i]) >= 0
                 for i in range(len(polygon)))
    distance = min(_segment_distance(centre, polygon[i], polygon[(i + 1) % len(polygon)])
                   for i in range(len(polygon)))
    return -distance - radius if inside else distance - radius


def _segment_distance(point, a, b):
    t = np.clip(np.dot(point - a, b - a) / np.dot(b - a, b - a), 0., 1.)
    return float(np.linalg.norm(point - (a + t * (b - a))))


def _gap(a, b):
    if a[0] == 'disc' and b[0] == 'disc':
        return float(np.linalg.norm(a[1] - b[1]) - a[2] - b[2])
    if a[0] == 'disc':
        return _disc_polygon_gap(a[1], a[2], b[1])
    if b[0] == 'disc':
        return _disc_polygon_gap(b[1], b[2], a[1])
    return _polygon_gap(a[1], b[1])


def _z_range(spec):
    half = family.collider_world_half(spec['asset_id'], spec['quaternion_wxyz'])
    return spec['position_m'][2] - half[2], spec['position_m'][2] + half[2]


def _project(camera, points):
    """Pixel coordinates in a world camera, as cameras.project_point computes them."""
    eye, target = np.asarray(camera['eye'], dtype=float), np.asarray(camera['target'], dtype=float)
    forward = (target - eye) / np.linalg.norm(target - eye)
    right = np.cross(forward, [0., 0., 1.])
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    width, height = camera['resolution']
    vertical = APERTURE_H_MM * height / width            # sensors.calibrate_camera
    delta = np.asarray(points, dtype=float) - eye
    depth = delta @ forward
    x = (delta @ right) / depth * camera['focal_length_mm']
    y = (delta @ up) / depth * camera['focal_length_mm']
    px, py = width / 2. + x / APERTURE_H_MM * width, height / 2. - y / vertical * height
    margin_x, margin_y = 0.02 * width, 0.02 * height
    inside = (depth > 0) & (px >= margin_x) & (px <= width - margin_x) & (py >= margin_y) & (py <= height - margin_y)
    return inside


def _box_samples(parts):
    samples = []
    for part in parts:
        centre, half = np.asarray(part['position_m']), np.asarray(part['half_extents_m'])
        samples.append(centre)
        samples.extend(centre + half * [sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1))
    return np.array(samples)


def _prop(prop_id):
    return next(prop for prop in SCENE['static_props'] if prop['id'] == prop_id)


# --- the family ----------------------------------------------------------------------

def test_family_validates():
    record = validate_family(SCENE)
    assert record['family'] == 'card_room'
    assert record['objects'] == len(SCENE['objects']) == len(family.LAYOUT)
    assert record['static_props'] == len(SCENE['static_props'])
    assert {s['id'] for s in record['supports']} == {'felt_top', 'xiangqi_board'}


def test_schema_and_family_name():
    assert SCENE['schema_version'] == 'isaacsim-scene-family-v1'
    assert family.FAMILY == SCENE['family'] == 'card_room'
    assert SCENE['title_zh'] and SCENE['description_zh']


def test_ids_are_unique_identifiers_and_usd_paths_do_not_collide():
    object_ids = [spec['id'] for spec in SCENE['objects']]
    assert len(object_ids) == len(set(object_ids)) and all(name.isidentifier() for name in object_ids)
    # scene.py builds the table, the surfaces and the static props under /World/Props/<id>.
    prop_ids = ['work_table'] + [s['id'] for s in SCENE['surfaces']] + [p['id'] for p in SCENE['static_props']]
    assert len(prop_ids) == len(set(prop_ids))
    for prop in SCENE['static_props']:
        part_ids = [part['id'] for part in prop['parts']]
        assert len(part_ids) == len(set(part_ids)), prop['id']


def test_layout_is_frozen_data_that_never_reads_the_library(monkeypatch):
    """The initial state must not depend on which manifest happens to be on disk."""
    def unavailable(*args, **kwargs):
        raise AssertionError('building the scene consulted the asset library')
    monkeypatch.setattr(family, 'load_manifest', unavailable)
    monkeypatch.setattr(family, 'verify_library', unavailable)
    assert family._build_objects() == SCENE['objects']
    assert not hasattr(family, 'plan_placement') and not hasattr(family, 'asset_sizes')
    for (object_id, asset_id, support, x, y, yaw, pose), spec in zip(family.LAYOUT, SCENE['objects']):
        assert spec['id'] == object_id and spec['asset_id'] == asset_id
        assert spec['position_m'][:2] == [x, y], 'positions are the literal layout table'


def test_manifest_digest_is_pinned_and_verified(manifest):
    digest = hashlib.sha256(MANIFEST_PATH.read_bytes()).hexdigest()
    assert family.MANIFEST_SHA256 == digest
    assert family.verify_library() == digest
    with pytest.raises(ValueError):
        family.verify_library(expected_sha256='0' * 64)
    with pytest.raises(FileNotFoundError):
        family.verify_library(asset_root=ROOT.parent / 'v_missing')
    with pytest.raises(FileNotFoundError):
        family.load_manifest(asset_root=ROOT.parent / 'v_missing')
    assert manifest['asset_version'] == family.ASSET_VERSION == ROOT.name
    assert all(spec['asset_root'] == family.ASSET_ROOT for spec in SCENE['objects'] + SCENE['fixtures'])


def test_manifest_records_family_license_and_texture_hashes(manifest):
    assert manifest['schema_version'] == 'scene_family_asset_library_v1'
    assert manifest['family'] == 'card_room' and manifest['units'] == 'metres'
    # The generator is the module path at build time (tests/test_room_libraries.py
    # checks that this module still rebuilds the pinned library).
    assert manifest['license'] == 'CC0-1.0' and manifest['generator'].endswith('/card_room.py')
    for relative, record in manifest['textures'].items():
        assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == record['sha256']
        assert record['license'] == 'CC0-1.0'
    assert family.BOARD_TEXTURE in manifest['textures']
    assert manifest['textures'][family.BOARD_TEXTURE]['board_size_m'] == list(family.BOARD_SIZE_XY)


# --- the authored assets -----------------------------------------------------------

def test_every_object_and_fixture_asset_resolves_with_verified_hashes(manifest):
    from bench.simulation.isaacsim_backend.asset_catalog import load_task_asset
    for spec in SCENE['objects']:
        asset = load_task_asset(spec['asset_id'], asset_root=spec['asset_root'], verify_hashes=True)
        assert asset['dynamic'] is True, spec['id']
        assert asset['visual_parts'] and asset['collision_parts'], spec['id']
    for spec in SCENE['fixtures']:
        asset = load_task_asset(spec['asset_id'], asset_root=spec['asset_root'], verify_hashes=True)
        assert asset['dynamic'] is False, 'a fixture is authored as immovable'
        assert asset['collision_parts']


def test_geometry_rule_mass_inertia_and_convex_collision(manifest):
    """Rule 1: positive mass, inertia from its own AABB, convex collision pieces."""
    import trimesh
    for asset_id, record in manifest['assets'].items():
        assert record['collision_parts'], asset_id
        for part in record['collision_parts']:
            assert part['geometry'] in ('box', 'cylinder', 'convex_mesh'), (asset_id, part['label'])
            if part['geometry'] == 'convex_mesh':
                vertices, faces = _load_obj(part['mesh_path'])
                assert trimesh.Trimesh(vertices, faces, process=True).is_convex, (asset_id, part['label'])
        if not record['dynamic']:
            continue
        assert record['mass_kg'] > 0, asset_id
        expected = uniform_box_inertia(record['mass_kg'], record['size_m'])
        assert record['inertia_diag_kg_m2'] == pytest.approx(expected, rel=1e-9), asset_id
        assert 'approximation' in record['inertia_method']


DESIGN_MASS_KG = {'poker_chip': 0.010, 'chip_stack': 0.030, 'mahjong_tile': 0.018, 'mahjong_die': 0.008,
                  'playing_card': 0.0018, 'card_stack': 0.036, 'teacup': 0.19, 'marker_ring': 0.030,
                  'xiangqi_red': 0.005, 'xiangqi_black': 0.005}


def test_masses_are_plausible_design_values(manifest):
    """Rule 5: contract reference values where given, real-material densities otherwise."""
    dynamic = {k for k, v in manifest['assets'].items() if v['dynamic']}
    assert dynamic == set(DESIGN_MASS_KG)
    for asset_id, mass in DESIGN_MASS_KG.items():
        record = manifest['assets'][asset_id]
        assert record['mass_kg'] == pytest.approx(mass, rel=1e-9), asset_id
        assert 'design value' in record['mass_design_note'].lower(), asset_id
        # Mass over the collider's box volume: never denser than clay, never lighter
        # than a hollow porcelain vessel or 3 mm card stock.
        lo, hi = _collision_aabb(record)
        density = record['mass_kg'] / float(np.prod(hi - lo))
        assert 80. <= density <= 2500., (asset_id, density)
    assert manifest['assets']['poker_chip']['mass_kg'] == pytest.approx(0.010)       # contract
    assert manifest['assets']['mahjong_tile']['mass_kg'] == pytest.approx(0.018)     # contract
    assert 0.5 * 0.28 <= manifest['assets']['teacup']['mass_kg'] <= 1.5 * 0.28      # vs a 0.28 kg mug


def test_thin_parts_have_analytic_colliders_of_at_least_3_mm(manifest):
    """Rule 4: chips, cards and tile plates get >= 3 mm boxes or cylinders."""
    for asset_id, record in manifest['assets'].items():
        for part in record['collision_parts']:
            if part['geometry'] == 'box':
                assert 2 * min(part['half_extents_m']) >= 0.003 - 1e-9, (asset_id, part['label'])
            elif part['geometry'] == 'cylinder':
                assert 2 * part['half_height_m'] >= 0.003 - 1e-9, (asset_id, part['label'])
    for asset_id in ('poker_chip', 'playing_card', 'card_stack', 'chip_stack'):
        parts = manifest['assets'][asset_id]['collision_parts']
        assert len(parts) == 1 and parts[0]['geometry'] in ('box', 'cylinder'), asset_id


def test_collider_sizes_in_the_module_match_the_manifest(manifest):
    """The frozen COLLIDER_HALF_M the layout uses is what was actually authored."""
    for asset_id, half in family.COLLIDER_HALF_M.items():
        lo, hi = _collision_aabb(manifest['assets'][asset_id])
        assert (-lo).tolist() == pytest.approx(list(half), abs=1e-9), asset_id
        assert hi.tolist() == pytest.approx(list(half), abs=1e-9), asset_id


def test_display_and_collision_agree_within_half_a_millimetre(manifest):
    for asset_id, record in manifest['assets'].items():
        if not record['dynamic']:
            continue
        clo, chi = _collision_aabb(record)
        vlo, vhi = _visual_aabb(record)
        deviation = max(np.abs(clo - vlo).max(), np.abs(chi - vhi).max())
        assert deviation <= DISPLAY_TOLERANCE_M + 1e-9, (asset_id, deviation)


def test_bore_rule_holds_where_declared(manifest):
    """Rule 2: a declared bore is sectors, each clear of the radius in true geometry."""
    declared = 0
    for asset_id, record in manifest['assets'].items():
        for part in record['collision_parts']:
            if part.get('bore') is not True:
                continue
            declared += 1
            radius = float(part['bore_radius_m'])
            assert part['geometry'] == 'convex_mesh' and part.get('sector'), (asset_id, part['label'])
            axis = np.asarray(part['bore_axis_xy_m'], dtype=float)
            vertices, _ = _load_obj(part['mesh_path'])
            polygon = _hull(vertices[:, :2] - axis)
            # The distance from the bore axis to the convex piece, edges included.
            distance = min(_segment_distance(np.zeros(2), polygon[i], polygon[(i + 1) % len(polygon)])
                           for i in range(len(polygon)))
            covers_axis = all(_cross2(polygon[(i + 1) % len(polygon)] - polygon[i], -polygon[i]) >= 0
                              for i in range(len(polygon)))
            assert not covers_axis, (asset_id, part['label'], 'the piece spans the bore axis')
            assert distance >= radius - 1e-7, (asset_id, part['label'], distance)
        bored = [p for p in record['collision_parts'] if p.get('bore') is True]
        if bored:
            assert len(bored) == len(record['collision_parts']) > 4, 'every piece of a bored ring is a sector'
    assert declared > 0, 'the dealer ring declares a real through bore'


def test_the_ring_bore_admits_a_chip_with_clearance(manifest):
    bore = manifest['assets']['marker_ring']['bore_radius_m']
    assert bore == family.RING_INNER
    assert bore - family.CHIP_RADIUS >= 0.0005, 'at least 1 mm of diametral clearance for a 39 mm chip'


def test_teacup_display_is_closed_and_matches_its_collider(manifest):
    """The cup has a floor in the render, and no invisible skin round its wall."""
    record = manifest['assets']['teacup']
    vessel = next(part for part in record['visual_parts'] if part['label'] == 'vessel')
    vertices, faces = _load_obj(vessel['mesh_path'])
    assert vessel['watertight'] is True
    height, floor = family.CUP_HEIGHT, family.CUP_FLOOR
    base = next(part for part in record['collision_parts'] if part['geometry'] == 'cylinder')
    base_top = base['position_m'][2] + base['half_height_m']
    assert base_top == pytest.approx(-height / 2. + floor, abs=1e-9)
    for radius in (0.0007, 0.005, 0.014, 0.020, family.CUP_INNER_RADIUS - 0.0007):
        hits = 0.2 - _ray_hits(vertices, faces, [radius, 0.0003, 0.2], [0., 0., -1.])
        assert len(hits) >= 2, f'a downward ray at r = {radius * 1000:.1f} mm misses the cup floor'
        assert hits[0] == pytest.approx(base_top, abs=DISPLAY_TOLERANCE_M), 'inner floor'
        assert hits[-1] == pytest.approx(-height / 2., abs=DISPLAY_TOLERANCE_M), 'underside'
    sectors = [_load_obj(part['mesh_path']) for part in record['collision_parts'] if part.get('sector')]
    assert len(sectors) > 4
    assert not any(part.get('bore') for part in record['collision_parts']), 'the base closes the vessel'
    for z in np.linspace(-height / 2. + floor + 0.001, height / 2. - 0.001, 7):
        for angle in np.radians(np.arange(0., 360., 9.7)):
            direction = [math.cos(angle), math.sin(angle), 0.]
            shown = _ray_hits(vertices, faces, [0., 0., z], direction)
            solid = np.concatenate([_ray_hits(v, f, [0., 0., z], direction) for v, f in sectors])
            assert abs(shown.max() - solid.max()) <= DISPLAY_TOLERANCE_M, ('outer wall', z, angle)
            assert abs(shown.min() - solid.min()) <= DISPLAY_TOLERANCE_M, ('inner wall', z, angle)


# --- the room ------------------------------------------------------------------------

def test_room_is_complete_and_closed():
    """Rule 8: floor, four walls with openings, ceiling, ceiling lights and a dome."""
    room = validate_room(SCENE['room'])
    assert room['enabled'] and room['size_m'] == list(family.ROOM_SIZE)
    assert set(room['surfaces']) == {'floor', 'wall', 'ceiling', 'baseboard'}
    assert all(name in SCENE['materials'] for name in room['surfaces'].values())
    assert {o['kind'] for o in room['openings']} == {'door', 'window'}
    assert room['ceiling_lights'] and room['dome']['enabled']
    assert room['collide']['floor'] is True


def test_built_room_has_floor_four_walls_ceiling_and_lights(built):
    paths = built['structured']['room']['paths']
    assert '/World/Room/Floor' in paths and '/World/Room/Ceiling' in paths
    for face in room_module.WALL_FACES:
        assert any(path.startswith(f'/World/Room/Wall_{face}_') for path in paths), face
    assert len(built['structured']['room']['lights']) == len(SCENE['room']['ceiling_lights'])
    assert built['structured']['room']['dome']


def _opening_world(room, opening):
    wall = room_module._wall_geometry(room, opening['wall'])
    (u, v), (su, sv) = opening['center_m'], opening['size_m']
    corners = [wall['center'] + (u + du * su / 2.) * wall['u_axis'] + (v + dv * sv / 2.) * wall['v_axis']
               for du in (-1, 1) for dv in (-1, 1)]
    return np.min(corners, axis=0), np.max(corners, axis=0), wall


def test_openings_use_the_wall_centre_convention():
    """A door reaches the floor and every opening is below the ceiling and inside its wall."""
    room = validate_room(SCENE['room'])
    for opening in room['openings']:
        lo, hi, wall = _opening_world(room, opening)
        assert lo[2] >= -1e-9 and hi[2] <= room['ceiling_height_m'] + 1e-9, opening['id']
        if opening['kind'] == 'door':
            assert lo[2] == pytest.approx(0.0, abs=1e-9), 'a door starts at the floor'
            assert 1.9 <= hi[2] - lo[2] <= 2.3
        else:
            assert lo[2] >= 0.6 and hi[2] <= room['ceiling_height_m'] - 0.2, 'a window sits in the wall'
        along = np.abs(wall['u_axis']) @ (hi - lo) / 2.
        centre_u = float(np.dot((lo + hi) / 2. - wall['center'], wall['u_axis']))
        assert abs(centre_u) + along <= wall['half'][0] + 1e-9, opening['id']
        mine = family.opening_world_rect(opening)
        assert mine['y'] == pytest.approx((lo[1], hi[1]), abs=1e-9)
        assert mine['z'] == pytest.approx((lo[2], hi[2]), abs=1e-9)


def test_openings_are_cut_out_of_the_built_walls(built):
    room = validate_room(SCENE['room'])
    records = built['records']
    for opening in room['openings']:
        lo, hi, wall = _opening_world(room, opening)
        quads = [path for path, record in records.items()
                 if path.startswith(f'/World/Room/Wall_{opening["wall"]}_') and record['kind'] == 'Mesh']
        assert quads, opening['wall']
        area = 0.
        for path in quads:
            points = records[path]['points']
            qlo, qhi = points.min(axis=0), points.max(axis=0)
            assert not _overlap_volume((qlo[[1, 2]], qhi[[1, 2]]), (lo[[1, 2]], hi[[1, 2]])), \
                f'{path} covers the {opening["id"]}'
            area += float(np.prod((qhi - qlo)[[1, 2]]))
        wall_area = 4. * wall['half'][0] * wall['half'][1]
        hole = float(np.prod(opening['size_m']))
        assert area == pytest.approx(wall_area - hole, abs=1e-5)      # uv_quad stores float32 points


def test_trim_frames_each_opening_from_the_room(built):
    """The casing flanks and heads each opening, never covers it, and sits on its wall."""
    room = validate_room(SCENE['room'])
    for opening, prop_id in (({o['id']: o for o in room['openings']}['door'], 'door_frame'),
                             ({o['id']: o for o in room['openings']}['window'], 'window_trim')):
        lo, hi, wall = _opening_world(room, opening)
        plane = float(wall['center'][0])
        parts = _prop(prop_id)['parts']
        boxes = [(np.asarray(p['position_m']) - p['half_extents_m'], np.asarray(p['position_m']) + p['half_extents_m'])
                 for p in parts]
        for (blo, bhi), part in zip(boxes, parts):
            assert not _overlap_volume((blo[[1, 2]], bhi[[1, 2]]), (lo[[1, 2]], hi[[1, 2]])), \
                f'{prop_id}/{part["id"]} intrudes into the {opening["id"]}'
            assert min(abs(blo[0] - plane), abs(bhi[0] - plane)) < 1e-9, f'{part["id"]} is off its wall'
        def covers(axis_lo, axis_hi, a, b):
            return axis_lo <= a + 1e-9 and axis_hi >= b - 1e-9
        assert any(abs(bhi[1] - lo[1]) < 1e-9 and covers(blo[2], bhi[2], lo[2], hi[2]) for blo, bhi in boxes)
        assert any(abs(blo[1] - hi[1]) < 1e-9 and covers(blo[2], bhi[2], lo[2], hi[2]) for blo, bhi in boxes)
        assert any(abs(blo[2] - hi[2]) < 1e-9 and covers(blo[1], bhi[1], lo[1], hi[1]) for blo, bhi in boxes)
        if opening['kind'] == 'window':
            assert any(abs(bhi[2] - lo[2]) < 1e-9 and covers(blo[1], bhi[1], lo[1], hi[1]) for blo, bhi in boxes), \
                'a window has a sill under it'


def test_every_static_box_lies_inside_the_interior():
    for name, centre, half in _static_boxes():
        assert np.all(centre - half >= INTERIOR_LO - 1e-9) and np.all(centre + half <= INTERIOR_HI + 1e-9), name


def test_room_colliders_as_built_stay_out_of_the_interior(built):
    """No solid the room authors may stand inside the room (a misplaced wall would)."""
    boxes = {path: box for path, box in _collider_boxes(built['records']).items()
             if path.startswith('/World/Room/') and 'RobotPedestal' not in path}
    assert '/World/Room/collider' in boxes, 'the floor is solid'
    floor_lo, floor_hi = boxes['/World/Room/collider']
    assert floor_hi[2] == pytest.approx(0.0, abs=1e-9)
    assert np.all(floor_lo[:2] <= INTERIOR_LO[:2]) and np.all(floor_hi[:2] >= INTERIOR_HI[:2])
    for path, box in boxes.items():
        assert not _overlap_volume(box, (INTERIOR_LO, INTERIOR_HI)), f'{path} stands inside the room'
    walls = [path for path in boxes if '/Wall_' in path]
    assert bool(walls) == SCENE['room']['collide']['walls']


# --- materials -----------------------------------------------------------------------

def test_materials_are_valid_and_their_files_exist():
    for name, spec in SCENE['materials'].items():
        validate_material(spec, name)


def test_every_material_builds_through_the_shared_usd_path():
    """visuals.textured_material is what the GPU build calls; it must not raise."""
    with _StubbedUsd():
        built, specs = room_module.room_materials(None, SCENE['materials'], prefix='mat')
    assert set(built) == set(SCENE['materials'])


def _buildable_texture_slots():
    """Texture slots the shared visuals code can author, probed one at a time."""
    image = SCENE['materials']['oak_wood']['base_color']
    buildable = set()
    for key in TEXTURE_KEYS:
        spec = {'kind': 'textured', 'base_color': image, key: image}
        with _StubbedUsd():
            try:
                # The original schema could not express some maps at all: a path in
                # 'roughness' failed float(), a 'normal' map raised NameError.
                room_module.room_materials(None, {f'probe_{key}': spec}, prefix='probe')
                buildable |= set(validate_material(spec, f'probe_{key}')['textures'])
            except (NameError, TypeError, ValueError):
                continue
    return buildable


def test_textured_materials_bind_only_maps_the_shared_code_can_build():
    """A map the shared code cannot author crashed the GPU build (the normal map did).

    What is bound is read from the shared validator itself, so the check holds for
    every material schema ``visuals.py`` has had: base_color only, and nothing that
    any version would read as a map (a ``*_map`` key, a bare slot name, or a
    path-valued roughness/metalness/opacity).
    """
    buildable = _buildable_texture_slots()
    assert 'base_color' in buildable
    for name, spec in SCENE['materials'].items():
        if spec['kind'] != 'textured':
            continue
        bound = set(validate_material(spec, name)['textures'])
        assert bound and bound <= buildable, f'{name} binds {sorted(bound - buildable)}'
        assert bound == {'base_color'}, f'{name} binds {sorted(bound)}; maps stay unbound until a GPU render check'
        assert not any(key.endswith('_map') for key in spec), name
        assert not {'normal', 'occlusion', 'emissive'} & set(spec), name
        assert not any(isinstance(spec.get(key), (str, int, float)) for key in ('roughness', 'metalness', 'opacity'))
        for path in spec.get('unbound_textures', {}).values():
            assert Path(path).is_file(), path


def test_the_recorded_normal_map_builds_through_the_shared_material_path():
    """The normal map kept in unbound_textures is buildable when it is bound later."""
    spec = dict(SCENE['materials']['oak_wood'], normal=SCENE['materials']['oak_wood']['unbound_textures']['normal_gl'])
    with _StubbedUsd():
        built, _ = room_module.room_materials(None, {'oak_wood_normal': spec}, prefix='mat')
    assert set(built) == {'oak_wood_normal'}


def test_every_material_is_used_and_every_reference_is_declared():
    referenced = set(SCENE['room']['surfaces'].values()) | {SCENE['table']['material']}
    referenced |= {spec['material'] for spec in SCENE['surfaces']}
    for prop in SCENE['static_props']:
        referenced |= {part['material'] for part in prop['parts']}
    referenced |= {'top', 'side'}            # bound to the robot stand by room.add_pedestal
    assert referenced == set(SCENE['materials']), (sorted(referenced ^ set(SCENE['materials'])))


def test_textured_materials_record_provenance():
    for name, spec in SCENE['materials'].items():
        if spec['kind'] != 'textured':
            continue
        assert spec['license'] == 'CC0-1.0', name
        path = Path(spec['base_color'])
        if spec['source'].startswith('https://ambientcg.com/'):
            source = load_json(path.parent / 'SOURCE.json')
            recorded = {item['file']: item['sha256'] for item in source['files']}
            for item in [path] + [Path(p) for p in spec.get('unbound_textures', {}).values()]:
                assert hashlib.sha256(item.read_bytes()).hexdigest() == recorded[item.name], (name, item.name)
            assert 'CC0' in source['license']
        else:
            assert path == ROOT / family.BOARD_TEXTURE, name


# --- furniture -----------------------------------------------------------------------

def test_no_box_is_declared_twice_or_interpenetrates_another():
    """The table slab, the felt and every prop part exist once; nothing overlaps."""
    boxes = _static_boxes()
    for (name_a, ca, ha), (name_b, cb, hb) in itertools.combinations(boxes, 2):
        assert not (np.allclose(ca, cb) and np.allclose(ha, hb)), f'{name_a} duplicates {name_b}'
        assert not _overlap_volume((ca - ha, ca + ha), (cb - hb, cb + hb)), f'{name_a} interpenetrates {name_b}'


def test_static_colliders_as_built_are_the_declared_boxes_once_each(built):
    boxes = _collider_boxes(built['records'])
    props = {path: box for path, box in boxes.items() if path.startswith('/World/Props/')}
    declared = 1 + len(SCENE['surfaces']) + sum(len(p['parts']) for p in SCENE['static_props'])
    assert len(props) == declared
    for (path_a, a), (path_b, b) in itertools.combinations(props.items(), 2):
        assert not _overlap_volume(a, b), f'{path_a} and {path_b} are coincident or overlapping colliders'
    for prop in SCENE['static_props']:
        for part in prop['parts']:
            lo, hi = props[f'/World/Props/{prop["id"]}/{part["id"]}/collider']
            centre, half = np.asarray(part['position_m']), np.asarray(part['half_extents_m'])
            assert lo == pytest.approx(centre - half, abs=1e-9) and hi == pytest.approx(centre + half, abs=1e-9)


def test_the_table_has_a_declared_material():
    """scene.py falls back to an undefined 'table' material; this family names one."""
    assert SCENE['table'].get('material') in SCENE['materials']


def test_the_table_is_a_real_card_table():
    table = SCENE['table']
    width, depth = (2 * v for v in table['half_extents_m'][:2])
    assert 0.80 <= width <= 0.90 and 0.80 <= depth <= 0.90, 'a mahjong / card table is ~0.8-0.9 m'
    assert table['position_m'][0] - table['half_extents_m'][0] >= 0.25, 'rule 7: the table starts at x >= 0.25'
    underside = table['position_m'][2] - table['half_extents_m'][2]
    for part in _prop('table_legs')['parts']:
        bottom = part['position_m'][2] - part['half_extents_m'][2]
        top = part['position_m'][2] + part['half_extents_m'][2]
        assert bottom == pytest.approx(0.0, abs=1e-9) and top == pytest.approx(underside, abs=1e-9), part['id']
    felt = next(s for s in SCENE['surfaces'] if s['id'] == 'felt_top')
    assert felt['position_m'][2] - felt['half_extents_m'][2] == pytest.approx(
        table['position_m'][2] + table['half_extents_m'][2], abs=1e-9), 'the felt lies on the slab'


def test_furniture_is_grounded_not_floating():
    """Every cluster of touching boxes reaches the floor, a wall or the ceiling."""
    boxes = _static_boxes()
    def touching(a, b, tol=1e-6):
        (ca, ha), (cb, hb) = a, b
        gap = np.abs(ca - cb) - (ha + hb)
        return any(abs(gap[k]) <= tol and all(gap[j] < -tol for j in range(3) if j != k) for k in range(3))
    def grounded(centre, half):
        lo, hi = centre - half, centre + half
        return (abs(lo[2]) < 1e-9 or abs(hi[2] - INTERIOR_HI[2]) < 1e-9
                or any(abs(lo[k] - INTERIOR_LO[k]) < 1e-9 or abs(hi[k] - INTERIOR_HI[k]) < 1e-9 for k in (0, 1)))
    reached = {i for i, (_, c, h) in enumerate(boxes) if grounded(c, h)}
    frontier = list(reached)
    while frontier:
        i = frontier.pop()
        for j, (_, c, h) in enumerate(boxes):
            if j not in reached and touching((boxes[i][1], boxes[i][2]), (c, h)):
                reached.add(j)
                frontier.append(j)
    floating = [boxes[i][0] for i in range(len(boxes)) if i not in reached]
    assert not floating, f'not attached to the room: {floating}'
    scroll = _prop('wall_scroll')['parts'][0]
    assert scroll['position_m'][1] + scroll['half_extents_m'][1] == pytest.approx(family.ROOM_HALF_Y), \
        'the scroll hangs flat on the wall'


def test_the_tea_set_rests_on_the_cabinet(built, manifest):
    spec = SCENE['fixtures'][0]
    top = max(part['position_m'][2] + part['half_extents_m'][2] for part in _prop('tea_cabinet')['parts'])
    assert spec['position_m'][2] == pytest.approx(top, abs=1e-9)
    records = built['records']
    collision = [p for p in records if p.startswith('/World/Fixtures/tea_set/collision_')]
    points = np.vstack([_world_points(records, p) for p in collision])
    assert points[:, 2].min() == pytest.approx(top, abs=1e-6), 'the tray sits on the cabinet top'
    cabinet_top = next(p for p in _prop('tea_cabinet')['parts'] if p['id'] == 'top')
    lo = np.asarray(cabinet_top['position_m']) - cabinet_top['half_extents_m']
    hi = np.asarray(cabinet_top['position_m']) + cabinet_top['half_extents_m']
    tray = points[points[:, 2] <= top + 0.026]
    assert np.all(tray[:, :2] >= lo[:2] - 1e-9) and np.all(tray[:, :2] <= hi[:2] + 1e-9)
    for name, centre, half in _static_boxes():
        assert not _overlap_volume(_aabb(tray), (centre - half, centre + half)), name


def test_the_room_reads_as_a_card_room():
    """Chairs at the table, a lamp over it, a tea cabinet, xiangqi, mahjong and poker."""
    props = {prop['id'] for prop in SCENE['static_props']}
    assert {'chair_far', 'chair_left', 'chair_right', 'pendant_lamp', 'tea_cabinet', 'chip_tray'} <= props
    assert 'parts_bin' not in props and not any(f['asset_id'] == 'parts_bin' for f in SCENE['fixtures'])
    assert any(spec['id'] == 'xiangqi_board' for spec in SCENE['surfaces'])
    counts = {}
    for spec in SCENE['objects']:
        counts[spec['asset_id']] = counts.get(spec['asset_id'], 0) + 1
    assert counts.get('xiangqi_red', 0) + counts.get('xiangqi_black', 0) >= 6
    assert counts['mahjong_tile'] >= 8 and counts['mahjong_die'] == 2
    assert counts['poker_chip'] >= 3 and counts['card_stack'] >= 1
    lamp = _prop('pendant_lamp')
    cx, cy = family.TABLE_CENTER_XY
    assert all(abs(p['position_m'][0] - cx) <= family.TABLE_HALF_XY[0] and
               abs(p['position_m'][1] - cy) <= family.TABLE_HALF_XY[1] for p in lamp['parts'])
    lights = {light['id']: light for light in SCENE['room']['ceiling_lights']}
    assert lights['lamp_table']['position_m'][2] < min(p['position_m'][2] + p['half_extents_m'][2]
                                                       for p in lamp['parts'] if p['id'] == 'lid'), \
        'the table light sits inside the shade'


def test_chairs_face_the_table_and_do_not_touch_it():
    table = SCENE['table']
    t_lo = np.asarray(table['position_m'][:2]) - table['half_extents_m'][:2]
    t_hi = np.asarray(table['position_m'][:2]) + table['half_extents_m'][:2]
    for chair_id, centre, yaw in family.CHAIRS:
        parts = {part['id']: part for part in _prop(chair_id)['parts']}
        seat = parts['seat']
        assert seat['position_m'][2] + seat['half_extents_m'][2] == pytest.approx(family.CHAIR_SEAT_TOP)
        back = np.asarray(parts['back_panel']['position_m'][:2])
        to_table = np.asarray(table['position_m'][:2]) - np.asarray(centre)
        assert np.dot(back - np.asarray(centre), to_table) < 0, f'{chair_id} backrest is away from the table'
        for part in parts.values():
            gap = _gap(_rect(part['position_m'][:2], part['half_extents_m'][:2]),
                       _rect((t_lo + t_hi) / 2., (t_hi - t_lo) / 2.))
            assert gap >= 0.05, f'{chair_id}/{part["id"]} is {gap:.3f} m from the table'


# --- the pieces ----------------------------------------------------------------------

def test_every_object_rests_exactly_on_a_declared_support(manifest):
    """Rule 3, from the authored collider: no floating and no sinking."""
    supports = {s['id']: s for s in support_surfaces(SCENE)}
    for spec in SCENE['objects']:
        lo, hi = _collision_aabb(manifest['assets'][spec['asset_id']])
        rotation = _quat_matrix(spec['quaternion_wxyz'])
        corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
        bottom = spec['position_m'][2] + (corners @ rotation.T)[:, 2].min()
        support = supports[spec['support']]
        assert bottom == pytest.approx(support['top_z_m'], abs=1e-6), spec['id']
        assert spec['rest_geometry']['size_m'][2] / 2. == pytest.approx(
            spec['position_m'][2] - bottom, abs=1e-6), 'rest_geometry is the collider at its pose'
        candidates = [s for s in support_surfaces(SCENE)
                      if s['xy_min'][0] <= spec['position_m'][0] <= s['xy_max'][0]
                      and s['xy_min'][1] <= spec['position_m'][1] <= s['xy_max'][1]]
        assert max(candidates, key=lambda s: s['top_z_m'])['id'] == spec['support'], spec['id']


def test_objects_as_built_by_scene_py_rest_on_their_supports(built):
    records = built['records']
    for spec in SCENE['objects']:
        base = f'/World/Objects/{spec["id"]}'
        collision = [p for p in records if p.startswith(base + '/collision_')]
        visual = [p for p in records if p.startswith(base + '/visual_')]
        assert len(collision) == built['objects'][spec['id']]['collision_parts'] and visual
        assert all(records[p]['collider'] for p in collision)
        z_collision = min(_world_points(records, p)[:, 2].min() for p in collision)
        z_visual = min(_world_points(records, p)[:, 2].min() for p in visual)
        top = family.SUPPORT_TOP_Z[spec['support']]
        assert z_collision == pytest.approx(top, abs=1e-6), f'{spec["id"]} as built'
        assert abs(z_visual - z_collision) <= DISPLAY_TOLERANCE_M, f'{spec["id"]} display vs collider'
        points = np.vstack([_world_points(records, p) for p in collision])
        half = family.collider_world_half(spec['asset_id'], spec['quaternion_wxyz'])
        assert (points.max(axis=0) - points.min(axis=0)) / 2. == pytest.approx(half, abs=2e-4)


def test_every_piece_lies_wholly_on_its_support():
    supports = {s['id']: s for s in SCENE['surfaces']}
    board = supports['xiangqi_board']
    board_rect = _rect(board['position_m'][:2], board['half_extents_m'][:2])
    for spec in SCENE['objects']:
        footprint = _footprint(spec)
        plate = supports[spec['support']]
        lo = np.asarray(plate['position_m'][:2]) - plate['half_extents_m'][:2]
        hi = np.asarray(plate['position_m'][:2]) + plate['half_extents_m'][:2]
        if footprint[0] == 'disc':
            pts = footprint[1] + footprint[2] * np.array([[1, 0], [-1, 0], [0, 1], [0, -1]])
        else:
            pts = footprint[1]
        assert np.all(pts >= lo - 1e-9) and np.all(pts <= hi + 1e-9), f'{spec["id"]} overhangs {plate["id"]}'
        if spec['support'] == 'felt_top':
            assert _gap(footprint, board_rect) >= 0.005, f'{spec["id"]} touches the board edge'


def test_no_two_objects_interpenetrate():
    for first, second in itertools.combinations(SCENE['objects'], 2):
        za, zb = _z_range(first), _z_range(second)
        if min(za[1], zb[1]) - max(za[0], zb[0]) <= 1e-9:
            continue
        assert _gap(_footprint(first), _footprint(second)) >= 0.0015, (first['id'], second['id'])


def test_no_object_intrudes_into_static_furniture():
    for spec in SCENE['objects']:
        bottom, top = _z_range(spec)
        footprint = _footprint(spec)
        for name, centre, half in _static_boxes():
            if centre[2] + half[2] <= bottom + 1e-9 or centre[2] - half[2] >= top - 1e-9:
                continue
            assert _gap(footprint, _rect(centre[:2], half[:2])) >= 0.0009, f'{spec["id"]} intrudes into {name}'


def test_chips_sit_centred_in_open_tray_wells():
    tray = _prop('chip_tray')
    for spec in SCENE['objects']:
        if not spec['id'].startswith('chip_well_'):
            continue
        centre = np.asarray(spec['position_m'][:2])
        assert spec['support'] == 'felt_top', 'the wells are cut through to the felt'
        for part in tray['parts']:
            gap = _disc_polygon_gap(centre, family.CHIP_RADIUS,
                                    _rect(part['position_m'][:2], part['half_extents_m'][:2])[1])
            assert gap >= 0.0009, (spec['id'], part['id'])
            assert part['position_m'][2] - part['half_extents_m'][2] >= family.FELT_TOP_Z - 1e-9


def test_pieces_stand_inside_the_work_envelope():
    """Rule 6: reachable, on the table, in front of the robot."""
    for spec in SCENE['objects']:
        x, y, z = spec['position_m']
        assert 0.30 <= x <= 0.80 and -0.10 <= y <= 0.45, spec['id']
        assert math.hypot(x, y) <= 0.80, f'{spec["id"]} is {math.hypot(x, y):.3f} m from the base'
        assert family.FELT_TOP_Z <= z <= 0.85, spec['id']


def test_xiangqi_pieces_sit_on_board_intersections():
    board = next(s for s in SCENE['surfaces'] if s['id'] == 'xiangqi_board')
    lo = np.asarray(board['position_m'][:2]) - board['half_extents_m'][:2]
    assert lo.tolist() == pytest.approx(list(family.BOARD_LO_XY), abs=1e-9)
    pieces = [spec for spec in SCENE['objects'] if spec['asset_id'].startswith('xiangqi_')]
    assert len(pieces) == 8
    for spec in pieces:
        rank = (spec['position_m'][0] - lo[0] - family.XQ_MARGIN) / family.XQ_SPACING
        file = (spec['position_m'][1] - lo[1] - family.XQ_MARGIN) / family.XQ_SPACING
        assert abs(rank - round(rank)) < 1e-6 and abs(file - round(file)) < 1e-6, spec['id']
        assert 0 <= round(rank) <= 9 and 0 <= round(file) <= 8
        side = 'red' if round(rank) <= 4 else 'black'
        assert spec['asset_id'] == f'xiangqi_{side}', f'{spec["id"]} is on the wrong side of the river'


def test_standing_tiles_face_the_robot():
    standing = [spec for spec in SCENE['objects'] if spec['quaternion_wxyz'] == list(family.STANDING_WXYZ)]
    assert len(standing) == 8
    for spec in standing:
        rotation = _quat_matrix(spec['quaternion_wxyz'])
        assert rotation @ [0., 0., 1.] == pytest.approx([-1., 0., 0.]), 'face plate toward the robot'
        assert rotation @ [0., 1., 0.] == pytest.approx([0., 0., 1.]), 'the 26 mm edge is vertical'
    others = [spec for spec in SCENE['objects'] if spec not in standing]
    for spec in others:
        assert spec['quaternion_wxyz'][1] == 0. and spec['quaternion_wxyz'][2] == 0., 'only yaw otherwise'


def test_description_matches_the_build():
    text = SCENE['description_zh']
    counts = {}
    for spec in SCENE['objects']:
        counts[spec['asset_id']] = counts.get(spec['asset_id'], 0) + 1
    def ids(prefix):
        return sum(spec['id'].startswith(prefix) for spec in SCENE['objects'])
    facts = {
        '象棋棋盘': any(s['id'] == 'xiangqi_board' for s in SCENE['surfaces']),
        '八枚棋子': counts['xiangqi_red'] + counts['xiangqi_black'] == 8,
        '立着一排八张麻将牌': ids('tile_hand_') == 8,
        '三张打出的明牌': ids('tile_discard_') == 3,
        '两枚骰子': counts['mahjong_die'] == 2,
        '三个格内各放一枚筹码': ids('chip_well_') == 3,
        '两叠筹码': counts['chip_stack'] == 2,
        '两枚散放筹码': ids('chip_loose_') == 2,
        '一副扑克牌': counts['card_stack'] == 1,
        '一张散牌': counts['playing_card'] == 1,
        '两只无柄瓷茶杯': counts['teacup'] == 2,
        '一枚庄家标记环': counts['marker_ring'] == 1,
        '各摆一把椅子': len(family.CHAIRS) == 3,
        '麻将吊灯': any(p['id'] == 'pendant_lamp' for p in SCENE['static_props']),
        '茶水柜': any(p['id'] == 'tea_cabinet' for p in SCENE['static_props']),
        '三只小茶杯': True,
        '麻将盒': any(part['id'] == 'mahjong_case' for part in _prop('cabinet_clutter')['parts']),
        '书法横幅': any(p['id'] == 'wall_scroll' for p in SCENE['static_props']),
        '木地板': SCENE['room']['surfaces']['floor'] == 'plank_floor',
        '右墙有门': any(o['kind'] == 'door' and o['wall'] == 'right' for o in SCENE['room']['openings']),
        '左墙有带窗台的窗户': any(o['kind'] == 'window' and o['wall'] == 'left' for o in SCENE['room']['openings']),
        '0.86米见方': all(abs(2 * h - 0.86) < 1e-9 for h in SCENE['table']['half_extents_m'][:2]),
    }
    for phrase, holds in facts.items():
        assert phrase in text, f'the description no longer mentions {phrase}'
        assert holds, f'the description says {phrase} but the scene does not build it'
    assert '地毯' not in text, 'there is no carpet'
    assert len(re.findall('骰子', text)) == 1


def test_the_tea_set_has_a_pot_and_three_cups(manifest):
    labels = [part['label'] for part in manifest['assets']['tea_set']['visual_parts']]
    assert sum(label.startswith('cup_') for label in labels) == 3 and 'pot_body' in labels


# --- cameras -------------------------------------------------------------------------

def test_cameras_resolve_with_world_views_and_an_aimed_wrist_camera():
    cameras = resolve_cameras(SCENE)
    assert [c['kind'] for c in cameras.values()].count('world') == 2
    wrist = cameras['wrist']
    assert wrist['kind'] == 'link' and wrist['mount_link'] == 'right_hand_C_MC'
    assert wrist['aim_source'] == 'declared' and wrist['standoff_m'] > 0
    assert wrist['aim_target_m'] == pytest.approx(family.WORK_TARGET)


def test_world_cameras_sit_inside_the_room():
    """Rule 8: eyes and targets inside the closed shell, clear of the walls."""
    for name, camera in resolve_cameras(SCENE).items():
        if camera['kind'] != 'world':
            continue
        for label in ('eye', 'target'):
            point = np.asarray(camera[label])
            assert np.all(point >= INTERIOR_LO + [0.1, 0.1, 0.0]), (name, label)
            assert np.all(point <= INTERIOR_HI - [0.1, 0.1, 0.1]), (name, label)


def test_third_person_camera_frames_the_work_table():
    """Rule 9: aimed at the table centre from 1.5-2.2 m, 0.6-1.4 m above its top."""
    camera = resolve_cameras(SCENE)['third_person']
    eye, target = np.asarray(camera['eye']), np.asarray(camera['target'])
    assert 1.5 <= np.linalg.norm(eye - target) <= 2.2
    assert 0.6 <= eye[2] - family.FELT_TOP_Z <= 1.4
    assert math.hypot(target[0] - SCENE['table']['position_m'][0],
                      target[1] - SCENE['table']['position_m'][1]) <= 0.05, 'aimed at the table centre'
    table = SCENE['table']
    assert _project(camera, _box_samples([table])).all(), 'the whole table top is in frame'
    assert _project(camera, [spec['position_m'] for spec in SCENE['objects']]).all()


def test_furnishings_are_inside_the_third_person_frame():
    """The view must read as a room: chairs, cabinet, tea set, lamp and robot in frame."""
    camera = resolve_cameras(SCENE)['third_person']
    fraction = {prop['id']: float(_project(camera, _box_samples(prop['parts'])).mean())
                for prop in SCENE['static_props']}
    for chair_id, _, _ in family.CHAIRS:
        assert fraction[chair_id] >= 0.6, (chair_id, fraction[chair_id])
    assert fraction['tea_cabinet'] >= 0.95 and fraction['cabinet_clutter'] >= 0.95
    assert fraction['pendant_lamp'] > 0.0, 'the lamp over the table shows at the top of the frame'
    tea_set = SCENE['fixtures'][0]
    assert _project(camera, [np.asarray(tea_set['position_m']) + [0., 0., 0.05]]).all()
    pedestal = SCENE['pedestal']
    assert _project(camera, [pedestal['position_m'], [0., 0., 1.2]]).all(), 'the robot is in frame'
    off_table = [pid for pid, f in fraction.items() if f > 0 and not pid.startswith(('table_', 'chip_'))]
    assert len(off_table) >= 6, off_table


def test_table_view_frames_every_piece():
    camera = resolve_cameras(SCENE)['table_view']
    assert _project(camera, [spec['position_m'] for spec in SCENE['objects']]).all()


# --- robot, pedestal, physics ----------------------------------------------------------

def test_robot_pins_the_release_hand_assembly():
    robot = SCENE['robot']
    assert robot['hand_id'] == 'sharpa_w01_right' and robot['assembly_version'] == 'isaacsim_6_1_v2'
    assert len(robot['arm_home_rad']) == 7
    assert 'mount_position_m' not in robot and 'mount_quaternion_wxyz' not in robot


def test_the_pedestal_stands_on_the_floor_beside_the_table():
    """Rule 7: the robot's own stand, clear of the table."""
    pedestal, table = SCENE['pedestal'], SCENE['table']
    gap = (table['position_m'][0] - table['half_extents_m'][0]) - (pedestal['position_m'][0] + pedestal['half_extents_m'][0])
    assert gap >= 0.0
    top = pedestal['position_m'][2] + pedestal['half_extents_m'][2]
    assert SCENE['robot']['base_position_m'][2] == pytest.approx(top, abs=1e-9)
    bottom = pedestal['position_m'][2] - pedestal['half_extents_m'][2] - pedestal['skirt_height_m']
    assert bottom == pytest.approx(0.0, abs=1e-9), 'the stand reaches the floor'


def test_physics_block_matches_the_bench_control_period():
    physics = SCENE['physics']
    assert physics['control_dt_s'] / physics['dt_s'] >= 2
    assert SCENE['isaacsim']['physics_dt_s'] == pytest.approx(physics['dt_s'])
    assert physics['gravity_m_s2'] == [0.0, 0.0, -9.81]
