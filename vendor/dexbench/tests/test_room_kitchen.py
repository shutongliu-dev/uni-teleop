"""CPU validation of the ``kitchen`` scene family and its authored asset library.

No GPU, no Isaac Sim, no USD. Every claim the scene makes about its own physics is
re-derived here in the frame Isaac actually builds: each collision and visual piece
is loaded through ``asset_catalog.load_task_asset`` (hashes verified) and placed
exactly as ``scene.add_task_asset`` places it - OBJ vertices scaled, then the
piece's own ``quaternion_xyzw`` and ``position_m`` applied on top, then the
object's pose. Nothing is measured on raw OBJ vertices alone: the v6 library
passed a raw-vertex test while its mug collision was built 26.5 mm off its visual.

Covered, per the frozen contract:
  * ``validate_family`` accepts the scene; the library is the pinned v8 and every
    object asset resolves;
  * rule 1: positive mass, uniform-box inertia, convex collision pieces whose
    as-built union reproduces the visual union within 1 mm;
  * rule 2: the mug handle bore and the vessel interiors are open as built, by the
    exact projected distance from each axis to every convex piece, and no sample of
    the visible finger hole lies inside any collision piece;
  * rule 3: the lowest as-built point of every object is on the work-table top;
  * rule 4: every collision piece is at least 3 mm thick (the knife blade included);
  * rule 5-7: design masses, reach, pedestal/table separation;
  * rule 8-9: the room is closed, the cameras resolve and the third-person camera
    sees the table from the required distance and height;
  * rule 10 and the static world: no two static colliders coincide or overlap
    (the table and its schema-forced ``surfaces`` twin excepted, and required to
    be identical), every bound material is declared;
  * the views read as a kitchen: hob, sink and tap, fridge, upper cabinets and the
    range hood are in the third-person frame and not hidden; the robot's arm at
    its home pose (URDF FK, collision meshes) hides no object from either world
    camera.
Negative controls show each check fails on the defect it was written for.
"""
from __future__ import annotations

import copy
import itertools
import math
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial import ConvexHull

from bench.control.kinematics import URDFKinematics
from bench.paths import load_json
from scenes.rooms import kitchen
from bench.simulation.isaacsim_backend.asset_catalog import load_obj_mesh, load_task_asset
from bench.simulation.isaacsim_backend.cameras import resolve_cameras
from bench.simulation.isaacsim_backend.scene_family import uniform_box_inertia, validate_family
from bench.simulation.isaacsim_backend.room import WALL_FACES, _wall_geometry, _wall_quads, validate_room

SCENE = kitchen.SCENE
ROOT = Path(kitchen.ASSET_ROOT)
V6_ROOT = ROOT.parent / 'v6'

# The contract's manipulation envelope and hard geometry floors.
REACH_X = (0.30, 0.80)
REACH_Y = (-0.10, 0.45)
REACH_RADIUS_M = 0.80
REST_TOLERANCE_M = 1e-3
AS_BUILT_TOLERANCE_M = 1e-3
THIN_PART_MIN_M = 0.003
# Pinhole model of the env's cameras: only the focal length is authored, so the
# USD default 20.955 mm horizontal aperture applies; the vertical aperture is the
# smaller of the USD default and the square-pixel value, i.e. the narrower frame.
H_APERTURE_MM = 20.955
V_APERTURE_DEFAULT_MM = 15.2908

ASSET_IDS = sorted({spec['asset_id'] for spec in SCENE['objects']})


# --- the as-built frame (scene.add_task_asset) --------------------------------

def _quat_xyzw_matrix(q):
    x, y, z, w = (float(v) for v in q)
    n = math.sqrt(x * x + y * y + z * z + w * w)
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def _quat_wxyz_matrix(q):
    return _quat_xyzw_matrix([q[1], q[2], q[3], q[0]])


def _local_points(part):
    """A piece's points before its own transform, as add_task_asset authors them."""
    geometry = part.get('geometry', 'convex_mesh')
    if geometry in ('mesh', 'convex_mesh'):
        vertices, _ = load_obj_mesh(part['mesh_path'])
        return np.asarray(vertices, dtype=float) * np.asarray(part.get('scale', [1, 1, 1]), dtype=float)
    if geometry == 'box':
        # UsdGeom.Cube of size 2 scaled by half_extents_m.
        half = np.asarray(part['half_extents_m'], dtype=float)
        return np.array([[a, b, c] for a in (-1, 1) for b in (-1, 1) for c in (-1, 1)], dtype=float) * half
    if geometry == 'cylinder':
        radius, half_h = float(part['radius_m']), float(part['half_height_m'])
        angles = np.arange(360) * 2 * math.pi / 360
        ring = np.c_[radius * np.cos(angles), radius * np.sin(angles)]
        return np.r_[np.c_[ring, np.full(360, -half_h)], np.c_[ring, np.full(360, half_h)]]
    if geometry == 'sphere':
        radius = float(part['radius_m'])
        return np.r_[np.eye(3), -np.eye(3)] * radius
    raise AssertionError(f'unknown collision geometry {geometry!r}')


def as_built_points(part):
    """A piece's points in the asset frame: own quaternion_xyzw, then position_m."""
    rotation = _quat_xyzw_matrix(part.get('quaternion_xyzw', [0., 0., 0., 1.]))
    return _local_points(part) @ rotation.T + np.asarray(part.get('position_m', [0., 0., 0.]), dtype=float)


def world_points(spec, points):
    """Asset-frame points placed by an object's position and wxyz quaternion."""
    rotation = _quat_wxyz_matrix(spec.get('quaternion_wxyz', [1., 0., 0., 0.]))
    return np.asarray(points, dtype=float) @ rotation.T + np.asarray(spec['position_m'], dtype=float)


def _bounds(point_sets):
    stacked = np.vstack(list(point_sets))
    return np.stack([stacked.min(axis=0), stacked.max(axis=0)])


def _load(asset_id, root=ROOT):
    return load_task_asset(asset_id, asset_root=str(root), verify_hashes=True)


def as_built_deviation(record):
    """Max |collision AABB - visual AABB| over the six faces, both as built."""
    visual = _bounds(as_built_points(part) for part in record['visual_parts'])
    collision = _bounds(as_built_points(part) for part in record['collision_parts'])
    return float(np.max(np.abs(collision - visual))), visual, collision


def projected_clearance(points, axis_point, axis_direction):
    """Exact distance from an infinite axis to a convex piece (0 if the axis pierces it)."""
    d = np.asarray(axis_direction, dtype=float)
    d /= np.linalg.norm(d)
    helper = np.array([1., 0., 0.]) if abs(d[0]) < 0.9 else np.array([0., 1., 0.])
    u = np.cross(d, helper)
    u /= np.linalg.norm(u)
    v = np.cross(d, u)
    rel = np.asarray(points, dtype=float) - np.asarray(axis_point, dtype=float)
    flat = np.c_[rel @ u, rel @ v]
    hull = ConvexHull(flat)
    if np.all(hull.equations[:, 2] <= 1e-12):          # origin inside every half-plane
        return 0.0
    polygon = flat[hull.vertices]
    best = np.inf
    for a, b in zip(polygon, np.roll(polygon, -1, axis=0)):
        edge = b - a
        t = float(np.clip(-(a @ edge) / (edge @ edge), 0.0, 1.0))
        best = min(best, float(np.linalg.norm(a + t * edge)))
    return best


def min_width(points):
    """Exact minimum width of a convex point set (face normals and edge-pair normals)."""
    hull = ConvexHull(points)
    directions = [eq[:3] for eq in hull.equations]
    edges = set()
    for simplex in hull.simplices:
        for i, j in ((0, 1), (1, 2), (0, 2)):
            edges.add(tuple(sorted((simplex[i], simplex[j]))))
    vectors = [points[j] - points[i] for i, j in edges]
    for a, b in itertools.combinations(vectors, 2):
        n = np.cross(a, b)
        if np.linalg.norm(n) > 1e-12:
            directions.append(n / np.linalg.norm(n))
    best = np.inf
    for n in directions:
        projection = points @ n
        best = min(best, float(projection.max() - projection.min()))
    return best


def thin_part_problems(record):
    problems = []
    for part in record['collision_parts']:
        points = as_built_points(part)
        if part.get('geometry') == 'box':
            width = 2 * float(min(part['half_extents_m']))
        elif part.get('geometry') == 'cylinder':
            width = min(2 * float(part['radius_m']), 2 * float(part['half_height_m']))
        else:
            width = min_width(points)
        if width < THIN_PART_MIN_M - 1e-9:
            problems.append(f'{part["label"]} is {width * 1000:.2f} mm thick')
    return problems


@pytest.fixture(scope='module')
def manifest():
    """The authored library must exist; the scene is not valid without it."""
    path = ROOT / 'manifest.json'
    assert path.is_file(), (
        f'the kitchen asset library is not built at {ROOT}; run '
        'scenes.rooms.kitchen.build_library() before validating the scene')
    return load_json(path)


@pytest.fixture(scope='module')
def records(manifest):
    return {asset_id: _load(asset_id) for asset_id in manifest['assets']}


# --- the scene validates and pins its library ------------------------------

def test_scene_passes_the_frozen_validator():
    result = validate_family(SCENE)
    assert result['family'] == 'kitchen'
    assert result['objects'] == len(SCENE['objects']) == 7
    assert result['static_props'] == len(SCENE['static_props'])


def test_schema_and_required_blocks():
    assert SCENE['schema_version'] == 'isaacsim-scene-family-v1'
    assert SCENE['family'] == 'kitchen'
    physics = SCENE['physics']
    assert physics['dt_s'] > 0 and physics['control_dt_s'] / physics['dt_s'] >= 2
    assert physics['gravity_m_s2'] == [0.0, 0.0, -9.81]


def test_scene_pins_its_library_and_every_asset_resolves(manifest):
    assert kitchen.ASSET_VERSION == 'v8' and ROOT.name == 'v8'
    assert manifest['asset_version'] == 'v8'
    for spec in SCENE['objects']:
        assert Path(spec['asset_root']) == ROOT, f'{spec["id"]} does not use the pinned v8 library'
        record = _load(spec['asset_id'], spec['asset_root'])       # hashes verified
        assert record['asset_id'] == spec['asset_id']
        for part in record['visual_parts'] + record['collision_parts']:
            if 'mesh_path' in part:
                assert Path(part['mesh_path']).is_file()


# --- rule 1: real geometry, in the frame Isaac builds -----------------------

def test_every_object_asset_has_mass_inertia_and_convex_collision(records):
    for spec in SCENE['objects']:
        record = records[spec['asset_id']]
        assert record['dynamic'] is True, f'{spec["id"]} must be a dynamic body'
        assert record['mass_kg'] > 0, f'{spec["id"]} has no mass'
        expected = uniform_box_inertia(record['mass_kg'], record['size_m'])
        assert np.allclose(record['inertia_diag_kg_m2'], expected, rtol=1e-9, atol=0), \
            f'{spec["id"]} inertia is not the uniform-density bounding-box value'
        assert 'not measured' in record['inertia_method']
        assert record['collision_parts'], f'{spec["id"]} has no collision part'
        for part in record['collision_parts']:
            assert part.get('geometry', 'convex_mesh') in ('box', 'cylinder', 'convex_mesh', 'sphere')
            if part.get('geometry', 'convex_mesh') == 'convex_mesh':
                assert_convex(f'{spec["id"]}/{part["label"]}', part['mesh_path'])


# OBJ export keeps 10 significant digits, so a planar bevel face of a sector can
# deviate from its plane by ~0.5 um; a real concavity is millimetres.
CONVEXITY_TOLERANCE_M = 2e-6


def assert_convex(name, mesh_path):
    """A convex piece: every vertex is a hull vertex, no vertex lies outside any
    face plane, and the closed mesh encloses its hull's volume."""
    vertices, faces = load_obj_mesh(mesh_path)
    vertices = np.asarray(vertices, dtype=float)
    hull = ConvexHull(vertices)
    assert len(set(hull.vertices)) == len(vertices), f'{name} has vertices inside its hull'
    for face in faces:
        a, b, c = vertices[face]
        normal = np.cross(b - a, c - a)
        if np.linalg.norm(normal) < 1e-14:
            continue
        signed = (vertices - a) @ (normal / np.linalg.norm(normal))
        assert signed.max() <= CONVEXITY_TOLERANCE_M or signed.min() >= -CONVEXITY_TOLERANCE_M, \
            f'{name} is not convex'
    triangles = vertices[np.asarray(faces)]
    volume = abs(float(np.einsum('ij,ij->i', triangles[:, 0],
                                 np.cross(triangles[:, 1], triangles[:, 2])).sum()) / 6.0)
    assert volume == pytest.approx(hull.volume, rel=1e-3), f'{name} does not enclose its hull'


def test_as_built_collision_union_matches_the_visual_within_1mm(records):
    """The check that catches the v6 defect: the collision Isaac builds (OBJ
    vertices plus position_m/quaternion) must be the object you see."""
    for asset_id, record in records.items():
        deviation, visual, collision = as_built_deviation(record)
        assert deviation <= AS_BUILT_TOLERANCE_M, (
            f'{asset_id}: as built, the collision AABB {collision.round(4).tolist()} is '
            f'{deviation * 1000:.2f} mm off the visual AABB {visual.round(4).tolist()}')


def test_meshed_collision_pieces_follow_the_asset_local_convention(records):
    """Meshed pieces are asset-local OBJs placed at the origin, so their vertices
    alone are their shape; analytic pieces carry their placement."""
    for asset_id, record in records.items():
        for part in record['collision_parts']:
            if part.get('geometry', 'convex_mesh') in ('mesh', 'convex_mesh'):
                assert np.allclose(part['position_m'], 0.0), \
                    f'{asset_id}/{part["label"]}: a meshed piece must sit at position_m 0'
                assert np.allclose(part['quaternion_xyzw'], [0, 0, 0, 1])


def test_as_built_check_catches_the_v6_double_shift():
    """Negative control on the retained v6 evidence: the same loader and the same
    measurements report the mug's 26.5 mm double shift, its sealed 17 mm handle
    bore and the handle hulls inside the drink, and pass the other six assets."""
    if not (V6_ROOT / 'manifest.json').is_file():
        pytest.skip('the v6 evidence library is not on disk')
    deviations = {asset_id: as_built_deviation(_load(asset_id, V6_ROOT))[0] for asset_id in ASSET_IDS}
    assert deviations['mug'] == pytest.approx(0.0265, abs=2e-4)
    assert all(value <= AS_BUILT_TOLERANCE_M for key, value in deviations.items() if key != 'mug')
    mug = _load('mug', V6_ROOT)
    # v6 states the post-recentre ring centre and a ring axis along x.
    bore = min(projected_clearance(as_built_points(part), mug['handle']['centre_m'], [1., 0., 0.])
               for part in mug['collision_parts'])
    assert bore < 1e-3, f'v6 bore measured {bore * 1000:.3f} mm; the check would not have caught it'
    base = next(part for part in mug['collision_parts'] if part['label'] == 'base')
    axis = [base['position_m'][0], base['position_m'][1], 0.0]
    interior = min(projected_clearance(as_built_points(part), axis, [0., 0., 1.])
                   for part in mug['collision_parts'] if part['label'] != 'base')
    assert interior < float(mug['inner_radius_m']) - 0.01


def test_module_guard_refuses_a_double_shifted_mesh():
    """build_library would refuse to write the v6 mug: _verify_as_built rebuilds
    the pieces as add_task_asset does."""
    mug = kitchen._recentre(kitchen._mug())
    shift = [0.0, -0.0265, 0.0]
    mug.collision = [(label, mesh, dict(component, position_m=shift) if mesh is not None else component)
                     for label, mesh, component in mug.collision]
    with pytest.raises(ValueError, match='deviates'):
        kitchen._verify_as_built('mug', mug)


def test_asset_aabb_is_centred_so_rest_pose_and_size_agree(records):
    """``validate_support`` uses ``position - size / 2``; that is only the real
    bottom when the as-built AABB is centred on the asset origin."""
    for asset_id, record in records.items():
        _, visual, collision = as_built_deviation(record)
        for bounds in (visual, collision):
            assert np.allclose(bounds[0], -bounds[1], atol=2e-4), \
                f'{asset_id}: as-built AABB {bounds.tolist()} is not centred on the origin'
        assert np.allclose(visual[1] - visual[0], record['size_m'], atol=1e-6)


def test_manifest_sizes_match_rest_geometry_and_design_fallbacks(records):
    for spec in SCENE['objects']:
        measured = np.asarray(records[spec['asset_id']]['size_m'], dtype=float)
        assert np.allclose(spec['rest_geometry']['size_m'], measured, atol=1e-6)
        assert spec['rest_geometry']['origin'] == 'centre'
    for asset_id, record in records.items():
        assert np.allclose(kitchen.DESIGN_SIZES[asset_id], record['size_m'], atol=1e-6), \
            f'{asset_id}: DESIGN_SIZES no longer matches the authored mesh'


def test_masses_are_plausible_design_values(records):
    expected = {'sponge': 0.008, 'mug': 0.28, 'ceramic_bowl': 0.25, 'cutting_board': 0.16,
                'paring_knife': 0.080, 'capped_bottle': 0.30, 'storage_jar': 0.30}
    for asset_id, mass in expected.items():
        record = records[asset_id]
        assert abs(record['mass_kg'] - mass) < 1e-9
        assert 'Design value' in record['mass_design_note']
        assert record['provenance']['source_kind'] == 'original_procedural'
        assert 'not manufacturer' in record['provenance']['dimension_status']


def test_every_visual_part_records_a_finish(records):
    """The loader's legacy default is metallic 0.65; a mug must not render as chrome."""
    for asset_id, record in records.items():
        for part in record['visual_parts']:
            assert 'roughness' in part and 'metallic' in part, f'{asset_id}/{part["label"]}'
    mug = records['mug']
    assert all(part['metallic'] == 0.0 for part in mug['visual_parts'])


# --- rule 2: bores and vessel interiors, as built ---------------------------

def test_declared_bores_are_sectors_with_a_stated_axis(records):
    bored_assets = [asset_id for asset_id, record in records.items()
                    if any(part.get('bore') is True for part in record['collision_parts'])]
    assert bored_assets == ['mug'], f'unexpected bore declarations: {bored_assets}'
    record = records['mug']
    bored = [part for part in record['collision_parts'] if part.get('bore') is True]
    assert len(bored) == kitchen.MUG_HANDLE_SECTORS
    for part in bored:
        assert part['geometry'] == 'convex_mesh', 'a bore is never one box'
        assert 'sector' in part, 'a bored ring must be split into sectors, not one hull'
        assert part['bore_radius_m'] == pytest.approx(record['handle']['hole_radius_m'])
    assert record['handle'].get('axis_direction') == [1.0, 0.0, 0.0], \
        'the manifest must state the bore axis it claims to keep open'
    # The sectors close the ring: every direction about the handle axis falls inside
    # some sector's angular span, so the hole is bounded all the way round.
    centre = np.asarray(record['handle']['centre_m'], dtype=float)
    spans = []
    for part in bored:
        rel = as_built_points(part)[:, 1:] - centre[1:]
        mid = math.atan2(rel[:, 1].mean(), rel[:, 0].mean())
        offsets = np.angle(np.exp(1j * (np.arctan2(rel[:, 1], rel[:, 0]) - mid)))
        spans.append((mid, float(np.abs(offsets).max())))
    for probe in np.linspace(0, 2 * math.pi, 720, endpoint=False):
        assert any(abs(np.angle(np.exp(1j * (probe - mid)))) <= half + 1e-9 for mid, half in spans), \
            f'the handle ring is open at {math.degrees(probe):.1f} deg'


def test_mug_handle_bore_is_open_as_built(records):
    """Exact projected distance from the handle axis to every convex piece."""
    record = records['mug']
    assert 'axis_direction' in record['handle'], 'the manifest must state the bore axis'
    centre, axis = record['handle']['centre_m'], record['handle']['axis_direction']
    bore = float(record['handle']['hole_radius_m'])
    assert bore >= 0.010, 'the handle bore is too small to be a finger hole'
    worst = min((projected_clearance(as_built_points(part), centre, axis), part['label'])
                for part in record['collision_parts'])
    assert worst[0] >= bore - 1e-6, (
        f'the {bore * 1000:.1f} mm bore is sealed as built: {worst[1]} reaches '
        f'{worst[0] * 1000:.3f} mm from the handle axis')


def test_no_finger_hole_sample_lies_inside_collision(records):
    """Sample the visible finger hole (disc of the bore, across the handle's
    thickness) and test every sample against every as-built convex piece."""
    record = records['mug']
    centre = np.asarray(record['handle']['centre_m'], dtype=float)
    bore = float(record['handle']['hole_radius_m'])
    half_t = float(record['handle']['minor_radius_m'])
    samples = []
    for r in np.linspace(0.0, bore - 5e-4, 9):
        for a in np.linspace(0, 2 * math.pi, 24, endpoint=False):
            for x in (-half_t * 0.9, 0.0, half_t * 0.9):
                samples.append(centre + [x, r * math.cos(a), r * math.sin(a)])
    samples = np.asarray(samples)
    for part in record['collision_parts']:
        hull = ConvexHull(as_built_points(part))
        inside = np.all(samples @ hull.equations[:, :3].T + hull.equations[:, 3] <= 1e-9, axis=1)
        assert not inside.any(), f'{part["label"]} covers {int(inside.sum())} finger-hole samples'


def test_vessels_keep_their_interiors_open_as_built(records):
    """No piece but the floor disc comes within the interior radius of the vessel
    axis, measured on every piece including the walls; the floor stays below the
    interior."""
    for asset_id in ('mug', 'ceramic_bowl'):
        record = records[asset_id]
        assert 'vessel' in record, f'{asset_id}: the manifest must state the vessel axis and interior'
        vessel = record['vessel']
        inner = float(vessel['inner_radius_m'])
        assert inner >= 0.03, f'{asset_id}: interior radius {inner} m is too small to hold anything'
        floors = [part for part in record['collision_parts'] if part.get('role') == 'vessel_floor']
        assert len(floors) == 1
        assert as_built_points(floors[0])[:, 2].max() <= vessel['floor_top_z_m'] + 1e-9
        worst = min((projected_clearance(as_built_points(part), vessel['axis_point_m'],
                                         vessel['axis_direction']), part['label'])
                    for part in record['collision_parts'] if part.get('role') != 'vessel_floor')
        assert worst[0] >= inner - 1e-6, (
            f'{asset_id}: {worst[1]} reaches {worst[0] * 1000:.2f} mm from the vessel axis, '
            f'inside the {inner * 1000:.1f} mm interior')
        # The axis really is the visual vessel's: the visual AABB in x-y is centred on it
        # for the bowl; for the mug the vessel body spans +-outer radius about it.
        visual = _bounds(as_built_points(part) for part in record['visual_parts']
                         if part['label'] in ('vessel', 'floor'))
        assert np.allclose((visual[0][:2] + visual[1][:2]) / 2, vessel['axis_point_m'][:2], atol=1e-6)


# --- rules 3-7: rest, thin parts, mass, reach, separation -------------------

def _support():
    surface = SCENE['surfaces'][0]
    centre = np.asarray(surface['position_m'], dtype=float)
    half = np.asarray(surface['half_extents_m'], dtype=float)
    return centre, half, float(centre[2] + half[2])


def test_no_object_floats_or_sinks_as_built(records):
    """The lowest as-built collision point (and visual point) of every object is
    on the work-table top within 1 mm, over the support footprint."""
    centre, half, top = _support()
    assert top == pytest.approx(kitchen.COUNTER_TOP_Z)
    for spec in SCENE['objects']:
        record = records[spec['asset_id']]
        for kind in ('collision', 'visual'):
            points = world_points(spec, np.vstack([as_built_points(part)
                                                   for part in record[f'{kind}_parts']]))
            lowest = float(points[:, 2].min())
            assert abs(lowest - top) <= REST_TOLERANCE_M, \
                f'{spec["id"]}: lowest {kind} point {lowest:.4f} m is not on the {top:.4f} m top'
            footprint = points[np.abs(points[:, 2] - lowest) < 1e-3][:, :2]
            assert np.all(footprint >= centre[:2] - half[:2] - 1e-6) and \
                np.all(footprint <= centre[:2] + half[:2] + 1e-6), f'{spec["id"]} overhangs its support'


def test_thin_parts_are_at_least_3mm_with_box_colliders(records):
    for asset_id, record in records.items():
        assert not thin_part_problems(record), f'{asset_id}: {thin_part_problems(record)}'
    knife = records['paring_knife']
    blade = next(p for p in knife['collision_parts'] if p['label'] == 'blade_box')
    handle = next(p for p in knife['collision_parts'] if p['label'] == 'handle_box')
    assert blade['geometry'] == 'box' and handle['geometry'] == 'box'
    assert 2 * blade['half_extents_m'][2] >= THIN_PART_MIN_M - 1e-9
    assert knife['blade_thickness_m'] >= THIN_PART_MIN_M - 1e-9
    # The blade box is the display blade, not a fattened stand-in.
    blade_visual = next(p for p in knife['visual_parts'] if p['label'] == 'blade')
    assert np.allclose(_bounds([as_built_points(blade_visual)]), _bounds([as_built_points(blade)]),
                       atol=1e-6)
    # The two boxes abut and share one bottom plane.
    assert blade['position_m'][1] - blade['half_extents_m'][1] == pytest.approx(
        handle['position_m'][1] + handle['half_extents_m'][1], abs=1e-9)
    assert blade['position_m'][2] - blade['half_extents_m'][2] == pytest.approx(
        handle['position_m'][2] - handle['half_extents_m'][2], abs=1e-9)
    board = records['cutting_board']
    assert [p['geometry'] for p in board['collision_parts']] == ['box']


def test_thin_part_check_catches_the_v6_blade():
    if not (V6_ROOT / 'manifest.json').is_file():
        pytest.skip('the v6 evidence library is not on disk')
    problems = thin_part_problems(_load('paring_knife', V6_ROOT))
    assert problems and 'blade_box' in problems[0] and '2.00 mm' in problems[0]


def test_objects_are_inside_the_reach_envelope():
    base = np.asarray(SCENE['robot']['base_position_m'], dtype=float)
    for spec in SCENE['objects']:
        position = np.asarray(spec['position_m'], dtype=float)
        assert REACH_X[0] <= position[0] <= REACH_X[1], f'{spec["id"]} x={position[0]}'
        assert REACH_Y[0] <= position[1] <= REACH_Y[1], f'{spec["id"]} y={position[1]}'
        assert np.linalg.norm(position - base) <= REACH_RADIUS_M, f'{spec["id"]} is out of reach'


def _object_world_boxes(records):
    boxes = {}
    for spec in SCENE['objects']:
        record = records[spec['asset_id']]
        points = world_points(spec, np.vstack([as_built_points(p) for p in record['collision_parts']]))
        boxes[spec['id']] = (points.min(axis=0), points.max(axis=0), points)
    return boxes


def test_objects_do_not_overlap_each_other_or_any_static_collider(records):
    boxes = _object_world_boxes(records)
    for (a, (lo_a, hi_a, pts_a)), (b, (lo_b, hi_b, pts_b)) in itertools.combinations(boxes.items(), 2):
        # Separating axis on the as-built footprints (each object's own hull in x-y).
        hull_a, hull_b = ConvexHull(pts_a[:, :2]), ConvexHull(pts_b[:, :2])
        normals = np.r_[hull_a.equations[:, :2], hull_b.equations[:, :2]]
        separated = any((pts_a[:, :2] @ n).max() <= (pts_b[:, :2] @ n).min() + 1e-9 or
                        (pts_b[:, :2] @ n).max() <= (pts_a[:, :2] @ n).min() + 1e-9 for n in normals)
        assert separated, f'{a} and {b} overlap on the counter'
    for name, (lo, hi, _) in boxes.items():
        for owner, part_id, s_lo, s_hi in static_boxes(SCENE):
            overlap = np.minimum(hi, s_hi) - np.maximum(lo, s_lo)
            assert not np.all(overlap > 1e-6), f'{name} interpenetrates {owner}/{part_id}'


def test_pedestal_and_table_are_separate_bodies():
    pedestal, table = SCENE['pedestal'], SCENE['table']
    p_lo = np.asarray(pedestal['position_m']) - np.asarray(pedestal['half_extents_m'])
    p_hi = np.asarray(pedestal['position_m']) + np.asarray(pedestal['half_extents_m'])
    t_lo = np.asarray(table['position_m']) - np.asarray(table['half_extents_m'])
    assert t_lo[0] >= 0.25 and t_lo[0] > p_hi[0], 'the work table must start in front of the pedestal'
    assert SCENE['robot']['base_position_m'][2] == pytest.approx(p_hi[2])
    assert p_lo[2] - pedestal['skirt_height_m'] == pytest.approx(0.0, abs=1e-9), \
        'the pedestal skirt must reach the floor'


# --- the static world: one collider per surface, every material bound -------

def static_boxes(scene):
    """Every static solid scene.py builds: props, work table, surfaces, pedestal."""
    boxes = []
    for spec in scene.get('static_props', []):
        for part in spec['parts']:
            c, h = np.asarray(part['position_m'], dtype=float), np.asarray(part['half_extents_m'], dtype=float)
            boxes.append((spec['id'], part['id'], c - h, c + h))
    table = scene['table']
    c, h = np.asarray(table['position_m'], dtype=float), np.asarray(table['half_extents_m'], dtype=float)
    boxes.append(('work_table', 'top', c - h, c + h))
    for spec in scene.get('surfaces', []):
        c, h = np.asarray(spec['position_m'], dtype=float), np.asarray(spec['half_extents_m'], dtype=float)
        boxes.append((spec['id'], 'plate', c - h, c + h))
    pedestal = scene['pedestal']
    c, h = np.asarray(pedestal['position_m'], dtype=float), np.asarray(pedestal['half_extents_m'], dtype=float)
    boxes.append(('pedestal', 'top', c - h, c + h))
    skirt = float(pedestal.get('skirt_height_m', 0.0))
    if skirt > 0:
        boxes.append(('pedestal', 'skirt', np.r_[c[:2] - h[:2], c[2] - h[2] - skirt],
                      np.r_[c[:2] + h[:2], c[2] - h[2]]))
    return boxes


SCHEMA_TWINS = {('work_table', 'counter_top')}


def static_collider_problems(scene):
    """Pairs of static colliders that coincide or overlap, across owners."""
    problems = []
    for (o_a, p_a, lo_a, hi_a), (o_b, p_b, lo_b, hi_b) in itertools.combinations(static_boxes(scene), 2):
        if o_a == o_b or tuple(sorted((o_a, o_b))) in {tuple(sorted(t)) for t in SCHEMA_TWINS}:
            continue
        overlap = np.minimum(hi_a, hi_b) - np.maximum(lo_a, lo_b)
        if np.all(overlap > 1e-6):
            problems.append(f'{o_a}/{p_a} overlaps {o_b}/{p_b}')
    # Within one prop, no two parts may be the same box (coplanar duplicates).
    for spec in scene.get('static_props', []):
        seen = {}
        for part in spec['parts']:
            key = tuple(np.round(part['position_m'], 6)) + tuple(np.round(part['half_extents_m'], 6))
            if key in seen:
                problems.append(f'{spec["id"]}/{part["id"]} duplicates {seen[key]}')
            seen[key] = part['id']
    # Nothing in a prop may coincide with the work-table box.
    table = scene['table']
    t_key = tuple(np.round(table['position_m'], 6)) + tuple(np.round(table['half_extents_m'], 6))
    for spec in scene.get('static_props', []):
        for part in spec['parts']:
            if tuple(np.round(part['position_m'], 6)) + tuple(np.round(part['half_extents_m'], 6)) == t_key:
                problems.append(f'{spec["id"]}/{part["id"]} coincides with the work table')
    return problems


def material_problems(scene):
    declared = set(scene['materials'])
    referenced = {('room', key): name for key, name in scene['room']['surfaces'].items()}
    referenced[('table', 'material')] = scene['table'].get('material', 'table')
    for spec in scene.get('surfaces', []):
        referenced[(spec['id'], 'material')] = spec.get('material', 'prop')
    for spec in scene.get('static_props', []):
        for part in spec['parts']:
            referenced[(spec['id'], part['id'])] = part.get('material', spec.get('material', 'prop'))
    return [f'{owner}/{key} binds undeclared material {name!r}'
            for (owner, key), name in referenced.items() if name not in declared]


def test_no_two_static_colliders_coincide_or_overlap():
    problems = static_collider_problems(SCENE)
    assert not problems, problems


def test_the_table_and_its_surface_twin_are_identical():
    """validate_family reads supports only from surfaces/static props, so the
    table box is restated once in surfaces: same box, same material, same friction."""
    table = SCENE['table']
    twins = [s for s in SCENE['surfaces']
             if np.allclose(s['position_m'], table['position_m']) and
             np.allclose(s['half_extents_m'], table['half_extents_m'])]
    assert len(twins) == 1 and len(SCENE['surfaces']) == 1
    twin = twins[0]
    assert twin.get('material') == table.get('material') == 'counter_quartz'
    assert twin.get('friction') == table.get('friction')
    supports = validate_family(SCENE)['supports']
    assert [s['id'] for s in supports if s['id'] == twin['id']] == [twin['id']]


def test_every_bound_material_is_declared_and_well_formed():
    from bench.simulation.isaacsim_backend.visuals import TEXTURE_KEYS, validate_material
    assert not material_problems(SCENE), material_problems(SCENE)
    for name, spec in SCENE['materials'].items():
        record = validate_material(spec, name)
        if record['kind'] == 'color':
            assert all(0.0 <= value <= 1.0 for value in record['color'])
        else:
            # Every declared map must resolve to a real file; the albedo map is required.
            assert 'base_color' in record['textures'], f'{name} has no albedo map'
            for key, path in record['textures'].items():
                assert key in TEXTURE_KEYS and Path(path).is_file()
            assert 0.0 <= record['roughness'] <= 1.0


def test_static_checks_catch_the_v6_worktop_and_material_defects():
    """Negative controls: the v6 top_main (a third box on the table) and the
    undeclared 'table' material are both reported."""
    broken = copy.deepcopy(SCENE)
    counter = next(spec for spec in broken['static_props'] if spec['id'] == 'counter_run')
    counter['parts'].append({'id': 'top_main', 'geometry': 'box', 'material': 'counter_quartz',
                             'position_m': list(SCENE['table']['position_m']),
                             'half_extents_m': list(SCENE['table']['half_extents_m'])})
    problems = static_collider_problems(broken)
    assert any('top_main' in p and 'work_table' in p for p in problems)
    broken['table']['material'] = 'table'
    assert any("'table'" in p for p in material_problems(broken))


def test_every_static_prop_is_immovable_boxes_inside_the_room():
    room = validate_room(SCENE['room'])
    low, high = _room_interior(room)
    for spec in SCENE['static_props']:
        assert spec['parts'] and 'asset_id' not in spec
        for part in spec['parts']:
            assert part.get('geometry', 'box') == 'box'
            c, h = np.asarray(part['position_m']), np.asarray(part['half_extents_m'])
            assert np.all(h > 0)
            assert np.all(c - h >= low - 1e-6) and np.all(c + h <= high + 1e-6), \
                f'{spec["id"]}/{part["id"]} passes through the room shell'
    dynamic_ids = {spec['id'] for spec in SCENE['objects']}
    assert not dynamic_ids & {spec['id'] for spec in SCENE['static_props']}
    assert SCENE['fixtures'] == []


def test_no_static_prop_overhangs_the_work_strip():
    """An overhead cabinet or hood over the work strip would collide with a lifted object."""
    for spec in SCENE['static_props']:
        for part in spec['parts']:
            lo = np.asarray(part['position_m']) - np.asarray(part['half_extents_m'])
            hi = np.asarray(part['position_m']) + np.asarray(part['half_extents_m'])
            over = lo[0] < 0.80 and hi[0] > 0.38 and lo[1] < REACH_Y[1] and hi[1] > REACH_Y[0]
            assert not (over and lo[2] > kitchen.COUNTER_TOP_Z - 1e-9), \
                f'{spec["id"]}/{part["id"]} overhangs the work strip at z={lo[2]}'


# --- the room is complete and closed ---------------------------------------

def _room_interior(room):
    size, origin = np.asarray(room['size_m'], dtype=float), np.asarray(room['origin_m'], dtype=float)
    low = np.array([origin[0] - size[0] / 2, origin[1] - size[1] / 2, origin[2]])
    high = np.array([origin[0] + size[0] / 2, origin[1] + size[1] / 2, origin[2] + room['ceiling_height_m']])
    return low, high


def test_room_has_floor_four_walls_ceiling_lights_and_dome():
    room = validate_family(SCENE)['room']
    assert room['enabled']
    for key in ('floor', 'wall', 'ceiling', 'baseboard'):
        assert room['surfaces'][key] in SCENE['materials']
    assert room['collide']['floor'] is True and room['collide']['walls'] is True
    by_face = {}
    for opening in room['openings']:
        by_face.setdefault(opening['wall'], []).append(opening)
        assert opening['kind'] in ('door', 'window')
    assert {o['kind'] for o in room['openings']} == {'door', 'window'}
    for face in WALL_FACES:
        wall = _wall_geometry(room, face)
        quads = _wall_quads(wall, by_face.get(face, []))
        assert quads, f'the {face} wall has no surface'
        area = sum((hu - lu) * (hv - lv) for lu, hu, lv, hv in quads)
        full = 4 * wall['half'][0] * wall['half'][1]
        holes = sum(o['size_m'][0] * o['size_m'][1] for o in by_face.get(face, []))
        assert area == pytest.approx(full - holes, rel=1e-9), f'the {face} wall is not closed around its openings'
        for opening in by_face.get(face, []):
            (u, v), (su, sv) = opening['center_m'], opening['size_m']
            assert abs(u) + su / 2 <= wall['half'][0] and abs(v) + sv / 2 <= wall['half'][1]
    assert room['ceiling_height_m'] > 2.0 and room['baseboard_m'] > 0
    assert room['ceiling_lights'] and all(l['position_m'][2] < room['ceiling_height_m']
                                          for l in room['ceiling_lights'])
    assert room['dome']['enabled']


# --- cameras ----------------------------------------------------------------

def test_every_camera_resolves():
    cameras = resolve_cameras(SCENE)
    assert set(cameras) == {'third_person', 'counter_view', 'wrist'}
    for camera in cameras.values():
        assert camera['focal_length_mm'] > 0
    wrist = cameras['wrist']
    assert wrist['kind'] == 'link' and wrist['mount_link'] == 'right_hand_C_MC'
    assert wrist['aim_source'] == 'default_work_table_top'
    assert np.allclose(wrist['aim_target_m'], kitchen.WORK_TABLE_TOP_CENTRE)


def test_third_person_camera_is_in_the_room_and_frames_the_table():
    room = validate_room(SCENE['room'])
    low, high = _room_interior(room)
    camera = resolve_cameras(SCENE)['third_person']
    eye, target = np.asarray(camera['eye']), np.asarray(camera['target'])
    assert np.all(eye > low) and np.all(eye < high), f'the eye {eye} is outside {low}..{high}'
    for owner, part_id, lo, hi in static_boxes(SCENE):
        assert not (np.all(eye > lo) and np.all(eye < hi)), f'the eye is inside {owner}/{part_id}'
    assert np.allclose(target, kitchen.WORK_TABLE_TOP_CENTRE), 'aim at the work-table centre'
    assert 1.5 <= float(np.linalg.norm(eye - target)) <= 2.2
    assert 0.6 <= float(eye[2] - kitchen.COUNTER_TOP_Z) <= 1.4


# --- what the cameras see -----------------------------------------------------

def _camera_frame(camera):
    eye = np.asarray(camera['eye'], dtype=float)
    forward = np.asarray(camera['target'], dtype=float) - eye
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, [0., 0., 1.])
    right /= np.linalg.norm(right)
    return eye, forward, right, np.cross(right, forward)


def in_frame(camera, points):
    eye, forward, right, up = _camera_frame(camera)
    rel = np.asarray(points, dtype=float) - eye
    depth = rel @ forward
    width, height = camera['resolution']
    v_aperture = min(V_APERTURE_DEFAULT_MM, H_APERTURE_MM * height / width)
    with np.errstate(divide='ignore', invalid='ignore'):
        x = (rel @ right) / depth * camera['focal_length_mm']
        y = (rel @ up) / depth * camera['focal_length_mm']
    return (depth > 1e-6) & (np.abs(x) <= H_APERTURE_MM / 2) & (np.abs(y) <= v_aperture / 2)


def segment_hits_box(eye, points, lo, hi, eps=1e-4):
    d = np.asarray(points, dtype=float) - eye
    with np.errstate(divide='ignore', invalid='ignore'):
        t1, t2 = (lo - eye) / d, (hi - eye) / d
    parallel = np.abs(d) < 1e-12
    inside = (eye >= lo) & (eye <= hi)
    t_min = np.where(parallel, np.where(inside, -np.inf, np.inf), np.minimum(t1, t2))
    t_max = np.where(parallel, np.where(inside, np.inf, -np.inf), np.maximum(t1, t2))
    enter, leave = t_min.max(axis=1), t_max.min(axis=1)
    return (enter < leave) & (leave > eps) & (enter < 1 - eps)


@pytest.fixture(scope='module')
def robot_links():
    """FR3 + SHARPA collision meshes posed at the scene's home pose (URDF FK)."""
    import trimesh
    robot = SCENE['robot']
    urdf = (Path(robot['assets_root']) / 'robots/assemblies/fr3_sharpa' / robot['assembly_version']
            / 'fr3_sharpa.urdf')
    if not urdf.is_file():
        pytest.skip(f'robot assembly not available: {urdf}')
    base = np.eye(4)
    base[:3, 3] = robot['base_position_m']
    home = {f'fr3_joint{i}': value for i, value in enumerate(robot['arm_home_rad'], start=1)}
    kinematics = URDFKinematics(urdf)
    for joint in kinematics.active:
        if joint.name in home:
            assert joint.lower <= home[joint.name] <= joint.upper, f'{joint.name} home is out of limits'
    poses = kinematics.forward(home, base)
    links = []
    for link in ET.parse(urdf).getroot().findall('link'):
        for element in link.findall('collision'):
            shape = element.find('geometry/mesh')
            if shape is None:
                continue
            local = np.eye(4)
            origin = element.find('origin')
            if origin is not None:
                from bench.geometry import rpy_matrix
                local[:3, 3] = np.fromstring(origin.get('xyz', '0 0 0'), sep=' ')
                local[:3, :3] = rpy_matrix(np.fromstring(origin.get('rpy', '0 0 0'), sep=' '))
            mesh = trimesh.load(urdf.parent / shape.get('filename'), force='mesh', process=False)
            vertices = np.asarray(mesh.vertices, dtype=float) * np.fromstring(shape.get('scale', '1 1 1'), sep=' ')
            world = (poses[link.get('name')] @ np.c_[vertices, np.ones(len(vertices))].T).T[:, :3]
            triangles = world[np.asarray(mesh.faces)]
            links.append((link.get('name'), world.min(axis=0), world.max(axis=0), triangles))
    assert len(links) >= 30
    return links


def segment_hits_robot(eye, points, links, eps=1e-4):
    hits = np.zeros(len(points), dtype=bool)
    for index, point in enumerate(np.asarray(points, dtype=float)):
        d = point - eye
        for _, lo, hi, triangles in links:
            if not segment_hits_box(eye, point[None], lo - 1e-3, hi + 1e-3)[0]:
                continue
            v0, e1, e2 = triangles[:, 0], triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
            h = np.cross(d, e2)
            a = np.einsum('ij,ij->i', e1, h)
            ok = np.abs(a) > 1e-14
            f = np.where(ok, 1.0 / np.where(ok, a, 1.0), 0.0)
            s = eye - v0
            u = f * np.einsum('ij,ij->i', s, h)
            q = np.cross(s, e1)
            v = f * (q @ d)
            t = f * np.einsum('ij,ij->i', e2, q)
            if np.any(ok & (u >= 0) & (u <= 1) & (v >= 0) & (u + v <= 1) & (t > eps) & (t < 1 - eps)):
                hits[index] = True
                break
    return hits


def _face_samples(lo, hi, eye, n):
    samples = []
    grid = (np.arange(n) + 0.5) / n
    for axis in range(3):
        others = [a for a in range(3) if a != axis]
        for side, sign in ((lo[axis], -1), (hi[axis], 1)):
            if (eye[axis] - side) * sign <= 0:
                continue                                  # a face turned away from the camera
            for a in grid:
                for b in grid:
                    p = np.empty(3)
                    p[axis] = side
                    p[others[0]] = lo[others[0]] + a * (hi[others[0]] - lo[others[0]])
                    p[others[1]] = lo[others[1]] + b * (hi[others[1]] - lo[others[1]])
                    samples.append(p)
    return np.asarray(samples).reshape(-1, 3)


def visible_fraction(camera, owner, links, object_boxes, *, parts=None, n=6):
    """Fraction of an owner's camera-facing surface samples in frame and unoccluded
    by any other static collider, any object or the robot at home."""
    eye = np.asarray(camera['eye'], dtype=float)
    boxes = static_boxes(SCENE)
    mine = [b for b in boxes if b[0] == owner and (parts is None or b[1] in parts)]
    assert mine, f'no static collider named {owner}'
    others = [b for b in boxes if b[0] != owner] + [(name, 'object', lo, hi)
                                                   for name, (lo, hi, _) in object_boxes.items()]
    samples = np.vstack([_face_samples(lo, hi, eye, n) for _, _, lo, hi in mine])
    visible = in_frame(camera, samples)
    for _, _, lo, hi in others:
        visible &= ~segment_hits_box(eye, samples, lo, hi)
    index = np.where(visible)[0]
    visible[index] &= ~segment_hits_robot(eye, samples[index], links)
    return float(visible.mean())


# Measured on the scene (third_person / counter_view): hob 1.00, sink and
# tap 0.54 / 0.56 (the tap alone 0.93 from third_person), fridge 0.32 / 0.30,
# upper cabinets 0.17, range hood 0.08. The oven door faces the robot, so neither
# world camera sees it; it is not required here. The floors below leave margin
# without letting a fixture fall out of view.
KITCHEN_FIXTURES = {
    'third_person': {'hob': 0.80, 'sink_basin': 0.40, 'fridge': 0.20, 'upper_cabinets': 0.10,
                     'range_hood': 0.04},
    'counter_view': {'sink_basin': 0.40, 'fridge': 0.20},
}


def kitchen_view_problems(camera_name, cameras, links, object_boxes):
    problems = []
    for owner, floor in KITCHEN_FIXTURES[camera_name].items():
        fraction = visible_fraction(cameras[camera_name], owner, links, object_boxes)
        if fraction < floor:
            problems.append(f'{camera_name}: {owner} is {fraction:.1%} visible (< {floor:.0%})')
    return problems


def test_kitchen_defining_fixtures_exist_as_static_props():
    ids = {spec['id']: spec for spec in SCENE['static_props']}
    for required in ('counter_run', 'hob', 'oven', 'range_hood', 'sink_basin', 'fridge',
                     'upper_cabinets', 'backsplash'):
        assert required in ids, f'a kitchen needs a {required}'
    assert {'tap_column', 'tap_spout'} <= {part['id'] for part in ids['sink_basin']['parts']}
    # The hob sits on the worktop and the hood hangs over it.
    glass = next(p for p in ids['hob']['parts'] if p['id'] == 'glass')
    assert glass['position_m'][2] - glass['half_extents_m'][2] == pytest.approx(kitchen.COUNTER_TOP_Z)
    canopy = next(p for p in ids['range_hood']['parts'] if p['id'] == 'canopy')
    assert abs(canopy['position_m'][1] - glass['position_m'][1]) < 0.05
    assert 0.6 <= canopy['position_m'][2] - canopy['half_extents_m'][2] - kitchen.COUNTER_TOP_Z <= 0.8
    # The fridge stands on the floor, tall, with door clearance from the run.
    body = next(p for p in ids['fridge']['parts'] if p['id'] == 'body')
    assert body['position_m'][2] - body['half_extents_m'][2] == pytest.approx(0.0)
    assert body['position_m'][2] + body['half_extents_m'][2] >= 1.7
    assert body['position_m'][1] - body['half_extents_m'][1] - kitchen.RUN_Y1 >= 0.6


def test_views_read_as_a_kitchen(robot_links, records):
    cameras = resolve_cameras(SCENE)
    boxes = _object_world_boxes(records)
    problems = []
    for name in KITCHEN_FIXTURES:
        problems += kitchen_view_problems(name, cameras, robot_links, boxes)
    assert not problems, problems
    tap = visible_fraction(cameras['third_person'], 'sink_basin', robot_links, boxes,
                           parts={'tap_column', 'tap_spout', 'tap_nozzle'})
    assert tap >= 0.5, f'the tap is only {tap:.0%} visible'


def test_view_check_catches_the_v6_camera(robot_links, records):
    """Negative control: the v6 third-person camera, aimed across the robot, misses
    the sink, fridge, cabinets and hood of this kitchen."""
    cameras = resolve_cameras(SCENE)
    cameras['third_person'] = dict(cameras['third_person'], eye=[-0.58, -0.82, 1.74],
                                   target=[0.58, 0.15, 0.93], focal_length_mm=24.0)
    problems = kitchen_view_problems('third_person', cameras, robot_links, _object_world_boxes(records))
    assert len(problems) >= 3, problems


def _object_samples(points, camera):
    """Corners, edge midpoints and the top centre of an object's as-built hull,
    drawn 5% towards its centre so a sample is never on a grazing silhouette."""
    lo, hi = points.min(axis=0), points.max(axis=0)
    centre = (lo + hi) / 2
    grid = [lo + (hi - lo) * np.array([a, b, c]) for a in (0, .5, 1) for b in (0, .5, 1) for c in (0, .5, 1)]
    grid = np.asarray([g for g in grid if not np.allclose(g, centre)])
    return centre + 0.95 * (grid - centre)


def test_no_object_is_hidden_by_props_other_objects_or_the_arm(robot_links, records):
    """At the home pose (URDF FK, collision meshes) the arm hides no object from
    either world camera, and no prop does."""
    cameras = resolve_cameras(SCENE)
    boxes = _object_world_boxes(records)
    for name in ('third_person', 'counter_view'):
        camera = cameras[name]
        eye = np.asarray(camera['eye'], dtype=float)
        for object_id, (lo, hi, points) in boxes.items():
            samples = _object_samples(points, camera)
            assert in_frame(camera, samples).all(), f'{name}: {object_id} leaves the frame'
            for owner, part_id, s_lo, s_hi in static_boxes(SCENE):
                blocked = segment_hits_box(eye, samples, s_lo, s_hi)
                assert not blocked.any(), f'{name}: {owner}/{part_id} hides {object_id}'
            top = np.array([[(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, hi[2] - 1e-3]])
            assert not segment_hits_robot(eye, top, robot_links)[0], \
                f'{name}: the arm at home hides the top of {object_id}'
            hidden = segment_hits_robot(eye, samples, robot_links)
            assert not hidden.any(), f'{name}: the arm at home hides {int(hidden.sum())} samples of {object_id}'


def test_arm_occlusion_check_has_teeth(robot_links):
    """Negative control: a point straight behind the palm, seen from the
    third-person eye, is reported hidden."""
    camera = resolve_cameras(SCENE)['third_person']
    eye = np.asarray(camera['eye'], dtype=float)
    palm = next(t for name, lo, hi, t in robot_links if name == 'right_hand_C_MC')
    centre = palm.reshape(-1, 3).mean(axis=0)
    behind = eye + 1.3 * (centre - eye)
    assert segment_hits_robot(eye, behind[None], robot_links)[0]


def test_robot_home_pose_is_clear_of_every_static_collider(robot_links):
    for name, lo, hi, triangles in robot_links:
        vertices = triangles.reshape(-1, 3)
        for owner, part_id, s_lo, s_hi in static_boxes(SCENE):
            if owner == 'pedestal':
                continue                                   # the base plate stands on it
            inside = np.all((vertices > s_lo + 1e-4) & (vertices < s_hi - 1e-4), axis=1)
            assert not inside.any(), f'{name} at home starts inside {owner}/{part_id}'
