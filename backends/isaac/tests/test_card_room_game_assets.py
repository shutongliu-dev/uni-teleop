"""CPU asset contracts; these checks do not establish native visual acceptance."""
import hashlib
import math
from pathlib import Path

import numpy as np
import pytest
import trimesh

from bench.paths import load_json
from scenes.board_games.hex import HexGeometry
from scenes.board_games.tabletop.geometry import GAME_IDS, geometry_for
from scenes.workrooms.card_room import game_assets as assets
from scenes.workrooms.card_room.tasks.hex7x7 import assets as heritage
from bench.simulation.assets.task_parts import load_task_asset


def _luminance(color):
    # USD's diffuseColor inputs here are linear material values, not rendered
    # pixels. This is a reflectance-order contract, not a camera contrast claim.
    return np.dot(color, [.2126, .7152, .0722])


@pytest.mark.parametrize('game_id', GAME_IDS)
@pytest.mark.parametrize('color,old_color', [('black', 'red'), ('white', 'blue')])
def test_piece_physics_is_exactly_the_existing_analytic_prior(game_id, color, old_color):
    g = geometry_for(game_id)
    current, original = assets.token(color, g), heritage.token(old_color, HexGeometry())
    assert current.collision == original.collision
    assert current.mass_kg == original.mass_kg == .060
    assert current.friction == original.friction == .65
    assert current.center_of_mass_m == original.center_of_mass_m
    for field in ('inertia_diag_kg_m2', 'inertia_method', 'design_density_kg_m3',
                  'mass_status', 'origin', 'design_dimensions'):
        assert current.data[field] == original.data[field]
    # Independent integration of the actual collision dimensions, rather than
    # assuming that a visual mesh or a bounding-box mass tensor is equivalent.
    cylinders = []
    for _, _, part in current.collision:
        body = trimesh.creation.cylinder(radius=part['radius_m'],
                                         height=2*part['half_height_m'], sections=256)
        body.apply_translation(part['position_m'])
        cylinders.append(body)
    body = trimesh.util.concatenate(cylinders)
    body.density = current.mass_kg/body.volume
    np.testing.assert_allclose(body.center_mass, current.center_of_mass_m, atol=1.e-12)
    np.testing.assert_allclose(np.diag(body.moment_inertia),
                               current.data['inertia_diag_kg_m2'], rtol=1.e-3)


@pytest.mark.parametrize('color', ['black', 'white'])
def test_grain_partitions_a_closed_token_surface_inside_the_original_envelope(color):
    current = assets.token(color, geometry_for('hex7x7'))
    original = heritage.token('red', HexGeometry())
    assert len(current.visual) == 6
    assert sum(len(mesh.faces) for _, mesh, _ in current.visual) < 20000
    for label, original_mesh, _ in original.visual:
        parts = [mesh for name, mesh, _ in current.visual if name.startswith(label+'_')]
        surface = trimesh.util.concatenate(parts)
        surface.merge_vertices()
        assert surface.is_watertight
        assert surface.is_winding_consistent
        assert surface.volume == pytest.approx(original_mesh.volume, abs=1.e-14)
        np.testing.assert_allclose(surface.bounds, original_mesh.bounds, atol=1.e-12)
        collider = next(part for name, _, part in current.collision if name == label)
        local = surface.vertices-np.asarray(collider['position_m'])
        radii = np.linalg.norm(local[:, :2], axis=1)
        assert radii.max() <= collider['radius_m']+1.e-12
        assert abs(local[:, 2]).max() <= collider['half_height_m']+1.e-12
        if label == 'handle':
            straight = local[np.isclose(radii, collider['radius_m'], atol=1.e-12), 2]
            assert np.ptp(straight) >= .036
    for roughness, metallic in current.visual_finish.values():
        assert .60 <= roughness <= .72
        assert metallic == 0


@pytest.mark.parametrize('sign', [-1, 1])
def test_pedestals_match_retreated_lowered_base_and_preserve_original_factory(sign):
    from scenes.workrooms.card_room.robot_layout import ROBOT_BASE_DISTANCE_M, ROBOT_BASE_HEIGHT_M
    before = heritage.stand(sign)
    current = assets.stand(sign)
    after = heritage.stand(sign)
    lowered = heritage.stand(sign, ROBOT_BASE_HEIGHT_M)
    offset = np.array([0., sign*(ROBOT_BASE_DISTANCE_M-.52), 0.])
    assert current.asset_id == ('stand_left' if sign < 0 else 'stand_right')
    assert before.collision == after.collision
    for (_, mesh, actual), (_, _, expected) in zip(current.collision, lowered.collision):
        assert mesh is None
        assert actual['half_extents_m'] == expected['half_extents_m']
        np.testing.assert_allclose(actual['position_m'], np.asarray(expected['position_m'])+offset)
    foot = current.collision[0][2]
    assert foot['position_m'][2]-foot['half_extents_m'][2] == pytest.approx(0.)
    mount = current.collision[-1][2]
    assert mount['position_m'][2]+mount['half_extents_m'][2] == pytest.approx(ROBOT_BASE_HEIGHT_M)
    assert mount['position_m'][1] == pytest.approx(sign*ROBOT_BASE_DISTANCE_M)
    assert current.friction == before.friction == .4
    assert current.mass_kg == before.mass_kg == 0
    assert current.dynamic is False
    assert after.data == before.data
    assert len(current.collision) == 4
    for (_, new_mesh, new_color), (_, old_mesh, old_color) in zip(current.visual, lowered.visual):
        np.testing.assert_allclose(new_mesh.vertices, old_mesh.vertices+offset)
        np.testing.assert_array_equal(new_mesh.faces, old_mesh.faces)
        assert new_color == old_color
    assert current.visual_finish == before.visual_finish


@pytest.mark.parametrize('game_id', GAME_IDS)
def test_board_encloses_every_cell_with_one_declared_solid_support(game_id):
    g = geometry_for(game_id)
    current = assets.board(g)
    half = np.asarray(g.board_size_xy_m)/2
    cell_vertices = np.asarray([g.world_to_local_xy(*v)
                                for q in range(g.size) for r in range(g.size)
                                for v in g.cell_vertices(q, r)])
    np.testing.assert_allclose(cell_vertices.min(0)+half, g.board_margin_m, atol=1.e-12)
    np.testing.assert_allclose(half-cell_vertices.max(0), g.board_margin_m, atol=1.e-12)
    assert len(current.collision) == 1
    _, collision_mesh, collider = current.collision[0]
    assert collision_mesh is None
    assert collider['geometry'] == 'box'
    np.testing.assert_allclose(collider['half_extents_m'], [*half, .006], atol=1.e-12)
    np.testing.assert_allclose(collider['position_m'], [0, 0, .006], atol=1.e-12)
    assert current.friction == .4
    assert current.mass_kg == 0
    assert current.dynamic is False
    for name, mesh, _ in current.visual:
        assert np.isfinite(mesh.vertices).all(), name
        assert (mesh.area_faces > 1.e-18).all(), name
        assert (mesh.bounds[0, :2] >= -half-1.e-12).all(), name
        assert (mesh.bounds[1, :2] <= half+1.e-12).all(), name
        assert mesh.bounds[0, 2] >= -1.e-12, name
        assert mesh.bounds[1, 2] <= .012+assets.MAXIMUM_BOARD_TOP_OFFSET_M+1.e-12, name
    # Geometry count guards against replacing wood finish with dense raised
    # stripe boxes; these native-loader-compatible veneers remain small.
    assert sum(len(mesh.faces) for _, mesh, _ in current.visual) < 12000


@pytest.mark.parametrize('game_id', GAME_IDS)
def test_board_markings_have_complete_cell_edges_and_readable_authored_width(game_id):
    g = geometry_for(game_id)
    current = assets.board(g)
    grid = next(mesh for name, mesh, _ in current.visual if name.endswith('_grid'))
    # Every marking is a four-corner flat ribbon with two triangles. Its short
    # side must retain the intended width after OBJ-compatible construction.
    ribbons = grid.vertices.reshape(-1, 4, 3)
    expected_count = 2*(g.size+1) if g.shape == 'square' else 3*g.size*g.size+4*g.size-1
    assert len(ribbons) == expected_count
    widths = np.linalg.norm(ribbons[:, 1]-ribbons[:, 0], axis=1)
    np.testing.assert_allclose(widths, .0022, atol=1.e-12)
    assert .0022/g.pitch_m >= .015
    assert len(current.data['coordinate_labels']) == 2*g.size
    if g.shape == 'square':
        extent = g.size*g.pitch_m/2
        centerlines = np.mean(ribbons.reshape(-1, 2, 2, 3), axis=2)[:, :, :2]
        assert np.allclose(np.linalg.norm(centerlines[:, 1]-centerlines[:, 0], axis=1), 2*extent)
    else:
        # A regular hex edge is pitch/sqrt(3); no clipped or missing edge can
        # masquerade as the complete Hex field's 174 boundaries.
        centerlines = np.mean(ribbons.reshape(-1, 2, 2, 3), axis=2)[:, :, :2]
        np.testing.assert_allclose(np.linalg.norm(centerlines[:, 1]-centerlines[:, 0], axis=1),
                                   g.pitch_m/math.sqrt(3), atol=2.e-9)


@pytest.mark.parametrize('game_id', GAME_IDS)
def test_white_black_and_walnut_have_separate_material_ranges(game_id):
    current = assets.board(geometry_for(game_id))
    field = [color for name, _, color in current.visual
             if name == 'walnut_playing_surface' or name.startswith('walnut_veneer')]
    white = [color for _, _, color in assets.token('white', geometry_for(game_id)).visual]
    black = [color for _, _, color in assets.token('black', geometry_for(game_id)).visual]
    lo, hi = min(map(_luminance, field)), max(map(_luminance, field))
    assert .10 < lo < hi < .25
    assert min(map(_luminance, white)) > 3*hi
    assert max(map(_luminance, black)) < lo/4
    assert _luminance(assets.GRID) < lo/4
    assert all(color[0] > color[1] > color[2] for color in field)
    assert all(finish[1] == 0 for finish in current.visual_finish.values())
    assert all(finish[0] >= .60 for finish in current.visual_finish.values())
    assert not any('ivory' in name or 'red' in name or 'blue' in name for name, _, _ in current.visual)


def test_hex_goal_edges_match_black_r_white_q_and_keep_wood_separation():
    g = geometry_for('hex7x7')
    current = assets.board(g)
    goals = current.data['goal_edges']
    assert {(row['color'], row['axis'], row['side']) for row in goals} == {
        ('black', 'r', 0), ('black', 'r', 6), ('white', 'q', 0), ('white', 'q', 6)}
    for color, axis in [('black', 1), ('white', 0)]:
        mesh = next(mesh for label, mesh, _ in current.visual if label == color+'_goal_edges')
        assert len(mesh.split(only_watertight=True)) == 2
        for band in mesh.split(only_watertight=True):
            center = band.bounds.mean(0)[:2]
            side = 0 if center[1 if axis else 0] < 0 else 6
            edge_cell = g.cell_local_xy(3, side) if axis else g.cell_local_xy(side, 3)
            direction = np.array([1., 0.]) if axis else np.array([.5, math.sqrt(3)/2])
            normal = np.array([-direction[1], direction[0]])
            if np.dot(normal, center) < 0:
                normal = -normal
            distance = np.min((band.vertices[:, :2]-edge_cell)@normal)
            assert distance-g.pitch_m/math.sqrt(3) > .0025
    for game_id in GAME_IDS[:2]:
        assert assets.board(geometry_for(game_id)).data['goal_edges'] == []


@pytest.fixture(scope='module', params=GAME_IDS)
def library(request, tmp_path_factory):
    root = tmp_path_factory.mktemp(request.param)/'test_v1'
    result = assets.build_assets(root, geometry_for(request.param))
    manifest = load_json(Path(result['manifest']))
    assert hashlib.sha256(Path(result['manifest']).read_bytes()).hexdigest() == result['manifest_sha256']
    return root, manifest, request.param


def test_all_generated_assets_load_with_verified_hashes_and_prior_physics(library):
    root, manifest, game_id = library
    assert set(manifest['assets']) == {'board', 'token_black', 'token_white', 'stand_left', 'stand_right', *geometry_for(game_id).reserve_stations()}
    for name in manifest['assets']:
        entry = load_task_asset(name, root, verify_hashes=True)
        assert all(part['classification'] == 'visual_only' for part in entry['visual_parts'])
        assert all(part['classification'] == 'physics' for part in entry['collision_parts'])
        assert all('mesh_path' not in part for part in entry['collision_parts'])
        assert all('roughness' in part and part['metallic'] >= 0 for part in entry['visual_parts'])
        if name.startswith('token_'):
            assert entry['mass_kg'] == .060
            assert entry['friction'] == .65
            np.testing.assert_allclose(entry['aabb_m'], [[-.028, -.028, -.007], [.028, .028, .047]], atol=1.e-12)
            assert len(entry['collision_parts']) == 2
        else:
            assert entry['mass_kg'] == 0
            assert entry['friction'] == .4
    assert manifest['geometry']['game_id'] == game_id
    assert manifest['appearance_version'] == assets.APPEARANCE
    assert manifest['license'] == 'CC0-1.0'
    assert len(manifest['recipe_sources_sha256']) == 6
    assert 'No external textures' in (root/'LICENSE.txt').read_text()
    assert 'Not established' in manifest['native_visual_acceptance']
    with pytest.raises(FileExistsError):
        assets.build_assets(root, geometry_for(game_id))


def test_building_new_styles_is_deterministic_and_preserves_old_palette():
    old_palette = dict(heritage.COLORS)
    old_version = heritage.APPEARANCE
    g = geometry_for('tic_tac_toe_3x3')
    for factory in (assets.board, lambda geometry: assets.token('white', geometry)):
        first, second = factory(g), factory(g)
        for (name, mesh, color), (other_name, other_mesh, other_color) in zip(first.visual, second.visual):
            assert name == other_name
            assert color == other_color
            np.testing.assert_array_equal(mesh.vertices, other_mesh.vertices)
            np.testing.assert_array_equal(mesh.faces, other_mesh.faces)
    assert heritage.COLORS == old_palette
    assert heritage.APPEARANCE == old_version


def _convex_footprint_gap(a, b):
    """Distance between convex display footprints; zero includes overlap."""
    edges = np.concatenate([np.roll(p, -1, axis=0)-p for p in (a, b)])
    normals = np.c_[-edges[:, 1], edges[:, 0]]
    normals /= np.linalg.norm(normals, axis=1)[:, None]
    ap, bp = a@normals.T, b@normals.T
    overlap = np.minimum(ap.max(0), bp.max(0))-np.maximum(ap.min(0), bp.min(0))
    if (overlap >= -1.e-12).all():
        return 0.
    distances = []
    for points, polygon in ((a, b), (b, a)):
        edge = np.roll(polygon, -1, axis=0)-polygon
        delta = points[:, None]-polygon
        t = np.clip(np.sum(delta*edge, axis=-1)/np.sum(edge*edge, axis=-1), 0., 1.)
        distances.append(np.linalg.norm(delta-t[:, :, None]*edge, axis=-1).min())
    return min(distances)


def test_hex_coordinate_strokes_clear_both_goal_bands_frame_and_board_edges():
    from bench.simulation.assets.inlay_geometry import convex_outline
    g = geometry_for('hex7x7')
    current = assets.board(g)
    obstacles = {name: [convex_outline(part.vertices[:, :2])
                        for part in mesh.split(only_watertight=True)]
                 for name, mesh, _ in current.visual
                 if name.endswith('goal_edges') or name == 'frame_inlay'}
    assert set(obstacles) == {'black_goal_edges', 'white_goal_edges', 'frame_inlay'}
    half = np.asarray(g.board_size_xy_m)/2
    for name, mesh, _ in current.visual:
        if not name.startswith('coordinate_'):
            continue
        # Include the actual stroke width and angled glyph segments. Checking
        # only text anchors previously missed goal bands cutting the glyphs.
        strokes = [convex_outline(points[:, :2]) for points in mesh.vertices.reshape(-1, 4, 3)]
        assert (mesh.bounds[0, :2] >= -half+.001).all(), name
        assert (mesh.bounds[1, :2] <= half-.001).all(), name
        for obstacle_name, polygons in obstacles.items():
            minimum_gap = min(_convex_footprint_gap(stroke, polygon)
                              for stroke in strokes for polygon in polygons)
            threshold = .001 if obstacle_name == 'frame_inlay' else .0015
            assert minimum_gap >= threshold, (name, obstacle_name, minimum_gap)
    labels = current.data['coordinate_labels']
    assert all(label['height_m'] == (.009 if label['axis'] == 'q' else .0105) for label in labels)
    assert current.data['appearance']['coordinate_layout'] == 'clear_goal_frame_v2'


@pytest.mark.parametrize('game_id', ['tic_tac_toe_3x3', 'connect_four_5x5'])
def test_square_coordinate_sizes_and_anchors_stay_unchanged(game_id):
    g = geometry_for(game_id)
    current = assets.board(g)
    for label in current.data['coordinate_labels']:
        assert label['height_m'] == .0105
        assert label['stroke_width_m'] == .00125
        index = label['index']
        expected = ([g.cell_local_xy(index, 0)[0], -g.board_size_xy_m[1]/2+.008]
                    if label['axis'] == 'q' else
                    [-g.board_size_xy_m[0]/2+.008, g.cell_local_xy(0, index)[1]])
        np.testing.assert_allclose(label['anchor_local_xy_m'], expected, atol=1.e-12)
