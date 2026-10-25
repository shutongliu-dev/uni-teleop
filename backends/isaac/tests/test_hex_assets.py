"""Independent metric checks for the task assets, not native contact acceptance."""
from dataclasses import replace
import hashlib
import math
from pathlib import Path

import numpy as np
import pytest
import trimesh

from bench.paths import load_json
from scenes.board_games.hex import HexGeometry, HexRules
from scenes.workrooms.card_room.tasks.hex7x7.assets import (
    board, build_assets, stand, token, token_mass_properties,
)


@pytest.mark.parametrize('geometry', [
    HexGeometry(),
    replace(HexGeometry(), handle_radius_m=.010, handle_height_m=.060, piece_mass_kg=.035),
    replace(HexGeometry(), piece_base_height_m=.020, handle_height_m=.025, piece_mass_kg=.120),
])
def test_analytic_mass_properties_match_closed_collision_volume_integration(geometry):
    """Integrate the actual collider records, independently of mass authoring.

    Cosmetic roundovers must not silently change the analytic physical inertia.
    The independent triangle approximation here is made from the collision
    shapes, rather than treating appearance meshes as collision geometry.
    """
    asset = token('red', geometry)
    cylinders = []
    for _, _, part in asset.collision:
        mesh = trimesh.creation.cylinder(radius=part['radius_m'], height=2*part['half_height_m'], sections=256)
        mesh.apply_translation(part['position_m'])
        cylinders.append(mesh)
    body = trimesh.util.concatenate(cylinders)
    assert body.is_watertight
    body.density = geometry.piece_mass_kg / body.volume
    analytic = token_mass_properties(geometry)
    np.testing.assert_allclose(analytic['center_of_mass_m'], body.center_mass, atol=1.e-12)
    # Triangle integration must remain close to the two true cylinders, not a
    # bounding-box tensor or a tensor calculated from the rounded appearance.
    np.testing.assert_allclose(np.diag(analytic['inertia_diag_kg_m2']),
                               body.moment_inertia, rtol=1.e-3, atol=1.e-15)
    assert analytic['center_of_mass_m'][2] > 0
    inertia = np.asarray(analytic['inertia_diag_kg_m2'])
    assert (inertia > 0).all()
    assert all(inertia[i] < sum(inertia) - inertia[i] for i in range(3))


@pytest.mark.parametrize('color', ['red', 'blue'])
def test_token_roundovers_preserve_envelope_and_straight_gripping_surface(color):
    g = HexGeometry(); asset = token(color, g)
    collisions = {label: part for label, _, part in asset.collision}
    for index, (label, mesh, _) in enumerate(asset.visual):
        collider = collisions[label]
        center = np.asarray(collider['position_m'])
        points = mesh.vertices-center
        radial = np.linalg.norm(points[:, :2], axis=1)
        assert np.isfinite(points).all()
        assert mesh.is_watertight and mesh.is_winding_consistent
        assert radial.max() == pytest.approx(collider['radius_m'], abs=1.e-12)
        assert radial.max() <= collider['radius_m']+1.e-12
        np.testing.assert_allclose([points[:, 2].min(), points[:, 2].max()],
                                   [-collider['half_height_m'], collider['half_height_m']], atol=1.e-12)
        straight = points[np.isclose(radial, collider['radius_m'], atol=1.e-12), 2]
        # The handle keeps its 40 mm diameter over at least 36 mm of height;
        # appearance cannot create a hidden narrower handle or a grasp fixture.
        if label == 'handle':
            assert np.ptp(straight) >= .036
        assert mesh.volume > .98*math.pi*collider['radius_m']**2*(2*collider['half_height_m'])
        roughness, metallic = asset.visual_finish[index]
        assert .18 <= roughness <= .30
        assert metallic == 0.


def test_mass_and_length_scaling_preserve_inertia_dimensions():
    g = HexGeometry(); baseline = token_mass_properties(g)
    heavier = token_mass_properties(replace(g, piece_mass_kg=2 * g.piece_mass_kg))
    np.testing.assert_allclose(heavier['center_of_mass_m'], baseline['center_of_mass_m'])
    np.testing.assert_allclose(heavier['inertia_diag_kg_m2'],
                               2 * np.array(baseline['inertia_diag_kg_m2']))
    scale = .8
    scaled = token_mass_properties(replace(g, piece_radius_m=g.piece_radius_m * scale,
        piece_base_height_m=g.piece_base_height_m * scale, handle_radius_m=g.handle_radius_m * scale,
        handle_height_m=g.handle_height_m * scale))
    np.testing.assert_allclose(scaled['center_of_mass_m'], scale * np.array(baseline['center_of_mass_m']))
    np.testing.assert_allclose(scaled['inertia_diag_kg_m2'],
                               scale**2 * np.array(baseline['inertia_diag_kg_m2']))


def test_board_encloses_all_49_complete_hexagons_and_fits_table():
    g = HexGeometry(); asset = board(g)
    vertices = np.array([point for q in range(7) for r in range(7)
                         for point in g.cell_vertices(q, r)])
    half = np.array(g.board_size_xy_m) / 2
    center = np.array(g.center_xy_m)
    # Derive the occupied grid envelope from the actual six vertices per cell.
    np.testing.assert_allclose(vertices.min(0) - (center - half), g.board_margin_m)
    np.testing.assert_allclose((center + half) - vertices.max(0), g.board_margin_m)
    assert ((center - half) > np.array([.15, -.5])).all()
    assert ((center + half) < np.array([1.15, .5])).all()
    assert g.board_size_xy_m == pytest.approx((.79, .5163139720814413))
    for label, mesh, _ in asset.visual:
        assert (mesh.bounds[0, :2] >= -half - 1.e-9).all(), label
        assert (mesh.bounds[1, :2] <= half + 1.e-9).all(), label
    assert len(asset.collision) == 1
    component = asset.collision[0][2]
    assert component['geometry'] == 'box'
    np.testing.assert_allclose(component['half_extents_m'], [*half, g.board_thickness_m / 2])
    for label, mesh, _ in asset.visual:
        assert mesh.bounds[0, 2] >= -1.e-12, label
        assert mesh.bounds[1, 2] <= g.board_thickness_m+.00013+1.e-12, label


def test_fine_inlays_still_trace_every_hex_edge_exactly_once():
    g = HexGeometry(); asset = board(g)
    mesh = next(mesh for name, mesh, _ in asset.visual if name == 'hex_grid')
    actual = mesh.split(only_watertight=True)
    expected = set()
    for q in range(7):
        for r in range(7):
            vertices = [np.array(p)-g.center_xy_m for p in g.cell_vertices(q, r)]
            for a, b in zip(vertices, vertices[1:]+vertices[:1]):
                expected.add(tuple(np.round((a+b)/2, 8)))
    assert len(actual) == len(expected)
    midpoints = np.array([part.vertices[:, :2].mean(0) for part in actual])
    error = np.linalg.norm(midpoints[:, None, :]-np.array(sorted(expected))[None, :, :], axis=-1)
    assert error.min(axis=0).max() < 6.e-9
    assert error.min(axis=1).max() < 6.e-9
    for part in actual:
        xy = part.vertices[:, :2]
        values = np.linalg.eigvalsh((xy-xy.mean(0)).T @ (xy-xy.mean(0)))
        # An oriented rectangle has eigenvalues N * half_length**2.
        width, length = 2*np.sqrt(values/len(xy))
        # At the task camera's roughly 620 pixels/metre board scale, the
        # initial 0.8 mm marks were subpixel. Retain at least 1.4 mm here.
        assert .0014 <= width <= .0016
        assert length == pytest.approx(g.pitch_m/math.sqrt(3), abs=2.e-9)
    grid_index = next(i for i, part in enumerate(asset.visual) if part[0] == 'hex_grid')
    assert max(asset.visual[grid_index][2]) <= .06
    assert asset.visual_finish[grid_index] == (.88, 0.)


def test_ivory_playing_field_contains_cells_without_covering_timber_corner_inlays():
    g = HexGeometry(); asset = board(g)
    parts = {name: mesh for name, mesh, _ in asset.visual}
    field = parts['ivory_playing_field']
    # Read the actual authored footprint from the top triangles' vertices.
    xy = field.vertices[:, :2]
    from scipy.spatial import ConvexHull
    hull = ConvexHull(xy)
    cells = np.array([np.array(p)-g.center_xy_m for q in range(7) for r in range(7)
                      for p in g.cell_vertices(q, r)])
    distance = cells @ hull.equations[:, :2].T + hull.equations[:, 2]
    assert distance.max() < -.0065
    frets = parts['corner_frets'].vertices[:, :2]
    assert ((frets @ hull.equations[:, :2].T + hull.equations[:, 2]).max(axis=1) > .02).all()
    grain = parts['walnut_grain'].vertices[:, :2]
    # Veneer grain remains on the wood, even outside the full light inset and
    # the nearby coordinate strip. It cannot be mistaken for another grid.
    assert ((grain @ hull.equations[:, :2].T + hull.equations[:, 2]).max(axis=1) > .013).all()


def test_visual_coordinate_inlays_follow_axes_and_clear_all_cell_footprints():
    from scipy.spatial import ConvexHull
    g = HexGeometry(); asset = board(g)
    labels = asset.data['coordinate_labels']
    assert len(labels) == 14
    assert {(x['text'], x['axis'], x['index']) for x in labels} == {
        *((chr(ord('A')+i), 'q', i) for i in range(7)),
        *((str(i+1), 'r', i) for i in range(7)),
    }
    hull = ConvexHull(np.array([np.array(p)-g.center_xy_m for q in range(7) for r in range(7)
                               for p in g.cell_vertices(q, r)]))
    parts = {name: mesh for name, mesh, _ in asset.visual}
    for label in labels:
        points = parts['coordinate_'+label['text']].vertices[:, :2]
        signed = points @ hull.equations[:, :2].T + hull.equations[:, 2]
        # Every letter stroke lies outside even the convex envelope of all 49
        # hexagons, with more than five millimetres of physical-footprint gap.
        assert (signed.max(axis=1) > .005).all(), label['text']
        assert np.ptp(points[:, 1]) >= .0105
        assert label['stroke_width_m'] >= .0012
        if label['axis'] == 'q':
            # A letter such as F has more strokes on its left; alignment is
            # determined by its writing envelope, not the vertex centroid.
            center_x = (points[:, 0].min()+points[:, 0].max())/2
            assert np.abs(center_x-g.cell_local_xy(label['index'], 0)[0]) < .001
        else:
            local = points.mean(0)-g.cell_local_xy(0, label['index'])
            assert np.dot(local, [math.sqrt(3)/2, -.5]) < -.05


def test_bottom_frame_leaves_clear_gaps_around_enlarged_coordinate_letters():
    g = HexGeometry(); asset = board(g)
    frame = next(mesh for name, mesh, _ in asset.visual if name == 'frame_pinstripe')
    bottom_points = frame.vertices[frame.vertices[:, 1] < -g.board_size_xy_m[1]/2+.007]
    for q in range(7):
        x = g.cell_local_xy(q, 0)[0]
        assert np.abs(bottom_points[:, 0]-x).min() >= .0064


@pytest.mark.parametrize('color,axis,direction', [('red', 'r', [1., 0.]),
                                                ('blue', 'q', [.5, math.sqrt(3) / 2])])
def test_visual_goal_edge_bands_agree_with_adjudicator(color, axis, direction):
    """Read authored mesh directions instead of repeating authoring axis logic."""
    assert HexRules.specification()['goals'][color] == f'connect {axis}=0 to {axis}=6'
    asset = board(HexGeometry())
    mesh = next(mesh for name, mesh, _ in asset.visual if name == color + '_goal_edges')
    strips = mesh.split(only_watertight=True)
    assert len(strips) == 2
    signs = []
    direction = np.asarray(direction)
    normal = np.array([-direction[1], direction[0]])
    for strip in strips:
        xy = strip.vertices[:, :2]
        centered = xy - xy.mean(0)
        _, vectors = np.linalg.eigh(centered.T @ centered)
        along = vectors[:, -1]
        assert abs(np.dot(along, direction)) > 1 - 1.e-10, color
        signs.append(np.sign(np.dot(xy.mean(0), normal)))
    assert sorted(signs) == [-1., 1.]


def _cylinder_box_gap(center, radius, z_low, z_high, box_center, half):
    """Return XY and Z gaps between an upright cylinder and axis-aligned box."""
    box_center = np.asarray(box_center); half = np.asarray(half)
    horizontal = np.linalg.norm(np.maximum(np.abs(center[:2] - box_center[:2]) - half[:2], 0.)) - radius
    vertical = max(box_center[2] - half[2] - z_high, z_low - box_center[2] - half[2])
    return horizontal, vertical


def test_all_fifty_reserves_have_table_support_board_clearance_and_no_pair_overlap():
    g = HexGeometry()
    positions = np.array([g.reserve_position(color, i) for color in ('red', 'blue') for i in range(25)])
    assert positions.shape == (50, 3)
    assert len(np.unique(positions, axis=0)) == 50
    assert (positions[:, 0] - g.piece_radius_m >= .15).all()
    assert (positions[:, 0] + g.piece_radius_m <= 1.15).all()
    assert (positions[:, 1] - g.piece_radius_m >= -.5).all()
    assert (positions[:, 1] + g.piece_radius_m <= .5).all()
    np.testing.assert_allclose(positions[:, 2] - g.piece_base_height_m / 2, g.table_top_z_m)
    distances = np.linalg.norm(positions[:, None, :2] - positions[None, :, :2], axis=-1)
    distances[np.diag_indices(50)] = np.inf
    assert distances.min() - 2 * g.piece_radius_m >= .010
    board_center = [*g.center_xy_m, g.table_top_z_m + g.board_thickness_m / 2]
    board_half = [*(np.array(g.board_size_xy_m) / 2), g.board_thickness_m / 2]
    for center in positions:
        xy_gap, _ = _cylinder_box_gap(center, g.piece_radius_m,
            center[2] - g.piece_base_height_m / 2, center[2] + g.piece_base_height_m / 2,
            board_center, board_half)
        assert xy_gap >= .010


def test_reserves_clear_both_actual_cantilevers_and_mount_plates():
    g = HexGeometry()
    for sign in (-1, 1):
        fixture = stand(sign)
        parts = {name: part for name, _, part in fixture.collision}
        assert set(parts) == {'foot', 'column', 'cantilever', 'mount_plate'}
        plate = parts['mount_plate']
        assert plate['position_m'][2] + plate['half_extents_m'][2] == pytest.approx(.95)
        # The floor-reaching column remains outside the one-metre tabletop.
        column = parts['column']
        assert abs(column['position_m'][1]) - column['half_extents_m'][1] > .5
        for color in ('red', 'blue'):
            for index in range(25):
                center = np.array(g.reserve_position(color, index))
                cylinders = [
                    (g.piece_radius_m, center[2] - g.piece_base_height_m / 2,
                     center[2] + g.piece_base_height_m / 2),
                    (g.handle_radius_m, center[2] + g.piece_base_height_m / 2,
                     center[2] + g.piece_base_height_m / 2 + g.handle_height_m),
                ]
                for radius, bottom, top in cylinders:
                    for name, part in parts.items():
                        gaps = _cylinder_box_gap(center, radius, bottom, top,
                                                 part['position_m'], part['half_extents_m'])
                        assert max(gaps) > .005, (sign, color, index, name, gaps)


@pytest.fixture(scope='module')
def library(tmp_path_factory):
    root = tmp_path_factory.mktemp('hex_assets') / 'test_v3'
    result = build_assets(root, HexGeometry())
    assert hashlib.sha256(Path(result['manifest']).read_bytes()).hexdigest() == result['manifest_sha256']
    return root, load_json(root / 'manifest.json')


def test_generated_token_manifest_keeps_exact_analytic_shapes_and_mass(library):
    root, manifest = library; g = HexGeometry()
    assert set(manifest['assets']) == {'hex_board', 'hex_token_red', 'hex_token_blue',
                                       'hex_stand_red', 'hex_stand_blue'}
    for color in ('red', 'blue'):
        entry = manifest['assets']['hex_token_' + color]
        assert entry['dynamic'] is True
        assert entry['mass_kg'] == .060
        assert entry['center_of_mass_m'] == pytest.approx([0, 0, .01601423487544484])
        assert entry['inertia_diag_kg_m2'] == pytest.approx([
            2.4043048340319903e-5, 2.4043048340319903e-5, 1.6687259786476868e-5])
        parts = entry['collision_parts']
        assert len(parts) == 2
        assert [part['geometry'] for part in parts] == ['cylinder', 'cylinder']
        assert all('mesh_path' not in part for part in parts)
        assert parts[0]['radius_m'] == g.piece_radius_m
        assert parts[1]['radius_m'] == g.handle_radius_m
        base_top = parts[0]['position_m'][2] + parts[0]['half_height_m']
        handle_bottom = parts[1]['position_m'][2] - parts[1]['half_height_m']
        assert base_top == pytest.approx(handle_bottom, abs=1.e-12)
        for part in entry['visual_parts']:
            filename = root / part['mesh_path']
            assert digest_file(filename) == part['sha256']
            assert part['classification'] == 'visual_only'
        np.testing.assert_allclose(entry['aabb_m'], [[-.028, -.028, -.007], [.028, .028, .047]])


def digest_file(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_generated_static_fixtures_keep_only_solid_box_collisions(library):
    _, manifest = library
    for name, count in [('hex_board', 1), ('hex_stand_red', 4), ('hex_stand_blue', 4)]:
        entry = manifest['assets'][name]
        assert entry['dynamic'] is False
        assert entry['mass_kg'] == 0
        assert len(entry['collision_parts']) == count
        for part in entry['collision_parts']:
            assert part['geometry'] == 'box'
            assert min(part['half_extents_m']) > 0
            assert part['classification'] == 'physics'
            assert 'mesh_path' not in part


@pytest.mark.parametrize('sign', [-1, 1])
def test_pedestal_collisions_are_exactly_the_original_four_boxes(sign):
    fixture = stand(sign)
    expected = {
        'foot': ([.65, sign*.74, .025], [.20, .19, .025]),
        'column': ([.65, sign*.74, .455], [.095, .095, .455]),
        'cantilever': ([.65, sign*.625, .915], [.11, .205, .015]),
        'mount_plate': ([.65, sign*.52, .9375], [.11, .10, .0125]),
    }
    assert len(fixture.collision) == len(expected)
    for label, mesh, part in fixture.collision:
        assert mesh is None
        assert part['geometry'] == 'box'
        np.testing.assert_allclose(part['position_m'], expected[label][0], atol=1.e-15)
        np.testing.assert_allclose(part['half_extents_m'], expected[label][1], atol=1.e-15)


def test_versioned_appearance_records_original_source_and_never_overwrites(library):
    root, manifest = library
    assert manifest['appearance_version'] == 'hex_heritage_v3'
    assert manifest['license'] == 'CC0-1.0'
    assert manifest['generator'] == 'scenes/workrooms/card_room/tasks/hex7x7/assets.py'
    assert len(manifest['geometry_helper_sha256']) == 64
    assert 'No external textures' in (root/'LICENSE.txt').read_text()
    with pytest.raises(FileExistsError):
        build_assets(root, HexGeometry())
