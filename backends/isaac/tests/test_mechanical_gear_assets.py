"""Actual L5 solid geometry, inertia and scene composition; no native claims."""
import math
import json
from copy import deepcopy
from itertools import product

import numpy as np
import pytest
from scipy.spatial import ConvexHull

from bench.geometry import quaternion_matrix
from bench.paths import DATA_ROOT, load_json
from bench.simulation.assets.scene_parts import write_library
from bench.simulation.isaacsim_backend.asset_catalog import load_task_asset
from scenes.workrooms.mechanical_assembly.task_scene import empty_task_scene
from scenes.workrooms.mechanical_assembly.tasks.gear_train_assembly import assets, task


@pytest.fixture(scope='module')
def solids():
    return dict(assets.make_assets())


@pytest.fixture(scope='module')
def library(tmp_path_factory, solids):
    root = tmp_path_factory.mktemp('gear-library')
    write_library(root, list(solids.items()), generator='CPU geometry regression',
                  extra={'source': {'schema': 'mechanical_gear_geometry_source.v1',
                                    'train_geometry': deepcopy(assets.TRAIN_GEOMETRY)}})
    return root


def projection_clearance(mesh):
    """Minimum XY radius of any point in the convex projected solid."""
    points = np.unique(np.asarray(mesh.vertices)[:, :2], axis=0)
    hull = ConvexHull(points)
    if np.all(hull.equations[:, 2] <= 0.):
        return 0.
    points = points[hull.vertices]
    following = np.roll(points, -1, axis=0)
    direction = following-points
    fraction = np.clip(-np.sum(points*direction, axis=1)/np.sum(direction*direction, axis=1), 0., 1.)
    return float(np.linalg.norm(points+fraction[:, None]*direction, axis=1).min())


def test_standard_gear_geometry_and_actual_full_cycle_mesh():
    report = assets.geometry_audit()
    assert report['passed']
    assert report['full_tooth_cycle_samples'] == 73
    assert report['maximum_intersecting_tooth_piece_pairs'] == 0
    assert report['tip_to_tip_negative_control_overlaps'] > 0
    assert report['maximum_sampled_involute_chord_error_m'] < 20.e-6
    assert report['nominal_pair_backlash_m'] == pytest.approx(.0008)
    assert report['nominal_radial_root_clearance_m'] == pytest.approx(.00125)
    assert report['minimum_radial_shaft_clearance_m'] == pytest.approx(.0003)
    assert 2*assets.GEOMETRY['pitch_radius_m'] == assets.GEOMETRY['module_m']*assets.GEOMETRY['teeth']


def test_nonconvex_tooth_must_not_be_silently_filled():
    with pytest.raises(ValueError, match='Nonconvex'):
        assets._prism([[0., 0.], [.02, 0.], [.005, .005], [.02, .02], [0., .02]], 0., .018)


def test_same_module_larger_idler_has_both_meshes_checked_and_wrong_phase_controls():
    report = assets.train_geometry_audit()
    assert report['passed']
    for row in report['pairs'].values():
        assert row['sample_count'] == 73
        assert row['center_distance_m'] == .13
        assert row['maximum_intersecting_tooth_piece_pairs'] == 0
        assert row['tip_to_tip_negative_control_overlaps'] > 0
        assert row['nominal_pair_tangential_backlash_m'] == pytest.approx(.0008)
        assert row['radial_tip_root_clearance_m'] == pytest.approx(.00125)
    assert max(report['maximum_sampled_involute_chord_error_m_by_teeth'].values()) < 20.e-6
    large = assets.gear_geometry(32)
    assert large['module_m'] == .005 and large['pitch_radius_m'] == .08
    assert large['outside_radius_m'] == .085
    assert large['bore_radius_m'] == assets.gear_geometry(20)['bore_radius_m']
    for value in (20., True, 31, 33):
        with pytest.raises(ValueError):
            assets.gear_geometry(value)


def test_actual_collision_pieces_closed_positive_convex_and_gpu_sized(solids):
    for name, asset in solids.items():
        for label, mesh, component in asset.collision:
            assert component['geometry'] == 'convex_mesh', (name, label)
            assert mesh.is_watertight and mesh.is_convex, (name, label)
            assert np.isfinite(mesh.volume) and mesh.volume > 1.e-15, (name, label)
            assert len(mesh.vertices) <= 64, (name, label)
            facets = mesh.facets
            polygons = len(facets)+len(mesh.faces)-sum(len(facet) for facet in facets)
            assert polygons <= 64, (name, label)
            assert all(len(np.unique(mesh.faces[facet])) <= 32 for facet in facets), (name, label)
    assert len(solids['input_gear'].collision) == 448


def test_tip_union_preserves_all_old_boundary_vertices_and_solid_area():
    original, merged = assets.tooth_polygons(split_tip=True), assets.tooth_polygons()
    for tooth in range(20):
        before = original[tooth*14+12:tooth*14+14]
        after = merged[tooth*13+12]
        hull = ConvexHull(np.vstack(before))
        assert ConvexHull(after).volume == pytest.approx(sum(ConvexHull(p).volume for p in before), abs=1.e-15)
        assert {tuple(p) for p in after} == {tuple(p) for p in np.vstack(before)[hull.vertices]}
        assert assets._prism(after, 0., .018).is_convex


@pytest.mark.parametrize('name', ['input_gear', 'idler_gear', 'output_gear'])
def test_every_gear_collision_piece_clears_real_24mm_bore(solids, name):
    clearances = [projection_clearance(mesh) for _, mesh, _ in solids[name].collision]
    assert min(clearances) == pytest.approx(.012, abs=1.e-10)
    assert all(value >= .012-1.e-10 for value in clearances)


def test_socket_keeps_real_through_hole_and_entry_cavity(solids):
    asset = solids['socket_crank']
    for label, mesh, _ in asset.collision:
        minimum = .029 if label.startswith('hex_socket_wall') else .013
        assert projection_clearance(mesh) >= minimum-1.e-10, label
    assert any(label.startswith('through_bore_roof') for label, _, _ in asset.collision)
    # The hole test would reject a whole-body convex approximation.
    vertices = np.vstack([mesh.vertices for _, mesh, _ in asset.collision])
    assert projection_clearance(assets._hull(vertices)) == 0.


def test_real_socket_full_yaw_clears_all_nonmating_assembled_meshes(solids):
    from scenes.workrooms.mechanical_assembly.tasks.gear_train_assembly.clearance import tool_sweep_audit
    report = tool_sweep_audit(solids)
    assert report['passed'] and report['yaw_interval_degrees'] == [0., 360.]
    assert report['groups']['middle_shaft']['lower_bound_m'] > .030
    assert report['groups']['idler_gear']['lower_bound_m'] > .0034
    assert report['fixed_reset_phase_neighbor_mesh_bounds']['idler_gear']['lower_bound_m'] > .004
    assert report['minimum_clearance_lower_bound_m'] >= .0013-1.e-12
    assert set(report['groups']) == {'input_shaft', 'middle_shaft', 'output_shaft',
                                    'workplate_support', 'idler_gear', 'output_gear'}


def test_neighbor_gears_may_also_rotate_through_any_phase(solids):
    from scenes.workrooms.mechanical_assembly.tasks.gear_train_assembly.clearance import yaw_sweep_pair_bound
    # Each gear's actual full-yaw sweep is contained in this square prism:
    # use the greatest radius of any real collision vertex and exact Z bounds.
    # This intentionally includes empty corners and the bore, so a positive
    # bound also proves clearance for every possible neighboring gear phase.
    for name, center in [('idler_gear', .13), ('output_gear', .26)]:
        points = np.vstack([m.vertices for _, m, _ in solids[name].collision])
        radius = np.linalg.norm(points[:, :2], axis=1).max()
        low, high = points[:, 2].min(), points[:, 2].max()
        enclosure = assets._box([2*radius, 2*radius, high-low], [0., center, .802+(high+low)/2])
        bounds = []
        for _, mesh, _ in solids['socket_crank'].collision:
            tool = mesh.copy(); tool.apply_translation([0., 0., .820])
            bounds.append(yaw_sweep_pair_bound(tool, enclosure))
        assert min(bounds) > .0034


def test_mesh_integrated_mass_and_non_diagonal_principal_inertia(solids):
    for name, asset in solids.items():
        if not asset.dynamic:
            continue
        info = asset.data
        values = np.array(info['inertia_diag_kg_m2'])
        axes = np.array(info['principal_axes_columns'])
        q = info['principal_axes_xyzw']
        rotation = quaternion_matrix([q[3], *q[:3]])
        assert np.allclose(axes.T@axes, np.eye(3), atol=1.e-12), name
        assert np.linalg.det(axes) == pytest.approx(1.)
        assert np.allclose(rotation, axes, atol=1.e-12)
        assert np.allclose(rotation@np.diag(values)@rotation.T,
                           info['inertia_body_frame_kg_m2'], atol=1.e-12), name
        assert np.all(values > 0.) and values[0]+values[1] >= values[2]-1.e-12
        assert asset.mass_kg == pytest.approx(sum(c['mass_kg'] for c in info['mass_components']))
    assert 1.3 < solids['input_gear'].mass_kg < 1.4
    assert 1.2 < solids['output_gear'].mass_kg < 1.3
    assert solids['idler_gear'].mass_kg == pytest.approx(2.956880398047562)
    assert solids['idler_gear'].data['gear_geometry']['teeth'] == 32
    crank = solids['socket_crank'].data['inertia_body_frame_kg_m2']
    assert abs(crank[0][2]) > 1.e-5  # The principal frame cannot be dropped.


def test_written_assets_preserve_inertia_and_verified_meshes(library, solids):
    for name, asset in solids.items():
        loaded = load_task_asset(name, library, verify_hashes=True)
        assert loaded['mass_kg'] == pytest.approx(asset.mass_kg)
        if asset.dynamic:
            assert loaded['principal_axes_xyzw'] == asset.data['principal_axes_xyzw']
            assert loaded['inertia_body_frame_kg_m2'] == asset.data['inertia_body_frame_kg_m2']


@pytest.mark.parametrize('name', ['input_gear', 'output_gear', 'socket_crank', 'open_end_wrench'])
def test_unmodified_bodies_keep_original_v2_mesh_bytes_and_physical_properties(library, name):
    old = load_task_asset(name, assets.LEGACY_ASSET_ROOT, verify_hashes=True)
    new = load_task_asset(name, library, verify_hashes=True)
    for key in ('mass_kg', 'center_of_mass_m', 'inertia_diag_kg_m2', 'principal_axes_xyzw',
                'inertia_body_frame_kg_m2', 'friction', 'aabb_m', 'size_m'):
        assert new[key] == old[key], key
    for kind in ('visual_parts', 'collision_parts'):
        assert len(new[kind]) == len(old[kind])
        for before, after in zip(old[kind], new[kind]):
            assert {k: v for k, v in after.items() if k != 'mesh_path'} == {
                k: v for k, v in before.items() if k != 'mesh_path'}


def test_task_preserves_shared_base_and_unassembled_reset(library):
    result = task.build_task_inputs(asset_root=library)
    scene, definition = result['scene'], result['task']
    base, initial, ready = empty_task_scene()
    for key in ('robot', 'workroom', 'isaacsim', 'physics', 'cameras'):
        assert scene[key] == base[key]
    assert result['initial_state'] == initial
    assert definition['ready_design'] == ready
    assert len(scene['objects']) == 6
    assert len(scene['fixtures']) == 2  # Functional workplate and shared robot stand only.
    assert all('rest' not in row['id'] for row in scene['fixtures'])
    assert not definition['physics_accepted'] and not definition['task_policy_success_claimed']
    for spec in scene['objects']:
        if spec['id'].endswith('_gear'):
            expected = {'input_gear': [.225, -.20], 'idler_gear': [.28, .34],
                        'output_gear': [.225, .10]}[spec['id']]
            assert spec['position_m'][:2] == expected
            assert spec['position_m'][2] == .7802
        if spec['asset_root'] == str(library):
            assert spec['friction'] == .3
            assert spec['contact_offset_m'] == .0001
            assert spec['rest_offset_m'] == 0.
            assert spec['collision_mesh_coordinate_scale'] == 1000.
    driver = next(row for row in scene['objects'] if row['id'] == 'straight_hex_screwdriver')
    assert driver['asset_root'] == str(task.THREADED_ROOT)
    assert not definition['public_rules']['tool_id_affects_score']
    for spec in scene['objects']:
        loaded = load_task_asset(spec['asset_id'], spec['asset_root'], verify_hashes=True)
        bounds = loaded.get('aabb_m', [loaded.get('bounds_min_m'), loaded.get('bounds_max_m')])
        corners = np.array(list(product(*np.array(bounds).T)))
        world = corners @ quaternion_matrix(spec['quaternion_wxyz']).T + spec['position_m']
        assert world[:, 2].min() == pytest.approx(.7802, abs=1.e-12)
    assert driver['position_m'][2] == pytest.approx(.8021999998807907)
    assert definition['evaluation']['shaft_centers_world_xy_m'] == [[.4, -.13], [.4, 0.], [.4, .13]]
    assert definition['evaluation']['input_equivalent_turn_weights'] == [1., 1.6, 1.]
    # AABB separation is a conservative proof that supplied bodies do not
    # overlap each other or the functional plate. The table is intentional.
    supplied = []
    for spec in [*scene['objects'], next(s for s in scene['fixtures'] if s['id'] == 'gear_workplate')]:
        loaded = load_task_asset(spec['asset_id'], spec['asset_root'], verify_hashes=True)
        bounds = loaded.get('aabb_m', [loaded.get('bounds_min_m'), loaded.get('bounds_max_m')])
        corners = np.array(list(product(*np.array(bounds).T)))
        world = corners @ quaternion_matrix(spec['quaternion_wxyz']).T + spec['position_m']
        supplied.append((spec['id'], world.min(axis=0), world.max(axis=0)))
    for i, (a, low, high) in enumerate(supplied):
        for b, other_low, other_high in supplied[i+1:]:
            assert np.any((high < other_low) | (other_high < low)), (a, b)


def test_reset_audit_is_published_and_rejects_table_or_object_overlap():
    result = task.build_task_inputs()
    audit = result['scene']['mechanical_task_provenance']['reset_audit']
    assert audit['passed']
    assert all(row['exact_initial_gap'] for row in audit['objects'].values())
    mutated = json.loads(json.dumps(result['scene']))
    mutated['objects'][1]['position_m'][0] = mutated['objects'][0]['position_m'][0]
    mutated['objects'][1]['position_m'][1] = mutated['objects'][0]['position_m'][1]
    failed = task.reset_audit(mutated)
    assert not failed['passed'] and failed['object_aabb_overlaps']


def test_supply_revision_rejects_historical_overhang_and_near_edge_reset():
    from scenes.workrooms.mechanical_assembly.tasks.gear_train_assembly.definition import TASK

    current = task.build_task_inputs()
    audit = current['scene']['mechanical_task_provenance']['reset_audit']
    assert audit['required_table_edge_clearance_m'] == .02
    assert audit['required_fixture_clearance_m'] == .02
    assert all(row['table_edge_clearance'] and row['fixture_clearance']
               for row in audit['objects'].values())
    assert audit['objects']['idler_gear']['minimum_table_edge_clearance_m'] > .037
    assert current['task']['supply_layout']['revision'] == 'table_inset_v1'
    assert TASK['design']['candidate_layout']['supply_centers_xy_m'] == current['task']['supply_layout']['gear_supply_positions_xy_m']
    assert assets.TRAIN_GEOMETRY['gear_supply_positions_xy_m'] == [[.21, -.20], [.25, .32], [.21, .10]]
    legacy = task.build_task_inputs(supply_layout_revision='legacy_v3_4')
    assert legacy['scene']['mechanical_task_provenance']['reset_audit']['passed']
    assert not task.reset_audit(legacy['scene'])['passed']
    idler = next(row for row in legacy['scene']['objects'] if row['id'] == 'idler_gear')
    idler['position_m'][:2] = [.16, 0.]
    failed = task.reset_audit(legacy['scene'])
    assert 'idler_gear' in failed['failures']
    assert not failed['objects']['idler_gear']['table_bounds']
    assert failed['objects']['idler_gear']['minimum_table_edge_clearance_m'] < -.08


def test_supply_revision_rejects_workplate_proximity_without_object_overlap():
    scene = task.build_task_inputs()['scene']
    idler = next(row for row in scene['objects'] if row['id'] == 'idler_gear')
    idler['position_m'][:2] = [.4, .31]
    failed = task.reset_audit(scene)
    assert failed['object_aabb_overlaps'] == []
    assert failed['objects']['idler_gear']['table_edge_clearance']
    assert not failed['objects']['idler_gear']['fixture_clearance']
    assert 'idler_gear' in failed['failures']


def test_old_asset_library_does_not_silently_use_new_layout(tmp_path):
    (tmp_path/'manifest.json').write_text(json.dumps({'source': {'schema': 'mechanical_gear_geometry_source.v1'}}))
    with pytest.raises(ValueError, match='20/32/20'):
        task.build_task_inputs(asset_root=tmp_path)


def test_table_placement_rejects_missing_or_nonfinite_bounds():
    for bounds in ({}, {'aabb_m': [[0., 0., float('nan')], [1., 1., 1.]]},
                   {'aabb_m': [[0., 0., 1.], [1., 1., 0.]]}):
        with pytest.raises(ValueError):
            task.tabletop_position(bounds, [0., 0.], [1., 0., 0., 0.])


def test_contact_probe_is_separate_passive_and_socket_clearance_is_explicit():
    probe = task.contact_probe_reset()
    assert probe['loaded_body'] == 'socket_crank'
    assert probe['all_gears_are_passive']
    assert not probe['native_physics_validated']
    assert probe['expected_signed_angle_ratios'] == {'idler_per_input': -.625, 'output_per_input': 1.}
    angle = math.radians(probe['nominal_socket_hub_half_free_angle_degrees'])
    hub_vertex_radius = .027/math.cos(math.pi/6)
    assert hub_vertex_radius*math.cos(math.pi/6-angle) == pytest.approx(.029)
    assert 8. < math.degrees(angle) < 9.
    poses = probe['object_overrides']
    assert poses['input_gear']['position_m'][2]+.040 < poses['socket_crank']['position_m'][2]+.025
    assert poses['socket_crank']['position_m'][2]+.025 < .78+.078
    assert .013 > assets.GEOMETRY['shaft_radius_m']


def test_asset_preparation_refuses_existing_version(tmp_path):
    with pytest.raises(FileExistsError):
        assets.prepare_assets(tmp_path)


def test_conditioned_screwdriver_binding_preserves_original_physical_properties():
    original_root = f'{DATA_ROOT}/assets/task_parts/mechanical_workshop/threaded_v2'
    old = load_task_asset(task.ASSET_IDS['screwdriver'], original_root, verify_hashes=True)
    new = load_task_asset(task.ASSET_IDS['screwdriver'], task.THREADED_ROOT, verify_hashes=True)
    for key in ('mass_kg', 'center_of_mass_m', 'inertia_diag_kg_m2', 'dimensions_m', 'source_scale'):
        assert new[key] == old[key], key
    original_manifest = load_json(task.Path(original_root)/'manifest.json')
    candidate_manifest = load_json(task.THREADED_ROOT/'manifest.json')
    assert candidate_manifest['per_role_spec_options']['screwdriver'] == original_manifest['per_role_spec_options']['screwdriver']
    inputs = task.build_task_inputs()
    assert inputs['scene']['mechanical_task_provenance']['screwdriver_manifest_sha256'] == task.THREADED_MANIFEST_SHA256
    assert not inputs['task']['physics_accepted']
