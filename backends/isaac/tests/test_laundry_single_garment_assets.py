"""Mesh invariants that distinguish a real unfolded shell from the old stack."""
import json
import numpy as np
import pytest

from bench.paths import load_json
from scenes.workrooms.laundry_room.garment_folding_assets import (
    candidate_metadata, geometry_report, validate_metadata, export_native_settled_reference, _rotate_xy,
    _shell_layer_pairs, FLAT_SHIRT_PROFILE, _same_native_source)
from bench.isaacsim.release import digest


@pytest.mark.parametrize('task,openings', [('shirt_folding', 4), ('trousers_folding', 3)])
def test_unfolded_sewn_surface_keeps_real_openings_and_rest_metric(task, openings):
    metadata = candidate_metadata(task)
    assert validate_metadata(metadata) == metadata
    assert metadata['initial_state_kind'] == 'unfolded_shell'
    assert metadata['goal_reference_status'] == 'provisional_projection_not_qualified'
    assert len(metadata['cloths']) == 1
    row = metadata['cloths'][0]
    rest, initial = map(np.asarray, (row['rest_vertices_m'], row['initial_vertices_m']))
    assert np.allclose(initial-rest, (initial-rest)[0])
    assert np.ptp(initial[:, 2]) > .005
    assert initial[:, 2].min() == pytest.approx(.93)
    assert initial[:, 0].min() > -.10 and initial[:, 0].max() < 1.40
    assert initial[:, 1].min() > -.94 and initial[:, 1].max() < -.16
    geometry = geometry_report(metadata)
    assert geometry['boundary_loops'] == openings
    assert geometry['nonmanifold_edges'] == 0
    assert geometry['min_edge_m'] >= .0014
    assert geometry['min_triangle_area_m2'] > 1.e-6
    assert geometry['self_intersections_checked'] is False
    assert row['fixed_vertices'] == []
    assert not np.allclose(initial, row['folded_reference_vertices_m'])
    flat, pairs = np.asarray(row['unfolded_reference_vertices_m']), np.asarray(row['folded_layer_pairs'])
    assert len(pairs) > 100
    assert np.min(np.linalg.norm(flat[pairs[:, 0], :2]-flat[pairs[:, 1], :2], axis=1)) >= .060


def test_official_shirt_topology_and_unit_conversion_are_preserved():
    m = candidate_metadata('shirt_folding')
    row = m['cloths'][0]
    assert len(row['rest_vertices_m']) == 6436
    assert len(row['triangles']) == 12736
    p = m['design']['representation']
    assert p['topology_changed'] is False and p['vertices_projected'] is False
    assert p['applied_uniform_scale'] == .01


def test_alternative_flat_shirt_is_a_sewn_shell_with_four_openings_and_no_slivers():
    from bench.evaluation.cloth.audit_cloth_self_intersection import self_intersections
    m = candidate_metadata('shirt_folding', mesh_profile=FLAT_SHIRT_PROFILE,
                            source_yaw_quarter_turns=1, source_center_xy_m=[.75, -.55])
    geometry = geometry_report(m)
    assert geometry['boundary_loops'] == 4 and geometry['nonmanifold_edges'] == 0
    assert geometry['min_triangle_area_m2'] > 1.e-6
    row = m['cloths'][0]
    q, triangles = np.asarray(row['initial_vertices_m']), np.asarray(row['triangles'])
    assert np.ptp(q[:, 2]) == pytest.approx(.003)
    assert np.allclose(np.ptp(q[:, :2], axis=0), [.61, .70])
    hits, _ = self_intersections(q, triangles)
    assert hits == []
    assert len(row['shell_layer_pairs']) > 1000
    assert m['design']['representation']['derived_from_official_worn_mesh'] is False
    assert m['design']['representation']['license'] == 'CC0-1.0'
    assert m['goal_reference_status'] == 'provisional_projection_not_qualified'
    assert validate_metadata(m) == m


def test_native_source_allows_only_tiny_derived_metric_noise_not_provenance_changes():
    from copy import deepcopy
    pinned = {'hash': 'unchanged', 'operation_credit': False,
              'settle_metrics': {'speed_p95_m_s': .02261967957019806}}
    observed = deepcopy(pinned)
    observed['settle_metrics']['speed_p95_m_s'] = .02261967584490776
    assert _same_native_source(observed, pinned)
    observed['operation_credit'] = True
    assert not _same_native_source(observed, pinned)
    observed = deepcopy(pinned)
    observed['settle_metrics']['speed_p95_m_s'] += 1.e-6
    assert not _same_native_source(observed, pinned)


@pytest.mark.parametrize('value', [0., .0001, .003, float('nan'), float('inf')])
def test_implausible_material_thickness_cannot_sneak_into_new_asset(value):
    with pytest.raises(ValueError):
        candidate_metadata('shirt_folding', physical_thickness_m=value)


def test_material_thickness_override_is_explicit_and_separate_from_shell_depth():
    m = candidate_metadata('trousers_folding', physical_thickness_m=.0008)
    assert m['physical_thickness_m'] == m['physics_options']['cloth_thickness_m'] == .0008
    assert m['design']['representation']['initial_panel_midplane_gap_m'] == .006
    assert validate_metadata(m) == m


@pytest.mark.parametrize('task', ['shirt_folding', 'trousers_folding'])
def test_natural_shell_clearance_is_distinct_from_fold_created_layers(task):
    m = candidate_metadata(task)
    row = m['cloths'][0]
    q = np.asarray(row['folded_reference_vertices_m'])
    pairs = np.asarray(row['shell_layer_pairs'])
    assert len(pairs) > 100
    gap = q[pairs[:, 1], 2]-q[pairs[:, 0], 2]
    assert np.all(gap > 0.)
    # Sampling cannot discard a compressed reference wall merely because its
    # gap would fail. Vertical fold/seam walls are excluded by normals instead.
    compressed = q.copy()
    compressed[:, 2] = .9+.1*(compressed[:, 2]-.9)
    retained = np.asarray(_shell_layer_pairs(task, np.asarray(row['rest_vertices_m']),
        np.asarray(row['triangles']), np.asarray(row['unfolded_reference_vertices_m']),
        compressed, m['physical_thickness_m']))
    assert len(retained) > 100
    retained_gap = compressed[retained[:, 1], 2]-compressed[retained[:, 0], 2]
    assert np.mean(retained_gap >= .45*m['physical_thickness_m']) < .85
    # Global compression can retain a reasonable outline; it must still lose
    # natural front/back layer clearance independently of semantic fold pairs.
    assert np.mean(gap*.1 >= .45*m['physical_thickness_m']) < .85
    if task == 'trousers_folding':
        half = len(q)//2
        assert np.all(abs(pairs[:, 1]-pairs[:, 0]) == half)


def test_reference_flattens_wrinkles_without_scaling_away_material_wall_gap(tmp_path):
    m = candidate_metadata('trousers_folding')
    row = m['cloths'][0]
    q = np.asarray(row['unfolded_reference_vertices_m'])
    # A paired-shell CPU fixture: the same broad wrinkle on both panels must
    # not turn their 1.2 mm material separation into a fraction of a millimetre.
    q[:, 2] += .07*np.sin(np.pi*(q[:, 1]-q[:, 1].min())/np.ptp(q[:, 1]))**2
    path = tmp_path/'synthetic_reference.json'
    path.write_text(json.dumps({'schema': 'laundry_unfolded_native_reference.v1',
        'task_id': 'trousers_folding', 'vertices_m': q.tolist(),
        'native_source': {'scope': 'Synthetic unit fixture only, not native success evidence'}}))
    target = candidate_metadata('trousers_folding', unfolded_reference_path=path)['cloths'][0]
    folded = np.asarray(target['folded_reference_vertices_m'])
    pairs = np.asarray(target['shell_layer_pairs'])
    gap = folded[pairs[:, 1], 2]-folded[pairs[:, 0], 2]
    assert len(pairs) > .80*len(q)/2
    assert np.mean(gap >= .45*row['physical_thickness_m']) > .95


def test_opposite_leg_fold_is_a_separate_complete_reference_at_same_target():
    m = candidate_metadata('trousers_folding')
    row = m['cloths'][0]
    assert len(row['folded_reference_variants']) == 1
    variant = row['folded_reference_variants'][0]
    assert set(variant) == {'id', 'folded_reference_vertices_m', 'folded_layer_pairs',
                            'shell_layer_pairs', 'support_bottom_vertex_indices'}
    reference, other = np.asarray(row['folded_reference_vertices_m']), np.asarray(variant['folded_reference_vertices_m'])
    assert np.allclose(reference[:, :2].min(0)+reference[:, :2].max(0),
                       other[:, :2].min(0)+other[:, :2].max(0))
    assert np.sqrt(np.mean(np.sum((reference-other)**2, axis=1))) > .06
    assert variant['folded_layer_pairs'] and variant['shell_layer_pairs']
    assert variant['support_bottom_vertex_indices']


def test_explicit_solver_candidates_are_frozen_without_changing_rest_geometry():
    original = candidate_metadata('trousers_folding')
    tuned = candidate_metadata('trousers_folding', physics_dt_s=.002,
                               solver_iterations=32, cloth_surface_bend_stiffness=0.)
    assert tuned['physics_options']['cloth_iterations'] == 32
    assert tuned['physics_options']['cloth_surface_bend_stiffness'] == 0.
    assert tuned['cloths'][0]['rest_vertices_m'] == original['cloths'][0]['rest_vertices_m']
    assert validate_metadata(tuned) == tuned


def test_explicit_friction_changes_only_the_native_cloth_material_prior():
    original = candidate_metadata('trousers_folding')
    lower = candidate_metadata('trousers_folding', cloth_dynamic_friction=.25)
    assert validate_metadata(lower) == lower
    assert lower['cloths'][0].pop('friction') == .25
    assert lower == original
    assert 'friction' not in original['cloths'][0]
    for value in (-.1, 1.1, float('nan'), float('inf')):
        with pytest.raises(ValueError, match='dynamic friction'):
            candidate_metadata('trousers_folding', cloth_dynamic_friction=value)


@pytest.mark.parametrize('task,turns', [('shirt_folding', 1), ('trousers_folding', 1),
                                      ('shirt_folding', 3), ('trousers_folding', 3)])
def test_quarter_turn_rotates_initial_and_fold_goal_without_material_changes(task, turns):
    original = candidate_metadata(task)
    rotated = candidate_metadata(task, source_yaw_quarter_turns=turns)
    a, b = original['cloths'][0], rotated['cloths'][0]
    assert a['rest_vertices_m'] == b['rest_vertices_m']
    for field in ('initial_vertices_m', 'folded_reference_vertices_m'):
        assert np.allclose(_rotate_xy(np.asarray(a[field]), turns, np.array([.65, -.55])), b[field])
    assert rotated['design']['fold_sequence_axis_hinge_sign_radius_m'][-1][0] == 0
    assert validate_metadata(rotated) == rotated


@pytest.mark.parametrize('turns', [0, 1, 3])
def test_general_fold_frame_matches_existing_quarter_turn_geometry(turns):
    legacy = candidate_metadata('trousers_folding', source_yaw_quarter_turns=turns)
    general = candidate_metadata('trousers_folding', source_yaw_degrees=90.*turns)
    assert 'source_yaw_degrees' not in legacy
    assert 'fold_coordinate_frame' not in legacy['design']
    assert general['design']['fold_sequence_axis_hinge_sign_radius_m'][0][0] == 0
    a, b = legacy['cloths'][0], general['cloths'][0]
    assert np.array_equal(a['initial_vertices_m'], b['initial_vertices_m'])
    assert a['rest_vertices_m'] == b['rest_vertices_m']
    assert np.allclose(a['folded_reference_vertices_m'], b['folded_reference_vertices_m'], atol=1.e-12)
    assert validate_metadata(general) == general


def test_general_yaw_is_rigid_and_does_not_stack_two_rotation_parameters():
    from scenes.workrooms.laundry_room.garment_folding_assets import transform_fold_frame
    original = candidate_metadata('trousers_folding')
    general = candidate_metadata('trousers_folding', source_yaw_degrees=70.)
    frame = general['design']['fold_coordinate_frame']
    row = general['cloths'][0]
    for field in ('initial_vertices_m', 'folded_reference_vertices_m'):
        assert np.allclose(transform_fold_frame(np.asarray(row[field]), frame, inverse=True),
                           original['cloths'][0][field], atol=1.e-12)
    for options in (dict(source_yaw_degrees=70., source_yaw_quarter_turns=1),
                    dict(source_yaw_degrees=float('nan')), dict(source_yaw_degrees=360.)):
        with pytest.raises(ValueError, match='General source'):
            candidate_metadata('trousers_folding', **options)


def test_native_reset_is_explicit_source_pinned_and_preserves_original_rest(tmp_path):
    original = candidate_metadata('trousers_folding')
    q = np.asarray(original['cloths'][0]['initial_vertices_m'])
    q[:, 2] -= q[:, 2].min()-.9006
    # A small allowed native material strain cannot be hidden as a rigid reset.
    q[:, 0] = .65+(q[:, 0]-.65)*1.002
    files = {}
    for name, data in dict(cloth_q=np.stack([q, q]), cloth_qd=np.zeros((2, len(q), 3)),
                           control_mask=np.zeros((2, len(q)), dtype=np.uint8), time_s=np.array([0., .5])).items():
        path = tmp_path/(name+'.npy')
        np.save(path, data)
        files[name] = dict(path=str(path), sha256=digest(path), shape=list(data.shape))
    report = tmp_path/'native_report.json'
    report.write_text(json.dumps(dict(schema='laundry_single_garment_native.v1', complete=True,
        mode='settle', cloth_position_velocity_writes=0, cloth_id='towel',
        states=dict(complete=True, valid_samples=2, files=files))))
    reference = tmp_path/'reference.json'
    export_native_settled_reference('trousers_folding', report, reference)
    ordinary = candidate_metadata('trousers_folding', unfolded_reference_path=reference)
    assert ordinary['initial_state_kind'] == 'unfolded_shell'
    assert ordinary['cloths'][0]['initial_vertices_m'] == original['cloths'][0]['initial_vertices_m']
    native = candidate_metadata('trousers_folding', unfolded_reference_path=reference,
                                reset_profile='physx_native_settled_unfolded_v1')
    assert native['initial_state_kind'] == 'native_settled_unfolded_shell'
    assert np.array_equal(native['cloths'][0]['initial_vertices_m'], q)
    assert native['cloths'][0]['rest_vertices_m'] == original['cloths'][0]['rest_vertices_m']
    assert native['native_settled_source']['sample_index'] == 1
    assert validate_metadata(native) == native
    rotated_reference = tmp_path/'rotated_reference.json'
    export_native_settled_reference('trousers_folding', report, rotated_reference,
                                    yaw_quarter_turns=1, rotation_pivot_xy_m=[.65, -.55])
    rotated = candidate_metadata('trousers_folding', unfolded_reference_path=rotated_reference,
        reset_profile='physx_native_settled_unfolded_v1', source_yaw_quarter_turns=1)
    assert np.array_equal(rotated['cloths'][0]['initial_vertices_m'],
                          _rotate_xy(q, 1, np.array([.65, -.55])))
    assert rotated['native_settled_source']['yaw_quarter_turns'] == 1
    assert validate_metadata(rotated) == rotated
    # A second source already authored with yaw=90 must use identity export,
    # while preserving cumulative orientation for the folded-reference axes.
    oriented_q = _rotate_xy(q, 1, np.array([.65, -.55]))
    oriented_stream = tmp_path/'oriented_q.npy'
    np.save(oriented_stream, np.stack([oriented_q, oriented_q]))
    oriented_files = {**files, 'cloth_q': dict(path=str(oriented_stream), sha256=digest(oriented_stream),
                                              shape=[2, len(q), 3])}
    input_metadata = tmp_path/'metadata.json'
    input_metadata.write_text(json.dumps(dict(source_yaw_quarter_turns=1, source_center_xy_m=[.65, -.55])))
    source_scene = tmp_path/'source_scene.json'
    source_scene.write_text(json.dumps({'laundry_task_provenance': {'asset_candidate': {
        'manifest_path': str(tmp_path/'manifest.json'),
        'files_sha256': {str(input_metadata): digest(input_metadata)}}}}))
    oriented_report = tmp_path/'oriented_native_report.json'
    native_record = load_json(report)
    native_record['states']['files'] = oriented_files
    native_record['scene_input'] = dict(path=str(source_scene), sha256=digest(source_scene))
    oriented_report.write_text(json.dumps(native_record))
    identity_reference = tmp_path/'identity_reference.json'
    export_native_settled_reference('trousers_folding', oriented_report, identity_reference)
    identity = candidate_metadata('trousers_folding', unfolded_reference_path=identity_reference,
        reset_profile='physx_native_settled_unfolded_v1', source_yaw_quarter_turns=1)
    assert np.array_equal(identity['cloths'][0]['initial_vertices_m'], oriented_q)
    assert identity['native_settled_source']['rotation_wxyz'] == [1., 0., 0., 0.]
    assert identity['native_settled_source']['result_yaw_quarter_turns'] == 1
    assert validate_metadata(identity) == identity
    # A genuine observed yaw90 source can be rotated -20 degrees, with its
    # source and extra transform independently pinned, to yield yaw70.
    general_reference = tmp_path/'general_reference.json'
    export_native_settled_reference('trousers_folding', oriented_report, general_reference,
        yaw_degrees=-20., rotation_pivot_xy_m=[.65, -.55])
    general = candidate_metadata('trousers_folding', unfolded_reference_path=general_reference,
        reset_profile='physx_native_settled_unfolded_v1', source_yaw_degrees=70.)
    assert general['native_settled_source']['yaw_degrees'] == -20.
    assert general['native_settled_source']['result_yaw_degrees'] == 70.
    assert general['native_settled_source']['input_orientation']['source_yaw_quarter_turns'] == 1
    from scenes.workrooms.laundry_room.garment_folding_assets import _rotate_degrees
    assert np.array_equal(general['cloths'][0]['initial_vertices_m'],
                          _rotate_degrees(oriented_q, -20., np.array([.65, -.55])))
    assert validate_metadata(general) == general
    altered = load_json(reference)
    altered['vertices_m'][0][0] += .001
    reference.write_text(json.dumps(altered))
    with pytest.raises(ValueError, match='differs from its pinned observed state'):
        candidate_metadata('trousers_folding', unfolded_reference_path=reference,
                            reset_profile='physx_native_settled_unfolded_v1')
