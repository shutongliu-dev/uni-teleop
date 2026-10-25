"""The bounded diagnostic must retain failed roots and native query evidence."""
import numpy as np
import pytest

from scripts.validation.gear_bounded_roots import (
    BISECTION_ITERATIONS, OFFSETS_M, bounded_zero_search, frozen_directions, load_frozen_field)
from scripts.validation.gear_sdf_checks import TOLERANCE_M, ZERO_RESIDUAL_M, project_all


def scalar_query(function):
    def query(points):
        assert points.dtype == np.float32
        return np.c_[np.ones(len(points)), np.zeros((len(points), 2)), function(points)]
    return query


def search(tmp_path, query, **kwargs):
    return bounded_zero_search([query], [[0., 0., 0.]], [np.array([[1., 0., 0.]])],
        output=tmp_path/'search', source_indices=[31], **kwargs)


def test_scalar_root_does_not_use_normalized_api_gradient_as_derivative(tmp_path):
    query = scalar_query(lambda p: 4*(p[:, 0]-13.e-6))
    result = search(tmp_path, query)
    assert result['numeric_passed']
    assert result['numeric_qualified_count'] == 1
    assert result['native_query_call_count'] == BISECTION_ITERATIONS+2
    assert result['maximum_selected_displacement_m'] <= TOLERANCE_M
    assert result['maximum_selected_residual_m'] <= ZERO_RESIDUAL_M
    assert not result['geometry_qualified']
    assert result['exterior_classification'] == 'ambiguous'
    # All starts, intervals, rounds, submitted coordinates and Nx4 raw columns survive.
    assert len(list((tmp_path/'search').glob('*_request.npz'))) == 18
    for request in (tmp_path/'search').glob('*_request.npz'):
        data = np.load(request)
        assert np.all(data['displacement_by_shape_m'][:, data['queried']] <= TOLERANCE_M)
        assert np.all(data['source_indices'] == 31)
    assert np.load(tmp_path/'search/00_response.npz')['all_native_raw'].shape == (1, 9, 4)
    final = np.load(result['result_npz'])
    assert final['same_submitted_coordinate'].tolist() == [True]
    # Old projection remains independently failed and cannot be replaced.
    original = project_all(lambda p: query(p.astype(np.float32)), np.zeros((1, 3)), 3, [0, 1, 2])
    assert not original['passed']


def test_all_sign_changing_subintervals_are_bisected_and_ties_are_deterministic(tmp_path):
    query = scalar_query(lambda p: abs(p[:, 0])-7.e-6)
    result = search(tmp_path, query)
    assert result['numeric_passed'] and result['initial_sign_changing_bracket_count'] == 2
    for iteration in range(1, 17):
        assert np.load(tmp_path/f'search/{iteration:02d}_request.npz')['queried'].tolist() == [True, True]
    final = np.load(result['result_npz'])
    assert final['selected_points'][0, 0] < 0.  # Equal roots: lower interval wins.
    rows = []
    for iteration in range(17):
        request = np.load(tmp_path/f'search/{iteration:02d}_request.npz')
        response = np.load(tmp_path/f'search/{iteration:02d}_response.npz')
        for row in np.flatnonzero(response['finite'] & (abs(response['selected_raw'][:, 3]) <= ZERO_RESIDUAL_M)):
            label = request['labels'][row]
            interval = min(label[2], 7) if iteration == 0 else label[2]
            key = (request['displacement_by_shape_m'][:, row].max(),
                abs(response['selected_raw'][row, 3]), label[1], interval, iteration)
            rows.append((key, iteration, row))
    _, expected_record, expected_row = min(rows)
    assert final['chosen_record'].tolist() == [expected_record]
    assert final['chosen_row'].tolist() == [expected_row]


def test_compound_recomputes_winner_and_retains_each_rotated_shape(tmp_path):
    angle = .34
    rotation = np.array([[np.cos(angle), -np.sin(angle), 0.],
                         [np.sin(angle), np.cos(angle), 0.], [0., 0., 1.]])
    # The owner switches between initial endpoints. The root occurs on shape 1.
    first = scalar_query(lambda p: .25*(p[:, 0]-8.e-6))
    def second(points):
        body = points.astype(float) @ rotation.T
        # Exercise a per-shape distance column other than 3.
        return np.c_[body[:, 0]-10.e-6, np.zeros((len(points), 2)), np.ones(len(points))]
    result = bounded_zero_search([first, second], [[0., 0., 0.]], [np.array([[1., 0., 0.]])],
        output=tmp_path/'search', rotations=[np.eye(3), rotation],
        distance_columns=[3, 0], gradient_columns=[[0, 1, 2], [3, 1, 2]])
    assert result['numeric_passed'] and result['native_query_call_count'] == 36
    first_response = np.load(tmp_path/'search/00_response.npz')
    assert set(first_response['winner']) == {0, 1}
    brackets = np.load(tmp_path/'search/initial_brackets.npz')
    assert first_response['winner'][brackets['left_row']].tolist() == [1]
    assert first_response['winner'][brackets['right_row']].tolist() == [0]
    final = np.load(result['result_npz'])
    assert final['winner'].tolist() == [1]
    assert final['actual_field_points'].shape == (2, 1, 3)
    assert np.all(np.linalg.norm(final['actual_field_points'], axis=2) <= TOLERANCE_M)


@pytest.mark.parametrize('function', [lambda p: p[:, 0]-30.e-6,
    lambda p: np.where(p[:, 0] < 7.e-6, -2.e-6, 2.e-6),
    lambda p: np.full(len(p), 5.e-6)])
def test_outside_missing_and_discontinuous_roots_remain_unqualified(tmp_path, function):
    result = search(tmp_path, scalar_query(function))
    assert not result['numeric_passed']
    assert result['unqualified_source_indices'] == [31]
    assert np.load(result['result_npz'])['qualified'].tolist() == [False]
    assert (tmp_path/'search/17_request.npz').exists()


def test_nonwinning_nan_column_invalidates_complete_ledger(tmp_path):
    calls = 0
    def bad(points):
        nonlocal calls
        calls += 1
        raw = np.c_[np.ones(len(points)), np.zeros((len(points), 2)), np.full(len(points), .01)]
        if calls == 2:
            raw[0, 1] = np.nan
        return raw
    result = bounded_zero_search([scalar_query(lambda p: p[:, 0]-7.e-6), bad],
        [[0., 0., 0.]], [np.array([[1., 0., 0.]])], output=tmp_path/'search')
    assert not result['all_query_receipts_finite'] and not result['numeric_passed']
    raw = np.load(tmp_path/'search/01_response.npz')
    assert np.isnan(raw['all_native_raw'][1, 0, 1])
    assert raw['winner'].tolist() == [-1]
    assert not np.load(result['result_npz'])['final_bracket_valid'].any()


def test_final_requery_cannot_inherit_an_earlier_pass(tmp_path):
    calls = 0
    def changing(points):
        nonlocal calls
        calls += 1
        raw = scalar_query(lambda p: p[:, 0]-7.e-6)(points)
        if calls == 18:
            raw[:, 3] = 3.e-6
        return raw
    result = search(tmp_path, changing)
    assert result['selected_candidate_count'] == 1
    assert not result['numeric_passed'] and result['numeric_qualified_count'] == 0
    assert np.load(result['result_npz'])['same_submitted_coordinate'].all()


def test_callback_exception_retains_attempted_request_and_prior_responses(tmp_path):
    def throwing(points):
        raise RuntimeError('native query failed')
    with pytest.raises(RuntimeError, match='native query failed'):
        search(tmp_path, throwing)
    assert (tmp_path/'search/00_request.npz').exists()
    assert (tmp_path/'search/00_response.npz').exists()
    assert 'native query failed' in (tmp_path/'search/00_error.json').read_text()


def test_float32_guard_checks_every_shape_before_native_submission(tmp_path):
    seen = []
    def query(points):
        seen.append(points.copy())
        return scalar_query(lambda p: np.ones(len(p)))(points)
    # Here the ideal point is in the radius, but float32 rounds away from x0.
    # A second rotated shape must be checked even if it never wins the min.
    origin = np.array([[1000.00002, 0., 0.]])
    result = bounded_zero_search([query, query], origin, [np.array([[1., 0., 0.]])],
        output=tmp_path/'search', rotations=[np.eye(3), np.diag([-1., -1., 1.])])
    initial = np.load(tmp_path/'search/00_request.npz')
    ideal = np.linalg.norm(initial['requested_points']-origin, axis=1)
    assert np.all(ideal <= TOLERANCE_M)
    assert np.any(~initial['bounded'])
    assert not result['all_attempted_coordinates_within_bound'] and not result['numeric_passed']
    for points in seen:
        # Both shapes have the same actual radius after their inverse transform.
        actual = np.c_[abs(points[:, 0]), points[:, 1:]]
        assert np.all(np.linalg.norm(actual-origin, axis=1) <= TOLERANCE_M)


def test_exact_initial_zero_and_duplicate_float32_midpoints(tmp_path):
    exact = search(tmp_path, scalar_query(lambda p: p[:, 0]))
    assert exact['numeric_passed'] and exact['initial_sign_changing_bracket_count'] == 0
    assert exact['native_query_call_count'] == 2
    assert exact['maximum_selected_displacement_m'] == 0.
    # A nonzero base makes midpoint collapse in float32 well before 16 rounds.
    result = bounded_zero_search([scalar_query(lambda p: p[:, 0]-np.float32(.030007))],
        [[.03, 0., 0.]], [np.array([[1., 0., 0.]])], output=tmp_path/'repeated')
    assert result['numeric_passed'] and result['native_query_call_count'] == 18
    a = np.load(tmp_path/'repeated/15_request.npz')['submitted_local_float32']
    b = np.load(tmp_path/'repeated/16_request.npz')['submitted_local_float32']
    assert np.array_equal(a, b)


def test_empty_population_has_no_native_queries_or_vacuous_pass(tmp_path):
    def never(points):
        raise AssertionError('empty population must not call native')
    result = bounded_zero_search([never], np.empty((0, 3)), [], output=tmp_path/'empty')
    assert result['native_query_call_count'] == 0 and not result['numeric_passed']
    assert result['origin_count'] == 0


def test_bad_shape_is_retained_and_cannot_pass(tmp_path):
    result = search(tmp_path, lambda p: np.zeros((len(p), 3)))
    assert not result['numeric_passed'] and not result['all_query_receipts_finite']
    raw = np.load(tmp_path/'search/00_response.npz')
    assert raw['native_raw_0'].shape == (9, 3)
    assert not raw['schema_valid'].any()


def test_load_frozen_population_checks_exact_initial_points_and_labels():
    origins = np.array([[.03, 0., 0.]])
    directions = np.array([[1., 0., 0.]])
    fields = {'piece_origins': origins, 'piece_origin_indices': np.array([77]),
        'piece_directions_0': directions, 'piece_points': origins+OFFSETS_M[:, None]*directions,
        'piece_labels': np.c_[np.zeros((9, 2), dtype=int), np.arange(9)]}
    loaded = load_frozen_field(fields, 'piece')
    assert loaded['source_indices'].tolist() == [77]
    fields['piece_points'][0, 0] += 1.e-10
    with pytest.raises(ValueError, match='Frozen initial'):
        load_frozen_field(fields, 'piece')


def test_directions_follow_incident_source_normals_without_new_geometry():
    import trimesh
    mesh = trimesh.creation.box()
    directions = frozen_directions(mesh, [0])[0]
    assert directions.shape == (67, 3)
    assert np.allclose(np.linalg.norm(directions, axis=1), 1.)
    assert {tuple(d) for d in directions[64:]} == {(1., 0., 0.), (0., 1., 0.), (0., 0., 1.)}


def test_existing_output_is_never_overwritten(tmp_path):
    search(tmp_path, scalar_query(lambda p: p[:, 0]))
    before = (tmp_path/'search/report.json').read_bytes()
    with pytest.raises(FileExistsError):
        search(tmp_path, scalar_query(lambda p: np.full(len(p), 1.)))
    assert (tmp_path/'search/report.json').read_bytes() == before
