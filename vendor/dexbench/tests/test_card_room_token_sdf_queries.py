import numpy as np

from scripts.validation.card_room_token_sdf import analytic_distance, analytic_witnesses, calibration, normalized_gradient_check, normal_zero_witnesses
from scripts.assets.prepare_card_room_token_sdf import sdf_profile, union_surface, geometry_review


def test_union_distance_omits_internal_disk_and_axis():
    points = [[0., 0., .007], [0., 0., .027], [.019, 0., .007],
              [.024, 0., .008], [.029, 0., .008], [.021, 0., .008]]
    np.testing.assert_allclose(analytic_distance(points), [-.014, -.020, -.001, .001, np.sqrt(2)*.001, .001], atol=1e-15)


def test_all_analytic_witnesses_on_boundary_and_rays_bracket():
    points, normals = analytic_witnesses(64)
    np.testing.assert_allclose(analytic_distance(points), 0., atol=1e-15)
    assert np.all(analytic_distance(points-.0005*normals) < 0)
    assert np.all(analytic_distance(points+.0005*normals) > 0)


def test_calibration_metric_and_normal_truth():
    points, expected, normals = calibration()
    np.testing.assert_allclose(analytic_distance(points), expected, atol=1e-15)
    h = 1e-6
    finite = np.column_stack([(analytic_distance(points+np.eye(3)[i]*h)-analytic_distance(points-np.eye(3)[i]*h))/(2*h) for i in range(3)])
    np.testing.assert_allclose(finite, normals, atol=1e-10)


def test_sparse_profile_requires_explicit_float32_storage():
    sparse, identity = sdf_profile('sparse1024_float32')
    assert sparse['sdfSubgridResolution'] == 6
    assert sparse['sdfBitsPerSubgridPixel'] == 'BitsPerPixel32'
    assert identity.endswith('_sparse32')
    assert sdf_profile('dense1024')[0]['sdfSubgridResolution'] == 0
    dense, identity = sdf_profile('dense512')
    assert dense['sdfResolution'] == 512 and dense['sdfSubgridResolution'] == 0
    assert identity == 'token_black_union_sdf512'
    assert sdf_profile('dense768')[0]['sdfResolution'] == 768
    assert sdf_profile('dense768')[1] == 'token_black_union_sdf768'
    finer, identity = sdf_profile('sparse1536_float32')
    assert finer['sdfResolution'] == 1536 and finer['sdfBitsPerSubgridPixel'] == 'BitsPerPixel32'
    assert identity == 'token_black_union_sdf1536_sparse32'
    parameters, identity = sdf_profile('sparse1250_float32_subgrid4')
    assert parameters['sdfResolution'] == 1250 and parameters['sdfSubgridResolution'] == 4
    assert identity.endswith('_subgrid4')
    parameters, identity = sdf_profile('sparse1250_float32_subgrid3')
    assert parameters['sdfResolution'] == 1250 and parameters['sdfSubgridResolution'] == 3
    assert parameters['sdfBitsPerSubgridPixel'] == 'BitsPerPixel32'
    assert identity.endswith('_subgrid3')


def test_best_observed_zero_retains_midpoint_failure_and_all_query_evidence():
    # A quantized field has a valid near-zero plateau but its final midpoint
    # can lie just below the step. Keep the actual earlier query, not a made-up
    # interpolated distance, while retaining the failed original midpoint.
    points = np.zeros((2,3)); normals = np.tile([1.,0.,0.],(2,1))
    def distance(p):
        return np.where(p[:,0]<-.99e-6,-2e-6,1e-8).astype(np.float32)
    result = normal_zero_witnesses(distance,points,normals,distance(points))
    assert result['query_distances_m'].shape == (2,17)
    np.testing.assert_array_equal(result['normal_offset_m'],0.)
    np.testing.assert_array_equal(result['residual_m'],distance(points))
    assert np.all(abs(result['original_midpoint_residual_m'])>1e-6)
    assert np.all(result['final_bracket_lower_distance_m']<0.)
    assert np.all(result['final_bracket_upper_distance_m']>0.)
    for col in range(17):
        np.testing.assert_array_equal(result['query_distances_m'][:,col],
            distance(points+result['query_normal_offsets_m'][:,col,None]*normals))


def test_best_observed_zero_cannot_hide_unresolved_sign_jump():
    points=np.zeros((1,3));normals=np.array([[1.,0.,0.]])
    def distance(p):
        return np.where(p[:,0]<0.,-1.3e-6,1.2e-6).astype(np.float32)
    result=normal_zero_witnesses(distance,points,normals,distance(points))
    assert abs(result['residual_m'][0])>1e-6
    assert np.all(abs(result['query_distances_m'])>1e-6)


def test_best_observed_zero_converges_on_continuous_metric_field():
    points=np.zeros((1,3));normals=np.array([[1.,0.,0.]])
    def distance(p):return (p[:,0]-12.3e-6).astype(np.float32)
    result=normal_zero_witnesses(distance,points,normals,distance(points))
    assert abs(result['normal_offset_m'][0]-12.3e-6)<31e-9
    assert abs(result['residual_m'][0])<31e-9


def test_native_normalized_gradient_requires_direction_not_derivative_length():
    def record(finite):
        return {'finite_difference_gradient': np.asarray([finite]), 'raw': np.array([[1.,0.,0.,0.]]),
            'gradient_columns_xyz': [0,1,2], 'maximum_gradient_finite_difference_error': .055,
            'checks': {'all_gradient_finite_difference_errors_within_005': False}}
    assert normalized_gradient_check(record([.945,0.,0.]))['passed']
    assert not normalized_gradient_check(record([0.,.945,0.]))['passed']
    assert not normalized_gradient_check(record([0.,0.,0.]))['passed']


def test_subdivision_preserves_external_polygon_surface_and_volume():
    vertices, faces = union_surface(512, .0005)
    review = geometry_review(vertices, faces, 512)
    original_v, original_f = union_surface(512)
    original = geometry_review(original_v, original_f, 512)
    assert review['triangles'] > original['triangles']
    assert abs(review['mesh_volume_m3']-original['mesh_volume_m3']) < 1e-16
    # New source vertices remain on the same physical axisymmetric boundary,
    # with the original512-sided chord deficit as the only circular error.
    assert np.max(abs(analytic_distance(vertices))) < 1e-15
    triangle_centers = vertices[faces].mean(axis=1)
    assert np.max(abs(analytic_distance(triangle_centers))) <= original['maximum_radial_chord_deficit_m']+1e-15
