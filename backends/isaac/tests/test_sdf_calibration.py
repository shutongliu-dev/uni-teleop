"""Reject ambiguous, incomplete or nonfinite native SDF calibration evidence."""
import numpy as np
import pytest

from scripts.validation.sdf_calibration import calibrate_layout


def sphere_query(points):
    radii = np.linalg.norm(points, axis=1)
    gradients = points/radii[:, None]
    # Exercise actual-distance-last and a nontrivial XYZ column permutation.
    return np.c_[gradients[:, 2], gradients[:, 0], gradients[:, 1], radii-.01]


def test_layout_uses_observed_distance_and_xyz_gradient_columns():
    points = np.array([[.0105, 0., 0.], [-.0095, 0., 0.], [0., .0105, 0.],
                       [0., -.0095, 0.], [0., 0., .0105], [0., 0., -.0095]])
    expected = np.linalg.norm(points, axis=1)-.01
    normals = points/np.linalg.norm(points, axis=1)[:, None]
    result = calibrate_layout(sphere_query, points, expected, normals)
    assert result['distance_column'] == 3
    assert result['gradient_columns_xyz'] == [1, 2, 0]
    assert result['passed']
    with pytest.raises(ValueError, match='uniquely calibrated'):
        calibrate_layout(lambda p: np.zeros((len(p), 4)), points, expected, normals)


def test_layout_can_be_identified_without_discarding_failed_bore_metric():
    points = np.array([[.0105, 0., 0.], [-.0095, 0., 0.], [0., .0105, 0.],
        [0., -.0095, 0.], [0., 0., .0105], [0., 0., -.0095], [.0105, 0., 0.],
        [-.0095, 0., 0.], [0., .0105, 0.], [0., -.0095, 0.]])
    expected = np.linalg.norm(points, axis=1)-.01
    normals = points/np.linalg.norm(points, axis=1)[:, None]
    def bad_bore(p):
        raw = sphere_query(p)
        raw[[4, 9], 3] += 38.e-6
        return raw
    result = calibrate_layout(bad_bore, points, expected, normals,
        identity_indices=[0, 1, 2, 3, 5, 6, 7, 8])
    assert result['distance_column'] == 3 and result['gradient_columns_xyz'] == [1, 2, 0]
    assert not result['passed']
    assert not result['checks']['all_distance_calibration_within_20um']
    assert result['metric_point_indices'] == list(range(10))
    assert len(result['raw']) == 10


def test_nonfinite_unused_gradient_in_derivative_receipt_fails():
    points = np.array([[.011, 0., 0.], [-.011, 0., 0.], [0., .011, 0.],
                       [0., -.011, 0.], [0., 0., .011], [0., 0., -.011]])
    normals = points / np.linalg.norm(points, axis=1)[:, None]
    calls = 0
    def query(p):
        nonlocal calls
        calls += 1
        radii = np.linalg.norm(p, axis=1)
        raw = np.c_[p/radii[:, None], radii-.01]
        if calls == 2:
            raw[:, 0] = np.nan
        return raw
    with pytest.raises(ValueError, match='Every calibration query'):
        calibrate_layout(query, points, np.full(6, .001), normals)
