"""Reject missing/ambiguous native SDF evidence, retaining every witness."""
import numpy as np
import pytest

from scripts.validation.gear_sdf_checks import (calibrate_layout, calibration_points,
    calibration_source_audit, project_all)
from scripts.validation.gear_union_geometry import gear_exterior


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


def test_all_projection_points_retained_and_bad_gradient_fails_closed():
    points = np.array([[.01001, 0., 0.], [0., .00999, 0.]])
    good = project_all(sphere_query, points, 3, [1, 2, 0])
    assert good['passed'] and good['qualified'].tolist() == [True, True]
    assert good['iterations_raw'].shape == (9, 2, 4)
    def bad(p):
        result = sphere_query(p)
        result[1, :3] = 0.
        return result
    failure = project_all(bad, points, 3, [1, 2, 0])
    assert not failure['passed']
    assert failure['qualified'].tolist() == [True, False]
    assert failure['failed'].tolist() == [False, True]
    assert len(failure['projected']) == len(points)


@pytest.mark.parametrize('input_hub', [False, True])
def test_calibration_points_are_metric_signed_distance_of_real_union(input_hub):
    mesh = gear_exterior(input_hub=input_hub)
    points, expected, _ = calibration_points(input_hub=input_hub)
    assert calibration_source_audit(mesh, points, expected, input_hub=input_hub)['passed']
