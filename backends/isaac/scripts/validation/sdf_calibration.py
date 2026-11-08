"""Read-only native SDF tensor calibration shared by current validators."""
from itertools import permutations

import numpy as np

TOLERANCE_M = 20.e-6


def calibrate_layout(query, points, expected, normals, *, identity_indices=None):
    """Separate column identity from unchanged all-point metric thresholds.

    Optional identity witnesses identify the distance and gradient columns.
    Every supplied witness still participates in the unchanged 20 um distance
    and .05 gradient accuracy gates, including witnesses outside that subset.
    """
    native_query = query
    def query(p):
        raw = native_query(p)
        if raw.shape != (len(p), 4) or not np.isfinite(raw).all():
            raise ValueError('Every calibration query requires finite Nx4 raw data')
        return raw
    raw = query(points)
    if raw.shape != (len(points), 4) or not np.isfinite(raw).all():
        raise ValueError('Invalid native SDF calibration tensor')
    identity = np.arange(len(points)) if identity_indices is None else np.asarray(identity_indices)
    candidates = [i for i in range(4) if np.max(abs(raw[identity, i]-expected[identity])) <= TOLERANCE_M]
    if len(candidates) != 1:
        raise ValueError('Native signed-distance column is not uniquely calibrated')
    column = candidates[0]
    h = 1.e-5
    finite = np.column_stack([(query(points+np.eye(3)[i]*h)[:, column]
        -query(points-np.eye(3)[i]*h)[:, column])/(2*h) for i in range(3)])
    choices = []
    for order in permutations(i for i in range(4) if i != column):
        if np.max(abs(raw[:, order][identity]-normals[identity])) <= .05:
            choices.append(order)
    if len(choices) != 1:
        raise ValueError('Native XYZ gradient columns are not uniquely calibrated')
    order = choices[0]
    checks = {'all_distance_calibration_within_20um': bool(np.max(abs(raw[:, column]-expected)) <= TOLERANCE_M),
        'all_gradient_finite_difference_errors_within_005': bool(np.max(abs(raw[:, order]-finite)) <= .05),
        'all_gradient_source_normal_errors_within_005': bool(np.max(abs(raw[:, order]-normals)) <= .05)}
    return {'passed': all(checks.values()), 'checks': checks,
        'layout_identity_point_indices': identity.tolist(), 'metric_point_indices': list(range(len(points))),
        'distance_column': column, 'gradient_columns_xyz': list(order),
        'maximum_distance_calibration_error_m': float(np.max(abs(raw[:, column]-expected))),
        'maximum_gradient_finite_difference_error': float(np.max(abs(raw[:, order]-finite))),
        'maximum_gradient_source_normal_error': float(np.max(abs(raw[:, order]-normals))),
        'raw': raw, 'finite_difference_gradient': finite}
