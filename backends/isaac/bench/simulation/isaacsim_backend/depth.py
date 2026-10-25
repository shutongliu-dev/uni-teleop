"""Validate native RTX optical-z depths without importing a simulator.

Isaac Sim 6.1.0 bundles Replicator 1.13.36. Its
``docs/annotators_details.rst`` documents ``distance_to_image_plane`` as an
H x W float32 image in meters, with zero denoting no hit. This is optical
camera z, unlike ``distance_to_camera`` (Euclidean ray range).
"""
from __future__ import annotations

import numpy as np


DEPTH_SOURCE = 'omni.replicator.core.distance_to_image_plane'


def validate_camera_z_depth(raw, *, width: int, height: int):
    """Return an owned float32 depth image plus explicit validity statistics.

    Preserve every positive finite native value. Zero is the documented
    no-hit value; positive infinity is also accepted as no hit and counted
    before conversion to zero. NaNs, negative depths, wrong types/shapes and
    completely empty images fail instead of being presented as sensor data.
    """
    array = np.asarray(raw)
    if array.dtype != np.dtype('float32'):
        raise RuntimeError(f'Native optical-z depth must be float32, got {array.dtype}')
    if array.shape != (height, width):
        raise RuntimeError(f'Native optical-z depth shape {array.shape} != {(height, width)}')
    if np.isnan(array).any() or np.isneginf(array).any() or (array < 0).any():
        raise RuntimeError('Native optical-z depth contains NaN, negative infinity or negative values')
    positive_infinity = np.isposinf(array)
    zero = array == 0
    valid = np.isfinite(array) & (array > 0)
    valid_count = int(np.count_nonzero(valid))
    if not valid_count:
        raise RuntimeError('Native optical-z depth contains no valid pixels')
    depth = np.array(array, dtype=np.float32, order='C', copy=True)
    depth[positive_infinity | zero] = 0.0
    values = depth[valid]
    return depth, {
        'pixel_count': int(depth.size),
        'valid_pixel_count': valid_count,
        'invalid_pixel_count': int(depth.size - valid_count),
        'raw_zero_pixel_count': int(np.count_nonzero(zero)),
        'raw_positive_infinity_pixel_count': int(np.count_nonzero(positive_infinity)),
        'valid_fraction': float(valid_count / depth.size),
        'minimum_valid_depth_m': float(values.min()),
        'maximum_valid_depth_m': float(values.max()),
        'invalid_mapping': 'raw_zero_or_positive_infinity_to_zero',
    }
