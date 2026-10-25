"""Reproducible model-facing views of validated public metric depth."""
from __future__ import annotations

from io import BytesIO

import numpy as np
from PIL import Image

from bench.sdk.depth import validate_metric_depth


PREVIEW_NEAR_M = 0.1
PREVIEW_FAR_M = 4.0


def depth_model_input(depth_m, *, grid_columns: int = 16, grid_rows: int = 12):
    """Make a fixed-scale preview and exact metric samples without altering raw z.

    Preview only: <=0.1 m is dark gray (1), >=4 m is white (255), and invalid
    zero pixels are black (0). The grid uses the original float32 pixel values,
    not region averages or preview values, rounded to 0.0001 m for concise
    model text; invalid samples become JSON null. The raw array is unchanged.
    """
    depth = validate_metric_depth(depth_m)
    for value in (grid_columns, grid_rows):
        if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 256:
            raise ValueError('Depth grid dimensions must be integers from 1 to 256')
    height, width = depth.shape
    valid = depth > 0
    gray = np.zeros(depth.shape, dtype=np.uint8)
    scaled = np.clip((depth[valid].astype(np.float64) - PREVIEW_NEAR_M)
                     / (PREVIEW_FAR_M - PREVIEW_NEAR_M), 0., 1.)
    gray[valid] = np.rint(1. + 254. * scaled).astype(np.uint8)
    rgb = np.repeat(gray[:, :, None], 3, axis=2)
    stream = BytesIO()
    Image.fromarray(rgb).save(stream, format='PNG')
    u = [min(width - 1, ((2 * column + 1) * width) // (2 * grid_columns))
         for column in range(grid_columns)]
    v = [min(height - 1, ((2 * row + 1) * height) // (2 * grid_rows))
         for row in range(grid_rows)]
    values = [[round(float(depth[y, x]), 4) if depth[y, x] > 0 else None for x in u] for y in v]
    valid_values = depth[valid]
    return {
        'preview_png_bytes': stream.getvalue(),
        'preview': {'encoding': 'grayscale_rgb_png', 'near_m': PREVIEW_NEAR_M,
                    'far_m': PREVIEW_FAR_M, 'near_intensity': 1, 'far_intensity': 255,
                    'invalid_intensity': 0, 'mapping': 'linear_near_dark_far_bright',
                    'clipping_scope': 'preview_only; original metric depth and samples are unclipped'},
        'grid': {'columns': grid_columns, 'rows': grid_rows, 'image_width': width,
                 'image_height': height, 'u_pixels': u, 'v_pixels': v, 'values_m': values,
                 'depth_semantics': 'camera_z', 'depth_unit': 'm', 'invalid_value': None,
                 'rounding_m': 0.0001,
                 'sampling': 'exact_cell_center_pixel_no_averaging',
                 'pixel_coordinates': 'zero_based_u_right_v_down'},
        'statistics': {'pixel_count': int(depth.size),
                       'valid_pixel_count': int(np.count_nonzero(valid)),
                       'invalid_pixel_count': int(np.count_nonzero(~valid)),
                       'valid_fraction': float(np.count_nonzero(valid) / depth.size),
                       'minimum_valid_depth_m': float(valid_values.min()),
                       'maximum_valid_depth_m': float(valid_values.max())},
    }
