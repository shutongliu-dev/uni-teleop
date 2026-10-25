"""Metric depth remains auditable while model previews have a fixed scale."""
from io import BytesIO
import json

import numpy as np
from PIL import Image
import pytest

from agent.model_client.depth import depth_model_input
from bench.sdk.depth import decode_metric_depth_npy


def npy(value, *, version=None):
    stream = BytesIO()
    if version is None:
        np.save(stream, value, allow_pickle=False)
    else:
        np.lib.format.write_array(stream, value, version=version, allow_pickle=False)
    return stream.getvalue()


@pytest.mark.parametrize('version', [(1, 0), (2, 0)])
def test_depth_decode_preserves_raw_values_and_zero_invalid(version):
    value = np.array([[.02345, .2, 4.], [0., 5., 7.654321]], dtype=np.float32)
    actual = decode_metric_depth_npy(npy(value, version=version), width=3, height=2)
    np.testing.assert_array_equal(actual, value)
    assert actual.dtype == np.float32


@pytest.mark.parametrize('kind', ['wrong_shape', 'float64', 'nan', 'infinity', 'negative',
                                 'all_invalid', 'trailing', 'truncated', 'oversized_header',
                                 'huge_declared_shape', 'object_dtype', 'version', 'magic'])
def test_depth_decode_rejects_invalid_or_unsafe_npy(kind):
    value = np.ones((2, 3), dtype=np.float32)
    if kind == 'wrong_shape': value = value.reshape(3, 2)
    if kind == 'float64': value = value.astype(np.float64)
    if kind == 'nan': value[0, 0] = np.nan
    if kind == 'infinity': value[0, 0] = np.inf
    if kind == 'negative': value[0, 0] = -.1
    if kind == 'all_invalid': value[:] = 0
    raw = npy(value)
    if kind == 'trailing': raw += b'junk'
    if kind == 'truncated': raw = raw[:-1]
    if kind == 'oversized_header': raw = raw[:8] + (60000).to_bytes(2, 'little') + raw[10:]
    if kind == 'huge_declared_shape': raw = raw.replace(b'(2, 3)', b'(2000000000, 3)')
    if kind == 'object_dtype': raw = raw.replace(b"'<f4'", b"'|O4'")
    if kind == 'version': raw = raw[:6] + b'\xff\x00' + raw[8:]
    if kind == 'magic': raw = b'not NPY'
    with pytest.raises(ValueError):
        decode_metric_depth_npy(raw, width=3, height=2)


def test_fixed_scale_preview_saturates_only_display_and_marks_invalid():
    raw = np.array([[0., .01, .1, 2.05, 4., 8.]], dtype=np.float32)
    before = raw.copy()
    value = depth_model_input(raw, grid_columns=6, grid_rows=1)
    np.testing.assert_array_equal(raw, before)
    with Image.open(BytesIO(value['preview_png_bytes'])) as preview:
        assert preview.mode == 'RGB' and preview.size == (6, 1)
        pixels = np.asarray(preview)
    np.testing.assert_array_equal(pixels[0, :, 0], [0, 1, 1, 128, 255, 255])
    np.testing.assert_array_equal(pixels[:, :, 0], pixels[:, :, 1])
    np.testing.assert_array_equal(pixels[:, :, 0], pixels[:, :, 2])
    assert value['grid']['values_m'] == [[None, .01, .1, 2.05, 4., 8.]]
    assert value['statistics']['maximum_valid_depth_m'] == 8.
    assert value['statistics']['invalid_pixel_count'] == 1
    assert value['preview']['near_m'] == .1 and value['preview']['far_m'] == 4.


def test_grid_samples_exact_640_by_480_pixel_centers_and_records_rounding():
    raw = np.full((480, 640), 1.234567, dtype=np.float32)
    raw[20, 20] = 0
    raw[460, 620] = 5.678912
    value = depth_model_input(raw)
    grid = value['grid']
    assert grid['u_pixels'] == list(range(20, 640, 40))
    assert grid['v_pixels'] == list(range(20, 480, 40))
    assert len(grid['values_m']) == 12 and all(len(row) == 16 for row in grid['values_m'])
    assert grid['values_m'][0][0] is None
    assert grid['values_m'][0][1] == 1.2346
    assert grid['values_m'][-1][-1] == 5.6789
    assert grid['rounding_m'] == .0001
    assert grid['depth_semantics'] == 'camera_z' and grid['depth_unit'] == 'm'
    json.dumps(grid, allow_nan=False)


def test_grid_generalizes_to_small_and_odd_images():
    raw = np.arange(1, 36, dtype=np.float32).reshape(5, 7)
    grid = depth_model_input(raw, grid_columns=3, grid_rows=2)['grid']
    assert grid['u_pixels'] == [1, 3, 5] and grid['v_pixels'] == [1, 3]
    assert grid['values_m'] == [[9., 11., 13.], [23., 25., 27.]]


@pytest.mark.parametrize('raw', [np.ones((2, 3), dtype=np.float64),
                               np.zeros((2, 3), dtype=np.float32),
                               np.array([[np.nan]], dtype=np.float32),
                               np.array([[-1]], dtype=np.float32)])
def test_preview_rejects_invalid_metric_data(raw):
    with pytest.raises(ValueError):
        depth_model_input(raw)
