"""Bounded decoding and validation of public metric camera-z depth arrays."""
from __future__ import annotations

from io import BytesIO

import numpy as np


MAX_DEPTH_PIXELS = 16_777_216
MAX_NPY_HEADER_BYTES = 4096


def validate_metric_depth(array):
    value = np.asarray(array)
    if value.dtype != np.dtype('float32') or value.ndim != 2 or not all(value.shape):
        raise ValueError('Depth must be a nonempty two-dimensional float32 array')
    if value.size > MAX_DEPTH_PIXELS:
        raise ValueError('Depth exceeds the pixel limit')
    if not np.isfinite(value).all() or (value < 0).any():
        raise ValueError('Depth must contain only finite nonnegative meters')
    if not (value > 0).any():
        raise ValueError('Depth contains no valid pixels')
    return value


def decode_metric_depth_npy(value: bytes, *, width: int, height: int):
    """Read a bounded, exact-shape float32 NPY without pickle or trailing data.

    Inspect the small header and declared payload length before np.load can
    allocate an array. Only NPY 1.0/2.0 are needed for this scalar image format.
    """
    if (isinstance(width, bool) or isinstance(height, bool)
            or not isinstance(width, int) or not isinstance(height, int)
            or width <= 0 or height <= 0 or width * height > MAX_DEPTH_PIXELS):
        raise ValueError('Invalid depth dimensions')
    if not isinstance(value, bytes) or len(value) < 10 or value[:6] != b'\x93NUMPY':
        raise ValueError('Depth response is not NPY')
    version = tuple(value[6:8])
    length_size = {(1, 0): 2, (2, 0): 4}.get(version)
    if length_size is None or len(value) < 8 + length_size:
        raise ValueError('Unsupported depth NPY version')
    header_size = int.from_bytes(value[8:8 + length_size], 'little')
    if not 0 < header_size <= MAX_NPY_HEADER_BYTES:
        raise ValueError('Depth NPY header exceeds the size limit')
    payload_offset = 8 + length_size + header_size
    if len(value) != payload_offset + width * height * 4:
        raise ValueError('Depth NPY payload length does not match its dimensions')
    stream = BytesIO(value)
    np.lib.format.read_magic(stream)
    read_header = (np.lib.format.read_array_header_1_0 if version == (1, 0)
                   else np.lib.format.read_array_header_2_0)
    shape, _, dtype = read_header(stream, max_header_size=MAX_NPY_HEADER_BYTES)
    if shape != (height, width) or dtype != np.dtype('float32') or dtype.hasobject:
        raise ValueError('Depth NPY shape or dtype does not match its observation')
    if stream.tell() != payload_offset:
        raise ValueError('Invalid depth NPY header length')
    depth = np.load(BytesIO(value), allow_pickle=False, max_header_size=MAX_NPY_HEADER_BYTES)
    return validate_metric_depth(depth)
