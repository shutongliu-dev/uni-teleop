"""Summarize native normal-impulse vectors without float32 reduction drift.

This module changes recording arithmetic only. It never normalizes or clamps a
native normal, rescales an impulse, or writes simulation state. Invalid normals
on nonzero impulses are rejected rather than hidden by magnitude aggregation.
"""
from __future__ import annotations

import numpy as np


AGGREGATION_VERSION = 'float64_point_normal_impulses_v1'
# Normals originate in PhysX float32 tensors. This is a unit-normal integrity
# gate, not a loosened contact/physics/task acceptance threshold.
NORMAL_UNIT_LENGTH_ATOL = float(32 * np.finfo(np.float32).eps)


def aggregation_metadata():
    return {'version': AGGREGATION_VERSION,
            'arithmetic_dtype': 'float64',
            'point_vector_definition': 'float64(native scalar normal impulse) * float64(native normal)',
            'impulse_vector_ns': 'Sum of the point vectors in float64.',
            'impulse_magnitude_sum_ns': 'Sum of the Euclidean lengths of those same point vectors in float64.',
            'raw_scalar_abs_impulse_sum_ns': 'Independent float64 sum of absolute native scalar normal impulses; retained without rescaling.',
            'normal_unit_length_atol': NORMAL_UNIT_LENGTH_ATOL,
            'normal_tolerance_source': '32 * numpy.finfo(numpy.float32).eps; applies to nonzero-impulse normals only.',
            'normal_integrity': 'All inputs finite; every nonzero-impulse normal must have length within tolerance of 1. Zero-impulse points may have a zero normal.',
            'normals_normalized_or_clamped': False,
            'solver_parameters_changed': False}


def summarize_normal_impulses(scalar_impulses, normals):
    """Return JSON-safe pair fields from finite Nx1 scalar and Nx3 directions.

    The scalar input is a one-dimensional N-vector. Signed native scalars are
    preserved in point vectors; both magnitude sums are nonnegative. Native
    zero-impulse points remain part of the caller's point count.
    """
    f = np.asarray(scalar_impulses, dtype=np.float64)
    n = np.asarray(normals, dtype=np.float64)
    if f.ndim != 1 or len(f) == 0 or n.shape != (len(f), 3):
        raise ValueError('Expected a nonempty scalar impulse vector and matching Nx3 normals')
    if not np.isfinite(f).all() or not np.isfinite(n).all():
        raise ValueError('Native contact impulses and normals must be finite')
    active = f != 0.
    active_normals = n[active]
    with np.errstate(over='ignore', invalid='ignore'):
        lengths = np.linalg.norm(active_normals, axis=1)
    normal_error = float(np.max(np.abs(lengths - 1.))) if len(lengths) else 0.
    if not np.isfinite(lengths).all() or normal_error > NORMAL_UNIT_LENGTH_ATOL:
        raise ValueError(f'Nonzero-impulse normal is not unit length: maximum error {normal_error:.17g}')
    with np.errstate(over='ignore', invalid='ignore'):
        point_vectors = f[:, None] * n
        vector = np.sum(point_vectors, axis=0, dtype=np.float64)
        magnitude_sum = float(np.sum(np.linalg.norm(point_vectors, axis=1), dtype=np.float64))
        raw_scalar_sum = float(np.sum(np.abs(f), dtype=np.float64))
    if not np.isfinite(point_vectors).all() or not np.isfinite(vector).all() or not np.isfinite([magnitude_sum, raw_scalar_sum]).all():
        raise ValueError('Native contact aggregate overflowed finite float64 arithmetic')
    return {'impulse_vector_ns': vector.tolist(),
            'impulse_magnitude_sum_ns': magnitude_sum,
            'raw_scalar_abs_impulse_sum_ns': raw_scalar_sum,
            'normal_length_max_abs_error': normal_error,
            'nonzero_impulse_normal_count': int(np.count_nonzero(active))}


def summarize_contact_groups(scalar_impulses, normals, starts, counts):
    """Batch equal-size manifolds, retaining each pair's float64 reduction order."""
    starts=np.asarray(starts,dtype=np.int64)
    counts=np.asarray(counts,dtype=np.int64)
    if starts.ndim!=1 or counts.shape!=starts.shape or np.any(counts<=0) or np.any(starts<0):
        raise ValueError('Expected nonempty contact ranges')
    f=np.asarray(scalar_impulses,dtype=np.float64)
    n=np.asarray(normals,dtype=np.float64)
    if f.ndim!=1 or n.shape!=(len(f),3) or np.any(starts+counts>len(f)):
        raise ValueError('Contact range exceeds native arrays')
    result=[None]*len(starts)
    for count in np.unique(counts):
        group=np.flatnonzero(counts==count)
        indices=starts[group,None]+np.arange(count)
        values,directions=f[indices],n[indices]
        if not np.isfinite(values).all() or not np.isfinite(directions).all():
            raise ValueError('Native contact impulses and normals must be finite')
        active=values!=0.
        with np.errstate(over='ignore',invalid='ignore'):
            lengths=np.linalg.norm(directions,axis=2)
            errors=np.where(active,np.abs(lengths-1.),0.).max(axis=1)
            vectors=values[:,:,None]*directions
            sums=np.sum(vectors,axis=1,dtype=np.float64)
            magnitudes=np.sum(np.linalg.norm(vectors,axis=2),axis=1,dtype=np.float64)
            raw=np.sum(np.abs(values),axis=1,dtype=np.float64)
        if not np.isfinite(errors).all() or np.any(errors>NORMAL_UNIT_LENGTH_ATOL):
            raise ValueError('Nonzero-impulse normal is not unit length')
        if not all(np.isfinite(x).all() for x in (vectors,sums,magnitudes,raw)):
            raise ValueError('Native contact aggregate overflowed finite float64 arithmetic')
        nonzero=np.count_nonzero(active,axis=1)
        for row,index in enumerate(group):
            result[index]={'impulse_vector_ns':sums[row].tolist(),
                'impulse_magnitude_sum_ns':float(magnitudes[row]),
                'raw_scalar_abs_impulse_sum_ns':float(raw[row]),
                'normal_length_max_abs_error':float(errors[row]),
                'nonzero_impulse_normal_count':int(nonzero[row])}
    return result
