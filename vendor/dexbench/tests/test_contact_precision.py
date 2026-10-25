"""Regression cases for actual float32 recording contradictions, not physics gates."""
import json
import math

import numpy as np
import pytest

from bench.simulation.isaacsim_backend.contact_precision import (
    NORMAL_UNIT_LENGTH_ATOL, aggregation_metadata, summarize_normal_impulses,
)


# Fixed float32 examples reproduce the former recording/audit contradiction.
THREE_FORCES = [0.09379968792200089, 0.08196552842855453, 0.06853123009204865]
THIRTY_FIVE_FORCES = [
    0.06480841338634491, 0.0066255852580070496, 0.06737454980611801,
    0.08794102817773819, 0.07739707827568054, 0.043983884155750275,
    0.07234929502010345, 0.06597242504358292, 0.07884527742862701,
    0.027225598692893982, 0.021340496838092804, 0.07179645448923111,
    0.06786992400884628, 0.014933206140995026, 0.09256798774003983,
    0.012631596066057682, 0.04877627640962601, 0.045038603246212006,
    0.09585648030042648, 0.07115751504898071, 0.038450226187705994,
    0.0661272406578064, 0.01773650385439396, 0.0969148576259613,
    0.06054627522826195, 0.059991154819726944, 0.0962473526597023,
    0.058070987462997437, 0.07809505611658096, 0.06662178039550781,
    0.043403346091508865, 0.048214320093393326, 0.0885157510638237,
    0.07068415731191635, 0.08344424515962601,
]


@pytest.mark.parametrize('forces,normal', [(THREE_FORCES, [.6, .8, 0]),
                                           (THIRTY_FIVE_FORCES, [0, 0, 1])])
def test_float32_reduction_counterexample_passes_unchanged_triangle_gate(forces, normal):
    f = np.asarray(forces, dtype=np.float32)
    n = np.tile(np.asarray(normal, dtype=np.float32), (len(f), 1))
    old_vector = np.sum(f[:, None] * n, axis=0)
    old_sum = float(np.abs(f).sum())
    assert math.hypot(*old_vector.tolist()) > old_sum + 1e-8

    result = summarize_normal_impulses(f, n)
    # Same gate as the existing audit, no new relative tolerance.
    assert math.hypot(*result['impulse_vector_ns']) <= result['impulse_magnitude_sum_ns'] + 1e-8
    # Independent scalar math.fsum/math.hypot oracle validates both semantics.
    points = [[float(a) * float(b) for b in direction] for a, direction in zip(f, n)]
    expected_vector = [math.fsum(point[k] for point in points) for k in range(3)]
    expected_magnitude = math.fsum(math.hypot(*point) for point in points)
    assert result['impulse_vector_ns'] == pytest.approx(expected_vector, rel=2e-15, abs=1e-17)
    assert result['impulse_magnitude_sum_ns'] == pytest.approx(expected_magnitude, rel=2e-15)
    assert result['raw_scalar_abs_impulse_sum_ns'] == math.fsum(abs(float(x)) for x in f)


def test_opposing_directions_retain_point_magnitudes_and_native_signs():
    result = summarize_normal_impulses([1., 1., -.5], [[1., 0, 0], [-1., 0, 0], [0, 1., 0]])
    assert result['impulse_vector_ns'] == [0., -.5, 0.]
    assert result['impulse_magnitude_sum_ns'] == 2.5
    assert result['raw_scalar_abs_impulse_sum_ns'] == 2.5


def test_small_native_normal_roundoff_is_recorded_not_normalized():
    length = 1. + NORMAL_UNIT_LENGTH_ATOL / 2
    result = summarize_normal_impulses([2.], [[0, 0, length]])
    assert result['impulse_vector_ns'] == [0., 0., 2. * length]
    assert result['impulse_magnitude_sum_ns'] == 2. * length
    assert result['raw_scalar_abs_impulse_sum_ns'] == 2.
    assert result['normal_length_max_abs_error'] == length - 1.


@pytest.mark.parametrize('normal', [[0, 0, 0], [0, 0, .5], [0, 0, 1.01],
                                    [0, 0, 1 + 2 * NORMAL_UNIT_LENGTH_ATOL]])
def test_nonunit_active_normals_are_rejected(normal):
    with pytest.raises(ValueError, match='not unit length'):
        summarize_normal_impulses([1.], [normal])


def test_zero_impulse_may_have_a_zero_normal():
    result = summarize_normal_impulses([0., 2.], [[0, 0, 0], [0, 0, 1.]])
    assert result['impulse_magnitude_sum_ns'] == 2.
    assert result['nonzero_impulse_normal_count'] == 1
    assert result['normal_length_max_abs_error'] == 0.
    assert summarize_normal_impulses([0.], [[0, 0, 0]])['impulse_vector_ns'] == [0., 0., 0.]


@pytest.mark.parametrize('forces,normals', [([float('nan')], [[0, 0, 1]]),
    ([float('inf')], [[0, 0, 1]]), ([1.], [[0, float('nan'), 1]]),
    ([0.], [[0, 0, float('nan')]]), ([1.], [[0, 0, float('inf')]])])
def test_nonfinite_input_is_rejected_even_on_zero_impulse(forces, normals):
    with pytest.raises(ValueError, match='finite'):
        summarize_normal_impulses(forces, normals)


@pytest.mark.parametrize('forces,normals', [([], []), ([1.], [[0, 0, 1], [0, 0, 1]]),
                                            ([[1.]], [[0, 0, 1]]), ([1.], [[0, 1]])])
def test_invalid_point_shapes_are_rejected(forces, normals):
    with pytest.raises(ValueError, match='matching Nx3'):
        summarize_normal_impulses(forces, normals)


def test_overflow_is_rejected_instead_of_recording_infinity():
    with pytest.raises(ValueError, match='overflowed'):
        summarize_normal_impulses([1e308], [[0, 0, 1.]])


def test_metadata_describes_source_arithmetic_without_changing_solver():
    metadata = aggregation_metadata()
    assert metadata['arithmetic_dtype'] == 'float64'
    assert metadata['normal_unit_length_atol'] == 32 * np.finfo(np.float32).eps
    assert not metadata['normals_normalized_or_clamped']
    assert not metadata['solver_parameters_changed']
    json.dumps(metadata, allow_nan=False)
    json.dumps(summarize_normal_impulses([1.], [[0, 0, 1.]]), allow_nan=False)


def test_batch_contact_groups_match_scalar_reductions_exactly():
    from bench.simulation.isaacsim_backend.contact_precision import summarize_contact_groups
    rng=np.random.default_rng(42)
    counts=np.array([1,4,35,3,4,1,35])
    starts=np.cumsum(np.r_[0,counts[:-1]])
    f=rng.normal(size=counts.sum()).astype(np.float32)
    n=rng.normal(size=(counts.sum(),3));n/=np.linalg.norm(n,axis=1)[:,None];n=n.astype(np.float32)
    f[::7]=0.;n[::7]=0.
    batch=summarize_contact_groups(f,n,starts,counts)
    assert batch==[summarize_normal_impulses(f[s:s+c],n[s:s+c]) for s,c in zip(starts,counts)]


@pytest.mark.parametrize('f,n', [([1.],[[0,0,2.]]),([float('nan')],[[0,0,1.]]),([1e308],[[0,0,1.]])])
def test_batch_rejects_invalid_or_overflowing_contacts(f,n):
    from bench.simulation.isaacsim_backend.contact_precision import summarize_contact_groups
    # Duplicate points make the finite individual impulse overflow on reduction.
    with pytest.raises(ValueError):
        summarize_contact_groups(f*2,n*2,[0],[2])
