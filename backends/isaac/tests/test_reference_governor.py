"""CPU reference tests; measured dynamics/contact physics need native acceptance."""
import numpy as np
import pytest

from bench.control.reference import ReferenceInfeasible, bounded_reference, stop_reference


@pytest.mark.parametrize('acceleration', [.3, 2., 3., 50.])
@pytest.mark.parametrize('dt', [.001, .02, .1])
def test_changing_targets_remain_stoppable_at_limits_and_deadline(acceleration, dt):
    rng = np.random.default_rng(51029)
    # Near-zero and negative boundaries exercise cancellation and float rounding.
    lower = np.array([-2.7, 0., -.02, 2.])
    upper = np.array([1.7, 1.5, .02, 2.1])
    vmax, amax = np.full(4, 1.), np.full(4, acceleration)
    q, v = lower.copy(), np.zeros(4)
    for remaining in range(400, 0, -1):
        desired = rng.choice([0., 1.], 4) * (upper - lower) + lower
        old_q, old_v = q.copy(), v.copy()
        q, v = bounded_reference(q, v, desired, lower, upper, vmax, amax, dt,
                                 steps_remaining=remaining)
        assert np.all(q >= lower) and np.all(q <= upper)
        assert np.all(abs(v) <= vmax + 1e-11)
        assert np.all(abs(v - old_v) <= amax * dt + 1e-11)
        np.testing.assert_allclose(v, (q - old_q) / dt, atol=1e-12, rtol=0)
        stop, _ = stop_reference(q, v, lower, upper, vmax, amax, dt)
        assert np.all(stop >= lower) and np.all(stop <= upper)
    np.testing.assert_array_equal(v, np.zeros(4))


@pytest.mark.parametrize('sign', [-1., 1.])
def test_discrete_braking_begins_before_the_recorded_infeasible_state(sign):
    q, v = np.array([sign * .7]), np.array([sign * 1.])
    lo, hi, vmax, amax = np.array([-1.]), np.array([1.]), np.array([1.]), np.array([3.])
    velocities = []
    for _ in range(80):
        q, v = bounded_reference(q, v, np.array([sign]), lo, hi, vmax, amax, .02)
        velocities.append(abs(v[0]))
        stop, _ = stop_reference(q, v, lo, hi, vmax, amax, .02)
        assert lo[0] <= stop[0] <= hi[0]
    assert min(velocities) == 0. and abs(q[0] - sign) < 1e-12


def test_impossible_state_is_rejected_without_unsafe_position_clip():
    q, v = np.array([.0001]), np.array([-.2])
    with pytest.raises(ReferenceInfeasible) as caught:
        bounded_reference(q, v, np.array([0.]), np.array([0.]), np.array([1.]),
                          np.array([1.]), np.array([3.]), .02)
    assert caught.value.joint_index == 0
    np.testing.assert_array_equal(q, [.0001])
    np.testing.assert_array_equal(v, [-.2])


def test_deadline_cannot_require_an_instant_velocity_reset():
    with pytest.raises(ReferenceInfeasible):
        bounded_reference(np.array([.5]), np.array([.2]), np.array([.5]),
                          np.array([0.]), np.array([1.]), np.array([1.]), np.array([3.]), .02,
                          steps_remaining=1)
