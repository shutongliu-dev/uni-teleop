import itertools
import unittest

import numpy as np

from sonic_mujoco.tactile_dynamics import (
    AR1NoiseState,
    HoldGapMarkovClock,
    SignedOUDriftState,
)


class HoldGapMarkovClockTest(unittest.TestCase):
    def test_one_state_clock_is_exact_periodic_and_has_no_event_cap(self) -> None:
        clock = HoldGapMarkovClock(
            [1.0],
            [[1.0]],
            record_period_s=0.02,
            seed=7,
        )

        self.assertEqual(tuple(clock.events_until(0.019)), ())
        np.testing.assert_allclose(
            tuple(clock.events_until(0.1)), np.arange(1, 6) * 0.02
        )
        self.assertEqual(clock.event_index, 5)

        clock.reset()
        events = tuple(clock.events_until(5001 * 0.02))
        self.assertEqual(len(events), 5001)
        self.assertAlmostEqual(events[-1], 5001 * 0.02)

    def test_seeded_events_are_invariant_to_query_chunking(self) -> None:
        initial = [0.2, 0.3, 0.5]
        transition = [
            [0.7, 0.2, 0.1],
            [0.1, 0.7, 0.2],
            [0.2, 0.3, 0.5],
        ]
        whole = HoldGapMarkovClock(initial, transition, seed=918)
        chunked = HoldGapMarkovClock(initial, transition, seed=918)
        other = HoldGapMarkovClock(initial, transition, seed=919)

        expected = tuple(whole.events_until(20.0))
        actual = tuple(
            itertools.chain.from_iterable(
                chunked.events_until(time) for time in (0.13, 1.7, 4.01, 9.8, 20.0)
            )
        )

        self.assertEqual(actual, expected)
        self.assertNotEqual(tuple(other.events_until(20.0)), expected)

        chunked.reset(seed=1234)
        first = tuple(chunked.events_until(3.0))
        chunked.reset(seed=1234)
        self.assertEqual(tuple(chunked.events_until(3.0)), first)

    def test_long_run_matches_markov_probabilities(self) -> None:
        initial = np.array([0.625, 0.375])
        transition = np.array([[0.85, 0.15], [0.25, 0.75]])
        period = 0.02
        clock = HoldGapMarkovClock(
            initial,
            transition,
            record_period_s=period,
            seed=51,
        )
        event_count = 120_001
        events = np.fromiter(
            itertools.islice(clock.events_until(1e9), event_count),
            dtype=np.float64,
        )
        ticks = np.rint(events / period).astype(np.int64)
        gap = np.diff(np.r_[0, ticks])
        state = gap - 1

        observed_initial = np.bincount(state, minlength=2) / len(state)
        np.testing.assert_allclose(observed_initial, initial, atol=0.01)
        for previous in range(2):
            mask = state[:-1] == previous
            observed = np.bincount(state[1:][mask], minlength=2) / np.count_nonzero(
                mask
            )
            np.testing.assert_allclose(observed, transition[previous], atol=0.01)

    def test_rejects_invalid_probabilities_time_and_mutation(self) -> None:
        with self.assertRaisesRegex(ValueError, "nonempty vector"):
            HoldGapMarkovClock([], np.empty((0, 0)))
        with self.assertRaisesRegex(ValueError, "sum to one"):
            HoldGapMarkovClock([0.4, 0.4], [[1.0, 0.0], [0.0, 1.0]])
        with self.assertRaisesRegex(ValueError, "nonnegative"):
            HoldGapMarkovClock([1.1, -0.1], [[1.0, 0.0], [0.0, 1.0]])
        with self.assertRaisesRegex(ValueError, "shape"):
            HoldGapMarkovClock([1.0, 0.0], [[1.0]])
        with self.assertRaisesRegex(ValueError, "row 1"):
            HoldGapMarkovClock([1.0, 0.0], [[1.0, 0.0], [0.2, 0.2]])
        with self.assertRaisesRegex(TypeError, "real numbers"):
            HoldGapMarkovClock([True], [[1.0]])
        with self.assertRaisesRegex(ValueError, "positive"):
            HoldGapMarkovClock([1.0], [[1.0]], record_period_s=0.0)

        initial = np.array([1.0, 0.0])
        transition = np.eye(2)
        clock = HoldGapMarkovClock(initial, transition)
        initial[:] = [0.0, 1.0]
        transition[:] = transition[::-1]
        self.assertEqual(clock.gap_frames, 1)
        self.assertEqual(clock.initial_probability.tolist(), [1.0, 0.0])
        with self.assertRaisesRegex(ValueError, "nondecreasing"):
            tuple(clock.events_until(-0.01))


class AR1NoiseStateTest(unittest.TestCase):
    def test_first_sample_is_stationary_and_reset_is_reproducible(self) -> None:
        standard_deviation = np.array([0.5, 2.0])
        noise = AR1NoiseState(
            2,
            marginal_std=standard_deviation,
            coefficient=[0.2, 0.8],
            seed=73,
        )
        expected = np.random.default_rng(73).standard_normal(2) * standard_deviation

        np.testing.assert_array_equal(noise.sample(), expected)
        sequence = [noise.sample() for _ in range(8)]
        noise.reset()
        np.testing.assert_array_equal(noise.sample(), expected)
        replay = [noise.sample() for _ in range(8)]
        for first, second in zip(sequence, replay, strict=True):
            np.testing.assert_array_equal(first, second)

    def test_long_run_has_requested_marginal_and_lag_one_correlation(self) -> None:
        marginal_std = 1.7
        coefficient = 0.65
        noise = AR1NoiseState(
            1,
            marginal_std=marginal_std,
            coefficient=coefficient,
            seed=2026,
        )
        samples = np.fromiter((noise.sample()[0] for _ in range(140_000)), float)
        samples = samples[2_000:]

        self.assertAlmostEqual(float(samples.mean()), 0.0, delta=0.04)
        self.assertAlmostEqual(float(samples.std()), marginal_std, delta=0.04)
        correlation = float(np.corrcoef(samples[:-1], samples[1:])[0, 1])
        self.assertAlmostEqual(correlation, coefficient, delta=0.015)

    def test_zero_noise_is_an_exact_degenerate_state(self) -> None:
        noise = AR1NoiseState((2, 3), marginal_std=0.0, coefficient=0.9, seed=3)
        for _ in range(20):
            np.testing.assert_array_equal(noise.sample(), np.zeros((2, 3)))
        self.assertEqual(noise.sample_index, 20)

        state = noise.state
        state.fill(12.0)
        np.testing.assert_array_equal(noise.state, np.zeros((2, 3)))

    def test_rejects_invalid_shape_parameters_and_seed(self) -> None:
        with self.assertRaisesRegex(ValueError, "dimensions must be positive"):
            AR1NoiseState(0, marginal_std=1.0)
        with self.assertRaisesRegex(ValueError, "nonnegative"):
            AR1NoiseState(2, marginal_std=[1.0, -1.0])
        with self.assertRaisesRegex(ValueError, "magnitude"):
            AR1NoiseState(1, marginal_std=1.0, coefficient=1.0)
        with self.assertRaisesRegex(ValueError, "broadcastable"):
            AR1NoiseState(3, marginal_std=[1.0, 2.0])
        with self.assertRaisesRegex(TypeError, "integer"):
            AR1NoiseState(1, marginal_std=1.0, seed=True)


class SignedOUDriftStateTest(unittest.TestCase):
    def test_uses_exact_discrete_transition(self) -> None:
        sigma = np.array([0.5, 2.0])
        tau = np.array([0.7, 3.0])
        drift = SignedOUDriftState(
            2,
            stationary_std=sigma,
            tau_s=tau,
            seed=29,
        )
        rng = np.random.default_rng(29)

        first_duration = 0.3
        first_scale = sigma * np.sqrt(-np.expm1(-2.0 * first_duration / tau))
        expected = first_scale * rng.standard_normal(2)
        np.testing.assert_allclose(drift.advance_to(first_duration), expected)

        second_duration = 0.8
        decay = np.exp(-second_duration / tau)
        scale = sigma * np.sqrt(-np.expm1(-2.0 * second_duration / tau))
        expected = decay * expected + scale * rng.standard_normal(2)
        np.testing.assert_allclose(
            drift.advance_to(first_duration + second_duration),
            expected,
        )

    def test_seeded_path_is_reproducible_across_loop_chunking(self) -> None:
        times = (0.02, 0.08, 0.13, 0.51, 0.72, 1.0)
        whole = SignedOUDriftState(3, stationary_std=0.8, tau_s=0.4, seed=90)
        chunked = SignedOUDriftState(3, stationary_std=0.8, tau_s=0.4, seed=90)

        expected = [whole.advance_to(time) for time in times]
        actual = []
        for group in (times[:2], times[2:5], times[5:]):
            actual.extend(chunked.advance_to(time) for time in group)

        for first, second in zip(expected, actual, strict=True):
            np.testing.assert_array_equal(first, second)

        chunked.reset(seed=901)
        first = [chunked.advance_to(time) for time in times]
        chunked.reset(seed=901)
        replay = [chunked.advance_to(time) for time in times]
        for left, right in zip(first, replay, strict=True):
            np.testing.assert_array_equal(left, right)

    def test_long_run_has_requested_stationary_statistics(self) -> None:
        sigma = 1.3
        tau = 0.8
        step = 0.1
        drift = SignedOUDriftState(1, stationary_std=sigma, tau_s=tau, seed=121)
        samples = np.empty(160_000)
        for index in range(len(samples)):
            samples[index] = drift.advance_to((index + 1) * step)[0]
        samples = samples[2_000:]

        self.assertAlmostEqual(float(samples.mean()), 0.0, delta=0.04)
        self.assertAlmostEqual(float(samples.std()), sigma, delta=0.04)
        correlation = float(np.corrcoef(samples[:-1], samples[1:])[0, 1])
        self.assertAlmostEqual(correlation, float(np.exp(-step / tau)), delta=0.01)
        self.assertLess(float(samples.min()), 0.0)
        self.assertGreater(float(samples.max()), 0.0)

    def test_zero_drift_and_zero_duration_are_exact_noops(self) -> None:
        drift = SignedOUDriftState(
            (2, 2),
            stationary_std=0.0,
            tau_s=10.0,
            seed=18,
            initial_time=2.0,
        )
        np.testing.assert_array_equal(drift.advance_to(2.0), np.zeros((2, 2)))
        self.assertEqual(drift.advance_index, 0)
        np.testing.assert_array_equal(drift.advance_to(1e6), np.zeros((2, 2)))
        self.assertEqual(drift.advance_index, 1)

        state = drift.state
        state.fill(9.0)
        np.testing.assert_array_equal(drift.state, np.zeros((2, 2)))
        with self.assertRaisesRegex(ValueError, "nondecreasing"):
            drift.advance_to(1.0)

    def test_rejects_invalid_parameters_and_times(self) -> None:
        with self.assertRaisesRegex(ValueError, "nonnegative"):
            SignedOUDriftState(1, stationary_std=-0.1, tau_s=1.0)
        with self.assertRaisesRegex(ValueError, "positive"):
            SignedOUDriftState(1, stationary_std=1.0, tau_s=0.0)
        with self.assertRaisesRegex(ValueError, "finite"):
            SignedOUDriftState(1, stationary_std=1.0, tau_s=np.inf)
        with self.assertRaisesRegex(ValueError, "finite"):
            SignedOUDriftState(
                1,
                stationary_std=1.0,
                tau_s=1.0,
                initial_time=np.nan,
            )


if __name__ == "__main__":
    unittest.main()
