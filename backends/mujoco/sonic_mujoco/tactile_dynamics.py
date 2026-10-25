"""Small stochastic dynamics used by hardware-shaped tactile sensors.

The classes in this module deliberately own independent random generators.
Changing a noise model therefore cannot perturb the delivery clock, and each
component can be reproduced from its seed in isolation.
"""

from __future__ import annotations

import math
from collections.abc import Iterator, Sequence
from numbers import Integral

import numpy as np

_UINT64_MAX = int(np.iinfo(np.uint64).max)
_PROBABILITY_TOLERANCE = 1e-12


class HoldGapMarkovClock:
    """Schedule observation deliveries on a finite-state hold-gap clock.

    State ``i`` means that the next packet is delivered after ``i + 1``
    record periods. The initial state schedules the first delivery after a
    reset; each delivery then draws its successor from the corresponding row
    of ``transition_probability``.

    Delivery times are represented by integer ticks relative to the reset
    origin. This avoids cumulative floating-point drift and makes a seeded
    clock invariant to how calls to :meth:`events_until` are chunked. The
    returned iterator is lazy, so a long simulation gap has no arbitrary
    event cap or mandatory event-list allocation.
    """

    def __init__(
        self,
        initial_probability: Sequence[float] | np.ndarray,
        transition_probability: Sequence[Sequence[float]] | np.ndarray,
        *,
        record_period_s: float = 0.02,
        seed: int = 0,
        initial_time: float = 0.0,
    ) -> None:
        initial = _probability_vector(initial_probability, "initial probability")
        transition = _transition_matrix(
            transition_probability,
            len(initial),
        )
        self.initial_probability = initial
        self.transition_probability = transition
        self.record_period_s = _positive_float(record_period_s, "record period")
        self.seed = _seed(seed, "hold-gap clock seed")
        self._rng = np.random.default_rng(self.seed)
        self._origin_time = 0.0
        self._last_query_time = 0.0
        self._state = 0
        self._next_tick = 0
        self._event_index = 0
        self.reset(time=initial_time)

    @property
    def state_index(self) -> int:
        """State whose gap ends at :attr:`next_event_time`."""

        return self._state

    @property
    def gap_frames(self) -> int:
        return self._state + 1

    @property
    def event_index(self) -> int:
        return self._event_index

    @property
    def next_event_time(self) -> float:
        return self._time_at_tick(self._next_tick)

    def reset(self, time: float = 0.0, *, seed: int | None = None) -> None:
        """Reset the clock and draw a new initial gap.

        Omitting ``seed`` replays the constructor seed. Passing an explicit
        seed reproduces that episode without changing the constructor seed.
        """

        origin = _finite_float(time, "hold-gap reset time")
        active_seed = self.seed if seed is None else _seed(seed, "hold-gap reset seed")
        self._rng = np.random.default_rng(active_seed)
        self._origin_time = origin
        self._last_query_time = origin
        self._state = self._draw_state(self.initial_probability)
        self._next_tick = self._state + 1
        self._event_index = 0

    def events_until(self, time: float) -> Iterator[float]:
        """Yield and consume every delivery at or before ``time``.

        Calls must use nondecreasing simulation times. Exact-boundary events
        are included. The iterator should normally be exhausted; if a caller
        stops early, the unconsumed backlog remains scheduled for a later
        nondecreasing query.
        """

        horizon = _finite_float(time, "hold-gap query time")
        if _strictly_before(horizon, self._last_query_time):
            raise ValueError("hold-gap query time must be nondecreasing")
        self._last_query_time = max(horizon, self._last_query_time)

        while _at_or_before(self.next_event_time, horizon):
            event_time = self.next_event_time
            probabilities = self.transition_probability[self._state]
            self._state = self._draw_state(probabilities)
            self._next_tick += self._state + 1
            self._event_index += 1
            yield event_time

    def _draw_state(self, probability: np.ndarray) -> int:
        possible = np.flatnonzero(probability > 0.0)
        if len(possible) == 1:
            return int(possible[0])
        return int(self._rng.choice(len(probability), p=probability))

    def _time_at_tick(self, tick: int) -> float:
        return self._origin_time + tick * self.record_period_s


class AR1NoiseState:
    """Source-sample AR(1) noise parameterized by marginal deviation.

    The first returned sample is drawn from the stationary marginal. Later
    samples use ``x[t] = rho*x[t-1] + sigma*sqrt(1-rho**2)*epsilon``. Noise is
    advanced only when :meth:`sample` is called, so record-rate reads between
    source samples remain exact holds.
    """

    def __init__(
        self,
        shape: int | tuple[int, ...],
        *,
        marginal_std: float | Sequence[float] | np.ndarray,
        coefficient: float | Sequence[float] | np.ndarray = 0.0,
        seed: int = 0,
    ) -> None:
        self.shape = _shape(shape)
        self.marginal_std = _broadcast_parameter(
            marginal_std,
            self.shape,
            "AR(1) marginal standard deviation",
        )
        if np.any(self.marginal_std < 0.0):
            raise ValueError("AR(1) marginal standard deviation must be nonnegative")
        self.coefficient = _broadcast_parameter(
            coefficient,
            self.shape,
            "AR(1) coefficient",
        )
        if np.any(np.abs(self.coefficient) >= 1.0):
            raise ValueError("AR(1) coefficient magnitude must be less than one")
        innovation = self.marginal_std * np.sqrt(1.0 - self.coefficient**2)
        innovation.setflags(write=False)
        self._innovation_std = innovation
        self.seed = _seed(seed, "AR(1) seed")
        self._rng = np.random.default_rng(self.seed)
        self._state = np.zeros(self.shape, dtype=np.float64)
        self._initialized = False
        self.sample_index = 0

    @property
    def state(self) -> np.ndarray:
        return self._state.copy()

    def reset(self, *, seed: int | None = None) -> None:
        active_seed = self.seed if seed is None else _seed(seed, "AR(1) reset seed")
        self._rng = np.random.default_rng(active_seed)
        self._state.fill(0.0)
        self._initialized = False
        self.sample_index = 0

    def sample(self) -> np.ndarray:
        """Advance one source sample and return a defensive state copy."""

        if not np.any(self.marginal_std):
            self._initialized = True
            self.sample_index += 1
            return self.state

        standard_normal = self._rng.standard_normal(self.shape)
        if self._initialized:
            self._state = (
                self.coefficient * self._state + self._innovation_std * standard_normal
            )
        else:
            self._state = self.marginal_std * standard_normal
            self._initialized = True
        self.sample_index += 1
        return self.state


class SignedOUDriftState:
    """Zero-mean signed Ornstein-Uhlenbeck drift with exact discretization.

    ``stationary_std`` is the long-run standard deviation and ``tau_s`` is the
    correlation time. Reset represents a freshly zeroed sensor, so the drift
    starts exactly at zero and approaches its stationary distribution. Calls
    use absolute, nondecreasing simulation time.
    """

    def __init__(
        self,
        shape: int | tuple[int, ...],
        *,
        stationary_std: float | Sequence[float] | np.ndarray,
        tau_s: float | Sequence[float] | np.ndarray,
        seed: int = 0,
        initial_time: float = 0.0,
    ) -> None:
        self.shape = _shape(shape)
        self.stationary_std = _broadcast_parameter(
            stationary_std,
            self.shape,
            "OU stationary standard deviation",
        )
        if np.any(self.stationary_std < 0.0):
            raise ValueError("OU stationary standard deviation must be nonnegative")
        self.tau_s = _broadcast_parameter(tau_s, self.shape, "OU time constant")
        if np.any(self.tau_s <= 0.0):
            raise ValueError("OU time constant must be positive")
        self.seed = _seed(seed, "OU seed")
        self._rng = np.random.default_rng(self.seed)
        self._state = np.zeros(self.shape, dtype=np.float64)
        self._time = 0.0
        self.advance_index = 0
        self.reset(time=initial_time)

    @property
    def state(self) -> np.ndarray:
        return self._state.copy()

    @property
    def time(self) -> float:
        return self._time

    def reset(self, time: float = 0.0, *, seed: int | None = None) -> None:
        active_seed = self.seed if seed is None else _seed(seed, "OU reset seed")
        self._rng = np.random.default_rng(active_seed)
        self._state.fill(0.0)
        self._time = _finite_float(time, "OU reset time")
        self.advance_index = 0

    def advance_to(self, time: float) -> np.ndarray:
        """Advance to an absolute time using the exact OU transition."""

        target_time = _finite_float(time, "OU target time")
        if _strictly_before(target_time, self._time):
            raise ValueError("OU target time must be nondecreasing")
        duration = max(target_time - self._time, 0.0)
        self._time = max(target_time, self._time)
        if duration == 0.0:
            return self.state

        self.advance_index += 1
        if not np.any(self.stationary_std):
            return self.state

        decay = np.exp(-duration / self.tau_s)
        innovation_std = self.stationary_std * np.sqrt(
            -np.expm1(-2.0 * duration / self.tau_s)
        )
        self._state = decay * self._state + innovation_std * self._rng.standard_normal(
            self.shape
        )
        return self.state


def _probability_vector(value: object, name: str) -> np.ndarray:
    probability = _numeric_array(value, name)
    if probability.ndim != 1 or len(probability) == 0:
        raise ValueError(f"{name} must be a nonempty vector")
    if np.any(probability < 0.0):
        raise ValueError(f"{name} must be nonnegative")
    _require_unit_sum(float(probability.sum()), name)
    probability = probability / probability.sum()
    probability.setflags(write=False)
    return probability


def _transition_matrix(value: object, state_count: int) -> np.ndarray:
    transition = _numeric_array(value, "transition probability")
    if transition.shape != (state_count, state_count):
        raise ValueError(
            f"transition probability must have shape ({state_count}, {state_count})"
        )
    if np.any(transition < 0.0):
        raise ValueError("transition probability must be nonnegative")
    row_sum = transition.sum(axis=1)
    for index, total in enumerate(row_sum):
        _require_unit_sum(float(total), f"transition probability row {index}")
    transition = transition / row_sum[:, None]
    transition.setflags(write=False)
    return transition


def _numeric_array(value: object, name: str) -> np.ndarray:
    raw = np.asarray(value)
    if raw.dtype.kind not in "iuf":
        raise TypeError(f"{name} must contain real numbers")
    result = np.asarray(raw, dtype=np.float64)
    if np.any(~np.isfinite(result)):
        raise ValueError(f"{name} must be finite")
    return result.copy()


def _require_unit_sum(total: float, name: str) -> None:
    if not math.isclose(
        total,
        1.0,
        rel_tol=_PROBABILITY_TOLERANCE,
        abs_tol=_PROBABILITY_TOLERANCE,
    ):
        raise ValueError(f"{name} must sum to one")


def _shape(value: int | tuple[int, ...]) -> tuple[int, ...]:
    if isinstance(value, Integral) and not isinstance(value, (bool, np.bool_)):
        result = (int(value),)
    elif isinstance(value, tuple) and all(
        isinstance(item, Integral) and not isinstance(item, (bool, np.bool_))
        for item in value
    ):
        result = tuple(int(item) for item in value)
    else:
        raise TypeError("state shape must be an integer or tuple of integers")
    if not result or any(item <= 0 for item in result):
        raise ValueError("state shape dimensions must be positive")
    return result


def _broadcast_parameter(
    value: object,
    shape: tuple[int, ...],
    name: str,
) -> np.ndarray:
    raw = _numeric_array(value, name)
    try:
        result = np.broadcast_to(raw, shape).astype(np.float64, copy=True)
    except ValueError as exc:
        raise ValueError(f"{name} is not broadcastable to state shape {shape}") from exc
    result.setflags(write=False)
    return result


def _positive_float(value: object, name: str) -> float:
    result = _finite_float(value, name)
    if result <= 0.0:
        raise ValueError(f"{name} must be positive")
    return result


def _finite_float(value: object, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.number)
    ):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _seed(value: object, name: str) -> int:
    if not isinstance(value, Integral) or isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if not 0 <= result <= _UINT64_MAX:
        raise ValueError(f"{name} must be in uint64 range")
    return result


def _time_tolerance(first: float, second: float) -> float:
    return 16.0 * np.finfo(np.float64).eps * max(1.0, abs(first), abs(second))


def _strictly_before(first: float, second: float) -> bool:
    return first < second - _time_tolerance(first, second)


def _at_or_before(first: float, second: float) -> bool:
    return first <= second + _time_tolerance(first, second)
