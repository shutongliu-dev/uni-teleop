"""Backend-independent monotonic budgets and action lifecycle.

The native adapter owns stepping/cancellation locks. This module never mutates
physics. A wall budget is checked before AND after a policy/network call or step.
"""
from __future__ import annotations

import math
import time
from collections.abc import Callable

from .schemas import ActionEvent, Budgets


class BudgetExceeded(RuntimeError):
    def __init__(self, outcome: str):
        self.outcome = outcome
        super().__init__(outcome)


class EpisodeClock:
    def __init__(self, budgets: Budgets, *, monotonic: Callable[[], float] = time.monotonic):
        self.budgets = budgets.model_copy(deep=True)
        self._monotonic = monotonic
        self._start = monotonic()
        self._stop: float | None = None
        self.sim_time_s = 0.0
        self.action_attempts = 0
        self.model_calls = 0
        self.model_latency_s = 0.0
        self._model_start: float | None = None

    @property
    def wall_time_s(self) -> float:
        return (self._stop if self._stop is not None else self._monotonic()) - self._start

    @property
    def remaining_wall_s(self) -> float | None:
        horizon = self.budgets.wall_time_s
        return None if horizon is None else max(0., horizon - self.wall_time_s)

    @property
    def wall_expired(self) -> bool:
        return self.remaining_wall_s == 0.

    @property
    def sim_expired(self) -> bool:
        horizon = self.budgets.sim_time_s
        return horizon is not None and self.sim_time_s >= horizon - 1e-9

    def check(self, *, next_sim_dt: float = 0.0) -> None:
        if not math.isfinite(next_sim_dt) or next_sim_dt < 0:
            raise ValueError("next simulation interval must be finite/nonnegative")
        if self._stop is not None:
            raise RuntimeError("episode clock is stopped")
        if self.wall_expired:
            raise BudgetExceeded("wall_timeout")
        # A final step landing exactly on the simulation horizon may be scored.
        if self.budgets.sim_time_s is not None and ((next_sim_dt == 0 and self.sim_expired)
                or self.sim_time_s + next_sim_dt > self.budgets.sim_time_s + 1e-9):
            raise BudgetExceeded("sim_timeout")

    def record_step(self, native_sim_time_s: float) -> None:
        if self._stop is not None:
            raise RuntimeError("episode clock is stopped")
        if not math.isfinite(native_sim_time_s) or native_sim_time_s <= self.sim_time_s:
            raise ValueError("native simulation time must increase monotonically")
        self.sim_time_s = native_sim_time_s
        if self.budgets.sim_time_s is not None and native_sim_time_s > self.budgets.sim_time_s + 1e-9:
            # Preserve the actual overshoot as evidence. Reporting the previous
            # step would hide a native clock/budget violation in the failure log.
            raise BudgetExceeded("sim_timeout")

    def action_attempt(self) -> None:
        self.check()
        if self.action_attempts >= self.budgets.max_actions:
            raise BudgetExceeded("action_budget")
        self.action_attempts += 1  # malformed, rejected and cancelled requests count

    def begin_model_call(self) -> None:
        self.check()
        if self._model_start is not None:
            raise RuntimeError("overlapping model calls require separate aggregation; v1 is sequential")
        if self.model_calls >= self.budgets.max_model_calls:
            raise BudgetExceeded("model_budget")
        self.model_calls += 1
        self._model_start = self._monotonic()

    def end_model_call(self) -> None:
        if self._model_start is None:
            raise RuntimeError("no model call is active")
        self.model_latency_s += self._monotonic() - self._model_start
        self._model_start = None
        self.check()

    def stop(self) -> None:
        if self._stop is None:
            now = self._monotonic()
            if self._model_start is not None:
                self.model_latency_s += now - self._model_start
                self._model_start = None
            self._stop = now

    def snapshot(self) -> dict:
        latency = self.model_latency_s
        if self._model_start is not None:
            latency += self._monotonic() - self._model_start
        return dict(elapsed_sim_s=self.sim_time_s, elapsed_wall_s=self.wall_time_s,
                    action_attempts=self.action_attempts, model_calls=self.model_calls,
                    model_latency_s=latency)


class ActionLifecycle:
    """One active action, unique IDs, explicit transition history.

    Cancellation becomes effective at the next physics control boundary. The
    paused native adapter retains its last acknowledged drive target; it does
    not write a new measured-position target or rewind object/cloth state.
    """
    _allowed = {"accepted": {"running", "cancelled", "failed"},
                "running": {"completed", "cancelled", "failed"}}

    def __init__(self):
        self.events: list[ActionEvent] = []
        self.statuses: dict[str, str] = {}
        self.active_action_id: str | None = None

    def accept(self, action_id: str, *, sim_time_s: float, wall_time_s: float) -> ActionEvent:
        if self.active_action_id is not None:
            raise ValueError("cancel or complete the active action before submitting another")
        if action_id in self.statuses:
            raise ValueError("duplicate action_id; retries must read existing status")
        if self.events and (sim_time_s < self.events[-1].sim_time_s or wall_time_s < self.events[-1].wall_time_s):
            raise ValueError("action event clocks must be monotonic")
        event = ActionEvent(action_id=action_id, status="accepted", sim_time_s=sim_time_s, wall_time_s=wall_time_s)
        self.active_action_id = action_id
        self.statuses[action_id] = "accepted"
        self.events.append(event)
        return event

    def transition(self, action_id: str, status: str, *, sim_time_s: float,
                   wall_time_s: float, detail: str = "") -> ActionEvent:
        current = self.statuses.get(action_id)
        if status not in self._allowed.get(current, set()):
            raise ValueError(f"invalid action transition {current!r} -> {status!r}")
        previous = self.events[-1]
        if sim_time_s < previous.sim_time_s or wall_time_s < previous.wall_time_s:
            raise ValueError("action event clocks must be monotonic")
        event = ActionEvent(action_id=action_id, status=status, sim_time_s=sim_time_s,
                            wall_time_s=wall_time_s, detail=detail)
        self.statuses[action_id] = status
        self.events.append(event)
        if status in ("completed", "cancelled", "failed"):
            self.active_action_id = None
        return event
