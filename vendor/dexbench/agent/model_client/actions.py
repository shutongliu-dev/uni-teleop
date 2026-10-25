"""Translate one model decision using only public proprioception/capabilities.

The delta limits bound target changes relative to observed joint positions. They
are not measured velocity/acceleration guarantees and do not make contact safe.
There are no task-specific trajectories, inverse kinematics, or object lookups.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
import re
from typing import Any

from bench.protocol.schemas import (AgentObservation, DirectJointAction, EpisodeSpec,
                                 MEASURED_JOINT_LIMIT_SAFETY_BAND_RAD, RobotCapabilities)
from bench.protocol.validation import validate_action


class ActionValidationError(ValueError):
    """A model decision cannot be translated without changing its intent."""

    def public_metadata(self):
        # Only this deliberately public validation type is eligible. Never call
        # this on transport/backend exceptions, which may contain private bodies.
        return {'code': 'action_validation_error', 'message': str(self)[:1024]}


# Contact can push a measured joint past its URDF bound. Inside the service
# safety band, the delta base of that joint is the measurement projected onto the
# bound; requested targets stay strictly inside the bounds and the public
# measurement itself is unchanged. Beyond the band the episode is a safety stop.
MEASURED_LIMIT_PROJECTION_BAND_RAD = MEASURED_JOINT_LIMIT_SAFETY_BAND_RAD


def _number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ActionValidationError(f"{label} must be a finite number, not a boolean")
    try:
        converted = float(value)
    except (ValueError, OverflowError):
        raise ActionValidationError(f"{label} must be a finite number, not a boolean") from None
    if not math.isfinite(converted):
        raise ActionValidationError(f"{label} must be a finite number, not a boolean")
    return converted


@dataclass(frozen=True)
class ActionLimits:
    min_duration_s: float = 0.2
    max_duration_s: float = 0.5
    max_hold_duration_s: float = 5.0
    max_arm_delta_rad: float = 0.05
    max_hand_delta_rad: float = 0.10
    # A delta may not push a commanded target farther than this from the
    # measured joint; contact or load is then resisting that joint.
    max_arm_target_gap_rad: float = 0.10
    max_hand_target_gap_rad: float = 0.40

    def __post_init__(self):
        for name in self.__dataclass_fields__:
            if _number(getattr(self, name), name) <= 0:
                raise ActionValidationError(f"{name} must be positive")
        if self.min_duration_s > self.max_duration_s:
            raise ActionValidationError("duration limits must be ordered")


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ActionValidationError("duplicate JSON keys are forbidden")
        result[key] = value
    return result


def _invalid_constant(_):
    raise ActionValidationError("non-finite JSON numbers are forbidden")


def parse_json_object(value: str | dict) -> dict:
    """Require one JSON object; a single outer JSON code fence is accepted."""
    if isinstance(value, str):
        if len(value) > 32768:
            raise ActionValidationError("model action is too large")
        value = value.strip()
        if value.startswith("```"):
            fence = re.fullmatch(r"```(?:json)?\s*\n([\s\S]*?)\n```", value)
            if fence is None:
                raise ActionValidationError("only one outer JSON code fence is allowed")
            value = fence.group(1)
        try:
            value = json.loads(value, object_pairs_hook=_unique_object,
                               parse_constant=_invalid_constant)
        except (ValueError, RecursionError):
            raise ActionValidationError("model action must be one strict JSON object") from None
    if not isinstance(value, dict):
        raise ActionValidationError("model action must be a JSON object")
    return value


def parse_model_action(value: str | dict) -> dict:
    value = parse_json_object(value)
    kind = value.get("action")
    if not isinstance(kind, str) or kind not in ("joint_delta", "hold", "stop"):
        raise ActionValidationError("action must be joint_delta, hold, or stop")
    allowed = {"action", "reason"}
    if kind != "stop":
        allowed.add("duration_s")
    if kind == "joint_delta":
        allowed.add("delta_rad")
    if set(value) - allowed:
        raise ActionValidationError("model action contains unadvertised fields")
    reason = value.get("reason", "")
    if not isinstance(reason, str) or len(reason) > 512:
        raise ActionValidationError("reason must be a public explanation of at most 512 characters")
    result = {"action": kind, "reason": reason}
    if kind != "stop":
        duration = _number(value.get("duration_s"), "duration_s")
        if duration <= 0:
            raise ActionValidationError("duration_s must be positive")
        result["duration_s"] = duration
    if kind == "joint_delta":
        deltas = value.get("delta_rad")
        if not isinstance(deltas, dict) or not deltas or len(deltas) > 256:
            raise ActionValidationError("delta_rad must contain between 1 and 256 named joints")
        if any(not isinstance(name, str) or not name for name in deltas):
            raise ActionValidationError("delta_rad keys must be joint names")
        result["delta_rad"] = {name: _number(delta, "joint delta") for name, delta in deltas.items()}
    return result


def public_observation(value: dict) -> AgentObservation:
    """Fail closed if a purported public observation carries private fields."""
    if not isinstance(value, dict) or any(not isinstance(name, str) or name.startswith("privileged_") for name in value):
        raise ActionValidationError("only public vision_proprio observations are allowed")
    try:
        result = AgentObservation.model_validate(value)
    except (ValueError, TypeError):
        raise ActionValidationError("invalid public observation contract") from None
    if result.profile != "vision_proprio":
        raise ActionValidationError("only public vision_proprio observations are allowed")
    # Pydantic float fields reject bool today; keep the boundary explicit.
    for row in (value["proprioception"]["joint_positions_rad"],
                value["proprioception"]["joint_velocities_rad_s"]):
        for number in row.values():
            _number(number, "proprioception")
    _number(value["sim_time_s"], "sim_time_s")
    return result


def build_joint_action(model_action: str | dict, observation: dict, capabilities: dict,
                       episode_spec: dict, *, action_id: str, expected_sequence: int,
                       actions_used: int = 0, limits: ActionLimits | None = None) -> dict | None:
    """Return a complete direct-joint command; stop is represented by None.

    ``expected_sequence`` is the revision seen by the model. Pass the latest
    public observation as ``observation`` so a stale decision is rejected. The
    server independently checks this revision again when accepting the command.
    ``actions_used`` must include all submitted attempts, not just successes.
    """
    decision = parse_model_action(model_action)
    obs = public_observation(observation)
    limits = limits or ActionLimits()
    if isinstance(expected_sequence, bool) or not isinstance(expected_sequence, int) or expected_sequence < 0:
        raise ActionValidationError("expected_sequence must be a nonnegative integer")
    if expected_sequence != obs.sequence:
        raise ActionValidationError("model decision uses a stale observation")
    if isinstance(actions_used, bool) or not isinstance(actions_used, int) or actions_used < 0:
        raise ActionValidationError("actions_used must be a nonnegative integer")
    try:
        caps = RobotCapabilities.model_validate(capabilities)
        spec = EpisodeSpec.model_validate(episode_spec)
    except (ValueError, TypeError):
        raise ActionValidationError("invalid advertised capabilities or episode specification") from None
    if (spec.episode_id != obs.episode_id or spec.track not in ("direct_joints", "free_tool_agent")
            or spec.observation_profile != "vision_proprio"):
        raise ActionValidationError("episode does not match the public direct-joint observation")
    if obs.last_action_status in ("accepted", "running"):
        raise ActionValidationError("previous action has not reached a decision boundary")
    positions = obs.proprioception.joint_positions_rad
    if set(positions) != set(caps.joint_limits):
        raise ActionValidationError("observation must contain all and only advertised joints")
    if decision["action"] == "stop":
        return None
    duration = decision["duration_s"]
    minimum = caps.control_period_s if decision['action'] == 'hold' else limits.min_duration_s
    maximum = limits.max_hold_duration_s if decision['action'] == 'hold' else limits.max_duration_s
    if not minimum <= duration <= maximum:
        raise ActionValidationError(f"duration_s must be between {minimum:g} and {maximum:g} seconds")
    if actions_used >= spec.budgets.max_actions:
        raise ActionValidationError("episode action budget is exhausted")
    if (spec.budgets.sim_time_s is not None
            and obs.sim_time_s + duration > spec.budgets.sim_time_s + 1e-9):
        raise ActionValidationError("action exceeds remaining episode simulation time")
    deltas = decision.get("delta_rad", {})
    if not set(deltas) <= set(caps.joint_limits):
        raise ActionValidationError("delta_rad contains an unknown joint")
    arm = set(caps.arm_joints)
    for name, delta in deltas.items():
        bound = limits.max_arm_delta_rad if name in arm else limits.max_hand_delta_rad
        if abs(delta) > bound:
            raise ActionValidationError(f"joint target delta exceeds the configured limit: arm <= {limits.max_arm_delta_rad:g} rad; hand <= {limits.max_hand_delta_rad:g} rad")
    # A position-controlled joint keeps its last commanded target: omitted joints
    # hold it and deltas move it. Re-anchoring at the measurement would release
    # a squeeze and let gravity sag ratchet the arm at every action boundary.
    commanded = obs.proprioception.commanded_joint_positions_rad
    targets = {}
    for name, position in positions.items():
        bound = caps.joint_limits[name]
        if max(bound.lower_rad - position, position - bound.upper_rad) > MEASURED_LIMIT_PROJECTION_BAND_RAD:
            raise ActionValidationError(
                f"joint {name} is measured beyond the {MEASURED_LIMIT_PROJECTION_BAND_RAD:g} rad "
                "joint-limit safety band; the robot state is invalid")
        reference = position if commanded is None else commanded[name]
        base = max(bound.lower_rad, min(bound.upper_rad, reference))
        targets[name] = base + deltas.get(name, 0.)
        if commanded is not None and name in deltas:
            gap = limits.max_arm_target_gap_rad if name in arm else limits.max_hand_target_gap_rad
            if abs(targets[name] - position) > max(gap, abs(base - position)):
                raise ActionValidationError(
                    f"joint {name} target would be {abs(targets[name] - position):.3f} rad from its measured "
                    f"position (limit {gap:g} rad); contact or load is resisting this joint")
    try:
        command = DirectJointAction(action_id=action_id, observation_sequence=obs.sequence,
            duration_s=duration, arm_joint_positions_rad={n: targets[n] for n in caps.arm_joints},
            hand_joint_positions_rad={n: targets[n] for n in caps.hand_joints})
        validate_action(command, caps, spec, observation_sequence=obs.sequence)
    except (ValueError, TypeError):
        raise ActionValidationError("joint command violates advertised limits, timing, or episode identity") from None
    return command.model_dump(mode="json")
