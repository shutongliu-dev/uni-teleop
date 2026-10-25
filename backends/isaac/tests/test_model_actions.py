"""Public decision conversion contracts; these are not physics acceptance tests."""
from copy import deepcopy

import pytest

from agent.model_client.actions import (ActionLimits, ActionValidationError,
    build_joint_action, parse_model_action)


@pytest.fixture
def public_inputs():
    caps = dict(robot_id="test_robot", robot_asset_sha256="a" * 64,
        arm_joints=["arm_1", "arm_2"], hand_joints=["thumb", "index"],
        joint_limits={n: {"lower_rad": -1.0, "upper_rad": 1.0}
                      for n in ("arm_1", "arm_2", "thumb", "index")},
        control_period_s=0.02, tracks=["direct_joints"], observation_profiles=["vision_proprio"])
    spec = dict(episode_id="ep1", agent_id="model", agent_version="v1",
        agent_config_sha256="b" * 64, task_id="task", task_release_sha256="c" * 64,
        evaluator_sha256="d" * 64, robot_id="test_robot", robot_asset_sha256="a" * 64,
        physics_profile_sha256="e" * 64, seed=0, track="direct_joints",
        observation_profile="vision_proprio", instruction="Move the object.",
        budgets={"sim_time_s": 10.0, "wall_time_s": 60.0, "max_actions": 5,
                 "max_model_calls": 6, "max_action_sim_s": 0.5})
    positions = {"arm_1": 0.1, "arm_2": -0.4, "thumb": 0.2, "index": 0.8}
    obs = dict(episode_id="ep1", sequence=2, sim_time_s=0.4, instruction="Move the object.",
        profile="vision_proprio", proprioception={"joint_positions_rad": positions,
            "joint_velocities_rad_s": dict.fromkeys(positions, 0.0)}, cameras=[],
        last_action_id="previous", last_action_status="completed")
    return obs, caps, spec


def build(inputs, decision=None, **kwargs):
    if decision is None:
        decision = {"action": "joint_delta", "delta_rad": {"arm_1": 0.03, "thumb": -0.1},
                    "duration_s": 0.2, "reason": "Move a small amount, then observe."}
    return build_joint_action(decision, *inputs, action_id="step_3",
                              expected_sequence=kwargs.pop("expected_sequence", 2), **kwargs)


def test_sparse_deltas_expand_to_complete_named_commands_without_mutation(public_inputs):
    before = deepcopy(public_inputs)
    result = build(public_inputs)
    assert result == dict(kind="direct_joints", action_id="step_3", observation_sequence=2,
        duration_s=0.2, arm_joint_positions_rad={"arm_1": 0.13, "arm_2": -0.4},
        hand_joint_positions_rad={"thumb": 0.1, "index": 0.8})
    assert public_inputs == before


def test_hold_preserves_all_observed_positions_and_stop_submits_nothing(public_inputs):
    result = build(public_inputs, {"action": "hold", "duration_s": 0.4})
    assert result["arm_joint_positions_rad"] | result["hand_joint_positions_rad"] == public_inputs[0]["proprioception"]["joint_positions_rad"]
    assert build(public_inputs, {"action": "stop", "reason": "Cannot identify a reliable next move."}) is None


@pytest.mark.parametrize("value", [
    "hello {\"action\":\"stop\"}", '{"action":"stop"} after',
    '{"action":"hold","duration_s":0.2,"duration_s":0.4}',
    '{"action":"joint_delta","duration_s":0.2,"delta_rad":{"arm_1":0.01,"arm_1":0.02}}',
    '{"action":"hold","duration_s":NaN}', '{"action":"hold","duration_s":Infinity}',
    "[]", "null", "true", {"action": "grasp"}, {"action": []},
    {"action": "stop", "duration_s": 0.2}, {"action": "hold", "duration_s": True},
    {"action": "hold", "duration_s": "0.2"}, {"action": "hold", "duration_s": float("inf")},
    {"action": "hold", "duration_s": 10 ** 400},
    {"action": "joint_delta", "delta_rad": {}, "duration_s": 0.2},
    {"action": "joint_delta", "delta_rad": {"arm_1": True}, "duration_s": 0.2},
    {"action": "joint_delta", "delta_rad": {"arm_1": float("nan")}, "duration_s": 0.2},
    {"action": "hold", "duration_s": 0.2, "object_pose": [0, 0, 0]},
    {"action": "hold", "duration_s": 0.2, "reason": {}},
    '```json\n{"action":"stop"}\n```\nExplanation',
    'Here is JSON:\n```json\n{"action":"stop"}\n```',
    '```json\n{"action":"stop"}\n```\n```json\n{"action":"stop"}\n```',
])
def test_parser_rejects_ambiguous_hidden_and_nonfinite_model_output(value):
    with pytest.raises(ActionValidationError):
        parse_model_action(value)


@pytest.mark.parametrize("fence", ["", "json"])
def test_single_outer_fence_is_accepted(fence):
    assert parse_model_action(f'```{fence}\n{{"action":"stop"}}\n```')["action"] == "stop"


@pytest.mark.parametrize("joint,delta", [("arm_1", 0.050001), ("arm_2", -0.051),
                                         ("thumb", 0.100001), ("index", -0.11), ("unknown", 0.01)])
def test_unknown_or_excess_delta_is_rejected_without_clamping(public_inputs, joint, delta):
    with pytest.raises(ActionValidationError):
        build(public_inputs, {"action": "joint_delta", "delta_rad": {joint: delta}, "duration_s": 0.2})


def test_target_limit_violation_is_rejected_not_clamped(public_inputs):
    public_inputs[0]["proprioception"]["joint_positions_rad"]["arm_1"] = 0.99
    with pytest.raises(ActionValidationError):
        build(public_inputs)


@pytest.mark.parametrize("joint,position,bound", [
    ("index", 1.000000143, 1.0), ("arm_2", -1.0000000256, -1.0),
])
def test_solver_roundoff_on_omitted_joint_does_not_reject_other_joint_motion(
        public_inputs, joint, position, bound):
    """Regression for the seven archived direct-joint hold-reference failures."""
    public_inputs[0]["proprioception"]["joint_positions_rad"][joint] = position
    before = deepcopy(public_inputs)
    result = build(public_inputs)
    targets = result["arm_joint_positions_rad"] | result["hand_joint_positions_rad"]
    assert targets[joint] == bound
    assert targets["arm_1"] == 0.13
    assert public_inputs == before


@pytest.mark.parametrize("measured", [1.0001, 1.0018969556549564, 1.0299])
def test_contact_excursion_inside_safety_band_is_projected_not_deadlocked(public_inputs, measured):
    public_inputs[0]["proprioception"]["joint_positions_rad"]["index"] = measured
    result = build(public_inputs)
    assert result["hand_joint_positions_rad"]["index"] == 1.0
    held = build(public_inputs, {"action": "joint_delta", "delta_rad": {"index": -0.05}, "duration_s": 0.2})
    assert held["hand_joint_positions_rad"]["index"] == pytest.approx(0.95)


def test_projection_does_not_hide_a_measured_state_beyond_the_safety_band(public_inputs):
    public_inputs[0]["proprioception"]["joint_positions_rad"]["index"] = 1.0301
    with pytest.raises(ActionValidationError, match="safety band"):
        build(public_inputs)


def commanded(public_inputs, values):
    public_inputs[0]["proprioception"]["commanded_joint_positions_rad"] = values
    return public_inputs


def test_omitted_joints_and_hold_keep_commanded_targets_not_sagged_measurements(public_inputs):
    # Measured arm_2 sags under load and the thumb is stopped by contact.
    commanded(public_inputs, {"arm_1": 0.1, "arm_2": -0.385, "thumb": 0.35, "index": 0.8})
    result = build(public_inputs)
    assert result["arm_joint_positions_rad"] == {"arm_1": 0.13, "arm_2": -0.385}
    assert result["hand_joint_positions_rad"] == {"thumb": pytest.approx(0.25), "index": 0.8}
    held = build(public_inputs, {"action": "hold", "duration_s": 0.4})
    assert held["arm_joint_positions_rad"] | held["hand_joint_positions_rad"] == {
        "arm_1": 0.1, "arm_2": -0.385, "thumb": 0.35, "index": 0.8}


def test_commanded_float32_roundoff_at_a_bound_stays_legal(public_inputs):
    commanded(public_inputs, {"arm_1": 0.1, "arm_2": -0.4, "thumb": 0.2, "index": 1.0000000149})
    assert build(public_inputs)["hand_joint_positions_rad"]["index"] == 1.0


def test_delta_cannot_wind_a_blocked_joint_beyond_the_target_gap(public_inputs):
    # Thumb measured 0.2 but commanded 0.55: an object resists it by 0.35 rad.
    commanded(public_inputs, {"arm_1": 0.1, "arm_2": -0.4, "thumb": 0.55, "index": 0.8})
    with pytest.raises(ActionValidationError, match="contact or load is resisting"):
        build(public_inputs, {"action": "joint_delta", "delta_rad": {"thumb": 0.1}, "duration_s": 0.2})
    relaxed = build(public_inputs, {"action": "joint_delta", "delta_rad": {"thumb": -0.1}, "duration_s": 0.2})
    assert relaxed["hand_joint_positions_rad"]["thumb"] == pytest.approx(0.45)
    # An existing larger gap is kept for omitted joints; only further windup is refused.
    commanded(public_inputs, {"arm_1": 0.1, "arm_2": -0.4, "thumb": 0.65, "index": 0.8})
    assert build(public_inputs, {"action": "hold", "duration_s": 0.4})["hand_joint_positions_rad"]["thumb"] == 0.65


def test_explicit_tiny_out_of_bounds_target_still_rejected(public_inputs):
    public_inputs[0]["proprioception"]["joint_positions_rad"]["index"] = 1.0
    with pytest.raises(ActionValidationError):
        build(public_inputs, {"action": "joint_delta", "delta_rad": {"index": 1e-7},
                              "duration_s": 0.2})


def test_disabled_total_time_budgets_keep_direct_joint_action_budget(public_inputs):
    public_inputs[2]["budgets"].update(sim_time_s=None, wall_time_s=None)
    public_inputs[0]["sim_time_s"] = 10000.0
    assert build(public_inputs)["duration_s"] == 0.2
    with pytest.raises(ActionValidationError, match="action budget"):
        build(public_inputs, actions_used=5)


@pytest.mark.parametrize("duration", [0.01, 5.01, 0.21])
def test_duration_window_and_native_period(public_inputs, duration):
    with pytest.raises(ActionValidationError):
        build(public_inputs, {"action": "hold", "duration_s": duration})


def test_custom_window_and_per_action_budget(public_inputs):
    decision = {"action": "hold", "duration_s": 0.6}
    limits = ActionLimits(min_duration_s=0.4, max_duration_s=0.8)
    with pytest.raises(ActionValidationError):
        build(public_inputs, decision, limits=limits)
    public_inputs[2]["budgets"]["max_action_sim_s"] = 0.8
    assert build(public_inputs, decision, limits=limits)["duration_s"] == 0.6


def test_remaining_time_and_action_budget(public_inputs):
    with pytest.raises(ActionValidationError):
        build(public_inputs, actions_used=5)
    public_inputs[0]["sim_time_s"] = 9.9
    with pytest.raises(ActionValidationError):
        build(public_inputs)


@pytest.mark.parametrize("mutate", [
    lambda obs: obs.update(sequence=3),
    lambda obs: obs.update(episode_id="different"),
    lambda obs: obs.update(last_action_status="running"),
    lambda obs: obs.update(last_action_status="accepted"),
    lambda obs: obs.update(privileged_object_poses=None),
    lambda obs: obs.update(evaluator={"success": True}),
    lambda obs: obs.update({7: "invalid observation key"}),
    lambda obs: obs["proprioception"]["joint_positions_rad"].pop("thumb"),
    lambda obs: obs["proprioception"]["joint_positions_rad"].update(passive_joint=0.0),
    lambda obs: obs["proprioception"]["joint_positions_rad"].update(thumb=True),
    lambda obs: obs["proprioception"]["joint_positions_rad"].update(thumb=float("nan")),
])
def test_stale_incomplete_or_private_observation_is_rejected(public_inputs, mutate):
    mutate(public_inputs[0])
    with pytest.raises(ActionValidationError):
        build(public_inputs)


def test_fresh_sequence_binds_exact_current_public_revision(public_inputs):
    public_inputs[0]["sequence"] = 3
    assert build(public_inputs, expected_sequence=3)["observation_sequence"] == 3


def test_pinned_robot_mismatch_is_rejected(public_inputs):
    public_inputs[1]["robot_asset_sha256"] = "f" * 64
    with pytest.raises(ActionValidationError):
        build(public_inputs)


@pytest.mark.parametrize("kwargs", [dict(max_arm_delta_rad=True), dict(max_hand_delta_rad=float("inf")),
                                  dict(min_duration_s=0.5, max_duration_s=0.2)])
def test_invalid_action_limit_configuration(kwargs):
    with pytest.raises(ActionValidationError):
        ActionLimits(**kwargs)
