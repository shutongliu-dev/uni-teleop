import numpy as np
import pytest
from bench.evaluator import evaluate_cloth, evaluate_rigid


def test_nonfinite_time_cannot_complete_cloth_task():
    initial = np.zeros((4, 3))
    target = initial + [.1, 0, 0]
    vertices = np.stack((initial, target, target))
    with pytest.raises(ValueError, match="Contact/time"):
        evaluate_cloth([0, .02, np.inf], vertices, [False, True, True],
                       initial_vertices=initial, target_vertices=target, task="cloth_drag")


def test_reset_only_contact_cannot_complete_gravity_drop():
    times = np.arange(0, 1.02, .02)
    initial, goal = np.eye(4), np.eye(4)
    initial[2, 3] = .04
    poses = np.tile(goal, (len(times), 1, 1))
    poses[:6, 2, 3] = np.linspace(.04, 0, 6)
    contact = np.zeros(len(times), bool)
    contact[0] = True
    result = evaluate_rigid(times, poses, contact, initial_pose=initial, goal_pose=goal)
    assert result["terminal_goal_dwell_s"] >= .3
    assert result["maximum_displacement_m"] >= .02
    assert not result["success"]
    assert result["post_reset_hand_contact_duration_s"] == 0


def test_cloth_threshold_changes_the_actual_score():
    initial = np.zeros((4, 3))
    target = initial+[.1, 0, 0]
    times = np.arange(0, 1.02, .02)
    states = np.tile(target+[.02, 0, 0], (len(times), 1, 1))
    contact = np.ones(len(times), bool)
    kwargs = dict(initial_vertices=initial, target_vertices=target, task="cloth_drag")
    assert evaluate_cloth(times, states, contact, tolerance_m=.025, **kwargs)["success"]
    assert not evaluate_cloth(times, states, contact, tolerance_m=.01, **kwargs)["success"]


def test_cloth_cannot_pass_with_explosive_motion_or_robot_limit_violation():
    initial = np.zeros((4, 3))
    target = initial+[.1, 0, 0]
    times = np.arange(0, 1.02, .02)
    states = np.tile(target, (len(times), 1, 1))
    contacts = np.ones(len(times), bool)
    kwargs = dict(initial_vertices=initial, target_vertices=target, task="cloth_drag")
    assert not evaluate_cloth(times, states, contacts, joint_limit_violation=True, **kwargs)["success"]
    states[0] = initial
    result = evaluate_cloth(times, states, contacts, **kwargs)
    assert result["maximum_vertex_speed_m_s"] == 5.
    assert result["terminal_goal_dwell_s"] >= .3
    assert not result["stable"] and not result["success"]
