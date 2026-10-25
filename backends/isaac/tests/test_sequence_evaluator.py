from copy import deepcopy

import numpy as np
import pytest

from bench.sequence_evaluator import SequenceEvaluator


def rigid(name, target=(1., 0., 0.), **extra):
    return {"kind": "rigid", "object": name, "position_m": list(target),
            "radius_m": .025, "max_speed_m_s": .05, **extra}


def sequence(*predicates, **extra):
    return {"id": "test", "max_observation_gap_s": .11,
            "steps": [{"id": f"stage_{i}", "hold_s": .2, "complete": [p]}
                      for i, p in enumerate(predicates)], **extra}


def object_state(position=(0., 0., 0.), quaternion=(1., 0., 0., 0.), velocity=(0., 0., 0.)):
    return {"position_m": list(position), "quaternion_wxyz": list(quaternion),
            "linear_velocity_m_s": list(velocity), "angular_velocity_rad_s": [0., 0., 0.]}


class Rollout:
    def __init__(self, task, *, vertices=None, **objects):
        self.evaluator = SequenceEvaluator(task)
        self.obs = {"time_s": -.05, "objects": {k: object_state(v) for k, v in objects.items()},
                    "metrics": {"state_finite": True}}
        if vertices is not None:
            self.cloth(vertices)

    def cloth(self, vertices):
        self.obs["cloth"] = {"vertices_m": np.asarray(vertices, dtype=float).tolist(),
                             "velocities_m_s": np.zeros_like(vertices, dtype=float).tolist()}

    def place(self, name, target=(1., 0., 0.)):
        self.obs["objects"][name]["position_m"] = list(target)

    def tick(self, duration=.05):
        self.obs["time_s"] += duration
        return self.evaluator.update(self.obs)

    def hold(self, frames=6):
        for _ in range(frames):
            result = self.tick()
        return result


def test_ordered_three_step_task_requires_separate_stable_arrivals():
    r = Rollout(sequence(rigid("a"), rigid("b"), rigid("c")), a=(0, 0, 0), b=(0, 0, 0), c=(0, 0, 0))
    assert r.tick()["reason"] == "baseline_recorded"
    for i, name in enumerate(("a", "b", "c"), 1):
        r.place(name)
        entry = r.tick()
        assert not entry["success"]
        result = r.hold(4)
        assert result["progress"] == pytest.approx(i / 3)
    assert result["success"]
    assert len(result["completion_times_s"]) == 3


def test_reset_at_all_goals_cannot_pass_even_after_long_dwell():
    r = Rollout(sequence(rigid("a"), rigid("b")), a=(1, 0, 0), b=(1, 0, 0))
    r.tick()
    result = r.hold(40)
    assert result["progress"] == 0
    assert result["reason"] == "waiting_for_fresh_entry"
    # Settling of a fast reset state is not fresh geometric entry either.
    r.obs["objects"]["a"]["linear_velocity_m_s"] = [1., 0., 0.]
    r.evaluator.reset()
    r.tick()
    r.obs["objects"]["a"]["linear_velocity_m_s"] = [0., 0., 0.]
    assert r.hold()["progress"] == 0


def test_future_stage_done_out_of_order_requires_reentry_after_activation():
    r = Rollout(sequence(rigid("a"), rigid("b")), a=(0, 0, 0), b=(0, 0, 0))
    r.tick()
    r.place("b")
    assert r.hold()["progress"] == 0
    r.place("a")
    assert r.hold()["progress"] == .5
    assert r.hold(20)["reason"] == "waiting_for_fresh_entry"
    r.place("b", (0., 0., 0.))
    r.tick()
    r.place("b")
    assert r.hold()["success"]


def test_one_frame_transit_and_high_speed_arrival_do_not_pass():
    r = Rollout(sequence(rigid("part")), part=(0, 0, 0))
    r.tick()
    r.place("part")
    assert not r.tick()["success"]
    r.place("part", (2., 0., 0.))
    assert r.hold()["progress"] == 0
    r.place("part")
    r.obs["objects"]["part"]["linear_velocity_m_s"] = [.1, 0., 0.]
    assert not r.hold()["success"]
    r.obs["objects"]["part"]["linear_velocity_m_s"] = [0., 0., 0.]
    assert r.hold()["success"]


def test_dwell_restarts_after_sample_gap_and_after_leaving_goal():
    r = Rollout(sequence(rigid("part")), part=(0, 0, 0))
    r.tick()
    r.place("part")
    r.hold(3)
    gap = r.tick(2.)
    assert gap["reason"] == "observation_gap:dwell_reset"
    assert gap["hold_elapsed_s"] == 0
    assert not r.hold(3)["success"]
    r.place("part", (0, 0, 0))
    r.tick()
    r.place("part")
    assert not r.hold(4)["success"]
    assert r.tick()["success"]


def test_lost_previous_condition_rolls_back_and_revokes_terminal_success():
    r = Rollout(sequence(rigid("a"), rigid("b")), a=(0, 0, 0), b=(0, 0, 0))
    r.tick()
    r.place("a")
    assert r.hold()["progress"] == .5
    r.place("b")
    r.tick()
    r.place("a", (0, 0, 0))
    failed = r.tick()
    assert failed["progress"] == 0
    assert failed["reason"].startswith("previous_step_lost:stage_0")
    r.place("a")
    r.hold()
    assert not r.hold()["success"]  # b was done before its reactivation.
    r.place("b", (0, 0, 0))
    r.tick()
    r.place("b")
    assert r.hold()["success"]
    r.place("a", (0, 0, 0))
    assert not r.tick()["success"]


def test_maintenance_can_allow_later_cloth_deformation():
    region = {"kind": "cloth_region", "bounds_m": [[-.1, -.1, -.01], [.1, .1, .01]],
              "min_span_m": [.15, .15, 0], "max_speed_m_s": .1}
    task = sequence(region, rigid("part"))
    task["steps"][0]["maintain"] = [{**region, "bounds_m": [[-.1, -.1, -.01], [.1, .1, .08]]}]
    flat = np.array([[-.1, -.1, 0], [-.1, .1, 0], [.1, -.1, 0], [.1, .1, 0]])
    r = Rollout(task, vertices=flat + [0, .3, 0], part=(0, 0, 0))
    r.tick()
    r.cloth(flat)
    assert r.hold()["progress"] == .5
    r.cloth(flat + [0, 0, .04])
    r.place("part")
    assert r.hold()["success"]


def test_cloth_translation_cannot_impersonate_a_fold_of_labelled_half():
    initial = np.array([[-.15, -.05, 0], [-.15, .05, 0], [.15, -.05, 0], [.15, .05, 0]])
    task = sequence({"kind": "cloth_region", "group": "moving", "bounds_m": [[.13, -.1, 0], [.17, .1, .03]],
                     "min_span_m": [0, .08, 0], "max_speed_m_s": .1},
                    vertex_groups={"moving": [0, 1], "anchor": [2, 3]})
    task["steps"][0]["complete"].append({"kind": "cloth_region", "group": "anchor",
        "bounds_m": [[.13, -.1, -.01], [.17, .1, .01]], "min_span_m": [0, .08, 0]})
    r = Rollout(task, vertices=initial)
    r.tick()
    r.cloth(initial + [.3, 0, 0])
    assert not r.hold()["success"]
    folded = initial.copy()
    folded[:2] += [.3, 0, .01]
    r.cloth(folded)
    assert r.hold()["success"]


def test_cloth_surface_requires_height_and_coverage_not_only_centroid():
    points = [[-.08, -.08, .06], [-.08, .08, .06], [.08, -.08, .06], [.08, .08, .06]]
    task = sequence({"kind": "cloth_samples", "points_m": points,
                     "xy_radius_m": .02, "z_tolerance_m": .01, "min_fraction": 1., "max_speed_m_s": .1})
    below = np.asarray(points) - [0, 0, .06]
    r = Rollout(task, vertices=below)
    r.tick()
    assert not r.hold()["success"]
    r.cloth(np.tile([0, 0, .06], (4, 1)))
    assert not r.hold()["success"]
    r.cloth(points)
    assert r.hold()["success"]


def test_cloth_region_centroid_spread_fraction_and_speed_are_enforced():
    target = np.array([[-.1, -.1, .02], [-.1, .1, .02], [.1, -.1, .02], [.1, .1, .02]])
    task = sequence({"kind": "cloth_region", "bounds_m": [[-.11, -.11, 0], [.11, .11, .04]],
                     "min_fraction": .75, "centroid_m": [0, 0, .02], "centroid_radius_m": .02,
                     "min_span_m": [.15, .15, 0], "max_speed_m_s": .1})
    r = Rollout(task, vertices=target + [1, 0, 0])
    r.tick()
    r.cloth(np.tile([0, 0, .02], (4, 1)))
    assert "cloth_span" in r.hold()["reason"]
    shifted = target.copy()
    shifted[-1, 0] += .15
    r.cloth(shifted)
    assert "cloth_centroid" in r.hold()["reason"]
    r.cloth(target)
    r.obs["cloth"]["velocities_m_s"][0][0] = .2
    assert "cloth_speed" in r.hold()["reason"]
    r.cloth(target)
    assert r.hold()["success"]


def test_rigid_height_axis_rotation_and_angular_speed():
    predicate = rigid("gear", xy_radius_m=.01, z_tolerance_m=.005,
                      axis_local=[0, 0, 1], axis_world=[0, 0, 1], axis_tolerance_rad=.1,
                      max_angular_speed_rad_s=.2)
    predicate.pop("radius_m")
    r = Rollout(sequence(predicate), gear=(0, 0, 0))
    r.tick()
    r.place("gear", (1, 0, .01))
    assert "height" in r.hold()["reason"]
    r.place("gear")
    r.obs["objects"]["gear"]["quaternion_wxyz"] = [np.sqrt(.5), np.sqrt(.5), 0, 0]
    assert "axis_orientation" in r.hold()["reason"]
    r.obs["objects"]["gear"]["quaternion_wxyz"] = [0, 0, 0, 1]  # Free yaw.
    r.obs["objects"]["gear"]["angular_velocity_rad_s"] = [0, 0, .3]
    assert "angular_speed" in r.hold()["reason"]
    r.obs["objects"]["gear"]["angular_velocity_rad_s"] = [0, 0, 0]
    assert r.hold()["success"]


def test_object_local_surface_samples_transform_with_fixture():
    task = sequence({"kind": "cloth_samples", "frame_object": "fixture", "points_m": [[.1, 0, .05]],
                     "xy_radius_m": .01, "z_tolerance_m": .01})
    r = Rollout(task, vertices=[[0, 0, 0]], fixture=(1, 2, 3))
    r.obs["objects"]["fixture"]["quaternion_wxyz"] = [np.sqrt(.5), 0, 0, np.sqrt(.5)]
    r.tick()
    r.cloth([[1, 2.1, 3.05]])
    assert r.hold()["success"]


def test_full_orientation_quaternion_sign_is_irrelevant():
    predicate = rigid("bar", quaternion_wxyz=[1, 0, 0, 0], orientation_tolerance_rad=.1)
    r = Rollout(sequence(predicate), bar=(0, 0, 0))
    r.tick()
    r.place("bar")
    r.obs["objects"]["bar"]["quaternion_wxyz"] = [0, 0, 0, 1]
    assert "orientation" in r.hold()["reason"]
    r.obs["objects"]["bar"]["quaternion_wxyz"] = [-1, 0, 0, 0]
    assert r.hold()["success"]


def test_evaluator_is_read_only_and_hidden_references_cannot_set_success():
    task = sequence(rigid("part"))
    r = Rollout(task, part=(0, 0, 0))
    r.obs["hidden_reference"] = {"objects": {"part": object_state((1, 0, 0))}, "success": True}
    original = deepcopy(r.obs)
    r.tick()
    baseline = r.evaluator.initial_observation
    baseline["objects"]["part"]["position_m"][0] = 999
    assert r.evaluator.initial_observation["objects"]["part"]["position_m"][0] == 0
    assert r.obs["objects"] == original["objects"]
    task["steps"][0]["complete"][0]["position_m"] = [0, 0, 0]
    assert not r.hold()["success"]
    r.place("part")
    before = deepcopy(r.obs)
    assert r.hold()["success"]
    assert r.obs["objects"] == before["objects"]
    assert r.obs["hidden_reference"] == before["hidden_reference"]


def test_condition_inspection_does_not_initialize_or_advance_sequence():
    r = Rollout(sequence(rigid("part")), part=(1, 0, 0))
    checks = r.evaluator.inspect_conditions(r.obs)
    assert checks[0]["complete_satisfied"]
    assert checks[0]["complete"][0]["geometry_satisfied"]
    assert r.evaluator.initial_observation is None
    assert r.tick()["reason"] == "baseline_recorded"
    assert r.hold()["progress"] == 0


def test_invalid_physics_revokes_progress_and_bad_timestamps_raise():
    r = Rollout(sequence(rigid("part")), part=(0, 0, 0))
    r.tick()
    with pytest.raises(ValueError, match="strictly increasing"):
        r.evaluator.update(r.obs)
    r.place("part")
    assert r.hold()["success"]
    r.obs["metrics"]["state_finite"] = False
    result = r.tick()
    assert not result["success"]
    assert result["progress"] == 0
    assert result["reason"].startswith("invalid_observation")


@pytest.mark.parametrize("bad", [
    {"kind": "unrecognised"},
    {"kind": "rigid", "object": "part", "position_m": [0, 0, 0]},
    {"kind": "cloth_region", "bounds_m": [[1, 0, 0], [0, 0, 0]]},
    {"kind": "cloth_region", "group": "missing", "bounds_m": [[0, 0, 0], [1, 1, 1]]},
    {"kind": "cloth_samples", "points_m": [], "xy_radius_m": .1, "z_tolerance_m": .1},
])
def test_invalid_task_configuration_rejected(bad):
    with pytest.raises(ValueError):
        SequenceEvaluator(sequence(bad))
