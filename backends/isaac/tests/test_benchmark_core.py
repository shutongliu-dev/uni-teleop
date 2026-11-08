from pathlib import Path
import numpy as np
import pytest

from bench.geometry import quaternion_matrix, matrix_quaternion, rotvec_matrix, rotation_distance
from bench.control.kinematics import URDFKinematics
from bench.paths import DATA_ROOT

ASSETS = Path(f"{DATA_ROOT}/assets")


def test_rotation_roundtrip_including_pi():
    for v in ([0, 0, 0], [np.pi, 0, 0], [0, -np.pi, 0], [.3, -.8, 1.2]):
        r = rotvec_matrix(v)
        assert np.allclose(quaternion_matrix(matrix_quaternion(r)), r, atol=1e-10)


def test_fr3_ik_reaches_known_reachable_pose_and_respects_bounds():
    arm = URDFKinematics(ASSETS / "robots/franka/fr3/fr3_no_hand.urdf")
    home = dict(zip(arm.names, [0, -.5, 0, -2.3, 0, 1.8, .8]))
    moved = {**home, "fr3_joint1": .12, "fr3_joint2": -.42, "fr3_joint6": 1.9}
    desired = arm.forward(moved)["fr3_link8"]
    result = arm.inverse(desired, "fr3_link8", home)
    assert result["converged"], result
    impossible = desired.copy()
    impossible[:3, 3] = [8, 8, 8]
    result = arm.inverse(impossible, "fr3_link8", home)
    assert not result["converged"]
    assert all(j.lower <= result["joint_positions_rad"][j.name] <= j.upper for j in arm.active)


@pytest.mark.parametrize("joint_values", [
    [.8, .5, -.7, -1.7, 1.1, 1.6, -.8],
    [-1., .8, .9, -1.1, -1.1, 2.1, 1.6],
    [2., -.8, -1.6, -2., 2., 2.8, -2.],
    [-2., 1., 1.8, -.3, -2., 1., 2.],
])
def test_fr3_ik_reachable_fk_targets_with_monotone_accepted_steps(joint_values):
    arm = URDFKinematics(ASSETS / "robots/franka/fr3/fr3_no_hand.urdf")
    home = dict(zip(arm.names, [0, -.5, 0, -2.3, 0, 1.8, .8]))
    desired = arm.forward(dict(zip(arm.names, joint_values)))["fr3_link8"]
    result = arm.inverse(desired, "fr3_link8", home, iterations=120,
                         multi_starts=8, return_trace=True)
    assert result["converged"], result
    fk = arm.forward(result["joint_positions_rad"])["fr3_link8"]
    assert np.linalg.norm(fk[:3, 3]-desired[:3, 3]) <= .002
    assert rotation_distance(fk[:3, :3], desired[:3, :3]) <= .04
    assert all(np.all(np.diff(trace) < 0) for trace in result["accepted_objective_traces"])
    assert result["weighted_error_m"] <= result["initial_weighted_error_m"]


def test_fr3_ik_unreachable_target_retains_best_pose():
    arm = URDFKinematics(ASSETS / "robots/franka/fr3/fr3_no_hand.urdf")
    home = dict(zip(arm.names, [0, -.5, 0, -2.3, 0, 1.8, .8]))
    desired = arm.forward(home)["fr3_link8"]
    desired[:3, 3] = [8., 8., 8.]
    result = arm.inverse(desired, "fr3_link8", home, return_trace=True)
    assert not result["converged"]
    assert result["weighted_error_m"] <= result["initial_weighted_error_m"]
    trace = result["accepted_objective_traces"][0]
    assert np.all(np.diff(trace) < 0)
    assert result["weighted_error_m"]**2 == pytest.approx(min(trace))
    assert all(j.lower <= result["joint_positions_rad"][j.name] <= j.upper for j in arm.active)


def test_fr3_ik_fork_frame_164_does_not_diverge_and_restart_recovers():
    # Frozen motion_v1 fork at 3.28 s: preceding commanded pose -> next source
    # root target. The former final-iterate solver produced 1.415124 m error.
    path = Path(f"{DATA_ROOT}/datasets/fr3_sharpa_bench/motion_v1/clips/oi2_08_fork_rearrange/trajectory.npz")
    with np.load(path, allow_pickle=False) as data:
        desired = data["desired_hand_root_T_world"][164]
        initial = dict(zip(data["joint_names"].tolist(), data["joint_positions_rad"][163]))
    arm = URDFKinematics(ASSETS / "robots/franka/fr3/fr3_no_hand.urdf")
    bounded = arm.inverse(desired, "fr3_link8", initial, return_trace=True)
    assert bounded["position_error_m"] < .1
    assert bounded["weighted_error_m"] < bounded["initial_weighted_error_m"]
    assert np.all(np.diff(bounded["accepted_objective_traces"][0]) < 0)
    recovered = arm.inverse(desired, "fr3_link8", initial, multi_starts=8)
    assert recovered["converged"], recovered
    assert recovered["position_error_m"] <= .002
    assert recovered["rotation_error_rad"] <= .04
    repeated = arm.inverse(desired, "fr3_link8", initial, multi_starts=8)
    assert repeated == recovered
