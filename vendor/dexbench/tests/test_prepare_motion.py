"""Motion schedule and independent artifact checks; no physical-success surrogate."""
from pathlib import Path

import numpy as np
import pytest

from bench.data_tools.prepare_motion import limit_targets, sampling_grid
from bench.control.kinematics import URDFKinematics
from bench.data_tools.reference import digest
from bench.paths import DATA_ROOT, load_json


def test_sampling_keeps_full_endpoint_and_fixed_physics_intervals():
    source = np.arange(164)/120
    times, left, right, alpha = sampling_grid(source)
    assert times[0] == 0
    np.testing.assert_allclose(np.diff(times), .02, atol=1e-14)
    assert source[-1] <= times[-1] < source[-1]+.02
    np.testing.assert_allclose(source[left]*(1-alpha)+source[right]*alpha, np.minimum(times, source[-1]))
    assert left[0] == 0 and alpha[0] == 0
    assert right[-1] == len(source)-1 and alpha[-1] == 1
    assert np.all((0 <= alpha) & (alpha <= 1))


def test_sampling_exact_boundary_does_not_add_extra_hold():
    times, _, _, _ = sampling_grid(np.arange(13)/120)
    np.testing.assert_allclose(times, np.arange(6)*.02)


@pytest.mark.parametrize("source", [[0, 0], [1, 2], [0, np.nan], [0], [0, -.1]])
def test_sampling_rejects_ambiguous_or_nonfinite_clock(source):
    with pytest.raises(ValueError):
        sampling_grid(source)


def test_joint_velocity_cap_is_per_joint_and_per_elapsed_interval():
    previous = np.array([.1, -.2, .3])
    raw = np.array([1., -1., .301])
    velocity = np.array([1., 3., 2.])
    target = limit_targets(raw, previous, velocity, .02)
    np.testing.assert_allclose(target, [.12, -.26, .301])
    assert np.all(np.abs(target-previous) <= velocity*.02+1e-14)


def test_generated_collection_has_consistent_reset_rates_hashes_and_fk():
    # motion_v2 is also a historical artifact: source paths name the old editable
    # checkout. Verify its original bytes in the retained pre-cleanup archive,
    # never against relocated/current code or by rewriting the recorded hashes.
    root = Path(f"{DATA_ROOT}/datasets/fr3_sharpa_bench/motion_v2")
    if not (root/"manifest.json").exists():
        pytest.skip("Motion artifact collection has not been generated")
    original_source_root = Path(f"{DATA_ROOT}/interface")
    archived_source_root = root.parent/"maintenance/project_cleanup_20261014_02/source_before"
    manifest = load_json(root/"manifest.json")
    assert manifest["complete_source_collection"] and manifest["clip_count"] == 15
    for clip in manifest["clips"]:
        assert digest(clip["metadata_path"]) == clip["metadata_sha256"]
        metadata = load_json(Path(clip["metadata_path"]))
        assert digest(clip["trajectory_path"]) == clip["trajectory_sha256"]
        assert metadata["physical_rollout_performed"] is False
        for record in metadata["provenance"]["source_code"]:
            archived = archived_source_root/Path(record["path"]).relative_to(original_source_root)
            assert digest(archived) == record["sha256"]
        for record in metadata["provenance"]["robot_urdfs"]:
            assert digest(record["path"]) == record["sha256"]
        with np.load(clip["trajectory_path"], allow_pickle=False) as data:
            q, names, time = data["joint_positions_rad"], data["joint_names"].tolist(), data["time_s"]
            assert q.shape == (len(time), 29) and len(set(names)) == 29
            assert np.isfinite(q).all()
            np.testing.assert_allclose(q[0], [metadata["bootstrap"]["initial_joint_positions_rad"][n] for n in names])
            np.testing.assert_allclose(data["initial_joint_positions_rad"], q[0])
            np.testing.assert_allclose(np.diff(time), .02, atol=1e-12)
            measured_velocity = np.diff(q, axis=0)/np.diff(time)[:, None]
            assert np.all(np.abs(measured_velocity) <= data["joint_velocity_limits_rad_s"]+1e-9)
            np.testing.assert_allclose(measured_velocity, data["joint_velocity_rad_s"][1:])
            assert data["reference_time_s"][-1] == metadata["sampling"]["source_last_time_s"]
            assert data["source_alpha"][-1] == 1
            arm = URDFKinematics(metadata["provenance"]["robot_urdfs"][0]["path"])
            mount = np.asarray(metadata["config"]["mount_transform"])
            for i in [0, len(q)//2, len(q)-1]:
                fk = arm.forward(dict(zip(names, q[i])))["fr3_link8"] @ mount
                np.testing.assert_allclose(fk, data["target_fk_hand_root_T_world"][i], atol=1e-12)
                error = np.linalg.norm(fk[:3, 3]-data["desired_hand_root_T_world"][i, :3, 3])
                assert error == pytest.approx(data["target_arm_position_error_m"][i])
