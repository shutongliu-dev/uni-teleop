"""Prepare hash-bound, complete 50 Hz FR3 + right Sharpa mocap target trajectories.

No simulator is imported. Object poses are observations/goals only, never actions.
Run with OPENBLAS_NUM_THREADS=1 python -m bench.data_tools.prepare_motion --workers 8.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
import json
from pathlib import Path
import numpy as np

from .agent_plan import build_plan
from bench.geometry import quaternion_matrix, rotation_distance, transform
from bench.control.keypoint_retarget import KeypointRetargetConfig, KeypointRetargeter
from bench.control.kinematics import URDFKinematics
from .reference import ReferenceClip, canonical_digest, digest
from bench.paths import DATA_ROOT, OUTPUT_ROOT, load_json

VERSION = "fr3_sharpa_decoded_mano_motion_v2"


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")


def sampling_grid(source_times, dt=.02):
    """Use fixed dt, clamping the final reference sample to cover the endpoint."""
    source_times = np.asarray(source_times, dtype=float)
    if (source_times.ndim != 1 or len(source_times) < 2 or
            not np.isfinite(source_times).all() or source_times[0] != 0 or
            not np.all(np.diff(source_times) > 0) or not 0 < dt <= .2):
        raise ValueError("Need finite increasing zero-based times and dt in (0,.2]")
    end = source_times[-1]
    times = np.arange(int(np.ceil(end/dt-1e-10))+1, dtype=float)*dt
    reference_times = np.minimum(times, end)
    right = np.clip(np.searchsorted(source_times, reference_times, side="right"), 1, len(source_times)-1)
    left = right-1
    alpha = (reference_times-source_times[left])/(source_times[right]-source_times[left])
    return times, left, right, alpha


def limit_targets(raw, previous, velocity_limits, dt):
    if dt <= 0 or not np.isfinite(dt):
        raise ValueError("Target interval must be positive and finite")
    step = np.asarray(velocity_limits)*dt
    return np.clip(raw, previous-step, previous+step)


def stats(array):
    array = np.asarray(array)
    return {"mean": float(np.mean(array)), "p95": float(np.percentile(array, 95)),
            "max": float(np.max(array))}


def prepare_clip(arguments):
    entry, decoded, decoded_manifest, task_dir, output_dir, dt = arguments
    task_path = Path(task_dir)/f"{entry['clip_id']}.json"
    task = load_json(task_path)
    ref = ReferenceClip(entry)
    if task["reference"]["reference_sha256"] != entry["reference_sha256"]:
        raise ValueError("Task/reference mismatch")
    for key in ("decoded", "metadata"):
        if digest(decoded[f"{key}_path"]) != decoded[f"{key}_sha256"]:
            raise ValueError(f"Decoded {key} hash mismatch")
    if decoded["source_reference_sha256"] != entry["reference_sha256"]:
        raise ValueError("Decoded/source reference mismatch")
    with np.load(decoded["decoded_path"], allow_pickle=False) as data:
        source_times = data["time_s"].copy()
        points = data["rh_joints_world_m"].copy()
        source_frame_ids = data["source_frame_ids"].copy()
        if data["joint_order"].tolist() != decoded_manifest["joint_order"]:
            raise ValueError("Unexpected decoded joint order")
    if (points.shape != (len(source_times), 21, 3) or not np.isfinite(points).all() or
            not np.array_equal(source_frame_ids, ref.arrays["source_frame_ids"]) or
            not np.array_equal(source_times, ref.arrays["time_s"])):
        raise ValueError("Invalid or misaligned decoded trajectory")
    robot = task["scene_spec"]["robot"]
    if np.any(np.asarray(robot["base_position_m"]) != 0):
        raise ValueError("This baseline requires the frozen zero-position robot base")
    assets = Path(robot["assets_root"])
    arm_path = assets/"robots/franka/fr3/fr3_no_hand.urdf"
    hand_path = assets/"robots/sharpa/wave_01/derived/right_sharpa_wave_with_flange.urdf"
    arm, hand = URDFKinematics(arm_path), URDFKinematics(hand_path)
    joints = arm.active+hand.active
    names = arm.names+hand.names
    if len(names) != 29 or len(set(names)) != 29:
        raise ValueError("Expected 7 FR3 + 22 right Sharpa joints")
    initial = dict(zip(arm.names, robot["arm_home_rad"]))
    initial.update({j.name: float(np.clip(0, j.lower, j.upper)) for j in hand.active})
    mount = transform(robot["mount_position_m"], quaternion_matrix(robot["mount_quaternion_wxyz"]))
    config = KeypointRetargetConfig()
    retarget = KeypointRetargeter(assets, initial, task["source_to_bench"], mount_transform=mount, config=config)
    velocity = np.array([min(j.velocity, config.maximum_arm_velocity_rad_s if j.name in arm.names
                            else config.maximum_hand_velocity_rad_s) for j in joints])
    times, left, right, alpha = sampling_grid(source_times, dt)
    sampled = points[left]*(1-alpha[:, None, None])+points[right]*alpha[:, None, None]
    # Bootstrap has no previous physical state; it is a reset, never a rate-limited action.
    root0, geometry0 = retarget.target_geometry(sampled[0])
    bootstrap_hand, bootstrap_hand_error = retarget.fit_hand(geometry0)
    bootstrap_ik = arm.inverse(root0 @ np.linalg.inv(mount), "fr3_link8", initial, iterations=240, multi_starts=8)
    bootstrap = {**bootstrap_ik["joint_positions_rad"], **bootstrap_hand}
    retarget.previous = bootstrap.copy()
    rows = []
    for index, human_points in enumerate(sampled):
        desired_root, geometry = retarget.target_geometry(human_points)
        if index == 0:
            fitted_hand, fit_error, ik = bootstrap_hand, bootstrap_hand_error, bootstrap_ik
        else:
            fitted_hand, fit_error = retarget.fit_hand(geometry)
            ik = arm.inverse(desired_root @ np.linalg.inv(mount), "fr3_link8", retarget.previous, multi_starts=2)
        raw_map = {**ik["joint_positions_rad"], **fitted_hand}
        raw = np.array([raw_map[n] for n in names])
        previous = np.array([retarget.previous[n] for n in names])
        elapsed = 0. if index == 0 else times[index]-times[index-1]
        target = raw if index == 0 else limit_targets(raw, previous, velocity, elapsed)
        target_map = dict(zip(names, target.tolist()))
        actual_root = arm.forward(target_map)["fr3_link8"] @ mount
        actual_hand = hand.forward(target_map)
        fit_points = np.array([actual_hand[n][:3, 3] for n in retarget.fitted_links])
        actual_world = fit_points @ actual_root[:3, :3].T+actual_root[:3, 3]
        desired_world = geometry @ desired_root[:3, :3].T+desired_root[:3, 3]
        rows.append({
            "joint_positions_rad": target, "raw_joint_positions_rad": raw,
            "rate_limited_joint_mask": np.abs(raw-target) > 1e-10,
            "joint_velocity_rad_s": np.zeros(29) if index == 0 else (target-previous)/elapsed,
            "desired_hand_root_T_world": desired_root,
            "target_fk_hand_root_T_world": actual_root,
            "raw_arm_position_error_m": ik["position_error_m"],
            "raw_arm_rotation_error_rad": ik["rotation_error_rad"],
            "raw_arm_ik_converged": ik["converged"], "raw_arm_ik_iterations": ik["iterations"],
            "raw_arm_ik_starts_attempted": ik["starts_attempted"],
            "raw_arm_ik_initial_weighted_error_m": ik["initial_weighted_error_m"],
            "raw_arm_ik_weighted_error_m": ik["weighted_error_m"],
            "raw_hand_geometry_rmse_m": fit_error,
            "target_arm_position_error_m": np.linalg.norm(actual_root[:3, 3]-desired_root[:3, 3]),
            "target_arm_rotation_error_rad": rotation_distance(actual_root[:3, :3], desired_root[:3, :3]),
            "target_hand_geometry_rmse_m": np.sqrt(np.mean(np.sum((fit_points-geometry)**2, axis=1))),
            "target_world_keypoint_rmse_m": np.sqrt(np.mean(np.sum((actual_world-desired_world)**2, axis=1))),
        })
        retarget.previous = target_map
    arrays = {key: np.asarray([row[key] for row in rows]) for key in rows[0]}
    arrays.update(time_s=times, interval_from_previous_s=np.r_[0., np.diff(times)],
                  reference_time_s=np.minimum(times, source_times[-1]),
                  joint_names=np.asarray(names), joint_velocity_limits_rad_s=velocity,
                  source_left_index=left, source_right_index=right, source_alpha=alpha,
                  source_left_frame_id=source_frame_ids[left], source_right_frame_id=source_frame_ids[right],
                  initial_joint_positions_rad=np.array([bootstrap[n] for n in names]))
    for key, value in arrays.items():
        if value.dtype.kind == "f" and not np.isfinite(value).all():
            raise ValueError(f"Nonfinite generated field {key}")
    q = arrays["joint_positions_rad"]
    if (np.any(q < np.array([j.lower for j in joints])-1e-10) or
            np.any(q > np.array([j.upper for j in joints])+1e-10) or
            np.any(np.abs(arrays["joint_velocity_rad_s"]) > velocity+1e-9)):
        raise ValueError("Generated targets violate declared joint or velocity limits")
    destination = Path(output_dir)/"clips"/entry["clip_id"]
    destination.mkdir(parents=True, exist_ok=True)
    trajectory_path = destination/"trajectory.npz"
    np.savez_compressed(trajectory_path, **arrays)
    plan_path = destination/"agent_plan.json"
    write_json(plan_path, build_plan(task, ref.metadata))
    source_files = [Path(__file__), Path(__file__).with_name("agent_plan.py"),
                    Path(__file__).parents[1] / "control/__init__.py",
                    Path(__file__).parents[1] / "control/keypoint_retarget.py",
                    Path(__file__).parents[1] / "control/kinematics.py",
                    Path(__file__).parents[1] / "geometry.py"]
    config_record = {"retarget": asdict(config), "control_dt_s": dt,
                     "bootstrap_arm_ik_iterations": 240, "bootstrap_arm_ik_fallback_starts": 8,
                     "rollout_arm_ik_iterations": 60, "rollout_arm_ik_fallback_starts": 2,
                     "arm_ik_algorithm": "bounded_monotone_lm_best_iterate_v2",
                     "arm_position_tolerance_m": .002, "arm_rotation_tolerance_rad": .04,
                     "source_to_bench": task["source_to_bench"], "mount_transform": mount.tolist(),
                     "initial_seed_joint_positions_rad": initial,
                     "joint_velocity_limits_rad_s": dict(zip(names, velocity.tolist()))}
    failed = np.flatnonzero(~arrays["raw_arm_ik_converged"])
    report = {
        "schema_version": VERSION, "clip_id": entry["clip_id"],
        "trajectory_path": str(trajectory_path), "trajectory_sha256": digest(trajectory_path),
        "agent_plan_path": str(plan_path), "agent_plan_sha256": digest(plan_path),
        "joint_names": names, "sample_count": len(times),
        "bootstrap": {"initial_joint_positions_rad": bootstrap, "source_time_s": 0.,
                      "arm_ik": {k: v for k, v in bootstrap_ik.items() if k != "joint_positions_rad"},
                      "hand_geometry_rmse_m": bootstrap_hand_error,
                      "collision_validated": False, "physical_settling_performed": False,
                      "usage": "Explicit robot reset configuration, before the first physics step."},
        "sampling": {"nominal_hz": 1/dt, "interpolation": "linear decoded world keypoints",
                     "source_frame_count": len(source_times), "source_first_frame_id": int(source_frame_ids[0]),
                     "source_last_frame_id": int(source_frame_ids[-1]), "first_time_s": float(times[0]),
                     "last_time_s": float(times[-1]), "source_last_time_s": float(source_times[-1]),
                     "final_endpoint_hold_s": float(times[-1]-source_times[-1]),
                     "last_interval_s": float(times[-1]-times[-2]),
                     "full_source_time_coverage": True, "cropped": False, "time_scaled": False,
                     "spatial_scale": 1., "terminal_behavior": "hold last target for task evaluator dwell",
                     "row_semantics": "Row 0 is reset target. For each i>=1, apply target q[i] for time_s[i]-time_s[i-1]. Fixed 20 ms steps; final reference sample clamps to source endpoint with less than 20 ms hold."},
        "config": config_record, "config_sha256": canonical_digest(config_record),
        "retarget_provenance": retarget.provenance(),
        "provenance": {"task_path": str(task_path), "task_sha256": digest(task_path),
                       "task_frozen_config_sha256": task["frozen_config_sha256"],
                       "source_reference_path": entry["reference_path"], "source_reference_sha256": entry["reference_sha256"],
                       "source_metadata_path": entry["metadata_path"], "source_metadata_sha256": entry["metadata_sha256"],
                       "decoded_path": decoded["decoded_path"], "decoded_sha256": decoded["decoded_sha256"],
                       "decoded_metadata_sha256": decoded["metadata_sha256"],
                       "model_and_decoder": decoded_manifest["model_and_decoder"],
                       "robot_urdfs": [{"path": str(p), "sha256": digest(p)} for p in (arm_path, hand_path)],
                       "source_code": [{"path": str(p), "sha256": digest(p)} for p in source_files],
                       "source_license": ref.metadata["license"], "numpy_version": np.__version__},
        "reachability": {"assessment": "local bounded IK over complete 50 Hz reference; not a global reachability proof",
                         "all_sampled_arm_ik_converged": len(failed) == 0,
                         "failed_sample_count": len(failed), "failed_sample_indices": failed.tolist(),
                         "converged_fraction": float(np.mean(arrays["raw_arm_ik_converged"])),
                         "rate_limited_sample_fraction": float(np.mean(np.any(arrays["rate_limited_joint_mask"], axis=1))),
                         "rate_limited_counts_by_joint": dict(zip(names, np.sum(arrays["rate_limited_joint_mask"], axis=0).tolist())),
                         "errors": {key: stats(value) for key, value in arrays.items() if "error" in key or "rmse" in key}},
        "qualification": "finite_bounded_kinematic_targets_only_not_benchmark_success",
        "physical_rollout_performed": False, "trained_rl": False, "online_model_api_called": False,
    }
    metadata_path = destination/"metadata.json"
    write_json(metadata_path, report)
    return {"clip_id": entry["clip_id"], "metadata_path": str(metadata_path), "metadata_sha256": digest(metadata_path),
            "trajectory_path": str(trajectory_path), "trajectory_sha256": report["trajectory_sha256"],
            "sample_count": len(times), "reachability": report["reachability"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", default=f"{DATA_ROOT}/datasets/hoi_bench/v1/manifest.json")
    parser.add_argument("--decoded-manifest", default=f"{DATA_ROOT}/datasets/hoi_bench/decoded_mano_v1/manifest.json")
    parser.add_argument("--task-dir", default=f"{DATA_ROOT}/datasets/fr3_sharpa_bench/staging_v2/tasks")
    parser.add_argument("--output-dir", default=f"{OUTPUT_ROOT}/motion_v2")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--clip-id", action="append")
    args = parser.parse_args()
    source = load_json(Path(args.source_manifest))
    decoded = load_json(Path(args.decoded_manifest))
    if not decoded["complete"] or decoded["source_manifest_sha256"] != digest(args.source_manifest):
        raise ValueError("Incomplete or mismatched decoded manifest")
    lookup = {entry["clip_id"]: entry for entry in decoded["clips"]}
    entries = [e for e in source["clips"] if not args.clip_id or e["clip_id"] in args.clip_id]
    if not entries or (args.clip_id and set(args.clip_id) != {e["clip_id"] for e in entries}):
        raise ValueError("Unknown or empty clip selection")
    work = [(entry, lookup[entry["clip_id"]], decoded, args.task_dir, args.output_dir, .02) for entry in entries]
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as pool:
        results = list(pool.map(prepare_clip, work))
    manifest = {"schema_version": VERSION, "source_manifest_path": args.source_manifest,
                "source_manifest_sha256": digest(args.source_manifest), "decoded_manifest_path": args.decoded_manifest,
                "decoded_manifest_sha256": digest(args.decoded_manifest), "clip_count": len(results),
                "complete_source_collection": len(results) == len(source["clips"]),
                "total_samples": sum(row["sample_count"] for row in results), "clips": results,
                "qualification": "kinematic preparation; no physical benchmark success claim"}
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    write_json(Path(args.output_dir)/"manifest.json", manifest)
    print(json.dumps({"manifest": str(Path(args.output_dir)/"manifest.json"), "clip_count": len(results),
                      "total_samples": manifest["total_samples"],
                      "all_ik_converged_clips": sum(r["reachability"]["all_sampled_arm_ik_converged"] for r in results)}))


if __name__ == "__main__":
    main()
