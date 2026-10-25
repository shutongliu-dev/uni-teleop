"""Offline reference-trajectory metrics; not the native Agent task scorer.

The caller supplies interval contact booleans, without native substep sampling
evidence. These versioned HOI/cloth metrics must not certify v7 simple tasks;
the current worker loads its separate evaluator from the selected release.

HOI object trajectories are scoring references only. A controller cannot report
success itself. Neither this module nor reference.py has access to simulator
state setters. All thresholds/configuration must be frozen before evaluation.
"""
from __future__ import annotations
from dataclasses import asdict, dataclass
import numpy as np
from .geometry import rotation_distance

VERSION = "physical_object_goal_v2"
CLOTH_VERSION = "cloth_goal_v3"


@dataclass(frozen=True)
class RigidEvaluationConfig:
    version: str = VERSION
    position_tolerance_m: float = .03
    rotation_tolerance_rad: float = .35
    goal_dwell_s: float = .3
    minimum_motion_m: float = .02
    maximum_speed_m_s: float = 4.
    require_hand_object_contact: bool = True
    minimum_hand_contact_duration_s: float = .02


@dataclass(frozen=True)
class ClothEvaluationConfig:
    version: str = CLOTH_VERSION
    tolerance_m: float = .025
    goal_dwell_s: float = .3
    maximum_vertex_speed_m_s: float = 4.


def evaluate_rigid(times_s, object_poses, hand_object_contacts, *, goal_pose,
                   initial_pose, reference_times_s=None, reference_poses=None,
                   joint_limit_violation=False, config=None):
    config = config or RigidEvaluationConfig()
    times = np.asarray(times_s, dtype=float)
    poses = np.asarray(object_poses, dtype=float)
    contacts = np.asarray(hand_object_contacts, dtype=bool)
    if len(times) < 2 or poses.shape != (len(times), 4, 4) or contacts.shape != times.shape:
        raise ValueError("Need at least two corresponding time/pose/contact samples")
    if not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError("Rollout timestamps must be finite and strictly increasing")
    if not np.isfinite(poses).all():
        return {"success": False, "invalid_reason": "nonfinite_physics", "evaluator": asdict(config)}
    goal, initial = np.asarray(goal_pose), np.asarray(initial_pose)
    errors = np.linalg.norm(poses[:, :3, 3] - goal[:3, 3], axis=1)
    rotations = np.array([rotation_distance(p[:3, :3], goal[:3, :3]) for p in poses])
    speeds = np.linalg.norm(np.diff(poses[:, :3, 3], axis=0), axis=1) / np.diff(times)
    movement = np.linalg.norm(poses[:, :3, 3] - initial[:3, 3], axis=1)
    in_goal = (errors <= config.position_tolerance_m) & (rotations <= config.rotation_tolerance_rad)
    last_bad = np.flatnonzero(~in_goal)
    start = int(last_bad[-1]+1) if len(last_bad) else 0
    dwell = float(times[-1]-times[start]) if start < len(times) else 0.
    # The reset observation is not evidence of controller-caused interaction.
    contact_duration = float(np.sum(np.diff(times)*contacts[1:]))
    contact_ok = bool(np.any(contacts[1:]) and contact_duration+1e-10 >= config.minimum_hand_contact_duration_s)
    contact_ok = contact_ok or not config.require_hand_object_contact
    stable = bool(np.max(speeds) <= config.maximum_speed_m_s and not joint_limit_violation)
    success = bool(dwell >= config.goal_dwell_s and movement.max() >= config.minimum_motion_m and contact_ok and stable)
    result = {"success": success, "final_position_error_m": float(errors[-1]),
              "implementation_version": VERSION,
              "final_rotation_error_rad": float(rotations[-1]), "terminal_goal_dwell_s": dwell,
              "maximum_displacement_m": float(movement.max()),
              "maximum_speed_m_s": float(speeds.max()),
              "hand_object_contact_observed": bool(np.any(contacts[1:])),
              "post_reset_hand_contact_duration_s": contact_duration,
              "hand_object_contact_sample_fraction": float(np.mean(contacts)),
              "joint_limit_violation": bool(joint_limit_violation), "stable": stable,
              "evaluator": asdict(config)}
    if reference_times_s is not None and reference_poses is not None:
        reference_times = np.asarray(reference_times_s)
        reference = np.asarray(reference_poses)
        expected = np.column_stack([np.interp(times, reference_times, reference[:, i, 3]) for i in range(3)])
        result["reference_translation_rmse_m"] = float(np.sqrt(np.mean(np.sum((poses[:, :3, 3]-expected)**2, axis=1))))
        result["reference_tracking_used_for_success"] = False
    return result


def evaluate_cloth(times_s, vertices, hand_cloth_contacts, *, initial_vertices,
                   target_vertices, task, tolerance_m=.025, dwell_s=.3,
                   maximum_vertex_speed_m_s=4., joint_limit_violation=False):
    """Correspondence score prevents a crumpled ball from passing a half fold.

Targets are fixed desired positions for labelled cloth vertices. Contacts come
from solver pairs. Grasp quality additionally needs force/area diagnostics in
future versions; this score reports physical lift/drag/fold geometry only.
    """
    times, states = np.asarray(times_s), np.asarray(vertices)
    initial, target = np.asarray(initial_vertices), np.asarray(target_vertices)
    contact = np.asarray(hand_cloth_contacts, dtype=bool)
    if task not in ("cloth_drag", "cloth_lift", "cloth_fold_half"):
        raise ValueError(f"Unknown cloth task: {task}")
    if (len(times) < 2 or target.ndim != 2 or target.shape[1] != 3 or len(target) == 0
            or states.shape != (len(times), *target.shape) or target.shape != initial.shape):
        raise ValueError("Cloth trajectory/target shape mismatch")
    if contact.shape != times.shape or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError("Contact/time alignment invalid")
    if not np.isfinite(initial).all() or not np.isfinite(target).all():
        raise ValueError("Initial and target vertices must be finite")
    if (not np.isfinite([tolerance_m, dwell_s, maximum_vertex_speed_m_s]).all()
            or tolerance_m <= 0 or dwell_s < 0 or maximum_vertex_speed_m_s <= 0):
        raise ValueError("Cloth thresholds must be finite and positive")
    if not np.isfinite(states).all():
        return {"success": False, "invalid_reason": "nonfinite_physics", "version": CLOTH_VERSION}
    errors = np.sqrt(np.mean(np.sum((states-target)**2, axis=-1), axis=-1))
    good = errors <= tolerance_m
    last_bad = np.flatnonzero(~good)
    start = int(last_bad[-1]+1) if len(last_bad) else 0
    dwell = float(times[-1]-times[start]) if start < len(times) else 0.
    movement = float(np.max(np.linalg.norm(states.mean(axis=1)-initial.mean(axis=0), axis=1)))
    moved = movement >= .02
    if task == "cloth_fold_half":
        moved = bool(np.max(np.linalg.norm(states-initial, axis=-1)) >= .05)
    speeds = np.linalg.norm(np.diff(states, axis=0), axis=-1) / np.diff(times)[:, None]
    stable = bool(np.max(speeds) <= maximum_vertex_speed_m_s and not joint_limit_violation)
    return {"success": bool(dwell >= dwell_s and contact[1:].any() and moved and stable),
            "version": CLOTH_VERSION, "task": task, "final_vertex_rmse_m": float(errors[-1]),
            "terminal_goal_dwell_s": dwell, "maximum_centroid_displacement_m": movement,
            "maximum_centroid_lift_m": float(np.max(states[:, :, 2].mean(axis=1)-initial[:, 2].mean())),
            "hand_cloth_contact_observed": bool(contact[1:].any()),
            "tolerance_m": tolerance_m, "required_dwell_s": dwell_s,
            "maximum_vertex_speed_m_s": float(np.max(speeds)),
            "allowed_vertex_speed_m_s": maximum_vertex_speed_m_s,
            "joint_limit_violation": bool(joint_limit_violation), "stable": stable,
            "stable_force_closure_grasp_claimed": False}
