"""Inspectable offline agent decisions for the mocap-reference baseline."""
from __future__ import annotations

from bench.geometry import rotation_distance


def build_plan(task, source_metadata):
    """Record the agent's actual policy choice, without implying online inference."""
    import numpy as np
    initial = np.asarray(task["initial_object_T_world"])
    goal = np.asarray(task["goal_object_T_world"])
    return {
        "schema_version": "astra_offline_motion_plan_v1",
        "task_id": task["task_id"],
        "planner": {"identity": "Astra", "execution": "offline authored deterministic plan",
                    "online_model_api_called": False, "trained_rl": False},
        "observations": {
            "source": task["reference"]["source"],
            "source_description": source_metadata.get("description", {}),
            "source_primitive": source_metadata.get("annotation", {}),
            "target_object_id": task["target_object_id"],
            "initial_object_T_world": initial.tolist(),
            "goal_object_T_world": goal.tolist(),
            "endpoint_translation_m": float(np.linalg.norm(goal[:3, 3]-initial[:3, 3])),
            "endpoint_rotation_rad": rotation_distance(goal[:3, :3], initial[:3, :3]),
        },
        "decision": {
            "goal": "Reproduce the complete right-hand rearrange primitive and evaluate physical object outcome.",
            "controller": "oracle decoded MANO keypoints -> morphology mapping -> bounded FR3/Sharpa IK -> rate-limited named joint targets",
            "rationale": "The audited source supplies exact licensed MANO decoded joints and complete object SE3; geometric fitting makes reference use inspectable before contact simulation.",
            "reset": "Fit the first decoded hand frame once as an explicit robot reset; initialize objects only from the frozen task.",
            "rollout": "Interpolate the complete source at 50 Hz, apply joint targets, and let contact physics determine object motion.",
            "reference_access": "oracle reference baseline",
            "object_reference_used_as_action": False,
            "feedback": "No online state feedback or replanning; precomputed joint target replay.",
        },
        "qualification": "kinematics only; physical success requires separate simulator rollout and evaluator",
        "limitations": ["No contact-preservation guarantee across hand morphologies.",
                        "Wrist anchor and flange mount are provisional and uncalibrated.",
                        "Local IK failures are diagnostic, not proofs of global unreachability.",
                        "Reset may contain collisions; contact-valid initialization must be assessed in simulation."],
    }
