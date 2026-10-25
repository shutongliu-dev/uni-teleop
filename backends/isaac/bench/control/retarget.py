"""Offline MANO-quaternion baseline; separate from the decoded-keypoint pipeline.

``prepare_motion`` uses ``keypoint_retarget``. This importable alternative is
retained for explicit reference comparisons, not current Agent execution.

MANO finger rotation magnitudes form a bounded flexion prior. This deliberately
does not claim fingertip/contact preservation or decode a licensed MANO mesh.
Wrist translation/rotation and FR3 URDF IK supply dense motion targets. A learned
System 0 or calibrated MANUS retargeter can replace this controller at the same
29-joint boundary while keeping the task/evaluator fixed.
"""
from __future__ import annotations
from dataclasses import asdict, dataclass
from pathlib import Path
import numpy as np
from bench.geometry import quaternion_matrix, transform
from .kinematics import URDFKinematics

VERSION = "mano_rotation_magnitude_fr3_ik_v1"
# MANO original order: root, index, middle, little, ring, thumb.
# Verified in manotorch/manolayer.py original joint-order diagram.
FINGER_INDICES = {"index": (1, 2, 3), "middle": (4, 5, 6),
                  "pinky": (7, 8, 9), "ring": (10, 11, 12), "thumb": (13, 14, 15)}


@dataclass(frozen=True)
class RetargetConfig:
    version: str = VERSION
    control_dt_s: float = .02
    arm_max_velocity_rad_s: float = 1.0
    hand_max_velocity_rad_s: float = 3.0
    ik_position_tolerance_m: float = .003
    ik_rotation_tolerance_rad: float = .06
    ik_rotation_weight_m: float = .12
    hand_flexion_gain: float = 1.0
    abduction_policy: str = "zero_no_anatomical_axis_estimate"
    wrist_alignment: str = "initial_reference_rotation_to_home_flange_fixed_per_clip"
    quality: str = "diagnostic_surrogate_not_contact_preserving"


class MotionReferenceController:
    def __init__(self, assets_root: str | Path, initial_joints: dict[str, float],
                 source_to_bench, reference_initial_wrist_quat,
                 *, mount_transform=None, config: RetargetConfig | None = None):
        root = Path(assets_root)
        self.arm = URDFKinematics(root / "robots/franka/fr3/fr3_no_hand.urdf")
        self.hand = URDFKinematics(root / "robots/sharpa/wave_01/derived/right_sharpa_wave_with_flange.urdf")
        self.config = config or RetargetConfig()
        self.previous = {name: float(initial_joints[name]) for name in self.arm.names+self.hand.names}
        self.source_to_bench = np.asarray(source_to_bench, dtype=float)
        self.mount = np.eye(4) if mount_transform is None else np.asarray(mount_transform, dtype=float)
        wrist_home = self.arm.forward(self.previous)["fr3_link8"] @ self.mount
        r_initial = self.source_to_bench[:3, :3] @ quaternion_matrix(reference_initial_wrist_quat)
        self.wrist_alignment = r_initial.T @ wrist_home[:3, :3]

    def hand_targets(self, quaternions):
        quats = np.asarray(quaternions, dtype=float)
        if quats.shape != (16, 4) or not np.isfinite(quats).all():
            raise ValueError("Expected finite MANO [16,4] quaternion annotation")
        norms = np.linalg.norm(quats, axis=1)
        if np.any(norms < 1e-8):
            raise ValueError("Zero MANO quaternion")
        q = quats / norms[:, None]
        # abs(w) makes q and -q equivalent and bounds rotation to [0,pi].
        angles = 2 * np.arctan2(np.linalg.norm(q[:, 1:], axis=1), np.abs(q[:, 0]))
        angles *= self.config.hand_flexion_gain
        result = {j.name: float(np.clip(0., j.lower, j.upper)) for j in self.hand.active}
        for finger, idx in FINGER_INDICES.items():
            suffixes = ("CMC_FE", "MCP_FE", "IP") if finger == "thumb" else ("MCP_FE", "PIP", "DIP")
            for index, suffix in zip(idx, suffixes, strict=True):
                result[f"right_{finger}_{suffix}"] = float(angles[index])
        return {j.name: float(np.clip(result[j.name], j.lower, j.upper)) for j in self.hand.active}

    def action(self, wrist_source, mano_quaternions, *, duration_s=None):
        duration_s = self.config.control_dt_s if duration_s is None else float(duration_s)
        if not 0 < duration_s <= .2:
            raise ValueError("Control duration must lie in (0,.2]")
        source_wrist = transform(wrist_source, quaternion_matrix(mano_quaternions[0]))
        desired_wrist = self.source_to_bench @ source_wrist
        desired_wrist[:3, :3] = desired_wrist[:3, :3] @ self.wrist_alignment
        desired_flange = desired_wrist @ np.linalg.inv(self.mount)
        ik = self.arm.inverse(desired_flange, "fr3_link8", self.previous,
            position_tolerance=self.config.ik_position_tolerance_m,
            rotation_tolerance=self.config.ik_rotation_tolerance_rad,
            rotation_weight=self.config.ik_rotation_weight_m)
        raw = {**ik["joint_positions_rad"], **self.hand_targets(mano_quaternions)}
        targets = {}
        for joint in self.arm.active+self.hand.active:
            cap = self.config.arm_max_velocity_rad_s if joint.name.startswith("fr3_") else self.config.hand_max_velocity_rad_s
            delta = min(cap, joint.velocity)*duration_s
            targets[joint.name] = float(np.clip(raw[joint.name], self.previous[joint.name]-delta,
                                                self.previous[joint.name]+delta))
        self.previous = targets
        return {"joint_positions_rad": targets, "duration_s": duration_s}, {
            "ik": {k:v for k,v in ik.items() if k != "joint_positions_rad"},
            "desired_wrist_T_world": desired_wrist.tolist(), "retarget_version": VERSION}

    def provenance(self):
        return {**asdict(self.config), "wrist_alignment_matrix": self.wrist_alignment.tolist(),
                "finger_order": FINGER_INDICES,
                "mano_model_used": False, "mano_betas_used": False,
                "source": "https://github.com/lixiny/manotorch/blob/master/manotorch/manolayer.py"}
