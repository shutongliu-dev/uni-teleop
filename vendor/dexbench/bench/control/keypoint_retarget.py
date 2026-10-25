"""MANO-keypoint to Sharpa geometry retargeting, with explicit morphology mapping.

The human palm basis sets hand orientation. Each human phalanx direction is
mapped to the corresponding robot phalanx length, then bounded geometric IK fits
the robot's fingertip/interphalangeal positions. This needs decoded MANO joints;
it never substitutes guessed keypoints for missing model weights.
"""
from __future__ import annotations
from dataclasses import asdict, dataclass
from pathlib import Path
import numpy as np
from bench.geometry import transform
from .kinematics import URDFKinematics

VERSION = "mano21_sharpa_bone_direction_ik_v1"
FINGERS = ("thumb", "index", "middle", "ring", "pinky")
HUMAN_INDICES = {name: list(range(1+4*i, 5+4*i)) for i,name in enumerate(FINGERS)}
ROBOT_LINKS = {name: [f"right_{name}_{suffix}" for suffix in
                     (("CMC_VL", "MCP_VL", "DP", "fingertip") if name == "thumb"
                      else ("PP", "MP", "DP", "fingertip"))] for name in FINGERS}


def unit(vector):
    length = np.linalg.norm(vector)
    if length < 1e-8:
        raise ValueError("Degenerate keypoint geometry")
    return vector/length


def palm_basis(wrist, index_mcp, middle_mcp, pinky_mcp):
    z = unit(middle_mcp-wrist)
    across = index_mcp-pinky_mcp
    y = unit(across-z*np.dot(across, z))
    x = unit(np.cross(y, z))
    return np.column_stack((x, y, z))


@dataclass(frozen=True)
class KeypointRetargetConfig:
    version: str = VERSION
    hand_iterations: int = 24
    damping: float = .008
    pose_regularization: float = .00002
    hand_step_limit_rad: float = .15
    maximum_hand_velocity_rad_s: float = 3.
    maximum_arm_velocity_rad_s: float = 1.
    wrist_anchor_link: str = "right_hand_C_MC"
    wrist_anchor_status: str = "fixed anatomical proxy, not hardware glove calibration"
    morphology: str = "preserve phalanx directions, use robot segment lengths and MCP anchors"


class KeypointRetargeter:
    def __init__(self, assets_root, initial_joints, source_to_bench, *, mount_transform=None, config=None):
        root = Path(assets_root)
        self.config = config or KeypointRetargetConfig()
        self.arm = URDFKinematics(root/"robots/franka/fr3/fr3_no_hand.urdf")
        self.hand = URDFKinematics(root/"robots/sharpa/wave_01/derived/right_sharpa_wave_with_flange.urdf")
        self.source_to_bench = np.asarray(source_to_bench, dtype=float)
        self.mount = np.eye(4) if mount_transform is None else np.asarray(mount_transform)
        self.previous = dict(initial_joints)
        zero = {j.name: float(np.clip(0, j.lower, j.upper)) for j in self.hand.active}
        neutral = self.hand.forward(zero)
        self.anchor = neutral[self.config.wrist_anchor_link][:3, 3].copy()
        self.robot_palm = palm_basis(self.anchor, neutral[ROBOT_LINKS["index"][0]][:3, 3],
                                      neutral[ROBOT_LINKS["middle"][0]][:3, 3],
                                      neutral[ROBOT_LINKS["pinky"][0]][:3, 3])
        self.segment_lengths = {}
        self.mcp_positions = {}
        self.fitted_links = []
        for finger, links in ROBOT_LINKS.items():
            p = np.array([neutral[name][:3, 3] for name in links])
            self.mcp_positions[finger] = p[0]
            self.segment_lengths[finger] = np.linalg.norm(np.diff(p, axis=0), axis=1)
            self.fitted_links.extend(links[1:])
        by_child = {j.child: j for j in self.hand.joints}
        self.ancestors = {}
        for link in self.fitted_links:
            ancestors, current = set(), link
            while current in by_child:
                joint = by_child[current]
                ancestors.add(joint.name)
                current = joint.parent
            self.ancestors[link] = ancestors

    def target_geometry(self, human_points):
        points = np.asarray(human_points, dtype=float)
        if points.shape != (21, 3) or not np.isfinite(points).all():
            raise ValueError("Decoded finite 21x3 MANO joints are required")
        points = points @ self.source_to_bench[:3, :3].T + self.source_to_bench[:3, 3]
        human_palm = palm_basis(points[0], points[5], points[9], points[17])
        robot_rotation = human_palm @ self.robot_palm.T
        desired_root = transform(points[0] - robot_rotation @ self.anchor, robot_rotation)
        targets = []
        for finger, indices in HUMAN_INDICES.items():
            directions = np.diff(points[indices], axis=0) @ robot_rotation
            directions = np.array([unit(d) for d in directions])
            positions = self.mcp_positions[finger] + np.cumsum(
                directions*self.segment_lengths[finger][:, None], axis=0)
            targets.extend(positions)
        return desired_root, np.asarray(targets)

    def fit_hand(self, target_points):
        q0 = np.array([self.previous[n] for n in self.hand.names])
        q = q0.copy()
        lower = np.array([j.lower for j in self.hand.active])
        upper = np.array([j.upper for j in self.hand.active])
        regularization = self.config.pose_regularization
        for _ in range(self.config.hand_iterations):
            poses = self.hand.forward(dict(zip(self.hand.names, q)))
            current = np.array([poses[n][:3, 3] for n in self.fitted_links])
            error = (target_points-current).ravel()
            jac = np.zeros((len(error), len(q)))
            for col, joint in enumerate(self.hand.active):
                jpose = poses[joint.parent] @ joint.origin
                axis = jpose[:3, :3] @ joint.axis
                for row, link in enumerate(self.fitted_links):
                    if joint.name in self.ancestors[link]:
                        jac[3*row:3*row+3, col] = np.cross(axis, current[row]-jpose[:3, 3])
            lhs = jac.T @ jac + (self.config.damping**2+regularization)*np.eye(len(q))
            dq = np.linalg.solve(lhs, jac.T @ error - regularization*(q-q0))
            q = np.clip(q + np.clip(dq, -self.config.hand_step_limit_rad, self.config.hand_step_limit_rad), lower, upper)
            if np.linalg.norm(dq) < 1e-5:
                break
        poses = self.hand.forward(dict(zip(self.hand.names, q)))
        fitted = np.array([poses[n][:3, 3] for n in self.fitted_links])
        return dict(zip(self.hand.names, q.tolist())), float(np.sqrt(np.mean(np.sum((fitted-target_points)**2, axis=1))))

    def action(self, human_points, duration_s=.02):
        if not 0 < duration_s <= .2:
            raise ValueError("Duration must lie in (0,.2]")
        root_target, target_points = self.target_geometry(human_points)
        hand, fit_error = self.fit_hand(target_points)
        ik = self.arm.inverse(root_target @ np.linalg.inv(self.mount), "fr3_link8", self.previous)
        raw = {**hand, **ik["joint_positions_rad"]}
        targets = {}
        for joint in self.arm.active+self.hand.active:
            cap = self.config.maximum_arm_velocity_rad_s if joint.name.startswith("fr3_") else self.config.maximum_hand_velocity_rad_s
            step = duration_s*min(cap, joint.velocity)
            targets[joint.name] = float(np.clip(raw[joint.name], self.previous[joint.name]-step,
                                               self.previous[joint.name]+step))
        self.previous = targets
        return {"joint_positions_rad": targets, "duration_s": duration_s}, {
            "version": VERSION, "hand_geometry_rmse_m": fit_error,
            "ik": {k:v for k,v in ik.items() if k != "joint_positions_rad"},
            "desired_robot_root_T_world": root_target.tolist()}

    def provenance(self):
        return {**asdict(self.config), "mano_decoded_joints_required": True,
                "human_joint_order": ["wrist"] + [f"{name}_{i}" for name in FINGERS for i in range(4)],
                "contact_preservation_guaranteed": False,
                "robot_segment_lengths_m": {k:v.tolist() for k,v in self.segment_lengths.items()}}
