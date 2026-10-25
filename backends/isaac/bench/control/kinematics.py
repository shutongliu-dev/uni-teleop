"""URDF-derived FK and bounded damped-least-squares FR3 inverse kinematics.

The result is a controller target, never a write to simulated body poses.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import time
import xml.etree.ElementTree as ET
import numpy as np
from bench.geometry import matrix_rotvec, rotvec_matrix, rpy_matrix, transform


@dataclass(frozen=True)
class Joint:
    name: str
    parent: str
    child: str
    kind: str
    origin: np.ndarray
    axis: np.ndarray
    lower: float
    upper: float
    velocity: float


class URDFKinematics:
    def __init__(self, path: str | Path):
        root = ET.parse(path).getroot()
        self.joints = []
        for j in root.findall("joint"):
            origin, axis, limit = j.find("origin"), j.find("axis"), j.find("limit")
            xyz = np.fromstring(origin.get("xyz", "0 0 0") if origin is not None else "0 0 0", sep=" ")
            rpy = np.fromstring(origin.get("rpy", "0 0 0") if origin is not None else "0 0 0", sep=" ")
            axis_v = np.fromstring(axis.get("xyz", "1 0 0") if axis is not None else "1 0 0", sep=" ")
            norm = np.linalg.norm(axis_v)
            if norm < 1e-12:
                if j.get("type") != "fixed":
                    raise ValueError(f"Zero axis for {j.get('name')}")
                axis_v = np.array([1., 0., 0.])
            else:
                axis_v /= norm
            self.joints.append(Joint(j.get("name"), j.find("parent").get("link"),
                j.find("child").get("link"), j.get("type"), transform(xyz, rpy_matrix(rpy)), axis_v,
                float(limit.get("lower", "-inf")) if limit is not None else 0.,
                float(limit.get("upper", "inf")) if limit is not None else 0.,
                float(limit.get("velocity", "inf")) if limit is not None else 0.))
        self.active = [j for j in self.joints if j.kind in ("revolute", "continuous", "prismatic")]
        self.names = [j.name for j in self.active]
        children = {j.child for j in self.joints}
        roots = {link.get("name") for link in root.findall("link")} - children
        if len(roots) != 1:
            raise ValueError("Expected a single-root URDF tree")
        self.root = roots.pop()

    def forward(self, positions: dict[str, float], base=None):
        poses = {self.root: np.eye(4) if base is None else np.asarray(base, dtype=float)}
        pending = self.joints.copy()
        while pending:
            progressed = False
            for j in pending.copy():
                if j.parent not in poses:
                    continue
                q = float(positions.get(j.name, 0.))
                delta = np.eye(4)
                if j.kind in ("revolute", "continuous"):
                    delta[:3, :3] = rotvec_matrix(j.axis * q)
                elif j.kind == "prismatic":
                    delta[:3, 3] = j.axis * q
                elif j.kind != "fixed":
                    raise ValueError(f"Unsupported URDF joint type {j.kind}")
                poses[j.child] = poses[j.parent] @ j.origin @ delta
                pending.remove(j)
                progressed = True
            if not progressed:
                raise ValueError("Disconnected/cyclic URDF")
        return poses

    def inverse(self, target, tip, initial, *, iterations=60, position_tolerance=.002,
                rotation_tolerance=.04, rotation_weight=.2, damping=.025,
                multi_starts=0, return_trace=False, deadline_ns=None):
        """Bounded monotone Levenberg-Marquardt with deterministic optional restarts.

        Every accepted step lowers ||dp||² + rotation_weight² ||log(dR)||².
        Failed searches keep the best state, including the supplied initial pose.
        Restarts run only after the warm start fails; they do not establish global
        feasibility or account for collisions. Tolerances are unchanged.
        An optional monotonic deadline cancels at an evaluation boundary with
        TimeoutError; it never returns an unverified intermediate target.
        """
        target = np.asarray(target, dtype=float)
        if target.shape != (4, 4) or not np.isfinite(target).all():
            raise ValueError("IK requires finite 4x4 target")
        if (iterations < 1 or multi_starts < 0 or damping <= 0 or rotation_weight <= 0 or
                position_tolerance <= 0 or rotation_tolerance <= 0):
            raise ValueError("Invalid IK iteration, damping or tolerance configuration")
        q_initial = np.array([initial[n] for n in self.names], dtype=float)
        if not np.isfinite(q_initial).all():
            raise ValueError("IK requires finite initial joint positions")
        lo, hi = np.array([j.lower for j in self.active]), np.array([j.upper for j in self.active])
        q_initial = np.clip(q_initial, lo, hi)
        ancestors, child = set(), tip
        by_child = {j.child: j for j in self.joints}
        while child in by_child:
            j = by_child[child]
            ancestors.add(j.name)
            child = j.parent

        def evaluate(q):
            if deadline_ns is not None and time.monotonic_ns() >= deadline_ns:
                raise TimeoutError('IK time budget exhausted')
            poses = self.forward(dict(zip(self.names, q)))
            current = poses[tip]
            dp = target[:3, 3]-current[:3, 3]
            dr = matrix_rotvec(target[:3, :3] @ current[:3, :3].T)
            error = np.r_[dp, rotation_weight*dr]
            return poses, error, float(error @ error)

        poses, error, best_cost = evaluate(q_initial)
        initial_cost = best_cost
        best_q, best_error = q_initial.copy(), error.copy()
        seeds = [q_initial]
        if multi_starts:
            # Fixed PRNG seed and bounded joint-space seeds make fallback reproducible.
            finite_lo, finite_hi = np.where(np.isfinite(lo), lo, -np.pi), np.where(np.isfinite(hi), hi, np.pi)
            rng = np.random.default_rng(0)
            seeds.append((finite_lo+finite_hi)/2)
            seeds.extend(rng.uniform(finite_lo, finite_hi) for _ in range(multi_starts-1))
        traces, total_iterations, accepted_steps, rejected_trials = [], 0, 0, 0
        for seed in seeds:
            q = seed.copy()
            poses, error, cost = evaluate(q)
            trace = [cost]
            traces.append(trace)
            current_damping = damping
            for _ in range(iterations):
                total_iterations += 1
                if cost < best_cost:
                    best_q, best_error, best_cost = q.copy(), error.copy(), cost
                if (np.linalg.norm(error[:3]) <= position_tolerance and
                        np.linalg.norm(error[3:])/rotation_weight <= rotation_tolerance):
                    # A converged iterate takes precedence over an unconverged lower
                    # weighted objective, since the two tolerances are independent.
                    best_q, best_error, best_cost = q.copy(), error.copy(), cost
                    break
                current = poses[tip]
                jac = np.zeros((6, len(q)))
                for i, j in enumerate(self.active):
                    if j.name not in ancestors:
                        continue
                    joint_pose = poses[j.parent] @ j.origin
                    axis = joint_pose[:3, :3] @ j.axis
                    if j.kind == "prismatic":
                        jac[:3, i] = axis
                    else:
                        jac[:3, i] = np.cross(axis, current[:3, 3]-joint_pose[:3, 3])
                        jac[3:, i] = rotation_weight*axis
                accepted = False
                for _attempt in range(5):
                    free = np.ones(len(q), dtype=bool)
                    # Do not waste a descent direction pushing saturated joints
                    # outward; resolve DLS with those columns removed.
                    for _active in range(len(q)+1):
                        jf = jac[:, free]
                        dq = np.zeros(len(q))
                        dq[free] = jf.T @ np.linalg.solve(jf @ jf.T + current_damping**2*np.eye(6), error)
                        outward = ((q <= lo+1e-10) & (dq < 0)) | ((q >= hi-1e-10) & (dq > 0))
                        if not np.any(outward & free):
                            break
                        free &= ~outward
                    dq = np.clip(dq, -.12, .12)
                    for scale in (1., .5, .25, .125, .0625, .03125):
                        trial_q = np.clip(q+scale*dq, lo, hi)
                        trial_poses, trial_error, trial_cost = evaluate(trial_q)
                        if trial_cost < cost-1e-15:
                            q, poses, error, cost = trial_q, trial_poses, trial_error, trial_cost
                            trace.append(cost)
                            accepted_steps += 1
                            current_damping = max(1e-5, current_damping*.7)
                            accepted = True
                            break
                        rejected_trials += 1
                    if accepted:
                        break
                    current_damping = min(10., current_damping*4)
                if not accepted:
                    break
            # Last accepted update can occur on the final allowed iteration.
            if cost < best_cost:
                best_q, best_error, best_cost = q.copy(), error.copy(), cost
            if (np.linalg.norm(best_error[:3]) <= position_tolerance and
                    np.linalg.norm(best_error[3:])/rotation_weight <= rotation_tolerance):
                break
        pe = float(np.linalg.norm(best_error[:3]))
        re = float(np.linalg.norm(best_error[3:])/rotation_weight)
        result = {"joint_positions_rad": dict(zip(self.names, best_q.tolist())),
                  "position_error_m": pe, "rotation_error_rad": re,
                  "converged": pe <= position_tolerance and re <= rotation_tolerance,
                  "iterations": total_iterations,
                  "algorithm": "bounded_monotone_lm_best_iterate_v2",
                  "starts_attempted": len(traces), "accepted_steps": accepted_steps,
                  "rejected_trials": rejected_trials,
                  "initial_weighted_error_m": float(np.sqrt(initial_cost)),
                  "weighted_error_m": float(np.sqrt(best_cost))}
        if return_trace:
            result["accepted_objective_traces"] = traces
        return result
