"""Task-independent FR3 + dexterous-hand position executor; no learned policy or contacts.

The wrist is the right_hand_flange frame: the hand frame of the pinned native
assembly (fr3_link8 itself for flange_axis_centred_v1; rotated 180 deg about
the flange axis for flange_axis_centred_v2, 1.9.0). Fingertip coordinates are
URDF virtual tip origins in this frame, not measured skin/contact locations.
This executor does not plan collision-free paths or certify a stable grasp.
The default embodiment is the right SHARPA W01; other registered hands are
selected with ``hand_id`` (see bench.control.embodiment).
"""
from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass, field
import copy
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np

from bench.protocol.policies import PolicyStep
from bench.protocol.schemas import (ActionDiagnostic, DirectControlHandoff, DurationPreflight, EffectiveGeometryTolerances, GeometryTolerancePolicy,
                                 MEASURED_JOINT_LIMIT_SAFETY_BAND_RAD, PolicyDescriptor, RobotExecutionState, RobotJointCommand)
from bench.control.reference import ReferenceInfeasible, bounded_reference, stop_reference
from bench.control.hand_posture import HandPosturePrior
from bench.geometry import matrix_quaternion, quaternion_matrix, rotation_distance, transform
from bench.control.kinematics import URDFKinematics


FINGERTIP_LINKS = {name: f'right_{name}_fingertip' for name in ('thumb', 'index', 'middle', 'ring')}
FINGERTIP_LINKS['little'] = 'right_pinky_fingertip'
HAND_FRAME = 'right_hand_flange'
ARM_TIP = 'fr3_link8'
CONTROL_DT = .02
# Contact can push a measured joint past its URDF bound. The executor accepts
# every state the service still treats as physically valid: inside this band the
# measured posture is clamped only for IK seeds and initial references. Raw
# feedback is unchanged and emitted targets stay strictly inside the bounds.
MEASURED_JOINT_LIMIT_TOLERANCE_RAD = MEASURED_JOINT_LIMIT_SAFETY_BAND_RAD
# Measured excursions above this are reported with a measured_limit_projection diagnostic.
MEASURED_LIMIT_REPORT_RAD = 1.e-3
PROJECT = Path(__file__).resolve().parents[2]


class GeometryIKError(ValueError):
    """A requested target did not satisfy IK tolerances; no target is executed."""
    def __init__(self, message, *, diagnostic=None):
        super().__init__(message)
        self.public_diagnostic = diagnostic or ActionDiagnostic(code='ik_solution_unverified')


def _finite_vector(value, size, name):
    array = np.asarray(value, dtype=float)
    if array.shape != (size,) or not np.isfinite(array).all():
        raise ValueError(f'{name} must be a finite {size}-vector')
    return array


def _pose_matrix(pose):
    value = pose.model_dump() if hasattr(pose, 'model_dump') else pose
    if not isinstance(value, dict) or set(value) != {'frame_id', 'position_m', 'quaternion_wxyz'}:
        raise ValueError('Wrist pose requires frame_id, position_m and quaternion_wxyz')
    if value['frame_id'] != 'world':
        raise ValueError('The wrist target must use the world frame')
    xyz = _finite_vector(value['position_m'], 3, 'Wrist position')
    quat = _finite_vector(value['quaternion_wxyz'], 4, 'Wrist quaternion')
    if abs(float(quat @ quat) - 1.) > 1e-5:
        raise ValueError('Wrist quaternion must be unit wxyz')
    return transform(xyz, quaternion_matrix(quat))


@dataclass(frozen=True)
class ExecutorLimits:
    arm_velocity_rad_s: float = .65
    hand_velocity_rad_s: float = 1.
    arm_acceleration_rad_s2: float = 2.
    hand_acceleration_rad_s2: float = 3.
    wrist_position_tolerance_m: float = .003
    wrist_rotation_tolerance_rad: float = .04
    fingertip_tolerance_m: float = .0015
    # Existing fixed preflight factor, exposed without adding a new tuning knob.
    ik_verification_scale: float = field(default=.5, init=False)
    joint_tracking_tolerance_rad: float = .03
    tracking_gain: float = .5
    integral_gain_per_s: float = 2.
    maximum_tracking_offset_rad: float = .04
    reference_tracking_gain_per_s: float = 8.
    settle_margin_s: float = .2
    # A move completes only once the measured wrist has settled below this speed.
    wrist_settle_speed_m_s: float = .002
    # The arm integral accumulates only while every arm joint is this still, so
    # it learns steady loads (gravity, contact) but not transient settling lag.
    arm_integral_still_speed_rad_s: float = .005
    # After the nominal ends, an objective that improves by less than this
    # fraction of its tolerance over the window, while the drive is no longer
    # changing, is stalled (normally blocked by contact) and ends early.
    stall_window_s: float = .3
    stall_progress_fraction: float = .05
    stall_drive_change_rad: float = .002
    # Closest IK solution accepted for a fingertip target just outside reach;
    # completion is still judged at the request with fingertip_tolerance_m.
    fingertip_ik_acceptance_m: float = .001
    # Within an action, a measured joint-limit excursion may not grow past
    # max(floor, excursion at begin + growth): the action fails instead.
    hand_excursion_floor_rad: float = .005
    hand_excursion_growth_rad: float = .002
    arm_excursion_floor_rad: float = .001
    arm_excursion_growth_rad: float = .001

    def __post_init__(self):
        for name, value in asdict(self).items():
            if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f'{name} must be finite and positive')


class Fr3SharpaGeometry:
    """URDF geometry and bounded IK using static robot configuration only.

    The class name is historical: ``hand_id`` selects any registered FR3 hand
    embodiment (default right SHARPA W01, whose path is unchanged).
    """
    def __init__(self, assets_root, base_position_m, base_quaternion_wxyz=(1., 0., 0., 0.), *, hand_posture_prior=None,
                 hand_id=None, assembly_version=None, embodiment=None):
        from bench.control.embodiment import load_embodiment
        self.embodiment = embodiment or load_embodiment(assets_root, hand_id, assembly_version)
        e = self.embodiment
        # arm_tip -> public hand frame; None keeps the original identity code path.
        self.wrist_offset = None if np.array_equal(e.wrist_offset, np.eye(4)) else np.asarray(e.wrist_offset, dtype=float)
        self.robot_id = e.robot_id
        self.arm_path, self.hand_path = e.asset_paths
        self.arm = e.arm
        self.hand = e.hand
        self.arm_tip, self.frame_id = e.arm_tip, e.frame_id
        self.fingertip_links = dict(e.tips)
        self.tip_offsets = {finger: np.asarray(offset, dtype=float) for finger, offset in e.tip_offsets.items()}
        self.hand_posture_prior = hand_posture_prior or e.posture_prior
        if len(self.arm.names) != 7 or (e.sharpa_joint_semantics and (self.hand.root != e.frame_id or len(self.hand.names) != 22)):
            raise ValueError('Executor requires a pinned FR3 + registered hand embodiment')
        quat = _finite_vector(base_quaternion_wxyz, 4, 'Robot base quaternion')
        if abs(float(quat @ quat) - 1.) > 1e-5:
            raise ValueError('Robot base quaternion must be unit wxyz')
        self.base = transform(_finite_vector(base_position_m, 3, 'Robot base position'), quaternion_matrix(quat))
        # World pose of the arm's own base link (a bimanual frame offsets each arm).
        self.arm_base = self.base if np.array_equal(e.arm_base, np.eye(4)) else self.base @ e.arm_base
        self.names = self.arm.names + self.hand.names
        self.joints = self.arm.active + self.hand.active
        self.lower = np.array([j.lower for j in self.joints])
        self.upper = np.array([j.upper for j in self.joints])
        self.by_name = {joint.name: joint for joint in self.joints}
        children = {joint.child: joint for joint in self.hand.joints}
        chains = {}
        for finger, tip in self.fingertip_links.items():
            chain, child = [], tip
            if child not in children:
                raise ValueError(f'URDF fingertip frame is missing: {tip}')
            while child in children:
                joint = children[child]
                if joint.name in self.hand.names:
                    chain.append(joint)
                child = joint.parent
            if child != self.hand.root or not chain:
                raise ValueError(f'URDF fingertip is disconnected: {tip}')
            chains[finger] = list(reversed(chain))
        # Joints on every finger's path (e.g. a hand wrist) belong to no finger:
        # they are maintained like an unselected chain unless a posture names them.
        shared = (set.intersection(*({j.name for j in chain} for chain in chains.values()))
                  if len(chains) > 1 else set())
        self.base_joints = [joint for joint in self.hand.active if joint.name in shared]
        self.finger_joints = {finger: [j for j in chain if j.name not in shared] for finger, chain in chains.items()}
        inventory = [j.name for chain in self.finger_joints.values() for j in chain] + [j.name for j in self.base_joints]
        if len(inventory) != len(set(inventory)) or set(inventory) != set(self.hand.names):
            raise ValueError('Finger chains and hand base joints must partition the independent hand joints')
        # Passive mimic followers on each finger's path add to their leader's Jacobian column.
        followers = getattr(self.hand, 'followers', {})
        self._chain_followers = {}
        for finger, tip in self.fingertip_links.items():
            path, child = set(), tip
            while child in children:
                path.add(children[child].name)
                child = children[child].parent
            self._chain_followers[finger] = {leader: [(joint, multiplier) for joint, multiplier in rows if joint.name in path]
                                             for leader, rows in followers.items()}
        self.asset_sha256 = {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                             for path in (self.arm_path, self.hand_path)}

    def _tip(self, poses, finger):
        """Public fingertip point: the tip frame origin, plus a pinned offset for hands without one."""
        pose = poses[self.fingertip_links[finger]]
        offset = self.tip_offsets.get(finger)
        return pose[:3, 3] if offset is None else pose[:3, :3] @ offset + pose[:3, 3]

    def checked_joints(self, positions, *, enforce_limits=False):
        if not isinstance(positions, dict) or set(positions) != set(self.names):
            raise ValueError('Supply exactly the public FR3 and SHARPA joint positions')
        q = _finite_vector([positions[name] for name in self.names], len(self.names), 'Joint positions')
        if enforce_limits and ((q < self.lower).any() or (q > self.upper).any()):
            raise ValueError('Requested joint reference is outside URDF limits')
        return dict(zip(self.names, q.tolist()))

    def measured_reference(self, positions):
        """Clamp a measured posture into the URDF bounds for internal references.

        Raw proprioception/FK and feedback remain unchanged. Explicit requested
        postures still pass the strict command-limit check, without tolerance.
        A state beyond the service safety band is invalid, not a control input.
        """
        measured = self.checked_joints(positions)
        q = np.array([measured[name] for name in self.names])
        violation = np.maximum(self.lower - q, q - self.upper)
        index = int(np.argmax(violation))
        if violation[index] > MEASURED_JOINT_LIMIT_TOLERANCE_RAD:
            raise GeometryIKError('Measured joint position exceeds the joint-limit safety band',
                diagnostic=ActionDiagnostic(code='invalid_robot_state', joint_name=self.names[index],
                                            joint_error_rad=float(violation[index])))
        bounded = np.clip(q, self.lower, self.upper)
        return dict(zip(self.names, bounded.tolist()))

    def public_geometry(self, joint_positions_rad):
        q = self.checked_joints(joint_positions_rad)
        wrist = self.arm_base @ self.arm.forward(q)[self.arm_tip]
        if self.wrist_offset is not None:
            wrist = wrist @ self.wrist_offset
        hand = self.hand.forward(q)
        return {'wrist': {'frame_id': 'world', 'position_m': wrist[:3, 3].tolist(),
                          'quaternion_wxyz': matrix_quaternion(wrist[:3, :3]).tolist()},
                'fingertip_frame_id': self.frame_id,
                'fingertip_positions_m': {name: self._tip(hand, name).tolist()
                                         for name in self.fingertip_links}}

    def public_finger_geometry(self, joint_positions_rad):
        """robot_geometry_v2: wrist/tip FK plus tip orientations and joint origins.

        Tip-frame axes and named joint centres have SHARPA semantics; other hands
        publish wrist/fingertip FK only (the optional v2 fields stay empty).
        """
        value = self.public_geometry(joint_positions_rad)
        if not self.embodiment.sharpa_joint_semantics:
            return value
        hand = self.hand.forward(self.checked_joints(joint_positions_rad))
        value['fingertip_quaternions_wxyz'] = {name: matrix_quaternion(hand[tip][:3, :3]).tolist()
                                               for name, tip in self.fingertip_links.items()}
        centers = {}
        for finger, chain in self.finger_joints.items():
            origin = lambda joint: (hand[joint.parent] @ joint.origin)[:3, 3].tolist()
            by_suffix = {joint.name.split('_', 2)[2]: joint for joint in chain}
            if finger == 'thumb':
                centers[finger] = {'base_m': origin(by_suffix['CMC_FE']), 'proximal_m': origin(by_suffix['MCP_AA']),
                                   'distal_m': origin(by_suffix['IP'])}
            else:
                centers[finger] = {'base_m': origin(by_suffix['MCP_FE']), 'proximal_m': origin(by_suffix['MCP_AA']),
                                   'middle_m': origin(by_suffix['PIP']), 'distal_m': origin(by_suffix['DIP'])}
        value['finger_joint_centers_m'] = centers
        return value

    def _finger_inverse(self, finger, point, reference, *, tolerance, acceptance=None, deadline_ns=None):
        """Bounded position IK with weak posture regularization for redundancy.

        Other finger chains are fixed exactly. The selected chain minimizes
        tip error plus a small distance to the supplied posture reference.
        A seed must first meet the strict tolerance; otherwise the closest
        solution of all seeds is accepted when it lies within ``acceptance``
        (the completion tolerance), i.e. at the edge of the reachable set.
        """
        joints = self.finger_joints[finger]
        names = [joint.name for joint in joints]
        lo, hi = np.array([j.lower for j in joints]), np.array([j.upper for j in joints])
        ref = np.array([reference[name] for name in names])
        project = lambda value: self.hand_posture_prior.project(finger, value, lo, hi, names)
        regularization = .0002

        coupled = self._chain_followers.get(finger, {})

        def evaluate(q):
            if deadline_ns is not None and time.monotonic_ns() >= deadline_ns:
                raise TimeoutError('IK time budget exhausted')
            positions = reference | dict(zip(names, q.tolist()))
            poses = self.hand.forward(positions)
            current = self._tip(poses, finger)
            dp = point - current
            error = np.r_[dp, regularization * (ref - q)]
            jacobian = np.empty((3, len(names)))
            for index, joint in enumerate(joints):
                joint_world = poses[joint.parent] @ joint.origin
                axis = joint_world[:3, :3] @ joint.axis
                jacobian[:, index] = np.cross(axis, current - joint_world[:3, 3])
                for follower, multiplier in coupled.get(joint.name, ()):
                    follower_world = poses[follower.parent] @ follower.origin
                    jacobian[:, index] += multiplier * np.cross(follower_world[:3, :3] @ follower.axis,
                                                                current - follower_world[:3, 3])
            return dp, error, np.vstack([jacobian, regularization * np.eye(len(names))])

        # Deterministic seeds only resolve local numerical ambiguity. A failure
        # means no verified solution was found, not a proof of infeasibility.
        rng = np.random.default_rng(0)
        seeds = [ref, (lo + hi) / 2] + [rng.uniform(lo, hi) for _ in range(3)]
        best = None
        for seed in seeds:
            q, damping = project(seed), .003
            for _ in range(120):
                dp, error, jac = evaluate(q)
                distance = float(np.linalg.norm(dp))
                cost = float(error @ error)
                if best is None or distance < best[0]:
                    best = (distance, q.copy())
                if distance <= min(tolerance * .25, .0002):
                    break
                free = np.ones(len(names), dtype=bool)
                for _active in range(len(names) + 1):
                    jf = jac[:, free]
                    dq = np.zeros(len(names))
                    dq[free] = np.linalg.solve(jf.T @ jf + damping**2 * np.eye(free.sum()), jf.T @ error)
                    outward = ((q <= lo + 1e-10) & (dq < 0)) | ((q >= hi - 1e-10) & (dq > 0))
                    if not np.any(outward & free):
                        break
                    free &= ~outward
                dq = np.clip(dq, -.2, .2)
                accepted = False
                for scale in (1., .5, .25, .125, .0625, .03125):
                    trial = project(q + scale * dq)
                    trial_dp, trial_error, _ = evaluate(trial)
                    if float(trial_error @ trial_error) < cost - 1e-16:
                        q = trial
                        damping = max(.00001, damping * .7)
                        accepted = True
                        if np.linalg.norm(trial_dp) < best[0]:
                            best = (float(np.linalg.norm(trial_dp)), q.copy())
                        break
                if not accepted:
                    damping *= 3
                    if damping > .3:
                        break
            if best[0] <= tolerance:
                return dict(zip(names, best[1].tolist())), best[0]
        if acceptance is not None and best[0] <= acceptance:
            return dict(zip(names, best[1].tolist())), best[0]
        limit = max(tolerance, acceptance or 0.)
        bound = [name for name, value, low, high in zip(names, best[1], lo, hi) if value <= low + 1e-6 or value >= high - 1e-6]
        found = self._tip(self.hand.forward(reference | dict(zip(names, best[1].tolist()))), finger)
        raise GeometryIKError(f'{finger} target has no verified IK solution within {limit:g} m; residual={best[0]:.6g} m',
            diagnostic=ActionDiagnostic(code='ik_solution_unverified', fingertip_errors_m={finger: float(best[0])},
                limiting_joint_names=bound, best_found_fingertips_m={finger: found.tolist()},
                geometry_tolerances=EffectiveGeometryTolerances(stage='ik_verification', fingertips_m={finger: limit})))

    def solve(self, measured_joints, *, wrist=None, fingertip_positions_m=None, hand_posture_rad=None,
              position_tolerance_m=.002, rotation_tolerance_rad=.03, fingertip_tolerance_m=.0015,
              fingertip_acceptance_m=None, deadline_ns=None):
        """Verify IK, optionally cancelling at a monotonic deadline without a target."""
        q = self.measured_reference(measured_joints)
        reference = dict(q)
        if hand_posture_rad is not None:
            if not isinstance(hand_posture_rad, dict) or not set(hand_posture_rad) <= set(self.hand.names):
                raise ValueError('Posture reference can contain only public hand joints')
            reference.update(hand_posture_rad)
        reference = self.checked_joints(reference, enforce_limits=True)
        # An explicit posture is an input, not a license to silently repair it.
        # Validate every finger touched by the supplied joint map after merging
        # unspecified joints with measured feedback; other fingers are unchanged.
        if hand_posture_rad:
            for finger, chain in self.finger_joints.items():
                if any(joint.name in hand_posture_rad for joint in chain):
                    violation = self.hand_posture_prior.violation(finger, reference)
                    if violation:
                        name, excess = violation
                        raise GeometryIKError('Explicit hand posture violates the advertised DIP/PIP coupling',
                            diagnostic=ActionDiagnostic(code='posture_constraint_violation',
                                                        joint_name=name, joint_error_rad=excess))
        tips = {} if fingertip_positions_m is None else fingertip_positions_m
        if not isinstance(tips, dict) or not set(tips) <= set(self.fingertip_links):
            raise ValueError('Unknown fingertip name')
        for value in (position_tolerance_m, rotation_tolerance_rad, fingertip_tolerance_m):
            if not math.isfinite(value) or value <= 0:
                raise ValueError('IK tolerances must be finite and positive')
        goal = dict(reference)
        diagnostic = {'fingertip_errors_m': {}, 'wrist_position_error_m': 0., 'wrist_rotation_error_rad': 0.}
        if wrist is not None:
            target = np.linalg.inv(self.arm_base) @ _pose_matrix(wrist)
            if self.wrist_offset is not None:
                target = target @ np.linalg.inv(self.wrist_offset)
            result = self.arm.inverse(target, self.arm_tip, q, iterations=140, multi_starts=4,
                                      position_tolerance=position_tolerance_m,
                                      rotation_tolerance=rotation_tolerance_rad, deadline_ns=deadline_ns)
            if not result['converged']:
                raise GeometryIKError('Wrist target has no verified IK solution; '
                    f'position residual={result["position_error_m"]:.6g} m, rotation residual={result["rotation_error_rad"]:.6g} rad',
                    diagnostic=ActionDiagnostic(code='ik_solution_unverified',
                        wrist_position_error_m=float(result['position_error_m']),
                        wrist_orientation_error_rad=float(result['rotation_error_rad']),
                        geometry_tolerances=EffectiveGeometryTolerances(stage='ik_verification',
                            wrist_position_m=position_tolerance_m, wrist_orientation_rad=rotation_tolerance_rad)))
            goal.update(result['joint_positions_rad'])
            diagnostic.update(wrist_position_error_m=result['position_error_m'],
                              wrist_rotation_error_rad=result['rotation_error_rad'])
        for finger, position in tips.items():
            point = _finite_vector(position, 3, f'{finger} target')
            solution, error = self._finger_inverse(finger, point, reference, tolerance=fingertip_tolerance_m,
                                                   acceptance=fingertip_acceptance_m, deadline_ns=deadline_ns)
            goal.update(solution)
            diagnostic['fingertip_errors_m'][finger] = error
        if deadline_ns is not None and time.monotonic_ns() >= deadline_ns:
            raise TimeoutError('IK time budget exhausted')
        return self.checked_joints(goal, enforce_limits=True), diagnostic


class ClassicalFingertipPolicy:
    """Public-joint-feedback executor implementing the System0 plugin boundary."""
    def __init__(self, assets_root, base_position_m, base_quaternion_wxyz=(1., 0., 0., 0.), *, limits=None,
                 hand_posture_prior=None, hand_id=None, assembly_version=None, embodiment=None):
        self.geometry = Fr3SharpaGeometry(assets_root, base_position_m, base_quaternion_wxyz,
                                         hand_posture_prior=hand_posture_prior, hand_id=hand_id,
                                         assembly_version=assembly_version, embodiment=embodiment)
        self.limits = limits or ExecutorLimits()
        source_paths = [Path(__file__), *[PROJECT/name for name in ('bench/control/__init__.py',
            'bench/control/reference.py','bench/control/hand_posture.py','bench/control/kinematics.py','bench/geometry.py')]]
        self.provenance = {
            'sources_sha256': {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in source_paths},
            'assets_sha256': self.geometry.asset_sha256, 'numpy_version': np.__version__,
            'world_from_robot_base': self.geometry.base.tolist(), 'limits': asdict(self.limits),
            'mount': self.geometry.embodiment.mount,
            'fingertip_links': self.geometry.fingertip_links, 'hand_frame': self.geometry.frame_id,
            'tip_semantics': self.geometry.embodiment.controller_tip_semantics,
            'collision_planning': False, 'force_control': False, 'trained_rl': False,
            'measured_reference_limit_tolerance_rad': MEASURED_JOINT_LIMIT_TOLERANCE_RAD,
            'measured_limit_scope':'Every skill accepts measured joints inside the service safety band; they are clamped only for IK seeds and initial references; raw q, FK and feedback unchanged; emitted targets strict',
            'completion_semantics':'Requested wrist/selected tips and posture-only joints; omitted hand tracking error is maintenance status, not primitive completion; stationary reference and settled measured wrist required',
            'fingertip_ik_acceptance':'strict seed search at half the completion tolerance, else the closest solution of all seeds within 1 mm; completion still judged at the request with the 1.5 mm tolerance',
            'measured_limit_band':'begin admits measured joints within the 0.03 rad service safety band (clamped only for IK seeds and anchors); an action fails if a hand excursion exceeds max(5 mrad, begin + 2 mrad) or an arm excursion exceeds max(1 mrad, begin + 1 mrad)',
            'recovery_integral':'a failed task integral is cleared at the recovery stop; arm joints keep the integral of the last normal completion',
            'stall_termination':'after the nominal ends, objective progress below 5% of tolerance over 0.3 s with drive change below 2 mrad ends the move as tracking_stalled; recovery keeps the executed drive',
            'recovery_drive_semantics':'After failure/cancel, committed continuous-stop target is held without reapplying P or I on unselected chains; explicit reselection restores normal tracking',
            'hand_posture_prior': self.geometry.hand_posture_prior.describe(),
            'hand_goal_semantics': {'protocol': 'accepted_hand_goal_retention_v2',
             'scope': 'move_geometry_and_hold',
             'selected_fingers': 'explicit_tip_keys_union_hand_posture_joint_chains',
             'normal_omission': 'previous_accepted_goal_or_persistent_fixed_drive',
             'initial_omission': 'measured_reference',
             'recovery_omission': 'committed_continuous_stop_fixed_drive_without_pi',
             'selected_partial_posture_omission': 'measured_reference',
             'wrist_omission': 'current_measured_pose',
             'selected_tip_ik': 'verified_position_may_change_redundant_joint_solution',
             'simultaneous_tip_posture': 'same_chain_posture_is_ik_reference_not_joint_objective',
             'recovery_hold': 'all_controlled_joints_fixed_drive',
             'geometry_arm': 'resume_tracking_requested_or_measured_wrist',
             'fixed_hand_exit': 'successful_explicit_chain_reselection_only',
             'rejected_begin': 'all_goal_mode_anchor_and_pending_state_unchanged',
             'completion': {'wrist': 'requested_or_measured_maintained_pose',
                            'fingertips': 'explicit_tip_fk_at_advertised_tolerance',
                            'joint_posture': 'explicit_entries_on_chains_without_simultaneous_tip',
                            'unselected_hand': 'maintenance_only_not_joint_completion',
                            'reference': 'all_acknowledged_command_velocities_stationary'},
             'measured_reference': {'epsilon_rad': MEASURED_JOINT_LIMIT_TOLERANCE_RAD,
                                    'scope': 'simulation_internal_reference_eligibility_not_hardware_safety',
                                    'raw_feedback_projected': False,
                                    'emitted_targets': 'strict_urdf_bounds'},
             'feedback_protocol': 'robot_execution_state_v1'},
            'measured_reference_projection': 'Internal trajectory/IK reference only; raw observations and feedback unchanged; explicit requested postures remain strict.',
            'command_limit_roundoff': 'Eight scaled float64 epsilons, then exact clamp; larger or nonfinite target is rejected.',
            'kinematic_trajectory': 'quintic_nominal_with_finite_response_continuous_discrete_braking_viable_reference_and_latched_terminal_stop',
            'reference_response': 'alpha=1-exp(-reference_tracking_gain_per_s*control_period_s); reference lag is not actuator saturation',
            'integrator': 'bounded_PI; arm_integrates_only_at_stationary_nominal_and_still_arm; preserve_normal_actions_and_hold; freeze_during_braking; clear_failed_task_integral_only_after_continuous_stop; conditional_anti_windup',
            'direct_control_handoff': DirectControlHandoff().model_dump(mode='json'),
            'normal_nominal_handoff': 'prior_nominal_goal_for_tracking_chains; committed_drive_for_recovered_fixed_chains; previous_drive_velocity_and_load_integral_preserved',
            'acknowledged_drive_hold': 'explicit mode; complete current stationary ACK required; fixed drives use existing governor; geometric reselection resumes tracking; no task or settling success inferred',
        }
        if self.geometry.robot_id != 'sharpa_w01_right':
            # Non-default embodiments add their identity; the default provenance keys are unchanged.
            self.provenance.update(robot_id=self.geometry.robot_id,
                tip_offsets_m={finger: offset.tolist() for finger, offset in self.geometry.tip_offsets.items()},
                hand_base_joints=[joint.name for joint in self.geometry.base_joints],
                mimic_followers=sorted(getattr(self.geometry.hand, 'mimic', {})),
                hand_base_joint_semantics='maintained like an unselected finger unless an explicit posture names them')
        digest = hashlib.sha256(json.dumps(self.provenance, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
        posture_profile = self.geometry.hand_posture_prior.describe()
        self.descriptor = PolicyDescriptor(policy_id='classical_fingertip', version='1.9.0',
            implementation_sha256=digest, trained=False, observation_profile='vision_proprio',
            skills=['move_geometry', 'hold'], hold_modes=['reference', 'acknowledged_drive'], control_period_s=CONTROL_DT,
            geometry_inputs=['wrist_pose', 'fingertip_positions', 'hand_posture'], force_objectives=False,
            robot_posture_profile=({key:value for key,value in posture_profile.items() if key != 'notes'}
                                   if posture_profile['joint_pairs'] else None),
            hand_goal_semantics=self.provenance['hand_goal_semantics'],
            direct_control_handoff=DirectControlHandoff(),
            geometry_tolerance_policy=GeometryTolerancePolicy(protocol='geometry_tolerance_policy_v2',
                wrist_position_completion_cap_m=self.limits.wrist_position_tolerance_m,
                wrist_orientation_completion_cap_rad=self.limits.wrist_rotation_tolerance_rad,
                fingertip_completion_m=self.limits.fingertip_tolerance_m,
                joint_posture_completion_rad=self.limits.joint_tracking_tolerance_rad,
                ik_verification_scale=self.limits.ik_verification_scale,
                wrist_rule='minimum_of_request_and_controller_cap',
                fingertip_ik_acceptance_m=self.limits.fingertip_ik_acceptance_m,
                wrist_settle_speed_m_s=self.limits.wrist_settle_speed_m_s,
                stall_window_s=self.limits.stall_window_s),
            camera_observation_mode='decision_only', supported_robot_ids=[self.geometry.robot_id])
        self._ready = False
        self._active = False
        self._last_step_time = None
        self._last_observation_sequence = None
        self.integral = np.zeros(len(self.geometry.names))
        self._arm_rest_integral = np.zeros(len(self.geometry.names))
        self._begin_excursion = np.zeros(len(self.geometry.names))
        self._limit_diagnostic = None
        self.previous = None
        self.velocity = None
        self._recovery_required = False
        self._has_goal = False
        self._fixed_drive_mask = np.zeros(len(self.geometry.names), dtype=bool)
        self._fixed_drive_goal = np.zeros(len(self.geometry.names))
        self._posture_objective_names = ()
        self._maintained_joint_names = ()
        self._pending_step_state = None
        self._direct_handoff_pending = False
        self._direct_velocity_known = False
        self._direct_ack_sequence = None
        self._direct_ack_time = None
        self._reach_diagnostic = None
        self._stall_window = deque()
        self._stall_stopping = False
        self._wrist_settled = None
        self._arm_mask = np.array([name in self.geometry.arm.names for name in self.geometry.names])

    def public_geometry(self, joint_positions_rad):
        return self.geometry.public_geometry(joint_positions_rad)

    def reset(self, spec, capabilities):
        if spec.observation_profile != 'vision_proprio' or capabilities.control_period_s != CONTROL_DT:
            raise ValueError('Classical executor requires public proprioception and a 20 ms period')
        if set(capabilities.arm_joints) != set(self.geometry.arm.names) or set(capabilities.hand_joints) != set(self.geometry.hand.names):
            raise ValueError('Classical executor joint inventory does not match this embodiment')
        for joint in self.geometry.joints:
            limit = capabilities.joint_limits[joint.name]
            if abs(limit.lower_rad - joint.lower) > 1e-7 or abs(limit.upper_rad - joint.upper) > 1e-7:
                raise ValueError('Public joint limits disagree with the pinned URDF')
        self._ready, self._active = True, False
        self._episode_id = spec.episode_id
        self._last_step_time = None
        self._last_observation_sequence = None
        self.integral = np.zeros(len(self.geometry.names))
        self._arm_rest_integral = np.zeros(len(self.geometry.names))
        self._begin_excursion = np.zeros(len(self.geometry.names))
        self._limit_diagnostic = None
        self.previous = None
        self.velocity = None
        self._recovery_required = False
        self._has_goal = False
        self._fixed_drive_mask = np.zeros(len(self.geometry.names), dtype=bool)
        self._fixed_drive_goal = np.zeros(len(self.geometry.names))
        self._posture_objective_names = ()
        self._maintained_joint_names = ()
        self._pending_step_state = None
        self._direct_handoff_pending = False
        self._direct_velocity_known = False
        self._direct_ack_sequence = None
        self._direct_ack_time = None
        self._reach_diagnostic = None
        self._stall_window = deque()
        self._stall_stopping = False
        self._wrist_settled = None

    def _measured(self, observation):
        if observation.profile != 'vision_proprio':
            raise ValueError('Classical executor accepts only public observations')
        if observation.episode_id != self._episode_id:
            raise ValueError('Public observation belongs to a different episode')
        try:
            raw = self.geometry.checked_joints(observation.proprioception.joint_positions_rad)
        except ValueError as error:
            raise GeometryIKError(str(error),diagnostic=ActionDiagnostic(code='invalid_robot_state')) from error
        self.geometry.measured_reference(raw)  # admissibility only; never replace raw feedback
        return raw

    def _reference_error(self, error, *, elapsed=None):
        index = error.joint_index
        return ActionDiagnostic(code=('invalid_robot_state' if 'nonfinite' in str(error)
                                      else 'trajectory_limit_guard'),
            joint_name=None if index is None else self.geometry.names[index], elapsed_sim_s=elapsed)

    def _acknowledged_reference(self, observation):
        """Read robot-only drive feedback; never substitute measured positions."""
        try:
            targets = observation.proprioception.commanded_joint_positions_rad
            if targets is None:
                raise ValueError('No acknowledged drive targets')
            targets = self.geometry.checked_joints(targets)
            raw = np.array([targets[name] for name in self.geometry.names])
            bounded = np.clip(raw, self.geometry.lower, self.geometry.upper)
            # A native float32 representation of an exact URDF boundary can
            # sit just outside its float64 value. Emit the strict-bound value
            # only if it authors exactly the same native float32 drive bits.
            if not np.array_equal(raw.astype(np.float32), bounded.astype(np.float32)):
                raise ValueError('Acknowledged drive exceeds strict native-equivalent bounds')
            return bounded
        except ValueError as error:
            raise GeometryIKError('Fixed-drive hold requires complete finite bounded acknowledged targets',
                diagnostic=ActionDiagnostic(code='control_handoff_unverified')) from error

    def begin(self, action, observation):
        if not self._ready:
            raise ValueError('Reset executor before starting an action')
        if self._active:
            raise ValueError('Cancel or finish the active action before beginning another')
        if self._pending_step_state is not None:
            raise ValueError('An unacknowledged command cannot be replaced by begin')
        if self._direct_handoff_pending and (not self._direct_velocity_known
                or observation.sequence != self._direct_ack_sequence
                or abs(float(observation.sim_time_s)-self._direct_ack_time)>1e-8):
            raise GeometryIKError('Classical reentry needs a current verified direct drive and velocity',
                diagnostic=ActionDiagnostic(code='control_handoff_unverified'))
        if action.observation_sequence != observation.sequence:
            raise ValueError('Geometry action references a stale observation sequence')
        is_hold = action.skill == 'hold'
        acknowledged_hold = is_hold and action.hold_mode == 'acknowledged_drive'
        q = self._measured(observation)
        begin_excursion = self._excursion(q)
        recovery_required = self._recovery_required
        # Construct the entire plan locally first. A rejected request does not
        # replace the last submitted reference, its velocity or the load bias.
        measured_reference = self.geometry.measured_reference(q)
        previous = (np.array([measured_reference[name] for name in self.geometry.names])
                    if self.previous is None else self.previous.copy())
        velocity = np.zeros_like(previous) if self.velocity is None else self.velocity.copy()
        if acknowledged_hold:
            acknowledged = self._acknowledged_reference(observation)
            # A fresh reset has a stationary drive. Later entry requires our
            # committed velocity, including any verified direct-control handoff.
            if ((self.previous is None and observation.sequence != 0)
                    or (self.previous is not None and self.velocity is None)
                    or not np.isfinite(velocity).all() or np.any(abs(velocity) > 1e-10)
                    or (self.previous is not None and not np.array_equal(
                        previous.astype(np.float32), acknowledged.astype(np.float32)))):
                raise GeometryIKError('Fixed-drive hold requires a current stationary acknowledged reference',
                    diagnostic=ActionDiagnostic(code='control_handoff_unverified'))
            # Native feedback may round the submitted float64 target to float32.
            # Anchor to the actual ACK, not that rounded-off requested reference.
            previous = acknowledged
            recovery_required = False
        arm_count = len(self.geometry.arm.names)
        vmax = np.array([min(j.velocity, self.limits.arm_velocity_rad_s if i < arm_count
                            else self.limits.hand_velocity_rad_s) for i, j in enumerate(self.geometry.joints)])
        amax = np.array([self.limits.arm_acceleration_rad_s2] * arm_count
                       + [self.limits.hand_acceleration_rad_s2] * len(self.geometry.hand.names))
        try:
            stopped, stop_steps = stop_reference(previous, velocity, self.geometry.lower,
                self.geometry.upper, vmax, amax, CONTROL_DT)
        except ReferenceInfeasible as error:
            raise GeometryIKError(str(error), diagnostic=self._reference_error(error)) from error
        recovery_steps = stop_steps if recovery_required else 0
        initial = stopped if recovery_required else previous.copy()
        if not recovery_required and self._has_goal:
            # A normal completed reference already includes its load compensation.
            # Continue the nominal geometric path from the preceding nominal goal,
            # while previous/velocity remain the committed drive reference. Adding
            # PI to a previous drive used as nominal would apply that bias twice.
            # Recovered fixed-drive chains deliberately retain drive coordinates;
            # their P/I bypass and explicit reselection behavior are unchanged.
            initial = np.where(self._fixed_drive_mask, previous, self.goal).copy()
        # Modes are computed transactionally, only installed after all IK/FK/deadline checks.
        fixed_drive_mask = (np.ones(len(previous), dtype=bool) if recovery_required
                            else self._fixed_drive_mask.copy())
        posture_objective_names = ()
        maintained_joint_names = tuple(self.geometry.names)
        if is_hold:
            deadline = float(action.hold_duration_s)
            if action.geometry is not None or deadline > action.timeout_sim_s:
                raise GeometryIKError('hold requires a duration within its timeout and no geometry objective')
            # A failed/cancelled objective must not be resumed by hold. Its safe
            # stop is a new robot reference; normal holds retain the last goal.
            goal_array = (initial.copy() if recovery_required or not self._has_goal else self.goal.copy())
            if acknowledged_hold:
                initial = goal_array = previous.copy()
                fixed_drive_mask = np.ones(len(previous), dtype=bool)
            goal = dict(zip(self.geometry.names, goal_array.tolist()))
            duration = deadline
            position_tolerance = self.limits.wrist_position_tolerance_m
            rotation_tolerance = self.limits.wrist_rotation_tolerance_rad
            requested_wrist = _pose_matrix(self.geometry.public_geometry(goal)['wrist'])
            requested_tips = {}
            ik_diagnostic = {}
            minimum_steps = stop_steps
        else:
            if action.skill != 'move_geometry' or action.geometry is None:
                raise GeometryIKError('Classical executor requires move_geometry or explicit timed hold')
            geometry = action.geometry
            # Arm/wrist omission keeps the existing measured-pose maintenance semantics.
            # Only omitted hand chains carry a recovered fixed-drive mode forward.
            fixed_drive_mask[:arm_count] = False
            tip_fingers = set(geometry.fingertip_positions_m or {})
            tip_joint_names = {j.name for f in tip_fingers for j in self.geometry.finger_joints[f]}
            # For a finger with a tip target, explicit posture is the existing IK
            # redundancy reference, not an independent conflicting joint objective.
            posture_objective_names = tuple(name for name in (geometry.hand_posture_rad or {})
                                            if name not in tip_joint_names)
            if geometry.contacts or geometry.wrist_trajectory:
                raise GeometryIKError('Contact constraints and wrist trajectories are not implemented by this executor')
            if geometry.fingertip_frame_id != self.geometry.frame_id:
                raise GeometryIKError(f'Fingertip targets must use the {self.geometry.frame_id} frame')
            position_tolerance = min(geometry.position_tolerance_m, self.limits.wrist_position_tolerance_m)
            rotation_tolerance = min(geometry.orientation_tolerance_rad, self.limits.wrist_rotation_tolerance_rad)
            try:
                goal, ik_diagnostic = self.geometry.solve(q, wrist=geometry.wrist,
                    fingertip_positions_m=geometry.fingertip_positions_m, hand_posture_rad=geometry.hand_posture_rad,
                    position_tolerance_m=position_tolerance * self.limits.ik_verification_scale,
                    rotation_tolerance_rad=rotation_tolerance * self.limits.ik_verification_scale,
                    fingertip_tolerance_m=self.limits.fingertip_tolerance_m * self.limits.ik_verification_scale,
                    fingertip_acceptance_m=self.limits.fingertip_ik_acceptance_m)
                # Omitted hand chains retain the last accepted command goal,
                # not a contact-deflected measurement. Recovery uses only the
                # stop reachable from committed references; a failed objective
                # or an older grip must never be silently resumed.
                anchor = (initial if recovery_required else self.goal if self._has_goal
                          else np.array([measured_reference[name] for name in self.geometry.names]))
                anchor = dict(zip(self.geometry.names, anchor.tolist()))
                posture_names = set(geometry.hand_posture_rad or {})
                maintained_joint_names = []
                for finger, joints in self.geometry.finger_joints.items():
                    selected = (finger in (geometry.fingertip_positions_m or {})
                                or any(joint.name in posture_names for joint in joints))
                    if not selected:
                        maintained_joint_names.extend(joint.name for joint in joints)
                        goal.update({joint.name: anchor[joint.name] for joint in joints})
                    else:
                        for joint in joints:fixed_drive_mask[self.geometry.names.index(joint.name)] = False
                base_joints = self.geometry.base_joints
                if base_joints and not any(joint.name in posture_names for joint in base_joints):
                    maintained_joint_names.extend(joint.name for joint in base_joints)
                    goal.update({joint.name: anchor[joint.name] for joint in base_joints})
                elif base_joints:
                    for joint in base_joints:fixed_drive_mask[self.geometry.names.index(joint.name)] = False
                goal = self.geometry.checked_joints(goal, enforce_limits=True)
            except GeometryIKError:
                raise
            except ValueError as error:
                raise GeometryIKError(str(error)) from error
            goal_array = np.array([goal[name] for name in self.geometry.names])
            delta = abs(goal_array - initial)
            minimum = max(float(np.max(1.875 * delta / vmax)),
                          float(np.sqrt(np.max(5.773503 * delta / amax))), CONTROL_DT)
            duration = math.ceil(minimum / CONTROL_DT) * CONTROL_DT
            deadline = float(action.timeout_sim_s)
            minimum_steps = recovery_steps + math.ceil((duration + self.limits.settle_margin_s) / CONTROL_DT - 1e-9)
            requested_wrist = (_pose_matrix(geometry.wrist) if geometry.wrist is not None
                               else _pose_matrix(self.geometry.public_geometry(q)['wrist']))
            requested_tips = {name: np.asarray(point, float) for name, point in geometry.fingertip_positions_m.items()}
        deadline_steps = round(deadline / CONTROL_DT)
        if abs(deadline_steps * CONTROL_DT - deadline) > 1e-8 or deadline_steps < max(1, minimum_steps):
            raise GeometryIKError('Requested timeout is too short for the bounded trajectory and settling margin',
                                  diagnostic=ActionDiagnostic(code='insufficient_duration',
                                      duration_preflight=DurationPreflight(skill=action.skill,
                                          requested_duration_s=deadline,
                                          minimum_duration_s=max(1, minimum_steps) * CONTROL_DT,
                                          control_period_s=CONTROL_DT,
                                          minimum_steps=max(1, minimum_steps),
                                          braking_steps=stop_steps if is_hold else recovery_steps,
                                          trajectory_steps=0 if is_hold else round(duration / CONTROL_DT),
                                          settling_steps=0 if is_hold else minimum_steps - recovery_steps - round(duration / CONTROL_DT))))
        target_geometry = self.geometry.public_geometry(goal)
        self.max_velocity, self.max_acceleration = vmax, amax
        self.previous, self.velocity = previous, velocity
        self.initial, self.goal = initial, goal_array
        self._fixed_drive_mask = fixed_drive_mask
        self._fixed_drive_goal = goal_array.copy()
        self._posture_objective_names = posture_objective_names
        self._maintained_joint_names = tuple(maintained_joint_names)
        self.duration, self.deadline, self.deadline_steps = duration, deadline, deadline_steps
        self._is_hold = is_hold
        self._acknowledged_hold = acknowledged_hold
        if acknowledged_hold:
            # The new nominal is the ACK itself, already containing any old
            # compensation. Its old nominal-relative bias must not be added a
            # second time when an explicit geometry request unlocks a chain.
            # This installation occurs only after every admission check passes.
            self.integral = np.zeros(len(previous))
            self._arm_rest_integral = np.zeros(len(previous))
        self._recovery_steps = recovery_steps
        self._recovering = self._recovery_required = recovery_required
        self._settling = False
        self.position_tolerance, self.rotation_tolerance = position_tolerance, rotation_tolerance
        self.requested_wrist, self.requested_tips = requested_wrist, requested_tips
        self.ik_diagnostic = ik_diagnostic
        # Targets just outside the reachable set (within the completion
        # tolerance) execute their closest solution and report the residual.
        boundary = {finger: float(error) for finger, error in ik_diagnostic.get('fingertip_errors_m', {}).items()
                    if error > self.limits.fingertip_tolerance_m * self.limits.ik_verification_scale}
        self._reach_diagnostic = None if not boundary else ActionDiagnostic(code='ik_target_at_reach_boundary',
            fingertip_errors_m=boundary, geometry_tolerances=EffectiveGeometryTolerances(stage='ik_verification',
                fingertips_m={finger: self.limits.fingertip_tolerance_m for finger in boundary}))
        self._objective_mask = ~np.isin(self.geometry.names, maintained_joint_names)
        self._stall_window = deque(maxlen=round(self.limits.stall_window_s / CONTROL_DT) + 1)
        self._stall_stopping = False
        self._wrist_settled = None
        self._begin_excursion = begin_excursion
        index = int(np.argmax(begin_excursion))
        self._limit_diagnostic = (None if begin_excursion[index] <= MEASURED_LIMIT_REPORT_RAD else
            ActionDiagnostic(code='measured_limit_projection', joint_name=self.geometry.names[index],
                             joint_error_rad=float(begin_excursion[index])))
        self.target_geometry = target_geometry
        self.start_time = float(observation.sim_time_s)
        self._last_step_time = None
        self._last_observation_sequence = None
        self._start_observation_sequence = observation.sequence
        self._active = self._has_goal = True

    def _reached(self, measured):
        q = np.array([measured[name] for name in self.geometry.names])
        indices = [self.geometry.names.index(name) for name in self._posture_objective_names]
        if indices and np.max(np.abs(q[indices] - self.goal[indices])) > self.limits.joint_tracking_tolerance_rad:
            return False
        actual = self.geometry.public_geometry(measured)
        wrist = _pose_matrix(actual['wrist'])
        if (np.linalg.norm(wrist[:3, 3] - self.requested_wrist[:3, 3]) > self.position_tolerance
                or rotation_distance(wrist[:3, :3], self.requested_wrist[:3, :3]) > self.rotation_tolerance):
            return False
        return all(np.linalg.norm(np.asarray(actual['fingertip_positions_m'][name]) - target)
                   <= self.limits.fingertip_tolerance_m for name, target in self.requested_tips.items())

    def _excursion(self, measured):
        """Per-joint measured distance beyond the URDF bounds (zero inside)."""
        q = np.array([measured[name] for name in self.geometry.names])
        return np.maximum(0., np.maximum(self.geometry.lower - q, q - self.geometry.upper))

    def _objective_ratio(self, measured):
        """Largest requested-objective residual as a multiple of its tolerance."""
        q = np.array([measured[name] for name in self.geometry.names])
        actual = self.geometry.public_geometry(measured)
        wrist = _pose_matrix(actual['wrist'])
        ratios = [np.linalg.norm(wrist[:3, 3] - self.requested_wrist[:3, 3]) / self.position_tolerance,
                  rotation_distance(wrist[:3, :3], self.requested_wrist[:3, :3]) / self.rotation_tolerance]
        ratios += [np.linalg.norm(np.asarray(actual['fingertip_positions_m'][name]) - target) / self.limits.fingertip_tolerance_m
                   for name, target in self.requested_tips.items()]
        indices = [self.geometry.names.index(name) for name in self._posture_objective_names]
        if indices:
            ratios.append(np.max(np.abs(q[indices] - self.goal[indices])) / self.limits.joint_tracking_tolerance_rad)
        return float(max(ratios))

    def _wrist_speed(self, measured, observation):
        """Measured wrist speed from raw arm joint velocities through URDF FK."""
        velocity = observation.proprioception.joint_velocities_rad_s
        ahead = {name: measured[name] + 1e-3 * velocity[name] for name in self.geometry.arm.names}
        if not all(math.isfinite(value) for value in ahead.values()):
            return math.inf
        now = self.geometry.arm.forward({name: measured[name] for name in self.geometry.arm.names})[self.geometry.arm_tip]
        later = self.geometry.arm.forward(ahead)[self.geometry.arm_tip]
        return float(np.linalg.norm(later[:3, 3] - now[:3, 3])) / 1e-3

    def execution_state(self, observation):
        """Describe the supplied acknowledged robot observation without advancing control.

        A pending candidate is not an executed reference. Status readers must
        pass the latest acknowledged public observation; the service independently
        enforces its timestamp and robot-name whitelist before publishing this.
        Maintenance residuals are not contact, load, or grasp certificates.
        """
        if not self._has_goal:
            raise ValueError('No accepted action has robot execution state')
        velocity = (self._pending_step_state['velocity'] if self._pending_step_state is not None
                    else self.velocity)
        stationary = None if velocity is None else bool(np.all(abs(velocity) <= 1e-10))
        objective = 'not_checked'
        errors = {}
        try:
            measured = self._measured(observation)
        except GeometryIKError:
            # Invalid feedback must never be projected to manufacture convergence.
            pass
        else:
            if not self._is_hold:
                objective = 'reached' if self._reached(measured) else 'not_reached'
            for name in self._maintained_joint_names:
                index = self.geometry.names.index(name)
                target = self._fixed_drive_goal[index] if self._fixed_drive_mask[index] else self.goal[index]
                errors[name] = abs(float(measured[name]) - float(target))
        return RobotExecutionState(protocol='robot_execution_state_v1',
            sim_time_s=float(observation.sim_time_s),
            completion_scope='timed_reference_maintenance' if self._is_hold else 'requested_geometry',
            objective_status=objective, reference_stationary=stationary,
            wrist_settled=None if self._active else self._wrist_settled,
            posture_objective_joint_names=list(self._posture_objective_names),
            recovery_fixed_drive_joint_names=[name for name, fixed in zip(
                self.geometry.names, self._fixed_drive_mask) if fixed],
            maintained_joint_error_rad=errors)

    def _terminal_failure(self, detail, diagnostic):
        self._active = False
        self._recovery_required = True
        return PolicyStep(status='failed', detail=detail, diagnostic=diagnostic)

    def _convergence_diagnostic(self, measured, elapsed):
        q = np.array([measured[name] for name in self.geometry.names])
        joint_error = abs(q - self.goal)
        indices = [self.geometry.names.index(name) for name in self._posture_objective_names]
        index = max(indices, key=lambda i:joint_error[i]) if indices else None
        actual = self.geometry.public_geometry(measured)
        wrist = _pose_matrix(actual['wrist'])
        return ActionDiagnostic(code='convergence_timeout', joint_name=None if index is None else self.geometry.names[index],
            joint_error_rad=None if index is None else float(joint_error[index]), elapsed_sim_s=elapsed,
            wrist_position_error_m=float(np.linalg.norm(wrist[:3, 3] - self.requested_wrist[:3, 3])),
            wrist_orientation_error_rad=float(rotation_distance(wrist[:3, :3], self.requested_wrist[:3, :3])),
            fingertip_errors_m={name: float(np.linalg.norm(np.asarray(actual['fingertip_positions_m'][name]) - target))
                               for name, target in self.requested_tips.items()},
            geometry_tolerances=None if self._is_hold else EffectiveGeometryTolerances(
                stage='completion', wrist_position_m=self.position_tolerance,
                wrist_orientation_rad=self.rotation_tolerance,
                joint_posture_rad=None if index is None else self.limits.joint_tracking_tolerance_rad,
                fingertips_m={name: self.limits.fingertip_tolerance_m for name in self.requested_tips}))

    def check_direct_handoff(self):
        if not self._ready or self._active or self._pending_step_state is not None:
            raise ValueError('Direct bypass requires an idle policy with no unacknowledged candidate')

    def adopt_direct_reference(self, command, *, velocity_rad_s, observation_sequence, sim_time_s):
        """Install an adapter-verified native ACK, never a requested/measured target.

        A direct command is not Q/V/A smoothed. Its real discrete drive velocity
        must be admitted by the existing stop planner before classical reentry.
        Old PI is frozen and inaccessible until that continuous stop clears it.
        """
        self.check_direct_handoff()
        q=self.geometry.checked_joints(command.joint_positions_rad,enforce_limits=True)
        if abs(command.duration_s-CONTROL_DT)>1e-9:
            raise ValueError('Direct ACK must cover one native period')
        if type(observation_sequence) is not int or observation_sequence < 1 or not math.isfinite(sim_time_s):
            raise ValueError('Invalid direct ACK identity')
        previous=np.array([q[name] for name in self.geometry.names])
        velocity=None if velocity_rad_s is None else _finite_vector(
            [velocity_rad_s[name] for name in self.geometry.names],len(previous),'Acknowledged command velocity')
        if velocity_rad_s is not None and set(velocity_rad_s)!=set(self.geometry.names):
            raise ValueError('Acknowledged velocity inventory mismatch')
        # All validation above is transactional. Omitted hand chains and hold
        # subsequently anchor only to this drive or its feasible continuous stop.
        self.previous=previous
        self.velocity=velocity
        self.initial=previous.copy()
        self.goal=previous.copy()
        self._fixed_drive_mask=np.ones(len(previous),dtype=bool)
        self._fixed_drive_goal=previous.copy()
        self._posture_objective_names=()
        self._maintained_joint_names=tuple(self.geometry.names)
        self._has_goal=False
        self._recovery_required=True
        self._last_step_time=None
        self._last_observation_sequence=None
        self._direct_handoff_pending=True
        self._direct_velocity_known=velocity is not None
        self._direct_ack_sequence=observation_sequence
        self._direct_ack_time=float(sim_time_s)
        self._arm_rest_integral=np.zeros(len(previous))

    def commit_pending_step(self):
        """Acknowledge that the adapter executed the last running command."""
        if self._pending_step_state is not None:
            self._direct_handoff_pending=False
        self._pending_step_state = None

    def discard_pending_step(self):
        """Roll back a candidate that the adapter did not execute."""
        if self._pending_step_state is not None:
            for name, value in self._pending_step_state.items():
                setattr(self, name, value)
            self._pending_step_state = None

    def step(self, observation):
        # The adapter must acknowledge worker.step before requesting another
        # candidate. An observation alone is not proof that a command executed.
        if self._pending_step_state is not None:
            raise ValueError('Commit or discard the preceding command before another policy step')
        fields = ('previous', 'velocity', 'integral', '_last_step_time',
                  '_last_observation_sequence', '_recovering', '_recovery_required',
                  '_settling', '_active', 'feedback_diagnostic', '_stall_window', '_stall_stopping')
        snapshot = {name: copy.deepcopy(getattr(self, name, None)) for name in fields}
        result = self._step(observation)
        if result.status == 'running':
            self._pending_step_state = snapshot
        # This describes observation, never the candidate command's future state.
        return result.model_copy(update={'robot_execution_state': self.execution_state(observation)})

    def _step(self, observation):
        if not self._active:
            raise ValueError('No active classical geometry action')
        try:
            measured = self._measured(observation)
            if self._acknowledged_hold and not np.array_equal(
                    self._acknowledged_reference(observation), self._fixed_drive_goal):
                raise GeometryIKError('Acknowledged drive changed during fixed-drive hold',
                    diagnostic=ActionDiagnostic(code='control_handoff_unverified'))
        except GeometryIKError as error:
            return self._terminal_failure(str(error), error.public_diagnostic)
        excursion = self._excursion(measured)
        allowed = np.where(self._arm_mask,
            np.maximum(self.limits.arm_excursion_floor_rad, self._begin_excursion + self.limits.arm_excursion_growth_rad),
            np.maximum(self.limits.hand_excursion_floor_rad, self._begin_excursion + self.limits.hand_excursion_growth_rad))
        if np.any(excursion > allowed):
            # The action keeps driving a joint past its limit (normally into
            # contact); stop this action. The next action may start from here.
            index = int(np.argmax(excursion - allowed))
            return self._terminal_failure('Measured joint-limit excursion grew during the action',
                ActionDiagnostic(code='invalid_robot_state', joint_name=self.geometry.names[index],
                                 joint_error_rad=float(excursion[index])))
        now = float(observation.sim_time_s)
        elapsed = now - self.start_time
        tick = observation.sequence - self._start_observation_sequence
        expected_sequence = (self._start_observation_sequence if self._last_observation_sequence is None
                             else self._last_observation_sequence + 1)
        if observation.sequence != expected_sequence:
            raise ValueError('Executor requires fresh observations with consecutive physics sequences')
        if (elapsed < -1e-9 or abs(elapsed - tick * CONTROL_DT) > 1e-8
                or (self._last_step_time is not None and abs(now - self._last_step_time - CONTROL_DT) > 1e-8)):
            raise ValueError('Executor requires fresh observations spaced by one control period')
        self._last_step_time = now
        self._last_observation_sequence = observation.sequence
        stopped = bool(np.all(abs(self.velocity) <= 1e-10))
        if self._recovering and tick >= self._recovery_steps:
            # Clear failed-task feedback only at a continuous, stationary stop.
            # No command is changed here; the governor also bounds the next step.
            if not stopped:
                return self._terminal_failure('Recovery did not reach a stationary reference',
                    ActionDiagnostic(code='trajectory_limit_guard', elapsed_sim_s=elapsed))
            # A failed task's integral is cleared, but the arm keeps the load
            # (gravity) compensation it held at its last normal completion.
            self.integral = np.where(self._arm_mask, self._arm_rest_integral, 0.)
            self._recovering = self._recovery_required = False
        nominal_done = (not self._is_hold and tick >= self._recovery_steps
                        and elapsed >= self._recovery_steps * CONTROL_DT + self.duration - 1e-9)
        reached = nominal_done and self._reached(measured)
        # Braking starts as soon as the pose is within tolerance, but a move
        # completes only once the measured wrist has also settled (or at the
        # deadline): a pose merely passing through the tolerance is not done.
        settled = reached and stopped and (tick >= self.deadline_steps or
            self._wrist_speed(measured, observation) <= self.limits.wrist_settle_speed_m_s)
        if (self._is_hold and tick >= self.deadline_steps and stopped) or settled:
            self._active = False
            self._wrist_settled = None if self._is_hold else (
                self._wrist_speed(measured, observation) <= self.limits.wrist_settle_speed_m_s)
            self._arm_rest_integral = np.where(self._arm_mask, self.integral, 0.)
            return PolicyStep(status='completed', detail=('Timed robot reference maintenance completed; no grasp/task success is inferred'
                if self._is_hold else 'Requested robot geometry reached with stationary reference; no grasp/contact/task success is inferred'),
                diagnostic=(self._reach_diagnostic if not self._is_hold and self._reach_diagnostic is not None
                            else self._limit_diagnostic))
        if nominal_done and not reached and not self._recovering:
            if self._stall_stopping and stopped:
                # Normally blocked by contact. The recovery stop keeps the
                # executed drive targets, so a finger pressing on an object keeps
                # pressing; only the wait for an unreachable pose ends.
                diagnostic = self._convergence_diagnostic(measured, elapsed).model_dump()
                return self._terminal_failure('Requested geometry stopped progressing before its objective',
                    ActionDiagnostic.model_validate({**diagnostic, 'code': 'tracking_stalled'}))
            if not self._stall_stopping:
                ratio = self._objective_ratio(measured)
                self._stall_window.append((ratio, self.previous.copy()))
                if len(self._stall_window) == self._stall_window.maxlen:
                    ratios = [row[0] for row in self._stall_window]
                    drive_change = float(np.max(np.abs(self._stall_window[-1][1] - self._stall_window[0][1])[self._objective_mask],
                                                initial=0.))
                    # Stalled: no progress and the drive (including PI) has stopped
                    # changing. Brake the reference to a stop, then report.
                    self._stall_stopping = (ratio > 1 and max(ratios) - min(ratios) < self.limits.stall_progress_fraction
                                            and drive_change < self.limits.stall_drive_change_rad)
        if tick >= self.deadline_steps:
            return self._terminal_failure('Public joint feedback did not reach requested geometry before timeout',
                                          self._convergence_diagnostic(measured, elapsed))
        # Reaching the tolerance starts a bounded stop, never a velocity reset.
        # Finish an entered stop before resuming tracking. A delayed measured
        # pose leaving tolerance must not reverse a still-moving reference.
        # Only measured convergence AND a stationary reference can complete.
        if self._settling and stopped:
            self._settling = False
        self._settling = self._settling or reached
        braking = self._recovering or self._settling or self._stall_stopping
        q = np.array([measured[name] for name in self.geometry.names])
        if self._is_hold:
            nominal, moving = self.goal.copy(), False
        else:
            s = float(np.clip((elapsed + CONTROL_DT - self._recovery_steps * CONTROL_DT) / self.duration, 0., 1.))
            alpha = 10 * s**3 - 15 * s**4 + 6 * s**5
            nominal, moving = self.initial + alpha * (self.goal - self.initial), s < 1.
        error = nominal - q
        # While the arm nominal moves, or the arm is still settling, its error is
        # the stiff drive's lag; integrating lag overshoots and, once braking
        # freezes it, leaves a lasting bias. Hand integrals are unchanged, and
        # nothing integrates while braking.
        candidate_integral = (self.integral.copy() if braking else np.clip(
            self.integral + self.limits.integral_gain_per_s * error * CONTROL_DT,
            -self.limits.maximum_tracking_offset_rad, self.limits.maximum_tracking_offset_rad))
        arm_speed = max(abs(observation.proprioception.joint_velocities_rad_s[name]) for name in self.geometry.arm.names)
        if moving or not arm_speed <= self.limits.arm_integral_still_speed_rad_s:
            candidate_integral = np.where(self._arm_mask, self.integral, candidate_integral)

        if not braking:
            candidate_integral = np.where(self._fixed_drive_mask, 0., candidate_integral)

        def bounded_command(integral):
            unbounded = nominal + self.limits.tracking_gain * error + integral
            correction = np.clip(unbounded - nominal, -self.limits.maximum_tracking_offset_rad,
                                 self.limits.maximum_tracking_offset_rad)
            desired = np.clip(nominal + correction, self.geometry.lower, self.geometry.upper)
            # These values already contain the old drive compensation. Applying
            # another proportional or integral layer would ratchet the target.
            desired = np.where(self._fixed_drive_mask, self._fixed_drive_goal, desired)
            response = -math.expm1(-self.limits.reference_tracking_gain_per_s * CONTROL_DT)
            filtered = self.previous + response * (desired - self.previous)
            command, velocity = bounded_reference(self.previous, self.velocity,
                self.previous if braking else filtered, self.geometry.lower, self.geometry.upper,
                self.max_velocity, self.max_acceleration, CONTROL_DT,
                steps_remaining=self.deadline_steps - tick)
            return unbounded, desired, filtered, velocity, command

        try:
            unbounded, desired, filtered, velocity, command = bounded_command(candidate_integral)
            increment = candidate_integral - self.integral
            # Anti-windup applies at actual offset/joint and motion bounds;
            # the intentional finite response lag alone is not saturation.
            blocked = ((increment * (unbounded - desired)) > 1e-14) | ((increment * (filtered - command)) > 1e-14)
            candidate_integral = np.where(blocked, self.integral, candidate_integral)
            _, desired, _, velocity, command = bounded_command(candidate_integral)
        except ReferenceInfeasible as error:
            return self._terminal_failure(str(error), self._reference_error(error, elapsed=elapsed))
        self.integral = candidate_integral
        self.feedback_diagnostic = {
            'maximum_abs_integral_rad': float(np.max(np.abs(self.integral))),
            'blocked_integrator_joint_names': [name for name, value in zip(self.geometry.names, blocked) if value],
            'maximum_abs_desired_tracking_offset_rad': float(np.max(np.abs(desired - nominal))),
            'recovery_fixed_drive_joint_names': [name for name, fixed in zip(self.geometry.names,self._fixed_drive_mask) if fixed],
            'reference_phase': 'recovery_stop' if self._recovering else 'stall_stop' if self._stall_stopping
                else ('settling_stop' if braking else 'tracking'),
        }
        self.previous, self.velocity = command, velocity
        return PolicyStep(status='running',
            command=RobotJointCommand(duration_s=CONTROL_DT,
            joint_positions_rad=dict(zip(self.geometry.names, command.tolist()))))

    def finish_hold_early(self):
        """End an active timed hold as a normal maintenance completion.

        Used by a composite (bimanual) executor when the other hand finishes:
        the reference stays the last accepted goal and the next action continues
        from the committed drive and velocity, exactly as after a completed hold.
        """
        if not (self._active and self._is_hold) or self._pending_step_state is not None:
            raise ValueError('Only an active hold with no unacknowledged candidate can finish early')
        self._active = False
        self._wrist_settled = None
        self._arm_rest_integral = np.where(self._arm_mask, self.integral, 0.)

    def cancel(self, reason):
        # The simulator is paused; keep its last submitted q and our command
        # velocity. Recovery consumes real bounded steps on the next action.
        self.discard_pending_step()
        self._active = False
        self._recovery_required = self.previous is not None
        self._last_step_time = None
        self._last_observation_sequence = None
