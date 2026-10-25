"""Relative XR motion to URDF-bounded robot targets; no task or object state."""
from __future__ import annotations

import time
import numpy as np
from scipy.spatial.transform import Rotation

from uni_teleop.mapping.franka.recovery import HandRecovery
from bench.control.fingertip import GeometryIKError
from uni_teleop.mapping.franka.controller_hand import ControllerHand
from bench.geometry import matrix_quaternion, quaternion_matrix

# OpenXR: palm=0, wrist=1; each finger includes its tip.
FINGERS = {'thumb': (2, 3, 4, 5), 'index': (6, 7, 8, 9, 10),
           'middle': (11, 12, 13, 14, 15), 'ring': (16, 17, 18, 19, 20),
           'little': (21, 22, 23, 24, 25)}
XR_TO_WORLD = np.array([[0., 0., -1.], [-1., 0., 0.], [0., 1., 0.]])


def pose_matrix(value):
    result = np.eye(4)
    result[:3, :3] = Rotation.from_quat(value[3:]).as_matrix()
    result[:3, 3] = value[:3]
    return result


def wrist_matrix(geometry):
    wrist = geometry['wrist']
    result = np.eye(4)
    result[:3, :3] = quaternion_matrix(wrist['quaternion_wxyz'])
    result[:3, 3] = wrist['position_m']
    return result


def palm_basis(index, middle, little, wrist):
    y = middle - wrist
    if np.linalg.norm(y) < 1e-5:
        raise ValueError('Degenerate hand calibration; show an open palm')
    y /= np.linalg.norm(y)
    x = index - little
    x -= y * np.dot(x, y)
    if np.linalg.norm(x) < 1e-5 or not np.isfinite(y).all():
        raise ValueError('Degenerate hand calibration; show an open palm')
    x /= np.linalg.norm(x)
    return np.column_stack((x, y, np.cross(x, y)))


def local_hand(joints):
    wrist = pose_matrix(joints[1])
    positions = np.array([joint[:3] if joint is not None else [np.nan]*3 for joint in joints])
    return (positions - wrist[:3, 3]) @ wrist[:3, :3]


class HandCalibration:
    def __init__(self, model, measured, joints):
        if any(joints[i] is None for indices in FINGERS.values() for i in indices):
            raise ValueError('Finger tracking incomplete — open hand to calibrate')
        self.human = local_hand(joints)
        geometry = model.public_finger_geometry(measured)
        self.tips = {name: np.array(value) for name, value in geometry['fingertip_positions_m'].items()}
        centers = geometry['finger_joint_centers_m']
        bases = {name: np.array(value['base_m']) for name, value in centers.items()}
        robot_basis = palm_basis(bases['index'], self.tips['middle'], bases['little'], np.zeros(3))
        human_basis = palm_basis(self.human[6], self.human[11], self.human[21], self.human[1])
        self.rotation = robot_basis @ human_basis.T
        self.scales = {}
        for finger, indices in FINGERS.items():
            human_length = np.linalg.norm(np.diff(self.human[list(indices)], axis=0), axis=1).sum()
            points = [np.array(v) for k, v in centers[finger].items() if k != 'proximal_m']
            points.append(self.tips[finger])
            robot_length = np.linalg.norm(np.diff(points, axis=0), axis=1).sum()
            if human_length < .015:
                raise ValueError('Degenerate tracked finger')
            self.scales[finger] = robot_length / human_length

    def targets(self, joints):
        current = local_hand(joints)
        return {finger: self.tips[finger] + self.scales[finger] * self.rotation @
                (current[indices[-1]]-self.human[indices[-1]]) for finger, indices in FINGERS.items()
                if all(joints[i] is not None for i in indices)}


def finger_step(model, reference, finger, target, *, deadline_ns):
    """Take bounded, improving steps toward a teleop tip target, even outside reach.

    This is approximate tracking, not the benchmark's verified endpoint IK.
    Joint boxes and the existing posture prior apply to every candidate.
    """
    chain = model.finger_joints[finger]
    names = [joint.name for joint in chain]
    lo, hi = np.array([j.lower for j in chain]), np.array([j.upper for j in chain])
    project = lambda q: model.hand_posture_prior.project(finger, q, lo, hi, names)
    q = project([reference[n] for n in names])

    def evaluate(value):
        poses = model.hand.forward(reference | dict(zip(names, value.tolist())))
        tip = poses[model.fingertip_links[finger]]
        point = tip[:3, 3] + tip[:3, :3] @ model.tip_offsets.get(finger, np.zeros(3))
        return poses, point, target-point

    poses, point, error = evaluate(q)
    for _ in range(4):
        if time.monotonic_ns() >= deadline_ns or np.linalg.norm(error) < .0002:
            break
        jacobian = np.column_stack([np.cross((poses[j.parent] @ j.origin)[:3, :3] @ j.axis,
                                             point-(poses[j.parent] @ j.origin)[:3, 3]) for j in chain])
        delta = jacobian.T @ np.linalg.solve(jacobian @ jacobian.T + .003**2*np.eye(3), error)
        largest = np.max(np.abs(delta))
        if largest > .15:
            delta *= .15/largest  # Preserve the descent direction across joints.
        for scale in (1., .5, .25, .125):
            candidate = project(q + scale*delta)
            new_poses, new_point, new_error = evaluate(candidate)
            if new_error @ new_error < error @ error:
                q, poses, point, error = candidate, new_poses, new_point, new_error
                break
        else:
            break
    return dict(zip(names, q.tolist())), float(np.linalg.norm(error))


class Retargeter:
    """One adapter for single, dual, or named FR3+SHARPA instances.

    Targets advance only after acknowledge(). Brief interruptions hold and
    re-anchor when opted in. Hands can wait for stable input or a spare-hand
    confirmation; explicit stops cancel recovery. Grip release clears its anchor.
    """
    def __init__(self, geometry, initial, bindings, *, mode='hands', scale=1., rotation=XR_TO_WORLD, tracking_grace_s=0., hands_recovery=False,
                 control_point='wrist', lock_orientation=False, arm_only=False):
        self.models = geometry.sides if hasattr(geometry, 'sides') else {'right': geometry}
        if not bindings or not set(bindings) <= self.models.keys() or not set(bindings.values()) <= {'left', 'right'}:
            raise ValueError('Bind robot hand names to left/right PICO inputs')
        if len(set(bindings.values())) != len(bindings):
            raise ValueError('Each PICO input controls at most one robot hand')
        if mode not in ('hands', 'controllers') or not np.isfinite(scale) or not 0 < scale <= 2:
            raise ValueError('Invalid input mode or position scale')
        if any(not model.embodiment.sharpa_joint_semantics for model in self.models.values()):
            raise ValueError('This PICO mapper supports the pinned SHARPA embodiments')
        if not np.isfinite(tracking_grace_s) or not 0 <= tracking_grace_s <= .5:
            raise ValueError('Tracking recovery window must be between 0 and 0.5 seconds')
        self.tracking_grace_ns = int(tracking_grace_s*1e9)
        self.last_tracked_ns = None
        self.interrupted = False
        self.bindings, self.mode, self.scale = dict(bindings), mode, scale
        self.rotation = np.asarray(rotation, dtype=float)
        if self.rotation.shape != (3, 3) or not np.allclose(self.rotation.T @ self.rotation, np.eye(3)) or np.linalg.det(self.rotation) < .99:
            raise ValueError('XR-to-world rotation must be a proper rotation')
        if control_point not in ('wrist', 'pinch'):
            raise ValueError('Control point must be wrist or pinch')
        self.control_point, self.lock_orientation = control_point, bool(lock_orientation)
        self.arm_only = arm_only
        self.acknowledged = dict(initial)
        self.anchors = {}
        self.hand_calibrations = {}
        self.hand_recovery = HandRecovery() if hands_recovery else None
        self.armed = False
        self.grip_recovery = None
        self.stream = None
        self.status = 'paused'
        self.diagnostics = {}
        self.finger_errors = {}
        self.presets = {}
        self.velocity = {}
        self.grasp_offsets = {}
        for name, model in self.models.items():
            self.presets[name] = ControllerHand(model)
            tips = model.public_geometry({n: initial[n] for n in model.names} | self.presets[name].prepose)['fingertip_positions_m']
            # Fixed in the wrist frame; trigger motion must not move the reference.
            self.grasp_offsets[name] = (np.array(tips['thumb']) + tips['index']) / 2
            for joint in model.joints:
                self.velocity[joint.name] = min(joint.velocity, .65 if joint.name in model.arm.names else 1.)

    def set_active(self, active):
        self.armed = bool(active)
        if self.hand_recovery:
            self.hand_recovery.clear()
        self.grip_recovery = None
        self.last_tracked_ns = None
        self.interrupted = False
        self.anchors.clear()
        self.status = 'calibrating' if active else 'paused'

    def control_points(self, measured):
        if self.mode != 'controllers':
            return []
        points = []
        for name in self.bindings:
            model = self.models[name]
            pose = wrist_matrix(model.public_geometry({n: measured[n] for n in model.names}))
            point = pose[:3, 3].copy()
            if self.control_point == 'pinch':
                point += pose[:3, :3] @ self.grasp_offsets[name]
            points.append(point)
        return points

    def set_controller_mapping(self, *, control_point=None, lock_orientation=None):
        if control_point is not None:
            if control_point not in ('wrist', 'pinch'):
                raise ValueError('Control point must be wrist or pinch')
            self.control_point = control_point
        if lock_orientation is not None:
            self.lock_orientation = bool(lock_orientation)
        # Re-anchor from measured pose on the next frame, without arming a pause.
        self.anchors.clear()

    @property
    def recovery_pending(self):
        return self.hand_recovery is not None and self.hand_recovery.pending

    def recalibrate(self):
        self.set_active(False)
        self.hand_calibrations.clear()
        self.status = 'hand calibration cleared — open hand, then Start'

    def _tracking_lost(self, now_ns, *, allow_grace=True):
        if (allow_grace and self.mode == 'hands' and self.hand_recovery
                and (self.armed or self.recovery_pending) and self.last_tracked_ns is not None):
            self.hand_recovery.interrupt(self.last_tracked_ns)
            self.hand_recovery.manual |= now_ns-self.hand_recovery.last_valid_ns > self.hand_recovery.automatic_ns
            self.armed = False
            self.anchors.clear()
            self.status = self.hand_recovery.status(self.bindings.values())
            return
        if (allow_grace and self.tracking_grace_ns and self.armed and self.last_tracked_ns is not None
                and 0 <= now_ns-self.last_tracked_ns <= self.tracking_grace_ns):
            self.interrupted = True
            self.anchors.clear()
            self.status = 'tracking interrupted — holding; waiting to re-anchor'
            return
        recover = self.mode == 'controllers' and (self.armed or self.grip_recovery is not None)
        self.set_active(False)
        if recover:
            self.grip_recovery = 'release'
        self.status = 'tracking lost — press Start to recalibrate'

    def set_mode(self, mode):
        if mode not in ('hands', 'controllers'):
            raise ValueError('Unknown PICO input mode')
        self.mode = mode
        self.set_active(False)

    def acknowledge(self, targets):
        self.acknowledged = dict(targets)

    def _ready(self, frame, now_ns):
        if frame is None or not 0 <= now_ns-frame.host_ns <= 150_000_000:
            self._tracking_lost(now_ns)
            return False
        if not frame.tracking.get('focused', True):
            self._tracking_lost(now_ns, allow_grace=False)
            self.grip_recovery = None
            self.status = 'PICO app not focused — press Start after returning'
            return False
        if self.stream is not None and self.stream != frame.stream:
            self.recalibrate()
        self.stream = frame.stream
        missing = [side for side in self.bindings.values() if side not in frame.tracking[self.mode]]
        if missing:
            self._tracking_lost(now_ns)
            if not self.interrupted:
                self.status = f'{", ".join(missing)} {self.mode} unavailable — check PICO tracking'
            status_key = 'hand_status' if self.mode == 'hands' else 'controller_status'
            self.diagnostics.update({side+'/tracking': frame.tracking.get(status_key, {}).get(side, 'input data absent')
                                     for side in missing})
            return False
        if (self.mode == 'hands' and self.hand_recovery and self.armed and self.last_tracked_ns is not None
                and frame.host_ns-self.last_tracked_ns > 150_000_000
                and not all((frame.continuous_since_ns or {}).get('hands/'+side, frame.host_ns)
                            <= self.last_tracked_ns for side in self.bindings.values())):
            self._tracking_lost(now_ns)
        if self.recovery_pending:
            if not self.hand_recovery.update(frame, self.bindings.values(), now_ns):
                self.status = self.hand_recovery.status(self.bindings.values())
                return False
            self.set_active(True)
        if self.interrupted:
            if now_ns-self.last_tracked_ns > self.tracking_grace_ns:
                self._tracking_lost(now_ns, allow_grace=False)
                return False
            self.interrupted = False
        if self.grip_recovery is not None:
            grips = [frame.tracking['controllers'][side]['grip'] for side in self.bindings.values()]
            if self.grip_recovery == 'release' and all(value < .5 for value in grips):
                self.grip_recovery = 'press'
            elif self.grip_recovery == 'press' and all(value >= .5 for value in grips):
                self.set_active(True)
            if self.grip_recovery is not None:
                self.status = f'tracking restored — {self.grip_recovery} grip to resume'
                return False
        if not self.armed:
            self.status = 'paused — press Start'
            return False
        self.last_tracked_ns = frame.host_ns
        if self.mode == 'controllers':
            for name, source in self.bindings.items():
                if frame.tracking[self.mode][source]['grip'] < .5:
                    self.anchors.pop(name, None)
        return True

    def hold(self, frame, *, now_ns):
        """Discard an expired solution without rearming or treating fresh input as lost."""
        if self._ready(frame, now_ns):
            self.status = 'slow IK — holding previous target'
        return dict(self.acknowledged)

    def command(self, frame, measured, *, now_ns):
        self.diagnostics = {}
        self.finger_errors = {}
        if not self._ready(frame, now_ns):
            return dict(self.acknowledged)
        deadline = time.monotonic_ns() + 80_000_000
        timed_out = False
        desired = dict(self.acknowledged)
        for name, source in self.bindings.items():
            model = self.models[name]
            sample = frame.tracking[self.mode][source]
            if self.mode == 'controllers' and sample['grip'] < .5:
                continue
            raw = sample['pose'] if self.mode == 'controllers' else sample[1]
            human = pose_matrix(raw)
            q = model.measured_reference({n: measured[n] for n in model.names})
            if name not in self.anchors:
                if (not self.arm_only and self.mode == 'hands' and name not in self.hand_calibrations
                        and any(sample[i] is None for indices in FINGERS.values() for i in indices)):
                    self.status = 'Finger tracking incomplete — open hand to calibrate'
                    return dict(self.acknowledged)
                try:
                    if not self.arm_only and self.mode == 'hands' and name not in self.hand_calibrations:
                        self.hand_calibrations[name] = HandCalibration(model, q, sample)
                    fingers = self.hand_calibrations.get(name) if self.mode == 'hands' else None
                except ValueError as error:
                    self.set_active(False)
                    self.status = str(error)
                    return dict(self.acknowledged)
                self.anchors[name] = (human, wrist_matrix(model.public_geometry(q)), fingers)
                continue
            human_zero, robot_zero, fingers = self.anchors[name]
            rotation = self.rotation
            target = robot_zero.copy()
            target[:3, 3] += self.scale * rotation @ (human[:3, 3]-human_zero[:3, 3])
            if self.mode != 'controllers' or not self.lock_orientation:
                target[:3, :3] = rotation @ human[:3, :3] @ human_zero[:3, :3].T @ rotation.T @ robot_zero[:3, :3]
            if self.mode == 'controllers' and self.control_point == 'pinch':
                target[:3, 3] += (robot_zero[:3, :3] - target[:3, :3]) @ self.grasp_offsets[name]
            wrist = {'frame_id': 'world', 'position_m': target[:3, 3].tolist(),
                     'quaternion_wxyz': matrix_quaternion(target[:3, :3]).tolist()}
            try:
                solved, _ = model.solve(q, wrist=wrist,
                    deadline_ns=min(deadline, time.monotonic_ns()+25_000_000))
                desired.update({n: solved[n] for n in model.arm.names})
            except (GeometryIKError, TimeoutError) as error:
                timed_out |= isinstance(error, TimeoutError)
                self.diagnostics[name+'/wrist'] = str(error)
            if self.arm_only:
                continue
            if self.mode == 'controllers':
                desired.update(self.presets[name].targets(sample['trigger']))
            else:
                points = fingers.targets(sample)
                for finger in FINGERS.keys()-points.keys():
                    self.diagnostics[name+'/'+finger] = 'finger tracking unavailable — holding'
                for index, (finger, point) in enumerate(points.items()):
                    now = time.monotonic_ns()
                    finger_deadline = now + max(0, deadline-now)//(len(points)-index)
                    solved, error = finger_step(model, q, finger, point, deadline_ns=finger_deadline)
                    desired.update(solved)
                    self.finger_errors[name+'/'+finger] = error
        self.status = 'tracking' if not self.diagnostics else 'tracking; affected targets held'
        if timed_out:
            self.status = 'tracking; IK budget exhausted — affected targets held'
        if self.mode == 'controllers' and not self.anchors:
            self.status = 'ready — hold grip to move'
        # Keep the native 20 ms period and original URDF limits. Never catch up a
        # slow render by skipping integration or applying a large joint jump.
        return {n: self.acknowledged[n] + float(np.clip(v-self.acknowledged[n],
                    -self.velocity[n]*.02, self.velocity[n]*.02)) for n, v in desired.items()}
