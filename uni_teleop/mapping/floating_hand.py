"""Relative PICO palm motion and independent MANUS fingers; no arm IK."""
import numpy as np
from scipy.spatial.transform import Rotation

# OpenXR: X right, Y up, -Z forward -> simulation X forward, Y left, Z up.
PICO_TO_WORLD = np.array([[0., 0., -1.], [-1., 0., 0.], [0., 1., 0.]])


def world_pose(pose):
    pose = np.asarray(pose, dtype=float)
    if pose.shape != (7,) or not np.isfinite(pose).all() or not .5 < np.linalg.norm(pose[3:]) < 1.5:
        raise ValueError('PICO pose requires finite xyz + xyzw')
    return PICO_TO_WORLD @ pose[:3], Rotation.from_matrix(PICO_TO_WORLD) * Rotation.from_quat(pose[3:])


class PalmMapper:
    def __init__(self, scale=1.):
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError('Position scale must be positive')
        self.scale = scale
        self.anchor = None
        self.alignment = Rotation.identity()

    def align_heading(self, head_pose, view_forward):
        """Freeze headset horizontal heading against the desktop camera."""
        _, head_rotation = world_pose(head_pose)
        forward = head_rotation.apply([0., 0., -1.])
        view = np.asarray(view_forward, dtype=float)
        if np.linalg.norm(forward[:2]) < .1 or np.linalg.norm(view[:2]) < .1:
            return False
        yaw = np.arctan2(view[1], view[0])-np.arctan2(forward[1], forward[0])
        self.alignment = Rotation.from_euler('z', yaw)
        self.clear()
        return True

    def clear(self):
        self.anchor = None

    def map(self, pose, held_position, held_quaternion):
        position, rotation = world_pose(pose)
        position = self.alignment.apply(position)
        rotation = self.alignment * rotation
        if self.anchor is None:
            self.anchor = (position, rotation, np.array(held_position), Rotation.from_quat(held_quaternion))
        p0, r0, target_p0, target_r0 = self.anchor
        return target_p0 + self.scale * (position-p0), (rotation*r0.inv()*target_r0).as_quat()


class HandController:
    """Clutch/re-anchor wrist on release or loss; fingers recover independently."""
    def __init__(self, model, position, quaternion, *, scale=1., max_age=.15, screen_aligned=False):
        self.model = model
        self.position, self.quaternion = np.array(position), np.array(quaternion)
        self.fingers = model.initial.copy()
        self.mapper = PalmMapper(scale)
        self.max_age_ns = int(max_age*1e9)
        self.active = False
        self.stream = None
        self.screen_aligned = screen_aligned
        self.heading_pending = screen_aligned
        self.wrist_status = self.finger_status = 'paused'

    def toggle(self):
        self.active = not self.active
        self.mapper.clear()
        self.heading_pending = self.screen_aligned

    def recenter(self):
        self.mapper.clear()
        self.heading_pending = self.screen_aligned

    def update(self, frame, fingers, now_ns, measured_pose, view_forward=None):
        if not self.active:
            self.wrist_status = self.finger_status = 'paused'
            return
        fresh = frame is not None and 0 <= now_ns-frame.host_ns <= self.max_age_ns
        if fresh and self.stream != frame.stream:
            self.mapper.clear()
            self.heading_pending = self.screen_aligned
            self.stream = frame.stream
        if self.heading_pending and fresh and frame.tracking['focused']:
            head = frame.tracking.get('head')
            if head is not None and view_forward is not None:
                self.heading_pending = not self.mapper.align_heading(head, view_forward)
        controller = frame.tracking['controllers'].get('right') if fresh and frame.tracking['focused'] else None
        if controller is None:
            self.mapper.clear()
            self.wrist_status = 'PICO missing/stale; wrist held'
        elif self.heading_pending:
            self.wrist_status = 'Look forward with headset to align; wrist held'
        elif controller['grip'] < .5:
            self.mapper.clear()
            self.wrist_status = 'clutch released; wrist held'
        else:
            if self.stream != frame.stream:
                self.mapper.clear()
            self.stream = frame.stream
            # Start at measured pose: a blocked hand must not inherit a target
            # on the far side of an obstacle when the operator re-clutches.
            self.position, self.quaternion = self.mapper.map(controller['pose'], *measured_pose)
            self.wrist_status = 'following PICO'
        if fingers is None:
            self.finger_status = 'MANUS missing/stale; fingers held'
        else:
            values = np.asarray([fingers[n] for n in self.model.names])
            if not np.isfinite(values).all():
                raise ValueError('Non-finite finger target')
            self.fingers = np.clip(values, self.model.lower, self.model.upper)
            self.finger_status = 'following MANUS'


def optimizer_targets(raw, candidate, *, now_ns, names, calibration_id, glove_id):
    if raw is None or candidate is None:
        return None
    if any(not 0 <= now_ns-p['capture_time_ns'] <= 150_000_000 for p in (raw, candidate)):
        return None
    if candidate['protocol'] != 'sharpa_optimizer_v1' or candidate['executable'] is not True:
        raise ValueError('Require executable SHARPA optimizer output')
    if (not raw['glove_tracking_valid'] or raw['device']['hand'] != 'right'
            or raw['device']['device_id'] != str(glove_id)
            or raw['calibration_id'] != calibration_id
            or any(candidate[k] != raw[k] for k in ('stream_id', 'clock_id', 'calibration_id', 'topology_id'))
            or candidate['device_id'] != raw['device']['device_id']
            or candidate['sequence'] > raw['sequence']
            or candidate['capture_time_ns'] > raw['capture_time_ns']):
        # The acquisition can restart before the asynchronous optimizer catches
        # up. Hold only fingers until both refer to the same source again.
        return None
    targets = candidate['joint_positions_rad']
    if set(targets) != set(names) or not np.isfinite(list(targets.values())).all():
        raise ValueError('MANUS optimizer joint names/values mismatch')
    return targets
