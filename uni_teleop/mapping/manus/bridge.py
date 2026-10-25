"""Deterministic normalization, freshness gating and morphology-independent retarget seam."""
from __future__ import annotations
import hashlib
import copy
import json
import math
from collections import OrderedDict
from dataclasses import dataclass
from typing import Callable
import numpy as np
from uni_teleop.devices.manus.schema import AcquisitionPacket, CaptureConfig, Pose


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def quaternion_matrix(q):
    w, x, y, z = q
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def multiply(a, b):
    w, x, y, z = a
    v, i, j, k = b
    return [w*v-x*i-y*j-z*k, w*i+x*v+y*k-z*j,
            w*j-x*k+y*v+z*i, w*k+x*j-y*i+z*v]


def compose(a: Pose, b: Pose) -> Pose:
    return Pose(position=(np.array(a.position)+quaternion_matrix(a.quaternion_wxyz) @ b.position).tolist(),
                quaternion_wxyz=multiply(a.quaternion_wxyz, b.quaternion_wxyz))


@dataclass
class Result:
    status: str
    reason: str
    frame: dict | None = None


class CaptureBridge:
    """One pinned glove/clock/calibration stream. New SDK session => new bridge."""
    def __init__(self, config: CaptureConfig):
        self.config = config.model_copy(deep=True)
        self._configuration_bytes = canonical(self.config.model_dump(mode="json"))
        self._identity = None
        self._node_signature = None
        self._last_sequence = -1
        self._last_time = -1
        self._digests = OrderedDict()

    def mapped_time(self, clock_id, value):
        if clock_id == self.config.target_clock_id:
            return value, 0
        for mapping in self.config.clock_mappings:
            if mapping.source_clock_id == clock_id:
                if mapping.uncertainty_ns <= self.config.max_clock_uncertainty_ns:
                    return value + mapping.offset_ns, mapping.uncertainty_ns
        return None, None

    def accept(self, packet: AcquisitionPacket, receive_time_ns: int) -> Result:
        if canonical(self.config.model_dump(mode="json")) != self._configuration_bytes:
            raise ValueError("capture configuration changed; start a new capture")
        payload = packet.model_dump(mode='json')
        digest = hashlib.sha256(canonical(payload)).hexdigest()
        identity = (packet.stream_id, packet.device.model_dump_json(), packet.clock_id,
                    packet.topology_id, packet.frame_id, packet.node_space, packet.position_unit)
        signature = tuple(sorted((n.node_id, n.parent_id or '', n.semantic) for n in packet.nodes))
        if packet.calibration_id != self.config.calibration_id:
            return Result('rejected', 'calibration_changed')
        if packet.frame_id != self.config.capture_frame_id:
            return Result('rejected', 'capture_frame_mismatch')
        if self._identity is not None and self._identity != identity:
            return Result('rejected', 'stream_identity_or_topology_changed')
        if packet.glove_tracking_valid and self._node_signature is not None and signature != self._node_signature:
            return Result('rejected', 'skeleton_topology_changed')
        previous = self._digests.get(packet.sequence)
        if previous is not None:
            return Result('duplicate' if previous == digest else 'rejected',
                          'already_recorded' if previous == digest else 'sequence_payload_conflict')
        if packet.sequence <= self._last_sequence or packet.capture_time_ns <= self._last_time:
            return Result('rejected', 'out_of_order')
        mapped, uncertainty = self.mapped_time(packet.clock_id, packet.capture_time_ns)
        age = None if mapped is None else receive_time_ns-mapped
        reasons = []
        if mapped is None:
            reasons.append('clock_unsynchronized')
        elif age < -uncertainty:
            reasons.append('future_timestamp')
        elif age + uncertainty > self.config.max_sample_age_ns:
            reasons.append('stale_glove')
        if not packet.glove_tracking_valid or not all(n.valid for n in packet.nodes):
            reasons.append('glove_tracking_lost')
        if not self.config.glove_calibrated:
            reasons.append('glove_uncalibrated')
        # Unknown clocks/stale data remain useful raw captures; never executable commands.
        scale = {'m': 1, 'cm': .01, 'mm': .001}[packet.position_unit]
        native = {n.node_id: Pose(position=[v*scale for v in n.pose.position],
                                 quaternion_wxyz=n.pose.quaternion_wxyz) for n in packet.nodes}
        nodes_by_id = {n.node_id: n for n in packet.nodes}
        resolved = {}
        def resolve(node_id):
            if node_id not in resolved:
                node = nodes_by_id[node_id]
                pose = native[node_id]
                if packet.node_space == 'parent_local' and node.parent_id is not None:
                    pose = compose(resolve(node.parent_id), pose)
                resolved[node_id] = pose
            return resolved[node_id]
        normalized = {}
        for node in packet.nodes:
            normalized[node.node_id] = {
                'parent_id': node.parent_id, 'semantic': node.semantic, 'valid': node.valid,
                **compose(self.config.world_from_capture, resolve(node.node_id)).model_dump()}
        wrist = None
        wrist_reasons = list(reasons)
        tracker = packet.tracker
        if tracker is None or tracker.status != 'tracked':
            wrist_reasons.append('wrist_6dof_unavailable')
        else:
            tracker_time, tracker_uncertainty = self.mapped_time(tracker.clock_id, tracker.capture_time_ns)
            if tracker_time is None or mapped is None:
                wrist_reasons.append('tracker_clock_unsynchronized')
            elif abs(tracker_time-mapped)+tracker_uncertainty+uncertainty > self.config.max_tracker_skew_ns:
                wrist_reasons.append('tracker_glove_time_skew')
            elif receive_time_ns-tracker_time+tracker_uncertainty > self.config.max_sample_age_ns:
                wrist_reasons.append('stale_tracker')
            elif tracker_time-receive_time_ns > tracker_uncertainty:
                wrist_reasons.append('future_tracker')
            if tracker.frame_id != self.config.capture_frame_id:
                wrist_reasons.append('tracker_frame_mismatch')
            if tracker.tracked_body_frame != self.config.tracker_body_frame:
                wrist_reasons.append('tracker_body_mismatch')
            if not self.config.world_calibrated or not self.config.wrist_offset_calibrated:
                wrist_reasons.append('wrist_transform_uncalibrated')
            if not wrist_reasons:
                pose = Pose(position=[v*scale for v in tracker.pose.position],
                            quaternion_wxyz=tracker.pose.quaternion_wxyz)
                wrist = compose(compose(self.config.world_from_capture, pose), self.config.tracker_from_wrist).model_dump()
        missing = 0 if self._last_sequence < 0 else packet.sequence-self._last_sequence-1
        frame = {
            'protocol': 'human_motion_v1', 'stream_id': packet.stream_id, 'sequence': packet.sequence,
            'device': packet.device.model_dump(), 'calibration_id': packet.calibration_id,
            'capture_time_ns': packet.capture_time_ns, 'capture_clock_id': packet.clock_id,
            'source_timestamp': packet.source_timestamp, 'source_timestamp_encoding': packet.source_timestamp_encoding,
            'receive_time_ns': receive_time_ns, 'receive_clock_id': self.config.target_clock_id,
            'mapped_capture_time_ns': mapped, 'clock_uncertainty_ns': uncertainty, 'callback_age_ns': age,
            'frame_id': self.config.world_frame_id, 'position_unit': 'm', 'rotation_order': 'wxyz',
            'topology_id': packet.topology_id, 'nodes': normalized,
            'ergonomics_rad': {k: math.radians(v) for k, v in packet.ergonomics_degrees.items()},
            'ergonomics_capture_time_ns': packet.ergonomics_capture_time_ns,
            'ergonomics_source_timestamp': packet.ergonomics_source_timestamp,
            'world_transform_calibrated': self.config.world_calibrated,
            'hand_valid_until_ns': None if mapped is None else mapped+self.config.max_sample_age_ns-uncertainty,
            'arm_valid_until_ns': None if wrist is None else min(mapped+self.config.max_sample_age_ns-uncertainty, tracker_time+self.config.max_sample_age_ns-tracker_uncertainty),
            'wrist': wrist, 'hand_control_valid': not reasons, 'arm_control_valid': not wrist_reasons,
            'hand_hold_reasons': reasons, 'arm_hold_reasons': wrist_reasons,
            'sequence_gap': missing, 'sdk_queue_drops_total': packet.sdk_queue_drops_total,
            'retarget_version': self.config.retarget_version, 'robot_profile_id': self.config.robot_profile_id,
        }
        self._identity = identity
        if packet.glove_tracking_valid:
            self._node_signature = signature
        self._last_sequence, self._last_time = packet.sequence, packet.capture_time_ns
        self._digests[packet.sequence] = digest
        if len(self._digests) > 4096:
            self._digests.popitem(last=False)
        return Result('accepted', 'recorded', frame)


class RetargetRegistry:
    """Inject measured/learned retargeters; human angles never become joint targets implicitly."""
    def __init__(self):
        self._entries = {}

    def register(self, profile_id: str, version: str, mapper: Callable):
        key = (profile_id, version)
        if key in self._entries:
            raise ValueError('retargeter already registered')
        self._entries[key] = mapper

    def map(self, frame, *, previous_targets, joint_limits, arm_joint_names, now_ns: int, now_clock_id: str):
        if set(previous_targets) != set(joint_limits) or not set(arm_joint_names) <= set(joint_limits):
            raise ValueError('robot profile joint names disagree')
        for name, value in previous_targets.items():
            lo, hi = joint_limits[name]
            if not math.isfinite(value) or not lo <= value <= hi:
                raise ValueError('invalid previous robot target')
        if now_clock_id != frame['receive_clock_id'] or now_ns < frame['receive_time_ns']:
            return dict(previous_targets), {'mode': 'hold_all', 'reasons': ['control_clock_mismatch']}
        if frame['hand_valid_until_ns'] is None or now_ns > frame['hand_valid_until_ns']:
            return dict(previous_targets), {'mode': 'hold_all', 'reasons': ['input_watchdog_expired']}
        if not frame['hand_control_valid']:
            return dict(previous_targets), {'mode': 'hold_all', 'reasons': frame['hand_hold_reasons']}
        key = (frame['robot_profile_id'], frame['retarget_version'])
        if key not in self._entries:
            raise ValueError('no retargeter registered for this profile/version')
        target = self._entries[key](copy.deepcopy(frame))
        if set(target) != set(joint_limits):
            raise ValueError('retargeter must supply exactly the robot joint set')
        target = dict(target)
        arm_valid = frame['arm_control_valid'] and frame['arm_valid_until_ns'] is not None and now_ns <= frame['arm_valid_until_ns']
        if not arm_valid:
            for name in arm_joint_names:
                target[name] = previous_targets[name]
        for name, value in target.items():
            lo, hi = joint_limits[name]
            if not math.isfinite(value) or not lo <= value <= hi:
                raise ValueError(f'invalid robot joint target: {name}')
        return target, {'mode': 'full' if arm_valid else 'hand_only',
                        'reasons': frame['arm_hold_reasons'], 'retarget_version': key[1]}
