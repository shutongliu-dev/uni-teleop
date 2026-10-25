"""Independent FACTR arm and calibrated MANUS fingers, with no hardware writes."""
from dataclasses import replace
import time

import numpy as np

from bench.adapters.manus.bridge import CaptureBridge
from bench.adapters.manus.schema import AcquisitionPacket, CaptureConfig
from bench.adapters.manus.sources import host_clock_id
from bench.adapters.pico.protocol import Frame
from bench.adapters.pico.retarget import Retargeter, HandCalibration, finger_step, FINGERS, palm_basis
from .retarget import JointRetargeter


# Semantic conversion into the existing finger solver's palm/wrist + finger layout.
# MANUS supplies a single hand root; it supplies no separate OpenXR palm pose.
SEMANTICS = ['Hand/Invalid', 'Hand/Invalid']
for finger in ('Thumb', 'Index', 'Middle', 'Ring', 'Pinky'):
    joints = ('Metacarpal', 'Proximal', 'Distal', 'Tip') if finger == 'Thumb' else (
        'Metacarpal', 'Proximal', 'Intermediate', 'Distal', 'Tip')
    SEMANTICS.extend('Finger'+finger+'/'+joint for joint in joints)


def finger_sample(frame):
    nodes = {node['semantic']: node for node in frame['nodes'].values()}
    if len(nodes) != len(frame['nodes']):
        raise ValueError('Ambiguous MANUS node semantics')
    result = []
    for semantic in SEMANTICS:
        node = nodes[semantic]
        w, x, y, z = node['quaternion_wxyz']
        result.append(node['position']+[x, y, z, w])
    return result


class ManusHandCalibration(HandCalibration):
    """Match CMC-to-tip for thumb and MCP-to-tip for the other fingers.

    MANUS Metacarpal->Proximal spans the palm on the four long fingers;
    SHARPA's published finger bases are already at their MCP joints.
    """
    def __init__(self, model, measured, joints):
        super().__init__(model, measured, joints)
        centers = model.public_finger_geometry(measured)['finger_joint_centers_m']
        robot_basis = palm_basis(np.array(centers['index']['base_m']), self.tips['middle'],
                                  np.array(centers['little']['base_m']), np.zeros(3))
        self.rotation = robot_basis @ palm_basis(self.human[7], self.human[12],
                                                 self.human[22], self.human[1]).T
        for finger, indices in FINGERS.items():
            start = 0 if finger == 'thumb' else 1
            human_length = np.linalg.norm(np.diff(self.human[list(indices[start:])], axis=0), axis=1).sum()
            points = [np.array(v) for k, v in centers[finger].items()
                      if finger == 'thumb' or k != 'proximal_m'] + [self.tips[finger]]
            robot_length = np.linalg.norm(np.diff(points, axis=0), axis=1).sum()
            if human_length < .015:
                raise ValueError('Degenerate MANUS finger bones')
            self.scales[finger] = robot_length / human_length


class CombinedReceiver:
    rejected = 0

    def __init__(self, leader, glove, optimizer=None):
        self.leader, self.glove, self.optimizer = leader, glove, optimizer

    def read(self):
        retarget=self.optimizer.latest if self.optimizer else None
        arm, hand = self.leader.read() if self.leader else None, self.glove.latest
        stamps = ([arm.host_ns] if arm else []) + ([hand['capture_time_ns']] if hand else [])
        if not stamps:
            return None
        return Frame('factr-manus', arm.sequence if arm else 0, max(stamps), '', 'FACTR+MANUS',
                     {'head': None, 'hands': {}, 'controllers': {},
                      'leader': arm.tracking['leader'] if arm else [],
                      'leader_host_ns': arm.host_ns if arm else 0,
                      'leader_id': arm.tracking.get('leader_id') if arm else None, 'manus': hand,
                      'manus_retarget': retarget})

    def health(self):
        return {'leader': self.leader.health() if self.leader else {'enabled': False},
                'manus': self.glove.health(),
                'optimizer': self.optimizer.health() if self.optimizer else {'enabled': False}}

    def close(self):
        if self.optimizer:
            self.optimizer.close()
        if self.leader:
            self.leader.close()
        self.glove.close()


class ManusJointRetargeter(Retargeter):
    control_space = 'joint'
    input_hint = 'Leader: arm joints; right MANUS: fingers. Open hand before Start.'

    def __init__(self, geometry, initial, bindings, *, glove_id, calibration_id, finger_speed=4., hand_only=False, optimizer_metadata=None, arm_calibration=None, fixed_reference=False, calibration_output=None, follow_on_start=False, practice_follow=False, unrestricted_arm=False, **kwargs):
        super().__init__(geometry, initial, bindings, **kwargs)
        self.leader = JointRetargeter(geometry, initial, bindings, arm_calibration=arm_calibration,
            fixed_reference=fixed_reference,calibration_output=calibration_output,follow_on_start=follow_on_start,
            practice_follow=practice_follow,unrestricted_arm=unrestricted_arm, **kwargs)
        self.optimizer_metadata=optimizer_metadata
        self.hand_only = hand_only
        if arm_calibration and arm_calibration.status == 'provisional':
            self.input_hint = 'Approximate arm reference + MANUS fingers; no per-axis calibration'
        if fixed_reference:self.input_hint=self.leader.input_hint+' MANUS: fingers.'
        if hand_only:
            self.input_hint = "Right MANUS: fingers only; robot arm held. Open hand before Start."
        self.name = self.leader.name
        self.model = self.models[self.name]
        if not np.isfinite(finger_speed) or finger_speed <= 0:
            raise ValueError('MANUS finger speed must be finite and positive')
        for joint in self.model.hand.active:
            self.velocity[joint.name] = min(joint.velocity, finger_speed)
        self.glove_id, self.calibration_id = str(glove_id), calibration_id
        self.bridge = self._new_bridge()
        self.hand_frame = self.calibration = None
        self.hand_enabled = False
        self.input_mapping = dict(self.leader.input_mapping, hand_source='manus_raw_fingertips',
                                  glove_id=self.glove_id, calibration_id=calibration_id,
                                  version='factr_manus_fingertips_v2', finger_speed_rad_s=finger_speed,
                                  arm_source='held' if hand_only else 'factr_readonly')
        if optimizer_metadata:
            from bench.adapters.manus.sharpa import kinematic_signature
            if (set(optimizer_metadata['joint_names']) != set(self.model.hand.names) or
                    optimizer_metadata['kinematic_signature'] != kinematic_signature(self.model.hand_path)):
                raise ValueError('Official SHARPA optimizer kinematics differ from the simulated hand')
            self.input_mapping.update(version='sharpa_official_v4',optimizer=optimizer_metadata,
                                      hand_source='manus_official_sharpa_optimizer')

    @property
    def issue(self):
        return None if self.hand_only else self.leader.issue

    @property
    def alignment(self):
        return self.leader.alignment

    def _new_bridge(self):
        return CaptureBridge(CaptureConfig(calibration_id=self.calibration_id,
            glove_calibrated=True, target_clock_id=host_clock_id(),
            robot_profile_id='sharpa', retarget_version='sharpa_official_v4' if self.optimizer_metadata else
            'factr_manus_fingertips_v2'))

    def set_active(self, active):
        super().set_active(active)
        self.leader.set_active(active and not self.hand_only)
        self.hand_enabled = bool(active)

    def recalibrate(self):
        self.set_active(False)
        self.leader.recalibrate()
        self.calibration = self.hand_frame = None
        self.bridge = self._new_bridge()

    def acknowledge(self, targets):
        super().acknowledge(targets)
        self.leader.acknowledge(targets)

    def capture_reference(self, *, require_saved_pose=False):
        saved=self.leader.capture_reference(require_saved_pose=require_saved_pose)
        status=self.leader.status
        if saved:self.set_active(False)
        self.input_mapping.update({k:v for k,v in self.leader.input_mapping.items()
                                   if k.startswith('arm_calibration') or k=='saved_reference_file'})
        self.status=status
        return saved

    def hold(self, frame, *, now_ns):
        # A slow control tick must not commit any target computed from expired input.
        return dict(self.acknowledged)

    def command(self, frame, measured, *, now_ns):
        self.diagnostics = {}
        candidate_stamp=None
        if not self.armed:
            # Keep fresh reference measurements available while the operator matches poses.
            if not self.hand_only:
                leader_frame=None if frame is None else replace(frame,host_ns=frame.tracking['leader_host_ns'])
                self.leader.command(leader_frame,measured,now_ns=now_ns)
            if not self.status.startswith(('Matched reference saved','Simulation aligned','Leader alignment refused')):
                self.status = 'Paused; press Start to follow leader and MANUS fingers'
            return dict(self.acknowledged)
        leader_frame = None
        desired = dict(self.acknowledged)
        if not self.hand_only:
            leader_frame = None if frame is None else replace(frame, host_ns=frame.tracking['leader_host_ns'])
            arm_targets = self.leader.command(leader_frame, measured, now_ns=now_ns)
            self.input_mapping.update({k:self.leader.input_mapping[k] for k in (
                'space','anchor_policy','arm_calibration','arm_calibration_sha256')})
            desired.update({n: arm_targets[n] for n in self.model.arm.names})
            if not self.leader.armed:
                self.diagnostics['arm'] = self.leader.status
                self.diagnostics.update(self.leader.diagnostics)
        raw = frame.tracking.get('manus') if frame else None
        try:
            if raw is None:
                raise ValueError('No MANUS data')
            packet = AcquisitionPacket.model_validate(raw)
            if packet.device.device_id != self.glove_id or packet.device.hand != 'right':
                raise ValueError('Wrong MANUS glove/side')
            result = self.bridge.accept(packet, now_ns)
            if result.status == 'accepted':
                self.hand_frame = result.frame
            elif result.status != 'duplicate':
                raise ValueError(result.reason)
            hand = self.hand_frame
            if (hand is None or not hand['hand_control_valid'] or
                    not 0 <= now_ns-hand['mapped_capture_time_ns'] <= 150_000_000):
                raise ValueError('MANUS stale/invalid; Pause then Start to resume fingers')
            if self.hand_enabled:
                if self.optimizer_metadata:
                    from bench.adapters.manus.sharpa import candidate_targets
                    target=candidate_targets(frame.tracking.get('manus_retarget'),raw,self.model,now_ns=now_ns)
                    if target is None:
                        self.diagnostics['hand']='Waiting for fresh official SHARPA targets; holding'
                    else:
                        candidate_stamp=frame.tracking['manus_retarget']['capture_time_ns']
                        desired.update(target)
                else:
                    joints = finger_sample(hand)
                    q = self.model.measured_reference({n: measured[n] for n in self.model.names})
                    if self.calibration is None:
                        self.calibration = ManusHandCalibration(self.model, q, joints)
                    else:
                        points = self.calibration.targets(joints)
                        deadline = time.monotonic_ns()+40_000_000
                        for finger, point in points.items():
                            solved, error = finger_step(self.model, q, finger, point, deadline_ns=deadline)
                            desired.update(solved)
                            self.finger_errors[self.name+'/'+finger] = error
            else:
                self.diagnostics['hand'] = 'MANUS held; Pause then Start to resume'
        except (ValueError, KeyError) as error:
            self.hand_enabled = False
            self.diagnostics['hand'] = str(error)
        finished = time.monotonic_ns()
        if leader_frame is not None and finished-leader_frame.host_ns > 150_000_000:
            self.leader.hold(leader_frame, now_ns=finished)
            desired.update({n: self.acknowledged[n] for n in self.model.arm.names})
        if (self.hand_frame is not None and self.hand_frame['mapped_capture_time_ns'] is not None and
                finished-self.hand_frame['mapped_capture_time_ns'] > 150_000_000):
            self.hand_enabled = False
            desired.update({n: self.acknowledged[n] for n in desired if n not in self.model.arm.names})
            self.diagnostics['hand'] = 'MANUS sample expired during mapping; holding'
        if candidate_stamp is not None and finished-candidate_stamp > 150_000_000:
            desired.update({n:self.acknowledged[n] for n in self.model.hand.names})
            self.diagnostics['hand']='Official SHARPA candidate expired during mapping; holding'
        self.status = ('MANUS fingers; arm held (hand-only)' if self.hand_only else
                       'FACTR arm + MANUS fingers') if not self.diagnostics else 'Affected input held'
        targets = {n: self.acknowledged[n]+float(np.clip(v-self.acknowledged[n],
                -self.velocity[n]*.02, self.velocity[n]*.02)) for n, v in desired.items()}
        if self.leader.unrestricted_arm:
            targets.update({n:desired[n] for n in self.model.arm.names})
        return targets
