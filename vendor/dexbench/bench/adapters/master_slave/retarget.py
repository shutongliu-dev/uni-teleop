"""Fixed joint mapping with a manual pre-start alignment gate; no motor writes."""
import time
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import numpy as np

from bench.adapters.pico.retarget import Retargeter
from .calibration import ArmCalibration, JointLimitError


class JointRetargeter(Retargeter):
    control_space = 'joint'
    input_hint = 'Fixed calibration. Start: manually match all 7 robot joints; arm holds until aligned.'

    def __init__(self, geometry, initial, bindings, *, trigger_sign=1, arm_calibration=None,
                 grip_range=.8, fixed_reference=False, calibration_output=None, follow_on_start=False,
                 practice_follow=False, unrestricted_arm=False, **kwargs):
        super().__init__(geometry, initial, bindings, **kwargs)
        if len(bindings) != 1:
            raise ValueError('One FACTR leader controls one bound arm')
        self.name = next(iter(bindings))
        self.model = self.models[self.name]
        if trigger_sign not in (-1, 1) or not 0 < grip_range <= 2:
            raise ValueError('Require trigger sign +1/-1 and a positive trigger range')
        self.trigger_sign, self.grip_range = trigger_sign, grip_range
        self.arm_calibration = arm_calibration
        if arm_calibration:
            arm_calibration.validate_robot(self.model)
            if arm_calibration.status == 'provisional':
                self.input_hint = 'Approximate reference; arm directions from preset, not individually calibrated'
        self.alignment_tolerance = .05
        self.alignment_dwell_ns = 250_000_000
        self.input_mapping = {'space': 'fixed_joint', 'grip_range_rad': grip_range,
            'trigger_sign': float(self.trigger_sign), 'hand_source': 'leader_trigger',
            'arm_calibration': arm_calibration.model_dump(mode='json') if arm_calibration else None,
            'arm_calibration_sha256': arm_calibration.sha256 if arm_calibration else None,
            'alignment_tolerance_rad': self.alignment_tolerance, 'alignment_dwell_s': .25,
            'alignment_stability_rad': .02}
        self.provisional = bool(arm_calibration and arm_calibration.status == 'provisional')
        self.fixed_reference=bool(fixed_reference)
        self.follow_on_start=bool(follow_on_start)
        self.practice_follow=bool(practice_follow)
        self.unrestricted_arm=bool(unrestricted_arm)
        if self.unrestricted_arm and not self.practice_follow:
            raise ValueError('Unrestricted simulated arm requires practice follow')
        if self.practice_follow and not (self.fixed_reference and self.follow_on_start):
            raise ValueError('Practice follow requires fixed joint follow-on-Start')
        self.input_mapping['practice_follow']=self.practice_follow
        self.input_mapping['unrestricted_simulated_arm']=self.unrestricted_arm
        if self.follow_on_start and not self.fixed_reference:
            raise ValueError('Follow on Start requires fixed joint calibration')
        self.calibration_output=Path(calibration_output) if calibration_output else None
        self.latest_reference=None
        self.input_mapping.update(space='relative_joint_practice' if self.provisional else 'fixed_joint',
                                  anchor_policy='measured_pose_on_Start' if self.provisional else 'saved_fixed_reference',
                                  alignment_dwell_s=0. if self.provisional else .25,
                                  alignment_tolerance_rad=None if self.provisional else self.alignment_tolerance)
        if self.fixed_reference:
            self.input_mapping.update(space='fixed_joint',anchor_policy='saved_fixed_reference',
                                      alignment_dwell_s=.25,alignment_tolerance_rad=self.alignment_tolerance)
            self.input_hint='Read leader -> align simulation, then Start. Saved joint zero stays fixed.'
        if self.follow_on_start:
            self.input_hint='Start: robot follows leader joints at bounded speed. Pause: robot holds.'
            self.input_mapping['startup']='bounded_joint_follow_on_Start'
        if self.unrestricted_arm:
            self.input_hint='Start: directly follow calibrated leader joints without simulated arm motion limits.'
            self.input_mapping['startup']='direct_joint_follow_on_Start'
        self.provisional_anchor_pending = False
        self.alignment_pending = False
        self.previous_raw = self.previous_stamp = self.previous_stream = None
        self.aligned_since = self.aligned_pose = self.trigger_zero = None
        self.alignment = {'state': 'paused'}
        self.issue = None

    @property
    def recovery_pending(self):
        return self.alignment_pending

    def set_active(self, active):
        super().set_active(False)
        self.alignment_pending = bool(active and self.arm_calibration is not None)
        self.provisional_anchor_pending = bool(active and self.provisional and not self.fixed_reference)
        self.previous_raw = self.previous_stamp = self.previous_stream = None
        self.aligned_since = self.aligned_pose = self.trigger_zero = None
        state = 'waiting' if self.alignment_pending else 'paused'
        self.alignment = {'state': state}
        self.issue = None
        self.status = ('Synchronizing robot to leader' if self.alignment_pending and self.follow_on_start else
                       'Align leader manually; robot held' if self.alignment_pending else
                       'Arm calibration missing; arm held' if active else 'Master/slave paused')

    def recalibrate(self):
        # Scene reset clears permission and transient alignment, never the saved zero.
        self.set_active(False)
        self.latest_reference=None

    def initial_state_from_input(self, frame, initial, *, now_ns):
        """Reset the simulated arm to a fresh calibrated leader pose, without rebasing."""
        if not self.fixed_reference or self.arm_calibration is None:
            raise ValueError('Reading the leader pose requires a fixed arm calibration')
        if frame is not None and 'leader_host_ns' in frame.tracking:
            frame=replace(frame,host_ns=frame.tracking['leader_host_ns'])
        if not self._fresh(frame,now_ns):
            raise ValueError(self.status)
        raw=frame.tracking['leader']
        target=self.arm_calibration.targets(raw,self.model)
        state=deepcopy(initial)
        joints=dict(zip(self.model.arm.names,target.tolist()))
        state['joint_positions_rad'].update(joints)
        receipt={'leader_host_ns':frame.host_ns,'leader_sequence':frame.sequence,
                 'leader_id':frame.tracking['leader_id'],'leader_raw_rad':list(raw),
                 'arm_calibration_sha256':self.arm_calibration.sha256,
                 'arm_calibration_status':self.arm_calibration.status,'arm_joint_positions_rad':joints}
        return state,receipt

    def capture_reference(self, *, require_saved_pose=False):
        """Operator declares the physically matched posture; never verifies directions."""
        if require_saved_pose and (self.armed or self.alignment_pending):
            self.status='Pause and reset the simulation before capturing the reference'
            return False
        if (self.latest_reference is None or self.calibration_output is None or
                time.monotonic_ns()-self.latest_reference[0]>150_000_000):
            self.status='Fresh leader/robot state required to save reference'
            return False
        stamp,raw,actual=self.latest_reference
        cal=self.arm_calibration
        if require_saved_pose and np.max(np.abs(np.array(actual)-cal.robot_reference_rad))>.05:
            self.status='Reset the simulation to the saved reference pose first'
            return False
        self.set_active(False)
        saved=ArmCalibration.from_reference(self.model,leader_id=cal.leader_id,
            leader_reference_rad=raw[:7],robot_reference_rad=actual,signs=cal.signs,
            directions_verified=cal.directions_verified,source_kind=cal.source_kind,
            status=cal.status)
        path=self.calibration_output/f'arm_reference_{stamp}.json'
        path.parent.mkdir(parents=True,exist_ok=True)
        saved.save(path)
        self.latest_reference=None
        self.arm_calibration=saved
        self.input_mapping.update(arm_calibration=saved.model_dump(mode='json'),
            arm_calibration_sha256=saved.sha256,saved_reference_file=str(path))
        self.status='Matched reference saved; Start to align and follow fixed joints'
        return True

    def _stop(self, reason, *, code='held', **details):
        self.set_active(False)
        self.status = reason+'; holding, Start to align again'
        self.alignment = {'state': 'held', 'reason': reason}
        self.issue = {'code':code, 'reason':reason, **details}

    def _fresh(self, frame, now_ns):
        if frame is None or not 0 <= now_ns-frame.host_ns <= 150_000_000:
            self._stop('Leader disconnected/stale', code='leader_stale',
                       age_ms=(now_ns-frame.host_ns)/1e6 if frame else None)
            return False
        raw = np.asarray(frame.tracking.get('leader', []), dtype=float)
        if raw.shape != (8,) or not np.isfinite(raw).all():
            self._stop('Invalid leader sample', code='invalid_leader')
            return False
        if self.arm_calibration and frame.tracking.get('leader_id') != self.arm_calibration.leader_id:
            self._stop('Leader identity differs from calibration', code='leader_identity')
            return False
        return True

    def hold(self, frame, *, now_ns):
        self._fresh(frame, now_ns)
        return dict(self.acknowledged)

    def command(self, frame, measured, *, now_ns):
        self.diagnostics = {}
        held = dict(self.acknowledged)
        if self.arm_calibration is None:
            self.set_active(False)
            self.status = 'Arm calibration missing; arm held'
            return held
        if not self._fresh(frame, now_ns):
            return held
        self.latest_reference=(frame.host_ns,list(frame.tracking['leader']),
                               [measured[n] for n in self.model.arm.names])
        if not (self.armed or self.alignment_pending):
            # Preserve the last hold reason until the operator explicitly restarts.
            return held
        raw = np.array(frame.tracking['leader'])
        if self.provisional_anchor_pending:
            cal = self.arm_calibration
            try:
                self.arm_calibration = ArmCalibration.from_reference(self.model,
                    leader_id=cal.leader_id, leader_reference_rad=raw[:7].tolist(),
                    robot_reference_rad=[measured[n] for n in self.model.arm.names],
                    signs=cal.signs, directions_verified=cal.directions_verified,
                    source_kind=cal.source_kind, status='provisional')
            except ValueError:
                self._stop('Cannot anchor from invalid measured arm pose')
                return held
            self.input_mapping.update(arm_calibration=self.arm_calibration.model_dump(mode='json'),
                                      arm_calibration_sha256=self.arm_calibration.sha256)
            self.provisional_anchor_pending = False
            self.previous_raw, self.previous_stamp, self.previous_stream = raw.copy(), frame.host_ns, frame.stream
            self.trigger_zero = raw[7]
            self.armed, self.alignment_pending = True, False
            self.alignment = {'state':'relative_anchored'}
            self.status = 'Relative practice ready; move leader'
            return held  # Explicit Start holds its first frame, then follows immediately.
        if self.previous_stamp is not None:
            if (frame.stream != self.previous_stream or frame.host_ns < self.previous_stamp or
                    (not self.practice_follow and np.max(np.abs(raw-self.previous_raw)) > .35)):
                axis = int(np.argmax(np.abs(raw-self.previous_raw)))
                self._stop(f'Leader stream/encoder jump at motor {axis+1}', code='encoder_jump',
                           motor_id=axis+1, delta_rad=float(raw[axis]-self.previous_raw[axis]),
                           threshold_rad=.35, stream_changed=frame.stream!=self.previous_stream)
                return held
            if frame.host_ns == self.previous_stamp:
                return held
            if self.alignment_pending and frame.host_ns-self.previous_stamp > 150_000_000:
                # Slow rendering must not count unsampled time towards alignment.
                self.aligned_since = self.aligned_pose = None
        self.previous_raw, self.previous_stamp, self.previous_stream = raw.copy(), frame.host_ns, frame.stream
        if self.practice_follow:
            reference=[held[n] for n in self.model.arm.names]
            if self.unrestricted_arm:
                arm,limits=self.arm_calibration.continuous_targets(raw,reference),[]
            else:
                arm,limits=self.arm_calibration.practice_targets(raw,self.model,reference)
            self.armed,self.alignment_pending=True,False
            self.alignment={'state':'limited' if limits else 'following','limited_joints':limits}
            self.issue=({'code':'joint_saturated','joint':limits[0]['joint'],
                         'limits':limits,'reason':'Only limited joints stop at their boundary; other joints follow'}
                        if limits else None)
            self.status='Practice following; joint boundary reached' if limits else 'Practice joint following'
            if self.trigger_zero is None:self.trigger_zero=raw[7]
            grip=float(np.clip(self.trigger_sign*(raw[7]-self.trigger_zero)/self.grip_range,0.,1.))
            desired=held|dict(zip(self.model.arm.names,arm.tolist()))|self.presets[self.name].targets(grip)
            targets={n:held[n]+float(np.clip(v-held[n],-self.velocity[n]*.02,self.velocity[n]*.02))
                     for n,v in desired.items()}
            if self.unrestricted_arm:
                targets.update(zip(self.model.arm.names,arm.tolist()))
            return targets
        try:
            arm = self.arm_calibration.targets(raw, self.model)
        except ValueError as error:
            self.aligned_since = self.aligned_pose = None
            if self.armed:
                self._stop(str(error))
            else:
                self.status = str(error)+'; mapped target outside limits, robot held'
                self.alignment = {'state': 'outside_limits'}
            self.issue = ({**error.details, 'reason':str(error)} if isinstance(error, JointLimitError) else
                          {'code':'mapping_error','reason':str(error)})
            return held
        if self.alignment_pending:
            actual = np.array([measured[n] for n in self.model.arm.names])
            error = arm-actual
            ack_error = arm-np.array([held[n] for n in self.model.arm.names])
            self.alignment = {'state': 'waiting', 'error_rad': dict(zip(self.model.arm.names, error.tolist()))}
            worst = int(np.argmax(np.abs(error)))
            self.status = f'Align: {self.model.arm.names[worst]} error {np.degrees(error[worst]):+.1f} deg; robot held'
            self.issue = {'code':'alignment_wait', 'joint':self.model.arm.names[worst],
                          'error_rad':float(error[worst]), 'tolerance_rad':self.alignment_tolerance,
                          'reason':self.status}
            self.diagnostics = {n: f'{n}: {np.degrees(v):+.1f} deg' for n, v in zip(self.model.arm.names, error)}
            if (not np.isfinite(actual).all() or np.max(np.abs(error)) > self.alignment_tolerance or
                    np.max(np.abs(ack_error)) > self.alignment_tolerance):
                self.aligned_since = self.aligned_pose = None
                if self.follow_on_start and np.isfinite(actual).all():
                    self.alignment['state']='synchronizing'
                    self.status=f'Synchronizing joints: {self.model.arm.names[worst]} {np.degrees(error[worst]):+.1f} deg'
                    self.issue.update(code='joint_synchronizing',reason=self.status)
                    desired=held|dict(zip(self.model.arm.names,arm.tolist()))
                    return {n:held[n]+float(np.clip(v-held[n],-self.velocity[n]*.02,self.velocity[n]*.02))
                            for n,v in desired.items()}
                return held
            if self.aligned_pose is None or np.max(np.abs(arm-self.aligned_pose)) > .02:
                self.aligned_since, self.aligned_pose = frame.host_ns, arm.copy()
            dwell = frame.host_ns-self.aligned_since
            self.alignment.update(state='settling', dwell_s=dwell/1e9)
            if dwell >= self.alignment_dwell_ns:
                self.armed, self.alignment_pending = True, False
                self.trigger_zero = raw[7]
                self.alignment['state'] = 'aligned'
                self.status = 'Aligned; fixed joint tracking ready'
                self.issue = None
                self.diagnostics = {}
            return held  # Even the enabling frame cannot move the robot.
        grip = float(np.clip(self.trigger_sign*(raw[7]-self.trigger_zero)/self.grip_range, 0., 1.))
        desired = held | dict(zip(self.model.arm.names, arm.tolist())) | self.presets[self.name].targets(grip)
        mode = 'Relative practice' if self.provisional and not self.fixed_reference else 'Fixed joint tracking'
        self.status = f'{mode}; trigger {grip:.0%}'
        return {n: held[n] + float(np.clip(v-held[n], -self.velocity[n]*.02, self.velocity[n]*.02))
                for n, v in desired.items()}
