"""Compose independent PICO wrist and existing MANUS finger controllers."""
from dataclasses import replace
import time

from uni_teleop.devices.pico.protocol import Frame
from uni_teleop.mapping.franka.retarget import Retargeter
from uni_teleop.mapping.factr_manus import ManusJointRetargeter


class PicoManusReceiver:
    rejected = 0

    def __init__(self, pico, glove, optimizer=None):
        self.pico, self.glove, self.optimizer = pico, glove, optimizer

    def read(self):
        candidate = self.optimizer.latest if self.optimizer else None
        pico, hand = self.pico.read(), self.glove.latest
        if pico is None and hand is None:
            return None
        base = pico or Frame('pico-missing', 0, 0, '', 'PICO+MANUS',
                             {'head':None, 'hands':{}, 'controllers':{}})
        # Envelope freshness never substitutes for either channel's own clock.
        return replace(base, host_ns=max(base.host_ns, hand['capture_time_ns'] if hand else 0),
            tracking=base.tracking | {'pico_host_ns':base.host_ns, 'pico_present':pico is not None,
                                      'manus':hand, 'manus_retarget':candidate})

    def health(self):
        return {'pico':self.pico.health(), 'manus':self.glove.health(),
                'optimizer':self.optimizer.health() if self.optimizer else {'enabled':False}}

    def close(self):
        if self.optimizer:
            self.optimizer.close()
        self.pico.close()
        self.glove.close()


class PicoManusRetargeter:
    input_hint = 'PICO grip: move arm; release: clutch. MANUS: fingers; PICO trigger unused.'

    def __init__(self, geometry, initial, bindings, *, glove_id, calibration_id,
                 finger_speed=4., optimizer_metadata=None, **kwargs):
        self.arm = Retargeter(geometry, initial, bindings, arm_only=True, **kwargs)
        self.hand = ManusJointRetargeter(geometry, initial, bindings, glove_id=glove_id,
            calibration_id=calibration_id, finger_speed=finger_speed, hand_only=True,
            optimizer_metadata=optimizer_metadata, **kwargs)
        self.session_active = False
        self.status, self.diagnostics = 'Paused', {}
        self.input_mapping = dict(self.hand.input_mapping, space='pico_wrist_manus_fingers',
                                  arm_source='pico', leader_enabled=False, pico_mode=self.arm.mode)

    def __getattr__(self, name):
        return getattr(self.arm, name)

    def set_active(self, active):
        self.session_active = bool(active)
        self.arm.set_active(active)
        self.hand.set_active(active)

    def set_mode(self, mode):
        self.set_active(False)
        self.arm.set_mode(mode)
        self.input_mapping['pico_mode'] = mode

    def recalibrate(self):
        self.set_active(False)
        self.arm.recalibrate()
        self.hand.recalibrate()

    def acknowledge(self, targets):
        self.arm.acknowledge(targets)
        self.hand.acknowledge(targets)

    @staticmethod
    def arm_frame(frame):
        if frame is None or not frame.tracking['pico_present']:
            return None
        return replace(frame, host_ns=frame.tracking['pico_host_ns'])

    def hold(self, frame, *, now_ns):
        self.arm.hold(self.arm_frame(frame), now_ns=now_ns)
        return dict(self.acknowledged)

    def command(self, frame, measured, *, now_ns):
        pico = self.arm_frame(frame)
        targets = self.arm.command(pico, measured, now_ns=now_ns)
        fingers = self.hand.command(frame, measured, now_ns=now_ns)
        finished = time.monotonic_ns()
        if pico is not None and finished-pico.host_ns > 150_000_000:
            self.arm.hold(pico, now_ns=finished)
            targets.update({n:self.acknowledged[n] for n in self.hand.model.arm.names})
        targets.update({n:fingers[n] for n in self.hand.model.hand.names})
        self.diagnostics = dict(self.arm.diagnostics)
        self.diagnostics.update(self.hand.diagnostics)
        self.status = 'PICO: '+self.arm.status+'; MANUS: '+self.hand.status
        return targets
