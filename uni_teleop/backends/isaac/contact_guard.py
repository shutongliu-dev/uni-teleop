"""Operator-only contact policy; benchmark safety gates remain unchanged."""
from copy import deepcopy

from bench.evaluation.rigid_safety import RigidContactSafetyGuard


class TeleopContactGuard:
    def __init__(self, *, practice=False):
        self.practice = practice
        self.trigger = self.warning = self.last_warning = None
        self.warning_windows = self.streak = 0
        self.previous_end = None

    def assert_can_step(self):
        if self.trigger is not None:
            raise ValueError('Teleoperation contact stop is latched; reset scene before stepping')

    def observe(self, observation):
        if self.trigger is not None:
            return deepcopy(self.trigger)
        # Keep the strict validator, including complete substep receipt checks.
        event = RigidContactSafetyGuard().observe(observation)
        self.warning = None
        if event is None:
            self.streak, self.previous_end = 0, None
            return None
        if not self.practice:
            self.trigger = event
            return deepcopy(event)
        start, end = event['window_start_sim_time_s'], event['window_end_sim_time_s']
        self.streak = self.streak + 1 if self.previous_end is not None and abs(start-self.previous_end) < 1e-8 else 1
        self.previous_end = end
        self.warning_windows += 1
        self.warning = event | {'schema': 'teleop_contact_warning.v1', 'consecutive_windows': self.streak,
                                'policy': 'practice_warning', 'formal_evaluation': False}
        self.last_warning = deepcopy(self.warning)
        return None

    def message(self, sim_time_s):
        if self.trigger:
            depth = -1000*self.trigger['minimum_separation_m']
            reason = self.trigger['reason'].replace('_', ' ')
            return f'Stopped: {reason} ({depth:.2f} mm); Reset scene, then Start'
        if self.last_warning and sim_time_s-self.last_warning['window_end_sim_time_s'] <= 2.:
            depth = -1000*self.last_warning['minimum_separation_m']
            return f'Contact warning: {depth:.2f} mm; practice continues'
        return ''
