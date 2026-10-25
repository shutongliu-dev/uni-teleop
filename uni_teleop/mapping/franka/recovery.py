"""Bounded hands-only resumption from fresh, stable tracking and a spare-hand clutch."""
from __future__ import annotations

import numpy as np


class HandRecovery:
    stable_ns = 250_000_000
    automatic_ns = 2_000_000_000

    def __init__(self):
        self.clear()

    def clear(self):
        self.last_valid_ns = None
        self.manual = False
        self.since_ns = None
        self.reference = None
        self.stamp = None
        self.gesture = 'release'
        self.pinch_since_ns = None
        self.confirmed = False

    @property
    def pending(self):
        return self.last_valid_ns is not None

    def interrupt(self, last_valid_ns):
        if not self.pending:
            self.last_valid_ns = last_valid_ns
        self.since_ns = self.reference = self.stamp = None
        self.gesture, self.pinch_since_ns, self.confirmed = 'release', None, False

    def update(self, frame, sources, now_ns):
        self.manual |= now_ns-self.last_valid_ns > self.automatic_ns
        if self.stamp is not None and frame.host_ns <= self.stamp:
            return False  # A cached packet cannot complete a dwell or gesture.
        if self.stamp is not None and frame.host_ns-self.stamp > 150_000_000:
            self.interrupt(self.last_valid_ns)
        self.stamp = frame.host_ns
        self.last_valid_ns = frame.host_ns
        wrists = np.array([frame.tracking['hands'][side][1] for side in sources])
        stable = (self.reference is not None
                  and np.all(np.linalg.norm(wrists[:, :3]-self.reference[:, :3], axis=1) <= .03)
                  and np.all(np.abs(np.sum(wrists[:, 3:]*self.reference[:, 3:], axis=1)) >= np.cos(np.deg2rad(7.5))))
        if not stable:
            self.reference, self.since_ns = wrists.copy(), frame.host_ns
        if self.manual:
            spare = {'left', 'right'}-set(sources)
            hand = frame.tracking['hands'].get(next(iter(spare))) if spare else None
            if hand is None or hand[5] is None or hand[10] is None:
                self.gesture, self.pinch_since_ns, self.confirmed = 'release', None, False
            else:
                gap = np.linalg.norm(np.array(hand[5][:3])-hand[10][:3])
                if gap >= .045:
                    self.gesture, self.pinch_since_ns = 'pinch', None
                elif self.gesture == 'pinch' and gap <= .025:
                    if self.pinch_since_ns is None:
                        self.pinch_since_ns = frame.host_ns
                    self.confirmed |= frame.host_ns-self.pinch_since_ns >= self.stable_ns
                else:
                    self.pinch_since_ns = None
        return frame.host_ns-self.since_ns >= self.stable_ns and (not self.manual or self.confirmed)

    def status(self, sources):
        if not self.manual:
            return 'tracking interrupted — hold hand steady to resume'
        spare = {'left', 'right'}-set(sources)
        if not spare:
            return 'tracking restored — press Start (both hands control robots)'
        return f'tracking restored — open then pinch {next(iter(spare))} thumb/index to resume'
