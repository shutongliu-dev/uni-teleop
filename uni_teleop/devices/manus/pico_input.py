"""PICO arm + MANUS hand input contract.

PICO remains the sole source for arm/wrist motion. MANUS is the sole source
for hand skeleton data; this object deliberately does not merge joint targets.
"""
from __future__ import annotations
from dataclasses import dataclass
import time

@dataclass(frozen=True)
class PicoManusSample:
    pico_frame: object
    manus_packet: dict | None
    manus_age_s: float | None
    hand_valid: bool
    hold_reason: str | None = None

class PicoManusInput:
    def __init__(self, manus_source, *, max_manus_age_s=.15):
        if not 0 < max_manus_age_s <= 1: raise ValueError("invalid MANUS watchdog")
        self.manus_source = manus_source
        self.max_age_s = max_manus_age_s

    def sample(self, pico_frame):
        packet = self.manus_source.latest
        age = None
        if packet is not None:
            age = max(0., time.monotonic_ns() - int(packet.get("receive_time_ns", time.monotonic_ns()))) / 1e9
        valid = packet is not None and age is not None and age <= self.max_age_s and bool(packet.get("glove_tracking_valid"))
        return PicoManusSample(pico_frame, packet, age, valid, None if valid else "manus_tracking_stale_or_missing")
