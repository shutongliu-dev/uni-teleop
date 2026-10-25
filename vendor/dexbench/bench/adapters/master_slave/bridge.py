"""Master-slave arm + MANUS hand aggregation.

This module contains no Dynamixel or robot SDK imports. A hardware-specific
leader reader supplies calibrated arm state; the existing isolated MANUS
process supplies hand packets. The PICO paths never enter this module.
"""
from __future__ import annotations
from dataclasses import dataclass
import math, time

@dataclass(frozen=True)
class LeaderState:
    joint_positions_rad: tuple[float, ...]
    gripper: float
    timestamp_monotonic: float

@dataclass(frozen=True)
class MasterSlaveSample:
    leader: LeaderState | None
    manus_packet: dict | None
    manus_age_s: float | None
    arm_valid: bool
    hand_valid: bool
    hold_reasons: tuple[str, ...]

class MasterSlaveBridge:
    """Combine only master-arm state and MANUS hand state.

    The bridge does not command the follower. The owner loop decides whether
    to send ``leader.joint_positions_rad`` after applying robot limits and
    acknowledgement rules.
    """
    def __init__(self, manus_source, *, arm_dof=7, max_leader_age_s=.15, max_manus_age_s=.15):
        if arm_dof < 1 or not 0 < max_leader_age_s <= 1 or not 0 < max_manus_age_s <= 1:
            raise ValueError("invalid master-slave watchdog configuration")
        self.manus_source = manus_source
        self.arm_dof = arm_dof
        self.max_leader_age_s = max_leader_age_s
        self.max_manus_age_s = max_manus_age_s

    def sample(self, leader: LeaderState | None) -> MasterSlaveSample:
        reasons = []
        now = time.monotonic()
        arm_valid = leader is not None and len(leader.joint_positions_rad) == self.arm_dof
        if arm_valid:
            arm_valid = all(math.isfinite(float(x)) for x in leader.joint_positions_rad)
            arm_valid = arm_valid and math.isfinite(leader.gripper)
            arm_valid = arm_valid and 0 <= leader.gripper <= 1
            arm_valid = arm_valid and 0 <= now-leader.timestamp_monotonic <= self.max_leader_age_s
        if not arm_valid: reasons.append("leader_arm_stale_or_invalid")
        packet = self.manus_source.latest
        age = self.manus_source.health().get("latest_age_s")
        hand_valid = packet is not None and age is not None and age <= self.max_manus_age_s and bool(packet.get("glove_tracking_valid"))
        if not hand_valid: reasons.append("manus_hand_stale_or_invalid")
        return MasterSlaveSample(leader, packet, age, arm_valid, hand_valid, tuple(reasons))
