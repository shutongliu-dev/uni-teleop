"""Simulator-independent five-finger hand conventions."""

from __future__ import annotations

import numpy as np

FINGERS = ("thumb", "index", "middle", "ring", "pinky")
JOINTS_PER_FINGER = 4
HAND_DOF = len(FINGERS) * JOINTS_PER_FINGER
DEXHAND_JOINT_NAMES = tuple(
    f"{side}_hand_{finger}_{joint}_joint"
    for side in ("left", "right")
    for finger in FINGERS
    for joint in range(JOINTS_PER_FINGER)
)
DEXHAND_DOF = len(DEXHAND_JOINT_NAMES)


def dexhand_controller_targets(
    left_trigger: float,
    left_grip: float,
    right_trigger: float,
    right_grip: float,
) -> np.ndarray:
    """Controller fallback: trigger pinches; grip closes the other digits."""

    def one_hand(trigger: float, grip: float, side: str) -> np.ndarray:
        trigger = float(np.clip(trigger, 0.0, 1.0))
        grip = float(np.clip(grip, 0.0, 1.0))
        close = max(trigger, grip)
        sign = -1.0 if side == "left" else 1.0
        thumb = np.array(
            [-0.55 * close, 0.25 * sign * close, sign * close, sign * close]
        )
        index = np.array(
            [0.0, 1.35 * sign * trigger, 1.45 * sign * trigger, 1.2 * sign * trigger]
        )
        other = np.array(
            [0.0, 1.35 * sign * grip, 1.45 * sign * grip, 1.2 * sign * grip]
        )
        return np.concatenate((thumb, index, other, other, other))

    return np.concatenate(
        (
            one_hand(left_trigger, left_grip, "left"),
            one_hand(right_trigger, right_grip, "right"),
        )
    )
