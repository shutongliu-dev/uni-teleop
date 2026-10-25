"""Joint-semantic mapping from the local hand to the Tesollo DG5F hand."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ..hand import HAND_DOF, JOINTS_PER_FINGER

Array = NDArray[np.float64]

DG5F_JOINT_NAMES = tuple(
    f"dg5f_joint_{finger}_{joint}" for finger in range(1, 6) for joint in range(1, 5)
)

# Poses are taken from the SuperDex 1.0 DG5F long-hand assets.  They keep the
# fingers slightly flexed instead of commanding the mechanical hard stops.
_DG5F_OPEN_RIGHT = np.array(
    [
        0.2617994,
        -0.9599311,
        0.2617994,
        0.2617994,
        -0.1047198,
        0.2617994,
        0.2617994,
        0.2617994,
        -0.0349066,
        0.2617994,
        0.2617994,
        0.2617994,
        0.0349066,
        0.2617994,
        0.2617994,
        0.2617994,
        0.0872665,
        0.1047198,
        0.2617994,
        0.2617994,
    ],
    dtype=np.float64,
)
_DG5F_OPEN_LEFT = _DG5F_OPEN_RIGHT.copy()
_DG5F_OPEN_LEFT[:4] *= -1.0
_DG5F_OPEN_LEFT[[4, 8, 12, 16, 17]] *= -1.0

_DG5F_MIN_RIGHT = np.array(
    [
        -0.383972,
        -3.141590,
        0.0,
        0.0,
        -0.418879,
        0.0,
        0.0,
        0.0,
        -0.610865,
        0.0,
        0.0,
        0.0,
        -0.05,
        0.0,
        0.0,
        0.0,
        -0.017453,
        -0.05,
        0.0,
        0.0,
    ],
    dtype=np.float64,
)
_DG5F_MAX_RIGHT = np.array(
    [
        0.890118,
        0.0,
        1.5708,
        1.5708,
        0.05,
        2.00713,
        1.5708,
        1.5708,
        0.610865,
        1.95477,
        1.5708,
        1.5708,
        0.418879,
        1.90241,
        1.5708,
        1.5708,
        1.0472,
        0.610865,
        1.5708,
        1.5708,
    ],
    dtype=np.float64,
)
_DG5F_MIN_LEFT = np.array(
    [
        -0.890118,
        0.0,
        -1.5708,
        -1.5708,
        -0.05,
        0.0,
        0.0,
        0.0,
        -0.610865,
        0.0,
        0.0,
        0.0,
        -0.418879,
        0.0,
        0.0,
        0.0,
        -1.0472,
        -0.610865,
        0.0,
        0.0,
    ],
    dtype=np.float64,
)
_DG5F_MAX_LEFT = np.array(
    [
        0.383972,
        3.141590,
        0.0,
        0.0,
        0.418879,
        2.00713,
        1.5708,
        1.5708,
        0.610865,
        1.95477,
        1.5708,
        1.5708,
        0.05,
        1.90241,
        1.5708,
        1.5708,
        0.017453,
        0.05,
        1.5708,
        1.5708,
    ],
    dtype=np.float64,
)


def _finite_hand(target: Array) -> Array:
    value = np.asarray(target, dtype=np.float64)
    if value.shape != (HAND_DOF,):
        raise ValueError(f"hand target must have shape ({HAND_DOF},)")
    if not np.isfinite(value).all():
        raise ValueError("hand target must contain finite values")
    return value


def _unit_flexion(value: float, side_sign: float, maximum: float) -> float:
    return float(np.clip(side_sign * value / maximum, 0.0, 1.0))


def dexhand_to_dg5f(target: Array, side: str) -> Array:
    """Map one local 20-DoF hand target into DG5F long-hand joint order.

    The local model uses one spread plus three flexion joints for every
    non-thumb digit.  DG5F uses two base joints and two flexion joints on the
    pinky, so its final two targets are flexion synergies instead of a raw
    positional copy.
    """

    if side not in {"left", "right"}:
        raise ValueError("side must be 'left' or 'right'")
    value = _finite_hand(target).reshape(5, JOINTS_PER_FINGER)
    side_sign = -1.0 if side == "left" else 1.0
    pose = (_DG5F_OPEN_LEFT if side == "left" else _DG5F_OPEN_RIGHT).copy()
    lower = _DG5F_MIN_LEFT if side == "left" else _DG5F_MIN_RIGHT
    upper = _DG5F_MAX_LEFT if side == "left" else _DG5F_MAX_RIGHT

    thumb_opposition = float(np.clip(-value[0, 0] / 0.65, 0.0, 1.0))
    thumb_spread = float(np.clip(side_sign * value[0, 1] / 0.8, -1.0, 1.0))
    pose[0] = side_sign * (0.2617994 + 0.48 * thumb_opposition)
    pose[1] = -side_sign * (0.9599311 - 0.28 * thumb_opposition + 0.32 * thumb_spread)
    pose[2] = side_sign * (
        0.2617994 + 1.16 * _unit_flexion(value[0, 2], side_sign, 1.5)
    )
    pose[3] = side_sign * (
        0.2617994 + 1.16 * _unit_flexion(value[0, 3], side_sign, 1.4)
    )

    spread_scales = (0.35, 0.30, 0.35)
    spread_ranges = (0.16, 0.20, 0.16)
    flexion_maxima = (1.5, 1.6, 1.4)
    for finger in range(1, 4):
        offset = finger * JOINTS_PER_FINGER
        spread = float(
            np.clip(
                side_sign * value[finger, 0] / spread_scales[finger - 1],
                -1.0,
                1.0,
            )
        )
        pose[offset] += side_sign * spread_ranges[finger - 1] * spread
        for joint, maximum in enumerate(flexion_maxima, start=1):
            pose[offset + joint] = 0.2617994 + 1.12 * _unit_flexion(
                value[finger, joint], side_sign, maximum
            )

    pinky_spread = float(np.clip(side_sign * value[4, 0] / 0.4, -1.0, 1.0))
    pinky_flexion = [
        _unit_flexion(value[4, joint], side_sign, maximum)
        for joint, maximum in enumerate(flexion_maxima, start=1)
    ]
    pose[16] += side_sign * 0.16 * pinky_spread
    pose[17] += side_sign * 0.14 * pinky_spread
    pose[18] = 0.2617994 + 1.12 * (0.65 * pinky_flexion[0] + 0.35 * pinky_flexion[1])
    pose[19] = 0.2617994 + 1.12 * (0.35 * pinky_flexion[1] + 0.65 * pinky_flexion[2])
    return np.clip(pose, lower, upper)
