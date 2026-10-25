"""OpenXR retargeting for the simulated five-finger dexterous hands."""

from __future__ import annotations

import numpy as np

from .. import hand as _hand

DEXHAND_DOF = _hand.DEXHAND_DOF
DEXHAND_JOINT_NAMES = _hand.DEXHAND_JOINT_NAMES
dexhand_controller_targets = _hand.dexhand_controller_targets

# OpenXR wrist-first hand layout after dropping joint 0 (palm) from its native
# 26-joint array.
WRIST = 0
THUMB = (1, 2, 3, 4)
INDEX = (5, 6, 7, 8, 9)
MIDDLE = (10, 11, 12, 13, 14)
RING = (15, 16, 17, 18, 19)
PINKY = (20, 21, 22, 23, 24)
FINGER_INDICES = (INDEX, MIDDLE, RING, PINKY)


def _positions(hand: np.ndarray) -> np.ndarray:
    value = np.asarray(hand, dtype=np.float64)
    if (
        value.ndim != 2
        or value.shape[0] not in (25, 26)
        or value.shape[1]
        not in (
            3,
            7,
        )
    ):
        raise ValueError("hand tracking must have shape (25|26, 3|7)")
    value = value[:, :3]
    if value.shape[0] == 26:
        value = value[1:]
    if not np.isfinite(value).all():
        raise ValueError("hand tracking must contain finite values")
    if np.linalg.norm(value[INDEX[0]] - value[PINKY[0]]) < 1e-4:
        raise ValueError("hand tracking has a degenerate palm")
    return value


def _unit(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    if norm < 1e-10:
        raise ValueError("hand tracking contains a zero-length bone")
    return vector / norm


def _flexion(first: np.ndarray, center: np.ndarray, last: np.ndarray) -> float:
    before = first - center
    after = last - center
    norm = np.linalg.norm(before) * np.linalg.norm(after)
    if norm < 1e-10:
        return 0.0
    angle = np.arccos(np.clip(np.dot(before, after) / norm, -1.0, 1.0))
    return float(np.pi - angle)


def _finger_target(
    hand: np.ndarray,
    indices: tuple[int, int, int, int, int],
    longitudinal: np.ndarray,
    lateral: np.ndarray,
    flexion_sign: float,
    spread_sign: float,
) -> np.ndarray:
    metacarpal, proximal, intermediate, distal, tip = indices
    direction = _unit(hand[proximal] - hand[metacarpal])
    spread = np.arctan2(np.dot(direction, lateral), np.dot(direction, longitudinal))
    return np.array(
        [
            spread_sign * np.clip(spread, -0.35, 0.35),
            flexion_sign
            * min(_flexion(hand[metacarpal], hand[proximal], hand[intermediate]), 1.5),
            flexion_sign
            * min(_flexion(hand[proximal], hand[intermediate], hand[distal]), 1.6),
            flexion_sign
            * min(_flexion(hand[intermediate], hand[distal], hand[tip]), 1.4),
        ]
    )


class DexHandRetargeter:
    """Retarget every tracked human digit to a 20-DoF simulated hand."""

    def __init__(self, low_pass_alpha: float = 0.2) -> None:
        if not 0.0 < low_pass_alpha <= 1.0:
            raise ValueError("low_pass_alpha must be in (0, 1]")
        self.low_pass_alpha = float(low_pass_alpha)
        self._previous: dict[str, np.ndarray] = {}

    def reset(self) -> None:
        self._previous.clear()

    def retarget_hand(self, hand: np.ndarray, side: str) -> np.ndarray:
        if side not in {"left", "right"}:
            raise ValueError("side must be 'left' or 'right'")
        positions = _positions(hand)
        longitudinal = _unit(positions[MIDDLE[0]] - positions[WRIST])
        lateral = _unit(positions[INDEX[0]] - positions[PINKY[0]])
        flexion_sign = -1.0 if side == "left" else 1.0
        spread_sign = -1.0 if side == "left" else 1.0

        palm_width = np.linalg.norm(positions[INDEX[0]] - positions[PINKY[0]])
        pinch_ratio = (
            np.linalg.norm(positions[THUMB[-1]] - positions[INDEX[-1]]) / palm_width
        )
        opposition = np.clip((1.15 - pinch_ratio) / 0.95, 0.0, 1.0)
        thumb_direction = _unit(positions[THUMB[1]] - positions[THUMB[0]])
        thumb_spread = np.arctan2(
            np.dot(thumb_direction, lateral),
            np.dot(thumb_direction, longitudinal),
        )
        thumb = np.array(
            [
                -0.65 * opposition,
                spread_sign * np.clip(thumb_spread, -0.8, 0.8),
                flexion_sign
                * min(
                    _flexion(
                        positions[THUMB[0]], positions[THUMB[1]], positions[THUMB[2]]
                    ),
                    1.5,
                ),
                flexion_sign
                * min(
                    _flexion(
                        positions[THUMB[1]], positions[THUMB[2]], positions[THUMB[3]]
                    ),
                    1.4,
                ),
            ]
        )
        target = np.concatenate(
            (
                thumb,
                *(
                    _finger_target(
                        positions,
                        indices,
                        longitudinal,
                        lateral,
                        flexion_sign,
                        spread_sign,
                    )
                    for indices in FINGER_INDICES
                ),
            )
        )
        previous = self._previous.get(side)
        if previous is not None:
            alpha = self.low_pass_alpha
            target = (1.0 - alpha) * previous + alpha * target
        self._previous[side] = target
        return target.copy()

    def retarget(self, left: np.ndarray, right: np.ndarray) -> np.ndarray:
        return np.concatenate(
            (self.retarget_hand(left, "left"), self.retarget_hand(right, "right"))
        )
