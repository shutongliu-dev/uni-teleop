from collections import deque
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ...envs.mujoco.g1 import RobotState
from .parameters import (
    DEFAULT_ANGLES,
    HISTORY_FRAMES,
    ISAACLAB_FROM_HARDWARE,
    OBSERVATION_DIM,
    TOKEN_DIM,
)

Array = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class _Frame:
    angular_velocity: Array
    joint_position: Array
    joint_velocity: Array
    last_action: Array
    gravity: Array


class SonicObservationBuilder:
    """Build the decoder observation with the reference C++ history semantics."""

    def __init__(self) -> None:
        self._history: deque[_Frame] = deque(maxlen=HISTORY_FRAMES)

    def reset(self) -> None:
        self._history.clear()

    def build(self, state: RobotState, token_state: Array, last_action: Array) -> Array:
        token = _vector("token_state", token_state, TOKEN_DIM)
        action = _vector("last_action", last_action, 29)
        quaternion = _vector("base_quaternion", state.base_quaternion, 4)
        frame = _Frame(
            angular_velocity=_vector(
                "base_angular_velocity", state.base_angular_velocity, 3
            ),
            joint_position=(
                _vector("joint_position", state.joint_position, 29)[
                    ISAACLAB_FROM_HARDWARE
                ]
                - DEFAULT_ANGLES[ISAACLAB_FROM_HARDWARE]
            ),
            joint_velocity=_vector("joint_velocity", state.joint_velocity, 29)[
                ISAACLAB_FROM_HARDWARE
            ],
            last_action=action,
            gravity=_projected_gravity(quaternion),
        )
        self._history.append(frame)

        missing = HISTORY_FRAMES - len(self._history)
        frames = [_zero_frame()] * missing + list(self._history)
        observation = np.concatenate(
            [token]
            + [np.concatenate([getattr(frame, field) for frame in frames])
               for field in (
                   "angular_velocity",
                   "joint_position",
                   "joint_velocity",
                   "last_action",
                   "gravity",
               )]
        )
        assert observation.shape == (OBSERVATION_DIM,)
        return observation


def _vector(name: str, value: Array, size: int) -> Array:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (size,) or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite vector of shape ({size},)")
    return array.copy()


def _projected_gravity(quaternion: Array) -> Array:
    w = quaternion[0]
    vector = -quaternion[1:]
    gravity = np.array([0.0, 0.0, -1.0])
    return (
        gravity * (2.0 * w * w - 1.0)
        + np.cross(vector, gravity) * w * 2.0
        + vector * np.dot(vector, gravity) * 2.0
    )


def _zero_frame() -> _Frame:
    return _Frame(
        angular_velocity=np.zeros(3),
        joint_position=np.zeros(29),
        joint_velocity=np.zeros(29),
        last_action=np.zeros(29),
        # The reference logger pads with quaternion [0, 0, 0, 0]. Its exact
        # quaternion rotation implementation maps gravity to +Z for that value.
        gravity=np.array([0.0, 0.0, 1.0]),
    )
