from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ..robot import RobotState, finite_vector

Array = NDArray[np.float64]
H2_DOF = 31
H2_BODY_DOF = 29
H2_HEAD_DOF = 2
H2_HEAD_SLICE = slice(H2_BODY_DOF, H2_DOF)


@dataclass(frozen=True, slots=True)
class H2Command:
    """Low-level command in this package's pinned H2 MJCF actuator order."""

    joint_position: Array
    joint_velocity: Array
    feedforward_torque: Array
    kp: Array
    kd: Array

    def __post_init__(self) -> None:
        for name in (
            "joint_position",
            "joint_velocity",
            "feedforward_torque",
            "kp",
            "kd",
        ):
            object.__setattr__(
                self,
                name,
                finite_vector(name, getattr(self, name), H2_DOF),
            )


__all__ = [
    "H2_BODY_DOF",
    "H2_DOF",
    "H2_HEAD_DOF",
    "H2_HEAD_SLICE",
    "H2Command",
    "RobotState",
]
