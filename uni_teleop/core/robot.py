"""Simulator-independent state and PD command contracts (radians, metres, wxyz)."""
from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray
Array = NDArray[np.float64]
G1_DOF, DEXHAND_DOF, NECK_DOF = 29, 40, 2


def _vector(name, value):
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (G1_DOF,) or not np.isfinite(array).all():
        raise ValueError(f'{name} must be finite with shape ({G1_DOF},)')
    return array.copy()


@dataclass(frozen=True, slots=True)
class RobotState:
    timestamp: float
    base_position: Array
    base_quaternion: Array
    base_linear_velocity: Array
    base_angular_velocity: Array
    joint_position: Array
    joint_velocity: Array
    joint_effort: Array
    imu_quaternion: Array
    imu_angular_velocity: Array
    imu_linear_acceleration: Array


@dataclass(frozen=True, slots=True)
class RobotCommand:
    joint_position: Array
    joint_velocity: Array
    feedforward_torque: Array
    kp: Array
    kd: Array
    hand_joint_position: Array | None = None
    neck_joint_position: Array | None = None

    def __post_init__(self) -> None:
        for name in (
            "joint_position",
            "joint_velocity",
            "feedforward_torque",
            "kp",
            "kd",
        ):
            object.__setattr__(self, name, _vector(name, getattr(self, name)))
        if self.hand_joint_position is not None:
            value = np.asarray(self.hand_joint_position, dtype=np.float64)
            if value.shape != (DEXHAND_DOF,):
                raise ValueError(
                    f"hand_joint_position must have shape ({DEXHAND_DOF},)"
                )
            if not np.isfinite(value).all():
                raise ValueError("hand_joint_position must contain finite values")
            object.__setattr__(self, "hand_joint_position", value.copy())
        if self.neck_joint_position is not None:
            value = np.asarray(self.neck_joint_position, dtype=np.float64)
            if value.shape != (NECK_DOF,):
                raise ValueError(f"neck_joint_position must have shape ({NECK_DOF},)")
            if not np.isfinite(value).all():
                raise ValueError("neck_joint_position must contain finite values")
            object.__setattr__(self, "neck_joint_position", value.copy())
