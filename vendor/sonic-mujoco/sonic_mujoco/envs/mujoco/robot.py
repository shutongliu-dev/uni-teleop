from dataclasses import dataclass
from typing import Protocol

import mujoco
import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class RobotSpec:
    """Scene-independent facts needed to address one robot model."""

    robot_id: str
    joint_names: tuple[str, ...]
    root_joint_name: str
    root_body_name: str
    imu_quaternion_sensor: str
    imu_angular_velocity_sensor: str
    imu_linear_acceleration_sensor: str
    camera_names: tuple[str, ...] = ()
    capabilities: frozenset[str] = frozenset()
    reset_keyframe: str | None = None

    def __post_init__(self) -> None:
        if not self.robot_id:
            raise ValueError("robot_id must not be empty")
        if not self.joint_names or len(set(self.joint_names)) != len(self.joint_names):
            raise ValueError("joint_names must be non-empty and unique")
        if len(set(self.camera_names)) != len(self.camera_names):
            raise ValueError("camera_names must be unique")

    @property
    def dof(self) -> int:
        return len(self.joint_names)


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


class JointCommand(Protocol):
    joint_position: Array
    joint_velocity: Array
    feedforward_torque: Array
    kp: Array
    kd: Array


@dataclass(frozen=True, slots=True)
class JointGroup:
    """Name-resolved one-DoF joints and their direct-drive actuators."""

    names: tuple[str, ...]
    joint_ids: tuple[int, ...]
    actuator_ids: tuple[int, ...]
    actuator_indices: NDArray[np.int32]
    qpos_indices: NDArray[np.int32]
    dof_indices: NDArray[np.int32]
    joint_range: Array
    joint_limited: NDArray[np.bool_]
    control_range: Array
    control_limited: NDArray[np.bool_]

    @classmethod
    def from_model(
        cls,
        model: mujoco.MjModel,
        names: tuple[str, ...],
        *,
        robot_id: str,
    ) -> "JointGroup":
        joint_ids = tuple(
            _required_id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in names
        )
        actuator_ids = tuple(
            _joint_actuator_id(model, joint_id, name, robot_id=robot_id)
            for joint_id, name in zip(joint_ids, names, strict=True)
        )
        if len(set(actuator_ids)) != len(actuator_ids):
            raise ValueError(f"{robot_id} joints must use distinct actuators")

        joint_indices = np.asarray(joint_ids, dtype=np.int32)
        actuator_indices = np.asarray(actuator_ids, dtype=np.int32)
        return cls(
            names=names,
            joint_ids=joint_ids,
            actuator_ids=actuator_ids,
            actuator_indices=actuator_indices,
            qpos_indices=model.jnt_qposadr[joint_indices].astype(np.int32),
            dof_indices=model.jnt_dofadr[joint_indices].astype(np.int32),
            joint_range=model.jnt_range[joint_indices].copy(),
            joint_limited=model.jnt_limited[joint_indices].astype(bool),
            control_range=model.actuator_ctrlrange[actuator_indices].copy(),
            control_limited=model.actuator_ctrllimited[actuator_indices].astype(bool),
        )

    @property
    def size(self) -> int:
        return len(self.names)

    def clipped_torque(self, data: mujoco.MjData, command: JointCommand) -> Array:
        torque = (
            command.feedforward_torque
            + command.kp * (command.joint_position - data.qpos[self.qpos_indices])
            + command.kd * (command.joint_velocity - data.qvel[self.dof_indices])
        )
        result = np.asarray(torque, dtype=np.float64).copy()
        if result.shape != (self.size,):
            raise ValueError(f"joint command must have shape ({self.size},)")
        limited = self.control_limited
        result[limited] = np.clip(
            result[limited],
            self.control_range[limited, 0],
            self.control_range[limited, 1],
        )
        return result

    def apply_position_pd(
        self,
        data: mujoco.MjData,
        target: Array,
        *,
        kp: float,
        kd: float,
    ) -> None:
        target = np.asarray(target, dtype=np.float64)
        if target.shape != (self.size,) or not np.isfinite(target).all():
            raise ValueError(f"joint target must be finite with shape ({self.size},)")
        clipped_target = target.copy()
        limited_joints = self.joint_limited
        clipped_target[limited_joints] = np.clip(
            clipped_target[limited_joints],
            self.joint_range[limited_joints, 0],
            self.joint_range[limited_joints, 1],
        )
        torque = (
            kp * (clipped_target - data.qpos[self.qpos_indices])
            - kd * data.qvel[self.dof_indices]
        )
        limited_actuators = self.control_limited
        torque[limited_actuators] = np.clip(
            torque[limited_actuators],
            self.control_range[limited_actuators, 0],
            self.control_range[limited_actuators, 1],
        )
        data.ctrl[self.actuator_indices] = torque


def finite_vector(name: str, value: Array, size: int) -> Array:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (size,):
        raise ValueError(f"{name} must have shape ({size},)")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain finite values")
    return array.copy()


def required_sensor_slice(model: mujoco.MjModel, name: str, robot_id: str) -> slice:
    sensor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, name)
    if sensor_id < 0:
        raise ValueError(f"missing {robot_id} sensor: {name}")
    start = int(model.sensor_adr[sensor_id])
    return slice(start, start + int(model.sensor_dim[sensor_id]))


def _required_id(model: mujoco.MjModel, object_type: int, name: str) -> int:
    object_id = mujoco.mj_name2id(model, object_type, name)
    if object_id < 0:
        raise ValueError(f"missing model object: {name}")
    return object_id


def _joint_actuator_id(
    model: mujoco.MjModel,
    joint_id: int,
    joint_name: str,
    *,
    robot_id: str,
) -> int:
    joint_transmissions = {
        int(mujoco.mjtTrn.mjTRN_JOINT),
        int(mujoco.mjtTrn.mjTRN_JOINTINPARENT),
    }
    candidates = [
        actuator_id
        for actuator_id in range(model.nu)
        if int(model.actuator_trnid[actuator_id, 0]) == joint_id
        and int(model.actuator_trntype[actuator_id]) in joint_transmissions
    ]
    if len(candidates) != 1:
        raise ValueError(
            f"{robot_id} joint {joint_name} must have exactly one direct actuator"
        )
    return candidates[0]
