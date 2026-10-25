from pathlib import Path

import mujoco

from ...contact import ContactRecorder
from .env_base import MujocoEnvBase
from .robot import (
    JointCommand,
    JointGroup,
    RobotSpec,
    RobotState,
    required_sensor_slice,
)


class MujocoRobotEnv(MujocoEnvBase):
    """Shared named-joint state and torque-control loop for humanoid models."""

    def __init__(
        self,
        xml_path: str | Path,
        spec: RobotSpec,
        timestep: float = 0.005,
        *,
        model: mujoco.MjModel | None = None,
    ) -> None:
        super().__init__(xml_path, timestep, model=model)
        self.robot_spec = spec
        self._body_joints = JointGroup.from_model(
            self.model,
            spec.joint_names,
            robot_id=spec.robot_id,
        )
        self._publish_main_joint_aliases()
        self._root_joint_id = self._validate_root_joint(spec.root_joint_name)
        self._root_qpos_address = int(self.model.jnt_qposadr[self._root_joint_id])
        self._root_dof_address = int(self.model.jnt_dofadr[self._root_joint_id])
        self._imu_quaternion = required_sensor_slice(
            self.model, spec.imu_quaternion_sensor, spec.robot_id
        )
        self._imu_angular_velocity = required_sensor_slice(
            self.model, spec.imu_angular_velocity_sensor, spec.robot_id
        )
        self._imu_linear_acceleration = required_sensor_slice(
            self.model, spec.imu_linear_acceleration_sensor, spec.robot_id
        )
        self._reset_keyframe_id = self._keyframe_id(spec.reset_keyframe)
        self._validate_cameras()
        self.contacts = ContactRecorder(self.model, robot_root=spec.root_body_name)

    def get_robot_state(self) -> RobotState:
        root_qpos = self._root_qpos_address
        root_dof = self._root_dof_address
        return RobotState(
            timestamp=self.time,
            base_position=self.data.qpos[root_qpos : root_qpos + 3].copy(),
            base_quaternion=self.data.qpos[root_qpos + 3 : root_qpos + 7].copy(),
            base_linear_velocity=self.data.qvel[root_dof : root_dof + 3].copy(),
            base_angular_velocity=self.data.qvel[root_dof + 3 : root_dof + 6].copy(),
            joint_position=self.data.qpos[self._qpos_indices].copy(),
            joint_velocity=self.data.qvel[self._dof_indices].copy(),
            joint_effort=self.data.actuator_force[self._actuator_indices].copy(),
            imu_quaternion=self.data.sensordata[self._imu_quaternion].copy(),
            imu_angular_velocity=self.data.sensordata[
                self._imu_angular_velocity
            ].copy(),
            imu_linear_acceleration=self.data.sensordata[
                self._imu_linear_acceleration
            ].copy(),
        )

    def step(self, command: JointCommand, steps: int = 1) -> None:
        if steps < 1:
            raise ValueError("steps must be positive")

        self.contacts.begin()
        self._begin_control_interval()
        for _ in range(steps):
            self._apply_command(command)
            mujoco.mj_step(self.model, self.data)
            self.contacts.update(self.data)
            self._after_physics_step()
        self.contacts.finish()
        self._finish_control_interval()

    def reset(self, seed: int | None = None) -> None:
        del seed
        if self._reset_keyframe_id is None:
            mujoco.mj_resetData(self.model, self.data)
        else:
            mujoco.mj_resetDataKeyframe(self.model, self.data, self._reset_keyframe_id)
        mujoco.mj_forward(self.model, self.data)
        self.contacts.reset()

    def _apply_command(self, command: JointCommand) -> None:
        self.data.ctrl[:] = 0.0
        self.data.ctrl[self._actuator_indices] = self._body_joints.clipped_torque(
            self.data, command
        )

    def _begin_control_interval(self) -> None:
        pass

    def _after_physics_step(self) -> None:
        pass

    def _finish_control_interval(self) -> None:
        pass

    def _publish_main_joint_aliases(self) -> None:
        self.joint_ids = self._body_joints.joint_ids
        self.actuator_ids = self._body_joints.actuator_ids
        self._actuator_indices = self._body_joints.actuator_indices
        self._qpos_indices = self._body_joints.qpos_indices
        self._dof_indices = self._body_joints.dof_indices
        self._control_range = self._body_joints.control_range

    def _validate_root_joint(self, name: str) -> int:
        joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint_id < 0 or self.model.jnt_type[joint_id] != mujoco.mjtJoint.mjJNT_FREE:
            raise ValueError(
                f"{self.robot_spec.robot_id} model must have a free joint named {name}"
            )
        return joint_id

    def _keyframe_id(self, name: str | None) -> int | None:
        if name is None:
            return None
        keyframe_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, name)
        if keyframe_id < 0:
            raise ValueError(
                f"missing {self.robot_spec.robot_id} reset keyframe: {name}"
            )
        return keyframe_id

    def _validate_cameras(self) -> None:
        for name in self.robot_spec.camera_names:
            camera_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, name)
            if camera_id < 0:
                raise ValueError(f"missing {self.robot_spec.robot_id} camera: {name}")
