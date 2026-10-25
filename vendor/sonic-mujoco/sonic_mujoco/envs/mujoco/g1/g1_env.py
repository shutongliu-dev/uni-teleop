from pathlib import Path

import mujoco
import numpy as np

from ....neck import NECK_DOF, NECK_JOINT_NAMES
from ....tactile import TactileRecorder
from ....tactile_calibration import TactileCalibrationProfile
from ....tactile_skin import (
    JuQiaoTactileAdapter,
    build_juqiao_skin_layout,
)
from ....teleop.dexhand import DEXHAND_JOINT_NAMES
from ..robot import JointGroup, RobotSpec
from ..robot_env import MujocoRobotEnv
from .interface import RobotCommand

SONIC_JOINT_NAMES = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)


G1_ROBOT_SPEC = RobotSpec(
    robot_id="unitree_g1_29dof_dexhand",
    joint_names=SONIC_JOINT_NAMES,
    root_joint_name="floating_base_joint",
    root_body_name="pelvis",
    imu_quaternion_sensor="imu_quat",
    imu_angular_velocity_sensor="imu_gyro",
    imu_linear_acceleration_sensor="imu_acc",
    camera_names=(
        "head_camera",
        "head_camera_scan",
        "head_camera_left",
        "head_camera_right",
        "left_wrist_camera",
        "right_wrist_camera",
    ),
    capabilities=frozenset(
        {
            "dexhand",
            "auxiliary_neck",
            "ego_rgb",
            "stereo_rgb",
            "wrist_rgb",
            "depth",
            "depth_normals",
            "juqiao_tactile",
        }
    ),
)


class MujocoG1Env(MujocoRobotEnv):
    """MuJoCo environment using the G1 hardware joint order."""

    hand_kp = 2.5
    hand_kd = 0.15
    neck_kp = 8.0
    neck_kd = 0.4

    def __init__(
        self,
        xml_path: str | Path,
        timestep: float = 0.005,
        *,
        model: mujoco.MjModel | None = None,
    ) -> None:
        super().__init__(xml_path, G1_ROBOT_SPEC, timestep, model=model)
        self._hand_joints = JointGroup.from_model(
            self.model, DEXHAND_JOINT_NAMES, robot_id=G1_ROBOT_SPEC.robot_id
        )
        self._neck_joints = JointGroup.from_model(
            self.model, NECK_JOINT_NAMES, robot_id=G1_ROBOT_SPEC.robot_id
        )
        self._publish_auxiliary_joint_aliases()
        self.tactile_skin_layout = build_juqiao_skin_layout(self.model)
        self.tactile = TactileRecorder(
            self.model,
            layout=self.tactile_skin_layout.taxels,
            mapping_radius=self.tactile_skin_layout.mapping_radius_m,
        )
        self.tactile_adapter = JuQiaoTactileAdapter(self.tactile_skin_layout)
        self.tactile_suit = self.tactile_adapter.last_frame

    def configure_tactile_profile(
        self,
        profile: TactileCalibrationProfile,
        *,
        seed: int = 0,
    ) -> None:
        """Replace the packet adapter and align it to the current simulation clock."""

        self.tactile_adapter = JuQiaoTactileAdapter(
            self.tactile_skin_layout,
            profile=profile,
            seed=seed,
            initial_time=self.time,
        )
        self.tactile_suit = self.tactile_adapter.last_frame

    def _begin_control_interval(self) -> None:
        self.tactile.begin()
        self._causal_tactile = self.tactile_adapter.profile.schema_version >= 2
        self._tactile_updated = np.zeros_like(self.tactile_suit.updated)

    def _apply_command(self, command: RobotCommand) -> None:
        super()._apply_command(command)
        if command.hand_joint_position is not None:
            self._hand_joints.apply_position_pd(
                self.data,
                command.hand_joint_position,
                kp=self.hand_kp,
                kd=self.hand_kd,
            )
        neck_target = (
            np.zeros(NECK_DOF, dtype=np.float64)
            if command.neck_joint_position is None
            else command.neck_joint_position
        )
        self._neck_joints.apply_position_pd(
            self.data,
            neck_target,
            kp=self.neck_kp,
            kd=self.neck_kd,
        )

    def _after_physics_step(self) -> None:
        normal_force = self.tactile.update(self.data)
        if self._causal_tactile:
            tactile_sample = self.tactile_adapter.update_normal_force(
                normal_force,
                self.time,
            )
            self._tactile_updated |= tactile_sample.updated

    def _finish_control_interval(self) -> None:
        tactile = self.tactile.finish()
        if self._causal_tactile:
            self.tactile_suit = self.tactile_adapter.snapshot(
                updated=self._tactile_updated
            )
        else:
            self.tactile_suit = self.tactile_adapter.update(tactile, self.time)

    def reset(self, seed: int | None = None) -> None:
        super().reset(seed)
        self.tactile.reset()
        self.tactile_adapter.reset(self.time, seed=seed)
        self.tactile_suit = self.tactile_adapter.last_frame

    def _publish_auxiliary_joint_aliases(self) -> None:
        """Keep the original public arrays while joint groups own their setup."""

        self.hand_joint_ids = self._hand_joints.joint_ids
        self.hand_actuator_ids = self._hand_joints.actuator_ids
        self._hand_actuator_indices = self._hand_joints.actuator_indices
        self._hand_qpos_indices = self._hand_joints.qpos_indices
        self._hand_dof_indices = self._hand_joints.dof_indices
        self._hand_joint_range = self._hand_joints.joint_range
        self._hand_control_range = self._hand_joints.control_range

        self.neck_joint_ids = self._neck_joints.joint_ids
        self.neck_actuator_ids = self._neck_joints.actuator_ids
        self._neck_actuator_indices = self._neck_joints.actuator_indices
        self._neck_qpos_indices = self._neck_joints.qpos_indices
        self._neck_dof_indices = self._neck_joints.dof_indices
        self._neck_joint_range = self._neck_joints.joint_range
        self._neck_control_range = self._neck_joints.control_range
