from pathlib import Path

import mujoco
import numpy as np

from ..robot import RobotSpec
from ..robot_env import MujocoRobotEnv
from .interface import H2Command

H2_JOINT_NAMES = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_roll_joint",
    "left_ankle_pitch_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_roll_joint",
    "right_ankle_pitch_joint",
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
    "head_pitch_joint",
    "head_yaw_joint",
)

H2_ACTUATOR_NAMES = tuple(name.removesuffix("_joint") for name in H2_JOINT_NAMES)
H2_ACTUATOR_ORDER_ID = "unitree_mujoco_h2_4134cb5_actuator_order"

H2_HOME_JOINT_POSITION = (
    -0.25,
    0.0,
    0.0,
    0.5,
    0.0,
    -0.25,
    -0.25,
    0.0,
    0.0,
    0.5,
    0.0,
    -0.25,
    0.0,
    0.0,
    0.0,
    0.35,
    0.18,
    0.0,
    0.87,
    0.0,
    0.0,
    0.0,
    0.35,
    -0.18,
    0.0,
    0.87,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
)

# Gains from Unitree's public H2 low-level example. They are a tested nominal
# hold default for this simulation model, not a safety or policy claim.
H2_HOME_KP = (
    150.0,
    150.0,
    150.0,
    250.0,
    60.0,
    90.0,
    150.0,
    150.0,
    150.0,
    250.0,
    60.0,
    90.0,
    200.0,
    200.0,
    200.0,
    90.0,
    60.0,
    20.0,
    60.0,
    4.0,
    4.0,
    4.0,
    90.0,
    60.0,
    20.0,
    60.0,
    4.0,
    4.0,
    4.0,
    30.0,
    30.0,
)

H2_HOME_KD = (
    2.0,
    2.0,
    2.0,
    2.0,
    0.3,
    0.1,
    2.0,
    2.0,
    2.0,
    2.0,
    0.3,
    0.1,
    2.5,
    5.0,
    5.0,
    2.0,
    1.0,
    0.4,
    1.0,
    0.2,
    0.2,
    0.2,
    2.0,
    1.0,
    0.4,
    1.0,
    0.2,
    0.2,
    0.2,
    1.0,
    1.0,
)

H2_ROBOT_SPEC = RobotSpec(
    robot_id="unitree_h2_31dof",
    joint_names=H2_JOINT_NAMES,
    root_joint_name="floating_base_joint",
    root_body_name="pelvis",
    imu_quaternion_sensor="imu_quat",
    imu_angular_velocity_sensor="imu_gyro",
    imu_linear_acceleration_sensor="imu_acc",
    camera_names=(
        "head_camera",
        "head_camera_left",
        "head_camera_right",
        "left_wrist_camera",
        "right_wrist_camera",
    ),
    capabilities=frozenset({"native_head_2dof", "ego_rgb", "stereo_rgb", "wrist_rgb"}),
    reset_keyframe="home",
)


class MujocoH2Env(MujocoRobotEnv):
    """Unitree H2 with the pinned official MJCF's 31-actuator ordering."""

    def __init__(
        self,
        xml_path: str | Path,
        timestep: float = 0.002,
        *,
        model: mujoco.MjModel | None = None,
    ) -> None:
        super().__init__(xml_path, H2_ROBOT_SPEC, timestep, model=model)
        self._validate_home_pose()

    def home_command(self) -> H2Command:
        zeros = np.zeros(H2_ROBOT_SPEC.dof, dtype=np.float64)
        return H2Command(
            joint_position=np.asarray(H2_HOME_JOINT_POSITION),
            joint_velocity=zeros,
            feedforward_torque=zeros,
            kp=np.asarray(H2_HOME_KP),
            kd=np.asarray(H2_HOME_KD),
        )

    def _validate_home_pose(self) -> None:
        assert self._reset_keyframe_id is not None
        home = self.model.key_qpos[self._reset_keyframe_id, self._qpos_indices]
        expected = np.asarray(H2_HOME_JOINT_POSITION)
        if not np.allclose(home, expected, atol=1e-12, rtol=0.0):
            raise ValueError(
                "H2 home keyframe does not match the pinned MJCF actuator order"
            )
        limited = self._body_joints.joint_limited
        within_range = (home[limited] >= self._body_joints.joint_range[limited, 0]) & (
            home[limited] <= self._body_joints.joint_range[limited, 1]
        )
        if not np.all(within_range):
            raise ValueError("H2 home keyframe exceeds a joint limit")
