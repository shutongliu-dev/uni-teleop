import unittest

import numpy as np

from sonic_mujoco.controllers.sonic.encoder import (
    ANCHOR_ORIENTATION_OFFSET,
    ENCODER_OBSERVATION_DIM,
    SMPL_JOINTS_OFFSET,
    WRIST_JOINTS,
    WRIST_JOINTS_OFFSET,
    SonicEncoderObservationBuilder,
)
from sonic_mujoco.envs.mujoco.g1 import RobotState
from sonic_mujoco.teleop import TeleopCommand


def robot_state(quaternion: np.ndarray | None = None) -> RobotState:
    zeros3 = np.zeros(3)
    zeros29 = np.zeros(29)
    return RobotState(
        timestamp=0.0,
        base_position=zeros3,
        base_quaternion=(
            np.array([1.0, 0.0, 0.0, 0.0]) if quaternion is None else quaternion
        ),
        base_linear_velocity=zeros3,
        base_angular_velocity=zeros3,
        joint_position=zeros29,
        joint_velocity=zeros29,
        joint_effort=zeros29,
        imu_quaternion=np.array([1.0, 0.0, 0.0, 0.0]),
        imu_angular_velocity=zeros3,
        imu_linear_acceleration=zeros3,
    )


def teleop_command(start: int, frames: int) -> TeleopCommand:
    frame_index = np.arange(start, start + frames)
    smpl = np.stack([np.full((24, 3), index) for index in frame_index])
    root = np.tile([1.0, 0.0, 0.0, 0.0], (frames, 1))
    joints = np.stack([np.arange(29) + index * 100 for index in frame_index])
    return TeleopCommand(frame_index, smpl, root, joints)


class SonicEncoderObservationTest(unittest.TestCase):
    def test_builds_mode_two_observation_and_clamps_future(self) -> None:
        builder = SonicEncoderObservationBuilder()
        builder.update(teleop_command(0, 5))
        observation = builder.build(robot_state())

        self.assertEqual(observation.shape, (ENCODER_OBSERVATION_DIM,))
        np.testing.assert_array_equal(observation[:4], [2.0, 0.0, 0.0, 0.0])
        smpl = observation[SMPL_JOINTS_OFFSET:ANCHOR_ORIENTATION_OFFSET].reshape(
            10, 24, 3
        )
        np.testing.assert_array_equal(smpl[:, 0, 0], [0, 1, 2, 3, 4, 4, 4, 4, 4, 4])

        identity_6d = np.tile([1.0, 0.0, 0.0, 1.0, 0.0, 0.0], (10, 1))
        anchors = observation[ANCHOR_ORIENTATION_OFFSET:WRIST_JOINTS_OFFSET]
        np.testing.assert_allclose(anchors.reshape(10, 6), identity_6d)

        wrists = observation[WRIST_JOINTS_OFFSET:].reshape(10, 6)
        expected = teleop_command(0, 5).joint_position[[0, 1, 2, 3, 4, 4, 4, 4, 4, 4]][
            :, WRIST_JOINTS
        ]
        np.testing.assert_array_equal(wrists, expected)
        self.assertTrue(np.all(observation[4:SMPL_JOINTS_OFFSET] == 0.0))

    def test_sliding_window_advances_with_ten_future_frames(self) -> None:
        builder = SonicEncoderObservationBuilder()
        builder.update(teleop_command(0, 5))
        builder.update(teleop_command(1, 11))

        first = builder.build(robot_state())
        builder.advance()
        second = builder.build(robot_state())
        first_value = first[SMPL_JOINTS_OFFSET]
        second_value = second[SMPL_JOINTS_OFFSET]
        self.assertEqual((first_value, second_value), (0.0, 1.0))

    def test_heading_increment_rotates_the_reference(self) -> None:
        builder = SonicEncoderObservationBuilder()
        builder.update(teleop_command(0, 5))
        command = teleop_command(1, 5)
        command = TeleopCommand(
            command.frame_index,
            command.smpl_joints,
            command.root_quaternion,
            command.joint_position,
            heading_increment=np.pi / 2.0,
        )
        builder.update(command)
        observation = builder.build(robot_state())

        first_anchor = observation[
            ANCHOR_ORIENTATION_OFFSET : ANCHOR_ORIENTATION_OFFSET + 6
        ]
        np.testing.assert_allclose(
            first_anchor, [0.0, -1.0, 1.0, 0.0, 0.0, 0.0], atol=1e-7
        )


if __name__ == "__main__":
    unittest.main()
