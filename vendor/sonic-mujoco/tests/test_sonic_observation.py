import unittest

import numpy as np

from sonic_mujoco.controllers.sonic import SonicController, SonicObservationBuilder
from sonic_mujoco.controllers.sonic.parameters import (
    DEFAULT_ANGLES,
    ISAACLAB_FROM_HARDWARE,
)
from sonic_mujoco.envs.mujoco.g1 import RobotState


def robot_state(frame: int) -> RobotState:
    quaternion = np.array(
        [1.0 + 0.01 * frame, 0.02 * frame, -0.015 * frame, 0.01 * (frame + 1)]
    )
    quaternion /= np.linalg.norm(quaternion)
    return RobotState(
        timestamp=0.02 * frame,
        base_position=np.zeros(3),
        base_quaternion=quaternion,
        base_linear_velocity=np.zeros(3),
        base_angular_velocity=np.array(
            [0.1 * frame, -0.2 + 0.03 * frame, 0.4 - 0.02 * frame]
        ),
        joint_position=-0.3 + 0.02 * frame + 0.005 * np.arange(29),
        joint_velocity=0.15 - 0.01 * frame - 0.003 * np.arange(29),
        joint_effort=np.zeros(29),
        imu_quaternion=np.array([1.0, 0.0, 0.0, 0.0]),
        imu_angular_velocity=np.zeros(3),
        imu_linear_acceleration=np.zeros(3),
    )


class SonicObservationTest(unittest.TestCase):
    def test_history_is_zero_padded_and_oldest_first(self) -> None:
        builder = SonicObservationBuilder()
        token = -0.25 + 0.01 * np.arange(64)
        observation = builder.build(robot_state(0), token, np.arange(29) * 0.01)

        self.assertEqual(observation.shape, (994,))
        np.testing.assert_array_equal(observation[:64], token)
        angular_velocity = observation[64:94].reshape(10, 3)
        np.testing.assert_array_equal(angular_velocity[:-1], 0.0)
        np.testing.assert_array_equal(
            angular_velocity[-1], robot_state(0).base_angular_velocity
        )
        gravity = observation[-30:].reshape(10, 3)
        np.testing.assert_array_equal(
            gravity[:-1], np.tile([0.0, 0.0, 1.0], (9, 1))
        )

    def test_joint_offsets_and_ten_frame_history(self) -> None:
        builder = SonicObservationBuilder()
        token = -0.25 + 0.01 * np.arange(64)
        for frame in range(10):
            action = -0.5 + 0.04 * frame + 0.01 * np.arange(29)
            observation = builder.build(robot_state(frame), token, action)

        positions = observation[94:384].reshape(10, 29)
        expected = np.stack(
            [
                robot_state(frame).joint_position[ISAACLAB_FROM_HARDWARE]
                - DEFAULT_ANGLES[ISAACLAB_FROM_HARDWARE]
                for frame in range(10)
            ]
        )
        np.testing.assert_allclose(positions, expected, atol=1e-15)
        actions = observation[674:964].reshape(10, 29)
        np.testing.assert_allclose(
            actions[-1], -0.5 + 0.04 * 9 + 0.01 * np.arange(29), atol=1e-15
        )

    def test_controller_logs_the_previous_action(self) -> None:
        controller = SonicController(lambda _: np.full(29, 0.25))
        token = np.zeros(64)

        controller.act(robot_state(0), token)
        np.testing.assert_array_equal(
            controller.last_observation[674:964].reshape(10, 29)[-1], 0.0
        )
        controller.act(robot_state(1), token)
        np.testing.assert_array_equal(
            controller.last_observation[674:964].reshape(10, 29)[-1], 0.25
        )
        self.assertEqual(controller.steps_per_action, 4)


if __name__ == "__main__":
    unittest.main()
