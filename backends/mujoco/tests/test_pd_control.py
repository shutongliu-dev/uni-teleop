import unittest

import mujoco
import numpy as np

from sonic_mujoco.envs.mujoco.g1 import MujocoG1EmptyEnv, RobotCommand


class PdControlTest(unittest.TestCase):
    def setUp(self) -> None:
        self.env = MujocoG1EmptyEnv()
        self.env.reset()

    def tearDown(self) -> None:
        self.env.close()

    def test_pd_formula_and_actuator_mapping(self) -> None:
        q = np.linspace(-0.2, 0.2, 29)
        dq = np.linspace(-0.1, 0.1, 29)
        self.env.data.qpos[self.env._qpos_indices] = q
        self.env.data.qvel[self.env._dof_indices] = dq
        mujoco.mj_forward(self.env.model, self.env.data)

        command = RobotCommand(
            joint_position=q + 0.1,
            joint_velocity=dq - 0.2,
            feedforward_torque=np.full(29, 0.3),
            kp=np.full(29, 2.0),
            kd=np.full(29, 0.5),
        )
        expected = np.full(29, 0.4)
        self.env.step(command)

        np.testing.assert_allclose(
            self.env.data.ctrl[self.env._actuator_indices], expected, atol=1e-12
        )
        hand_ids = np.setdiff1d(
            np.arange(self.env.model.nu), self.env._actuator_indices
        )
        np.testing.assert_array_equal(self.env.data.ctrl[hand_ids], 0.0)

    def test_effort_clipping(self) -> None:
        zeros = np.zeros(29)
        command = RobotCommand(zeros, zeros, np.full(29, 1e6), zeros, zeros)
        self.env.step(command)
        np.testing.assert_allclose(
            self.env.data.ctrl[self.env._actuator_indices],
            self.env._control_range[:, 1],
        )

    def test_hand_target_drives_five_finger_actuators(self) -> None:
        zeros = np.zeros(29)
        target = np.linspace(-0.3, 0.3, 40)
        command = RobotCommand(
            zeros,
            zeros,
            zeros,
            zeros,
            zeros,
            hand_joint_position=target,
        )
        clipped_target = np.clip(
            target,
            self.env._hand_joint_range[:, 0],
            self.env._hand_joint_range[:, 1],
        )

        self.env.step(command)

        np.testing.assert_allclose(
            self.env.data.ctrl[self.env._hand_actuator_indices],
            np.clip(
                self.env.hand_kp * clipped_target,
                self.env._hand_control_range[:, 0],
                self.env._hand_control_range[:, 1],
            ),
            atol=1e-12,
        )

    def test_neck_target_drives_two_axis_pd_control(self) -> None:
        zeros = np.zeros(29)
        target = np.array([0.25, -0.2])
        command = RobotCommand(
            zeros,
            zeros,
            zeros,
            zeros,
            zeros,
            neck_joint_position=target,
        )

        self.env.step(command)

        np.testing.assert_allclose(
            self.env.data.ctrl[self.env._neck_actuator_indices],
            self.env.neck_kp * target,
            atol=1e-12,
        )

    def test_command_rejects_wrong_shape(self) -> None:
        zeros = np.zeros(29)
        with self.assertRaises(ValueError):
            RobotCommand(np.zeros(28), zeros, zeros, zeros, zeros)
        with self.assertRaises(ValueError):
            RobotCommand(zeros, zeros, zeros, zeros, zeros, np.zeros(39))
        with self.assertRaises(ValueError):
            RobotCommand(
                zeros,
                zeros,
                zeros,
                zeros,
                zeros,
                neck_joint_position=np.zeros(3),
            )


if __name__ == "__main__":
    unittest.main()
