import unittest

import numpy as np

from sonic_mujoco.envs.mujoco.g1 import MujocoG1EmptyEnv


class RobotStateTest(unittest.TestCase):
    def test_state_uses_hardware_order_and_copies_data(self) -> None:
        env = MujocoG1EmptyEnv()
        self.addCleanup(env.close)
        env.reset()

        state = env.get_robot_state()
        self.assertEqual(state.base_position.shape, (3,))
        self.assertEqual(state.base_quaternion.shape, (4,))
        self.assertEqual(state.base_linear_velocity.shape, (3,))
        self.assertEqual(state.base_angular_velocity.shape, (3,))
        self.assertEqual(state.joint_position.shape, (29,))
        self.assertEqual(state.joint_velocity.shape, (29,))
        self.assertEqual(state.joint_effort.shape, (29,))
        self.assertEqual(state.imu_quaternion.shape, (4,))
        self.assertEqual(state.imu_angular_velocity.shape, (3,))
        self.assertEqual(state.imu_linear_acceleration.shape, (3,))
        np.testing.assert_allclose(state.joint_position, env.data.qpos[env._qpos_indices])
        self.assertFalse(np.shares_memory(state.joint_position, env.data.qpos))


if __name__ == "__main__":
    unittest.main()
