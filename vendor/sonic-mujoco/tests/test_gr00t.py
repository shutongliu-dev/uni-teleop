import unittest

import numpy as np

from sonic_mujoco.controllers.sonic.parameters import DEFAULT_ANGLES
from sonic_mujoco.envs.mujoco.g1 import MujocoG1SweepEnv
from sonic_mujoco.gr00t import PROMPT, build_observation


class Gr00tObservationTest(unittest.TestCase):
    def test_sweep_observation_matches_policy_modalities(self) -> None:
        env = MujocoG1SweepEnv()
        try:
            env.reset(seed=0)
            image = np.zeros((48, 64, 3), dtype=np.uint8)
            observation = build_observation(
                env.get_robot_state(),
                {"ego_view_left": image, "ego_view_right": image},
            )
        finally:
            env.close()

        self.assertEqual(observation["video"]["ego_view_left"].shape, (1, 1, 48, 64, 3))
        self.assertEqual(observation["state"]["left_leg"].shape, (1, 1, 6))
        self.assertEqual(observation["state"]["right_arm"].shape, (1, 1, 7))
        self.assertEqual(observation["state"]["projected_gravity"].shape, (1, 1, 3))
        np.testing.assert_allclose(
            observation["state"]["left_leg"][0, 0], -DEFAULT_ANGLES[:6]
        )
        self.assertEqual(observation["tactile"]["vest"].shape, (1, 1, 256))
        self.assertEqual(
            observation["language"]["annotation.human.task_description"], [[PROMPT]]
        )

    def test_observation_uses_real_compatible_tactile_packets(self) -> None:
        env = MujocoG1SweepEnv()
        try:
            env.reset(seed=0)
            image = np.zeros((48, 64, 3), dtype=np.uint8)
            values = env.tactile_suit.values.copy()
            values[0, 194] = 17
            values[1, 128] = 23
            values[2, 3] = 31
            tactile = type(env.tactile_suit)(
                values=values,
                source_time=env.tactile_suit.source_time,
                updated=env.tactile_suit.updated,
            )

            observation = build_observation(
                env.get_robot_state(),
                {"ego_view_left": image, "ego_view_right": image},
                tactile=tactile,
            )
        finally:
            env.close()

        self.assertEqual(observation["tactile"]["vest"].dtype, np.uint8)
        self.assertEqual(observation["tactile"]["vest"][0, 0, 194], 17)
        self.assertEqual(observation["tactile"]["left_arm"][0, 0, 128], 23)
        self.assertEqual(observation["tactile"]["right_arm"][0, 0, 3], 31)


if __name__ == "__main__":
    unittest.main()
