import unittest

import mujoco
import numpy as np

from sonic_mujoco.envs.mujoco.g1 import MujocoG1ChairLeanEnv


class ChairLeanEnvTest(unittest.TestCase):
    def setUp(self) -> None:
        self.env = MujocoG1ChairLeanEnv()

    def tearDown(self) -> None:
        self.env.close()

    def test_reference_dimensions_and_dynamic_chair(self) -> None:
        chair_id = mujoco.mj_name2id(
            self.env.model, mujoco.mjtObj.mjOBJ_BODY, "chair"
        )
        joint_id = mujoco.mj_name2id(
            self.env.model, mujoco.mjtObj.mjOBJ_JOINT, "chair_joint"
        )
        self.assertGreater(chair_id, 0)
        self.assertEqual(
            self.env.model.jnt_type[joint_id], mujoco.mjtJoint.mjJNT_FREE
        )
        self.assertAlmostEqual(self.env.model.body_mass[chair_id], 11.0)
        visual_id = mujoco.mj_name2id(
            self.env.model, mujoco.mjtObj.mjOBJ_GEOM, "chair_visual"
        )
        back_id = mujoco.mj_name2id(
            self.env.model, mujoco.mjtObj.mjOBJ_GEOM, "chair_back"
        )
        self.assertEqual(
            self.env.model.geom_type[visual_id], mujoco.mjtGeom.mjGEOM_MESH
        )
        self.assertEqual(self.env.model.geom_contype[visual_id], 0)
        self.assertNotEqual(self.env.model.geom_contype[back_id], 0)

    def test_reset_is_seeded_and_randomizes_task_variations(self) -> None:
        self.env.reset(seed=17)
        first = self.env.get_scene_state()
        robot_position = self.env.get_robot_state().base_position.copy()

        self.env.reset(seed=17)
        repeated = self.env.get_scene_state()
        np.testing.assert_allclose(first.chair_position, repeated.chair_position)
        np.testing.assert_allclose(first.chair_quaternion, repeated.chair_quaternion)
        self.assertEqual(first.backrest_width, repeated.backrest_width)
        np.testing.assert_allclose(
            robot_position, self.env.get_robot_state().base_position
        )

        self.env.reset(seed=18)
        changed = self.env.get_scene_state()
        self.assertFalse(np.allclose(first.chair_position, changed.chair_position))
        self.assertNotEqual(first.backrest_softness, changed.backrest_softness)

    def test_randomization_stays_near_reference_chair(self) -> None:
        for seed in range(20):
            self.env.reset(seed=seed)
            state = self.env.get_scene_state()
            self.assertTrue(0.48 <= state.backrest_width <= 0.54)
            self.assertTrue(0.29 <= state.backrest_height <= 0.34)
            self.assertTrue(np.deg2rad(7) <= state.backrest_tilt <= np.deg2rad(14))
            self.assertTrue(0.025 <= state.backrest_softness <= 0.050)

    def test_chair_moves_under_external_force(self) -> None:
        self.env.reset(seed=4)
        chair_id = mujoco.mj_name2id(
            self.env.model, mujoco.mjtObj.mjOBJ_BODY, "chair"
        )
        initial_x = self.env.get_scene_state().chair_position[0]
        self.env.data.xfrc_applied[chair_id, 0] = 40.0
        for _ in range(10):
            mujoco.mj_step(self.env.model, self.env.data)
        self.assertGreater(self.env.get_scene_state().chair_position[0], initial_x)


if __name__ == "__main__":
    unittest.main()
