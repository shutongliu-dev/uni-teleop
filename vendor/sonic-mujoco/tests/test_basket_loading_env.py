import unittest

import mujoco
import numpy as np

from sonic_mujoco.envs.mujoco.g1 import MujocoG1BasketLoadingEnv


class BasketLoadingEnvTest(unittest.TestCase):
    def setUp(self) -> None:
        self.env = MujocoG1BasketLoadingEnv()

    def tearDown(self) -> None:
        self.env.close()

    def test_basket_is_an_open_dynamic_container(self) -> None:
        joint_id = mujoco.mj_name2id(
            self.env.model, mujoco.mjtObj.mjOBJ_JOINT, "loading_basket_joint"
        )
        self.assertEqual(self.env.model.jnt_type[joint_id], mujoco.mjtJoint.mjJNT_FREE)
        for name in (
            "basket_bottom",
            "basket_back",
            "basket_front",
            "basket_left",
            "basket_right",
        ):
            geom_id = mujoco.mj_name2id(self.env.model, mujoco.mjtObj.mjOBJ_GEOM, name)
            self.assertNotEqual(self.env.model.geom_contype[geom_id], 0)

    def test_reset_is_seeded_and_randomizes_load(self) -> None:
        self.env.reset(seed=18)
        first = self.env.get_scene_state()
        first_inertia = self.env.model.body_inertia.copy()
        self.env.reset(seed=18)
        repeated = self.env.get_scene_state()

        np.testing.assert_allclose(first.basket_position, repeated.basket_position)
        np.testing.assert_allclose(first.object_masses, repeated.object_masses)
        np.testing.assert_allclose(first_inertia, self.env.model.body_inertia)
        self.assertTrue(1.0 <= first.basket_mass <= 1.8)
        self.assertTrue(np.all(first.object_masses >= 0.25))
        self.assertTrue(np.all(first.object_masses <= 2.8))

    def test_objects_load_one_at_a_time(self) -> None:
        self.env.reset(seed=4)
        self.env.start_loading()
        self.env.advance_loading(4.9)
        self.assertEqual(self.env.get_scene_state().loaded_count, 0)

        self.env.advance_loading(5.0)
        first = self.env.get_scene_state()
        self.assertEqual(first.loaded_count, 1)

        self.env.advance_loading(16.0)
        self.assertEqual(self.env.get_scene_state().loaded_count, 3)

    def test_dropped_object_falls_into_basket(self) -> None:
        self.env.reset(seed=7)
        self.env.start_loading()
        basket = self.env.get_scene_state().basket_position
        self.env.advance_loading(5.0)
        for _ in range(100):
            mujoco.mj_step(self.env.model, self.env.data)

        object_positions = self.env.data.xpos[
            [
                mujoco.mj_name2id(
                    self.env.model, mujoco.mjtObj.mjOBJ_BODY, f"load_{name}"
                )
                for name in self.env._OBJECTS
            ]
        ]
        distances = np.linalg.norm(object_positions[:, :2] - basket[:2], axis=1)
        self.assertLess(np.min(distances), 0.20)


if __name__ == "__main__":
    unittest.main()
