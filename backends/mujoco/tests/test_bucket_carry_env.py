import unittest

import mujoco
import numpy as np

from sonic_mujoco.envs.mujoco.g1 import MujocoG1BucketCarryEnv


class BucketCarryEnvTest(unittest.TestCase):
    def setUp(self) -> None:
        self.env = MujocoG1BucketCarryEnv()

    def tearDown(self) -> None:
        self.env.close()

    def test_two_identical_dynamic_tables(self) -> None:
        body_ids = [
            mujoco.mj_name2id(self.env.model, mujoco.mjtObj.mjOBJ_BODY, name)
            for name in ("source_table", "target_table")
        ]
        joint_ids = [
            mujoco.mj_name2id(self.env.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            for name in ("source_table_joint", "target_table_joint")
        ]
        np.testing.assert_allclose(
            self.env.model.body_mass[body_ids[0]],
            self.env.model.body_mass[body_ids[1]],
        )
        for joint_id in joint_ids:
            self.assertEqual(
                self.env.model.jnt_type[joint_id], mujoco.mjtJoint.mjJNT_FREE
            )

    def test_scanned_visuals_keep_primitive_collisions(self) -> None:
        for visual, collision in (
            ("source_table_visual", "source_table_top"),
            ("jug_visual", "jug_body"),
        ):
            visual_id = mujoco.mj_name2id(
                self.env.model, mujoco.mjtObj.mjOBJ_GEOM, visual
            )
            collision_id = mujoco.mj_name2id(
                self.env.model, mujoco.mjtObj.mjOBJ_GEOM, collision
            )
            self.assertEqual(
                self.env.model.geom_type[visual_id], mujoco.mjtGeom.mjGEOM_MESH
            )
            self.assertEqual(self.env.model.geom_contype[visual_id], 0)
            self.assertNotEqual(self.env.model.geom_contype[collision_id], 0)

    def test_reset_is_seeded(self) -> None:
        self.env.reset(seed=12)
        first = self.env.get_scene_state()
        robot_position = self.env.get_robot_state().base_position.copy()

        self.env.reset(seed=12)
        repeated = self.env.get_scene_state()
        np.testing.assert_allclose(
            first.source_table_position, repeated.source_table_position
        )
        np.testing.assert_allclose(first.container_position, repeated.container_position)
        np.testing.assert_allclose(
            first.container_center_of_mass, repeated.container_center_of_mass
        )
        np.testing.assert_allclose(
            robot_position, self.env.get_robot_state().base_position
        )

    def test_randomization_stays_in_physical_ranges(self) -> None:
        for seed in range(20):
            self.env.reset(seed=seed)
            state = self.env.get_scene_state()
            self.assertTrue(0.25 <= state.container_size[0] <= 0.30)
            self.assertTrue(0.42 <= state.container_size[1] <= 0.50)
            self.assertTrue(5.0 <= state.container_mass <= 10.0)
            self.assertTrue(0.48 <= state.container_friction <= 0.82)
            self.assertGreater(
                state.target_table_position[1] - state.source_table_position[1],
                1.5,
            )

    def test_container_moves_under_external_force(self) -> None:
        self.env.reset(seed=3)
        body_id = mujoco.mj_name2id(
            self.env.model, mujoco.mjtObj.mjOBJ_BODY, "carry_jug"
        )
        initial_x = self.env.get_scene_state().container_position[0]
        self.env.data.xfrc_applied[body_id, 0] = 30.0
        for _ in range(10):
            mujoco.mj_step(self.env.model, self.env.data)
        self.assertGreater(
            self.env.get_scene_state().container_position[0], initial_x
        )


if __name__ == "__main__":
    unittest.main()
