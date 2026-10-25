import unittest

import mujoco
import numpy as np

from sonic_mujoco.envs.mujoco.g1 import MujocoG1SweepEnv
from sonic_mujoco.envs.mujoco.g1.sweep_env import OBJECT_NAMES


class SweepEnvTest(unittest.TestCase):
    def setUp(self) -> None:
        self.env = MujocoG1SweepEnv()

    def tearDown(self) -> None:
        self.env.close()

    def test_scene_contains_table_target_and_free_objects(self) -> None:
        table = mujoco.mj_name2id(self.env.model, mujoco.mjtObj.mjOBJ_BODY, "table")
        table_joint = mujoco.mj_name2id(
            self.env.model, mujoco.mjtObj.mjOBJ_JOINT, "table_joint"
        )
        divider = mujoco.mj_name2id(
            self.env.model, mujoco.mjtObj.mjOBJ_GEOM, "divider_tape"
        )
        target = mujoco.mj_name2id(
            self.env.model, mujoco.mjtObj.mjOBJ_SITE, "sweep_target"
        )
        self.assertGreaterEqual(table, 0)
        self.assertEqual(
            self.env.model.jnt_type[table_joint], mujoco.mjtJoint.mjJNT_FREE
        )
        self.assertAlmostEqual(self.env.model.body_mass[table], 30.0)
        self.assertGreaterEqual(divider, 0)
        self.assertGreaterEqual(target, 0)
        self.assertEqual(self.env.model.geom_bodyid[divider], table)
        self.assertEqual(self.env.model.site_bodyid[target], table)
        for name in OBJECT_NAMES:
            joint = mujoco.mj_name2id(
                self.env.model, mujoco.mjtObj.mjOBJ_JOINT, f"{name}_joint"
            )
            self.assertEqual(self.env.model.jnt_type[joint], mujoco.mjtJoint.mjJNT_FREE)
        half_size = self.env.model.site_size[target]
        self.assertGreater(half_size[0], half_size[1])
        self.assertGreater(self.env.model.site_pos[target, 1], 0.0)
        self.assertEqual(len(OBJECT_NAMES), 5)

    def test_scene_only_has_rear_curtain(self) -> None:
        curtain = mujoco.mj_name2id(
            self.env.model, mujoco.mjtObj.mjOBJ_MATERIAL, "curtain"
        )
        curtain_geoms = np.flatnonzero(self.env.model.geom_matid == curtain)
        names = [
            mujoco.mj_id2name(self.env.model, mujoco.mjtObj.mjOBJ_GEOM, geom)
            for geom in curtain_geoms
        ]
        self.assertEqual(names, ["rear_curtain"])

    def test_seeded_reset_is_reproducible(self) -> None:
        self.env.reset(seed=7)
        first = self.env.get_scene_state().object_position
        self.env.reset(seed=7)
        second = self.env.get_scene_state().object_position
        self.env.reset(seed=8)
        third = self.env.get_scene_state().object_position

        np.testing.assert_array_equal(first, second)
        self.assertFalse(np.array_equal(first, third))
        self.assertTrue(np.all(first[:, 1] < 0.0))
        self.assertFalse(self.env.is_success())

    def test_seeded_physics_randomization_is_reproducible(self) -> None:
        self.env.reset(seed=7)
        first_mass = self.env.model.body_mass[self.env._physics_body_ids].copy()
        first_friction = self.env.model.geom_friction[
            self.env._physics_geom_ids
        ].copy()
        self.env.reset(seed=7)
        second_mass = self.env.model.body_mass[self.env._physics_body_ids].copy()
        second_friction = self.env.model.geom_friction[
            self.env._physics_geom_ids
        ].copy()
        self.env.reset(seed=8)
        third_mass = self.env.model.body_mass[self.env._physics_body_ids].copy()

        np.testing.assert_array_equal(first_mass, second_mass)
        np.testing.assert_array_equal(first_friction, second_friction)
        self.assertFalse(np.array_equal(first_mass, third_mass))
        self.assertTrue(np.all(first_mass[1:] > 0.0))

    def test_success_when_all_objects_are_in_target(self) -> None:
        self.env.reset(seed=0)
        site_id = mujoco.mj_name2id(
            self.env.model, mujoco.mjtObj.mjOBJ_SITE, "sweep_target"
        )
        center = self.env.data.site_xpos[site_id]
        x_offsets = np.linspace(-0.24, 0.24, len(OBJECT_NAMES))
        for offset, name in zip(x_offsets, OBJECT_NAMES):
            joint_id = mujoco.mj_name2id(
                self.env.model, mujoco.mjtObj.mjOBJ_JOINT, f"{name}_joint"
            )
            address = self.env.model.jnt_qposadr[joint_id]
            self.env.data.qpos[address : address + 3] = center + [
                offset,
                0.0,
                0.024,
            ]
        mujoco.mj_forward(self.env.model, self.env.data)

        self.assertTrue(self.env.is_success())
        self.assertTrue(self.env.get_scene_state().success)


if __name__ == "__main__":
    unittest.main()
