import unittest

import mujoco
import numpy as np

from sonic_mujoco.envs.mujoco.g1 import MujocoG1DoorElbowEnv


class DoorElbowEnvTest(unittest.TestCase):
    def setUp(self) -> None:
        self.env = MujocoG1DoorElbowEnv()

    def tearDown(self) -> None:
        self.env.close()

    def test_reference_door_has_a_resisted_hinge(self) -> None:
        body_id = mujoco.mj_name2id(
            self.env.model, mujoco.mjtObj.mjOBJ_BODY, "door"
        )
        joint_id = mujoco.mj_name2id(
            self.env.model, mujoco.mjtObj.mjOBJ_JOINT, "door_hinge"
        )
        dof = int(self.env.model.jnt_dofadr[joint_id])
        leaf_id = mujoco.mj_name2id(
            self.env.model, mujoco.mjtObj.mjOBJ_GEOM, "door_leaf"
        )

        self.assertEqual(
            self.env.model.jnt_type[joint_id], mujoco.mjtJoint.mjJNT_HINGE
        )
        np.testing.assert_allclose(
            2 * self.env.model.geom_size[leaf_id], (0.044, 0.90, 2.04)
        )
        self.assertAlmostEqual(self.env.model.body_mass[body_id], 25.0)
        self.assertGreater(self.env.model.dof_damping[dof], 0.0)
        self.assertGreater(self.env.model.dof_frictionloss[dof], 0.0)
        visual_id = mujoco.mj_name2id(
            self.env.model, mujoco.mjtObj.mjOBJ_GEOM, "door_visual_mesh"
        )
        self.assertEqual(
            self.env.model.geom_type[visual_id], mujoco.mjtGeom.mjGEOM_MESH
        )
        self.assertEqual(self.env.model.geom_contype[visual_id], 0)
        self.assertNotEqual(self.env.model.geom_contype[leaf_id], 0)

    def test_door_opens_into_a_bounded_room_with_clear_aisle(self) -> None:
        mujoco.mj_forward(self.env.model, self.env.data)
        for name in (
            "room_floor",
            "room_left_wall",
            "room_right_wall",
            "room_back_wall",
        ):
            geom_id = mujoco.mj_name2id(
                self.env.model, mujoco.mjtObj.mjOBJ_GEOM, name
            )
            self.assertGreaterEqual(geom_id, 0)

        for name in ("room_sofa_base", "room_table"):
            geom_id = mujoco.mj_name2id(
                self.env.model, mujoco.mjtObj.mjOBJ_GEOM, name
            )
            self.assertGreater(abs(self.env.data.geom_xpos[geom_id, 1]), 0.8)

    def test_reset_is_seeded_and_randomizes_task_variations(self) -> None:
        self.env.reset(seed=12)
        first = self.env.get_scene_state()
        robot = self.env.get_robot_state().base_position.copy()
        self.env.reset(seed=12)
        repeated = self.env.get_scene_state()

        self.assertEqual(first, repeated)
        np.testing.assert_allclose(robot, self.env.get_robot_state().base_position)
        self.env.reset(seed=13)
        changed = self.env.get_scene_state()
        self.assertNotEqual(first.hinge_friction, changed.hinge_friction)

    def test_randomization_stays_in_physical_ranges(self) -> None:
        sides = set()
        for seed in range(30):
            self.env.reset(seed=seed)
            state = self.env.get_scene_state()
            self.assertTrue(np.deg2rad(1.0) <= state.door_angle <= np.deg2rad(10.0))
            self.assertTrue(20.0 <= state.door_mass <= 30.0)
            self.assertTrue(0.7 <= state.hinge_damping <= 2.4)
            self.assertTrue(1.0 <= state.hinge_friction <= 4.0)
            self.assertTrue(0.45 <= state.surface_friction <= 0.75)
            sides.add(state.execution_side)
        self.assertEqual(sides, {-1, 1})

    def test_applied_hinge_torque_opens_the_door(self) -> None:
        self.env.reset(seed=4)
        initial = self.env.get_scene_state().door_angle
        joint_id = mujoco.mj_name2id(
            self.env.model, mujoco.mjtObj.mjOBJ_JOINT, "door_hinge"
        )
        dof = int(self.env.model.jnt_dofadr[joint_id])
        self.env.data.qfrc_applied[dof] = -12.0
        for _ in range(120):
            mujoco.mj_step(self.env.model, self.env.data)

        state = self.env.get_scene_state()
        self.assertGreater(state.door_angle, initial + np.deg2rad(5.0))
        self.assertLessEqual(state.door_angle, np.deg2rad(100.1))

    def test_hinge_resistance_changes_the_opening_response(self) -> None:
        angles = []
        for damping, friction in ((0.7, 1.0), (2.4, 4.0)):
            self.env.reset(seed=4)
            joint_id = mujoco.mj_name2id(
                self.env.model, mujoco.mjtObj.mjOBJ_JOINT, "door_hinge"
            )
            qpos = int(self.env.model.jnt_qposadr[joint_id])
            dof = int(self.env.model.jnt_dofadr[joint_id])
            self.env.data.qpos[qpos] = -np.deg2rad(5.0)
            self.env.model.dof_damping[dof] = damping
            self.env.model.dof_frictionloss[dof] = friction
            mujoco.mj_forward(self.env.model, self.env.data)
            self.env.data.qfrc_applied[dof] = -10.0
            for _ in range(200):
                mujoco.mj_step(self.env.model, self.env.data)
            angles.append(-self.env.data.qpos[qpos])

        self.assertGreater(angles[0], angles[1] + np.deg2rad(5.0))


if __name__ == "__main__":
    unittest.main()
