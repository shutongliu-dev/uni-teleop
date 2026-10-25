import unittest

import mujoco
import numpy as np

from sonic_mujoco.envs.mujoco.h2 import (
    H2_ACTUATOR_NAMES,
    H2_DOF,
    H2_HEAD_SLICE,
    H2_HOME_JOINT_POSITION,
    H2_JOINT_NAMES,
    H2Command,
    MujocoH2EmptyEnv,
)
from sonic_mujoco.scan import mujoco_camera_pose


class H2ModelTest(unittest.TestCase):
    def setUp(self) -> None:
        self.env = MujocoH2EmptyEnv()

    def tearDown(self) -> None:
        self.env.close()

    def test_official_model_dimensions_and_pinned_actuator_order(self) -> None:
        self.assertEqual((self.env.model.nq, self.env.model.nv), (38, 37))
        self.assertEqual(self.env.model.nu, H2_DOF)
        self.assertEqual(self.env.robot_spec.dof, H2_DOF)
        self.assertEqual(len(set(self.env.joint_ids)), H2_DOF)
        self.assertEqual(self.env.actuator_ids, tuple(range(H2_DOF)))

        actuator_names = tuple(
            mujoco.mj_id2name(self.env.model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_id)
            for actuator_id in self.env.actuator_ids
        )
        self.assertEqual(actuator_names, H2_ACTUATOR_NAMES)
        for joint_name, joint_id, actuator_id in zip(
            H2_JOINT_NAMES,
            self.env.joint_ids,
            self.env.actuator_ids,
            strict=True,
        ):
            self.assertEqual(
                mujoco.mj_id2name(self.env.model, mujoco.mjtObj.mjOBJ_JOINT, joint_id),
                joint_name,
            )
            self.assertEqual(self.env.model.actuator_trnid[actuator_id, 0], joint_id)

    def test_named_home_keyframe_is_valid_and_reset_loads_it(self) -> None:
        self.env.reset()
        expected = np.asarray(H2_HOME_JOINT_POSITION)
        np.testing.assert_allclose(
            self.env.get_robot_state().joint_position, expected, atol=1e-12
        )
        np.testing.assert_allclose(
            self.env.get_robot_state().base_position, [0, 0, 1.03]
        )
        limited = self.env._body_joints.joint_limited
        joint_range = self.env._body_joints.joint_range
        self.assertTrue(np.all(expected[limited] >= joint_range[limited, 0]))
        self.assertTrue(np.all(expected[limited] <= joint_range[limited, 1]))

        # Head joints are early in qpos tree order but last in the pinned MJCF order.
        self.assertEqual(tuple(self.env._qpos_indices[H2_HEAD_SLICE]), (22, 23))
        self.assertEqual(tuple(self.env._actuator_indices[H2_HEAD_SLICE]), (29, 30))

    def test_state_uses_actuator_order_and_owns_its_arrays(self) -> None:
        self.env.reset()
        state = self.env.get_robot_state()
        self.assertEqual(state.joint_position.shape, (H2_DOF,))
        self.assertEqual(state.joint_velocity.shape, (H2_DOF,))
        self.assertEqual(state.joint_effort.shape, (H2_DOF,))
        self.assertEqual(state.imu_quaternion.shape, (4,))
        self.assertEqual(state.imu_angular_velocity.shape, (3,))
        self.assertEqual(state.imu_linear_acceleration.shape, (3,))
        np.testing.assert_allclose(
            state.joint_position, self.env.data.qpos[self.env._qpos_indices]
        )
        self.assertFalse(np.shares_memory(state.joint_position, self.env.data.qpos))

    def test_pd_control_maps_to_model_actuators_and_clips_effort(self) -> None:
        self.env.reset()
        state = self.env.get_robot_state()
        zeros = np.zeros(H2_DOF)
        command = H2Command(
            joint_position=state.joint_position + 0.01,
            joint_velocity=zeros,
            feedforward_torque=np.full(H2_DOF, 1e6),
            kp=zeros,
            kd=zeros,
        )
        self.env.step(command)
        np.testing.assert_allclose(
            self.env.data.ctrl[self.env._actuator_indices],
            self.env._control_range[:, 1],
        )

    def test_home_hold_stays_finite_during_smoke_window(self) -> None:
        self.env.reset()
        self.env.step(self.env.home_command(), 250)
        self.assertTrue(np.isfinite(self.env.data.qpos).all())
        self.assertTrue(np.isfinite(self.env.data.qvel).all())
        self.assertGreater(self.env.get_robot_state().base_position[2], 0.8)

    def test_head_and_wrist_camera_contract(self) -> None:
        self.env.reset()
        head_id = mujoco.mj_name2id(
            self.env.model, mujoco.mjtObj.mjOBJ_BODY, "head_yaw_link"
        )
        left_wrist_id = mujoco.mj_name2id(
            self.env.model, mujoco.mjtObj.mjOBJ_BODY, "left_wrist_yaw_link"
        )
        right_wrist_id = mujoco.mj_name2id(
            self.env.model, mujoco.mjtObj.mjOBJ_BODY, "right_wrist_yaw_link"
        )
        expected_bodies = {
            "head_camera": head_id,
            "head_camera_left": head_id,
            "head_camera_right": head_id,
            "left_wrist_camera": left_wrist_id,
            "right_wrist_camera": right_wrist_id,
        }
        for name, expected_body in expected_bodies.items():
            camera_id = mujoco.mj_name2id(
                self.env.model, mujoco.mjtObj.mjOBJ_CAMERA, name
            )
            self.assertEqual(self.env.model.cam_bodyid[camera_id], expected_body)

        left = mujoco_camera_pose(self.env.model, self.env.data, "head_camera_left")
        right = mujoco_camera_pose(self.env.model, self.env.data, "head_camera_right")
        self.assertAlmostEqual(np.linalg.norm(left[:3, 3] - right[:3, 3]), 0.064)
        np.testing.assert_allclose(left[:3, :3], right[:3, :3], atol=1e-12)

        before = mujoco_camera_pose(self.env.model, self.env.data, "head_camera")
        np.testing.assert_allclose(-before[:3, 2], [1.0, 0.0, 0.0], atol=1e-5)
        self.env.data.qpos[self.env._qpos_indices[H2_HEAD_SLICE]] = [0.2, 0.3]
        mujoco.mj_forward(self.env.model, self.env.data)
        after = mujoco_camera_pose(self.env.model, self.env.data, "head_camera")
        self.assertGreater(np.linalg.norm(after - before), 0.1)

    def test_camera_views_start_outside_their_mounting_link(self) -> None:
        self.env.reset()
        renderer = mujoco.Renderer(self.env.model, height=96, width=128)
        renderer.enable_segmentation_rendering()
        try:
            for name in self.env.robot_spec.camera_names:
                camera_id = mujoco.mj_name2id(
                    self.env.model, mujoco.mjtObj.mjOBJ_CAMERA, name
                )
                mounting_body = self.env.model.cam_bodyid[camera_id]
                mounting_geoms = np.flatnonzero(
                    self.env.model.geom_bodyid == mounting_body
                )
                renderer.update_scene(self.env.data, camera=name)
                segmentation = renderer.render()
                mounting_link_visible = (
                    segmentation[..., 1] == mujoco.mjtObj.mjOBJ_GEOM
                ) & np.isin(segmentation[..., 0], mounting_geoms)
                self.assertLess(
                    mounting_link_visible.mean(),
                    0.01,
                    f"{name} starts inside its mounting link",
                )
        finally:
            renderer.close()

    def test_native_head_is_controlled_in_the_main_31dof_command(self) -> None:
        self.env.reset()
        target = np.asarray(H2_HOME_JOINT_POSITION).copy()
        target[H2_HEAD_SLICE] = [0.1, -0.2]
        zeros = np.zeros(H2_DOF)
        kp = zeros.copy()
        kp[H2_HEAD_SLICE] = 10.0
        self.env.step(H2Command(target, zeros, zeros, kp, zeros))

        np.testing.assert_allclose(
            self.env.data.ctrl[self.env._actuator_indices[H2_HEAD_SLICE]],
            [1.0, -2.0],
            atol=1e-12,
        )

    def test_command_rejects_wrong_shape_and_nonfinite_values(self) -> None:
        zeros = np.zeros(H2_DOF)
        with self.assertRaises(ValueError):
            H2Command(np.zeros(H2_DOF - 1), zeros, zeros, zeros, zeros)
        invalid = zeros.copy()
        invalid[0] = np.nan
        with self.assertRaises(ValueError):
            H2Command(zeros, invalid, zeros, zeros, zeros)


if __name__ == "__main__":
    unittest.main()
