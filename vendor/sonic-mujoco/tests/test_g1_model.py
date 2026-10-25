import unittest

import mujoco
import numpy as np

from sonic_mujoco.envs.mujoco.g1 import MujocoG1EmptyEnv
from sonic_mujoco.scan import mujoco_camera_pose


class G1ModelTest(unittest.TestCase):
    def setUp(self) -> None:
        self.env = MujocoG1EmptyEnv()

    def tearDown(self) -> None:
        self.env.close()

    def test_model_dimensions(self) -> None:
        self.assertEqual((self.env.model.nq, self.env.model.nv), (78, 77))
        self.assertEqual(self.env.model.nu, 71)
        self.assertEqual(len(self.env.joint_ids), 29)
        self.assertEqual(len(self.env.hand_joint_ids), 40)
        self.assertEqual(len(self.env.neck_joint_ids), 2)

    def test_head_cameras_move_with_the_neck(self) -> None:
        head_id = mujoco.mj_name2id(
            self.env.model, mujoco.mjtObj.mjOBJ_BODY, "head_link"
        )
        for name in (
            "head_camera",
            "head_camera_scan",
            "head_camera_left",
            "head_camera_right",
        ):
            camera_id = mujoco.mj_name2id(
                self.env.model, mujoco.mjtObj.mjOBJ_CAMERA, name
            )
            self.assertEqual(self.env.model.cam_bodyid[camera_id], head_id)

    def test_open2dof_head_uses_a_serial_neck_and_real_inertias(self) -> None:
        neck_id = mujoco.mj_name2id(
            self.env.model, mujoco.mjtObj.mjOBJ_BODY, "neck_link"
        )
        head_id = mujoco.mj_name2id(
            self.env.model, mujoco.mjtObj.mjOBJ_BODY, "head_link"
        )
        yaw_id = mujoco.mj_name2id(
            self.env.model, mujoco.mjtObj.mjOBJ_JOINT, "neck_yaw_joint"
        )
        pitch_id = mujoco.mj_name2id(
            self.env.model, mujoco.mjtObj.mjOBJ_JOINT, "neck_pitch_joint"
        )

        self.assertEqual(self.env.model.body_parentid[head_id], neck_id)
        self.assertEqual(self.env.model.jnt_bodyid[yaw_id], neck_id)
        self.assertEqual(self.env.model.jnt_bodyid[pitch_id], head_id)
        self.assertAlmostEqual(self.env.model.body_mass[neck_id], 0.140262)
        self.assertAlmostEqual(self.env.model.body_mass[head_id], 0.499294)
        for mesh_name in (
            "open2dof_neck_base",
            "open2dof_neck",
            "open2dof_head",
        ):
            self.assertGreaterEqual(
                mujoco.mj_name2id(self.env.model, mujoco.mjtObj.mjOBJ_MESH, mesh_name),
                0,
            )

    def test_neck_axes_preserve_the_teleoperation_sign_convention(self) -> None:
        self.env.reset()
        self.env.data.qpos[self.env._neck_qpos_indices] = [0.2, 0.0]
        mujoco.mj_forward(self.env.model, self.env.data)
        yawed = mujoco_camera_pose(self.env.model, self.env.data, "head_camera_scan")
        self.assertGreater((-yawed[:3, 2])[1], 0.1)

        self.env.reset()
        self.env.data.qpos[self.env._neck_qpos_indices] = [0.0, 0.2]
        mujoco.mj_forward(self.env.model, self.env.data)
        pitched = mujoco_camera_pose(self.env.model, self.env.data, "head_camera_scan")
        self.assertLess((-pitched[:3, 2])[2], -0.1)

    def test_head_cameras_start_outside_the_open2dof_mesh(self) -> None:
        self.env.reset()
        head_id = mujoco.mj_name2id(
            self.env.model, mujoco.mjtObj.mjOBJ_BODY, "head_link"
        )
        head_geoms = np.flatnonzero(self.env.model.geom_bodyid == head_id)
        renderer = mujoco.Renderer(self.env.model, height=96, width=128)
        renderer.enable_segmentation_rendering()
        try:
            for name in (
                "head_camera",
                "head_camera_scan",
                "head_camera_left",
                "head_camera_right",
            ):
                renderer.update_scene(self.env.data, camera=name)
                segmentation = renderer.render()
                head_visible = (
                    segmentation[..., 1] == mujoco.mjtObj.mjOBJ_GEOM
                ) & np.isin(segmentation[..., 0], head_geoms)
                self.assertLess(
                    head_visible.mean(),
                    0.01,
                    f"{name} starts inside the head mesh",
                )
        finally:
            renderer.close()

    def test_vr_cameras_form_a_parallel_64_mm_stereo_pair(self) -> None:
        self.env.reset()
        view_body_id = mujoco.mj_name2id(
            self.env.model, mujoco.mjtObj.mjOBJ_BODY, "vr_view"
        )
        mocap_id = self.env.model.body_mocapid[view_body_id]
        base_pose = mujoco_camera_pose(self.env.model, self.env.data, "vr_camera_base")
        self.env.data.mocap_pos[mocap_id] = base_pose[:3, 3]
        mujoco.mju_mat2Quat(
            self.env.data.mocap_quat[mocap_id],
            base_pose[:3, :3].reshape(-1),
        )
        mujoco.mj_forward(self.env.model, self.env.data)

        left = mujoco_camera_pose(self.env.model, self.env.data, "vr_camera_left")
        right = mujoco_camera_pose(self.env.model, self.env.data, "vr_camera_right")

        np.testing.assert_allclose((left[:3, 3] + right[:3, 3]) / 2.0, base_pose[:3, 3])
        self.assertAlmostEqual(np.linalg.norm(right[:3, 3] - left[:3, 3]), 0.064)
        np.testing.assert_allclose(left[:3, :3], base_pose[:3, :3])
        np.testing.assert_allclose(right[:3, :3], base_pose[:3, :3])

    def test_reset(self) -> None:
        self.env.reset()
        self.assertAlmostEqual(self.env.time, 0.0)

    def test_robot_cameras_have_documented_virtual_mounts(self) -> None:
        self.env.reset()
        left = mujoco_camera_pose(self.env.model, self.env.data, "head_camera_left")
        right = mujoco_camera_pose(self.env.model, self.env.data, "head_camera_right")
        self.assertAlmostEqual(np.linalg.norm(left[:3, 3] - right[:3, 3]), 0.060)
        np.testing.assert_allclose(left[:3, :3], right[:3, :3])
        forward = -left[:3, 2]
        self.assertAlmostEqual(np.arcsin(-forward[2]), np.deg2rad(20), places=4)
        for side in ("left", "right"):
            camera = self.env.model.camera(f"{side}_wrist_camera").id
            self.assertEqual(
                self.env.model.cam_bodyid[camera],
                self.env.model.body(f"{side}_wrist_yaw_link").id,
            )


if __name__ == "__main__":
    unittest.main()
