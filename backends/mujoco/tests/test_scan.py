import unittest

import numpy as np

from sonic_mujoco.scan import CameraIntrinsics, RelativeCameraMapper


class ScanCameraTest(unittest.TestCase):
    level_camera_rotation = np.array(
        (
            (0.0, 0.0, -1.0),
            (-1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
        )
    )

    def test_computes_intrinsics_from_vertical_fov(self) -> None:
        intrinsics = CameraIntrinsics.from_vertical_fov(640, 480, 90.0)

        self.assertAlmostEqual(intrinsics.fx, 240.0)
        self.assertAlmostEqual(intrinsics.fy, 240.0)
        self.assertEqual(intrinsics.cx, 320.0)
        self.assertEqual(intrinsics.cy, 240.0)

    def test_maps_relative_motion_onto_scan_anchor(self) -> None:
        scan_anchor = np.eye(4)
        scan_anchor[:3, :3] = self.level_camera_rotation
        scan_anchor[:3, 3] = (2.0, 3.0, 4.0)
        mujoco_anchor = np.eye(4)
        mujoco_anchor[:3, :3] = self.level_camera_rotation
        mujoco_anchor[:3, 3] = (10.0, 0.0, 0.0)
        mapper = RelativeCameraMapper(
            scan_anchor,
            mujoco_anchor,
            scan_units_per_meter=0.25,
        )
        moved = mujoco_anchor.copy()
        moved[:3, 3] += (0.0, 2.0, 0.0)

        mapped = mapper.map(moved)

        self.assertEqual(mapped.shape, (4, 4))
        np.testing.assert_allclose(mapped[:3, 3], (2.0, 3.5, 4.0))
        np.testing.assert_allclose(mapped[:3, :3], self.level_camera_rotation)

    def test_keeps_world_vertical_despite_camera_pitch(self) -> None:
        scan_anchor = np.eye(4)
        scan_anchor[:3, :3] = self.level_camera_rotation
        mujoco_anchor = np.eye(4)
        angle = np.deg2rad(45.0)
        mujoco_anchor[:3, :3] = np.array(
            (
                (0.0, np.cos(angle), -np.cos(angle)),
                (-1.0, 0.0, 0.0),
                (0.0, np.sin(angle), np.sin(angle)),
            )
        )
        mapper = RelativeCameraMapper(
            scan_anchor,
            mujoco_anchor,
            scan_units_per_meter=0.2,
        )
        raised = mujoco_anchor.copy()
        raised[2, 3] = 1.0

        np.testing.assert_allclose(mapper.map(mujoco_anchor), scan_anchor, atol=1e-12)
        np.testing.assert_allclose(mapper.map(raised)[:3, 3], (0.0, 0.0, 0.2))

    def test_rejects_invalid_pose_and_scale(self) -> None:
        with self.assertRaisesRegex(ValueError, "scan_units_per_meter"):
            RelativeCameraMapper(np.eye(4), np.eye(4), scan_units_per_meter=0.0)
        with self.assertRaisesRegex(ValueError, "shape"):
            RelativeCameraMapper(np.eye(3), np.eye(4), scan_units_per_meter=1.0)


if __name__ == "__main__":
    unittest.main()
