import unittest

import numpy as np

from sonic_mujoco.teleop import neck_joint_targets


class NeckRetargetingTest(unittest.TestCase):
    def test_head_yaw_and_pitch_map_to_neck_axes(self) -> None:
        pose = np.zeros((2, 21, 3), dtype=np.float64)
        pose[0, 14, 1] = 0.4
        pose[1, 14, 0] = -0.3

        target = neck_joint_targets(pose)

        np.testing.assert_allclose(target[0], [0.4, 0.0], atol=1e-7)
        np.testing.assert_allclose(target[1], [0.0, -0.3], atol=1e-7)

    def test_neck_and_head_rotations_are_composed_and_clipped(self) -> None:
        pose = np.zeros((21, 3), dtype=np.float64)
        pose[11, 1] = 0.8
        pose[14, 1] = 0.8
        pose[14, 0] = 0.8

        target = neck_joint_targets(pose)

        self.assertEqual(target.shape, (2,))
        self.assertLessEqual(target[0], 1.2)
        self.assertLessEqual(target[1], 0.55)

    def test_invalid_pose_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "shape"):
            neck_joint_targets(np.zeros((24, 3)))


if __name__ == "__main__":
    unittest.main()
