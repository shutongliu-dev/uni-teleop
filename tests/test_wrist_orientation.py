import math
import unittest
import numpy as np
from scipy.spatial.transform import Rotation
from uni_teleop.video.wrist_orientation import WristDisplayRotation


class WristOrientationTests(unittest.TestCase):
    def test_roll_aligns_projected_right_and_forward(self):
        control = np.eye(4)
        control[:3, :3] = Rotation.from_euler('xyz', [50, -10, 100], degrees=True).as_matrix()
        aligned = np.column_stack((control[:3, 0], -control[:3, 2], control[:3, 1]))
        for roll in (-150, -90, 0, 50, 179):
            camera = np.eye(4)
            camera[:3, :3] = aligned @ Rotation.from_euler('z', roll, degrees=True).as_matrix()
            angle = WristDisplayRotation().update(camera, control, now=0)
            rotation = Rotation.from_euler('z', angle, degrees=True).as_matrix()[:2, :2]
            axes = camera[:3, :2].T @ np.column_stack((control[:3, 0], -control[:3, 2]))
            np.testing.assert_allclose(rotation @ axes, np.eye(2), atol=1e-12)

    def test_ambiguous_projection_holds_and_motion_is_rate_limited(self):
        control = np.eye(4)
        camera = np.eye(4)
        camera[:3, :3] = np.column_stack(([1, 0, 0], [0, 0, -1], [0, 1, 0]))
        display = WristDisplayRotation()
        self.assertAlmostEqual(display.update(camera, control, now=0), 0)
        rotated = camera.copy()
        rotated[:3, :3] = camera[:3, :3] @ Rotation.from_euler('z', 100, degrees=True).as_matrix()
        angle = display.update(rotated, control, now=.05)
        self.assertGreater(angle, 0)
        self.assertLessEqual(angle, 4.5 + 1e-8)
        # Facing the opposite side makes right/forward projection a reflection;
        # no image roll can satisfy both, so hold rather than flip.
        camera[:3, :3] = np.column_stack(([1, 0, 0], [0, 0, 1], [0, -1, 0]))
        self.assertEqual(display.update(camera, control, now=.10), angle)
