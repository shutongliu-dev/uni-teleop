import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from sonic_mujoco.teleop.desktop_video import fit_viewport
from sonic_mujoco.teleop.pico_video import (
    RelativeHeadsetView,
    _recording_overlay,
    _stereo_frame,
    _tracking_warning_overlay,
    _validate_scan_alignment,
)


class PicoVideoTest(unittest.TestCase):
    def test_scan_alignment_rejects_mismatched_anchor_or_floor(self) -> None:
        renderer = {
            "anchor_image": "frame.jpg",
            "anchor_camera_to_world": np.eye(4).tolist(),
            "scan_up": [0.0, 0.0, 1.0],
        }
        scene = dict(renderer)

        _validate_scan_alignment(renderer, scene)
        with self.assertRaisesRegex(ValueError, "different anchors"):
            _validate_scan_alignment(renderer, {**scene, "anchor_image": "other.jpg"})
        with self.assertRaisesRegex(ValueError, "anchor poses differ"):
            moved = np.eye(4)
            moved[0, 3] = 0.01
            _validate_scan_alignment(
                renderer,
                {**scene, "anchor_camera_to_world": moved.tolist()},
            )
        with self.assertRaisesRegex(ValueError, "floor normals differ"):
            _validate_scan_alignment(renderer, {**scene, "scan_up": [0.0, 1.0, 0.0]})

    def test_stereo_frame_keeps_distinct_left_and_right_images(self) -> None:
        left = np.full((3, 4, 3), 17, dtype=np.uint8)
        right = np.full((3, 4, 3), 93, dtype=np.uint8)

        stereo = _stereo_frame(left, right)

        self.assertEqual(stereo.shape, (3, 8, 3))
        np.testing.assert_array_equal(stereo[:, :4], left)
        np.testing.assert_array_equal(stereo[:, 4:], right)
        with self.assertRaisesRegex(ValueError, "matching dimensions"):
            _stereo_frame(left, right[:, :3])

    def test_headset_view_is_relative_to_first_valid_pose(self) -> None:
        tracker = RelativeHeadsetView()
        reference = np.array((1.0, 1.5, -0.4, 0.0, 0.0, 0.0, 1.0))
        moved = reference.copy()
        moved[:3] += (0.08, -0.03, -0.12)
        moved[3:] = Rotation.from_euler("y", 20, degrees=True).as_quat()

        np.testing.assert_allclose(tracker.update(np.zeros(7)), np.eye(4))
        np.testing.assert_allclose(tracker.update(reference), np.eye(4), atol=1e-12)
        relative = tracker.update(moved)

        np.testing.assert_allclose(relative[:3, 3], (0.08, -0.03, -0.12))
        np.testing.assert_allclose(
            relative[:3, :3],
            Rotation.from_euler("y", 20, degrees=True).as_matrix(),
        )
        self.assertTrue(tracker.tracking_active)
        tracker.recenter()
        self.assertFalse(tracker.tracking_active)
        np.testing.assert_allclose(tracker.update(moved), np.eye(4), atol=1e-12)

    def test_desktop_viewport_preserves_camera_aspect_ratio(self) -> None:
        self.assertEqual(fit_viewport(640, 480, 1200, 600), (200, 0, 800, 600))
        self.assertEqual(fit_viewport(640, 480, 600, 900), (0, 225, 600, 450))

    def test_recording_overlay_draws_timer_without_mutating_source(self) -> None:
        frame = np.full((80, 240, 3), 17, dtype=np.uint8)

        first = _recording_overlay(frame, 1.8)
        second = _recording_overlay(frame, 61.2)

        self.assertTrue(np.all(frame == 17))
        np.testing.assert_array_equal(first[12, 12], [190, 0, 0])
        self.assertTrue(np.any(np.all(first == 255, axis=2)))
        self.assertFalse(np.array_equal(first, second))
        self.assertTrue(np.all(first[50:] == 17))

    def test_tracking_warning_is_visible_without_mutating_source(self) -> None:
        frame = np.full((90, 240, 3), 17, dtype=np.uint8)

        warning = _tracking_warning_overlay(frame)

        self.assertTrue(np.all(frame == 17))
        np.testing.assert_array_equal(warning[40, 12], [190, 70, 0])
        self.assertTrue(np.any(np.all(warning == 255, axis=2)))
        self.assertTrue(np.all(warning[:39] == 17))


if __name__ == "__main__":
    unittest.main()
