import unittest

import numpy as np

from sonic_mujoco.teleop import DexHandRetargeter, dexhand_controller_targets


def open_hand() -> np.ndarray:
    hand = np.zeros((25, 3), dtype=np.float64)
    hand[0] = [0.0, 0.0, 0.0]
    hand[1:5] = [
        [0.025, 0.025, 0.0],
        [0.045, 0.045, 0.0],
        [0.060, 0.065, 0.0],
        [0.075, 0.085, 0.0],
    ]
    for start, x in ((5, 0.030), (10, 0.0), (15, -0.020), (20, -0.035)):
        hand[start : start + 5] = [
            [x, 0.040, 0.0],
            [x, 0.070, 0.0],
            [x, 0.100, 0.0],
            [x, 0.125, 0.0],
            [x, 0.145, 0.0],
        ]
    return hand


class DexHandRetargeterTest(unittest.TestCase):
    def test_openxr_and_wrist_first_layouts_match(self) -> None:
        hand = open_hand()
        openxr = np.vstack((np.array([[0.0, 0.04, 0.0]]), hand))

        left = DexHandRetargeter(low_pass_alpha=1.0).retarget_hand(hand, "left")
        openxr_left = DexHandRetargeter(low_pass_alpha=1.0).retarget_hand(
            openxr, "left"
        )

        np.testing.assert_allclose(left, openxr_left)
        self.assertEqual(left.shape, (20,))

    def test_index_curl_maps_to_mirrored_hand_joints(self) -> None:
        hand = open_hand()
        hand[7] = hand[6] + [0.025, 0.0, 0.0]
        hand[8] = hand[7] + [0.0, -0.025, 0.0]
        hand[9] = hand[8] + [-0.020, 0.0, 0.0]

        left = DexHandRetargeter(low_pass_alpha=1.0).retarget_hand(hand, "left")
        right = DexHandRetargeter(low_pass_alpha=1.0).retarget_hand(hand, "right")

        self.assertLess(left[5], -1.0)
        self.assertLess(left[6], -1.0)
        self.assertGreater(right[5], 1.0)
        self.assertGreater(right[6], 1.0)

    def test_controller_fallback_separates_index_and_middle(self) -> None:
        target = dexhand_controller_targets(1.0, 0.0, 0.0, 1.0)

        self.assertLess(target[5], 0.0)
        self.assertEqual(target[9], 0.0)
        self.assertGreater(target[29], 0.0)
        self.assertEqual(target[25], 0.0)


if __name__ == "__main__":
    unittest.main()
