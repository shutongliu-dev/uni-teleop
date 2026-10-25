import unittest

import numpy as np

from sonic_mujoco.superdex import (
    DG5F_JOINT_NAMES,
    SuperDexHandCommand,
    dexhand_to_dg5f,
)
from sonic_mujoco.teleop import dexhand_controller_targets


class SuperDexProtocolTest(unittest.TestCase):
    def test_hand_command_round_trip(self) -> None:
        target = np.linspace(-1.0, 1.0, 40)
        command = SuperDexHandCommand(7, 123456, target)

        decoded = SuperDexHandCommand.decode(command.encode())

        self.assertEqual(decoded.sequence, 7)
        self.assertEqual(decoded.monotonic_ns, 123456)
        np.testing.assert_allclose(decoded.joint_position, target, rtol=1e-6)

    def test_hand_command_rejects_invalid_shape(self) -> None:
        with self.assertRaisesRegex(ValueError, "shape"):
            SuperDexHandCommand(0, 0, np.zeros(20))


class Dg5fMappingTest(unittest.TestCase):
    def test_mapping_has_one_target_per_dg5f_joint(self) -> None:
        self.assertEqual(len(DG5F_JOINT_NAMES), 20)
        target = dexhand_to_dg5f(np.zeros(20), "right")
        self.assertEqual(target.shape, (20,))
        self.assertTrue(np.isfinite(target).all())

    def test_closing_controller_targets_flexes_both_hands(self) -> None:
        target = dexhand_controller_targets(1.0, 1.0, 1.0, 1.0)
        left_open = dexhand_to_dg5f(np.zeros(20), "left")
        right_open = dexhand_to_dg5f(np.zeros(20), "right")
        left_closed = dexhand_to_dg5f(target[:20], "left")
        right_closed = dexhand_to_dg5f(target[20:], "right")

        self.assertGreater(left_closed[6], left_open[6])
        self.assertGreater(right_closed[6], right_open[6])
        self.assertLess(left_closed[2], left_open[2])
        self.assertGreater(right_closed[2], right_open[2])

    def test_left_and_right_thumb_targets_are_mirrored(self) -> None:
        target = dexhand_controller_targets(1.0, 0.0, 1.0, 0.0)
        left = dexhand_to_dg5f(target[:20], "left")
        right = dexhand_to_dg5f(target[20:], "right")

        np.testing.assert_allclose(left[:4], -right[:4], atol=1e-7)


if __name__ == "__main__":
    unittest.main()
