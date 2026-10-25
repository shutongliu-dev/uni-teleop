import json
import unittest

import numpy as np

from sonic_mujoco.teleop import DexHandRetargeter, PicoPoseConverter, PicoTeleop


def identity_body() -> np.ndarray:
    body = np.zeros((24, 7), dtype=np.float64)
    body[:, 6] = 1.0
    return body


def tracked_hand() -> np.ndarray:
    hand = np.zeros((26, 7), dtype=np.float64)
    hand[:, 6] = 1.0
    wrist_first = np.zeros((25, 3), dtype=np.float64)
    wrist_first[1:5] = [
        [0.025, 0.025, 0.0],
        [0.045, 0.045, 0.0],
        [0.060, 0.065, 0.0],
        [0.075, 0.085, 0.0],
    ]
    for start, x in ((5, 0.030), (10, 0.0), (15, -0.020), (20, -0.035)):
        wrist_first[start : start + 5] = [
            [x, 0.040, 0.0],
            [x, 0.070, 0.0],
            [x, 0.100, 0.0],
            [x, 0.125, 0.0],
            [x, 0.145, 0.0],
        ]
    hand[1:, :3] = wrist_first
    return hand


class FakeSdk:
    def __init__(self) -> None:
        self.timestamp = 0
        self.closed = False
        self.buttons = {name: False for name in "ABXY"}
        self.left_grip = 0.0
        self.device_commands = []
        self.left_hand_active = False
        self.left_hand = tracked_hand()
        self.body_available = True
        self.headset = np.array((0.1, 1.6, -0.2, 0.0, 0.0, 0.0, 2.0))

    def init(self) -> None:
        pass

    def close(self) -> None:
        self.closed = True

    def is_body_data_available(self) -> bool:
        return self.body_available

    def get_headset_pose(self) -> np.ndarray:
        return self.headset

    def get_time_stamp_ns(self) -> int:
        self.timestamp += 20_000_000
        return self.timestamp

    def get_body_joints_pose(self) -> np.ndarray:
        return identity_body()

    def get_left_axis(self) -> tuple[float, float]:
        return 0.0, 0.0

    def get_right_axis(self) -> tuple[float, float]:
        return 0.5, 0.0

    def get_A_button(self) -> bool:
        return self.buttons["A"]

    def get_B_button(self) -> bool:
        return self.buttons["B"]

    def get_X_button(self) -> bool:
        return self.buttons["X"]

    def get_Y_button(self) -> bool:
        return self.buttons["Y"]

    def get_left_grip(self) -> float:
        return self.left_grip

    def get_left_hand_is_active(self) -> int:
        return int(self.left_hand_active)

    def get_left_hand_tracking_state(self) -> np.ndarray:
        return self.left_hand

    def device_control_json(self, device_id: str, command: str) -> None:
        self.device_commands.append((device_id, json.loads(command)))


class PicoDirectTest(unittest.TestCase):
    def test_headset_pose_updates_without_full_body_tracking(self) -> None:
        sdk = FakeSdk()
        sdk.body_available = False
        teleop = PicoTeleop(sdk, start_service=False)

        self.assertIsNone(teleop.read())

        assert teleop.headset_pose is not None
        np.testing.assert_allclose(teleop.headset_pose[:3], sdk.headset[:3])
        np.testing.assert_allclose(teleop.headset_pose[3:], (0.0, 0.0, 0.0, 1.0))
        np.testing.assert_allclose(sdk.headset[3:], (0.0, 0.0, 0.0, 2.0))

    def test_pose_conversion_is_finite_and_has_sonic_shapes(self) -> None:
        frame = PicoPoseConverter().convert(identity_body())

        self.assertEqual(frame.pose.shape, (21, 3))
        self.assertEqual(frame.joints.shape, (24, 3))
        self.assertEqual(frame.root_quaternion.shape, (4,))
        self.assertTrue(np.isfinite(frame.joints).all())
        self.assertAlmostEqual(np.linalg.norm(frame.root_quaternion), 1.0)

    def test_direct_reader_builds_a_five_frame_command(self) -> None:
        sdk = FakeSdk()
        teleop = PicoTeleop(sdk, start_service=False)

        command = None
        for _ in range(6):
            command = teleop.read()

        self.assertIsNotNone(command)
        assert command is not None
        self.assertEqual(command.smpl_joints.shape, (5, 24, 3))
        self.assertEqual(command.joint_position.shape, (5, 29))
        self.assertEqual(command.hand_joint_position.shape, (5, 40))
        self.assertEqual(command.neck_joint_position.shape, (5, 2))
        np.testing.assert_array_equal(command.neck_joint_position, 0.0)
        self.assertAlmostEqual(command.heading_increment, -0.015)
        teleop.close()
        self.assertTrue(sdk.closed)

    def test_button_combinations_are_edge_triggered(self) -> None:
        sdk = FakeSdk()
        teleop = PicoTeleop(sdk, start_service=False)

        sdk.buttons.update(A=True, B=True, X=True, Y=True)
        teleop.read()
        events = teleop.pop_events()
        self.assertTrue(events.start_stop)
        self.assertFalse(events.toggle_pose)

        teleop.read()
        self.assertFalse(teleop.pop_events().start_stop)
        sdk.buttons.update(B=False, Y=False)
        teleop.read()
        self.assertFalse(teleop.pop_events().toggle_pose)
        sdk.buttons.update(A=False, X=False)
        teleop.read()
        sdk.buttons.update(A=True, X=True)
        teleop.read()
        self.assertTrue(teleop.pop_events().toggle_pose)

        sdk.buttons.update(A=False, X=False)
        teleop.read()
        sdk.left_grip = 1.0
        sdk.buttons["A"] = True
        teleop.read()
        self.assertTrue(teleop.pop_events().toggle_recording)

        sdk.buttons["A"] = False
        teleop.read()
        sdk.buttons["B"] = True
        teleop.read()
        self.assertTrue(teleop.pop_events().abort_recording)

        sdk.buttons["B"] = False
        teleop.read()
        sdk.buttons["X"] = True
        teleop.read()
        events = teleop.pop_events()
        self.assertTrue(events.reset_scene)
        self.assertFalse(events.toggle_pose)

        teleop.read()
        self.assertFalse(teleop.pop_events().reset_scene)

    def test_optical_hand_tracking_overrides_controller_fallback(self) -> None:
        sdk = FakeSdk()
        sdk.left_hand_active = True
        teleop = PicoTeleop(sdk, start_service=False)

        command = None
        for _ in range(6):
            command = teleop.read()

        assert command is not None and command.hand_joint_position is not None
        expected = DexHandRetargeter(low_pass_alpha=1.0).retarget_hand(
            sdk.left_hand, "left"
        )
        np.testing.assert_allclose(command.hand_joint_position[-1, :20], expected)
        np.testing.assert_array_equal(command.hand_joint_position[-1, 20:], 0.0)

    def test_dexhand_can_be_disabled(self) -> None:
        teleop = PicoTeleop(FakeSdk(), start_service=False, enable_dexhand=False)

        command = None
        for _ in range(6):
            command = teleop.read()

        assert command is not None
        self.assertIsNone(command.hand_joint_position)

    def test_haptic_command_uses_xrobotoolkit_device_control(self) -> None:
        sdk = FakeSdk()
        teleop = PicoTeleop(sdk, start_service=False)

        sent = teleop.send_haptics("pico-sn", 0.2, 0.7, 60, 150)

        self.assertTrue(sent)
        self.assertEqual(sdk.device_commands[0][0], "pico-sn")
        command = sdk.device_commands[0][1]
        self.assertEqual(command["functionName"], "HapticImpulse")
        value = json.loads(command["value"])
        self.assertEqual(value["right"], 0.7)
        teleop.close()


if __name__ == "__main__":
    unittest.main()
