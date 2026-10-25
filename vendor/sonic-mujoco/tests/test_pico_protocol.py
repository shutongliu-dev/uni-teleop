import json
import unittest

import numpy as np

from sonic_mujoco.teleop import TeleopCommand, decode_pose_message
from sonic_mujoco.teleop.pico import HEADER_SIZE


def pack_pose(fields: dict[str, np.ndarray], version: int = 3) -> bytes:
    dtype_names = {
        np.dtype("float32"): "f32",
        np.dtype("float64"): "f64",
        np.dtype("int32"): "i32",
        np.dtype("int64"): "i64",
        np.dtype("bool"): "bool",
    }
    descriptions = [
        {"name": name, "dtype": dtype_names[value.dtype], "shape": list(value.shape)}
        for name, value in fields.items()
    ]
    header = json.dumps(
        {"v": version, "endian": "le", "count": 1, "fields": descriptions},
        separators=(",", ":"),
    ).encode()
    payload = b"".join(value.tobytes() for value in fields.values())
    return b"pose" + header.ljust(HEADER_SIZE, b"\0") + payload


def pose_fields(frames: int = 5) -> dict[str, np.ndarray]:
    return {
        "smpl_pose": np.zeros((frames, 21, 3), dtype=np.float32),
        "smpl_joints": np.arange(frames * 72, dtype=np.float32).reshape(frames, 24, 3),
        "body_quat_w": np.tile([1.0, 0.0, 0.0, 0.0], (frames, 1)).astype(np.float32),
        "joint_pos": np.arange(frames * 29, dtype=np.float64).reshape(frames, 29),
        "joint_vel": np.zeros((frames, 29), dtype=np.float64),
        "frame_index": np.arange(10, 10 + frames, dtype=np.int64),
        "left_trigger": np.array([0.5], dtype=np.float32),
        "heading_increment": np.array([0.1], dtype=np.float32),
    }


class PicoProtocolTest(unittest.TestCase):
    def test_decodes_existing_protocol_v3_layout(self) -> None:
        fields = pose_fields()
        command = decode_pose_message(pack_pose(fields))

        np.testing.assert_array_equal(command.frame_index, fields["frame_index"])
        np.testing.assert_allclose(command.smpl_joints, fields["smpl_joints"])
        np.testing.assert_allclose(command.root_quaternion, fields["body_quat_w"])
        np.testing.assert_allclose(command.joint_position, fields["joint_pos"])
        np.testing.assert_array_equal(command.neck_joint_position, 0.0)
        self.assertAlmostEqual(command.heading_increment, 0.1, places=6)

    def test_decodes_explicit_neck_targets(self) -> None:
        fields = pose_fields()
        fields["neck_joint_pos"] = np.linspace(-0.2, 0.3, 10).reshape(5, 2)

        command = decode_pose_message(pack_pose(fields))

        np.testing.assert_allclose(
            command.neck_joint_position, fields["neck_joint_pos"]
        )

    def test_decodes_optional_hand_targets(self) -> None:
        fields = pose_fields()
        fields["hand_joint_pos"] = np.arange(200, dtype=np.float32).reshape(5, 40)

        command = decode_pose_message(pack_pose(fields))

        np.testing.assert_allclose(
            command.hand_joint_position, fields["hand_joint_pos"]
        )

    def test_rejects_wrong_protocol_version(self) -> None:
        with self.assertRaisesRegex(ValueError, "protocol v3"):
            decode_pose_message(pack_pose(pose_fields(), version=4))

    def test_command_rejects_non_monotonic_frames(self) -> None:
        fields = pose_fields(frames=2)
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            TeleopCommand(
                frame_index=np.array([2, 2]),
                smpl_joints=fields["smpl_joints"],
                root_quaternion=fields["body_quat_w"],
                joint_position=fields["joint_pos"],
            )


if __name__ == "__main__":
    unittest.main()
