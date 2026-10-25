import json
from math import prod

import numpy as np

from .base import TeleopBase, TeleopCommand
from .dexhand import DexHandRetargeter
from .neck import neck_joint_targets

HEADER_SIZE = 1280
TOPIC = b"pose"
DTYPES = {
    "f32": np.dtype("<f4"),
    "f64": np.dtype("<f8"),
    "i32": np.dtype("<i4"),
    "i64": np.dtype("<i8"),
    "bool": np.dtype("?"),
    "u8": np.dtype("u1"),
}


def decode_pose_message(message: bytes) -> TeleopCommand:
    """Decode the protocol-v3 packet produced by the existing PICO manager."""

    if not message.startswith(TOPIC) or len(message) < len(TOPIC) + HEADER_SIZE:
        raise ValueError("not a packed pose message")

    header_start = len(TOPIC)
    header_end = header_start + HEADER_SIZE
    try:
        header = json.loads(message[header_start:header_end].rstrip(b"\0"))
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise ValueError("invalid pose message header") from error

    if header.get("v") != 3 or header.get("endian") != "le":
        raise ValueError("PICO pose messages must use little-endian protocol v3")

    payload = memoryview(message)[header_end:]
    offset = 0
    values: dict[str, np.ndarray] = {}
    for field in header.get("fields", []):
        try:
            name = field["name"]
            dtype = DTYPES[field["dtype"]]
            shape = tuple(int(size) for size in field["shape"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("invalid pose field description") from error
        if any(size < 0 for size in shape):
            raise ValueError("pose field dimensions must be non-negative")
        size = prod(shape) * dtype.itemsize
        if offset + size > len(payload):
            raise ValueError(f"pose field {name!r} exceeds the payload")
        values[name] = np.frombuffer(
            payload[offset : offset + size], dtype=dtype, count=prod(shape)
        ).reshape(shape)
        offset += size
    if offset != len(payload):
        raise ValueError("pose message has trailing payload bytes")

    required = {
        "smpl_pose",
        "smpl_joints",
        "body_quat_w",
        "joint_pos",
        "joint_vel",
        "frame_index",
    }
    missing = required.difference(values)
    if missing:
        raise ValueError(f"pose message is missing: {', '.join(sorted(missing))}")

    frame_index = values["frame_index"]
    frames = frame_index.size
    if values["smpl_pose"].shape != (frames, 21, 3):
        raise ValueError("smpl_pose must have shape (N, 21, 3)")
    if values["joint_vel"].shape != (frames, 29):
        raise ValueError("joint_vel must have shape (N, 29)")

    root_quaternion = values["body_quat_w"]
    if root_quaternion.shape == (frames, 1, 4):
        root_quaternion = root_quaternion[:, 0]
    heading_increment = values.get("heading_increment", np.zeros(1))
    if heading_increment.shape != (1,):
        raise ValueError("heading_increment must have shape (1,)")
    hand_joint_position = values.get("hand_joint_pos")
    if hand_joint_position is None:
        left_hand = values.get("left_hand_joints")
        right_hand = values.get("right_hand_joints")
        if left_hand is not None or right_hand is not None:
            if left_hand is None or right_hand is None:
                raise ValueError(
                    "left_hand_joints and right_hand_joints must be provided together"
                )
            if left_hand.shape[0] != frames or right_hand.shape[0] != frames:
                raise ValueError("hand tracking batch must match frame_index")
            retargeter = DexHandRetargeter(low_pass_alpha=1.0)
            hand_joint_position = np.stack(
                [
                    retargeter.retarget(left, right)
                    for left, right in zip(left_hand, right_hand)
                ]
            )
    neck_joint_position = values.get("neck_joint_pos")
    if neck_joint_position is None:
        neck_joint_position = neck_joint_targets(values["smpl_pose"])
    return TeleopCommand(
        frame_index=frame_index,
        smpl_joints=values["smpl_joints"],
        root_quaternion=root_quaternion,
        joint_position=values["joint_pos"],
        heading_increment=float(heading_increment[0]),
        hand_joint_position=hand_joint_position,
        neck_joint_position=neck_joint_position,
    )


class PicoZmqTeleop(TeleopBase):
    """Receive PICO poses from an external ZMQ publisher."""

    def __init__(self, endpoint: str = "tcp://127.0.0.1:5556") -> None:
        try:
            import zmq
        except ImportError as error:
            raise RuntimeError(
                "install sonic-mujoco[teleop] to receive PICO messages"
            ) from error

        self._zmq = zmq
        self._socket = zmq.Context.instance().socket(zmq.SUB)
        self._socket.setsockopt(zmq.SUBSCRIBE, TOPIC)
        self._socket.setsockopt(zmq.RCVHWM, 3)
        self._socket.connect(endpoint)

    def read(self) -> TeleopCommand | None:
        try:
            message = self._socket.recv(flags=self._zmq.NOBLOCK)
        except self._zmq.Again:
            return None
        return decode_pose_message(message)

    def close(self) -> None:
        self._socket.close(linger=0)
