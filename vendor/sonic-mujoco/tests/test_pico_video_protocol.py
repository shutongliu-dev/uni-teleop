import runpy
import struct
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

BRIDGE = Path(__file__).parents[1] / "scripts/pico_video_bridge.py"
PROTOCOL = runpy.run_path(str(BRIDGE))
FRAME_HEADER = PROTOCOL["FRAME_HEADER"]
VIDEO_BRIDGE = PROTOCOL["VideoBridge"]


class _Buffer:
    @classmethod
    def new_allocate(
        cls, _memory: object, size: int, _allocator: object
    ) -> "_Buffer":
        return cls(size)

    def __init__(self, size: int) -> None:
        self.data = bytes(size)

    def fill(self, _offset: int, data: bytes) -> None:
        self.data = bytes(data)


class _Gst:
    Buffer = _Buffer


class PicoVideoProtocolTest(unittest.TestCase):
    def test_decodes_xrrobotkit_open_camera_packet(self) -> None:
        camera = b"head_camera"
        ip = b"192.168.3.10"
        data = (
            b"\xca\xfe\x01"
            + struct.pack("<7i", 1280, 720, 30, 8_000_000, 0, 0, 23456)
            + bytes([len(camera)])
            + camera
            + bytes([len(ip)])
            + ip
        )
        command = b"OPEN_CAMERA\0"
        packet = (
            struct.pack("<i", len(command))
            + command
            + struct.pack("<i", len(data))
            + data
        )

        name, payload = PROTOCOL["parse_command"](packet)
        config = PROTOCOL["parse_camera_config"](payload)

        self.assertEqual(name, "OPEN_CAMERA")
        self.assertEqual((config.width, config.height, config.fps), (1280, 720, 30))
        self.assertEqual((config.ip, config.port), ("192.168.3.10", 23456))
        self.assertFalse(config.hevc)

    def test_rejects_truncated_command(self) -> None:
        with self.assertRaisesRegex(ValueError, "too short"):
            PROTOCOL["parse_command"](b"OPEN")

    def test_repeats_last_frame_to_keep_stream_alive(self) -> None:
        payload = b"same-frame"
        bridge = object.__new__(VIDEO_BRIDGE)
        bridge._frames = FRAME_HEADER.pack(b"SMVF", 2, 1, 1, len(payload)) + payload
        bridge._pipeline = Mock()
        bridge.Gst = _Gst

        self.assertTrue(bridge._push_frame())
        self.assertTrue(bridge._push_frame())

        source = bridge._pipeline.get_by_name.return_value
        self.assertEqual(source.emit.call_count, 2)
        for call in source.emit.call_args_list:
            self.assertEqual(call.args[0], "push-buffer")
            self.assertEqual(call.args[1].data, payload)

    def test_stream_has_no_timeout_after_connecting(self) -> None:
        connection = Mock()
        with patch.object(
            PROTOCOL["socket"], "create_connection", return_value=connection
        ) as create_connection:
            result = PROTOCOL["connect_video"]("192.168.3.87", 12345)

        self.assertIs(result, connection)
        create_connection.assert_called_once_with(("192.168.3.87", 12345), timeout=3)
        connection.settimeout.assert_called_once_with(None)


if __name__ == "__main__":
    unittest.main()
