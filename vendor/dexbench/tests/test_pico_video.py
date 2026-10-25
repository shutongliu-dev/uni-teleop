"""Actual Unity video request framing, independent of Isaac and GStreamer."""
import struct
import pytest
from scripts.teleop.pico_video import parse_command, parse_camera_config


def test_unity_camera_open_packet():
    strings = bytes([4])+b'pico'+bytes([9])+b'127.0.0.1'
    data = b'\xca\xfe\x01'+struct.pack('<7i', 1280, 480, 30, 5_000_000, 0, 1, 12345)+strings
    command = b'OPEN_CAMERA\0'
    packet = struct.pack('<i',len(command))+command+struct.pack('<i',len(data))+data
    name, decoded = parse_command(packet)
    assert name == 'OPEN_CAMERA'
    config = parse_camera_config(decoded)
    assert (config.width, config.height, config.ip, config.port) == (1280,480,'127.0.0.1',12345)


@pytest.mark.parametrize('packet', [b'', b'\0'*8, struct.pack('<i', -1)+b'\0'*4])
def test_invalid_video_command_is_rejected(packet):
    with pytest.raises(ValueError):
        name, data = parse_command(packet)
        parse_camera_config(data)


def test_control_marker_projection_stereo_and_hidden_points():
    import numpy as np
    from bench.adapters.pico.video import control_point_overlay
    frame = np.zeros((100, 200, 3), dtype=np.uint8)
    camera = np.eye(4)
    center = control_point_overlay(frame, [[0., 0., -1.]], camera, 1.)
    assert center[50, 100].tolist() == [255, 220, 60]
    assert not frame.any()
    assert not control_point_overlay(frame, [[0, 0, 1], [5, 0, -1]], camera, 1.).any()
    camera[0, 3] = .1
    shifted = control_point_overlay(frame, [[0., 0., -1.]], camera, 1.)
    assert shifted[50, 80].tolist() == [255, 220, 60]
    assert not shifted[50, 100].any()
