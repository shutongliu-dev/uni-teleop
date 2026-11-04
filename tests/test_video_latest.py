from types import SimpleNamespace
import unittest
from scripts.teleop.pico_video import VideoBridge, FRAME_HEADER


class VideoLatestTests(unittest.TestCase):
    def test_only_new_complete_frames_enter_encoder(self):
        bridge = VideoBridge.__new__(VideoBridge)
        pushed = []
        source = SimpleNamespace(emit=lambda signal, data: pushed.append(data))
        bridge._pipeline = SimpleNamespace(get_by_name=lambda _: source)
        bridge._last_sequence = None
        bridge.Gst = SimpleNamespace(Buffer=SimpleNamespace(new_allocate=lambda *args:
            SimpleNamespace(fill=lambda *args: None)))
        bridge._frames = bytearray(FRAME_HEADER.pack(b'SMVF', 2, 2, 1, 6)+bytes(6))
        for _ in range(5):
            self.assertTrue(bridge._push_frame())
        self.assertEqual(len(pushed), 1)
        bridge._frames[:FRAME_HEADER.size] = FRAME_HEADER.pack(b'SMVF', 3, 2, 1, 6)
        bridge._push_frame()
        self.assertEqual(len(pushed), 1)
        bridge._frames[:FRAME_HEADER.size] = FRAME_HEADER.pack(b'SMVF', 4, 2, 1, 6)
        bridge._push_frame()
        self.assertEqual(len(pushed), 2)

class VideoRecoveryTests(unittest.TestCase):
    def bridge(self):
        bridge = VideoBridge.__new__(VideoBridge)
        self.scheduled = []
        self.cancelled = []
        bridge.GLib = SimpleNamespace(timeout_add=lambda *args: self.scheduled.append(args) or 7,
                                     source_remove=lambda timer: self.cancelled.append(timer))
        bridge._pipeline = object()
        bridge._requested_config = object()
        bridge._retry_timer = None
        bridge._stop_stream = lambda: setattr(bridge, '_pipeline', None)
        return bridge

    def test_failed_stream_reconnects_but_explicit_close_cancels_retry(self):
        bridge = self.bridge()
        failed = bridge._pipeline
        bridge._reconnect(failed)
        self.assertEqual(len(self.scheduled), 1)
        self.assertIsNone(bridge._pipeline)
        bridge._reconnect(failed)
        self.assertEqual(len(self.scheduled), 1)
        bridge._close()
        self.assertEqual(self.cancelled, [7])
        self.assertIsNone(bridge._requested_config)
        bridge._open = lambda _: self.fail('Explicitly closed stream reopened')
        bridge._retry()

    def test_stale_stream_failure_does_not_close_new_stream(self):
        bridge = self.bridge()
        current = bridge._pipeline
        bridge._reconnect(object())
        self.assertIs(bridge._pipeline, current)
        self.assertEqual(self.scheduled, [])
