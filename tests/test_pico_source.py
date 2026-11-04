import os
from pathlib import Path
import select
import signal
import subprocess
import sys
import unittest


class PicoSourceShutdownTests(unittest.TestCase):
    def test_sigterm_releases_sdk_without_tracking_frames(self):
        # Exercise a real signal while the reader is waiting for its first
        # frame; no native SDK or device is needed for this lifecycle test.
        script = '''
import queue
import sys
from uni_teleop.devices.pico import source
class FakeSDK:
    def __init__(self, library):
        self.pending = queue.Queue()
        self.dropped = 0
    def start(self):
        pass
    def close(self):
        print("SDK_CLOSED", flush=True)
source.SDK = FakeSDK
sys.argv = ["source", "--library", "unused"]
source.main()
'''
        process = subprocess.Popen(
            [sys.executable, '-u', '-c', script],
            cwd=Path(__file__).resolve().parents[1],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )
        try:
            readable, _, _ = select.select([process.stdout], [], [], 10)
            self.assertTrue(readable, 'reader did not start')
            self.assertIn('PICO bridge ready', process.stdout.readline())
            os.kill(process.pid, signal.SIGTERM)
            output, errors = process.communicate(timeout=5)
            self.assertEqual(process.returncode, 0, errors)
            self.assertEqual(output.count('SDK_CLOSED'), 1)
            self.assertIn('"frames": 0', output)
        finally:
            if process.poll() is None:
                process.kill()
            process.communicate()
