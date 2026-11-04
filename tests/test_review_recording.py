import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from concurrent.futures import Future

import numpy as np
from scripts.teleop.review import Recorder


class ReviewRecordingTests(unittest.TestCase):
    def test_rate_backpressure_timestamps_and_no_catchup_duplicates(self):
        with tempfile.TemporaryDirectory() as directory:
            recorder = Recorder(Path(directory))
            pixels = np.zeros((8, 8, 3), dtype=np.uint8)
            try:
                self.assertEqual(recorder.image_fps, 15)
                with patch('scripts.teleop.review.time.monotonic', return_value=10.):
                    recorder.image(pixels, {'sim_time_s': .02}, 0, captured_ns=10_000_000_000)
                recorder.pending.result()
                self.assertFalse(recorder.image_due(10.03))
                self.assertTrue(recorder.image_due(10.07))
                recorder.pending = Future()
                self.assertFalse(recorder.image_due(11.))
                recorder.pending.set_result(None)
                with patch('scripts.teleop.review.time.monotonic', return_value=11.):
                    recorder.image(pixels, {'sim_time_s': .04}, 0, captured_ns=11_000_000_000)
                recorder.pending.result()
                self.assertFalse(recorder.image_due(11.01))
                self.assertEqual(recorder.image_stats()['actual_fps'], 1.)
                self.assertEqual(recorder.count, 2)
            finally:
                recorder.close()
            rows = [json.loads(line) for line in (Path(directory)/'review_images.jsonl').read_text().splitlines()]
            self.assertEqual([row['host_ns'] for row in rows], [10_000_000_000, 11_000_000_000])
            self.assertTrue(all((Path(directory)/row['file']).is_file() for row in rows))

    def test_invalid_rates_rejected_before_opening_output(self):
        for rate in (0, -1, float('nan'), float('inf'), 51):
            with self.subTest(rate=rate), self.assertRaises(ValueError):
                Recorder(Path('/unused'), image_fps=rate)
