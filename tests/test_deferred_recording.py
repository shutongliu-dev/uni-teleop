import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from uni_teleop.recording.deferred import StateArchive


class DeferredRecordingTests(unittest.TestCase):
    def test_archive_keeps_every_frame_and_copies_before_async_write(self):
        with tempfile.TemporaryDirectory() as directory:
            archive = StateArchive(Path(directory), {'schema': 'test'})
            state = np.zeros(3)
            for i in range(20):
                state[:] = i
                archive.append({'joint_q': state}, {'sim_time_s': .04*i, 'action': {'target': i}})
            state[:] = -1
            archive.close()
            root = Path(directory)/'replay'
            rows = [json.loads(line) for line in (root/'frames.jsonl').read_text().splitlines()]
            self.assertEqual(len(rows), 20)
            for i, row in enumerate(rows):
                self.assertEqual(row['frame'], i)
                self.assertEqual(row['action']['target'], i)
                with np.load(root/row['file'], allow_pickle=False) as data:
                    np.testing.assert_array_equal(data['joint_q'], [i]*3)
            self.assertEqual(list(root.glob('*.partial')), [])
