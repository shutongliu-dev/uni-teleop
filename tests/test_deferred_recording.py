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

    def test_independent_episodes_skip_empty_and_seal_before_next(self):
        from uni_teleop.recording.deferred import EpisodeArchives
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            for name in ('scene.json', 'initial_state.json'):
                (output/name).write_text('{}')
            archives = EpisodeArchives(output, {'schema': 'test', 'cameras': ['left']}, {'seed': 3})
            state = {'joint_q': np.zeros(3)}
            archives.append(state, {'episode': 0, 'action': None})
            self.assertFalse((output/'episodes').exists())
            for episode in (1, 2):
                archives.append(state, {'episode': episode, 'action': {'target': episode}})
                archives.append(state, {'episode': episode, 'action': None, 'event': 'terminal'})
                root = output/'episodes'/f'{episode:06d}'
                receipt = json.loads((root/'session.json').read_text())
                self.assertEqual(receipt['deferred_recording']['status'], 'saved')
                self.assertEqual(receipt['deferred_recording']['frames'], 2)
                self.assertEqual(receipt['settings']['seed'], 3)
                self.assertEqual((root/'scene.json').read_text(), '{}')
                rows = [json.loads(x) for x in (root/'replay/frames.jsonl').read_text().splitlines()]
                self.assertEqual([r['frame'] for r in rows], [0, 1])
                self.assertEqual({r['episode'] for r in rows}, {episode})
            archives.close()
            self.assertEqual(archives.count, 4)
            self.assertEqual(len(json.loads((output/'episodes.json').read_text())), 2)
            self.assertFalse((output/'replay_images').exists())
