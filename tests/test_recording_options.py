import unittest
from uni_teleop.runtime.arguments import parser
from uni_teleop.recording.options import configure_recording


class RecordingOptionsTests(unittest.TestCase):
    def test_test_mode_overrides_recipe_regardless_of_argument_order(self):
        for options in (
            ['--deferred-recording', '--profile', '--no-recording'],
            ['--no-recording', '--profile', '--deferred-recording'],
        ):
            args = parser().parse_args(options + ['--family', 'study', '--task', 'stationery_sorting',
                                                '--operator-view', 'rear', '--video-fps', '20'])
            self.assertFalse(configure_recording(args))
            self.assertFalse(args.deferred_recording)
            self.assertFalse(args.profile)
            self.assertEqual(args.family, 'study')
            self.assertEqual(args.task, 'stationery_sorting')
            self.assertEqual(args.video_fps, 20)
            self.assertEqual(args.operator_view, 'rear')

    def test_recording_defaults_remain_enabled(self):
        args = parser().parse_args(['--deferred-recording', '--profile'])
        self.assertTrue(configure_recording(args))
        self.assertTrue(args.deferred_recording)
        self.assertTrue(args.profile)

    def test_review_campaign_conflict_is_explicit(self):
        args = parser().parse_args(['--no-recording', '--review-dir', '/unused'])
        with self.assertRaisesRegex(ValueError, '--review-dir'):
            configure_recording(args)
