from types import SimpleNamespace
import unittest
import numpy as np
from scripts.teleop.pico_isaac import update_control_view


class ControlViewStartTests(unittest.TestCase):
    def test_start_alignment_does_not_wait_for_video_frame(self):
        view = SimpleNamespace(reference=None, control_base=np.eye(4))
        def update(head, poses):
            if head is not None and view.reference is None:
                view.reference = head
        view.update = update
        mapper = SimpleNamespace(armed=True, anchors={'right': object()}, rotation=None)
        frame = SimpleNamespace(tracking={'head': [0, 0, 0, 0, 0, 0, 1]})
        update_control_view(view, mapper, frame, {})
        self.assertIsNotNone(view.reference)
        self.assertTrue(mapper.armed)
        self.assertEqual(mapper.anchors, {})
        mapper.anchors['right'] = 'new anchor'
        update_control_view(view, mapper, frame, {})
        self.assertEqual(mapper.anchors['right'], 'new anchor')

    def test_missing_head_does_not_falsely_mark_alignment_ready(self):
        view = SimpleNamespace(reference=None, update=lambda *_: None)
        mapper = SimpleNamespace(anchors={'right': 'anchor'})
        update_control_view(view, mapper, None, {})
        self.assertIsNone(view.reference)
        self.assertEqual(mapper.anchors['right'], 'anchor')
