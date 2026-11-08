from copy import deepcopy
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from scipy.spatial.transform import Rotation

from uni_teleop.video.isaac import StereoView
from uni_teleop.video.isaac_camera import rear_camera_pose, table_closeup_pose
from uni_teleop.mapping.franka.retarget import view_alignment_rotation, relative_pose_target


class OperatorViewTests(unittest.TestCase):
    def test_controller_axes_and_rotations_match_oblique_headset_view(self):
        camera = np.eye(4)
        camera[:3, :3] = Rotation.from_euler('xyz', [55, -12, 145], degrees=True).as_matrix()
        head = Rotation.from_euler('xyz', [-15, 70, 8], degrees=True)
        basis = view_alignment_rotation(camera, [0, 0, 0, *head.as_quat()])
        human_zero, robot_zero = np.eye(4), np.eye(4)
        human_zero[:3, :3] = Rotation.from_euler('xyz', [35, 10, -70], degrees=True).as_matrix()
        robot_zero[:3, :3] = Rotation.from_euler('xyz', [90, 15, 20], degrees=True).as_matrix()
        np.testing.assert_allclose(relative_pose_target(human_zero, human_zero, robot_zero, basis), robot_zero, atol=1e-12)
        for axis in ([1, 0, 0], [0, 1, 0], [0, 0, -1]):
            with self.subTest(axis=axis):
                axis = np.asarray(axis)
                human = human_zero.copy()
                human[:3, 3] += head.apply(axis)*.1
                human[:3, :3] = Rotation.from_rotvec(head.apply(axis)*.2).as_matrix() @ human_zero[:3, :3]
                target = relative_pose_target(human, human_zero, robot_zero, basis)
                np.testing.assert_allclose(camera[:3, :3].T @ target[:3, 3], axis*.1, atol=1e-12)
                delta = target[:3, :3] @ robot_zero[:3, :3].T
                np.testing.assert_allclose(camera[:3, :3].T @ Rotation.from_matrix(delta).as_rotvec(), axis*.2, atol=1e-12)

    def test_headset_view_does_not_replace_recorded_composite(self):
        view = StereoView.__new__(StereoView)
        view.wrist_display_rotation = SimpleNamespace(render=lambda pixels, *args: np.rot90(pixels, k=-1))
        view.inset_op = SimpleNamespace(Get=lambda: np.eye(4))
        view.operator_base = np.eye(4)
        view.arm_visibility = None
        original_left, original_right, inset = object(), object(), object()
        rear_left, rear_right = object(), object()
        frames = dict(pico_left=original_left, pico_right=original_right,
                      pico_inset=inset, pico_operator_left=rear_left,
                      pico_operator_right=rear_right)
        view.env = SimpleNamespace(render_many=lambda names: {n: frames[n] for n in names})
        view.inset, view.inset_label = True, 'WRIST'
        view.rear_op = None
        view.operator_ops = {'left': object(), 'right': object()}
        view.operator_inset_op = None
        view.operator_wrist = False
        with patch('uni_teleop.video.isaac.picture_in_picture', side_effect=lambda eye, small, label: (eye, small)):
            recorded = view.render()
        self.assertEqual(recorded, ((original_left, inset), (original_right, inset)))
        self.assertEqual(view.operator_eyes, (rear_left, rear_right))
        closeup = object()
        frames['pico_operator_table'] = closeup
        view.operator_inset_op = object()
        with patch('uni_teleop.video.isaac.picture_in_picture', side_effect=lambda eye, small, label: (eye, small)), \
                patch('uni_teleop.video.isaac.operator_panels', side_effect=lambda eye, small, label: (eye, small)):
            recorded_with_closeup = view.render()
        self.assertEqual(recorded_with_closeup, recorded)
        self.assertEqual(view.operator_eyes, ((rear_left, closeup), (rear_right, closeup)))
        with patch('uni_teleop.video.isaac.operator_panels', side_effect=lambda eye, small, label: (eye, small)):
            self.assertEqual(view.render(include_recording=False), (None, None))
        self.assertEqual(set(view.frames), {'pico_operator_left', 'pico_operator_right', 'pico_operator_table'})
        self.assertEqual(view.operator_eyes, ((rear_left, closeup), (rear_right, closeup)))

    def test_rear_pose_is_behind_base_and_does_not_edit_scene(self):
        scene = {'robot': {'base_position_m': [.65, 0., .78]},
                 'table': {'position_m': [.45, -.95, .8775],
                           'half_extents_m': [1.4, .4, .0225]}}
        before = deepcopy(scene)
        eye, target = rear_camera_pose(scene, lateral_m=0.)
        base = np.array(scene['robot']['base_position_m'])
        self.assertLess(np.dot(eye[:2]-base[:2], target[:2]-base[:2]), 0.)
        self.assertGreater(eye[2], target[2])
        self.assertEqual(scene, before)
        near_eye, near_target = table_closeup_pose(scene)
        far_eye, far_target = rear_camera_pose(scene, lateral_m=.65, height_m=1.40)
        np.testing.assert_allclose((near_target-near_eye)[:2]/np.linalg.norm((near_target-near_eye)[:2]),
                                   (far_target-far_eye)[:2]/np.linalg.norm((far_target-far_eye)[:2]))
        self.assertAlmostEqual(near_eye[2]-(.8775+.0225), .26)
        self.assertLess(abs(near_target[2]-near_eye[2])/np.linalg.norm((near_target-near_eye)[:2]), .25)
        self.assertEqual(scene, before)

    def test_dual_detail_panels_preserve_main_and_recording_channels(self):
        from uni_teleop.video.isaac import operator_panels
        main = np.full((480, 640, 3), [255, 0, 0], dtype=np.uint8)
        front = np.full((240, 320, 3), [0, 255, 0], dtype=np.uint8)
        side = np.full((240, 320, 3), [0, 0, 255], dtype=np.uint8)
        result = operator_panels(main, front, 'TABLE DETAIL', second=side)
        self.assertEqual(result.shape, main.shape)
        np.testing.assert_array_equal(result[100, 320], [255, 0, 0])
        np.testing.assert_array_equal(result[400, 160], [0, 255, 0])
        np.testing.assert_array_equal(result[400, 480], [0, 0, 255])
        view = StereoView.__new__(StereoView)
        view.wrist_display_rotation = SimpleNamespace(render=lambda pixels, *args: np.rot90(pixels, k=-1))
        view.inset_op = SimpleNamespace(Get=lambda: np.eye(4))
        view.operator_base = np.eye(4)
        view.arm_visibility = None
        view.inset = False
        view.rear_op = None
        view.operator_ops = {'left': object(), 'right': object()}
        view.operator_inset_op = object()
        view.operator_wrist = True
        frames = dict(pico_operator_left=main, pico_operator_right=main,
                      pico_operator_table=front, pico_inset=side)
        view.env = SimpleNamespace(render_many=lambda names: {n: frames[n] for n in names})
        self.assertEqual(view.render(include_recording=False), (None, None))
        self.assertEqual(set(view.frames), set(frames))
        np.testing.assert_array_equal(view.operator_eyes[0],
                                      operator_panels(main, front, 'TABLE DETAIL', second=np.rot90(side, k=-1)))
        np.testing.assert_array_equal(view.frames['pico_inset'], side)

        view.operator_inset_op = None
        self.assertEqual(view.render(include_recording=False), (None, None))
        self.assertNotIn('pico_operator_table', view.frames)
        np.testing.assert_array_equal(view.operator_eyes[0],
                                      operator_panels(main, np.rot90(side, k=-1), 'WRIST'))

    def test_hidden_arm_pass_is_isolated_and_restored_on_render_failure(self):
        from contextlib import contextmanager
        view = StereoView.__new__(StereoView)
        view.wrist_display_rotation = SimpleNamespace(render=lambda pixels, *args: np.rot90(pixels, k=-1))
        view.inset_op = SimpleNamespace(Get=lambda: np.eye(4))
        view.operator_base = np.eye(4)
        view.inset = False
        view.rear_op = view.operator_inset_op = None
        view.operator_wrist = False
        view.operator_ops = {'left': object(), 'right': object()}
        hidden = False
        calls = []
        @contextmanager
        def hide():
            nonlocal hidden
            hidden = True
            try:
                yield
            finally:
                hidden = False
        view.arm_visibility = SimpleNamespace(hidden=hide)
        def render(names):
            calls.append((tuple(names), hidden))
            if hidden:
                raise RuntimeError('render failed')
            return {name: object() for name in names}
        view.env = SimpleNamespace(render_many=render)
        with self.assertRaisesRegex(RuntimeError, 'render failed'):
            view.render(include_recording=True)
        self.assertFalse(hidden)
        self.assertEqual(calls, [(('pico_left', 'pico_right'), False),
                                 (('pico_operator_left', 'pico_operator_right'), True)])

    def test_hidden_wrist_shares_preview_pass_but_preserves_recording(self):
        from contextlib import contextmanager
        view = StereoView.__new__(StereoView)
        view.wrist_display_rotation = SimpleNamespace(render=lambda pixels, *args: np.rot90(pixels, k=-1))
        view.inset_op = SimpleNamespace(Get=lambda: np.eye(4))
        view.operator_base = np.eye(4)
        view.inset, view.inset_label = True, 'WRIST'
        view.rear_op = view.operator_inset_op = None
        view.operator_wrist = True
        view.operator_ops = {'left': object(), 'right': object()}
        hidden = False
        calls = []
        @contextmanager
        def hide():
            nonlocal hidden
            hidden = True
            try:
                yield
            finally:
                hidden = False
        view.arm_visibility = SimpleNamespace(hidden=hide)
        wrist_buffer = np.zeros((4, 6, 3), dtype=np.uint8)
        def render(names):
            calls.append((tuple(names), hidden))
            wrist_buffer[:] = 20 if hidden else 100
            return {name: wrist_buffer if name == 'pico_inset' else np.zeros((4, 6, 3), dtype=np.uint8)
                    for name in names}
        view.env = SimpleNamespace(render_many=render)
        with patch('uni_teleop.video.isaac.operator_panels', side_effect=lambda main, wrist, label: wrist), \
                patch('uni_teleop.video.isaac.picture_in_picture', side_effect=lambda main, wrist, label: wrist):
            self.assertEqual(view.render(include_recording=False), (None, None))
            self.assertEqual(calls, [(('pico_operator_left', 'pico_operator_right', 'pico_inset'), True)])
            self.assertTrue((view.operator_eyes[0] == 20).all())
            calls.clear()
            recorded = view.render(include_recording=True)
            self.assertEqual(len(calls), 2)
            self.assertFalse(calls[0][1])
            self.assertTrue(calls[1][1])
            self.assertTrue((recorded[0] == 100).all())
            self.assertTrue((view.operator_eyes[0] == 20).all())
            self.assertFalse(hidden)


    def test_operator_main_is_forward_of_base_without_crossing_workspace(self):
        from uni_teleop.video.isaac_camera import operator_rear_pose
        for distance in (.2, 1., 2.):
            scene = {'robot': {'base_position_m': [0., 0., .78]},
                     'table': {'position_m': [distance, 0., .8], 'half_extents_m': [.5, .5, .02]}}
            before = deepcopy(scene)
            eye, target = operator_rear_pose(scene)
            self.assertGreater(eye[0], 0.)
            self.assertLess(eye[0], target[0])
            self.assertGreater(eye[2], target[2])
            self.assertEqual(scene, before)
