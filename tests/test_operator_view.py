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
        self.assertAlmostEqual(near_eye[2]-(.8775+.0225), .16)
        self.assertLess(abs(near_target[2]-near_eye[2])/np.linalg.norm((near_target-near_eye)[:2]), .1)
        self.assertEqual(scene, before)
