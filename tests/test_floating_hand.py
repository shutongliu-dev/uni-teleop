import unittest
from types import SimpleNamespace
from pathlib import Path
from dataclasses import replace

import numpy as np
from scipy.spatial.transform import Rotation

from uni_teleop.mapping.floating_hand import HandController, PalmMapper, optimizer_targets
from uni_teleop.devices.pico.protocol import Frame
from uni_teleop.control.floating_hand import wrist_wrench
from uni_teleop.catalog import recipes, load_profile


class FloatingHandTests(unittest.TestCase):
    def model(self):
        return SimpleNamespace(names=('finger',), lower=np.array([0.]), upper=np.array([1.]), initial=np.array([0.]))

    def frame(self, xyz=(0., 0., 0.), *, stamp=1_000_000_000, stream='one', grip=1.):
        return Frame(stream, 1, stamp, '1', 'test', dict(focused=True,
            controllers={'right': dict(pose=[*xyz, 0., 0., 0., 1.], grip=grip)}))

    def test_relative_translation_rotation_and_no_jump(self):
        mapper = PalmMapper()
        p0, q0 = np.array([.1, .2, 1.]), Rotation.from_euler('x', 90, degrees=True).as_quat()
        first = np.r_[[.3, .4, .5], Rotation.from_euler('z', 60, degrees=True).as_quat()]
        p, q = mapper.map(first, p0, q0)
        np.testing.assert_allclose(p, p0)
        self.assertLess((Rotation.from_quat(q)*Rotation.from_quat(q0).inv()).magnitude(), 1e-12)
        moved = first.copy(); moved[1] += .1
        p, _ = mapper.map(moved, p0, q0)
        np.testing.assert_allclose(p, p0+[0, 0, .1])
        # A 30 degree rotation about SDK +Y becomes world +Z, independent of grip angle.
        moved[3:] = (Rotation.from_euler('y', 30, degrees=True)*Rotation.from_quat(first[3:])).as_quat()
        _, q = mapper.map(moved, p0, q0)
        np.testing.assert_allclose((Rotation.from_quat(q)*Rotation.from_quat(q0).inv()).as_rotvec(),
                                   [0., 0., np.pi/6], atol=1e-12)

    def test_channel_loss_and_clutch_do_not_stop_other_channel(self):
        pose = (np.array([0., 0., 1.]), np.array([0., 0., 0., 1.]))
        c = HandController(self.model(), *pose); c.toggle()
        c.update(self.frame(), {'finger': .2}, 1_000_000_000, pose)
        c.update(self.frame((0, .1, 0)), None, 1_000_000_000, pose)
        np.testing.assert_allclose(c.position, [0., 0., 1.1]); self.assertEqual(c.fingers[0], .2)
        c.update(None, {'finger': .8}, 1_000_000_000, pose)
        np.testing.assert_allclose(c.position, [0., 0., 1.1]); self.assertEqual(c.fingers[0], .8)
        # Returning far away reanchors to the measured hand, without jumping.
        c.update(self.frame((1, 1, 1)), None, 1_000_000_000, pose)
        np.testing.assert_allclose(c.position, pose[0])
        c.update(self.frame((2, 2, 2), grip=0), {'finger': .4}, 1_000_000_000, pose)
        np.testing.assert_allclose(c.position, pose[0]); self.assertEqual(c.fingers[0], .4)
        c.toggle()
        c.update(self.frame(), {'finger': .9}, 1_000_000_000, pose)
        self.assertEqual(c.fingers[0], .4)

    def test_screen_alignment_axes_rotation_and_frozen_heading(self):
        from uni_teleop.mapping.floating_hand import PICO_TO_WORLD
        # Headset faces sideways in its room; desktop camera is also oblique.
        head = Rotation.from_euler('y', 70, degrees=True)
        view = np.array([-.6, .8, -.3])
        right = np.cross(view, [0, 0, 1]); right /= np.linalg.norm(right)
        forward = view.copy(); forward[2] = 0
        pose = (np.array([0., 0., 1.]), np.array([0., 0., 0., 1.]))
        c = HandController(self.model(), *pose, screen_aligned=True); c.toggle()
        def frame(xyz, head_rotation=head):
            f = self.frame(xyz)
            return replace(f, tracking={**f.tracking, 'head': [0, 0, 0, *head_rotation.as_quat()]})
        c.update(frame([0, 0, 0]), None, 1_000_000_000, pose, view)
        np.testing.assert_allclose(c.position, pose[0])
        for source, expected in [([1, 0, 0], right), ([0, 0, -1], forward), ([0, 1, 0], [0, 0, 1])]:
            f = frame(head.apply(source)*.1, Rotation.identity())
            c.update(f, None, 1_000_000_000, pose, -view)
            np.testing.assert_allclose(c.position, pose[0]+.1*np.array(expected), atol=1e-12)
        # Spatial rotation must use the same basis as translation.
        f = frame([0, 0, 0]); axis = head.apply([1, 0, 0])
        f.tracking['controllers']['right']['pose'][3:] = Rotation.from_rotvec(.2*axis).as_quat().tolist()
        c.update(f, None, 1_000_000_000, pose, view)
        np.testing.assert_allclose(Rotation.from_quat(c.quaternion).as_rotvec(), .2*right, atol=1e-12)
        c.recenter()
        c.update(self.frame(), {'finger': .7}, 1_000_000_000, pose, view)
        self.assertTrue(c.heading_pending)
        self.assertEqual(c.fingers[0], .7)
        c.update(frame([1, 2, 3]), None, 1_000_000_000, pose, view)
        np.testing.assert_allclose(c.position, pose[0])

    def test_stream_restart_focus_and_old_packet_reanchor(self):
        pose = (np.zeros(3), np.array([0., 0., 0., 1.]))
        c = HandController(self.model(), *pose); c.toggle()
        c.update(self.frame(), None, 1_000_000_000, pose)
        c.update(self.frame((1, 1, 1), stream='new'), None, 1_000_000_000, pose)
        np.testing.assert_allclose(c.position, pose[0])
        c.update(self.frame((2, 2, 2), stamp=0), None, 1_000_000_000, pose)
        self.assertIsNone(c.mapper.anchor)
        unfocused = replace(self.frame(), tracking=dict(focused=False, controllers={}))
        c.update(unfocused, None, 1_000_000_000, pose)
        self.assertIsNone(c.mapper.anchor)

    def test_optimizer_identity_freshness_and_joint_set(self):
        raw = dict(capture_time_ns=1_000_000_000, stream_id='s', clock_id='c', calibration_id='a',
                   topology_id='t', sequence=5, glove_tracking_valid=True,
                   device=dict(hand='right', device_id='42'))
        candidate = {k: raw[k] for k in ('capture_time_ns', 'stream_id', 'clock_id', 'calibration_id', 'topology_id', 'sequence')}
        candidate.update(protocol='sharpa_optimizer_v1', executable=True, device_id='42', joint_positions_rad={'finger': .5})
        kw = dict(now_ns=1_000_000_000, names=('finger',), calibration_id='a', glove_id=42)
        self.assertEqual(optimizer_targets(raw, candidate, **kw), {'finger': .5})
        self.assertIsNone(optimizer_targets(raw, candidate, **{**kw, 'now_ns': 2_000_000_000}))
        for update in (dict(stream_id='other'), dict(sequence=6)):
            self.assertIsNone(optimizer_targets(raw, {**candidate, **update}, **kw))
        self.assertIsNone(optimizer_targets({**raw, 'glove_tracking_valid': False}, candidate, **kw))
        for update in (dict(executable=False), dict(joint_positions_rad={'wrong': .3})):
            with self.subTest(update=update), self.assertRaises(ValueError):
                optimizer_targets(raw, {**candidate, **update}, **kw)

    def test_impedance_sign_and_bounds(self):
        f, t = wrist_wrench(np.zeros(3), [0, 0, 0, 1], np.zeros(6), [.01, 0, 0],
                             Rotation.from_euler('z', .1).as_quat())
        np.testing.assert_allclose(f, [6, 0, 0]); np.testing.assert_allclose(t, [0, 0, .8])
        f, t = wrist_wrench(np.zeros(3), [0, 0, 0, 1], np.ones(6), [100, 100, 100], [0, 0, 1, 0])
        self.assertLessEqual(max(abs(f)), 60); self.assertLessEqual(max(abs(t)), 8)

    def test_standalone_routes_do_not_start_arm_pipeline(self):
        for backend in ('isaac', 'mujoco'):
            name = backend+'-pico-manus-sharpa-hand'
            self.assertEqual(recipes()[name]['arm'], 'none')
            profile = load_profile(name)
            self.assertIn('uni_teleop.runtime.floating_hand', profile['command'])
            self.assertNotIn('FACTR_PORT', profile['settings'])
            self.assertNotIn('--workbench-only', profile['command'])


if __name__ == '__main__': unittest.main()
