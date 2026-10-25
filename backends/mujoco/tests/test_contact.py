import unittest
from dataclasses import replace

import mujoco
import numpy as np

from sonic_mujoco.contact import MAX_CONTACTS, ContactRecorder
from sonic_mujoco.envs.mujoco.g1 import (
    MujocoG1PlushCarryEnv,
    MujocoG1SweepEnv,
    RobotCommand,
)


def zero_command() -> RobotCommand:
    zeros = np.zeros(29)
    return RobotCommand(zeros, zeros, zeros, zeros, zeros)


class ContactRecorderTest(unittest.TestCase):
    def setUp(self) -> None:
        self.env = MujocoG1SweepEnv()
        self.env.reset(seed=0)

    def tearDown(self) -> None:
        self.env.close()

    def test_contact_frame_has_fixed_empty_shape(self) -> None:
        frame = ContactRecorder(self.env.model).last_frame

        self.assertEqual(frame.robot_body_id.shape, (MAX_CONTACTS,))
        self.assertEqual(frame.position.shape, (MAX_CONTACTS, 3))
        self.assertEqual(frame.count, 0)
        self.assertTrue(np.all(frame.robot_body_id == -1))

    def test_reset_clears_contact_and_tactile_history(self) -> None:
        hand_body = mujoco.mj_name2id(
            self.env.model, mujoco.mjtObj.mjOBJ_BODY, "right_wrist_yaw_link"
        )
        object_joint = mujoco.mj_name2id(
            self.env.model, mujoco.mjtObj.mjOBJ_JOINT, "sweep_object_0_joint"
        )
        address = self.env.model.jnt_qposadr[object_joint]
        self.env.data.qpos[address : address + 3] = self.env.data.xpos[hand_body]
        mujoco.mj_forward(self.env.model, self.env.data)
        self.env.step(zero_command(), steps=2)
        self.assertGreater(self.env.contacts.last_frame.count, 0)

        self.env.reset(seed=0)

        self.assertEqual(self.env.contacts.last_frame.count, 0)
        self.assertFalse(np.any(self.env.tactile.last_frame.contact))
        self.assertFalse(np.any(self.env.tactile_suit.values))

    def test_robot_object_contact_is_accumulated(self) -> None:
        hand_body = mujoco.mj_name2id(
            self.env.model, mujoco.mjtObj.mjOBJ_BODY, "right_wrist_yaw_link"
        )
        object_joint = mujoco.mj_name2id(
            self.env.model, mujoco.mjtObj.mjOBJ_JOINT, "sweep_object_0_joint"
        )
        address = self.env.model.jnt_qposadr[object_joint]
        self.env.data.qpos[address : address + 3] = self.env.data.xpos[hand_body]
        mujoco.mj_forward(self.env.model, self.env.data)

        self.env.step(zero_command(), steps=2)
        frame = self.env.contacts.last_frame
        active = frame.robot_body_id >= 0

        self.assertGreater(frame.count, 0)
        self.assertTrue(np.any(frame.other_body_id[active] > 0))
        self.assertGreater(frame.normal_impulse[active].sum(), 0.0)

    def test_control_interval_or_merges_substep_tactile_updates(self) -> None:
        profile = replace(
            self.env.tactile_adapter.profile,
            schema_version=2,
            profile_id="test-causal-v2",
        )
        self.env.configure_tactile_profile(profile, seed=0)
        substep_updates = []
        update_normal_force = self.env.tactile_adapter.update_normal_force

        def track_update(normal_force: np.ndarray, time: float):
            frame = update_normal_force(normal_force, time)
            substep_updates.append(frame.updated.copy())
            return frame

        self.env.tactile_adapter.update_normal_force = track_update
        self.env.step(zero_command(), steps=2)

        self.assertEqual(len(substep_updates), 2)
        self.assertTrue(substep_updates[0][0])
        self.assertFalse(substep_updates[1][0])
        np.testing.assert_array_equal(
            self.env.tactile_suit.updated,
            np.logical_or.reduce(substep_updates),
        )
        np.testing.assert_array_equal(
            self.env.tactile_adapter.last_frame.updated,
            self.env.tactile_suit.updated,
        )
        self.assertAlmostEqual(
            self.env.tactile.last_frame.duration,
            2.0 * self.env.timestep,
        )

    def test_v1_tactile_profile_keeps_control_interval_adapter_path(self) -> None:
        interval_updates = []
        substep_updates = []
        update = self.env.tactile_adapter.update
        update_normal_force = self.env.tactile_adapter.update_normal_force

        def track_interval(frame, time: float):
            interval_updates.append((frame.duration, time))
            return update(frame, time)

        def track_substep(normal_force: np.ndarray, time: float):
            substep_updates.append(time)
            return update_normal_force(normal_force, time)

        self.env.tactile_adapter.update = track_interval
        self.env.tactile_adapter.update_normal_force = track_substep
        self.env.step(zero_command(), steps=2)

        self.assertEqual(interval_updates, [(2.0 * self.env.timestep, self.env.time)])
        self.assertFalse(substep_updates)

    def test_robot_flex_contact_is_accumulated(self) -> None:
        plush_env = MujocoG1PlushCarryEnv()
        self.addCleanup(plush_env.close)
        plush_env.reset(seed=0)
        hand_body = mujoco.mj_name2id(
            plush_env.model,
            mujoco.mjtObj.mjOBJ_BODY,
            "right_wrist_yaw_link",
        )
        plush_joint = mujoco.mj_name2id(
            plush_env.model, mujoco.mjtObj.mjOBJ_JOINT, "carry_plush_joint"
        )
        address = plush_env.model.jnt_qposadr[plush_joint]
        plush_env.data.qpos[address : address + 3] = plush_env.data.xpos[hand_body]
        mujoco.mj_forward(plush_env.model, plush_env.data)

        plush_env.step(zero_command(), steps=2)
        frame = plush_env.contacts.last_frame
        active = frame.robot_body_id >= 0

        self.assertGreater(frame.count, 0)
        self.assertTrue(np.any(frame.other_body_id[active] > 0))
        self.assertGreater(frame.normal_impulse[active].sum(), 0.0)


if __name__ == "__main__":
    unittest.main()
