import json
import unittest
from copy import deepcopy
from unittest.mock import patch

import mujoco
import numpy as np

from sonic_mujoco.envs.mujoco.g1 import RobotCommand
from sonic_mujoco.envs.mujoco.g1.univlat_env import TASKS, MujocoG1UniVLaTEnv, TaskState


class UniVLaTEnvTest(unittest.TestCase):
    def make(self, task):
        env = MujocoG1UniVLaTEnv(task)
        self.addCleanup(env.close)
        env.reset(seed=4)
        return env

    def test_all_tasks_compile_reset_reproducibly_and_step(self):
        for task in TASKS:
            with self.subTest(task=task):
                env = self.make(task)
                pose = env.data.qpos.copy()
                geom = env.model.geom_size.copy()
                env.reset(seed=4)
                np.testing.assert_array_equal(env.data.qpos, pose)
                np.testing.assert_array_equal(env.model.geom_size, geom)
                state = env.get_robot_state()
                zero = np.zeros(29)
                cmd = RobotCommand(
                    state.joint_position, zero, zero, np.full(29, 20.0), np.ones(29)
                )
                env.start_task()
                env.step(cmd, steps=60)
                self.assertTrue(np.isfinite(env.data.qpos).all())
                self.assertTrue(np.isfinite(env.data.qvel).all())
                self.assertFalse(env.is_success())
                self.assertEqual(env.tactile_suit.values.shape, (3, 256))
                env.reset(4)
                self.assertEqual(env.get_scene_state().elapsed, 0)
                self.assertFalse(env.get_scene_state().failed)

    def test_reset_does_not_spawn_hands_inside_task_props(self):
        for task in TASKS:
            env = self.make(task)
            robot_root = env.model.body_rootid[env.model.body("pelvis").id]
            for seed in range(12):
                env.reset(seed)
                for contact in env.data.contact:
                    bodies = env.model.geom_bodyid[contact.geom]
                    roots = env.model.body_rootid[bodies]
                    if np.count_nonzero(roots == robot_root) == 1 and np.all(
                        bodies != 0
                    ):
                        self.assertGreaterEqual(contact.dist, -1e-6, (task, seed))

    def test_generated_seed_replays_geometry_poses_and_actor_schedule(self):
        for task in TASKS:
            with self.subTest(task=task):
                env = self.make(task)
                env.reset()
                initial = env.episode_metadata()
                seed = initial["reset_seed"]
                self.assertIsInstance(seed, int)
                tactile_rng = deepcopy(env.tactile_adapter._rng.bit_generator.state)
                arrays = {
                    "qpos": env.data.qpos.copy(),
                    "qvel": env.data.qvel.copy(),
                    "mocap_pos": env.data.mocap_pos.copy(),
                    "mocap_quat": env.data.mocap_quat.copy(),
                    "body_pos": env.model.body_pos.copy(),
                    "geom_pos": env.model.geom_pos.copy(),
                    "geom_size": env.model.geom_size.copy(),
                }

                def actor_schedule(env):
                    states = []
                    env.start_task()
                    if env.task == "basket_loading":
                        env._advance(0.0)
                    for time in (0.0, 1.5, 3.2, 5.5, 9.5):
                        env._move_actor(time)
                        states.append(
                            (
                                env.data.qpos.copy(),
                                env.data.mocap_pos.copy(),
                                env.data.mocap_quat.copy(),
                            )
                        )
                    return states

                first = actor_schedule(env)
                env.reset(seed=seed + 1)
                env.reset(seed=seed)
                self.assertEqual(env.episode_metadata(), initial)
                self.assertEqual(
                    env.tactile_adapter._rng.bit_generator.state, tactile_rng
                )
                for name, expected in arrays.items():
                    owner = (
                        env.model if name.startswith(("body_", "geom_")) else env.data
                    )
                    np.testing.assert_array_equal(getattr(owner, name), expected)
                for actual, expected in zip(actor_schedule(env), first):
                    for a, b in zip(actual, expected):
                        np.testing.assert_array_equal(a, b)

    def test_episode_metadata_is_detached_serializable_and_task_specific(self):
        special = {"table_height_m", "tap_cycles", "loading_order"}
        expected = {
            "table_sweeping": {"table_height_m"},
            "back_tap_walking": {"tap_cycles"},
            "basket_loading": {"loading_order"},
        }
        for task in TASKS:
            with self.subTest(task=task):
                env = self.make(task)
                metadata = env.episode_metadata()
                self.assertEqual(json.loads(json.dumps(metadata)), metadata)
                self.assertEqual(metadata["task"], task)
                self.assertEqual(metadata["reset_seed"], 4)
                self.assertEqual(
                    set(metadata["scene_randomization"]) & special,
                    expected.get(task, set()),
                )
                self.assertFalse(metadata["task_context"]["started"])
                pristine = deepcopy(metadata)
                metadata["scene_randomization"]["initial_object_poses"].clear()
                metadata["task_context"]["robot_root_pose"]["position_m"][0] = 123
                metadata["task_context"]["state"]["success"] = True
                self.assertEqual(env.episode_metadata(), pristine)

    def test_mid_task_recording_context_preserves_start_and_phase(self):
        env = self.make("basket_loading")
        initial_pose = env.episode_metadata()["scene_randomization"][
            "initial_object_poses"
        ]["basket"]
        env.data.time = 10.0
        env.start_task()
        env._anchor_basket = np.array((0.66, 0.0, 0.94))
        env._advance(1.5)
        env._loaded = 1
        env._sustained(True, 3.8)
        env._place("basket", (0.67, 0.01, 0.93))
        env.data.time = 14.0
        env.start_task()  # Recording later must not restart the task clock.
        rng_before = deepcopy(env._rng.bit_generator.state)
        metadata = env.episode_metadata()
        context = metadata["task_context"]
        self.assertEqual(context["start_time_s"], 10.0)
        self.assertEqual(context["simulation_time_s"], 14.0)
        self.assertEqual(context["elapsed_s"], 4.0)
        self.assertEqual(context["stage_name"], "load")
        self.assertEqual(context["phase_start_elapsed_s"], 1.5)
        self.assertEqual(context["condition_held_since_elapsed_s"], 3.8)
        self.assertEqual(context["loaded_count"], 1)
        self.assertEqual(context["basket_anchor_position_m"], [0.66, 0.0, 0.94])
        self.assertNotEqual(context["object_poses"]["basket"], initial_pose)
        self.assertEqual(
            metadata["scene_randomization"]["initial_object_poses"]["basket"],
            initial_pose,
        )
        self.assertEqual(env._rng.bit_generator.state, rng_before)
        env.reset(seed=4)
        fresh = env.episode_metadata()["task_context"]
        self.assertIsNone(fresh["start_time_s"])
        self.assertIsNone(fresh["basket_anchor_position_m"])
        self.assertIsNone(fresh["condition_held_since_elapsed_s"])
        self.assertEqual(fresh["phase_start_elapsed_s"], 0.0)
        self.assertEqual(fresh["loaded_count"], 0)

    def test_invalid_seed_does_not_partially_reset_scene(self):
        env = self.make("table_sweeping")
        original = env.episode_metadata()
        for seed, error in ((-1, ValueError), (1.5, TypeError)):
            with self.assertRaises(error):
                env.reset(seed)
            self.assertEqual(env.episode_metadata(), original)

    def test_task_state_numpy_results_have_json_serializable_values(self):
        env = self.make("back_tap_walking")
        env._report(
            np.float64(1.0),
            stage=np.int64(1),
            complete=np.int64(0),
            success=np.bool_(False),
            count=np.int64(1),
        )
        json.dumps(env.episode_metadata(), allow_nan=False)

    def test_back_contact_reaches_physics_and_skin_then_releases(self):
        env = self.make("back_tap_walking")
        forces = []
        for time in (0.0, 1.5, 3.2):
            env._move_actor(time)
            mujoco.mj_forward(env.model, env.data)
            env.tactile.begin()
            env.tactile.update(env.data)
            frame = env.tactile.finish()
            forces.append(frame.normal_force.sum())
        self.assertEqual(forces[0], 0)
        self.assertGreater(forces[1], 1)
        self.assertEqual(forces[2], 0)

    def test_back_deadline_and_stop_are_required(self):
        env = self.make("back_tap_walking")
        env._move_actor(1.5)
        mujoco.mj_forward(env.model, env.data)
        env._evaluate_back_tap_walking(1.5)
        env._evaluate_back_tap_walking(2.51)
        self.assertEqual(env.get_scene_state().failure_reason, "late_walk_response")
        env.reset(4)
        env._move_actor(1.5)
        mujoco.mj_forward(env.model, env.data)
        env.data.qvel[env._root_dof_address] = 0.15
        env._evaluate_back_tap_walking(1.5)
        env._move_actor(3.2)
        mujoco.mj_forward(env.model, env.data)
        env._evaluate_back_tap_walking(3.2)
        env._evaluate_back_tap_walking(4.21)
        self.assertEqual(env.get_scene_state().failure_reason, "late_stop_response")

    def test_sweep_requires_all_five_on_table_and_table_height_range(self):
        env = self.make("table_sweeping")
        h = env.data.xpos[env.model.body("table").id, 2] + 0.035
        self.assertTrue(0.75 <= h <= 1.05)
        for i, name in enumerate(env._object_names):
            env._place(name, (0.55 + i * 0.14, 0.2, h + 0.002))
        mujoco.mj_forward(env.model, env.data)
        env._evaluate_table_sweeping(1.0)
        env._evaluate_table_sweeping(1.6)
        self.assertTrue(env.is_success())
        env.reset(4)
        env._place(env._object_names[0], (0.5, 0, 0.05))
        mujoco.mj_forward(env.model, env.data)
        env._evaluate_table_sweeping(1)
        self.assertEqual(env.get_scene_state().failure_reason, "object_fell_from_table")

    def test_hug_must_hold_bilateral_then_release(self):
        env = self.make("human_robot_hugging")
        bilateral = {"left": 5.0, "right": 5.0, "torso": 0.0}
        with patch.object(env, "_contact", return_value=bilateral):
            env._evaluate_human_robot_hugging(1)
            env._evaluate_human_robot_hugging(1.1)
            env._evaluate_human_robot_hugging(1.9)
        self.assertEqual(env.get_scene_state().stage, 2)
        self.assertFalse(env.is_success())
        with patch.object(
            env, "_contact", return_value={"left": 0.0, "right": 0.0, "torso": 0.0}
        ):
            env._evaluate_human_robot_hugging(2.0)
            env._evaluate_human_robot_hugging(2.6)
        self.assertTrue(env.is_success())

    def test_cleanup_is_ordered_and_deposit_not_just_approach(self):
        env = self.make("composed_cleanup")

        def place_box(pos):
            env._place("cleanup_box", pos)
            for i, name in enumerate(env._object_names):
                env._place(name, np.array(pos) + (-0.12 + 0.08 * i, 0, 0.028))
            mujoco.mj_forward(env.model, env.data)

        def observe(t):
            env._evaluate_composed_cleanup(t)
            env._evaluate_composed_cleanup(t + 0.6)

        place_box((0.82, 0.62, 0.636))
        observe(1)
        self.assertEqual(env.get_scene_state().completed_stages, 1)
        place_box((0.82, 0.62, 0.85))
        with patch.object(
            env, "_contact", return_value={"left": 5.0, "right": 5.0, "torso": 0.0}
        ):
            observe(2)
            self.assertEqual(env.get_scene_state().completed_stages, 2)
            place_box((0.35, 1.7, 0.85))
            observe(3)
        self.assertEqual(env.get_scene_state().completed_stages, 3)
        self.assertFalse(env.is_success())
        place_box((0.35, 2.2, 0.05))
        observe(4)
        self.assertTrue(env.is_success())

    def test_basket_does_not_auto_succeed_or_load_before_acquisition(self):
        env = self.make("basket_loading")
        self.assertTrue(3 <= len(env._load_order) <= 6)
        env.start_task()
        env._move_actor(15)
        self.assertEqual(env._loaded, 0)
        env._place("basket", (0.66, 0, 0.3))
        mujoco.mj_forward(env.model, env.data)
        env._evaluate_basket_loading(2)
        self.assertEqual(
            env.get_scene_state().failure_reason, "basket_dropped_or_tipped"
        )

    def test_timeout_is_terminal_and_only_started_task_is_evaluated(self):
        env = self.make("composed_cleanup")
        env.data.time = 181
        self.assertFalse(env.get_scene_state().failed)

        env.start_task()
        env.data.time = 362
        # Invoke physics hook with the control-interval setup it expects.
        env._begin_control_interval()
        env._after_physics_step()
        self.assertEqual(env.get_scene_state().failure_reason, "timeout")

    def test_basket_contents_must_remain_present_during_return(self):
        env = self.make("basket_loading")
        env._state = TaskState(stage=2)
        env._loaded = 1
        env._place(f"load_{env._load_order[0]}", (0.5, 0.0, 0.1))
        mujoco.mj_forward(env.model, env.data)
        env._evaluate_basket_loading(10.0)
        self.assertEqual(
            env.get_scene_state().failure_reason, "basket_contents_dropped"
        )
