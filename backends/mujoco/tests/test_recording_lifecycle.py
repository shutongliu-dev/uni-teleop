import json
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from scripts.run_pico_teleop import _TeleopSession
from uni_teleop.recording.episode import EpisodeRecorder
from uni_teleop.backends.mujoco.pipeline import PicoControls, PicoEvents, TeleopMode
from tests.test_teleop_control import command, contacts


def append(recorder, time=0.02, **changes):
    args = {
        "time": time,
        "qpos": np.arange(40, dtype=float),
        "qvel": np.zeros(39),
        "ctrl": np.zeros(29),
        "command": command(),
        "token": np.zeros(64),
        "action": np.zeros(29),
        "controls": PicoControls(),
        "contacts": contacts(),
    }
    args.update(changes)
    recorder.append(**args)


class RecordingLifecycleTest(unittest.TestCase):
    def test_invalid_recording_fps_is_rejected_at_configuration(self):
        for fps in (0, -1, 2.5, True, float("nan")):
            with (
                self.subTest(fps=fps),
                self.assertRaisesRegex(ValueError, "positive integer"),
            ):
                EpisodeRecorder("unused", "empty", record_video=False, fps=fps)

    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.recorder = EpisodeRecorder(directory.name, "sweep", record_video=False)
        self.addCleanup(self.recorder.abort)

    def test_double_start_does_not_discard_buffered_frames(self):
        self.recorder.start()
        append(self.recorder)
        with self.assertRaisesRegex(RuntimeError, "already active"):
            self.recorder.start()
        self.assertEqual(self.recorder.frame_count, 1)
        self.assertIsNotNone(self.recorder.finish())

    def test_invalid_state_or_time_is_rejected_before_writing(self):
        self.recorder.start()
        append(self.recorder)
        for time, changes in (
            (0.04, {"qpos": np.full(40, np.nan)}),
            (0.02, {}),
            (0.0, {}),
            (0.08, {}),
            (0.04, {"ctrl": np.zeros(30)}),
        ):
            with self.subTest(time=time, changes=tuple(changes)):
                with self.assertRaises(ValueError):
                    append(self.recorder, time, **changes)
                self.assertEqual(self.recorder.frame_count, 1)
        append(self.recorder, 0.04)
        self.assertIsNotNone(self.recorder.finish())

    def test_schema_cannot_silently_change_between_episodes(self):
        self.recorder.start()
        append(self.recorder)
        self.recorder.finish()
        self.recorder.start()
        with self.assertRaisesRegex(ValueError, "fields, shapes or dtypes"):
            append(self.recorder, task_state={"success": True})
        self.assertEqual(self.recorder.frame_count, 0)

    def test_task_context_is_per_episode_snapshot_and_outcomes_are_explicit(self):
        context = {"reset_seed": 42, "scene_randomization": {"table_height_m": 0.9}}
        self.recorder.start(task_context=context)
        context["scene_randomization"]["table_height_m"] = 1.0
        append(
            self.recorder,
            task_state={"success": False, "failed": False, "failure_reason": ""},
        )
        path = self.recorder.finish()
        self.recorder.start(task_context={"reset_seed": 43})
        append(
            self.recorder,
            task_state={"success": False, "failed": True, "failure_reason": "timeout"},
        )
        self.recorder.finish()
        episodes = [
            json.loads(line)
            for line in (path.parents[2] / "meta/episodes.jsonl")
            .read_text()
            .splitlines()
        ]
        self.assertEqual(
            episodes[0]["task_context"]["scene_randomization"]["table_height_m"], 0.9
        )
        self.assertEqual([e["task_context"]["reset_seed"] for e in episodes], [42, 43])
        self.assertEqual(
            [e["task_outcome"]["status"] for e in episodes], ["incomplete", "failed"]
        )

    def test_partial_parquet_failure_does_not_touch_saved_episode(self):
        self.recorder.start()
        append(self.recorder)
        first = self.recorder.finish()
        original = first.read_bytes()
        metadata = (first.parents[2] / "meta/info.json").read_bytes()
        self.recorder.start()
        append(self.recorder)

        def fail(path, arrays):
            path.write_bytes(b"incomplete")
            raise OSError("test disk full")

        with (
            patch.object(self.recorder, "_write_parquet", side_effect=fail),
            self.assertRaisesRegex(OSError, "disk full"),
        ):
            self.recorder.finish()
        with self.assertRaisesRegex(RuntimeError, "failed"):
            self.recorder.finish()
        self.recorder.abort()
        self.assertEqual(first.read_bytes(), original)
        self.assertEqual((first.parents[2] / "meta/info.json").read_bytes(), metadata)
        self.assertFalse(list(first.parents[2].rglob("*.partial")))
        self.assertFalse(list(first.parents[2].rglob("episode_000001*")))

    def test_preview_failure_does_not_turn_saved_episode_into_failure(self):
        self.recorder.start()
        append(self.recorder)
        with (
            patch.object(
                self.recorder,
                "_write_contact_preview",
                side_effect=OSError("preview disk full"),
            ),
            self.assertWarnsRegex(UserWarning, "Episode saved"),
        ):
            path = self.recorder.finish()
        self.assertTrue(path.is_file())
        self.assertFalse(self.recorder.active)
        self.assertIsNone(self.recorder.last_preview)


class TaskStartControlTest(unittest.TestCase):
    def make_session(self, task_start="recording", seed=7):
        env = SimpleNamespace(
            start_task=Mock(),
            reset=Mock(),
            episode_metadata=lambda: {"reset_seed": seed},
            tactile_adapter=SimpleNamespace(metadata={"sensor": "test"}),
        )
        resources = SimpleNamespace(
            args=SimpleNamespace(task_start=task_start, scene_seed=seed),
            env=env,
            recorder=Mock(active=False),
            cameras=Mock(),
            video=None,
            teleop=object(),
            haptics=None,
            encoder=Mock(),
            controller=Mock(),
        )
        return _TeleopSession(resources)

    def test_default_task_does_not_consume_time_during_pose_setup(self):
        session = self.make_session()
        session._start_task("pose")
        session.resources.env.start_task.assert_not_called()
        session._update_recording(PicoEvents(toggle_recording=True))
        session.resources.env.start_task.assert_called_once()
        session.resources.recorder.start.assert_called_once_with(
            tactile_metadata={"sensor": "test"}, task_context={"reset_seed": 7}
        )

    def test_reset_preserves_fixed_seed_and_restarts_only_pose_trigger(self):
        for trigger in ("pose", "recording"):
            with self.subTest(trigger=trigger):
                session = self.make_session(trigger, seed=11)
                session.mode = TeleopMode.POSE
                session._reset_scene()
                session.resources.env.reset.assert_called_once_with(seed=11)
                self.assertEqual(
                    session.resources.env.start_task.call_count, int(trigger == "pose")
                )
                session.resources.cameras.reset.assert_called_once()


if __name__ == "__main__":
    unittest.main()
