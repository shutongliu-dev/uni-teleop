import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from scripts.run_pico_teleop import _append_recording
from sonic_mujoco.contact import MAX_CONTACTS, ContactFrame
from sonic_mujoco.envs.mujoco.g1 import MujocoG1EmptyEnv
from uni_teleop.recording.episode import EpisodeRecorder, _fit_video_frame
from sonic_mujoco.tactile import TaxelLayout
from uni_teleop.backends.mujoco.pipeline import (
    PicoControls,
    PicoEvents,
    TeleopCommand,
    TeleopMode,
    next_mode,
)
from uni_teleop.video.mujoco import RenderedCameraFrame


def command() -> TeleopCommand:
    return TeleopCommand(
        frame_index=np.arange(5),
        smpl_joints=np.zeros((5, 24, 3)),
        root_quaternion=np.tile([1.0, 0.0, 0.0, 0.0], (5, 1)),
        joint_position=np.zeros((5, 29)),
    )


def contacts() -> ContactFrame:
    robot = np.full(MAX_CONTACTS, -1, dtype=np.int32)
    other = np.full(MAX_CONTACTS, -1, dtype=np.int32)
    robot[0], other[0] = 1, 2
    normal_force = np.zeros(MAX_CONTACTS, dtype=np.float32)
    normal_impulse = np.zeros(MAX_CONTACTS, dtype=np.float32)
    normal_force[0], normal_impulse[0] = 12.0, 0.06
    return ContactFrame(
        robot_body_id=robot,
        other_body_id=other,
        position=np.zeros((MAX_CONTACTS, 3), dtype=np.float32),
        normal_force=normal_force,
        tangent_force=np.zeros(MAX_CONTACTS, dtype=np.float32),
        normal_impulse=normal_impulse,
        sample_count=np.r_[2, np.zeros(MAX_CONTACTS - 1)].astype(np.int32),
        count=1,
    )


def append_frame(
    recorder: EpisodeRecorder,
    video_frame: np.ndarray | None = None,
    camera_pose: np.ndarray | None = None,
    appearance_camera_pose: np.ndarray | None = None,
) -> None:
    recorder.append(
        time=0.02,
        qpos=np.arange(40),
        qvel=np.arange(39),
        ctrl=np.arange(29),
        command=command(),
        token=np.arange(64),
        action=np.arange(29),
        controls=PicoControls(left_grip=0.8, a=True),
        contacts=contacts(),
        video_frame=video_frame,
        camera_pose=camera_pose,
        appearance_camera_pose=appearance_camera_pose,
    )


def append_tactile_frame(recorder: EpisodeRecorder, env: MujocoG1EmptyEnv) -> None:
    recorder.append(
        time=env.time,
        qpos=env.data.qpos,
        qvel=env.data.qvel,
        ctrl=env.data.ctrl,
        command=command(),
        token=np.arange(64),
        action=np.arange(29),
        controls=PicoControls(),
        contacts=env.contacts.last_frame,
        tactile=env.tactile.last_frame,
        tactile_suit=env.tactile_adapter.last_frame,
    )


class TeleopControlTest(unittest.TestCase):
    def test_video_letterbox_preserves_widescreen_aspect(self) -> None:
        frame = np.full((720, 1280, 3), 127, dtype=np.uint8)

        fitted = _fit_video_frame(frame)

        self.assertEqual(fitted.shape, (480, 640, 3))
        self.assertTrue(np.all(fitted[60:420] == 127))
        self.assertTrue(np.all(fitted[:60] == 0))
        self.assertTrue(np.all(fitted[420:] == 0))

    def test_start_and_pose_combinations_follow_the_real_robot_flow(self) -> None:
        mode = next_mode(TeleopMode.OFF, PicoEvents(start_stop=True))
        self.assertIs(mode, TeleopMode.READY)
        mode = next_mode(mode, PicoEvents(toggle_pose=True))
        self.assertIs(mode, TeleopMode.POSE)
        mode = next_mode(mode, PicoEvents(toggle_pose=True))
        self.assertIs(mode, TeleopMode.READY)
        mode = next_mode(mode, PicoEvents(start_stop=True))
        self.assertIs(mode, TeleopMode.OFF)

    def test_episode_recorder_saves_aligned_arrays(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            recorder = EpisodeRecorder(
                directory,
                "sweep",
                body_names=("world", "right_wrist", "sweep_object"),
                record_video=False,
            )
            recorder.start()
            append_frame(recorder)
            self.assertEqual(recorder.frame_count, 1)
            self.assertAlmostEqual(recorder.duration_seconds, 0.02)
            path = recorder.finish()

            self.assertIsNotNone(path)
            assert path is not None
            self.assertTrue(Path(path).is_file())
            table = pq.read_table(path)
            self.assertEqual(table.num_rows, 1)
            self.assertIn("observation.qpos", table.column_names)
            self.assertIn("observation.contact.normal_force", table.column_names)
            session = path.parents[2]
            self.assertTrue((session / "meta/info.json").is_file())
            self.assertTrue((session / "meta/modality.json").is_file())
            self.assertTrue((session / "meta/episodes.jsonl").is_file())
            self.assertTrue(recorder.last_preview.is_file())
            self.assertEqual(recorder.frame_count, 0)

            recorder.start()
            append_frame(recorder)
            second = recorder.finish()
            info = json.loads((session / "meta/info.json").read_text())
            self.assertIsNotNone(second)
            assert second is not None
            self.assertEqual(second.name, "episode_000001.parquet")
            self.assertEqual(info["total_episodes"], 2)
            self.assertEqual(info["total_frames"], 2)
            self.assertEqual(pq.read_table(second)["index"][0].as_py(), 1)

    def test_aborted_recording_does_not_write_a_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            recorder = EpisodeRecorder(directory, "sweep", record_video=False)
            recorder.start()
            append_frame(recorder)
            recorder.abort()

            self.assertFalse(recorder.active)
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_recorder_writes_supplied_reconstructed_scene_frame(self) -> None:
        frame = np.full((480, 640, 3), (23, 71, 149), dtype=np.uint8)
        with (
            tempfile.TemporaryDirectory() as directory,
            patch("uni_teleop.recording.episode._VideoWriter") as writer_class,
        ):
            recorder = EpisodeRecorder(
                directory,
                "lab_scan",
                model=object(),
                data=object(),
                video_source="reconstructed_scene_composite",
                video_metadata={"renderer": "mujoco_3dgs_depth_composite"},
            )
            recorder.start()
            camera_pose = np.eye(4)
            appearance_camera_pose = np.eye(4)
            appearance_camera_pose[0, 3] = 1.25
            append_frame(
                recorder,
                frame,
                camera_pose,
                appearance_camera_pose,
            )
            path = recorder.finish()

            written = writer_class.return_value.add_frame.call_args.args[0]
            np.testing.assert_array_equal(written, frame)
            assert path is not None
            info = json.loads((path.parents[2] / "meta/info.json").read_text())
            self.assertEqual(
                info["script_config"]["video_source"],
                "reconstructed_scene_composite",
            )
            self.assertEqual(
                info["script_config"]["video_metadata"]["renderer"],
                "mujoco_3dgs_depth_composite",
            )
            table = pq.read_table(path)
            self.assertIn("observation.camera_pose", table.column_names)
            self.assertIn("observation.appearance_camera_pose", table.column_names)

    def test_reconstructed_scene_recording_rejects_missing_rgb_frame(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            recorder = EpisodeRecorder(
                directory,
                "lab_scan",
                model=object(),
                data=object(),
                video_source="reconstructed_scene_composite",
            )
            recorder.start()

            with self.assertRaisesRegex(RuntimeError, "composite RGB frame"):
                append_frame(recorder)

    def test_recorder_preserves_real_tactile_schema_and_physical_sidecar(
        self,
    ) -> None:
        env = MujocoG1EmptyEnv()
        self.addCleanup(env.close)
        env.reset()
        with tempfile.TemporaryDirectory() as directory:
            recorder = EpisodeRecorder(
                directory,
                "empty",
                body_names=env.contacts.body_names,
                record_video=False,
                tactile_layout=env.tactile_skin_layout,
                tactile_metadata=env.tactile_adapter.metadata,
            )
            with self.assertRaisesRegex(ValueError, "current sensor metadata"):
                recorder.start()
            recorder.start(tactile_metadata=env.tactile_adapter.metadata)
            append_tactile_frame(recorder, env)
            path = recorder.finish()

            assert path is not None
            table = pq.read_table(path)
            for device in ("vest", "left_arm", "right_arm"):
                key = f"observation.tactile_{device}"
                self.assertIn(key, table.column_names)
                self.assertEqual(len(table[key][0].as_py()), 256)
                self.assertTrue(pa.types.is_list(table.schema.field(key).type))
                self.assertTrue(
                    pa.types.is_uint8(table.schema.field(key).type.value_type)
                )
            self.assertIn("observation.tactile.normal_force_n", table.column_names)
            self.assertIn(
                "observation.tactile.tangent_force_body_n", table.column_names
            )
            self.assertIn("observation.tactile.other_geom_id", table.column_names)
            self.assertEqual(
                len(table["observation.tactile.normal_force_n"][0].as_py()),
                624,
            )
            session = path.parents[2]
            layout = json.loads((session / "meta/tactile_layout.json").read_text())
            info = json.loads((session / "meta/info.json").read_text())
            modality = json.loads((session / "meta/modality.json").read_text())
            self.assertEqual(layout["wired_taxels"], 624)
            self.assertEqual(
                layout["sha256"], info["script_config"]["tactile"]["layout_sha256"]
            )
            self.assertEqual(
                modality["tactile"]["vest"]["original_key"],
                "observation.tactile_vest",
            )
            self.assertEqual(
                set(modality["tactile"]), {"vest", "left_arm", "right_arm"}
            )
            self.assertIn("physical_force", modality["simulation_tactile"])
            self.assertEqual(
                info["features"]["observation.tactile_vest"]["names"],
                [f"raw_{index:03d}" for index in range(1, 257)],
            )

            with self.assertRaisesRegex(ValueError, "record rate"):
                EpisodeRecorder(
                    directory,
                    "empty",
                    fps=25,
                    record_video=False,
                    tactile_layout=env.tactile_skin_layout,
                    tactile_metadata=env.tactile_adapter.metadata,
                )

            bad_layout = TaxelLayout(
                body_id=env.tactile.last_frame.layout.body_id.copy(),
                geom_id=env.tactile.last_frame.layout.geom_id.copy(),
                local_center=env.tactile.last_frame.layout.local_center.copy(),
                local_normal=env.tactile.last_frame.layout.local_normal.copy(),
                region_id=env.tactile.last_frame.layout.region_id.copy(),
                channel_id=np.roll(
                    env.tactile.last_frame.layout.channel_id.copy(),
                    1,
                ),
                region_names=env.tactile.last_frame.layout.region_names,
            )
            recorder.start(tactile_metadata=env.tactile_adapter.metadata)
            with self.assertRaisesRegex(ValueError, "does not match layout"):
                recorder.append(
                    time=env.time,
                    qpos=env.data.qpos,
                    qvel=env.data.qvel,
                    ctrl=env.data.ctrl,
                    command=command(),
                    token=np.arange(64),
                    action=np.arange(29),
                    controls=PicoControls(),
                    contacts=env.contacts.last_frame,
                    tactile=replace(env.tactile.last_frame, layout=bad_layout),
                    tactile_suit=env.tactile_suit,
                )

    def test_recorder_tracks_realized_tactile_profile_per_episode(self) -> None:
        env = MujocoG1EmptyEnv()
        self.addCleanup(env.close)
        with tempfile.TemporaryDirectory() as directory:
            recorder = EpisodeRecorder(
                directory,
                "empty",
                body_names=env.contacts.body_names,
                record_video=False,
                tactile_layout=env.tactile_skin_layout,
                tactile_metadata=env.tactile_adapter.metadata,
            )
            paths = []
            for seed in (11, 12):
                env.tactile_adapter.reset(seed=seed)
                recorder.start(tactile_metadata=env.tactile_adapter.metadata)
                append_tactile_frame(recorder, env)
                paths.append(recorder.finish())

            assert paths[-1] is not None
            session = paths[-1].parents[2]
            episodes = [
                json.loads(line)
                for line in (session / "meta/episodes.jsonl").read_text().splitlines()
            ]
            info = json.loads((session / "meta/info.json").read_text())

            self.assertEqual(
                [episode["tactile"]["episode_seed"] for episode in episodes],
                [11, 12],
            )
            self.assertNotEqual(
                episodes[0]["tactile"]["episode_profile"]["sha256"],
                episodes[1]["tactile"]["episode_profile"]["sha256"],
            )
            self.assertEqual(
                set(episodes[0]["tactile"]),
                {
                    "episode_profile",
                    "episode_seed",
                    "realized_device_sample_rate_hz",
                    "taxel_gain_sha256",
                },
            )
            self.assertEqual(
                info["script_config"]["tactile"]["episode_metadata"],
                "meta/episodes.jsonl[].tactile",
            )
            self.assertNotIn(
                "episode_seed",
                info["script_config"]["tactile"],
            )

            changed_profile = dict(env.tactile_adapter.metadata)
            changed_profile["profile_sha256"] = "0" * 64
            with self.assertRaisesRegex(ValueError, "cannot change within one session"):
                recorder.start(tactile_metadata=changed_profile)

            static_only = dict(env.tactile_adapter.metadata)
            for key in (
                "episode_profile",
                "episode_seed",
                "realized_device_sample_rate_hz",
                "taxel_gain_sha256",
            ):
                static_only.pop(key)
            with self.assertRaisesRegex(ValueError, "metadata is missing"):
                recorder.start(tactile_metadata=static_only)

    def test_video_throttle_holds_rgb_without_dropping_tactile_rows(self) -> None:
        env = MujocoG1EmptyEnv()
        self.addCleanup(env.close)
        env.reset()
        rendered = RenderedCameraFrame(
            rgb=np.zeros((480, 640, 3), dtype=np.uint8),
            camera_to_world=np.eye(4),
            appearance_camera_to_world=np.eye(4),
        )
        video = SimpleNamespace(
            render=lambda **_: None,
            latest_rendered=rendered,
        )
        with (
            tempfile.TemporaryDirectory() as directory,
            patch("uni_teleop.recording.episode._VideoWriter"),
        ):
            recorder = EpisodeRecorder(
                directory,
                "lab_scan",
                model=env.model,
                data=env.data,
                video_source="reconstructed_scene_composite",
                tactile_layout=env.tactile_skin_layout,
                tactile_metadata=env.tactile_adapter.metadata,
            )
            recorder.start(tactile_metadata=env.tactile_adapter.metadata)
            for _ in range(2):
                env.data.time += 0.02
                _append_recording(
                    recorder,
                    env,
                    command(),
                    np.arange(64),
                    np.arange(29),
                    PicoControls(),
                    video,
                )

            self.assertEqual(recorder.frame_count, 2)
            self.assertEqual(recorder._frames[0]["sim_time"], 0.02)
            self.assertEqual(recorder._frames[1]["sim_time"], 0.04)
            np.testing.assert_array_equal(
                recorder._frames[0]["observation.tactile_vest"],
                env.tactile_suit.values[0],
            )
            recorder.abort()


if __name__ == "__main__":
    unittest.main()
