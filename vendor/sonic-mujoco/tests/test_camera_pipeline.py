import json
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import av
import mujoco
import numpy as np
import pyarrow.parquet as pq

from sonic_mujoco.cameras import CameraRig, CameraSampler, depth_to_normals, scan_mapper
from sonic_mujoco.envs.mujoco.g1 import MujocoG1EmptyEnv
from sonic_mujoco.envs.mujoco.g1.univlat_env import MujocoG1UniVLaTEnv
from sonic_mujoco.recording import EpisodeRecorder
from sonic_mujoco.scan import CameraIntrinsics
from sonic_mujoco.teleop import PicoControls, TeleopCommand
from sonic_mujoco.teleop.pico_video import PicoVideo


class MemoryBridge:
    def __init__(self, **kwargs):
        self.frames = []

    def ready(self):
        return True

    def publish_stereo(self, left, right):
        self.frames.append((left.copy(), right.copy()))

    def close(self):
        pass


def append_camera_row(recorder, env, frames):
    recorder.append(
        time=env.time,
        qpos=env.data.qpos,
        qvel=env.data.qvel,
        ctrl=env.data.ctrl,
        command=TeleopCommand(
            frame_index=np.arange(1),
            smpl_joints=np.zeros((1, 24, 3)),
            root_quaternion=np.array([[1.0, 0, 0, 0]]),
            joint_position=np.zeros((1, 29)),
        ),
        token=np.zeros(64),
        action=np.zeros(29),
        controls=PicoControls(),
        contacts=env.contacts.last_frame,
        camera_frames=frames,
        task_state={"stage": 0, "failure_reason": "", "success": False},
    )


class CameraPipelineTest(unittest.TestCase):
    def test_robot_translation_changes_each_camera_view_and_pose(self):
        env = MujocoG1UniVLaTEnv("table_sweeping")
        self.addCleanup(env.close)
        env.reset(4)
        rig = CameraRig(env.model, env.data, width=128, height=96)
        self.addCleanup(rig.close)
        before = rig.capture()
        delta = np.array((0.10, 0.15, 0.0))
        env.data.qpos[env._root_qpos_address : env._root_qpos_address + 3] += delta
        after = rig.capture()
        for name in before:
            np.testing.assert_allclose(
                after[name].camera_to_world[:3, 3]
                - before[name].camera_to_world[:3, 3],
                delta,
            )
            self.assertGreater(
                np.abs(after[name].rgb.astype(float) - before[name].rgb).mean(), 1
            )

    def test_fractional_camera_rate_stays_anchored_to_simulation_time(self):
        rig = SimpleNamespace(data=SimpleNamespace(time=0.0))
        rig.capture = lambda: {"stamp": rig.data.time}
        sampler = CameraSampler(rig, fps=30)
        stamps = []
        for i in range(50):
            rig.data.time = i / 50
            stamps.append(sampler.capture()["stamp"])
        self.assertEqual(len(set(stamps)), 30)
        self.assertLessEqual(
            max(i / 50 - stamp for i, stamp in enumerate(stamps)), 0.04
        )
        sampler.reset()
        rig.data.time = 0
        self.assertEqual(sampler.capture()["stamp"], 0)

    def test_analytic_normals_and_discontinuities(self):
        k = CameraIntrinsics.from_vertical_fov(32, 24, 60)
        depth = np.full((24, 32), 2.0, np.float32)
        valid = np.ones_like(depth, bool)
        normals, mask = depth_to_normals(depth, valid, k)
        np.testing.assert_allclose(
            normals[mask], np.tile([0, 0, -1], (mask.sum(), 1)), atol=1e-6
        )
        depth[:, 16:] = 3
        normals, mask = depth_to_normals(depth, valid, k)
        self.assertFalse(mask[:, 15:17].any())
        self.assertTrue(np.all(normals[~mask] == 0))

    def test_axial_depth_and_capture_timestamp(self):
        model = mujoco.MjModel.from_xml_string("""<mujoco><worldbody>
          <geom type="plane" size="3 3 .1"/><camera name="eye" pos="0 0 2" fovy="60"/>
          </worldbody></mujoco>""")
        data = mujoco.MjData(model)
        rig = CameraRig(model, data, width=64, height=48, camera_names={"test": "eye"})
        self.addCleanup(rig.close)
        frame = rig.capture()["test"]
        np.testing.assert_allclose(frame.depth_m, 2.0, atol=1e-4)
        self.assertTrue(frame.valid_depth.all())
        self.assertEqual(frame.timestamp, data.time)

    def test_pico_stereo_tracks_neck_and_keeps_hud_out_of_data(self):
        env = MujocoG1EmptyEnv()
        self.addCleanup(env.close)
        env.reset()
        with patch("sonic_mujoco.teleop.pico_video.PicoFrameBridge", MemoryBridge):
            video = PicoVideo(env.model, env.data, width=320, height=180)
        self.addCleanup(video.close)
        first = video.render(recording=True)
        left, right = video._bridge.frames[-1]
        self.assertGreater(np.abs(left.astype(float) - right).mean(), 1)
        self.assertFalse(np.array_equal(left, first.rgb))
        env.data.qpos[env._neck_qpos_indices] = (0.3, 0.2)
        second = video.render()
        self.assertGreater(np.abs(first.rgb.astype(float) - second.rgb).mean(), 5)
        self.assertFalse(np.allclose(first.camera_to_world, second.camera_to_world))

    def test_multicamera_roundtrip_hold_and_abort(self):
        env = MujocoG1EmptyEnv()
        self.addCleanup(env.close)
        env.reset()
        rig = CameraRig(env.model, env.data, width=64, height=48)
        self.addCleanup(rig.close)
        sampler = CameraSampler(rig, fps=10)
        with tempfile.TemporaryDirectory() as directory:
            recorder = EpisodeRecorder(
                directory, "camera_test", camera_metadata=sampler.metadata
            )
            recorder.start()
            first = None
            for i in range(46):
                env.data.time = i * 0.02
                env.data.qpos[env._neck_qpos_indices] = (i * 0.002, 0)
                frames = sampler.capture()
                if i == 0:
                    first = frames
                append_camera_row(recorder, env, frames)
            path = recorder.finish()
            session = path.parents[2]
            rows = pq.read_table(path)
            key = "observation.camera.ego_view_left"
            self.assertEqual(rows.num_rows, 46)
            self.assertEqual(rows[f"{key}.sample_index"][4].as_py(), 0)
            self.assertAlmostEqual(rows[f"{key}.age_s"][4].as_py(), 0.08)
            self.assertFalse(rows[f"{key}.updated"][4].as_py())
            self.assertEqual(rows[f"{key}.sample_index"][5].as_py(), 1)
            geometry = sorted((session / "geometry/episode_000000").glob("*.npz"))
            self.assertEqual(len(geometry), 2)
            with np.load(geometry[0]) as saved:
                np.testing.assert_array_equal(
                    saved["ego_view_left.depth_m"][0], first["ego_view_left"].depth_m
                )
                np.testing.assert_array_equal(
                    saved["ego_view_left.normals_camera"][0],
                    first["ego_view_left"].normals_camera,
                )
                self.assertEqual(len(saved["ego_view_left.timestamp"]), 8)
            info = json.loads((session / "meta/info.json").read_text())
            self.assertEqual(info["total_videos"], 4)
            preview = recorder.last_preview.read_text()
            self.assertIn("observation.images.wrist_view_right/", preview)
            self.assertNotIn("observation.images.ego_view/", preview)
            for video in (session / "videos").rglob("*.mp4"):
                with av.open(str(video)) as container:
                    decoded = list(container.decode(video=0))
                self.assertEqual(len(decoded), 46)
                self.assertEqual(
                    decoded[0].to_ndarray(format="rgb24").shape, (48, 64, 3)
                )
            recorder.start()
            append_camera_row(recorder, env, sampler.capture())
            recorder.abort()
            self.assertFalse(list(session.rglob("episode_000001.mp4")))
            self.assertEqual(len(list(session.rglob("episode_000000.mp4"))), 4)

    def test_recorded_robot_camera_does_not_follow_vr_mocap(self):
        env = MujocoG1EmptyEnv()
        self.addCleanup(env.close)
        env.reset()
        rig = CameraRig(env.model, env.data, width=64, height=48)
        self.addCleanup(rig.close)
        before = rig.capture()
        index = env.model.body_mocapid[env.model.body("vr_view").id]
        env.data.mocap_pos[index] = (3, 4, 2)
        after = rig.capture()
        for name in before:
            np.testing.assert_array_equal(before[name].rgb, after[name].rgb)
            np.testing.assert_array_equal(
                before[name].camera_to_world, after[name].camera_to_world
            )

    def test_scan_composition_uses_scene_world_rotation_and_scale(self):
        env = MujocoG1EmptyEnv()
        self.addCleanup(env.close)
        env.reset()
        info = {"anchor_camera_to_world": np.eye(4), "scan_up": [0, 1, 0]}
        mapper = scan_mapper(env.model, info, 0.2)
        requested = []

        def render(pose, k):
            requested.append(pose.copy())
            return np.full((k.height, k.width, 3), 80, np.uint8), np.full(
                (k.height, k.width), 0.4, np.float32
            )

        client = SimpleNamespace(render_rgbd=render)
        rig = CameraRig(
            env.model, env.data, width=64, height=48, scan_client=client, mapper=mapper
        )
        self.addCleanup(rig.close)
        frames = rig.capture()
        rotation = mapper.scan_rotation_to_mujoco(np.eye(3)).T
        for frame, pose in zip(frames.values(), requested):
            np.testing.assert_allclose(
                pose[:3, :3], rotation @ frame.camera_to_world[:3, :3]
            )
            self.assertTrue(np.any(np.isclose(frame.depth_m, 2.0)))
        self.assertEqual(rig.metadata["depth_units"], "estimated_meters")
