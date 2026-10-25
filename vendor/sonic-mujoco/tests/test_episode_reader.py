import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from sonic_mujoco.cameras import ROBOT_CAMERAS, CameraFrame
from sonic_mujoco.contact import MAX_CONTACTS, ContactFrame
from sonic_mujoco.episode_reader import EpisodeReader, EpisodeReadError
from sonic_mujoco.recording import EpisodeRecorder
from sonic_mujoco.teleop import PicoControls, TeleopCommand


def record_session(directory, legacy=False, chunk_size=8):
    metadata = {
        "cameras": {name: {"height": 12, "width": 16} for name in ROBOT_CAMERAS},
        "depth_units": "meters",
    }
    recorder = EpisodeRecorder(
        directory,
        "inspection_test",
        model=object(),
        data=object(),
        video_source="external_rgb",
        camera_metadata=None if legacy else metadata,
    )
    command = TeleopCommand(
        frame_index=np.arange(1),
        smpl_joints=np.zeros((1, 24, 3)),
        root_quaternion=np.array([[1.0, 0, 0, 0]]),
        joint_position=np.zeros((1, 29)),
    )
    contacts = ContactFrame(
        robot_body_id=np.full(MAX_CONTACTS, -1),
        other_body_id=np.full(MAX_CONTACTS, -1),
        position=np.zeros((MAX_CONTACTS, 3)),
        normal_force=np.zeros(MAX_CONTACTS),
        tangent_force=np.zeros(MAX_CONTACTS),
        normal_impulse=np.zeros(MAX_CONTACTS),
        sample_count=np.zeros(MAX_CONTACTS, int),
        count=0,
    )
    recorder.start()
    with patch("sonic_mujoco.camera_recording.GEOMETRY_CHUNK_SIZE", chunk_size):
        for row in range(19):
            sample = row // 2
            pose = np.eye(4)
            pose[0, 3] = sample / 10
            frames = {
                name: CameraFrame(
                    rgb=np.full((12, 16, 3), 20 + sample * 10 + i, np.uint8),
                    depth_m=np.full((12, 16), sample + 0.5, np.float32),
                    normals_camera=np.tile(
                        np.array([0, 0, -1], np.float32), (12, 16, 1)
                    ),
                    valid_depth=np.ones((12, 16), bool),
                    valid_normals=np.ones((12, 16), bool),
                    camera_to_world=pose.copy(),
                    appearance_camera_to_world=None,
                    timestamp=2 + sample * 0.04,
                )
                for i, name in enumerate(ROBOT_CAMERAS)
            }
            recorder.append(
                time=2 + row * 0.02,
                qpos=np.zeros(3),
                qvel=np.zeros(3),
                ctrl=np.zeros(3),
                command=command,
                token=np.zeros(64),
                action=np.zeros(29),
                controls=PicoControls(),
                contacts=contacts,
                video_frame=frames["ego_view_left"].rgb if legacy else None,
                camera_frames=None if legacy else frames,
            )
        path = recorder.finish()
    return path.parents[2], path


class EpisodeReaderTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)

    def test_roundtrip_resolves_held_geometry_not_row_index(self):
        session, _ = record_session(self.directory)
        reader = EpisodeReader(session)
        sample = reader.read(17)
        self.assertEqual(set(sample.cameras), set(ROBOT_CAMERAS))
        for frame in sample.cameras.values():
            self.assertEqual(frame.sample_index, 8)
            self.assertFalse(frame.updated)
            self.assertAlmostEqual(frame.timestamp, 2.32)
            self.assertAlmostEqual(frame.age_s, 0.02)
            np.testing.assert_array_equal(
                frame.depth_m, np.full((12, 16), 8.5, np.float32)
            )
            self.assertAlmostEqual(frame.camera_to_world[0, 3], 0.8)
            self.assertGreater(frame.rgb.mean(), 90)
            self.assertLess(frame.rgb.mean(), 110)
        report = reader.audit(sample)
        self.assertTrue(report["temporal"]["ok"])
        self.assertEqual(report["cameras"]["ego_view_left"]["capture_count"], 10)
        self.assertEqual(
            report["selected"]["cameras"]["ego_view_left"]["depth_range_m"], [8.5, 8.5]
        )

    def test_lookup_uses_metadata_chunk_size(self):
        session, _ = record_session(self.directory, chunk_size=3)
        reader = EpisodeReader(session)
        self.assertEqual(reader.samples_per_chunk, 3)
        frame = reader.read(17).cameras["ego_view_left"]
        self.assertEqual(frame.sample_index, 8)
        np.testing.assert_allclose(frame.depth_m, 8.5)

    def test_missing_and_corrupt_geometry_fail_actionably(self):
        session, _ = record_session(self.directory)
        path = session / "geometry/episode_000000/chunk_000001.npz"
        path.unlink()
        with self.assertRaisesRegex(EpisodeReadError, "capture 8.*chunk_000001"):
            EpisodeReader(session).read(17)
        path.write_bytes(b"not a geometry archive")
        with self.assertRaisesRegex(EpisodeReadError, "recording finished"):
            EpisodeReader(session).read(17)

    def test_audit_detects_camera_clock_and_sample_corruption(self):
        session, path = record_session(self.directory)
        table = pq.read_table(path)
        for field, value in (("age_s", 0.9), ("sample_index", 4), ("timestamp", 3.0)):
            key = f"observation.camera.ego_view_left.{field}"
            values = table[key].to_pylist()
            values[5] = value
            table = table.set_column(
                table.schema.get_field_index(key), key, pa.array(values)
            )
        pq.write_table(table, path)
        report = EpisodeReader(session).audit()
        self.assertFalse(report["temporal"]["ok"])
        errors = " ".join(error["message"] for error in report["temporal"]["errors"])
        self.assertIn("timestamp/age", errors)
        self.assertIn("sample_index", errors)
        self.assertIn("not synchronized", errors)
        with self.assertRaisesRegex(EpisodeReadError, "Geometry timestamp"):
            EpisodeReader(session).read(5)

    def test_legacy_rgb_only_does_not_invent_geometry(self):
        session, _ = record_session(self.directory, legacy=True)
        reader = EpisodeReader(session)
        frame = reader.read(3)
        self.assertEqual(tuple(frame.cameras), ("ego_view",))
        self.assertIsNone(frame.cameras["ego_view"].depth_m)
        self.assertIsNone(frame.cameras["ego_view"].timestamp)
        report = reader.audit(frame)
        self.assertFalse(report["geometry_available"])
        self.assertIn("RGB-only", report["geometry_note"])

    def test_audit_detects_skipped_control_rows_even_with_increasing_clocks(self):
        session, path = record_session(self.directory, legacy=True)
        table = pq.read_table(path)
        values = table["sim_time"].to_pylist()
        values[5:] = [value + 0.02 for value in values[5:]]
        table = table.set_column(
            table.schema.get_field_index("sim_time"), "sim_time", pa.array(values)
        )
        pq.write_table(table, path)
        report = EpisodeReader(session).audit()
        self.assertFalse(report["temporal"]["ok"])
        self.assertIn("Simulation cadence", str(report["temporal"]["errors"]))

    def test_tactile_summary_and_task_context(self):
        session, path = record_session(self.directory)
        table = pq.read_table(path)
        table = table.append_column(
            "observation.tactile_vest", pa.array([[0, 3, 4]] * 19)
        )
        table = table.append_column(
            "observation.tactile.normal_force_n", pa.array([[1.5, 2.5]] * 19)
        )
        table = table.append_column(
            "simulation.tactile_source_time", pa.array([[2.0, -1.0, 2.2]] * 19)
        )
        pq.write_table(table, path)
        episode = {
            "episode_index": 0,
            "task_context": {"seed": 4},
            "task_outcome": {"success": False},
        }
        (session / "meta/episodes.jsonl").write_text(json.dumps(episode) + "\n")
        reader = EpisodeReader(session)
        report = reader.audit(reader.read(17))
        self.assertEqual(report["task_context"], {"seed": 4})
        self.assertEqual(report["selected"]["tactile"]["vest"]["nonzero_channels"], 2)
        self.assertEqual(
            report["selected"]["tactile"]["physical"]["normal_force_sum_n"], 4
        )
        self.assertIsNone(report["selected"]["tactile"]["device_age_s"][1])

    def test_cli_writes_preview_without_modifying_dataset(self):
        session, _ = record_session(self.directory / "recordings")
        path = Path(__file__).resolve().parents[1] / "scripts/inspect_episode.py"
        spec = importlib.util.spec_from_file_location("inspect_episode", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        before = {
            p: (p.stat().st_size, p.stat().st_mtime_ns)
            for p in session.rglob("*")
            if p.is_file()
        }
        output = self.directory / "inspection"
        module.main([str(session), "--row", "17", "--output", str(output)])
        self.assertTrue((output / "contact_sheet.png").is_file())
        self.assertTrue(
            json.loads((output / "audit.json").read_text())["temporal"]["ok"]
        )
        after = {
            p: (p.stat().st_size, p.stat().st_mtime_ns)
            for p in session.rglob("*")
            if p.is_file()
        }
        self.assertEqual(before, after)
        with self.assertRaises(SystemExit) as caught:
            module.main([str(session), "--output", str(session / "inspection")])
        self.assertEqual(caught.exception.code, 2)

    def test_row_bounds(self):
        session, _ = record_session(self.directory)
        reader = EpisodeReader(session)
        for row in (-1, 19):
            with self.assertRaisesRegex(EpisodeReadError, "outside episode range"):
                reader.read(row)

    def test_unpublished_episode_is_not_read_even_if_files_exist(self):
        session, _ = record_session(self.directory)
        path = session / "meta/info.json"
        info = json.loads(path.read_text())
        info["total_episodes"] = 0
        path.write_text(json.dumps(info))
        with self.assertRaisesRegex(EpisodeReadError, "not published"):
            EpisodeReader(session)

    def test_missing_camera_metadata_does_not_silently_downgrade_to_legacy(self):
        session, _ = record_session(self.directory)
        (session / "meta/cameras.json").unlink()
        with self.assertRaisesRegex(EpisodeReadError, "meta/cameras.json is missing"):
            EpisodeReader(session)
