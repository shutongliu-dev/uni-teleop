import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest


spec = importlib.util.spec_from_file_location(
    "collect_sleeve_calibration",
    Path(__file__).resolve().parents[1] / "scripts/collect_sleeve_calibration.py",
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
BaselineCapture = module.SleeveCapture


class BaselineCaptureTest(unittest.TestCase):
    def test_press_release_cues_and_mount_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            capture = BaselineCapture(Path(directory), 60, {"mount_condition": "flat_on_table"})
            capture.start(100, "press_release")
            self.assertEqual(capture.duration, 30)
            for elapsed, expected in ((0, "release"), (5, "press"), (8, "release"),
                                      (13, "press"), (16, "release"), (21, "press"), (24, "release")):
                self.assertEqual(capture.stage(elapsed)[0], expected)
            capture.finish("interrupted")
            metadata = json.loads((capture.directory / "metadata.json").read_text())
            self.assertEqual(metadata["mount_condition"], "flat_on_table")
            self.assertEqual(metadata["capture_kind"], "press_release")

    def test_raw_values_and_timestamps_survive_without_subtraction(self):
        with tempfile.TemporaryDirectory() as directory:
            capture = BaselineCapture(Path(directory), 30, {})
            capture.start(100)
            sample = SimpleNamespace(
                raw=bytes(range(256)), gyro=bytes(16), sensor_type=2,
                packet1_time=1000, packet2_time=1000.01,
            )
            capture.add(sample, 1000.01, 100.01)
            capture.add(sample, 1030, 130)  # At the boundary: excluded.
            capture.update(130)
            rows = (capture.directory / "samples.jsonl").read_text().splitlines()
            self.assertEqual(len(rows), 1)
            row = json.loads(rows[0])
            self.assertEqual(row["raw"], list(range(256)))
            self.assertEqual(row["packet2_host_unix_s"], 1000.01)
            metadata = json.loads((capture.directory / "metadata.json").read_text())
            self.assertEqual(metadata["status"], "complete")
            self.assertFalse(metadata["baseline_subtracted"])
            self.assertFalse(metadata["quiet_condition_verified"])

    def test_empty_capture_is_not_complete_and_repeat_uses_new_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            capture = BaselineCapture(Path(directory), 30, {})
            capture.start(100)
            first = capture.directory
            capture.update(130)
            self.assertEqual(json.loads((first / "metadata.json").read_text())["status"], "no_data")
            capture.start(131)
            self.assertNotEqual(capture.directory, first)
            capture.finish("interrupted")
            self.assertEqual(
                json.loads((capture.directory / "metadata.json").read_text())["status"],
                "interrupted",
            )
