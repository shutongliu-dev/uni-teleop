import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from sonic_mujoco.tactile_audit import (
    DeviceAuditAccumulator,
    audit_tactile_sources,
)


class TactileAuditTest(unittest.TestCase):
    def test_accumulator_reports_exact_packet_statistics(self) -> None:
        packets = np.zeros((5, 256), dtype=np.uint8)
        packets[1:3, 4] = 3
        packets[3:, 4] = 7
        accumulator = DeviceAuditAccumulator(deadband_counts=2)
        accumulator.begin_episode()
        accumulator.update(packets[:3])
        accumulator.update(packets[3:])
        threshold = accumulator.quiet_threshold(0.5)
        accumulator.update_quiet(packets, threshold)

        report = accumulator.report(fps=50.0, quiet_quantile=0.5)

        self.assertEqual(report["frames"], 5)
        self.assertAlmostEqual(report["repeat_fraction"], 0.5)
        self.assertAlmostEqual(report["observed_change_rate_hz_lower_bound"], 25.0)
        self.assertAlmostEqual(report["zero_packet_fraction"], 0.2)
        self.assertEqual(report["positive_count_quantiles"]["p50"], 3.0)
        self.assertEqual(report["active_channel_quantiles"]["p90"], 1.0)
        self.assertEqual(report["quiet_packet_sum_maximum"], 3)
        self.assertEqual(
            report["quiet_channel_mean_counts_including_zero"][4],
            2.0,
        )
        self.assertEqual(report["quiet_nonzero_channel_mean_counts"][4], 3.0)

    def test_quiet_statistics_do_not_hide_zero_packets(self) -> None:
        packets = np.zeros((100, 256), dtype=np.uint8)
        packets[-1, 7] = 1
        accumulator = DeviceAuditAccumulator()
        accumulator.update(packets)
        threshold = accumulator.quiet_threshold(0.2)
        accumulator.update_quiet(packets, threshold)

        report = accumulator.report(fps=50.0, quiet_quantile=0.2)

        self.assertEqual(report["quiet_frames_including_zero"], 100)
        self.assertEqual(report["quiet_nonzero_frames"], 1)
        self.assertAlmostEqual(
            report["quiet_channel_mean_counts_including_zero"][7],
            0.01,
        )
        self.assertEqual(report["quiet_nonzero_channel_mean_counts"][7], 1.0)

    def test_single_parquet_requires_an_explicit_record_rate(self) -> None:
        try:
            import pyarrow as pa
            from pyarrow import parquet
        except ImportError:
            self.skipTest("pyarrow is not installed")

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "packets.parquet"
            flat = pa.array(np.zeros(256, dtype=np.uint8), type=pa.uint8())
            packet = pa.FixedSizeListArray.from_arrays(flat, 256)
            parquet.write_table(
                pa.table(
                    {
                        column: packet
                        for column in (
                            "observation.tactile_vest",
                            "observation.tactile_left_arm",
                            "observation.tactile_right_arm",
                        )
                    }
                ),
                path,
            )

            with self.assertRaisesRegex(ValueError, "pass fps explicitly"):
                audit_tactile_sources([path])

            report = audit_tactile_sources([path], fps=20.0)
            self.assertEqual(report["sources"][0]["fps"], 20.0)

            int16_path = Path(directory) / "wrong-dtype.parquet"
            int16_flat = pa.array(np.zeros(256, dtype=np.int16), type=pa.int16())
            int16_packet = pa.FixedSizeListArray.from_arrays(int16_flat, 256)
            parquet.write_table(
                pa.table(
                    {
                        column: int16_packet
                        for column in (
                            "observation.tactile_vest",
                            "observation.tactile_left_arm",
                            "observation.tactile_right_arm",
                        )
                    }
                ),
                int16_path,
            )
            with self.assertRaisesRegex(ValueError, "list<uint8>"):
                audit_tactile_sources([int16_path], fps=20.0)

            variable_path = Path(directory) / "real-compatible-list.parquet"
            variable_packet = pa.array(
                [np.zeros(256, dtype=np.uint8).tolist()],
                type=pa.list_(pa.uint8()),
            )
            parquet.write_table(
                pa.table(
                    {
                        column: variable_packet
                        for column in (
                            "observation.tactile_vest",
                            "observation.tactile_left_arm",
                            "observation.tactile_right_arm",
                        )
                    }
                ),
                variable_path,
            )
            variable_report = audit_tactile_sources([variable_path], fps=20.0)
            self.assertEqual(
                variable_report["sources"][0]["devices"]["vest"]["frames"], 1
            )

    def test_episode_boundaries_are_not_counted_as_updates(self) -> None:
        packet = np.ones((1, 256), dtype=np.uint8)
        accumulator = DeviceAuditAccumulator()
        accumulator.begin_episode()
        accumulator.update(packet)
        accumulator.begin_episode()
        accumulator.update(packet)

        report = accumulator.report(fps=50.0, quiet_quantile=0.2)

        self.assertEqual(report["repeat_fraction"], 0.0)
        self.assertEqual(report["observed_change_rate_hz_lower_bound"], 0.0)

    def test_non_integer_packets_are_rejected_instead_of_truncated(self) -> None:
        accumulator = DeviceAuditAccumulator()

        with self.assertRaisesRegex(ValueError, "uint8"):
            accumulator.update(np.full((1, 256), 1.5))

    def test_dataset_audit_omits_paths_and_skips_discarded_episodes(self) -> None:
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError:
            self.skipTest("pyarrow is not installed")

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "private_operator_name"
            data = root / "data" / "chunk-000"
            meta = root / "meta"
            data.mkdir(parents=True)
            meta.mkdir()
            (meta / "info.json").write_text(
                json.dumps(
                    {
                        "fps": 50,
                        "discarded_episode_indices": [1],
                    }
                )
            )
            for episode, value in ((0, 4), (1, 99)):
                columns = {}
                for device in ("vest", "left_arm", "right_arm"):
                    flat = pa.array(
                        np.full(3 * 256, value, dtype=np.uint8), type=pa.uint8()
                    )
                    columns[f"observation.tactile_{device}"] = (
                        pa.FixedSizeListArray.from_arrays(flat, 256)
                    )
                pq.write_table(
                    pa.table(columns), data / f"episode_{episode:06d}.parquet"
                )

            report = audit_tactile_sources([root], labels=["clean-real"])
            encoded = json.dumps(report)

            self.assertNotIn(str(root), encoded)
            self.assertNotIn("private_operator_name", encoded)
            source = report["sources"][0]
            self.assertEqual(source["label"], "clean-real")
            self.assertEqual(source["discarded_episode_files_skipped"], 1)
            self.assertEqual(source["devices"]["vest"]["frames"], 3)
            self.assertEqual(
                source["devices"]["vest"]["positive_count_quantiles"]["p50"],
                4.0,
            )
            self.assertIn("cannot determine", report["interpretation"]["force_mapping"])


if __name__ == "__main__":
    unittest.main()
