import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from sonic_mujoco.tactile import TactileFrame, TaxelLayout
from sonic_mujoco.tactile_calibration import (
    FORCE_MAPPING_STATUS,
    TactileCalibrationProfile,
    TactileDeviceProfile,
    TactileEpisodeRandomization,
    TactileObservationDeliveryProfile,
    TactileProfileSampler,
    TactileSensorDynamicsProfile,
    TactileTransferProfile,
    UniformRange,
    asymmetric_first_order_filter,
    tactile_input,
)
from sonic_mujoco.tactile_skin import (
    TACTILE_DEVICE_NAMES,
    JuQiaoSkinLayout,
    JuQiaoTactileAdapter,
    SleeveMount,
)


def _skin_layout() -> JuQiaoSkinLayout:
    region_id = np.concatenate(
        (
            np.zeros(112, dtype=np.int16),
            np.ones(256, dtype=np.int16),
            np.full(256, 2, dtype=np.int16),
        )
    )
    channel_id = np.concatenate(
        (
            np.arange(112, dtype=np.int32),
            np.arange(256, dtype=np.int32),
            np.arange(256, dtype=np.int32),
        )
    )
    count = len(region_id)
    normals = np.zeros((count, 3), dtype=np.float64)
    normals[:, 0] = 1.0
    taxels = TaxelLayout(
        body_id=np.zeros(count, dtype=np.int32),
        geom_id=np.zeros(count, dtype=np.int32),
        local_center=np.zeros((count, 3), dtype=np.float64),
        local_normal=normals,
        region_id=region_id,
        channel_id=channel_id,
        region_names=TACTILE_DEVICE_NAMES,
    )
    return JuQiaoSkinLayout(
        taxels=taxels,
        area_m2=np.full(count, 0.002, dtype=np.float64),
        mapping_radius_m=np.full(count, 0.04, dtype=np.float64),
        subregion=tuple("test" for _ in range(count)),
        body_name=tuple("test_body" for _ in range(count)),
        left_mount=SleeveMount(),
        right_mount=SleeveMount(),
    )


def _frame(layout: TaxelLayout, normal_force: np.ndarray) -> TactileFrame:
    force = -normal_force[:, None] * layout.local_normal
    vectors = np.zeros((len(layout), 3), dtype=np.float64)
    scalars = np.zeros(len(layout), dtype=np.float64)
    return TactileFrame(
        layout=layout,
        force=force,
        normal_force=normal_force,
        tangent_force=vectors.copy(),
        impulse=vectors.copy(),
        normal_impulse=scalars.copy(),
        tangent_impulse=vectors.copy(),
        contact=normal_force > 0.0,
        other_body_id=np.full(len(layout), -1, dtype=np.int32),
        other_geom_id=np.full(len(layout), -1, dtype=np.int32),
        sample_count=np.zeros(len(layout), dtype=np.int32),
        duration=0.02,
        mapped_contact_count=0,
        unmapped_contact_count=0,
    )


def _devices(
    *,
    rates: tuple[float, float, float] = (14.0, 14.0, 14.0),
    stale: tuple[float, float, float] = (0.1, 0.1, 0.1),
    dropout: tuple[float, float, float] = (0.0, 0.0, 0.0),
    noise: tuple[float, float, float] = (0.0, 0.0, 0.0),
    phases: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> tuple[TactileDeviceProfile, ...]:
    return tuple(
        TactileDeviceProfile(
            name=name,
            sample_rate_hz=rates[index],
            stale_timeout_s=stale[index],
            dropout_probability=dropout[index],
            noise_std_counts=noise[index],
            phase_offset_s=phases[index],
        )
        for index, name in enumerate(TACTILE_DEVICE_NAMES)
    )


def _profile(
    skin: JuQiaoSkinLayout,
    *,
    transfer: TactileTransferProfile | None = None,
    devices: tuple[TactileDeviceProfile, ...] | None = None,
    randomization: TactileEpisodeRandomization | None = None,
    fixed_gain_variation: float = 0.0,
    fixed_rate_variation: float = 0.0,
    schema_version: int = 1,
) -> TactileCalibrationProfile:
    return TactileCalibrationProfile(
        schema_version=schema_version,
        profile_id=f"test-provisional-v{schema_version}",
        layout_sha256=skin.sha256,
        transfer=transfer or TactileTransferProfile(),
        devices=devices or _devices(),
        randomization=randomization or TactileEpisodeRandomization(),
        fixed_taxel_gain_variation=fixed_gain_variation,
        fixed_device_rate_variation=fixed_rate_variation,
    )


class TactileCalibrationProfileTest(unittest.TestCase):
    def setUp(self) -> None:
        self.skin = _skin_layout()

    def test_profile_round_trip_file_hash_and_strict_rejection(self) -> None:
        profile = _profile(self.skin)
        payload = profile.as_dict()
        parsed = TactileCalibrationProfile.from_dict(payload)

        self.assertEqual(parsed.as_dict(), payload)
        self.assertEqual(parsed.sha256, profile.sha256)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "profile.json"
            raw = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
            path.write_bytes(raw)
            loaded = TactileCalibrationProfile.from_file(path)

            self.assertEqual(loaded.as_dict(), payload)
            self.assertEqual(
                loaded.source_file_sha256,
                hashlib.sha256(raw).hexdigest(),
            )
            self.assertNotIn(str(path), json.dumps(loaded.as_dict()))

            nan_payload = json.loads(json.dumps(payload))
            nan_payload["transfer"]["gain"] = float("nan")
            path.write_text(json.dumps(nan_payload))
            with self.assertRaisesRegex(ValueError, "valid UTF-8 JSON"):
                TactileCalibrationProfile.from_file(path)

            path.write_text('{"schema_version": 1, "schema_version": 2}')
            with self.assertRaisesRegex(ValueError, "valid UTF-8 JSON"):
                TactileCalibrationProfile.from_file(path)

        unknown = dict(payload)
        unknown["unexpected"] = True
        with self.assertRaisesRegex(ValueError, "unknown fields"):
            TactileCalibrationProfile.from_dict(unknown)

        missing = dict(payload)
        del missing["schema_version"]
        with self.assertRaisesRegex(ValueError, "missing fields"):
            TactileCalibrationProfile.from_dict(missing)

        wrong_order = json.loads(json.dumps(payload))
        wrong_order["devices"][0], wrong_order["devices"][1] = (
            wrong_order["devices"][1],
            wrong_order["devices"][0],
        )
        with self.assertRaisesRegex(ValueError, "device order"):
            JuQiaoTactileAdapter(
                self.skin,
                profile=TactileCalibrationProfile.from_dict(wrong_order),
            )

        with self.assertRaisesRegex(ValueError, "must not exceed 1000"):
            TactileDeviceProfile(name="vest", sample_rate_hz=1000.1)

        with self.assertRaisesRegex(ValueError, "realized tactile sample rate"):
            _profile(
                self.skin,
                randomization=TactileEpisodeRandomization(
                    sample_rate_scale=UniformRange(1.0, 1e308)
                ),
            )

    def test_v2_sensor_dynamics_are_strict_and_seeded(self) -> None:
        dynamics = TactileSensorDynamicsProfile(
            spatial_first_ring_fraction=UniformRange(0.10, 0.20),
            spatial_second_ring_fraction=UniformRange(0.01, 0.04),
            randomize_source_phase=True,
            noise_ar1_rho=UniformRange(0.10, 0.30),
            drift_std_counts=UniformRange(0.4, 0.9),
            drift_time_constant_s=UniformRange(15.0, 60.0),
            observation_delivery=TactileObservationDeliveryProfile(
                initial_probabilities=(0.2, 0.8),
                transition_probabilities=((0.7, 0.3), (0.2, 0.8)),
            ),
        )
        profile = TactileCalibrationProfile(
            schema_version=2,
            profile_id="test-provisional-v2",
            layout_sha256=self.skin.sha256,
            devices=_devices(),
            sensor_dynamics=dynamics,
        )

        parsed = TactileCalibrationProfile.from_dict(profile.as_dict())
        self.assertEqual(parsed.as_dict(), profile.as_dict())
        first, _ = TactileProfileSampler(profile, len(self.skin.taxels)).sample(seed=42)
        replay, _ = TactileProfileSampler(profile, len(self.skin.taxels)).sample(
            seed=42
        )
        self.assertEqual(first.sha256, replay.sha256)
        np.testing.assert_array_equal(first.source_phase_s, replay.source_phase_s)
        self.assertTrue(np.all(first.source_phase_s >= 0.0))
        self.assertTrue(np.all(first.source_phase_s < 1.0 / 14.0))
        self.assertGreaterEqual(first.spatial_first_ring_fraction, 0.10)
        self.assertLessEqual(first.spatial_first_ring_fraction, 0.20)

        unknown = profile.as_dict()
        unknown["sensor_dynamics"]["unknown"] = True
        with self.assertRaisesRegex(ValueError, "unknown fields"):
            TactileCalibrationProfile.from_dict(unknown)

        with self.assertRaisesRegex(ValueError, "sum to <= 1"):
            TactileSensorDynamicsProfile(
                spatial_first_ring_fraction=UniformRange(0.8, 0.8),
                spatial_second_ring_fraction=UniformRange(0.3, 0.3),
            )

    def test_pressure_input_uses_taxel_area_without_mutating_force(self) -> None:
        force = np.array([2.0, 4.0, -1.0])
        area = np.array([0.002, 0.004, 0.001])
        original = force.copy()

        pressure = tactile_input(force, area, "pressure")

        np.testing.assert_allclose(pressure, np.array([1.0, 1.0, 0.0]))
        np.testing.assert_array_equal(force, original)
        np.testing.assert_array_equal(tactile_input(force, area, "force"), [2, 4, 0])

    def test_attack_and_release_use_distinct_exact_time_constants(self) -> None:
        initial = np.zeros(2)
        target = np.full(2, 10.0)
        attacked = asymmetric_first_order_filter(
            initial,
            target,
            0.1,
            attack_tau_s=0.1,
            release_tau_s=0.2,
        )
        released = asymmetric_first_order_filter(
            attacked,
            initial,
            0.1,
            attack_tau_s=0.1,
            release_tau_s=0.2,
        )

        np.testing.assert_allclose(attacked, 10.0 * (1.0 - np.exp(-1.0)))
        np.testing.assert_allclose(released, attacked * np.exp(-0.5))

    def test_same_explicit_seed_reproduces_realization_and_packets(self) -> None:
        randomization = TactileEpisodeRandomization(
            gain_scale=UniformRange(0.8, 1.2),
            sample_rate_scale=UniformRange(0.9, 1.1),
            noise_std_counts=UniformRange(0.2, 0.8),
            preload=UniformRange(0.0, 0.2),
            residual_bias_counts=UniformRange(0.0, 2.0),
        )
        profile = _profile(
            self.skin,
            devices=_devices(rates=(20.0, 25.0, 30.0), noise=(0.5, 0.5, 0.5)),
            randomization=randomization,
            fixed_gain_variation=0.05,
            fixed_rate_variation=0.04,
        )
        adapter = JuQiaoTactileAdapter(self.skin, profile=profile, seed=31)
        frame = _frame(
            self.skin.taxels,
            np.linspace(0.0, 12.0, len(self.skin.taxels)),
        )

        def run() -> tuple[dict[str, object], list[tuple[np.ndarray, ...]]]:
            adapter.reset(seed=918)
            episode = adapter.metadata["episode_profile"]
            samples = []
            for time in (0.0, 0.02, 0.05, 0.10, 0.17):
                sample = adapter.update(frame, time)
                samples.append(
                    (
                        sample.values.copy(),
                        sample.source_time.copy(),
                        sample.updated.copy(),
                    )
                )
            return episode, samples

        first_episode, first_packets = run()
        second_episode, second_packets = run()

        self.assertNotEqual(first_episode["index"], second_episode["index"])
        self.assertEqual(first_episode["sha256"], second_episode["sha256"])
        for first, second in zip(first_packets, second_packets):
            for first_value, second_value in zip(first, second):
                np.testing.assert_array_equal(first_value, second_value)

        episode_index = adapter.metadata["episode_profile"]["index"]
        for invalid_seed in (-1, 1 << 64):
            with self.assertRaisesRegex(ValueError, "uint64"):
                adapter.reset(seed=invalid_seed)
        self.assertEqual(
            adapter.metadata["episode_profile"]["index"],
            episode_index,
        )

    def test_dropout_holds_then_stale_clears_one_device_independently(self) -> None:
        profile = _profile(
            self.skin,
            transfer=TactileTransferProfile(gain=1.0),
            devices=_devices(
                rates=(10.0, 10.0, 10.0),
                stale=(0.15, 1.0, 1.0),
                dropout=(0.5, 0.0, 0.0),
                phases=(0.0, 100.0, 100.0),
            ),
        )
        adapter = JuQiaoTactileAdapter(self.skin, profile=profile, seed=0)
        adapter.reset(seed=0)
        frame = _frame(self.skin.taxels, np.full(len(self.skin.taxels), 7.0))

        initial = adapter.update(frame, 0.0)
        held = adapter.update(frame, 0.05)
        first_drop = adapter.update(frame, 0.1)
        stale = adapter.update(frame, 0.2)
        recovered = adapter.update(frame, 0.41)

        self.assertTrue(initial.updated[0])
        self.assertTrue(np.any(initial.values[0]))
        self.assertFalse(held.updated[0])
        np.testing.assert_array_equal(held.values[0], initial.values[0])
        self.assertFalse(first_drop.updated[0])
        np.testing.assert_array_equal(first_drop.values[0], initial.values[0])
        self.assertFalse(stale.updated[0])
        self.assertFalse(np.any(stale.values[0]))
        self.assertTrue(recovered.updated[0])
        self.assertTrue(np.any(recovered.values[0]))
        self.assertTrue(np.all(initial.source_time[1:] == -1.0))

    def test_residual_bias_is_post_transfer_and_unwired_vest_stays_zero(self) -> None:
        randomization = TactileEpisodeRandomization(
            preload=UniformRange(2.0, 2.0),
            residual_bias_counts=UniformRange(3.0, 3.0),
        )
        profile = _profile(
            self.skin,
            transfer=TactileTransferProfile(gain=1.0),
            randomization=randomization,
        )
        adapter = JuQiaoTactileAdapter(self.skin, profile=profile, seed=7)
        frame = _frame(self.skin.taxels, np.zeros(len(self.skin.taxels)))
        physical_force = frame.force.copy()
        physical_impulse = frame.impulse.copy()

        sample = adapter.update(frame, 0.0)
        wired = self.skin.taxels.channel_id[self.skin.taxels.region_id == 0]
        unwired = np.setdiff1d(np.arange(256), wired)

        self.assertEqual(sample.values.shape, (3, 256))
        self.assertEqual(sample.values.dtype, np.uint8)
        self.assertTrue(np.all(sample.values[0, wired] == 5))
        self.assertTrue(np.all(sample.values[0, unwired] == 0))
        np.testing.assert_array_equal(frame.force, physical_force)
        np.testing.assert_array_equal(frame.impulse, physical_impulse)
        self.assertEqual(adapter.metadata["calibration_status"], FORCE_MAPPING_STATUS)
        self.assertFalse(adapter.metadata["paired_force_calibrated"])
        self.assertEqual(
            adapter.metadata["physical_preload_input"]["identifiability"],
            "not_identifiable_from_post_baseline_quiet_frames",
        )
        self.assertEqual(
            adapter.metadata["post_baseline_residual_count"]["identifiability"],
            "estimable_from_post_baseline_quiet_frames",
        )

    def test_filtered_response_is_sampled_at_the_reported_device_time(self) -> None:
        profile = _profile(
            self.skin,
            transfer=TactileTransferProfile(gain=1.0, attack_tau_s=0.1),
            devices=_devices(phases=(0.05, 100.0, 100.0)),
        )
        adapter = JuQiaoTactileAdapter(self.skin, profile=profile, seed=3)
        frame = _frame(self.skin.taxels, np.full(len(self.skin.taxels), 10.0))

        sample = adapter.update(frame, 0.1)

        self.assertEqual(sample.source_time[0], 0.05)
        self.assertTrue(np.all(sample.values[0, :112] == 4))

    def test_packet_events_do_not_depend_on_update_batching(self) -> None:
        profile = _profile(
            self.skin,
            transfer=TactileTransferProfile(gain=1.0, attack_tau_s=0.1),
            devices=_devices(
                rates=(13.0, 14.0, 15.0),
                dropout=(0.2, 0.2, 0.2),
                noise=(0.5, 0.5, 0.5),
                phases=(0.01, 0.02, 0.03),
            ),
        )
        frame = _frame(self.skin.taxels, np.full(len(self.skin.taxels), 10.0))
        coarse = JuQiaoTactileAdapter(self.skin, profile=profile, seed=11)
        fine = JuQiaoTactileAdapter(self.skin, profile=profile, seed=11)

        coarse_sample = coarse.update(frame, 0.3)
        for time in np.arange(0.01, 0.301, 0.01):
            fine_sample = fine.update(frame, float(time))

        np.testing.assert_array_equal(coarse_sample.values, fine_sample.values)
        np.testing.assert_allclose(coarse_sample.source_time, fine_sample.source_time)

    def test_default_v1_constant_input_bitstream_is_unchanged(self) -> None:
        adapter = JuQiaoTactileAdapter(self.skin, seed=0)
        frame = _frame(
            self.skin.taxels,
            np.full(len(self.skin.taxels), 11.0),
        )
        packets = []

        for time in np.arange(0.0, 0.301, 0.02):
            packets.append(adapter.update(frame, float(time)).values.tobytes())

        self.assertEqual(
            hashlib.sha256(b"".join(packets)).hexdigest(),
            "dd1c72d46da79b68814581646126eb185d753cfe047c4df91589436f2727adc6",
        )

    def test_force_step_cannot_reach_an_earlier_device_sample(self) -> None:
        profile = _profile(
            self.skin,
            schema_version=2,
            transfer=TactileTransferProfile(gain=1.0),
            devices=_devices(
                rates=(10.0, 10.0, 10.0),
                phases=(0.015, 100.0, 100.0),
            ),
        )
        adapter = JuQiaoTactileAdapter(self.skin, profile=profile, seed=5)
        zero = np.zeros(len(self.skin.taxels))
        force = np.full(len(self.skin.taxels), 10.0)

        adapter.update_normal_force(zero, 0.01)
        before_step = adapter.update_normal_force(force, 0.02)

        self.assertTrue(before_step.updated[0])
        self.assertEqual(before_step.source_time[0], 0.015)
        self.assertFalse(np.any(before_step.values[0]))

        after_step = adapter.update_normal_force(force, 0.12)
        self.assertEqual(after_step.source_time[0], 0.115)
        self.assertTrue(np.all(after_step.values[0, :112] == 10))

    def test_constant_causal_trajectory_is_bit_exact_across_batching(self) -> None:
        profile = _profile(
            self.skin,
            schema_version=2,
            transfer=TactileTransferProfile(gain=1.0, attack_tau_s=0.1),
            devices=_devices(
                rates=(13.0, 14.0, 15.0),
                dropout=(0.2, 0.2, 0.2),
                noise=(0.5, 0.5, 0.5),
                phases=(0.01, 0.02, 0.03),
            ),
        )
        force = np.full(len(self.skin.taxels), 10.0)
        coarse = JuQiaoTactileAdapter(self.skin, profile=profile, seed=11)
        fine = JuQiaoTactileAdapter(self.skin, profile=profile, seed=11)

        coarse_updated = coarse.update_normal_force(force, 0.0).updated.copy()
        coarse_sample = coarse.update_normal_force(force, 0.3)
        coarse_updated |= coarse_sample.updated
        coarse_sample = coarse.snapshot(updated=coarse_updated)

        fine_updated = fine.update_normal_force(force, 0.0).updated.copy()
        for time in np.arange(0.01, 0.301, 0.01):
            fine_sample = fine.update_normal_force(force, float(time))
            fine_updated |= fine_sample.updated
        fine_sample = fine.snapshot(updated=fine_updated)

        np.testing.assert_array_equal(coarse_sample.values, fine_sample.values)
        np.testing.assert_array_equal(
            coarse_sample.source_time,
            fine_sample.source_time,
        )
        np.testing.assert_array_equal(coarse_sample.updated, fine_sample.updated)

    def test_history_or_merges_device_updates(self) -> None:
        profile = _profile(
            self.skin,
            schema_version=2,
            devices=_devices(
                rates=(10.0, 10.0, 10.0),
                phases=(0.005, 0.015, 100.0),
            ),
        )
        adapter = JuQiaoTactileAdapter(self.skin, profile=profile, seed=2)
        force = np.zeros(len(self.skin.taxels))

        sample = adapter.update_normal_force_history(((0.01, force), (0.02, force)))

        np.testing.assert_array_equal(sample.updated, (True, True, False))

    def test_excessive_sample_backlog_is_rejected(self) -> None:
        profile = _profile(
            self.skin,
            devices=_devices(rates=(1000.0, 1000.0, 1000.0)),
        )
        adapter = JuQiaoTactileAdapter(self.skin, profile=profile, seed=5)
        frame = _frame(self.skin.taxels, np.zeros(len(self.skin.taxels)))

        with self.assertRaisesRegex(RuntimeError, "too many tactile samples"):
            adapter.update(frame, 2.0)


if __name__ == "__main__":
    unittest.main()
