import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

from sonic_mujoco.envs.mujoco.g1 import MujocoG1EmptyEnv
from sonic_mujoco.tactile_calibration import (
    TactileCalibrationProfile,
    TactileDeviceProfile,
    TactileObservationDeliveryProfile,
    TactileSensorDynamicsProfile,
    TactileTransferProfile,
    UniformRange,
    load_tactile_calibration_profile,
)
from sonic_mujoco.tactile_skin import (
    TACTILE_DEVICE_NAMES,
    JuQiaoSkinLayout,
    JuQiaoTactileAdapter,
)


def _devices(
    *,
    rate: tuple[float, float, float] = (10.0, 10.0, 10.0),
    noise: tuple[float, float, float] = (0.0, 0.0, 0.0),
    dropout: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> tuple[TactileDeviceProfile, ...]:
    return tuple(
        TactileDeviceProfile(
            name=name,
            sample_rate_hz=rate[index],
            stale_timeout_s=0.1,
            noise_std_counts=noise[index],
            dropout_probability=dropout[index],
        )
        for index, name in enumerate(TACTILE_DEVICE_NAMES)
    )


def _profile(
    layout: JuQiaoSkinLayout,
    *,
    devices: tuple[TactileDeviceProfile, ...] | None = None,
    dynamics: TactileSensorDynamicsProfile | None = None,
) -> TactileCalibrationProfile:
    return TactileCalibrationProfile(
        schema_version=2,
        profile_id="sensor-dynamics-test-v2",
        layout_sha256=layout.sha256,
        transfer=TactileTransferProfile(gain=1.0),
        devices=devices or _devices(),
        sensor_dynamics=dynamics or TactileSensorDynamicsProfile(),
    )


class TactileSensorDynamicsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.env = MujocoG1EmptyEnv()
        cls.layout = cls.env.tactile_skin_layout

    @classmethod
    def tearDownClass(cls) -> None:
        cls.env.close()

    def test_garment_spread_conserves_load_without_mutating_input(self) -> None:
        dynamics = TactileSensorDynamicsProfile(
            spatial_first_ring_fraction=UniformRange(0.2, 0.2),
        )
        adapter = JuQiaoTactileAdapter(
            self.layout,
            profile=_profile(self.layout, dynamics=dynamics),
            seed=7,
        )
        normal_force = np.zeros(len(self.layout.taxels))
        # An interior point of the 6x8 front-chest grid has four neighbors.
        normal_force[2 * 8 + 3] = 100.0
        original = normal_force.copy()

        sample = adapter.update_normal_force(normal_force, 0.0)

        np.testing.assert_array_equal(normal_force, original)
        self.assertEqual(int(sample.values[0].sum()), 100)
        self.assertEqual(np.count_nonzero(sample.values[0]), 5)
        self.assertFalse(np.any(sample.values[1:]))

    def test_unwired_vest_slots_stay_zero_with_noise_and_drift(self) -> None:
        dynamics = TactileSensorDynamicsProfile(
            noise_ar1_rho=UniformRange(0.2, 0.2),
            drift_std_counts=UniformRange(1.0, 1.0),
            drift_time_constant_s=UniformRange(1.0, 1.0),
        )
        adapter = JuQiaoTactileAdapter(
            self.layout,
            profile=_profile(
                self.layout,
                devices=_devices(noise=(1.0, 1.0, 1.0)),
                dynamics=dynamics,
            ),
            seed=19,
        )
        force = np.zeros(len(self.layout.taxels))
        wired = self.layout.taxels.channel_id[self.layout.taxels.region_id == 0]
        unwired = np.setdiff1d(np.arange(256), wired)

        for time in np.arange(0.0, 2.01, 0.02):
            sample = adapter.update_normal_force(force, float(time))
            self.assertFalse(np.any(sample.values[0, unwired]))

    def test_device_random_streams_are_isolated(self) -> None:
        dynamics = TactileSensorDynamicsProfile(
            randomize_source_phase=True,
            noise_ar1_rho=UniformRange(0.2, 0.2),
            drift_std_counts=UniformRange(0.6, 0.6),
            drift_time_constant_s=UniformRange(2.0, 2.0),
        )
        baseline = _devices(
            rate=(11.0, 13.0, 15.0),
            noise=(0.5, 0.5, 0.5),
            dropout=(0.2, 0.2, 0.2),
        )
        changed = (
            *baseline[:2],
            replace(
                baseline[2],
                sample_rate_hz=23.0,
                noise_std_counts=4.0,
                dropout_probability=0.95,
            ),
        )
        first = JuQiaoTactileAdapter(
            self.layout,
            profile=_profile(self.layout, devices=baseline, dynamics=dynamics),
            seed=31,
        )
        second = JuQiaoTactileAdapter(
            self.layout,
            profile=_profile(self.layout, devices=changed, dynamics=dynamics),
            seed=31,
        )
        force = np.full(len(self.layout.taxels), 5.0)

        for time in np.arange(0.0, 1.01, 0.01):
            left = first.update_normal_force(force, float(time))
            right = second.update_normal_force(force, float(time))
            np.testing.assert_array_equal(left.values[:2], right.values[:2])
            np.testing.assert_array_equal(left.source_time[:2], right.source_time[:2])
            np.testing.assert_array_equal(left.updated[:2], right.updated[:2])

    def test_dropped_source_still_advances_electronics_state(self) -> None:
        dynamics = TactileSensorDynamicsProfile(
            noise_ar1_rho=UniformRange(0.2, 0.2),
            drift_std_counts=UniformRange(0.5, 0.5),
            drift_time_constant_s=UniformRange(1.0, 1.0),
        )
        adapter = JuQiaoTactileAdapter(
            self.layout,
            profile=_profile(
                self.layout,
                devices=_devices(
                    noise=(0.5, 0.5, 0.5),
                    dropout=(1.0, 0.0, 0.0),
                ),
                dynamics=dynamics,
            ),
            seed=23,
        )
        force = np.zeros(len(self.layout.taxels))

        adapter.update_normal_force(force, 0.0)
        sample = adapter.update_normal_force(force, 0.1)

        self.assertEqual(adapter._noise_states[0].sample_index, 2)
        self.assertEqual(adapter._drift_states[0].advance_index, 1)
        self.assertEqual(sample.source_time[0], -1.0)
        self.assertFalse(np.any(sample.values[0]))

    def test_strictly_earlier_source_cannot_observe_current_force(self) -> None:
        source_time = 0.019999999999
        devices = tuple(
            replace(device, phase_offset_s=source_time) for device in _devices()
        )
        adapter = JuQiaoTactileAdapter(
            self.layout,
            profile=_profile(self.layout, devices=devices),
        )
        zero = np.zeros(len(self.layout.taxels))
        contact = np.full(len(self.layout.taxels), 10.0)

        adapter.update_normal_force(zero, 0.01)
        sample = adapter.update_normal_force(contact, 0.02)

        np.testing.assert_array_equal(sample.source_time, np.full(3, source_time))
        np.testing.assert_array_equal(sample.updated, np.ones(3, dtype=bool))
        self.assertFalse(np.any(sample.values))

    def test_repository_profile_replays_and_describes_provisional_dynamics(
        self,
    ) -> None:
        path = (
            Path(__file__).resolve().parents[1]
            / "configs/tactile/juqiao_g1_sim2real_provisional_v2.json"
        )
        profile = load_tactile_calibration_profile(path)
        first = JuQiaoTactileAdapter(self.layout, profile=profile, seed=42)
        second = JuQiaoTactileAdapter(self.layout, profile=profile, seed=42)
        force = np.linspace(0.0, 8.0, len(self.layout.taxels))

        for time in np.arange(0.005, 0.501, 0.005):
            left = first.update_normal_force(force, float(time))
            right = second.update_normal_force(force, float(time))
            np.testing.assert_array_equal(left.values, right.values)
            np.testing.assert_array_equal(left.source_time, right.source_time)
            np.testing.assert_array_equal(left.updated, right.updated)

        metadata = first.metadata
        self.assertEqual(metadata["schema_version"], 2)
        self.assertEqual(
            metadata["schema_version"], metadata["profile"]["schema_version"]
        )
        self.assertEqual(self.layout.sha256, profile.layout_sha256)
        self.assertEqual(len(metadata["sensor_dynamics"]["topology_sha256"]), 64)
        self.assertEqual(
            metadata["sensor_dynamics"]["status"],
            "provisional_unpaired_sensor_dynamics",
        )
        self.assertEqual(
            metadata["sensor_dynamics"]["baseline_pipeline"]["status"],
            "not_simulated",
        )
        self.assertEqual(
            metadata["sensor_dynamics"]["mount_calibration"]["status"],
            "unverified_absolute_registration",
        )
        self.assertTrue(metadata["episode_profile"]["source_phase_s"])

    def test_observation_delivery_fails_closed_until_separately_identified(
        self,
    ) -> None:
        dynamics = TactileSensorDynamicsProfile(
            observation_delivery=TactileObservationDeliveryProfile()
        )
        with self.assertRaisesRegex(NotImplementedError, "observation_delivery"):
            JuQiaoTactileAdapter(
                self.layout,
                profile=_profile(self.layout, dynamics=dynamics),
            )


if __name__ == "__main__":
    unittest.main()
