"""JuQiao skin-suit layout and hardware-compatible tactile sampling."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from itertools import pairwise
from typing import Literal

import mujoco
import numpy as np

from .tactile import TactileFrame, TaxelLayout
from .tactile_calibration import (
    FORCE_MAPPING_STATUS,
    SENSOR_DYNAMICS_STATUS,
    TactileCalibrationProfile,
    TactileEpisodeRandomization,
    TactileInputQuantity,
    TactileProfileSampler,
    asymmetric_first_order_filter,
    make_legacy_tactile_profile,
    power_law_response,
    quantize_counts,
    tactile_input,
)
from .tactile_dynamics import AR1NoiseState, SignedOUDriftState
from .tactile_spatial import GarmentLoadSpreadKernel

TACTILE_DEVICE_NAMES = ("vest", "left_arm", "right_arm")
TACTILE_DEVICE_DIM = 256
TACTILE_VALID_COUNT = 624
_MAPPING_RADIUS_MIN_M = 0.03
_MAPPING_RADIUS_MAX_M = 0.06
_MAPPING_RADIUS_MARGIN_M = 0.01
_MAX_SAMPLE_EVENTS_PER_UPDATE = 4096
_SAMPLE_TIME_DECIMALS = 12
_AR1_RNG_TAG = 0x415231
_DRIFT_RNG_TAG = 0x4F5521
_DROPOUT_RNG_TAG = 0x44524F50
_SLEEVE_RAW_CHANNELS = np.asarray(
    tuple(range(128, TACTILE_DEVICE_DIM)) + tuple(range(128)),
    dtype=np.int32,
).reshape(16, 16)
_SLEEVE_RAW_CHANNELS.setflags(write=False)
# Preserve the legacy per-channel gain draw after moving the vest sidecar from
# its historical shoulder-first order to the vendor's canonical arm-first order.
_LEGACY_FIXED_GAIN_INDEX = np.r_[
    np.arange(88),
    np.arange(92, 100),
    np.arange(88, 92),
    np.arange(104, 112),
    np.arange(100, 104),
    np.arange(112, TACTILE_VALID_COUNT),
]
_LEGACY_FIXED_GAIN_INDEX.setflags(write=False)


def _channels(*rows: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(channel for row in rows for channel in row)


# The vest controller emits a sparse 256-byte packet. Each row below is a
# physical garment row; values are the vendor's one-based electrical slots.
_FRONT_CHEST_CHANNELS = _channels(
    (195, 211, 227, 243, 3, 19, 35, 51),
    (196, 212, 228, 244, 4, 20, 36, 52),
    (197, 213, 229, 245, 5, 21, 37, 53),
    (198, 214, 230, 246, 6, 22, 38, 54),
    (199, 215, 231, 247, 7, 23, 39, 55),
    (200, 216, 232, 248, 8, 24, 40, 56),
)
_BACK_CHANNELS = _channels(
    (58, 42, 26, 10, 250, 234, 218, 202),
    (59, 43, 27, 11, 251, 235, 219, 203),
    (60, 44, 28, 12, 252, 236, 220, 204),
    (61, 45, 29, 13, 253, 237, 221, 205),
    (62, 46, 30, 14, 254, 238, 222, 206),
)
_VEST_REGIONS = (
    ("front_chest", 6, 8, _FRONT_CHEST_CHANNELS),
    ("back", 5, 8, _BACK_CHANNELS),
    ("left_arm", 2, 4, (79, 95, 111, 127, 80, 96, 112, 128)),
    ("left_shoulder", 1, 4, (9, 25, 41, 57)),
    ("right_arm", 2, 4, (177, 162, 146, 130, 178, 161, 145, 129)),
    ("right_shoulder", 1, 4, (249, 233, 217, 201)),
)

_TORSO_PROFILE = (
    (0.020, 0.078, 0.112),
    (0.060, 0.082, 0.120),
    (0.100, 0.086, 0.128),
    (0.140, 0.090, 0.136),
    (0.180, 0.093, 0.143),
    (0.220, 0.095, 0.145),
    (0.260, 0.096, 0.150),
    (0.290, 0.094, 0.155),
    (0.310, 0.091, 0.158),
    (0.328, 0.086, 0.160),
    (0.342, 0.080, 0.160),
)


@dataclass(frozen=True, slots=True)
class SleeveMount:
    """Installation-dependent mapping from a raw 16x16 sleeve to the arm."""

    axial_axis: Literal["rows", "cols"] = "rows"
    flip_axial: bool = False
    flip_circular: bool = False
    circular_shift: int = 0

    def __post_init__(self) -> None:
        if self.axial_axis not in {"rows", "cols"}:
            raise ValueError("sleeve axial_axis must be 'rows' or 'cols'")
        if not isinstance(self.circular_shift, int):
            raise TypeError("sleeve circular_shift must be an integer")

    def physical_indices(self, row: int, column: int) -> tuple[int, int]:
        axial, circular = (row, column) if self.axial_axis == "rows" else (column, row)
        if self.flip_axial:
            axial = 15 - axial
        if self.flip_circular:
            circular = 15 - circular
        return axial, (circular + self.circular_shift) % 16

    def as_dict(self) -> dict[str, object]:
        return {
            "axial_axis": self.axial_axis,
            "flip_axial": self.flip_axial,
            "flip_circular": self.flip_circular,
            "circular_shift": self.circular_shift,
        }


@dataclass(frozen=True, slots=True)
class JuQiaoSkinLayout:
    """Exact 624 wired channels embedded on the MuJoCo G1 upper body."""

    taxels: TaxelLayout
    area_m2: np.ndarray
    mapping_radius_m: np.ndarray
    subregion: tuple[str, ...]
    body_name: tuple[str, ...]
    left_mount: SleeveMount
    right_mount: SleeveMount

    @property
    def sha256(self) -> str:
        payload = self.as_dict(include_taxels=False)
        # Calibration status is provenance, not part of the spatial layout.
        payload.pop("mount_calibration")
        payload["taxels"] = self._taxel_records(include_runtime_ids=False)
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(canonical).hexdigest()

    def matches_taxels(self, layout: TaxelLayout) -> bool:
        """Return whether a physical frame uses this exact ordered layout."""

        return _same_layout(layout, self.taxels)

    def as_dict(self, *, include_taxels: bool = True) -> dict[str, object]:
        result: dict[str, object] = {
            "schema_version": 2,
            "sha256_semantics": "scene_invariant_hardware_layout",
            "hardware": "JuQiao V1.0 triple-device tactile skin",
            "devices": list(TACTILE_DEVICE_NAMES),
            "device_shape": [TACTILE_DEVICE_DIM],
            "device_dtype": "uint8",
            "wired_taxels": len(self.taxels),
            "vest_wired_taxels": 112,
            "left_mount": self.left_mount.as_dict(),
            "right_mount": self.right_mount.as_dict(),
            "mount_calibration": {
                "status": "unverified_absolute_registration",
                "left_verified": False,
                "right_verified": False,
                "required_fixture": (
                    "point_press_proximal_distal_and_circular_seam_per_sleeve"
                ),
            },
            "coordinate_frame": "owning_mujoco_body_local",
            "contact_routing": "all_collidable_geoms_on_owning_body",
            "mapping_radius_m": {
                "minimum": float(np.min(self.mapping_radius_m)),
                "maximum": float(np.max(self.mapping_radius_m)),
                "rule": "nearest_collision_vertex_plus_10mm",
            },
            "channel_indexing": "zero_based_raw_device_slot",
            "source": (
                "GR00T-WholeBodyControl/JuQiao hardware mapping and "
                "G1 tactile garment geometry"
            ),
        }
        if include_taxels:
            result["taxels"] = self._taxel_records(include_runtime_ids=True)
        return result

    def _taxel_records(self, *, include_runtime_ids: bool) -> list[dict[str, object]]:
        records = []
        for index in range(len(self.taxels)):
            record: dict[str, object] = {
                "index": index,
                "device": self.taxels.region_names[int(self.taxels.region_id[index])],
                "channel": int(self.taxels.channel_id[index]),
                "subregion": self.subregion[index],
                "body_name": self.body_name[index],
                "center_m": self.taxels.local_center[index].tolist(),
                "normal": self.taxels.local_normal[index].tolist(),
                "area_m2": float(self.area_m2[index]),
                "mapping_radius_m": float(self.mapping_radius_m[index]),
            }
            if include_runtime_ids:
                record["body_id"] = int(self.taxels.body_id[index])
                record["geom_id"] = int(self.taxels.geom_id[index])
            records.append(record)
        return records


@dataclass(frozen=True, slots=True)
class JuQiaoTactileFrame:
    """One 50 Hz-compatible sample of the three real skin-suit packets."""

    values: np.ndarray
    source_time: np.ndarray
    updated: np.ndarray

    def __post_init__(self) -> None:
        if self.values.shape != (len(TACTILE_DEVICE_NAMES), TACTILE_DEVICE_DIM):
            raise ValueError("JuQiao values must have shape (3, 256)")
        if self.values.dtype != np.uint8:
            raise TypeError("JuQiao values must use uint8")
        if self.source_time.shape != (len(TACTILE_DEVICE_NAMES),):
            raise ValueError("JuQiao source_time must have shape (3,)")
        if self.updated.shape != (len(TACTILE_DEVICE_NAMES),):
            raise ValueError("JuQiao updated must have shape (3,)")

    def device(self, name: str) -> np.ndarray:
        try:
            index = TACTILE_DEVICE_NAMES.index(name)
        except ValueError as exc:
            raise KeyError(f"unknown tactile device: {name}") from exc
        return self.values[index].copy()


class JuQiaoTactileAdapter:
    """Convert physical taxel force to the real robot's asynchronous packets.

    The real hardware does not provide a force-to-count calibration curve. The
    configurable power-law conversion is therefore deliberately separated from
    the physical force sidecar and labelled provisional in :attr:`metadata`.
    """

    def __init__(
        self,
        layout: JuQiaoSkinLayout,
        *,
        sample_rate_hz: float | Sequence[float] = 14.0,
        stale_timeout_s: float | Sequence[float] = 0.1,
        gain_counts_per_newton: float = 4.0,
        gamma: float = 1.0,
        force_offset_n: float = 0.0,
        gain_variation: float = 0.08,
        sample_rate_jitter: float = 0.08,
        seed: int = 0,
        initial_time: float = 0.0,
        input_quantity: TactileInputQuantity = "force",
        dropout_probability: float | Sequence[float] = 0.0,
        noise_std_counts: float | Sequence[float] = 0.0,
        attack_tau_s: float = 0.0,
        release_tau_s: float = 0.0,
        output_deadband_counts: float = 0.0,
        saturation_count: int = 255,
        randomization: TactileEpisodeRandomization | None = None,
        profile: TactileCalibrationProfile | None = None,
    ) -> None:
        self.layout = layout
        self._legacy_compatibility = profile is None
        self.profile = profile or make_legacy_tactile_profile(
            len(TACTILE_DEVICE_NAMES),
            device_names=TACTILE_DEVICE_NAMES,
            layout_sha256=layout.sha256,
            sample_rate_hz=sample_rate_hz,
            stale_timeout_s=stale_timeout_s,
            dropout_probability=dropout_probability,
            noise_std_counts=noise_std_counts,
            input_quantity=input_quantity,
            gain=gain_counts_per_newton,
            gamma=gamma,
            offset=force_offset_n,
            attack_tau_s=attack_tau_s,
            release_tau_s=release_tau_s,
            output_deadband_counts=output_deadband_counts,
            saturation_count=saturation_count,
            fixed_taxel_gain_variation=gain_variation,
            fixed_device_rate_variation=sample_rate_jitter,
            randomization=randomization,
        )
        self._uses_sensor_dynamics = self.profile.schema_version >= 2
        if (
            self._uses_sensor_dynamics
            and self.profile.sensor_dynamics.observation_delivery is not None
        ):
            raise NotImplementedError(
                "observation_delivery is not yet supported by the JuQiao adapter"
            )
        device_names = tuple(device.name for device in self.profile.devices)
        if device_names != TACTILE_DEVICE_NAMES:
            raise ValueError(
                "JuQiao profile device order must be vest, left_arm, right_arm"
            )
        if (
            self.profile.layout_sha256 is not None
            and self.profile.layout_sha256 != layout.sha256
        ):
            raise ValueError("tactile profile layout_sha256 does not match G1 skin")

        transfer = self.profile.transfer
        nominal_rates = np.asarray(
            [device.sample_rate_hz for device in self.profile.devices]
        )
        stale_timeouts = np.asarray(
            [device.stale_timeout_s for device in self.profile.devices]
        )
        self.sample_rate_hz = _uniform_or_list(nominal_rates)
        self.stale_timeout_s = _uniform_or_list(stale_timeouts)
        self.gain_counts_per_newton = (
            float(transfer.gain) if transfer.input_quantity == "force" else None
        )
        self.gamma = float(transfer.gamma)
        self.force_offset_n = (
            float(transfer.offset) if transfer.input_quantity == "force" else None
        )
        self.gain_variation = float(self.profile.fixed_taxel_gain_variation)
        self.sample_rate_jitter = float(self.profile.fixed_device_rate_variation)
        self.seed = int(seed)
        self._profile_sampler = TactileProfileSampler(
            self.profile,
            len(layout.taxels),
            seed=seed,
            legacy_fixed_stream=self._legacy_compatibility,
            fixed_taxel_permutation=(
                _LEGACY_FIXED_GAIN_INDEX if self._legacy_compatibility else None
            ),
        )
        self._phase = np.asarray(
            [device.phase_offset_s for device in self.profile.devices],
            dtype=np.float64,
        )
        self._stale_timeout = stale_timeouts
        self._dropout_probability = np.asarray(
            [device.dropout_probability for device in self.profile.devices],
            dtype=np.float64,
        )
        self._device_taxel_indices = tuple(
            np.flatnonzero(layout.taxels.region_id == device_id)
            for device_id in range(len(TACTILE_DEVICE_NAMES))
        )
        self._garment_topology = (
            _garment_grid_topology(layout) if self._uses_sensor_dynamics else None
        )
        self._garment_topology_sha256 = (
            _topology_sha256(
                self._garment_topology,
                layout.taxels.channel_id,
            )
            if self._garment_topology is not None
            else None
        )
        self._spread_kernel: GarmentLoadSpreadKernel | None = None
        self._noise_states: tuple[AR1NoiseState, ...] = ()
        self._drift_states: tuple[SignedOUDriftState, ...] = ()
        self._dropout_rngs: tuple[np.random.Generator, ...] = ()
        self._held = np.zeros(
            (len(TACTILE_DEVICE_NAMES), TACTILE_DEVICE_DIM), dtype=np.uint8
        )
        self._source_time = np.full(len(TACTILE_DEVICE_NAMES), -1.0)
        self._next_sample = np.zeros(len(TACTILE_DEVICE_NAMES), dtype=np.float64)
        self._sample_origin = np.zeros(len(TACTILE_DEVICE_NAMES), dtype=np.float64)
        self._next_sample_index = np.zeros(len(TACTILE_DEVICE_NAMES), dtype=np.int64)
        self._response = np.zeros(len(layout.taxels), dtype=np.float64)
        self._causal_target = np.zeros(len(layout.taxels), dtype=np.float64)
        self._last_update_time = float(initial_time)
        self._activate_episode(initial_time, seed)

    @property
    def metadata(self) -> dict[str, object]:
        transfer = self.profile.transfer
        randomization = self.profile.randomization
        metadata: dict[str, object] = {
            "schema_version": self.profile.schema_version,
            "mode": "triple",
            "devices": list(TACTILE_DEVICE_NAMES),
            "shape": [TACTILE_DEVICE_DIM],
            "dtype": "uint8",
            "record_rate_hz": 50.0,
            "device_sample_rate_hz": self.sample_rate_hz,
            "realized_device_sample_rate_hz": self._device_sample_rate.tolist(),
            "fixed_device_rate_variation": self.sample_rate_jitter,
            "sampling": "independent_phase_sample_and_hold",
            "stale_timeout_s": self.stale_timeout_s,
            "dropout_probability": self._dropout_probability.tolist(),
            "normal_force_only": True,
            "mapping_radius_m": {
                "minimum": float(np.min(self.layout.mapping_radius_m)),
                "maximum": float(np.max(self.layout.mapping_radius_m)),
                "rule": "nearest_collision_vertex_plus_10mm",
            },
            "output_semantics": "post_baseline_nonnegative_raw_count",
            "real_robot_baseline": "100_packet_per_channel_mean",
            "policy_deadband_counts": 2,
            "gain_counts_per_newton": self.gain_counts_per_newton,
            "gamma": self.gamma,
            "force_offset_n": self.force_offset_n,
            "taxel_gain_variation": self.gain_variation,
            "random_seed": self.seed,
            "episode_seed": self._episode.seed,
            "episode_profile": self._episode.as_dict(),
            "taxel_gain_sha256": hashlib.sha256(
                self._taxel_gain.astype("<f8", copy=False).tobytes()
            ).hexdigest(),
            "calibration_status": (
                "provisional_until_force_fixture_calibration"
                if self._legacy_compatibility
                else FORCE_MAPPING_STATUS
            ),
            "paired_force_calibrated": False,
            "force_mapping_warning": (
                "N/kPa to count mapping has no synchronized paired-force fixture "
                "calibration; task-data marginals are not a force calibration"
            ),
            "input_quantity": transfer.input_quantity,
            "input_unit": transfer.input_unit,
            "gain_counts_per_input_unit_power": transfer.gain,
            "input_offset": transfer.offset,
            "profile_id": self.profile.profile_id,
            "profile_sha256": self.profile.sha256,
            "profile_file_sha256": self.profile.source_file_sha256,
            "profile": self.profile.as_dict(),
            "physical_preload_input": {
                "unit": transfer.input_unit,
                "episode_range": randomization.preload.as_dict(),
                "identifiability": ("not_identifiable_from_post_baseline_quiet_frames"),
            },
            "post_baseline_residual_count": {
                "unit": "count",
                "episode_range": randomization.residual_bias_counts.as_dict(),
                "identifiability": "estimable_from_post_baseline_quiet_frames",
            },
            "layout_sha256": self.layout.sha256,
            "source_time": {
                "unit": "s",
                "clock": "mujoco_sim_time",
                "invalid_sentinel": -1.0,
            },
            "updated_semantics": "non_dropped_device_packet_published_on_this_record_row",
        }
        if self._uses_sensor_dynamics:
            metadata["sensor_dynamics"] = {
                "status": SENSOR_DYNAMICS_STATUS,
                "force_spread_order": (
                    "normal_force_before_input_conversion_and_transfer"
                ),
                "noise_semantics": "AR(1)_advanced_once_per_source_sample",
                "drift_semantics": "signed_OU_advanced_at_source_sample_time",
                "dropout_semantics": (
                    "source_state_advances_before_independent_device_dropout"
                ),
                "observation_delivery": "disabled_unidentified_from_current_data",
                "baseline_pipeline": {
                    "status": "not_simulated",
                    "simulated_domain": "post_baseline_nonnegative_count",
                    "hardware_startup_baseline_packets": 100,
                    "hardware_recalibration_packets": 50,
                    "recalibration_blackout": "not_simulated",
                },
                "sleeve_elbow_coupling": (
                    "continuous_16x16_grid_across_two_links_provisional"
                ),
                "mount_calibration": self.layout.as_dict(include_taxels=False)[
                    "mount_calibration"
                ],
                "topology_sha256": self._garment_topology_sha256,
            }
        return metadata

    def reset(self, time: float = 0.0, *, seed: int | None = None) -> None:
        self._activate_episode(time, seed)

    def _activate_episode(self, time: float, seed: int | None) -> None:
        if not np.isfinite(time):
            raise ValueError("reset time must be finite")
        self._episode, self._rng = self._profile_sampler.sample(seed=seed)
        self._device_sample_rate = self._episode.device_sample_rate_hz
        self._taxel_gain = self._episode.taxel_gain_scale
        self._period = 1.0 / self._device_sample_rate
        self._held.fill(0)
        self._source_time.fill(-1.0)
        phase = self._phase
        if self._uses_sensor_dynamics:
            self._activate_sensor_dynamics(time)
            assert self._episode.source_phase_s is not None
            phase = self._episode.source_phase_s
        self._sample_origin = float(time) + phase
        self._next_sample = self._sample_origin.copy()
        self._next_sample_index.fill(0)
        self._response.fill(0.0)
        self._causal_target = self._normal_force_target(
            np.zeros(len(self.layout.taxels), dtype=np.float64)
        )
        self._last_update_time = float(time)
        self.last_frame = self._frame(np.zeros(3, dtype=bool))

    def update(self, frame: TactileFrame, time: float) -> JuQiaoTactileFrame:
        """Convert an interval-mean physical frame using the legacy API.

        This path intentionally retains the historical assumption that the
        frame's mean force applies over the whole interval ending at ``time``.
        Physics-step callers should use :meth:`update_normal_force` so a new
        force observation cannot affect an earlier device sample.
        """

        if not _same_layout(frame.layout, self.layout.taxels):
            raise ValueError("tactile frame uses a different taxel layout")
        target = self._normal_force_target(frame.normal_force)
        self._validate_sample_time(time)
        updated = np.zeros(len(TACTILE_DEVICE_NAMES), dtype=bool)
        response_time = self._last_update_time
        for sample_time, device_id in self._sample_events(time):
            self._advance_response(target, sample_time - response_time)
            response_time = sample_time
            source_time = time if self._legacy_compatibility else sample_time
            self._publish_sample(device_id, source_time, updated)

        self._advance_response(target, time - response_time)
        self._causal_target = target
        return self._finish_update(time, updated)

    def update_normal_force(
        self,
        normal_force_n: np.ndarray,
        time: float,
    ) -> JuQiaoTactileFrame:
        """Advance from one timestamped physics-step force observation.

        Device events strictly before ``time`` use the preceding observation.
        An event exactly at ``time`` may use the new observation. This makes
        the adapter causal while preserving asynchronous device clocks.
        """

        target = self._normal_force_target(normal_force_n)
        self._validate_sample_time(time)
        updated = np.zeros(len(TACTILE_DEVICE_NAMES), dtype=bool)
        boundary_events: list[tuple[float, int]] = []
        response_time = self._last_update_time
        boundary_time = (
            round(float(time), _SAMPLE_TIME_DECIMALS)
            if self._uses_sensor_dynamics
            else float(time)
        )
        for sample_time, device_id in self._sample_events(time):
            if sample_time == boundary_time:
                boundary_events.append((sample_time, device_id))
                continue
            self._advance_response(self._causal_target, sample_time - response_time)
            response_time = sample_time
            self._publish_sample(device_id, sample_time, updated)

        self._advance_response(self._causal_target, time - response_time)
        self._causal_target = target
        self._advance_response(target, 0.0)
        for sample_time, device_id in boundary_events:
            self._publish_sample(device_id, sample_time, updated)
        return self._finish_update(time, updated)

    def update_normal_force_history(
        self,
        history: Iterable[tuple[float, np.ndarray]],
    ) -> JuQiaoTactileFrame:
        """Advance a chronological ``(time, normal_force_n)`` history.

        ``updated`` in the returned frame is the per-device OR across every
        observation in this call; packet values remain sample-and-hold.
        """

        updated = np.zeros(len(TACTILE_DEVICE_NAMES), dtype=bool)
        for time, normal_force_n in history:
            frame = self.update_normal_force(normal_force_n, time)
            updated |= frame.updated
        return self.snapshot(updated=updated)

    def snapshot(self, *, updated: np.ndarray | None = None) -> JuQiaoTactileFrame:
        """Return held packets, optionally replacing the row update flags."""

        if updated is None:
            updated = self.last_frame.updated
        self.last_frame = self._frame(updated)
        return self.last_frame

    def _normal_force_target(self, normal_force_n: np.ndarray) -> np.ndarray:
        transfer = self.profile.transfer
        force = normal_force_n
        if self._uses_sensor_dynamics:
            assert self._spread_kernel is not None
            force = tactile_input(normal_force_n, self.layout.area_m2, "force")
            force = self._spread_kernel.spread(force)
        physical_input = tactile_input(
            force,
            self.layout.area_m2,
            transfer.input_quantity,
        )
        return power_law_response(physical_input, transfer, self._episode)

    def _activate_sensor_dynamics(self, time: float) -> None:
        episode = self._episode
        assert self._garment_topology is not None
        assert episode.spatial_first_ring_fraction is not None
        assert episode.spatial_second_ring_fraction is not None
        assert episode.device_noise_ar1_rho is not None
        assert episode.device_drift_std_counts is not None
        assert episode.device_drift_time_constant_s is not None
        self._spread_kernel = GarmentLoadSpreadKernel(
            *self._garment_topology,
            first_ring_fraction=episode.spatial_first_ring_fraction,
            second_ring_fraction=episode.spatial_second_ring_fraction,
        )
        self._noise_states = tuple(
            AR1NoiseState(
                len(indices),
                marginal_std=float(episode.device_noise_std_counts[device_id]),
                coefficient=float(episode.device_noise_ar1_rho[device_id]),
                seed=_component_seed(episode.seed, _AR1_RNG_TAG, device_id),
            )
            for device_id, indices in enumerate(self._device_taxel_indices)
        )
        self._drift_states = tuple(
            SignedOUDriftState(
                len(indices),
                stationary_std=float(episode.device_drift_std_counts[device_id]),
                tau_s=float(episode.device_drift_time_constant_s[device_id]),
                seed=_component_seed(episode.seed, _DRIFT_RNG_TAG, device_id),
                initial_time=time,
            )
            for device_id, indices in enumerate(self._device_taxel_indices)
        )
        self._dropout_rngs = tuple(
            np.random.default_rng(
                _component_seed(episode.seed, _DROPOUT_RNG_TAG, device_id)
            )
            for device_id in range(len(TACTILE_DEVICE_NAMES))
        )

    def _validate_sample_time(self, time: float) -> None:
        if not np.isfinite(time):
            raise ValueError("sample time must be finite")
        if time < self._last_update_time:
            raise ValueError("tactile sample time must be nondecreasing")

    def _sample_events(self, time: float) -> list[tuple[float, int]]:
        if self.profile.schema_version == 1:
            return self._legacy_sample_events(time)

        events: list[tuple[float, int]] = []
        next_indices = self._next_sample_index.copy()
        for device_id in range(len(TACTILE_DEVICE_NAMES)):
            index = int(next_indices[device_id])
            while True:
                sample_time = round(
                    float(
                        self._sample_origin[device_id] + index * self._period[device_id]
                    ),
                    _SAMPLE_TIME_DECIMALS,
                )
                if sample_time > time:
                    break
                if len(events) >= _MAX_SAMPLE_EVENTS_PER_UPDATE:
                    raise RuntimeError(
                        "too many tactile samples elapsed; reset the adapter after "
                        "a simulation-time discontinuity"
                    )
                events.append((sample_time, device_id))
                index += 1
            next_indices[device_id] = index

        self._next_sample_index = next_indices
        events.sort()
        return events

    def _legacy_sample_events(self, time: float) -> list[tuple[float, int]]:
        """Retain the published v1 floating-point clock exactly."""

        events: list[tuple[float, int]] = []
        for device_id in range(len(TACTILE_DEVICE_NAMES)):
            if time + 1e-12 < self._next_sample[device_id]:
                continue
            elapsed = time - self._next_sample[device_id]
            period = self._period[device_id]
            periods = math.floor(max(elapsed, 0.0) / period) + 1
            if len(events) + periods > _MAX_SAMPLE_EVENTS_PER_UPDATE:
                raise RuntimeError(
                    "too many tactile samples elapsed; reset the adapter after "
                    "a simulation-time discontinuity"
                )
            events.extend(
                (
                    min(
                        self._next_sample[device_id] + index * period,
                        float(time),
                    ),
                    device_id,
                )
                for index in range(periods)
            )
            self._next_sample[device_id] += periods * period

        events.sort()
        return events

    def _publish_sample(
        self,
        device_id: int,
        source_time: float,
        updated: np.ndarray,
    ) -> None:
        if self._uses_sensor_dynamics:
            packet = self._sample_v2_device(device_id, source_time)
            if (
                self._dropout_rngs[device_id].random()
                < self._dropout_probability[device_id]
            ):
                return
            self._held[device_id] = packet
            self._source_time[device_id] = source_time
            updated[device_id] = True
            return

        if self._rng.random() < self._dropout_probability[device_id]:
            return
        self._held[device_id] = self._sample_device(
            device_id,
            self.layout.taxels.region_id,
            self.layout.taxels.channel_id,
        )
        self._source_time[device_id] = source_time
        updated[device_id] = True

    def _finish_update(
        self,
        time: float,
        updated: np.ndarray,
    ) -> JuQiaoTactileFrame:
        self._last_update_time = float(time)
        for device_id in range(len(TACTILE_DEVICE_NAMES)):
            if time - self._source_time[device_id] > self._stale_timeout[device_id]:
                self._held[device_id].fill(0)
        self.last_frame = self._frame(updated)
        return self.last_frame

    def _advance_response(self, target: np.ndarray, duration: float) -> None:
        transfer = self.profile.transfer
        self._response = asymmetric_first_order_filter(
            self._response,
            target,
            float(duration),
            attack_tau_s=transfer.attack_tau_s,
            release_tau_s=transfer.release_tau_s,
        )

    def _sample_device(
        self,
        device_id: int,
        region_ids: np.ndarray,
        channels: np.ndarray,
    ) -> np.ndarray:
        packet = np.zeros(TACTILE_DEVICE_DIM, dtype=np.uint8)
        mask = region_ids == device_id
        response = self._response[mask] + self._episode.taxel_residual_bias_counts[mask]
        packet[channels[mask]] = quantize_counts(
            response,
            noise_std_counts=float(self._episode.device_noise_std_counts[device_id]),
            transfer=self.profile.transfer,
            rng=self._rng,
        )
        return packet

    def _sample_v2_device(self, device_id: int, source_time: float) -> np.ndarray:
        packet = np.zeros(TACTILE_DEVICE_DIM, dtype=np.uint8)
        indices = self._device_taxel_indices[device_id]
        response = (
            self._response[indices]
            + self._episode.taxel_residual_bias_counts[indices]
            + self._noise_states[device_id].sample()
            + self._drift_states[device_id].advance_to(source_time)
        )
        channels = self.layout.taxels.channel_id[indices]
        packet[channels] = quantize_counts(
            response,
            noise_std_counts=0.0,
            transfer=self.profile.transfer,
            rng=None,
        )
        return packet

    def _frame(self, updated: np.ndarray) -> JuQiaoTactileFrame:
        return JuQiaoTactileFrame(
            values=self._held.copy(),
            source_time=self._source_time.copy(),
            updated=np.asarray(updated, dtype=bool).copy(),
        )


@dataclass(frozen=True, slots=True)
class _SkinTaxel:
    body_name: str
    device_id: int
    channel: int
    center: np.ndarray
    normal: np.ndarray
    area_m2: float
    subregion: str


def build_juqiao_skin_layout(
    model: mujoco.MjModel,
    *,
    left_mount: SleeveMount | None = None,
    right_mount: SleeveMount | None = None,
) -> JuQiaoSkinLayout:
    """Build the current real-robot 3-device skin layout on a MuJoCo G1."""

    left_mount = left_mount or SleeveMount()
    right_mount = right_mount or SleeveMount()
    taxels = [
        *_vest_torso_taxels(),
        *_vest_arm_taxels(),
        *_sleeve_taxels("left", 1, left_mount),
        *_sleeve_taxels("right", 2, right_mount),
    ]
    return _compile_skin_layout(model, taxels, left_mount, right_mount)


def _vest_torso_taxels() -> Iterator[_SkinTaxel]:
    for key, rows, columns, one_based_channels in _VEST_REGIONS[:2]:
        front = key == "front_chest"
        width = 0.208
        height = 0.188 if front else 0.180
        z_center = 0.205
        for row, column in np.ndindex(rows, columns):
            direction = -1.0 if front else 1.0
            y = direction * ((column + 0.5) * width / columns - width / 2.0)
            z = z_center + height / 2.0 - (row + 0.5) * height / rows
            center, normal = _torso_surface(1.0 if front else -1.0, z, y)
            channel = one_based_channels[row * columns + column] - 1
            yield _SkinTaxel(
                body_name="torso_link",
                device_id=0,
                channel=channel,
                center=center,
                normal=normal,
                area_m2=width * height / (rows * columns),
                subregion=key,
            )


def _vest_arm_taxels() -> Iterator[_SkinTaxel]:
    for key, rows, columns, one_based_channels in _VEST_REGIONS[2:]:
        side, region = key.split("_", maxsplit=1)
        band = 0 if region == "shoulder" else 1
        for row, column in np.ndindex(rows, columns):
            angle = math.radians(-70.0 + column * (140.0 / 3.0))
            z = 0.025 - band * 0.046 - row * 0.020
            radius = 0.051
            center = np.array((radius * math.cos(angle), radius * math.sin(angle), z))
            normal = np.array((math.cos(angle), math.sin(angle), 0.0))
            channel = one_based_channels[row * columns + column] - 1
            yield _SkinTaxel(
                body_name=f"{side}_shoulder_yaw_link",
                device_id=0,
                channel=channel,
                center=center,
                normal=normal,
                area_m2=0.025 * 0.018,
                subregion=key,
            )


def _sleeve_taxels(
    side: Literal["left", "right"],
    device_id: int,
    mount: SleeveMount,
) -> Iterator[_SkinTaxel]:
    for row, column in np.ndindex(16, 16):
        axial, circular = mount.physical_indices(row, column)
        angle = 2.0 * math.pi * circular / 16.0
        if axial < 8:
            body_name = f"{side}_shoulder_yaw_link"
            radius = 0.051
            center = np.array(
                (
                    radius * math.cos(angle),
                    radius * math.sin(angle),
                    0.002 - axial * 0.0165,
                )
            )
            normal = np.array((math.cos(angle), math.sin(angle), 0.0))
            subregion = f"{side}_sleeve_upper"
        else:
            body_name = f"{side}_elbow_link"
            radius = 0.048
            center = np.array(
                (
                    0.002 + (axial - 8) * 0.0165,
                    radius * math.cos(angle),
                    radius * math.sin(angle),
                )
            )
            normal = np.array((0.0, math.cos(angle), math.sin(angle)))
            subregion = f"{side}_sleeve_forearm"
        yield _SkinTaxel(
            body_name=body_name,
            device_id=device_id,
            channel=int(_SLEEVE_RAW_CHANNELS[row, column]),
            center=center,
            normal=normal,
            area_m2=0.0165 * (2.0 * math.pi * radius / 16.0),
            subregion=subregion,
        )


def _compile_skin_layout(
    model: mujoco.MjModel,
    taxels: Iterable[_SkinTaxel],
    left_mount: SleeveMount,
    right_mount: SleeveMount,
) -> JuQiaoSkinLayout:
    body_ids: list[int] = []
    geom_ids: list[int] = []
    centers: list[np.ndarray] = []
    normals: list[np.ndarray] = []
    device_ids: list[int] = []
    channel_ids: list[int] = []
    areas: list[float] = []
    subregions: list[str] = []
    body_names: list[str] = []
    mounts: dict[str, tuple[int, int]] = {}

    for taxel in taxels:
        mount = mounts.get(taxel.body_name)
        if mount is None:
            body_id = _body_id(model, taxel.body_name)
            mount = body_id, _primary_collision_geom(model, body_id)
            mounts[taxel.body_name] = mount
        body_id, geom_id = mount
        normal = np.asarray(taxel.normal, dtype=np.float64)
        normal /= np.linalg.norm(normal)
        body_ids.append(body_id)
        geom_ids.append(geom_id)
        centers.append(np.asarray(taxel.center, dtype=np.float64))
        normals.append(normal)
        device_ids.append(taxel.device_id)
        channel_ids.append(taxel.channel)
        areas.append(taxel.area_m2)
        subregions.append(taxel.subregion)
        body_names.append(taxel.body_name)

    layout = TaxelLayout(
        body_id=_readonly(np.asarray(body_ids, dtype=np.int32)),
        geom_id=_readonly(np.asarray(geom_ids, dtype=np.int32)),
        local_center=_readonly(np.asarray(centers, dtype=np.float64)),
        local_normal=_readonly(np.asarray(normals, dtype=np.float64)),
        region_id=_readonly(np.asarray(device_ids, dtype=np.int16)),
        channel_id=_readonly(np.asarray(channel_ids, dtype=np.int32)),
        region_names=TACTILE_DEVICE_NAMES,
    )
    area_m2 = _readonly(np.asarray(areas, dtype=np.float64))
    mapping_radius_m = _mapping_radii(model, layout)
    _validate_layout(layout, area_m2, mapping_radius_m)
    return JuQiaoSkinLayout(
        taxels=layout,
        area_m2=area_m2,
        mapping_radius_m=mapping_radius_m,
        subregion=tuple(subregions),
        body_name=tuple(body_names),
        left_mount=left_mount,
        right_mount=right_mount,
    )


def _torso_surface(side: float, z: float, y: float) -> tuple[np.ndarray, np.ndarray]:
    radius_x, radius_y = _torso_radii(z)

    def x_at(sample_y: float) -> float:
        normalized = float(np.clip(sample_y / radius_y, -1.0, 1.0))
        magnitude = abs(normalized)
        ellipse = radius_x * math.sqrt(max(1.0 - normalized**2, 0.0))
        flat = radius_x - 0.004 * normalized**2
        blend = float(np.clip((magnitude - 0.70) / 0.27, 0.0, 1.0))
        blend = blend * blend * (3.0 - 2.0 * blend)
        return side * (flat * (1.0 - blend) + ellipse * blend)

    epsilon = 1e-4
    derivative = (x_at(y + epsilon) - x_at(y - epsilon)) / (2.0 * epsilon)
    normal = np.array((side, -side * derivative, 0.0))
    normal /= np.linalg.norm(normal)
    return np.array((x_at(y), y, z)), normal


def _torso_radii(z: float) -> tuple[float, float]:
    lower, upper = _TORSO_PROFILE[0], _TORSO_PROFILE[-1]
    for start, end in pairwise(_TORSO_PROFILE):
        if start[0] <= z <= end[0]:
            lower, upper = start, end
            break
    blend = float(np.clip((z - lower[0]) / max(upper[0] - lower[0], 1e-8), 0.0, 1.0))
    return (
        lower[1] * (1.0 - blend) + upper[1] * blend,
        lower[2] * (1.0 - blend) + upper[2] * blend,
    )


def _body_id(model: mujoco.MjModel, name: str) -> int:
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    if body_id < 0:
        raise ValueError(f"missing G1 body for tactile skin: {name}")
    return body_id


def _primary_collision_geom(model: mujoco.MjModel, body_id: int) -> int:
    candidates = [
        geom_id
        for geom_id in range(model.ngeom)
        if int(model.geom_bodyid[geom_id]) == body_id
        and (
            int(model.geom_contype[geom_id]) != 0
            or int(model.geom_conaffinity[geom_id]) != 0
        )
    ]
    if not candidates:
        raise ValueError(f"body {body_id} has no collision geom for tactile skin")
    return max(
        candidates,
        key=lambda geom_id: float(np.prod(2.0 * model.geom_aabb[geom_id, 3:])),
    )


def _mapping_radii(model: mujoco.MjModel, layout: TaxelLayout) -> np.ndarray:
    """Size each contact gate from its garment point to the compiled shell."""

    nearest = np.full(len(layout), np.inf, dtype=np.float64)
    for body_id in np.unique(layout.body_id):
        surface_samples = []
        for geom_id in range(model.ngeom):
            if int(model.geom_bodyid[geom_id]) != int(body_id):
                continue
            if not (
                int(model.geom_contype[geom_id]) != 0
                or int(model.geom_conaffinity[geom_id]) != 0
            ):
                continue
            geom_type = int(model.geom_type[geom_id])
            if geom_type == int(mujoco.mjtGeom.mjGEOM_MESH):
                mesh_id = int(model.geom_dataid[geom_id])
                address = int(model.mesh_vertadr[mesh_id])
                count = int(model.mesh_vertnum[mesh_id])
                samples = np.asarray(
                    model.mesh_vert[address : address + count],
                    dtype=np.float64,
                )
            else:
                center = np.asarray(model.geom_aabb[geom_id, :3])
                half = np.asarray(model.geom_aabb[geom_id, 3:])
                corners = np.asarray(
                    [
                        center + half * (x, y, z)
                        for x in (-1.0, 1.0)
                        for y in (-1.0, 1.0)
                        for z in (-1.0, 1.0)
                    ]
                )
                faces = np.concatenate(
                    tuple(
                        center[None, :] + np.eye(3)[axis][None, :] * sign * half[axis]
                        for axis in range(3)
                        for sign in (-1.0, 1.0)
                    )
                )
                samples = np.concatenate((corners, faces))
            rotation = np.zeros(9, dtype=np.float64)
            mujoco.mju_quat2Mat(rotation, model.geom_quat[geom_id])
            surface_samples.append(
                samples @ rotation.reshape(3, 3).T + model.geom_pos[geom_id]
            )
        if not surface_samples:
            raise ValueError(
                f"body {body_id} has no collision surface for tactile skin"
            )
        samples = np.concatenate(surface_samples)
        indices = np.flatnonzero(layout.body_id == body_id)
        for index in indices:
            nearest[index] = np.min(
                np.linalg.norm(samples - layout.local_center[index], axis=1)
            )
    if np.any(~np.isfinite(nearest)):
        raise AssertionError(
            "could not measure every tactile point to a collision shell"
        )
    return _readonly(
        np.clip(
            nearest + _MAPPING_RADIUS_MARGIN_M,
            _MAPPING_RADIUS_MIN_M,
            _MAPPING_RADIUS_MAX_M,
        )
    )


def _validate_layout(
    layout: TaxelLayout,
    area_m2: np.ndarray,
    mapping_radius_m: np.ndarray,
) -> None:
    if len(layout) != TACTILE_VALID_COUNT:
        raise AssertionError(f"JuQiao layout has {len(layout)} taxels, expected 624")
    if area_m2.shape != (len(layout),) or np.any(area_m2 <= 0.0):
        raise AssertionError("JuQiao taxel areas must be positive and fixed-shape")
    if mapping_radius_m.shape != (len(layout),) or np.any(mapping_radius_m <= 0.0):
        raise AssertionError("JuQiao mapping radii must be positive and fixed-shape")
    keys = list(zip(layout.region_id.tolist(), layout.channel_id.tolist()))
    if len(set(keys)) != len(keys):
        raise AssertionError("JuQiao device/channel pairs must be unique")
    if np.any(layout.channel_id < 0) or np.any(layout.channel_id >= TACTILE_DEVICE_DIM):
        raise AssertionError("JuQiao channels must be zero-based uint8 packet slots")
    vest = layout.channel_id[layout.region_id == 0]
    if len(vest) != 112:
        raise AssertionError("JuQiao vest must expose exactly 112 wired channels")
    for device_id in (1, 2):
        channels = np.sort(layout.channel_id[layout.region_id == device_id])
        if not np.array_equal(channels, np.arange(TACTILE_DEVICE_DIM)):
            raise AssertionError("JuQiao sleeve must expose every channel once")


def _garment_grid_topology(
    layout: JuQiaoSkinLayout,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return garment-grid coordinates without changing layout serialization.

    The vest consists of six electrically distinct rectangular patches. Each
    sleeve is one continuous 16x16 patch, including its circular seam and the
    upper-arm/forearm boundary. Coordinates follow the physical mount rather
    than raw controller row order.
    """

    device_id = np.asarray(layout.taxels.region_id, dtype=np.int64).copy()
    patch_id = np.full(len(layout.taxels), -1, dtype=np.int64)
    grid_row = np.full(len(layout.taxels), -1, dtype=np.int64)
    grid_col = np.full(len(layout.taxels), -1, dtype=np.int64)
    wrap_columns = np.zeros(len(layout.taxels), dtype=bool)

    vest_patches = ((6, 8), (5, 8), (2, 4), (1, 4), (2, 4), (1, 4))
    vest_indices = np.flatnonzero(device_id == 0)
    if len(vest_indices) != sum(rows * columns for rows, columns in vest_patches):
        raise ValueError("JuQiao vest topology requires exactly 112 taxels")

    vest_coordinates: dict[tuple[str, int], tuple[int, int, int]] = {}
    for patch, (name, rows, columns, channels) in enumerate(_VEST_REGIONS):
        for position, one_based_channel in enumerate(channels):
            vest_coordinates[(name, one_based_channel - 1)] = (
                patch,
                position // columns,
                position % columns,
            )
    known_vest_taxels = [
        (layout.subregion[index], int(layout.taxels.channel_id[index]))
        in vest_coordinates
        for index in vest_indices
    ]
    if any(known_vest_taxels) and not all(known_vest_taxels):
        raise ValueError("JuQiao vest topology is only partially identified")
    if all(known_vest_taxels):
        for index in vest_indices:
            coordinate = vest_coordinates[
                (layout.subregion[index], int(layout.taxels.channel_id[index]))
            ]
            patch_id[index], grid_row[index], grid_col[index] = coordinate
    else:
        # Minimal synthetic layouts used by adapter tests may omit physical
        # subregion names. Their unique channels still provide stable ordering.
        ordered = vest_indices[np.argsort(layout.taxels.channel_id[vest_indices])]
        offset = 0
        for patch, (rows, columns) in enumerate(vest_patches):
            count = rows * columns
            indices = ordered[offset : offset + count]
            positions = np.arange(count, dtype=np.int64)
            patch_id[indices] = patch
            grid_row[indices] = positions // columns
            grid_col[indices] = positions % columns
            offset += count

    raw_sleeve_coordinates = {
        int(channel): divmod(raw_index, 16)
        for raw_index, channel in enumerate(_SLEEVE_RAW_CHANNELS.flat)
    }

    for patch, (device, mount) in enumerate(
        ((1, layout.left_mount), (2, layout.right_mount)),
        start=len(vest_patches),
    ):
        indices = np.flatnonzero(device_id == device)
        if len(indices) != TACTILE_DEVICE_DIM:
            raise ValueError("JuQiao sleeve topology requires exactly 256 taxels")
        for taxel_index in indices:
            channel = int(layout.taxels.channel_id[taxel_index])
            try:
                raw_row, raw_column = raw_sleeve_coordinates[channel]
            except KeyError as exc:
                raise ValueError(
                    "JuQiao sleeve channel is outside uint8 range"
                ) from exc
            axial, circular = mount.physical_indices(raw_row, raw_column)
            patch_id[taxel_index] = patch
            grid_row[taxel_index] = axial
            grid_col[taxel_index] = circular
            wrap_columns[taxel_index] = True

    if np.any(patch_id < 0) or np.any(grid_row < 0) or np.any(grid_col < 0):
        raise ValueError("JuQiao garment topology does not cover every taxel")
    return tuple(
        _readonly(value)
        for value in (device_id, patch_id, grid_row, grid_col, wrap_columns)
    )


def _topology_sha256(
    topology: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    channel_id: np.ndarray,
) -> str:
    records = np.column_stack(
        (
            *topology[:-1],
            topology[-1].astype(np.int64),
            np.asarray(channel_id, dtype=np.int64),
        )
    )
    order = np.lexsort(tuple(records[:, column] for column in reversed(range(6))))
    digest = hashlib.sha256()
    digest.update(records[order].astype("<i8", copy=False).tobytes())
    return digest.hexdigest()


def _component_seed(episode_seed: int, tag: int, device_id: int) -> int:
    """Derive a stable uint64 stream seed without consuming another stream."""

    state = np.random.SeedSequence(
        [
            episode_seed & 0xFFFFFFFF,
            episode_seed >> 32,
            tag,
            device_id,
        ]
    ).generate_state(2, dtype=np.uint32)
    return int(state[0]) | (int(state[1]) << 32)


def _readonly(array: np.ndarray) -> np.ndarray:
    array.setflags(write=False)
    return array


def _uniform_or_list(values: np.ndarray) -> float | list[float]:
    values = np.asarray(values, dtype=np.float64)
    if np.all(values == values[0]):
        return float(values[0])
    return values.tolist()


def _same_layout(left: TaxelLayout, right: TaxelLayout) -> bool:
    if left is right:
        return True
    return (
        left.region_names == right.region_names
        and np.array_equal(left.body_id, right.body_id)
        and np.array_equal(left.geom_id, right.geom_id)
        and np.array_equal(left.region_id, right.region_id)
        and np.array_equal(left.channel_id, right.channel_id)
        and np.array_equal(left.local_center, right.local_center)
        and np.array_equal(left.local_normal, right.local_normal)
    )
