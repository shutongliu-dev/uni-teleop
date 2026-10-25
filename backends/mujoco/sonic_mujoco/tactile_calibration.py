"""Versioned Sim2Real profiles for tactile sensor electronics.

This module deliberately starts *after* MuJoCo contact projection. Physical
force and impulse remain in :class:`sonic_mujoco.tactile.TactileFrame`; the
profiles below only describe the provisional conversion into hardware-shaped
integer packets.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from numbers import Integral, Real
from pathlib import Path
from typing import Literal

import numpy as np

TACTILE_CALIBRATION_PROFILE_VERSION = 2
_SUPPORTED_PROFILE_VERSIONS = frozenset({1, 2})
FORCE_MAPPING_STATUS = "provisional_unpaired_force_mapping"
SENSOR_DYNAMICS_STATUS = "provisional_unpaired_sensor_dynamics"
_PROFILE_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_UINT64_MAX = np.iinfo(np.uint64).max
_MAX_DEVICE_SAMPLE_RATE_HZ = 1000.0

TactileInputQuantity = Literal["force", "pressure"]


@dataclass(frozen=True, slots=True)
class UniformRange:
    """Inclusive range used for deterministic per-episode randomization."""

    low: float
    high: float

    def __post_init__(self) -> None:
        if not np.isfinite(self.low) or not np.isfinite(self.high):
            raise ValueError("randomization range must be finite")
        if self.low > self.high:
            raise ValueError("randomization range low must not exceed high")

    def sample(
        self,
        rng: np.random.Generator,
        size: int | tuple[int, ...],
    ) -> np.ndarray:
        if self.low == self.high:
            return np.full(size, self.low, dtype=np.float64)
        return rng.uniform(self.low, self.high, size=size)

    def as_dict(self) -> dict[str, object]:
        return {"low": float(self.low), "high": float(self.high)}


@dataclass(frozen=True, slots=True)
class TactileTransferProfile:
    """Static analog transfer parameters before packet sampling.

    ``force`` input is measured in newtons per logical taxel. ``pressure`` is
    derived from the taxel area and measured in kilopascals. ``gain`` therefore
    has units of counts per ``input_unit ** gamma``. Neither mode is considered
    calibrated until a synchronized applied-force fixture is available.
    """

    input_quantity: TactileInputQuantity = "force"
    gain: float = 4.0
    gamma: float = 1.0
    offset: float = 0.0
    attack_tau_s: float = 0.0
    release_tau_s: float = 0.0
    output_deadband_counts: float = 0.0
    saturation_count: int = 255

    def __post_init__(self) -> None:
        if self.input_quantity not in {"force", "pressure"}:
            raise ValueError("tactile input_quantity must be 'force' or 'pressure'")
        values = (
            self.gain,
            self.gamma,
            self.offset,
            self.attack_tau_s,
            self.release_tau_s,
            self.output_deadband_counts,
        )
        if not all(np.isfinite(value) for value in values):
            raise ValueError("tactile transfer parameters must be finite")
        if self.gain <= 0.0 or self.gamma <= 0.0:
            raise ValueError("tactile gain and gamma must be positive")
        if self.offset < 0.0:
            raise ValueError("tactile input offset must be nonnegative")
        if self.attack_tau_s < 0.0 or self.release_tau_s < 0.0:
            raise ValueError("tactile response time constants must be nonnegative")
        if self.output_deadband_counts < 0.0:
            raise ValueError("tactile output deadband must be nonnegative")
        if not isinstance(self.saturation_count, int):
            raise TypeError("tactile saturation_count must be an integer")
        if not 1 <= self.saturation_count <= 255:
            raise ValueError("tactile saturation_count must be in [1, 255]")

    @property
    def input_unit(self) -> str:
        return "N" if self.input_quantity == "force" else "kPa"

    def as_dict(self) -> dict[str, object]:
        return {
            "input_quantity": self.input_quantity,
            "gain": float(self.gain),
            "gamma": float(self.gamma),
            "offset": float(self.offset),
            "attack_tau_s": float(self.attack_tau_s),
            "release_tau_s": float(self.release_tau_s),
            "output_deadband_counts": float(self.output_deadband_counts),
            "saturation_count": self.saturation_count,
        }


@dataclass(frozen=True, slots=True)
class TactileDeviceProfile:
    """Sampling and link behavior for one independently clocked device."""

    name: str
    sample_rate_hz: float = 14.0
    stale_timeout_s: float = 0.1
    dropout_probability: float = 0.0
    noise_std_counts: float = 0.0
    phase_offset_s: float = 0.0

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not _PROFILE_ID_PATTERN.fullmatch(
            self.name
        ):
            raise ValueError("tactile device name must be a plain stable identifier")
        values = (
            self.sample_rate_hz,
            self.stale_timeout_s,
            self.dropout_probability,
            self.noise_std_counts,
            self.phase_offset_s,
        )
        if not all(np.isfinite(value) for value in values):
            raise ValueError("tactile device parameters must be finite")
        if self.sample_rate_hz <= 0.0:
            raise ValueError("tactile sample_rate_hz must be positive")
        if self.sample_rate_hz > _MAX_DEVICE_SAMPLE_RATE_HZ:
            raise ValueError(
                f"tactile sample_rate_hz must not exceed {_MAX_DEVICE_SAMPLE_RATE_HZ:g}"
            )
        if self.stale_timeout_s <= 0.0:
            raise ValueError("tactile stale_timeout_s must be positive")
        if not 0.0 <= self.dropout_probability <= 1.0:
            raise ValueError("tactile dropout_probability must be in [0, 1]")
        if self.noise_std_counts < 0.0:
            raise ValueError("tactile noise_std_counts must be nonnegative")
        if self.phase_offset_s < 0.0:
            raise ValueError("tactile phase_offset_s must be nonnegative")

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "sample_rate_hz": float(self.sample_rate_hz),
            "stale_timeout_s": float(self.stale_timeout_s),
            "dropout_probability": float(self.dropout_probability),
            "noise_std_counts": float(self.noise_std_counts),
            "phase_offset_s": float(self.phase_offset_s),
        }


@dataclass(frozen=True, slots=True)
class TactileEpisodeRandomization:
    """Random variables redrawn for each simulated episode.

    Gain and preload are drawn independently per taxel. Rate and noise are
    drawn independently per device. Degenerate ranges disable a perturbation.
    """

    gain_scale: UniformRange = field(default_factory=lambda: UniformRange(1.0, 1.0))
    sample_rate_scale: UniformRange = field(
        default_factory=lambda: UniformRange(1.0, 1.0)
    )
    noise_std_counts: UniformRange = field(
        default_factory=lambda: UniformRange(0.0, 0.0)
    )
    preload: UniformRange = field(default_factory=lambda: UniformRange(0.0, 0.0))
    residual_bias_counts: UniformRange = field(
        default_factory=lambda: UniformRange(0.0, 0.0)
    )

    def __post_init__(self) -> None:
        if self.gain_scale.low <= 0.0:
            raise ValueError("episode gain scale must be positive")
        if self.sample_rate_scale.low <= 0.0:
            raise ValueError("episode sample-rate scale must be positive")
        if self.noise_std_counts.low < 0.0:
            raise ValueError("episode noise must be nonnegative")
        if self.preload.low < 0.0:
            raise ValueError("episode preload must be nonnegative")
        if self.residual_bias_counts.low < 0.0:
            raise ValueError("episode residual count bias must be nonnegative")

    def as_dict(self) -> dict[str, object]:
        return {
            "gain_scale": self.gain_scale.as_dict(),
            "sample_rate_scale": self.sample_rate_scale.as_dict(),
            "noise_std_counts": self.noise_std_counts.as_dict(),
            "preload": self.preload.as_dict(),
            "residual_bias_counts": self.residual_bias_counts.as_dict(),
        }


@dataclass(frozen=True, slots=True)
class TactileObservationDeliveryProfile:
    """50 Hz observation-layer delivery cadence measured from robot data.

    The states are hold lengths of one through five recorded rows. This models
    the legacy publisher/exporter delivery pattern, not the physical sensing
    rate. Source acquisition remains an independent clock.
    """

    record_rate_hz: float = 50.0
    initial_probabilities: tuple[float, ...] = (0.0, 0.0, 0.0, 1.0, 0.0)
    transition_probabilities: tuple[tuple[float, ...], ...] = (
        (0.0, 0.0, 0.0, 1.0, 0.0),
        (0.0, 0.0, 0.0, 1.0, 0.0),
        (0.0, 0.0, 0.0, 1.0, 0.0),
        (0.0, 0.0, 0.0, 1.0, 0.0),
        (0.0, 0.0, 0.0, 1.0, 0.0),
    )

    def __post_init__(self) -> None:
        if not np.isfinite(self.record_rate_hz) or self.record_rate_hz <= 0.0:
            raise ValueError("observation record_rate_hz must be finite and positive")
        if self.record_rate_hz > _MAX_DEVICE_SAMPLE_RATE_HZ:
            raise ValueError(
                "observation record_rate_hz must not exceed "
                f"{_MAX_DEVICE_SAMPLE_RATE_HZ:g}"
            )
        initial = _probability_vector(
            self.initial_probabilities,
            "observation initial probabilities",
        )
        transition = np.asarray(self.transition_probabilities, dtype=np.float64)
        if transition.shape != (len(initial), len(initial)):
            raise ValueError("observation transition matrix must be square")
        if np.any(~np.isfinite(transition)) or np.any(transition < 0.0):
            raise ValueError("observation transition probabilities must be finite")
        if not np.allclose(transition.sum(axis=1), 1.0, atol=1e-9, rtol=0.0):
            raise ValueError("each observation transition row must sum to one")
        object.__setattr__(self, "initial_probabilities", tuple(initial.tolist()))
        object.__setattr__(
            self,
            "transition_probabilities",
            tuple(tuple(row.tolist()) for row in transition),
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "record_rate_hz": float(self.record_rate_hz),
            "initial_probabilities": list(self.initial_probabilities),
            "transition_probabilities": [
                list(row) for row in self.transition_probabilities
            ],
        }


@dataclass(frozen=True, slots=True)
class TactileSensorDynamicsProfile:
    """Provisional garment, clock, correlated-noise, and drift behavior."""

    spatial_first_ring_fraction: UniformRange = field(
        default_factory=lambda: UniformRange(0.0, 0.0)
    )
    spatial_second_ring_fraction: UniformRange = field(
        default_factory=lambda: UniformRange(0.0, 0.0)
    )
    randomize_source_phase: bool = False
    noise_ar1_rho: UniformRange = field(default_factory=lambda: UniformRange(0.0, 0.0))
    drift_std_counts: UniformRange = field(
        default_factory=lambda: UniformRange(0.0, 0.0)
    )
    drift_time_constant_s: UniformRange = field(
        default_factory=lambda: UniformRange(30.0, 30.0)
    )
    observation_delivery: TactileObservationDeliveryProfile | None = None

    def __post_init__(self) -> None:
        if type(self.randomize_source_phase) is not bool:
            raise TypeError("randomize_source_phase must be a boolean")
        first = self.spatial_first_ring_fraction
        second = self.spatial_second_ring_fraction
        if first.low < 0.0 or second.low < 0.0 or first.high + second.high > 1.0:
            raise ValueError(
                "spatial ring fractions must be nonnegative and sum to <= 1"
            )
        if not 0.0 <= self.noise_ar1_rho.low <= self.noise_ar1_rho.high < 1.0:
            raise ValueError("noise AR(1) rho must be in [0, 1)")
        if self.drift_std_counts.low < 0.0:
            raise ValueError("drift standard deviation must be nonnegative")
        if self.drift_time_constant_s.low <= 0.0:
            raise ValueError("drift time constant must be positive")

    def as_dict(self) -> dict[str, object]:
        return {
            "status": SENSOR_DYNAMICS_STATUS,
            "spatial_first_ring_fraction": self.spatial_first_ring_fraction.as_dict(),
            "spatial_second_ring_fraction": self.spatial_second_ring_fraction.as_dict(),
            "randomize_source_phase": self.randomize_source_phase,
            "noise_ar1_rho": self.noise_ar1_rho.as_dict(),
            "drift_std_counts": self.drift_std_counts.as_dict(),
            "drift_time_constant_s": self.drift_time_constant_s.as_dict(),
            "observation_delivery": (
                None
                if self.observation_delivery is None
                else self.observation_delivery.as_dict()
            ),
        }


@dataclass(frozen=True, slots=True)
class TactileCalibrationProfile:
    """Versioned configuration for a provisional tactile Sim2Real model."""

    profile_id: str = "inline-provisional-v1"
    layout_sha256: str | None = None
    packet_size: int = 256
    packet_dtype: str = "uint8"
    transfer: TactileTransferProfile = field(default_factory=TactileTransferProfile)
    devices: tuple[TactileDeviceProfile, ...] = field(default_factory=tuple)
    randomization: TactileEpisodeRandomization = field(
        default_factory=TactileEpisodeRandomization
    )
    sensor_dynamics: TactileSensorDynamicsProfile = field(
        default_factory=TactileSensorDynamicsProfile
    )
    fixed_taxel_gain_variation: float = 0.0
    fixed_device_rate_variation: float = 0.0
    # Programmatic construction stays on the legacy behavior unless a caller
    # deliberately opts into the newer sensor-dynamics schema.
    schema_version: int = 1
    source_file_sha256: str | None = field(default=None, compare=False, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.profile_id, str) or not _PROFILE_ID_PATTERN.fullmatch(
            self.profile_id
        ):
            raise ValueError("tactile profile_id must be a plain stable identifier")
        if self.layout_sha256 is not None and (
            not isinstance(self.layout_sha256, str)
            or not _SHA256_PATTERN.fullmatch(self.layout_sha256)
        ):
            raise ValueError("tactile layout_sha256 must be 64 lowercase hex digits")
        if self.source_file_sha256 is not None and (
            not isinstance(self.source_file_sha256, str)
            or not _SHA256_PATTERN.fullmatch(self.source_file_sha256)
        ):
            raise ValueError("tactile source file SHA256 is invalid")
        if self.packet_size != 256 or self.packet_dtype != "uint8":
            raise ValueError("tactile packet contract must be uint8[256]")
        if self.schema_version not in _SUPPORTED_PROFILE_VERSIONS:
            raise ValueError(
                "unsupported tactile calibration profile version: "
                f"{self.schema_version}"
            )
        if (
            self.schema_version == 1
            and self.sensor_dynamics != TactileSensorDynamicsProfile()
        ):
            raise ValueError("profile v1 cannot configure sensor_dynamics")
        if not self.devices:
            raise ValueError("tactile calibration profile needs at least one device")
        names = tuple(device.name for device in self.devices)
        if len(set(names)) != len(names):
            raise ValueError("tactile profile device names must be unique")
        variations = (
            self.fixed_taxel_gain_variation,
            self.fixed_device_rate_variation,
        )
        if not all(np.isfinite(value) for value in variations):
            raise ValueError("fixed tactile variations must be finite")
        if not 0.0 <= self.fixed_taxel_gain_variation < 1.0:
            raise ValueError("fixed taxel gain variation must be in [0, 1)")
        if not 0.0 <= self.fixed_device_rate_variation < 1.0:
            raise ValueError("fixed device rate variation must be in [0, 1)")
        maximum_rate = max(device.sample_rate_hz for device in self.devices)
        maximum_rate *= 1.0 + self.fixed_device_rate_variation
        maximum_rate *= self.randomization.sample_rate_scale.high
        if not np.isfinite(maximum_rate) or maximum_rate > _MAX_DEVICE_SAMPLE_RATE_HZ:
            raise ValueError(
                "realized tactile sample rate must not exceed "
                f"{_MAX_DEVICE_SAMPLE_RATE_HZ:g} Hz"
            )

    @property
    def sha256(self) -> str:
        payload = json.dumps(
            self.as_dict(),
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        return hashlib.sha256(payload).hexdigest()

    def as_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "schema_version": self.schema_version,
            "profile_id": self.profile_id,
            "layout_sha256": self.layout_sha256,
            "packet_size": self.packet_size,
            "packet_dtype": self.packet_dtype,
            "force_mapping_status": FORCE_MAPPING_STATUS,
            "paired_force_calibrated": False,
            "transfer": self.transfer.as_dict(),
            "devices": [device.as_dict() for device in self.devices],
            "randomization": self.randomization.as_dict(),
            "fixed_taxel_gain_variation": float(self.fixed_taxel_gain_variation),
            "fixed_device_rate_variation": float(self.fixed_device_rate_variation),
        }
        if self.schema_version >= 2:
            payload["sensor_dynamics"] = self.sensor_dynamics.as_dict()
        return payload

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> TactileCalibrationProfile:
        """Parse one strict external profile without accepting silent extras."""

        payload = _strict_mapping(value, "tactile profile")
        if "schema_version" not in payload:
            raise ValueError("tactile profile has missing fields: schema_version")
        schema_version = payload["schema_version"]
        if type(schema_version) is not int:  # bool is not a schema version.
            raise ValueError("tactile profile schema_version must be an integer")
        if schema_version not in _SUPPORTED_PROFILE_VERSIONS:
            raise ValueError(
                f"unsupported tactile calibration profile version: {schema_version}"
            )
        expected = {
            "schema_version",
            "profile_id",
            "layout_sha256",
            "packet_size",
            "packet_dtype",
            "force_mapping_status",
            "paired_force_calibrated",
            "transfer",
            "devices",
            "randomization",
            "fixed_taxel_gain_variation",
            "fixed_device_rate_variation",
        }
        if schema_version >= 2:
            expected.add("sensor_dynamics")
        _require_exact_keys(payload, expected, "tactile profile")
        if payload["force_mapping_status"] != FORCE_MAPPING_STATUS:
            raise ValueError("tactile profile cannot claim a paired force mapping")
        if payload["paired_force_calibrated"] is not False:
            raise ValueError("tactile profiles do not support paired-force calibration")
        devices_raw = payload["devices"]
        if not isinstance(devices_raw, list):
            raise TypeError("tactile profile devices must be a list")
        return cls(
            schema_version=schema_version,
            profile_id=_plain_string(payload["profile_id"], "profile_id"),
            layout_sha256=_optional_string(payload["layout_sha256"], "layout_sha256"),
            packet_size=_strict_integer(payload["packet_size"], "packet_size"),
            packet_dtype=_plain_string(payload["packet_dtype"], "packet_dtype"),
            transfer=_transfer_from_dict(payload["transfer"]),
            devices=tuple(_device_from_dict(item) for item in devices_raw),
            randomization=_randomization_from_dict(payload["randomization"]),
            sensor_dynamics=(
                _sensor_dynamics_from_dict(payload["sensor_dynamics"])
                if schema_version >= 2
                else TactileSensorDynamicsProfile()
            ),
            fixed_taxel_gain_variation=_strict_float(
                payload["fixed_taxel_gain_variation"],
                "fixed_taxel_gain_variation",
            ),
            fixed_device_rate_variation=_strict_float(
                payload["fixed_device_rate_variation"],
                "fixed_device_rate_variation",
            ),
        )

    @classmethod
    def from_file(cls, path: str | Path) -> TactileCalibrationProfile:
        """Load strict JSON and retain only its exact SHA256, never its path."""

        raw = Path(path).read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        try:
            value = json.loads(
                raw.decode("utf-8"),
                object_pairs_hook=_unique_json_object,
                parse_constant=_reject_json_constant,
            )
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
            raise ValueError("tactile profile must be valid UTF-8 JSON") from exc
        profile = cls.from_dict(value)
        return replace(profile, source_file_sha256=digest)


@dataclass(frozen=True, slots=True)
class RealizedTactileEpisode:
    """Concrete deterministic parameters sampled for one episode."""

    index: int
    seed: int
    taxel_gain_scale: np.ndarray
    device_sample_rate_hz: np.ndarray
    device_noise_std_counts: np.ndarray
    taxel_preload: np.ndarray
    taxel_residual_bias_counts: np.ndarray
    source_phase_s: np.ndarray | None = None
    device_noise_ar1_rho: np.ndarray | None = None
    device_drift_std_counts: np.ndarray | None = None
    device_drift_time_constant_s: np.ndarray | None = None
    spatial_first_ring_fraction: float | None = None
    spatial_second_ring_fraction: float | None = None

    def __post_init__(self) -> None:
        _validate_seed(self.seed, "realized tactile episode seed")
        if self.index < 0:
            raise ValueError("realized tactile episode index must be nonnegative")
        optional_arrays = (
            self.source_phase_s,
            self.device_noise_ar1_rho,
            self.device_drift_std_counts,
            self.device_drift_time_constant_s,
        )
        if any(value is None for value in optional_arrays) != all(
            value is None for value in optional_arrays
        ):
            raise ValueError(
                "realized sensor dynamics arrays must be configured together"
            )
        fractions = (
            self.spatial_first_ring_fraction,
            self.spatial_second_ring_fraction,
        )
        if (fractions[0] is None) != (fractions[1] is None):
            raise ValueError("realized spatial fractions must be configured together")

    @property
    def sha256(self) -> str:
        digest = hashlib.sha256()
        digest.update(np.asarray(self.seed, dtype="<u8").tobytes())
        for value in (
            self.taxel_gain_scale,
            self.device_sample_rate_hz,
            self.device_noise_std_counts,
            self.taxel_preload,
            self.taxel_residual_bias_counts,
        ):
            digest.update(value.astype("<f8", copy=False).tobytes())
        if self.source_phase_s is not None:
            digest.update(b"sensor-dynamics-v2")
            for value in (
                self.source_phase_s,
                self.device_noise_ar1_rho,
                self.device_drift_std_counts,
                self.device_drift_time_constant_s,
            ):
                assert value is not None
                digest.update(value.astype("<f8", copy=False).tobytes())
            digest.update(
                np.asarray(
                    [
                        self.spatial_first_ring_fraction,
                        self.spatial_second_ring_fraction,
                    ],
                    dtype="<f8",
                ).tobytes()
            )
        return digest.hexdigest()

    def as_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "index": self.index,
            "seed": self.seed,
            "sha256": self.sha256,
            "device_sample_rate_hz": self.device_sample_rate_hz.tolist(),
            "device_noise_std_counts": self.device_noise_std_counts.tolist(),
            "taxel_gain_scale": _array_summary(self.taxel_gain_scale),
            "taxel_preload": _array_summary(self.taxel_preload),
            "taxel_residual_bias_counts": _array_summary(
                self.taxel_residual_bias_counts
            ),
        }
        if self.source_phase_s is not None:
            payload["source_phase_s"] = self.source_phase_s.tolist()
            payload["device_noise_ar1_rho"] = self.device_noise_ar1_rho.tolist()
            payload["device_drift_std_counts"] = self.device_drift_std_counts.tolist()
            payload["device_drift_time_constant_s"] = (
                self.device_drift_time_constant_s.tolist()
            )
            payload["spatial_first_ring_fraction"] = self.spatial_first_ring_fraction
            payload["spatial_second_ring_fraction"] = self.spatial_second_ring_fraction
        return payload


class TactileProfileSampler:
    """Realize fixed manufacturing and per-episode profile variation."""

    def __init__(
        self,
        profile: TactileCalibrationProfile,
        taxel_count: int,
        *,
        seed: int = 0,
        legacy_fixed_stream: bool = False,
        fixed_taxel_permutation: np.ndarray | None = None,
    ) -> None:
        if taxel_count < 1:
            raise ValueError("tactile profile sampler needs at least one taxel")
        self.profile = profile
        self.taxel_count = int(taxel_count)
        self.seed = _validate_seed(seed, "tactile profile base seed")
        self.episode_index = -1

        fixed_seed = (
            self.seed
            if legacy_fixed_stream
            else np.random.SeedSequence([self.seed, 0xF17ED])
        )
        fixed_rng = np.random.default_rng(fixed_seed)
        gain_variation = profile.fixed_taxel_gain_variation
        rate_variation = profile.fixed_device_rate_variation
        self._fixed_device_rate = fixed_rng.uniform(
            1.0 - rate_variation,
            1.0 + rate_variation,
            len(profile.devices),
        )
        self._fixed_taxel_gain = fixed_rng.uniform(
            1.0 - gain_variation,
            1.0 + gain_variation,
            self.taxel_count,
        )
        if fixed_taxel_permutation is not None:
            permutation = np.asarray(fixed_taxel_permutation, dtype=np.int64)
            if permutation.shape != (self.taxel_count,) or not np.array_equal(
                np.sort(permutation), np.arange(self.taxel_count)
            ):
                raise ValueError("fixed taxel permutation must contain every index")
            self._fixed_taxel_gain = self._fixed_taxel_gain[permutation]
        self._make_readonly(self._fixed_taxel_gain)
        self._make_readonly(self._fixed_device_rate)

    def sample(
        self, *, seed: int | None = None
    ) -> tuple[RealizedTactileEpisode, np.random.Generator]:
        """Draw one episode and return its runtime RNG.

        An explicit seed reproduces the same realized parameters and subsequent
        noise/dropout stream regardless of earlier resets. Without one, the
        base seed and monotonically increasing episode index define the stream.
        """

        episode_index = self.episode_index + 1
        episode_seed = self._episode_seed(seed, episode_index)
        rng = np.random.default_rng(episode_seed)
        randomization = self.profile.randomization
        taxel_gain = self._fixed_taxel_gain * randomization.gain_scale.sample(
            rng, self.taxel_count
        )
        nominal_rate = np.asarray(
            [device.sample_rate_hz for device in self.profile.devices],
            dtype=np.float64,
        )
        device_rate = (
            nominal_rate
            * self._fixed_device_rate
            * randomization.sample_rate_scale.sample(rng, len(self.profile.devices))
        )
        base_noise = np.asarray(
            [device.noise_std_counts for device in self.profile.devices],
            dtype=np.float64,
        )
        device_noise = base_noise + randomization.noise_std_counts.sample(
            rng, len(self.profile.devices)
        )
        preload = randomization.preload.sample(rng, self.taxel_count)
        residual_bias = randomization.residual_bias_counts.sample(rng, self.taxel_count)
        values = [
            taxel_gain,
            device_rate,
            device_noise,
            preload,
            residual_bias,
        ]
        dynamics_values: dict[str, object] = {}
        if self.profile.schema_version >= 2:
            dynamics = self.profile.sensor_dynamics
            device_count = len(self.profile.devices)
            if dynamics.randomize_source_phase:
                source_phase = rng.uniform(0.0, 1.0, device_count) / device_rate
            else:
                source_phase = np.asarray(
                    [device.phase_offset_s for device in self.profile.devices],
                    dtype=np.float64,
                )
            noise_rho = dynamics.noise_ar1_rho.sample(rng, device_count)
            drift_std = dynamics.drift_std_counts.sample(rng, device_count)
            drift_tau = dynamics.drift_time_constant_s.sample(rng, device_count)
            first_ring = float(dynamics.spatial_first_ring_fraction.sample(rng, 1)[0])
            second_ring = float(dynamics.spatial_second_ring_fraction.sample(rng, 1)[0])
            values.extend((source_phase, noise_rho, drift_std, drift_tau))
            dynamics_values = {
                "source_phase_s": source_phase,
                "device_noise_ar1_rho": noise_rho,
                "device_drift_std_counts": drift_std,
                "device_drift_time_constant_s": drift_tau,
                "spatial_first_ring_fraction": first_ring,
                "spatial_second_ring_fraction": second_ring,
            }
        for value in values:
            self._make_readonly(value)
        episode = RealizedTactileEpisode(
            index=episode_index,
            seed=episode_seed,
            taxel_gain_scale=taxel_gain,
            device_sample_rate_hz=device_rate,
            device_noise_std_counts=device_noise,
            taxel_preload=preload,
            taxel_residual_bias_counts=residual_bias,
            **dynamics_values,
        )
        self.episode_index = episode_index
        return episode, rng

    def _episode_seed(self, explicit: int | None, episode_index: int) -> int:
        if explicit is not None:
            return _validate_seed(explicit, "tactile episode seed")
        state = np.random.SeedSequence(
            [self.seed, episode_index, 0xE9150DE]
        ).generate_state(2, dtype=np.uint32)
        return int(state[0]) | (int(state[1]) << 32)

    @staticmethod
    def _make_readonly(value: np.ndarray) -> None:
        value.setflags(write=False)


def make_legacy_tactile_profile(
    device_count: int,
    *,
    device_names: Sequence[str] | None = None,
    profile_id: str = "juqiao-inline-provisional-v1",
    layout_sha256: str | None = None,
    sample_rate_hz: float | Sequence[float] = 14.0,
    stale_timeout_s: float | Sequence[float] = 0.1,
    dropout_probability: float | Sequence[float] = 0.0,
    noise_std_counts: float | Sequence[float] = 0.0,
    input_quantity: TactileInputQuantity = "force",
    gain: float = 4.0,
    gamma: float = 1.0,
    offset: float = 0.0,
    attack_tau_s: float = 0.0,
    release_tau_s: float = 0.0,
    output_deadband_counts: float = 0.0,
    saturation_count: int = 255,
    fixed_taxel_gain_variation: float = 0.08,
    fixed_device_rate_variation: float = 0.08,
    randomization: TactileEpisodeRandomization | None = None,
) -> TactileCalibrationProfile:
    """Build a versioned profile from the adapter's backward-compatible kwargs."""

    if device_count < 1:
        raise ValueError("device_count must be positive")
    names = (
        tuple(device_names)
        if device_names is not None
        else tuple(f"device_{index}" for index in range(device_count))
    )
    if len(names) != device_count:
        raise ValueError("device_names must match device_count")
    rates = _per_device(sample_rate_hz, device_count, "sample_rate_hz")
    stale = _per_device(stale_timeout_s, device_count, "stale_timeout_s")
    dropout = _per_device(
        dropout_probability,
        device_count,
        "dropout_probability",
    )
    noise = _per_device(noise_std_counts, device_count, "noise_std_counts")
    reference_rate = float(np.mean(rates))
    devices = tuple(
        TactileDeviceProfile(
            name=names[index],
            sample_rate_hz=float(rates[index]),
            stale_timeout_s=float(stale[index]),
            dropout_probability=float(dropout[index]),
            noise_std_counts=float(noise[index]),
            phase_offset_s=index / (device_count * reference_rate),
        )
        for index in range(device_count)
    )
    return TactileCalibrationProfile(
        schema_version=1,
        profile_id=profile_id,
        layout_sha256=layout_sha256,
        transfer=TactileTransferProfile(
            input_quantity=input_quantity,
            gain=gain,
            gamma=gamma,
            offset=offset,
            attack_tau_s=attack_tau_s,
            release_tau_s=release_tau_s,
            output_deadband_counts=output_deadband_counts,
            saturation_count=saturation_count,
        ),
        devices=devices,
        randomization=randomization or TactileEpisodeRandomization(),
        fixed_taxel_gain_variation=fixed_taxel_gain_variation,
        fixed_device_rate_variation=fixed_device_rate_variation,
    )


def tactile_input(
    normal_force_n: np.ndarray,
    area_m2: np.ndarray,
    quantity: TactileInputQuantity,
) -> np.ndarray:
    """Return force in N or nominal pressure in kPa without mutating sidecars."""

    force = np.asarray(normal_force_n, dtype=np.float64)
    area = np.asarray(area_m2, dtype=np.float64)
    if force.shape != area.shape:
        raise ValueError("tactile force and area must have matching shapes")
    if np.any(~np.isfinite(force)):
        raise ValueError("tactile normal force must be finite")
    force = np.maximum(force, 0.0)
    if quantity == "force":
        return force
    if quantity != "pressure":
        raise ValueError("tactile input quantity must be 'force' or 'pressure'")
    if np.any(~np.isfinite(area)) or np.any(area <= 0.0):
        raise ValueError("pressure input requires finite positive taxel areas")
    return force / area / 1000.0


def power_law_response(
    sensor_input: np.ndarray,
    transfer: TactileTransferProfile,
    realization: RealizedTactileEpisode,
) -> np.ndarray:
    """Convert physical input into pre-noise floating-point counts."""

    value = np.asarray(sensor_input, dtype=np.float64)
    if value.shape != realization.taxel_gain_scale.shape:
        raise ValueError("tactile input does not match realized taxel profile")
    effective = np.maximum(value + realization.taxel_preload - transfer.offset, 0.0)
    return transfer.gain * realization.taxel_gain_scale * effective**transfer.gamma


def asymmetric_first_order_filter(
    previous: np.ndarray,
    target: np.ndarray,
    dt: float,
    *,
    attack_tau_s: float,
    release_tau_s: float,
) -> np.ndarray:
    """Apply an exact discrete asymmetric first-order sensor response."""

    if dt < 0.0 or not np.isfinite(dt):
        raise ValueError("tactile filter dt must be finite and nonnegative")
    old = np.asarray(previous, dtype=np.float64)
    goal = np.asarray(target, dtype=np.float64)
    if old.shape != goal.shape:
        raise ValueError("tactile filter arrays must have matching shapes")
    tau = np.where(goal > old, attack_tau_s, release_tau_s)
    immediate = tau <= 0.0
    alpha = np.ones_like(goal)
    delayed = ~immediate
    alpha[delayed] = -np.expm1(-dt / tau[delayed])
    return old + alpha * (goal - old)


def quantize_counts(
    response: np.ndarray,
    *,
    noise_std_counts: float,
    transfer: TactileTransferProfile,
    rng: np.random.Generator | None,
) -> np.ndarray:
    """Add per-sample electronics noise, deadband, saturation, and quantize."""

    value = np.asarray(response, dtype=np.float64)
    if noise_std_counts > 0.0:
        if rng is None:
            raise ValueError("count noise requires a random generator")
        value = value + rng.normal(0.0, noise_std_counts, size=value.shape)
    value = np.maximum(value - transfer.output_deadband_counts, 0.0)
    return np.clip(
        np.rint(value),
        0,
        transfer.saturation_count,
    ).astype(np.uint8)


def _per_device(
    value: float | Sequence[float],
    count: int,
    name: str,
) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim == 0:
        array = np.full(count, float(array), dtype=np.float64)
    if array.shape != (count,):
        raise ValueError(f"{name} must be scalar or have shape ({count},)")
    return array


def load_tactile_calibration_profile(path: str | Path) -> TactileCalibrationProfile:
    """Load a strict versioned JSON profile."""

    return TactileCalibrationProfile.from_file(path)


def _transfer_from_dict(value: object) -> TactileTransferProfile:
    payload = _strict_mapping(value, "tactile transfer")
    _require_exact_keys(
        payload,
        {
            "input_quantity",
            "gain",
            "gamma",
            "offset",
            "attack_tau_s",
            "release_tau_s",
            "output_deadband_counts",
            "saturation_count",
        },
        "tactile transfer",
    )
    quantity = _plain_string(payload["input_quantity"], "input_quantity")
    if quantity not in {"force", "pressure"}:
        raise ValueError("tactile input_quantity must be 'force' or 'pressure'")
    return TactileTransferProfile(
        input_quantity=quantity,  # type: ignore[arg-type]
        gain=_strict_float(payload["gain"], "gain"),
        gamma=_strict_float(payload["gamma"], "gamma"),
        offset=_strict_float(payload["offset"], "offset"),
        attack_tau_s=_strict_float(payload["attack_tau_s"], "attack_tau_s"),
        release_tau_s=_strict_float(payload["release_tau_s"], "release_tau_s"),
        output_deadband_counts=_strict_float(
            payload["output_deadband_counts"],
            "output_deadband_counts",
        ),
        saturation_count=_strict_integer(
            payload["saturation_count"],
            "saturation_count",
        ),
    )


def _device_from_dict(value: object) -> TactileDeviceProfile:
    payload = _strict_mapping(value, "tactile device")
    _require_exact_keys(
        payload,
        {
            "name",
            "sample_rate_hz",
            "stale_timeout_s",
            "dropout_probability",
            "noise_std_counts",
            "phase_offset_s",
        },
        "tactile device",
    )
    return TactileDeviceProfile(
        name=_plain_string(payload["name"], "device name"),
        sample_rate_hz=_strict_float(payload["sample_rate_hz"], "sample_rate_hz"),
        stale_timeout_s=_strict_float(
            payload["stale_timeout_s"],
            "stale_timeout_s",
        ),
        dropout_probability=_strict_float(
            payload["dropout_probability"],
            "dropout_probability",
        ),
        noise_std_counts=_strict_float(
            payload["noise_std_counts"],
            "noise_std_counts",
        ),
        phase_offset_s=_strict_float(payload["phase_offset_s"], "phase_offset_s"),
    )


def _randomization_from_dict(value: object) -> TactileEpisodeRandomization:
    payload = _strict_mapping(value, "tactile episode randomization")
    _require_exact_keys(
        payload,
        {
            "gain_scale",
            "sample_rate_scale",
            "noise_std_counts",
            "preload",
            "residual_bias_counts",
        },
        "tactile episode randomization",
    )
    return TactileEpisodeRandomization(
        gain_scale=_range_from_dict(payload["gain_scale"], "gain_scale"),
        sample_rate_scale=_range_from_dict(
            payload["sample_rate_scale"],
            "sample_rate_scale",
        ),
        noise_std_counts=_range_from_dict(
            payload["noise_std_counts"],
            "noise_std_counts",
        ),
        preload=_range_from_dict(payload["preload"], "preload"),
        residual_bias_counts=_range_from_dict(
            payload["residual_bias_counts"],
            "residual_bias_counts",
        ),
    )


def _sensor_dynamics_from_dict(value: object) -> TactileSensorDynamicsProfile:
    payload = _strict_mapping(value, "tactile sensor dynamics")
    _require_exact_keys(
        payload,
        {
            "status",
            "spatial_first_ring_fraction",
            "spatial_second_ring_fraction",
            "randomize_source_phase",
            "noise_ar1_rho",
            "drift_std_counts",
            "drift_time_constant_s",
            "observation_delivery",
        },
        "tactile sensor dynamics",
    )
    if payload["status"] != SENSOR_DYNAMICS_STATUS:
        raise ValueError("tactile sensor dynamics must remain marked provisional")
    randomize_source_phase = payload["randomize_source_phase"]
    if type(randomize_source_phase) is not bool:
        raise TypeError("randomize_source_phase must be a boolean")
    delivery = payload["observation_delivery"]
    return TactileSensorDynamicsProfile(
        spatial_first_ring_fraction=_range_from_dict(
            payload["spatial_first_ring_fraction"],
            "spatial_first_ring_fraction",
        ),
        spatial_second_ring_fraction=_range_from_dict(
            payload["spatial_second_ring_fraction"],
            "spatial_second_ring_fraction",
        ),
        randomize_source_phase=randomize_source_phase,
        noise_ar1_rho=_range_from_dict(payload["noise_ar1_rho"], "noise_ar1_rho"),
        drift_std_counts=_range_from_dict(
            payload["drift_std_counts"],
            "drift_std_counts",
        ),
        drift_time_constant_s=_range_from_dict(
            payload["drift_time_constant_s"],
            "drift_time_constant_s",
        ),
        observation_delivery=(
            None if delivery is None else _observation_delivery_from_dict(delivery)
        ),
    )


def _observation_delivery_from_dict(
    value: object,
) -> TactileObservationDeliveryProfile:
    payload = _strict_mapping(value, "tactile observation delivery")
    _require_exact_keys(
        payload,
        {
            "record_rate_hz",
            "initial_probabilities",
            "transition_probabilities",
        },
        "tactile observation delivery",
    )
    initial = _strict_float_sequence(
        payload["initial_probabilities"],
        "observation initial probabilities",
    )
    rows_raw = payload["transition_probabilities"]
    if not isinstance(rows_raw, list):
        raise TypeError("observation transition probabilities must be a list")
    rows = tuple(
        _strict_float_sequence(row, "observation transition row") for row in rows_raw
    )
    return TactileObservationDeliveryProfile(
        record_rate_hz=_strict_float(
            payload["record_rate_hz"],
            "observation record_rate_hz",
        ),
        initial_probabilities=initial,
        transition_probabilities=rows,
    )


def _range_from_dict(value: object, name: str) -> UniformRange:
    payload = _strict_mapping(value, name)
    _require_exact_keys(payload, {"low", "high"}, name)
    return UniformRange(
        _strict_float(payload["low"], f"{name}.low"),
        _strict_float(payload["high"], f"{name}.high"),
    )


def _strict_float_sequence(value: object, name: str) -> tuple[float, ...]:
    if not isinstance(value, list):
        raise TypeError(f"{name} must be a list")
    return tuple(_strict_float(item, name) for item in value)


def _probability_vector(value: Sequence[float], name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.ndim != 1 or len(result) < 1:
        raise ValueError(f"{name} must be a nonempty vector")
    if np.any(~np.isfinite(result)) or np.any(result < 0.0):
        raise ValueError(f"{name} must be finite and nonnegative")
    if not np.isclose(result.sum(), 1.0, atol=1e-9, rtol=0.0):
        raise ValueError(f"{name} must sum to one")
    return result


def _strict_mapping(value: object, name: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be an object")
    if not all(isinstance(key, str) for key in value):
        raise ValueError(f"{name} keys must be strings")
    return dict(value)


def _require_exact_keys(
    payload: Mapping[str, object],
    expected: set[str],
    name: str,
) -> None:
    missing = sorted(expected - payload.keys())
    unknown = sorted(payload.keys() - expected)
    if missing:
        raise ValueError(f"{name} is missing fields: {', '.join(missing)}")
    if unknown:
        raise ValueError(f"{name} has unknown fields: {', '.join(unknown)}")


def _strict_float(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a number")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _strict_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    return int(value)


def _plain_string(value: object, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    return value


def _optional_string(value: object, name: str) -> str | None:
    return None if value is None else _plain_string(value, name)


def _validate_seed(value: object, name: str) -> int:
    seed = _strict_integer(value, name)
    if not 0 <= seed <= _UINT64_MAX:
        raise ValueError(f"{name} must be in uint64 range")
    return seed


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant is forbidden: {value}")


def _unique_json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON field is forbidden: {key}")
        result[key] = value
    return result


def _array_summary(value: np.ndarray) -> dict[str, float]:
    return {
        "minimum": float(np.min(value)),
        "maximum": float(np.max(value)),
        "mean": float(np.mean(value)),
    }
