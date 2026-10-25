"""Privacy-preserving statistics for real-robot tactile packets."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .tactile_skin import TACTILE_DEVICE_DIM, TACTILE_DEVICE_NAMES

TACTILE_COLUMNS = tuple(
    f"observation.tactile_{device}" for device in TACTILE_DEVICE_NAMES
)
_EPISODE_FILE = re.compile(r"episode_(\d+)\.parquet$")
_COUNT_BINS = 256
_PACKET_SUM_BINS = TACTILE_DEVICE_DIM * 255 + 1


def _histogram_quantile(histogram: np.ndarray, quantile: float) -> float:
    total = int(histogram.sum())
    if total == 0:
        return 0.0
    target = max(1, int(np.ceil(quantile * total)))
    cumulative = np.cumsum(histogram, dtype=np.int64)
    return float(np.searchsorted(cumulative, target, side="left"))


def _quantiles(
    histogram: np.ndarray,
    values: tuple[float, ...],
) -> dict[str, float]:
    return {
        f"p{round(100 * quantile):02d}": _histogram_quantile(histogram, quantile)
        for quantile in values
    }


@dataclass(slots=True)
class DeviceAuditAccumulator:
    """Accumulate exact packet statistics without retaining raw frames."""

    deadband_counts: int = 2
    frames: int = 0
    zero_packets: int = 0
    repeated_transitions: int = 0
    comparable_transitions: int = 0
    value_histogram: np.ndarray = field(
        default_factory=lambda: np.zeros(_COUNT_BINS, dtype=np.int64)
    )
    active_histogram: np.ndarray = field(
        default_factory=lambda: np.zeros(TACTILE_DEVICE_DIM + 1, dtype=np.int64)
    )
    packet_sum_histogram: np.ndarray = field(
        default_factory=lambda: np.zeros(_PACKET_SUM_BINS, dtype=np.int64)
    )
    channel_histogram: np.ndarray = field(
        default_factory=lambda: np.zeros(
            (TACTILE_DEVICE_DIM, _COUNT_BINS), dtype=np.int64
        )
    )
    quiet_channel_histogram_including_zero: np.ndarray = field(
        default_factory=lambda: np.zeros(
            (TACTILE_DEVICE_DIM, _COUNT_BINS), dtype=np.int64
        )
    )
    quiet_nonzero_channel_histogram: np.ndarray = field(
        default_factory=lambda: np.zeros(
            (TACTILE_DEVICE_DIM, _COUNT_BINS), dtype=np.int64
        )
    )
    quiet_frames_including_zero: int = 0
    quiet_nonzero_frames: int = 0
    _previous: np.ndarray | None = field(default=None, repr=False)

    def begin_episode(self) -> None:
        self._previous = None

    def update(self, packets: np.ndarray) -> None:
        values = _validate_packets(packets)
        if len(values) == 0:
            return
        self.frames += len(values)
        packet_sum = values.sum(axis=1, dtype=np.int32)
        active = np.count_nonzero(values > self.deadband_counts, axis=1)
        self.zero_packets += int(np.count_nonzero(packet_sum == 0))
        self.value_histogram += np.bincount(values.reshape(-1), minlength=_COUNT_BINS)
        self.active_histogram += np.bincount(active, minlength=TACTILE_DEVICE_DIM + 1)
        self.packet_sum_histogram += np.bincount(packet_sum, minlength=_PACKET_SUM_BINS)
        for channel in range(TACTILE_DEVICE_DIM):
            self.channel_histogram[channel] += np.bincount(
                values[:, channel], minlength=_COUNT_BINS
            )

        if self._previous is not None:
            first_repeated = bool(np.array_equal(values[0], self._previous))
            self.repeated_transitions += int(first_repeated)
            self.comparable_transitions += 1
        if len(values) > 1:
            repeated = np.all(values[1:] == values[:-1], axis=1)
            self.repeated_transitions += int(np.count_nonzero(repeated))
            self.comparable_transitions += len(repeated)
        self._previous = values[-1].copy()

    def update_quiet(self, packets: np.ndarray, maximum_packet_sum: int) -> None:
        values = _validate_packets(packets)
        packet_sum = values.sum(axis=1, dtype=np.int32)
        quiet = values[packet_sum <= maximum_packet_sum]
        quiet_nonzero = quiet[np.any(quiet != 0, axis=1)]
        self.quiet_frames_including_zero += len(quiet)
        self.quiet_nonzero_frames += len(quiet_nonzero)
        for channel in range(TACTILE_DEVICE_DIM):
            self.quiet_channel_histogram_including_zero[channel] += np.bincount(
                quiet[:, channel], minlength=_COUNT_BINS
            )
            self.quiet_nonzero_channel_histogram[channel] += np.bincount(
                quiet_nonzero[:, channel], minlength=_COUNT_BINS
            )

    def quiet_threshold(self, quantile: float) -> int:
        histogram = self.packet_sum_histogram.copy()
        histogram[0] = 0
        return int(_histogram_quantile(histogram, quantile))

    def report(self, *, fps: float, quiet_quantile: float) -> dict[str, object]:
        if self.frames == 0:
            raise ValueError("cannot report an empty tactile stream")
        positive = self.value_histogram.copy()
        positive[0] = 0
        above_deadband = self.value_histogram.copy()
        above_deadband[: self.deadband_counts + 1] = 0
        repeat_fraction = (
            self.repeated_transitions / self.comparable_transitions
            if self.comparable_transitions
            else 0.0
        )
        observed_change_rate = (
            fps
            * (self.comparable_transitions - self.repeated_transitions)
            / self.comparable_transitions
            if self.comparable_transitions
            else 0.0
        )
        channel_nonzero = 1.0 - self.channel_histogram[:, 0] / self.frames
        channel_above_deadband = (
            self.channel_histogram[:, self.deadband_counts + 1 :].sum(axis=1)
            / self.frames
        )
        quiet_mean = _histogram_mean(self.quiet_channel_histogram_including_zero)
        quiet_p90 = _row_quantile(self.quiet_channel_histogram_including_zero, 0.9)
        quiet_nonzero_mean = _histogram_mean(self.quiet_nonzero_channel_histogram)
        quiet_nonzero_p90 = _row_quantile(self.quiet_nonzero_channel_histogram, 0.9)
        return {
            "frames": self.frames,
            "repeat_fraction": repeat_fraction,
            "observed_change_rate_hz_lower_bound": observed_change_rate,
            "zero_packet_fraction": self.zero_packets / self.frames,
            "element_nonzero_fraction": int(positive.sum())
            / (self.frames * TACTILE_DEVICE_DIM),
            "element_above_deadband_fraction": int(above_deadband.sum())
            / (self.frames * TACTILE_DEVICE_DIM),
            "deadband_counts": self.deadband_counts,
            "positive_count_quantiles": _quantiles(
                positive, (0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99)
            ),
            "active_channel_quantiles": _quantiles(
                self.active_histogram, (0.1, 0.5, 0.9, 0.95, 0.99)
            ),
            "packet_sum_quantiles": _quantiles(
                self.packet_sum_histogram, (0.1, 0.5, 0.9, 0.95, 0.99)
            ),
            "channel_nonzero_rate": channel_nonzero.tolist(),
            "channel_above_deadband_rate": channel_above_deadband.tolist(),
            "quiet_frames_including_zero": self.quiet_frames_including_zero,
            "quiet_nonzero_frames": self.quiet_nonzero_frames,
            "quiet_packet_sum_quantile": quiet_quantile,
            "quiet_packet_sum_maximum": self.quiet_threshold(quiet_quantile),
            "quiet_channel_mean_counts_including_zero": quiet_mean.tolist(),
            "quiet_channel_p90_counts_including_zero": quiet_p90.tolist(),
            "quiet_nonzero_channel_mean_counts": quiet_nonzero_mean.tolist(),
            "quiet_nonzero_channel_p90_counts": quiet_nonzero_p90.tolist(),
        }


def audit_tactile_sources(
    sources: list[Path],
    *,
    labels: list[str] | None = None,
    fps: float | None = None,
    deadband_counts: int = 2,
    quiet_quantile: float = 0.2,
) -> dict[str, object]:
    """Audit LeRobot tactile parquet sources and return anonymous summaries."""

    if not sources:
        raise ValueError("at least one tactile source is required")
    if labels is not None and len(labels) != len(sources):
        raise ValueError("labels must match the number of sources")
    if fps is not None and (not np.isfinite(fps) or fps <= 0.0):
        raise ValueError("fps must be finite and positive")
    if not 0 <= deadband_counts < 255:
        raise ValueError("deadband_counts must be in [0, 255)")
    if not 0.0 < quiet_quantile < 1.0:
        raise ValueError("quiet_quantile must be in (0, 1)")

    reports = []
    for index, source in enumerate(sources):
        label = labels[index] if labels is not None else f"source_{index + 1:03d}"
        reports.append(
            _audit_source(
                source,
                label=label,
                fps_override=fps,
                deadband_counts=deadband_counts,
                quiet_quantile=quiet_quantile,
            )
        )
    return {
        "schema_version": 1,
        "kind": "tactile_sim2real_distribution_audit",
        "sources": reports,
        "interpretation": {
            "force_mapping": (
                "unidentified: task marginal distributions cannot determine "
                "newtons-to-counts calibration"
            ),
            "observed_change_rate": (
                "lower bound: a newly published hardware packet can equal the "
                "previous packet"
            ),
            "quiet_channel_statistics": (
                "post-baseline raw-count residuals; they may mix electronics "
                "noise, garment preload, baseline drift, unlabeled contact, and "
                "stale-link zero fill; inclusive-zero and conditional-nonzero "
                "statistics are reported separately"
            ),
            "privacy": "raw tactile frames and absolute source paths are omitted",
        },
    }


def write_audit_report(path: Path, report: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")


def _audit_source(
    source: Path,
    *,
    label: str,
    fps_override: float | None,
    deadband_counts: int,
    quiet_quantile: float,
) -> dict[str, object]:
    files, metadata, skipped = _source_files(source)
    metadata_fps = metadata.get("fps")
    if fps_override is None and metadata_fps is None:
        raise ValueError(f"dataset fps is unknown for {label!r}; pass fps explicitly")
    source_fps = float(fps_override if fps_override is not None else metadata_fps)
    if not np.isfinite(source_fps) or source_fps <= 0.0:
        raise ValueError(f"invalid dataset fps for {label!r}")
    accumulators = [
        DeviceAuditAccumulator(deadband_counts=deadband_counts)
        for _ in TACTILE_DEVICE_NAMES
    ]
    _scan_files(files, accumulators)
    thresholds = [item.quiet_threshold(quiet_quantile) for item in accumulators]
    _scan_quiet_files(files, accumulators, thresholds)
    return {
        "label": label,
        "fingerprint_sha256": _fingerprint(files),
        "fps": source_fps,
        "parquet_files": len(files),
        "discarded_episode_files_skipped": skipped,
        "devices": {
            device: accumulator.report(fps=source_fps, quiet_quantile=quiet_quantile)
            for device, accumulator in zip(
                TACTILE_DEVICE_NAMES, accumulators, strict=True
            )
        },
    }


def _source_files(source: Path) -> tuple[list[Path], dict[str, object], int]:
    source = Path(source)
    if source.is_file():
        if source.suffix != ".parquet":
            raise ValueError(f"tactile source is not parquet: {source.name}")
        return [source], {}, 0
    if not source.is_dir():
        raise FileNotFoundError(f"tactile source not found: {source}")
    info_path = source / "meta" / "info.json"
    metadata = json.loads(info_path.read_text()) if info_path.is_file() else {}
    discarded = {int(value) for value in metadata.get("discarded_episode_indices", [])}
    candidates = sorted(source.glob("data/**/*.parquet"))
    files = [path for path in candidates if _episode_index(path) not in discarded]
    if not files:
        raise ValueError(f"no usable tactile parquet files in {source.name}")
    return files, metadata, len(candidates) - len(files)


def _episode_index(path: Path) -> int | None:
    match = _EPISODE_FILE.search(path.name)
    return int(match.group(1)) if match else None


def _scan_files(
    files: list[Path],
    accumulators: list[DeviceAuditAccumulator],
) -> None:
    parquet = _import_parquet()
    for path in files:
        for accumulator in accumulators:
            accumulator.begin_episode()
        for batch in parquet.ParquetFile(path).iter_batches(
            batch_size=4096, columns=list(TACTILE_COLUMNS)
        ):
            names = batch.schema.names
            for column, accumulator in zip(TACTILE_COLUMNS, accumulators, strict=True):
                if column not in names:
                    raise ValueError(f"missing tactile column {column} in {path.name}")
                accumulator.update(_arrow_packets(batch.column(names.index(column))))


def _scan_quiet_files(
    files: list[Path],
    accumulators: list[DeviceAuditAccumulator],
    thresholds: list[int],
) -> None:
    parquet = _import_parquet()
    for path in files:
        for batch in parquet.ParquetFile(path).iter_batches(
            batch_size=4096, columns=list(TACTILE_COLUMNS)
        ):
            names = batch.schema.names
            for column, accumulator, threshold in zip(
                TACTILE_COLUMNS, accumulators, thresholds, strict=True
            ):
                accumulator.update_quiet(
                    _arrow_packets(batch.column(names.index(column))), threshold
                )


def _arrow_packets(array) -> np.ndarray:
    import pyarrow as pa

    array_type = array.type
    is_fixed = pa.types.is_fixed_size_list(array_type)
    is_list = pa.types.is_list(array_type)
    wrong_fixed_size = is_fixed and array_type.list_size != TACTILE_DEVICE_DIM
    if (
        not (is_fixed or is_list)
        or wrong_fixed_size
        or not pa.types.is_uint8(array_type.value_type)
    ):
        raise ValueError("tactile parquet columns must use list<uint8> packets of 256")
    if array.null_count:
        raise ValueError("tactile packet columns must not contain nulls")
    try:
        packets = np.asarray(array.to_pylist())
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("tactile packet columns must contain uint8 values") from exc
    return _validate_packets(packets)


def _validate_packets(packets: np.ndarray) -> np.ndarray:
    values = np.asarray(packets)
    if values.shape != (len(values), TACTILE_DEVICE_DIM):
        raise ValueError("tactile packets must have shape (frames, 256)")
    if values.dtype != np.uint8:
        if not np.issubdtype(values.dtype, np.number):
            raise ValueError("tactile packets must contain uint8 values")
        if (
            np.any(~np.isfinite(values))
            or np.any((values < 0) | (values > 255))
            or np.any(values != np.rint(values))
        ):
            raise ValueError("tactile packets must contain uint8 values")
        values = values.astype(np.uint8)
    return values


def _fingerprint(files: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in files:
        file_digest = hashlib.sha256()
        with path.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                file_digest.update(chunk)
        digest.update(path.name.encode())
        digest.update(file_digest.digest())
    return digest.hexdigest()


def _histogram_mean(histogram: np.ndarray) -> np.ndarray:
    counts = histogram.sum(axis=1)
    weights = np.arange(histogram.shape[1], dtype=np.float64)
    total = histogram @ weights
    return np.divide(total, counts, out=np.zeros_like(total), where=counts > 0)


def _row_quantile(histogram: np.ndarray, quantile: float) -> np.ndarray:
    return np.asarray(
        [_histogram_quantile(row, quantile) for row in histogram], dtype=np.float64
    )


def _import_parquet():
    try:
        from pyarrow import parquet
    except ImportError as exc:
        raise RuntimeError(
            "tactile parquet audit requires: uv sync --extra recording"
        ) from exc
    return parquet
