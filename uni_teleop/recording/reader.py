"""Read recorded RGB/geometry bundles without loading a complete episode."""

import json
from dataclasses import dataclass
from pathlib import Path
from zipfile import BadZipFile

import numpy as np


class EpisodeReadError(ValueError):
    """A dataset is missing an expected file or contains inconsistent data."""


@dataclass(frozen=True)
class RecordedCamera:
    rgb: np.ndarray
    timestamp: float | None = None
    age_s: float | None = None
    sample_index: int | None = None
    updated: bool | None = None
    depth_m: np.ndarray | None = None
    normals_camera: np.ndarray | None = None
    valid_depth: np.ndarray | None = None
    valid_normals: np.ndarray | None = None
    camera_to_world: np.ndarray | None = None
    appearance_camera_to_world: np.ndarray | None = None


@dataclass(frozen=True)
class RecordedRow:
    index: int
    sim_time: float
    timestamp: float
    cameras: dict[str, RecordedCamera]
    values: dict


def _json(path):
    try:
        value = json.loads(path.read_text())
        if not isinstance(value, dict):
            raise TypeError("expected a JSON object")
        return value
    except (OSError, ValueError, TypeError) as exc:
        raise EpisodeReadError(f"Cannot read JSON metadata {path}: {exc}") from exc


class EpisodeReader:
    """Read one dataset episode; RGB is indexed by row, geometry by capture.

    Parquet is streamed in small batches and video is decoded one frame at a time.
    Each geometry field decompresses at most one recording chunk (normally eight
    captures), rather than stacking every capture in an episode.
    """

    def __init__(self, session: str | Path, episode: int = 0):
        import pyarrow.parquet as pq

        self.session = Path(session).resolve()
        self.episode = episode
        if episode < 0:
            raise EpisodeReadError("Episode index must be nonnegative")
        self.info = _json(self.session / "meta/info.json")
        published = self.info.get("total_episodes")
        if not isinstance(published, int) or published < 0:
            raise EpisodeReadError("meta/info.json needs a nonnegative total_episodes")
        if episode >= published:
            raise EpisodeReadError(
                f"Episode {episode} is not published: meta/info.json lists {published} completed episodes"
            )
        camera_path = self.session / "meta/cameras.json"
        self.camera_metadata = _json(camera_path) if camera_path.exists() else None
        features = self.info.get("features", {})
        self.camera_names = tuple(
            key.removeprefix("observation.images.")
            for key, spec in features.items()
            if key.startswith("observation.images.") and spec.get("dtype") == "video"
        )
        if self.camera_metadata is not None:
            if set(self.camera_metadata["cameras"]) != set(self.camera_names):
                raise EpisodeReadError(
                    "Camera metadata and video feature names disagree"
                )
            storage = self.camera_metadata.get("geometry_storage", {})
            self.samples_per_chunk = storage.get("samples_per_chunk")
            if (
                not isinstance(self.samples_per_chunk, int)
                or self.samples_per_chunk < 1
            ):
                raise EpisodeReadError(
                    "meta/cameras.json needs positive samples_per_chunk"
                )
        self.data_path = self._path(self.info["data_path"])
        try:
            self._parquet = pq.ParquetFile(self.data_path)
        except (OSError, ValueError) as exc:
            raise EpisodeReadError(
                f"Cannot open episode rows {self.data_path}: {exc}"
            ) from exc
        self.length = self._parquet.metadata.num_rows
        self.columns = set(self._parquet.schema_arrow.names)
        self.fps = float(self.info.get("fps", 0))
        if not np.isfinite(self.fps) or self.fps <= 0:
            raise EpisodeReadError("meta/info.json needs a finite positive fps")
        if self.camera_metadata is None and any(
            key.startswith("observation.camera.") for key in self.columns
        ):
            raise EpisodeReadError(
                "Camera capture columns exist but meta/cameras.json is missing; "
                "copy the camera metadata with the dataset instead of treating it as RGB-only"
            )
        required = {"sim_time", "timestamp", "frame_index"}
        if self.camera_metadata is not None:
            required.update(
                f"observation.camera.{name}.{field}"
                for name in self.camera_names
                for field in ("timestamp", "age_s", "sample_index", "updated")
            )
        missing = required - self.columns
        if missing:
            raise EpisodeReadError(
                f"Episode is missing columns: {', '.join(sorted(missing))}"
            )

    def _path(self, template, **kwargs):
        try:
            relative = template.format(
                episode_index=self.episode,
                episode_chunk=self.episode // self.info.get("chunks_size", 1000),
                **kwargs,
            )
        except (KeyError, ValueError) as exc:
            raise EpisodeReadError(
                f"Invalid dataset path template {template!r}"
            ) from exc
        path = (self.session / relative).resolve()
        if not path.is_relative_to(self.session):
            raise EpisodeReadError(f"Dataset path escapes the session: {relative}")
        return path

    def _rows(self, columns=None):
        try:
            for batch in self._parquet.iter_batches(batch_size=32, columns=columns):
                yield from batch.to_pylist()
        except (OSError, ValueError) as exc:
            raise EpisodeReadError(
                f"Cannot read episode rows {self.data_path}: {exc}"
            ) from exc

    def read(self, row: int = 0) -> RecordedRow:
        if row < 0 or row >= self.length:
            raise EpisodeReadError(
                f"Row {row} outside episode range [0, {self.length})"
            )
        values = next(value for index, value in enumerate(self._rows()) if index == row)
        cameras = {}
        for name in self.camera_names:
            rgb = self._rgb(name, row)
            if self.camera_metadata is None:
                cameras[name] = RecordedCamera(rgb)
                continue
            prefix = f"observation.camera.{name}"
            sample = values[f"{prefix}.sample_index"]
            if not isinstance(sample, int) or sample < 0:
                raise EpisodeReadError(
                    f"Invalid {name} sample_index at row {row}: {sample}"
                )
            geometry = self._geometry(name, sample, values)
            cameras[name] = RecordedCamera(
                rgb=rgb,
                timestamp=float(values[f"{prefix}.timestamp"]),
                age_s=float(values[f"{prefix}.age_s"]),
                sample_index=sample,
                updated=bool(values[f"{prefix}.updated"]),
                **geometry,
            )
        return RecordedRow(
            row, float(values["sim_time"]), float(values["timestamp"]), cameras, values
        )

    def _rgb(self, name, row):
        import av

        path = self._path(
            self.info["video_path"], video_key=f"observation.images.{name}"
        )
        try:
            with av.open(str(path)) as container:
                for index, frame in enumerate(container.decode(video=0)):
                    if index == row:
                        rgb = frame.to_ndarray(format="rgb24")
                        expected = self.info["features"][f"observation.images.{name}"][
                            "shape"
                        ]
                        if list(rgb.shape) != expected:
                            raise EpisodeReadError(
                                f"Video dimensions disagree with metadata: {path}"
                            )
                        return rgb
        except (OSError, ValueError, av.error.FFmpegError) as exc:
            raise EpisodeReadError(
                f"Cannot decode RGB row {row} from {path}: {exc}"
            ) from exc
        raise EpisodeReadError(f"Video ends before row {row}: {path}")

    def _geometry(self, name, sample, values):
        chunk, offset = divmod(sample, self.samples_per_chunk)
        storage = self.camera_metadata["geometry_storage"]
        path = self._path(storage["path"], chunk_index=chunk)
        config = self.camera_metadata["cameras"][name]
        h, w = config["height"], config["width"]
        shapes = {
            "depth_m": (h, w),
            "normals_camera": (h, w, 3),
            "valid_depth": (h, w),
            "valid_normals": (h, w),
            "camera_to_world": (4, 4),
            "appearance_camera_to_world": (4, 4),
            "timestamp": (),
        }
        fields = {}
        try:
            with np.load(path, allow_pickle=False) as saved:
                for key, shape in shapes.items():
                    array = saved[f"{name}.{key}"]
                    if (
                        array.shape[1:] != shape
                        or not offset < len(array) <= self.samples_per_chunk
                    ):
                        raise ValueError(
                            f"invalid shape or capture offset for {name}.{key}"
                        )
                    value = array[offset].copy()
                    if not np.isfinite(value).all():
                        raise ValueError(f"nonfinite {name}.{key}")
                    if key.startswith("valid_") and value.dtype != np.bool_:
                        raise ValueError(
                            f"{name}.{key} must be a boolean validity mask"
                        )
                    fields[key] = value
        except (OSError, ValueError, TypeError, KeyError, EOFError, BadZipFile) as exc:
            raise EpisodeReadError(
                f"Cannot read {name} capture {sample} from {path}: {exc}. "
                "Check that the recording finished and its geometry directory was copied."
            ) from exc
        prefix = f"observation.camera.{name}"
        if not np.isclose(
            fields.pop("timestamp"), values[f"{prefix}.timestamp"], atol=1e-7, rtol=0
        ):
            raise EpisodeReadError(
                f"Geometry timestamp disagrees with row for {name} capture {sample}: {path}"
            )
        for key in ("camera_to_world", "appearance_camera_to_world"):
            recorded = values.get(f"{prefix}.{key}")
            if recorded is not None and not np.allclose(
                fields[key], np.asarray(recorded).reshape(4, 4)
            ):
                raise EpisodeReadError(
                    f"Geometry pose disagrees with row for {name} capture {sample}: {path}"
                )
        return fields

    def audit(self, selected: RecordedRow | None = None) -> dict:
        """Check all row clocks, without decoding all video/geometry captures.

        This is a temporal audit, not a full file-integrity scan. If a selected
        row is supplied, include its decoded camera and tactile statistics.
        """
        columns = ["sim_time", "timestamp", "frame_index"]
        columns += sorted(
            key
            for key in self.columns
            if key.startswith("observation.camera.")
            and not key.endswith("camera_to_world")
        )
        errors, count, previous = [], 0, None
        stats = {
            name: {
                "capture_count": 0 if self.camera_metadata is not None else None,
                "max_age_s": 0.0 if self.camera_metadata is not None else None,
            }
            for name in self.camera_names
        }

        def report(index, message):
            nonlocal count
            count += 1
            if len(errors) < 100:
                errors.append({"row": index, "message": message})

        for index, row in enumerate(self._rows(columns)):
            now = row["sim_time"]
            if not np.isfinite(now) or not np.isfinite(row["timestamp"]):
                report(index, "Nonfinite row time")
            if row["frame_index"] != index:
                report(index, "frame_index does not match video row")
            if not np.isclose(row["timestamp"], index / self.fps, rtol=2e-7, atol=1e-6):
                report(index, "Video row timestamp does not match dataset fps")
            if previous is not None and (
                now <= previous["sim_time"] or row["timestamp"] <= previous["timestamp"]
            ):
                report(index, "Row clocks are not strictly increasing")
            if previous is not None and not np.isclose(
                now - previous["sim_time"], 1 / self.fps, rtol=0, atol=1e-6
            ):
                report(index, "Simulation cadence differs from dataset fps")
            stamps = []
            if self.camera_metadata is not None:
                for name in self.camera_names:
                    prefix = f"observation.camera.{name}"
                    stamp, age = row[f"{prefix}.timestamp"], row[f"{prefix}.age_s"]
                    sample, updated = (
                        row[f"{prefix}.sample_index"],
                        row[f"{prefix}.updated"],
                    )
                    stamps.append(stamp)
                    if (
                        not np.isfinite(stamp)
                        or not np.isfinite(age)
                        or stamp > now + 1e-7
                        or age < 0
                        or not np.isclose(age, now - stamp, rtol=0, atol=1e-6)
                    ):
                        report(index, f"{name}: invalid capture timestamp/age")
                    stats[name]["max_age_s"] = max(stats[name]["max_age_s"], age)
                    if previous is None:
                        valid = sample == 0 and updated
                    else:
                        last = previous[f"{prefix}.sample_index"]
                        last_stamp = previous[f"{prefix}.timestamp"]
                        valid = (
                            sample == last + 1 and updated and stamp > last_stamp
                        ) or (sample == last and not updated and stamp == last_stamp)
                    if not valid:
                        report(
                            index,
                            f"{name}: sample_index, updated and capture time disagree",
                        )
                    stats[name]["capture_count"] += int(bool(updated))
                if len(set(stamps)) > 1:
                    report(index, "Camera bundle capture times are not synchronized")
            previous = row
        result = {
            "session": str(self.session),
            "episode": self.episode,
            "rows": self.length,
            "geometry_available": self.camera_metadata is not None,
            "rgb_storage": "Decoded H264 RGB, lossy; geometry sidecars are lossless.",
            "scope": "All row timing; RGB and geometry decoded only for the selected row, not full media integrity.",
            "temporal": {"ok": count == 0, "error_count": count, "errors": errors},
            "cameras": stats,
        }
        if self.camera_metadata is None:
            result["geometry_note"] = (
                "No geometry metadata: RGB-only legacy recording, depth/normals unavailable."
            )
        episodes_path = self.session / "meta/episodes.jsonl"
        if episodes_path.exists():
            try:
                with episodes_path.open() as stream:
                    for line in stream:
                        episode = json.loads(line)
                        if episode.get("episode_index") == self.episode:
                            for key in ("tasks", "task_context", "task_outcome"):
                                if key in episode:
                                    result[key] = episode[key]
                            break
            except (OSError, ValueError) as exc:
                raise EpisodeReadError(
                    f"Cannot read episode metadata {episodes_path}: {exc}"
                ) from exc
        if selected is not None:
            result["selected"] = self._selected_summary(selected)
        return result

    def _selected_summary(self, selected):
        views = {}
        for name, frame in selected.cameras.items():
            entry = {
                "shape": list(frame.rgb.shape),
                "capture_timestamp": frame.timestamp,
                "age_s": frame.age_s,
                "sample_index": frame.sample_index,
                "updated": frame.updated,
                "geometry_available": frame.depth_m is not None,
            }
            if frame.depth_m is not None:
                valid = frame.depth_m[frame.valid_depth]
                entry.update(
                    depth_units=self.camera_metadata.get("depth_units", "unspecified"),
                    valid_depth_fraction=float(frame.valid_depth.mean()),
                    valid_normal_fraction=float(frame.valid_normals.mean()),
                    depth_range_m=[float(valid.min()), float(valid.max())]
                    if valid.size
                    else None,
                )
            views[name] = entry
        tactile = {}
        for name in ("vest", "left_arm", "right_arm"):
            key = f"observation.tactile_{name}"
            if key in selected.values:
                values = np.asarray(selected.values[key])
                tactile[name] = {
                    "channels": values.size,
                    "nonzero_channels": int(np.count_nonzero(values)),
                    "maximum_raw": int(values.max()),
                    "note": "Raw uint8 channels, not force-calibrated contact counts.",
                }
        force = selected.values.get("observation.tactile.normal_force_n")
        if force is not None:
            force = np.asarray(force)
            tactile["physical"] = {
                "taxels": force.size,
                "normal_force_sum_n": float(force.sum()),
                "maximum_taxel_force_n": float(force.max()),
            }
        source_time = selected.values.get("simulation.tactile_source_time")
        if source_time is not None:
            tactile["device_source_time"] = source_time
            tactile["device_age_s"] = [
                selected.sim_time - stamp if stamp >= 0 else None
                for stamp in source_time
            ]
        return {
            "row": selected.index,
            "sim_time": selected.sim_time,
            "timestamp": selected.timestamp,
            "cameras": views,
            "tactile": tactile,
        }
