"""Bounded-memory RGB video and lossless geometry recording for camera bundles."""

import json
from copy import deepcopy
from fractions import Fraction
from pathlib import Path

import numpy as np

GEOMETRY_CHUNK_SIZE = 8
GEOMETRY_FIELDS = ("depth_m", "normals_camera", "valid_depth", "valid_normals")


class CameraEpisodeWriter:
    def __init__(self, session: Path, episode: int, metadata: dict, fps: int):
        import av

        self._av = av
        self.metadata = deepcopy(metadata)
        self.cameras = self.metadata["cameras"]
        self._session = session
        self._episode = episode
        self._fps = fps
        self._streams = {}
        self._files = []
        self._geometry = []
        self._capture_count = 0
        self._row = 0
        self._last_time = None
        self._chunk_count = 0

    def append(self, frames, time):
        if set(frames) != set(self.cameras):
            raise ValueError("camera bundle does not match configured views")
        timestamps = {frame.timestamp for frame in frames.values()}
        if len(timestamps) != 1:
            raise ValueError("camera bundle must have one synchronized capture time")
        stamp = timestamps.pop()
        if not np.isfinite(stamp) or stamp > time + 1e-9:
            raise ValueError("camera timestamp must be finite and not in the future")
        if self._last_time is not None and stamp < self._last_time:
            raise ValueError("camera timestamps must be monotonic")
        for name, frame in frames.items():
            config = self.cameras[name]
            h, w = config["height"], config["width"]
            for attribute, shape, dtype in (
                ("rgb", (h, w, 3), np.uint8),
                ("depth_m", (h, w), np.float32),
                ("normals_camera", (h, w, 3), np.float32),
                ("valid_depth", (h, w), np.bool_),
                ("valid_normals", (h, w), np.bool_),
                ("camera_to_world", (4, 4), np.float64),
            ):
                value = getattr(frame, attribute)
                if (
                    value.shape != shape
                    or value.dtype != dtype
                    or not np.isfinite(value).all()
                ):
                    raise ValueError(
                        f"invalid {name}.{attribute} shape, dtype or values"
                    )
            appearance = frame.appearance_camera_to_world
            if appearance is not None and (
                appearance.shape != (4, 4) or not np.isfinite(appearance).all()
            ):
                raise ValueError(f"invalid {name}.appearance_camera_to_world")
        updated = self._last_time != stamp
        if updated:
            self._geometry.append(
                {
                    name: {
                        **{key: getattr(frame, key).copy() for key in GEOMETRY_FIELDS},
                        "timestamp": stamp,
                        "camera_to_world": frame.camera_to_world.copy(),
                        "appearance_camera_to_world": (
                            frame.camera_to_world
                            if frame.appearance_camera_to_world is None
                            else frame.appearance_camera_to_world
                        ).copy(),
                    }
                    for name, frame in frames.items()
                }
            )
            self._capture_count += 1
            self._last_time = stamp
            if len(self._geometry) == GEOMETRY_CHUNK_SIZE:
                self._flush_geometry()
        row = {}
        for name, frame in frames.items():
            if name not in self._streams:
                self._open_video(name)
            container, stream = self._streams[name]
            video = self._av.VideoFrame.from_ndarray(frame.rgb, format="rgb24")
            video.pts = self._row
            video.time_base = Fraction(1, self._fps)
            for packet in stream.encode(video):
                container.mux(packet)
            prefix = f"observation.camera.{name}"
            row.update(
                {
                    f"{prefix}.timestamp": stamp,
                    f"{prefix}.age_s": max(0.0, time - stamp),
                    f"{prefix}.sample_index": self._capture_count - 1,
                    f"{prefix}.updated": updated,
                    f"{prefix}.camera_to_world": frame.camera_to_world.copy(),
                    f"{prefix}.appearance_camera_to_world": (
                        frame.camera_to_world
                        if frame.appearance_camera_to_world is None
                        else frame.appearance_camera_to_world
                    ).copy(),
                }
            )
        self._row += 1
        return row

    def _open_video(self, name):
        config = self.cameras[name]
        path = (
            self._session
            / "videos/chunk-000"
            / f"observation.images.{name}"
            / f"episode_{self._episode:06d}.mp4"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        self._files.append(path)
        container = self._av.open(str(path), mode="w")
        stream = container.add_stream("libx264", rate=self._fps)
        stream.width, stream.height = config["width"], config["height"]
        stream.pix_fmt = "yuv420p"
        stream.options = {"preset": "ultrafast", "crf": "18"}
        self._streams[name] = container, stream

    def _flush_geometry(self):
        if not self._geometry:
            return
        path = (
            self._session
            / "geometry"
            / f"episode_{self._episode:06d}"
            / f"chunk_{self._chunk_count:06d}.npz"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays = {
            f"{name}.{field}": np.stack(
                [sample[name][field] for sample in self._geometry]
            )
            for name in self.cameras
            for field in self._geometry[0][name]
        }
        self._files.append(path)
        np.savez_compressed(path, **arrays)
        self._geometry.clear()
        self._chunk_count += 1

    def finish(self):
        self._flush_geometry()
        for container, stream in self._streams.values():
            for packet in stream.encode():
                container.mux(packet)
            container.close()
        self._streams.clear()
        return {
            "capture_count": self._capture_count,
            "geometry_chunks": self._chunk_count,
        }

    def abort(self):
        for container, _ in self._streams.values():
            container.close()
        self._streams.clear()
        self._geometry.clear()
        for path in self._files:
            path.unlink(missing_ok=True)


def camera_features(metadata, fps):
    return {
        f"observation.images.{name}": {
            "dtype": "video",
            "shape": [config["height"], config["width"], 3],
            "names": ["height", "width", "channel"],
            "info": {
                "video.height": config["height"],
                "video.width": config["width"],
                "video.codec": "h264",
                "video.pix_fmt": "yuv420p",
                "video.fps": fps,
                "video.channels": 3,
                "video.is_depth_map": False,
                "has_audio": False,
            },
        }
        for name, config in metadata["cameras"].items()
    }


def write_camera_metadata(meta, metadata):
    value = {
        **metadata,
        "geometry_storage": {
            "format": "npz_lossless_float32_and_bool",
            "path": "geometry/episode_{episode_index:06d}/chunk_{chunk_index:06d}.npz",
            "samples_per_chunk": GEOMETRY_CHUNK_SIZE,
            "lookup": "chunk=sample_index//8; offset=sample_index%8",
            "keys": "{camera}.{depth_m,normals_camera,valid_depth,valid_normals,timestamp,camera_to_world,appearance_camera_to_world}",
        },
    }
    (meta / "cameras.json").write_text(json.dumps(value, indent=2) + "\n")
