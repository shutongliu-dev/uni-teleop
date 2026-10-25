import html
import json
import shutil
import tempfile
import warnings
from collections import Counter
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from fractions import Fraction
from pathlib import Path

import numpy as np

from uni_teleop.recording.camera import (
    CameraEpisodeWriter,
    camera_features,
    write_camera_metadata,
)
from sonic_mujoco.contact import MAX_CONTACTS, ContactFrame
from sonic_mujoco.tactile import TactileFrame
from sonic_mujoco.tactile_skin import (
    TACTILE_DEVICE_DIM,
    TACTILE_DEVICE_NAMES,
    JuQiaoSkinLayout,
    JuQiaoTactileFrame,
)
from uni_teleop.core.command import TeleopCommand
from uni_teleop.devices.pico.xr import PicoControls

DEFAULT_TASKS = {
    "empty": "move the G1 with full-body teleoperation",
    "sweep": (
        "Use your forearm to sweep all objects across the divider from one side "
        "of the table to the other."
    ),
    "chair_lean": (
        "Walk to the chair from the front, turn around, adjust your position, "
        "and gently lean back against the backrest."
    ),
    "door_elbow": "Open the door using your elbow and shoulder.",
    "basket_loading": (
        "Hold the basket steady while objects are progressively loaded into it."
    ),
    "bucket_carry": (
        "Carry the container to the designated area while holding it securely "
        "against your torso."
    ),
    "plush_carry": (
        "Carry the soft plush toy to the designated area while holding it "
        "securely against your torso."
    ),
    "univlat_sweep": "Sweep all five objects across the blue divider without dropping them.",
    "univlat_basket": "Receive the basket, hold it steady while it is loaded, and release when it is taken.",
    "back_tap": "Walk forward while your back is touched and stop when the touch is released.",
    "human_hug": "Approach the person, gently embrace with both arms, then release and stand still.",
    "composed_cleanup": "Sweep the rubbish into the box, lift and carry the box, then deposit it in the bin.",
}

_CONTACT_FIELDS = (
    "robot_body_id",
    "other_body_id",
    "position",
    "normal_force",
    "tangent_force",
    "normal_impulse",
    "sample_count",
    "count",
)

_EPISODE_TACTILE_METADATA_FIELDS = (
    "episode_profile",
    "episode_seed",
    "realized_device_sample_rate_hz",
    "taxel_gain_sha256",
)


def _static_tactile_metadata(metadata: dict) -> dict:
    return {
        key: value
        for key, value in metadata.items()
        if key not in _EPISODE_TACTILE_METADATA_FIELDS
    }


@dataclass(frozen=True, slots=True)
class _TactileField:
    name: str
    key: str
    attribute: str
    dtype: type[np.generic]


_TACTILE_FIELDS = (
    _TactileField("physical_force", "force_body_n", "force", np.float32),
    _TactileField(
        "physical_normal_force", "normal_force_n", "normal_force", np.float32
    ),
    _TactileField(
        "physical_tangent_force",
        "tangent_force_body_n",
        "tangent_force",
        np.float32,
    ),
    _TactileField("physical_impulse", "impulse_body_ns", "impulse", np.float32),
    _TactileField(
        "physical_normal_impulse",
        "normal_impulse_ns",
        "normal_impulse",
        np.float32,
    ),
    _TactileField(
        "physical_tangent_impulse",
        "tangent_impulse_body_ns",
        "tangent_impulse",
        np.float32,
    ),
    _TactileField("contact", "contact", "contact", np.bool_),
    _TactileField("other_body_id", "other_body_id", "other_body_id", np.int32),
    _TactileField("other_geom_id", "other_geom_id", "other_geom_id", np.int32),
    _TactileField("sample_count", "sample_count", "sample_count", np.int32),
)


def _fit_video_frame(
    image: np.ndarray,
    *,
    width: int = 640,
    height: int = 480,
) -> np.ndarray:
    """Resize and letterbox RGB without changing the camera aspect ratio."""

    source = np.asarray(image, dtype=np.uint8)
    if source.ndim != 3 or source.shape[2] != 3 or min(source.shape[:2]) < 1:
        raise ValueError("recorded video frame must have shape (height, width, 3)")
    if source.shape == (height, width, 3):
        return np.ascontiguousarray(source)
    scale = min(width / source.shape[1], height / source.shape[0])
    resized_width = max(1, round(source.shape[1] * scale))
    resized_height = max(1, round(source.shape[0] * scale))
    x_index = np.minimum(
        (np.arange(resized_width) * source.shape[1] / resized_width).astype(int),
        source.shape[1] - 1,
    )
    y_index = np.minimum(
        (np.arange(resized_height) * source.shape[0] / resized_height).astype(int),
        source.shape[0] - 1,
    )
    resized = source[y_index[:, None], x_index[None, :]]
    output = np.zeros((height, width, 3), dtype=np.uint8)
    x_offset = (width - resized_width) // 2
    y_offset = (height - resized_height) // 2
    output[
        y_offset : y_offset + resized_height,
        x_offset : x_offset + resized_width,
    ] = resized
    return output


class _VideoWriter:
    def __init__(
        self,
        path: Path,
        model,
        data,
        fps: int,
        *,
        render_fallback: bool,
    ) -> None:
        try:
            import av
        except ImportError as exc:
            raise RuntimeError(
                "video recording requires: uv sync --extra recording"
            ) from exc

        path.parent.mkdir(parents=True, exist_ok=True)
        self._av = av
        self._data = data
        self._renderer = None
        if render_fallback:
            try:
                import mujoco
            except ImportError as exc:
                raise RuntimeError(
                    "video recording requires: uv sync --extra recording"
                ) from exc
            self._renderer = mujoco.Renderer(model, height=480, width=640)
        self._container = av.open(str(path), mode="w")
        self._stream = self._container.add_stream("libx264", rate=fps)
        self._stream.width = 640
        self._stream.height = 480
        self._stream.pix_fmt = "yuv420p"
        self._stream.options = {"preset": "ultrafast", "crf": "23"}
        self._time_base = Fraction(1, fps)
        self._frame_index = 0

    def add_frame(self, image: np.ndarray | None = None) -> None:
        if image is None:
            if self._renderer is None:
                raise RuntimeError("an external RGB frame is required")
            self._renderer.update_scene(self._data, camera="head_camera")
            image = self._renderer.render()
        image = _fit_video_frame(image)
        frame = self._av.VideoFrame.from_ndarray(image, format="rgb24")
        frame.pts = self._frame_index
        frame.time_base = self._time_base
        for packet in self._stream.encode(frame):
            self._container.mux(packet)
        self._frame_index += 1

    def close(self) -> None:
        for packet in self._stream.encode():
            self._container.mux(packet)
        self._container.close()
        if self._renderer is not None:
            self._renderer.close()

    def cancel(self, path: Path) -> None:
        self._container.close()
        if self._renderer is not None:
            self._renderer.close()
        path.unlink(missing_ok=True)


class EpisodeRecorder:
    """Write aligned MuJoCo episodes in the real-robot LeRobot layout."""

    def __init__(
        self,
        directory: str | Path,
        scene: str,
        *,
        model=None,
        data=None,
        body_names: tuple[str, ...] = (),
        fps: int = 50,
        task: str | None = None,
        record_video: bool = True,
        video_source: str = "mujoco_head_camera",
        video_metadata: dict | None = None,
        tactile_layout: JuQiaoSkinLayout | None = None,
        tactile_metadata: dict | None = None,
        camera_metadata: dict | None = None,
        task_metadata: dict | None = None,
    ) -> None:
        if not isinstance(fps, (int, np.integer)) or isinstance(fps, bool) or fps <= 0:
            raise ValueError("recording fps must be a positive integer")
        if record_video and camera_metadata is None and (model is None or data is None):
            raise ValueError("model and data are required for video recording")
        if camera_metadata is not None and not record_video:
            raise ValueError("camera bundles require record_video=True")
        self.directory = Path(directory).resolve()
        self.camera_metadata = deepcopy(camera_metadata)
        self.task_metadata = deepcopy(task_metadata)
        self._camera_writer = None
        self.scene = scene
        self.model = model
        self.data = data
        self.body_names = body_names
        self.fps = int(fps)
        self.task = task or DEFAULT_TASKS.get(scene, scene)
        self.record_video = record_video
        self.video_source = video_source
        self.video_metadata = dict(video_metadata or {})
        self.tactile_layout = tactile_layout
        if tactile_layout is not None and tactile_metadata is None:
            raise ValueError("tactile layout requires sensor metadata")
        self.tactile_metadata = dict(tactile_metadata or {})
        self._validate_tactile_metadata(self.tactile_metadata)
        self._frames: list[dict[str, np.ndarray | float | int]] = []
        self._session: Path | None = None
        self._video: _VideoWriter | None = None
        self._video_path: Path | None = None
        self._episodes: list[dict] = []
        self._stats: list[dict] = []
        self._features: dict | None = None
        self._active_tactile_metadata: dict | None = None
        self._active_task_context: dict | None = None
        self._frame_schema: dict | None = None
        self._write_failed = False
        self.active = False
        self.last_preview: Path | None = None

    @property
    def session_directory(self) -> Path | None:
        return self._session

    @property
    def frame_count(self) -> int:
        return len(self._frames)

    @property
    def duration_seconds(self) -> float:
        return self.frame_count / self.fps

    @property
    def write_failed(self) -> bool:
        """Whether an interrupted write requires aborting the current episode."""
        return self._write_failed

    def start(
        self, *, tactile_metadata: dict | None = None, task_context: dict | None = None
    ) -> None:
        if self.active:
            raise RuntimeError("recording is already active; finish or abort it first")
        if tactile_metadata is not None and self.tactile_layout is None:
            raise ValueError("tactile metadata requires a configured tactile layout")
        if self.tactile_layout is not None and tactile_metadata is None:
            raise ValueError("each tactile episode requires current sensor metadata")
        episode_metadata = tactile_metadata or {}
        self._validate_tactile_metadata(episode_metadata)
        missing_episode_fields = [
            key
            for key in _EPISODE_TACTILE_METADATA_FIELDS
            if key not in episode_metadata
        ]
        if self.tactile_layout is not None and missing_episode_fields:
            raise ValueError(
                "tactile episode metadata is missing: "
                + ", ".join(missing_episode_fields)
            )
        if _static_tactile_metadata(episode_metadata) != _static_tactile_metadata(
            self.tactile_metadata
        ):
            raise ValueError("tactile profile cannot change within one session")
        self._active_tactile_metadata = (
            {
                key: deepcopy(episode_metadata[key])
                for key in _EPISODE_TACTILE_METADATA_FIELDS
                if key in episode_metadata
            }
            if self.tactile_layout is not None
            else None
        )
        self._frames.clear()
        self._active_task_context = deepcopy(task_context)
        self._write_failed = False
        self.active = True

    def append(
        self,
        *,
        time: float,
        qpos: np.ndarray,
        qvel: np.ndarray,
        ctrl: np.ndarray,
        command: TeleopCommand,
        token: np.ndarray,
        action: np.ndarray,
        controls: PicoControls,
        contacts: ContactFrame,
        tactile: TactileFrame | None = None,
        tactile_suit: JuQiaoTactileFrame | None = None,
        video_frame: np.ndarray | None = None,
        camera_pose: np.ndarray | None = None,
        appearance_camera_pose: np.ndarray | None = None,
        camera_frames: dict | None = None,
        task_state: dict | None = None,
    ) -> None:
        if not self.active:
            return
        if self._write_failed:
            raise RuntimeError(
                "episode media write failed; abort before recording again"
            )
        if (
            self.record_video
            and self.video_source == "reconstructed_scene_composite"
            and self.camera_metadata is None
            and video_frame is None
        ):
            raise RuntimeError(
                "reconstructed-scene recording requires a composite RGB frame"
            )
        if self.tactile_layout is not None and (
            tactile is None or tactile_suit is None
        ):
            raise RuntimeError("tactile recording requires physical and suit frames")
        if self.tactile_layout is None and (
            tactile is not None or tactile_suit is not None
        ):
            raise RuntimeError("tactile frames require a configured tactile layout")
        if (self.camera_metadata is None) != (camera_frames is None):
            raise ValueError(
                "camera frames and camera metadata must be configured together"
            )
        frame = {
            "observation.qpos": np.asarray(qpos).copy(),
            "observation.qvel": np.asarray(qvel).copy(),
            "action.ctrl": np.asarray(ctrl).copy(),
            "teleop.smpl_joints": command.smpl_joints[-1].reshape(-1).copy(),
            "teleop.body_quat_w": command.root_quaternion[-1].copy(),
            "teleop.reference_joint_position": command.joint_position[-1].copy(),
            "teleop.hand_joint_position": (
                command.hand_joint_position[-1].copy()
                if command.hand_joint_position is not None
                else np.zeros(40, dtype=np.float64)
            ),
            "teleop.neck_joint_position": (
                command.neck_joint_position[-1].copy()
                if command.neck_joint_position is not None
                else np.zeros(2, dtype=np.float64)
            ),
            "action.motion_token": np.asarray(token).copy(),
            "action.policy": np.asarray(action).copy(),
            "teleop.pico_input": self._pico_input(controls),
            "observation.contact.robot_body_id": contacts.robot_body_id.copy(),
            "observation.contact.other_body_id": contacts.other_body_id.copy(),
            "observation.contact.position": contacts.position.reshape(-1).copy(),
            "observation.contact.normal_force": contacts.normal_force.copy(),
            "observation.contact.tangent_force": contacts.tangent_force.copy(),
            "observation.contact.normal_impulse": contacts.normal_impulse.copy(),
            "observation.contact.sample_count": contacts.sample_count.copy(),
            "observation.contact.count": int(contacts.count),
            "sim_time": float(time),
        }
        if camera_pose is not None:
            frame["observation.camera_pose"] = np.asarray(camera_pose).copy()
        if appearance_camera_pose is not None:
            frame["observation.appearance_camera_pose"] = np.asarray(
                appearance_camera_pose
            ).copy()
        if tactile is not None and tactile_suit is not None:
            self._append_tactile(frame, tactile, tactile_suit)
        if task_state is not None:
            frame.update({f"task.{key}": value for key, value in task_state.items()})
        self._validate_frame(frame)
        try:
            self._ensure_episode_started()
            if self._camera_writer is not None:
                frame.update(self._camera_writer.append(camera_frames, time))
            if self._video is not None:
                self._video.add_frame(video_frame)
            self._frames.append(frame)
        except BaseException:
            # A codec can fail after writing part of a frame. Never publish the
            # episode as valid when video and control rows may have diverged.
            self._write_failed = True
            raise

    @staticmethod
    def _schema(frame: dict) -> dict:
        return {
            key: (
                np.asarray(value).shape,
                "string"
                if np.asarray(value).dtype.kind in "US"
                else str(np.asarray(value).dtype),
            )
            for key, value in frame.items()
            if not key.startswith("observation.camera.")
        }

    def _validate_frame(self, frame: dict) -> None:
        for key, value in frame.items():
            array = np.asarray(value)
            if array.dtype.kind not in "biufUS" or (
                array.dtype.kind in "biuf" and not np.isfinite(array).all()
            ):
                raise ValueError(f"invalid recording values: {key}")
        if self._frames:
            interval = frame["sim_time"] - self._frames[-1]["sim_time"]
            if interval <= 0:
                raise ValueError(
                    "recorded simulation time must increase; reset starts a new episode"
                )
            if not np.isclose(interval, 1 / self.fps, rtol=0, atol=1e-6):
                raise ValueError(
                    "simulation interval must match recording fps; a control row was skipped"
                )
        expected = self._schema(self._frames[0]) if self._frames else self._frame_schema
        if expected is not None and self._schema(frame) != expected:
            raise ValueError(
                "recording fields, shapes or dtypes changed within the session"
            )

    def finish(self) -> Path | None:
        if not self.active:
            return None
        if self._write_failed:
            raise RuntimeError(
                "episode media write failed; abort instead of saving incomplete data"
            )
        if not self._frames:
            self.abort()
            return None
        assert self._session is not None
        episode_index = len(self._episodes)
        previous_stats_count = len(self._stats)
        previous_features = self._features
        try:
            arrays = self._stack_frames(episode_index)
            data_path = self._episode_path("data/chunk-000", episode_index, "parquet")
            data_path.parent.mkdir(parents=True, exist_ok=True)
            episode = {
                "episode_index": episode_index,
                "tasks": [self.task],
                "length": len(self._frames),
            }
            if self._video is not None:
                self._video.close()
                self._video = None
            if self._camera_writer is not None:
                episode["cameras"] = self._camera_writer.finish()
            temporary = data_path.with_suffix(".parquet.partial")
            self._write_parquet(temporary, arrays)
            temporary.replace(data_path)
            if self._active_tactile_metadata is not None:
                episode["tactile"] = self._active_tactile_metadata
            if self._active_task_context is not None:
                episode["task_context"] = self._active_task_context
            outcome = {
                key.removeprefix("task."): np.asarray(value).tolist()
                for key, value in self._frames[-1].items()
                if key.startswith("task.")
            }
            if outcome:
                episode["task_outcome"] = {
                    "status": "success"
                    if outcome.get("success")
                    else "failed"
                    if outcome.get("failed")
                    else "incomplete",
                    "final_state": outcome,
                }
            stats = {
                "episode_index": episode_index,
                "stats": self._episode_stats(arrays),
            }
            self._features = self._features or self._build_features(arrays)
            self._episodes.append(episode)
            self._stats.append(stats)
            self._write_metadata()
        except BaseException:
            del self._episodes[episode_index:]
            del self._stats[previous_stats_count:]
            self._features = previous_features
            self._write_failed = True
            raise
        self.active = False
        self._camera_writer = None
        self._frame_schema = self._schema(self._frames[0])
        self._frames.clear()
        self._video_path = None
        self._active_tactile_metadata = None
        self._active_task_context = None
        self.last_preview = None
        try:
            self.last_preview = self._write_contact_preview(episode_index, arrays)
        except OSError as exc:
            warnings.warn(
                f"Episode saved, but preview could not be written: {exc}", stacklevel=2
            )
        return data_path

    def abort(self) -> None:
        self.active = False
        if self._camera_writer is not None:
            self._camera_writer.abort()
            self._camera_writer = None
        self._frames.clear()
        self._active_tactile_metadata = None
        self._active_task_context = None
        self._write_failed = False
        if self._video_path is not None:
            if self._video is not None:
                self._video.cancel(self._video_path)
            else:
                self._video_path.unlink(missing_ok=True)
            self._video = None
            self._video_path = None
        if self._session is not None:
            index = len(self._episodes)
            self._episode_path("data/chunk-000", index, "parquet.partial").unlink(
                missing_ok=True
            )
            self._episode_path("data/chunk-000", index, "parquet").unlink(
                missing_ok=True
            )
        if self._session is not None and not self._episodes:
            shutil.rmtree(self._session, ignore_errors=True)
            self._session = None

    def _ensure_episode_started(self) -> None:
        if self._session is None:
            self._session = self._new_session_directory()
        if self.camera_metadata is not None:
            if self._camera_writer is None:
                self._camera_writer = CameraEpisodeWriter(
                    self._session, len(self._episodes), self.camera_metadata, self.fps
                )
            return
        if self.record_video and self._video is None:
            index = len(self._episodes)
            self._video_path = self._episode_path(
                "videos/chunk-000/observation.images.ego_view", index, "mp4"
            )
            self._video = _VideoWriter(
                self._video_path,
                self.model,
                self.data,
                self.fps,
                render_fallback=self.video_source == "mujoco_head_camera",
            )

    def _new_session_directory(self) -> Path:
        self.directory.mkdir(parents=True, exist_ok=True)
        name = datetime.now().astimezone().strftime("%Y-%m-%d-%H-%M-%S")
        path = self.directory / name
        suffix = 1
        while True:
            try:
                path.mkdir()
                return path
            except FileExistsError:
                path = self.directory / f"{name}-{suffix:02d}"
                suffix += 1

    def _episode_path(self, group: str, index: int, suffix: str) -> Path:
        assert self._session is not None
        return self._session / group / f"episode_{index:06d}.{suffix}"

    def _stack_frames(self, episode_index: int) -> dict[str, np.ndarray]:
        arrays = {
            key: np.stack([np.asarray(frame[key]) for frame in self._frames])
            for key in self._frames[0]
        }
        length = len(self._frames)
        arrays.update(
            {
                "timestamp": np.arange(length, dtype=np.float32) / self.fps,
                "frame_index": np.arange(length, dtype=np.int64),
                "episode_index": np.full(length, episode_index, dtype=np.int64),
                "index": np.arange(
                    sum(item["length"] for item in self._episodes),
                    sum(item["length"] for item in self._episodes) + length,
                    dtype=np.int64,
                ),
                "task_index": np.zeros(length, dtype=np.int64),
            }
        )
        return arrays

    @staticmethod
    def _write_parquet(path: Path, arrays: dict[str, np.ndarray]) -> None:
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise RuntimeError(
                "parquet recording requires: uv sync --extra recording"
            ) from exc

        columns = {}
        tactile_keys = {f"observation.tactile_{name}" for name in TACTILE_DEVICE_NAMES}
        for key, values in arrays.items():
            values = np.asarray(values)
            if values.ndim == 1:
                columns[key] = pa.array(values)
                continue
            if key in tactile_keys:
                columns[key] = pa.array(
                    values.tolist(),
                    type=pa.list_(pa.uint8()),
                )
                continue
            width = int(np.prod(values.shape[1:]))
            flat = pa.array(values.reshape(-1))
            columns[key] = pa.FixedSizeListArray.from_arrays(flat, width)
        pq.write_table(pa.table(columns), path, compression="zstd")

    def _build_features(self, arrays: dict[str, np.ndarray]) -> dict:
        features = {}
        if self.camera_metadata is not None:
            features.update(camera_features(self.camera_metadata, self.fps))
        elif self.record_video:
            features["observation.images.ego_view"] = {
                "dtype": "video",
                "shape": [480, 640, 3],
                "names": ["height", "width", "channel"],
                "info": {
                    "video.height": 480,
                    "video.width": 640,
                    "video.codec": "h264",
                    "video.pix_fmt": "yuv420p",
                    "video.is_depth_map": False,
                    "video.fps": self.fps,
                    "video.channels": 3,
                    "has_audio": False,
                },
            }
        for key, values in arrays.items():
            shape = list(values.shape[1:]) or [1]
            names = (
                [f"raw_{index:03d}" for index in range(1, TACTILE_DEVICE_DIM + 1)]
                if key
                in {f"observation.tactile_{name}" for name in TACTILE_DEVICE_NAMES}
                else None
            )
            features[key] = {
                "dtype": "string" if values.dtype.kind in "US" else str(values.dtype),
                "shape": shape,
                "names": names,
            }
        return features

    def _write_metadata(self) -> None:
        assert self._session is not None and self._features is not None
        meta = self._session / "meta"
        meta.mkdir(parents=True, exist_ok=True)
        # Individual replacements are atomic, not the whole manifest set. Keep
        # originals until publication finishes so exceptions can roll it back.
        staged = Path(tempfile.mkdtemp(prefix=".metadata-", dir=self._session))
        keep_recovery = False
        try:
            self._write_metadata_to(staged)
            manifests = sorted(
                staged.iterdir(), key=lambda item: item.name == "info.json"
            )
            previous = staged / "previous"
            previous.mkdir()
            restore = staged / "restore"
            restore.mkdir()
            for path in manifests:
                current = meta / path.name
                if current.exists():
                    shutil.copy2(current, previous / path.name)
            attempted = []
            try:
                for path in manifests:
                    attempted.append(path.name)
                    path.replace(meta / path.name)
            except BaseException as publish_error:
                failures = []
                for name in reversed(attempted):
                    try:
                        original = previous / name
                        if original.exists():
                            replacement = restore / name
                            shutil.copy2(original, replacement)
                            replacement.replace(meta / name)
                        else:
                            (meta / name).unlink(missing_ok=True)
                    except BaseException as rollback_error:  # noqa: BLE001
                        # Preserve recovery copies even if rollback is interrupted.
                        failures.append(f"{name}: {rollback_error}")
                if failures:
                    keep_recovery = True
                    raise RuntimeError(
                        "Metadata publication and rollback failed; prior manifests "
                        f"are retained at {previous}. Restore before using the dataset. "
                        + "; ".join(failures)
                    ) from publish_error
                raise
        finally:
            if not keep_recovery:
                shutil.rmtree(staged, ignore_errors=True)

    def _write_metadata_to(self, meta: Path) -> None:
        if self.camera_metadata is not None:
            write_camera_metadata(meta, self.camera_metadata)
        if self.tactile_layout is not None:
            layout = self.tactile_layout.as_dict(include_taxels=True)
            layout["sha256"] = self.tactile_layout.sha256
            (meta / "tactile_layout.json").write_text(
                json.dumps(layout, indent=2, ensure_ascii=False) + "\n"
            )
        total_frames = sum(item["length"] for item in self._episodes)
        info = {
            "codebase_version": "v2.1",
            "robot_type": "g1_29dof_mujoco",
            "total_episodes": len(self._episodes),
            "total_frames": total_frames,
            "total_tasks": 1,
            "total_videos": (
                len(self._episodes) * len(self.camera_metadata["cameras"])
                if self.camera_metadata is not None
                else len(self._episodes)
                if self.record_video
                else 0
            ),
            "total_chunks": 1,
            "chunks_size": 1000,
            "fps": self.fps,
            "splits": {"train": f"0:{len(self._episodes)}"},
            "data_path": (
                "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
            ),
            "video_path": (
                "videos/chunk-{episode_chunk:03d}/{video_key}/"
                "episode_{episode_index:06d}.mp4"
            ),
            "features": self._features,
            "script_config": {
                "scene": self.scene,
                "video_source": self.video_source,
                "video_metadata": self.video_metadata,
                "cameras": self.camera_metadata,
                "task_protocol": self.task_metadata,
                "geometry_sidecar": (
                    "meta/cameras.json" if self.camera_metadata is not None else None
                ),
                "contact_source": "mujoco",
                "max_contacts_per_frame": MAX_CONTACTS,
                "contact_body_names": list(self.body_names),
                "tactile": (
                    {
                        **_static_tactile_metadata(self.tactile_metadata),
                        "layout_sha256": self.tactile_layout.sha256,
                        "physical_units": {
                            "force": "N",
                            "impulse": "N*s",
                            "frame": "owning_mujoco_body_local",
                        },
                        "hardware_compatible_fields": [
                            f"observation.tactile_{name}"
                            for name in TACTILE_DEVICE_NAMES
                        ],
                        "episode_metadata": "meta/episodes.jsonl[].tactile",
                    }
                    if self.tactile_layout is not None
                    else None
                ),
            },
            "discarded_episode_indices": [],
        }
        (meta / "info.json").write_text(
            json.dumps(info, indent=2, ensure_ascii=False) + "\n"
        )
        modality = {
            "state": {
                "qpos": {"original_key": "observation.qpos"},
                "qvel": {"original_key": "observation.qvel"},
            },
            "action": {
                "ctrl": {"original_key": "action.ctrl"},
                "motion_token": {"original_key": "action.motion_token"},
                "policy": {"original_key": "action.policy"},
            },
            "video": (
                {
                    name: {"original_key": f"observation.images.{name}"}
                    for name in self.camera_metadata["cameras"]
                }
                if self.camera_metadata is not None
                else (
                    {"ego_view": {"original_key": "observation.images.ego_view"}}
                    if self.record_video
                    else {}
                )
            ),
            "annotation": {"human.task_description": {"original_key": "task_index"}},
            "contact": {
                name: {"original_key": f"observation.contact.{name}"}
                for name in _CONTACT_FIELDS
            },
            "tactile": (
                {
                    name: {"original_key": f"observation.tactile_{name}"}
                    for name in TACTILE_DEVICE_NAMES
                }
                if self.tactile_layout is not None
                else {}
            ),
            "simulation_tactile": (
                {
                    **{
                        field.name: {"original_key": f"observation.tactile.{field.key}"}
                        for field in _TACTILE_FIELDS
                    },
                    "duration": {"original_key": "observation.tactile.duration_s"},
                    "mapped_contact_count": {
                        "original_key": "observation.tactile.mapped_contact_count"
                    },
                    "unmapped_contact_count": {
                        "original_key": "observation.tactile.unmapped_contact_count"
                    },
                    "source_time": {"original_key": "simulation.tactile_source_time"},
                    "updated": {"original_key": "simulation.tactile_updated"},
                }
                if self.tactile_layout is not None
                else {}
            ),
        }
        (meta / "modality.json").write_text(
            json.dumps(modality, indent=2, ensure_ascii=False) + "\n"
        )
        self._write_jsonl(meta / "tasks.jsonl", [{"task_index": 0, "task": self.task}])
        self._write_jsonl(meta / "episodes.jsonl", self._episodes)
        self._write_jsonl(meta / "episodes_stats.jsonl", self._stats)

    def _validate_tactile_metadata(self, metadata: dict) -> None:
        if self.tactile_layout is None:
            return
        record_rate = metadata.get("record_rate_hz")
        if record_rate is not None and not np.isclose(float(record_rate), self.fps):
            raise ValueError("tactile record rate must match episode fps")
        layout_sha256 = metadata.get("layout_sha256")
        if layout_sha256 is not None and layout_sha256 != self.tactile_layout.sha256:
            raise ValueError("tactile metadata does not match the configured layout")

    @staticmethod
    def _write_jsonl(path: Path, rows: list[dict]) -> None:
        text = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
        path.write_text(text)

    @staticmethod
    def _episode_stats(arrays: dict[str, np.ndarray]) -> dict:
        stats = {}
        for key, values in arrays.items():
            if key in {"episode_index", "index", "task_index"}:
                continue
            values = np.asarray(values)
            if values.dtype.kind in "US":
                continue
            stats[key] = {
                "min": np.min(values, axis=0).reshape(-1).tolist(),
                "max": np.max(values, axis=0).reshape(-1).tolist(),
                "mean": np.mean(values, axis=0).reshape(-1).tolist(),
                "std": np.std(values, axis=0).reshape(-1).tolist(),
                "count": [len(values)],
            }
        return stats

    def _write_contact_preview(
        self, episode_index: int, arrays: dict[str, np.ndarray]
    ) -> Path:
        assert self._session is not None
        preview_dir = self._session / "previews"
        preview_dir.mkdir(parents=True, exist_ok=True)
        robot_ids = arrays["observation.contact.robot_body_id"]
        other_ids = arrays["observation.contact.other_body_id"]
        interaction = (robot_ids >= 0) & (other_ids != 0)
        normal = np.where(
            interaction, arrays["observation.contact.normal_force"], 0.0
        ).sum(axis=1)
        impulses = arrays["observation.contact.normal_impulse"]
        impulse = np.where(interaction, impulses, 0.0).sum(axis=1)
        counts = interaction.sum(axis=1)
        pairs = Counter()
        for row in range(len(normal)):
            for slot in range(MAX_CONTACTS):
                if interaction[row, slot]:
                    pair = int(robot_ids[row, slot]), int(other_ids[row, slot])
                    pairs[pair] += float(impulses[row, slot])
        summary = {
            "episode_index": episode_index,
            "frames": len(normal),
            "duration_sec": len(normal) / self.fps,
            "contact_frames": int(np.count_nonzero(counts)),
            "max_normal_force_n": float(np.max(normal)),
            "total_normal_impulse_ns": float(np.sum(impulse)),
            "top_contact_pairs": [
                {
                    "robot_body": self._body_name(pair[0]),
                    "other_body": self._body_name(pair[1]),
                    "normal_impulse_ns": value,
                }
                for pair, value in pairs.most_common(5)
            ],
        }
        stem = f"episode_{episode_index:06d}_contact"
        (preview_dir / f"{stem}.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False) + "\n"
        )
        svg_path = preview_dir / f"{stem}.svg"
        svg_path.write_text(self._contact_svg(normal, counts, summary))
        html_path = preview_dir / f"episode_{episode_index:06d}.html"
        html_path.write_text(self._preview_html(episode_index, stem, summary))
        return html_path

    def _body_name(self, body_id: int) -> str:
        if 0 <= body_id < len(self.body_names):
            return self.body_names[body_id]
        return f"body_{body_id}"

    def _append_tactile(
        self,
        output: dict[str, np.ndarray | float | int],
        tactile: TactileFrame,
        suit: JuQiaoTactileFrame,
    ) -> None:
        assert self.tactile_layout is not None
        expected = len(self.tactile_layout.taxels)
        if tactile.normal_force.shape != (
            expected,
        ) or not self.tactile_layout.matches_taxels(tactile.layout):
            raise ValueError("physical tactile frame does not match layout")
        if suit.values.shape != (len(TACTILE_DEVICE_NAMES), TACTILE_DEVICE_DIM):
            raise ValueError("hardware tactile frame must have shape (3, 256)")
        for index, name in enumerate(TACTILE_DEVICE_NAMES):
            output[f"observation.tactile_{name}"] = suit.values[index].copy()
        for field in _TACTILE_FIELDS:
            value = getattr(tactile, field.attribute)
            output[f"observation.tactile.{field.key}"] = np.asarray(
                value, dtype=field.dtype
            ).copy()
        output.update(
            {
                "observation.tactile.duration_s": float(tactile.duration),
                "observation.tactile.mapped_contact_count": int(
                    tactile.mapped_contact_count
                ),
                "observation.tactile.unmapped_contact_count": int(
                    tactile.unmapped_contact_count
                ),
                "simulation.tactile_source_time": suit.source_time.copy(),
                "simulation.tactile_updated": suit.updated.copy(),
            }
        )

    def _preview_html(self, episode_index: int, stem: str, summary: dict) -> str:
        rows = "".join(
            "<tr>"
            f"<td>{html.escape(item['robot_body'])}</td>"
            f"<td>{html.escape(item['other_body'])}</td>"
            f"<td>{item['normal_impulse_ns']:.4f} N·s</td>"
            "</tr>"
            for item in summary["top_contact_pairs"]
        )
        video = ""
        if self.record_video:
            views = (
                self.camera_metadata["cameras"]
                if self.camera_metadata is not None
                else ("ego_view",)
            )
            video = "".join(
                f'<h2>{html.escape(name)}</h2><video controls preload="metadata" '
                f'src="../videos/chunk-000/observation.images.{html.escape(name)}/'
                f'episode_{episode_index:06d}.mp4"></video>'
                for name in views
            )
        return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>MuJoCo contact episode</title>
<style>
body{{font:16px sans-serif;background:#0f172a;color:#e5e7eb;margin:32px}}
main{{max-width:1050px;margin:auto}} video,img{{width:100%;margin:12px 0}}
table{{border-collapse:collapse;width:100%}}
td,th{{padding:8px;border-bottom:1px solid #334155}}
</style></head><body><main>
<h1>Episode {episode_index:06d}</h1>
<p>{summary["frames"]} frames · {summary["duration_sec"]:.2f} s ·
{summary["contact_frames"]} interaction-contact frames</p>
{video}
<img src="{stem}.svg" alt="MuJoCo contact force timeline">
<h2>Top contact pairs</h2>
<table><tr><th>G1 body</th><th>Scene body</th><th>Normal impulse</th></tr>{rows}</table>
</main></body></html>
"""

    @staticmethod
    def _contact_svg(normal: np.ndarray, counts: np.ndarray, summary: dict) -> str:
        width, height = 1000, 420
        left, top, plot_width, plot_height = 70, 70, 880, 230
        maximum = max(float(np.max(normal)), 1e-6)
        denominator = max(len(normal) - 1, 1)
        points = " ".join(
            f"{left + plot_width * i / denominator:.1f},"
            f"{top + plot_height * (1.0 - value / maximum):.1f}"
            for i, value in enumerate(normal)
        )
        contact_bars = "".join(
            f'<rect x="{left + plot_width * i / len(counts):.1f}" y="315" '
            f'width="{max(1.0, plot_width / len(counts)):.1f}" height="12" '
            f'fill="#ff9f43" opacity="0.9"/>'
            for i, count in enumerate(counts)
            if count > 0
        )
        pair_text = (
            " · ".join(
                f"{item['robot_body']} → {item['other_body']}"
                for item in summary["top_contact_pairs"][:3]
            )
            or "no robot-scene contact"
        )
        title = html.escape(f"Episode {summary['episode_index']:06d} MuJoCo contact")
        pair_text = html.escape(pair_text)
        summary_text = (
            f"frames={summary['frames']}  "
            f"contact_frames={summary['contact_frames']}  "
            f"max_force={summary['max_normal_force_n']:.3f} N  "
            f"impulse={summary['total_normal_impulse_ns']:.4f} N·s"
        )
        return "\n".join(
            [
                (
                    f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
                    f'height="{height}">'
                ),
                '<rect width="100%" height="100%" fill="#111827"/>',
                (
                    f'<text x="{left}" y="36" fill="white" font-size="22" '
                    f'font-family="sans-serif">{title}</text>'
                ),
                (
                    f'<text x="{left}" y="58" fill="#9ca3af" font-size="13" '
                    f'font-family="sans-serif">{pair_text}</text>'
                ),
                (
                    f'<rect x="{left}" y="{top}" width="{plot_width}" '
                    f'height="{plot_height}" fill="#1f2937"/>'
                ),
                (
                    f'<line x1="{left}" y1="{top + plot_height}" '
                    f'x2="{left + plot_width}" y2="{top + plot_height}" '
                    'stroke="#6b7280"/>'
                ),
                (
                    f'<polyline points="{points}" fill="none" stroke="#34d399" '
                    'stroke-width="2"/>'
                ),
                (
                    f'<text x="12" y="{top + 15}" fill="#34d399" '
                    f'font-size="13" font-family="sans-serif">{maximum:.2f} N</text>'
                ),
                (
                    f'<text x="{left}" y="348" fill="#ff9f43" font-size="13" '
                    'font-family="sans-serif">contact-active control frames</text>'
                ),
                contact_bars,
                (
                    f'<text x="{left}" y="385" fill="#d1d5db" font-size="14" '
                    f'font-family="sans-serif">{summary_text}</text>'
                ),
                "</svg>",
                "",
            ]
        )

    @staticmethod
    def _pico_input(controls: PicoControls) -> np.ndarray:
        return np.array(
            [
                controls.left_trigger,
                controls.right_trigger,
                controls.left_grip,
                controls.right_grip,
                controls.a,
                controls.b,
                controls.x,
                controls.y,
                controls.menu,
            ],
            dtype=np.float64,
        )
