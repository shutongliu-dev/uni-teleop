import mmap
import os
import struct
import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from ..cameras import scan_mapper
from ..scan import (
    CameraIntrinsics,
    ScanRendererClient,
    mujoco_camera_pose,
)
from .desktop_video import DesktopFrameViewer

FRAME_HEADER = struct.Struct("<4sIIII")


@dataclass(frozen=True)
class RenderedCameraFrame:
    """One RGB observation aligned with physics and appearance camera poses."""

    rgb: np.ndarray
    camera_to_world: np.ndarray
    appearance_camera_to_world: np.ndarray

    def copy(self) -> "RenderedCameraFrame":
        return RenderedCameraFrame(
            rgb=self.rgb.copy(),
            camera_to_world=self.camera_to_world.copy(),
            appearance_camera_to_world=self.appearance_camera_to_world.copy(),
        )


def _stereo_frame(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left = np.asarray(left, dtype=np.uint8)
    right = np.asarray(right, dtype=np.uint8)
    if left.ndim != 3 or left.shape[2] != 3:
        raise ValueError("left video frame must have shape (height, width, 3)")
    if right.shape != left.shape:
        raise ValueError("left and right video frames must have matching dimensions")
    return np.ascontiguousarray(np.concatenate((left, right), axis=1))


def _validate_scan_alignment(renderer_info: dict, scene_metadata: dict) -> None:
    if renderer_info.get("anchor_image") != scene_metadata.get("anchor_image"):
        raise ValueError("scan renderer and collision scene use different anchors")
    renderer_anchor = np.asarray(
        renderer_info.get("anchor_camera_to_world"),
        dtype=np.float64,
    )
    scene_anchor = np.asarray(
        scene_metadata.get("anchor_camera_to_world"),
        dtype=np.float64,
    )
    if (
        renderer_anchor.shape != (4, 4)
        or scene_anchor.shape != (4, 4)
        or not np.allclose(renderer_anchor, scene_anchor, atol=1e-5)
    ):
        raise ValueError("scan renderer and collision scene anchor poses differ")
    renderer_up = np.asarray(renderer_info.get("scan_up"), dtype=np.float64)
    scene_up = np.asarray(scene_metadata.get("scan_up"), dtype=np.float64)
    if renderer_up.shape != (3,) or scene_up.shape != (3,):
        raise ValueError("scan renderer and collision scene need floor normals")
    renderer_norm = np.linalg.norm(renderer_up)
    scene_norm = np.linalg.norm(scene_up)
    if min(renderer_norm, scene_norm) < 1e-8:
        raise ValueError("scan floor normals must be non-zero")
    agreement = float(np.dot(renderer_up / renderer_norm, scene_up / scene_norm))
    if agreement < np.cos(np.deg2rad(1.0)):
        raise ValueError("scan renderer and collision scene floor normals differ")


class RelativeHeadsetView:
    """Convert absolute XR poses into motion relative to the first sample."""

    def __init__(self) -> None:
        self._reference: np.ndarray | None = None
        self._relative = np.eye(4, dtype=np.float64)

    @property
    def tracking_active(self) -> bool:
        return self._reference is not None

    def update(self, value: object | None) -> np.ndarray:
        pose = self._pose(value)
        if pose is None:
            return self._relative.copy()
        if self._reference is None:
            self._reference = pose
        self._relative = np.linalg.inv(self._reference) @ pose
        return self._relative.copy()

    def recenter(self) -> None:
        self._reference = None
        self._relative = np.eye(4, dtype=np.float64)

    @staticmethod
    def _pose(value: object | None) -> np.ndarray | None:
        if value is None:
            return None
        raw = np.asarray(value, dtype=np.float64)
        if raw.shape != (7,) or not np.isfinite(raw).all():
            return None
        norm = np.linalg.norm(raw[3:])
        if norm < 1e-8:
            return None
        pose = np.eye(4, dtype=np.float64)
        # The PICO client emits a right-handed X-right, Y-up, Z-in frame;
        # forward is -Z, matching an OpenGL camera's local axes.
        pose[:3, :3] = Rotation.from_quat(raw[3:] / norm).as_matrix()
        pose[:3, 3] = raw[:3]
        return pose


# Minimal 5x7 font keeps the live overlay dependency-free.
_GLYPHS = {
    " ": (0, 0, 0, 0, 0, 0, 0),
    ".": (0, 0, 0, 0, 0, 0, 0b00100),
    ":": (0, 0b00100, 0b00100, 0, 0b00100, 0b00100, 0),
    "0": (0b01110, 0b10001, 0b10011, 0b10101, 0b11001, 0b10001, 0b01110),
    "1": (0b00100, 0b01100, 0b00100, 0b00100, 0b00100, 0b00100, 0b01110),
    "2": (0b01110, 0b10001, 0b00001, 0b00010, 0b00100, 0b01000, 0b11111),
    "3": (0b11110, 0b00001, 0b00001, 0b01110, 0b00001, 0b00001, 0b11110),
    "4": (0b00010, 0b00110, 0b01010, 0b10010, 0b11111, 0b00010, 0b00010),
    "5": (0b11111, 0b10000, 0b10000, 0b11110, 0b00001, 0b00001, 0b11110),
    "6": (0b01110, 0b10000, 0b10000, 0b11110, 0b10001, 0b10001, 0b01110),
    "7": (0b11111, 0b00001, 0b00010, 0b00100, 0b01000, 0b01000, 0b01000),
    "8": (0b01110, 0b10001, 0b10001, 0b01110, 0b10001, 0b10001, 0b01110),
    "9": (0b01110, 0b10001, 0b10001, 0b01111, 0b00001, 0b00001, 0b01110),
    "A": (0b01110, 0b10001, 0b10001, 0b11111, 0b10001, 0b10001, 0b10001),
    "C": (0b01111, 0b10000, 0b10000, 0b10000, 0b10000, 0b10000, 0b01111),
    "D": (0b11110, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b11110),
    "E": (0b11111, 0b10000, 0b10000, 0b11110, 0b10000, 0b10000, 0b11111),
    "F": (0b11111, 0b10000, 0b10000, 0b11110, 0b10000, 0b10000, 0b10000),
    "H": (0b10001, 0b10001, 0b10001, 0b11111, 0b10001, 0b10001, 0b10001),
    "O": (0b01110, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01110),
    "R": (0b11110, 0b10001, 0b10001, 0b11110, 0b10100, 0b10010, 0b10001),
}


def _label_overlay(
    frame: np.ndarray,
    label: str,
    *,
    top: int,
    background: tuple[int, int, int],
) -> np.ndarray:
    image = np.asarray(frame, dtype=np.uint8).copy()
    scale, padding = 2, 6
    width = len(label) * 6 * scale - scale + 2 * padding
    height = 7 * scale + 2 * padding
    image[top : top + height, 12 : 12 + width] = background
    for index, character in enumerate(label):
        x = 12 + padding + index * 6 * scale
        for row, bits in enumerate(_GLYPHS[character]):
            for column in range(5):
                if bits & (1 << (4 - column)):
                    y0 = top + padding + row * scale
                    x0 = x + column * scale
                    image[y0 : y0 + scale, x0 : x0 + scale] = 255
    return image


def _recording_overlay(frame: np.ndarray, elapsed_seconds: float) -> np.ndarray:
    elapsed = min(max(float(elapsed_seconds), 0.0), 5999.9)
    minutes = int(elapsed // 60)
    seconds = elapsed - minutes * 60
    return _label_overlay(
        frame,
        f"REC {minutes:02d}:{seconds:04.1f}",
        top=12,
        background=(190, 0, 0),
    )


def _tracking_warning_overlay(frame: np.ndarray) -> np.ndarray:
    return _label_overlay(
        frame,
        "HEAD OFF",
        top=40,
        background=(190, 70, 0),
    )


class PicoFrameBridge:
    """Expose side-by-side RGB frames to the dependency-free PICO bridge."""

    def __init__(
        self,
        *,
        listen: str = "0.0.0.0:13579",
        width: int = 640,
        height: int = 480,
        fps: int = 30,
    ) -> None:
        self._frame_path = Path(f"/dev/shm/sonic_mujoco_pico_{os.getpid()}")
        self._frame_size = 2 * width * height * 3
        self._file = self._frame_path.open("w+b")
        self._file.truncate(FRAME_HEADER.size + self._frame_size)
        self._memory = mmap.mmap(self._file.fileno(), 0)
        self._sequence = 0
        self._period = 1.0 / fps
        self._last_render = 0.0
        empty = np.zeros((height, width, 3), dtype=np.uint8)
        self._write_stereo(empty, empty)

        bridge = Path(__file__).parents[2] / "scripts/pico_video_bridge.py"
        self._process = subprocess.Popen(
            [
                "/usr/bin/python3",
                str(bridge),
                "--frames",
                str(self._frame_path),
                "--listen",
                listen,
            ]
        )

    def ready(self) -> bool:
        now = time.monotonic()
        if now - self._last_render < self._period:
            return False
        self._last_render = now
        return True

    def publish(self, frame: np.ndarray) -> None:
        self._write_stereo(frame, frame)

    def publish_stereo(self, left: np.ndarray, right: np.ndarray) -> None:
        self._write_stereo(left, right)

    def _write_stereo(self, left: np.ndarray, right: np.ndarray) -> None:
        frame = _stereo_frame(left, right)
        if frame.nbytes != self._frame_size:
            raise ValueError("video frame dimensions changed")
        self._sequence += 1
        width, height = frame.shape[1], frame.shape[0]
        self._memory[: FRAME_HEADER.size] = FRAME_HEADER.pack(
            b"SMVF", self._sequence, width, height, frame.nbytes
        )
        self._memory[FRAME_HEADER.size :] = frame.tobytes()
        self._sequence += 1
        self._memory[: FRAME_HEADER.size] = FRAME_HEADER.pack(
            b"SMVF", self._sequence, width, height, frame.nbytes
        )

    def close(self) -> None:
        self._process.terminate()
        try:
            self._process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            self._process.kill()
        self._memory.close()
        self._file.close()
        self._frame_path.unlink(missing_ok=True)


class PicoVideo:
    """Render a MuJoCo camera and expose it to the PICO video bridge."""

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        *,
        camera: str = "head_camera_left",
        right_camera: str = "head_camera_right",
        listen: str = "0.0.0.0:13579",
        width: int = 1280,
        height: int = 720,
        fps: int = 30,
        desktop: bool = False,
    ) -> None:
        self._data = data
        self._model = model
        self._camera = camera
        self._right_camera = right_camera
        self.fps = fps
        model.vis.global_.offwidth = max(model.vis.global_.offwidth, width)
        model.vis.global_.offheight = max(model.vis.global_.offheight, height)
        self._renderer = mujoco.Renderer(model, height=height, width=width)
        self._latest: RenderedCameraFrame | None = None
        self.metadata = {
            "camera": camera,
            "renderer": "mujoco",
            "width": width,
            "height": height,
            "fps": fps,
            "stereo": True,
            "right_camera": right_camera,
            "view_source": "robot_mounted",
            "recorded_eye": "left",
        }
        self._bridge = PicoFrameBridge(
            listen=listen,
            width=width,
            height=height,
            fps=fps,
        )
        self._desktop = (
            DesktopFrameViewer(
                width=width, height=height, title="PICO Robot Camera (left eye)"
            )
            if desktop
            else None
        )

    @property
    def latest_frame(self) -> np.ndarray | None:
        return None if self._latest is None else self._latest.rgb.copy()

    @property
    def latest_rendered(self) -> RenderedCameraFrame | None:
        return None if self._latest is None else self._latest.copy()

    def render(
        self, *, recording: bool = False, elapsed_seconds: float = 0.0
    ) -> RenderedCameraFrame | None:
        if not self._bridge.ready():
            return None
        mujoco.mj_forward(self._model, self._data)
        camera_to_world = mujoco_camera_pose(self._model, self._data, self._camera)
        self._renderer.update_scene(self._data, camera=self._camera)
        self._latest = RenderedCameraFrame(
            rgb=self._renderer.render().copy(),
            camera_to_world=camera_to_world.copy(),
            appearance_camera_to_world=camera_to_world.copy(),
        )
        frame = self._latest.rgb
        if recording:
            frame = _recording_overlay(frame, elapsed_seconds)
        self._renderer.update_scene(self._data, camera=self._right_camera)
        right = self._renderer.render().copy()
        if recording:
            right = _recording_overlay(right, elapsed_seconds)
        self._bridge.publish_stereo(frame, right)
        if self._desktop is not None and not self._desktop.publish(frame):
            self._desktop = None
        return self._latest

    def close(self) -> None:
        self._renderer.close()
        self._bridge.close()
        if self._desktop is not None:
            self._desktop.close()


class ScanPicoVideo:
    """Render a reconstructed scan with MuJoCo robot geometry in the foreground."""

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        *,
        endpoint: str,
        scan_units_per_meter: float,
        camera: str = "vr_camera_base",
        left_camera: str = "vr_camera_left",
        right_camera: str = "vr_camera_right",
        headset_pose_provider: Callable[[], np.ndarray | None] | None = None,
        alignment_metadata: dict | None = None,
        listen: str = "0.0.0.0:13579",
        width: int = 1280,
        height: int = 720,
        fps: int = 30,
        desktop: bool = True,
    ) -> None:
        camera_ids = {
            name: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, name)
            for name in (camera, left_camera, right_camera)
        }
        missing = [name for name, camera_id in camera_ids.items() if camera_id < 0]
        if missing:
            raise ValueError(f"MuJoCo camera not found: {missing[0]}")
        view_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "vr_view")
        if view_body_id < 0 or model.body_mocapid[view_body_id] < 0:
            raise ValueError("MuJoCo mocap body not found: vr_view")
        mujoco.mj_forward(model, data)
        self._model = model
        self._data = data
        self._camera = camera
        self._eye_cameras = (left_camera, right_camera)
        self._view_mocap_id = int(model.body_mocapid[view_body_id])
        self._headset_pose_provider = headset_pose_provider
        self._headset_view = RelativeHeadsetView()
        self.fps = fps
        model.vis.global_.offwidth = max(model.vis.global_.offwidth, width)
        model.vis.global_.offheight = max(model.vis.global_.offheight, height)
        self._renderer = mujoco.Renderer(model, height=height, width=width)
        self._latest: RenderedCameraFrame | None = None
        self._client = ScanRendererClient(endpoint)
        info = self._client.info()
        if alignment_metadata is not None:
            _validate_scan_alignment(info, alignment_metadata)
        self._mapper = scan_mapper(model, info, scan_units_per_meter)
        self._intrinsics = CameraIntrinsics.from_vertical_fov(
            width,
            height,
            float(model.cam_fovy[camera_ids[left_camera]]),
        )
        left_offset = model.cam_pos[camera_ids[left_camera]]
        right_offset = model.cam_pos[camera_ids[right_camera]]
        eye_baseline = float(np.linalg.norm(right_offset - left_offset))
        self.metadata = {
            "camera": left_camera,
            "base_camera": camera,
            "left_camera": left_camera,
            "right_camera": right_camera,
            "renderer": "mujoco_3dgs_depth_composite",
            "intrinsics": self._intrinsics.as_dict(),
            "fps": fps,
            "stereo": True,
            "eye_baseline_meters": eye_baseline,
            "recorded_eye": "left",
            "headset_pose_source": "xrobotoolkit_relative_6dof",
            "scan_units_per_meter": scan_units_per_meter,
            "anchor_image": info["anchor_image"],
            "checkpoint": info["checkpoint"],
            "checkpoint_step": info["step"],
        }
        if alignment_metadata is not None:
            self.metadata.update(
                {
                    "collision_manifest_version": alignment_metadata.get("version"),
                    "scale_calibration": alignment_metadata.get("scale_calibration"),
                    "geometry_provenance": alignment_metadata.get(
                        "geometry_provenance"
                    ),
                }
            )
        self._bridge = PicoFrameBridge(
            listen=listen,
            width=width,
            height=height,
            fps=fps,
        )
        self._desktop = (
            DesktopFrameViewer(width=width, height=height) if desktop else None
        )

    @property
    def latest_frame(self) -> np.ndarray | None:
        return None if self._latest is None else self._latest.rgb.copy()

    @property
    def latest_rendered(self) -> RenderedCameraFrame | None:
        return None if self._latest is None else self._latest.copy()

    @property
    def headset_tracking_active(self) -> bool:
        return self._headset_view.tracking_active

    def recenter_headset(self) -> None:
        self._headset_view.recenter()

    def render(
        self, *, recording: bool = False, elapsed_seconds: float = 0.0
    ) -> RenderedCameraFrame | None:
        if not self._bridge.ready():
            return None
        mujoco.mj_forward(self._model, self._data)
        base_camera_to_world = mujoco_camera_pose(
            self._model,
            self._data,
            self._camera,
        )
        headset_pose = (
            self._headset_pose_provider()
            if self._headset_pose_provider is not None
            else None
        )
        relative_headset_pose = self._headset_view.update(headset_pose)
        self._set_view_pose(base_camera_to_world @ relative_headset_pose)

        rendered_eyes = [
            self._render_eye(camera_name) for camera_name in self._eye_cameras
        ]
        left, right = rendered_eyes
        self._latest = RenderedCameraFrame(
            rgb=left[0].copy(),
            camera_to_world=left[1].copy(),
            appearance_camera_to_world=left[2].copy(),
        )
        left_frame, right_frame = left[0], right[0]
        if not self.headset_tracking_active:
            left_frame = _tracking_warning_overlay(left_frame)
            right_frame = _tracking_warning_overlay(right_frame)
        if recording:
            left_frame = _recording_overlay(left_frame, elapsed_seconds)
            right_frame = _recording_overlay(right_frame, elapsed_seconds)
        self._bridge.publish_stereo(left_frame, right_frame)
        if self._desktop is not None and not self._desktop.publish(left_frame):
            self._desktop = None
        return self._latest

    def _set_view_pose(self, camera_to_world: np.ndarray) -> None:
        self._data.mocap_pos[self._view_mocap_id] = camera_to_world[:3, 3]
        quaternion = Rotation.from_matrix(camera_to_world[:3, :3]).as_quat()
        self._data.mocap_quat[self._view_mocap_id] = np.roll(quaternion, 1)
        mujoco.mj_forward(self._model, self._data)

    def _render_eye(
        self,
        camera: str,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        physics_camera_to_world = mujoco_camera_pose(self._model, self._data, camera)
        camera_to_world = self._mapper.map(physics_camera_to_world)
        rotation = self._mapper.scan_rotation_to_mujoco(np.eye(3)).T
        camera_to_world[:3, :3] = rotation @ physics_camera_to_world[:3, :3]
        background, scan_depth = self._client.render_rgbd(
            camera_to_world,
            self._intrinsics,
        )
        robot_rgb, robot_mask, robot_depth = self._render_mujoco_foreground(camera)
        scan_depth_meters = scan_depth / self._mapper.scan_units_per_meter
        robot_mask &= robot_depth <= scan_depth_meters + 0.02
        frame = background
        frame[robot_mask] = robot_rgb[robot_mask]
        return (
            np.ascontiguousarray(frame, dtype=np.uint8),
            physics_camera_to_world,
            camera_to_world,
        )

    def _render_mujoco_foreground(
        self,
        camera: str,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        self._renderer.update_scene(self._data, camera=camera)
        rgb = self._renderer.render().copy()
        self._renderer.enable_depth_rendering()
        try:
            self._renderer.update_scene(self._data, camera=camera)
            depth = self._renderer.render().copy()
        finally:
            self._renderer.disable_depth_rendering()
        self._renderer.enable_segmentation_rendering()
        try:
            self._renderer.update_scene(self._data, camera=camera)
            segmentation = self._renderer.render().copy()
        finally:
            self._renderer.disable_segmentation_rendering()

        object_ids = segmentation[..., 0]
        is_geom = segmentation[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM)
        valid = is_geom & (object_ids >= 0)
        foreground = np.zeros(object_ids.shape, dtype=bool)
        foreground[valid] = self._model.geom_bodyid[object_ids[valid]] != 0
        return rgb, foreground, depth

    def close(self) -> None:
        self._renderer.close()
        self._client.close()
        self._bridge.close()
        if self._desktop is not None:
            self._desktop.close()
