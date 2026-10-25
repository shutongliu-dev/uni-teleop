"""Camera alignment and transport for reconstructed-scene renderers."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]


def _pose(value: object, *, name: str) -> Array:
    pose = np.asarray(value, dtype=np.float64)
    if pose.shape != (4, 4):
        raise ValueError(f"{name} must have shape (4, 4)")
    if not np.isfinite(pose).all():
        raise ValueError(f"{name} must contain finite values")
    if not np.allclose(pose[3], (0.0, 0.0, 0.0, 1.0)):
        raise ValueError(f"{name} must be a homogeneous transform")
    return pose.copy()


@dataclass(frozen=True, slots=True)
class CameraIntrinsics:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float

    def __post_init__(self) -> None:
        if self.width < 1 or self.height < 1:
            raise ValueError("camera dimensions must be positive")
        values = (self.fx, self.fy, self.cx, self.cy)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("camera intrinsics must be finite")
        if self.fx <= 0.0 or self.fy <= 0.0:
            raise ValueError("camera focal lengths must be positive")

    @classmethod
    def from_vertical_fov(
        cls,
        width: int,
        height: int,
        vertical_fov_degrees: float,
    ) -> CameraIntrinsics:
        if not 0.0 < vertical_fov_degrees < 180.0:
            raise ValueError("vertical field of view must be between 0 and 180")
        focal = 0.5 * height / math.tan(math.radians(vertical_fov_degrees) / 2.0)
        return cls(width, height, focal, focal, width / 2.0, height / 2.0)

    def as_dict(self) -> dict[str, int | float]:
        return {
            "width": self.width,
            "height": self.height,
            "fx": self.fx,
            "fy": self.fy,
            "cx": self.cx,
            "cy": self.cy,
        }


class RelativeCameraMapper:
    """Align MuJoCo motion with a level floor and an anchored scan camera."""

    def __init__(
        self,
        scan_anchor: object,
        mujoco_anchor: object,
        *,
        scan_units_per_meter: float,
        scan_up: object = (0.0, 0.0, 1.0),
    ) -> None:
        if not math.isfinite(scan_units_per_meter) or scan_units_per_meter <= 0.0:
            raise ValueError("scan_units_per_meter must be positive and finite")
        self._scan_anchor = _pose(scan_anchor, name="scan_anchor")
        self._mujoco_anchor = _pose(mujoco_anchor, name="mujoco_anchor")
        self._scale = float(scan_units_per_meter)
        scan_up_vector = self._unit_vector(scan_up, name="scan_up")
        mujoco_up = np.array((0.0, 0.0, 1.0), dtype=np.float64)
        scan_basis = self._level_basis(self._scan_anchor, scan_up_vector)
        mujoco_basis = self._level_basis(self._mujoco_anchor, mujoco_up)
        self._scan_from_mujoco_rotation = scan_basis @ mujoco_basis.T
        aligned_anchor = self._scan_from_mujoco_rotation @ self._mujoco_anchor[:3, :3]
        self._camera_correction = aligned_anchor.T @ self._scan_anchor[:3, :3]

    def map(self, mujoco_camera_to_world: object) -> Array:
        mujoco_pose = _pose(
            mujoco_camera_to_world,
            name="mujoco_camera_to_world",
        )
        mapped = np.eye(4, dtype=np.float64)
        mapped[:3, :3] = (
            self._scan_from_mujoco_rotation
            @ mujoco_pose[:3, :3]
            @ self._camera_correction
        )
        displacement = mujoco_pose[:3, 3] - self._mujoco_anchor[:3, 3]
        mapped[:3, 3] = self._scan_anchor[:3, 3] + self._scale * (
            self._scan_from_mujoco_rotation @ displacement
        )
        return mapped

    @property
    def scan_units_per_meter(self) -> float:
        return self._scale

    def scan_points_to_mujoco(self, points: object) -> Array:
        scan_points = np.asarray(points, dtype=np.float64)
        if scan_points.ndim < 1 or scan_points.shape[-1] != 3:
            raise ValueError("scan points must end with dimension 3")
        if not np.isfinite(scan_points).all():
            raise ValueError("scan points must contain finite values")
        displacement = scan_points - self._scan_anchor[:3, 3]
        return (
            self._mujoco_anchor[:3, 3]
            + (displacement @ self._scan_from_mujoco_rotation) / self._scale
        )

    def scan_rotation_to_mujoco(self, rotation: object) -> Array:
        scan_rotation = np.asarray(rotation, dtype=np.float64)
        if scan_rotation.shape != (3, 3):
            raise ValueError("scan rotation must have shape (3, 3)")
        if not np.isfinite(scan_rotation).all():
            raise ValueError("scan rotation must contain finite values")
        return self._scan_from_mujoco_rotation.T @ scan_rotation

    @staticmethod
    def _unit_vector(value: object, *, name: str) -> Array:
        vector = np.asarray(value, dtype=np.float64)
        if vector.shape != (3,) or not np.isfinite(vector).all():
            raise ValueError(f"{name} must be a finite 3-vector")
        norm = np.linalg.norm(vector)
        if norm < 1e-8:
            raise ValueError(f"{name} must be non-zero")
        return vector / norm

    @classmethod
    def _level_basis(cls, camera_pose: Array, up: Array) -> Array:
        forward = -camera_pose[:3, 2]
        forward -= up * np.dot(forward, up)
        forward = cls._unit_vector(forward, name="horizontal camera forward")
        lateral = np.cross(up, forward)
        lateral = cls._unit_vector(lateral, name="camera lateral")
        return np.column_stack((forward, lateral, up))


def mujoco_camera_pose(model, data, camera: str) -> Array:
    """Return MuJoCo's OpenGL camera-to-world transform."""
    import mujoco

    camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera)
    if camera_id < 0:
        raise ValueError(f"MuJoCo camera not found: {camera}")
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = np.asarray(data.cam_xmat[camera_id]).reshape(3, 3)
    pose[:3, 3] = data.cam_xpos[camera_id]
    return pose


class ScanRendererClient:
    """Small synchronous client for a renderer running in an isolated ML env."""

    def __init__(self, endpoint: str, *, timeout_ms: int = 10_000) -> None:
        import zmq

        self._zmq = zmq
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.REQ)
        self._socket.setsockopt(zmq.RCVTIMEO, timeout_ms)
        self._socket.setsockopt(zmq.SNDTIMEO, timeout_ms)
        self._socket.setsockopt(zmq.LINGER, 0)
        self._socket.connect(endpoint)

    def info(self) -> dict:
        header, payloads = self._call({"operation": "info"})
        if payloads != [b""]:
            raise RuntimeError("renderer returned an unexpected info payload")
        return header

    def render(
        self,
        camera_to_world: object,
        intrinsics: CameraIntrinsics,
    ) -> NDArray[np.uint8]:
        pose = _pose(camera_to_world, name="camera_to_world")
        request = {
            "operation": "render",
            "camera_to_world": pose[:3].tolist(),
            **intrinsics.as_dict(),
        }
        _, payloads = self._call(request)
        payload = payloads[0]
        expected_shape = (intrinsics.height, intrinsics.width, 3)
        image = np.frombuffer(payload, dtype=np.uint8)
        if image.size != math.prod(expected_shape):
            raise RuntimeError("renderer returned an invalid RGB payload")
        return image.reshape(expected_shape).copy()

    def render_rgbd(
        self,
        camera_to_world: object,
        intrinsics: CameraIntrinsics,
    ) -> tuple[NDArray[np.uint8], NDArray[np.float32]]:
        pose = _pose(camera_to_world, name="camera_to_world")
        request = {
            "operation": "render",
            "camera_to_world": pose[:3].tolist(),
            **intrinsics.as_dict(),
        }
        _, payloads = self._call(request)
        if len(payloads) != 2:
            raise RuntimeError("renderer did not return RGB and depth")
        rgb_shape = (intrinsics.height, intrinsics.width, 3)
        depth_shape = (intrinsics.height, intrinsics.width)
        rgb = np.frombuffer(payloads[0], dtype=np.uint8)
        depth = np.frombuffer(payloads[1], dtype=np.float32)
        if rgb.size != math.prod(rgb_shape) or depth.size != math.prod(depth_shape):
            raise RuntimeError("renderer returned an invalid RGB-D payload")
        return rgb.reshape(rgb_shape).copy(), depth.reshape(depth_shape).copy()

    def close(self) -> None:
        self._socket.close(linger=0)
        self._context.term()

    def _call(self, request: dict) -> tuple[dict, list[bytes]]:
        self._socket.send_json(request)
        parts = self._socket.recv_multipart()
        if len(parts) < 2:
            raise RuntimeError("renderer returned an invalid multipart response")
        header = json.loads(parts[0])
        if header.get("status") != "ok":
            raise RuntimeError(header.get("error", "scan renderer failed"))
        return header, parts[1:]
