"""Synchronized robot-mounted RGB, axial depth and depth-derived surface normals."""

from dataclasses import dataclass

import mujoco
import numpy as np

from .scan import CameraIntrinsics, RelativeCameraMapper, mujoco_camera_pose

ROBOT_CAMERAS = {
    "ego_view_left": "head_camera_left",
    "ego_view_right": "head_camera_right",
    "wrist_view_left": "left_wrist_camera",
    "wrist_view_right": "right_wrist_camera",
}


@dataclass(frozen=True)
class CameraFrame:
    rgb: np.ndarray
    depth_m: np.ndarray
    normals_camera: np.ndarray
    valid_depth: np.ndarray
    valid_normals: np.ndarray
    camera_to_world: np.ndarray
    appearance_camera_to_world: np.ndarray | None
    timestamp: float


def depth_to_normals(depth, valid, intrinsics, *, edge_threshold_m=0.05):
    """Camera-facing normals in optical +X right, +Y down, +Z forward axes.

    Pixels bordering invalid depth or a >5cm depth step have no normal.
    Zero vectors are invalid, never interpreted as a surface direction.
    """
    h, w = depth.shape
    y, x = np.mgrid[:h, :w]
    points = np.stack(
        (
            (x + 0.5 - intrinsics.cx) * depth / intrinsics.fx,
            (y + 0.5 - intrinsics.cy) * depth / intrinsics.fy,
            depth,
        ),
        axis=-1,
    )
    dx = points[1:-1, 2:] - points[1:-1, :-2]
    dy = points[2:, 1:-1] - points[:-2, 1:-1]
    n = np.cross(dy, dx)
    norm = np.linalg.norm(n, axis=-1)
    mask = valid[1:-1, 1:-1].copy()
    center = depth[1:-1, 1:-1]
    for d, v in (
        (depth[1:-1, 2:], valid[1:-1, 2:]),
        (depth[1:-1, :-2], valid[1:-1, :-2]),
        (depth[2:, 1:-1], valid[2:, 1:-1]),
        (depth[:-2, 1:-1], valid[:-2, 1:-1]),
    ):
        mask &= v & (np.abs(d - center) <= edge_threshold_m)
    mask &= norm > 1e-12
    n = np.divide(n, norm[..., None], out=np.zeros_like(n), where=mask[..., None])
    n[np.sum(n * points[1:-1, 1:-1], axis=-1) > 0] *= -1
    output = np.zeros((h, w, 3), dtype=np.float32)
    output[1:-1, 1:-1] = n
    valid_normal = np.zeros((h, w), dtype=bool)
    valid_normal[1:-1, 1:-1] = mask
    return output, valid_normal


def scan_mapper(model, info, scale):
    reference = mujoco.MjData(model)
    mujoco.mj_forward(model, reference)
    return RelativeCameraMapper(
        info["anchor_camera_to_world"],
        mujoco_camera_pose(model, reference, "head_camera_scan"),
        scan_units_per_meter=scale,
        scan_up=info["scan_up"],
    )


class CameraRig:
    def __init__(
        self,
        model,
        data,
        *,
        width=640,
        height=480,
        camera_names=None,
        scan_client=None,
        mapper=None,
    ):
        if min(width, height) < 4 or width % 2 or height % 2:
            raise ValueError("camera dimensions must be even and at least four pixels")
        if (scan_client is None) != (mapper is None):
            raise ValueError("scan client and mapper must be configured together")
        self.model, self.data = model, data
        self.camera_names = dict(
            ROBOT_CAMERAS if camera_names is None else camera_names
        )
        self._intrinsics = {}
        camera_metadata = {}
        for name, camera in self.camera_names.items():
            camera_id = model.camera(camera).id
            k = CameraIntrinsics.from_vertical_fov(
                width, height, model.cam_fovy[camera_id]
            )
            self._intrinsics[name] = k
            camera_metadata[name] = {
                "camera_name": camera,
                "width": width,
                "height": height,
                "intrinsics": k.as_dict(),
                "mount_body": model.body(int(model.cam_bodyid[camera_id])).name,
                "position_body_m": model.cam_pos[camera_id].tolist(),
                "quaternion_body_wxyz": model.cam_quat[camera_id].tolist(),
                "calibration_status": "virtual_mount_not_real_camera_calibrated",
            }
        self.metadata = {
            "schema_version": 1,
            "cameras": camera_metadata,
            "source": "mujoco" if scan_client is None else "mujoco_3dgs_composite",
            "pose_convention": "camera_to_world_opengl_x_right_y_up_minus_z_forward",
            "depth_units": "meters" if scan_client is None else "estimated_meters",
            "depth_quantity": "optical_axis_z",
            "normal_frame": "optical_x_right_y_down_z_forward",
            "normal_source": "depth_finite_difference_camera_facing",
            "normal_edge_threshold_m": 0.05,
            "invalid_depth": 0,
            "invalid_normal": [0, 0, 0],
            "clock": "mujoco_sim_time",
            "view_source": "robot_mounted",
        }
        if mapper is not None:
            self.metadata["scan_units_per_meter"] = mapper.scan_units_per_meter
        model.vis.global_.offwidth = max(model.vis.global_.offwidth, width)
        model.vis.global_.offheight = max(model.vis.global_.offheight, height)
        self._renderer = mujoco.Renderer(model, height=height, width=width)
        self._client, self._mapper = scan_client, mapper

    def capture(self, names=None):
        # mj_step's derived camera/body transforms precede its final integration.
        # Refresh poses before associating RGB with the post-step timestamp/state.
        mujoco.mj_forward(self.model, self.data)
        return {
            name: self._capture(name)
            for name in (self.camera_names if names is None else names)
        }

    def _capture(self, name):
        renderer, model, data = self._renderer, self.model, self.data
        camera = self.camera_names[name]
        pose = mujoco_camera_pose(model, data, camera)
        renderer.update_scene(data, camera=camera)
        rgb = renderer.render().copy()
        renderer.enable_depth_rendering()
        try:
            depth = renderer.render().copy()
        finally:
            renderer.disable_depth_rendering()
        near = model.vis.map.znear * model.stat.extent
        far = model.vis.map.zfar * model.stat.extent
        valid = np.isfinite(depth) & (depth > near) & (depth < far * 0.999)
        appearance = None
        if self._client is not None:
            appearance = self._mapper.map(pose)
            # Collisions use a world rotation; local anchor corrections must not
            # rotate wrist cameras differently from the physical scene.
            world_rotation = self._mapper.scan_rotation_to_mujoco(np.eye(3)).T
            appearance[:3, :3] = world_rotation @ pose[:3, :3]
            background, scan_depth = self._client.render_rgbd(
                appearance, self._intrinsics[name]
            )
            scan_depth = scan_depth / self._mapper.scan_units_per_meter
            scan_valid = np.isfinite(scan_depth) & (scan_depth > near)
            renderer.enable_segmentation_rendering()
            try:
                segmentation = renderer.render()
            finally:
                renderer.disable_segmentation_rendering()
            ids = segmentation[..., 0]
            foreground = (segmentation[..., 1] == mujoco.mjtObj.mjOBJ_GEOM) & (ids >= 0)
            foreground[foreground] &= model.geom_bodyid[ids[foreground]] != 0
            foreground &= valid & (~scan_valid | (depth <= scan_depth))
            rgb = np.where(foreground[..., None], rgb, background).astype(np.uint8)
            depth = np.where(foreground, depth, scan_depth)
            valid = np.where(foreground, valid, scan_valid)
        depth = np.where(valid, depth, 0).astype(np.float32)
        normals, normal_valid = depth_to_normals(depth, valid, self._intrinsics[name])
        return CameraFrame(
            rgb, depth, normals, valid, normal_valid, pose, appearance, float(data.time)
        )

    def close(self):
        self._renderer.close()


class CameraSampler:
    """Capture on simulation time; held observations retain their original poses."""

    def __init__(self, rig, fps=10):
        if not np.isfinite(fps) or not 0 < fps <= 50:
            raise ValueError("camera fps must be in (0, 50]")
        self.rig, self.fps = rig, fps
        self.reset()

    @property
    def metadata(self):
        return {
            **self.rig.metadata,
            "requested_capture_fps": self.fps,
            "sampling": "simulation_clock_latest_capture_hold",
        }

    def reset(self):
        self._frames = None
        self._origin = None
        self._tick = 0

    def capture(self):
        time = float(self.rig.data.time)
        if self._origin is None:
            self._origin = time
        if self._frames is None or time + 1e-10 >= self._origin + self._tick / self.fps:
            self._frames = self.rig.capture()
            # Keep the schedule anchored when capture and control rates differ.
            # Missed instants are skipped, never synthesized retroactively.
            self._tick = int(np.floor((time - self._origin + 1e-10) * self.fps)) + 1
        return self._frames
