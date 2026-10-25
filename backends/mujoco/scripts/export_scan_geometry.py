"""Export scan-derived geometry from a trained Nerfstudio Gaussian scene."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--camera-stride", type=int, default=2)
    parser.add_argument("--resolution-scale", type=float, default=0.125)
    parser.add_argument("--pixel-stride", type=int, default=2)
    parser.add_argument("--min-accumulation", type=float, default=0.98)
    parser.add_argument("--max-depth", type=float, default=3.0)
    parser.add_argument("--voxel-size", type=float, default=0.01)
    parser.add_argument("--anchor-image", default="frame_00138.jpg")
    return parser.parse_args()


def ordered_cameras(pipeline):
    entries = []
    for dataset in (
        pipeline.datamanager.train_dataset,
        pipeline.datamanager.eval_dataset,
    ):
        if dataset is None:
            continue
        entries.extend(
            (path.name, dataset.cameras[index : index + 1])
            for index, path in enumerate(dataset.image_filenames)
        )
    return sorted(entries, key=lambda item: item[0])


def unproject(
    camera,
    depth,
    rgb,
    accumulation,
    *,
    pixel_stride,
    min_accumulation,
    max_depth,
):
    import numpy as np

    depth = depth[::pixel_stride, ::pixel_stride, 0]
    rgb = rgb[::pixel_stride, ::pixel_stride]
    accumulation = accumulation[::pixel_stride, ::pixel_stride, 0]
    height, width = depth.shape
    rows, columns = np.mgrid[0:height, 0:width]
    rows = rows * pixel_stride
    columns = columns * pixel_stride
    valid = (
        np.isfinite(depth)
        & (depth > 0.02)
        & (depth < max_depth)
        & (accumulation >= min_accumulation)
    )
    depth = depth[valid]
    x = (columns[valid] - float(camera.cx.item())) * depth / float(camera.fx.item())
    y = -(rows[valid] - float(camera.cy.item())) * depth / float(camera.fy.item())
    camera_points = np.column_stack((x, y, -depth))
    camera_to_world = camera.camera_to_worlds[0].detach().cpu().numpy()
    world_points = camera_points @ camera_to_world[:3, :3].T
    world_points += camera_to_world[:3, 3]
    return world_points, rgb[valid]


def estimate_floor(point_cloud, voxel_size: float) -> dict:
    import numpy as np
    import open3d as o3d

    points = np.asarray(point_cloud.points)
    o3d.utility.random.seed(0)
    low_cutoff = float(np.quantile(points[:, 2], 0.35))
    low_points = points[points[:, 2] <= low_cutoff]
    candidate = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(low_points))
    plane, inliers = candidate.segment_plane(
        distance_threshold=max(voxel_size * 1.5, 0.005),
        ransac_n=3,
        num_iterations=2_000,
    )
    normal = np.asarray(plane[:3], dtype=np.float64)
    normal /= np.linalg.norm(normal)
    if normal[2] < 0.0:
        normal *= -1.0
        plane = [-value for value in plane]
    floor_z = -float(plane[3]) / float(plane[2])
    return {
        "plane": [float(value) for value in plane],
        "normal": normal.tolist(),
        "z_at_origin": floor_z,
        "inlier_count": len(inliers),
        "candidate_count": len(low_points),
        "normal_alignment": float(normal[2]),
    }


def main() -> None:
    import numpy as np
    import open3d as o3d
    import torch
    from nerfstudio.utils.eval_utils import eval_setup

    args = parse_args()
    if args.camera_stride < 1 or args.pixel_stride < 1:
        raise SystemExit("camera and pixel stride must be positive")
    if not 0.0 < args.resolution_scale <= 1.0:
        raise SystemExit("resolution scale must be in (0, 1]")
    if not 0.0 <= args.min_accumulation <= 1.0:
        raise SystemExit("minimum accumulation must be in [0, 1]")
    if args.max_depth <= 0.0 or args.voxel_size <= 0.0:
        raise SystemExit("maximum depth and voxel size must be positive")

    torch.set_float32_matmul_precision("high")
    _, pipeline, checkpoint, step = eval_setup(
        args.config,
        test_mode="inference",
    )
    camera_entries = ordered_cameras(pipeline)
    anchors = [camera for name, camera in camera_entries if name == args.anchor_image]
    if len(anchors) != 1:
        raise SystemExit(
            f"anchor image not found in reconstruction: {args.anchor_image}"
        )
    anchor_pose = np.eye(4, dtype=np.float64)
    anchor_pose[:3] = anchors[0].camera_to_worlds[0].detach().cpu().numpy()
    cameras = [camera for _, camera in camera_entries[:: args.camera_stride]]
    points = []
    colors = []
    for index, source_camera in enumerate(cameras, start=1):
        camera = source_camera.to(pipeline.device)
        camera.rescale_output_resolution(args.resolution_scale)
        with torch.no_grad():
            outputs = pipeline.model.get_outputs_for_camera(camera)
        world_points, point_colors = unproject(
            camera,
            outputs["depth"].detach().cpu().numpy(),
            outputs["rgb"].detach().cpu().numpy(),
            outputs["accumulation"].detach().cpu().numpy(),
            pixel_stride=args.pixel_stride,
            min_accumulation=args.min_accumulation,
            max_depth=args.max_depth,
        )
        points.append(world_points)
        colors.append(point_colors)
        print(f"Rendered geometry view {index}/{len(cameras)}")

    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(np.concatenate(points))
    point_cloud.colors = o3d.utility.Vector3dVector(np.concatenate(colors))
    point_cloud = point_cloud.voxel_down_sample(args.voxel_size)
    point_cloud, _ = point_cloud.remove_statistical_outlier(
        nb_neighbors=20,
        std_ratio=2.0,
    )
    floor = estimate_floor(point_cloud, args.voxel_size)
    bounds = np.stack(
        (
            point_cloud.get_min_bound(),
            point_cloud.get_max_bound(),
        )
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not o3d.io.write_point_cloud(str(args.output), point_cloud):
        raise RuntimeError(f"failed to write point cloud: {args.output}")
    summary = {
        "source_config": str(args.config.resolve()),
        "checkpoint": str(checkpoint),
        "step": step,
        "camera_count": len(cameras),
        "anchor_image": args.anchor_image,
        "anchor_camera_to_world": anchor_pose.tolist(),
        "point_count": len(point_cloud.points),
        "bounds": bounds.tolist(),
        "floor": floor,
        "parameters": {
            "camera_stride": args.camera_stride,
            "resolution_scale": args.resolution_scale,
            "pixel_stride": args.pixel_stride,
            "min_accumulation": args.min_accumulation,
            "max_depth": args.max_depth,
            "voxel_size": args.voxel_size,
        },
    }
    summary_path = args.output.with_suffix(".json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Wrote {len(point_cloud.points):,} scan points to {args.output}")
    print(f"Estimated floor z={floor['z_at_origin']:.4f}")
    print(f"Wrote geometry summary to {summary_path}")


if __name__ == "__main__":
    main()
