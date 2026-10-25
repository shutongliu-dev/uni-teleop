"""Fuse dense Splatfacto RGB-D renders into a cleaned TSDF surface mesh."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--camera-stride", type=int, default=2)
    parser.add_argument("--resolution-scale", type=float, default=0.25)
    parser.add_argument("--min-accumulation", type=float, default=0.98)
    parser.add_argument("--max-depth", type=float, default=3.0)
    parser.add_argument("--voxel-size", type=float, default=0.01)
    parser.add_argument("--sdf-truncation", type=float, default=0.04)
    parser.add_argument(
        "--depth-discontinuity-threshold",
        type=float,
        default=0.0,
        help=(
            "Reject pixels whose depth differs from an adjacent valid pixel by "
            "more than this value; zero disables the filter."
        ),
    )
    parser.add_argument(
        "--valid-mask-erosion",
        type=int,
        default=0,
        help="Erode the valid depth mask by this many pixels before integration.",
    )
    parser.add_argument("--minimum-component-triangles", type=int, default=200)
    parser.add_argument("--target-triangles", type=int, default=100_000)
    parser.add_argument(
        "--evaluation-pixel-stride",
        type=int,
        default=8,
        help="Pixel stride used for post-export depth-consistency evaluation.",
    )
    parser.add_argument(
        "--evaluation-min-accumulation",
        type=float,
        default=0.98,
        help="Fixed confidence threshold for comparable post-export evaluation.",
    )
    parser.add_argument(
        "--evaluation-tolerance",
        type=float,
        default=0.04,
        help="Depth tolerance used to classify covered and free-space pixels.",
    )
    parser.add_argument(
        "--bounds-min",
        type=float,
        nargs=3,
        default=(-1.45, -0.65, -0.70),
    )
    parser.add_argument(
        "--bounds-max",
        type=float,
        nargs=3,
        default=(2.20, 1.70, 0.80),
    )
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


def open3d_extrinsic(camera_to_world):
    """Convert a Nerfstudio/OpenGL camera pose to Open3D world-to-camera."""

    import numpy as np

    camera_to_world = np.asarray(camera_to_world, dtype=np.float64)
    if camera_to_world.shape != (4, 4):
        raise ValueError("camera_to_world must have shape (4, 4)")
    camera_axis_conversion = np.diag((1.0, -1.0, -1.0, 1.0))
    return np.linalg.inv(camera_to_world @ camera_axis_conversion)


def _camera_to_world(camera):
    import numpy as np

    pose = np.eye(4, dtype=np.float64)
    pose[:3] = camera.camera_to_worlds[0].detach().cpu().numpy()
    return pose


def _erode_mask(mask, iterations):
    import numpy as np

    result = np.asarray(mask, dtype=bool).copy()
    for _ in range(iterations):
        padded = np.pad(result, 1, mode="constant", constant_values=False)
        neighbors = [
            padded[row : row + result.shape[0], column : column + result.shape[1]]
            for row in range(3)
            for column in range(3)
        ]
        result = np.logical_and.reduce(neighbors)
    return result


def _depth_continuity_mask(depth, valid, threshold):
    import numpy as np

    if threshold <= 0.0:
        return valid
    result = valid.copy()
    for row_offset, column_offset in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        shifted_depth = np.roll(depth, (row_offset, column_offset), axis=(0, 1))
        shifted_valid = np.roll(valid, (row_offset, column_offset), axis=(0, 1))
        if row_offset < 0:
            shifted_valid[-1] = False
        elif row_offset > 0:
            shifted_valid[0] = False
        if column_offset < 0:
            shifted_valid[:, -1] = False
        elif column_offset > 0:
            shifted_valid[:, 0] = False
        comparable = valid & shifted_valid
        result &= ~comparable | (np.abs(depth - shifted_depth) <= threshold)
    return result


def _integrate_camera(volume, camera, outputs, args):
    import numpy as np
    import open3d as o3d

    depth = outputs["depth"].squeeze(-1).detach().cpu().numpy().astype(np.float32)
    accumulation = outputs["accumulation"].squeeze(-1).detach().cpu().numpy()
    finite_depth = np.isfinite(depth) & (depth > 0.02) & (depth < args.max_depth)
    evaluation_valid = finite_depth & (accumulation >= args.evaluation_min_accumulation)
    integration_valid = finite_depth & (accumulation >= args.min_accumulation)
    integration_valid = _depth_continuity_mask(
        depth,
        integration_valid,
        args.depth_discontinuity_threshold,
    )
    integration_valid = _erode_mask(
        integration_valid,
        args.valid_mask_erosion,
    )
    height, width = depth.shape
    intrinsic_parameters = {
        "width": width,
        "height": height,
        "fx": float(camera.fx.item()),
        "fy": float(camera.fy.item()),
        "cx": float(camera.cx.item()),
        "cy": float(camera.cy.item()),
    }
    camera_to_world = _camera_to_world(camera)
    if volume is not None:
        filtered_depth = np.where(integration_valid, depth, 0.0).astype(np.float32)
        color = outputs["rgb"].clamp(0.0, 1.0).mul(255.0).byte().detach().cpu().numpy()
        intrinsic = o3d.camera.PinholeCameraIntrinsic(
            intrinsic_parameters["width"],
            intrinsic_parameters["height"],
            intrinsic_parameters["fx"],
            intrinsic_parameters["fy"],
            intrinsic_parameters["cx"],
            intrinsic_parameters["cy"],
        )
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(np.ascontiguousarray(color, dtype=np.uint8)),
            o3d.geometry.Image(np.ascontiguousarray(filtered_depth)),
            depth_scale=1.0,
            depth_trunc=args.max_depth,
            convert_rgb_to_intensity=False,
        )
        volume.integrate(
            rgbd,
            intrinsic,
            open3d_extrinsic(camera_to_world),
        )
    return {
        "camera_to_world": camera_to_world,
        "depth": depth,
        "valid": evaluation_valid,
        "intrinsic": intrinsic_parameters,
        "finite_depth_count": int(finite_depth.sum()),
        "integration_valid_count": int(integration_valid.sum()),
        "evaluation_valid_count": int(evaluation_valid.sum()),
        "integrated": volume is not None,
    }


def _component_statistics(mesh, minimum_component_triangles):
    import numpy as np

    if len(mesh.triangles) == 0:
        return {
            "component_count": 0,
            "small_component_count": 0,
            "small_component_triangle_count": 0,
            "largest_component_triangle_fraction": 0.0,
        }
    _, counts, _ = mesh.cluster_connected_triangles()
    counts = np.asarray(counts, dtype=np.int64)
    triangle_count = int(counts.sum())
    small = counts < minimum_component_triangles
    return {
        "component_count": len(counts),
        "small_component_count": int(small.sum()),
        "small_component_triangle_count": int(counts[small].sum()),
        "largest_component_triangle_fraction": (
            float(counts.max() / triangle_count) if triangle_count else 0.0
        ),
    }


def _mesh_quality(mesh):
    import numpy as np

    triangles = np.asarray(mesh.triangles)
    vertices = np.asarray(mesh.vertices)
    if len(triangles) == 0:
        return {
            "surface_area": 0.0,
            "median_triangle_area": 0.0,
            "p95_triangle_area": 0.0,
        }
    first = vertices[triangles[:, 0]]
    second = vertices[triangles[:, 1]]
    third = vertices[triangles[:, 2]]
    areas = 0.5 * np.linalg.norm(np.cross(second - first, third - first), axis=1)
    return {
        "surface_area": float(areas.sum()),
        "median_triangle_area": float(np.median(areas)),
        "p95_triangle_area": float(np.percentile(areas, 95)),
    }


def _evaluate_mesh_depth_consistency(mesh, evaluation_views, args):
    import numpy as np
    import open3d as o3d

    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    hit_depths = []
    reference_depths = []
    valid_reference_count = 0
    sampled_pixel_count = 0
    stride = args.evaluation_pixel_stride
    camera_axis_conversion = np.diag((1.0, -1.0, -1.0, 1.0))

    for view in evaluation_views:
        depth = view["depth"]
        valid = view["valid"]
        height, width = depth.shape
        rows = np.arange(stride // 2, height, stride)
        columns = np.arange(stride // 2, width, stride)
        pixel_x, pixel_y = np.meshgrid(columns, rows)
        sampled_valid = valid[pixel_y, pixel_x]
        sampled_pixel_count += int(sampled_valid.size)
        if not sampled_valid.any():
            continue

        intrinsic = view["intrinsic"]
        directions_camera = np.stack(
            (
                (pixel_x - intrinsic["cx"]) / intrinsic["fx"],
                (pixel_y - intrinsic["cy"]) / intrinsic["fy"],
                np.ones_like(pixel_x),
            ),
            axis=-1,
        )
        camera_to_world = view["camera_to_world"] @ camera_axis_conversion
        directions_world = directions_camera @ camera_to_world[:3, :3].T
        origins = np.broadcast_to(camera_to_world[:3, 3], directions_world.shape)
        rays = np.concatenate((origins, directions_world), axis=-1).astype(np.float32)
        result = scene.cast_rays(o3d.core.Tensor(rays))["t_hit"].numpy()
        hit_depths.append(result[sampled_valid])
        reference_depths.append(depth[pixel_y, pixel_x][sampled_valid])
        valid_reference_count += int(sampled_valid.sum())

    if not reference_depths:
        raise RuntimeError("no valid reference pixels for mesh evaluation")
    hit_depth = np.concatenate(hit_depths)
    reference_depth = np.concatenate(reference_depths)
    hit = np.isfinite(hit_depth)
    error = hit_depth[hit] - reference_depth[hit]
    absolute_error = np.abs(error)
    tolerance = args.evaluation_tolerance
    covered = hit & (np.abs(hit_depth - reference_depth) <= tolerance)
    intrusion = hit & (hit_depth < reference_depth - tolerance)
    behind = hit & (hit_depth > reference_depth + tolerance)
    intrusion_depth = reference_depth[intrusion] - hit_depth[intrusion]

    return {
        "sampled_pixel_count": sampled_pixel_count,
        "reference_valid_pixel_count": valid_reference_count,
        "mesh_hit_rate": float(hit.sum() / valid_reference_count),
        "surface_coverage_rate": float(covered.sum() / valid_reference_count),
        "free_space_intrusion_rate": float(intrusion.sum() / valid_reference_count),
        "behind_surface_rate": float(behind.sum() / valid_reference_count),
        "median_absolute_depth_error": (
            float(np.median(absolute_error)) if len(absolute_error) else None
        ),
        "p95_absolute_depth_error": (
            float(np.percentile(absolute_error, 95)) if len(absolute_error) else None
        ),
        "median_free_space_intrusion_depth": (
            float(np.median(intrusion_depth)) if len(intrusion_depth) else None
        ),
        "p95_free_space_intrusion_depth": (
            float(np.percentile(intrusion_depth, 95)) if len(intrusion_depth) else None
        ),
        "tolerance": tolerance,
        "pixel_stride": stride,
        "evaluation_min_accumulation": args.evaluation_min_accumulation,
    }


def _clean_mesh(mesh, args):
    import numpy as np
    import open3d as o3d

    mesh = mesh.crop(
        o3d.geometry.AxisAlignedBoundingBox(args.bounds_min, args.bounds_max)
    )
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()
    if len(mesh.triangles) == 0:
        raise RuntimeError("TSDF extraction produced an empty mesh")

    raw_component_statistics = _component_statistics(
        mesh,
        args.minimum_component_triangles,
    )
    clusters, counts, _ = mesh.cluster_connected_triangles()
    clusters = np.asarray(clusters)
    counts = np.asarray(counts)
    remove = counts[clusters] < args.minimum_component_triangles
    removed_triangle_count = int(remove.sum())
    mesh.remove_triangles_by_mask(remove)
    mesh.remove_unreferenced_vertices()
    if len(mesh.triangles) == 0:
        raise RuntimeError("all TSDF components were removed as noise")

    before_simplification = len(mesh.triangles)
    if 0 < args.target_triangles < before_simplification:
        mesh = mesh.simplify_quadric_decimation(args.target_triangles)
        mesh.remove_degenerate_triangles()
        mesh.remove_duplicated_triangles()
        mesh.remove_unreferenced_vertices()
    post_simplification_statistics = _component_statistics(
        mesh,
        args.minimum_component_triangles,
    )
    clusters, counts, _ = mesh.cluster_connected_triangles()
    clusters = np.asarray(clusters)
    counts = np.asarray(counts)
    remove = counts[clusters] < args.minimum_component_triangles
    removed_after_simplification = int(remove.sum())
    mesh.remove_triangles_by_mask(remove)
    mesh.remove_unreferenced_vertices()
    if len(mesh.triangles) == 0:
        raise RuntimeError("simplification left no usable TSDF components")
    mesh.compute_vertex_normals()
    cleanup = {
        "raw": raw_component_statistics,
        "removed_small_component_triangles": removed_triangle_count,
        "post_simplification": post_simplification_statistics,
        "removed_post_simplification_triangles": removed_after_simplification,
        "cleaned": _component_statistics(
            mesh,
            args.minimum_component_triangles,
        ),
    }
    cleanup["cleaned"].update(_mesh_quality(mesh))
    return mesh, before_simplification, cleanup


def main() -> None:
    import numpy as np
    import open3d as o3d
    import torch
    from nerfstudio.utils.eval_utils import eval_setup

    args = parse_args()
    if args.camera_stride < 1:
        raise SystemExit("camera stride must be positive")
    if not 0.0 < args.resolution_scale <= 1.0:
        raise SystemExit("resolution scale must be in (0, 1]")
    if not 0.0 <= args.min_accumulation <= 1.0:
        raise SystemExit("minimum accumulation must be in [0, 1]")
    if not 0.0 <= args.evaluation_min_accumulation <= 1.0:
        raise SystemExit("evaluation minimum accumulation must be in [0, 1]")
    if min(args.max_depth, args.voxel_size, args.sdf_truncation) <= 0.0:
        raise SystemExit("depth, voxel size, and SDF truncation must be positive")
    if args.depth_discontinuity_threshold < 0.0:
        raise SystemExit("depth discontinuity threshold cannot be negative")
    if args.valid_mask_erosion < 0:
        raise SystemExit("valid mask erosion cannot be negative")
    if args.evaluation_pixel_stride < 1:
        raise SystemExit("evaluation pixel stride must be positive")
    if args.evaluation_tolerance <= 0.0:
        raise SystemExit("evaluation tolerance must be positive")
    if np.any(np.asarray(args.bounds_min) >= np.asarray(args.bounds_max)):
        raise SystemExit("each minimum mesh bound must be below its maximum")

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
    cameras = camera_entries[:: args.camera_stride]
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=args.voxel_size,
        sdf_trunc=args.sdf_truncation,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
    )
    evaluation_views = []
    integrated_count = 0
    for source_index, (image_name, source_camera) in enumerate(camera_entries):
        camera = source_camera.to(pipeline.device)
        camera.rescale_output_resolution(args.resolution_scale)
        with torch.no_grad():
            outputs = pipeline.model.get_outputs_for_camera(camera)
        should_integrate = source_index % args.camera_stride == 0
        evaluation_views.append(
            _integrate_camera(
                volume if should_integrate else None,
                camera,
                outputs,
                args,
            )
        )
        if should_integrate:
            integrated_count += 1
            print(
                f"Integrated TSDF view {integrated_count}/{len(cameras)}: {image_name}"
            )

    mesh, triangles_before_simplification, component_cleanup = _clean_mesh(
        volume.extract_triangle_mesh(),
        args,
    )
    depth_consistency = _evaluate_mesh_depth_consistency(
        mesh,
        evaluation_views,
        args,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not o3d.io.write_triangle_mesh(str(args.output), mesh):
        raise RuntimeError(f"failed to write TSDF mesh: {args.output}")

    bounds = np.stack((mesh.get_min_bound(), mesh.get_max_bound()))
    summary = {
        "version": 2,
        "source_config": str(args.config.resolve()),
        "checkpoint": str(checkpoint),
        "step": step,
        "camera_count": len(cameras),
        "evaluation_camera_count": len(evaluation_views),
        "anchor_image": args.anchor_image,
        "anchor_camera_to_world": _camera_to_world(anchors[0]).tolist(),
        "geometry_source": "splatfacto_expected_depth_tsdf",
        "metric_status": "estimated",
        "vertex_count": len(mesh.vertices),
        "triangle_count": len(mesh.triangles),
        "triangles_before_simplification": triangles_before_simplification,
        "bounds": bounds.tolist(),
        "component_cleanup": component_cleanup,
        "depth_consistency": depth_consistency,
        "depth_filtering": {
            "finite_depth_pixels": sum(
                view["finite_depth_count"]
                for view in evaluation_views
                if view["integrated"]
            ),
            "integration_valid_pixels": sum(
                view["integration_valid_count"]
                for view in evaluation_views
                if view["integrated"]
            ),
            "evaluation_valid_pixels": sum(
                view["evaluation_valid_count"] for view in evaluation_views
            ),
        },
        "parameters": {
            "camera_stride": args.camera_stride,
            "resolution_scale": args.resolution_scale,
            "min_accumulation": args.min_accumulation,
            "max_depth": args.max_depth,
            "voxel_size": args.voxel_size,
            "sdf_truncation": args.sdf_truncation,
            "depth_discontinuity_threshold": (args.depth_discontinuity_threshold),
            "valid_mask_erosion": args.valid_mask_erosion,
            "minimum_component_triangles": args.minimum_component_triangles,
            "target_triangles": args.target_triangles,
            "evaluation_pixel_stride": args.evaluation_pixel_stride,
            "evaluation_min_accumulation": (args.evaluation_min_accumulation),
            "evaluation_tolerance": args.evaluation_tolerance,
            "crop_bounds": [args.bounds_min, args.bounds_max],
        },
    }
    summary_path = args.output.with_suffix(".json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(
        f"Wrote TSDF mesh with {len(mesh.vertices):,} vertices and "
        f"{len(mesh.triangles):,} triangles to {args.output}"
    )
    print(f"Wrote TSDF summary to {summary_path}")


if __name__ == "__main__":
    main()
