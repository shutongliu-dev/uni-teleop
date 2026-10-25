"""Build validated convex collision meshes from reconstructed scan geometry.

The reconstruction scale is estimated unless ``--metric-scale`` is supplied.
Consequently, distances described as meters in this script are pseudo-metric:
they are derived from ``--scan-units-per-meter`` and keep that provenance in
the output manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path

SCHEMA = "sonic_mujoco.scan_collision_convex.v1"


@dataclass(frozen=True)
class FloorFrame:
    """Orthonormal floor frame expressed in reconstruction coordinates."""

    origin: object
    basis: object
    plane: object


@dataclass(frozen=True)
class Partition:
    """A deterministic group of occupied pseudo-metric voxels."""

    component_id: int
    chunk_key: tuple[int, int, int]
    voxels: object


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--mesh-dir",
        type=Path,
        help="OBJ directory; defaults to a meshes/ folder beside --output",
    )
    parser.add_argument(
        "--geometry-summary",
        type=Path,
        help="JSON containing floor.plane; defaults to the input sidecar",
    )
    parser.add_argument("--floor-plane", type=float, nargs=4)
    parser.add_argument("--scan-units-per-meter", type=float, default=0.198)
    parser.add_argument(
        "--metric-scale",
        action="store_true",
        help="mark scale as measured instead of estimated",
    )
    parser.add_argument("--voxel-size-m", type=float, default=0.05)
    parser.add_argument("--minimum-height-m", type=float, default=0.05)
    parser.add_argument("--maximum-height-m", type=float, default=2.8)
    parser.add_argument(
        "--anchor-clearance-m",
        type=float,
        default=0.75,
        help="horizontal collision-free radius around the anchor camera",
    )
    parser.add_argument("--u-bounds-m", type=float, nargs=2)
    parser.add_argument("--v-bounds-m", type=float, nargs=2)
    parser.add_argument("--minimum-component-voxels", type=int, default=12)
    parser.add_argument("--initial-chunk-size-m", type=float, default=1.0)
    parser.add_argument(
        "--partition-fraction",
        type=float,
        default=0.85,
        help="fraction of the hull budget reserved for initial partitions",
    )
    parser.add_argument("--maximum-hulls", type=int, default=256)
    parser.add_argument("--maximum-hulls-per-partition", type=int, default=2)
    parser.add_argument("--maximum-vertices-per-hull", type=int, default=64)
    parser.add_argument("--vhacd-resolution", type=int, default=50_000)
    parser.add_argument("--vhacd-max-recursion-depth", type=int, default=8)
    parser.add_argument(
        "--vhacd-volume-error-percent",
        type=float,
        default=1.0,
    )
    parser.add_argument("--maximum-dense-grid-cells", type=int, default=40_000_000)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_args(args: argparse.Namespace) -> None:
    import numpy as np

    if not args.input.is_file():
        raise SystemExit(f"input geometry does not exist: {args.input}")
    positive = {
        "scan units per meter": args.scan_units_per_meter,
        "voxel size": args.voxel_size_m,
        "initial chunk size": args.initial_chunk_size_m,
        "VHACD volume error": args.vhacd_volume_error_percent,
    }
    invalid = [
        name
        for name, value in positive.items()
        if not np.isfinite(value) or value <= 0.0
    ]
    if invalid:
        raise SystemExit(f"values must be positive: {', '.join(invalid)}")
    if not 0.0 < args.partition_fraction <= 1.0:
        raise SystemExit("partition fraction must be in (0, 1]")
    if not np.isfinite(args.minimum_height_m) or args.minimum_height_m < 0.0:
        raise SystemExit("minimum height cannot be negative")
    if not np.isfinite(args.maximum_height_m):
        raise SystemExit("maximum height must be finite")
    if not np.isfinite(args.anchor_clearance_m) or args.anchor_clearance_m < 0.0:
        raise SystemExit("anchor clearance cannot be negative")
    if args.maximum_height_m <= args.minimum_height_m:
        raise SystemExit("maximum height must exceed minimum height")
    integer_positive = {
        "minimum component voxels": args.minimum_component_voxels,
        "maximum hulls": args.maximum_hulls,
        "maximum hulls per partition": args.maximum_hulls_per_partition,
        "maximum vertices per hull": args.maximum_vertices_per_hull,
        "VHACD resolution": args.vhacd_resolution,
        "VHACD recursion depth": args.vhacd_max_recursion_depth,
        "maximum dense grid cells": args.maximum_dense_grid_cells,
    }
    invalid = [name for name, value in integer_positive.items() if value < 1]
    if invalid:
        raise SystemExit(f"integer values must be positive: {', '.join(invalid)}")
    if args.maximum_vertices_per_hull < 4:
        raise SystemExit("maximum vertices per hull must be at least four")
    for name, bounds in (("u", args.u_bounds_m), ("v", args.v_bounds_m)):
        if bounds is not None and (
            not np.isfinite(bounds).all() or bounds[0] >= bounds[1]
        ):
            raise SystemExit(f"invalid {name} bounds: {bounds}")


def _load_geometry(path: Path):
    import numpy as np
    import open3d as o3d

    mesh = o3d.io.read_triangle_mesh(str(path))
    if len(mesh.triangles):
        points = np.asarray(mesh.vertices, dtype=np.float64)
        source_kind = "triangle_mesh"
        face_count = len(mesh.triangles)
    else:
        cloud = o3d.io.read_point_cloud(str(path))
        points = np.asarray(cloud.points, dtype=np.float64)
        source_kind = "point_cloud"
        face_count = 0
    if len(points) < 4:
        raise RuntimeError(f"input geometry has fewer than four vertices: {path}")
    if not np.isfinite(points).all():
        raise RuntimeError("input geometry contains non-finite coordinates")
    return points, source_kind, face_count


def _summary_floor_plane(args: argparse.Namespace):
    if args.floor_plane is not None:
        return args.floor_plane, "command_line"
    summary_path = args.geometry_summary
    if summary_path is None and args.input.with_suffix(".json").is_file():
        summary_path = args.input.with_suffix(".json")
    if summary_path is None:
        return None, "estimated_from_geometry"
    summary = json.loads(summary_path.read_text())
    if "floor" in summary and "plane" in summary["floor"]:
        return summary["floor"]["plane"], str(summary_path.resolve())
    if "floor_plane" in summary:
        return summary["floor_plane"], str(summary_path.resolve())
    return None, "estimated_from_geometry"


def _estimate_floor_plane(points, voxel_size_scan: float):
    import numpy as np
    import open3d as o3d

    low_cutoff = float(np.quantile(points[:, 2], 0.35))
    candidates = points[points[:, 2] <= low_cutoff]
    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(candidates))
    o3d.utility.random.seed(0)
    plane, _ = cloud.segment_plane(
        distance_threshold=max(voxel_size_scan * 1.5, 0.005),
        ransac_n=3,
        num_iterations=2_000,
    )
    return plane


def _floor_frame(plane) -> FloorFrame:
    import numpy as np

    plane = np.asarray(plane, dtype=np.float64)
    if plane.shape != (4,) or not np.isfinite(plane).all():
        raise ValueError("floor plane must contain four finite values")
    norm = float(np.linalg.norm(plane[:3]))
    if norm < 1e-12:
        raise ValueError("floor plane has a zero normal")
    plane = plane / norm
    normal = plane[:3]
    if normal[2] < 0.0:
        normal = -normal
        plane = -plane
    axes = np.eye(3)
    reference = axes[int(np.argmin(np.abs(axes @ normal)))]
    first = reference - normal * float(reference @ normal)
    first /= np.linalg.norm(first)
    second = np.cross(normal, first)
    basis = np.column_stack((first, second, normal))
    origin = -float(plane[3]) * normal
    return FloorFrame(origin=origin, basis=basis, plane=plane)


def _pseudo_metric_voxels(
    points,
    frame: FloorFrame,
    args: argparse.Namespace,
    anchor_scan,
):
    import numpy as np

    metric_points = (points - frame.origin) @ frame.basis
    metric_points /= args.scan_units_per_meter
    valid = np.isfinite(metric_points).all(axis=1)
    valid &= metric_points[:, 2] >= args.minimum_height_m
    valid &= metric_points[:, 2] <= args.maximum_height_m
    if args.u_bounds_m is not None:
        valid &= metric_points[:, 0] >= args.u_bounds_m[0]
        valid &= metric_points[:, 0] <= args.u_bounds_m[1]
    if args.v_bounds_m is not None:
        valid &= metric_points[:, 1] >= args.v_bounds_m[0]
        valid &= metric_points[:, 1] <= args.v_bounds_m[1]
    if anchor_scan is not None and args.anchor_clearance_m > 0.0:
        anchor_metric = (anchor_scan - frame.origin) @ frame.basis
        anchor_metric /= args.scan_units_per_meter
        distance_squared = np.sum(
            (metric_points[:, :2] - anchor_metric[:2]) ** 2,
            axis=1,
        )
        valid &= distance_squared >= args.anchor_clearance_m**2
    metric_points = metric_points[valid]
    if len(metric_points) < 4:
        raise RuntimeError("height and bounds filters removed all scan geometry")
    voxels = np.floor(metric_points / args.voxel_size_m).astype(np.int32)
    voxels = np.unique(voxels, axis=0)
    return (
        voxels,
        len(metric_points),
        np.stack((metric_points.min(axis=0), metric_points.max(axis=0))),
    )


def _dense_components(voxels, minimum_size: int):
    import numpy as np
    from scipy import ndimage

    minimum = voxels.min(axis=0)
    shifted = voxels - minimum
    shape = tuple((shifted.max(axis=0) + 1).tolist())
    occupancy = np.zeros(shape, dtype=bool)
    occupancy[tuple(shifted.T)] = True
    labels, _ = ndimage.label(
        occupancy,
        structure=np.ones((3, 3, 3), dtype=np.uint8),
    )
    voxel_labels = labels[tuple(shifted.T)]
    counts = np.bincount(voxel_labels)
    groups = [
        voxels[voxel_labels == label]
        for label in range(1, len(counts))
        if counts[label] >= minimum_size
    ]
    return groups, len(counts) - 1


def _sparse_components(voxels, minimum_size: int):
    import numpy as np

    remaining = {tuple(int(value) for value in voxel) for voxel in voxels}
    neighbors = [
        (x, y, z)
        for x in (-1, 0, 1)
        for y in (-1, 0, 1)
        for z in (-1, 0, 1)
        if (x, y, z) != (0, 0, 0)
    ]
    groups = []
    component_count = 0
    while remaining:
        seed = min(remaining)
        remaining.remove(seed)
        pending = [seed]
        component = [seed]
        while pending:
            current = pending.pop()
            for offset in neighbors:
                neighbor = tuple(current[axis] + offset[axis] for axis in range(3))
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    pending.append(neighbor)
                    component.append(neighbor)
        component_count += 1
        if len(component) >= minimum_size:
            groups.append(np.asarray(component, dtype=np.int32))
    return groups, component_count


def _connected_components(voxels, args: argparse.Namespace):
    import numpy as np

    shape = voxels.max(axis=0) - voxels.min(axis=0) + 1
    grid_cells = int(np.prod(shape, dtype=np.int64))
    if grid_cells <= args.maximum_dense_grid_cells:
        groups, raw_count = _dense_components(
            voxels,
            args.minimum_component_voxels,
        )
        backend = "scipy_ndimage_26_connected"
    else:
        groups, raw_count = _sparse_components(
            voxels,
            args.minimum_component_voxels,
        )
        backend = "sparse_26_connected"
    groups.sort(
        key=lambda group: (
            -len(group),
            tuple(int(value) for value in group.min(axis=0)),
        )
    )
    if len(groups) > args.maximum_hulls:
        groups = groups[: args.maximum_hulls]
    if not groups:
        raise RuntimeError("no collision components survived occupancy filtering")
    return groups, raw_count, grid_cells, backend


def _partition_components(components, chunk_voxels: int) -> list[Partition]:
    import numpy as np

    partitions = []
    for component_id, component in enumerate(components):
        keys = np.floor_divide(component, chunk_voxels)
        unique_keys, inverse = np.unique(keys, axis=0, return_inverse=True)
        for index, key in enumerate(unique_keys):
            partitions.append(
                Partition(
                    component_id=component_id,
                    chunk_key=tuple(int(value) for value in key),
                    voxels=component[inverse == index],
                )
            )
    partitions.sort(
        key=lambda partition: (
            partition.component_id,
            partition.chunk_key,
        )
    )
    return partitions


def _adaptive_partitions(components, args: argparse.Namespace):
    target = max(
        len(components),
        math.floor(args.maximum_hulls * args.partition_fraction),
    )
    target = min(target, args.maximum_hulls)
    chunk_voxels = max(
        1,
        math.ceil(args.initial_chunk_size_m / args.voxel_size_m),
    )
    partitions = _partition_components(components, chunk_voxels)
    for _ in range(64):
        if len(partitions) <= target:
            return partitions, chunk_voxels, target
        chunk_voxels += max(1, math.ceil(chunk_voxels * 0.15))
        partitions = _partition_components(components, chunk_voxels)
    raise RuntimeError(
        f"could not reduce {len(partitions)} partitions to the {target} budget"
    )


def _partition_allocations(partitions, args: argparse.Namespace) -> list[int]:
    allocations = [1] * len(partitions)
    remaining = args.maximum_hulls - len(partitions)
    order = sorted(
        range(len(partitions)),
        key=lambda index: (-len(partitions[index].voxels), index),
    )
    while remaining:
        changed = False
        for index in order:
            if allocations[index] >= args.maximum_hulls_per_partition:
                continue
            allocations[index] += 1
            remaining -= 1
            changed = True
            if remaining == 0:
                break
        if not changed:
            break
    return allocations


def _box_fallback(centers, voxel_size: float):
    import numpy as np
    import trimesh

    lower = centers.min(axis=0) - voxel_size / 2.0
    upper = centers.max(axis=0) + voxel_size / 2.0
    transform = np.eye(4)
    transform[:3, 3] = (lower + upper) / 2.0
    return trimesh.creation.box(extents=upper - lower, transform=transform)


def _validated_convex_mesh(vertices, faces, maximum_vertices: int):
    import numpy as np
    import trimesh

    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    if (
        vertices.ndim != 2
        or vertices.shape[1] != 3
        or faces.ndim != 2
        or faces.shape[1] != 3
        or len(vertices) < 4
        or len(faces) < 4
        or not np.isfinite(vertices).all()
        or np.any(faces < 0)
        or np.any(faces >= len(vertices))
    ):
        raise ValueError("invalid vertices or faces returned by convex decomposition")
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=True)
    if not mesh.is_convex:
        mesh = mesh.convex_hull
    if (
        len(mesh.vertices) < 4
        or len(mesh.faces) < 4
        or len(mesh.vertices) > maximum_vertices
        or not np.isfinite(mesh.vertices).all()
        or not np.isfinite(mesh.faces).all()
        or not mesh.is_watertight
        or not mesh.is_convex
        or abs(float(mesh.volume)) <= 1e-12
    ):
        raise ValueError("convex mesh failed geometry validation")
    return mesh


def _decompose_partition(
    partition: Partition,
    allocation: int,
    args: argparse.Namespace,
):
    import numpy as np
    import trimesh
    import vhacdx

    centers = (np.asarray(partition.voxels, dtype=np.float64) + 0.5) * args.voxel_size_m
    source_mesh = trimesh.voxel.ops.points_to_marching_cubes(
        centers,
        pitch=args.voxel_size_m,
    )
    fallback = False
    try:
        raw_hulls = vhacdx.compute_vhacd(
            np.ascontiguousarray(source_mesh.vertices, dtype=np.float64),
            np.ascontiguousarray(source_mesh.faces, dtype=np.uint32),
            maxConvexHulls=allocation,
            resolution=args.vhacd_resolution,
            minimumVolumePercentErrorAllowed=args.vhacd_volume_error_percent,
            maxRecursionDepth=args.vhacd_max_recursion_depth,
            shrinkWrap=True,
            fillMode="flood",
            maxNumVerticesPerCH=args.maximum_vertices_per_hull,
            asyncACD=False,
            minEdgeLength=2,
            findBestPlane=False,
        )
        hulls = [
            _validated_convex_mesh(
                vertices,
                faces,
                args.maximum_vertices_per_hull,
            )
            for vertices, faces in raw_hulls[:allocation]
        ]
        if not hulls:
            raise ValueError("VHACD returned no hulls")
    except (RuntimeError, ValueError):
        fallback = True
        box = _box_fallback(centers, args.voxel_size_m)
        hulls = [
            _validated_convex_mesh(
                box.vertices,
                box.faces,
                args.maximum_vertices_per_hull,
            )
        ]
    return hulls, fallback, len(source_mesh.vertices), len(source_mesh.faces)


def _write_obj(path: Path, name: str, vertices, faces) -> None:
    lines = [f"# Convex collision hull generated by {Path(__file__).name}", f"o {name}"]
    lines.extend(
        f"v {vertex[0]:.12g} {vertex[1]:.12g} {vertex[2]:.12g}" for vertex in vertices
    )
    lines.extend(f"f {face[0] + 1} {face[1] + 1} {face[2] + 1}" for face in faces)
    path.write_text("\n".join(lines) + "\n")


def _reload_validate_obj(path: Path, maximum_vertices: int):
    import numpy as np
    import trimesh

    mesh = trimesh.load_mesh(path, force="mesh", process=True)
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(f"OBJ did not reload as a triangle mesh: {path}")
    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.faces)
    if (
        len(vertices) < 4
        or len(vertices) > maximum_vertices
        or len(faces) < 4
        or not np.isfinite(vertices).all()
        or not np.isfinite(faces).all()
        or np.any(faces < 0)
        or np.any(faces >= len(vertices))
        or not mesh.is_watertight
        or not mesh.is_convex
        or abs(float(mesh.volume)) <= 1e-12
    ):
        raise RuntimeError(f"written OBJ failed reload validation: {path}")
    return mesh


def _relative_mesh_path(mesh_path: Path, manifest_path: Path) -> str:
    relative = Path(os.path.relpath(mesh_path, manifest_path.parent))
    if ".." in relative.parts:
        raise ValueError("mesh directory must be inside the manifest directory")
    return relative.as_posix()


def _point_inside_any_mesh(point_scan, meshes, manifest_path: Path, scale: float):
    import numpy as np
    import trimesh
    from scipy.spatial import ConvexHull

    for mesh in meshes:
        path = manifest_path.parent / mesh["file"]
        geometry = trimesh.load_mesh(path, force="mesh", process=True)
        local_point_m = (
            np.asarray(point_scan, dtype=np.float64)
            - np.asarray(mesh["center_scan"], dtype=np.float64)
        ) / scale
        equations = ConvexHull(np.asarray(geometry.vertices)).equations
        if np.all(equations[:, :3] @ local_point_m + equations[:, 3] <= 1e-8):
            return True, mesh["name"]
    return False, None


def _prepare_outputs(args: argparse.Namespace) -> Path:
    mesh_dir = args.mesh_dir or args.output.parent / "meshes"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    mesh_dir.mkdir(parents=True, exist_ok=True)
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(
            f"output manifest exists; pass --overwrite to replace it: {args.output}"
        )
    if not args.overwrite:
        existing = sorted(mesh_dir.glob("scan_collision_hull_*.obj"))
        if existing:
            raise FileExistsError(
                f"collision OBJ already exists; pass --overwrite: {existing[0]}"
            )
    return mesh_dir


def _build_hulls(
    partitions,
    allocations,
    frame: FloorFrame,
    args: argparse.Namespace,
    mesh_dir: Path,
):
    import numpy as np

    records = []
    hull_index = 0
    fallback_count = 0
    meshing_vertices = 0
    meshing_faces = 0
    for partition_id, (partition, allocation) in enumerate(
        zip(partitions, allocations, strict=True)
    ):
        hulls, fallback, source_vertices, source_faces = _decompose_partition(
            partition,
            allocation,
            args,
        )
        fallback_count += int(fallback)
        meshing_vertices += source_vertices
        meshing_faces += source_faces
        for piece_id, hull in enumerate(hulls):
            metric_vertices = np.asarray(hull.vertices, dtype=np.float64)
            world_vertices = (
                frame.origin
                + (metric_vertices * args.scan_units_per_meter) @ frame.basis.T
            )
            center_scan = (
                world_vertices.min(axis=0) + world_vertices.max(axis=0)
            ) / 2.0
            local_vertices_m = (
                world_vertices - center_scan
            ) / args.scan_units_per_meter
            faces = np.asarray(hull.faces, dtype=np.int64)
            name = f"scan_collision_hull_{hull_index:03d}"
            path = mesh_dir / f"{name}.obj"
            _write_obj(path, name, local_vertices_m, faces)
            reloaded = _reload_validate_obj(
                path,
                args.maximum_vertices_per_hull,
            )
            aabb_size_m = np.ptp(np.asarray(reloaded.vertices), axis=0)
            records.append(
                {
                    "name": name,
                    "file": _relative_mesh_path(path, args.output),
                    "sha256": _sha256(path),
                    "center_scan": center_scan.tolist(),
                    "vertex_count": len(reloaded.vertices),
                    "face_count": len(reloaded.faces),
                    "aabb_size_m": aabb_size_m.tolist(),
                    "volume_pseudo_metric_m3": abs(float(reloaded.volume)),
                    "bounds_local_m": np.asarray(reloaded.bounds).tolist(),
                    "bounds_world_scan": np.stack(
                        (world_vertices.min(axis=0), world_vertices.max(axis=0))
                    ).tolist(),
                    "component_id": partition.component_id,
                    "partition_id": partition_id,
                    "partition_piece_id": piece_id,
                    "occupied_voxel_count": len(partition.voxels),
                    "decomposition": "axis_aligned_box_fallback"
                    if fallback
                    else "vhacdx",
                }
            )
            hull_index += 1
            if hull_index > args.maximum_hulls:
                raise RuntimeError("convex decomposition exceeded maximum hull count")
        print(
            f"Built partition {partition_id + 1}/{len(partitions)} "
            f"({len(hulls)} hulls, {len(partition.voxels)} voxels)"
        )
    return records, fallback_count, meshing_vertices, meshing_faces


def main() -> None:
    import numpy as np

    args = parse_args()
    _validate_args(args)
    mesh_dir = _prepare_outputs(args)
    source_hash = _sha256(args.input)
    points, source_kind, source_face_count = _load_geometry(args.input)
    summary_path = args.geometry_summary
    if summary_path is None and args.input.with_suffix(".json").is_file():
        summary_path = args.input.with_suffix(".json")
    geometry_summary = (
        json.loads(summary_path.read_text()) if summary_path is not None else {}
    )
    anchor_pose = geometry_summary.get("anchor_camera_to_world")
    anchor_scan = None
    if anchor_pose is None:
        raise ValueError(
            "geometry summary must define anchor_camera_to_world for runtime alignment"
        )
    anchor_pose = np.asarray(anchor_pose, dtype=np.float64)
    if anchor_pose.shape != (4, 4) or not np.isfinite(anchor_pose).all():
        raise ValueError("anchor_camera_to_world must be a finite 4x4 matrix")
    anchor_scan = anchor_pose[:3, 3]
    plane, floor_source = _summary_floor_plane(args)
    if plane is None:
        plane = _estimate_floor_plane(
            points,
            args.voxel_size_m * args.scan_units_per_meter,
        )
    frame = _floor_frame(plane)
    voxels, filtered_point_count, metric_bounds = _pseudo_metric_voxels(
        points,
        frame,
        args,
        anchor_scan,
    )
    components, raw_component_count, grid_cells, component_backend = (
        _connected_components(voxels, args)
    )
    kept_voxel_count = sum(len(component) for component in components)
    partitions, chunk_voxels, target_partitions = _adaptive_partitions(
        components,
        args,
    )
    allocations = _partition_allocations(partitions, args)
    hulls, fallback_count, meshing_vertices, meshing_faces = _build_hulls(
        partitions,
        allocations,
        frame,
        args,
        mesh_dir,
    )
    if not 0 < len(hulls) <= args.maximum_hulls:
        raise RuntimeError("invalid final hull count")
    hull_world_bounds = np.asarray(
        [hull["bounds_world_scan"] for hull in hulls],
        dtype=np.float64,
    )
    hull_bounds_scan = np.stack(
        (
            hull_world_bounds[:, 0].min(axis=0),
            hull_world_bounds[:, 1].max(axis=0),
        )
    )
    hull_aabb_size_m = (
        hull_bounds_scan[1] - hull_bounds_scan[0]
    ) / args.scan_units_per_meter
    total_hull_volume = sum(hull["volume_pseudo_metric_m3"] for hull in hulls)
    anchor_inside_hull = False
    anchor_containing_hull = None
    if anchor_scan is not None:
        anchor_inside_hull, anchor_containing_hull = _point_inside_any_mesh(
            anchor_scan,
            hulls,
            args.output,
            args.scan_units_per_meter,
        )
    combined_hash = hashlib.sha256(
        json.dumps(
            [(hull["file"], hull["sha256"]) for hull in hulls],
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    manifest = {
        "schema": SCHEMA,
        "version": 2,
        "scan_units_per_meter": args.scan_units_per_meter,
        "scale_calibration": {
            "status": "metric" if args.metric_scale else "estimated",
            "method": "supplied_scan_units_per_meter",
            "has_measured_reference": args.metric_scale,
        },
        "anchor_image": geometry_summary.get("anchor_image"),
        "anchor_camera_to_world": anchor_pose.tolist(),
        "scan_up": frame.basis[:, 2].tolist(),
        "floor_plane": frame.plane.tolist(),
        "source": {
            "path": str(args.input.resolve()),
            "sha256": source_hash,
            "kind": source_kind,
            "vertex_or_point_count": len(points),
            "face_count": source_face_count,
            "surface_sampling": "mesh_vertices" if source_face_count else "points",
            "geometry_summary": str(summary_path.resolve())
            if summary_path is not None
            else None,
        },
        "coordinate_frame": {
            "mesh_vertex_units": "meters",
            "mesh_vertices_are_local": True,
            "mesh_axes": "scan_axes",
            "mesh_local_origin": "center_scan",
            "center_units": "scan_units",
            "scan_units_per_meter": args.scan_units_per_meter,
            "scale_status": "metric" if args.metric_scale else "estimated",
            "occupancy_units": "pseudo_metric_meters"
            if not args.metric_scale
            else "meters",
        },
        "floor": {
            "plane_scan": frame.plane.tolist(),
            "origin_scan": frame.origin.tolist(),
            "basis_floor_to_scan": frame.basis.tolist(),
            "source": floor_source,
        },
        "parameters": {
            "voxel_size_m": args.voxel_size_m,
            "minimum_height_m": args.minimum_height_m,
            "maximum_height_m": args.maximum_height_m,
            "anchor_clearance_m": args.anchor_clearance_m,
            "u_bounds_m": args.u_bounds_m,
            "v_bounds_m": args.v_bounds_m,
            "minimum_component_voxels": args.minimum_component_voxels,
            "initial_chunk_size_m": args.initial_chunk_size_m,
            "partition_fraction": args.partition_fraction,
            "effective_chunk_size_m": chunk_voxels * args.voxel_size_m,
            "target_partition_count": target_partitions,
            "maximum_hulls": args.maximum_hulls,
            "maximum_hulls_per_partition": args.maximum_hulls_per_partition,
            "maximum_vertices_per_hull": args.maximum_vertices_per_hull,
            "vhacd_resolution": args.vhacd_resolution,
            "vhacd_max_recursion_depth": args.vhacd_max_recursion_depth,
            "vhacd_volume_error_percent": args.vhacd_volume_error_percent,
            "maximum_dense_grid_cells": args.maximum_dense_grid_cells,
        },
        "statistics": {
            "source_filtered_point_count": filtered_point_count,
            "metric_bounds": metric_bounds.tolist(),
            "raw_occupied_voxel_count": len(voxels),
            "kept_occupied_voxel_count": kept_voxel_count,
            "raw_component_count": raw_component_count,
            "kept_component_count": len(components),
            "component_backend": component_backend,
            "dense_grid_cell_count": grid_cells,
            "partition_count": len(partitions),
            "allocated_hull_budget": sum(allocations),
            "hull_count": len(hulls),
            "fallback_partition_count": fallback_count,
            "partition_surface_vertex_count": meshing_vertices,
            "partition_surface_face_count": meshing_faces,
            "total_hull_vertex_count": sum(hull["vertex_count"] for hull in hulls),
            "total_hull_face_count": sum(hull["face_count"] for hull in hulls),
            "total_hull_volume_pseudo_metric_m3": total_hull_volume,
            "hull_bounds_scan": hull_bounds_scan.tolist(),
            "hull_aabb_size_m": hull_aabb_size_m.tolist(),
            "occupied_voxel_volume_m3": kept_voxel_count * args.voxel_size_m**3,
            "hull_to_occupied_voxel_volume_ratio": total_hull_volume
            / (kept_voxel_count * args.voxel_size_m**3),
            "anchor_inside_hull": anchor_inside_hull,
            "anchor_containing_hull": anchor_containing_hull,
        },
        "validation": {
            "passed": True,
            "hull_limit_passed": len(hulls) <= args.maximum_hulls,
            "all_files_hashed": all(len(hull["sha256"]) == 64 for hull in hulls),
            "all_vertices_finite": True,
            "all_faces_valid": True,
            "all_meshes_convex": True,
            "all_meshes_watertight": True,
            "combined_mesh_sha256": combined_hash,
        },
        "meshes": hulls,
    }
    args.output.write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        f"Wrote {len(hulls)} validated convex hulls from "
        f"{kept_voxel_count:,} occupied voxels"
    )
    print(f"Wrote collision manifest to {args.output}")


if __name__ == "__main__":
    main()
