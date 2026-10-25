"""Build conservative box collisions from a Gaussian-splat point cloud."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gaussians", type=Path, required=True)
    parser.add_argument("--geometry-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preview", type=Path)
    parser.add_argument("--scan-units-per-meter", type=float, default=0.198)
    parser.add_argument(
        "--scale-method",
        default="anchor_camera_height_alignment",
        help="provenance label for the reconstruction-to-metric scale",
    )
    parser.add_argument(
        "--metric-scale",
        action="store_true",
        help="mark the supplied scale as measured rather than estimated",
    )
    parser.add_argument("--cell-size", type=float, default=0.04)
    parser.add_argument("--opacity", type=float, default=0.7)
    parser.add_argument("--minimum-points", type=int, default=12)
    parser.add_argument("--minimum-height", type=float, default=0.08)
    parser.add_argument("--maximum-height", type=float, default=0.55)
    parser.add_argument("--u-bounds", type=float, nargs=2, default=(-1.45, 1.65))
    parser.add_argument("--v-bounds", type=float, nargs=2, default=(-0.6, 1.45))
    return parser.parse_args()


def load_gaussians(path: Path):
    import numpy as np

    with path.open("rb") as stream:
        names = []
        count = 0
        while True:
            line = stream.readline().decode("ascii").strip()
            if line.startswith("element vertex"):
                count = int(line.split()[-1])
            elif line.startswith("property float"):
                names.append(line.split()[-1])
            elif line == "end_header":
                offset = stream.tell()
                break
        if count < 1 or not {"x", "y", "z", "opacity"}.issubset(names):
            raise ValueError("input is not a Nerfstudio Gaussian PLY")
        data_type = np.dtype([(name, "<f4") for name in names])
        stream.seek(offset)
        vertices = np.fromfile(stream, dtype=data_type, count=count)
    points = np.column_stack((vertices["x"], vertices["y"], vertices["z"])).astype(
        np.float64
    )
    opacity = 1.0 / (1.0 + np.exp(-vertices["opacity"].astype(np.float64)))
    return points, opacity


def floor_basis(plane):
    import numpy as np

    plane = np.asarray(plane, dtype=np.float64)
    normal = plane[:3] / np.linalg.norm(plane[:3])
    if normal[2] < 0.0:
        normal *= -1.0
        plane *= -1.0
    first = np.array((1.0, 0.0, 0.0))
    first -= normal * np.dot(first, normal)
    first /= np.linalg.norm(first)
    second = np.cross(normal, first)
    return first, second, normal, float(plane[3])


def largest_rectangle(mask):
    import numpy as np

    rows, columns = mask.shape
    heights = np.zeros(columns, dtype=np.int32)
    best = (0, 0, 0, 0, 0)
    for row in range(rows):
        heights = np.where(mask[row], heights + 1, 0)
        stack = []
        for column in range(columns + 1):
            current = heights[column] if column < columns else 0
            start = column
            while stack and stack[-1][1] > current:
                left, height = stack.pop()
                area = int(height * (column - left))
                if area > best[0]:
                    best = (area, row + 1 - height, row + 1, left, column)
                start = left
            if current and (not stack or stack[-1][1] < current):
                stack.append((start, current))
    return best


def rectangle_cover(occupancy, heights, *, cell_size, u_min, v_min):
    import numpy as np

    remaining = occupancy.copy()
    rectangles = []
    while remaining.any():
        area, row_start, row_stop, column_start, column_stop = largest_rectangle(
            remaining
        )
        if area < 1:
            raise RuntimeError("failed to cover occupied collision cells")
        region = heights[row_start:row_stop, column_start:column_stop]
        evidence = occupancy[row_start:row_stop, column_start:column_stop]
        height = float(np.quantile(region[evidence], 0.9))
        rectangles.append(
            {
                "center_uv": [
                    u_min + 0.5 * (column_start + column_stop) * cell_size,
                    v_min + 0.5 * (row_start + row_stop) * cell_size,
                ],
                "size_uv": [
                    (column_stop - column_start) * cell_size,
                    (row_stop - row_start) * cell_size,
                ],
                "height": height,
                "cell_count": area,
            }
        )
        remaining[row_start:row_stop, column_start:column_stop] = False
    return rectangles


def build_collision(args, points, opacity, geometry):
    import numpy as np
    from scipy import ndimage

    first, second, normal, plane_offset = floor_basis(geometry["floor"]["plane"])
    u = points @ first
    v = points @ second
    height = points @ normal + plane_offset
    u_min, u_max = args.u_bounds
    v_min, v_max = args.v_bounds
    columns = int(np.ceil((u_max - u_min) / args.cell_size))
    rows = int(np.ceil((v_max - v_min) / args.cell_size))
    valid = (
        (opacity >= args.opacity)
        & (height >= 0.025)
        & (height <= args.maximum_height)
        & (u >= u_min)
        & (u < u_max)
        & (v >= v_min)
        & (v < v_max)
    )
    column = ((u[valid] - u_min) / args.cell_size).astype(np.int32)
    row = ((v[valid] - v_min) / args.cell_size).astype(np.int32)
    counts = np.zeros((rows, columns), dtype=np.int32)
    heights = np.zeros((rows, columns), dtype=np.float64)
    np.add.at(counts, (row, column), 1)
    np.maximum.at(heights, (row, column), height[valid])
    occupancy = (counts >= args.minimum_points) & (heights >= args.minimum_height)
    occupancy = ndimage.binary_closing(
        occupancy,
        structure=np.ones((2, 2)),
        iterations=1,
    )
    occupancy = ndimage.binary_opening(
        occupancy,
        structure=np.ones((2, 2)),
        iterations=1,
    )
    labels, _ = ndimage.label(occupancy)
    component_sizes = np.bincount(labels.ravel())
    occupancy &= component_sizes[labels] >= 4

    anchor = np.asarray(geometry["anchor_camera_to_world"], dtype=np.float64)
    anchor_u = float(anchor[:3, 3] @ first)
    anchor_v = float(anchor[:3, 3] @ second)
    grid_rows, grid_columns = np.ogrid[:rows, :columns]
    cell_u = u_min + (grid_columns + 0.5) * args.cell_size
    cell_v = v_min + (grid_rows + 0.5) * args.cell_size
    occupancy[(cell_u - anchor_u) ** 2 + (cell_v - anchor_v) ** 2 < 0.15**2] = False
    rectangles = rectangle_cover(
        occupancy,
        heights,
        cell_size=args.cell_size,
        u_min=u_min,
        v_min=v_min,
    )
    floor_origin = -plane_offset * normal
    rotation = np.column_stack((first, second, normal))
    boxes = []
    for index, rectangle in enumerate(rectangles):
        center = (
            floor_origin
            + first * rectangle["center_uv"][0]
            + second * rectangle["center_uv"][1]
            + normal * rectangle["height"] / 2.0
        )
        boxes.append(
            {
                "name": f"scan_collision_{index:03d}",
                "center": center.tolist(),
                "center_uv": rectangle["center_uv"],
                "rotation": rotation.tolist(),
                "size": [
                    rectangle["size_uv"][0],
                    rectangle["size_uv"][1],
                    rectangle["height"],
                ],
                "cell_count": rectangle["cell_count"],
            }
        )
    return boxes, occupancy, heights, (first, second, normal), (anchor_u, anchor_v)


def write_preview(path, occupancy, heights, args, boxes, anchor_uv):
    import matplotlib.pyplot as plt
    from matplotlib import patches

    extent = (*args.u_bounds, *args.v_bounds)
    figure, axes = plt.subplots(1, 2, figsize=(13, 6), constrained_layout=True)
    axes[0].imshow(
        heights,
        origin="lower",
        extent=extent,
        cmap="viridis",
        vmin=0.0,
        vmax=args.maximum_height,
    )
    axes[0].set_title("Maximum scan-derived obstacle height")
    axes[1].imshow(occupancy, origin="lower", extent=extent, cmap="gray_r")
    for box in boxes:
        width, height = box["size"][:2]
        center_u, center_v = box["center_uv"]
        axes[1].add_patch(
            patches.Rectangle(
                (center_u - width / 2.0, center_v - height / 2.0),
                width,
                height,
                fill=False,
                edgecolor="tab:blue",
                linewidth=0.5,
            )
        )
    axes[1].plot(*anchor_uv, "ro", label="initial G1 camera")
    axes[1].legend(loc="upper left")
    axes[1].set_title(f"{len(boxes)} conservative collision boxes")
    for axis in axes:
        axis.set_aspect("equal")
        axis.set_xlabel("floor x (scan units)")
        axis.set_ylabel("floor y (scan units)")
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)


def main() -> None:
    args = parse_args()
    if not 0.0 < args.opacity < 1.0:
        raise SystemExit("opacity threshold must be in (0, 1)")
    if args.scan_units_per_meter <= 0.0 or args.cell_size <= 0.0:
        raise SystemExit("scale and cell size must be positive")
    points, opacity = load_gaussians(args.gaussians)
    geometry = json.loads(args.geometry_summary.read_text())
    boxes, occupancy, heights, basis, anchor_uv = build_collision(
        args,
        points,
        opacity,
        geometry,
    )
    output = {
        "version": 2,
        "source_gaussians": str(args.gaussians.resolve()),
        "source_geometry": str(args.geometry_summary.resolve()),
        "scan_units_per_meter": args.scan_units_per_meter,
        "scale_calibration": {
            "status": "metric" if args.metric_scale else "estimated",
            "method": args.scale_method,
            "has_measured_reference": args.metric_scale,
        },
        "geometry_provenance": {
            "appearance": "nerfstudio_splatfacto_3dgs",
            "surface_geometry": "depth_rendered_from_trained_3dgs",
            "training_depth": "colmap_sparse_sfm",
            "training_depth_is_metric": False,
        },
        "anchor_image": geometry["anchor_image"],
        "anchor_camera_to_world": geometry["anchor_camera_to_world"],
        "scan_up": basis[2].tolist(),
        "floor_plane": geometry["floor"]["plane"],
        "boxes": boxes,
        "parameters": {
            "cell_size": args.cell_size,
            "opacity": args.opacity,
            "minimum_points": args.minimum_points,
            "minimum_height": args.minimum_height,
            "maximum_height": args.maximum_height,
            "u_bounds": args.u_bounds,
            "v_bounds": args.v_bounds,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    if args.preview is not None:
        write_preview(args.preview, occupancy, heights, args, boxes, anchor_uv)
    print(f"Wrote {len(boxes)} scan-derived collision boxes to {args.output}")


if __name__ == "__main__":
    main()
