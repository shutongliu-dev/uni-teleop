"""Read-only GPU particle surfacing for native and stopped-timeline rendering.

The field uses a compact, weighted local particle centre (Zhu/Bridson, 2005),
with a nearest-particle distance cap to avoid bridging empty space. This is a
visual reconstruction, not a fluid solver or a volume/mass measurement.

Algorithm references (no third-party implementation is vendored):
https://www.cs.ubc.ca/~rbridson/docs/zhu-siggraph05-sandfluid.pdf
https://github.com/newton-physics/newton/pull/3349
https://github.com/NVIDIA/warp/blob/v1.16.0/warp/examples/core/example_marching_cubes.py
"""

from dataclasses import dataclass
import hashlib
import math
from typing import Any

import numpy as np
from scipy.spatial import cKDTree

try:
    import warp as wp
except ImportError:  # CPU catalog discovery does not require the native runtime.
    wp = None


if wp is not None:

    @wp.kernel(enable_backward=False)
    def _make_surface_field(
        points: wp.array(dtype=wp.vec3),
        grid: wp.uint64,
        lower: wp.vec3,
        voxel_size: float,
        support_radius: float,
        surface_radius: float,
        distance_cap: float,
        field: wp.array3d(dtype=wp.float32),
    ):
        i, j, k = wp.tid()
        pos = lower + voxel_size * wp.vec3(float(i), float(j), float(k))
        query = wp.hash_grid_query(grid, pos, support_radius)
        weight_sum = float(0.0)
        weighted_offset = wp.vec3(0.0)
        nearest_sq = support_radius * support_radius
        for particle in query:
            offset = points[particle] - pos
            distance_sq = wp.dot(offset, offset)
            if distance_sq < support_radius * support_radius:
                nearest_sq = wp.min(nearest_sq, distance_sq)
                weight = 1.0 - distance_sq / (support_radius * support_radius)
                weight = weight * weight * weight
                weight_sum += weight
                weighted_offset += weight * offset
        value = support_radius
        if weight_sum > 1.0e-12:
            centre_distance = wp.length(weighted_offset / weight_sum)
            value = wp.max(centre_distance - surface_radius, wp.sqrt(nearest_sq) - distance_cap)
        field[i, j, k] = value


@dataclass(frozen=True)
class ParticleSurfaceMesh:
    vertices: np.ndarray
    triangles: np.ndarray
    normals: np.ndarray
    metadata: dict[str, Any]


def _positive(value: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f'{name} must be finite and positive')
    return value


def _components(points: np.ndarray, radius: float) -> list[np.ndarray]:
    """Keep distant droplets out of each other's reconstruction grids."""
    if not len(points):
        return []
    tree = cKDTree(points)
    pair_count = (int(tree.count_neighbors(tree, radius)) - len(points)) // 2
    if pair_count > 1_000_000:
        raise ValueError('Particle neighbourhood exceeds the bounded reconstruction budget')
    parent = np.arange(len(points), dtype=np.int32)

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = int(parent[i])
        return i

    for left, right in tree.query_pairs(radius, output_type='ndarray'):
        a, b = root(int(left)), root(int(right))
        if a != b:
            parent[b] = a
    groups = {}
    for i in range(len(points)):
        groups.setdefault(root(i), []).append(i)
    return [np.asarray(ids, dtype=np.int32) for ids in groups.values()]


def _orient_and_measure(vertices, triangles):
    """Remove zero-area faces and return a closed, outward-facing mesh."""
    v = np.asarray(vertices, dtype=np.float64)
    f = np.asarray(triangles, dtype=np.int32).reshape(-1, 3)
    if not len(v) or not len(f) or not np.isfinite(v).all():
        raise RuntimeError('Particle surface extraction produced no finite mesh')
    # A zero crossing exactly on a grid node can be emitted once per incident
    # edge. Weld those numerical duplicates before removing collapsed faces.
    # Coordinates are spacing-normalized: the weld tolerance is one millionth
    # of the particle spacing, far below the visual grid resolution.
    _, first, remap = np.unique(np.round(v, decimals=6), axis=0, return_index=True, return_inverse=True)
    v, f = v[first], remap[f].astype(np.int32)
    crosses = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])
    f = f[np.linalg.norm(crosses, axis=1) > 1.0e-12]
    if not len(f):
        raise RuntimeError('Particle surface contains only degenerate triangles')
    used, inverse = np.unique(f, return_inverse=True)
    v, f = v[used], inverse.reshape(-1, 3).astype(np.int32)
    edges = np.sort(np.concatenate((f[:, (0, 1)], f[:, (1, 2)], f[:, (2, 0)])), axis=1)
    _, counts = np.unique(edges, axis=0, return_counts=True)
    if np.any(counts != 2):
        raise RuntimeError('Particle surface is not a closed two-manifold mesh')
    centred = v - v.mean(axis=0)
    volume = float(np.einsum('ij,ij->i', centred[f[:, 0]], np.cross(centred[f[:, 1]], centred[f[:, 2]])).sum() / 6.0)
    if abs(volume) <= 1.0e-12:
        raise RuntimeError('Particle surface has zero enclosed volume')
    if volume < 0:
        f = f[:, (0, 2, 1)]
        volume = -volume
    crosses = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])
    normals = np.zeros_like(v)
    for corner in range(3):
        np.add.at(normals, f[:, corner], crosses)
    lengths = np.linalg.norm(normals, axis=1)
    if np.any(lengths <= 1.0e-12):
        raise RuntimeError('Particle surface contains undefined vertex normals')
    normals /= lengths[:, None]
    return v, f, normals, volume


def reconstruct_particle_surface(
    points,
    spacing_m: float,
    *,
    device,
    voxel_size_m: float | None = None,
    kernel_radius_m: float | None = None,
    surface_radius_m: float | None = None,
    max_grid_nodes: int = 2_000_000,
    max_particles: int = 16_384,
) -> ParticleSurfaceMesh:
    """Build visual triangles without writing particle positions or physics.

    ``device`` must explicitly identify a CUDA device; no CPU fallback or GPU
    selection is performed. The caller owns CUDA visibility/device selection.
    ``source_sha256`` hashes an owned C-contiguous copy in the input dtype.
    World coordinates are centred and normalized before GPU evaluation so
    translating a vessel does not reduce the effective grid precision.

    All groups' grid sizes are checked before allocating GPU fields. No points
    are silently dropped or the voxel size silently coarsened to fit a budget.
    The resulting visual volume must never replace the native water ledger.
    """
    source = np.array(points, copy=True, order='C')
    if source.ndim != 2 or source.shape[1] != 3:
        raise ValueError('points must have shape (N, 3)')
    if not np.issubdtype(source.dtype, np.number) or np.iscomplexobj(source):
        raise ValueError('points must contain real numbers')
    points64 = np.asarray(source, dtype=np.float64)
    if not np.isfinite(points64).all():
        raise ValueError('points must all be finite')
    for name, value in [('max_grid_nodes', max_grid_nodes), ('max_particles', max_particles)]:
        if not isinstance(value, (int, np.integer)) or isinstance(value, bool) or value <= 0:
            raise ValueError(f'{name} must be a positive integer')
    if len(source) > max_particles:
        raise ValueError('Particle count exceeds max_particles')
    spacing = _positive(spacing_m, 'spacing_m')
    voxel = _positive(spacing / 4.0 if voxel_size_m is None else voxel_size_m, 'voxel_size_m')
    support = _positive(1.75 * spacing if kernel_radius_m is None else kernel_radius_m, 'kernel_radius_m')
    radius = _positive(0.5 * spacing if surface_radius_m is None else surface_radius_m, 'surface_radius_m')
    # The centre of a cubic spacing cell is sqrt(3)/2 spacings from a sample.
    # A smaller cap would puncture an otherwise fully sampled bulk liquid.
    distance_cap = 0.9 * spacing
    if radius > 0.75 * spacing or voxel > radius / 2.0 or support < spacing or support > 2.0 * spacing:
        raise ValueError('Require radius <= 0.75*spacing, voxel <= radius/2, and spacing <= kernel <= 2*spacing')
    if not str(device).startswith('cuda'):
        raise ValueError('Particle surface reconstruction requires an explicit CUDA device')
    metadata = {
        'method': 'weighted_local_centre_sdf_with_distance_cap_v1',
        'source_point_count': len(source),
        'source_sha256': hashlib.sha256(source.tobytes()).hexdigest(),
        'source_dtype': str(source.dtype),
        'source_shape': list(source.shape),
        'spacing_m': spacing, 'voxel_size_m': voxel,
        'kernel_radius_m': support, 'surface_radius_m': radius,
        'nearest_particle_distance_cap_m': distance_cap,
        'max_grid_nodes': int(max_grid_nodes), 'groups': [],
        'visual_only': True, 'visual_volume_is_mass_ledger': False,
    }
    if not len(source):
        metadata.update(group_count=0, grid_nodes=0, vertex_count=0, triangle_count=0, visual_volume_m3=0.0)
        return ParticleSurfaceMesh(np.empty((0, 3)), np.empty((0, 3), dtype=np.int32), np.empty((0, 3)), metadata)

    # A diameter cap of 1.8 spacing bounds all rendered connections. Connected
    # components prevent one far spilled drop from allocating a room-sized grid.
    shifted = (points64 - points64[0]) / spacing
    if not np.isfinite(shifted).all() or np.max(np.abs(shifted)) > 1.0e12:
        raise ValueError('Particle coordinate range exceeds the supported finite domain')
    groups = _components(shifted, 1.8)
    plans = []
    total_nodes = 0
    step = voxel / spacing
    for ids in groups:
        anchor = points64[int(ids[0])].copy()
        local = (points64[ids] - anchor) / spacing
        lower = local.min(axis=0) - 0.9 - 2.0 * step
        spans = local.max(axis=0) + 0.9 + 2.0 * step - lower
        if np.any(spans / step > max_grid_nodes):
            raise ValueError('Particle surface grid exceeds max_grid_nodes')
        shape = tuple(int(math.ceil(s / step)) + 1 for s in spans)
        nodes = math.prod(shape)
        total_nodes += nodes
        if total_nodes > max_grid_nodes:
            raise ValueError('Particle surface grids exceed max_grid_nodes')
        upper = lower + step * (np.asarray(shape) - 1)
        plans.append((ids, anchor, local, lower, upper, shape))

    if wp is None:
        raise RuntimeError('Warp from the native runtime is required for GPU surface reconstruction')
    selected = wp.get_device(device)
    if not selected.is_cuda:
        raise ValueError('Particle surface reconstruction requires CUDA')
    metadata['device'] = str(selected)
    vertices, triangles, normals = [], [], []
    vertex_offset = 0
    total_volume = 0.0
    for ids, anchor, local, lower, upper, shape in plans:
        gpu_points = wp.array(local.astype(np.float32), dtype=wp.vec3, device=selected)
        grid = wp.HashGrid(16, 16, 16, device=selected)
        grid.build(gpu_points, support / spacing)
        field = wp.empty(shape, dtype=wp.float32, device=selected)
        wp.launch(_make_surface_field, dim=shape,
                  inputs=[gpu_points, grid.id, wp.vec3(*lower), step, support / spacing,
                          radius / spacing, 0.9, field], device=selected)
        mc = wp.MarchingCubes(*shape, domain_bounds_lower_corner=tuple(lower), domain_bounds_upper_corner=tuple(upper))
        mc.surface(field, 0.0)
        v, f, n, volume = _orient_and_measure(mc.verts.numpy(), mc.indices.numpy())
        vertices.append(v * spacing + anchor)
        triangles.append(f + vertex_offset)
        normals.append(n)
        vertex_offset += len(v)
        total_volume += volume * spacing**3
        metadata['groups'].append({'source_indices': ids.tolist(), 'grid_shape': list(shape),
                                   'grid_nodes': math.prod(shape), 'vertex_count': len(v), 'triangle_count': len(f),
                                   'visual_volume_m3': volume * spacing**3})
    result_vertices = np.concatenate(vertices)
    result_triangles = np.concatenate(triangles)
    metadata.update(group_count=len(groups), grid_nodes=total_nodes, vertex_count=len(result_vertices),
                    triangle_count=len(result_triangles), visual_volume_m3=total_volume)
    return ParticleSurfaceMesh(result_vertices, result_triangles, np.concatenate(normals), metadata)
